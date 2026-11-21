"""許可したギルド・チャンネルだけで動く Discord アダプター。"""

import asyncio
import io
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import discord
from discord import app_commands

from ..chat import ChatService, RetrievedItem
from ..code_output import PreparedReply
from ..config import Settings
from ..discord_context import DiscordContext
from ..discord_users import author_context, public_user_names, remember_users, sdk_user
from ..errors import AIKunError, InvalidInputError
from ..file_types import attachment_kind, is_document_trigger
from ..history import sync_guild_members
from ..images import image_data_url
from ..observability import metrics
from ..request_diagnostics import RequestTrace, current_trace, phase
from ..streaming import current_text_update
from ..text_files import FILE_TOKEN_LIMIT
from ..work_queue import WorkQueue
from .reply_chain import collect_reply_chain, resolve_reply
from .stream_reply import StreamReply

logger = logging.getLogger(__name__)
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
REQUEST_HEARTBEAT_SECONDS = 15
TYPING_TIMEOUT_SECONDS = 3


def allowed(settings: Settings, guild_id: int | None) -> bool:
    """チャンネル単位の絞り込みは廃止。設定したサーバー（GUILD_ID）内なら全チャンネル対象。"""
    return guild_id is not None and guild_id == settings.discord.guild_id


def split_message(text: str) -> list[str]:
    # Discord の2000文字制限に対して、絵文字（UTF-16で2単位）も余裕をもって分割。
    chunks, current, units = [], [], 0
    for char in text:
        width = len(char.encode("utf-16-le")) // 2
        if units + width > 1900:
            chunks.append("".join(current))
            current, units = [], 0
        current.append(char)
        units += width
    if current:
        chunks.append("".join(current))
    return chunks or ["返答を作れませんでした。"]


class RestrictedTree(app_commands.CommandTree):
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not allowed(self.client.settings, interaction.guild_id):
            await interaction.response.send_message("このチャンネルでは利用できません。", ephemeral=True)
            return False
        return True

    async def on_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        if isinstance(error, app_commands.MissingPermissions):
            message = "この操作には「サーバー管理」権限が必要です。"
        else:
            logger.warning("command failed: %s", type(error).__name__)
            message = "操作に失敗しました。設定や権限を確認してください。"
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)


class AIKunClient(discord.Client):
    def __init__(self, settings: Settings, service: ChatService):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(
            intents=intents,
            application_id=settings.discord.application_id,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.service = service
        self._response_queue = WorkQueue()
        self._parallel_responses: set[asyncio.Task] = set()
        self._voice_sessions = {}
        self._voice_starting: set[int] = set()
        self._voice_tts_locks: dict[int, asyncio.Lock] = {}
        self._voice_notice_channels = {}
        self._message_tasks: set[asyncio.Task] = set()
        self._history_tasks: set[asyncio.Task] = set()
        self._history_task_channels: dict[asyncio.Task, int] = {}
        self._history_loop_task: asyncio.Task | None = None
        self._members_synced = False
        self.tree = RestrictedTree(self)
        self._register_commands()

    async def setup_hook(self) -> None:
        guild = discord.Object(id=self.settings.discord.guild_id)
        self.tree.copy_global_to(guild=guild)
        synced = await self.tree.sync(guild=guild)
        logger.info(
            "Discord guild commands synced: guild_id=%s count=%d commands=%s",
            self.settings.discord.guild_id,
            len(synced),
            ",".join(command.name for command in synced),
        )
        if self.settings.asr.enabled:
            asyncio.create_task(self._prepare_asr_model())

    async def _prepare_asr_model(self) -> None:
        try:
            await self.service.prepare_asr_model()
            logger.info("ASR model ready: %s", self.settings.asr.model)
        except Exception as exc:
            logger.warning("ASR model download/load failed: %s", type(exc).__name__)

    async def on_ready(self) -> None:
        logger.info("Discord connected; configured guild commands ready")
        metrics.increment("ai_kun.discord_connections_total")
        if not self._members_synced:
            try:
                count = await sync_guild_members(self.settings.discord.guild_id, self.settings.discord.bot_token)
                logger.info("Discord member cache synced: members=%d", count)
                self._members_synced = True
            except Exception as exc:
                logger.warning("Discord member cache sync failed: %s", type(exc).__name__)
        if self.settings.memory.enabled and (self._history_loop_task is None or self._history_loop_task.done()):
            self._history_loop_task = asyncio.create_task(self._history_loop())

    async def _history_loop(self):
        while not self.is_closed():
            try:
                self._sweep_history()
            except Exception as exc:
                logger.warning("history sweep failed: %s", type(exc).__name__)
            await asyncio.sleep(60)

    def _sweep_history(self):
        guild = self.get_guild(self.settings.discord.guild_id)
        if guild is None or guild.me is None:
            return
        pending = set(self._history_task_channels.values())
        channels = {channel.id: channel for channel in [*guild.channels, *guild.threads]}
        for channel in channels.values():
            if not hasattr(channel, "history") or channel.id in pending:
                continue
            permissions = channel.permissions_for(guild.me)
            if permissions.view_channel and permissions.read_message_history:
                self._schedule_backfill(channel.id)

    async def _run(
        self,
        operation: Callable[[], Awaitable[str]],
        *,
        request_id: str | None = None,
        publisher: StreamReply | None = None,
    ) -> str:
        trace = RequestTrace(request_id or uuid.uuid4().hex[:12])
        logger.info("request queued: request=%s", trace.request_id)

        async def execute():
            token = current_trace.set(trace)
            update_token = current_text_update.set(publisher.update if publisher else None)
            if publisher:
                publisher.start()
            phase("starting")
            logger.info(
                "request started: request=%s queue_seconds=%.3f",
                trace.request_id,
                time.monotonic() - trace.accepted_at,
            )
            heartbeat = asyncio.create_task(self._request_heartbeat(trace))
            started = time.monotonic()
            try:
                result = await self._execute(operation)
                if publisher:
                    visible = public_user_names(str(result), self.settings.discord.guild_id)
                    result = result.with_text(visible) if isinstance(result, PreparedReply) else visible
                    await publisher.finish(result, failed=trace.failed)
                return result
            except asyncio.CancelledError:
                logger.warning("request cancelled: request=%s phase=%s", trace.request_id, trace.phase)
                raise
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
                logger.info(
                    "request finished: request=%s phase=%s seconds=%.3f",
                    trace.request_id,
                    trace.phase,
                    time.monotonic() - started,
                )
                if publisher:
                    await publisher.close()
                current_text_update.reset(update_token)
                current_trace.reset(token)

        if not self.settings.llm.parallel_requests:
            return await self._response_queue.submit(execute)
        task = asyncio.current_task()
        if task is not None:
            self._parallel_responses.add(task)
        try:
            return await execute()
        finally:
            if task is not None:
                self._parallel_responses.discard(task)

    async def _request_heartbeat(self, trace):
        while True:
            await asyncio.sleep(REQUEST_HEARTBEAT_SECONDS)
            logger.info(
                "request pending: request=%s phase=%s total_seconds=%.3f",
                trace.request_id,
                trace.phase,
                time.monotonic() - trace.accepted_at,
            )

    @asynccontextmanager
    async def _typing(self, channel):
        manager = channel.typing()
        entered = False
        try:
            await asyncio.wait_for(manager.__aenter__(), timeout=TYPING_TIMEOUT_SECONDS)
            entered = True
        except Exception as exc:
            logger.warning("typing unavailable: error=%s", type(exc).__name__)
        try:
            yield
        finally:
            if entered:
                try:
                    await asyncio.wait_for(manager.__aexit__(None, None, None), timeout=TYPING_TIMEOUT_SECONDS)
                except Exception as exc:
                    logger.warning("typing cleanup failed: error=%s", type(exc).__name__)

    async def _execute(self, operation: Callable[[], Awaitable[str]]) -> str:
        # キュー待機時間は実行タイムアウトへ含めない。
        try:
            return await asyncio.wait_for(operation(), timeout=self.settings.operational.request_timeout_seconds)
        except AIKunError as exc:
            trace = current_trace.get()
            if trace:
                trace.failed = True
            logger.warning(
                "request failed: request=%s phase=%s error=%s service=%s",
                trace.request_id if trace else "unknown",
                trace.phase if trace else "unknown",
                type(exc).__name__,
                getattr(exc, "service", "internal"),
            )
            return exc.user_message()
        except TimeoutError:
            trace = current_trace.get()
            if trace:
                trace.failed = True
            logger.warning(
                "request timed out: request=%s phase=%s timeout=%s",
                trace.request_id if trace else "unknown",
                trace.phase if trace else "unknown",
                self.settings.operational.request_timeout_seconds,
            )
            return "時間がかかりすぎたため中断しました。少し待ってから試してください。"
        except Exception as exc:
            # 例外本文は外部レスポンスや秘密を含む場合があるため出さない。
            trace = current_trace.get()
            if trace:
                trace.failed = True
            logger.warning(
                "request failed: request=%s phase=%s error=%s",
                trace.request_id if trace else "unknown",
                trace.phase if trace else "unknown",
                type(exc).__name__,
            )
            metrics.increment("ai_kun.failures_total")
            return "処理に失敗しました。管理者に設定の確認をお願いしてください。"

    async def drain(self):
        if self._history_loop_task is not None:
            self._history_loop_task.cancel()
            await asyncio.gather(self._history_loop_task, return_exceptions=True)
        # 接続終了後も、受付済みメッセージから後続の取り込みがキューへ入るまで待つ。
        if self._message_tasks:
            await asyncio.gather(*self._message_tasks, return_exceptions=True)
        if self._history_tasks:
            await asyncio.gather(*self._history_tasks, return_exceptions=True)
        for guild_id in list(self._voice_sessions):
            await self._stop_voice_stt(guild_id, announce=False)
        if self._parallel_responses:
            await asyncio.gather(*tuple(self._parallel_responses), return_exceptions=True)
        await self._response_queue.join()
        await self.service.drain()

    async def _answer(self, interaction: discord.Interaction, operation, *, ephemeral=False) -> None:
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)
        request_id = f"interaction-{getattr(interaction, 'id', 'unknown')}"

        async def send(text, *, files=None):
            return await interaction.followup.send(
                text,
                ephemeral=ephemeral,
                allowed_mentions=discord.AllowedMentions.none(),
                wait=True,
                **({"files": files} if files else {}),
            )

        publisher = StreamReply(
            send,
            split_message,
            request_id=request_id,
            kind="followup",
            code_token_limit=self.settings.operational.max_code_tokens,
            prose_limit=self.settings.operational.max_output_chars,
        )
        await self._run(operation, request_id=request_id, publisher=publisher)

    def _register_commands(self) -> None:
        @self.tree.command(name="ask", description="AIくんに質問する（会話は自動保存しません）")
        @app_commands.describe(question="質問内容")
        async def ask(interaction: discord.Interaction, question: str):
            async def operation():
                user = getattr(interaction, "user", None)
                if user is not None:
                    await asyncio.to_thread(remember_users, self.settings.discord.guild_id, [sdk_user(user)])
                context = (
                    DiscordContext(author_id=str(user.id), author=author_context(user)) if user is not None else None
                )
                answer = await self.service.generate_reply(
                    question,
                    channel_id=interaction.channel_id,
                    discord_context=context,
                )
                return PreparedReply(answer.content, files=getattr(answer, "files", ()))

            await self._answer(interaction, operation)

        @self.tree.command(name="search_memory", description="保存記憶を検索して管理者にだけ表示する")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        @app_commands.describe(query="検索語", user="発言した人で絞る", channel="発言先チャンネルで絞る")
        async def search_memory(
            interaction: discord.Interaction,
            query: str,
            user: discord.User | None = None,
            channel: discord.TextChannel | None = None,
        ):
            async def operation():
                items = await self.service.search_memory(
                    query,
                    user_id=user.id if user else None,
                    channel_id=channel.id if channel else None,
                )
                if not items:
                    return "条件に合う記憶は見つからなかったよ。"
                return "\n\n".join(f"〔{item.label}〕\n{item.text}" for item in items)

            await self._answer(interaction, operation, ephemeral=True)

        @self.tree.command(name="remember", description="指定した内容をサーバー共通の記憶に保存する")
        @app_commands.describe(text="保存するテキスト。許可チャンネルの利用者全員が参照できます")
        async def remember(interaction: discord.Interaction, text: str):
            async def operation():
                result = await self.service.remember(text, channel_id=interaction.channel_id)
                return f"保存しました。文書ID: `{result['document_id']}`（{result['chunks']}チャンク）"

            await self._answer(interaction, operation, ephemeral=True)

        @self.tree.command(name="ingest", description="テキスト資料をサーバー共通の記憶へ登録する（管理者）")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        async def ingest(interaction: discord.Interaction, file: discord.Attachment):
            async def operation():
                if file.size > MAX_ATTACHMENT_BYTES:
                    raise InvalidInputError("添付は8 MiBまでです。")
                result = await self.service.ingest_document(
                    file.filename,
                    await file.read(),
                    channel_id=interaction.channel_id,
                    content_type=getattr(file, "content_type", None),
                )
                note = "（先頭4,096トークン以内のみ登録）" if result.get("truncated") else ""
                return f"登録しました。文書ID: `{result['document_id']}`（{result['chunks']}チャンク）{note}"

            await self._answer(interaction, operation, ephemeral=True)

        @self.tree.command(name="forget", description="文書IDを指定して記憶から削除する（管理者）")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        async def forget(interaction: discord.Interaction, document_id: str):
            async def operation():
                count = await self.service.delete_memory(document_id)
                return f"削除しました（{count}チャンク）。"

            await self._answer(interaction, operation, ephemeral=True)

        @self.tree.command(name="vc_stt_start", description="今いるVCの音声を文字起こししてRAGへ保存する（管理者）")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        async def vc_stt_start(interaction: discord.Interaction):
            if not self.settings.asr.enabled:
                await interaction.response.send_message("ASR_ENABLED=true が必要です。", ephemeral=True)
                return
            if not self.settings.memory.enabled:
                await interaction.response.send_message("RAG記憶が無効です。", ephemeral=True)
                return
            try:
                await self.service.prepare_asr_model()
            except Exception:
                await interaction.response.send_message(
                    "Whisperモデルを取得できませんでした。ネットワークとモデル設定を確認してください。", ephemeral=True
                )
                return
            voice_state = getattr(interaction.user, "voice", None)
            channel = getattr(voice_state, "channel", None)
            if not isinstance(channel, discord.VoiceChannel):
                await interaction.response.send_message("先に通常のVCへ入ってから実行してください。", ephemeral=True)
                return
            guild = interaction.guild
            if guild.id in self._voice_sessions or guild.id in self._voice_starting:
                await interaction.response.send_message(
                    "このサーバーではVC文字起こしがすでに動いています。", ephemeral=True
                )
                return
            if guild.voice_client is not None:
                await interaction.response.send_message("Botはすでに別のVCセッションを使用中です。", ephemeral=True)
                return
            me = guild.me
            if me is None or not channel.permissions_for(me).connect:
                await interaction.response.send_message("BotにVCの接続権限がありません。", ephemeral=True)
                return
            try:
                from discord.ext import voice_recv

                from ..voice_stt import VoiceTranscriptSession
            except ImportError:
                await interaction.response.send_message(
                    "音声受信依存がありません。voice extraを入れてください。", ephemeral=True
                )
                return

            self._voice_starting.add(guild.id)
            notice_channel = interaction.channel
            try:
                await interaction.response.defer(ephemeral=True)
                await notice_channel.send(
                    "🎙️ VC文字起こしを開始します。音声はローカルWhisperで処理し、音声ファイルは保存せず、"
                    "話者別の文字起こしをRAG記憶へ登録します。停止は `/vc_stt_stop`。"
                )
                voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)

                async def announce_transcript(author: dict, text: str):
                    name = author.get("nick") or author.get("global_name") or author.get("username") or "メンバー"
                    name = str(name).replace("@", "@\u200b")[:80]
                    for part in split_message(text):
                        await notice_channel.send(
                            f"🎙️ **{name}**: {part}", allowed_mentions=discord.AllowedMentions.none()
                        )

                session = VoiceTranscriptSession(voice_client, self.service, guild.id, channel.id, announce_transcript)
                await session.start()
                self._voice_sessions[guild.id] = session
                self._voice_notice_channels[guild.id] = notice_channel
                await interaction.followup.send(f"{channel.mention} の文字起こしを開始しました。", ephemeral=True)
            except Exception as exc:
                logger.warning("VC STT start failed: %s", type(exc).__name__)
                current = guild.voice_client
                if current is not None and current.is_connected():
                    await current.disconnect(force=True)
                await interaction.followup.send(
                    "VCへ接続できませんでした。Botの接続権限を確認してください。", ephemeral=True
                )
            finally:
                self._voice_starting.discard(guild.id)

        @self.tree.command(name="vc_stt_stop", description="VC文字起こしを停止して未処理分をRAGへ保存する（管理者）")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        async def vc_stt_stop(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)
            if await self._stop_voice_stt(interaction.guild_id):
                await interaction.followup.send(
                    "VC文字起こしを停止しました。未処理分もRAGへ保存しました。", ephemeral=True
                )
            else:
                await interaction.followup.send("VC文字起こしは動いていません。", ephemeral=True)

    async def _stop_voice_stt(self, guild_id: int | None, *, announce: bool = True) -> bool:
        if guild_id is None:
            return False
        session = self._voice_sessions.pop(guild_id, None)
        notice_channel = self._voice_notice_channels.pop(guild_id, None)
        if session is None:
            return False
        try:
            await session.stop()
        finally:
            voice_client = session.voice_client
            if voice_client.is_connected():
                await voice_client.disconnect(force=True)
        if announce and notice_channel is not None:
            try:
                await notice_channel.send("⏹️ VC文字起こしを停止しました。文字起こしはRAG記憶に保存済みです。")
            except Exception as exc:
                logger.warning("VC STT stop announcement failed: %s", type(exc).__name__)
        return True

    async def on_voice_state_update(self, member, before, after) -> None:
        guild_id = getattr(getattr(member, "guild", None), "id", None)
        session = self._voice_sessions.get(guild_id)
        if session is None:
            return
        if self.user is not None and member.id == self.user.id and after.channel is None:
            await self._stop_voice_stt(guild_id, announce=True)
            return
        voice_channel = session.voice_client.channel
        if before.channel is voice_channel and after.channel is not voice_channel:
            await asyncio.sleep(0.5)
            if not any(not participant.bot for participant in voice_channel.members):
                await self._stop_voice_stt(guild_id, announce=True)

    async def on_message(self, message: discord.Message) -> None:
        task = asyncio.current_task()
        self._message_tasks.add(task)
        try:
            await self._handle_message(message)
        finally:
            self._message_tasks.discard(task)

    def _schedule_backfill(self, channel_id: int):
        if not self.settings.memory.enabled:
            return
        task = asyncio.create_task(self._run_backfill(channel_id))
        self._history_tasks.add(task)
        self._history_task_channels[task] = channel_id

        def finished(task):
            self._history_tasks.discard(task)
            self._history_task_channels.pop(task, None)

        task.add_done_callback(finished)

    async def _run_backfill(self, channel_id: int):
        try:
            result = await self.service.backfill_channel(channel_id)
            if result["imported"]:
                logger.info(
                    "auto backfill filled gaps channel=%s scanned=%d imported=%d documents=%d",
                    channel_id,
                    result["scanned"],
                    result["imported"],
                    result["documents"],
                )
        except Exception as exc:
            logger.warning(
                "auto backfill failed: %s (%s)",
                type(exc).__name__,
                exc.user_message() if isinstance(exc, AIKunError) else "詳細非表示",
            )

    async def _handle_message(self, message: discord.Message) -> None:
        if message.author.bot or not allowed(self.settings, message.guild.id if message.guild else None):
            return

        # 全メッセージ収集：フック（メンション・!ai・/ask）の有無や内容に関係なく、
        # サーバー内の人間の発言は受付順にRAG記憶へ取り込む。失敗しても返信処理は続行。
        try:
            await self.service.ingest_message(message)
        except Exception as exc:  # 取り込み失敗を返信やログには出す。本文は出さない。
            logger.warning(
                "live history ingest failed: %s (%s)",
                type(exc).__name__,
                exc.user_message() if isinstance(exc, AIKunError) else "詳細非表示",
            )

        # 取得要求はすべて受け付けるが、返信は過去ログ取得の完了を待たない。
        self._schedule_backfill(message.channel.id)

        trigger = self.settings.voice_tts.trigger
        trigger_index = message.content.casefold().find(trigger.casefold()) if trigger else -1
        if trigger_index >= 0:
            prompt = message.content[trigger_index + len(trigger) :].lstrip(" \t\r\n:：,、")
            if prompt:
                await self._speak_tts_in_author_vc(message, prompt)
            return

        content = message.content.strip()
        reply_target, reply_stop = await resolve_reply(message, message.channel, message.guild.id)
        is_bot_reply = (
            reply_target is not None
            and self.user is not None
            and getattr(getattr(reply_target, "author", None), "id", None) == self.user.id
        )
        target_text = str(getattr(reply_target, "content", "") or "")
        refresh_reply = is_bot_reply and (target_text.startswith("考え中…") or target_text.endswith("\n…（生成中）"))
        prefixes = (f"<@{self.user.id}>", f"<@!{self.user.id}>") if self.user else ()
        if content == "!ai" or content.startswith("!ai "):
            content = content[3:].strip()
        elif content.startswith(prefixes):
            for prefix in prefixes:
                content = content.removeprefix(prefix).strip()
        elif is_bot_reply:
            pass
        elif not any(
            is_document_trigger(attachment.filename, getattr(attachment, "content_type", None))
            for attachment in message.attachments
        ):
            return

        async def operation():
            # キュー待機中はtypingを維持しない。typing APIの停止も推論を妨げない。
            async with self._typing(message.channel):
                phase("reply_chain")
                target, stop = reply_target, reply_stop
                if refresh_reply:
                    # 先行する生成の完了後に、途中表示への返信を最新の本文へ解決する。
                    target, stop = await resolve_reply(message, message.channel, message.guild.id, refresh=True)
                context = await collect_reply_chain(message, target, stop)
                phase("attachments")
                return await self._process_attachments(
                    content, message.attachments, message.channel.id, discord_context=context
                )

        request_id = str(getattr(message, "id", uuid.uuid4().hex[:12]))

        async def send(text, *, files=None):
            return await message.reply(
                text,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
                **({"files": files} if files else {}),
            )

        publisher = StreamReply(
            send,
            split_message,
            request_id=request_id,
            code_token_limit=self.settings.operational.max_code_tokens,
            prose_limit=self.settings.operational.max_output_chars,
        )
        await self._run(operation, request_id=request_id, publisher=publisher)

    async def _speak_tts_in_author_vc(self, message: discord.Message, prompt: str) -> None:
        guild = message.guild
        state = getattr(message.author, "voice", None)
        channel = getattr(state, "channel", None)
        if guild is None or not isinstance(channel, discord.VoiceChannel):
            return
        lock = self._voice_tts_locks.setdefault(guild.id, asyncio.Lock())
        if lock.locked():
            return

        async with lock:
            voice_client = guild.voice_client
            disconnect_after = False
            try:
                voice = self.settings.voice_tts.voice

                async def operation():
                    async with self._typing(message.channel):
                        phase("reply_chain")
                        target, stop = await resolve_reply(message, message.channel, guild.id)
                        context = await collect_reply_chain(message, target, stop)
                        phase("attachments")
                        answer = await self.service.generate_reply(
                            prompt,
                            channel_id=message.channel.id,
                            discord_context=context,
                        )
                        return answer.content

                text = str(await self._run(operation, request_id=str(message.id))).strip()
                if not text:
                    return
                from edge_tts import Communicate

                audio = bytearray()
                async for chunk in Communicate(text, voice).stream():
                    if chunk.get("type") == "audio":
                        audio.extend(chunk["data"])
                        if len(audio) > 8 * 1024 * 1024:
                            raise ValueError("TTS audio exceeded size limit")
                if not audio:
                    raise ValueError("TTS returned no audio")

                if voice_client is not None and voice_client.is_connected():
                    if getattr(voice_client, "channel", None) != channel:
                        logger.info("VC TTS skipped: bot is connected to a different channel")
                        return
                else:
                    me = guild.me
                    permissions = channel.permissions_for(me) if me else None
                    if permissions is None or not permissions.connect or not permissions.speak:
                        logger.info("VC TTS skipped: bot lacks connect or speak permission")
                        return
                    voice_client = await channel.connect()
                    disconnect_after = True

                loop = asyncio.get_running_loop()
                finished = loop.create_future()

                def after_playback(error):
                    def complete():
                        if finished.done():
                            return
                        if error:
                            finished.set_exception(error)
                        else:
                            finished.set_result(None)

                    loop.call_soon_threadsafe(complete)

                source = discord.FFmpegPCMAudio(io.BytesIO(bytes(audio)), pipe=True)
                voice_client.play(source, after=after_playback)
                await asyncio.wait_for(finished, timeout=600)
            except Exception as exc:
                logger.warning("VC TTS failed: %s", type(exc).__name__)
            finally:
                if disconnect_after and voice_client is not None and voice_client.is_connected():
                    try:
                        await voice_client.disconnect(force=True)
                    except Exception as exc:
                        logger.warning("VC TTS disconnect failed: %s", type(exc).__name__)

    async def _process_attachments(
        self, content: str, attachments, channel_id: int, *, discord_context: DiscordContext | None = None
    ) -> str:
        text = content
        documents = []
        reply_kwargs = {"channel_id": channel_id}
        if discord_context and (
            discord_context.message_id
            or discord_context.messages
            or discord_context.stop_reason
            or discord_context.stickers
        ):
            reply_kwargs["discord_context"] = discord_context
        images = []
        notices = []
        prepared = []
        audio_count = 0
        tokens_remaining = FILE_TOKEN_LIMIT
        # 全ファイルの形式・サイズを先に確認し、テキストを音声認識へ送らない。
        for attachment in attachments:
            if attachment.size > MAX_ATTACHMENT_BYTES:
                raise InvalidInputError("添付ファイルは1つ8 MiBまでです。")
            kind = attachment_kind(attachment.filename, getattr(attachment, "content_type", None))
            if kind == "unsupported":
                raise InvalidInputError(
                    "この添付形式には対応していません。テキスト資料・画像・音声ファイルを送ってください。"
                )
            if kind == "audio":
                audio_count += 1
            # 不明な形式は実データを検証し、読めるテキストだけを資料にする。
            prepared.append((attachment, "document" if kind == "unknown" else kind))
        if audio_count > 1:
            raise InvalidInputError("音声ファイルは1つずつ送ってください。テキスト資料は複数添付できます。")
        loaded = []
        documents_remaining = sum(kind == "document" for _, kind in prepared)
        for attachment, kind in prepared:
            data = await attachment.read()
            if len(data) > MAX_ATTACHMENT_BYTES:
                raise InvalidInputError("添付ファイルは1つ8 MiBまでです。")
            document = None
            if kind == "document":
                document = self.service.read_document(
                    attachment.filename,
                    data,
                    token_limit=tokens_remaining // documents_remaining,
                    content_type=getattr(attachment, "content_type", None),
                )
                tokens_remaining -= document.tokens
                documents_remaining -= 1
            loaded.append((attachment, kind, data, document))
        for attachment, kind, data, document in loaded:
            if kind == "audio":
                transcript = await self.service.transcribe_attachment(data, attachment.filename)
                text = f"{text}\n{transcript}".strip()
            elif kind == "image":
                images.append(image_data_url(data))
            else:
                label = f"attachment:{attachment.filename}"
                if self.settings.memory.enabled:
                    result = await self.service.ingest_document(
                        attachment.filename, document.text.encode("utf-8"), channel_id=channel_id
                    )
                    label = f"document:{result['document_id']} ({attachment.filename})"
                if document.truncated:
                    notices.append(f"{attachment.filename}: 先頭部分のみ参照しました（添付合計4,096トークン上限）。")
                documents.append(RetrievedItem(label=label, text=document.text))
        if not text.strip():
            if images:
                text = "添付画像の内容を説明してください。"
            elif documents:
                text = "添付資料の要点を短くまとめてください。"
            elif discord_context and discord_context.stickers:
                text = "今回のスタンプの内容を説明してください。取得できなかった画像の内容は推測しないでください。"
            elif discord_context and discord_context.messages:
                text = "返信チェーンの発言と、取得できた画像を参照して、返信元の質問や依頼に答えてください。取得できなかった画像の内容は推測しないでください。"
        if text.strip():
            if images:
                answer = await self.service.generate_reply(text, documents=documents, images=images, **reply_kwargs)
            elif documents:
                answer = await self.service.generate_reply(text, documents=documents, **reply_kwargs)
            else:
                answer = await self.service.generate_reply(text, **reply_kwargs)
            return PreparedReply("\n\n".join([answer.content, *notices]), files=getattr(answer, "files", ()))
        raise InvalidInputError("質問か、テキスト資料・画像・音声ファイルを送ってください。")
