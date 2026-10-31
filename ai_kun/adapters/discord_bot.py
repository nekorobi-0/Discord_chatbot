"""許可したギルド・チャンネルだけで動く Discord アダプター。"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import discord
from discord import app_commands

from ..chat import ChatService, RetrievedItem
from ..config import Settings
from ..discord_context import DiscordContext
from ..discord_users import public_user_names, remember_users, sdk_user
from ..errors import AIKunError, InvalidInputError
from ..file_types import attachment_kind, is_document_trigger
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
        super().__init__(
            intents=intents,
            application_id=settings.discord.application_id,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.service = service
        self._response_queue = WorkQueue()
        self._message_tasks: set[asyncio.Task] = set()
        self._history_tasks: set[asyncio.Task] = set()
        self._history_task_channels: dict[asyncio.Task, int] = {}
        self._history_loop_task: asyncio.Task | None = None
        self.tree = RestrictedTree(self)
        self._register_commands()

    async def setup_hook(self) -> None:
        guild = discord.Object(id=self.settings.discord.guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)

    async def on_ready(self) -> None:
        logger.info("Discord connected; configured guild commands ready")
        metrics.increment("ai_kun.discord_connections_total")
        guild = self.get_guild(self.settings.discord.guild_id)
        members = getattr(guild, "members", ()) if guild else ()
        if members:
            await asyncio.to_thread(
                remember_users, self.settings.discord.guild_id, [sdk_user(user) for user in members]
            )
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
                    result = public_user_names(result, self.settings.discord.guild_id)
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

        return await self._response_queue.submit(execute)

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
        await self._response_queue.join()
        await self.service.drain()

    async def _answer(self, interaction: discord.Interaction, operation, *, ephemeral=False) -> None:
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)
        request_id = f"interaction-{getattr(interaction, 'id', 'unknown')}"

        async def send(text):
            return await interaction.followup.send(
                text,
                ephemeral=ephemeral,
                allowed_mentions=discord.AllowedMentions.none(),
                wait=True,
            )

        publisher = StreamReply(send, split_message, request_id=request_id, kind="followup")
        await self._run(operation, request_id=request_id, publisher=publisher)

    def _register_commands(self) -> None:
        @self.tree.command(name="ask", description="AIくんに質問する（会話は自動保存しません）")
        @app_commands.describe(question="質問内容")
        async def ask(interaction: discord.Interaction, question: str):
            async def operation():
                user = getattr(interaction, "user", None)
                context = DiscordContext(author_id=str(user.id)) if user is not None else None
                return (
                    await self.service.generate_reply(
                        question,
                        channel_id=interaction.channel_id,
                        discord_context=context,
                    )
                ).content

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

        async def send(text):
            return await message.reply(text, mention_author=False, allowed_mentions=discord.AllowedMentions.none())

        publisher = StreamReply(send, split_message, request_id=request_id)
        await self._run(operation, request_id=request_id, publisher=publisher)

    async def _process_attachments(
        self, content: str, attachments, channel_id: int, *, discord_context: DiscordContext | None = None
    ) -> str:
        text = content
        documents = []
        reply_kwargs = {"channel_id": channel_id}
        if discord_context and (discord_context.message_id or discord_context.messages or discord_context.stop_reason):
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
            elif discord_context and discord_context.messages:
                text = "返信チェーンの発言と、取得できた画像を参照して、返信元の質問や依頼に答えてください。取得できなかった画像の内容は推測しないでください。"
        if text.strip():
            if images:
                answer = await self.service.generate_reply(text, documents=documents, images=images, **reply_kwargs)
            elif documents:
                answer = await self.service.generate_reply(text, documents=documents, **reply_kwargs)
            else:
                answer = await self.service.generate_reply(text, **reply_kwargs)
            return "\n\n".join([answer.content, *notices])
        raise InvalidInputError("質問か、テキスト資料・画像・音声ファイルを送ってください。")
