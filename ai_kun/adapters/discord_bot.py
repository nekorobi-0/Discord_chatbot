"""許可したギルド・チャンネルだけで動く Discord アダプター。"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import discord
from discord import app_commands

from ..chat import ChatService, RetrievedItem
from ..config import Settings
from ..errors import AIKunError, InvalidInputError
from ..images import IMAGE_EXTENSIONS, image_data_url
from ..memory import DOCUMENT_EXTENSIONS, VOICE_EXTENSIONS
from ..observability import metrics
from ..text_files import FILE_TOKEN_LIMIT
from ..work_queue import WorkQueue

logger = logging.getLogger(__name__)
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024


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
        self.tree = RestrictedTree(self)
        self._register_commands()

    async def setup_hook(self) -> None:
        guild = discord.Object(id=self.settings.discord.guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)

    async def on_ready(self) -> None:
        logger.info("Discord connected; configured guild commands ready")
        metrics.increment("ai_kun.discord_connections_total")

    async def _run(self, operation: Callable[[], Awaitable[str]]) -> str:
        return await self._response_queue.submit(lambda: self._execute(operation))

    async def _execute(self, operation: Callable[[], Awaitable[str]]) -> str:
        # キュー待機時間は実行タイムアウトへ含めない。
        try:
            return await asyncio.wait_for(operation(), timeout=240)
        except AIKunError as exc:
            return exc.user_message()
        except TimeoutError:
            return "時間がかかりすぎたため中断しました。少し待ってから試してください。"
        except Exception as exc:
            # 例外本文は外部レスポンスや秘密を含む場合があるため出さない。
            logger.warning("request failed: %s", type(exc).__name__)
            metrics.increment("ai_kun.failures_total")
            return "処理に失敗しました。管理者に設定の確認をお願いしてください。"

    async def drain(self):
        # 接続終了後も、受付済みメッセージから後続の取り込みがキューへ入るまで待つ。
        if self._message_tasks:
            await asyncio.gather(*self._message_tasks, return_exceptions=True)
        if self._history_tasks:
            await asyncio.gather(*self._history_tasks, return_exceptions=True)
        await self._response_queue.join()
        await self.service.drain()

    async def _answer(self, interaction: discord.Interaction, operation, *, ephemeral=False) -> None:
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)
        result = await self._run(operation)
        for chunk in split_message(result):
            await interaction.followup.send(chunk, ephemeral=ephemeral, allowed_mentions=discord.AllowedMentions.none())

    def _register_commands(self) -> None:
        @self.tree.command(name="ask", description="AIくんに質問する（会話は自動保存しません）")
        @app_commands.describe(question="質問内容")
        async def ask(interaction: discord.Interaction, question: str):
            async def operation():
                return (await self.service.generate_reply(question, channel_id=interaction.channel_id)).content

            await self._answer(interaction, operation)

        @self.tree.command(name="search_memory", description="保存記憶を検索して管理者にだけ表示する")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        @app_commands.describe(query="検索語")
        async def search_memory(interaction: discord.Interaction, query: str):
            async def operation():
                items = await self.service.search_memory(query)
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
                    file.filename, await file.read(), channel_id=interaction.channel_id
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
        task = asyncio.create_task(self._run_backfill(channel_id))
        self._history_tasks.add(task)
        task.add_done_callback(self._history_tasks.discard)

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

    async def _bot_reply_target(self, message):
        reference = getattr(message, "reference", None)
        if reference is None or self.user is None:
            return None
        if getattr(reference, "type", discord.MessageReferenceType.default) != discord.MessageReferenceType.default:
            return None
        if getattr(reference, "guild_id", message.guild.id) not in (None, message.guild.id):
            return None
        if getattr(reference, "channel_id", message.channel.id) not in (None, message.channel.id):
            return None
        target = getattr(reference, "resolved", None) or getattr(reference, "cached_message", None)
        if isinstance(target, discord.DeletedReferencedMessage):
            return None
        if target is None and getattr(reference, "message_id", None) is not None:
            try:
                target = await message.channel.fetch_message(reference.message_id)
            except discord.HTTPException as exc:
                logger.warning("reply target fetch failed: status=%s code=%s", exc.status, exc.code)
                return None
        author = getattr(target, "author", None)
        return target if author is not None and author.id == self.user.id else None

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
        reply_target = await self._bot_reply_target(message)
        prefixes = (f"<@{self.user.id}>", f"<@!{self.user.id}>") if self.user else ()
        if content == "!ai" or content.startswith("!ai "):
            content = content[3:].strip()
        elif content.startswith(prefixes):
            for prefix in prefixes:
                content = content.removeprefix(prefix).strip()
        elif reply_target is not None:
            pass
        elif not any(
            attachment.filename.lower().endswith(tuple(DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS))
            for attachment in message.attachments
        ):
            return

        async def operation():
            return await self._process_attachments(
                content, message.attachments, message.channel.id, reply_target=reply_target
            )

        async with message.channel.typing():
            result = await self._run(operation)
        for chunk in split_message(result):
            await message.reply(chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none())

    async def _process_attachments(self, content: str, attachments, channel_id: int, *, reply_target=None) -> str:
        text = content
        documents = []
        if reply_target is not None:
            documents.append(
                RetrievedItem(
                    label=f"今回の返信先Botメッセージ (message_id={reply_target.id})",
                    text=str(reply_target.content or "")[:4000],
                )
            )
        images = []
        notices = []
        prepared = []
        audio_count = 0
        tokens_remaining = FILE_TOKEN_LIMIT
        # 全ファイルの形式・サイズを先に確認し、テキストを音声認識へ送らない。
        for attachment in attachments:
            name = attachment.filename.lower()
            if attachment.size > MAX_ATTACHMENT_BYTES:
                raise InvalidInputError("添付ファイルは1つ8 MiBまでです。")
            if name.endswith(tuple(DOCUMENT_EXTENSIONS)):
                prepared.append((attachment, "document"))
            elif name.endswith(tuple(IMAGE_EXTENSIONS)):
                prepared.append((attachment, "image"))
            elif name.endswith(tuple(VOICE_EXTENSIONS)):
                audio_count += 1
                prepared.append((attachment, "audio"))
            else:
                raise InvalidInputError(
                    "この添付形式には対応していません。テキスト資料・画像・音声ファイルを送ってください。"
                )
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
                    attachment.filename, data, token_limit=tokens_remaining // documents_remaining
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
        if text.strip():
            if images:
                answer = await self.service.generate_reply(
                    text, documents=documents, images=images, channel_id=channel_id
                )
            elif documents:
                answer = await self.service.generate_reply(text, documents=documents, channel_id=channel_id)
            else:
                answer = await self.service.generate_reply(text, channel_id=channel_id)
            return "\n\n".join([answer.content, *notices])
        raise InvalidInputError("質問か、テキスト資料・画像・音声ファイルを送ってください。")
