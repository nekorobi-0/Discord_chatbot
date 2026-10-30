"""許可したギルド・チャンネルだけで動く Discord アダプター。"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import discord
from discord import app_commands

from ..chat import ChatService
from ..config import Settings
from ..errors import AIKunError, InvalidInputError
from ..observability import metrics

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
        if isinstance(error, app_commands.CommandOnCooldown):
            message = "少し待ってからもう一回試してください。"
        elif isinstance(error, app_commands.MissingPermissions):
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
        self.busy = asyncio.Lock()
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
        if self.busy.locked():
            return "いま別の返事を考えています。少し待ってから試してください。"
        async with self.busy:
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

    async def _answer(self, interaction: discord.Interaction, operation, *, ephemeral=False) -> None:
        await interaction.response.defer(thinking=True, ephemeral=ephemeral)
        result = await self._run(operation)
        for chunk in split_message(result):
            await interaction.followup.send(chunk, ephemeral=ephemeral, allowed_mentions=discord.AllowedMentions.none())

    def _register_commands(self) -> None:
        @self.tree.command(name="ask", description="AIくんに質問する（会話は自動保存しません）")
        @app_commands.describe(question="質問内容")
        @app_commands.checks.cooldown(1, 5, key=lambda i: i.user.id)
        async def ask(interaction: discord.Interaction, question: str):
            async def operation():
                return (await self.service.generate_reply(question)).content

            await self._answer(interaction, operation)

        @self.tree.command(name="search_memory", description="保存記憶を検索して管理者にだけ表示する")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        @app_commands.checks.cooldown(1, 5, key=lambda i: i.user.id)
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
        @app_commands.checks.cooldown(1, 10, key=lambda i: i.user.id)
        async def remember(interaction: discord.Interaction, text: str):
            async def operation():
                result = await self.service.remember(text)
                return f"保存しました。文書ID: `{result['document_id']}`（{result['chunks']}チャンク）"

            await self._answer(interaction, operation, ephemeral=True)

        @self.tree.command(name="ingest", description="テキスト資料をサーバー共通の記憶へ登録する（管理者）")
        @app_commands.default_permissions(manage_guild=True)
        @app_commands.checks.has_permissions(manage_guild=True)
        async def ingest(interaction: discord.Interaction, file: discord.Attachment):
            async def operation():
                if file.size > MAX_ATTACHMENT_BYTES:
                    raise InvalidInputError("添付は8 MiBまでです。")
                result = await self.service.ingest_document(file.filename, await file.read())
                return f"登録しました。文書ID: `{result['document_id']}`（{result['chunks']}チャンク）"

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
        if message.author.bot or not allowed(self.settings, message.guild.id if message.guild else None):
            return

        # 全メッセージ収集：フック（メンション・!ai・/ask）の有無や内容に関係なく、
        # サーバー内の人間の発言は即座にRAG記憶へ取り込む。失敗しても返信処理は続行。
        try:
            await self.service.ingest_message(message)
        except Exception as exc:  # 取り込み失敗を返信やログには出す。本文は出さない。
            logger.warning(
                "live history ingest failed: %s (%s)",
                type(exc).__name__,
                exc.user_message() if isinstance(exc, AIKunError) else "詳細非表示",
            )

        # 全自動バックフィル：観測したチャンネルの未保存分を、保存済みカーソルから
        # 古い順に巻き戻して即取り込みする（フック有無とは無関係・常時）。
        try:
            result = await self.service.backfill_channel(message.channel.id)
            if result["imported"]:
                logger.info(
                    "auto backfill filled gaps channel=%s scanned=%d imported=%d documents=%d",
                    message.channel.id,
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

        content = message.content.strip()
        prefixes = (f"<@{self.user.id}>", f"<@!{self.user.id}>") if self.user else ()
        if content == "!ai" or content.startswith("!ai "):
            content = content[3:].strip()
        elif content.startswith(prefixes):
            for prefix in prefixes:
                content = content.removeprefix(prefix).strip()
        else:
            return

        async def operation():
            text = content
            if message.attachments:
                if len(message.attachments) != 1:
                    raise InvalidInputError("音声添付は1つずつ送ってください。資料は /ingest を使ってください。")
                attachment = message.attachments[0]
                if attachment.size > MAX_ATTACHMENT_BYTES:
                    raise InvalidInputError("音声添付は8 MiBまでです。")
                transcript = await self.service.transcribe_attachment(await attachment.read(), attachment.filename)
                text = f"{text}\n{transcript}".strip()
            return (await self.service.generate_reply(text)).content

        async with message.channel.typing():
            result = await self._run(operation)
        for chunk in split_message(result):
            await message.reply(chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
