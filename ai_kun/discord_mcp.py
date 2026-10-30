"""設定サーバーのDiscord APIを読むMCPサーバー。Bot内はメモリ転送、外部クライアントはstdio。"""

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from .config import DiscordSettings
from .errors import AIKunError, InvalidInputError
from .history import DiscordHistoryAPI, shared_limiter

logger = logging.getLogger(__name__)
_ID = {"type": "string", "pattern": "^[0-9]{5,25}$"}
_MAX_RESULT_CHARS = 24_000
DISCORD_TOOLS = [
    Tool(
        name="discord_list_channels",
        description="設定済みDiscordサーバーのチャンネル一覧を読む。チャンネル名からIDを調べるときに使う。",
        inputSchema={
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 40},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="discord_read_messages",
        description=(
            "設定済みサーバーのチャンネル発言を読む。既定は直近20件。"
            "古い発言はbefore_message_id、新しい発言はafter_message_idで続けて取得できる。"
            "取得した範囲だけに基づいて答える。URLは明示的に求められた場合だけ回答に出す。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": _ID,
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                "before_message_id": _ID,
                "after_message_id": _ID,
            },
            "required": ["channel_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="discord_get_message",
        description="設定済みサーバーの特定の発言をメッセージIDから読む。送信や編集は行わない。",
        inputSchema={
            "type": "object",
            "properties": {"channel_id": _ID, "message_id": _ID},
            "required": ["channel_id", "message_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="discord_list_active_threads",
        description="設定済みDiscordサーバーの、Botが参照できるアクティブなスレッド一覧を読む。",
        inputSchema={
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 40},
            },
            "additionalProperties": False,
        },
    ),
]
for _tool in DISCORD_TOOLS:
    if _tool.name == "discord_read_messages":
        _tool.inputSchema["not"] = {"required": ["before_message_id", "after_message_id"]}
    _tool.annotations = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    )


def snowflake(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{5,25}", value):
        raise InvalidInputError("Discord IDは5〜25桁の数字の文字列で指定してください。")
    return value


def paged_items(rows: list[dict], arguments: dict, key: str) -> dict:
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit", 40)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidInputError("offsetは0以上の整数で指定してください。")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise InvalidInputError("取得件数は1〜100件で指定してください。")
    selected = []
    for row in rows[offset : offset + limit]:
        if len(json.dumps(selected + [row], ensure_ascii=False)) > _MAX_RESULT_CHARS:
            break
        selected.append(row)
    next_offset = offset + len(selected)
    return {key: selected, "total": len(rows), "next_offset": next_offset if next_offset < len(rows) else None}


class DiscordReader:
    def __init__(self, settings: DiscordSettings, *, api=None):
        self.settings = settings
        self.api = api

    @asynccontextmanager
    async def connection(self):
        if self.api is not None:
            yield self.api
        else:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                yield DiscordHistoryAPI(self.settings.bot_token, client, shared_limiter)

    async def channel(self, api, channel_id):
        channel = await api.get(channel_id)
        if str(channel.get("guild_id")) != str(self.settings.guild_id):
            raise InvalidInputError("設定されたサーバー以外のチャンネルやDMは読み取れません。")
        return channel

    def message(self, data, channel_id):
        author = data.get("author") or {}
        message_id = str(data.get("id", ""))
        reference = data.get("message_reference") or {}
        return {
            "id": message_id,
            "channel_id": channel_id,
            "author": {
                "id": str(author.get("id", "")),
                "name": str(author.get("global_name") or author.get("username") or "")[:100],
                "bot": bool(author.get("bot")),
            },
            "content": str(data.get("content") or "")[:4000],
            "timestamp": data.get("timestamp"),
            "reply_to_message_id": reference.get("message_id"),
            "attachments": [
                {
                    "filename": str(item.get("filename", ""))[:256],
                    "size": item.get("size"),
                    "content_type": str(item.get("content_type") or "")[:100],
                }
                for item in (data.get("attachments") or [])[:10]
            ],
            "url": f"https://discord.com/channels/{self.settings.guild_id}/{channel_id}/{message_id}",
        }

    async def read(self, name: str, arguments: dict) -> dict:
        if name not in {tool.name for tool in DISCORD_TOOLS}:
            raise InvalidInputError("このMCPはDiscordの読み取り専用です。その操作はありません。")
        async with self.connection() as api:
            if name == "discord_list_channels":
                channels = await api.get(f"guilds/{self.settings.guild_id}/channels", expect_list=True)
                rows = [
                    {
                        "id": str(channel.get("id", "")),
                        "name": str(channel.get("name", ""))[:100],
                        "type": channel.get("type"),
                        "parent_id": channel.get("parent_id"),
                        "topic": str(channel.get("topic") or "")[:300],
                    }
                    for channel in channels
                ]
                return paged_items(rows, arguments, "channels")
            if name == "discord_list_active_threads":
                data = await api.get(f"guilds/{self.settings.guild_id}/threads/active")
                rows = [
                    {
                        "id": str(thread.get("id", "")),
                        "name": str(thread.get("name", ""))[:100],
                        "type": thread.get("type"),
                        "parent_id": thread.get("parent_id"),
                        "owner_id": thread.get("owner_id"),
                    }
                    for thread in data.get("threads", [])
                    if str(thread.get("guild_id")) == str(self.settings.guild_id)
                ]
                return paged_items(rows, arguments, "threads")
            channel_id = snowflake(arguments.get("channel_id"))
            await self.channel(api, channel_id)
            if name == "discord_get_message":
                message_id = snowflake(arguments.get("message_id"))
                data = await api.get(f"channels/{channel_id}/messages/{message_id}")
                return {"message": self.message(data, channel_id)}
            limit = arguments.get("limit", 20)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                raise InvalidInputError("取得件数は1〜100件で指定してください。")
            before = arguments.get("before_message_id")
            after = arguments.get("after_message_id")
            if before is not None and after is not None:
                raise InvalidInputError("before_message_idとafter_message_idは同時には指定できません。")
            if before is not None:
                before = snowflake(before)
            if after is not None:
                after = snowflake(after)
            data = await api.get(channel_id, messages=True, limit=limit, before=before, after=after)
            messages = []
            for item in data[:limit]:
                candidate = self.message(item, channel_id)
                if len(json.dumps(messages + [candidate], ensure_ascii=False)) > _MAX_RESULT_CHARS:
                    break
                messages.append(candidate)
            ids = [message["id"] for message in messages if message["id"].isdigit()]
            return {
                "channel_id": channel_id,
                "messages": messages,
                "returned_count": len(messages),
                "limited": len(messages) < len(data) or len(data) == limit,
                "next_before_message_id": min(ids, key=int) if ids else None,
                "newest_message_id": max(ids, key=int) if ids else None,
            }


def create_server(reader: DiscordReader) -> Server:
    server = Server("ai-kun-discord-read")

    @server.list_tools()
    async def list_tools():
        return DISCORD_TOOLS

    @server.call_tool()
    async def call_tool(name, arguments):
        try:
            data = await reader.read(name, arguments)
            return {"untrusted_data": data}
        except AIKunError as exc:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=exc.user_message())])
        except Exception as exc:
            logger.warning("Discord MCP read failed: %s", type(exc).__name__)
            return CallToolResult(
                isError=True, content=[TextContent(type="text", text="Discordの読み取りに失敗しました。")]
            )

    return server


@asynccontextmanager
async def memory_session(server: Server):
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        task = asyncio.create_task(server.run(*server_streams, server.create_initialization_options()))
        try:
            async with ClientSession(*client_streams, read_timeout_seconds=timedelta(seconds=60)) as session:
                await session.initialize()
                yield session
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class DiscordMCP:
    def __init__(self, settings: DiscordSettings, *, reader: DiscordReader | None = None):
        self.server = create_server(reader or DiscordReader(settings))

    def specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
            }
            for tool in DISCORD_TOOLS
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        async with memory_session(self.server) as session:
            result = await session.call_tool(name, arguments)
        if result.isError:
            return json.dumps(
                {"error": " ".join(item.text for item in result.content if item.type == "text")}, ensure_ascii=False
            )
        return json.dumps(result.structuredContent or {"untrusted_data": []}, ensure_ascii=False)


async def serve_stdio(settings: DiscordSettings):
    server = create_server(DiscordReader(settings))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    load_dotenv(override=False)
    asyncio.run(serve_stdio(DiscordSettings.from_env()))


if __name__ == "__main__":
    main()
