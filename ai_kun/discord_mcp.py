"""設定サーバーのDiscord APIを読むMCPサーバー。Bot内はメモリ転送、外部クライアントはstdio。"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, ImageContent, TextContent, Tool, ToolAnnotations

from .config import DiscordSettings
from .db import init_db
from .discord_users import message_users, remember_users, resolve_users, search_users
from .errors import AIKunError, InvalidInputError
from .history import DiscordHistoryAPI, shared_limiter
from .images import IMAGE_EXTENSIONS, image_data_url
from .tool_result import ToolResult

logger = logging.getLogger(__name__)
_ID = {"type": "string", "pattern": "^[0-9]{5,25}$"}
_MAX_RESULT_CHARS = 24_000
DISCORD_TOOLS = [
    Tool(
        name="discord_search_memory",
        description=(
            "保存済みDiscord発言をRAG検索する。過去の話題・人物の発言・特定チャンネルの話はまずこのツールで調べる。"
            "user_idとchannel_idは任意のAND条件。名前はdiscord_search_usersでIDを確認し、推測しない。"
            "ユーザー指定時はその人の本文だけを返す。保存済み範囲だけの検索であり、未保存分はDiscord読み取りツールで補う。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "user_id": _ID,
                "channel_id": _ID,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 24, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="discord_resolve_users",
        description="設定サーバーでキャッシュ済みのユーザーIDから表示名を引く。ネットワーク取得しない。IDやmember-<ID>は回答に出さず名前を使う。未解決なら名前を推測しない。",
        inputSchema={
            "type": "object",
            "properties": {
                "user_ids": {"type": "array", "items": _ID, "minItems": 1, "maxItems": 100},
            },
            "required": ["user_ids"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="discord_search_users",
        description="設定サーバーのキャッシュ内で名前からユーザー候補をtop-k検索する。NFKC・大文字小文字を正規化し、長さの差も含むハミング距離で並べる。あいまいな候補を本人だと断定しない。返すIDは内部照合専用で回答には表示名を使う。",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 100},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
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
    def __init__(self, settings: DiscordSettings, *, api=None, memory=None, image_transport=None, clock=time.monotonic):
        init_db()
        self.settings = settings
        self.api = api
        self.memory = memory
        self.image_transport = image_transport
        self.clock = clock
        self._channels = {}

    @asynccontextmanager
    async def connection(self):
        if self.api is not None:
            yield self.api
        else:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                yield DiscordHistoryAPI(self.settings.bot_token, client, shared_limiter)

    async def channel(self, api, channel_id):
        cached = self._channels.get(channel_id)
        if cached is not None and cached[0] > self.clock():
            return cached[1]
        channel = await api.get(channel_id)
        if str(channel.get("guild_id")) != str(self.settings.guild_id):
            raise InvalidInputError("設定されたサーバー以外のチャンネルやDMは読み取れません。")
        # 所属情報だけを60秒再利用。本文・権限エラーはキャッシュしない。
        if len(self._channels) >= 1024:
            self._channels.clear()
        self._channels[channel_id] = (self.clock() + 60, channel)
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
                    "url": str(item.get("url") or ""),
                }
                for item in (data.get("attachments") or [])[:10]
            ],
            "url": f"https://discord.com/channels/{self.settings.guild_id}/{channel_id}/{message_id}",
        }

    async def image_contents(self, data: dict) -> list:
        """所属確認済みメッセージの画像のみ。失敗しても本文は返す。"""
        messages = data.get("messages", [data["message"]] if "message" in data else [])
        selected = []
        for message in messages:
            for attachment in message.get("attachments", []):
                if PurePosixPath(attachment["filename"]).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if len(selected) >= 4:
                    attachment["image_status"] = "omitted: maximum 4 images per call"
                    continue
                selected.append((message["id"], attachment))
        if not selected:
            return []
        async with httpx.AsyncClient(timeout=8, follow_redirects=False, transport=self.image_transport) as client:

            async def fetch(message_id, attachment):
                parsed = urlparse(attachment["url"])
                if (
                    parsed.scheme != "https"
                    or parsed.hostname not in {"cdn.discordapp.com", "media.discordapp.net"}
                    or parsed.port not in (None, 443)
                    or parsed.username is not None
                    or not parsed.path.startswith("/attachments/")
                ):
                    attachment["image_status"] = "unavailable: invalid CDN URL"
                    return []
                try:
                    if int(attachment.get("size") or 0) > 8 * 1024 * 1024:
                        raise InvalidInputError("画像は8MiBまで")
                    async with client.stream("GET", attachment["url"]) as response:
                        response.raise_for_status()
                        blob = bytearray()
                        async for chunk in response.aiter_bytes():
                            blob.extend(chunk)
                            if len(blob) > 8 * 1024 * 1024:
                                raise InvalidInputError("画像は8MiBまで")
                    uri = image_data_url(bytes(blob))
                    header, encoded = uri.split(",", 1)
                    attachment["image_status"] = "included"
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "message_id": message_id,
                                    "filename": attachment["filename"],
                                    "source": "untrusted Discord attachment",
                                },
                                ensure_ascii=False,
                            ),
                        ),
                        ImageContent(type="image", mimeType=header[5:].split(";")[0], data=encoded),
                    ]
                except (httpx.HTTPError, AIKunError, ValueError):
                    attachment["image_status"] = "unavailable: download or image format error"
                    return []

            async def bounded_fetch(message_id, attachment):
                try:
                    return await asyncio.wait_for(fetch(message_id, attachment), timeout=10)
                except (TimeoutError, ValueError):
                    attachment["image_status"] = "unavailable: timeout or invalid URL"
                    return []

            results = await asyncio.gather(*(bounded_fetch(*item) for item in selected))
        return [part for result in results for part in result]

    async def read(self, name: str, arguments: dict) -> dict:
        if name not in {tool.name for tool in DISCORD_TOOLS}:
            raise InvalidInputError("このMCPはDiscordの読み取り専用です。その操作はありません。")
        if name == "discord_search_memory":
            query, top_k = arguments.get("query"), arguments.get("top_k", 5)
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
                raise InvalidInputError("検索語は1〜500文字で指定してください。")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 24:
                raise InvalidInputError("top_kは1〜24で指定してください。")
            filters = {key: snowflake(arguments[key]) for key in ("user_id", "channel_id") if key in arguments}
            if self.memory is None:
                # stdioでもEmbedding設定だけでDBを検索できる。Chat APIは使わない。
                from .config import ASRSettings, EmbeddingSettings, LLMSettings, MemorySettings
                from .memory import MemoryStore

                embedding = EmbeddingSettings.from_env()
                self.memory = MemoryStore(
                    embedding,
                    LLMSettings(base_url=embedding.base_url, model=embedding.model),
                    ASRSettings(),
                    MemorySettings.from_env(),
                )
            hits = await self.memory.search(query, top_k=top_k, guild_id=self.settings.guild_id, **filters)
            results = []
            remaining = 20_000
            for hit in hits:
                excerpt = hit.text[: min(remaining, 8000)]
                if not excerpt:
                    break
                results.append({"label": hit.label, "text": excerpt, "truncated": excerpt != hit.text})
                remaining -= len(excerpt)
            return {
                "source": "saved_discord_memory",
                "filters": filters,
                "returned_count": len(results),
                "results": results,
            }
        if name == "discord_resolve_users":
            values = arguments.get("user_ids")
            if not isinstance(values, list) or not 1 <= len(values) <= 100:
                raise InvalidInputError("user_idsは1〜100件で指定してください。")
            return await asyncio.to_thread(
                resolve_users, self.settings.guild_id, [snowflake(value) for value in values]
            )
        if name == "discord_search_users":
            query, top_k = arguments.get("query"), arguments.get("top_k", 5)
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 100:
                raise InvalidInputError("名前は1〜100文字で指定してください。")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
                raise InvalidInputError("top_kは1〜20で指定してください。")
            return await asyncio.to_thread(search_users, self.settings.guild_id, query, top_k)
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
                remember_users(self.settings.guild_id, message_users([data]))
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
            remember_users(self.settings.guild_id, message_users(data[:limit]))
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
            started = time.monotonic()
            data = await reader.read(name, arguments)
            read_seconds = time.monotonic() - started
            images = await reader.image_contents(data)
            logger.info(
                "Discord MCP %s: read_seconds=%.3f image_seconds=%.3f images=%d",
                name,
                read_seconds,
                time.monotonic() - started - read_seconds,
                sum(part.type == "image" for part in images),
            )
            structured = {"untrusted_data": data}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(structured, ensure_ascii=False)), *images],
                structuredContent=structured,
            )
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
    def __init__(self, settings: DiscordSettings, *, reader: DiscordReader | None = None, memory=None):
        self.server = create_server(reader or DiscordReader(settings, memory=memory))

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
        parts = []
        for item in result.content[1:]:
            if item.type == "text":
                parts.append({"type": "text", "text": item.text})
            elif item.type == "image":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{item.mimeType};base64,{item.data}",
                            "detail": "low",
                        },
                    }
                )
        return ToolResult(json.dumps(result.structuredContent or {"untrusted_data": []}, ensure_ascii=False), parts)


async def serve_stdio(settings: DiscordSettings):
    server = create_server(DiscordReader(settings))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    load_dotenv(override=False)
    asyncio.run(serve_stdio(DiscordSettings.from_env()))


if __name__ == "__main__":
    main()
