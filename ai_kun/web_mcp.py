"""Exa検索・URL本文取得の読み取り専用MCP。Bot内とstdioの両方で利用できる。"""

import asyncio
import ipaddress
import json
import logging
import time
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from .config import SearchSettings
from .mcp_session import memory_session

logger = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 25


class WebToolError(ValueError):
    """外部応答を含めず、そのまま返せる固定の検証メッセージ。"""


def public_url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or any(ord(c) < 33 for c in value):
        raise WebToolError("公開WebのHTTP/HTTPS URLを指定してください。")
    parsed = urlparse(value)
    host = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 80, 443)
        or "\\" in value
        or "%" in host
    ):
        raise WebToolError("公開WebのHTTP/HTTPS URLを指定してください。")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if (
            "." not in host
            or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid"))
            or host == "localhost"
        ):
            raise WebToolError("公開WebのURLを指定してください。") from None
        if not all(part and all(c.isalnum() or c == "-" for c in part) for part in host.split(".")):
            raise WebToolError("URLのホスト名が不正です。") from None
    else:
        if not address.is_global:
            raise WebToolError("公開WebのURLを指定してください。")
    return value


def available_tools(settings):
    tools = []
    if settings.enabled and settings.configured:
        tools.append(
            Tool(
                name="web_search",
                description="公開Webを検索する。Exaを優先し、最新情報を調べる。取得した出典URLを回答に付ける。",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 500}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        )
    if settings.fetch_enabled and settings.exa_api_key:
        tools.append(
            Tool(
                name="web_fetch",
                description="既知の公開HTTP/HTTPS URLの本文をExaで取得する。検索結果を詳しく読むときにも使う。抽出本文は長さ上限があり、未取得部分を読んだと主張しない。",
                inputSchema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "minLength": 1, "maxLength": 2048}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            )
        )
    for tool in tools:
        tool.annotations = ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
        )
    return tools


class WebReader:
    def __init__(self, settings, *, transport=None):
        self.settings = settings
        self.transport = transport

    async def request(self, endpoint, headers, payload):
        options = {"transport": self.transport} if self.transport is not None else {}
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=20, follow_redirects=False, **options) as client:
                async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    if int(response.headers.get("content-length") or 0) > MAX_RESPONSE_BYTES:
                        raise WebToolError("検索サービスの応答が大きすぎます。")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise WebToolError("検索サービスの応答が大きすぎます。")
                        body.extend(chunk)
        data = json.loads(body)
        if not isinstance(data, dict) or data.get("error") or not isinstance(data.get("results"), list):
            raise WebToolError("検索サービスから正しい結果を取得できませんでした。")
        return data

    async def read(self, name, args):
        if name not in {tool.name for tool in available_tools(self.settings)}:
            raise WebToolError("そのWebツールは今は使えません。EXA_API_KEYと有効設定を確認してください。")
        if name == "web_search":
            if (
                set(args) != {"query"}
                or not isinstance(args.get("query"), str)
                or not 1 <= len(args["query"].strip()) <= 500
            ):
                raise WebToolError("検索語は1〜500文字で指定してください。")
            query = args["query"].strip()
            if self.settings.exa_api_key:
                provider = "exa"
                data = await self.request(
                    "https://api.exa.ai/search",
                    {"x-api-key": self.settings.exa_api_key},
                    {
                        "query": query,
                        "type": "auto",
                        "numResults": self.settings.max_results,
                        "contents": {"highlights": {"maxCharacters": 1500}},
                    },
                )
            else:
                provider = "tavily"
                data = await self.request(
                    "https://api.tavily.com/search",
                    {"Authorization": f"Bearer {self.settings.api_key}"},
                    {
                        "query": query,
                        "max_results": self.settings.max_results,
                        "include_raw_content": False,
                    },
                )
            limit, count = 1500, self.settings.max_results
        else:
            if set(args) != {"url"}:
                raise WebToolError("web_fetchにはurlだけを指定してください。")
            url = public_url(args.get("url"))
            provider = "exa"
            limit, count = self.settings.fetch_max_chars, 1
            data = await self.request(
                "https://api.exa.ai/contents",
                {"x-api-key": self.settings.exa_api_key},
                {
                    "urls": [url],
                    "text": {"maxCharacters": limit + 1},
                    "maxAgeHours": 0,
                    "livecrawlTimeout": 10000,
                },
            )
        results = []
        for item in data["results"][:count]:
            if not isinstance(item, dict):
                continue
            try:
                result_url = public_url(item.get("url"))
            except ValueError:
                continue
            highlights = item.get("highlights")
            text = (
                "\n".join(h for h in highlights[:10] if isinstance(h, str))
                if isinstance(highlights, list) and name == "web_search"
                else ""
            )
            text = text or item.get("text") or item.get("content") or ""
            if not isinstance(text, str):
                continue
            if name == "web_fetch" and not text.strip():
                continue
            results.append(
                {
                    "title": str(item.get("title") or "")[:200],
                    "url": result_url,
                    "content": text[:limit],
                    "truncated": len(text) > limit,
                    "published_date": str(item.get("publishedDate") or "")[:40],
                }
            )
        if name == "web_fetch" and not results:
            raise WebToolError("ページ本文を取得できませんでした。取得できたと断言しないでください。")
        return {"provider": provider, "untrusted_data": results}


def create_server(reader):
    server = Server("ai-kun-web-readonly")

    @server.list_tools()
    async def list_tools():
        return available_tools(reader.settings)

    @server.call_tool()
    async def call_tool(name, arguments):
        try:
            started = time.monotonic()
            data = await reader.read(name, arguments)
            logger.info(
                "Web MCP %s completed: provider=%s results=%d seconds=%.3f",
                name,
                data["provider"],
                len(data["untrusted_data"]),
                time.monotonic() - started,
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False))], structuredContent=data
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("Web MCP request failed: tool=%s status=%s", name, status)
            message = (
                "APIキーの認証に失敗しました。EXA_API_KEY等の設定を確認してください。"
                if status in {401, 403}
                else "検索サービスの利用枠・残高を確認してください。"
                if status == 402
                else "検索サービスが混雑しています。少し待ってください。"
                if status == 429
                else "Web取得サービスが応答しませんでした。取得できたと断言しないでください。"
            )
        except WebToolError as exc:
            message = str(exc)
        except (ValueError, TypeError, AttributeError):
            message = "Webの結果を読み取れませんでした。"
        except (httpx.HTTPError, TimeoutError, OSError):
            message = "Web取得がタイムアウトまたは失敗しました。取得できたと断言しないでください。"
        except Exception as exc:
            logger.warning("Web MCP failed: tool=%s error=%s", name, type(exc).__name__)
            message = "Web取得に失敗しました。取得できたと断言しないでください。"
        return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])

    return server


class WebMCP:
    def __init__(self, settings, *, reader=None):
        self.settings = settings
        self.server = create_server(reader or WebReader(settings))

    def specs(self):
        return [
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
            }
            for tool in available_tools(self.settings)
        ]

    async def execute(self, name, arguments):
        async with memory_session(self.server) as session:
            result = await session.call_tool(name, arguments)
        if result.isError:
            return json.dumps(
                {"error": " ".join(item.text for item in result.content if item.type == "text")}, ensure_ascii=False
            )
        return json.dumps(result.structuredContent or {"untrusted_data": []}, ensure_ascii=False)


async def serve_stdio(settings):
    server = create_server(WebReader(settings))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    load_dotenv(override=False)
    asyncio.run(serve_stdio(SearchSettings.from_env()))


if __name__ == "__main__":
    main()
