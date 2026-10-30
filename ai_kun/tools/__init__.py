"""読み取り専用のWeb検索とDiscord MCPをLLMへ公開する。"""

import json
import logging
from collections.abc import Mapping
from urllib.parse import urlparse

import httpx

from ..config import DiscordSettings, SearchSettings
from ..observability import metrics

logger = logging.getLogger(__name__)


class ToolHub:
    def __init__(self, search: SearchSettings, memory=None, *, discord: DiscordSettings | None = None):
        self.search = search
        self.discord_mcp = None
        if discord is not None and discord.bot_token and discord.guild_id:
            from ..discord_mcp import DiscordMCP

            self.discord_mcp = DiscordMCP(discord)

    def specs(self) -> list[dict]:
        tools = self.discord_mcp.specs() if self.discord_mcp else []
        if not self.search.enabled or not self.search.configured:
            return tools
        return tools + [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "公開Webを検索する。最新情報にはこの検索を使い、回答に出典URLを付ける。",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    async def execute(self, name: str, args: Mapping[str, object]) -> str:
        if self.discord_mcp and name in {spec["function"]["name"] for spec in self.discord_mcp.specs()}:
            try:
                result = await self.discord_mcp.execute(name, dict(args))
                metrics.increment("ai_kun.discord_mcp_calls_total")
                return result
            except Exception as exc:
                logger.warning("Discord MCP call failed: %s", type(exc).__name__)
                metrics.increment("ai_kun.discord_mcp_failures_total")
                return '{"error":"Discordの読み取りが応答しませんでした。取得できたと断言しないでください。"}'
        if name != "web_search" or not (self.search.enabled and self.search.configured):
            return json.dumps({"error": "そのツールは今は使えません。"}, ensure_ascii=False)
        query = args.get("query")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            return '{"error":"検索語は1〜500文字で指定してください。"}'
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {self.search.api_key}"},
                    json={
                        "query": query,
                        "max_results": self.search.max_results,
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
            results = []
            for item in data.get("results", [])[: self.search.max_results]:
                url = str(item.get("url", ""))
                if urlparse(url).scheme not in {"http", "https"}:
                    continue
                results.append(
                    {
                        "title": str(item.get("title", ""))[:200],
                        "url": url,
                        "content": str(item.get("content", ""))[:1500],
                    }
                )
            metrics.increment("ai_kun.web_searches_total")
            return json.dumps({"untrusted_data": results}, ensure_ascii=False)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            metrics.increment("ai_kun.web_search_failures_total")
            return '{"error":"Web検索が応答しませんでした。検索できたと断言しないでください。"}'
