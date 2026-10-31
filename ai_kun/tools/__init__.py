"""読み取り専用のWeb MCPとDiscord MCPをLLMへ公開する。"""

import json
import logging
from collections.abc import Mapping

from ..config import DiscordSettings, SearchSettings
from ..observability import metrics
from ..web_mcp import WebMCP

logger = logging.getLogger(__name__)


class ToolHub:
    def __init__(self, search: SearchSettings, memory=None, *, discord: DiscordSettings | None = None):
        self.search = search
        self.web_mcp = WebMCP(search)
        self.discord_mcp = None
        if discord is not None and discord.bot_token and discord.guild_id:
            from ..discord_mcp import DiscordMCP

            self.discord_mcp = DiscordMCP(discord, memory=memory)

    def specs(self) -> list[dict]:
        return (self.discord_mcp.specs() if self.discord_mcp else []) + self.web_mcp.specs()

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
        if name in {spec["function"]["name"] for spec in self.web_mcp.specs()}:
            try:
                result = await self.web_mcp.execute(name, dict(args))
                failed = "error" in json.loads(result)
                metric = (
                    ("ai_kun.web_search_failures_total" if failed else "ai_kun.web_searches_total")
                    if name == "web_search"
                    else ("ai_kun.web_fetch_failures_total" if failed else "ai_kun.web_fetches_total")
                )
                metrics.increment(metric)
                return result
            except Exception as exc:
                logger.warning("Web MCP call failed: tool=%s error=%s", name, type(exc).__name__)
                return '{"error":"Web取得が応答しませんでした。取得できたと断言しないでください。"}'
        return json.dumps({"error": "そのツールは今は使えません。"}, ensure_ascii=False)
