"""読み取り専用 Web 検索。LLM に任意URL取得・記憶変更は許さない。"""

import json
from collections.abc import Mapping
from urllib.parse import urlparse

import httpx

from ..config import SearchSettings
from ..observability import metrics


class ToolHub:
    def __init__(self, search: SearchSettings, memory=None):
        self.search = search

    def specs(self) -> list[dict]:
        if not self.search.enabled or not self.search.configured:
            return []
        return [
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
        if name != "web_search" or not self.specs():
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
