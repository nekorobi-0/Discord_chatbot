import json
from dataclasses import replace

import httpx

from ai_kun.tools import ToolHub


async def test_disabled_tool(settings):
    hub = ToolHub(settings.search)
    assert hub.specs() == []
    assert "error" in json.loads(await hub.execute("web_search", {"query": "test"}))


async def test_tavily_mock(settings, monkeypatch):
    calls = []

    def handler(request):
        assert request.url == "https://api.tavily.com/search"
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "結果", "url": "https://example.com", "content": "alice@example.com"},
                    {"url": "javascript:alert(1)", "content": "invalid"},
                ]
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    hub = ToolHub(replace(settings.search, enabled=True, api_key="fake-key"))
    result = json.loads(await hub.execute("web_search", {"query": "alice@example.com を検索"}))
    assert "alice@example.com" in str(calls)
    assert len(result["untrusted_data"]) == 1
    assert result["untrusted_data"][0]["content"] == "alice@example.com"


async def test_tavily_failure_and_invalid_arguments(settings, monkeypatch):
    def handler(request):
        return httpx.Response(429)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    hub = ToolHub(replace(settings.search, enabled=True, api_key="fake-key"))
    assert "error" in json.loads(await hub.execute("web_search", {"query": "test"}))
    assert "error" in json.loads(await hub.execute("web_search", {"query": 42}))
    assert "error" in json.loads(await hub.execute("unknown", {}))
