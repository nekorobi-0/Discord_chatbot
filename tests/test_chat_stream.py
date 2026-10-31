import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_kun.errors import RateLimitError, UpstreamError


def event(delta=None, *, finish=None, usage=None):
    frame = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        frame["usage"] = usage
    return ("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode()


class Events(httpx.AsyncByteStream):
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for item in self.events:
            if isinstance(item, float):
                await asyncio.sleep(item)
            else:
                yield item

    async def aclose(self):
        self.closed = True


def transport(monkeypatch, handler):
    original = httpx.AsyncClient

    class MockClient(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)


async def test_sse_content_reasoning_usage_and_utf8_byte_boundaries(store, monkeypatch):
    encoded = event({"content": "こんにちは。"})
    stream = Events(
        [
            event({"role": "assistant", "content": ""}),
            event({"reasoning_content": "PRIVATE_REASONING"}),
            encoded[:47],
            encoded[47:50],
            encoded[50:],
            event(finish="stop", usage={"prompt_tokens": 12, "completion_tokens": 5}),
            b"data: [DONE]\n\n",
        ]
    )

    def handle(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, stream=stream)

    transport(monkeypatch, handle)
    updates = AsyncMock()
    turn = await store.complete_chat([{"role": "user", "content": "hi"}], on_text=updates)
    assert turn.content == "こんにちは。"
    assert turn.prompt_tokens == 12 and turn.completion_tokens == 5
    updates.assert_awaited_once_with("こんにちは。")
    assert stream.closed


async def test_interleaved_fragmented_tool_calls_are_assembled_without_publishing_arguments(store, monkeypatch):
    stream = Events(
        [
            event(
                {
                    "tool_calls": [
                        {"index": 0, "id": "call-a", "function": {"name": "discord_search_", "arguments": '{"query":'}}
                    ]
                }
            ),
            event(
                {
                    "tool_calls": [
                        {"index": 1, "id": "call-b", "function": {"name": "discord_list_channels", "arguments": "{}"}}
                    ]
                }
            ),
            event(
                {"tool_calls": [{"index": 0, "function": {"name": "memory", "arguments": '"締切","user_id":"12345"}'}}]}
            ),
            event(finish="tool_calls"),
            b"data: [DONE]\n\n",
        ]
    )
    transport(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    updates = AsyncMock()
    turn = await store.complete_chat([{"role": "user", "content": "search"}], on_text=updates)
    assert [call.name for call in turn.tool_calls] == ["discord_search_memory", "discord_list_channels"]
    assert json.loads(turn.tool_calls[0].arguments) == {"query": "締切", "user_id": "12345"}
    updates.assert_not_awaited()


async def test_first_delta_can_take_longer_than_idle_timeout_and_active_stream_can_exceed_it(store, monkeypatch):
    store._llm = replace(
        store._llm, timeout_seconds=0.025, first_token_timeout_seconds=0.15, stream_timeout_seconds=0.4
    )
    stream = Events(
        [
            event({"role": "assistant", "content": ""}),
            0.04,
            b": keepalive\n\n",
            0.02,
            event({"content": "first "}),
            0.015,
            event({"content": "second "}),
            0.015,
            event({"content": "third."}),
            event(finish="stop"),
            b"data: [DONE]\n\n",
        ]
    )
    transport(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    updates = AsyncMock()
    turn = await store.complete_chat([{"role": "user", "content": "hi"}], on_text=updates)
    assert turn.content == "first second third."
    assert updates.await_count == 3 and stream.closed


@pytest.mark.parametrize("mode", ["first", "idle", "total"])
async def test_first_idle_and_total_timeouts_close_http_stream(store, monkeypatch, mode):
    store._llm = replace(
        store._llm, timeout_seconds=0.03, first_token_timeout_seconds=0.03, stream_timeout_seconds=0.06
    )
    if mode == "first":
        events = [b": heartbeat\n\n", 0.02] * 8
    elif mode == "idle":
        events = [event({"content": "first "}), 0.2]
    else:
        store._llm = replace(store._llm, timeout_seconds=0.1)
        events = [event({"content": "word "}), 0.015] * 8
    stream = Events(events)
    transport(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    with pytest.raises(UpstreamError, match="タイムアウト"):
        await store.complete_chat([{"role": "user", "content": "hi"}], on_text=AsyncMock())
    assert stream.closed


async def test_truncated_stream_reports_failure_after_retaining_received_text(store, monkeypatch):
    stream = Events([event({"content": "途中まで受信。"})])
    transport(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    updates = AsyncMock()
    with pytest.raises(UpstreamError, match="途中で切断"):
        await store.complete_chat([{"role": "user", "content": "hi"}], on_text=updates)
    updates.assert_awaited_once_with("途中まで受信。")
    assert stream.closed


@pytest.mark.parametrize("finish,args", [("length", '{"query":"test"}'), ("tool_calls", '{"query":')])
async def test_incomplete_tool_call_is_never_returned_as_executable(store, monkeypatch, finish, args):
    stream = Events(
        [
            event(
                {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "function": {"name": "discord_search_memory", "arguments": args}}
                    ]
                }
            ),
            event(finish=finish),
            b"data: [DONE]\n\n",
        ]
    )
    transport(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    with pytest.raises(UpstreamError):
        await store.complete_chat([{"role": "user", "content": "hi"}], on_text=AsyncMock())
    assert stream.closed


@pytest.mark.parametrize("status", [400, 429, 500])
async def test_stream_http_errors_are_safe_and_not_retried(store, monkeypatch, status, caplog):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "PRIVATE_REQUEST_ECHO test-key"})

    transport(monkeypatch, handle)
    with pytest.raises(RateLimitError if status == 429 else UpstreamError):
        await store.complete_chat([{"role": "user", "content": "hi"}], on_text=AsyncMock())
    assert len(requests) == 1
    assert "PRIVATE_REQUEST_ECHO" not in caplog.text and "test-key" not in caplog.text


async def test_json_response_from_server_ignoring_stream_is_accepted(store, monkeypatch):
    transport(
        monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "JSON reply"}}]})
    )
    updates = AsyncMock()
    assert (await store.complete_chat([{"role": "user", "content": "hi"}], on_text=updates)).content == "JSON reply"
    updates.assert_awaited_once_with("JSON reply")


async def test_stream_can_be_disabled_for_legacy_backends(store, monkeypatch):
    store._llm = replace(store._llm, stream_enabled=False)
    request = AsyncMock(return_value=httpx.Response(200, json={"choices": [{"message": {"content": "reply"}}]}))
    monkeypatch.setattr(store, "_request", request)
    assert (await store.complete_chat([{"role": "user", "content": "hi"}], on_text=AsyncMock())).content == "reply"
    assert "stream" not in request.await_args.kwargs["json"]


async def test_stream_keeps_endpoint_guard_and_does_not_follow_redirects(store, monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "http://8.8.8.8/private"}, content=b"not JSON")

    transport(monkeypatch, handle)
    with pytest.raises(UpstreamError):
        await store.complete_chat([{"role": "user", "content": "hi"}], on_text=AsyncMock())
    assert len(requests) == 1
    requests.clear()
    store._llm = replace(store._llm, base_url="http://8.8.8.8/v1")
    with pytest.raises(UpstreamError):
        await store.complete_chat([{"role": "user", "content": "hi"}], on_text=AsyncMock())
    assert requests == []
