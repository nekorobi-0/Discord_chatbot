import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.chat import ChatService
from ai_kun.errors import UpstreamError
from ai_kun.request_diagnostics import current_trace, phase
from tests.test_adapters import Typing, file_message


@pytest.fixture
def client(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    return AIKunClient(settings, service)


async def test_only_running_request_types_and_timeout_does_not_block_next(client, monkeypatch, caplog):
    monkeypatch.setattr("ai_kun.adapters.discord_bot.RESPONSE_TIMEOUT_SECONDS", 0.04)
    monkeypatch.setattr("ai_kun.adapters.discord_bot.REQUEST_HEARTBEAT_SECONDS", 0.01)
    calls = 0
    entered = asyncio.Event()
    stopped = []

    class TrackedTyping(Typing):
        async def __aenter__(self):
            entered.set()

        async def __aexit__(self, *args):
            stopped.append(True)

    async def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            phase("llm_http")
            await asyncio.Event().wait()
        return SimpleNamespace(content="次の要求は完了")

    monkeypatch.setattr(client.service, "generate_reply", generate)
    first, second = file_message("!ai first", []), file_message("!ai second", [])
    first.id, second.id = "9001", "9002"
    first.channel.typing, second.channel.typing = TrackedTyping, AsyncMock()
    # 2件目のtypingコンテキストは開始時だけ記録する。
    second_manager = Typing()
    second.channel.typing = lambda: second_manager
    second_enter = AsyncMock(wraps=second_manager.__aenter__)
    second_manager.__aenter__ = second_enter
    caplog.set_level(logging.INFO)
    task1 = asyncio.create_task(client.on_message(first))
    await entered.wait()
    task2 = asyncio.create_task(client.on_message(second))
    try:
        await asyncio.sleep(0.005)
        second_enter.assert_not_awaited()
        await asyncio.wait_for(asyncio.gather(task1, task2), 1)
        assert stopped == [True]
        assert "時間がかかりすぎた" in first.reply.await_args.args[0]
        assert second.reply.await_args.args[0] == "次の要求は完了"
        assert "request pending: request=9001 phase=llm_http" in caplog.text
        assert "request timed out: request=9001 phase=llm_http" in caplog.text
        assert "Discord reply completed: request=9002" in caplog.text
        assert current_trace.get() is None
    finally:
        await client.drain()
        await client.close()


async def test_broken_typing_does_not_prevent_answer(client, monkeypatch, caplog):
    monkeypatch.setattr("ai_kun.adapters.discord_bot.TYPING_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client.service, "generate_reply", AsyncMock(return_value=SimpleNamespace(content="回答")))

    class StuckTyping(Typing):
        async def __aenter__(self):
            await asyncio.Event().wait()

    message = file_message("!ai 質問", [])
    message.channel.typing = StuckTyping
    try:
        await asyncio.wait_for(client.on_message(message), 1)
        assert message.reply.await_args.args[0] == "回答"
        assert "typing unavailable: error=TimeoutError" in caplog.text
    finally:
        await client.drain()
        await client.close()


async def test_discord_reply_timeout_is_logged(client, monkeypatch, caplog):
    monkeypatch.setattr("ai_kun.adapters.discord_bot.DISCORD_IO_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client.service, "generate_reply", AsyncMock(return_value=SimpleNamespace(content="回答")))
    message = file_message("!ai 質問", [])
    message.reply = AsyncMock(side_effect=lambda *args, **kwargs: None)

    async def stuck(*args, **kwargs):
        await asyncio.Event().wait()

    message.reply.side_effect = stuck
    try:
        await asyncio.wait_for(client.on_message(message), 1)
        assert "Discord reply failed: request=9003 error=TimeoutError" in caplog.text
    finally:
        await client.drain()
        await client.close()


@pytest.mark.parametrize("service,path", [("LLM", "/chat/completions"), ("Embedding", "/embeddings")])
async def test_transport_failure_identifies_service_without_logging_secrets(store, monkeypatch, caplog, service, path):
    original = httpx.AsyncClient

    def handler(request):
        raise httpx.ConnectError("private upstream body SECRET", request=request)

    monkeypatch.setattr(
        "ai_kun.memory.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    base = store._llm.base_url if service == "LLM" else store._embedding.base_url
    with pytest.raises(UpstreamError) as caught:
        await store._request("POST", base + path)
    assert service in caught.value.user_message()
    assert f"service={service} error=ConnectError" in caplog.text
    assert "SECRET" not in caplog.text
    assert "test-key" not in caplog.text


async def test_dripping_http_response_has_total_deadline(store, monkeypatch, caplog):
    original = httpx.AsyncClient
    store._llm = replace(store._llm, timeout_seconds=0.025)
    closed = asyncio.Event()

    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b" "
                await asyncio.sleep(0.005)

        async def aclose(self):
            closed.set()

    monkeypatch.setattr(
        "ai_kun.memory.httpx.AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Drip())),
            **kwargs,
        ),
    )
    with pytest.raises(UpstreamError, match="LLM（応答タイムアウト）"):
        await asyncio.wait_for(store._request("POST", store._llm.base_url + "/chat/completions"), 1)
    assert closed.is_set()
    assert "service=LLM error=TimeoutError" in caplog.text
