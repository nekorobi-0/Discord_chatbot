from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from ai_kun.adapters.discord_bot import AIKunClient, allowed, split_message
from ai_kun.chat import ChatService
from ai_kun.errors import UpstreamError
from ai_kun.http_server import create_app
from ai_kun.memory import ModelTurn


class Typing:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


def test_guild_filter(settings):
    # チャンネル単位の絞り込みは廃止済み。サーバー一致だけを確認する。
    assert allowed(settings, 23456)
    assert not allowed(settings, 99999)
    assert not allowed(settings, None)


def test_emoji_split():
    text = "こんにちは😀" * 1000
    parts = split_message(text)
    assert "".join(parts) == text
    assert all(len(p.encode("utf-16-le")) // 2 <= 1900 for p in parts)


async def test_commands_registered_and_disallowed_interaction(settings, store):
    client = AIKunClient(settings, ChatService(settings, store))
    assert {cmd.name for cmd in client.tree.get_commands()} == {
        "ask",
        "remember",
        "ingest",
        "forget",
        "search_memory",
    }
    interaction = SimpleNamespace(guild_id=99999, channel_id=34567, response=SimpleNamespace(send_message=AsyncMock()))
    assert not await client.tree.interaction_check(interaction)
    interaction.response.send_message.assert_awaited_once()
    await client.close()


@pytest.mark.parametrize(
    "bot,guild_id,channel_id,content",
    [
        (True, 23456, 34567, "!ai hello"),
        (False, 99999, 34567, "!ai hello"),
        (False, None, 34567, "!ai hello"),
    ],
)
async def test_ignored_messages(settings, store, monkeypatch, bot, guild_id, channel_id, content):
    service = ChatService(settings, store)
    reply = AsyncMock()
    monkeypatch.setattr(service, "generate_reply", reply)
    client = AIKunClient(settings, service)
    message = SimpleNamespace(
        author=SimpleNamespace(bot=bot),
        guild=SimpleNamespace(id=guild_id) if guild_id else None,
        channel=SimpleNamespace(id=channel_id),
        content=content,
    )
    await client.on_message(message)
    reply.assert_not_called()
    await client.close()


async def test_all_messages_ingested_without_hooks(settings, store, monkeypatch):
    # フックとは無関係な通常会話も、受け付けると同時にRAG記憶へ収集される。
    ingested = []

    async def fake_ingest(message_obj):
        ingested.append(str(message_obj.id))
        return {"scanned": 1, "imported": 1, "skipped": 0, "documents": 1}

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", fake_ingest)
    reply = AsyncMock()
    monkeypatch.setattr(service, "generate_reply", reply)
    client = AIKunClient(settings, service)
    message = SimpleNamespace(
        id="9001",
        author=SimpleNamespace(bot=False, id="111"),
        guild=SimpleNamespace(id=23456),
        channel=SimpleNamespace(id=45678, typing=Typing),
        content="ただの雑談（フックなし）",
        webhook_id=None,
        created_at=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
        attachments=[],
    )
    await client.on_message(message)
    assert ingested == ["9001"]
    # フックが無いので返信生成は走らない。
    reply.assert_not_called()
    await client.close()


async def test_ingest_failure_does_not_break_hook_reply(settings, store, monkeypatch):
    service = ChatService(settings, store)

    async def broken_ingest(message_obj):
        raise UpstreamError("Embedding")

    monkeypatch.setattr(service, "ingest_message", broken_ingest)

    async def complete(*args, **kwargs):
        return ModelTurn(content="こんにちは！")

    monkeypatch.setattr(store, "complete_chat", complete)
    client = AIKunClient(settings, service)
    reply = AsyncMock()
    message = SimpleNamespace(
        id="9002",
        author=SimpleNamespace(bot=False, id="111"),
        guild=SimpleNamespace(id=23456),
        channel=SimpleNamespace(id=45678, typing=Typing),
        content="!ai こんにちは",
        webhook_id=None,
        created_at=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
        attachments=[],
        reply=reply,
    )
    await client.on_message(message)
    # 取り込みは失敗しても、フックへの返信は通常どおり届く。
    reply.assert_awaited_once()
    assert "こんにちは" in reply.await_args.args[0]
    await client.close()


async def test_mention_and_prefix_reply(settings, store, monkeypatch):
    service = ChatService(settings, store)
    generate = AsyncMock(return_value=SimpleNamespace(content="こんにちは"))
    monkeypatch.setattr(service, "generate_reply", generate)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=12345)

    class Typing:
        async def __aenter__(self):
            pass

        async def __aexit__(self, *args):
            pass

    for content in ("!ai テスト", "<@12345> テスト", "<@!12345> テスト"):
        message = SimpleNamespace(
            author=SimpleNamespace(bot=False),
            guild=SimpleNamespace(id=23456),
            channel=SimpleNamespace(id=34567, typing=Typing),
            content=content,
            attachments=[],
            reply=AsyncMock(),
        )
        await client.on_message(message)
        generate.assert_awaited_with("テスト")
        message.reply.assert_awaited_once()
        assert message.reply.call_args.kwargs["allowed_mentions"].everyone is False
    await client.close()


async def test_http_dashboard_and_metrics_without_auth(settings):
    app = create_app(settings, SimpleNamespace(is_ready=lambda: True))
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 200
        for path in ("/", "/dashboard"):
            response = await client.get(path)
            assert response.status == 200
            assert "AIくん 運用情報" in await response.text()
        response = await client.get("/metrics/v1")
        assert response.status == 200
        assert set(await response.json()) == {"counters"}


async def test_http_unconfigured_and_disconnected(settings):
    app = create_app(settings, SimpleNamespace(is_ready=lambda: False))
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 503
        assert (await client.get("/metrics/v1")).status == 200
        assert (await client.get("/dashboard")).status == 200
