import base64
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from ai_kun.adapters.discord_bot import AIKunClient, allowed, split_message
from ai_kun.chat import ChatService
from ai_kun.http_server import create_app


def test_channel_and_guild_filter(settings):
    assert allowed(settings, 23456, 34567)
    assert not allowed(settings, 23456, 45678)
    assert not allowed(settings, 99999, 34567)
    assert not allowed(settings, None, 34567)
    assert not allowed(settings, 23456, None)


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
        (False, 23456, 45678, "!ai hello"),
        (False, None, 34567, "!ai hello"),
        (False, 23456, 34567, "普通の会話"),
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


async def test_http_auth_and_health(settings):
    operational = replace(settings.operational, dashboard_username="admin", dashboard_password="secret")
    app = create_app(replace(settings, operational=operational), SimpleNamespace(is_ready=lambda: True))
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 200
        assert (await client.get("/metrics/v1")).status == 401
        assert (
            await client.get(
                "/dashboard", headers={"Authorization": "Basic " + base64.b64encode(b"admin:wrong").decode()}
            )
        ).status == 401
        response = await client.get(
            "/metrics/v1", headers={"Authorization": "Basic " + base64.b64encode(b"admin:secret").decode()}
        )
        assert response.status == 200
        assert set(await response.json()) == {"counters"}
        assert (
            await client.get(
                "/dashboard", headers={"Authorization": "Basic " + base64.b64encode(b"admin:secret").decode()}
            )
        ).status == 200


async def test_http_unconfigured_and_disconnected(settings):
    app = create_app(settings, SimpleNamespace(is_ready=lambda: False))
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 503
        assert (await client.get("/metrics/v1")).status == 404
