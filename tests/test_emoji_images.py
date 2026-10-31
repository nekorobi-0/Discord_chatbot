import asyncio
import base64
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_kun import emoji_images
from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.chat import ChatService
from ai_kun.discord_mcp import DiscordReader
from ai_kun.emoji_images import find_emojis, load_emoji_parts
from ai_kun.memory import ModelTurn
from tests.test_adapters import file_message
from tests.test_discord_mcp import FakeAPI
from tests.test_reply_chain import ancestor, reference
from tests.test_reply_images import PNG


@pytest.fixture(autouse=True)
def clear_image_cache():
    emoji_images._cache.clear()
    yield
    emoji_images._cache.clear()


def test_custom_and_unicode_emoji_are_extracted_as_whole_images():
    images = find_emojis("a <:happy:123456> <a:dance:987654> 👩🏽‍💻 🇯🇵 ❤️ 1️⃣ 🏳️‍🌈")
    assert len(images) == 7
    assert images[0].url == "https://cdn.discordapp.com/emojis/123456.png?size=128&quality=lossless"
    assert not images[0].animated and images[1].animated
    assert images[2].url.endswith("/1f469-1f3fd-200d-1f4bb.png")
    assert images[3].url.endswith("/1f1ef-1f1f5.png")
    assert images[4].url.endswith("/2764.png")
    assert images[5].url.endswith("/31-20e3.png")
    assert images[6].url.endswith("/1f3f3-fe0f-200d-1f308.png")
    assert find_emojis("abc 123 :happy: <@123456> https://example.com") == []


async def test_duplicate_images_cached_but_source_message_mapping_preserved():
    calls = []

    def download(request):
        calls.append(request)
        return httpx.Response(200, content=PNG)

    transport = httpx.MockTransport(download)
    parts = await load_emoji_parts(
        [("60002", "<a:dance:123456> 😀 😀"), ("60001", "<:other_name:123456> 😀")], transport=transport
    )
    assert len(calls) == 2
    images = [part for part in parts if part["type"] == "image_url"]
    assert len(images) == 2
    assert base64.b64decode(images[0]["image_url"]["url"].split(",")[1]) == PNG
    assert "60002" in parts[0]["text"] and "60001" in parts[0]["text"]
    assert "static frame" in parts[0]["text"]
    await load_emoji_parts([("60003", "😀 <:dance:123456>")], transport=transport)
    assert len(calls) == 2
    assert all("authorization" not in request.headers for request in calls)


@pytest.mark.parametrize("failure", ["404", "redirect", "bad_image", "oversized", "false_size", "timeout"])
async def test_download_failure_retains_text_without_image_and_is_cached(failure, monkeypatch):
    monkeypatch.setattr(emoji_images, "MAX_EMOJI_BYTES", len(PNG))
    monkeypatch.setattr(emoji_images, "EMOJI_TIMEOUT_SECONDS", 0.01)
    calls = []

    async def download(request):
        calls.append(request)
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "404":
            return httpx.Response(404)
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        data = b"not an image" if failure == "bad_image" else PNG + b"a"
        headers = {"content-length": "1"} if failure == "false_size" else {}
        return httpx.Response(200, content=data, headers=headers)

    transport = httpx.MockTransport(download)
    parts = await load_emoji_parts([("60001", "<:happy:123456>")], transport=transport)
    assert len(parts) == 1 and "unavailable" in parts[0]["text"]
    await load_emoji_parts([("60002", "<:happy:123456>")], transport=transport)
    assert len(calls) == 1


async def test_count_limit_prioritizes_current_and_nearby_messages():
    calls = []

    def download(request):
        calls.append(request)
        return httpx.Response(200, content=PNG)

    sources = [("60003", "😀"), ("60002", "😃"), ("60001", "😄")]
    parts = await load_emoji_parts(sources, max_images=2, transport=httpx.MockTransport(download))
    assert len(calls) == 2 and "😀" in parts[0]["text"] and "😃" in parts[2]["text"]
    limited = await load_emoji_parts(sources, max_images=0, transport=httpx.MockTransport(download))
    assert len(limited) == 1 and "上限" in limited[0]["text"]
    assert len(calls) == 2


async def test_current_and_reply_chain_emojis_reach_llm_without_changing_trigger(settings, store, monkeypatch):
    async def download(client, image):
        return emoji_images.image_data_url(PNG)

    monkeypatch.setattr(emoji_images, "_download", download)
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content="絵文字についての回答"))
    monkeypatch.setattr(store, "complete_chat", complete)
    root = ancestor(60001, "<a:dance:123456> これは何？")
    message = file_message("<@222> 😀", [])
    message.reference = reference(root.id, resolved=root)
    client = AIKunClient(settings, service)
    from types import SimpleNamespace

    client._connection.user = SimpleNamespace(id=222)
    try:
        await client.on_message(message)
        parts = complete.await_args.args[0][-1]["content"]
        assert len([part for part in parts if part["type"] == "image_url"]) == 2
        assert any("60001" in part.get("text", "") and "dance" in part["text"] for part in parts)
        assert message.reply.await_args.args[0] == "絵文字についての回答"
        complete.reset_mock()
        await client.on_message(file_message("<:dance:123456> 😀", []))
        complete.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_discord_mcp_returns_emoji_as_image_content(settings):
    calls = []

    def download(request):
        calls.append(request)
        return httpx.Response(200, content=PNG)

    reader = DiscordReader(settings.discord, api=FakeAPI(), image_transport=httpx.MockTransport(download))
    message = reader.message({"id": "60001", "content": "<:happy:123456> 👩🏽‍💻"}, "34567")
    parts = await reader.image_contents({"message": message})
    assert len(calls) == 2 and sum(part.type == "image" for part in parts) == 2
    assert any(part.type == "text" and "60001" in part.text for part in parts)
    assert all(part.mimeType == "image/png" for part in parts if part.type == "image")


async def test_mcp_attachments_take_priority_over_emojis(settings):
    def download(request):
        assert request.url.path.startswith("/attachments/")
        return httpx.Response(200, content=PNG)

    reader = DiscordReader(settings.discord, api=FakeAPI(), image_transport=httpx.MockTransport(download))
    message = {
        "id": "60001",
        "content": "😀",
        "attachments": [
            {
                "filename": "test.png",
                "url": f"https://cdn.discordapp.com/attachments/12345/{i}/test.png",
                "size": len(PNG),
            }
            for i in range(4)
        ],
    }
    parts = await reader.image_contents({"message": message})
    assert sum(part.type == "image" for part in parts) == 4
