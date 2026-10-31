import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import httpx
import pytest
from PIL import Image

from ai_kun import sticker_images
from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.chat import ChatService
from ai_kun.discord_mcp import DiscordReader
from ai_kun.memory import ModelTurn
from ai_kun.sticker_images import load_sticker_parts, render_sticker, sticker_record
from tests.test_adapters import file_message
from tests.test_discord_mcp import FakeAPI
from tests.test_reply_chain import ancestor, reference
from tests.test_reply_images import PNG

LOTTIE = json.dumps(
    {
        "v": "5.7.4",
        "fr": 30,
        "ip": 0,
        "op": 30,
        "w": 32,
        "h": 32,
        "assets": [],
        "layers": [
            {
                "ddd": 0,
                "ind": 1,
                "ty": 1,
                "nm": "Red",
                "sr": 1,
                "sw": 32,
                "sh": 32,
                "sc": "#ff0000",
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [0, 0, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]},
                },
                "ao": 0,
                "ip": 0,
                "op": 30,
                "st": 0,
                "bm": 0,
            }
        ],
    }
).encode()


@pytest.fixture(autouse=True)
def clear_sticker_cache():
    sticker_images._cache.clear()
    yield
    sticker_images._cache.clear()


def sticker(sticker_id=70001, format_type=1, name="スタンプ"):
    return {"id": str(sticker_id), "name": name, "format_type": format_type}


@pytest.mark.parametrize("format_type", [1, 2, 3, 4])
async def test_all_discord_sticker_formats_render_to_cpu_png(format_type):
    if format_type == 3:
        data = LOTTIE
    else:
        with Image.new("RGBA", (32, 32), (255, 0, 0, 255)) as image, io.BytesIO() as output:
            image.save(output, format="GIF" if format_type == 4 else "PNG", save_all=format_type == 2)
            data = output.getvalue()
    uri = await render_sticker(data, format_type)
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1]))) as image:
        assert image.size == (32, 32)
        assert image.convert("RGB").getpixel((16, 16)) == (255, 0, 0)
        assert not getattr(image, "is_animated", False)


@pytest.mark.parametrize("format_type", [2, 4])
async def test_animated_sticker_sends_only_first_frame_and_preserves_size_bound(format_type):
    with (
        Image.new("RGB", (640, 640), "red") as first,
        Image.new("RGB", (640, 640), "blue") as second,
        io.BytesIO() as output,
    ):
        first.save(
            output,
            format="GIF" if format_type == 4 else "PNG",
            save_all=True,
            append_images=[second],
            duration=100,
            loop=0,
        )
        uri = await render_sticker(output.getvalue(), format_type)
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1]))) as image:
        assert image.size == (320, 320) and image.getpixel((160, 160)) == (255, 0, 0)


async def test_lottie_external_assets_and_invalid_data_are_rejected():
    document = json.loads(LOTTIE)
    document["assets"] = [{"id": "image", "u": "/etc/", "p": "passwd"}]
    with pytest.raises(ValueError):
        await render_sticker(json.dumps(document).encode(), 3)
    with pytest.raises(ValueError):
        await render_sticker(b"not an image", 1)


async def test_url_formats_duplicate_mapping_and_cache():
    calls = []

    def download(request):
        calls.append(request)
        return httpx.Response(200, content=LOTTIE if request.url.path.endswith(".json") else PNG)

    transport = httpx.MockTransport(download)
    sources = [("60002", [sticker(70001), sticker(70002, 3)]), ("60001", [sticker(70001), sticker(70003, 4)])]
    parts = await load_sticker_parts(sources, transport=transport)
    assert len(calls) == 3 and sum(p["type"] == "image_url" for p in parts) == 3
    assert [str(request.url) for request in calls] == [
        "https://cdn.discordapp.com/stickers/70001.png",
        "https://cdn.discordapp.com/stickers/70002.json",
        "https://media.discordapp.net/stickers/70003.gif",
    ]
    assert "60002" in parts[0]["text"] and "60001" in parts[0]["text"]
    await load_sticker_parts(sources, transport=transport)
    assert len(calls) == 3
    assert all("authorization" not in request.headers for request in calls)


@pytest.mark.parametrize("failure", ["404", "redirect", "oversized", "false_size", "timeout", "bad_image"])
async def test_failed_sticker_is_marked_unavailable_and_negative_cached(failure, monkeypatch):
    monkeypatch.setattr(sticker_images, "MAX_STICKER_BYTES", len(PNG))
    monkeypatch.setattr(sticker_images, "DOWNLOAD_TIMEOUT_SECONDS", 0.01)
    calls = []

    async def download(request):
        calls.append(request)
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "404":
            return httpx.Response(404)
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        return httpx.Response(
            200,
            content=b"bad" if failure == "bad_image" else PNG + b"a",
            headers={"content-length": "1"} if failure == "false_size" else {},
        )

    sources = [("60001", [sticker()])]
    parts = await load_sticker_parts(sources, transport=httpx.MockTransport(download))
    assert len(parts) == 1 and "unavailable" in parts[0]["text"]
    await load_sticker_parts(sources, transport=httpx.MockTransport(download))
    assert len(calls) == 1


async def test_sticker_limit_and_invalid_id_never_download_arbitrary_url():
    def download(request):
        assert request.url.path == "/stickers/70001.png"
        return httpx.Response(200, content=PNG)

    sources = [("60002", [sticker()]), ("60001", [sticker(70002)])]
    parts = await load_sticker_parts(sources, max_images=1, transport=httpx.MockTransport(download))
    assert sum(p["type"] == "image_url" for p in parts) == 1
    assert "上限" in parts[-1]["text"]
    invalid = await load_sticker_parts(
        [("60001", [sticker("https://evil.example/")])], transport=httpx.MockTransport(download)
    )
    assert len(invalid) == 1 and "不正" in invalid[0]["text"]


def test_sdk_sticker_record_preserves_format_and_name():
    result = sticker_record(SimpleNamespace(id=70001, name="こんにちは", format=discord.StickerFormatType.apng))
    assert result == sticker(70001, 2, "こんにちは")


@pytest.mark.parametrize("current_sticker", [False, True])
async def test_sticker_only_invoked_reply_and_chain_reach_model(settings, store, monkeypatch, current_sticker):
    async def download(client, item):
        return sticker_images.image_data_url(PNG)

    monkeypatch.setattr(sticker_images, "download_sticker", download)
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content="スタンプの説明"))
    monkeypatch.setattr(store, "complete_chat", complete)
    root = ancestor(60001, "スタンプについて教えて", bot=current_sticker)
    root.author.id = 222 if current_sticker else 11111
    current = file_message("" if current_sticker else "<@222>", [])
    current.reference = reference(root.id, resolved=root)
    (current if current_sticker else root).stickers = [
        SimpleNamespace(id=70001, name="スタンプ", format=discord.StickerFormatType.png)
    ]
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=222)
    try:
        await client.on_message(current)
        parts = complete.await_args.args[0][-1]["content"]
        assert sum(p["type"] == "image_url" for p in parts) == 1
        assert any("70001" in p.get("text", "") and "スタンプ" in p.get("text", "") for p in parts)
        assert current.reply.await_args.args[0] == "スタンプの説明"
        complete.reset_mock()
        standalone = file_message("", [])
        standalone.stickers = current.stickers if current_sticker else root.stickers
        await client.on_message(standalone)
        complete.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_mcp_sticker_items_become_image_content(settings):
    reader = DiscordReader(
        settings.discord,
        api=FakeAPI(),
        image_transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PNG)),
    )
    record = reader.message({"id": "60001", "content": "", "sticker_items": [sticker()]}, "34567")
    assert record["stickers"] == [sticker()]
    parts = await reader.image_contents({"message": record})
    assert sum(p.type == "image" for p in parts) == 1
    assert any(p.type == "text" and "70001" in p.text and "60001" in p.text for p in parts)


async def test_render_timeout_kills_and_reaps_worker(monkeypatch):
    monkeypatch.setattr(sticker_images, "RENDER_TIMEOUT_SECONDS", 0.01)

    async def communicate(data):
        await asyncio.Event().wait()

    process = SimpleNamespace(returncode=None, communicate=communicate, wait=AsyncMock())

    def kill():
        process.returncode = -9

    from unittest.mock import Mock

    process.kill = Mock(side_effect=kill)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises(TimeoutError):
        await render_sticker(PNG, 1)
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()
