import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.adapters.reply_chain import collect_reply_chain
from ai_kun.chat import ChatService
from ai_kun.memory import ModelTurn
from tests.test_adapters import file_message
from tests.test_reply_chain import ancestor, reference

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=")


def picture(attachment_id=70001, *, data=PNG, error=None):
    return SimpleNamespace(
        id=attachment_id,
        filename="image.png",
        content_type="image/png",
        size=len(data),
        read=AsyncMock(side_effect=error, return_value=data),
    )


async def test_nested_human_reply_image_reaches_llm_when_current_post_only_mentions_bot(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content="元の画像について答えるよ。"))
    monkeypatch.setattr(store, "complete_chat", complete)
    root = ancestor(60001, "この画像は何でしょう？")
    root.attachments = [picture()]
    middle = ancestor(60002, "この問題を見て", parent=reference(root.id))
    message = file_message("<@222>", [])
    message.reference = reference(middle.id, resolved=middle)
    message.channel.fetch_message = AsyncMock(return_value=root)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=222)
    try:
        await client.on_message(message)
        messages = complete.await_args.args[0]
        assert any("この画像は何でしょう？" in m["content"] for m in messages if isinstance(m["content"], str))
        parts = messages[-1]["content"]
        assert isinstance(parts, list)
        assert "返信元の質問" in parts[0]["text"]
        assert any(p["type"] == "text" and "60001" in p["text"] for p in parts)
        image = next(p for p in parts if p["type"] == "image_url")
        assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == PNG
        root.attachments[0].read.assert_awaited_once_with(use_cached=True)
        assert message.reply.await_args.args[0] == "元の画像について答えるよ。"
        assert message.reply.await_args.kwargs["mention_author"] is False
        message.channel.send.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_reference_image_failure_preserves_text_context_and_marks_unavailable():
    root = ancestor(60001, "画像の質問文は保持する")
    root.attachments = [picture(data=b"not an image")]
    message = file_message("!ai 見て", [])
    context = await collect_reply_chain(message, root)
    assert context.images == ()
    assert context.messages[0]["content"] == "画像の質問文は保持する"
    assert context.messages[0]["attachments"][0]["image_status"].startswith("unavailable")


async def test_expired_attachment_url_is_refreshed_from_the_same_message():
    expired = discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), {"message": "expired", "code": 0})
    root = ancestor(60001, "古い画像")
    root.attachments = [picture(error=expired)]
    fresh = ancestor(60001, "古い画像")
    fresh.attachments = [picture()]
    message = file_message("!ai 見て", [])
    message.channel.fetch_message = AsyncMock(return_value=fresh)
    context = await collect_reply_chain(message, root)
    assert len(context.images) == 1
    assert context.images[0].message_id == "60001"
    message.channel.fetch_message.assert_awaited_once_with(60001)
    fresh.attachments[0].read.assert_awaited_once_with()


async def test_refresh_does_not_cross_guild_boundary():
    expired = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), {"message": "expired", "code": 0})
    root = ancestor(60001, "画像")
    root.attachments = [picture(error=expired)]
    fresh = ancestor(60001, "別サーバー")
    fresh.guild = SimpleNamespace(id=99999)
    fresh.attachments = [picture()]
    message = file_message("!ai 見て", [])
    message.channel.fetch_message = AsyncMock(return_value=fresh)
    context = await collect_reply_chain(message, root)
    assert context.images == ()
    fresh.attachments[0].read.assert_not_awaited()


async def test_image_download_timeout_does_not_block_remaining_context(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.reply_images.IMAGE_TIMEOUT_SECONDS", 0.01)
    root = ancestor(60001, "画像の質問")
    image = picture()

    async def stuck(**kwargs):
        await asyncio.Event().wait()

    image.read.side_effect = stuck
    root.attachments = [image]
    context = await asyncio.wait_for(collect_reply_chain(file_message("!ai", []), root), 1)
    assert context.images == ()
    assert context.messages[0]["attachments"][0]["image_status"] == "unavailable"


async def test_chain_image_limits_prefer_nearby_images_and_keep_all_text(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.reply_images.MAX_REPLY_IMAGES", 1)
    root = ancestor(60001, "一番古い画像")
    root.attachments = [picture(70001)]
    middle = ancestor(60002, "近い画像", parent=reference(root.id, resolved=root))
    middle.attachments = [picture(70002)]
    context = await collect_reply_chain(file_message("!ai", []), middle)
    assert [record["message_id"] for record in context.messages] == ["60001", "60002"]
    assert len(context.images) == 1 and context.images[0].message_id == "60002"
    root.attachments[0].read.assert_not_awaited()
    assert context.messages[0]["attachments"][0]["image_status"].startswith("omitted")


async def test_empty_current_text_can_use_plain_reply_context(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content="返信元への回答"))
    monkeypatch.setattr(store, "complete_chat", complete)
    root = ancestor(60001, "元の質問です。")
    message = file_message("<@222>", [])
    message.reference = reference(root.id, resolved=root)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=222)
    try:
        await client.on_message(message)
        assert message.reply.await_args.args[0] == "返信元への回答"
        assert "元の質問です。" in complete.await_args.args[0][1]["content"]
    finally:
        await client.drain()
        await client.close()


def test_java_source_is_supported(store):
    source = 'public class Main { public static void main(String[] args) { System.out.println("hello"); } }\n'
    assert store.read_document("Main.java", source.encode(), content_type="application/octet-stream") == source


@pytest.mark.parametrize("false_metadata", [False, True])
async def test_oversized_chain_image_is_not_forwarded(monkeypatch, false_metadata):
    monkeypatch.setattr("ai_kun.adapters.reply_images.MAX_IMAGE_BYTES", len(PNG) - 1)
    root = ancestor(60001, "画像")
    image = picture()
    if false_metadata:
        image.size = 1
    root.attachments = [image]
    context = await collect_reply_chain(file_message("!ai", []), root)
    assert context.images == ()
    if false_metadata:
        image.read.assert_awaited_once()
    else:
        image.read.assert_not_awaited()


async def test_chain_images_obey_total_byte_budget(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.reply_images.MAX_REPLY_IMAGE_BYTES", len(PNG))
    root = ancestor(60001, "画像")
    root.attachments = [picture(70001), picture(70002)]
    context = await collect_reply_chain(file_message("!ai", []), root)
    assert len(context.images) == 1
    assert context.messages[0]["attachments"][1]["image_status"] == "omitted: total image size limit"
