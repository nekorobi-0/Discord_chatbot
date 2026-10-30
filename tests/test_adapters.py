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


@pytest.fixture(autouse=True)
def no_history_network(monkeypatch):
    monkeypatch.setattr(
        ChatService,
        "backfill_channel",
        AsyncMock(return_value={"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}),
    )


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
        channel=SimpleNamespace(id=45678, typing=Typing, send=AsyncMock()),
        content="!ai こんにちは",
        webhook_id=None,
        created_at=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
        attachments=[],
        reply=reply,
    )
    await client.on_message(message)
    # 取り込みは失敗しても、フックへの返信は通常どおり届く。
    reply.assert_awaited_once()
    message.channel.send.assert_not_awaited()
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
            channel=SimpleNamespace(id=34567, typing=Typing, send=AsyncMock()),
            content=content,
            attachments=[],
            reply=AsyncMock(),
        )
        await client.on_message(message)
        generate.assert_awaited_with("テスト", channel_id=34567)
        message.reply.assert_awaited_once()
        message.channel.send.assert_not_awaited()
        assert message.reply.call_args.kwargs["allowed_mentions"].everyone is False
        assert message.reply.call_args.kwargs["mention_author"] is False
    await client.close()


async def test_http_dashboard_and_metrics_without_auth(settings):
    app = create_app(settings, SimpleNamespace(is_ready=lambda: True))
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 200
        for path in ("/", "/dashboard"):
            response = await client.get(path)
            assert response.status == 200
            assert "Discord → RAG 取り込み状況" in await response.text()
        response = await client.get("/metrics/v1")
        assert response.status == 200
        assert set(await response.json()) == {"counters"}


async def test_http_unconfigured_and_disconnected(settings):
    app = create_app(settings, SimpleNamespace(is_ready=lambda: False))
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/healthz")).status == 503
        assert (await client.get("/metrics/v1")).status == 200
        assert (await client.get("/dashboard")).status == 200


async def test_reply_requests_wait_in_queue_instead_of_being_rejected(settings, store):
    import asyncio

    client = AIKunClient(settings, ChatService(settings, store))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation(index):
        calls.append(index)
        entered.set()
        await release.wait()
        return f"reply-{index}"

    tasks = [asyncio.create_task(client._run(lambda: operation(0)))]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for index in range(1, 4):
            tasks.append(asyncio.create_task(client._run(lambda index=index: operation(index))))
            await asyncio.sleep(0)
        assert calls == [0]
        assert not any(task.done() for task in tasks)
        release.set()
        assert await asyncio.gather(*tasks) == [f"reply-{index}" for index in range(4)]
        assert calls == [0, 1, 2, 3]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.drain()
        await client.close()


async def test_shutdown_waits_for_accepted_message_handlers(settings, store, monkeypatch):
    import asyncio

    client = AIKunClient(settings, ChatService(settings, store))
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(message):
        entered.set()
        await release.wait()

    monkeypatch.setattr(client, "_handle_message", handle)
    task = asyncio.create_task(client.on_message(SimpleNamespace()))
    draining = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        draining = asyncio.create_task(client.drain())
        await asyncio.sleep(0)
        assert not draining.done()
        release.set()
        await task
        await draining
        assert client._message_tasks == set()
    finally:
        release.set()
        await task
        if draining is not None:
            await draining
        await client.close()


def file_attachment(name, text):
    data = text.encode("utf-8")
    return SimpleNamespace(filename=name, size=len(data), read=AsyncMock(return_value=data))


def file_message(content, attachments):
    return SimpleNamespace(
        id="9003",
        author=SimpleNamespace(bot=False, id="111"),
        guild=SimpleNamespace(id=23456),
        channel=SimpleNamespace(id=34567, typing=Typing, send=AsyncMock()),
        content=content,
        attachments=attachments,
        reply=AsyncMock(),
        webhook_id=None,
        created_at=datetime(2026, 10, 31, tzinfo=UTC),
    )


@pytest.mark.parametrize("content", ["", "!ai", "!ai この資料を要約して", "この資料を要約して"])
async def test_text_attachment_uses_document_ingestion_without_asr(settings, store, monkeypatch, content):
    from sqlmodel import Session, select

    from ai_kun.db import MemoryDocumentRecord, get_engine

    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts]))
    complete = AsyncMock(return_value=ModelTurn(content="資料の要約です"))
    monkeypatch.setattr(store, "complete_chat", complete)
    service = ChatService(settings, store)
    transcribe = AsyncMock()
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    client = AIKunClient(settings, service)
    attachment = file_attachment("資料.TXT", "納期は金曜日。連絡先 alice@example.com")
    message = file_message(content, [attachment])
    try:
        await client.on_message(message)
        transcribe.assert_not_awaited()
        attachment.read.assert_awaited_once()
        message.channel.send.assert_not_awaited()
        message.reply.assert_awaited()
        sent = "".join(call.args[0] for call in message.reply.await_args_list)
        assert "登録しました" not in sent
        assert "RAG" not in sent
        with Session(get_engine()) as session:
            documents = session.exec(select(MemoryDocumentRecord).where(MemoryDocumentRecord.source == "upload")).all()
            assert len(documents) == 1
            assert "納期は金曜日" in documents[0].content
        complete.assert_awaited_once()
        messages = complete.await_args.args[0]
        assert any("納期は金曜日" in m["content"] and m["role"] == "system" for m in messages)
        assert sum(m["content"].count("納期は金曜日") for m in messages) == 1
        assert "資料の要約です" in sent
        status = await service.ingestion_status()
        assert status["stored"]["documents"] >= 1
        assert any(job["kind"] == "document" for job in status["recent"])
    finally:
        await client.drain()
        await client.close()


async def test_multiple_files_share_4096_token_budget(settings, store, monkeypatch):
    from sqlmodel import Session, select

    from ai_kun.db import MemoryDocumentRecord, get_engine
    from ai_kun.text_files import FILE_TOKEN_LIMIT, file_encoding

    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts]))
    complete = AsyncMock(return_value=ModelTurn(content="確認しました"))
    monkeypatch.setattr(store, "complete_chat", complete)
    service = ChatService(settings, store)
    client = AIKunClient(settings, service)
    message = file_message(
        "!ai 比較して",
        [file_attachment("a.txt", "猫と犬の記録。" * 3000), file_attachment("b.md", "青と赤の記録。" * 3000)],
    )
    try:
        await client.on_message(message)
        with Session(get_engine()) as session:
            documents = session.exec(select(MemoryDocumentRecord).where(MemoryDocumentRecord.source == "upload")).all()
            assert len(documents) == 2
            assert sum(len(file_encoding().encode(doc.content)) for doc in documents) <= FILE_TOKEN_LIMIT
        sent = "".join(call.args[0] for call in message.reply.await_args_list)
        assert "先頭部分のみ" in sent
        assert "4,096" in sent
        assert "確認しました" in sent
    finally:
        await client.drain()
        await client.close()


async def test_unsupported_attachment_is_not_sent_to_asr(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    transcribe = AsyncMock()
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    client = AIKunClient(settings, service)
    attachment = file_attachment("archive.zip", "not an audio file")
    message = file_message("!ai 見て", [attachment])
    try:
        await client.on_message(message)
        assert "対応していません" in message.reply.await_args.args[0]
        transcribe.assert_not_awaited()
        attachment.read.assert_not_awaited()
    finally:
        await client.close()


async def test_audio_attachment_still_uses_transcription(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    transcribe = AsyncMock(return_value="文字起こしです")
    generate = AsyncMock(return_value=SimpleNamespace(content="音声の返答"))
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    monkeypatch.setattr(service, "generate_reply", generate)
    client = AIKunClient(settings, service)
    attachment = file_attachment("voice.ogg", "audio bytes")
    message = file_message("!ai", [attachment])
    try:
        await client.on_message(message)
        transcribe.assert_awaited_once_with(b"audio bytes", "voice.ogg")
        generate.assert_awaited_once()
        assert generate.await_args.args == ("文字起こしです",)
        assert generate.await_args.kwargs["channel_id"] == 34567
        assert generate.await_args.kwargs["discord_context"].message_id == "9003"
        assert message.reply.await_args.args[0] == "音声の返答"
        message.channel.send.assert_not_awaited()
    finally:
        await client.close()


async def test_text_attachment_can_be_read_without_rag_or_asr(settings, store, monkeypatch):
    from dataclasses import replace

    settings = replace(settings, memory=replace(settings.memory, enabled=False))
    service = ChatService(settings, store)
    complete = AsyncMock(return_value=ModelTurn(content="本文を読みました"))
    monkeypatch.setattr(store, "complete_chat", complete)
    transcribe = AsyncMock()
    ingest = AsyncMock()
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    monkeypatch.setattr(service, "ingest_document", ingest)
    client = AIKunClient(settings, service)
    message = file_message("!ai 要点は？", [file_attachment("memo.txt", "議題は次のリリース日です。")])
    try:
        await client.on_message(message)
        ingest.assert_not_awaited()
        transcribe.assert_not_awaited()
        assert any("議題は次のリリース日" in m["content"] for m in complete.await_args.args[0])
        assert "本文を読みました" in "".join(call.args[0] for call in message.reply.await_args_list)
    finally:
        await client.drain()
        await client.close()


async def test_oversized_file_is_rejected_before_download_or_asr(settings, store, monkeypatch):
    service = ChatService(settings, store)
    transcribe = AsyncMock()
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    client = AIKunClient(settings, service)
    attachment = file_attachment("large.txt", "small fixture")
    attachment.size = 8 * 1024 * 1024 + 1
    message = file_message("", [attachment])
    try:
        await client.on_message(message)
        assert "8 MiB" in message.reply.await_args.args[0]
        attachment.read.assert_not_awaited()
        transcribe.assert_not_awaited()
    finally:
        await client.close()


@pytest.mark.parametrize("content", ["", "!ai この画像を説明して", "画像を説明して"])
async def test_image_attachment_reaches_multimodal_llm_without_asr(settings, store, monkeypatch, content):
    import base64

    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII="
    )
    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts]))
    complete = AsyncMock(return_value=ModelTurn(content="画像を確認しました"))
    monkeypatch.setattr(store, "complete_chat", complete)
    service = ChatService(settings, store)
    transcribe = AsyncMock()
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    client = AIKunClient(settings, service)
    attachment = SimpleNamespace(filename="sample.PNG", size=len(image), read=AsyncMock(return_value=image))
    message = file_message(content, [attachment])
    try:
        await client.on_message(message)
        transcribe.assert_not_awaited()
        parts = complete.await_args.args[0][-1]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["detail"] == "low"
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == image
        assert message.reply.await_args.args[0] == "画像を確認しました"
        message.channel.send.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_reply_does_not_wait_for_background_backfill(settings, store, monkeypatch):
    import asyncio

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "generate_reply", AsyncMock(return_value=SimpleNamespace(content="すぐ返答")))
    entered, release = asyncio.Event(), asyncio.Event()

    async def backfill(channel):
        entered.set()
        await release.wait()
        return {"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}

    monkeypatch.setattr(service, "backfill_channel", backfill)
    client = AIKunClient(settings, service)
    message = file_message("!ai 質問", [])
    try:
        await asyncio.wait_for(client.on_message(message), timeout=1)
        await entered.wait()
        assert not release.is_set()
        message.reply.assert_awaited_once()
    finally:
        release.set()
        await client.drain()
        await client.close()


@pytest.mark.parametrize("cached", [True, False])
async def test_reply_to_own_bot_message_triggers_without_mention_and_includes_context(
    settings, store, monkeypatch, cached
):
    import discord

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    complete = AsyncMock(return_value=ModelTurn(content="定例は15時からだよ。"))
    monkeypatch.setattr(store, "complete_chat", complete)
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=12345)
    target = SimpleNamespace(id=60001, author=SimpleNamespace(id=12345, bot=True), content="次回の定例は金曜日です。")
    message = file_message("何時から？", [])
    message.mentions = []
    message.reference = SimpleNamespace(
        message_id=60001,
        channel_id=34567,
        guild_id=23456,
        type=discord.MessageReferenceType.default,
        resolved=target if cached else None,
    )
    message.channel.fetch_message = AsyncMock(return_value=target)
    try:
        await client.on_message(message)
        complete.assert_awaited_once()
        messages = complete.await_args.args[0]
        assert messages[-1]["content"] == "何時から？"
        assert any("次回の定例は金曜日です。" in m["content"] for m in messages)
        assert message.reply.await_args.args[0] == "定例は15時からだよ。"
        assert message.reply.await_args.kwargs["mention_author"] is False
        message.channel.send.assert_not_awaited()
        if cached:
            message.channel.fetch_message.assert_not_awaited()
        else:
            message.channel.fetch_message.assert_awaited_once_with(60001)
    finally:
        await client.drain()
        await client.close()


@pytest.mark.parametrize(
    "author_id,guild,type_name",
    [
        (99999, 23456, "default"),
        (12345, 99999, "default"),
        (12345, 23456, "forward"),
    ],
)
async def test_reply_to_others_other_guild_or_forward_does_not_trigger(
    settings, store, monkeypatch, author_id, guild, type_name
):
    import discord

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    generate = AsyncMock()
    monkeypatch.setattr(service, "generate_reply", generate)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=12345)
    message = file_message("普通の返信", [])
    message.reference = SimpleNamespace(
        message_id=60001,
        channel_id=34567,
        guild_id=guild,
        type=getattr(discord.MessageReferenceType, type_name),
        resolved=SimpleNamespace(author=SimpleNamespace(id=author_id), content="previous"),
    )
    message.channel.fetch_message = AsyncMock()
    try:
        await client.on_message(message)
        generate.assert_not_awaited()
        message.reply.assert_not_awaited()
        message.channel.fetch_message.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_unavailable_reply_target_is_ignored_without_ping_or_crash(settings, store, monkeypatch):
    import discord

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    generate = AsyncMock()
    monkeypatch.setattr(service, "generate_reply", generate)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=12345)
    message = file_message("普通の返信", [])
    message.reference = SimpleNamespace(
        message_id=60001, channel_id=34567, guild_id=23456, type=discord.MessageReferenceType.default, resolved=None
    )
    message.channel.fetch_message = AsyncMock(
        side_effect=discord.Forbidden(
            SimpleNamespace(status=403, reason="Forbidden"), {"message": "Missing Access", "code": 50001}
        )
    )
    try:
        await client.on_message(message)
        generate.assert_not_awaited()
        message.reply.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_bot_reply_passes_complete_chain_and_current_message_id_to_llm(settings, store, monkeypatch):
    from tests.test_reply_chain import ancestor, reference

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content="チェーン全体を踏まえた返答"))
    monkeypatch.setattr(store, "complete_chat", complete)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=12345)
    root = ancestor(60001, "最初の質問")
    human = ancestor(60002, "追加の質問", parent=reference(60001, resolved=root))
    bot = ancestor(60003, "前の回答", bot=True, parent=reference(60002, resolved=human))
    message = file_message("その続きは？", [])
    message.id = 90003
    message.reference = reference(60003, resolved=bot)
    try:
        await client.on_message(message)
        messages = complete.await_args.args[0]
        assert "今回のDiscordメッセージID: 90003" in messages[0]["content"]
        assert "今回の返信先メッセージID: 60003" in messages[0]["content"]
        chain = next(item["content"] for item in messages if "今回の返信チェーン" in item["content"])
        assert chain.index("最初の質問") < chain.index("追加の質問") < chain.index("前の回答")
        for message_id in ("60001", "60002", "60003"):
            assert message_id in chain
        assert messages[-1]["content"] == "その続きは？"
        assert message.reply.await_args.kwargs["mention_author"] is False
    finally:
        await client.drain()
        await client.close()
