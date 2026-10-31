import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import httpx
import pytest

from ai_kun.adapters.discord_bot import AIKunClient, split_message
from ai_kun.adapters.stream_reply import StreamReply
from ai_kun.chat import ChatService
from ai_kun.discord_users import remember_users
from ai_kun.memory import ModelTurn
from ai_kun.streaming import current_text_update
from tests.test_adapters import file_message
from tests.test_chat_stream import Events, event, transport


class Published:
    def __init__(self):
        self.contents = []
        self.edit = AsyncMock(side_effect=self._edit)

    async def _edit(self, **kwargs):
        self.contents.append(kwargs["content"])
        assert not kwargs["allowed_mentions"].everyone
        return self


async def until(condition):
    async with asyncio.timeout(1):
        while not condition():
            await asyncio.sleep(0.001)


async def test_progress_edits_one_reply_then_commits_final_text(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.01)
    message = Published()
    send = AsyncMock(return_value=message)
    publisher = StreamReply(send, split_message, request_id="r1")
    publisher.start()
    try:
        await publisher.update("最初の文。")
        await until(lambda: send.await_count == 1)
        assert "最初の文。" in send.await_args.args[0]
        await publisher.update("最初の文。次の文。")
        await until(lambda: bool(message.contents))
        await publisher.finish("最初の文。次の文。完了。")
        assert send.await_count == 1
        assert message.contents[-1] == "最初の文。次の文。完了。"
        assert "生成中" not in message.contents[-1]
        assert publisher._task.done()
    finally:
        await publisher.close()


async def test_slow_discord_send_does_not_block_token_receipt_or_duplicate_reply(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.001)
    entered, release = asyncio.Event(), asyncio.Event()
    message = Published()

    async def send(text):
        entered.set()
        await release.wait()
        return message

    sender = AsyncMock(side_effect=send)
    publisher = StreamReply(sender, split_message, request_id="r2")
    publisher.start()
    try:
        await entered.wait()
        await asyncio.wait_for(publisher.update("本文はDiscord通信と独立して進む。"), 0.05)
        finishing = asyncio.create_task(publisher.finish("最終回答"))
        await asyncio.sleep(0)
        release.set()
        await finishing
        assert sender.await_count == 1
        assert message.contents[-1] == "最終回答"
    finally:
        release.set()
        await publisher.close()


async def test_failure_preserves_published_partial_text(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.001)
    message = Published()
    publisher = StreamReply(AsyncMock(return_value=message), split_message, request_id="r3")
    publisher.start()
    try:
        await publisher.update("ここまでは生成済み。")
        await until(lambda: publisher.message is not None)
        await publisher.finish("LLMが応答しませんでした。", failed=True)
        assert message.contents[-1].startswith("ここまでは生成済み。")
        assert "生成が中断" in message.contents[-1]
        assert "LLMが応答しませんでした" in message.contents[-1]
    finally:
        await publisher.close()


async def test_fast_reply_skips_placeholder_and_splits_utf16_safely():
    sender = AsyncMock(return_value=Published())
    publisher = StreamReply(sender, split_message, request_id="r4")
    publisher.start()
    try:
        text = "😀" * 1500
        await publisher.finish(text)
        parts = [call.args[0] for call in sender.await_args_list]
        assert "".join(parts) == text
        assert all(len(part.encode("utf-16-le")) // 2 <= 1900 for part in parts)
        assert all("考え中" not in part for part in parts)
    finally:
        await publisher.close()


async def test_deleted_preview_falls_back_to_another_reply_to_original(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.001)
    first, second = Published(), Published()
    first.edit.side_effect = discord.NotFound(SimpleNamespace(status=404, reason="missing"), {"code": 10008})
    sender = AsyncMock(side_effect=[first, second])
    publisher = StreamReply(sender, split_message, request_id="r5")
    publisher.start()
    try:
        await until(lambda: publisher.message is not None)
        await publisher.finish("最終回答")
        assert sender.await_count == 2
        assert sender.await_args.args[0] == "最終回答"
    finally:
        await publisher.close()


async def test_stream_preview_withholds_partial_ids_and_unsolicited_footer(settings, store, monkeypatch):
    monkeypatch.setattr("ai_kun.chat.STREAM_PREVIEW_INTERVAL_SECONDS", 0)
    user_id = "677414946837102592"
    remember_users(23456, [{"id": user_id, "nick": "はみんぐ"}])
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))

    async def complete(messages, **kwargs):
        update = kwargs["on_text"]
        await update("最初の文。member-677414")
        await update("最初の文。member-" + user_id + " が話した。")
        await update("最初の文。member-" + user_id + " が話した。\n根拠データ：\n")
        return ModelTurn(content="最初の文。member-" + user_id + " が話した。")

    monkeypatch.setattr(store, "complete_chat", complete)
    updates = AsyncMock()
    result = await ChatService(settings, store).generate_reply("誰の話？", on_update=updates)
    previews = [call.args[0] for call in updates.await_args_list]
    assert previews[0] == "最初の文。"
    assert all("677414" not in text and "member-" not in text and "根拠データ" not in text for text in previews)
    assert "はみんぐ" in previews[1] and "はみんぐ" in result.content


async def test_cancelling_waiter_does_not_cancel_accepted_reply_delivery(settings, store, monkeypatch):
    client = AIKunClient(settings, ChatService(settings, store))
    entered, release = asyncio.Event(), asyncio.Event()
    sender = AsyncMock(return_value=Published())
    publisher = StreamReply(sender, split_message, request_id="r6")

    async def operation():
        entered.set()
        await release.wait()
        return "受付済みの回答"

    task = asyncio.create_task(client._run(operation, publisher=publisher))
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await client.drain()
        sender.assert_awaited_once_with("受付済みの回答")
        assert current_text_update.get() is None
    finally:
        release.set()
        await client.drain()
        await client.close()


async def test_real_sse_to_discord_edits_before_generation_completes(settings, store, monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr("ai_kun.chat.STREAM_PREVIEW_INTERVAL_SECONDS", 0)
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    stream = Events(
        [
            event({"content": "最初の文。"}),
            0.03,
            event({"content": "次の文。"}),
            0.03,
            event(finish="stop"),
            b"data: [DONE]\n\n",
        ]
    )
    transport(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    client = AIKunClient(settings, service)
    message = file_message("!ai 質問", [])
    published = Published()
    message.reply.return_value = published
    task = asyncio.create_task(client.on_message(message))
    try:
        await until(lambda: message.reply.await_count > 0)
        assert not task.done()
        assert "最初の文。" in message.reply.await_args.args[0]
        await task
        assert published.contents[-1] == "最初の文。次の文。"
        assert message.reply.await_count == 1
        assert message.reply.await_args.kwargs["mention_author"] is False
        message.channel.send.assert_not_awaited()
        assert stream.closed
    finally:
        await client.drain()
        await client.close()


async def test_streaming_tool_round_trip_resolves_names_before_final_answer(settings, store, monkeypatch):
    monkeypatch.setattr("ai_kun.chat.STREAM_PREVIEW_INTERVAL_SECONDS", 0)
    user_id = "677414946837102592"
    remember_users(23456, [{"id": user_id, "nick": "はみんぐ"}])
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    requests = []

    def handle(request):
        data = json.loads(request.content)
        requests.append(data)
        if len(requests) == 1:
            stream = Events(
                [
                    event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "lookup",
                                    "function": {
                                        "name": "discord_resolve_users",
                                        "arguments": json.dumps({"user_ids": [user_id]}),
                                    },
                                }
                            ]
                        }
                    ),
                    event(finish="tool_calls"),
                    b"data: [DONE]\n\n",
                ]
            )
        else:
            assert any(m["role"] == "tool" and "はみんぐ" in m["content"] for m in data["messages"])
            stream = Events([event({"content": "はみんぐの発言だね。"}), event(finish="stop"), b"data: [DONE]\n\n"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    transport(monkeypatch, handle)
    updates = AsyncMock()
    reply = await ChatService(settings, store).generate_reply("誰の発言？", on_update=updates)
    assert reply.content == "はみんぐの発言だね。"
    assert len(requests) == 2 and all(request["stream"] for request in requests)
    assert all(user_id not in call.args[0] for call in updates.await_args_list)


async def test_transient_edit_connection_error_does_not_abort_generation(monkeypatch):
    import aiohttp

    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.005)
    message = Published()
    message.edit.side_effect = [aiohttp.ClientConnectionError("private error"), message]
    publisher = StreamReply(AsyncMock(return_value=message), split_message, request_id="retry-edit")
    publisher.start()
    try:
        await publisher.update("最初の文。")
        await until(lambda: publisher.message is not None)
        await publisher.update("次の文まで生成した。")
        await until(lambda: publisher.disabled)
        await publisher.update("Discord通信に関係なく生成は進む。")
        await publisher.finish("生成完了の本文")
        assert message.edit.await_args.kwargs["content"] == "生成完了の本文"
        assert message.edit.await_count == 2
    finally:
        await publisher.close()


async def test_reply_to_in_progress_post_reads_final_text_when_its_turn_starts(settings, store, monkeypatch):
    from tests.test_reply_chain import ancestor, reference

    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content="確定した回答に沿った返答"))
    monkeypatch.setattr(store, "complete_chat", complete)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=12345)
    cached = ancestor(60001, "考え中…", bot=True)
    final = ancestor(60001, "先行する質問への確定した回答", bot=True)
    message = file_message("続けて説明して", [])
    message.reference = reference(60001, resolved=cached)
    message.channel.fetch_message = AsyncMock(return_value=final)
    try:
        await client.on_message(message)
        messages = complete.await_args.args[0]
        assert any("先行する質問への確定した回答" in m["content"] for m in messages if isinstance(m["content"], str))
        message.channel.fetch_message.assert_awaited_once_with(60001)
    finally:
        await client.drain()
        await client.close()
