from dataclasses import replace

import httpx
import pytest

from ai_kun.chat import ChatService
from ai_kun.errors import InvalidInputError, RateLimitError, UpstreamError
from ai_kun.memory import ModelTurn, ToolCall


async def test_reply_end_to_end_mock_api(settings, store, monkeypatch):
    sent = []

    async def request(method, url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "こんにちは！"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    monkeypatch.setattr(store, "_request", request)
    service = ChatService(settings, store)
    result = await service.generate_reply("連絡先は alice@example.com です")
    assert result.content == "こんにちは！"
    assert "alice@example.com" in str(sent)
    assert len(sent[0]["messages"]) == 2
    await service.generate_reply("こんにちは")
    assert len(sent[1]["messages"]) == 2  # 履歴を追加しない


async def test_tool_round_trip(settings, store, monkeypatch):
    service = ChatService(replace(settings, search=replace(settings.search, enabled=True, api_key="test")), store)
    messages_seen = []
    turns = iter(
        [
            ModelTurn(tool_calls=(ToolCall("call-1", "web_search", '{"query":"天気"}'),)),
            ModelTurn(content="検索結果です"),
        ]
    )

    async def complete(messages, **kwargs):
        messages_seen.append([dict(message) for message in messages])
        return next(turns)

    async def execute(name, args):
        assert name == "web_search" and args == {"query": "天気"}
        return '{"untrusted_data": []}'

    monkeypatch.setattr(store, "complete_chat", complete)
    monkeypatch.setattr(service._tools, "execute", execute)
    assert (await service.generate_reply("今日の天気を検索して")).content == "検索結果です"
    assert messages_seen[1][-1]["role"] == "tool"
    assert messages_seen[1][-1]["tool_call_id"] == "call-1"


async def test_rag_failure_still_replies(settings, store, monkeypatch):
    async def search(*args, **kwargs):
        raise UpstreamError("記憶")

    async def complete(messages, **kwargs):
        assert len(messages) == 2
        return ModelTurn(content="今は資料を参照できません。")

    monkeypatch.setattr(store, "search", search)
    monkeypatch.setattr(store, "complete_chat", complete)
    assert (await ChatService(settings, store).generate_reply("資料を教えて")).content


async def test_rag_context(settings, store, monkeypatch):
    async def embed(texts):
        return [[1.0] * 64 for text in texts]

    async def complete(messages, **kwargs):
        assert "金曜" in messages[1]["content"]
        return ModelTurn(content="金曜です。")

    monkeypatch.setattr(store, "embed", embed)
    monkeypatch.setattr(store, "complete_chat", complete)
    service = ChatService(settings, store)
    result = await service.remember("プロジェクトの締切は金曜")
    reply = await service.generate_reply("プロジェクトの締切は？")
    assert reply.citations == (f"document:{result['document_id']}",)


async def test_output_limit_preserves_personal_information(settings, store, monkeypatch):
    async def complete(*args, **kwargs):
        return ModelTurn(content="alice@example.com " + "あ" * 3000)

    monkeypatch.setattr(store, "complete_chat", complete)
    result = await ChatService(settings, store).generate_reply("こんにちは")
    assert len(result.content) <= settings.operational.max_output_chars
    assert result.truncated and "alice@example.com" in result.content


@pytest.mark.parametrize("content", ["", " " * 4, "x" * 8001])
async def test_invalid_input(settings, store, content):
    with pytest.raises(InvalidInputError):
        await ChatService(settings, store).generate_reply(content)


async def test_opt_out_phrases_no_longer_skip(settings, store, monkeypatch):
    # オプトアウト廃止：旧マーカー入りの内容も普通に保存・検索される。
    async def embed(texts):
        return [[1.0] + [0.0] * 63 for text in texts]

    monkeypatch.setattr(store, "embed", embed)
    result = await ChatService(settings, store).remember("覚えないで sayaka@example.com の締切は金曜日")
    hits = await store.search("締切")
    assert result["chunks"] >= 1
    assert hits and "金曜日" in hits[0].text

    async def complete(*args, **kwargs):
        return ModelTurn(content="こんにちは！")

    monkeypatch.setattr(store, "complete_chat", complete)
    reply = await ChatService(settings, store).generate_reply("覚えないで プロジェクトの締切は？")
    assert reply.content


async def test_tool_loop_is_bounded(settings, store, monkeypatch):
    async def complete(*args, **kwargs):
        return ModelTurn(tool_calls=(ToolCall("x", "unknown", "{}"),))

    monkeypatch.setattr(store, "complete_chat", complete)
    with pytest.raises(UpstreamError):
        await ChatService(settings, store).generate_reply("こんにちは")


@pytest.mark.parametrize("status,error", [(429, RateLimitError), (401, UpstreamError), (500, UpstreamError)])
async def test_upstream_errors(settings, store, monkeypatch, status, error):
    async def request(*args, **kwargs):
        return httpx.Response(status, json={"error": "private upstream details"})

    monkeypatch.setattr(store, "_request", request)
    with pytest.raises(error) as caught:
        await ChatService(settings, store).generate_reply("こんにちは")
    assert "private" not in caught.value.user_message()


async def test_real_http_client_with_mock_transport(settings, store, monkeypatch):
    captured = []

    def handler(request):
        assert request.url == "https://llm.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        captured.append(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "応答"}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    assert (await ChatService(settings, store).generate_reply("こんにちは")).content == "応答"
    assert len(captured) == 1


async def test_local_llm_without_key_and_without_budget(settings, store, monkeypatch):
    import os
    from pathlib import Path

    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "1")  # 古い環境設定も生成を制限しない
    store._llm = replace(settings.llm, base_url="http://127.0.0.1:11434/v1", api_key="")
    calls = []

    def handler(request):
        assert "Authorization" not in request.headers
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "応答"}}],
                "usage": {"prompt_tokens": 9_000_000, "completion_tokens": 100},
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    service = ChatService(settings, store)
    for _ in range(2):
        assert (await service.generate_reply("こんにちは")).content == "応答"
    assert len(calls) == 2
    assert not (Path(os.environ["DATA_DIR"]) / "budget.sqlite3").exists()


async def test_backfill_queues_all_requests_including_same_channel(settings, store, monkeypatch):
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    totals = {"scanned": 1, "imported": 1, "skipped": 0, "documents": 1}

    async def backfill(settings, memory, channel_id, *, api=None):
        calls.append(channel_id)
        entered.set()
        await release.wait()
        return dict(totals)

    monkeypatch.setattr("ai_kun.history.backfill_channel", backfill)
    service = ChatService(settings, store)
    tasks = [asyncio.create_task(service.backfill_channel(34567))]
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        for channel in (34567, 34568):
            tasks.append(asyncio.create_task(service.backfill_channel(channel)))
            await asyncio.sleep(0)
        assert calls == [34567]
        assert not any(task.done() for task in tasks)
        pending = store.ingestion.snapshot(23456)
        assert len(pending["queued"]) == 2
        assert len(pending["active"]) == 1
        release.set()
        assert await asyncio.gather(*tasks) == [totals, totals, totals]
        assert calls == [34567, 34567, 34568]
        assert store.ingestion.snapshot(23456)["queued"] == []
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_uploaded_document_is_truncated_before_ingestion(settings, store, monkeypatch):
    from unittest.mock import AsyncMock

    from sqlmodel import Session, select

    from ai_kun.db import MemoryDocumentRecord, get_engine
    from ai_kun.text_files import FILE_TOKEN_LIMIT, file_encoding

    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts]))
    service = ChatService(settings, store)
    result = await service.ingest_document("long.txt", ("ファイル入力の速度確認。" * 4000).encode(), channel_id=34567)
    assert result["truncated"] is True
    assert result["tokens"] <= FILE_TOKEN_LIMIT
    with Session(get_engine()) as session:
        document = session.exec(select(MemoryDocumentRecord)).one()
        assert 0 < len(file_encoding().encode(document.content)) <= FILE_TOKEN_LIMIT
    status = await service.ingestion_status()
    assert status["stored"]["documents"] == 1
    assert status["stored"]["chunks"] == result["chunks"]
    assert status["recent"][0]["kind"] == "document"


async def test_live_message_burst_is_batched_without_losing_messages(settings, store, monkeypatch):
    import asyncio
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sqlmodel import Session, select

    from ai_kun.db import HistoryMessageRecord, get_engine

    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts]))
    original = store.ingest_history_page
    page = AsyncMock(wraps=original)
    monkeypatch.setattr(store, "ingest_history_page", page)
    service = ChatService(settings, store)
    messages = [
        SimpleNamespace(
            id=index,
            content=f"新着の発言{index}",
            author=SimpleNamespace(id=42, bot=False),
            guild=SimpleNamespace(id=23456),
            channel=SimpleNamespace(id=34567),
            created_at=datetime(2026, 10, 31, tzinfo=UTC),
            webhook_id=None,
        )
        for index in range(100, 120)
    ]
    results = await asyncio.gather(*[service.ingest_message(message) for message in messages])
    await service.drain()
    assert len(results) == 20
    assert all(result["imported"] == 20 for result in results)
    page.assert_awaited_once()
    assert len(page.await_args.args[2]) == 20
    with Session(get_engine()) as session:
        assert len(session.exec(select(HistoryMessageRecord)).all()) == 20


async def test_backfill_burst_shares_scan_but_completes_every_request(settings, store, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    result = {"scanned": 10, "imported": 10, "skipped": 0, "documents": 1}
    scan = AsyncMock(return_value=result)
    monkeypatch.setattr("ai_kun.history.backfill_channel", scan)
    service = ChatService(settings, store)
    results = await asyncio.gather(*[service.backfill_channel(34567) for _ in range(100)])
    await service.drain()
    assert results == [result] * 100
    scan.assert_awaited_once()
    status = await service.ingestion_status()
    assert len(status["recent"]) == 100
    assert all(job["status"] == "completed" and job["result"]["shared_requests"] == 100 for job in status["recent"])


@pytest.mark.parametrize("header", ["根拠データ：", "**根拠データ：**", "**根拠データ**：", "参照メッセージ:"])
async def test_unsolicited_discord_source_footer_is_removed(settings, store, monkeypatch, header):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        store,
        "complete_chat",
        AsyncMock(
            return_value=ModelTurn(
                content=f"定例は金曜日だよ。\n\n{header}\n[#雑談](https://discord.com/channels/23456/34567/60001)"
            )
        ),
    )
    assert (await ChatService(settings, store).generate_reply("定例はいつ？")).content == "定例は金曜日だよ。"


async def test_discord_source_link_is_kept_when_explicitly_requested(settings, store, monkeypatch):
    from unittest.mock import AsyncMock

    response = "元の発言: https://discord.com/channels/23456/34567/60001"
    monkeypatch.setattr(store, "complete_chat", AsyncMock(return_value=ModelTurn(content=response)))
    result = await ChatService(settings, store).generate_reply("元の発言のリンクを出して")
    assert result.content == response


async def test_discord_source_footer_is_removed_when_user_declines_links(settings, store, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        store,
        "complete_chat",
        AsyncMock(
            return_value=ModelTurn(
                content="定例は金曜日。\n\n根拠データ：\nhttps://discord.com/channels/23456/34567/60001"
            )
        ),
    )
    result = await ChatService(settings, store).generate_reply("定例いつ？リンクはいらない")
    assert result.content == "定例は金曜日。"
