import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from ai_kun.chat import ChatService
from ai_kun.errors import UpstreamError
from ai_kun.http_server import create_app
from ai_kun.ingestion import IngestionMonitor


def message(message_id="100"):
    return {
        "id": message_id,
        "content": "取り込み確認",
        "author": {"id": "42"},
        "timestamp": "2026-10-31T00:00:00+00:00",
    }


async def test_pending_then_committed_ingestion_visible_in_api(settings, store, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def embed(texts):
        entered.set()
        await release.wait()
        return [[1.0] + [0.0] * 63 for _ in texts]

    monkeypatch.setattr(store, "embed", embed)
    service = ChatService(settings, memory=store)
    app = create_app(settings, SimpleNamespace(service=service, is_ready=lambda: True))
    task = asyncio.create_task(store.ingest_history_page(23456, 34567, [message()]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/api/ingestion")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            pending = await response.json()
            assert pending["discord_ready"] is True
            assert pending["stored"]["searchable"] == 0
            assert pending["active"][0]["status"] == "running"
            assert pending["active"][0]["messages"] == 1
            release.set()
            await task
            status = await (await client.get("/api/ingestion")).json()
            assert status["active"] == []
            assert status["recent"][0]["result"]["imported"] == 1
            assert status["stored"]["searchable"] == 1
            assert status["stored"]["processed"] == 1
            assert status["stored"]["documents"] == 1
            assert status["stored"]["chunks"] == 1
            assert status["stored"]["channels"] == [
                {"channel_id": "34567", "processed": 1, "searchable": 1, "last_message_id": "100"}
            ]
    finally:
        release.set()
        await task


async def test_failure_retry_duplicates_and_restart_counts(settings, store, monkeypatch):
    service = ChatService(settings, memory=store)
    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=UpstreamError("secret upstream response")))
    with pytest.raises(UpstreamError):
        await store.ingest_history_page(23456, 34567, [message()])
    failed = await service.ingestion_status()
    assert not failed["active"]
    assert failed["stored"]["processed"] == 0
    assert failed["recent"][0]["status"] == "failed"
    assert failed["recent"][0]["error"] == "UpstreamError"
    assert "secret upstream response" not in str(failed)
    monkeypatch.setattr(store, "embed", AsyncMock(return_value=[[1.0] + [0.0] * 63]))
    await store.ingest_history_page(23456, 34567, [message()])
    await store.ingest_history_page(23456, 34567, [message()])
    restarted = ChatService(settings)
    status = await restarted.ingestion_status()
    assert status["stored"]["searchable"] == 1
    assert status["stored"]["processed"] == 1
    assert status["recent"] == []
    changed = replace(settings, embedding=replace(settings.embedding, model="different-model"))
    changed_status = await ChatService(changed).ingestion_status()
    assert changed_status["stored"]["searchable"] == 0
    assert changed_status["stored"]["chunks"] == 0
    assert changed_status["stored"]["processed"] == 1
    assert changed_status["stored"]["documents"] == 1


async def test_deleted_skipped_and_other_guild_are_not_searchable(settings, store, monkeypatch):
    monkeypatch.setattr(store, "embed", AsyncMock(return_value=[[1.0] + [0.0] * 63]))
    await store.ingest_history_page(23456, 34567, [message()])
    await store.ingest_history_page(999, 34568, [message("101")])
    skipped = {**message("102"), "author": {"id": "42", "bot": True}}
    await store.ingest_history_page(23456, 34567, [skipped])
    service = ChatService(settings, memory=store)
    status = await service.ingestion_status()
    assert status["stored"]["processed"] == 2
    assert status["stored"]["searchable"] == 1
    assert len(status["stored"]["channels"]) == 1
    assert all(job["guild_id"] == "23456" for job in status["recent"])
    from sqlmodel import Session

    from ai_kun.db import HistoryMessageRecord, get_engine

    with Session(get_engine()) as session:
        document_id = session.get(HistoryMessageRecord, "100").document_id
    await store.delete_document(document_id)
    deleted = await service.ingestion_status()
    assert deleted["stored"]["processed"] == 2
    assert deleted["stored"]["searchable"] == 0
    assert deleted["stored"]["documents"] == 0
    assert deleted["stored"]["chunks"] == 0


def test_monitor_history_is_bounded():
    monitor = IngestionMonitor()
    for _ in range(110):
        monitor.finish(monitor.start(1, 2, "messages", 1), result={"imported": 1})
    assert len(monitor.snapshot(1)["recent"]) == 100
    assert monitor.snapshot(2) == {"active": [], "recent": []}


async def test_cancelled_ingestion_does_not_remain_active(settings, store, monkeypatch):
    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await store.ingest_history_page(23456, 34567, [message()])
    status = await ChatService(settings, memory=store).ingestion_status()
    assert status["active"] == []
    assert status["recent"][0]["error"] == "CancelledError"


async def test_backfill_failure_is_visible(settings, store, monkeypatch):
    monkeypatch.setattr("ai_kun.history.backfill_channel", AsyncMock(side_effect=UpstreamError("Discord履歴取得")))
    service = ChatService(settings, memory=store)
    with pytest.raises(UpstreamError):
        await service.backfill_channel(34567)
    status = await service.ingestion_status()
    assert status["active"] == []
    assert status["recent"][0]["kind"] == "backfill"
    assert status["recent"][0]["status"] == "failed"
