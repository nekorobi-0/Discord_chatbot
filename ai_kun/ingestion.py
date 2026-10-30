"""Discord取り込みの実行状況と、保存済みRAGの集計。"""

from collections import deque
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from sqlalchemy import func
from sqlmodel import Session, select

from .db import HistoryCursorRecord, HistoryMessageRecord, MemoryChunkRecord, MemoryDocumentRecord, get_engine


class IngestionMonitor:
    def __init__(self):
        self._lock = Lock()
        self._active = {}
        self._recent = deque(maxlen=100)

    def start(self, guild_id: int, channel_id: int, kind: str, count: int = 0) -> str:
        job_id = uuid4().hex
        with self._lock:
            self._active[job_id] = {
                "id": job_id,
                "guild_id": str(guild_id),
                "channel_id": str(channel_id),
                "kind": kind,
                "status": "running",
                "messages": count,
                "started_at": datetime.now(UTC).isoformat(),
            }
        return job_id

    def finish(self, job_id: str, *, result=None, error: str | None = None):
        with self._lock:
            job = self._active.pop(job_id)
            job.update(
                status="failed" if error else "completed",
                finished_at=datetime.now(UTC).isoformat(),
                result=dict(result or {}),
                error=error,
            )
            self._recent.appendleft(job)

    def snapshot(self, guild_id: int) -> dict:
        with self._lock:
            return {
                "active": [dict(job) for job in self._active.values() if job["guild_id"] == str(guild_id)],
                "recent": [dict(job) for job in self._recent if job["guild_id"] == str(guild_id)],
            }


def stored_status(guild_id: int, model: str, dimension: int) -> dict:
    """現在のモデルで検索できる発言を数える。削除済み文書は数えない。"""
    history = HistoryMessageRecord
    chunk = MemoryChunkRecord
    guild = str(guild_id)
    document_ids = select(history.document_id).where(history.guild_id == guild, history.document_id.is_not(None))
    searchable = (
        select(chunk.document_id)
        .join(MemoryDocumentRecord, MemoryDocumentRecord.document_id == chunk.document_id)
        .where(chunk.model == model, chunk.dimension == dimension)
        .distinct()
        .subquery()
    )
    with Session(get_engine()) as session:
        rows = session.exec(
            select(history.channel_id, func.count(history.message_id), func.count(searchable.c.document_id))
            .outerjoin(searchable, searchable.c.document_id == history.document_id)
            .where(history.guild_id == guild)
            .group_by(history.channel_id)
            .order_by(history.channel_id)
        ).all()
        cursors = {
            row.channel_key.split(":", 1)[1]: row.last_message_id
            for row in session.exec(
                select(HistoryCursorRecord).where(HistoryCursorRecord.channel_key.startswith(f"{guild}:"))
            )
        }
        documents = session.exec(
            select(func.count())
            .select_from(MemoryDocumentRecord)
            .where(MemoryDocumentRecord.document_id.in_(document_ids))
        ).one()
        chunks = session.exec(
            select(func.count())
            .select_from(chunk)
            .where(chunk.document_id.in_(document_ids), chunk.model == model, chunk.dimension == dimension)
        ).one()
    channels = [
        {"channel_id": channel, "processed": processed, "searchable": ready, "last_message_id": cursors.get(channel)}
        for channel, processed, ready in rows
    ]
    return {
        "processed": sum(row["processed"] for row in channels),
        "searchable": sum(row["searchable"] for row in channels),
        "documents": documents,
        "chunks": chunks,
        "channels": channels,
    }
