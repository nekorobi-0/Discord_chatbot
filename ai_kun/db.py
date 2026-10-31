"""SQLite 永続化。本文・埋め込みベクトルを保存する。"""

# 検索は保存済みの埋め込みベクトルに対して NumPy のバッチ演算で行う。

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Field, SQLModel, create_engine


def data_dir() -> Path:
    path = Path(os.environ.get("DATA_DIR", "./data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


class MemoryDocumentRecord(SQLModel, table=True):
    document_id: str = Field(primary_key=True)
    source: str
    title: str
    content: str
    created_at: datetime


class MemoryChunkRecord(SQLModel, table=True):
    chunk_id: str = Field(primary_key=True)
    document_id: str = Field(index=True, foreign_key="memorydocumentrecord.document_id")
    content: str
    embedding: bytes
    model: str
    dimension: int
    created_at: datetime


class EmbeddingCacheRecord(SQLModel, table=True):
    digest: str = Field(primary_key=True)
    model: str
    vector: bytes


@lru_cache(maxsize=1)
def get_engine():
    engine = create_engine(f"sqlite:///{data_dir() / 'memory.sqlite3'}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")

    return engine


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())


class HistoryCursorRecord(SQLModel, table=True):
    channel_key: str = Field(primary_key=True)
    last_message_id: str


class HistoryScanCursorRecord(SQLModel, table=True):
    """APIで欠落なく走査し、保存し終えた位置。ライブ観測位置とは分離する。"""

    channel_key: str = Field(primary_key=True)
    last_message_id: str


class DiscordUserRecord(SQLModel, table=True):
    user_key: str = Field(primary_key=True)
    guild_id: str = Field(index=True)
    user_id: str
    username: str = ""
    global_name: str = ""
    nickname: str | None = None


class HistoryMessageRecord(SQLModel, table=True):
    # 削除した文書の対応も残し、再取り込みで復活させない。
    message_id: str = Field(primary_key=True)
    guild_id: str
    channel_id: str = Field(index=True)
    document_id: str | None = Field(default=None, index=True)


class HistoryMessageSpanRecord(SQLModel, table=True):
    """会話をまとめた文書内の、各発言の作者と範囲。"""

    message_id: str = Field(primary_key=True)
    document_id: str = Field(index=True, foreign_key="memorydocumentrecord.document_id", ondelete="CASCADE")
    author_id: str = Field(index=True)
    start: int
    end: int
