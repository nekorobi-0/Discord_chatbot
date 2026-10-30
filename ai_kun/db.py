"""SQLite 永続化。本文・ベクトルは呼び出し側で暗号化する。"""

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Column, LargeBinary, event
from sqlmodel import Field, SQLModel, create_engine


def data_dir() -> Path:
    path = Path(os.environ.get("DATA_DIR", "./data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


class MemoryDocumentRecord(SQLModel, table=True):
    document_id: str = Field(primary_key=True)
    source: str
    title: str
    content_ciphertext: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    created_at: datetime


class MemoryChunkRecord(SQLModel, table=True):
    chunk_id: str = Field(primary_key=True)
    document_id: str = Field(index=True, foreign_key="memorydocumentrecord.document_id")
    content_ciphertext: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    embedding_ciphertext: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    model: str
    dimension: int
    created_at: datetime


class EmbeddingCacheRecord(SQLModel, table=True):
    digest: str = Field(primary_key=True)
    model: str
    vector_ciphertext: bytes = Field(sa_column=Column(LargeBinary, nullable=False))


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


class MetadataEnvelopeRecord(SQLModel, table=True):
    """話者ID・チャンネルID・添付種別などの目録は、暗号化した1本の封書としてだけ保存する。

    カラムは envelope_id と sealed の2本だけ。sealed は exclude=True かつ repr=False にして
    あり、model_dump・str・repr・ログ出力のいずれにも平文（および暗号文）を出さない。
    """

    envelope_id: str = Field(primary_key=True)
    sealed: bytes = Field(exclude=True, repr=False, sa_column=Column(LargeBinary, nullable=False))


class MetadataDocumentLinkRecord(SQLModel, table=True):
    """どの封書がどの保存文書に対応するか。ここにも話者ID・チャンネルIDは平文で置かない。

    ID同士を結ぶだけで目録の内容は持たないため、repr に出す機密はない。
    """

    envelope_id: str = Field(primary_key=True, foreign_key="metadataenveloperecord.envelope_id")
    document_id: str = Field(index=True, foreign_key="memorydocumentrecord.document_id")


class HistoryCursorRecord(SQLModel, table=True):
    channel_key: str = Field(primary_key=True)
    last_message_id: str


class HistoryMessageRecord(SQLModel, table=True):
    # 削除した文書の対応も残し、再取り込みで復活させない。
    message_id: str = Field(primary_key=True)
    guild_id: str
    channel_id: str = Field(index=True)
    document_id: str | None = Field(default=None, index=True)
