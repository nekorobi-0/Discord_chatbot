"""AIくんの記憶（RAG）レイヤー。

- 検索は NumPy のバッチ演算で埋め込み近傍を計算する。総当たりの純Python演算は使わない
- 保存した本文・ベクトルは平文のまま検索対象として扱う
- LLM / Embedding への要求は設定済みの社内・ローカル中継先だけに限定し、数値アドレスはプライベート範囲へ固定する
- LLM / Embedding は互換API、ASRはローカル推論
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
import numpy as np
from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import Session, select

from .chat_stream import read_chat_stream
from .config import ASRSettings, EmbeddingSettings, LLMSettings, MemorySettings, _is_internal_host
from .db import (
    EmbeddingCacheRecord,
    HistoryCursorRecord,
    HistoryMessageRecord,
    HistoryMessageSpanRecord,
    MemoryChunkRecord,
    MemoryDocumentRecord,
    get_engine,
    init_db,
)
from .discord_users import message_users, remember_users
from .errors import InvalidInputError, RateLimitError, UpstreamError
from .file_types import DOCUMENT_EXTENSIONS as DOCUMENT_EXTENSIONS
from .file_types import VOICE_EXTENSIONS, decode_text_file
from .ingestion import IngestionMonitor
from .request_diagnostics import phase, request_id
from .streaming import TextUpdate
from .work_queue import WorkQueue

logger = logging.getLogger("ai_kun.memory")

_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_CHUNK_MARKER_RE = re.compile(r"^\s*\[\[\d{1,3}/\d{1,3}\]\]\s*$", re.MULTILINE)
_CONTROL_CHARS_RE = re.compile(r"[^\S\n\r\t\u00a0-\u1680\u2000-\u2fff]+")
_HISTORY_HEADER = re.compile(r"(?m)^\[\d{4}-\d{2}-\d{2}T[^\]\n]+\] member-([0-9]{5,25}): ")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelTurn:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class RetrievedChunk:
    label: str
    text: str


def _sanitize_text(text: str) -> str:
    """プロンプトインジェクションの足しになるもの（埋め込みマーカー・制御文字）を薄める。"""
    cleaned = _CONTROL_CHARS_RE.sub(" ", (text or "").replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = _CHUNK_MARKER_RE.sub("", cleaned)
    for marker in ("system:", "assistant:", "<|im_start|>", "<|im_end|>", "<|", "|>"):
        cleaned = cleaned.replace(marker, " [区切り] ")
    return cleaned.strip()


def _looks_like_literal(value: str) -> bool:
    """数値アドレス（ゾーン指定付きを含む）かどうかを判定する。名前は True を返さない。"""
    candidate = value.strip("[]")
    if "%" in candidate:  # fe80::1%eth0 のような系統指定付きは数値表記として扱う
        candidate = candidate.split("%", 1)[0]
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return True


class MemoryStore:
    """記憶の保存・検索・推論をまとめる唯一の窓口。"""

    def __init__(
        self, embedding: EmbeddingSettings, llm: LLMSettings, asr: ASRSettings, memory: MemorySettings
    ) -> None:
        self._embedding = embedding
        self._llm = llm
        self._asr = asr
        self._memory = memory
        self.ingestion = IngestionMonitor()
        self.ingestion_queue = WorkQueue()
        init_db()

    # ---------- HTTP（SSRF 対策つき）----------
    # 平文HTTPの送信先は、設定時と同じループバック・RFC1918の範囲で検証する。
    @staticmethod
    def _is_internal_ip(literal: str) -> bool:
        """設定検証と同じ宛先ポリシーで判定する。"""
        return _is_internal_host(literal)

    def _validate_upstream(self, url: str) -> str:
        base = None
        for candidate, suffix in (
            (self._llm.base_url, "/chat/completions"),
            (self._embedding.base_url, "/embeddings"),
        ):
            if url == candidate.rstrip("/") + suffix:
                base = candidate.rstrip("/")
                break
        if base is None:
            raise UpstreamError("外部サービス")

        # 設定済みの宛先そのものへしか要求を飛ばさない。HTTPS はそのまま通し、
        # 平文 HTTP は設定値が数値アドレスのときだけ社内・ローカル範囲であることを必須条件にする。
        target = urlparse(url)
        if target.scheme not in {"http", "https"}:
            raise UpstreamError("外部サービス")
        literal = (urlparse(base).hostname or "").lower()
        if target.scheme == "http" and _looks_like_literal(literal) and not self._is_internal_ip(literal):
            raise UpstreamError("外部サービス")
        return "LLM" if url.endswith("/chat/completions") else "Embedding"

    async def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        service = self._validate_upstream(url)
        phase(f"{service.lower()}_http")
        started = time.monotonic()
        timeout = float(self._llm.timeout_seconds)
        logger.info("upstream started: request=%s service=%s timeout=%.1f", request_id(), service, timeout)
        try:
            # httpxのread timeoutは無通信時間。少量ずつ返り続ける応答にも総時間の上限を設ける。
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                    response = await client.request(method, url, **kwargs)
            logger.info(
                "upstream completed: request=%s service=%s status=%d seconds=%.3f",
                request_id(),
                service,
                response.status_code,
                time.monotonic() - started,
            )
            phase(f"{service.lower()}_response")
            return response
        except (TimeoutError, httpx.TimeoutException) as exc:
            logger.warning(
                "upstream timeout: request=%s service=%s error=%s seconds=%.3f",
                request_id(),
                service,
                type(exc).__name__,
                time.monotonic() - started,
            )
            raise UpstreamError(f"{service}（応答タイムアウト）") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "upstream transport error: request=%s service=%s error=%s seconds=%.3f",
                request_id(),
                service,
                type(exc).__name__,
                time.monotonic() - started,
            )
            raise UpstreamError(f"{service}（接続エラー）") from exc

    async def complete_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_text: TextUpdate | None = None,
    ) -> ModelTurn:
        payload: dict[str, object] = {
            "model": self._llm.model,
            "messages": list(messages),
            "temperature": self._llm.temperature if temperature is None else temperature,
            "max_tokens": self._llm.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self._llm.api_key}"} if self._llm.api_key else {}
        if on_text is not None and self._llm.stream_enabled:
            self._validate_upstream(f"{self._llm.base_url.rstrip('/')}/chat/completions")
            data = await read_chat_stream(self._llm, payload, on_text)
        else:
            response = await self._request(
                "POST", f"{self._llm.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers
            )
            if response.status_code == 429:
                raise RateLimitError()
            if response.status_code >= 400:
                logger.warning("llm http error: status=%s model=%s", response.status_code, self._llm.model)
                raise UpstreamError(f"LLM HTTP {response.status_code} (model={self._llm.model})")
            try:
                data = response.json()
            except ValueError as exc:
                raise UpstreamError("おしゃべりの仕組み") from exc
        choices = data.get("choices") or []
        if not choices:
            raise UpstreamError("おしゃべりの仕組み")
        message = choices[0].get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            name = str(function.get("name") or "")[:64]
            if name:
                calls.append(
                    ToolCall(
                        id=str(item.get("id") or uuid.uuid4().hex)[:64],
                        name=name,
                        arguments=str(function.get("arguments") or "{}")[:8_000],
                    )
                )
        usage = data.get("usage") or {}
        return ModelTurn(
            content=(str(message.get("content")) if message.get("content") is not None else None),
            tool_calls=tuple(calls),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    # ---------- Embedding（キャッシュ付き・鍵は外に出さない）----------
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        digests = [
            hashlib.sha256(
                f"{self._embedding.model}:{self._embedding.dimension}\n{text}".encode("utf-8", "ignore")
            ).hexdigest()
            for text in texts
        ]
        unique = dict(zip(digests, texts, strict=True))
        vectors = {}
        keys = list(unique)
        with Session(get_engine()) as session:
            for offset in range(0, len(keys), 256):
                for cached in session.exec(
                    select(EmbeddingCacheRecord).where(EmbeddingCacheRecord.digest.in_(keys[offset : offset + 256]))
                ):
                    vector = _decode_vector(cached.vector, self._embedding.dimension)
                    if vector is not None:
                        vectors[cached.digest] = vector
        missing = [digest for digest in unique if digest not in vectors]
        for offset in range(0, len(missing), 16):
            batch = missing[offset : offset + 16]
            result = await self._embed_batch_uncached([unique[digest] for digest in batch])
            vectors.update(zip(batch, result, strict=True))
        if missing:
            with Session(get_engine()) as session:
                try:
                    for offset in range(0, len(missing), 128):
                        rows = [
                            {
                                "digest": digest,
                                "model": self._embedding.model,
                                "vector": _encode_vector(vectors[digest]),
                            }
                            for digest in missing[offset : offset + 128]
                        ]
                        statement = sqlite_insert(EmbeddingCacheRecord).values(rows)
                        session.execute(
                            statement.on_conflict_do_update(
                                index_elements=["digest"],
                                set_={"model": statement.excluded.model, "vector": statement.excluded.vector},
                            )
                        )
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.warning("embedding cache write failed")
        return [vectors[digest] for digest in digests]

    async def _embed_uncached(self, text: str) -> list[float]:
        return (await self._embed_batch_uncached([text]))[0]

    async def _embed_batch_uncached(self, texts: Sequence[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._embedding.api_key}"} if self._embedding.api_key else {}
        started = time.perf_counter()
        response = await self._request(
            "POST",
            f"{self._embedding.base_url.rstrip('/')}/embeddings",
            json={
                "model": self._embedding.model,
                "input": texts[0] if len(texts) == 1 else list(texts),
                "dimensions": self._embedding.dimension,
            },
            headers=headers,
        )
        if response.status_code == 429:
            raise RateLimitError()
        if response.status_code >= 400:
            logger.warning(
                "embedding http error: status=%s model=%s dimensions=%s inputs=%s",
                response.status_code,
                self._embedding.model,
                self._embedding.dimension,
                len(texts),
            )
            raise UpstreamError(f"Embedding HTTP {response.status_code} (model={self._embedding.model})")
        try:
            data = response.json()
            items = data["data"]
            if len(items) != len(texts):
                raise ValueError("embedding count mismatch")
            indexed = {item.get("index", 0 if len(texts) == 1 else None): item for item in items}
            if set(indexed) != set(range(len(texts))):
                raise ValueError("embedding indices mismatch")
            vectors = [[float(value) for value in indexed[index]["embedding"]] for index in range(len(texts))]
            if any(len(vector) != self._embedding.dimension or not np.isfinite(vector).all() for vector in vectors):
                raise ValueError("embedding dimension mismatch or non-finite vector")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            logger.warning(
                "invalid embedding response: expected_inputs=%s dimension=%s model=%s",
                len(texts),
                self._embedding.dimension,
                self._embedding.model,
            )
            raise UpstreamError("記憶のしくみ") from exc
        logger.info("embedding batch complete: inputs=%d seconds=%.3f", len(texts), time.perf_counter() - started)
        return vectors

    # ---------- 音声（CPU / faster-whisper / ローカルモデルのみ）----------
    async def transcribe(self, data: bytes, filename: str | None) -> str:
        if not self._asr.enabled:
            raise UpstreamError("音声認識")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise InvalidInputError("音声が長すぎたよ。短く区切ってもう一回おねがい。")
        name = (filename or "voice.ogg").lower()
        if not name.endswith(tuple(VOICE_EXTENSIONS)):
            raise InvalidInputError("この音声はうまく扱えなかったよ。")
        return await asyncio.to_thread(self._transcribe_blocking, data)

    def _transcribe_blocking(self, data: bytes) -> str:
        from faster_whisper import WhisperModel

        model_path = self._asr.model_path
        if not os.path.isfile(os.path.join(model_path, "model.bin")):
            logger.warning("ASR model assets missing")
            raise UpstreamError("音声認識")

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
            path = handle.name
        try:
            with open(path, "wb") as sink:
                sink.write(data)
            model = WhisperModel(model_path, device=self._asr.device, compute_type=self._asr.compute_type)
            segments, _info = model.transcribe(path, language=self._asr.language or None, vad_filter=True)
            return _sanitize_text(" ".join(str(segment.text).strip() for segment in segments))
        except Exception as exc:
            logger.warning("asr failed: %s", type(exc).__name__)
            raise UpstreamError("音声認識") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ---------- 資料の取り込み ----------
    def read_document(self, filename: str | None, data: bytes, *, content_type: str | None = None) -> str:
        if len(data) > _MAX_UPLOAD_BYTES:
            raise InvalidInputError("ファイルが大きすぎたよ。もっと小さくしてほしいな。")
        text = decode_text_file(filename, data, content_type)
        if not text.strip():
            raise InvalidInputError("からっぽの資料だったよ。")

        return text

    async def ingest_document(
        self, filename: str | None, data: bytes, *, guild_id: int | None = None, channel_id: int | None = None
    ) -> dict[str, int | str]:
        job = (
            self.ingestion.start(guild_id, channel_id, "document", queued=True)
            if guild_id is not None and channel_id is not None
            else None
        )

        async def operation():
            if job:
                self.ingestion.running(job)
            try:
                result = await self._ingest_document(filename, data)
            except BaseException as exc:
                if job:
                    self.ingestion.finish(job, error=type(exc).__name__)
                raise
            if job:
                self.ingestion.finish(job, result={"documents": 1, "chunks": result["chunks"]})
            return result

        return await self.ingestion_queue.submit(operation)

    async def _ingest_document(self, filename: str | None, data: bytes) -> dict[str, int | str]:
        if not self._memory.enabled:
            raise UpstreamError("記憶")
        text = self.read_document(filename, data)
        if len(text) > self._memory.max_chars_per_document:
            raise InvalidInputError("資料が長すぎたよ。分割してほしいな。")

        document_id = uuid.uuid4().hex
        chunks: list[tuple[str, str]] = []
        step = max(1, self._memory.chunk_size - self._memory.chunk_overlap)
        for start in range(0, len(text), step):
            piece = text[start : start + self._memory.chunk_size]
            if piece.strip():
                chunks.append((uuid.uuid4().hex, piece))
            if start + self._memory.chunk_size >= len(text):
                break

        vectors = await self.embed([piece for _, piece in chunks])
        with Session(get_engine()) as session:
            session.add(
                MemoryDocumentRecord(
                    document_id=document_id,
                    source="upload",
                    title="uploaded document",
                    content=text,
                    created_at=datetime.now(UTC),
                )
            )
            # 文書を先にflushし、チャンクと合わせて1トランザクションで確定する。
            session.flush()
            for (chunk_id, piece), vector in zip(chunks, vectors, strict=True):
                session.add(
                    MemoryChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=piece,
                        embedding=_encode_vector(vector),
                        model=self._embedding.model,
                        dimension=self._embedding.dimension,
                        created_at=datetime.now(UTC),
                    )
                )
            session.commit()
        return {"chunks": len(chunks), "document_id": document_id}

    async def ingest_history_page(self, guild_id: int, channel_id: int, messages: list[dict]) -> dict[str, int]:
        job = self.ingestion.start(guild_id, channel_id, "messages", len(messages), queued=True)

        async def operation():
            self.ingestion.running(job)
            try:
                result = await self._ingest_history_page(guild_id, channel_id, messages)
            except BaseException as exc:
                self.ingestion.finish(job, error=type(exc).__name__)
                raise
            self.ingestion.finish(job, result=result)
            return result

        return await self.ingestion_queue.submit(operation)

    async def _ingest_history_page(self, guild_id: int, channel_id: int, messages: list[dict]) -> dict[str, int]:
        """20発言・30分・文字数を目安にまとめる。ページ保存と進捗は原子的に更新。"""
        if not self._memory.enabled:
            raise UpstreamError("記憶")
        if not messages:
            return {"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}
        remember_users(guild_id, message_users(messages))
        ids = [str(message["id"]) for message in messages]
        with Session(get_engine()) as session:
            seen = set(
                session.exec(
                    select(HistoryMessageRecord.message_id).where(HistoryMessageRecord.message_id.in_(ids))
                ).all()
            )
        groups = []
        current = []
        length = 0
        previous = None
        skipped = []
        encountered = set()
        duplicate_count = 0
        for message in messages:
            message_id = str(message["id"])
            if message_id in seen or message_id in encountered:
                duplicate_count += 1
                continue
            encountered.add(message_id)
            author = message.get("author") or {}
            # 問答無用で全ログをRAGへ。オプトアウトも秘密検出による除外も廃止した。
            text = str(message.get("content") or "").strip()
            if author.get("bot") or message.get("webhook_id") or message.get("type", 0) not in {0, 19} or not text:
                skipped.append(message_id)
                continue
            text = _sanitize_text(text)
            if not text:
                skipped.append(message_id)
                continue
            try:
                timestamp = datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timezone missing")
            except (KeyError, TypeError, ValueError):
                raise UpstreamError("Discord履歴の日時") from None
            member = str(author.get("id", "unknown"))
            line = f"[{timestamp.isoformat()}] member-{member}: {text}"
            if current and (
                len(current) >= 20
                or length + len(line) > self._memory.max_chars_per_document
                or (previous and (timestamp - previous).total_seconds() > 1800)
            ):
                groups.append(current)
                current, length = [], 0
            current.append((message_id, line))
            length += len(line) + 1
            previous = timestamp
        if current:
            groups.append(current)

        prepared = []
        for group in groups:
            document_id = uuid.uuid4().hex
            text = "\n".join(line for _, line in group)
            pieces = []
            step = self._memory.chunk_size - self._memory.chunk_overlap
            for start in range(0, len(text), step):
                pieces.append(text[start : start + self._memory.chunk_size])
                if start + self._memory.chunk_size >= len(text):
                    break
            prepared.append((document_id, text, group, pieces))
        all_vectors = await self.embed([piece for _, _, _, pieces in prepared for piece in pieces])
        offset = 0
        with_vectors = []
        for document_id, text, group, pieces in prepared:
            with_vectors.append((document_id, text, group, pieces, all_vectors[offset : offset + len(pieces)]))
            offset += len(pieces)
        prepared = with_vectors

        # Embedding失敗時にはここへ到達しないため、カーソルも進まず同じページから再開できる。
        now = datetime.now(UTC)
        authors = {}
        for message in messages:
            authors.setdefault(str(message["id"]), str((message.get("author") or {}).get("id", "")))
        with Session(get_engine()) as session:
            for document_id, text, group, pieces, vectors in prepared:
                session.add(
                    MemoryDocumentRecord(
                        document_id=document_id,
                        source="discord_history",
                        title=f"channel:{channel_id}",
                        content=text,
                        created_at=now,
                    )
                )
            # 文書・チャンク・カーソルを1トランザクションで確定する。
            session.flush()
            for document_id, text, group, pieces, vectors in prepared:
                for piece, vector in zip(pieces, vectors, strict=True):
                    session.add(
                        MemoryChunkRecord(
                            chunk_id=uuid.uuid4().hex,
                            document_id=document_id,
                            content=piece,
                            embedding=_encode_vector(vector),
                            model=self._embedding.model,
                            dimension=self._embedding.dimension,
                            created_at=now,
                        )
                    )
                start = 0
                for message_id, line in group:
                    session.add(
                        HistoryMessageRecord(
                            message_id=message_id,
                            guild_id=str(guild_id),
                            channel_id=str(channel_id),
                            document_id=document_id,
                        )
                    )
                    session.add(
                        HistoryMessageSpanRecord(
                            message_id=message_id,
                            document_id=document_id,
                            author_id=authors[message_id],
                            start=start,
                            end=start + len(line),
                        )
                    )
                    start += len(line) + 1
            for message_id in skipped:
                session.add(
                    HistoryMessageRecord(message_id=message_id, guild_id=str(guild_id), channel_id=str(channel_id))
                )
            key = f"{guild_id}:{channel_id}"
            checkpoint = session.get(HistoryCursorRecord, key)
            cursor = str(max(int(message["id"]) for message in messages))
            if checkpoint:
                checkpoint.last_message_id = str(max(int(cursor), int(checkpoint.last_message_id)))
            else:
                checkpoint = HistoryCursorRecord(channel_key=key, last_message_id=cursor)
            session.add(checkpoint)
            session.commit()
        return {
            "scanned": len(messages),
            "imported": sum(len(group) for group in groups),
            "skipped": len(skipped) + duplicate_count,
            "documents": len(groups),
        }

    def _ensure_history_spans(self, scoped_documents):
        """旧文書の作者範囲だけを補完する。本文やEmbeddingは変更しない。"""
        cursor = ""
        while True:
            with Session(get_engine()) as session:
                batch = session.exec(
                    select(MemoryDocumentRecord)
                    .where(
                        MemoryDocumentRecord.document_id.in_(scoped_documents),
                        MemoryDocumentRecord.document_id.not_in(select(HistoryMessageSpanRecord.document_id)),
                        MemoryDocumentRecord.document_id > cursor,
                    )
                    .order_by(MemoryDocumentRecord.document_id)
                    .limit(128)
                ).all()
                if not batch:
                    break
                records = session.exec(
                    select(HistoryMessageRecord).where(
                        HistoryMessageRecord.document_id.in_([doc.document_id for doc in batch])
                    )
                ).all()
                by_document = {}
                for record in records:
                    by_document.setdefault(record.document_id, []).append(record)
                values = []
                for doc in batch:
                    messages = sorted(by_document.get(doc.document_id, []), key=lambda item: int(item.message_id))
                    headers = list(_HISTORY_HEADER.finditer(doc.content))
                    # 一対一対応を確認できない旧文書は推測で作者を割り当てない。
                    if not headers or headers[0].start() != 0 or len(headers) != len(messages):
                        logger.warning("legacy history author metadata unavailable: document=%s", doc.document_id)
                        continue
                    for index, (record, header) in enumerate(zip(messages, headers, strict=True)):
                        values.append(
                            {
                                "message_id": record.message_id,
                                "document_id": doc.document_id,
                                "author_id": header[1],
                                "start": header.start(),
                                "end": headers[index + 1].start() if index + 1 < len(headers) else len(doc.content),
                            }
                        )
                if values:
                    # executemanyでSQLiteの変数数制限を超えない。並行した補完とも競合しない。
                    statement = sqlite_insert(HistoryMessageSpanRecord).on_conflict_do_nothing(
                        index_elements=["message_id"]
                    )
                    session.execute(statement, values)
                session.commit()
                cursor = batch[-1].document_id

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        guild_id: int | str | None = None,
        channel_id: int | str | None = None,
        user_id: int | str | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """メタデータで候補を絞ってから既存ベクトルを検索する。条件はAND。"""
        if not self._memory.enabled:
            return ()
        if (channel_id is not None or user_id is not None) and guild_id is None:
            raise InvalidInputError("条件付き検索にはサーバーIDが必要です。")
        text = _sanitize_text(query)
        if not text:
            return ()
        limit = min(max(1, int(top_k or self._memory.top_k)), 24)
        conditions = [
            MemoryChunkRecord.model == self._embedding.model,
            MemoryChunkRecord.dimension == self._embedding.dimension,
        ]
        if guild_id is not None:
            scoped = select(HistoryMessageRecord.document_id).where(HistoryMessageRecord.guild_id == str(guild_id))
            if channel_id is not None:
                scoped = scoped.where(HistoryMessageRecord.channel_id == str(channel_id))
            conditions.append(MemoryChunkRecord.document_id.in_(scoped))
            if user_id is not None:
                await asyncio.to_thread(self._ensure_history_spans, scoped)
                conditions.append(
                    MemoryChunkRecord.document_id.in_(
                        select(HistoryMessageSpanRecord.document_id).where(
                            HistoryMessageSpanRecord.author_id == str(user_id)
                        )
                    )
                )
        with Session(get_engine()) as session:
            rows = list(
                session.exec(
                    select(MemoryChunkRecord, MemoryDocumentRecord)
                    .join(
                        MemoryDocumentRecord,
                        MemoryDocumentRecord.document_id == MemoryChunkRecord.document_id,
                    )
                    .where(*conditions)
                ).all()
            )
            spans = {}
            if user_id is not None:
                for span in session.exec(
                    select(HistoryMessageSpanRecord).where(
                        HistoryMessageSpanRecord.author_id == str(user_id),
                        HistoryMessageSpanRecord.document_id.in_(
                            select(MemoryChunkRecord.document_id).where(*conditions)
                        ),
                    )
                ):
                    spans.setdefault(span.document_id, []).append(span)
        candidates = []
        for row, document in rows:
            if user_id is None:
                candidates.append((row, row.content, None))
            else:
                fragment, message_id = _author_fragment(document.content, row.content, spans.get(row.document_id, []))
                if fragment:
                    candidates.append((row, fragment, message_id))
        if not candidates:
            return ()
        query_vector = (await self.embed([text]))[0]
        matrix = np.empty((len(candidates), self._embedding.dimension), dtype=np.float32)
        usable = np.zeros(len(candidates), dtype=bool)
        for index, (row, _, _) in enumerate(candidates):
            vector = _decode_vector(row.embedding, self._embedding.dimension)
            if vector is None:
                logger.warning("stored memory entry unreadable")
                continue
            matrix[index] = vector
            usable[index] = True
        if not usable.any():
            return ()
        scores = _batched_cosine(query_vector, matrix[usable])
        positions = np.flatnonzero(usable)
        ranked = np.argsort(-scores, kind="stable")
        results = []
        seen = set()
        for score_index in ranked.tolist():
            if scores[score_index] < self._memory.min_score:
                continue
            row, fragment, message_id = candidates[int(positions[score_index])]
            if (row.document_id, fragment) in seen:
                continue
            seen.add((row.document_id, fragment))
            with Session(get_engine()) as session:
                origin_query = select(HistoryMessageRecord).where(HistoryMessageRecord.document_id == row.document_id)
                if message_id is not None:
                    origin_query = origin_query.where(HistoryMessageRecord.message_id == message_id)
                origin = session.exec(origin_query).first()
            label = f"document:{row.document_id}"
            if origin:
                label += f" https://discord.com/channels/{origin.guild_id}/{origin.channel_id}/{origin.message_id}"
            results.append(RetrievedChunk(label=label, text=fragment))
            if len(results) >= limit:
                break
        return tuple(results)

    async def delete_document(self, document_id: str) -> int:
        doc_id = str(document_id)[:64]
        with Session(get_engine()) as session:
            chunks = (
                session.exec(delete(MemoryChunkRecord).where(MemoryChunkRecord.document_id == doc_id)).rowcount or 0
            )
            docs = (
                session.exec(delete(MemoryDocumentRecord).where(MemoryDocumentRecord.document_id == doc_id)).rowcount
                or 0
            )
            # 文書削除時は埋め込みキャッシュも消す（別文書のキャッシュは再生成可能）。
            session.exec(delete(EmbeddingCacheRecord))
            session.commit()
        if not docs:
            raise InvalidInputError("その名前の記憶は見つからなかったよ。")
        return int(chunks)


def _encode_vector(vector: Sequence[float]) -> bytes:
    """埋め込みを float32 のリトルエンディアン列として直列化する。"""
    return np.asarray(vector, dtype="<f4").tobytes(order="C")


def _decode_vector(payload: object, dimension: int) -> list[float] | None:
    """保存されたバイト列を次元数検証とともに float32 ベクトルへ戻す。"""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None
    raw = bytes(payload)
    if len(raw) != dimension * 4:
        return None
    try:
        values = np.frombuffer(raw, dtype="<f4", count=dimension)
    except ValueError:
        return None
    if not bool(np.all(np.isfinite(values))):
        return None
    return [float(value) for value in values]


def _batched_cosine(query: Sequence[float], stored: np.ndarray) -> np.ndarray:
    """1本の問い合わせベクトルと保存済み全ベクトルのコサイン類似度を一度の演算で求める。

    BLAS の行列積を使うため、以前のように1件ずつ Python ループで掛算・加算を繰り返さない。
    ノルムがゼロの行はスコア0として扱い、他の行の順序や値に影響を与えない。
    """
    query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if query_vector.size == 0 or stored.size == 0:
        return np.zeros(stored.shape[0], dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if not query_norm:
        return np.zeros(stored.shape[0], dtype=np.float32)
    norms = np.maximum(np.linalg.norm(stored, axis=1), 1e-12)
    normalized = stored / norms[:, None]
    scores = normalized @ (query_vector / query_norm)
    return np.where(np.isfinite(scores), scores, np.float32(0.0)).astype(np.float32, copy=False)


def _author_fragment(document: str, chunk: str, spans) -> tuple[str, str | None]:
    """一致する作者範囲との交差部分だけを返す。別人の本文を混ぜない。"""
    if not chunk:
        return "", None
    fragments = []
    message_id = None
    offset = document.find(chunk)
    while offset >= 0:
        for span in sorted(spans, key=lambda item: item.start):
            header = _HISTORY_HEADER.match(document, span.start)
            start = max(offset, header.end() if header else span.start)
            end = min(offset + len(chunk), span.end)
            if start < end:
                piece = document[start:end].strip()
                if piece and piece not in fragments:
                    fragments.append(piece)
                    message_id = message_id or span.message_id
        offset = document.find(chunk, offset + 1)
    return "\n".join(fragments), message_id
