"""AIくんの記憶（RAG）レイヤー。

- 保存するテキストは暗号化し、平文はディスクに残さない
- 検索はベクトル近傍のみ。スコア下限より下なら「覚えていない」と答える
- LLM / Embedding は互換API、ASRはローカル推論
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy import delete
from sqlmodel import Session, select

from .config import ASRSettings, EmbeddingSettings, LLMSettings, MemorySettings, mask_pii
from .crypto import KeyProvider
from .db import (
    EmbeddingCacheRecord,
    HistoryCursorRecord,
    HistoryMessageRecord,
    MemoryChunkRecord,
    MemoryDocumentRecord,
    MetadataDocumentLinkRecord,
    MetadataEnvelopeRecord,
    get_engine,
    init_db,
)
from .errors import ConfigurationError, InvalidInputError, RateLimitError, UpstreamError
from .metadata import MetadataEnvelope, MetadataGrant, detect_flags
from .observability import metrics

logger = logging.getLogger("ai_kun.memory")

_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_DOC_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".log", ".csv", ".json", ".yaml", ".yml"})
_VOICE_EXTENSIONS = frozenset({".ogg", ".mp4", ".webm", ".mkv", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus"})
_CHUNK_MARKER_RE = re.compile(r"^\s*\[\[\d{1,3}/\d{1,3}\]\]\s*$", re.MULTILINE)
_CONTROL_CHARS_RE = re.compile(r"[^\S\n\r\t\u00a0-\u1680\u2000-\u2fff]+")


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


class MemoryStore:
    """暗号化記憶の保存・検索・推論をまとめる唯一の窓口。"""

    def __init__(
        self, embedding: EmbeddingSettings, llm: LLMSettings, asr: ASRSettings, memory: MemorySettings
    ) -> None:
        self._embedding = embedding
        self._llm = llm
        self._asr = asr
        self._memory = memory
        self._crypto = KeyProvider() if memory.enabled else None
        self._envelope = MetadataEnvelope(self._crypto) if memory.enabled else None
        init_db()
        if self._crypto:
            with Session(get_engine()) as session:
                sample = session.exec(select(MemoryDocumentRecord)).first()
            if sample:
                try:
                    self._open(sample.content_ciphertext)
                except Exception as exc:
                    raise ConfigurationError(
                        "保存済み記憶を復号できません。元の暗号鍵とDBを確認してください。"
                    ) from exc

    # ---------- 暗号化 ----------
    def _seal(self, text: str) -> bytes:
        return self._seal_bytes(text.encode("utf-8"))

    def _open(self, sealed: bytes | str) -> str:
        if self._crypto is None:
            raise UpstreamError("記憶")
        return self._crypto.open(sealed).decode("utf-8")

    # ---------- HTTP（SSRF 対策つき）----------
    async def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        if not any(
            url == base.rstrip("/") + suffix
            for base, suffix in ((self._llm.base_url, "/chat/completions"), (self._embedding.base_url, "/embeddings"))
        ):
            raise UpstreamError("外部サービス")
        try:
            async with httpx.AsyncClient(timeout=float(self._llm.timeout_seconds), follow_redirects=False) as client:
                return await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            logger.warning("upstream transport error: %s", type(exc).__name__)
            raise UpstreamError("外部サービス") from exc

    # ---------- LLM 推論 ----------
    async def complete_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        response = await self._request(
            "POST", f"{self._llm.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers
        )
        if response.status_code == 429:
            raise RateLimitError()
        if response.status_code >= 400:
            logger.warning("llm http error", extra={"status": response.status_code})
            raise UpstreamError("おしゃべりの仕組み")
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
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(
                f"{self._embedding.model}:{self._embedding.dimension}\n{text}".encode("utf-8", "ignore")
            ).hexdigest()
            with Session(get_engine()) as session:
                cached = session.exec(select(EmbeddingCacheRecord).where(EmbeddingCacheRecord.digest == digest)).first()
            if cached is not None:
                try:
                    vector = json.loads(self._open(cached.vector_ciphertext))
                except Exception:
                    logger.warning("embedding cache entry unreadable")
                    vector = None
                if isinstance(vector, list) and len(vector) == self._embedding.dimension:
                    vectors.append([float(v) for v in vector])
                    continue
            vector = await self._embed_uncached(text)
            payload = json.dumps(vector).encode("ascii")
            with Session(get_engine()) as session:
                session.add(
                    EmbeddingCacheRecord(
                        digest=digest, model=self._embedding.model, vector_ciphertext=self._seal_bytes(payload)
                    )
                )
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.warning("embedding cache write failed")
            vectors.append(vector)
        return vectors

    def _seal_bytes(self, payload: bytes) -> bytes:
        if self._crypto is None:
            raise UpstreamError("記憶")
        return self._crypto.seal(payload)

    async def _embed_uncached(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self._embedding.api_key}"} if self._embedding.api_key else {}
        response = await self._request(
            "POST",
            f"{self._embedding.base_url.rstrip('/')}/embeddings",
            json={"model": self._embedding.model, "input": mask_pii(text)[0], "dimensions": self._embedding.dimension},
            headers=headers,
        )
        if response.status_code == 429:
            raise RateLimitError()
        if response.status_code >= 400:
            logger.warning("embedding http error", extra={"status": response.status_code})
            raise UpstreamError("記憶のしくみ")
        try:
            data = response.json()
        except ValueError as exc:
            raise UpstreamError("記憶のしくみ") from exc
        items = data.get("data") or []
        if not items or not isinstance(items[0].get("embedding"), list):
            raise UpstreamError("記憶のしくみ")
        vector = [float(v) for v in items[0]["embedding"]]
        if len(vector) != self._embedding.dimension:
            logger.warning(
                "embedding dimension mismatch", extra={"expected": self._embedding.dimension, "got": len(vector)}
            )
            raise UpstreamError("記憶のしくみ")
        return vector

    # ---------- 音声（CPU / faster-whisper / ローカルモデルのみ）----------
    async def transcribe(self, data: bytes, filename: str | None) -> str:
        if not self._asr.enabled:
            raise UpstreamError("音声認識")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise InvalidInputError("音声が長すぎたよ。短く区切ってもう一回おねがい。")
        name = (filename or "voice.ogg").lower()
        if not name.endswith(tuple(_VOICE_EXTENSIONS)):
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
    async def ingest_document(self, filename: str | None, data: bytes) -> dict[str, int | str]:
        if not self._memory.enabled:
            raise UpstreamError("記憶")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise InvalidInputError("ファイルが大きすぎたよ。もっと小さくしてほしいな。")
        name = (filename or "document.txt").lower()
        if not name.endswith(tuple(_DOC_EXTENSIONS)):
            raise InvalidInputError("この形式の資料はまだ扱えないよ。テキスト形式にしてくれる？")
        try:
            text = io.BytesIO(data).read().decode("utf-8", "replace")
        except Exception as exc:
            raise InvalidInputError("資料を読み取れなかったよ。") from exc
        if len(text) > self._memory.max_chars_per_document:
            raise InvalidInputError("資料が長すぎたよ。分割してほしいな。")
        text = mask_pii(_sanitize_text(text))[0][: self._memory.max_chars_per_document]
        if not text.strip():
            raise InvalidInputError("からっぽの資料だったよ。")

        document_id = uuid.uuid4().hex
        chunks: list[tuple[str, str]] = []
        step = max(1, self._memory.chunk_size - self._memory.chunk_overlap)
        for start in range(0, len(text), step):
            piece = text[start : start + self._memory.chunk_size].strip()
            if piece:
                chunks.append((uuid.uuid4().hex, piece))
            if start + self._memory.chunk_size >= len(text):
                break

        vectors = await self.embed([piece for _, piece in chunks])
        # 話者ID・チャンネルIDは使わないが、「どの文書か」を暗号化した目録を作り、検索時の照合に使う。
        envelope_sealed = self._envelope.seal({"document_id": document_id, "flags": 0})
        with Session(get_engine()) as session:
            session.add(
                MemoryDocumentRecord(
                    document_id=document_id,
                    source="upload",
                    title="uploaded document",
                    content_ciphertext=self._seal(text),
                    created_at=datetime.now(UTC),
                )
            )
            session.flush()
            # 目録封書の主キーは別枠のランダムID。文書との対応はリンク行だけで表す。
            envelope_id = uuid.uuid4().hex
            session.add(MetadataEnvelopeRecord(envelope_id=envelope_id, sealed=envelope_sealed))
            # 封書行を先に確定させてからリンクを書く（挿入順序の違いで外部キーに引っ掛けない）。
            session.flush()
            session.add(MetadataDocumentLinkRecord(envelope_id=envelope_id, document_id=document_id))
            for (chunk_id, piece), vector in zip(chunks, vectors, strict=True):
                masked, _ = mask_pii(piece)
                session.add(
                    MemoryChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content_ciphertext=self._seal(masked),
                        embedding_ciphertext=self._seal_bytes(json.dumps(vector).encode("ascii")),
                        model=self._embedding.model,
                        dimension=self._embedding.dimension,
                        created_at=datetime.now(UTC),
                    )
                )
            session.commit()
        return {"chunks": len(chunks), "document_id": document_id}

    async def ingest_history_page(self, guild_id: int, channel_id: int, messages: list[dict]) -> dict[str, int]:
        """20発言・30分・文字数を目安にまとめる。ページ保存と進捗は原子的に更新。"""
        from .chat import _is_memory_opt_out, _looks_like_secret

        if not self._memory.enabled:
            raise UpstreamError("記憶")
        if not messages:
            return {"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}
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
        for message in messages:
            message_id = str(message["id"])
            if message_id in seen:
                continue
            author = message.get("author") or {}
            text = str(message.get("content") or "").strip()
            if (
                author.get("bot")
                or message.get("webhook_id")
                or message.get("type", 0) not in {0, 19}
                or not text
                or _is_memory_opt_out(text)
            ):
                skipped.append(message_id)
                continue
            text = mask_pii(_sanitize_text(re.sub(r"<@!?[0-9]+>", "[メンバー]", text)))[0]
            if _looks_like_secret(text):
                skipped.append(message_id)
                continue
            try:
                timestamp = datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timezone missing")
            except (KeyError, TypeError, ValueError):
                raise UpstreamError("Discord履歴の日時") from None
            member = hashlib.sha256(str(author.get("id", "unknown")).encode()).hexdigest()[:8]
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
            vectors = await self.embed(pieces)
            prepared.append((document_id, text, group, pieces, vectors))

        # Embedding失敗時にはここへ到達しないため、カーソルも進まず同じページから再開できる。
        now = datetime.now(UTC)
        flags = detect_flags(messages)
        authors = []
        for message in messages:
            author = str((message.get("author") or {}).get("id") or "")
            if author and author not in authors:
                authors.append(author)
        payload = {
            "guild_id": str(guild_id),
            "channel_id": str(channel_id),
            "authors": authors,
            "flags": flags,
        }
        # 目録は1本の暗号化封書としてだけ保存する。話者ID・チャンネルIDは平文で行を作らない。
        sealed_envelopes = [self._envelope.seal(payload) for _ in prepared]
        with Session(get_engine()) as session:
            for document_id, text, group, pieces, vectors in prepared:
                session.add(
                    MemoryDocumentRecord(
                        document_id=document_id,
                        source="discord_history",
                        title=f"channel:{channel_id}",
                        content_ciphertext=self._seal(text),
                        created_at=now,
                    )
                )
            session.flush()
            # 文書行を確定させたうえで、目録封書と対応付けを同じトランザクションに書く。
            for sealed, (document_id, *_rest) in zip(sealed_envelopes, prepared, strict=True):
                envelope_id = uuid.uuid4().hex
                session.add(MetadataEnvelopeRecord(envelope_id=envelope_id, sealed=sealed))
                # 封書行を先に確定させてからリンクを書く（挿入順序の違いで外部キーに引っ掛けない）。
                session.flush()
                session.add(MetadataDocumentLinkRecord(envelope_id=envelope_id, document_id=document_id))
            for document_id, text, group, pieces, vectors in prepared:
                for piece, vector in zip(pieces, vectors, strict=True):
                    session.add(
                        MemoryChunkRecord(
                            chunk_id=uuid.uuid4().hex,
                            document_id=document_id,
                            content_ciphertext=self._seal(piece),
                            embedding_ciphertext=self._seal_bytes(json.dumps(vector).encode()),
                            model=self._embedding.model,
                            dimension=self._embedding.dimension,
                            created_at=now,
                        )
                    )
                for message_id, _ in group:
                    session.add(
                        HistoryMessageRecord(
                            message_id=message_id,
                            guild_id=str(guild_id),
                            channel_id=str(channel_id),
                            document_id=document_id,
                        )
                    )
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
            "skipped": len(skipped) + len(seen),
            "documents": len(groups),
        }

    async def search(
        self, query: str, grant: MetadataGrant, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        """埋め込み近傍検索。目録（話者ID・チャンネルID・添付種別）の条件を満たす項目だけ返す。

        grant が空でも「条件なし」として全件対象になるだけ。目録がある項目は必ず
        matches() を通らなければ結果から落とすため、絞り込み要求を無視して返さない。
        """
        if not self._memory.enabled:
            return ()
        if not isinstance(grant, MetadataGrant):
            raise InvalidInputError("検索の絞り込み条件を読み取れなかったよ。もう一回おねがい。")
        text = mask_pii(_sanitize_text(query))[0]
        if not text:
            return ()
        limit = min(int(top_k or self._memory.top_k), 24)
        with Session(get_engine()) as session:
            rows = session.exec(
                select(MemoryChunkRecord).where(
                    MemoryChunkRecord.model == self._embedding.model,
                    MemoryChunkRecord.dimension == self._embedding.dimension,
                )
            ).all()
        if not rows:
            return ()
        query_vector = (await self.embed([text]))[0]
        scored: list[tuple[float, MemoryChunkRecord]] = []
        for row in rows:
            try:
                vector = json.loads(self._open(row.embedding_ciphertext))
            except Exception:
                logger.warning("stored memory entry unreadable")
                continue
            if not isinstance(vector, list) or len(vector) != self._embedding.dimension:
                continue
            score = _cosine(query_vector, [float(v) for v in vector])
            if score < self._memory.min_score:
                continue
            with Session(get_engine()) as session:
                linked = session.exec(
                    select(MetadataEnvelopeRecord)
                    .join(MetadataDocumentLinkRecord, MetadataDocumentLinkRecord.envelope_id == MetadataEnvelopeRecord.envelope_id)
                    .where(MetadataDocumentLinkRecord.document_id == row.document_id)
                ).first()
            if linked is not None:
                # 目録がある項目は封書を検証して条件照合。読めない封書の項目は安全側で落とす。
                try:
                    payload = self._envelope.open(linked.sealed)
                except Exception:
                    logger.warning("stored memory index unreadable")
                    continue
                if not MetadataEnvelope.matches(payload, grant):
                    metrics.increment("ai_kun.retrieval_filtered_out_total")
                    continue
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievedChunk] = []
        for _, row in scored[:limit]:
            try:
                text_chunk = self._open(row.content_ciphertext)
            except Exception:
                logger.warning("stored memory entry unreadable")
                continue
            with Session(get_engine()) as session:
                origin = session.exec(
                    select(HistoryMessageRecord).where(HistoryMessageRecord.document_id == row.document_id)
                ).first()
            label = f"document:{row.document_id}"
            if origin:
                label += f" https://discord.com/channels/{origin.guild_id}/{origin.channel_id}/{origin.message_id}"
            results.append(RetrievedChunk(label=label, text=text_chunk))
        return tuple(results)

    async def delete_document(self, document_id: str) -> int:
        doc_id = str(document_id)[:64]
        with Session(get_engine()) as session:
            # 先に目録封書と対応付けを消す（文書行より先に消さないと外部キーで削除自体が止まる）。
            linked = session.exec(
                select(MetadataDocumentLinkRecord).where(MetadataDocumentLinkRecord.document_id == doc_id)
            ).all()
            for row in linked:
                session.delete(row)
                session.flush()
                envelope = session.get(MetadataEnvelopeRecord, row.envelope_id)
                if envelope is not None:
                    session.delete(envelope)
            session.flush()
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


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
