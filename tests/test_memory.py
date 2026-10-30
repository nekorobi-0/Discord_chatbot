import base64
import json
from dataclasses import replace

import httpx
import pytest
from sqlmodel import Session, select

from ai_kun.db import EmbeddingCacheRecord, MemoryChunkRecord, MemoryDocumentRecord, get_engine
from ai_kun.errors import ConfigurationError, InvalidInputError, UpstreamError
from ai_kun.memory import MemoryStore


async def test_ingest_search_restart_cache_delete(store, settings, monkeypatch):
    calls = []

    async def request(method, url, **kwargs):
        calls.append(kwargs["json"]["input"])
        return httpx.Response(200, json={"data": [{"embedding": [1.0] + [0.0] * 63}]})

    monkeypatch.setattr(store, "_request", request)
    text = "連絡先 alice@example.com。プロジェクトの締切は金曜日です。"
    result = await store.ingest_document("plan.md", text.encode())
    assert len(result["document_id"]) == 32
    assert all("alice@example.com" not in value for value in calls)
    hits = await store.search("プロジェクトの締切")
    assert "金曜日" in hits[0].text
    assert "[EMAIL]" in hits[0].text
    assert result["document_id"] in hits[0].label
    before = len(calls)
    await store.search("プロジェクトの締切")
    assert len(calls) == before  # 永続キャッシュで再送しない
    restarted = MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)
    monkeypatch.setattr(restarted, "_request", request)
    assert (await restarted.search("プロジェクトの締切"))[0].text == hits[0].text
    assert len(calls) == before
    with Session(get_engine()) as db:
        doc = db.exec(select(MemoryDocumentRecord)).first()
        assert "金曜日".encode() not in doc.content_ciphertext
        assert doc.title == "uploaded document"
    assert await restarted.delete_document(result["document_id"]) == result["chunks"]
    with Session(get_engine()) as db:
        assert db.exec(select(MemoryChunkRecord)).all() == []
        assert db.exec(select(EmbeddingCacheRecord)).all() == []
    assert await restarted.search("プロジェクトの締切") == ()
    with pytest.raises(InvalidInputError):
        await restarted.delete_document(result["document_id"])


async def test_wrong_key_fails_at_startup(store, settings, monkeypatch):
    async def embed(texts):
        return [[1.0] * 64 for text in texts]

    monkeypatch.setattr(store, "embed", embed)
    await store.ingest_document("test.txt", b"hello")
    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", base64.b64encode(b"b" * 32).decode())
    with pytest.raises(ConfigurationError, match="復号"):
        MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)


async def test_documents_masked_before_embedding(store, monkeypatch):
    captured = []

    async def embed(texts):
        captured.extend(texts)
        return [[1.0] * 64 for text in texts]

    monkeypatch.setattr(store, "embed", embed)
    result = await store.ingest_document("private.txt", b"api_key=topsecret alice@example.com")
    assert "topsecret" not in json.dumps(captured)
    assert "alice@" not in json.dumps(captured)
    assert result["chunks"] == 1


@pytest.mark.parametrize("name,data", [("x.exe", b"hello"), ("x.txt", b""), ("x.txt", b"a" * 12001)])
async def test_invalid_upload(store, name, data):
    with pytest.raises(InvalidInputError):
        await store.ingest_document(name, data)


async def test_asr_disabled(store):
    with pytest.raises(UpstreamError):
        await store.transcribe(b"voice", "voice.ogg")


async def test_only_configured_endpoints(store):
    with pytest.raises(UpstreamError):
        await store._request("GET", "https://other.example/private")


async def test_model_change_excludes_old_vectors(store, settings, monkeypatch):
    async def embed(texts):
        return [[1.0] * 64 for text in texts]

    monkeypatch.setattr(store, "embed", embed)
    await store.ingest_document("x.txt", b"old-model data")
    changed = MemoryStore(replace(settings.embedding, model="new-model"), settings.llm, settings.asr, settings.memory)
    monkeypatch.setattr(changed, "embed", embed)
    assert await changed.search("data") == ()


async def test_low_similarity_is_not_retrieved(store, monkeypatch):
    async def embed(texts):
        return [[1.0] + [0.0] * 63 for text in texts]

    monkeypatch.setattr(store, "embed", embed)
    await store.ingest_document("a.txt", b"stored text")

    async def other_embed(texts):
        return [[0.0, 1.0] + [0.0] * 62 for text in texts]

    monkeypatch.setattr(store, "embed", other_embed)
    assert await store.search("unrelated question") == ()


async def test_local_embedding_without_key(store, settings, monkeypatch):
    store._embedding = replace(settings.embedding, base_url="http://127.0.0.1:11434/v1", api_key="")

    def handler(request):
        assert request.url == "http://127.0.0.1:11434/v1/embeddings"
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * 64}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    assert await store.embed(["記憶する文章"]) == [[1.0] * 64]
