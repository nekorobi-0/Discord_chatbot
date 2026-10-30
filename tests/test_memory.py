import json
from dataclasses import replace

import httpx
import pytest
from sqlmodel import Session, select

from ai_kun.db import (
    EmbeddingCacheRecord,
    MemoryChunkRecord,
    MemoryDocumentRecord,
    get_engine,
)
from ai_kun.errors import InvalidInputError, UpstreamError
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
    assert any("alice@example.com" in value for value in calls)
    hits = await store.search("プロジェクトの締切")
    assert "金曜日" in hits[0].text
    assert "alice@example.com" in hits[0].text
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
        assert "alice@example.com" in doc.content
        assert doc.title == "uploaded document"
    assert await restarted.delete_document(result["document_id"]) == result["chunks"]
    with Session(get_engine()) as db:
        assert db.exec(select(MemoryChunkRecord)).all() == []
        assert db.exec(select(EmbeddingCacheRecord)).all() == []
    assert await restarted.search("プロジェクトの締切") == ()
    with pytest.raises(InvalidInputError):
        await restarted.delete_document(result["document_id"])


async def test_documents_preserve_original_text_before_embedding(store, monkeypatch):
    captured = []

    async def embed(texts):
        captured.extend(texts)
        return [[1.0] * 64 for text in texts]

    monkeypatch.setattr(store, "embed", embed)
    result = await store.ingest_document("private.txt", b"api_key=topsecret alice@example.com")
    assert "topsecret" in json.dumps(captured)
    assert "alice@example.com" in json.dumps(captured)
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


async def test_embedding_error_reports_status_and_model_without_response_body(store, monkeypatch, caplog):
    async def request(method, url, **kwargs):
        return httpx.Response(404, json={"error": "private echoed request alice@example.com"})

    monkeypatch.setattr(store, "_request", request)
    with pytest.raises(UpstreamError, match="Embedding HTTP 404"):
        await store._embed_uncached("alice@example.com")
    assert "status=404" in caplog.text
    assert "model=embedding-test" in caplog.text
    assert "dimensions=64" in caplog.text
    assert "alice@example.com" not in caplog.text


async def test_embedding_batches_deduplicate_preserve_order_and_reuse_cache(store, monkeypatch):
    calls = []

    async def request(method, url, **kwargs):
        data = kwargs["json"]["input"]
        texts = [data] if isinstance(data, str) else data
        calls.append(list(texts))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(text.split("-")[1])] + [0.0] * 63}
                    for index, text in reversed(list(enumerate(texts)))
                ]
            },
        )

    monkeypatch.setattr(store, "_request", request)
    texts = [f"text-{index}" for index in range(20)] + ["text-3", "text-0"]
    vectors = await store.embed(texts)
    assert [vector[0] for vector in vectors] == list(range(20)) + [3, 0]
    assert len(calls) == 2
    assert sum(map(len, calls)) == 20
    assert await store.embed(list(reversed(texts))) == list(reversed(vectors))
    assert len(calls) == 2


@pytest.mark.parametrize(
    "data",
    [
        {"data": [{"index": 0, "embedding": [1.0] * 64}]},
        {"data": [{"index": 0, "embedding": [1.0] * 64}, {"index": 0, "embedding": [1.0] * 64}]},
        {"data": [{"index": 0, "embedding": [1.0] * 63}, {"index": 1, "embedding": [1.0] * 64}]},
    ],
)
async def test_invalid_batch_embeddings_are_not_saved(store, monkeypatch, data):
    async def request(method, url, **kwargs):
        return httpx.Response(200, json=data)

    monkeypatch.setattr(store, "_request", request)
    with pytest.raises(UpstreamError):
        await store.embed(["first", "second"])
    with Session(get_engine()) as session:
        assert session.exec(select(EmbeddingCacheRecord)).all() == []
