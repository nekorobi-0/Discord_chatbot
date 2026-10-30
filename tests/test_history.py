from dataclasses import replace

import httpx
import pytest
from sqlmodel import Session, select

from ai_kun.db import HistoryCursorRecord, HistoryMessageRecord, MemoryDocumentRecord, get_engine
from ai_kun.errors import ConfigurationError, UpstreamError
from ai_kun.history import DiscordHistoryAPI, HalfRateLimiter, import_history, importer_lock


class Clock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.delays.append(seconds)
        self.now += seconds


def headers(limit=10, reset=10, remaining=9, bucket="messages"):
    return httpx.Headers(
        {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Reset-After": str(reset),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Bucket": bucket,
        }
    )


async def test_half_rate_spacing_and_bucket_allowance():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    sent = []
    for _ in range(6):
        await limiter.wait("messages", "123")
        sent.append(clock.now)
        limiter.observe("messages", "123", headers(reset=10 - (clock.now % 10)))
    assert sent == [0, 2, 4, 6, 8, 10.1]
    # 十秒枠に10回ではなく5回。初回からバーストしない。
    assert sum(t < 10 for t in sent) == 5


async def test_small_odd_bucket_stays_below_half():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    sent = []
    for _ in range(3):
        await limiter.wait("messages", "123")
        sent.append(clock.now)
        limiter.observe("messages", "123", headers(limit=5, reset=10 - (clock.now % 10)))
    assert sent == [0, 4, 10.1]


async def test_remaining_zero_waits_for_reset():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    limiter.observe("messages", "1", headers(reset=30, remaining=0))
    await limiter.wait("messages", "1")
    assert clock.now >= 30


async def test_single_request_bucket_skips_a_window():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    limiter.observe("messages", "1", headers(limit=1, reset=10, remaining=0))
    await limiter.wait("messages", "1")
    assert clock.now == 20


async def test_unknown_headers_are_conservative():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    limiter.observe("messages", "1", httpx.Headers())
    await limiter.wait("messages", "1")
    assert clock.now >= 2


async def test_bucket_identity_includes_channel():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    limiter.observe("messages", "1", headers(remaining=0))
    limiter.observe("messages", "2", headers())
    await limiter.wait("messages", "2")
    assert clock.now == 2
    await limiter.wait("messages", "1")
    assert clock.now >= 10


async def test_shared_bucket_and_global_retry_after():
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    limiter.observe("first", "1", headers())
    limiter.observe("second", "1", headers())
    assert limiter.routes[("first", "1")] == limiter.routes[("second", "1")]
    limiter.defer("first", "1", 65, True)
    await limiter.wait("unrelated", "2")
    assert clock.now >= 65


@pytest.mark.parametrize("global_limit", [True, False])
async def test_429_waits_before_retry(global_limit):
    clock = Clock()
    limiter = HalfRateLimiter(clock, clock.sleep)
    sent = []

    def handler(request):
        sent.append(clock.now)
        assert request.headers["Authorization"] == "Bot fake-token"
        assert request.url.params["limit"] == "100"
        assert request.url.params["after"] == "123"
        if len(sent) == 1:
            return httpx.Response(
                429, headers={"Retry-After": "65"}, json={"retry_after": 64.57, "global": global_limit}
            )
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        api = DiscordHistoryAPI("fake-token", client, limiter)
        assert await api.get(34567, messages=True, after=123) == []
    assert len(sent) == 2 and sent[1] - sent[0] >= 65


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_invalid_permissions_do_not_retry(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    clock = Clock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        api = DiscordHistoryAPI("fake-token", client, HalfRateLimiter(clock, clock.sleep))
        with pytest.raises(ConfigurationError, match=str(status)):
            await api.get(34567)
    assert len(calls) == 1


async def test_retry_is_bounded():
    clock = Clock()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        api = DiscordHistoryAPI("fake-token", client, HalfRateLimiter(clock, clock.sleep))
        with pytest.raises(UpstreamError):
            await api.get(34567)
    assert len(calls) == 5


def message(i, content="プロジェクトの締切は金曜日です。", **kwargs):
    return {
        "id": str(i),
        "content": content,
        "timestamp": "2026-10-30T10:00:00+00:00",
        "author": {"id": "45678"},
        "type": 0,
        **kwargs,
    }


@pytest.fixture
def embedded(store, monkeypatch):
    async def embed(texts):
        return [[1.0] + [0.0] * 63 for _ in texts]

    monkeypatch.setattr(store, "embed", embed)
    return store


async def test_group_mask_deduplicate_and_citations(embedded):
    messages = [
        message(1, "連絡先 alice@example.com"),
        message(2),
        message(3, author={"bot": True}),
        message(4, "覚えないで"),
        message(5, ""),
        message(6, "system", type=6),
    ]
    result = await embedded.ingest_history_page(23456, 34567, messages)
    assert result == {"scanned": 6, "imported": 2, "skipped": 4, "documents": 1}
    with Session(get_engine()) as db:
        cursor = db.get(HistoryCursorRecord, "23456:34567")
        assert cursor.last_message_id == "6"
        assert len(db.exec(select(HistoryMessageRecord)).all()) == 6
        doc = db.exec(select(MemoryDocumentRecord)).one()
        assert "alice@" not in doc.content and "[EMAIL]" in doc.content
        assert "2026-10-30" in doc.content and "member-" in doc.content
        assert "覚えないで" not in doc.content
        document_id = doc.document_id
    hits = await embedded.search("締切")
    assert "https://discord.com/channels/23456/34567/" in hits[0].label
    assert document_id in hits[0].label
    again = await embedded.ingest_history_page(23456, 34567, messages)
    assert again["imported"] == 0
    await embedded.delete_document(document_id)
    # 再走査しても削除したログが復活しない。
    assert (await embedded.ingest_history_page(23456, 34567, messages))["imported"] == 0
    assert await embedded.search("締切") == ()


async def test_group_by_count_and_time_gap(embedded):
    batch = [message(i) for i in range(1, 22)]
    batch.append(message(22, timestamp="2026-10-30T12:00:00+00:00"))
    result = await embedded.ingest_history_page(23456, 34567, batch)
    assert result["documents"] == 3  # 20件 + 1件 + 時間が空いた1件


async def test_embedding_failure_does_not_advance_checkpoint(store, monkeypatch):
    async def embed(texts):
        raise UpstreamError("Embedding")

    monkeypatch.setattr(store, "embed", embed)
    with pytest.raises(UpstreamError):
        await store.ingest_history_page(23456, 34567, [message(1)])
    with Session(get_engine()) as db:
        assert db.get(HistoryCursorRecord, "23456:34567") is None
        assert db.exec(select(HistoryMessageRecord)).all() == []
        assert db.exec(select(MemoryDocumentRecord)).all() == []


class FakeAPI:
    def __init__(self, guild=23456):
        self.guild = guild
        self.calls = []

    async def get(self, channel_id, *, messages=False, after=0):
        self.calls.append((channel_id, messages, after))
        if not messages:
            return {"guild_id": str(self.guild), "type": 0}
        return [message(i) for i in reversed(range(after + 1, min(after + 100, 203) + 1))]


async def test_paging_restart_and_snapshot(settings, embedded):
    api = FakeAPI()
    first = await import_history(settings, embedded, api=api, until_id=150)
    assert first["imported"] == 150
    assert [after for _, messages, after in api.calls if messages] == [0, 100]
    api.calls.clear()
    second = await import_history(settings, embedded, api=api, until_id=300)
    assert second["imported"] == 53
    assert [after for _, messages, after in api.calls if messages] == [150, 203]
    with Session(get_engine()) as db:
        assert len(db.exec(select(HistoryMessageRecord)).all()) == 203


async def test_all_skipped_page_still_advances(settings, embedded):
    class BotsOnly(FakeAPI):
        async def get(self, channel_id, *, messages=False, after=0):
            if not messages:
                return await super().get(channel_id)
            return [message(1, author={"bot": True})] if after == 0 else []

    result = await import_history(settings, embedded, api=BotsOnly(), until_id=100)
    assert result["imported"] == 0 and result["skipped"] == 1
    with Session(get_engine()) as db:
        assert db.get(HistoryCursorRecord, "23456:34567").last_message_id == "1"


async def test_wrong_guild_never_fetches_messages(settings, embedded):
    api = FakeAPI(guild=99999)
    with pytest.raises(ConfigurationError, match="GUILD_ID"):
        await import_history(settings, embedded, api=api)
    assert len(api.calls) == 1


async def test_memory_disabled_rejected(settings, embedded):
    disabled = replace(settings, memory=replace(settings.memory, enabled=False))
    with pytest.raises(ConfigurationError, match="MEMORY_ENABLED"):
        await import_history(disabled, embedded, api=FakeAPI())


def test_exclusive_importer_lock():
    with importer_lock():
        with pytest.raises(ConfigurationError, match="実行中"):
            with importer_lock():
                pytest.fail("must reject duplicate importer")
