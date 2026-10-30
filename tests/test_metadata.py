"""目録（メタデータ）による検索絞り込みの実装確認。

話者ID・チャンネルID・添付ファイルの有無といった付属条件でしか引けないこと、
条件を満たさない項目が検索結果から落ちること、平文が目録として残らないことを確認する。
"""

import base64
import io
import logging

import pytest
from sqlmodel import Session, select

from ai_kun.chat import ChatService
from ai_kun.crypto import KeyProvider
from ai_kun.db import (
    MemoryChunkRecord,
    MemoryDocumentRecord,
    MetadataDocumentLinkRecord,
    MetadataEnvelopeRecord,
    get_engine,
)
from ai_kun.errors import InvalidInputError
from ai_kun.metadata import ATTACHMENT_IMAGE, MetadataEnvelope, MetadataGrant, classify_attachment, detect_flags


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


def test_attachment_classification_is_kind_only():
    assert classify_attachment("https://cdn.discordapp.com/attachments/1/2/photo.PNG") == ATTACHMENT_IMAGE
    assert detect_flags([message(1, attachments=[{"url": "https://cdn.discordapp.com/x/voice_message.ogg"}])]) & 4
    # 想定外のURL（CDN以外）は添付として数えない。
    assert classify_attachment("https://evil.example/private.png") == 0


def test_envelope_record_never_serializes_or_logs_plaintext():
    """目録レコードは model_dump・str・repr・ログ出力のどれにも平文を出さない。"""
    secret = "SUPER-SECRET-PLAINTEXT"
    record = MetadataEnvelopeRecord(envelope_id="e1", sealed=secret.encode())
    carriers = [record.model_dump(), str(record), repr(record)]
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    probe = logging.getLogger("ai_kun.test.probe")
    probe.addHandler(handler)
    probe.setLevel(logging.INFO)
    try:
        probe.info("record: %s", record)
    finally:
        probe.removeHandler(handler)
    carriers.append(buf.getvalue())
    assert all(secret not in str(carrier) for carrier in carriers)


async def test_history_metadata_sealed_and_not_plaintext(embedded):
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    with Session(get_engine()) as db:
        envelopes = db.exec(select(MetadataEnvelopeRecord)).all()
        links = db.exec(select(MetadataDocumentLinkRecord)).all()
    assert len(envelopes) == 1 and len(links) == 1
    sealed = envelopes[0].sealed
    # 暗号化された封書に、話者ID・チャンネルIDが平文で残らない。
    for secret in (b"45678", b"34567", b"23456"):
        assert secret not in bytes(sealed)
    assert "sealed" not in MetadataEnvelopeRecord(envelope_id="x", sealed=b"y").model_dump()


async def test_search_requires_a_grant_object(embedded):
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    with pytest.raises(InvalidInputError):
        await embedded.search("締切", "not-a-grant")  # type: ignore[arg-type]


async def test_unfiltered_search_still_works(embedded):
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    hits = await embedded.search("締切", MetadataGrant())
    assert hits and "金曜日" in hits[0].text
    assert "https://discord.com/channels/23456/34567/1" in hits[0].label
    with Session(get_engine()) as db:
        assert embedded._open(db.exec(select(MemoryDocumentRecord)).first().content_ciphertext)


async def test_search_filtered_by_user_id(embedded):
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    match = await embedded.search("締切", MetadataGrant(user_ids=frozenset({"45678"})))
    assert match and "金曜日" in match[0].text
    # 別ユーザーだけを対象にすると何も出てこない（条件を満たさない項目は落とす）。
    assert await embedded.search("締切", MetadataGrant(user_ids=frozenset({"99999"}))) == ()


async def test_two_speakers_are_isolated_per_document(embedded):
    """話者が違えば別文書＝別目録。話者IDで絞ると該当発言だけが出て、他話者は漏れない。"""
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    await embedded.ingest_history_page(23456, 34567, [message(2, author={"id": "88888"})])
    both = await embedded.search("締切", MetadataGrant())
    assert {hit.label.split("/")[-1] for hit in both} == {"1", "2"}
    only_a = await embedded.search("締切", MetadataGrant(user_ids=frozenset({"45678"})))
    assert {hit.label.split("/")[-1] for hit in only_a} == {"1"}
    only_b = await embedded.search("締切", MetadataGrant(user_ids=frozenset({"88888"})))
    assert {hit.label.split("/")[-1] for hit in only_b} == {"2"}


async def test_search_filtered_by_channel_id(embedded):
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    assert await embedded.search("締切", MetadataGrant(channel_ids=frozenset({"34567"})))
    assert await embedded.search("締切", MetadataGrant(channel_ids=frozenset({"44444"}))) == ()


async def test_search_filtered_by_attachment_kind(embedded):
    with_image = message(1, attachments=[{"url": "https://cdn.discordapp.com/attachments/1/2/a.png"}])
    without = message(2)
    await embedded.ingest_history_page(23456, 34567, [with_image, without])
    only_images = await embedded.search("締切", MetadataGrant(require_image=True))
    assert only_images and "https://discord.com/channels/23456/34567/1" in only_images[0].label
    # 画像添付のない記録は、画像指定の検索では返さない。
    assert all("/34567/2" not in hit.label for hit in only_images)
    assert await embedded.search("締切", MetadataGrant(require_video=True)) == ()


async def test_unreadable_envelope_fails_closed(embedded):
    """目録が読めない項目は、条件なし検索でも安全側（非表示）で扱う。"""
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    assert await embedded.search("締切", MetadataGrant())
    with Session(get_engine()) as db:
        for row in db.exec(select(MetadataEnvelopeRecord)).all():
            row.sealed = base64.b64encode(b"unreadable")
        db.commit()
    assert await embedded.search("締切", MetadataGrant()) == ()


async def test_grant_normalization_rejects_too_many_ids():
    too_many = ",".join(str(i) for i in range(33))
    with pytest.raises(InvalidInputError):
        MetadataEnvelope.normalize_user_ids(too_many)
    assert MetadataEnvelope.normalize_channel_ids("1;2, 3") == frozenset({"1", "2", "3"})
    assert MetadataGrant().empty


async def test_delete_purges_envelopes_and_links(embedded):
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    with Session(get_engine()) as db:
        doc_id = db.exec(select(MemoryDocumentRecord)).first().document_id
    await embedded.delete_document(doc_id)
    with Session(get_engine()) as db:
        assert db.exec(select(MetadataEnvelopeRecord)).all() == []
        assert db.exec(select(MetadataDocumentLinkRecord)).all() == []
        assert db.exec(select(MemoryChunkRecord)).all() == []


async def test_uploaded_documents_are_indexed_for_filtering(embedded):
    result = await embedded.ingest_document("plan.md", "締切は金曜日".encode())
    with Session(get_engine()) as db:
        assert db.exec(select(MetadataEnvelopeRecord)).all() != []
        assert db.exec(select(MetadataDocumentLinkRecord)).all() != []
    # 資料は話者・チャンネル目録を持たないため、ID指定の絞り込みでは出てこない。
    assert await embedded.search("締切", MetadataGrant(user_ids=frozenset({"45678"}))) == ()
    assert result["document_id"] in (await embedded.search("締切", MetadataGrant()))[0].label


async def test_service_grant_and_search_memory(settings, store, embedded):
    service = ChatService(settings, embedded)
    grant = service.build_grant(user_ids="45678", has_image=False)
    assert grant.user_ids == frozenset({"45678"}) and not grant.empty
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    hits = await service.search_memory("締切", grant)
    assert hits and "金曜日" in hits[0].text
    assert await service.search_memory("締切", MetadataGrant(user_ids=frozenset({"1"}))) == ()


async def test_envelopes_are_readable_only_with_the_key(embedded, monkeypatch):
    """封書は設定済みの暗号鍵でだけ読める。別の鍵では目録を読み取れない。"""
    await embedded.ingest_history_page(23456, 34567, [message(1)])
    with Session(get_engine()) as db:
        sealed = db.exec(select(MetadataEnvelopeRecord)).first().sealed
    payload = embedded._envelope.open(sealed)
    assert payload["channel_id"] == "34567" and payload["authors"] == ["45678"]

    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", base64.b64encode(b"b" * 32).decode())
    other_envelope = MetadataEnvelope(KeyProvider())
    with pytest.raises(Exception):
        other_envelope.open(sealed)  # 別の鍵では読み取れない
