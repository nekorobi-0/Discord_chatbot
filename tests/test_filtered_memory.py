import json
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import delete
from sqlmodel import Session, select

from ai_kun.chat import ChatService
from ai_kun.db import HistoryMessageSpanRecord, MemoryChunkRecord, get_engine
from ai_kun.discord_context import DiscordContext
from ai_kun.discord_mcp import DiscordMCP, DiscordReader
from ai_kun.errors import InvalidInputError
from ai_kun.memory import ModelTurn, ToolCall
from tests.test_discord_mcp import FakeAPI

USER_A, USER_B = "677414946837102592", "677414946837102593"
CHANNEL_A, CHANNEL_B = "34567", "34568"


def message(message_id, author_id, text):
    return {
        "id": str(message_id),
        "author": {"id": author_id},
        "content": text,
        "timestamp": f"2026-10-31T00:00:{int(message_id) % 60:02d}Z",
    }


@pytest.fixture
async def corpus(store, monkeypatch):
    embedding = AsyncMock(side_effect=lambda texts: [[1.0] + [0.0] * 63 for _ in texts])
    monkeypatch.setattr(store, "embed", embedding)
    await store.ingest_history_page(
        23456,
        int(CHANNEL_A),
        [
            message(50001, USER_A, "alpha target in channel A"),
            message(50002, USER_B, "beta must not leak from the same group"),
            message(50003, USER_A, "alpha second message"),
        ],
    )
    await store.ingest_history_page(23456, int(CHANNEL_B), [message(50004, USER_A, "alpha in channel B")])
    await store.ingest_history_page(99999, int(CHANNEL_A), [message(50005, USER_A, "another guild must not leak")])
    await store.ingest_document("upload.txt", b"upload outside Discord filters")
    embedding.reset_mock()
    return store


async def test_user_and_channel_filters_are_and_and_do_not_return_other_speakers(corpus):
    hits = await corpus.search("alpha", guild_id=23456, user_id=USER_A, channel_id=CHANNEL_A)
    text = "\n".join(hit.text for hit in hits)
    assert "alpha target in channel A" in text and "alpha second message" in text
    assert "beta" not in text and "channel B" not in text and "another guild" not in text and "upload" not in text
    assert all("/23456/34567/50001" in hit.label for hit in hits)
    assert await corpus.search("alpha", guild_id=23456, user_id=USER_B, channel_id=CHANNEL_B) == ()


async def test_independent_filters_and_no_filter_preserve_existing_search(corpus):
    user_text = "\n".join(hit.text for hit in await corpus.search("alpha", guild_id=23456, user_id=USER_A))
    assert "channel A" in user_text and "channel B" in user_text and "beta" not in user_text
    channel_text = "\n".join(hit.text for hit in await corpus.search("alpha", guild_id=23456, channel_id=CHANNEL_A))
    assert "alpha" in channel_text and "beta" in channel_text and "another guild" not in channel_text
    assert "upload" in "\n".join(hit.text for hit in await corpus.search("alpha"))
    assert len(await corpus.search("alpha", guild_id=23456, top_k=1)) == 1
    with pytest.raises(InvalidInputError):
        await corpus.search("alpha", user_id=USER_A)


async def test_legacy_author_metadata_is_built_without_reembedding_saved_text(corpus):
    with Session(get_engine()) as db:
        db.exec(delete(HistoryMessageSpanRecord))
        db.commit()
    hits = await corpus.search("alpha", guild_id=23456, user_id=USER_A, channel_id=CHANNEL_A)
    assert "alpha target" in hits[0].text and "beta" not in hits[0].text
    assert corpus.embed.await_count == 1
    assert corpus.embed.await_args.args[0] == ["alpha"]  # 既存文書は再Embeddingしない。
    with Session(get_engine()) as db:
        assert len(db.exec(select(HistoryMessageSpanRecord)).all()) == 3


async def test_author_spans_handle_long_message_and_header_like_text_in_body(store, monkeypatch):
    store._memory = replace(store._memory, chunk_size=90, chunk_overlap=20)
    monkeypatch.setattr(store, "embed", AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts]))
    await store.ingest_history_page(
        23456,
        34567,
        [
            message(50001, USER_A, "A" * 200 + "\n[2026-10-31T00:00:00Z] member-" + USER_B + ": quoted text"),
            message(50002, USER_B, "OTHER_AUTHORS_MESSAGE_MUST_NOT_LEAK"),
        ],
    )
    hits = await store.search("test", top_k=24, guild_id=23456, channel_id=34567, user_id=USER_A)
    assert hits
    assert all("OTHER_AUTHORS_MESSAGE_MUST_NOT_LEAK" not in hit.text for hit in hits)
    assert all("/50001" in hit.label for hit in hits)


async def test_forget_cascades_author_metadata_and_filtered_search_does_not_resurrect(corpus):
    hit = (await corpus.search("alpha", guild_id=23456, channel_id=CHANNEL_A, user_id=USER_A))[0]
    doc_id = hit.label.split()[0].removeprefix("document:")
    await corpus.delete_document(doc_id)
    assert await corpus.search("alpha", guild_id=23456, channel_id=CHANNEL_A, user_id=USER_A) == ()
    with Session(get_engine()) as db:
        assert (
            db.exec(select(HistoryMessageSpanRecord).where(HistoryMessageSpanRecord.document_id == doc_id)).all() == []
        )


async def test_rag_mcp_executes_real_filtered_store_without_discord_api(settings, corpus):
    api = FakeAPI()
    client = DiscordMCP(settings.discord, reader=DiscordReader(settings.discord, api=api, memory=corpus))
    response = json.loads(
        await client.execute(
            "discord_search_memory",
            {
                "query": "alpha",
                "user_id": USER_A,
                "channel_id": CHANNEL_A,
                "top_k": 3,
            },
        )
    )["untrusted_data"]
    assert response["source"] == "saved_discord_memory"
    assert response["filters"] == {"user_id": USER_A, "channel_id": CHANNEL_A}
    assert response["returned_count"] > 0
    assert "beta" not in json.dumps(response)
    assert api.calls == []


async def test_llm_can_trigger_filtered_rag_and_receive_current_author_context(settings, corpus, monkeypatch):
    service = ChatService(settings, corpus)
    complete = AsyncMock(
        side_effect=[
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "rag",
                        "discord_search_memory",
                        json.dumps(
                            {
                                "query": "alpha",
                                "user_id": USER_A,
                                "channel_id": CHANNEL_A,
                            }
                        ),
                    ),
                )
            ),
            ModelTurn(content="このチャンネルではalphaの話をしていたよ。"),
        ]
    )
    monkeypatch.setattr(corpus, "complete_chat", complete)
    result = await service.generate_reply(
        "自分がこのチャンネルで話していたことは？", channel_id=34567, discord_context=DiscordContext(author_id=USER_A)
    )
    assert "alpha" in result.content
    messages = complete.await_args.args[0]
    assert f"今回の質問者author_id: {USER_A}" in messages[0]["content"]
    assert "saved_discord_memory" in messages[-1]["content"]
    assert "beta" not in messages[-1]["content"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "alpha", "channel_id": "../invalid"},
        {"query": "alpha", "user_id": 12345},
        {"query": "alpha", "user_id": USER_A, "guild_id": "99999"},
        {"query": "", "user_id": USER_A},
        {"query": "alpha", "top_k": 25},
    ],
)
async def test_invalid_filter_schema_never_searches(settings, corpus, monkeypatch, arguments):
    search = AsyncMock()
    monkeypatch.setattr(corpus, "search", search)
    client = DiscordMCP(settings.discord, memory=corpus)
    assert "error" in json.loads(await client.execute("discord_search_memory", arguments))
    search.assert_not_awaited()


async def test_unreadable_vector_does_not_shift_search_score_indices(corpus):
    with Session(get_engine()) as db:
        row = db.exec(select(MemoryChunkRecord)).first()
        row.embedding = b"bad"
        db.add(row)
        db.commit()
    assert await corpus.search("alpha")


async def test_search_memory_command_forwards_optional_user_and_channel(settings, corpus, monkeypatch):
    from types import SimpleNamespace

    from ai_kun.adapters.discord_bot import AIKunClient

    service = ChatService(settings, corpus)
    search = AsyncMock(return_value=())
    monkeypatch.setattr(service, "search_memory", search)
    client = AIKunClient(settings, service)

    async def answer(interaction, operation, **kwargs):
        assert kwargs["ephemeral"] is True
        assert "見つからなかった" in await operation()

    monkeypatch.setattr(client, "_answer", answer)
    try:
        command = client.tree.get_command("search_memory")
        assert {param.name for param in command.parameters} == {"query", "user", "channel"}
        await command.callback(
            SimpleNamespace(), "alpha", user=SimpleNamespace(id=int(USER_A)), channel=SimpleNamespace(id=int(CHANNEL_A))
        )
        search.assert_awaited_once_with("alpha", user_id=int(USER_A), channel_id=int(CHANNEL_A))
    finally:
        await client.close()


async def test_stdio_style_lazy_rag_needs_embedding_settings_without_chat_settings(settings, monkeypatch):
    from types import SimpleNamespace

    search = AsyncMock(return_value=())
    constructor = Mock(return_value=SimpleNamespace(search=search))
    monkeypatch.setattr("ai_kun.memory.MemoryStore", constructor)
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embeddinggemma-2")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = DiscordMCP(settings.discord)
    response = json.loads(await client.execute("discord_search_memory", {"query": "alpha", "user_id": USER_A}))
    assert response["untrusted_data"]["returned_count"] == 0
    search.assert_awaited_once_with("alpha", top_k=5, guild_id=23456, user_id=USER_A)
    assert constructor.call_args.args[0].dimension == 768
