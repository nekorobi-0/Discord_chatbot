import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_kun.chat import ChatService
from ai_kun.db import init_db
from ai_kun.discord_mcp import DiscordMCP, DiscordReader
from ai_kun.discord_users import public_user_names, remember_users, resolve_users, search_users
from ai_kun.memory import ModelTurn, ToolCall
from tests.test_discord_mcp import FakeAPI

USER_ID = "677414946837102592"
OTHER_ID = "677414946837102593"


def seed():
    init_db()
    remember_users(
        23456,
        [
            {"id": USER_ID, "username": "humming", "global_name": "ハミング", "nick": "はみんぐ"},
            {"id": OTHER_ID, "username": "other", "global_name": "ハミング"},
        ],
    )


async def test_user_lookup_uses_persistent_guild_cache_without_discord_api(settings):
    seed()
    remember_users(99999, [{"id": USER_ID, "nick": "別サーバーの名前"}])
    api = FakeAPI()
    mcp = DiscordMCP(settings.discord, reader=DiscordReader(settings.discord, api=api))
    result = json.loads(await mcp.execute("discord_resolve_users", {"user_ids": [USER_ID, "888888888888888888"]}))
    users = result["untrusted_data"]["users"]
    assert users == [
        {"user_id": USER_ID, "display_name": "はみんぐ", "resolved": True},
        {"user_id": "888888888888888888", "display_name": None, "resolved": False},
    ]
    assert api.calls == []
    # 別のクライアントでもDBキャッシュを再利用する。
    restarted = DiscordMCP(settings.discord, reader=DiscordReader(settings.discord, api=api))
    result = json.loads(await restarted.execute("discord_search_users", {"query": "ＨＵＭＭＩＮＧ", "top_k": 1}))
    assert result["untrusted_data"]["candidates"][0]["user_id"] == USER_ID
    assert result["untrusted_data"]["candidates"][0]["distance"] == 0
    assert api.calls == []


def test_reverse_lookup_hamming_top_k_and_ambiguity_even_at_k_one():
    seed()
    ranked = search_users(23456, "はみんく", 2)
    assert ranked["candidates"][0]["user_id"] == USER_ID
    assert ranked["candidates"][0]["distance"] == 1
    assert len(ranked["candidates"]) == 2
    tied = search_users(23456, "ハミング", 1)
    assert tied["candidates"][0]["distance"] == 0
    assert tied["ambiguous"] is True
    assert search_users(99999, "はみんぐ", 5)["candidates"] == []


def test_missing_nickname_does_not_erase_cached_nickname_but_explicit_clear_does():
    seed()
    remember_users(23456, [{"id": USER_ID, "username": "renamed", "global_name": "新表示名"}])
    assert resolve_users(23456, [USER_ID])["users"][0]["display_name"] == "はみんぐ"
    remember_users(23456, [{"id": USER_ID, "nick": None}])
    assert resolve_users(23456, [USER_ID])["users"][0]["display_name"] == "新表示名"
    remember_users(23456, [{"id": USER_ID, "global_name": None}])
    assert resolve_users(23456, [USER_ID])["users"][0]["display_name"] == "renamed"


def test_public_answer_replaces_markers_mentions_raw_user_ids_but_keeps_message_ids():
    seed()
    unknown = "888888888888888888"
    message_id = "999999999999999999"
    text = f"member-{USER_ID} / <@!{USER_ID}> / {USER_ID} / member-{unknown} / {unknown} / message={message_id}"
    rendered = public_user_names(text, 23456, sources=[{"author_id": unknown, "message_id": message_id}])
    assert rendered.count("はみんぐ") == 3
    assert rendered.count("あるメンバー") == 2
    assert USER_ID not in rendered and unknown not in rendered
    assert message_id in rendered
    assert public_user_names(f"member-{USER_ID}", 99999) == "あるメンバー"


async def test_live_message_caches_nickname_even_without_text(settings, store):
    service = ChatService(settings, store)
    message = SimpleNamespace(
        author=SimpleNamespace(id=int(USER_ID), name="humming", global_name="ハミング", nick="はみんぐ", bot=False),
        content="",
        created_at=datetime.now(UTC),
    )
    await service.ingest_message(message)
    assert resolve_users(23456, [USER_ID])["users"][0]["display_name"] == "はみんぐ"


async def test_duplicate_history_scan_can_learn_author_name_without_reembedding(store, monkeypatch):
    monkeypatch.setattr(store, "embed", AsyncMock(return_value=[[1.0] * 64]))
    message = {"id": "50001", "author": {"id": USER_ID}, "content": "test", "timestamp": "2026-10-31T00:00:00Z"}
    await store.ingest_history_page(23456, 34567, [message])
    message["author"]["global_name"] = "ハミング"
    result = await store.ingest_history_page(23456, 34567, [message])
    assert result["imported"] == 0
    assert resolve_users(23456, [USER_ID])["users"][0]["display_name"] == "ハミング"
    assert sum(bool(call.args[0]) for call in store.embed.await_args_list) == 1


async def test_mcp_message_reads_populate_author_cache(settings):
    api = FakeAPI()
    mcp = DiscordMCP(settings.discord, reader=DiscordReader(settings.discord, api=api))
    await mcp.execute("discord_get_message", {"channel_id": "34567", "message_id": "50001"})
    response = json.loads(await mcp.execute("discord_resolve_users", {"user_ids": ["11111"]}))
    assert response["untrusted_data"]["users"][0]["display_name"] == "member"
    assert len(api.calls) == 2


async def test_llm_tool_lookup_and_final_output_cannot_expose_user_ids(settings, store, monkeypatch):
    seed()
    service = ChatService(settings, store)
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(
        side_effect=[
            ModelTurn(tool_calls=(ToolCall("who", "discord_resolve_users", json.dumps({"user_ids": [USER_ID]})),)),
            ModelTurn(content=f"member-{USER_ID}（{USER_ID}）の発言だね。<@{USER_ID}>"),
        ]
    )
    monkeypatch.setattr(store, "complete_chat", complete)
    reply = await service.generate_reply("誰の発言？")
    assert USER_ID not in reply.content
    assert reply.content == "はみんぐ（はみんぐ）の発言だね。はみんぐ"
    assert "display_name" in complete.await_args.args[0][-1]["content"]


@pytest.mark.parametrize(
    "name,args",
    [
        ("discord_resolve_users", {"user_ids": ["../invalid"]}),
        ("discord_resolve_users", {"user_ids": []}),
        ("discord_resolve_users", {"user_ids": [USER_ID], "guild_id": "99999"}),
        ("discord_search_users", {"query": "", "top_k": 5}),
        ("discord_search_users", {"query": "name", "top_k": 21}),
        ("discord_search_users", {"query": "name", "top_k": True}),
    ],
)
async def test_invalid_user_tools_do_not_call_discord(settings, name, args):
    api = FakeAPI()
    mcp = DiscordMCP(settings.discord, reader=DiscordReader(settings.discord, api=api))
    assert "error" in json.loads(await mcp.execute(name, args))
    assert api.calls == []
