import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.adapters.reply_chain import collect_reply_chain
from ai_kun.chat import ChatService
from ai_kun.db import init_db
from ai_kun.discord_mcp import DiscordReader
from ai_kun.discord_users import author_context, prepare_author_contexts, remember_users
from ai_kun.memory import ModelTurn
from tests.test_adapters import file_message
from tests.test_discord_mcp import FakeAPI
from tests.test_reply_chain import ancestor, reference

USER_ID = "677414946837102592"


def member(user_id=USER_ID, *, nick="サーバーの呼び名", username="account_name", global_name="全体表示名", bot=False):
    return SimpleNamespace(
        id=int(user_id),
        name=username,
        global_name=global_name,
        nick=nick,
        display_name=nick or global_name or username,
        bot=bot,
    )


@pytest.mark.parametrize(
    "nick,global_name,expected",
    [
        ("サーバーの呼び名", "全体表示名", "サーバーの呼び名"),
        (None, "全体表示名", "全体表示名"),
        (None, None, "account_name"),
    ],
)
def test_sdk_and_rest_share_profile_format_and_display_name_priority(nick, global_name, expected):
    sdk = author_context(member(nick=nick, global_name=global_name))
    rest = author_context(
        {"id": USER_ID, "username": "account_name", "global_name": global_name}, member={"nick": nick}
    )
    assert sdk == rest
    assert sdk["display_name"] == expected
    assert sdk["nick"] == nick and sdk["global_name"] == global_name and sdk["username"] == "account_name"


def test_missing_author_names_are_unknown_without_synthetic_member_id():
    profile = author_context({"id": USER_ID})
    assert profile["display_name"] is None and profile["nick"] is None and profile["username"] is None
    assert "member-" not in str(profile)
    assert author_context(None)["display_name"] is None
    assert author_context(SimpleNamespace(bot=False))["id"] == ""


def test_rest_cache_fills_missing_fields_and_respects_explicit_nickname_removal():
    init_db()
    remember_users(
        23456, [{"id": USER_ID, "username": "old_account", "global_name": "全体表示名", "nick": "サーバーの呼び名"}]
    )
    remember_users(99999, [{"id": USER_ID, "nick": "別サーバーの名前"}])
    first = prepare_author_contexts(23456, [{"id": USER_ID, "username": "new_account"}])[0]
    assert first["display_name"] == "サーバーの呼び名" and first["username"] == "new_account"
    cleared = prepare_author_contexts(23456, [{"id": USER_ID, "nick": None}])[0]
    assert cleared["nick"] is None and cleared["display_name"] == "全体表示名"
    cleared_global = prepare_author_contexts(23456, [{"id": USER_ID, "global_name": None}])[0]
    assert cleared_global["global_name"] is None and cleared_global["display_name"] == "new_account"
    assert prepare_author_contexts(99999, [{"id": USER_ID}])[0]["display_name"] == "別サーバーの名前"


async def test_all_reply_ancestors_and_current_author_keep_their_own_profiles():
    root = ancestor(60001, "最初の発言")
    root.author = member("11111", nick="最初の人", username="root")
    middle = ancestor(60002, "Botの返答", parent=reference(root.id, resolved=root))
    middle.author = member("22222", nick="AIくん", username="bot_account", bot=True)
    current = file_message("!ai 続けて", [])
    current.channel.fetch_message = AsyncMock()
    current.author = member("33333", nick="今回の人", username="current")
    current.reference = reference(middle.id, resolved=middle)
    context = await collect_reply_chain(current, middle)
    assert context.author["nick"] == "今回の人"
    assert [m["author"]["nick"] for m in context.messages] == ["最初の人", "AIくん"]
    assert [m["author"]["username"] for m in context.messages] == ["root", "bot_account"]
    assert context.messages[-1]["author"]["bot"] is True
    current.channel.fetch_message.assert_not_awaited()


async def test_current_and_chain_profile_json_reach_llm_without_leaking_user_id(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(return_value=ModelTurn(content=f"{USER_ID}への返答"))
    monkeypatch.setattr(store, "complete_chat", complete)
    root = ancestor(60001, "前の質問")
    root.author = member("11111", nick="先の質問者", username="previous")
    current = file_message("<@222> 続けて", [])
    current.author = member(nick="今回の質問者")
    current.reference = reference(root.id, resolved=root)
    client = AIKunClient(settings, service)
    client._connection.user = SimpleNamespace(id=222)
    try:
        await client.on_message(current)
        messages = complete.await_args.args[0]
        current_metadata = next(
            m["content"]
            for m in messages
            if isinstance(m["content"], str) and "今回の質問者プロフィール" in m["content"]
        )
        profile = json.loads(current_metadata[current_metadata.index("{") :])["author"]
        assert profile["nick"] == "今回の質問者" and profile["username"] == "account_name"
        history = next(
            m["content"] for m in messages if isinstance(m["content"], str) and "今回の返信チェーン" in m["content"]
        )
        assert json.loads(history[history.index("[{") :])[0]["author"]["nick"] == "先の質問者"
        assert USER_ID not in current.reply.await_args.args[0]
    finally:
        await client.drain()
        await client.close()


async def test_mcp_member_nickname_and_cached_profile_without_extra_discord_requests(settings):
    class API(FakeAPI):
        @staticmethod
        def message(message_id):
            value = FakeAPI.message(message_id)
            value["author"] = {"id": USER_ID, "username": "account_name", "global_name": "全体表示名"}
            if message_id == "50002":
                value["member"] = {"nick": "サーバーの呼び名"}
            return value

    api = API()
    reader = DiscordReader(settings.discord, api=api)
    data = await reader.read("discord_read_messages", {"channel_id": "34567", "limit": 2})
    assert len(api.calls) == 2
    assert [m["author"]["display_name"] for m in data["messages"]] == ["サーバーの呼び名"] * 2
    assert data["messages"][0]["author"]["nick"] == "サーバーの呼び名"
    single = await reader.read("discord_get_message", {"channel_id": "34567", "message_id": "50001"})
    assert single["message"]["author"]["nick"] == "サーバーの呼び名"
    assert len(api.calls) == 3


async def test_slash_question_includes_member_profile(settings, store, monkeypatch):
    service = ChatService(settings, store)
    generate = AsyncMock(return_value=SimpleNamespace(content="返答"))
    monkeypatch.setattr(service, "generate_reply", generate)
    client = AIKunClient(settings, service)
    interaction = SimpleNamespace(user=member(), channel_id=34567)

    async def answer(interaction, operation):
        return await operation()

    monkeypatch.setattr(client, "_answer", answer)
    try:
        command = client.tree.get_command("ask")
        await command.callback(interaction, "質問")
        context = generate.await_args.kwargs["discord_context"]
        assert context.author["nick"] == "サーバーの呼び名"
        assert context.author["global_name"] == "全体表示名"
    finally:
        await client.close()
