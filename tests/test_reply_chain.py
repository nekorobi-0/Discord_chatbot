from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord

from ai_kun.adapters.reply_chain import collect_reply_chain, resolve_reply


def reference(message_id, *, resolved=None, channel_id=34567, guild_id=23456):
    return SimpleNamespace(
        message_id=message_id,
        resolved=resolved,
        channel_id=channel_id,
        guild_id=guild_id,
        type=discord.MessageReferenceType.default,
    )


def ancestor(message_id, content, *, bot=False, parent=None):
    return SimpleNamespace(
        id=message_id,
        content=content,
        author=SimpleNamespace(id=12345 if bot else 11111, bot=bot, display_name="AIくん" if bot else "利用者"),
        reference=parent,
        created_at=datetime(2026, 10, 31, tzinfo=UTC),
        attachments=[],
    )


async def test_collects_entire_mixed_reply_chain_oldest_first_with_ids():
    root = ancestor(60001, "最初の質問")
    human = ancestor(60002, "次の質問", parent=reference(60001))
    bot = ancestor(60003, "途中の返答", bot=True, parent=reference(60002))
    lookup = {60001: root, 60002: human}
    channel = SimpleNamespace(id=34567, fetch_message=AsyncMock(side_effect=lambda message_id: lookup[message_id]))
    message = SimpleNamespace(
        id=90003, guild=SimpleNamespace(id=23456), channel=channel, reference=reference(60003, resolved=bot)
    )
    first, reason = await resolve_reply(message, channel, 23456)
    context = await collect_reply_chain(message, first, reason)
    assert context.message_id == "90003"
    assert context.reply_to_message_id == "60003"
    assert context.stop_reason is None
    assert [item["message_id"] for item in context.messages] == ["60001", "60002", "60003"]
    assert [item["content"] for item in context.messages] == ["最初の質問", "次の質問", "途中の返答"]
    assert all(item["guild_id"] == "23456" and item["channel_id"] == "34567" for item in context.messages)
    assert context.messages[-1]["author_is_bot"] is True
    assert context.messages[1]["reply_to_message_id"] == "60001"
    assert [call.args[0] for call in channel.fetch_message.await_args_list] == [60002, 60001]


async def test_cycle_stops_without_duplicate_fetch_or_duplicate_context():
    bot = ancestor(60001, "循環する参照", bot=True, parent=reference(60001))
    channel = SimpleNamespace(id=34567, fetch_message=AsyncMock())
    message = SimpleNamespace(id=90003, guild=SimpleNamespace(id=23456), channel=channel, reference=reference(60001))
    context = await collect_reply_chain(message, bot)
    assert context.stop_reason == "cycle"
    assert [item["message_id"] for item in context.messages] == ["60001"]
    channel.fetch_message.assert_not_awaited()


async def test_missing_ancestor_preserves_available_context_and_unread_id():
    bot = ancestor(60003, "参照できる返答", bot=True, parent=reference(60002))
    channel = SimpleNamespace(
        id=34567,
        fetch_message=AsyncMock(
            side_effect=discord.NotFound(
                SimpleNamespace(status=404, reason="Not Found"), {"message": "Unknown Message", "code": 10008}
            )
        ),
    )
    message = SimpleNamespace(id=90003, guild=SimpleNamespace(id=23456), channel=channel, reference=reference(60003))
    context = await collect_reply_chain(message, bot)
    assert context.stop_reason == "http_404"
    assert context.messages[0]["reply_to_message_id"] == "60002"
    assert len(context.messages) == 1


async def test_other_channel_ancestor_is_not_fetched():
    bot = ancestor(60003, "ここまで", bot=True, parent=reference(60002, channel_id=45678))
    channel = SimpleNamespace(id=34567, fetch_message=AsyncMock())
    message = SimpleNamespace(id=90003, guild=SimpleNamespace(id=23456), channel=channel, reference=reference(60003))
    context = await collect_reply_chain(message, bot)
    assert context.stop_reason == "outside_channel"
    assert len(context.messages) == 1
    channel.fetch_message.assert_not_awaited()
