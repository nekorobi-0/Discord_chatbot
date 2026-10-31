"""返信参照を同じサーバー・チャンネル内でルートまでたどる。"""

import logging

import discord

from ..discord_context import DiscordContext
from ..sticker_images import sticker_record
from .reply_images import load_reply_images

logger = logging.getLogger(__name__)


def reply_id(message) -> str | None:
    reference = getattr(message, "reference", None)
    if (
        reference is not None
        and getattr(reference, "type", discord.MessageReferenceType.default) != discord.MessageReferenceType.default
    ):
        return None
    value = getattr(reference, "message_id", None)
    return str(value) if value is not None else None


async def resolve_reply(message, channel, guild_id: int, *, refresh=False):
    reference = getattr(message, "reference", None)
    if (
        reference is None
        or getattr(reference, "type", discord.MessageReferenceType.default) != discord.MessageReferenceType.default
    ):
        return None, None
    if getattr(reference, "guild_id", guild_id) not in (None, guild_id):
        return None, "outside_guild"
    if getattr(reference, "channel_id", channel.id) not in (None, channel.id):
        return None, "outside_channel"
    target = None if refresh else (getattr(reference, "resolved", None) or getattr(reference, "cached_message", None))
    if isinstance(target, discord.DeletedReferencedMessage):
        return None, "deleted"
    if target is None and getattr(reference, "message_id", None) is not None:
        try:
            target = await channel.fetch_message(reference.message_id)
        except discord.HTTPException as exc:
            logger.warning("reply target fetch failed: status=%s code=%s", exc.status, exc.code)
            return None, f"http_{exc.status}"
    if target is None:
        return None, "unavailable"
    target_guild = getattr(target, "guild", None)
    target_channel = getattr(target, "channel", None)
    if target_guild is not None and target_guild.id != guild_id:
        return None, "outside_guild"
    if target_channel is not None and target_channel.id != channel.id:
        return None, "outside_channel"
    return target, None


def message_record(message, *, channel_id: int, guild_id: int) -> dict:
    author = getattr(message, "author", None)
    timestamp = getattr(message, "created_at", None)
    return {
        "message_id": str(message.id),
        "guild_id": str(guild_id),
        "channel_id": str(channel_id),
        "reply_to_message_id": reply_id(message),
        "author_id": str(author.id) if author is not None else None,
        "author_name": str(getattr(author, "display_name", None) or getattr(author, "name", None) or ""),
        "author_is_bot": bool(getattr(author, "bot", False)),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "content": str(getattr(message, "content", "") or ""),
        "stickers": [sticker_record(sticker) for sticker in getattr(message, "stickers", ())],
        "attachments": [
            {
                "attachment_id": str(attachment.id) if getattr(attachment, "id", None) is not None else None,
                "filename": attachment.filename,
                "content_type": getattr(attachment, "content_type", None),
                "size": attachment.size,
            }
            for attachment in getattr(message, "attachments", ())
        ],
    }


async def collect_reply_chain(message, first_target, first_stop_reason=None) -> DiscordContext:
    message_id = str(message.id) if getattr(message, "id", None) is not None else None
    seen = {message_id} if message_id is not None else set()
    records = []
    targets = []
    target = first_target
    reason = first_stop_reason
    while target is not None:
        target_id = str(target.id)
        if target_id in seen:
            reason = "cycle"
            break
        seen.add(target_id)
        record = message_record(target, channel_id=message.channel.id, guild_id=message.guild.id)
        records.append(record)
        targets.append((target, record))
        # 既知の参照だけで循環を判定し、同じ発言を二度取得しない。
        next_id = reply_id(target)
        if next_id is not None and next_id in seen:
            reason = "cycle"
            break
        target, reason = await resolve_reply(target, message.channel, message.guild.id)
    images = await load_reply_images(targets, message.channel, message.guild.id)
    return DiscordContext(
        message_id=message_id,
        reply_to_message_id=reply_id(message),
        messages=tuple(reversed(records)),
        stop_reason=reason,
        author_id=str(message.author.id) if getattr(getattr(message, "author", None), "id", None) is not None else None,
        images=images,
        stickers=tuple(sticker_record(sticker) for sticker in getattr(message, "stickers", ())),
    )
