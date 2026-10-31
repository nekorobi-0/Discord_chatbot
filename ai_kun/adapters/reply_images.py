"""検証済みの返信チェーンにある画像を、近い発言から優先して読み込む。"""

import asyncio
import logging
import time

import aiohttp
import discord

from ..discord_context import DiscordImage
from ..errors import InvalidInputError
from ..file_types import attachment_kind
from ..images import image_data_url
from ..request_diagnostics import phase

logger = logging.getLogger(__name__)
MAX_REPLY_IMAGES = 8
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REPLY_IMAGE_BYTES = 24 * 1024 * 1024
IMAGE_TIMEOUT_SECONDS = 10


async def load_reply_images(targets, channel, guild_id) -> tuple[DiscordImage, ...]:
    selected = []
    seen = set()
    for target, record in targets:
        for attachment, metadata in zip(getattr(target, "attachments", ()), record["attachments"], strict=True):
            if attachment_kind(attachment.filename, getattr(attachment, "content_type", None)) != "image":
                continue
            key = str(
                getattr(attachment, "id", None)
                or getattr(attachment, "url", None)
                or f"{target.id}:{attachment.filename}"
            )
            if key in seen:
                metadata["image_status"] = "duplicate"
            elif len(selected) >= MAX_REPLY_IMAGES:
                metadata["image_status"] = "omitted: image count limit"
            elif (getattr(attachment, "size", 0) or 0) > MAX_IMAGE_BYTES:
                metadata["image_status"] = "omitted: image size limit"
            else:
                seen.add(key)
                selected.append((target, attachment, metadata))
    if not selected:
        return ()
    phase("reply_chain_images")
    started = time.monotonic()
    semaphore = asyncio.Semaphore(3)

    async def download(target, attachment, metadata):
        async with semaphore:
            try:
                async with asyncio.timeout(IMAGE_TIMEOUT_SECONDS):
                    try:
                        data = await attachment.read(use_cached=True)
                    except discord.HTTPException as exc:
                        if exc.status not in {403, 404}:
                            raise
                        # キャッシュ済み添付URLの期限切れは、同じ発言を再取得して更新する。
                        fresh = await channel.fetch_message(target.id)
                        fresh_guild, fresh_channel = getattr(fresh, "guild", None), getattr(fresh, "channel", None)
                        if (fresh_guild is not None and fresh_guild.id != guild_id) or (
                            fresh_channel is not None and fresh_channel.id != channel.id
                        ):
                            raise InvalidInputError("返信先の取得範囲が一致しません。")
                        current = next(
                            (
                                item
                                for item in fresh.attachments
                                if getattr(item, "id", None) == getattr(attachment, "id", None)
                                and item.filename == attachment.filename
                            ),
                            None,
                        )
                        if current is None:
                            raise InvalidInputError("返信先の画像が見つかりません。")
                        data = await current.read()
                if len(data) > MAX_IMAGE_BYTES:
                    raise InvalidInputError("画像は8 MiBまでです。")
                return data
            except (discord.HTTPException, aiohttp.ClientError, OSError, TimeoutError, InvalidInputError) as exc:
                metadata["image_status"] = "unavailable"
                logger.warning(
                    "reply image failed: message=%s attachment=%s error=%s",
                    target.id,
                    getattr(attachment, "id", None),
                    type(exc).__name__,
                )
                return None

    blobs = await asyncio.gather(*(download(*item) for item in selected))
    images = []
    used = 0
    for (target, attachment, metadata), data in zip(selected, blobs, strict=True):
        if data is None:
            continue
        if used + len(data) > MAX_REPLY_IMAGE_BYTES:
            metadata["image_status"] = "omitted: total image size limit"
            continue
        try:
            uri = image_data_url(data)
        except InvalidInputError:
            metadata["image_status"] = "unavailable: image format"
            logger.warning("reply image format invalid: message=%s attachment=%s", target.id, metadata["attachment_id"])
            continue
        used += len(data)
        metadata["image_status"] = "included"
        images.append(
            DiscordImage(
                message_id=str(target.id),
                attachment_id=metadata["attachment_id"],
                filename=attachment.filename,
                data_url=uri,
            )
        )
    logger.info(
        "reply chain images loaded: images=%d bytes=%d seconds=%.3f", len(images), used, time.monotonic() - started
    )
    return tuple(images)
