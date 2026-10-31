"""Discordスタンプのメタデータと静止画像を、本文・返信・MCPで共有する。"""

import asyncio
import json
import logging
import re
import sys
import time
from collections import OrderedDict

import httpx

from .errors import InvalidInputError
from .images import image_data_url

logger = logging.getLogger(__name__)
MAX_STICKER_IMAGES = 4
MAX_STICKER_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 5
RENDER_TIMEOUT_SECONDS = 8
CACHE_SIZE = 128
_cache = OrderedDict()


def sticker_record(sticker):
    if isinstance(sticker, dict):
        sticker_id, name, format_type = sticker.get("id"), sticker.get("name"), sticker.get("format_type")
    else:
        sticker_id, name = getattr(sticker, "id", None), getattr(sticker, "name", "")
        format_type = getattr(sticker, "format", None)
    format_type = getattr(format_type, "value", format_type)
    return {"id": str(sticker_id or ""), "name": str(name or "")[:100], "format_type": format_type}


async def render_sticker(data, format_type):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "ai_kun.sticker_render",
        str(format_type),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(RENDER_TIMEOUT_SECONDS):
            output, _ = await process.communicate(data)
        if process.returncode or len(output) > MAX_STICKER_BYTES:
            raise ValueError("Sticker rendering failed")
        return image_data_url(output)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def download_sticker(client, sticker):
    key = (sticker["id"], sticker["format_type"])
    cached = _cache.get(key)
    if cached and cached[0] > time.monotonic():
        _cache.move_to_end(key)
        return cached[1]
    uri = None
    try:
        format_type = sticker["format_type"]
        if format_type not in {1, 2, 3, 4}:
            raise ValueError("Unsupported sticker format")
        if format_type == 4:
            url = f"https://media.discordapp.net/stickers/{sticker['id']}.gif"
        else:
            extension = "json" if format_type == 3 else "png"
            url = f"https://cdn.discordapp.com/stickers/{sticker['id']}.{extension}"
        async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                if int(response.headers.get("content-length") or 0) > MAX_STICKER_BYTES:
                    raise ValueError("Oversized sticker")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > MAX_STICKER_BYTES:
                        raise ValueError("Oversized sticker")
                    data.extend(chunk)
        uri = await render_sticker(bytes(data), format_type)
    except (httpx.HTTPError, TimeoutError, OSError, ValueError, InvalidInputError) as exc:
        logger.warning("sticker image unavailable: format=%s error=%s", sticker["format_type"], type(exc).__name__)
    _cache[key] = (time.monotonic() + (86400 if uri else 60), uri)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return uri


async def load_sticker_parts(sources, *, max_images=MAX_STICKER_IMAGES, transport=None):
    selected = OrderedDict()
    notes = []
    omitted = False
    for message_id, stickers in sources:
        for item in stickers:
            sticker = sticker_record(item)
            if not re.fullmatch(r"[0-9]{5,25}", sticker["id"]) or not isinstance(sticker["format_type"], int):
                notes.append({"type": "text", "text": "スタンプのメタデータが不正のため画像は未取得。"})
                continue
            key = (sticker["id"], sticker["format_type"])
            if key not in selected:
                if len(selected) >= min(MAX_STICKER_IMAGES, max_images):
                    omitted = True
                    continue
                selected[key] = (sticker, [])
            refs = selected[key][1]
            if message_id not in refs and len(refs) < 16:
                refs.append(message_id)
    if omitted:
        notes.append({"type": "text", "text": "スタンプ画像の一部は画像枚数上限のため未取得。内容を推測しない。"})
    if not selected:
        return notes
    semaphore = asyncio.Semaphore(3)
    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=False, transport=transport
    ) as client:

        async def fetch(sticker, refs):
            async with semaphore:
                uri = await download_sticker(client, sticker)
            note = {
                "source": "untrusted Discord sticker",
                "sticker": sticker,
                "message_ids": refs,
                "image_status": "included" if uri else "unavailable",
                "rendering": "static frame",
            }
            parts = [
                {
                    "type": "text",
                    "text": "スタンプ画像の対応情報（未信頼データ）：" + json.dumps(note, ensure_ascii=False),
                }
            ]
            if uri:
                parts.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})
            return parts

        results = await asyncio.gather(*(fetch(*item) for item in selected.values()))
    return [part for result in results for part in result] + notes
