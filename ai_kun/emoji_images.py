"""Discord本文の絵文字を小さい静止画像としてマルチモーダル入力へ渡す。"""

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

import emoji
import httpx

from .errors import InvalidInputError
from .images import image_data_url

logger = logging.getLogger(__name__)
CUSTOM_EMOJI = re.compile(r"<(?P<animated>a?):(?P<name>[A-Za-z0-9_]{1,64}):(?P<id>[0-9]{5,25})>")
TWEMOJI_BASE = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.1/assets/72x72"
MAX_EMOJI_IMAGES = 4
MAX_EMOJI_BYTES = 256 * 1024
EMOJI_TIMEOUT_SECONDS = 5
CACHE_SIZE = 128
# 画像は24時間、取得失敗は1分キャッシュ。同じID・Unicode列のみをキーにする。
_cache = OrderedDict()


@dataclass(frozen=True)
class EmojiImage:
    key: str
    text: str
    url: str
    animated: bool = False


def find_emojis(text: str):
    """ZWJ・肌色・国旗・異体字セレクタを含むUnicode列を分断しない。"""
    matches = []
    for match in CUSTOM_EMOJI.finditer(text):
        value = EmojiImage(
            "custom:" + match["id"],
            match[0],
            f"https://cdn.discordapp.com/emojis/{match['id']}.png?size=128&quality=lossless",
            bool(match["animated"]),
        )
        matches.append((match.start(), value))
    for match in emoji.emoji_list(CUSTOM_EMOJI.sub(lambda m: " " * len(m[0]), text)):
        value = match["emoji"]
        # Twemojiは単体・keycapからFE0Fを除去し、ZWJ列では保持する。
        code = "-".join(f"{ord(char):x}" for char in value if char != "\ufe0f" or "\u200d" in value)
        matches.append((match["match_start"], EmojiImage("unicode:" + code, value, f"{TWEMOJI_BASE}/{code}.png")))
    return [value for _, value in sorted(matches, key=lambda item: item[0])]


async def _download(client, image):
    cached = _cache.get(image.key)
    if cached and cached[0] > time.monotonic():
        _cache.move_to_end(image.key)
        return cached[1]
    uri = None
    try:
        async with asyncio.timeout(EMOJI_TIMEOUT_SECONDS):
            async with client.stream("GET", image.url) as response:
                response.raise_for_status()
                if int(response.headers.get("content-length") or 0) > MAX_EMOJI_BYTES:
                    raise InvalidInputError("絵文字画像が大きすぎます。")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > MAX_EMOJI_BYTES:
                        raise InvalidInputError("絵文字画像が大きすぎます。")
                    data.extend(chunk)
            uri = image_data_url(bytes(data))
    except (httpx.HTTPError, TimeoutError, OSError, ValueError, InvalidInputError) as exc:
        logger.warning("emoji image unavailable: kind=%s error=%s", image.key.split(":")[0], type(exc).__name__)
    _cache[image.key] = (time.monotonic() + (86400 if uri else 60), uri)
    _cache.move_to_end(image.key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return uri


async def load_emoji_parts(sources, *, max_images=MAX_EMOJI_IMAGES, transport=None):
    """(message_id, content)を近い順に受け取り、対応表と画像を返す。外部URLは受け取らない。"""
    selected = OrderedDict()
    omitted = False
    for message_id, text in sources:
        for image in find_emojis(str(text or "")):
            if image.key not in selected:
                if len(selected) >= min(MAX_EMOJI_IMAGES, max_images):
                    omitted = True
                    continue
                selected[image.key] = (image, [])
            refs = selected[image.key][1]
            if message_id not in refs and len(refs) < 16:
                refs.append(message_id)
    limit_note = (
        [{"type": "text", "text": "絵文字画像の一部は画像枚数上限のため未取得。見たと主張せず、本文のみ参照する。"}]
        if omitted
        else []
    )
    if not selected:
        return limit_note
    semaphore = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=EMOJI_TIMEOUT_SECONDS, follow_redirects=False, transport=transport) as client:

        async def fetch(image, refs):
            async with semaphore:
                uri = await _download(client, image)
            note = {
                "source": "untrusted Discord emoji",
                "emoji": image.text,
                "message_ids": refs,
                "image_status": "included" if uri else "unavailable",
                "rendering": "static frame" if image.animated else "static",
            }
            if image.key.startswith("unicode:"):
                note["artwork"] = "Twemoji 17.0.1 (CC-BY 4.0; jdecked/twemoji)"
            parts = [
                {
                    "type": "text",
                    "text": "絵文字画像の対応情報（未信頼データ）：" + json.dumps(note, ensure_ascii=False),
                }
            ]
            if uri:
                parts.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})
            return parts

        results = await asyncio.gather(*(fetch(*item) for item in selected.values()))
    return [part for parts in results for part in parts] + limit_note
