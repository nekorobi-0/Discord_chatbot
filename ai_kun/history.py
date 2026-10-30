"""Discord履歴を100件ずつ取得。レート制限ヘッダーに追従する単独CLI用。"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlmodel import Session

from .db import HistoryCursorRecord, data_dir, get_engine
from .errors import ConfigurationError, UpstreamError

logger = logging.getLogger(__name__)
API = "https://discord.com/api/v10"


async def interruptible_sleep(seconds: float) -> None:
    while seconds > 0:
        step = min(seconds, 60)
        await asyncio.sleep(step)
        seconds -= step


@contextmanager
def importer_lock():
    # コンテナをまたいでも同じボリュームでは一件だけ。終了・クラッシュ時にOSが解放。
    import fcntl

    with (data_dir() / "history-import.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigurationError("過去ログの取り込みはすでに実行中です。") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def number(value, default=0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed >= 0 else default
    except (TypeError, ValueError):
        return default


@dataclass
class Bucket:
    next_at: float = 0
    reset_at: float = 0
    window: float = 0
    used: int = 0
    limit: int = 1


class HalfRateLimiter:
    """既知bucketの約半分以下。未知bucketは2秒間隔でプローブする。"""

    def __init__(self, clock=time.monotonic, sleep=interruptible_sleep):
        self.clock = clock
        self.sleep = sleep
        self.routes: dict[tuple[str, str], tuple[str, str]] = {}
        self.buckets: dict[tuple[str, str], Bucket] = {}
        self.global_next = 0.0

    async def wait(self, route: str, major: str) -> None:
        key = self.routes.get((route, major), (route, major))
        bucket = self.buckets.get(key)
        deadline = max(self.global_next, bucket.next_at if bucket else 0)
        await self.sleep(max(0.0, deadline - self.clock()))
        self.global_next = self.clock() + 1 / 25  # 公式のglobal 50req/sに対して50%。

    def observe(self, route: str, major: str, headers: httpx.Headers) -> None:
        now = self.clock()
        old_key = self.routes.get((route, major), (route, major))
        bucket_id = headers.get("X-RateLimit-Bucket", route)
        key = (bucket_id, major)
        self.routes[(route, major)] = key
        bucket = self.buckets.setdefault(key, self.buckets.get(old_key, Bucket()))
        limit = number(headers.get("X-RateLimit-Limit"))
        reset = number(headers.get("X-RateLimit-Reset-After"), -1)
        if reset < 0:
            reset = max(0.0, number(headers.get("X-RateLimit-Reset")) - time.time())
        if not limit or reset <= 0:
            # 上限不明のときはバーストしない。
            bucket.next_at = now + max(2.0, bucket.window * 2 / bucket.limit)
            self.global_next = max(self.global_next, now + 2.0)
            return
        if now >= bucket.reset_at or abs(bucket.reset_at - (now + reset)) > 0.5:
            bucket.used = 0
        bucket.limit = max(1, int(limit))
        bucket.window = max(bucket.window, reset)
        bucket.reset_at = now + reset
        bucket.used += 1
        bucket.next_at = now + max(1 / 25, 2 * bucket.window / bucket.limit)
        half_allowance = max(1, bucket.limit // 2)
        if bucket.used >= half_allowance or number(headers.get("X-RateLimit-Remaining"), 0) <= 0:
            bucket.next_at = max(bucket.next_at, bucket.reset_at + 0.1)

    def defer(self, route: str, major: str, seconds: float, global_limit: bool) -> None:
        deadline = self.clock() + seconds + 0.1
        key = self.routes.get((route, major), (route, major))
        bucket = self.buckets.setdefault(key, Bucket())
        bucket.next_at = max(bucket.next_at, deadline)
        if global_limit:
            self.global_next = max(self.global_next, deadline)


class DiscordHistoryAPI:
    def __init__(self, token: str, client: httpx.AsyncClient, limiter=None):
        self.client = client
        self.token = token
        self.limiter = limiter or HalfRateLimiter()

    async def get(self, channel_id: int, *, messages=False, after=0):
        route = "messages" if messages else "channel"
        path = f"/channels/{channel_id}" + ("/messages" if messages else "")
        major = str(channel_id)
        for attempt in range(5):
            await self.limiter.wait(route, major)
            try:
                response = await self.client.get(
                    API + path,
                    params={"limit": 100, "after": str(after)} if messages else None,
                    headers={"Authorization": f"Bot {self.token}", "User-Agent": "AIKunHistory/0.1.0"},
                )
            except httpx.TransportError:
                self.limiter.defer(route, major, 2**attempt, True)
                continue
            self.limiter.observe(route, major, response.headers)
            if response.status_code == 429:
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                if not isinstance(data, dict):
                    data = {}
                retry = max(number(data.get("retry_after")), number(response.headers.get("Retry-After")), 1)
                global_limit = bool(data.get("global")) or response.headers.get("X-RateLimit-Global") == "true"
                self.limiter.defer(route, major, retry, global_limit)
                logger.warning("Discord history rate limited; waiting %.2fs", retry)
                continue
            if response.status_code in {401, 403, 404}:
                raise ConfigurationError(
                    f"Discord履歴取得がHTTP {response.status_code}で停止しました。トークン・対象・閲覧/履歴権限を確認してください。"
                )
            if response.status_code >= 500:
                self.limiter.defer(route, major, 2**attempt, True)
                continue
            if response.status_code != 200:
                raise UpstreamError("Discord履歴取得")
            try:
                data = response.json()
            except ValueError:
                raise UpstreamError("Discord履歴取得") from None
            if (messages and not isinstance(data, list)) or (not messages and not isinstance(data, dict)):
                raise UpstreamError("Discord履歴取得")
            return data
        raise UpstreamError("Discord履歴取得（再試行上限）")


async def import_history(settings, store, *, api=None, until_id=None) -> dict[str, int]:
    """全許可チャンネルを古い順に走査。ページの保存とカーソル更新は同じDBトランザクション。"""
    if not settings.memory.enabled:
        raise ConfigurationError("過去ログ取り込みには MEMORY_ENABLED=true が必要です。")
    if until_id is None:
        until_id = (int(datetime.now(UTC).timestamp() * 1000) - 1420070400000) << 22
    totals = {"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}
    with importer_lock():
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            api = api or DiscordHistoryAPI(settings.discord.bot_token, client)
            for channel_id in settings.discord.allowed_channel_ids:
                channel = await api.get(channel_id)
                if str(channel.get("guild_id")) != str(settings.discord.guild_id):
                    raise ConfigurationError("許可チャンネルのサーバーIDがGUILD_IDと一致しません。")
                if channel.get("type") not in {0, 5, 10, 11, 12}:
                    raise ConfigurationError("履歴取り込みはテキストチャンネル・個別スレッドのみ対応しています。")
                key = f"{settings.discord.guild_id}:{channel_id}"
                with Session(get_engine()) as session:
                    checkpoint = session.get(HistoryCursorRecord, key)
                    cursor = int(checkpoint.last_message_id) if checkpoint else 0
                while cursor < until_id:
                    page = await api.get(channel_id, messages=True, after=cursor)
                    try:
                        page = sorted(page, key=lambda message: int(message["id"]))
                        eligible = [m for m in page if cursor < int(m["id"]) <= until_id]
                    except (KeyError, TypeError, ValueError):
                        raise UpstreamError("Discord履歴の形式") from None
                    if not eligible:
                        break
                    result = await store.ingest_history_page(settings.discord.guild_id, channel_id, eligible)
                    cursor = int(eligible[-1]["id"])
                    for name in totals:
                        totals[name] += result[name]
                    logger.info(
                        "history progress channel=%s scanned=%d imported=%d skipped=%d documents=%d",
                        channel_id,
                        totals["scanned"],
                        totals["imported"],
                        totals["skipped"],
                        totals["documents"],
                    )
                    if any(int(m["id"]) > until_id for m in page):
                        break
    return totals
