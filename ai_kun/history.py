"""Discord履歴を100件ずつ取得。稼働中のBotが自動バックフィルで使うレート制限ヘッダー追従のみを行う。"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
from sqlmodel import Session

from .db import HistoryScanCursorRecord, data_dir, get_engine
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
    limit: int = 1
    remaining: int | None = None


class DiscordRateLimiter:
    """全体50req/s以内で、bucketの残り枠・リセット・429に従う。"""

    def __init__(self, clock=time.monotonic, sleep=interruptible_sleep):
        self.clock = clock
        self.sleep = sleep
        self.routes: dict[tuple[str, str], tuple[str, str]] = {}
        self.buckets: dict[tuple[str, str], Bucket] = {}
        self.global_next = 0.0
        self._reservation_lock = asyncio.Lock()

    async def wait(self, route: str, major: str) -> None:
        while True:
            async with self._reservation_lock:
                now = self.clock()
                key = self.routes.get((route, major), (route, major))
                bucket = self.buckets.get(key)
                if bucket and bucket.remaining is not None and now >= bucket.reset_at:
                    bucket.remaining = bucket.limit
                    bucket.reset_at = float("inf")  # 次の応答で実際のリセット時刻を更新。
                deadline = max(self.global_next, bucket.next_at if bucket else 0)
                if bucket and bucket.remaining == 0:
                    deadline = max(deadline, bucket.reset_at + 0.05)
                if deadline <= now:
                    self.global_next = now + 1 / 50
                    if bucket and bucket.remaining is not None:
                        bucket.remaining -= 1
                    return
            # 待機中は他のbucketを止めず、起床後に予約状況を再確認する。
            await self.sleep(deadline - now)

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
            # ヘッダーがないだけで人工的な2秒待機を足さない。429はdeferで扱う。
            return
        new_window = bucket.remaining is None or (
            bucket.reset_at != float("inf") and (now >= bucket.reset_at or abs(bucket.reset_at - (now + reset)) > 0.5)
        )
        bucket.limit = max(1, int(limit))
        bucket.reset_at = now + reset
        remaining = int(number(headers.get("X-RateLimit-Remaining"), 0))
        bucket.remaining = remaining if new_window or bucket.remaining is None else min(bucket.remaining, remaining)

    def defer(self, route: str, major: str, seconds: float, global_limit: bool) -> None:
        deadline = self.clock() + seconds + 0.1
        key = self.routes.get((route, major), (route, major))
        bucket = self.buckets.setdefault(key, Bucket())
        bucket.next_at = max(bucket.next_at, deadline)
        if global_limit:
            self.global_next = max(self.global_next, deadline)


# 稼働中Botの自動取り込み（バックフィル）もこの共通制限器を共有する。
shared_limiter = DiscordRateLimiter()


class DiscordHistoryAPI:
    def __init__(self, token: str, client: httpx.AsyncClient, limiter=None):
        self.client = client
        self.token = token
        self.limiter = limiter or DiscordRateLimiter()

    async def get(self, resource, *, messages=False, after=None, before=None, limit=100, expect_list=False):
        """チャンネルIDまたはDiscord APIパス（例: guilds/<id>/channels）へのGET。"""
        route = "messages" if messages else "resource"
        channel_id = resource
        path = (
            f"/channels/{channel_id}/messages"
            if messages
            else (f"/channels/{channel_id}" if str(resource).isdigit() else f"/{resource}")
        )
        params = None
        if messages:
            params = {"limit": limit}
            if after is not None:
                params["after"] = str(after)
            if before is not None:
                params["before"] = str(before)
        major = str(channel_id)
        parts = str(resource).split("/")
        if len(parts) == 4 and parts[0] == "channels" and parts[2] == "messages":
            # メッセージIDをmajorにすると、同一チャンネルでも別bucketになってしまう。
            route, major = "message", parts[1]
        for attempt in range(5):
            started = time.monotonic()
            await self.limiter.wait(route, major)
            sent = time.monotonic()
            try:
                response = await self.client.get(
                    API + path,
                    params=params,
                    headers={"Authorization": f"Bot {self.token}", "User-Agent": "AIKunHistory/0.1.0"},
                )
            except httpx.TransportError:
                self.limiter.defer(route, major, 2**attempt, True)
                continue
            logger.info(
                "Discord API GET: route=%s wait_seconds=%.3f request_seconds=%.3f status=%d",
                route,
                sent - started,
                time.monotonic() - sent,
                response.status_code,
            )
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
            if ((messages or expect_list) and not isinstance(data, list)) or (
                not (messages or expect_list) and not isinstance(data, dict)
            ):
                raise UpstreamError("Discord履歴取得")
            return data
        raise UpstreamError("Discord履歴取得（再試行上限）")


async def backfill_channel(settings, store, channel_id: int, *, api=None) -> dict[str, int]:
    """走査専用カーソルから古い順に読む。旧版のライブ観測位置は信用しない。"""
    totals = {"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}
    key = f"{settings.discord.guild_id}:{channel_id}"
    # 自動バックフィルも単独実行だけ。二重起動はカーソルを進めず、取り込みを重複させない。
    with importer_lock():
        with Session(get_engine()) as session:
            checkpoint = session.get(HistoryScanCursorRecord, key)
            cursor = int(checkpoint.last_message_id) if checkpoint else 0
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            api = api or DiscordHistoryAPI(settings.discord.bot_token, client, shared_limiter)
            channel = await api.get(channel_id)
            if str(channel.get("guild_id")) != str(settings.discord.guild_id):
                raise ConfigurationError("チャンネルのサーバーIDがGUILD_IDと一致しません。")
            while True:
                page = await api.get(channel_id, messages=True, after=cursor)
                try:
                    ordered = sorted(page, key=lambda message: int(message["id"]))
                    eligible = [m for m in ordered if cursor < int(m["id"])]
                except (KeyError, TypeError, ValueError):
                    raise UpstreamError("Discord履歴の形式") from None
                if not eligible:
                    break
                result = await store.ingest_history_page(settings.discord.guild_id, channel_id, eligible)
                cursor = int(eligible[-1]["id"])
                # 保存に成功したページだけ確定。ライブ投入の最新IDは参照しない。
                with Session(get_engine()) as session:
                    checkpoint = session.get(HistoryScanCursorRecord, key)
                    if checkpoint is None:
                        checkpoint = HistoryScanCursorRecord(channel_key=key, last_message_id=str(cursor))
                    else:
                        checkpoint.last_message_id = str(cursor)
                    session.add(checkpoint)
                    session.commit()
                for name in totals:
                    totals[name] += result[name]
                logger.info(
                    "auto backfill progress channel=%s scanned=%d imported=%d skipped=%d documents=%d",
                    channel_id,
                    totals["scanned"],
                    totals["imported"],
                    totals["skipped"],
                    totals["documents"],
                )
    return totals
