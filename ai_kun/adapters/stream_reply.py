"""生成とは独立した間隔でDiscordの返信を更新する。"""

import asyncio
import logging
import time

import aiohttp
import discord

from ..request_diagnostics import phase

logger = logging.getLogger("ai_kun.adapters.discord_bot")
EDIT_INTERVAL_SECONDS = 2.0
DISCORD_IO_TIMEOUT_SECONDS = 30


class StreamReply:
    def __init__(self, send, split, *, request_id: str, kind="reply"):
        self.send = send
        self.split = split
        self.request_id = request_id
        self.kind = kind
        self.latest = ""
        self.message = None
        self.shown = None
        self.disabled = False
        self._stop = asyncio.Event()
        self._task = None
        self._started_at = 0.0

    def start(self):
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._pump())

    async def update(self, text: str):
        # HTTPの受信をDiscordの編集待ちで止めない。
        self.latest = text

    async def _pump(self):
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), EDIT_INTERVAL_SECONDS)
                return
            except TimeoutError:
                if self.disabled:
                    continue
                elapsed = int((time.monotonic() - self._started_at) // 15) * 15
                waiting = f"考え中…（{elapsed}秒経過）" if elapsed else "考え中…"
                preview = self.split(self.latest)[0] + "\n…（生成中）" if self.latest.strip() else waiting
                await self._render(preview)

    async def _deliver(self, text, *, edit=False):
        verb = "edit" if edit else self.kind
        logger.info("Discord %s started: request=%s", verb, self.request_id)
        try:
            async with asyncio.timeout(DISCORD_IO_TIMEOUT_SECONDS):
                if edit:
                    message = await self.message.edit(content=text, allowed_mentions=discord.AllowedMentions.none())
                else:
                    message = await self.send(text)
            logger.info("Discord %s completed: request=%s", verb, self.request_id)
            return True, message
        except (TimeoutError, discord.HTTPException, aiohttp.ClientError, OSError) as exc:
            logger.warning(
                "Discord %s failed: request=%s error=%s status=%s code=%s",
                verb,
                self.request_id,
                type(exc).__name__,
                getattr(exc, "status", None),
                getattr(exc, "code", None),
            )
            self.disabled = True
            return False, None

    async def _render(self, text):
        if self.message is not None and text == self.shown:
            return True
        success, message = await self._deliver(text, edit=self.message is not None)
        if success:
            self.message = message or self.message
            self.shown = text
        return success

    async def finish(self, text: str, *, failed=False):
        self._stop.set()
        if self._task is not None:
            # 編集のHTTP処理をキャンセルすると、送信済みIDを受け取れず重複投稿し得る。
            await self._task
        if failed and self.latest.strip():
            text = self.latest.rstrip() + "\n\n（生成が中断されました）\n" + text
        chunks = self.split(text)
        phase("discord_send")
        success = await self._render(chunks[0])
        if not success and self.message is not None:
            # 途中表示の投稿が削除された場合なども、元投稿への返信で最終結果を届ける。
            self.message, self.shown = None, None
            success = await self._render(chunks[0])
        if success:
            for chunk in chunks[1:]:
                success, _ = await self._deliver(chunk)
                if not success:
                    break

    async def close(self):
        self._stop.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
