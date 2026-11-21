"""Receive Discord voice audio, split it by speaking state, and store local STT in RAG."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

try:
    from discord.ext import voice_recv
except ImportError:  # The optional voice extra is required to activate this feature.
    voice_recv = None

logger = logging.getLogger("ai_kun.voice_stt")

SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH


@dataclass(frozen=True)
class VoiceSegment:
    pcm: bytes
    author: dict
    started_at: datetime


@dataclass
class _SpeakerBuffer:
    author: dict
    pcm: bytearray
    started_at: datetime


if voice_recv is not None:

    class _TranscriptSink(voice_recv.AudioSink):
        def __init__(self, emit: Callable[[VoiceSegment], None], *, max_seconds: int = 16):
            super().__init__()
            self._emit = emit
            self._max_bytes = BYTES_PER_SECOND * max_seconds
            self._buffers: dict[int, _SpeakerBuffer] = {}
            self._speaking: set[int] = set()
            self._lock = threading.RLock()

        def wants_opus(self) -> bool:
            return False

        @staticmethod
        def _author(user) -> dict:
            return {
                "id": str(getattr(user, "id", "")),
                "username": str(getattr(user, "name", "") or ""),
                "global_name": str(getattr(user, "global_name", "") or ""),
                "nick": str(getattr(user, "nick", "") or ""),
            }

        def _take(self, user_id: int) -> VoiceSegment | None:
            state = self._buffers.pop(user_id, None)
            if state is None or len(state.pcm) < BYTES_PER_SECOND * 0.22:
                return None
            return VoiceSegment(bytes(state.pcm), state.author, state.started_at)

        @voice_recv.AudioSink.listener("voice_member_speaking_start")
        def on_voice_member_speaking_start(self, user) -> None:
            if user is None or getattr(user, "bot", False):
                return
            author = self._author(user)
            if not author["id"].isdigit():
                return
            user_id = int(author["id"])
            with self._lock:
                self._speaking.add(user_id)
                self._buffers.setdefault(
                    user_id,
                    _SpeakerBuffer(author, bytearray(), datetime.now(UTC)),
                )

        @voice_recv.AudioSink.listener("voice_member_speaking_stop")
        def on_voice_member_speaking_stop(self, user) -> None:
            user_id = getattr(user, "id", None)
            if user_id is None:
                return
            segment = None
            with self._lock:
                self._speaking.discard(int(user_id))
                segment = self._take(int(user_id))
            if segment is not None:
                self._emit(segment)

        def write(self, user, data) -> None:
            if user is None or getattr(user, "bot", False) or data.pcm is None:
                return
            author = self._author(user)
            if not author["id"].isdigit():
                return
            pcm = bytes(data.pcm)
            user_id = int(author["id"])
            segments = []
            with self._lock:
                if user_id not in self._speaking:
                    return
                state = self._buffers.get(user_id)
                if state is None:
                    state = _SpeakerBuffer(author, bytearray(), datetime.now(UTC))
                    self._buffers[user_id] = state
                state.author = author
                state.pcm.extend(pcm)
                while len(state.pcm) >= self._max_bytes:
                    segment = self._take(user_id)
                    if segment is not None:
                        segments.append(segment)
                    if user_id in self._speaking:
                        state = _SpeakerBuffer(author, bytearray(), datetime.now(UTC))
                        self._buffers[user_id] = state
                    else:
                        break
            for segment in segments:
                self._emit(segment)

        def flush_all(self) -> list[VoiceSegment]:
            with self._lock:
                self._speaking.clear()
                return [segment for user_id in list(self._buffers) if (segment := self._take(user_id)) is not None]

        def cleanup(self) -> None:
            # Audio is held only in memory and is discarded by the session after final flush.
            return

else:

    class _TranscriptSink:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("discord-ext-voice-recv is not installed")


class VoiceTranscriptSession:
    """One announced VC capture; raw audio is transient and only transcripts enter RAG."""

    def __init__(self, voice_client, service, guild_id: int, channel_id: int, on_transcript):
        if voice_recv is None:
            raise RuntimeError("discord-ext-voice-recv is not installed")
        self.voice_client = voice_client
        self.service = service
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.on_transcript = on_transcript
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[VoiceSegment | None] = asyncio.Queue(maxsize=16)
        self.sink = _TranscriptSink(self._emit)
        self._stopping = False
        self._reader_stopped = self.loop.create_future()
        self._workers: list[asyncio.Task] = []

    async def start(self, *, workers: int = 2) -> None:
        self._workers = [
            asyncio.create_task(self._consume(), name=f"voice-stt-{self.guild_id}-{index + 1}")
            for index in range(max(1, workers))
        ]
        try:
            self.voice_client.listen(self.sink, after=self._after)
        except Exception:
            self._stopping = True
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            raise

    def _emit(self, segment: VoiceSegment) -> None:
        try:
            self.loop.call_soon_threadsafe(self._enqueue, segment)
        except RuntimeError:
            logger.warning("voice transcript segment discarded: event loop closed")

    def _enqueue(self, segment: VoiceSegment) -> None:
        if self._stopping:
            return
        try:
            self.queue.put_nowait(segment)
        except asyncio.QueueFull:
            logger.warning("voice transcript queue full; dropping segment channel=%s", self.channel_id)

    def _after(self, error: Exception | None) -> None:
        if error:
            logger.warning("voice audio receive stopped: %s", type(error).__name__)
        self.loop.call_soon_threadsafe(self._mark_reader_stopped)

    def _mark_reader_stopped(self) -> None:
        if not self._reader_stopped.done():
            self._reader_stopped.set_result(None)

    @staticmethod
    def _wav(pcm: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        return output.getvalue()

    async def _consume(self) -> None:
        while True:
            segment = await self.queue.get()
            try:
                if segment is None:
                    return
                await self._process(segment)
            except Exception as exc:
                logger.warning("voice STT segment failed: %s", type(exc).__name__)
            finally:
                self.queue.task_done()

    async def _process(self, segment: VoiceSegment) -> None:
        transcript = await self.service.transcribe_attachment(self._wav(segment.pcm), "voice.wav")
        if not transcript.strip():
            return
        result = await self.service.ingest_voice_transcript(
            self.channel_id, segment.author, segment.started_at, transcript
        )
        if not result.get("imported"):
            return
        logger.info(
            "voice transcript ingested: channel=%s user=%s chars=%d chunks=%s",
            self.channel_id,
            segment.author["id"],
            len(transcript),
            result.get("chunks", 0),
        )
        try:
            await self.on_transcript(segment.author, transcript)
        except Exception as exc:
            logger.warning("voice transcript announcement failed: %s", type(exc).__name__)

    async def stop(self) -> None:
        if self._stopping:
            return
        if self.voice_client.is_listening():
            self.voice_client.stop_listening()
            try:
                await asyncio.wait_for(asyncio.shield(self._reader_stopped), timeout=5)
            except TimeoutError:
                logger.warning("voice receive stop timed out: channel=%s", self.channel_id)
        self._stopping = True
        for segment in self.sink.flush_all():
            self._enqueue_before_stop(segment)
        await self.queue.join()
        if self._workers:
            for _worker in self._workers:
                await self.queue.put(None)
            await asyncio.gather(*self._workers, return_exceptions=True)

    def _enqueue_before_stop(self, segment: VoiceSegment) -> None:
        try:
            self.queue.put_nowait(segment)
        except asyncio.QueueFull:
            logger.warning("voice transcript queue full during stop; dropping segment channel=%s", self.channel_id)
