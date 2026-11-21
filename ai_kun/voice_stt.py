"""Receive Discord voice audio, split it by speaker and silence, and store local STT in RAG."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import wave
from array import array
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
    last_voice_at: float
    trailing_silence_bytes: int = 0


if voice_recv is not None:

    class _TranscriptSink(voice_recv.AudioSink):
        def __init__(self, emit: Callable[[VoiceSegment], None], *, max_seconds: int = 16):
            super().__init__()
            self._emit = emit
            self._max_bytes = BYTES_PER_SECOND * max_seconds
            self._buffers: dict[int, _SpeakerBuffer] = {}
            self._lock = threading.RLock()

        def wants_opus(self) -> bool:
            return False

        @staticmethod
        def _rms(pcm: bytes) -> float:
            samples = array("h")
            samples.frombytes(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH])
            if not samples:
                return 0.0
            return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5

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
            if state is None or len(state.pcm) < BYTES_PER_SECOND * 0.35:
                return None
            return VoiceSegment(bytes(state.pcm), state.author, state.started_at)

        def write(self, user, data) -> None:
            if user is None or getattr(user, "bot", False) or data.pcm is None:
                return
            author = self._author(user)
            if not author["id"].isdigit():
                return
            now = time.monotonic()
            pcm = bytes(data.pcm)
            loud = self._rms(pcm) >= 240
            segment = None
            user_id = int(author["id"])
            with self._lock:
                state = self._buffers.get(user_id)
                if loud:
                    if state is None:
                        state = _SpeakerBuffer(author, bytearray(), datetime.now(UTC), now)
                        self._buffers[user_id] = state
                    state.author = author
                    state.last_voice_at = now
                    state.trailing_silence_bytes = 0
                    state.pcm.extend(pcm)
                elif state is not None:
                    next_silence = state.trailing_silence_bytes + len(pcm)
                    if next_silence <= BYTES_PER_SECOND // 4:
                        state.pcm.extend(pcm)
                        state.trailing_silence_bytes = next_silence
                if state is not None and len(state.pcm) >= self._max_bytes:
                    segment = self._take(user_id)
            if segment is not None:
                self._emit(segment)

        def flush_idle(self, *, idle_seconds: float = 1.2) -> list[VoiceSegment]:
            now = time.monotonic()
            with self._lock:
                return [
                    segment
                    for user_id, state in list(self._buffers.items())
                    if now - state.last_voice_at >= idle_seconds
                    if (segment := self._take(user_id)) is not None
                ]

        def flush_all(self) -> list[VoiceSegment]:
            with self._lock:
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
        self._worker: asyncio.Task | None = None
        self._flusher: asyncio.Task | None = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._consume(), name=f"voice-stt-{self.guild_id}")
        self._flusher = asyncio.create_task(self._flush_loop(), name=f"voice-stt-flush-{self.guild_id}")
        try:
            self.voice_client.listen(self.sink, after=self._after)
        except Exception:
            self._stopping = True
            self._flusher.cancel()
            self._worker.cancel()
            await asyncio.gather(self._flusher, self._worker, return_exceptions=True)
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

    async def _flush_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(0.4)
            for segment in self.sink.flush_idle():
                self._enqueue(segment)

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
        if self._flusher is not None:
            self._flusher.cancel()
            await asyncio.gather(self._flusher, return_exceptions=True)
        for segment in self.sink.flush_all():
            self._enqueue_before_stop(segment)
        await self.queue.join()
        if self._worker is not None:
            await self.queue.put(None)
            await self._worker

    def _enqueue_before_stop(self, segment: VoiceSegment) -> None:
        try:
            self.queue.put_nowait(segment)
        except asyncio.QueueFull:
            logger.warning("voice transcript queue full during stop; dropping segment channel=%s", self.channel_id)
