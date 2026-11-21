"""Bounded in-memory fan-out for live voice transcript SSE clients."""

from __future__ import annotations

import asyncio
from collections import deque


class VoiceTranscriptStream:
    def __init__(self, *, history_size: int = 100, subscriber_queue_size: int = 100):
        self._history = deque(maxlen=history_size)
        self._subscribers: set[asyncio.Queue] = set()
        self._subscriber_queue_size = subscriber_queue_size
        self._sequence = 0

    def publish(self, event: dict) -> None:
        self._sequence += 1
        event = {**event, "id": self._sequence}
        self._history.append(event)
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> tuple[asyncio.Queue, tuple[dict, ...]]:
        queue = asyncio.Queue(maxsize=self._subscriber_queue_size)
        self._subscribers.add(queue)
        return queue, tuple(self._history)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
