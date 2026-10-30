"""受付済みの処理をFIFOで実行する。呼び出し元のキャンセルは待機中の仕事へ伝播しない。"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class WorkQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker = None

    async def submit(self, operation: Callable[[], Awaitable[T]]) -> T:
        future = asyncio.get_running_loop().create_future()
        # 呼び出し元が待機をやめても、後で失敗したFutureの例外を回収する。
        future.add_done_callback(lambda result: result.exception() if not result.cancelled() else None)
        self._queue.put_nowait((operation, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._consume())
        return await asyncio.shield(future)

    async def _consume(self):
        while not self._queue.empty():
            operation, future = self._queue.get_nowait()
            try:
                result = await operation()
            except asyncio.CancelledError:
                future.cancel()
                if asyncio.current_task().cancelling():
                    raise
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._queue.task_done()

    async def join(self):
        await self._queue.join()
        if self._worker is not None:
            await self._worker


class MessageQueue:
    """連続する同一チャンネルの発言を短い時間窓でまとめる。全Futureを完了させる。"""

    def __init__(self, handler, *, window: float = 0.025, batch_size: int = 100):
        self._handler = handler
        self._window = window
        self._batch_size = batch_size
        self._queue = asyncio.Queue()
        self._worker = None

    async def submit(self, guild_id: int, channel_id: int, message: dict) -> dict:
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(lambda result: result.exception() if not result.cancelled() else None)
        self._queue.put_nowait(((guild_id, channel_id), message, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._consume())
        return await asyncio.shield(future)

    async def _consume(self):
        from itertools import groupby

        while not self._queue.empty():
            await asyncio.sleep(self._window)
            entries = [self._queue.get_nowait() for _ in range(min(self._queue.qsize(), self._batch_size))]
            for key, group in groupby(entries, key=lambda entry: entry[0]):
                batch = list(group)
                try:
                    result = await self._handler(*key, [entry[1] for entry in batch])
                except asyncio.CancelledError:
                    for _, _, future in batch:
                        future.cancel()
                    if asyncio.current_task().cancelling():
                        raise
                except Exception as exc:
                    for _, _, future in batch:
                        future.set_exception(exc)
                else:
                    for _, _, future in batch:
                        future.set_result(dict(result))
                finally:
                    for _ in batch:
                        self._queue.task_done()

    async def join(self):
        await self._queue.join()
        if self._worker is not None:
            await self._worker
