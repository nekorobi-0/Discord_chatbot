import asyncio

import pytest

from ai_kun.work_queue import WorkQueue


async def test_fifo_continues_after_failure_and_accepts_all_requests():
    queue = WorkQueue()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def operation(index):
        calls.append(index)
        if index == 0:
            entered.set()
            await release.wait()
        if index == 1:
            raise ValueError("test failure")
        return index

    tasks = [asyncio.create_task(queue.submit(lambda: operation(0)))]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for index in range(1, 8):
            tasks.append(asyncio.create_task(queue.submit(lambda index=index: operation(index))))
            await asyncio.sleep(0)
        assert calls == [0]
        assert not any(task.done() for task in tasks)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert results[0] == 0
        assert isinstance(results[1], ValueError)
        assert results[2:] == list(range(2, 8))
        await queue.join()
        assert calls == list(range(8))
        # 空になったキューへ後から受け付けた処理も実行される。
        assert await queue.submit(lambda: operation(8)) == 8
        assert calls == list(range(9))
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.join()


async def test_cancelled_waiter_does_not_cancel_accepted_work():
    queue = WorkQueue()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def first():
        entered.set()
        await release.wait()
        calls.append("first")

    async def second():
        calls.append("second")

    first_task = asyncio.create_task(queue.submit(first))
    second_task = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second_task = asyncio.create_task(queue.submit(second))
        await asyncio.sleep(0)
        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task
        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task
        release.set()
        await queue.join()
        assert calls == ["first", "second"]
    finally:
        release.set()
        await asyncio.gather(first_task, *([second_task] if second_task else []), return_exceptions=True)
        await queue.join()
