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


async def test_message_queue_batches_without_dropping_requests_and_preserves_channel_order():
    from ai_kun.work_queue import MessageQueue

    calls = []

    async def handler(guild, channel, messages):
        calls.append((channel, [m["id"] for m in messages]))
        if channel == 2:
            raise ValueError("test failure")
        return {"imported": len(messages)}

    queue = MessageQueue(handler, window=0)
    channels = [1, 1, 2, 1, 3, 3]
    results = await asyncio.gather(
        *[queue.submit(10, channel, {"id": index}) for index, channel in enumerate(channels)], return_exceptions=True
    )
    await queue.join()
    assert calls == [(1, [0, 1]), (2, [2]), (1, [3]), (3, [4, 5])]
    assert isinstance(results[2], ValueError)
    assert [result["imported"] for index, result in enumerate(results) if index != 2] == [2, 2, 1, 2, 2]


async def test_message_queue_keeps_cancelled_waiter_payload():
    from ai_kun.work_queue import MessageQueue

    calls = []

    async def handler(guild, channel, messages):
        calls.extend(message["id"] for message in messages)
        return {"imported": len(messages)}

    queue = MessageQueue(handler, window=0.01)
    task = asyncio.create_task(queue.submit(1, 2, {"id": 100}))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await queue.join()
    assert calls == [100]
