import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.chat import ChatService


async def test_ready_sweeps_idle_channels_and_does_not_duplicate_loop(settings, store, monkeypatch):
    service = ChatService(settings, store)
    backfill = AsyncMock(return_value={"scanned": 0, "imported": 0, "skipped": 0, "documents": 0})
    monkeypatch.setattr(service, "backfill_channel", backfill)
    client = AIKunClient(settings, service)

    def channel(channel_id, readable=True):
        return SimpleNamespace(
            id=channel_id,
            history=Mock(),
            permissions_for=lambda member: SimpleNamespace(view_channel=readable, read_message_history=readable),
        )

    guild = SimpleNamespace(me=object(), channels=[channel(34567), channel(34568, False)], threads=[channel(45678)])
    monkeypatch.setattr(client, "get_guild", lambda guild_id: guild if guild_id == settings.discord.guild_id else None)
    try:
        await client.on_ready()
        loop = client._history_loop_task
        await client.on_ready()
        assert client._history_loop_task is loop
        for _ in range(5):
            await asyncio.sleep(0)
        assert {call.args[0] for call in backfill.await_args_list} == {34567, 45678}
        assert not loop.done()  # 初回だけで終了せず、次の巡回を待つ。
        await asyncio.gather(*client._history_tasks)
        client._sweep_history()  # 発言が来なくても、次回の巡回で再確認する。
        await asyncio.gather(*client._history_tasks)
        assert backfill.await_count == 4
    finally:
        await client.drain()
        await client.close()
    assert loop.done()


async def test_periodic_sweep_does_not_queue_duplicate_scans_while_running(settings, store, monkeypatch):
    service = ChatService(settings, store)
    release = asyncio.Event()

    async def scan(channel_id):
        await release.wait()
        return {"scanned": 0, "imported": 0, "skipped": 0, "documents": 0}

    backfill = AsyncMock(side_effect=scan)
    monkeypatch.setattr(service, "backfill_channel", backfill)
    client = AIKunClient(settings, service)
    channel = SimpleNamespace(
        id=34567,
        history=Mock(),
        permissions_for=lambda member: SimpleNamespace(
            view_channel=True,
            read_message_history=True,
        ),
    )
    monkeypatch.setattr(
        client, "get_guild", lambda guild_id: SimpleNamespace(me=object(), channels=[channel], threads=[])
    )
    try:
        client._sweep_history()
        await asyncio.sleep(0)
        client._sweep_history()
        assert backfill.await_count == 1
        assert len(client._history_tasks) == 1
    finally:
        release.set()
        await client.drain()
        await client.close()
