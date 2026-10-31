"""Bot内のMCPクライアントとサーバーをメモリ転送で接続する。"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession
from mcp.server import Server
from mcp.shared.memory import create_client_server_memory_streams


@asynccontextmanager
async def memory_session(server: Server):
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        task = asyncio.create_task(server.run(*server_streams, server.create_initialization_options()))
        try:
            async with ClientSession(*client_streams, read_timeout_seconds=timedelta(seconds=60)) as session:
                await session.initialize()
                yield session
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
