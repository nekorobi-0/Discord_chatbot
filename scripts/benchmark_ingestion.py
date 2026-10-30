"""一時DBで取り込みを測る。例: uv run python -m scripts.benchmark_ingestion --mode live"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy import event, func
from sqlmodel import Session, select

from ai_kun.chat import ChatService
from ai_kun.config import (
    ASRSettings,
    DiscordSettings,
    EmbeddingSettings,
    LLMSettings,
    MemorySettings,
    OperationalSettings,
    SearchSettings,
    Settings,
)
from ai_kun.db import HistoryMessageRecord, MemoryChunkRecord, get_engine
from ai_kun.memory import MemoryStore


async def benchmark(args):
    store = MemoryStore(
        EmbeddingSettings(args.embedding_url, os.getenv("EMBEDDING_API_KEY", ""), args.model, args.dimension),
        LLMSettings("https://unused.example/v1", "", "unused"),
        ASRSettings(),
        MemorySettings(),
    )
    await store._embed_uncached("ウォームアップです。")
    counts = Counter()

    def query(conn, cursor, statement, parameters, context, executemany):
        counts[statement.split()[0].upper()] += 1

    def commit(conn):
        counts["COMMIT"] += 1

    original = store._request

    async def request(method, url, **kwargs):
        counts["HTTP"] += 1
        return await original(method, url, **kwargs)

    store._request = request
    engine = get_engine()
    event.listen(engine, "before_cursor_execute", query)
    event.listen(engine, "commit", commit)
    messages = [
        {
            "id": str(1000 + index),
            "content": f"性能測定用の記録{index:03d}。"
            + "週次レビューでは実装と運用の確認を行い、対応する項目と担当を明確にします。" * 2,
            "timestamp": "2026-10-31T00:00:00+00:00",
            "author": {"id": str(42 + index % 5)},
        }
        for index in range(args.messages)
    ]
    started = time.perf_counter()
    try:
        if args.mode == "live":
            settings = Settings(
                DiscordSettings("unused", 12345, 23456),
                store._llm,
                store._embedding,
                ASRSettings(),
                SearchSettings(enabled=False),
                MemorySettings(),
                OperationalSettings(),
            )
            service = ChatService(settings, store)
            requests = [
                service.ingest_message(
                    SimpleNamespace(
                        id=message["id"],
                        content=message["content"],
                        created_at=datetime(2026, 10, 31, tzinfo=UTC),
                        author=SimpleNamespace(id=message["author"]["id"], bot=False),
                        guild=SimpleNamespace(id=23456),
                        channel=SimpleNamespace(id=34567),
                        webhook_id=None,
                    )
                )
                for message in messages
            ]
            await asyncio.gather(*requests)
            await service.drain()
        else:
            await store.ingest_history_page(23456, 34567, messages)
        elapsed = time.perf_counter() - started
    finally:
        event.remove(engine, "before_cursor_execute", query)
        event.remove(engine, "commit", commit)
    with Session(engine) as session:
        stored = session.exec(select(func.count()).select_from(HistoryMessageRecord)).one()
        chunks = session.exec(select(func.count()).select_from(MemoryChunkRecord)).one()
    assert stored == args.messages, "一部の発言が保存されていません"
    print(
        json.dumps(
            {
                "mode": args.mode,
                "messages": stored,
                "chunks": chunks,
                "elapsed_seconds": round(elapsed, 3),
                "operations": dict(counts),
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="一時DBを使うRAG取り込みベンチマーク（Discord APIは呼ばない）")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18434/v1")
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--messages", type=int, default=100)
    parser.add_argument("--mode", choices=("live", "page"), default="live")
    args = parser.parse_args()
    if args.messages < 1:
        parser.error("--messages は1以上")
    previous = os.environ.get("DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="ai-kun-benchmark-") as directory:
        os.environ["DATA_DIR"] = directory
        get_engine.cache_clear()
        try:
            asyncio.run(benchmark(args))
        finally:
            get_engine().dispose()
            get_engine.cache_clear()
            if previous is None:
                os.environ.pop("DATA_DIR", None)
            else:
                os.environ["DATA_DIR"] = previous


if __name__ == "__main__":
    main()
