"""内容を保存しないメトリクス・監査ログ。"""

import json
from collections import Counter
from datetime import UTC, datetime
from threading import Lock

from .db import data_dir


class Metrics:
    def __init__(self):
        self._counts = Counter()
        self._lock = Lock()

    def increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            self._counts[name] += count

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counts)


metrics = Metrics()


class AuditLog:
    _lock = Lock()

    @classmethod
    def record(cls, event: str, **counts: int) -> None:
        # 本文・ID・入力ハッシュは記録しない。サイズでローテート。
        with cls._lock:
            path = data_dir() / "audit.jsonl"
            if path.exists() and path.stat().st_size >= 5_000_000:
                path.replace(path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": datetime.now(UTC).isoformat(), "event": event, **counts}) + "\n")
