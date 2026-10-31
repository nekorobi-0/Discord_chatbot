"""本文や認証情報を記録せず、応答処理の現在位置を共有する。"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestTrace:
    request_id: str
    phase: str = "queued"
    accepted_at: float = field(default_factory=time.monotonic)


current_trace: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)


def phase(name: str):
    trace = current_trace.get()
    if trace is not None:
        trace.phase = name


def request_id() -> str:
    trace = current_trace.get()
    return trace.request_id if trace is not None else "background"
