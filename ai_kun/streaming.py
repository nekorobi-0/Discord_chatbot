"""Discordへの表示と分離した、LLMの本文更新コールバック。"""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar

TextUpdate = Callable[[str], Awaitable[None]]
current_text_update: ContextVar[TextUpdate | None] = ContextVar("text_update", default=None)
