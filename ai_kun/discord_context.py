"""今回のDiscord発言と返信チェーンの文脈。会話をリクエスト間では保持しない。"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscordImage:
    message_id: str
    attachment_id: str | None
    filename: str
    data_url: str


@dataclass(frozen=True)
class DiscordContext:
    message_id: str | None = None
    reply_to_message_id: str | None = None
    messages: tuple[Mapping[str, object], ...] = ()
    stop_reason: str | None = None
    author_id: str | None = None
    images: tuple[DiscordImage, ...] = ()
