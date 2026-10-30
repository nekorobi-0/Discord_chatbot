"""保存する目録（メタデータ）を1本の暗号化封書にまとめる共通処理。

話者ID・チャンネルID・添付ファイルの有無は、JSON を AES-GCM で保護した
Base64 文字列としてだけ扱う。平文のまま DB に行を作らないため、保存用モデルは
常に単一のカラムしか持たない（検証済み: exclude=True の bytes は SQLAlchemy の
カラムになり得ず、model_dump・repr・ログに平文が現れない）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .crypto import KeyProvider
from .errors import InvalidInputError

# 検出した添付の種別。検索時の絞り込み条件としてだけ使い、拡張子やファイル名は保存しない。
ATTACHMENT_NONE = 0
ATTACHMENT_IMAGE = 1
ATTACHMENT_VIDEO = 2
ATTACHMENT_AUDIO = 3
ATTACHMENT_FILE = 4

_CDN_HOSTS = ("cdn.discordapp.com", "media.discordapp.net", "media-spoiler")
_IMAGE_MARKERS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".heic")
_VIDEO_MARKERS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", "video_", "/videos/")
_AUDIO_MARKERS = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac", "voice_", "audio_")

_MAX_GRANT_ITEMS = 32


def classify_attachment(url: object, proxy: bool = False) -> int:
    """URL・プロキシ名から種別だけを推定する。手がかりが無ければ『その他ファイル』。"""
    name = str(url or "").lower()
    if not name:
        return ATTACHMENT_NONE
    if proxy or any(host in name for host in _CDN_HOSTS):
        for markers, kind in (
            (_IMAGE_MARKERS, ATTACHMENT_IMAGE),
            (_VIDEO_MARKERS, ATTACHMENT_VIDEO),
            (_AUDIO_MARKERS, ATTACHMENT_AUDIO),
        ):
            if any(marker in name for marker in markers):
                return kind
        return ATTACHMENT_FILE
    return ATTACHMENT_NONE


def detect_flags(messages: Sequence[Mapping[str, object]]) -> int:
    """ページ全体について画像・動画・その他ファイルの有無をビットでまとめる。"""
    flags = 0
    for message in messages:
        for attachment in message.get("attachments") or ():
            if not isinstance(attachment, Mapping):
                continue
            kind = classify_attachment(attachment.get("url"), proxy=bool(attachment.get("proxy_url")))
            if kind == ATTACHMENT_IMAGE:
                flags |= 1
            elif kind == ATTACHMENT_VIDEO:
                flags |= 2
            elif kind == ATTACHMENT_AUDIO:
                flags |= 4
            else:
                flags |= 8
    return flags


@dataclass(frozen=True)
class MetadataGrant:
    """検索時に利用者が指定できる絞り込み条件。"""

    user_ids: frozenset[str] = frozenset()
    channel_ids: frozenset[str] = frozenset()
    require_image: bool = False
    require_video: bool = False
    require_other_file: bool = False

    @property
    def empty(self) -> bool:
        return not (
            self.user_ids or self.channel_ids or self.require_image or self.require_video or self.require_other_file
        )


class MetadataEnvelope:
    """封書の作成と検証。鍵が無い状態ではメタデータを一切扱わない。"""

    def __init__(self, crypto: KeyProvider | None) -> None:
        if crypto is None:
            raise InvalidInputError("記憶の暗号鍵がまだ準備できていないよ。管理者にたずねてくれる？")
        self._crypto = crypto

    def seal(self, payload: Mapping[str, object]) -> bytes:
        return self._crypto.seal(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def open(self, sealed: object) -> dict[str, object]:
        if not isinstance(sealed, (bytes, str)):
            raise InvalidInputError("記憶の目録を読み取れなかったよ。")
        try:
            parsed = json.loads(self._crypto.open(sealed).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise InvalidInputError("記憶の目録を読み取れなかったよ。") from exc
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def matches(payload: Mapping[str, object], grant: MetadataGrant) -> bool:
        """条件を1つでも満たさない項目は検索結果から落とす。"""
        flags = int(payload.get("flags") or 0)
        if grant.require_image and not flags & 1:
            return False
        if grant.require_video and not flags & 2:
            return False
        if grant.require_other_file and not flags & 12:
            return False
        if grant.channel_ids and str(payload.get("channel_id") or "") not in grant.channel_ids:
            return False
        if grant.user_ids:
            authors = payload.get("authors")
            names = (
                [str(item) for item in authors]
                if isinstance(authors, Sequence) and not isinstance(authors, (str, bytes))
                else []
            )
            if not grant.user_ids.intersection(names):
                return False
        return True

    @staticmethod
    def _normalize(values: object, limit: int, label: str) -> frozenset[str]:
        if values is None or (isinstance(values, str) and not values.strip()):
            return frozenset()
        if isinstance(values, str):
            items = values.replace(";", ",").split(",")
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            items = list(values)
        else:
            items = [values]
        cleaned = {str(item).strip() for item in items if str(item or "").strip()}
        if len(cleaned) > limit:
            raise InvalidInputError(f"{label}の指定が多すぎるよ。{limit}件までにしてくれる？")
        return frozenset(cleaned)

    @classmethod
    def normalize_user_ids(cls, values: object) -> frozenset[str]:
        return cls._normalize(values, _MAX_GRANT_ITEMS, "話者")

    @classmethod
    def normalize_channel_ids(cls, values: object) -> frozenset[str]:
        return cls._normalize(values, _MAX_GRANT_ITEMS, "チャンネル")
