"""保存データを平文でディスクに置かないための暗号化ユーティリティ。

- AES-256-GCM（鍵は起動環境からしか得ない）
- 鍵の指紋をログ・DB に残さない
- 復号失敗時は「読めない」だけで中身は出さない
"""

from __future__ import annotations

import base64
import binascii
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import ConfigurationError

_NONCE_BYTES = 12
_KEY_BYTES = 32


def _key_material() -> bytes:
    raw = os.environ.get("MEMORY_ENCRYPTION_KEY", "").strip().strip('"').strip("'")
    if not raw or "PASTE_YOUR" in raw.upper():
        raise ConfigurationError("記憶を保護する鍵がまだ設定されていないよ。管理者にたずねてくれる？")
    for decoder in (base64.b64decode, base64.b32decode):
        try:
            key = decoder(raw)
        except (ValueError, binascii.Error):  # 候補のエンコーディングが違う場合は次へ
            continue  # noqa: S112
        if len(key) == _KEY_BYTES:
            return key
    try:
        key = base64.b16decode(raw.upper())
    except (ValueError, binascii.Error):
        key = b""
    if len(key) == _KEY_BYTES:
        return key
    raise ConfigurationError("記憶を保護する鍵の形が正しくないみたい。32 バイト分の鍵を用意してほしいな。")


class KeyProvider:
    """起動環境から鍵を読み、プロセスのメモリ上で保持する。"""

    def __init__(self) -> None:
        self._key = _key_material()

    def seal(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return base64.b64encode(nonce + AESGCM(self._key).encrypt(nonce, plaintext, None))

    def open(self, sealed: bytes | str) -> bytes:
        encoded = sealed.encode("ascii") if isinstance(sealed, str) else bytes(sealed)
        blob = base64.b64decode(encoded, validate=True)
        if len(blob) <= _NONCE_BYTES:
            raise ValueError("sealed payload too short")
        return AESGCM(self._key).decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], None)
