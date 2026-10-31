"""設定読み込み。

このファイルは環境変数からしか値を読まない。
.env はコンテナのビルドには一切渡さず、実行時にのみ必要になる。
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .errors import ConfigurationError

_TRUE = {"1", "true", "yes", "on", "はい", "やる"}
_FALSE = {"0", "false", "no", "off", "いいえ", "やらない", ""}
_SNOWFLAKE_RE = re.compile(r"^\d{5,25}$")
_ALLOWED_EMBEDDING_DIMS = frozenset({64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072})
_MAX_LIST_ITEMS = 64


def _raw(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def _secret(name: str) -> str:
    """秘密鍵专用。前後の空白と引用符だけ落として中身はそのまま返す。"""
    value = os.environ.get(name)
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def _required_secret(name: str) -> str:
    value = _secret(name)
    if not value or "PASTE_YOUR" in value.upper():
        raise ConfigurationError(f"{name} が設定されてないみたい。管理者にたずねてくれる？")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE and raw != "":
        return False
    if raw == "":
        return default
    raise ConfigurationError(f"{name} は true か false で教えてほしいな。")


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = _raw(name)
    if raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name} は数字で指定してほしいな。") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} は {minimum}〜{maximum} のあいだでおねがい。")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = _raw(name)
    if raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name} は数字で指定してほしいな。") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} は {minimum}〜{maximum} のあいだでおねがい。")
    return value


# 平文HTTPはループバックとRFC1918のプライベートIPv4アドレスに許可する。
_RFC1918_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
_INTERNAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-loopback"})


def _is_internal_host(host: str) -> bool:
    """ループバックまたはRFC1918かを、名前解決せずに判定する。"""
    candidate = host.strip("[]").lower()
    if "%" in candidate:  # ゾーン指定付き（fe80::1%eth0 など）は結線に使えないため承認しない
        return False
    if candidate in _INTERNAL_HOSTNAMES:  # 承認済み名前は数値表記でなくてもよい
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False  # 名前解決が使えない環境では、未知のホスト名は平文の送信先にならない
    return address.is_loopback or any(address in network for network in _RFC1918_NETWORKS)


def _https_base_url(name: str, default: str = "") -> str:
    raw = _raw(name) or default
    if not raw:
        raise ConfigurationError(f"{name} が設定されてないみたい。管理者にたずねてくれる？")
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()

    try:
        _ = parsed.port
    except ValueError as exc:
        raise ConfigurationError(f"{name} のポート番号が不正です。") from exc
    valid = host and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
    # https は常時可。http は RFC 1918 を含むプライベート・ローカル範囲のときだけ許す。
    if valid and (parsed.scheme == "https" or (parsed.scheme == "http" and _is_internal_host(host))):
        return raw.rstrip("/")
    raise ConfigurationError(
        f"{name} は HTTPS URL、または社内・ローカル（プライベートアドレス）向けの HTTP URL で指定してください。"
    )


def _model_id(name: str, default: str = "") -> str:
    raw = _raw(name) or default
    if not raw or len(raw) > 200 or any(ch.isspace() for ch in raw):
        raise ConfigurationError(f"{name} はモデル ID 1 個だけを書いてほしいな。")
    return raw


def _embedding_dimension(name: str, default: int) -> int:
    value = _int(name, default, minimum=64, maximum=3072)
    if value not in _ALLOWED_EMBEDDING_DIMS:
        raise ConfigurationError(f"{name} は 64〜3072 の一般的な次元数で指定してほしいな。")
    return value


@dataclass(frozen=True)
class DiscordSettings:
    """Discord 連携の設定。トークンは repr に出さない。"""

    bot_token: str = field(repr=False, default="")
    application_id: int = 0
    guild_id: int = 0

    @classmethod
    def from_env(cls) -> DiscordSettings:
        token = _required_secret("DISCORD_BOT_TOKEN")
        app_raw = _raw("DISCORD_APPLICATION_ID")
        guild_raw = _raw("GUILD_ID")
        if not _SNOWFLAKE_RE.fullmatch(app_raw):
            raise ConfigurationError("DISCORD_APPLICATION_ID が設定されてないみたい。数字だけで教えてくれる？")
        if not _SNOWFLAKE_RE.fullmatch(guild_raw):
            raise ConfigurationError("GUILD_ID が設定されてないみたい。数字だけで教えてくれる？")
        return cls(
            bot_token=token,
            application_id=int(app_raw),
            guild_id=int(guild_raw),
        )


@dataclass(frozen=True)
class LLMSettings:
    """テキスト生成モデルの設定。"""

    base_url: str = ""
    api_key: str = field(repr=False, default="")
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 10240
    timeout_seconds: float = 90.0
    first_token_timeout_seconds: float = 300.0
    stream_timeout_seconds: float = 900.0
    stream_enabled: bool = True

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls(
            base_url=_https_base_url("LLM_BASE_URL"),
            api_key=_secret("LLM_API_KEY"),
            model=_model_id("LLM_MODEL"),
            temperature=_float("LLM_TEMPERATURE", 0.7, 0.0, 2.0),
            max_tokens=_int("LLM_MAX_TOKENS", 10240, 64, 65536),
            timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 90.0, 5.0, 300.0),
            first_token_timeout_seconds=_float("LLM_FIRST_TOKEN_TIMEOUT_SECONDS", 300.0, 5.0, 1800.0),
            stream_timeout_seconds=_float("LLM_STREAM_TIMEOUT_SECONDS", 900.0, 30.0, 3600.0),
            stream_enabled=_bool("LLM_STREAM_ENABLED", True),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """記憶の埋め込み用モデルの設定。"""

    base_url: str = ""
    api_key: str = field(repr=False, default="")
    model: str = ""
    dimension: int = 1536

    @classmethod
    def from_env(cls) -> EmbeddingSettings:
        base = _raw("EMBEDDING_BASE_URL") or _raw("LLM_BASE_URL")
        key = _secret("EMBEDDING_API_KEY") or _secret("LLM_API_KEY")
        return cls(
            base_url=_https_base_url("EMBEDDING_BASE_URL", default=base),
            api_key=key,
            model=_model_id("EMBEDDING_MODEL"),
            dimension=_embedding_dimension("EMBEDDING_DIMENSION", 1536),
        )


@dataclass(frozen=True)
class ASRSettings:
    """添付音声の文字起こし設定。既定はオフ、モデルはイメージに含めない。"""

    enabled: bool = False
    model_path: str = "/app/data/asr-model"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "ja"

    @classmethod
    def from_env(cls) -> ASRSettings:
        return cls(
            enabled=_bool("ASR_ENABLED", False),
            model_path=_raw("ASR_MODEL_PATH") or "/app/data/asr-model",
            device="cuda" if _raw("ASR_DEVICE").lower().startswith("cuda") else "cpu",
            compute_type=_raw("ASR_COMPUTE_TYPE") or "int8",
            language=(_raw("ASR_LANGUAGE") or "ja")[:8],
        )


@dataclass(frozen=True)
class SearchSettings:
    """Web 検索ツール。キーが無ければツールは安全に停止する。"""

    api_key: str = field(repr=False, default="")
    enabled: bool = True
    max_results: int = 5
    exa_api_key: str = field(repr=False, default="")
    fetch_enabled: bool = True
    fetch_max_chars: int = 12000

    @classmethod
    def from_env(cls) -> SearchSettings:
        return cls(
            api_key=_secret("TAVILY_API_KEY"),
            enabled=_bool("WEB_SEARCH_ENABLED", True),
            max_results=_int("WEB_SEARCH_MAX_RESULTS", 5, 1, 10),
            exa_api_key=_secret("EXA_API_KEY"),
            fetch_enabled=_bool("WEB_FETCH_ENABLED", True),
            fetch_max_chars=_int("WEB_FETCH_MAX_CHARS", 12000, 500, 32000),
        )

    @property
    def configured(self) -> bool:
        return bool(self.exa_api_key or self.api_key)


@dataclass(frozen=True)
class MemorySettings:
    """RAG 記憶の挙動。"""

    enabled: bool = True
    min_score: float = 0.35
    top_k: int = 8
    max_chars_per_document: int = 12_000
    chunk_size: int = 900
    chunk_overlap: int = 150

    @classmethod
    def from_env(cls) -> MemorySettings:
        settings = cls(
            enabled=_bool("MEMORY_ENABLED", True),
            min_score=_float("RETRIEVAL_MIN_SCORE", 0.35, 0.0, 1.0),
            top_k=_int("RETRIEVAL_TOP_K", 8, 1, 24),
            max_chars_per_document=_int("INGEST_MAX_CHARS_PER_DOC", 12_000, 1_000, 200_000),
            chunk_size=_int("INGEST_CHUNK_SIZE", 900, 100, 4_000),
            chunk_overlap=_int("INGEST_CHUNK_OVERLAP", 150, 0, 1_000),
        )

        if settings.chunk_overlap >= settings.chunk_size:
            raise ConfigurationError("INGEST_CHUNK_OVERLAP は CHUNK_SIZE より小さくしてください。")
        return settings


@dataclass(frozen=True)
class OperationalSettings:
    audit_log_enabled: bool = True
    max_output_chars: int = 1_800
    http_enabled: bool = True
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    request_timeout_seconds: float = 1200.0
    max_code_tokens: int = 8192

    @classmethod
    def from_env(cls) -> OperationalSettings:
        return cls(
            audit_log_enabled=_bool("AUDIT_LOG_ENABLED", True),
            max_output_chars=_int("MAX_OUTPUT_CHARS", 1_800, 200, 32_000),
            http_enabled=_bool("HTTP_ENABLED", True),
            http_host=_raw("HTTP_HOST") or "0.0.0.0",
            http_port=_int("HTTP_PORT", 8080, 1, 65535),
            request_timeout_seconds=_float("REQUEST_TIMEOUT_SECONDS", 1200.0, 30.0, 7200.0),
            max_code_tokens=_int("MAX_CODE_TOKENS", 8192, 256, 32768),
        )


@dataclass(frozen=True)
class Settings:
    discord: DiscordSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    asr: ASRSettings
    search: SearchSettings
    memory: MemorySettings
    operational: OperationalSettings

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            discord=DiscordSettings.from_env(),
            llm=LLMSettings.from_env(),
            embedding=EmbeddingSettings.from_env() if _bool("MEMORY_ENABLED", True) else EmbeddingSettings(),
            asr=ASRSettings.from_env(),
            search=SearchSettings.from_env(),
            memory=MemorySettings.from_env(),
            operational=OperationalSettings.from_env(),
        )


def load_settings() -> Settings:
    """環境変数から設定を作る。失敗時はuser-safe なメッセージだけを返す。"""
    return Settings.from_env()
