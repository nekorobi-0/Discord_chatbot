from dataclasses import replace

import httpx
import pytest
from dotenv import dotenv_values

from ai_kun.config import LLMSettings, MemorySettings, OperationalSettings, Settings
from ai_kun.errors import ConfigurationError, UpstreamError


def test_example_can_be_configured(monkeypatch):
    for key, value in dotenv_values(".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-discord")
    monkeypatch.setenv("LLM_API_KEY", "fake-llm")
    settings = Settings.from_env()
    assert settings.llm.base_url == "http://127.0.0.1:11434/v1"
    assert settings.embedding.base_url == settings.llm.base_url
    assert settings.discord.guild_id == 000000000000000000


@pytest.mark.parametrize(
    "url",
    [
        "https://llm.example/v1",
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://host.docker.internal:11434/v1",
        # RFC 1918 プライベート範囲とローカル・予約済み範囲は HTTP を許す。
        "http://10.0.0.5:11434/v1",
        "http://172.16.4.20:8000/v1",
        "http://192.168.1.50:11434/v1",
        "http://169.254.169.254:11434/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_valid_base_urls(monkeypatch, url):
    monkeypatch.setenv("LLM_BASE_URL", url)
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL", "model")
    assert LLMSettings.from_env().base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        # プライベート範囲外（グローバル単一縮約含む）への平文は拒否。
        "http://8.8.8.8:11434/v1",
        "http://169.253.0.1:11434/v1",
        "http://172.32.1.1:11434/v1",
        # 100.64.0.0/10 は共有アドレス域でプライベートではない。
        "http://100.64.0.9:11434/v1",
        "https:///v1",
        "https://u:p@example.com/v1",
        "https://example.com/v1?key=x",
        "https://example.com:bad/v1",
        # 名前だけでは解決先が判定できないため、数値アドレスでないと拒否する。
        "http://internal-gpu.lan:11434/v1",
        "http://ollama:11434/v1",
    ],
)
def test_invalid_base_urls(monkeypatch, url):
    monkeypatch.setenv("LLM_BASE_URL", url)
    with pytest.raises(ConfigurationError):
        LLMSettings.from_env()


def test_invalid_chunk_overlap(monkeypatch):
    monkeypatch.setenv("INGEST_CHUNK_SIZE", "100")
    monkeypatch.setenv("INGEST_CHUNK_OVERLAP", "100")
    with pytest.raises(ConfigurationError):
        MemorySettings.from_env()


def test_dashboard_requires_both_credentials(monkeypatch):
    monkeypatch.setenv("DASHBOARD_USERNAME", "admin")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    with pytest.raises(ConfigurationError):
        OperationalSettings.from_env()


def test_local_config_without_api_keys(monkeypatch):
    for key, value in dotenv_values(".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-discord")
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "invalid-old-value")
    settings = Settings.from_env()
    assert settings.llm.api_key == ""
    assert settings.embedding.api_key == ""


@pytest.mark.parametrize(
    "url",
    [
        # RFC 1918・ループバック・リンクローカル・予約済み範囲への平文は許可する。
        "http://10.0.0.5:11434/v1",
        "http://172.16.4.20:8000/v1",
        "http://192.168.1.50:11434/v1",
        "http://169.254.169.254:11434/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8080/v1",
        "http://100.64.0.9:11434/v1",
        "http://[fd12:3456::7]:8080/v1",
        "http://[fe80::2]:11434/v1",
    ],
)
def test_plaintext_to_internal_ranges_is_allowed(url):
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("LLM_BASE_URL", url)
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("LLM_MODEL", "model")
        assert LLMSettings.from_env().base_url == url
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize(
    "url",
    [
        # プライベート範囲外の平文は拒否する。
        "http://8.8.8.8:11434/v1",
        "http://203.0.113.9:11434/v1",
        "http://169.253.0.1:11434/v1",
        "http://172.32.1.1:11434/v1",
        # 試験用・予約済み・6to4・IPv4 変換範囲は is_private が True でも社内扱いしない。
        "http://192.0.2.5:11434/v1",
        "http://198.51.100.9:11434/v1",
        "http://192.0.0.1:11434/v1",
        "http://198.18.0.1:11434/v1",
        "http://240.0.0.1:11434/v1",
        "http://0.0.0.0:11434/v1",
        "http://[2001:db8::1]:11434/v1",
        "http://[ff02::1]:11434/v1",
        "http://[64:ff9b::7f00:1]:11434/v1",
        "http://[2002:7f00:1::1]:11434/v1",
    ],
)
def test_plaintext_to_external_literals_is_rejected(url):
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("LLM_BASE_URL", url)
        with pytest.raises(ConfigurationError):
            LLMSettings.from_env()
    finally:
        monkeypatch.undo()


def test_zone_scoped_literal_is_rejected():
    # ゾーン指定付きアドレスは結線に使えないため拒否する。
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("LLM_BASE_URL", "http://[fe80::1%eth0]:11434/v1")
        with pytest.raises(ConfigurationError):
            LLMSettings.from_env()
    finally:
        monkeypatch.undo()


async def test_plaintext_request_to_global_literal_is_blocked(settings, store, monkeypatch):
    # 設定値がグローバル地址なら、平文要求は結線前に遮断する。名前解決も実接続もさせない。
    from ai_kun.memory import MemoryStore

    async def fail_if_called(self, method, url, **kwargs):
        raise AssertionError(f"request escaped the guard: {url}")

    monkeypatch.setattr(httpx.AsyncClient, "request", fail_if_called)
    guarded = MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)
    guarded._llm = replace(settings.llm, base_url="http://203.0.113.9:11434/v1", api_key="")
    with pytest.raises(UpstreamError):
        await guarded.complete_chat([{"role": "user", "content": "こんにちは"}])


async def test_plaintext_request_to_internal_literal_still_works(settings, store, monkeypatch):
    # ループバック上のEmbeddingは従来どおり使え、応答をそのまま復号して取り扱う。
    import httpx

    seen = []

    async def handler(self, method, url, **kwargs):
        seen.append(url)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0] + [0.0] * (settings.embedding.dimension - 1)}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", handler)
    store._embedding = replace(settings.embedding, base_url="http://127.0.0.1:11434/v1", api_key="")
    vectors = await store.embed(["こんにちは"])
    assert seen == ["http://127.0.0.1:11434/v1/embeddings"]
    assert len(vectors) == 1 and vectors[0][0] == 1.0


async def test_https_to_configured_host_is_unchanged(settings, store, monkeypatch):
    # HTTPS の中継先は従来どおり検証せず、設定された名前そのままへ要求する。
    import httpx

    seen = []

    async def handler(self, method, url, **kwargs):
        seen.append(url)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * settings.embedding.dimension}]})

    monkeypatch.setattr(httpx.AsyncClient, "request", handler)
    vectors = await store.embed(["こんにちは"])
    assert seen == ["https://embed.example/v1/embeddings"]
    assert len(vectors) == 1 and vectors[0][0] == 1.0


@pytest.mark.parametrize(
    "url",
    [
        # 設定値が試験用・予約済み範囲なら、結線直前でも平文要求を遮断する。
        "http://203.0.113.9:11434/v1",
        "http://198.51.100.9:11434/v1",
        "http://192.0.2.5:11434/v1",
        "http://240.0.0.1:11434/v1",
        "http://[2001:db8::1]:11434/v1",
    ],
)
async def test_plaintext_request_to_non_internal_literal_is_blocked(settings, store, monkeypatch, url):
    from dataclasses import replace as _replace

    from ai_kun.memory import MemoryStore

    async def fail_if_called(self, method, target, **kwargs):
        raise AssertionError(f"request escaped the guard: {target}")

    monkeypatch.setattr(httpx.AsyncClient, "request", fail_if_called)
    guarded = MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)
    guarded._llm = _replace(settings.llm, base_url=url, api_key="")
    with pytest.raises(UpstreamError):
        await guarded.complete_chat([{"role": "user", "content": "こんにちは"}])


async def test_plaintext_request_to_internal_literal_is_not_rewritten(settings, store, monkeypatch):
    # 承認済み範囲への平文要求はそのまま透過し、設定値がそのまま使われることを確認する。
    import httpx

    seen = []

    async def handler(self, method, target, **kwargs):
        seen.append(target)
        return httpx.Response(200, json={"choices": [{"message": {"content": "うん"}}]})

    from ai_kun.memory import MemoryStore

    monkeypatch.setattr(httpx.AsyncClient, "request", handler)
    guarded = MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)
    guarded._llm = replace(settings.llm, base_url="http://10.0.0.5:11434/v1", api_key="")
    turn = await guarded.complete_chat([{"role": "user", "content": "こんにちは"}])
    assert seen == ["http://10.0.0.5:11434/v1/chat/completions"]
    assert turn.content == "うん"


async def test_guard_runs_before_any_socket_is_opened(settings, store, monkeypatch):
    # 遮断は接続確立より前。トランスポート生成時点で要求が外へ出ることを許さない。
    from dataclasses import replace as _replace

    from ai_kun.memory import MemoryStore

    def boom(*args, **kwargs):
        raise AssertionError("an HTTP client was built for a rejected destination")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", boom)
    guarded = MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)
    guarded._llm = _replace(settings.llm, base_url="http://203.0.113.9:11434/v1", api_key="")
    with pytest.raises(UpstreamError):
        await guarded.complete_chat([{"role": "user", "content": "こんにちは"}])
