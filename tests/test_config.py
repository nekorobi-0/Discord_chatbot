import pytest
from dotenv import dotenv_values

from ai_kun.config import LLMSettings, MemorySettings, OperationalSettings, Settings
from ai_kun.errors import ConfigurationError


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
        "https:///v1",
        "https://u:p@example.com/v1",
        "https://example.com/v1?key=x",
        "https://example.com:bad/v1",
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
