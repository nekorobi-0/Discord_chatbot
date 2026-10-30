import pytest

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
from ai_kun.db import get_engine
from ai_kun.memory import MemoryStore


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_engine.cache_clear()
    yield
    get_engine().dispose()
    get_engine.cache_clear()


@pytest.fixture
def settings():
    return Settings(
        discord=DiscordSettings("test-token", 12345, 23456, (34567,)),
        llm=LLMSettings("https://llm.example/v1", "test-key", "test-model"),
        embedding=EmbeddingSettings("https://embed.example/v1", "test-key", "embedding-test", 64),
        asr=ASRSettings(),
        search=SearchSettings(enabled=False),
        memory=MemorySettings(),
        operational=OperationalSettings(),
    )


@pytest.fixture
def store(settings):
    return MemoryStore(settings.embedding, settings.llm, settings.asr, settings.memory)
