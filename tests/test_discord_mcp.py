import json
import sys
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_kun.chat import ChatService
from ai_kun.discord_mcp import DiscordMCP, DiscordReader, memory_session
from ai_kun.history import DiscordHistoryAPI, HalfRateLimiter
from ai_kun.memory import ModelTurn, ToolCall
from ai_kun.tools import ToolHub


class FakeAPI:
    def __init__(self, guild_id="23456", count=2):
        self.guild_id = guild_id
        self.count = count
        self.calls = []

    async def get(self, resource, **kwargs):
        self.calls.append((str(resource), kwargs))
        if kwargs.get("expect_list"):
            return [{"id": str(34567 + index), "name": f"ch-{index}", "type": 0} for index in range(self.count)]
        if kwargs.get("messages"):
            return [self.message(str(50002 - index)) for index in range(min(self.count, kwargs.get("limit", 20)))]
        if str(resource).endswith("threads/active"):
            return {"threads": [{"id": "45678", "name": "thread", "guild_id": self.guild_id, "parent_id": "34567"}]}
        if "/messages/" in str(resource):
            return self.message(str(resource).split("/")[-1])
        return {"id": str(resource), "guild_id": self.guild_id}

    @staticmethod
    def message(message_id):
        return {
            "id": message_id,
            "content": "締切は金曜日です。alice@example.com",
            "channel_id": "34567",
            "author": {"id": "11111", "username": "member"},
            "timestamp": "2026-10-31T00:00:00Z",
        }


def client_for(settings, api):
    return DiscordMCP(settings.discord, reader=DiscordReader(settings.discord, api=api))


async def test_real_mcp_initialize_list_and_read_messages(settings):
    api = FakeAPI()
    client = client_for(settings, api)
    async with memory_session(client.server) as session:
        listed = await session.list_tools()
        assert {tool.name for tool in listed.tools} == {
            "discord_list_channels",
            "discord_read_messages",
            "discord_get_message",
            "discord_list_active_threads",
        }
        assert all(tool.annotations.readOnlyHint and not tool.annotations.destructiveHint for tool in listed.tools)
        result = await session.call_tool("discord_read_messages", {"channel_id": "34567", "limit": 2})
        assert not result.isError
        data = result.structuredContent["untrusted_data"]
        assert data["returned_count"] == 2
        assert data["next_before_message_id"] == "50001"
        assert "alice@example.com" in data["messages"][0]["content"]
    assert len(api.calls) == 2
    assert api.calls[1][1]["after"] is None  # 既定でafter=0を送らず、直近の発言を読む。


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("discord_read_messages", {"channel_id": "../34567"}),
        ("discord_read_messages", {"channel_id": "34567", "limit": 101}),
        ("discord_read_messages", {"channel_id": "34567", "limit": True}),
        ("discord_read_messages", {"channel_id": "34567", "method": "POST", "content": "send"}),
        ("discord_read_messages", {"channel_id": "34567", "before_message_id": "50001", "after_message_id": "50002"}),
        ("discord_get_message", {"channel_id": "34567"}),
        ("discord_send_message", {"channel_id": "34567", "content": "send"}),
        ("discord_delete_message", {"channel_id": "34567", "message_id": "50001"}),
    ],
)
async def test_mcp_rejects_invalid_and_write_operations_before_api(settings, name, arguments):
    api = FakeAPI()
    data = json.loads(await client_for(settings, api).execute(name, arguments))
    assert "error" in data
    assert api.calls == []


@pytest.mark.parametrize("guild", ["99999", None])
async def test_mcp_rejects_other_guild_and_dm(settings, guild):
    api = FakeAPI(guild_id=guild)
    data = json.loads(await client_for(settings, api).execute("discord_read_messages", {"channel_id": "34567"}))
    assert "error" in data and "サーバー以外" in data["error"]
    assert len(api.calls) == 1  # 所属確認のGETだけ。メッセージを取得しない。


async def test_mcp_lists_channels_threads_and_reads_one_message(settings):
    api = FakeAPI(count=3)
    client = client_for(settings, api)
    channels = json.loads(await client.execute("discord_list_channels", {"limit": 1}))["untrusted_data"]
    assert channels["channels"][0]["id"] == "34567"
    assert channels["next_offset"] == 1
    second = json.loads(await client.execute("discord_list_channels", {"limit": 1, "offset": 1}))["untrusted_data"]
    assert second["channels"][0]["id"] == "34568"
    threads = json.loads(await client.execute("discord_list_active_threads", {}))["untrusted_data"]
    assert threads["threads"][0]["id"] == "45678"
    message = json.loads(await client.execute("discord_get_message", {"channel_id": "34567", "message_id": "50001"}))
    assert message["untrusted_data"]["message"]["id"] == "50001"


async def test_discord_mcp_completes_llm_tool_round_trip(settings, store, monkeypatch):
    service = ChatService(settings, store)
    api = FakeAPI()
    service._tools.discord_mcp = client_for(settings, api)
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    complete = AsyncMock(
        side_effect=[
            ModelTurn(tool_calls=(ToolCall("discord-1", "discord_read_messages", '{"channel_id":"34567"}'),)),
            ModelTurn(content="直近の話では、締切は金曜日とのこと。"),
        ]
    )
    monkeypatch.setattr(store, "complete_chat", complete)
    result = await service.generate_reply("このチャンネルの直近の話題は？", channel_id=34567)
    assert "金曜日" in result.content
    messages = complete.await_args.args[0]
    assert "今回のDiscordチャンネルID: 34567" in messages[0]["content"]
    assert any(message["role"] == "tool" and "untrusted_data" in message["content"] for message in messages)
    assert len(api.calls) == 2


async def test_discord_tools_available_without_web_search(settings):
    hub = ToolHub(settings.search, discord=settings.discord)
    assert len(hub.specs()) == 4
    assert "error" in json.loads(await hub.execute("web_search", {"query": "test"}))


async def test_history_api_pagination_uses_only_get():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        return httpx.Response(200, json=[])

    async def no_wait(seconds):
        pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        api = DiscordHistoryAPI("fake", http, HalfRateLimiter(sleep=no_wait))
        await api.get("34567", messages=True, limit=2, before="50001")
        await api.get("34567", messages=True, limit=2)
        await api.get("guilds/23456/channels", expect_list=True)
    assert requests[0].url.params["before"] == "50001"
    assert "after" not in requests[0].url.params
    assert set(requests[1].url.params) == {"limit"}


async def test_stdio_mcp_server_lists_tools_without_discord_network(settings):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ai_kun.discord_mcp"],
        env={
            "DISCORD_BOT_TOKEN": "fake-token",
            "DISCORD_APPLICATION_ID": "12345",
            "GUILD_ID": "23456",
        },
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert len(tools.tools) == 4


async def test_reply_chain_message_id_can_drive_mcp_get_message(settings, store, monkeypatch):
    from ai_kun.discord_context import DiscordContext

    service = ChatService(settings, store)
    api = FakeAPI()
    service._tools.discord_mcp = client_for(settings, api)
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    context = DiscordContext(
        message_id="90003",
        reply_to_message_id="50002",
        messages=({"message_id": "50002", "channel_id": "34567", "guild_id": "23456", "content": "前の発言"},),
    )
    calls = 0

    async def complete(messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert "今回のDiscordメッセージID: 90003" in messages[0]["content"]
            assert "今回の返信先メッセージID: 50002" in messages[0]["content"]
            assert any('"message_id": "50002"' in message["content"] for message in messages)
            return ModelTurn(
                tool_calls=(ToolCall("reply-id", "discord_get_message", '{"channel_id":"34567","message_id":"50002"}'),)
            )
        return ModelTurn(content="返信先の発言も確認しました。")

    monkeypatch.setattr(store, "complete_chat", complete)
    result = await service.generate_reply("返信先を詳しく確認して", channel_id=34567, discord_context=context)
    assert result.content == "返信先の発言も確認しました。"
    assert api.calls[-1][0] == "channels/34567/messages/50002"
