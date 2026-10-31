import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_kun.adapters.discord_bot import AIKunClient, split_message
from ai_kun.adapters.stream_reply import StreamReply
from ai_kun.chat import ChatService
from ai_kun.code_output import prepare_reply
from ai_kun.memory import ModelTurn
from ai_kun.text_files import file_encoding
from tests.test_adapters import file_message
from tests.test_stream_reply import until


@pytest.mark.parametrize(
    "language,extension,label",
    [
        ("python", ".py", "Python"),
        ("js", ".js", "JavaScript"),
        ("typescript", ".ts", "TypeScript"),
        ("tsx", ".tsx", "TSX"),
        ("java", ".java", "Java"),
        ("cpp", ".cpp", "C++"),
        ("c#", ".cs", "C#"),
        ("go", ".go", "Go"),
        ("rust", ".rs", "Rust"),
        ("sql", ".sql", "SQL"),
        ("yaml", ".yaml", "YAML"),
    ],
)
def test_code_is_an_attachment_in_correct_format(language, extension, label):
    body = '    value = "日本語😀"\n'
    result = prepare_reply(f"説明。\n```{language}\n{body}```\n")
    assert len(result.files) == 1
    assert result.files[0].filename.endswith(extension)
    assert result.files[0].content == body
    assert result.files[0].complete
    assert f"{label}コード[" in result
    assert body not in str(result)


def test_java_public_class_name_and_explicit_filename_are_preserved():
    result = prepare_reply("```java\npublic class Main { }\n```")
    assert result.files[0].filename == "Main.java"
    result = prepare_reply('```python filename="../../main.py"\nprint(1)\n```')
    assert result.files[0].filename == "main.py"


def test_nested_fences_inline_blocks_and_midline_opening_do_not_lose_code():
    body = "```python\nprint(1)\n```\n"
    result = prepare_reply("````markdown\n" + body + "````")
    assert result.files[0].content == body
    assert result.files[0].filename.endswith(".md")
    assert prepare_reply("例：```print(1)```").files[0].content == "print(1)"
    assert prepare_reply("例：```python\nprint(1)\n```").files[0].content == "print(1)\n"


def test_unclosed_code_is_a_partial_file_not_a_successful_complete_file():
    result = prepare_reply("```python\ndef unfinished():\n    pass\n")
    assert result.truncated
    assert result.files[0].filename.endswith(".partial.py")
    assert result.files[0].content == "def unfinished():\n    pass\n"
    assert "未完" in result


def test_code_does_not_consume_the_prose_character_budget():
    body = 'print("hello")\n' * 500
    result = prepare_reply("説明文。\n```python\n" + body + "```\n補足。", prose_limit=20)
    assert result.files[0].content == body
    assert len(body) > 1800
    assert not result.truncated
    assert "補足。" in result
    long = prepare_reply("説明。" * 1000 + "\n```python\n" + body + "```", prose_limit=100)
    assert long.files[0].content == body and long.truncated


def test_all_blocks_share_the_code_token_budget_and_truncation_is_explicit():
    body = 'print("日本語")\n' * 1000
    result = prepare_reply("```python\n" + body + "```\n```js\nconsole.log(1);\n```", code_token_limit=128)
    assert sum(file.tokens for file in result.files) <= 128
    assert sum(len(file_encoding().encode(file.content, disallowed_special=())) for file in result.files) <= 128
    assert result.truncated and not result.files[0].complete
    assert "上限" in result
    assert result.files[0].content.startswith("print(")


def test_preview_uses_dynamic_character_counts_even_without_whitespace_in_code():
    first = prepare_reply("```python\nabcdefgh", preview=True)
    second = prepare_reply("```python\nabcdefghijklmnop", preview=True)
    assert str(first) == "Pythonコード[8文字]"
    assert str(second) == "Pythonコード[16文字]"
    assert "abcdefgh" not in str(first)
    assert first.source == "```python\nabcdefgh"
    assert not first.files


async def test_chat_preserves_long_code_and_reserves_enough_model_output_tokens(settings, store, monkeypatch):
    body = 'print("hello")\n' * 600
    complete = AsyncMock(return_value=ModelTurn(content="生成しました。\n```python\n" + body + "```"))
    monkeypatch.setattr(store, "complete_chat", complete)
    result = await ChatService(settings, store).generate_reply("Pythonコードを書いて")
    assert len(body) > settings.operational.max_output_chars
    assert result.files[0].content == body and not result.truncated
    assert complete.await_args.kwargs["max_tokens"] >= 8192 + 2048
    assert "print(" not in result.content


async def test_progress_sanitizes_prose_without_changing_role_words_inside_code(settings, store, monkeypatch):
    monkeypatch.setattr("ai_kun.chat.STREAM_PREVIEW_INTERVAL_SECONDS", 0)
    source = "system:\n    assistant: <|im_start|>\n根拠データ：\nhttps://discord.com/channels/23456/34567/60001\n"

    async def complete(messages, **kwargs):
        await kwargs["on_text"]("```yaml\n" + source)
        return ModelTurn(content="```yaml\n" + source + "```")

    monkeypatch.setattr(store, "complete_chat", complete)
    updates = AsyncMock()
    result = await ChatService(settings, store).generate_reply("このYAMLを出して", on_update=updates)
    assert result.files[0].content == source
    assert "system:" not in updates.await_args.args[0]
    assert "YAMLコード[" in updates.await_args.args[0]


async def test_discord_stream_updates_counts_then_attaches_code_to_same_reply(settings, store, monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr("ai_kun.chat.STREAM_PREVIEW_INTERVAL_SECONDS", 0)
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(store, "search", AsyncMock(return_value=()))
    uploaded = []
    edits = []

    class Message:
        async def edit(self, **kwargs):
            edits.append(kwargs["content"])
            for file in kwargs.get("attachments", []):
                uploaded.append((file.filename, file.fp.read().decode()))
            return self

    async def complete(messages, **kwargs):
        await kwargs["on_text"]("```python\nprint(1)\n")
        await asyncio.sleep(0.02)
        await kwargs["on_text"]("```python\nprint(1)\nprint(2)\n")
        await asyncio.sleep(0.02)
        return ModelTurn(content="```python\nprint(1)\nprint(2)\n```")

    monkeypatch.setattr(store, "complete_chat", complete)
    client = AIKunClient(settings, service)
    message = file_message("!ai コードを書いて", [])
    message.reply.return_value = Message()
    try:
        await client.on_message(message)
        assert message.reply.await_count == 1
        assert "Pythonコード[9文字]" in message.reply.await_args.args[0]
        assert any("Pythonコード[18文字]" in value for value in edits)
        assert all("print(" not in value for value in edits)
        assert uploaded == [("code_01.py", "print(1)\nprint(2)\n")]
        assert message.reply.await_args.kwargs["mention_author"] is False
    finally:
        await client.drain()
        await client.close()


async def test_failure_salvages_partial_code_as_a_file(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.005)
    uploaded = []
    contents = []

    class Message:
        async def edit(self, **kwargs):
            contents.append(kwargs["content"])
            for file in kwargs.get("attachments", []):
                uploaded.append((file.filename, file.fp.read().decode()))
            return self

    publisher = StreamReply(AsyncMock(return_value=Message()), split_message, request_id="partial")
    publisher.start()
    try:
        await publisher.update(prepare_reply("```python\nprint(1)\n", preview=True))
        await until(lambda: publisher.message is not None)
        await publisher.finish("LLM応答が中断", failed=True)
        assert uploaded == [("code_01.partial.py", "print(1)\n")]
        assert "生成が中断" in contents[-1] and "未完" in contents[-1]
    finally:
        await publisher.close()


async def test_more_than_ten_files_are_sent_in_batches_and_streams_are_closed():
    result = prepare_reply("\n".join(f"```python\nprint({i})\n```" for i in range(12)))
    posted = []
    file_objects = []

    async def send(text, **kwargs):
        files = kwargs.get("files", [])
        posted.append([file.filename for file in files])
        file_objects.extend(files)
        assert all(not file.fp.closed for file in files)
        return SimpleNamespace(edit=AsyncMock())

    publisher = StreamReply(send, split_message, request_id="batch")
    await publisher.finish(result)
    assert [len(batch) for batch in posted] == [10, 2]
    assert len(file_objects) == 12 and all(file.fp.closed for file in file_objects)


def test_default_budget_really_allows_eight_thousand_code_tokens():
    body = 'value = "日本語"\n' * 4000
    result = prepare_reply("```python\n" + body + "```", prose_limit=1800)
    file = result.files[0]
    assert 8000 <= file.tokens <= 8192
    assert len(file.content) > 1800
    assert body.startswith(file.content)
    assert result.truncated and file.filename.endswith(".partial.py")


async def test_active_counter_is_visible_even_when_prose_fills_first_message(monkeypatch):
    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.005)
    send = AsyncMock(return_value=SimpleNamespace(edit=AsyncMock()))
    publisher = StreamReply(send, split_message, request_id="visible")
    publisher.start()
    try:
        await publisher.update(
            prepare_reply("説明" * 4000 + "\n```python\n" + "x" * 10000, preview=True, prose_limit=10000)
        )
        await until(lambda: send.await_count > 0)
        content = send.await_args.args[0]
        assert "Pythonコード[10,000文字]" in content
        assert len(content.encode("utf-16-le")) // 2 <= 2000
        assert "xxxx" not in content
    finally:
        await publisher.close()


async def test_attachment_retry_recreates_file_stream_after_deleted_preview(monkeypatch):
    import discord

    monkeypatch.setattr("ai_kun.adapters.stream_reply.EDIT_INTERVAL_SECONDS", 0.005)
    received = []
    upload_objects = []

    class Deleted:
        async def edit(self, **kwargs):
            upload_objects.extend(kwargs["attachments"])
            raise discord.NotFound(SimpleNamespace(status=404, reason="deleted"), {"code": 10008})

    async def send(text, **kwargs):
        for upload in kwargs.get("files", []):
            received.append(upload.fp.read().decode())
            upload_objects.append(upload)
        return Deleted()

    publisher = StreamReply(send, split_message, request_id="retry-file")
    publisher.start()
    try:
        await publisher.update(prepare_reply("```python\nprint(1)\n", preview=True))
        await until(lambda: publisher.message is not None)
        await publisher.finish(prepare_reply("```python\nprint(1)\n```"))
        assert received == ["print(1)\n"]
        assert len(upload_objects) == 2
        assert all(upload.fp.closed for upload in upload_objects)
    finally:
        await publisher.close()
