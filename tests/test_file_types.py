import codecs
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_kun.adapters.discord_bot import AIKunClient
from ai_kun.chat import ChatService
from ai_kun.errors import InvalidInputError
from ai_kun.file_types import attachment_kind, decode_text_file, is_document_trigger
from ai_kun.text_files import FILE_TOKEN_LIMIT, file_encoding
from tests.test_adapters import file_message


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("main.py", "application/octet-stream"),
        ("script.sh", "application/x-executable"),
        ("index.ts", "video/mp2t"),
        ("component.tsx", None),
        ("main.cpp", None),
        ("lib.rs", None),
        ("Main.java", None),
        ("app.go", None),
        ("main.cs", None),
        ("Dockerfile", None),
        ("Makefile", None),
        ("CMakeLists.txt", None),
        (".gitignore", None),
        ("config.toml", None),
        ("snippet.unusual", "text/plain; charset=utf-8"),
        ("payload", "application/json"),
        ("schema", "application/vnd.example+json"),
    ],
)
def test_text_metadata_and_source_names(filename, content_type):
    assert attachment_kind(filename, content_type) == "document"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32"])
def test_unicode_bom_and_source_formatting(encoding):
    source = '    system: str = "assistant: <|im_start|>"\n    print("日本語")\n'
    assert decode_text_file("main.py", source.encode(encoding)) == source


@pytest.mark.parametrize("charset", ['charset="utf-16-le"', "charset='utf-16-le'"])
def test_metadata_charset_without_bom(charset):
    source = 'print("日本語")\n'
    assert decode_text_file("main.py", source.encode("utf-16-le"), "text/plain; " + charset) == source


@pytest.mark.parametrize(
    "filename,metadata",
    [("image.svg", "image/svg+xml"), ("image.svg", None), ("photo.png", "text/plain"), ("image", "image/png")],
)
def test_metadata_and_svg_do_not_reenable_automatic_image_replies(filename, metadata):
    assert not is_document_trigger(filename, metadata)


@pytest.mark.parametrize(
    "metadata,source",
    [
        ("text/plain; charset=shift_jis", 'print("日本語")\n'),
        (None, '# coding: cp932\nprint("日本語")\n'),
        (None, 'print("日本語")\n'),
    ],
)
def test_japanese_legacy_encoding(metadata, source):
    assert decode_text_file("main.py", source.encode("cp932"), metadata) == source


@pytest.mark.parametrize(
    "filename,data,metadata",
    [
        ("main.py", b"\x89PNG\r\n\x1a\nimage bytes", "text/plain"),
        ("source.txt", b"\x00\x01\x02BINARY", "text/plain"),
        ("file.unknown", b"PK\x03\x04archive", "text/plain"),
        ("file.unknown", b"%PDF-1.7", "application/octet-stream"),
        ("fake.txt", b"\xff" * 32, "text/plain"),
        ("snippet", b"\xff\xfe\xfd", "application/octet-stream"),
        ("program.exe", b"plain bytes", "text/plain"),
    ],
)
def test_binary_data_cannot_be_forced_to_text_by_metadata(filename, data, metadata):
    with pytest.raises(InvalidInputError):
        decode_text_file(filename, data, metadata)


@pytest.mark.parametrize(
    "filename,metadata,hook",
    [
        ("main.py", "application/octet-stream", ""),
        ("index.ts", "video/mp2t", ""),
        ("Dockerfile", None, ""),
        ("snippet.custom", "text/x-python; charset=utf-8", ""),
        ("snippet", "application/octet-stream", "!ai コードを読んで"),
    ],
)
async def test_source_attachment_reaches_context_and_never_asr(settings, store, monkeypatch, filename, metadata, hook):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(service, "ingest_document", AsyncMock(return_value={"document_id": "test-doc"}))
    transcribe = AsyncMock()
    monkeypatch.setattr(service, "transcribe_attachment", transcribe)
    generate = AsyncMock(return_value=SimpleNamespace(content="コードの説明"))
    monkeypatch.setattr(service, "generate_reply", generate)
    source = '    system: str = "assistant: <|im_start|>"\n    print("日本語")\n'
    data = source.encode("utf-8")
    attachment = SimpleNamespace(
        filename=filename, content_type=metadata, size=len(data), read=AsyncMock(return_value=data)
    )
    client = AIKunClient(settings, service)
    message = file_message(hook, [attachment])
    try:
        await client.on_message(message)
        assert generate.await_args.kwargs["documents"][0].text == source
        assert message.reply.await_args.args[0] == "コードの説明"
        transcribe.assert_not_awaited()
    finally:
        await client.drain()
        await client.close()


async def test_multiple_source_files_share_the_existing_token_budget(settings, store, monkeypatch):
    service = ChatService(settings, store)
    monkeypatch.setattr(service, "ingest_message", AsyncMock())
    monkeypatch.setattr(service, "backfill_channel", AsyncMock(return_value={"imported": 0}))
    monkeypatch.setattr(service, "ingest_document", AsyncMock(return_value={"document_id": "test-doc"}))
    generate = AsyncMock(return_value=SimpleNamespace(content="要点"))
    monkeypatch.setattr(service, "generate_reply", generate)
    attachments = []
    for filename in ("main.py", "app.tsx"):
        data = ('print("日本語と絵文字😀")\n' * 3000).encode()
        attachments.append(
            SimpleNamespace(
                filename=filename, size=len(data), content_type="text/plain", read=AsyncMock(return_value=data)
            )
        )
    client = AIKunClient(settings, service)
    try:
        await client.on_message(file_message("!ai 説明して", attachments))
        texts = [doc.text for doc in generate.await_args.kwargs["documents"]]
        assert len(texts) == 2
        assert sum(len(file_encoding().encode(text, disallowed_special=())) for text in texts) <= FILE_TOKEN_LIMIT
    finally:
        await client.drain()
        await client.close()


async def test_code_is_data_and_is_not_executed(store, monkeypatch, tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"    import pathlib\n    pathlib.Path({str(marker)!r}).touch()\n"
    embed = AsyncMock(side_effect=lambda texts: [[1.0] * 64 for _ in texts])
    monkeypatch.setattr(store, "embed", embed)
    await store.ingest_document("main.py", codecs.BOM_UTF8 + source.encode())
    assert embed.await_args.args[0] == [source]
    assert not marker.exists()
