import pytest

from ai_kun.text_files import FILE_TOKEN_LIMIT, file_encoding, limit_file_text


@pytest.mark.parametrize("text", ["日本語と絵文字😀🐈" * 3000, "alpha beta gamma " * 5000, "<|endoftext|>" * 1000])
def test_file_limit_counts_tokens_and_preserves_valid_unicode(text):
    limited = limit_file_text(text)
    assert limited.truncated
    assert 0 < limited.tokens <= FILE_TOKEN_LIMIT
    assert len(file_encoding().encode(limited.text, disallowed_special=())) == limited.tokens
    assert text.startswith(limited.text)
    assert "\ufffd" not in limited.text


def test_short_file_is_preserved():
    text = "連絡先 alice@example.com、納期2026-11-01"
    limited = limit_file_text(text)
    assert limited.text == text
    assert not limited.truncated
