"""添付テキストの入力予算。全モデル共通のcl100k_base基準で数える。"""

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

FILE_TOKEN_LIMIT = 4096


@lru_cache(maxsize=1)
def file_encoding():
    return tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class FileText:
    text: str
    tokens: int
    truncated: bool


def limit_file_text(text: str, limit: int = FILE_TOKEN_LIMIT) -> FileText:
    encoding = file_encoding()
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= limit:
        return FileText(text, len(tokens), False)
    clipped = encoding.decode(tokens[:limit], errors="ignore")
    clipped_tokens = encoding.encode(clipped, disallowed_special=())
    while len(clipped_tokens) > limit:
        clipped = clipped[:-1]
        clipped_tokens = encoding.encode(clipped, disallowed_special=())
    return FileText(clipped, len(clipped_tokens), True)
