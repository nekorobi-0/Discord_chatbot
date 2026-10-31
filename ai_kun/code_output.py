"""コード本文をDiscord本文から分離し、文字数表示と添付ファイルにする。"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .file_types import DOCUMENT_EXTENSIONS
from .text_files import limit_file_text

CODE_TOKEN_LIMIT = 8192
_OPEN = re.compile(r"(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)(?:\r?\n|$)")
_BOUNDARY = re.compile(r"[\s、。，．！？!?;；:：]")
_LANGUAGES = {
    "python": ("Python", ".py"),
    "javascript": ("JavaScript", ".js"),
    "typescript": ("TypeScript", ".ts"),
    "jsx": ("JSX", ".jsx"),
    "tsx": ("TSX", ".tsx"),
    "java": ("Java", ".java"),
    "kotlin": ("Kotlin", ".kt"),
    "c": ("C", ".c"),
    "cpp": ("C++", ".cpp"),
    "csharp": ("C#", ".cs"),
    "go": ("Go", ".go"),
    "rust": ("Rust", ".rs"),
    "ruby": ("Ruby", ".rb"),
    "php": ("PHP", ".php"),
    "swift": ("Swift", ".swift"),
    "bash": ("Shell", ".sh"),
    "powershell": ("PowerShell", ".ps1"),
    "sql": ("SQL", ".sql"),
    "json": ("JSON", ".json"),
    "yaml": ("YAML", ".yaml"),
    "toml": ("TOML", ".toml"),
    "html": ("HTML", ".html"),
    "css": ("CSS", ".css"),
    "xml": ("XML", ".xml"),
    "svg": ("SVG", ".svg"),
    "markdown": ("Markdown", ".md"),
    "dockerfile": ("Dockerfile", ".dockerfile"),
    "makefile": ("Makefile", ".mk"),
    "diff": ("Diff", ".patch"),
    "text": ("", ".txt"),
}
_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "ts": "typescript",
    "c++": "cpp",
    "c#": "csharp",
    "cs": "csharp",
    "rs": "rust",
    "rb": "ruby",
    "kt": "kotlin",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "ps1": "powershell",
    "yml": "yaml",
    "md": "markdown",
    "txt": "text",
    "plaintext": "text",
}


@dataclass(frozen=True)
class CodeFile:
    filename: str
    content: str
    language: str
    complete: bool
    tokens: int


@dataclass(frozen=True)
class Fence:
    info: str
    body: str
    raw: str
    complete: bool


class PreparedReply(str):
    def __new__(cls, text, *, files=(), source=None, progress=None, truncated=False):
        result = super().__new__(cls, text)
        result.files = tuple(files)
        result.source = source
        result.progress = progress
        result.truncated = truncated
        return result

    def with_text(self, text):
        return PreparedReply(
            text, files=self.files, source=self.source, progress=self.progress, truncated=self.truncated
        )


def parse_fences(text: str) -> list[str | Fence]:
    parts = []
    cursor = 0
    while opening := _OPEN.search(text, cursor):
        if opening.start() > cursor:
            parts.append(text[cursor : opening.start()])
        marker = opening["fence"]
        inline = re.search(re.escape(marker[0]) + "{" + str(len(marker)) + ",}", opening["info"])
        if inline:
            end = opening.start("info") + inline.end()
            parts.append(Fence("", opening["info"][: inline.start()], text[opening.start() : end], True))
            cursor = end
            continue
        close = re.compile(r"(?m)^ {0,3}" + re.escape(marker[0]) + r"{" + str(len(marker)) + r",}[ \t]*(?:\r?\n|$)")
        ending = close.search(text, opening.end())
        if ending is None:
            ending = re.compile(re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*(?:\r?\n|$)").search(
                text, opening.end()
            )
        end = ending.end() if ending else len(text)
        parts.append(
            Fence(
                opening["info"].strip(),
                text[opening.end() : ending.start() if ending else end],
                text[opening.start() : end],
                ending is not None,
            )
        )
        cursor = end
        if ending is None:
            break
    if cursor < len(text):
        parts.append(text[cursor:])
    return parts


def _file_info(fence: Fence, index: int) -> tuple[str, str]:
    first = fence.info.split()[0].strip("{}.").lower() if fence.info.split() else "text"
    language = _ALIASES.get(first, first)
    filename = None
    hint = re.search(r"""(?:filename|file|title)\s*=\s*["']?([^\s"']+)""", fence.info)
    candidates = [hint[1]] if hint else fence.info.split()
    for candidate in candidates:
        basename = PurePosixPath(candidate.replace("\\", "/")).name
        if PurePosixPath(basename).suffix.lower() in DOCUMENT_EXTENSIONS:
            filename = re.sub(r"[^\w.\-]", "_", basename).strip(".")[:120]
            if language not in _LANGUAGES:
                suffix = PurePosixPath(basename).suffix.lower()
                language = next((key for key, (_, ext) in _LANGUAGES.items() if ext == suffix), suffix[1:])
            break
    if language not in _LANGUAGES:
        if "." + language in DOCUMENT_EXTENSIONS:
            label, extension = language.capitalize(), "." + language
        else:
            label, extension = _LANGUAGES["text"]
    else:
        label, extension = _LANGUAGES[language]
    if filename is None:
        java = re.search(
            r"\bpublic\s+(?:(?:final|abstract|sealed)\s+)*(?:class|record|interface|enum)\s+(\w+)", fence.body
        )
        if language == "java" and java:
            filename = java[1] + ".java"
        elif language == "dockerfile" and index == 1:
            filename = "Dockerfile"
        else:
            filename = f"code_{index:02d}{extension}"
    return label, filename


def _limit_code(body: str, budget: int) -> tuple[str, int, bool]:
    """長い一行でもtokenizerを巨大な入力で止めない。約8k枠を小分けに計数する。"""
    pieces = []
    used = 0
    for offset in range(0, len(body), 2048):
        piece = body[offset : offset + 2048]
        remaining = budget - used
        if remaining <= 0:
            break
        limited = limit_file_text(piece, remaining)
        pieces.append(limited.text)
        used += limited.tokens
        if limited.truncated:
            break
    text = "".join(pieces)
    return text, used, text != body


def prepare_reply(
    text: str, *, prose_limit=None, code_token_limit=CODE_TOKEN_LIMIT, preview=False, prose_transform=None
) -> PreparedReply:
    parts = parse_fences(text)
    source = []
    processed = []
    for index, part in enumerate(parts):
        if isinstance(part, str):
            value = prose_transform(part) if prose_transform else part
            if preview and index == len(parts) - 1:
                end = max((match.end() for match in _BOUNDARY.finditer(value)), default=0)
                value = value[:end]
            processed.append(value)
            source.append(value)
        else:
            if source and not source[-1].endswith("\n"):
                source.append("\n")
            source.append(part.raw)
            processed.append(part)
    prose_size = sum(len(part) for part in processed if isinstance(part, str))
    prose_cut = prose_limit is not None and prose_size > prose_limit
    suffix = "\n…（長いため省略）" if prose_cut else ""
    remaining_prose = max(0, prose_limit - len(suffix)) if prose_cut else prose_limit
    code_budget = code_token_limit
    display, files = [], []
    truncated = prose_cut
    progress = None
    code_index = 0
    for part in processed:
        if isinstance(part, str):
            if remaining_prose is None:
                display.append(part)
            else:
                display.append(part[:remaining_prose])
                remaining_prose = max(0, remaining_prose - len(part))
                if remaining_prose == 0 and suffix:
                    display.append(suffix)
                    suffix = ""
            continue
        code_index += 1
        label, filename = _file_info(part, code_index)
        body, tokens, clipped = (part.body, 0, False) if preview else _limit_code(part.body, code_budget)
        code_budget -= tokens
        complete = part.complete and not clipped
        if not preview and not complete:
            stem = PurePosixPath(filename)
            filename = f"{stem.stem}.partial{stem.suffix}"
        marker = f"{label}コード[{len(body):,}文字]"
        progress = marker
        if preview:
            display.append("\n" + marker + "\n")
        else:
            detail = "（上限で省略・未完）" if clipped else "（未完）" if not complete else ""
            display.append(f"\n{marker}{detail}" + (f" → `{filename}`" if body else "") + "\n")
            truncated = truncated or not complete
            if body:
                files.append(CodeFile(filename, body, label, complete, tokens))
    content = "".join(display)
    if files or any(isinstance(part, Fence) for part in processed):
        content = content.strip()
    return PreparedReply(
        content, files=files, source="".join(source) if preview else None, progress=progress, truncated=truncated
    )
