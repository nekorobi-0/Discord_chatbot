"""添付メタデータで形式を推定し、テキストは実データでも検証する。"""

import codecs
import re
from pathlib import PurePosixPath

from .errors import InvalidInputError
from .images import IMAGE_EXTENSIONS

DOCUMENT_EXTENSIONS = frozenset(
    """
    .adoc .asm .astro .bash .bat .bib .c .cc .cfg .cjs .clj .cljc .cljs .cmake .cmd .conf .config .cpp .cs
    .css .csv .cts .cxx .dart .diff .dockerfile .edn .elm .erl .ex .exs .f .f90 .fish .fs .fsx .go .gql
    .gradle .graphql .h .hh .hpp .hrl .hs .htm .html .http .hxx .ini .ipynb .java .jl .js .json .jsonl .jsx
    .kt .kts .less .lhs .lock .log .lua .m .make .markdown .md .mjs .mk .mm .mts .ndjson .nix .pas .patch .php
    .pl .pm .properties .proto .ps1 .py .pyi .pyw .r .rb .rs .rst .s .sass .sc .scala .scss .sh .sql .svelte
    .svg .swift .tex .tf .tfvars .toml .ts .tsv .tsx .txt .v .vb .vue .xml .xsd .xsl .yaml .yml .zig .zsh
    """.split()
)
VOICE_EXTENSIONS = frozenset({".ogg", ".mp4", ".webm", ".mkv", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus"})
_TEXT_NAMES = {
    "dockerfile",
    "containerfile",
    "makefile",
    "gnumakefile",
    "cmakelists.txt",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".editorconfig",
    ".env",
    "license",
    "readme",
}
_BLOCKED_EXTENSIONS = {
    ".zip",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".tar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".a",
    ".lib",
    ".wasm",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pkl",
    ".pickle",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}
_TEXT_MIMES = {
    "application/json",
    "application/ld+json",
    "application/x-ndjson",
    "application/jsonl",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/typescript",
    "application/x-javascript",
    "application/x-typescript",
    "application/x-python",
    "application/x-python-code",
    "application/x-sh",
    "application/x-shellscript",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/sql",
    "application/graphql",
    "application/x-ruby",
    "application/x-perl",
    "application/x-httpd-php",
}
_BLOCKED_MIMES = {
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/vnd.rar",
    "application/x-executable",
    "application/x-dosexec",
    "application/x-sharedlib",
}
_BINARY_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"RIFF",
    b"OggS",
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ",
    b"7z\xbc\xaf",
    b"Rar!\x1a\x07",
    b"%PDF-",
    b"\x7fELF",
    b"\x00asm",
    b"SQLite format 3\x00",
)
_BINARY_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")


def is_text_name(filename: str | None) -> bool:
    name = PurePosixPath(filename or "").name.lower()
    return (
        PurePosixPath(name).suffix in DOCUMENT_EXTENSIONS
        or name in _TEXT_NAMES
        or name.startswith((".env.", "dockerfile.", "containerfile."))
    )


def attachment_kind(filename: str | None, content_type: str | None = None) -> str:
    """unknownは呼びかけ時だけダウンロードして本文判定。画像単体を起動条件にしない。"""
    suffix = PurePosixPath(filename or "").suffix.lower()
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VOICE_EXTENSIONS:
        return "audio"
    if suffix in _BLOCKED_EXTENSIONS:
        return "unsupported"
    # .tsのvideo/mp2t誤推定にも対応する。ソース形式は本文で最終確認する。
    if is_text_name(filename) or mime.startswith("text/") or mime in _TEXT_MIMES or mime.endswith(("+json", "+xml")):
        return "document"
    if mime in _BLOCKED_MIMES:
        return "unsupported"
    if mime in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        return "image"
    if mime.startswith(("image/", "audio/", "video/")):
        return "unsupported"
    return "unknown"


def decode_text_file(filename: str | None, data: bytes, content_type: str | None = None) -> str:
    if attachment_kind(filename, content_type) in {"image", "audio", "unsupported"} or data.startswith(_BINARY_MAGIC):
        raise InvalidInputError("この形式の資料はまだ扱えないよ。テキストやソースコードを送ってください。")
    encoding = None
    for bom, name in (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if data.startswith(bom):
            encoding = name
            break
    if not encoding:
        charset = re.search(r'charset\s*=\s*["\x27]?([\w.-]+)', content_type or "", re.I)
        cookie = re.search(rb"(?m)^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", b"\n".join(data[:512].splitlines()[:2]))
        encoding = charset[1] if charset else cookie[1].decode("ascii") if cookie else "utf-8"
    try:
        text = data.decode(encoding, "strict")
    except UnicodeDecodeError as exc:
        # 文字コードの指定がない日本語の既知テキストだけCP932へフォールバックする。
        if encoding != "utf-8" or attachment_kind(filename, content_type) != "document":
            raise InvalidInputError("テキストの文字コードを読み取れません。UTF-8で保存して送ってください。") from exc
        try:
            text = data.decode("cp932", "strict")
        except UnicodeDecodeError as error:
            raise InvalidInputError("テキストの文字コードを読み取れません。UTF-8で保存して送ってください。") from error
        if not re.search(r"[\u3000-\u30ff\u3400-\u9fff\uff01-\uffef]", text):
            raise InvalidInputError("テキストの文字コードを判定できません。UTF-8で保存して送ってください。")
    except (LookupError, ValueError) as exc:
        raise InvalidInputError("指定された文字コードには対応していません。UTF-8で保存して送ってください。") from exc
    if "\x00" in text:
        raise InvalidInputError("バイナリデータをテキストとして読み込めません。ソースコードを送ってください。")
    for count, _ in enumerate(_BINARY_CONTROLS.finditer(text), start=1):
        if count > len(text) * 0.01:
            raise InvalidInputError("バイナリデータをテキストとして読み込めません。ソースコードを送ってください。")
    return text


def is_document_trigger(filename: str | None, content_type: str | None = None) -> bool:
    """テキストの自動受付。SVGも含め、画像投稿は呼びかけを必要とする。"""
    mime = (content_type or "").strip().lower()
    return (
        attachment_kind(filename, content_type) == "document"
        and not mime.startswith("image/")
        and PurePosixPath(filename or "").suffix.lower() != ".svg"
    )
