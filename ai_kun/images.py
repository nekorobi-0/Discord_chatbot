"""画像添付をOpenAI互換のマルチモーダル入力へ変換する。"""

import base64

from .errors import InvalidInputError

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})


def image_data_url(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        media_type = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise InvalidInputError("画像を読み取れませんでした。PNG・JPEG・WebP・GIF形式で送ってください。")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"
