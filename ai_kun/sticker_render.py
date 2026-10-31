"""スタンプをCPUでPNGへ変換する子プロセス。入力・描画サイズと実行時間を制限する。"""

import io
import json
import resource
import sys

MAX_BYTES = 1024 * 1024


def render(data: bytes, format_type: int) -> bytes:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 4 * 1024 * 1024
    if format_type == 3:
        from rlottie_python import LottieAnimation

        document = json.loads(data)
        if not isinstance(document, dict) or not document.get("layers"):
            raise ValueError("Invalid Lottie document")
        # Discordのベクタースタンプのみ。画像・外部ファイル参照は読まない。
        for asset in document.get("assets", []):
            if not isinstance(asset, dict) or asset.get("p") or asset.get("u"):
                raise ValueError("External assets are unsupported")
        width, height = int(document.get("w", 0)), int(document.get("h", 0))
        if not (0 < width <= 2048 and 0 < height <= 2048):
            raise ValueError("Invalid dimensions")
        scale = min(1, 320 / max(width, height))
        with LottieAnimation(data=json.dumps(document), resource_path="") as animation:
            image = animation.render_pillow_frame(width=max(1, int(width * scale)), height=max(1, int(height * scale)))
    elif format_type in {1, 2, 4}:
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in {"PNG", "GIF", "WEBP"}:
                raise ValueError("Unsupported sticker image")
            if original.width * original.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("Oversized image")
            original.seek(0)
            image = original.convert("RGBA")
            image.thumbnail((320, 320))
    else:
        raise ValueError("Unsupported sticker format")
    # 透過スタンプの輪郭もencoderが認識できるよう白地にする。
    with image:
        background = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image)
        with background, io.BytesIO() as output:
            background.save(output, format="PNG")
            return output.getvalue()


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (4, 5))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    data = sys.stdin.buffer.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Oversized sticker")
    sys.stdout.buffer.write(render(data, int(sys.argv[1])))


if __name__ == "__main__":
    main()
