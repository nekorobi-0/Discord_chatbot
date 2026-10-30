"""JSON文字列との互換性を保ちつつ、ツールの画像入力も運ぶ。"""


class ToolResult(str):
    def __new__(cls, text: str, image_parts=()):
        result = super().__new__(cls, text)
        result.image_parts = tuple(image_parts)
        return result
