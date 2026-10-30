"""アプリ全体で使う安全な例外。メッセージは利用者にそのまま見せてよい文面のみ。"""


class AIKunError(Exception):
    """AIくん内部で発生した、対処可能なエラーの基底クラス。"""

    def user_message(self) -> str:
        return str(self) or "ごめん、いまちょっと手が止まっちゃった。もう一回試してくれる？"


class ConfigurationError(AIKunError):
    """設定や環境が足りなくて動けない状態。"""

    def __init__(self, message: str = "AIくんの設定が足りなくて動けないよ。管理者にたずねてくれる？") -> None:
        super().__init__(message)
        self._message = message

    def user_message(self) -> str:
        return self._message


class UpstreamError(AIKunError):
    """LLM / Embedding / 検索などの外部サービスが失敗した状態。"""

    def __init__(self, service: str) -> None:
        super().__init__(f"{service}がいま応答できないみたい。ごめん、少し待ってからもう一回どうぞ。")
        self.service = service


class RateLimitError(AIKunError):
    """推論サーバーがレート制限を返した状態。"""

    def __init__(self) -> None:
        super().__init__("推論サーバーが混み合っているみたい。少し待ってからもう一回お願い。")


class InvalidInputError(AIKunError):
    """リクエストが規約を満たさず、そもそも応答できない状態。"""

    def user_message(self) -> str:
        return str(self) or "うまく読み取れなかったよ。"
