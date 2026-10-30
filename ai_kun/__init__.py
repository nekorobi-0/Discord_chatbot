"""AIくん — Discord 用 RAG チャットボット。

設計方針:
- stateless な generate_reply() が唯一の応答経路（replay しても履歴は残らない）
- 過去会話・ドキュメントは埋め込みインデックス経由でのみ参照する RAG
- 個人情報らしい文字列は API 送信前にマスキング（検出には限界あり）
"""

from .chat import generate_reply

__all__ = ["generate_reply"]
