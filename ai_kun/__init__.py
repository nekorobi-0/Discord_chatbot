"""AIくん — Discord 用 RAG チャットボット。

設計方針:
- stateless な generate_reply() が唯一の応答経路（replay しても履歴は残らない）
- 過去会話・ドキュメントは埋め込みインデックス経由でのみ参照する RAG
- 会話・資料・応答の個人情報は自動マスクしない
"""

from .chat import generate_reply

__all__ = ["generate_reply"]
