"""AIくん — Discord 用 RAG チャットボット。

設計方針:
- stateless な generate_reply() が唯一の応答経路（replay しても履歴は残らない）
- 過去会話は RAG で検索し、今回の添付資料・画像は応答時に直接参照する
- 会話・資料・応答の個人情報は自動マスクしない
"""

from .chat import generate_reply

__all__ = ["generate_reply"]
