# RAG取り込みの原因調査と測定

## 再現した問題

既定の900文字の日本語チャンクが594トークンになり、llama.cppの既定ubatch 512を超えてHTTP 500になった。
サーバーログは `input (594 tokens) is too large to process. increase the physical batch size (current batch size: 512)`。
短い発言だけの疎通試験では再現せず、ページやテキスト資料で止まる構成だった。

また、各チャンクでEmbeddingへのHTTP接続・キャッシュSELECT・キャッシュCOMMITを繰り返していた。
新着発言も1件ずつ文書にしていたため、連投100件では100回のEmbeddingと300回のDBコミットになっていた。
返信前には過去ログ取得の完了も待っており、Discord APIのレート待ちが返信経路を止めていた。

## 変更

- サーバーのbatch/ubatchを8192、並列スロットを4に指定。CPU専用、重みは同じBF16。
- 新着の連続した同一チャンネルの発言を25msの窓でまとめる。すべての受付Futureへ結果を返す。
- 1ページ内の全チャンクをまとめ、最大16入力ずつEmbedding APIへ送る。応答indexの整合性を検証する。
- キャッシュをまとめて検索・upsertし、元の入力順へ戻す。重複入力も1回だけ計算する。
- 文書・チャンク・カーソルはflush後に1回でコミットし、全体を原子的に保存する。
- 同じチャンネルの過去ログ要求は取得を共有する。全要求を完了させ、結果に共有要求数を残す。
- 過去ログ取得は返信から切り離す。Discord APIのレート制限は維持する。
- ログにEmbedding入力数と所要秒数、Web UIに実行時間・待機時間を表示する。

## CPU実測（2026-10-31）

AMD Ryzen 7 9700X、8スレッド、同じEmbeddingGemma 2 BF16 / 768次元。
日本語の新着100件を一斉に投入し、空の一時DBを毎回使った。モデルのウォームアップは計測外。
変更前の取り込み処理は `df10820` のもの。比較の両側でbatch/ubatch 8192・4スロットを使い、
HTTP 500を解消した条件で比較した。Discord APIやLLM応答時間はこの測定には含まない。

| 指標 | 変更前 | 変更後 |
| --- | ---: | ---: |
| 受付・保存した発言 | 100 | 100 |
| Embedding HTTP回数 | 100 | 2 |
| DBコミット回数 | 300 | 2 |
| 生成チャンク | 100 | 20 |
| 経過時間（3回の中央値） | 3.884秒 | 3.405秒 |

CPU推論が支配的なこの環境では、経過時間は約12%短縮。
HTTP回数やDBコミットの削減率を、そのまま経過時間の改善率として扱わない。
本番環境のCPU・ディスク・API待ち時間によって改善幅は変わる。

再測定は、別途起動したEmbeddingサーバーに対して以下を実行する。
実DBは使わず、DiscordのトークンやAPIも使わない。

```bash
uv run python -m scripts.benchmark_ingestion --embedding-url http://127.0.0.1:18434/v1 --mode live
uv run python -m scripts.benchmark_ingestion --embedding-url http://127.0.0.1:18434/v1 --mode page
```
