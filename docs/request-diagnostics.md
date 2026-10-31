# 入力中が長く続く／外部サービスエラーの調査

以前は、FIFOキューの待機中もDiscordの入力中表示を更新していた。実行時間の240秒上限にはキュー待ちが含まれないため、要求がたまると長時間入力中になる。入力中表示を実行中だけに移し、表示用APIの開始・終了待機も各3秒で打ち切る。表示APIが失敗しても回答処理は続行する。

LLM・EmbeddingへのHTTP要求は、従来のhttpxタイムアウトに加えて総時間の上限を適用する。`LLM_TIMEOUT_SECONDS`（既定90秒）を使う。httpxのread timeoutだけでは、少量ずつ応答が届き続ける場合に総時間が延びるため。応答処理全体は従来どおり実行開始から240秒、Discordへの返信・slashコマンドのfollowup送信は30秒を上限とする。キュー待ちによって受付済み要求を破棄しない。

以前の `外部サービスがいま応答できないみたい` は、LLMとEmbeddingのHTTP通信失敗をまとめた表現だった。通信タイムアウトと接続エラーについて、現在は `LLM（応答タイムアウト）`、`Embedding（接続エラー）` などと区別する。処理側で捕捉した既知のエラー・全体タイムアウトもWARNINGを記録する。

## ログ

DiscordメッセージIDを `request` として、以下の処理を追跡する。slashコマンドではinteraction IDを使う。本文・画像・認証ヘッダー・外部例外の本文は記録しない。

| ログ | 内容 |
| --- | --- |
| `request queued` | 受付 |
| `request started` / `queue_seconds` | 実行開始とキュー待ち時間 |
| `request pending` / `phase` | 15秒ごとの処理中の位置 |
| `upstream started` | LLMまたはEmbeddingへの要求開始・タイムアウト設定 |
| `upstream completed` | HTTPステータスと所要時間 |
| `upstream timeout` / `upstream transport error` | 対象サービス、例外の型、失敗までの時間 |
| `request failed` / `request timed out` | 利用者へエラーを返す理由と処理位置 |
| `request finished` | 応答の生成終了。Discordへの送信完了とは別 |
| `Discord reply completed` / `Discord followup completed` | Discordへの送信完了 |
| `Discord reply failed` / `Discord followup failed` | 送信タイムアウトまたはDiscordのHTTPエラー |

```bash
docker compose logs --since 10m bot | rg 'request |upstream |Discord reply|Discord followup|typing '
```

`phase` は `reply_chain`（返信参照）、`attachments`（添付）、`reply_lock`（推論順番待ち）、`rag_search`、`embedding_http`、`llm_http`、`tool:<ツール名>` など。HTTP終了後の `*_response` は応答の処理中を表す。

2026-10-31の確認では、192.168.0.123:8080のhealthは正常、短いChat要求はHTTP 200・約0.66秒で返った。利用者が失敗した要求そのものは再現できておらず、長い文脈・画像付き要求のタイムアウトや、Bot実行環境からの接続失敗までは否定できない。再発時は上記のログでサービスと処理位置を確認する。
