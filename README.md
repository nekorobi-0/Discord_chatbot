# AIくん — Discord RAGチャットボット

許可したDiscordサーバー・チャンネルで、日本語の質問に答えるPython製Botです。保存を指定したテキストや資料を検索して回答に使います。通常の会話は自動保存しません。

## できること

- `/ask`、Botへの先頭メンション、`!ai 質問` への返信
- `/remember` による明示的な記憶登録、管理者による資料取り込み・削除
- 許可チャンネルの過去ログを時系列でまとめて取り込むCLI（中断・再開対応）
- SQLiteへの保存と埋め込みキャッシュ、NumPyの行列演算を使ったコサイン類似度検索（1件ずつの純Python演算ではなく一括で計算）
- TavilyによるWeb検索（任意）。APIキー未設定ならツールを無効化
- 添付音声のローカル文字起こし（任意、faster-whisperとモデルが必要）
- 連続ツール呼び出しの制限、メンション通知の抑止
- 任意の接続状態・メトリクス確認用HTTPエンドポイント

1プロセス・1サーバーでの小規模運用を想定しています。許可チャンネル間で記憶を共有します。チャンネルやユーザーごとの記憶分離、VCへの参加、PDF/Wordの取り込みには対応していません。

## 最短の起動手順

Python 3.11〜3.14と[uv](https://docs.astral.sh/uv/)を用意してください。開発・テストはPython 3.13で確認しています。

```bash
uv sync --locked
cp .env.example .env
chmod 600 .env
```

`.env` の次の項目を実値に置き換えます。

| 設定 | 内容 |
| --- | --- |
| `DISCORD_BOT_TOKEN` | Botトークン |
| `DISCORD_APPLICATION_ID` | DiscordアプリのApplication ID |
| `GUILD_ID` | 利用するサーバーID |
| `ALLOWED_CHANNEL_IDS` | 許可チャンネルID。複数ならカンマ区切り |
| `LLM_BASE_URL` | Chat Completions互換APIのベースURL。例: `http://127.0.0.1:11434/v1` |
| `LLM_API_KEY` / `LLM_MODEL` | 認証が必要な場合のキー・導入したモデルID |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | 埋め込みモデルIDと出力次元。例の値は1536次元 |

`EMBEDDING_BASE_URL` と `EMBEDDING_API_KEY` が空なら、LLMと同じ設定を使います。埋め込みサービスは `/embeddings` と `dimensions` パラメータに対応し、指定次元の数値配列を返す必要があります。モデル例は利用先で使えるものに変更してください。Chat Completions互換でないAPIはそのまま接続できません。

記憶を使わない場合は `MEMORY_ENABLED=false` にすると、Embedding設定は不要です。

```bash
# 外部に接続せず、設定・モデル設定・DBを検証
uv run python -m ai_kun --check

# 起動（終了は Ctrl+C）
uv run python -m ai_kun
```

`uv run ai-kun` でも起動できます。`.env` はカレントディレクトリから読み込み、既存の環境変数が優先されます。別の設定ファイルは `--env-file /path/to/bot.env` で指定できます。`DATA_DIR` などの相対パスもカレントディレクトリ基準です。

uvを使わない場合:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m ai_kun --check
python -m ai_kun
```

## ローカル推論サーバーの設定

Chat Completions互換のローカル推論サーバーを起動し、導入済みのモデルIDと接続先を `.env` に設定してください。Bot自身がモデルをダウンロード・起動する処理はありません。以下のポート番号とモデル名は環境に合わせて変更します。

```dotenv
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=
LLM_MODEL=YOUR_LOCAL_CHAT_MODEL
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL=YOUR_LOCAL_EMBEDDING_MODEL
EMBEDDING_DIMENSION=1536
```

認証不要のサーバーではAPIキーは空欄で動き、Authorizationヘッダーを送信しません。埋め込みの次元数は実際のモデルの出力に合わせてください。Embeddingを使わない場合は `MEMORY_ENABLED=false` に設定します。Web検索が不要なら `WEB_SEARCH_ENABLED=false` にします。Discordへの接続にはネットワーク接続とBotトークンが必要です。

## Discord側の設定

1. [Developer Portal](https://discord.com/developers/applications)でApplicationとBotを作成し、トークンを設定します。
2. **Bot → Privileged Gateway Intents → Message Content Intent** を有効にします。
3. Guild Install用の招待URLに `bot` と `applications.commands` のスコープを含め、対象サーバーへ招待します。
4. 対象チャンネルで「チャンネルを見る」「メッセージを送信」「メッセージ履歴を読む」を許可します。スレッドを使う場合は「スレッドでメッセージを送信」も必要です。
5. Discordの開発者モードでサーバー・チャンネルIDをコピーし、`.env` に設定します。

Botの起動時に設定したサーバーへスラッシュコマンドを同期します。DM、他サーバー、許可リスト外のチャンネルでは質問・記憶操作を受け付けません。親チャンネルを許可しても、そのスレッドは自動許可しません。必要なスレッドIDを個別に追加してください。

実装で使用するAPIは [discord.py Quickstart](https://discordpy.readthedocs.io/en/stable/quickstart.html) と [Interactions API](https://discordpy.readthedocs.io/en/stable/interactions/api.html) を参照しています。

## 使い方

| 操作 | 用途 | 権限 |
| --- | --- | --- |
| `/ask question:来週の予定は？` | 保存記憶を検索して質問に回答 | 許可チャンネルの利用者 |
| `@AIくん こんにちは` | 先頭メンションで質問 | 同上 |
| `!ai こんにちは` | テキストで質問 | 同上 |
| `/remember text:プロジェクトの締切は金曜日` | テキストを記憶へ保存 | 同上 |
| `/ingest file:...` | UTF-8のテキスト資料を登録 | サーバー管理権限 |
| `/forget document_id:...` | 文書とその検索チャンクを削除 | サーバー管理権限 |
| `/search_memory query:...` | 保存済みの記憶を検索し、管理者だけに結果を表示 | サーバー管理権限 |

登録時に文書IDを本人だけに表示します。削除する場合はそのIDを管理者へ伝えてください。`/remember` の保存内容は、登録結果が非公開でも許可チャンネルの利用者全員が参照できます。会話の文脈は自動的に引き継がないため、「それについて」のような省略を避けて質問してください。

通常の発言・Botの回答の常時自動保存は行いません。過去ログCLIは明示的な一括取り込みです。`覚えないで`、`学習しないで`、`memory off` などが含まれる質問では記憶の参照をスキップし、同じ表現を含む `/remember` は保存を拒否します。既存の記憶を消す操作は `/forget` です。

資料の対応形式は `.txt`, `.md`, `.markdown`, `.log`, `.csv`, `.json`, `.yaml`, `.yml`。上限は8 MiB、既定で12,000文字です。既定では900文字ずつ、150文字重ねて分割します。検索スコアが `RETRIEVAL_MIN_SCORE` 未満の記憶は回答に渡しません。類似度計算は NumPy のバッチ演算で行い、モデルや次元数が設定と違う記憶は対象外です。モデルや次元を変更した場合は資料を再登録してください。古いモデルの記憶は検索対象から除きます。

Botは同時に1件を処理し、処理中の追加呼びかけには待つよう返します。1処理の上限は240秒です。

## 過去ログの取り込み

設定した `GUILD_ID` の `ALLOWED_CHANNEL_IDS` 全件が対象です。許可リスト外や別サーバーへは取り込みを広げません。スレッドは自身のIDを許可リストへ追加します。フォーラム全体やスレッドの自動探索は行いません。

**Dockerでは専用の `history-import` サービスを同じイメージに含めています。** `.env` とローカルEmbeddingサーバーを準備したら次を実行します。

```bash
# 他のBot処理とAPI枠を競合させないよう、通常Botを一旦停止
docker compose stop bot
# イメージをビルドし、同じデータボリュームへ取り込む
docker compose --profile tools run --rm --build history-import
# 完了後に通常Botを起動
docker compose up -d bot
```

ローカルで直接動かす場合:

```bash
uv run python -m ai_kun --import-history
```

`--check` と `--import-history` は同時指定できません。取り込みにLLM生成は使わず、ローカルEmbeddingで索引を作ります。コンテナからEmbeddingサーバーに到達できるURLを設定してください。コンテナの `127.0.0.1` はホストを指しません（Docker節を参照）。

- 1回100件ずつ古い順に取得し、開始時点までのログを処理します。次回は保存済み位置以降の新規ログから続きます。
- 20発言・発言間隔30分・設定文字数を目安に会話をまとめ、日時・匿名化した話者識別子とともに保存します。長い単独発言は全文を分割して索引化します。取得ページ境界では会話が分かれます。
- Bot/Webhook、システム通知、空本文、記憶を拒否する表現、マスキング後も秘密鍵らしい本文は除外します。添付ファイル・音声・埋め込みはダウンロードしません。
- 本文のマスキングは通常の資料登録と同じです。検索で見つかった履歴には文書IDと元発言へのDiscordリンクが付きます。本文やユーザー名を進捗ログへ出しません。
- 100件の保存と進捗更新は一つのDBトランザクションです。Embedding失敗やCtrl+Cでは次回、最後に保存したページから再開します。同じボリュームへの二重実行は拒否します。
- メッセージIDで重複を防ぎます。`/forget` で削除した履歴は、再走査しても復活させません。既存メッセージの編集・Discord側の削除の追従、モデル変更後の履歴再索引は行いません。

取り込んだ内容は、既存の記憶と同様に許可チャンネル間で共有します。Botに `VIEW_CHANNEL` と `READ_MESSAGE_HISTORY`、アプリにMessage Content Intentが必要です。特に履歴閲覧権限がない場合、Discordはエラーではなく空配列を返すことがあります。件数が0なら権限とIntentを確認してください。

### 取得速度の根拠

[Discord公式のレート制限仕様](https://docs.discord.com/developers/topics/rate-limits)に従い、固定の取得回数を上限だと仮定せず、レスポンスの `X-RateLimit-Bucket`、`Limit`、`Remaining`、`Reset-After` を読みます。同じbucketでもチャンネルIDが違えば別枠として管理します。

読み取れた枠の約50%以下を目標に等間隔で取得します。例えば10秒に10回なら約2秒間隔、奇数枠は切り捨てて余裕を取ります。上限不明の初回応答後は2秒間隔に抑え、残量0ならリセットまで待機します。全体にも公式の50回/秒に対して25回/秒の上限を設けています。Embeddingの処理時間などにより、実際はこれより遅くなります。

429では `Retry-After` とJSONの `retry_after` の長い方以上を待ち、global制限なら他チャンネルも止めます。401/403/404は再試行せず停止し、通信失敗・5xx・429の再試行は最大5回です。上限の変更や他プロセスとの枠共有もあるため、429が絶対に発生しない保証ではありません。同じBotトークンの別プロセスを止めて取り込むのが確実です。

履歴APIの最大100件・ページング・権限の仕様は [Get Channel Messages](https://docs.discord.com/developers/resources/message#get-channel-messages) を参照しています。

## Web検索（任意）

`TAVILY_API_KEY` を設定し、`WEB_SEARCH_ENABLED=true` にすると、LLMが必要に応じて `web_search` を呼びます。LLM側もChat Completionsのfunction callingに対応している必要があります。Web検索は質問に応じたモデルの判断であり、毎回必ず行うものではありません。キーがなければ通常の会話だけで動きます。

[Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)を利用します。検索結果に含まれるページをBot自身が追加取得する機能はありません。

## 音声文字起こし（任意）

```bash
uv sync --locked --extra asr
```

ダウンロード済みのCTranslate2形式Whisperモデル一式（`model.bin`、設定、トークナイザーなど）を `data/asr-model/` に置き、次の値を設定して再起動します。

```dotenv
ASR_ENABLED=true
ASR_MODEL_PATH=./data/asr-model
ASR_DEVICE=cpu
ASR_COMPUTE_TYPE=int8
ASR_LANGUAGE=ja
```

`!ai` または先頭メンションとともに音声を1つ添付すると、文字起こしを質問に加えます。対応拡張子は `.ogg`, `.mp4`, `.webm`, `.mkv`, `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.opus` です。上限8 MiB。VCに接続する機能ではありません。標準DockerイメージにはASR依存関係とモデルを含めていません。

音声は一時ファイルとしてOSの一時領域へ書き、処理後に削除します。スレッドでの推論はタイムアウト後も完了まで続く場合があるため、短い音声を使ってください。

## データと運用

`DATA_DIR`（既定 `./data`）に以下を保存します。

- `memory.sqlite3` とSQLiteの補助ファイル: 登録した文書・検索チャンク・埋め込みキャッシュを保存。ID、日時、モデル、キャッシュのダイジェスト、履歴の取得進捗、メッセージIDと文書IDの対応も併せて記録されます。
- `audit.jsonl`: 有効時のみ、応答日時・文字数・マスキング件数。本文は記録せず、約5 MBで1世代ローテート

起動時にファイル作成権限を所有者限定にします。バックアップはBotを停止してデータディレクトリをコピーしてください。`/forget` は現行DBの論理削除で、過去のバックアップやDiscordに投稿した本文は消しません。

メール・電話・鍵らしい文字列は、LLM/Embeddingへ送信する前に正規表現でマスキングします。検出は完全ではなく、人名やすべての機密情報を除去するものではありません。会話や登録資料のマスキング後の内容は設定したAPIへ、検索語はTavilyへ送信されます。応答もマスキングし、Discordへのメンション通知を無効化しています。

LLMの利用量に応じた日次制限はありません。

`MAX_OUTPUT_CHARS` は200〜1900文字、既定1800文字。絵文字を含む返答はDiscord上限に合わせて複数投稿になる場合があります。

## 運用HTTP（任意）

```dotenv
HTTP_ENABLED=true
HTTP_HOST=127.0.0.1
HTTP_PORT=8080
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=十分に長い専用パスワード
```

- `GET /healthz`: 認証不要。Discord接続完了なら200、未接続なら503
- `GET /dashboard`: Basic認証付きの運用リンク画面
- `GET /metrics/v1`: Basic認証付きのJSONメトリクス

ユーザー名とパスワードの両方が空なら、画面とメトリクスは404です。どちらか片方だけの設定は起動エラーになります。メトリクスのカウンターはプロセス再起動でリセットされます。HTTPを有効にしてもLLM生成APIは公開しません。標準はlocalhostのみ。外部公開する場合はTLS終端とアクセス制限を別途用意してください。

## Docker

Docker Composeが使える環境で、先に `.env` を設定してください。

```bash
docker compose build
docker compose run --rm bot python -m ai_kun --check
docker compose up -d
docker compose logs -f bot
# 停止（データは名前付きボリュームへ保持）
docker compose down
```

コンテナは非rootで動き、`.env` はビルド対象から除外して起動時に渡します。データは `bot-data` 名前付きボリュームです。ホストの `./data` とは別です。ボリューム削除は記憶も削除します。

Compose構成はHTTPポートを公開しません。利用する場合はHTTPホストを `0.0.0.0` に変更し、ホストのlocalhostに限定するポートマッピング（例: `127.0.0.1:8080:8080`）を追加してください。

APIベースURLはHTTPSか、`localhost` / loopback IP / `host.docker.internal` へのHTTPを受け付けます。ホスト側のローカル推論サーバーへ接続する場合は `.env` を次のように設定します。

```dotenv
LLM_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
```

Composeには `host.docker.internal:host-gateway` のマッピングを設定しています。これは [Docker公式のhost-gateway指定](https://docs.docker.com/reference/cli/docker/container/run/#add-entries-to-container-hosts-file---add-host)に従っています。推論サーバーはDockerブリッジから到達できるインターフェースで待ち受ける必要があります。ホストの `127.0.0.1` にしか待ち受けていないサーバーには、この設定だけでは接続できません。ポート番号も実際の環境に合わせてください。

## systemd

`deploy/systemd/ai-kun.service` を用意しています。専用ユーザーと `/opt/ai-kun` への配置を前提にした例です。実際のユーザー・パスに変更し、仮想環境、`.env`、書き込み可能な `data` ディレクトリを用意してからインストールしてください。サービスの自動登録・起動は行いません。

## 開発・確認

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check ai_kun tests
uv run ruff format --check ai_kun tests
```

テストは一時DBと模擬APIを使用します。設定検証、登録・再起動・検索・削除、履歴のページング・重複防止・再開、Discordのレート制限追従・429待機、API送信前のマスキング、ツール呼び出し、Discordの許可判定、HTTP認証を確認します。Discord実サーバー・外部LLM・Tavily・ASRモデルの実機疎通は別途必要です。

## 困ったとき

| 症状 | 確認すること |
| --- | --- |
| `--check` が設定エラー | `.env` のプレースホルダー、APIベースURL、ID を確認 |
| Botがオンラインにならない | トークン、ネットワーク、Message Content Intentを確認 |
| コマンドが見えない | Application ID・GUILD_ID・招待スコープ、起動時の同期エラーを確認 |
| 返信しない | ギルドとチャンネルIDが一致するか、先頭メンション/`!ai`を使っているか確認 |
| 外部サービスが応答できない | 利用先モデル・API形式・キー・次元・タイムアウトを確認 |
| 記憶が見つからない | 登録の完了、モデル変更、検索スコア、記憶参照を拒否する表現の有無を確認 |

## 構成

```text
ai_kun/
  __main__.py             起動・設定検証
  adapters/discord_bot.py Discord接続・コマンド・アクセス制御
  chat.py                 応答生成・記憶参照・ツールループ
  memory.py               資料・履歴登録・Embedding・NumPy検索・LLM・ASR
  history.py              Discord履歴取得・速度調整・進捗再開
  db.py                   SQLiteスキーマ
  tools/__init__.py       Tavily検索
  observability.py        メトリクス・監査
  http_server.py          任意の運用HTTP
  config.py / errors.py   設定・例外
```
