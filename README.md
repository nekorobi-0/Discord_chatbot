# AIくん — Discord RAGチャットボット

許可したDiscordサーバー（GUILD_ID）内の全チャンネルで、日本語の質問に答えるPython製Botです。**人間の発言は、Botが返信する・しない（メンション・`!ai`・`/ask` などのフック有無）を問わず、届いた瞬間に全件RAG記憶として保存・索引されます**。さらに**稼働中のBot自体が自動バックフィルを実行します**：取り込み済みメッセージのIDと進捗はDBに記録済みで、観測したチャンネルのページング結果と比較して未保存分を特定し、古い順に巻き戻して即座に取り込みます（手動CLIを待たずに過去ログ全体が自動的にRAGへ集約される）。オプトアウトや秘密検出による除外もありません。チャンネル単位の許可リスト（旧 ALLOWED_CHANNEL_IDS）と「覚えないで」系の記憶拒否は廃止しました。

## できること

- `/ask`、Botへの先頭メンション、`!ai 質問` への返信
- `/remember` による明示的な記憶登録、管理者による資料取り込み・削除
- 設定サーバー内の全テキストチャンネルの過去ログを、稼働中のBotが自動で巻き戻し取り込み（手動CLI・専用サービスは廃止済み）
- SQLiteへの保存と埋め込みキャッシュ、NumPyの行列演算を使ったコサイン類似度検索（1件ずつの純Python演算ではなく一括で計算）
- TavilyによるWeb検索（任意）。APIキー未設定ならツールを無効化
- 添付音声のローカル文字起こし（任意、faster-whisperとモデルが必要）
- 連続ツール呼び出しの制限、メンション通知の抑止
- 任意の接続状態・メトリクス確認用HTTPエンドポイント

1プロセス・1サーバーでの小規模運用を想定しています。設定サーバー内の全チャンネルで記憶を共有します。チャンネルやユーザーごとの記憶分離、VCへの参加、PDF/Wordの取り込みには対応していません。

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
| `LLM_BASE_URL` | Chat Completions互換APIのベースURL。例: `http://127.0.0.1:11434/v1` |
| `LLM_API_KEY` / `LLM_MODEL` | 認証が必要な場合のキー・導入したモデルID |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | 埋め込みモデルIDと出力次元。既定は `embeddinggemma-2`（768次元） |

`EMBEDDING_BASE_URL` と `EMBEDDING_API_KEY` が空なら、LLMと同じ設定を使います。埋め込みサービスは `/embeddings` と `dimensions` パラメータに対応し、指定次元の数値配列を返す必要があります。Embeddingの既定モデルは `embeddinggemma-2` です。Chat Completions互換でないAPIはそのまま接続できません。

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

Chat Completions互換のローカル推論サーバーを起動し、導入済みのモデルIDと接続先を `.env` に設定してください。Bot自身がモデルをダウンロード・起動する処理はありません。**埋め込みは compose の `embedding` サービス（ループバック `127.0.0.1:11434`）へだけ向けます。** このサービスは GPU デバイスノードを受け渡さないため CPU でだけ推論し、GPU は使いません。既定モデルは `embeddinggemma-2`（768次元）のままです。

```dotenv
LLM_BASE_URL=http://127.0.0.1:11435/v1
LLM_API_KEY=
LLM_MODEL=YOUR_LOCAL_CHAT_MODEL
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_API_KEY=
EMBEDDING_MODEL=embeddinggemma-2
EMBEDDING_DIMENSION=768
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
| `/ask question:来週の予定は？` | 保存記憶を検索して質問に回答 | 設定サーバー内の利用者 |
| `@AIくん こんにちは` | 先頭メンションで質問 | 同上 |
| `!ai こんにちは` | テキストで質問 | 同上 |
| `/remember text:プロジェクトの締切は金曜日` | テキストを記憶へ保存（内容に関係なく必ず保存） | 同上 |
| `/ingest file:...` | UTF-8のテキスト資料を登録 | サーバー管理権限 |
| `/forget document_id:...` | 文書とその検索チャンクを削除 | サーバー管理権限 |
| `/search_memory query:...` | 保存済みの記憶を検索し、管理者だけに結果を表示 | サーバー管理権限 |

登録時に文書IDを本人だけに表示します。削除する場合はそのIDを管理者へ伝えてください。`/remember` の保存内容は、登録結果が非公開でも設定サーバー内の利用者全員が参照できます。「記憶しないで」と言っても保存されます。会話の文脈は自動的に引き継がないため、「それについて」のような省略を避けて質問してください。

人間の発言はすべてRAG記憶になります。**メンション・`!ai`・`/ask` などのフックとは無関係な通常会話も、届いた瞬間に自動で収集・索引されます**（返信はフックに対してのみ行います）。加えてBotは常時、保存済みカーソルとDiscord履歴のページングを照合し、**取り込み漏れ・未導入分の過去ログを自動で巻き戻し取り込みします**。手動で実行する取り込みコマンドや専用サービスは廃止済みで、この自動バックフィルが唯一の手段です。「覚えないで」「memory off」などのオプトアウト表現は廃止済みで、含まれていても通常どおり保存・索引され、質問時も常に記憶検索を行います。Bot自身の回答とシステム通知、空本文だけが構造的に除外されます。既存の記憶を消す操作は `/forget` です。

資料の対応形式は `.txt`, `.md`, `.markdown`, `.log`, `.csv`, `.json`, `.yaml`, `.yml`。上限は8 MiB、既定で12,000文字です。既定では900文字ずつ、150文字重ねて分割します。検索スコアが `RETRIEVAL_MIN_SCORE` 未満の記憶は回答に渡しません。類似度計算は NumPy のバッチ演算で行い、モデルや次元数が設定と違う記憶は対象外です。モデルや次元を変更した場合は資料を再登録してください。古いモデルの記憶は検索対象から除きます。

Botは同時に1件を処理し、処理中の追加呼びかけには待つよう返します。1処理の上限は240秒です。

## 過去ログの取り込み

設定した `GUILD_ID` のサーバーに見つかった全テキストチャンネルが対象です（チャンネルIDの指定は不要）。別サーバーへは取り込みを広げません。スレッドは親チャンネルの時系列に含まれる分だけを処理します。フォーラム全体やスレッドの自動探索は行いません。

**過去ログの手動取り込み用CLIとDockerサービスは廃止しました。稼働中のBotが自動的に全件を取り込みます。** 手動で走らせるコマンドも、専用のコンテナサービスも不要です。Botを起動すれば、保存済みカーソルとDiscord履歴のページング結果を照合して欠落を割り出し、取り込み済み・未取り込みを区別したうえで**古い順に即座に自動補完**します。`bot` サービスを起動するだけです。

```bash
docker compose up -d bot
```

- 1回100件ずつ古い順に取得し、未取得分をすべて処理します。ページ境界でも会話を分割したまま取り込み、最後に保存した位置から自動で継続します。
- 20発言・発言間隔30分・設定文字数を目安に会話をまとめ、日時・匿名化した話者識別子とともに保存します。長い単独発言は全文を分割して索引化します。取得ページ境界では会話が分かれます。
- Bot/Webhookの発言、システム通知、空本文だけが除外されます。人間の発言はオプトアウト表現や秘密鍵らしき内容を含んだままでも取り込みます（PIIマスキングは従来どおり送信前に適用）。添付ファイル・音声・埋め込みはダウンロードしません。
- 本文のマスキングは通常の資料登録と同じです。検索で見つかった履歴には文書IDと元発言へのDiscordリンクが付きます。本文やユーザー名を進捗ログへ出しません。
- 100件の保存と進捗更新は一つのDBトランザクションです。Embedding失敗やCtrl+Cでは次回、最後に保存したページから自動で再開します。同じデータ領域への二重実行はファイルロックで拒否します。
- メッセージIDで重複を防ぎます。`/forget` で削除した履歴は、再走査しても復活させません。既存メッセージの編集・Discord側の削除の追従、モデル変更後の履歴再索引は行いません。`/forget` で削除した履歴は、再走査しても復活させません。既存メッセージの編集・Discord側の削除の追従、モデル変更後の履歴再索引は行いません。

取り込んだ内容は、既存の記憶と同様に設定サーバー内の利用者全員で共有します。Botに `VIEW_CHANNEL` と `READ_MESSAGE_HISTORY`、アプリにMessage Content Intentが必要です。特に履歴閲覧権限がない場合、Discordはエラーではなく空配列を返すことがあります。件数が0なら権限とIntentを確認してください。

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

## Web UI（デフォルト有効・認証不要）

起動後、`http://localhost:8080/` または `http://サーバーのIP:8080/` で運用画面を開けます。ユーザー名・パスワードは不要です。

```dotenv
HTTP_ENABLED=true
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
```

- `GET /` / `GET /dashboard`: DiscordメッセージのRAG取り込み状況（2秒ごとに自動更新）
- `GET /healthz`: Discord接続完了なら200、未接続なら503
- `GET /metrics/v1`: JSONメトリクス
- `GET /api/ingestion`: 取り込み状況のJSON（キャッシュ無効）

画面では現在処理中の新着発言・過去ログ取得、成功／失敗した直近100件の処理、チャンネル別の処理済み発言数・検索可能な発言数・最終処理メッセージIDを確認できます。各チャンネルからDiscordへ移動できます。失敗した処理は例外の種類を表示します。

発言数・文書数・チャンク数はDBから集計するため、Bot再起動後も保持されます。「検索可能」は現在設定されているモデル・次元のベクトルがあり、文書が削除されていないDiscord発言の件数です。処理済み件数にはBot発言などの対象外・削除済みの記録も含みます。処理中と直近の履歴は起動中のみ保持されます。Discord全体の総発言数は取得していないため、残件数や進捗率は表示しません。

すべて認証不要です。無効にする場合は `HTTP_ENABLED=false` を設定します。旧 `.env` にこの値がある場合も引き続き無効になるので、有効化には `true` に変更してください。旧 `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD` は使用しません。メトリクスのカウンターはプロセス再起動でリセットされます。この画面からLLMへのチャットや設定変更はできません。

## Docker

Docker Compose（BuildKit対応）が使える環境で、先に `.env` を設定してください。`HF_TOKEN` にHugging Faceのread権限のあるトークンを設定します。`EMBEDDING_MODEL=embeddinggemma-2` / `EMBEDDING_DIMENSION=768` を使用してください。

```bash
docker compose build
docker compose run --rm bot python -m ai_kun --check
docker compose up -d
docker compose logs -f bot
# 停止（データは名前付きボリュームへ保持）
docker compose down
```

コンテナは非rootで動き、`.env` はビルドコンテキストから除外します。埋め込みイメージの取得ステップにだけBuildKit secretとしてマウントし、`HF_TOKEN` を読み取ります。トークンをDockerのARG・ENV・レイヤーに保存せず、起動時のモデル取得も行いません。Botの設定は起動時に渡します。データは `bot-data` 名前付きボリュームです。ホストの `./data` とは別です。ボリューム削除は記憶も削除します。

ComposeはWeb UIのポート（既定8080）を公開します。Botとネットワーク名前空間を共有しているため、ポートマッピングは `embedding` サービスに設定しています。コンテナ内のHTTP待ち受けは `0.0.0.0` です。

APIベースURLはHTTPS、またはループバック・RFC1918のプライベートIPv4アドレスへのHTTPを指定できます。HTTPで許可する範囲は `127.0.0.0/8`、`::1`、`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16` と、承認済みホスト名 `localhost` / `localhost.localdomain` / `ip6-loopback` です。例えば `http://192.168.0.123:8080/v1` を使用できます。CGNAT・リンクローカル・公開IP・その他のホスト名はHTTPでは許可しません。設定時とAPI送信時には同じ宛先判定を使用し、URLの書き換えやリダイレクトは行いません。

```dotenv
LLM_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
```

Composeには `host.docker.internal:host-gateway` のマッピングを設定しています。これは [Docker公式のhost-gateway指定](https://docs.docker.com/reference/cli/docker/container/run/#add-entries-to-container-hosts-file---add-host)に従っています。推論サーバーはDockerブリッジから到達できるインターフェースで待ち受ける必要があります。ホストの `127.0.0.1` にしか待ち受けていないサーバーには、この設定だけでは接続できません。ポート番号も実際の環境に合わせてください。

## 埋め込みサービス（CPUのみ）

`compose.yaml` の `embedding` サービスは、CPU用のllama.cppで [EmbeddingGemma 2のテキスト用GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)（BF16）を提供します。`docker compose build` の際に `.env` の `HF_TOKEN` で認証し、固定リビジョンの重みをイメージの `/models/embeddinggemma-2.gguf` に組み込みます。取得失敗やトークン未設定ならビルドは失敗します。

Botは埋め込みサービスのネットワーク名前空間を共有し、`http://127.0.0.1:11434/v1` へ接続します。埋め込みAPIはホストやLANにポート公開せず、ヘルスチェック成功後にBotを起動します。GPUは使いません。埋め込みコンテナは非root、読み取り専用ファイルシステムで動作します。モデル用ボリュームは不要です。

旧設定の `gemma2` / 3072次元を使用していた場合は `.env` も変更してください。モデル・次元が異なる既存ベクトルは検索対象外になるため、既存資料の再取り込みが必要です。

## systemd

`deploy/systemd/ai-kun.service` を用意しています。専用ユーザーと `/opt/ai-kun` への配置を前提にした例です。実際のユーザー・パスに変更し、仮想環境、`.env`、書き込み可能な `data` ディレクトリを用意してからインストールしてください。サービスの自動登録・起動は行いません。

## 開発・確認

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check ai_kun tests
uv run ruff format --check ai_kun tests
```

テストは一時DBと模擬APIを使用します。設定検証、登録・再起動・検索・削除、履歴のページング・重複防止・再開、Discordのレート制限追従・429待機、API送信前のマスキング、ツール呼び出し、Discordの許可判定、認証不要のWeb UIを確認します。Discord実サーバー・外部LLM・Tavily・ASRモデルの実機疎通は別途必要です。

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
