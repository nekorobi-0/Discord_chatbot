"""python -m ai_kun / ai-kun エントリーポイント。"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

from .chat import configure
from .config import load_settings
from .errors import ConfigurationError


def main() -> None:
    parser = argparse.ArgumentParser(description="Discord RAG Bot: AIくん")
    parser.add_argument("--env-file", default=".env", help="読み込む設定ファイル（環境変数が優先）")
    parser.add_argument("--check", action="store_true", help="設定とDBを検証し、外部接続せず終了")
    args = parser.parse_args()
    os.umask(0o077)
    load_dotenv(Path(args.env_file), override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = load_settings()
        service = configure(settings)
    except ConfigurationError as exc:
        print(f"設定エラー: {exc.user_message()}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.check:
        print("設定とDB: OK（Discord / LLM / Embedding の疎通確認は行っていません）")
        return
    try:
        asyncio.run(run(settings, service))
    except ConfigurationError as exc:
        print(f"取り込みエラー: {exc.user_message()}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("中断しました。過去ログは次回、保存済みの進捗から再開します。")
    except Exception as exc:
        logging.getLogger(__name__).error(
            "起動失敗: %s。設定・ネットワーク・Discord権限を確認してください。", type(exc).__name__
        )
        raise SystemExit(1) from None


async def run(settings, service) -> None:
    from .adapters.discord_bot import AIKunClient

    client = AIKunClient(settings, service)
    runner = None
    try:
        if settings.operational.http_enabled:
            from .http_server import create_app

            runner = web.AppRunner(create_app(settings, client), access_log=None)
            await runner.setup()
            await web.TCPSite(runner, settings.operational.http_host, settings.operational.http_port).start()
        async with client:
            await client.start(settings.discord.bot_token)
    finally:
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    main()
