"""任意のローカル運用エンドポイント。生成・登録APIは公開しない。"""

import base64
import hmac

from aiohttp import web

from .observability import metrics


def create_app(settings, client):
    operational = settings.operational

    @web.middleware
    async def auth(request, handler):
        if request.path != "/healthz":
            username, password = operational.dashboard_username, operational.dashboard_password
            if not username or not password:
                raise web.HTTPNotFound()
            expected = base64.b64encode(f"{username}:{password}".encode()).decode()
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Basic {expected}".encode()):
                raise web.HTTPUnauthorized(headers={"WWW-Authenticate": 'Basic realm="AI-kun"'})
        return await handler(request)

    app = web.Application(middlewares=[auth])

    async def health(request):
        ready = client.is_ready()
        return web.json_response(
            {"status": "ok" if ready else "connecting", "discord_ready": ready}, status=200 if ready else 503
        )

    async def snapshot(request):
        return web.json_response(
            {
                "counters": metrics.snapshot(),
            }
        )

    async def dashboard(request):
        return web.Response(
            text="""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>AIくん</title><h1>AIくん 運用情報</h1><p><a href="/healthz">接続状態</a></p>
<p><a href="/metrics/v1">メトリクス（JSON）</a></p></html>""",
            content_type="text/html",
        )

    app.router.add_get("/healthz", health)
    app.router.add_get("/metrics/v1", snapshot)
    app.router.add_get("/dashboard", dashboard)
    return app
