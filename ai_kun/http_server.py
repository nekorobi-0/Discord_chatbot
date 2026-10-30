"""認証不要の運用エンドポイント。生成・登録APIは公開しない。"""

from pathlib import Path

from aiohttp import web

from .observability import metrics


def create_app(settings, client):
    app = web.Application()

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

    async def ingestion(request):
        data = await client.service.ingestion_status()
        data["discord_ready"] = client.is_ready()
        return web.json_response(data, headers={"Cache-Control": "no-store"})

    dashboard_html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    async def dashboard(request):
        return web.Response(text=dashboard_html, content_type="text/html")

    app.router.add_get("/", dashboard)
    app.router.add_get("/healthz", health)
    app.router.add_get("/metrics/v1", snapshot)
    app.router.add_get("/api/ingestion", ingestion)
    app.router.add_get("/dashboard", dashboard)
    return app
