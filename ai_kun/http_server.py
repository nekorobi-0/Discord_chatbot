"""認証不要の運用エンドポイント。生成・登録APIは公開しない。"""

import asyncio
import json
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

    async def voice_transcripts(request):
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )
        queue, history = client.voice_transcripts.subscribe()

        async def send(event):
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            await response.write(f"event: transcript\ndata: {payload}\n\n".encode("utf-8"))

        try:
            await response.prepare(request)
            for event in history:
                await send(event)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    await response.write(b": keep-alive\n\n")
                    continue
                try:
                    await send(event)
                finally:
                    queue.task_done()
        except (asyncio.CancelledError, ConnectionResetError, OSError, RuntimeError):
            pass
        finally:
            client.voice_transcripts.unsubscribe(queue)
        return response

    dashboard_html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    async def dashboard(request):
        return web.Response(text=dashboard_html, content_type="text/html")

    app.router.add_get("/", dashboard)
    app.router.add_get("/healthz", health)
    app.router.add_get("/metrics/v1", snapshot)
    app.router.add_get("/api/ingestion", ingestion)
    app.router.add_get("/api/voice/transcripts", voice_transcripts)
    app.router.add_get("/dashboard", dashboard)
    return app
