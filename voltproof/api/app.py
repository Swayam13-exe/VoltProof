"""The demo app (SPEC §6.13): one FastAPI app serves the page, the replays and (step 3) the live API.

Run locally (PowerShell, from the repo root):
    python -m uvicorn voltproof.api.app:app --port 7860 --no-access-log

The page (`web/`) is static and also works on its own (replay only). Every response carries a strict
content-security policy: scripts only from this origin and the pinned Chart.js file on cdnjs, no
inline script or style, connections only to this origin. Frames: only this origin and Hugging Face
(a Space shows the app inside its page).
"""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from voltproof.config import PROJECT_ROOT

WEB_DIR = PROJECT_ROOT / "web"
CHART_JS_ORIGIN = "https://cdnjs.cloudflare.com"
FRAME_ANCESTORS = "'self' https://huggingface.co"
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        f"script-src 'self' {CHART_JS_ORIGIN}",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "object-src 'none'",
        f"frame-ancestors {FRAME_ANCESTORS}",
    )
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def create_app(web_dir: Path = WEB_DIR) -> FastAPI:
    app = FastAPI(title="VoltProof AI", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")  # last: the API routes come first
    return app


app = create_app()
