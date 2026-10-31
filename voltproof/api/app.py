"""The demo app (SPEC §6.13): one FastAPI app serves the page, the replays and the live API.

Run locally (PowerShell, from the repo root):
    python -m uvicorn voltproof.api.app:create_app --factory --port 7860 --no-access-log

The page (`web/`) is static and also works on its own (replay only). Every response carries a strict
content-security policy: scripts only from this origin and the pinned Chart.js file on cdnjs, no
inline script or style, connections only to this origin. Frames: only this origin and Hugging Face
(a Space shows the app inside its page).

Live runs (typed goals) run the frozen A4 unchanged (see live.py) behind: LIVE_MODE, the per-IP,
daily and concurrency limits, and the demo budget cap; anything else returns a reason the page turns
into a labelled replay. The uvicorn access log stays off, and nothing here logs a client address.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Iterator, Mapping

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints

from voltproof.agents.loop import ChatLLM
from voltproof.api.live import RUN_RESERVE_USD, Limits, LiveState, RunStore, client_ip, run_live
from voltproof.api.view import day_card, site_card
from voltproof.bench.episode import DayBuilder, processed_day_builder
from voltproof.config import PROJECT_ROOT, Settings, get_settings
from voltproof.data.days import load_splits, split_dates
from voltproof.data.sites import SITES
from voltproof.llm.client import LLMClient, read_ledger_totals
from voltproof.models import Site

WEB_DIR = PROJECT_ROOT / "web"
RESULTS_DIR = PROJECT_ROOT / "results"
MAX_GOAL_CHARS = 500
HEARTBEAT_S = 15.0
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
REFUSAL_STATUS = {"rate_limit": 429, "daily_cap": 429, "busy": 429}


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site: str
    day: date
    goal: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_GOAL_CHARS)]


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str


def split_days(sites: Mapping[str, Site], splits: Mapping[str, Any]) -> dict[str, list[tuple[date, str]]]:
    """Every site's demo days: its dev and test split days (a held-out site has test days only)."""
    out: dict[str, list[tuple[date, str]]] = {}
    for site_id in sites:
        days = [(d, split) for split in ("dev", "test") for d in split_dates(splits, split, site_id)]
        out[site_id] = sorted(days)
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def benchmark_summary(results_dir: Path) -> dict[str, Any]:
    """The headline tables for the evidence tab (read from the committed results)."""
    main = read_json(results_dir / "report" / "summary.json") or {}
    e2 = read_json(results_dir / "e2" / "summary.json") or {}
    e3 = read_json(results_dir / "e3" / "summary.json") or {}
    e4 = read_json(results_dir / "e4" / "summary.json") or {}
    tables = main.get("tables", {})
    return {
        "main": {name: tables.get(name) for name in ("goal_satisfaction", "goal_satisfaction_bootstrap", "error_decomposition", "a4_spec_by_language")},
        "e2": (e2.get("tables") or {}).get("goal_satisfaction"),
        "e3": e3.get("summary"),
        "e4": (e4.get("tables") or {}).get("e4_by_effort"),
    }


def configure_logging() -> None:
    """Send the app's own INFO lines (one per live run) to stderr, where the host shows them; uvicorn
    configures only its own loggers. Idempotent; the lines never hold an IP or a goal."""
    logger = logging.getLogger("voltproof")
    logger.setLevel(logging.INFO)
    if not any(getattr(h, "_voltproof", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler._voltproof = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def create_app(
    *,
    settings: Settings | None = None,
    make_client: Callable[[Settings], ChatLLM] | None = None,
    clock: Callable[[], float] | None = None,
    sites: Mapping[str, Site] | None = None,
    splits: Mapping[str, Any] | None = None,
    day_builder: DayBuilder | None = None,
    limits: Limits | None = None,
    web_dir: Path = WEB_DIR,
    results_dir: Path = RESULTS_DIR,
) -> FastAPI:
    """The app (uvicorn builds it with --factory, so importing this module reads no settings). Every
    dependency can be injected, so the tests run offline with a scripted LLM."""
    configure_logging()
    settings = settings or get_settings()
    clock = clock or time.time
    sites = sites if sites is not None else SITES
    days = split_days(sites, splits if splits is not None else load_splits())
    build = day_builder or processed_day_builder()
    factory = make_client or (lambda s: LLMClient(s))
    state = LiveState(settings.live_mode, lambda: factory(settings), clock)
    limits = limits or Limits(clock)
    store = RunStore(clock)
    workers = ThreadPoolExecutor(max_workers=limits.concurrent, thread_name_prefix="live")

    app = FastAPI(title="VoltProof AI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.live, app.state.limits, app.state.store = state, limits, store

    def demo_spend() -> float:
        return read_ledger_totals(settings.ledger_path).demo_usd

    def caller(request: Request) -> str:
        peer = request.client.host if request.client else None
        return client_ip(peer, request.headers.get("x-forwarded-for"), settings.trusted_proxy_hops)

    def site_or_422(site_id: str) -> Site:
        if site_id not in sites:
            raise HTTPException(422, "unknown site")
        return sites[site_id]

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    def status(request: Request) -> dict[str, Any]:
        remaining = limits.remaining(caller(request))
        return {**state.status(), "runs_left_this_hour": remaining["this_hour"], "runs_left_today": remaining["today"], "demo_cap_usd": settings.demo_budget_usd_cap}

    @app.get("/api/sites")
    def api_sites() -> list[dict[str, Any]]:
        cards = []
        for site_id, site in sites.items():
            first = days[site_id][0][0] if days.get(site_id) else None
            if first is None:
                continue
            _, tariff = build(site, first)
            cards.append(site_card(site, tariff))
        return cards

    @app.get("/api/days")
    def api_days(site: str = Query(...)) -> list[dict[str, Any]]:
        chosen = site_or_422(site)
        out = []
        for d, split in days.get(site, []):
            day, _ = build(chosen, d)
            out.append({**day_card(day), "split": split})
        return out

    @app.get("/api/scenarios")
    def api_scenarios() -> Any:
        return read_json(web_dir / "replays" / "index.json") or {"chips": []}

    @app.get("/api/benchmark")
    def api_benchmark() -> dict[str, Any]:
        return benchmark_summary(results_dir)

    @app.post("/api/plan")
    def api_plan(body: PlanRequest, request: Request) -> JSONResponse:
        site = site_or_422(body.site)
        if body.day not in {d for d, _ in days.get(body.site, [])}:
            raise HTTPException(422, "unknown day for this site")
        unavailable = state.unavailable()
        if unavailable:
            return JSONResponse({"reason": unavailable, **state.status()}, status_code=503)
        if demo_spend() + RUN_RESERVE_USD > settings.demo_budget_usd_cap:
            return JSONResponse({"reason": "budget"}, status_code=503)
        refusal = limits.admit(caller(request))
        if refusal:
            headers = {"Retry-After": str(refusal.retry_after_s)} if refusal.retry_after_s else {}
            return JSONResponse({"reason": refusal.reason, "retry_after_s": refusal.retry_after_s}, status_code=REFUSAL_STATUS[refusal.reason], headers=headers)
        day, tariff = build(site, body.day)
        run = store.new()

        def work() -> None:
            try:
                run_live(run, state, site=site, day=day, tariff=tariff, goal=body.goal, model=settings.llm_model, effort=settings.llm_reasoning_effort, demo_spend=demo_spend, demo_cap=settings.demo_budget_usd_cap)
            finally:
                limits.release()

        workers.submit(work)
        return JSONResponse({"run_id": run.run_id})

    def get_run(run_id: str):  # type: ignore[no-untyped-def]
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return run

    @app.get("/api/plan/{run_id}/events")
    def api_events(run_id: str) -> StreamingResponse:
        run = get_run(run_id)

        def stream() -> Iterator[str]:
            sent = 0
            while True:
                with run.cond:
                    if sent == len(run.events) and run.status == "running":
                        run.cond.wait(timeout=HEARTBEAT_S)
                    events = run.events[sent:]
                    finished = run.status != "running"
                for event in events:
                    yield f"event: {event['kind']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                sent += len(events)
                if finished and sent == len(run.events):
                    yield f"event: end\ndata: {json.dumps({'status': run.status, 'reason': run.reason})}\n\n"
                    return
                if not events:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/plan/{run_id}")
    def api_run(run_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        return {"status": run.status, "reason": run.reason, "view": run.view}

    @app.post("/api/replay")
    def api_replay(body: ReplayRequest) -> dict[str, Any]:
        run = get_run(body.run_id)
        if run.view is None or run.view.get("same_day") is None:
            raise HTTPException(409, "this run has no plan to replay")
        return run.view["same_day"]

    @app.post("/api/tariff-from-image")
    def api_tariff_from_image() -> JSONResponse:
        return JSONResponse({"reason": "not_built"}, status_code=501)

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")  # last: the API routes come first
    return app

