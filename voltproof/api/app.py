"""The demo app (SPEC §6.13): one FastAPI app serves the page, the replays and the live API.

Run locally (PowerShell, from the repo root):
    python -m uvicorn voltproof.api.app:create_app --factory --port 7860 --no-access-log

The page (`web/`) is static and also works on its own (replay only). Every response carries a strict
content-security policy: scripts only from this origin and the pinned Chart.js file on cdnjs, no
inline script or style, connections only to this origin. Frames: only this origin and Hugging Face
(a Space shows the app inside its page).

Live runs (typed goals) run the frozen A4 unchanged (see live.py) behind: LIVE_MODE, the global
hourly, per-visitor, daily and concurrency limits, and the demo budget cap; anything else returns a
reason the page turns into a labelled replay. The uvicorn access log stays off, and nothing here logs
a client address. DEBUG_CLIENT=true adds GET /api/debug/client, which shows the caller its own address
headers (to choose CLIENT_IP_SOURCE on a new host); otherwise that path is a 404.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Iterator, Mapping

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints

from voltproof.agents.loop import ChatLLM
from voltproof.agents.voltproof import MAX_REPAIRS
from voltproof.core.metrics import KW_TOLERANCE, PCT_TOLERANCE
from voltproof.api.live import RUN_RESERVE_USD, Limits, LiveState, RunStore, client_ip, run_live
from voltproof.api.view import day_card, site_card
from voltproof.bench.episode import DayBuilder, processed_day_builder
from voltproof.config import PROJECT_ROOT, Settings, get_settings
from voltproof.data.days import load_splits, split_dates
from voltproof.data.sites import SITES
from voltproof.llm.client import BudgetExceeded, LLMClient, LLMError, LLMFatalError, read_ledger_totals
from voltproof.models import Site
from voltproof.vision.extract import read_bill
from voltproof.vision.image import OUTPUT_MIME, BillImageError, decode_upload, sanitize_image
from voltproof.vision.validate import apply_bill, check_bill

WEB_DIR = PROJECT_ROOT / "web"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLES_DIR = PROJECT_ROOT / "data" / "vision_samples"
SAMPLE_BILLS = ("bill_1_tou_inr", "bill_2_flat_kes", "bill_3_tou_eur")  # one per site, drawn by scripts/make_vision_samples.py
RECORDED_LABEL = "Recorded read of the sample bill"  # never "read just now": it was read earlier, by the probe
RECORDED_READS_FILE = "results/vision_probe/results.json"
BILL_RESERVE_USD = 0.05  # what one bill read may cost at most, for the pre-check against the demo cap
MAX_IMAGE_B64_CHARS = 2_900_000  # a 2 MB image in base64, with a little room
bill_log = logging.getLogger("voltproof.bill")
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
REFUSAL_STATUS = {"hourly_cap": 429, "rate_limit": 429, "daily_cap": 429, "busy": 429}
# What /api/debug/client shows besides X-Forwarded-For (only these, and only when present)
DEBUG_CLIENT_HEADERS = ("X-Real-IP", "Forwarded", "CF-Connecting-IP", "True-Client-IP", "X-Envoy-External-Address", "Fly-Client-IP")


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site: str
    day: date
    goal: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_GOAL_CHARS)]
    tariff: BillTariffIn | None = None  # a tariff confirmed from a bill (live runs only)


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str


class BillTariffIn(BaseModel):
    """A tariff the user confirmed from a bill; always re-validated on the server."""

    model_config = ConfigDict(extra="forbid")
    kind: str
    periods: list[dict[str, Any]]
    export_rate: float | str | None = None  # as typed: check_bill says what is wrong with a non-number
    currency: str | None = None
    fixed_charge: float | None = None

    def as_answer(self, site_currency: str) -> dict[str, Any]:
        return {"currency": self.currency or site_currency, "tariff_kind": self.kind, "periods": self.periods, "export_rate": self.export_rate, "fixed_charge": self.fixed_charge}


class BillReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_id: str
    day: date | None = None
    sample_id: str | None = None
    image_b64: Annotated[str, StringConstraints(max_length=MAX_IMAGE_B64_CHARS)] | None = None
    mime: str | None = None


class TariffCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_id: str
    day: date | None = None
    tariff: BillTariffIn


def split_days(sites: Mapping[str, Site], splits: Mapping[str, Any]) -> dict[str, list[tuple[date, str]]]:
    """Every site's demo days: its dev and test split days (a held-out site has test days only)."""
    out: dict[str, list[tuple[date, str]]] = {}
    for site_id in sites:
        days = [(d, split) for split in ("dev", "test") for d in split_dates(splits, split, site_id)]
        out[site_id] = sorted(days)
    return out


ASSET_LINK = re.compile(r'(<(?:script|link)\b[^>]*?\b(?:src|href)=")([\w./-]+\.(?:css|js))(")')


def with_asset_versions(page: str, web_dir: Path) -> str:
    """index.html with each same-origin stylesheet and script linked as name?v=<first 10 hex of its
    sha256>, so a deploy that changes a file changes its URL and no browser keeps the old copy (the
    static files are served without Cache-Control, which lets browsers reuse them heuristically).
    External links (the pinned Chart.js) and files that do not exist are left as they are."""

    def versioned(match: re.Match[str]) -> str:
        path = web_dir / match.group(2)
        if not path.is_file():
            return match.group(0)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
        return f"{match.group(1)}{match.group(2)}?v={digest}{match.group(3)}"

    return ASSET_LINK.sub(versioned, page)


def with_image_tags(page: str, base_url: str) -> str:
    """index.html with og:image and twitter:image pointing at {base_url}/og.png; unchanged without a base."""
    if not base_url:
        return page
    image = html.escape(f"{base_url}/og.png", quote=True)
    tags = "".join(
        f"  {tag}\n"
        for tag in (
            f'<meta property="og:image" content="{image}">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            f'<meta name="twitter:image" content="{image}">',
        )
    )
    return page.replace("</head>", tags + "</head>", 1)


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
        "main": {name: tables.get(name) for name in ("run_summary", "goal_satisfaction", "goal_satisfaction_bootstrap", "outage_safety", "error_decomposition", "a4_spec_by_language")},
        "e2": (e2.get("tables") or {}).get("goal_satisfaction"),
        "e3": e3.get("summary"),
        "e4": (e4.get("tables") or {}).get("e4_by_effort"),
        # facts from the frozen code the page states (it never types a number by hand)
        "code": {
            "max_repairs": MAX_REPAIRS,
            "max_repairs_source": "voltproof/agents/voltproof.py (MAX_REPAIRS)",
            "spec_match_pct_points": PCT_TOLERANCE,
            "spec_match_kw": KW_TOLERANCE,
            "spec_match_source": "voltproof/core/metrics.py (PCT_TOLERANCE, KW_TOLERANCE)",
        },
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
    bill_limits: Limits | None = None,
    web_dir: Path = WEB_DIR,
    results_dir: Path = RESULTS_DIR,
    samples_dir: Path = SAMPLES_DIR,
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
    limits = limits or Limits(clock, hourly=settings.live_runs_per_hour)
    bill_limits = bill_limits or Limits(clock, per_ip=settings.bill_reads_per_visitor, hourly=settings.bill_reads_per_hour)
    store = RunStore(clock)
    workers = ThreadPoolExecutor(max_workers=limits.concurrent, thread_name_prefix="live")

    app = FastAPI(title="VoltProof AI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.live, app.state.limits, app.state.store = state, limits, store

    def demo_spend() -> float:
        return read_ledger_totals(settings.ledger_path).demo_usd

    def caller(request: Request) -> str | None:
        peer = request.client.host if request.client else None
        return client_ip(peer, request.headers, settings.client_ip_source, settings.trusted_proxy_hops)

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
        return {
            **state.status(),
            "per_visitor_limit": settings.client_ip_source != "none",
            "per_visitor_per_hour": limits.per_ip,
            "demo_video_url": settings.demo_video_url or None,  # the page shows "Watch video" only when set
            "runs_left_this_hour": remaining["this_hour"],
            "runs_left_everyone_this_hour": remaining["everyone_this_hour"],
            "runs_left_today": remaining["today"],
            "demo_cap_usd": settings.demo_budget_usd_cap,
        }

    if settings.debug_client:

        @app.get("/api/debug/client")
        def debug_client(request: Request) -> JSONResponse:
            """Diagnosis only (DEBUG_CLIENT=true): the caller's own address headers, returned to the
            caller and never logged."""
            raw = request.headers.get("x-forwarded-for")
            entries = [p.strip() for p in raw.split(",")] if raw else []
            body = {
                "peer": request.client.host if request.client else None,
                "x_forwarded_for": entries,
                "x_forwarded_for_count": len(entries),
                "headers": {name: request.headers[name] for name in DEBUG_CLIENT_HEADERS if name in request.headers},
                "client_ip": caller(request),
                "client_ip_source": settings.client_ip_source,
                "trusted_proxy_hops": settings.trusted_proxy_hops,
            }
            return JSONResponse(body, headers={"Cache-Control": "no-store"})

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
        day, tariff = build(site, body.day)
        tariff_source = "site"
        if body.tariff is not None:  # the bill's tariff: checked again here (never trusted), then in place of the site's prices
            check = check_bill(body.tariff.as_answer(site.currency), site.currency, average_price(day))
            if not check["ok"]:
                return JSONResponse({"reason": "tariff_invalid", "errors": check["errors"]}, status_code=422)
            day, tariff = apply_bill(day, tariff, check)
            tariff_source = "bill"
        unavailable = state.unavailable()
        if unavailable:
            return JSONResponse({"reason": unavailable, **state.status()}, status_code=503)
        if demo_spend() + RUN_RESERVE_USD > settings.demo_budget_usd_cap:
            return JSONResponse({"reason": "budget"}, status_code=503)
        refusal = limits.admit(caller(request))
        if refusal:
            headers = {"Retry-After": str(refusal.retry_after_s)} if refusal.retry_after_s else {}
            return JSONResponse({"reason": refusal.reason, "retry_after_s": refusal.retry_after_s}, status_code=REFUSAL_STATUS[refusal.reason], headers=headers)
        run = store.new()

        def work() -> None:
            try:
                run_live(run, state, site=site, day=day, tariff=tariff, goal=body.goal, model=settings.llm_model, effort=settings.llm_reasoning_effort, demo_spend=demo_spend, demo_cap=settings.demo_budget_usd_cap, tariff_source=tariff_source)
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

    # --- the bill reader (SPEC §6.12): read it, then confirm it ------------------------------------------

    def average_price(day: Any) -> float:
        prices = list(day.import_price_per_kwh)
        return sum(prices) / len(prices)

    def reference_day(site: Site, day: date | None) -> Any:
        known = [d for d, _ in days.get(site.site_id, [])]
        chosen = day if day in known else (known[0] if known else None)
        if chosen is None:
            raise HTTPException(422, "no day for this site")
        return build(site, chosen)[0]

    def sample_index() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for sample_id in SAMPLE_BILLS:
            truth_path, image_path = samples_dir / f"{sample_id}.json", samples_dir / f"{sample_id}.png"
            if not (truth_path.is_file() and image_path.is_file()):
                continue
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            site = next((s for s in sites.values() if s.currency == truth["currency"]), None)
            if site is None:
                continue
            words = {"time_of_use": "time-of-use", "flat": "flat"}.get(truth["tariff_kind"], truth["tariff_kind"])
            out[sample_id] = {"id": sample_id, "site_id": site.site_id, "currency": truth["currency"], "kind": truth["tariff_kind"], "label": f"Sample bill for {site.name} ({truth['currency']}, {words})"}
        return out

    samples = sample_index()
    recorded_reads = read_json(results_dir / "vision_probe" / "results.json") or {}

    @app.get("/api/bill-samples")
    def api_bill_samples() -> list[dict[str, Any]]:
        return list(samples.values())

    @app.get("/api/bill-samples/{sample_id}.png")
    def api_bill_sample_image(sample_id: str) -> Response:
        if sample_id not in samples:  # only the known samples: no path from the request reaches the disk
            raise HTTPException(404, "unknown sample")
        return Response((samples_dir / f"{sample_id}.png").read_bytes(), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/tariff/check")
    def api_tariff_check(body: TariffCheckRequest) -> dict[str, Any]:
        """The confirm table's edits, checked with the same functions as the server's own check."""
        site = site_or_422(body.site_id)
        return check_bill(body.tariff.as_answer(site.currency), site.currency, average_price(reference_day(site, body.day)))

    @app.post("/api/tariff-from-image")
    def api_tariff_from_image(body: BillReadRequest, request: Request) -> JSONResponse:
        """Read a sample bill or an uploaded one. The image is sanitized in memory, sent once, and dropped:
        never written to disk, logged or hashed.

        A sample bill never dead-ends: whenever a live read cannot happen (live AI off or paused, the demo
        budget, a bill-read limit, a failed call) it returns the sample's recorded read, labelled
        RECORDED_LABEL, with the reason in `why`. Only an upload is refused, with the reason."""
        site = site_or_422(body.site_id)
        reference = average_price(reference_day(site, body.day))
        sample = body.sample_id
        if sample is not None and sample not in samples:
            raise HTTPException(422, "unknown sample")
        if sample is None and (not body.image_b64 or not body.mime):
            raise HTTPException(422, "send an image (PNG or JPEG) or pick a sample bill")

        def recorded_read(why: str) -> JSONResponse:
            recorded = recorded_reads.get(f"{sample}.png") or {}
            check = check_bill(recorded.get("answer"), site.currency, reference)
            return JSONResponse({"source": "recorded", "label": RECORDED_LABEL, "why": why, "recorded_in": RECORDED_READS_FILE, "sample_id": sample, "answer": recorded.get("answer"), "check": check, "cost_usd": recorded.get("cost_usd")})

        def refused(reason: str, status: int, message: str, **extra: Any) -> JSONResponse:
            if sample is not None:
                return recorded_read(reason)
            headers = {"Retry-After": str(extra["retry_after_s"])} if extra.get("retry_after_s") else {}
            return JSONResponse({"reason": reason, "message": message, **extra}, status_code=status, headers=headers)

        unavailable = state.unavailable()
        if unavailable:
            return refused(unavailable, 503, "Reading your own bill needs the live AI, which is not available here right now. Try a sample bill.")
        if sample is not None:
            data, mime = (samples_dir / f"{sample}.png").read_bytes(), "image/png"
        else:
            try:
                data, mime = decode_upload(body.image_b64 or ""), body.mime or ""
            except BillImageError as exc:
                return JSONResponse({"reason": "image", "message": str(exc)}, status_code=422)
        try:
            clean = sanitize_image(data, mime)
        except BillImageError as exc:
            return JSONResponse({"reason": "image", "message": str(exc)}, status_code=422)
        del data
        if demo_spend() + BILL_RESERVE_USD > settings.demo_budget_usd_cap:
            return refused("budget", 503, "The demo's AI budget is used up. Try a sample bill.")
        refusal = bill_limits.admit(caller(request))
        if refusal:
            return refused(refusal.reason, REFUSAL_STATUS[refusal.reason], "No bill reads are left for now. Try a sample bill.", retry_after_s=refusal.retry_after_s, per_visitor_per_hour=bill_limits.per_ip)
        try:
            read = read_bill(state.client(), clean, OUTPUT_MIME, tags={"demo": True, "purpose": "bill"})
        except BudgetExceeded:
            return refused("budget", 503, "The demo's AI budget is used up. Try a sample bill.")
        except LLMFatalError as exc:
            state.switch_off(type(exc).__name__)
            return refused("live_off", 503, "The live AI is switched off here. Try a sample bill.")
        except LLMError:
            return refused("error", 503, "The bill could not be read right now. Try again or use a sample bill.")
        finally:
            bill_limits.release()
            del clean
        check = check_bill(read["answer"], site.currency, reference)
        bill_log.info("bill read: ok=%s unreadable=%s cost_usd=%.4f", check["ok"], check["unreadable"], read["cost_usd"])  # never the image, its hash or the visitor
        return JSONResponse({"source": "live", "label": f"Read by Muse Spark Open in {read['latency_s']:.1f} s", "sample_id": sample, "answer": read["answer"], "check": check, "json": read["json"], "latency_s": read["latency_s"], "cost_usd": read["cost_usd"]})

    # The page, with the link-preview image tags added when the public address is known: og:image must be
    # an absolute URL, so without PUBLIC_BASE_URL there are none (never a relative one).
    page_html = with_asset_versions((web_dir / "index.html").read_text(encoding="utf-8"), web_dir) if (web_dir / "index.html").is_file() else None  # versions computed once, at startup

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def index_page() -> Response:
        if page_html is None:
            raise HTTPException(404, "no page")
        return HTMLResponse(with_image_tags(page_html, settings.public_base_url), headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")  # last: the API routes come first
    return app

