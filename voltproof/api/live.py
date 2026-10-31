"""Live runs of the frozen A4 for typed goals (SPEC §6.13), behind the demo's limits.

- `run_a4` is imported and called unchanged, at the configured reasoning effort (medium), with
  tags {"demo": True, "run": ...} so every call (agent turns and the reviewer) counts against
  DEMO_BUDGET_USD_CAP in the client's own budget guard.
- `ObservingLLM` wraps the client and emits the steps while the agent works (tool calls, compact tool
  results, the reviewer's verdict); model reasoning text is never emitted. The authoritative record is
  the trace `run_a4` returns, which the final view is built from.
- `Limits` (in memory only): 10 live runs per hour per IP, 150 a day, 2 at once. IPs are dictionary
  keys in memory; they are never logged or written anywhere.
- `LiveState`: an account error (missing or rejected key, HTTP 401/402/403) switches live runs off until
  restart; a 429 that outlasts the retries pauses them for 10 minutes, after which a fresh client is
  built (the old one latched after the fatal error).
- One log line per live run: its cost and the running totals since start (never the IP or the goal).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from voltproof.agents.gate import parse_review
from voltproof.agents.loop import ChatLLM
from voltproof.agents.tools import EpisodeContext
from voltproof.agents.voltproof import run_a4
from voltproof.api.view import view_model
from voltproof.llm.client import LLMClientStopped, LLMFatalError, LLMRateLimitPersisted, LLMResponse, MissingAPIKey
from voltproof.models import DayData, Site, Tariff

log = logging.getLogger("voltproof.live")

PER_IP_PER_HOUR = 10
DAILY_RUNS = 150
MAX_CONCURRENT = 2
UPSTREAM_PAUSE_S = 600.0
RUN_TTL_S = 3600.0
MAX_RUNS = 50
RUN_RESERVE_USD = 0.10  # what one live run may cost at most, for the pre-check against the demo cap

Clock = Callable[[], float]


# --- limits ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    reason: str  # rate_limit | daily_cap | busy
    retry_after_s: int | None = None


class Limits:
    """10 per hour per IP, 150 per UTC day, 2 at once; in memory, thread-safe."""

    def __init__(self, clock: Clock = time.time, per_ip: int = PER_IP_PER_HOUR, daily: int = DAILY_RUNS, concurrent: int = MAX_CONCURRENT) -> None:
        self.clock, self.per_ip, self.daily, self.concurrent = clock, per_ip, daily, concurrent
        self._lock = threading.Lock()
        self._by_ip: dict[str, deque[float]] = {}
        self._day: str | None = None
        self._today = 0
        self._running = 0

    def _prune(self, now: float) -> None:
        for ip in list(self._by_ip):
            stamps = self._by_ip[ip]
            while stamps and stamps[0] <= now - 3600.0:
                stamps.popleft()
            if not stamps:
                del self._by_ip[ip]
        day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
        if day != self._day:
            self._day, self._today = day, 0

    def admit(self, ip: str) -> Refusal | None:
        with self._lock:
            now = self.clock()
            self._prune(now)
            stamps = self._by_ip.get(ip, deque())
            if len(stamps) >= self.per_ip:
                return Refusal("rate_limit", int(stamps[0] + 3600.0 - now) + 1)
            if self._today >= self.daily:
                return Refusal("daily_cap")
            if self._running >= self.concurrent:
                return Refusal("busy")
            stamps.append(now)
            self._by_ip[ip] = stamps
            self._today += 1
            self._running += 1
            return None

    def release(self) -> None:
        with self._lock:
            self._running = max(0, self._running - 1)

    def remaining(self, ip: str) -> dict[str, int]:
        with self._lock:
            self._prune(self.clock())
            return {"this_hour": self.per_ip - len(self._by_ip.get(ip, ())), "today": self.daily - self._today}


def client_ip(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """The client's address: the peer, or the forwarded-for entry the trusted proxy added (counted
    from the right, so a client cannot choose it)."""
    if trusted_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= trusted_hops:
            return parts[-trusted_hops]
    return peer or "unknown"


# --- live on / off / paused ---------------------------------------------------------------------------


class LiveState:
    """Whether live runs are allowed, the shared client, and the running totals since start."""

    def __init__(self, live_mode: bool, make_client: Callable[[], ChatLLM], clock: Clock = time.time) -> None:
        self.live_mode = live_mode
        self.make_client = make_client
        self.clock = clock
        self._lock = threading.Lock()
        self._client: ChatLLM | None = None
        self.off_reason: str | None = None
        self.paused_until: float | None = None
        self.runs = 0
        self.cost_usd = 0.0

    def unavailable(self) -> str | None:
        """Why a live run cannot start now (None: it can)."""
        with self._lock:
            if not self.live_mode:
                return "live_mode_off"
            if self.off_reason:
                return "live_off"
            if self.paused_until is not None:
                if self.clock() < self.paused_until:
                    return "paused"
                self.paused_until, self._client = None, None  # pause over: a fresh client next time
            return None

    def client(self) -> ChatLLM:
        with self._lock:
            if self._client is None:
                self._client = self.make_client()
            return self._client

    def pause(self) -> None:
        with self._lock:
            self.paused_until = self.clock() + UPSTREAM_PAUSE_S

    def switch_off(self, reason: str) -> None:
        with self._lock:
            self.off_reason = reason

    def record(self, cost_usd: float) -> tuple[int, float]:
        with self._lock:
            self.runs += 1
            self.cost_usd += cost_usd
            return self.runs, self.cost_usd

    def status(self) -> dict[str, Any]:
        reason = self.unavailable()
        until = self.paused_until
        return {
            "live": reason is None,
            "reason": reason,
            "paused_until": datetime.fromtimestamp(until, timezone.utc).isoformat(timespec="seconds") if reason == "paused" and until else None,
        }


# --- one run ----------------------------------------------------------------------------------------------


@dataclass
class LiveRun:
    run_id: str
    created: float
    status: str = "running"  # running | done | unavailable
    reason: str | None = None
    view: dict[str, Any] | None = None
    cost_usd: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, kind: str, data: Mapping[str, Any]) -> None:
        with self.cond:
            self.events.append({"kind": kind, "data": dict(data)})
            self.cond.notify_all()

    def finish(self, status: str, reason: str | None = None, view: dict[str, Any] | None = None) -> None:
        with self.cond:
            self.status, self.reason, self.view = status, reason, view
            self.cond.notify_all()


class RunStore:
    """The last runs, in memory (at most 50, for an hour)."""

    def __init__(self, clock: Clock = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._runs: dict[str, LiveRun] = {}

    def new(self) -> LiveRun:
        with self._lock:
            now = self.clock()
            for run_id in [r for r, run in self._runs.items() if run.status != "running" and run.created < now - RUN_TTL_S]:
                del self._runs[run_id]
            while len(self._runs) >= MAX_RUNS:
                oldest = min(self._runs.values(), key=lambda r: r.created)
                del self._runs[oldest.run_id]
            run = LiveRun(uuid.uuid4().hex, now)
            self._runs[run.run_id] = run
            return run

    def get(self, run_id: str) -> LiveRun | None:
        with self._lock:
            return self._runs.get(run_id)


# --- streaming the agent's steps ------------------------------------------------------------------------


def _compact_result(name: str | None, result: Any) -> dict[str, Any]:
    """What the page shows of a tool result while the agent works (no forecast arrays)."""
    if not isinstance(result, dict):
        return {}
    if "error" in result:
        return {"error": str(result.get("error")), "spec_errors": result.get("spec_errors")}
    if name == "set_checklist":
        return {"checklist": [item.get("text") for item in result.get("checklist", [])]}
    if name == "solve_plan":
        return {k: result.get(k) for k in ("plan_id", "valid", "energy_cost", "currency")} | {"shortfalls": len(result.get("shortfalls", []))}
    if name == "finalize":
        return {k: result.get(k) for k in ("accepted", "plan_id", "escalated")} | {"failed_checks": [f.get("check") for f in result.get("failed_checks", [])]}
    return {}


class ObservingLLM:
    """A ChatLLM that reports the agent's steps as they happen; it changes nothing it passes on."""

    def __init__(self, inner: ChatLLM, emit: Callable[[str, Mapping[str, Any]], None]) -> None:
        self.inner, self.emit = inner, emit
        self._seen = 0
        self._names: dict[str, str] = {}

    def chat(self, messages: Sequence[Mapping[str, Any]], tools: Any = None, tool_choice: Any = None, response_format: Any = None, *, purpose: str, tags: Mapping[str, Any] | None = None) -> LLMResponse:
        if purpose == "agent":
            history = list(messages)
            for message in history[self._seen :]:
                if message.get("role") == "tool":
                    name = self._names.get(str(message.get("tool_call_id")))
                    try:
                        result = json.loads(message.get("content") or "{}")
                    except ValueError:
                        result = {}
                    self.emit("tool_result", {"name": name, "result": _compact_result(name, result)})
            self._seen = len(history)
        response = self.inner.chat(messages, tools, tool_choice, response_format, purpose=purpose, tags=tags)
        if purpose == "agent":
            self.emit("llm_call", {"turn": (tags or {}).get("turn"), "tokens_out": response.usage.completion})
            for call in response.tool_calls:
                self._names[str(call.id)] = call.name
                try:
                    args = json.loads(call.arguments_json or "{}")
                except ValueError:
                    args = {}
                shown = {"items": args.get("items")} if call.name == "set_checklist" else {"spec": args.get("spec")} if call.name == "solve_plan" else {}
                self.emit("tool_call", {"name": call.name, **shown})
        elif purpose == "review":
            verdict = parse_review(response.content)
            self.emit("review", {"approved": verdict.approve if verdict else None, "issues": [i.model_dump() for i in verdict.issues] if verdict else []})
        return response


# --- the worker -------------------------------------------------------------------------------------------


def run_live(run: LiveRun, state: LiveState, *, site: Site, day: DayData, tariff: Tariff, goal: str, model: str, effort: str | None, demo_spend: Callable[[], float], demo_cap: float) -> None:
    """Run the frozen A4 for one typed goal and finish `run`; never raises."""
    status, reason, view, plan_source = "unavailable", "error", None, None
    started = time.monotonic()
    try:
        observer = ObservingLLM(state.client(), run.emit)
        ctx = EpisodeContext(f"live-{run.run_id}", site, day, tariff, goal)
        arm = run_a4(observer, ctx, tags={"demo": True, "run": f"live-{run.run_id}"})
        run.cost_usd = arm.totals.cost_usd
        plan_source = arm.plan_source
        if arm.stop_reason == "budget_exceeded":
            reason = "budget"
        elif arm.stop_reason.startswith("llm_error"):
            reason = "error"
        else:
            meta = {
                "run_id": run.run_id,
                "model": model,
                "reasoning_effort": effort,
                "llm_cost_usd": arm.totals.cost_usd,
                "latency_s": round(time.monotonic() - started, 1),
                "tokens_in": arm.totals.tokens_in,
                "tokens_out": arm.totals.tokens_out,
                "plan_source": arm.plan_source,
            }
            limits = {"export_limit_kw": arm.export_limit_kw, "import_limit_kw": arm.import_limit_kw, "protected_soc_kwh": arm.protected_soc_kwh}
            view = view_model(
                kind="live", site=site, day=day, tariff=tariff, goal=goal, language=None, events=arm.trace.events,
                logged_battery_kw=arm.battery_kw, spec=arm.spec if arm.battery_kw is not None else None, meta=meta,
                on_mismatch="log", logged_limits=limits,
            )
            status, reason = "done", None
    except LLMRateLimitPersisted:
        state.pause()
        reason = "busy_upstream"
    except LLMClientStopped:  # the shared client latched: after a 429 (a pause is running) or an account error
        if state.paused_until is not None:
            reason = "busy_upstream"
        else:
            state.switch_off("LLMClientStopped")
            reason = "live_off"
    except (LLMFatalError, MissingAPIKey) as exc:
        state.switch_off(type(exc).__name__)
        reason = "live_off"
    except Exception:  # anything else: this run fails, the app keeps serving
        log.exception("live run %s failed", run.run_id)
        reason = "error"
    runs, total = state.record(run.cost_usd)
    log.info(
        "live run %s: status=%s reason=%s plan_source=%s cost_usd=%.4f | since start: runs=%d cost_usd=%.4f | demo spend %.4f of %.2f",
        run.run_id, status, reason, plan_source, run.cost_usd, runs, total, demo_spend(), demo_cap,
    )
    run.finish(status, reason, view)
