"""The live API (SPEC §6.13): validation, limits, pauses, streaming, the run view, the log line.

Offline: a synthetic site and day, the real A4 on a scripted fake LLM, a fake clock; no network, no
held-out site."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.fakes_data import TOU_TEMPLATE
from tests.fakes_llm import agent_day, agent_site
from tests.test_counterfactual import a4_llm
from voltproof.api import view as vw
from voltproof.api.app import create_app
from voltproof.api.live import UPSTREAM_PAUSE_S, Limits, client_ip
from voltproof.config import Settings
from voltproof.data.tariff import build_tariff
from voltproof.llm.client import BudgetExceeded, LLMAccountError, LLMRateLimitPersisted

SITE = agent_site()
DAY = date(2024, 6, 5)
SPLITS = {"dev": [{"site_id": SITE.site_id, "date": DAY.isoformat()}], "test": []}
GOAL = "Power cut from 17:00 to 20:00. Keep the critical load running."


def build(site, d):
    return agent_day().model_copy(update={"date": d, "site_id": site.site_id}), build_tariff(TOU_TEMPLATE, d, seed=2026, stream=site.site_id)


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


class Raising:
    """A client whose every call raises `error` (and counts the calls)."""

    def __init__(self, error: BaseException) -> None:
        self.error, self.calls = error, 0

    def chat(self, *args: Any, **kwargs: Any):
        self.calls += 1
        raise self.error


class Blocking:
    """A client that waits until released, so runs stay in progress."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def chat(self, *args: Any, **kwargs: Any):
        self.release.wait(10)
        raise BudgetExceeded("released")


def make(settings: Settings, client_factory, *, live: bool = True, hops: int = 1, clock: Clock | None = None, limits: Limits | None = None):
    s = dataclasses.replace(settings, live_mode=live, trusted_proxy_hops=hops)
    clock = clock or Clock()
    app = create_app(settings=s, make_client=client_factory, clock=clock, sites={SITE.site_id: SITE}, splits=SPLITS, day_builder=build, limits=limits or Limits(clock))
    return TestClient(app), app, clock


def post(client: TestClient, ip: str = "203.0.113.5", goal: str = GOAL, **body: Any):
    return client.post("/api/plan", json={"site": SITE.site_id, "day": DAY.isoformat(), "goal": goal, **body}, headers={"x-forwarded-for": f"198.51.100.1, {ip}"})


def wait(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        result = client.get(f"/api/plan/{run_id}").json()
        if result["status"] != "running":
            return result
        time.sleep(0.05)
    raise AssertionError("the run did not finish")


def test_status_sites_days_and_static_endpoints(settings: Settings) -> None:
    client, _, _ = make(settings, lambda s: a4_llm())
    status = client.get("/api/status", headers={"x-forwarded-for": "203.0.113.5"}).json()
    assert status["live"] is True and status["runs_left_this_hour"] == 10 and status["runs_left_today"] == 150 and status["demo_cap_usd"] == settings.demo_budget_usd_cap
    assert [c["site_id"] for c in client.get("/api/sites").json()] == [SITE.site_id]
    days = client.get("/api/days", params={"site": SITE.site_id}).json()
    assert [(d["date"], d["split"]) for d in days] == [(DAY.isoformat(), "dev")]
    assert client.get("/api/days", params={"site": "nowhere"}).status_code == 422
    assert client.post("/api/tariff-from-image").status_code == 501
    assert set(client.get("/api/benchmark").json()) == {"main", "e2", "e3", "e4"}


def test_goal_site_and_day_are_validated(settings: Settings) -> None:
    client, _, _ = make(settings, lambda s: a4_llm())
    assert post(client, goal="   ").status_code == 422
    assert post(client, goal="x" * 501).status_code == 422
    assert client.post("/api/plan", json={"site": "nowhere", "day": DAY.isoformat(), "goal": GOAL}).status_code == 422
    assert client.post("/api/plan", json={"site": SITE.site_id, "day": "2024-06-06", "goal": GOAL}).status_code == 422
    assert post(client, extra="x").status_code == 422


def test_live_mode_off_is_the_kill_switch(settings: Settings) -> None:
    client, _, _ = make(settings, lambda s: a4_llm(), live=False)
    response = post(client)
    assert response.status_code == 503 and response.json()["reason"] == "live_mode_off"
    assert client.get("/api/status").json()["live"] is False


def test_a_live_run_streams_its_steps_and_ends_with_the_view(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    llm = a4_llm()
    client, _, _ = make(settings, lambda s: llm)
    caplog.set_level(logging.INFO, logger="voltproof.live")
    run_id = post(client).json()["run_id"]
    result = wait(client, run_id)
    assert result["status"] == "done" and result["reason"] is None
    view = result["view"]
    assert view["kind"] == "live" and view["label"] == vw.LIVE_LABEL and view["checked_against"] == vw.AGENT_SPEC_NOTE
    assert view["spec"] == {"grid_outages": [{"start_hour": 17, "end_hour": 20}]} and view["plan"]["verified"] and "plan_notice" not in view
    assert view["same_day"]["agent"]["goals_met"] is True and view["meta"]["reasoning_effort"] == settings.llm_reasoning_effort
    assert all(call["tags"].get("demo") is True and call["tags"]["run"] == f"live-{run_id}" for call in llm.calls)
    stream = client.get(f"/api/plan/{run_id}/events").text
    kinds = [line.split(": ", 1)[1] for line in stream.splitlines() if line.startswith("event: ")]
    assert kinds[-1] == "end" and "review" in kinds and kinds.index("tool_call") < kinds.index("tool_result")
    end = json.loads(stream.strip().splitlines()[-1].split(": ", 1)[1])
    assert end == {"status": "done", "reason": None}
    assert "reasoning" not in stream
    assert client.post("/api/replay", json={"run_id": run_id}).json() == view["same_day"]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(f"live run {run_id}"))
    assert "cost_usd=" in line and "since start: runs=1" in line and "demo spend" in line
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "203.0.113.5" not in logged and GOAL not in logged


def test_a_plan_mismatch_in_live_mode_shows_the_logged_plan(settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    original = vw.solve

    def shifted(*args: Any, **kwargs: Any):
        plan = original(*args, **kwargs)
        return plan.model_copy(update={"battery_kw": [b + 0.5 for b in plan.battery_kw]})

    # only the view's re-solve is shifted; the agent's plan book uses its own (frozen) import of solve
    monkeypatch.setattr(vw, "solve", shifted)
    caplog.set_level(logging.WARNING, logger="voltproof.live")
    client, _, _ = make(settings, lambda s: a4_llm())
    view = wait(client, post(client).json()["run_id"])["view"]
    assert view["plan_notice"] == vw.PLAN_NOTICE and view["plan"] is not None
    assert any("differs from the logged plan" in r.getMessage() for r in caplog.records)


def test_per_ip_limit_frees_up_after_an_hour_and_other_ips_are_separate(settings: Settings) -> None:
    client, _, clock = make(settings, lambda s: Raising(BudgetExceeded("cap")))
    for _ in range(10):
        assert post(client, ip="203.0.113.5").status_code == 200
        clock.now += 1
        time.sleep(0.01)
    refused = post(client, ip="203.0.113.5")
    assert refused.status_code == 429 and refused.json()["reason"] == "rate_limit" and int(refused.headers["retry-after"]) > 0
    assert post(client, ip="203.0.113.9").status_code == 200
    assert client.get("/api/status", headers={"x-forwarded-for": "203.0.113.5"}).json()["runs_left_this_hour"] == 0
    clock.now += 3600
    assert post(client, ip="203.0.113.5").status_code == 200


def test_daily_cap_and_two_at_once(settings: Settings) -> None:
    blocking = Blocking()
    clock = Clock()
    client, _, _ = make(settings, lambda s: blocking, clock=clock, limits=Limits(clock, daily=3))
    first, second = post(client, ip="203.0.113.1"), post(client, ip="203.0.113.2")
    assert first.status_code == 200 and second.status_code == 200
    busy = post(client, ip="203.0.113.3")
    assert busy.status_code == 429 and busy.json()["reason"] == "busy"
    blocking.release.set()
    wait(client, first.json()["run_id"]), wait(client, second.json()["run_id"])
    assert post(client, ip="203.0.113.4").status_code == 200
    capped = post(client, ip="203.0.113.5")
    assert capped.status_code == 429 and capped.json()["reason"] == "daily_cap"


def test_demo_budget_precheck(settings: Settings) -> None:
    settings.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ledger_path.write_text(json.dumps({"cost_usd": settings.demo_budget_usd_cap - 0.05, "tags": {"demo": True}}) + "\n", encoding="utf-8")
    client, _, _ = make(settings, lambda s: a4_llm())
    response = post(client)
    assert response.status_code == 503 and response.json()["reason"] == "budget"


def test_a_budget_stop_marks_the_run_unavailable(settings: Settings) -> None:
    client, _, _ = make(settings, lambda s: Raising(BudgetExceeded("demo cap")))
    result = wait(client, post(client).json()["run_id"])
    assert result == {"status": "unavailable", "reason": "budget", "view": None}


def test_a_persistent_429_pauses_live_runs_then_builds_a_fresh_client(settings: Settings) -> None:
    built: list[Raising] = []

    def factory(s: Settings):
        built.append(Raising(LLMRateLimitPersisted("429", status_code=429)))
        return built[-1]

    client, _, clock = make(settings, factory)
    result = wait(client, post(client).json()["run_id"])
    assert result["status"] == "unavailable" and result["reason"] == "busy_upstream"
    status = client.get("/api/status").json()
    assert status["live"] is False and status["reason"] == "paused" and status["paused_until"]
    paused = post(client)
    assert paused.status_code == 503 and paused.json()["reason"] == "paused"
    clock.now += UPSTREAM_PAUSE_S + 1
    assert client.get("/api/status").json()["live"] is True
    wait(client, post(client).json()["run_id"])
    assert len(built) == 2  # a fresh client after the pause


def test_an_account_error_switches_live_runs_off_until_restart(settings: Settings) -> None:
    client, _, clock = make(settings, lambda s: Raising(LLMAccountError("401", status_code=401)))
    result = wait(client, post(client).json()["run_id"])
    assert result["reason"] == "live_off"
    clock.now += 10 * UPSTREAM_PAUSE_S
    refused = post(client)
    assert refused.status_code == 503 and refused.json()["reason"] == "live_off"


def test_client_ip_uses_the_hop_the_trusted_proxy_added() -> None:
    assert client_ip("10.0.0.1", "1.1.1.1, 2.2.2.2", 1) == "2.2.2.2"  # a client can only prepend entries
    assert client_ip("10.0.0.1", "1.1.1.1, 2.2.2.2", 0) == "10.0.0.1"
    assert client_ip("10.0.0.1", None, 1) == "10.0.0.1"
    assert client_ip(None, None, 0) == "unknown"
