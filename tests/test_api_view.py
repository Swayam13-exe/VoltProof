"""The page's view model and the precomputed replays (SPEC §6.13).

Offline: the A4 run comes from the real agent on the scripted fake LLM (see test_counterfactual);
synthetic site, no held-out site. The committed web/replays/*.json are checked for shape only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_counterfactual import a4_llm, episode, write_run
from voltproof.agents.voltproof import run_a4
from voltproof.api import replays as rp
from voltproof.api import view as vw
from voltproof.bench.episode import baseline_results, evaluate
from voltproof.models import HOURS

SERIES = ("battery_kw", "soc_pct", "charge_kw", "discharge_kw", "grid_import_kw", "grid_export_kw", "curtail_kw", "pv_forecast_kw", "load_forecast_kw", "import_price_per_kwh")
KEYS = {"schema_version", "kind", "label", "checked_against", "meta", "site", "day", "goal", "checklist", "spec", "assumptions", "spec_assumptions", "explanation", "coverage", "gate", "trace", "plan", "same_day"}


@pytest.fixture()
def recorded(tmp_path: Path):
    ep = episode()
    run = run_a4(a4_llm(), ep.context())
    rows = write_run(tmp_path / "run", ep, {"A4": run})
    return ep, run, tmp_path / "run", {(r["episode_id"], r["arm"]): r for r in rows}


def live_view(ep, run) -> dict:
    return vw.view_model(
        kind="live", site=ep.site, day=ep.day, tariff=ep.tariff, goal=ep.goal.text, language="en",
        events=run.trace.events, logged_battery_kw=run.battery_kw, spec=run.spec, meta={"run_id": "x"},
    )


def test_view_model_shows_the_accepted_plan_and_the_same_day_comparison(recorded) -> None:
    ep, run, _, rows = recorded
    view = live_view(ep, run)
    assert set(view) == KEYS and view["kind"] == "live" and view["label"] == vw.LIVE_LABEL
    assert view["checked_against"] == vw.AGENT_SPEC_NOTE  # a live run is checked against the agent's own spec
    assert all(len(view["plan"][k]) == HOURS for k in SERIES)
    assert view["plan"]["battery_kw"] == [round(b, 3) or 0.0 for b in run.battery_kw]
    assert view["plan"]["goals"]["outage_hours"] == [17, 18, 19] and view["plan"]["verified"]
    assert view["checklist"] == ["grid off 17:00-20:00", "keep the bill low"] and view["explanation"] == "Grid off 17:00 to 20:00."
    assert view["gate"]["outcome"]["decision"] == "accepted" and view["gate"]["reviews"][0]["approved"] is True
    benchmark = evaluate(ep, run)
    assert view["same_day"]["agent"]["realized_cost"] == round(benchmark.realized_cost, 2)
    assert view["same_day"]["agent"]["goals_met"] is benchmark.gold_satisfied
    a0 = next(r for r in baseline_results(ep) if r.arm == "A0")
    assert view["same_day"]["rule_based"]["realized_cost"] == round(a0.realized_cost, 2)
    assert not any(k in e["data"] for e in view["trace"] for k in vw.TRACE_DROP if e["kind"] == "llm_call")
    assert rp.dumps(view) == rp.dumps(live_view(ep, run))  # deterministic


def test_the_plan_shown_must_equal_the_logged_plan(recorded) -> None:
    ep, run, _, _ = recorded
    tampered = list(run.battery_kw)
    tampered[3] += 1e-9
    with pytest.raises(vw.LoggedPlanMismatch):
        vw.logged_plan(run.spec, ep.day, ep.site, tampered)


def test_a_run_without_a_plan_shows_the_trace_only(recorded) -> None:
    ep, run, _, _ = recorded
    view = vw.view_model(kind="live", site=ep.site, day=ep.day, tariff=ep.tariff, goal="g", language=None, events=run.trace.events, logged_battery_kw=None, spec=None, meta={})
    assert view["plan"] is None and view["same_day"] is None and view["trace"]


def test_replay_reproduces_its_benchmark_row(recorded) -> None:
    ep, _, run_dir, rows = recorded
    row = rows[(ep.episode_id, "A4")]
    view = rp.build_replay(ep, run_dir, row, {"settings": {"llm_model": "m", "llm_reasoning_effort": "medium"}})
    assert view["kind"] == "replay" and view["label"] == vw.REPLAY_LABEL and view["checked_against"] == vw.GOLD_MATCH_NOTE
    assert view["meta"]["reasoning_effort"] == "medium" and view["meta"]["benchmark_realized_cost"] == row["realized_cost"]
    for field, change in (("realized_cost", 1e-6), ("unserved_critical_kwh", 1e-6), ("gold_satisfied", None)):
        bad = dict(row)
        bad[field] = (not row[field]) if change is None else row[field] + change
        with pytest.raises(rp.ReplayCheckFailed, match=field):
            rp.build_replay(ep, run_dir, bad, {})


def test_build_all_writes_the_files_and_the_index(recorded, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ep, _, run_dir, _ = recorded
    monkeypatch.setattr(rp, "run_episodes", lambda _run_dir: [ep])
    (run_dir / "config.json").write_text(json.dumps({"settings": {"llm_model": "m"}}), encoding="utf-8")
    chips = (rp.Chip("test-outage-en", "run", ep.episode_id, "english"),)
    entries = rp.build_all(chips, run_dir.parent, tmp_path / "replays")
    assert [e["id"] for e in entries] == ["test-outage-en"]
    index = json.loads((tmp_path / "replays" / "index.json").read_text(encoding="utf-8"))
    assert index["chips"] == entries and (tmp_path / "replays" / "test-outage-en.json").is_file()


def test_committed_replays_have_the_page_shape() -> None:
    index = json.loads((rp.REPLAY_DIR / "index.json").read_text(encoding="utf-8"))
    files = {p.name for p in rp.REPLAY_DIR.glob("*.json")} - {"index.json"}
    assert {e["file"] for e in index["chips"]} == files and len(files) == len(rp.CHIPS)
    sites = {e["site_id"] for e in index["chips"]}
    for entry in index["chips"]:
        view = json.loads((rp.REPLAY_DIR / entry["file"]).read_text(encoding="utf-8"))
        assert set(view) == KEYS and view["kind"] == "replay" and view["label"] == vw.REPLAY_LABEL
        assert all(len(view["plan"][k]) == HOURS for k in SERIES) and len(view["same_day"]["agent"]["soc_pct"]) == HOURS
        assert view["goal"]["text"] == entry["goal"] and view["meta"]["reasoning_effort"] == "medium"
        assert view["same_day"]["agent"]["goals_met"] and view["plan"]["verified"]
    assert len(sites) == 3
    for site in sites:  # one English goal, one hard goal and other languages per site
        kinds = {e["kind"] for e in index["chips"] if e["site_id"] == site}
        assert kinds == {"english", "hard", "multilingual"}
    assert {e["language"] for e in index["chips"]} >= {"en", "es", "hi", "pt", "de"}
