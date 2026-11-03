"""After a live run the page's pipeline strip shows all six stages done (Part C2).

The fixture is a real recorded A4 run with one repair round (copied unchanged from the pilot run
runs/pilot-20261030-170121, episode ahmedabad_cold_store_2024-06-24_import_cap_afternoon_p1: attempt 1
failed the grounded-explanation check, attempt 2 passed and was accepted). The test plays that run's
recorded model replies through the real live path (ObservingLLM inside run_live, the frozen A4, the same
site and day), takes the event stream the page would receive, and folds it with the page's own
pipelineStep (web/app.js) in Node. No network, no LLM call.

The bug it guards: the observer reports a tool's result only at the agent's next call, and after the
accepting finalize there is none, so "Accepted" never arrived and stayed "waiting"."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.fakes_llm import ScriptedLLM, call, reply
from tests.test_web_logic import pure
from voltproof.api.live import LiveRun, LiveState, run_live
from voltproof.bench.episode import processed_day_builder
from voltproof.data.sites import SITES

FIXTURE = Path(__file__).parent / "fixtures" / "a4_trace_with_repair.jsonl"
SITE_ID, DAY = "ahmedabad_cold_store", date(2024, 6, 24)
GOAL = "From 13:00 to 17:00 don't draw more than 30 kW from the grid."  # the episode's goal (episodes_plan.json)
STREAMED = ("tool_call", "tool_result", "review", "gate")
STAGES = ("rules", "optimizer", "checks", "review", "accepted", "replay")


def recorded() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def replies_from(events: list[dict[str, Any]]) -> ScriptedLLM:
    """The run's own model replies: each agent turn's tool calls, and the reviewer's raw verdicts."""
    turns = [e["turn"] for e in events if e["kind"] == "llm_call" and e["name"] == "agent"]
    agent = [reply(None, [call(e["name"], e["data"]["args"], e["call_id"]) for e in events if e["kind"] == "tool_call" and e["turn"] == turn]) for turn in turns]
    reviews = [reply(e["data"]["raw"]) for e in events if e["kind"] == "review"]
    return ScriptedLLM(*agent, review=reviews)


@pytest.fixture(scope="module")
def live_run() -> tuple[LiveRun, list[dict[str, Any]]]:
    site = SITES[SITE_ID]
    day, tariff = processed_day_builder()(site, DAY)
    llm = replies_from(recorded())
    run = LiveRun("replayed-trace", 0.0)
    run_live(run, LiveState(True, lambda: llm), site=site, day=day, tariff=tariff, goal=GOAL, model="recorded", effort=None, demo_spend=lambda: 0.0, demo_cap=10.0)
    stream = [{"kind": e["kind"], "name": e["data"].get("name"), "data": e["data"]} for e in run.events if e["kind"] in STREAMED]
    return run, stream


def fold(events: list[dict[str, Any]]) -> dict[str, Any]:
    return pure(state=f"({json.dumps(events)}).reduce(T.pipelineStep, T.pipelineStart())")["state"]


def ending(run: LiveRun) -> list[dict[str, Any]]:
    """What the page appends: the stream's end, then the fetched result (api/plan/{id})."""
    accepted = run.view["gate"]["outcome"]["decision"] == "accepted"
    return [{"kind": "end", "data": {"status": run.status}}, {"kind": "replayed", "data": {"ok": bool(run.view["plan"]), "accepted": accepted}}]


def test_the_recorded_run_replays_the_same_way_through_the_live_path(live_run) -> None:
    run, _ = live_run
    original = recorded()
    assert run.status == "done" and run.view["plan"] is not None
    gate = run.view["gate"]
    assert gate["repair_rounds"] == 1 and gate["outcome"]["decision"] == "accepted"
    assert [a["failed_checks"] for a in gate["attempts"]] == [[{"check": f["check"], "message": f["message"]} for f in e["data"]["failed_checks"]] for e in original if e["kind"] == "gate" and e["name"] == "finalize"]


def test_the_stream_now_carries_the_gate_decision(live_run) -> None:
    _, stream = live_run
    finalize_results = [e["data"]["result"] for e in stream if e["kind"] == "tool_result" and e["name"] == "finalize"]
    assert finalize_results and not any(r.get("accepted") for r in finalize_results)  # the accepting result is never observed: the cause
    assert [e["data"]["decision"] for e in stream if e["kind"] == "gate"] == ["accepted"]
    assert stream[-1]["kind"] == "gate"  # sent before the stream ends


def test_a_successful_live_run_shows_all_six_stages_done_after_one_repair(live_run) -> None:
    run, stream = live_run
    streamed = fold(stream + ending(run)[:1])
    assert {k: streamed[k] for k in STAGES[:5]} == dict.fromkeys(STAGES[:5], "done") and streamed["replay"] == "active"
    final = fold(stream + ending(run))
    assert {k: final[k] for k in STAGES} == dict.fromkeys(STAGES, "done")
    assert final["repairs"] == 1 and final["failed"] == ["grounded_explanation"]


def test_without_the_gate_event_the_final_result_still_closes_every_stage(live_run) -> None:
    run, stream = live_run
    old = [e for e in stream if e["kind"] != "gate"]  # the stream as an older server sent it
    assert fold(old + ending(run)[:1])["accepted"] != "done"  # the bug
    assert {k: fold(old + ending(run))[k] for k in STAGES} == dict.fromkeys(STAGES, "done")  # the fallback


def test_a_failed_or_errored_run_never_shows_everything_done(live_run) -> None:
    _, stream = live_run
    before_gate = [e for e in stream if e["kind"] != "gate"]
    errored = fold(before_gate + [{"kind": "end", "data": {"status": "unavailable"}}])
    assert errored["accepted"] != "done" and errored["replay"] == "fail"
    no_plan = fold(before_gate + [{"kind": "end", "data": {"status": "done"}}, {"kind": "replayed", "data": {"ok": False, "accepted": False}}])
    assert no_plan["accepted"] != "done" and no_plan["replay"] == "fail"
    escalated = fold(before_gate + [{"kind": "gate", "data": {"decision": "escalated_with_plan"}}, {"kind": "end", "data": {"status": "done"}}, {"kind": "replayed", "data": {"ok": True, "accepted": False}}])
    assert escalated["accepted"] == "fail" and not all(escalated[k] == "done" for k in STAGES)


def test_a_recorded_replay_of_the_same_trace_also_ends_all_done() -> None:
    view = {"trace": [{"kind": e["kind"], "name": e["name"], "data": e["data"]} for e in recorded()], "plan": {}, "gate": {"outcome": {"decision": "accepted"}}}
    out = pure(state=f"[...T.traceEvents({json.dumps(view)}), {{kind: 'end', data: {{status: 'done'}}}}, {{kind: 'replayed', data: {{ok: true, accepted: true}}}}].reduce(T.pipelineStep, T.pipelineStart())")
    assert {k: out["state"][k] for k in STAGES} == dict.fromkeys(STAGES, "done") and out["state"]["repairs"] == 1
