"""E4 reasoning-effort sweep: cache keys, routing, the E4 cap, the episode selection, the dry run and the
runner options. Offline: scripted fakes and synthetic sites; no held-out site."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.fakes_llm import ScriptedLLM, reply
from tests.test_runner import SITES, SPLITS, StubArms, day_builder, run_main
from voltproof.bench import effort_sweep as es
from voltproof.bench import runner
from voltproof.bench.intents import LANGUAGES
from voltproof.config import Settings
from voltproof.llm.client import BudgetExceeded, cache_key

KEY_ARGS: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": None, "tool_choice": None, "response_format": None, "max_tokens": 8192}


def test_cache_key_differs_when_only_the_reasoning_effort_differs() -> None:
    keys = {effort: cache_key(reasoning_effort=effort, **KEY_ARGS) for effort in ("low", "medium", "high")}
    assert len(set(keys.values())) == 3
    assert cache_key(reasoning_effort="low", **KEY_ARGS) == keys["low"]  # deterministic


def test_effort_llm_sends_only_the_agents_own_turns_to_the_swept_client() -> None:
    agent, other = ScriptedLLM(agent=[reply("a"), reply("b")]), ScriptedLLM(review=[reply("r")])
    llm = es.EffortLLM(agent, other)
    llm.chat([], purpose="agent")
    llm.chat([], purpose="review")
    llm.chat([], purpose="agent")  # a repair turn is another agent turn
    assert agent.purposes() == ["agent", "agent"] and other.purposes() == ["review"]


def test_effort_llm_builds_two_clients_and_refuses_other_efforts(settings: Settings) -> None:
    llm = es.effort_llm(settings, "high")
    assert llm.agent._settings.llm_reasoning_effort == "high" and llm.other._settings.llm_reasoning_effort == "medium"
    with pytest.raises(ValueError):
        es.effort_llm(settings, "medium")


def ledger_line(run: str, cost: float) -> str:
    return json.dumps({"tags": {"run": run}, "cost_usd": cost}) + "\n"


def test_cap_counts_only_e4_runs_and_refuses_before_passing_it(settings: Settings) -> None:
    settings.ledger_path.parent.mkdir(parents=True)
    settings.ledger_path.write_text(ledger_line("test-x", 30.0) + ledger_line("e4-probe-low-1", 0.5) + ledger_line("e4-low-2", 4.48), encoding="utf-8")
    assert es.e4_spent_usd(settings.ledger_path) == pytest.approx(4.98)
    inner = ScriptedLLM(agent=[reply("ok")])
    capped = es.CappedLLM(inner, settings)
    with pytest.raises(BudgetExceeded, match="E4 cap"):  # 4.98 + one worst-case call (> $0.03) > 5
        capped.chat([{"role": "user", "content": "x" * 20000}], purpose="agent")
    assert inner.calls == []
    roomy = es.CappedLLM(inner, settings, cap_usd=5.0)
    settings.ledger_path.write_text(ledger_line("e4-low-2", 1.0), encoding="utf-8")
    assert roomy.chat([{"role": "user", "content": "x"}], purpose="agent").content == "ok"
    with pytest.raises(ValueError):
        es.CappedLLM(inner, settings, cap_usd=6.0)


def filled_hard():
    from voltproof.bench.intents_hard import load_hard_intents

    return load_hard_intents()


def test_selection_is_stratified_deterministic_and_free_of_duplicate_requests() -> None:
    episodes = es.e4_episodes(None, filled_hard(), SPLITS, SITES, day_builder)
    assert episodes == es.e4_episodes(None, filled_hard(), SPLITS, SITES, day_builder)
    main = [e for e in episodes if es.subset(e.episode_id) == "main"]
    hard = [e for e in episodes if es.subset(e.episode_id) == "hard"]
    assert len(main) == 30 and len(hard) == 30
    for site_id in SITES:
        mine = [e for e in main if e.site.site_id == site_id]
        assert len(mine) == 10 and len({e.goal.template_id for e in mine}) == 10
        ml = [e for e in mine if es.is_multilingual(e)]
        assert len(ml) == 2 and len({e.goal.language for e in ml}) == 2 and all(e.goal.language in LANGUAGES for e in ml)
    by_template: dict[str, list] = {}
    for e in hard:
        by_template.setdefault(e.goal.template_id, []).append(e)
    assert all(len(v) == 3 for v in by_template.values()) and len(by_template) == 10
    assert all(len({e.site.site_id for e in v}) == min(3, len({e.site.site_id for e in v})) for v in by_template.values())
    keys = [(e.site.site_id, e.day.date, e.goal.text) for e in episodes]
    assert len(keys) == len(set(keys))


def test_duplicate_requests_are_the_english_twins_of_multilingual_sources() -> None:
    from voltproof.bench.intents_hard import build_hard_episodes

    hard = build_hard_episodes(filled_hard(), SPLITS, SITES, day_builder)
    duplicates = es.duplicate_requests(hard)
    for eid in duplicates:
        twin = next(e for e in hard if e.episode_id == eid)
        assert any(es.is_multilingual(e) and e.goal.language == "en" and e.goal.text == twin.goal.text and e.day.date == twin.day.date for e in hard)
    assert not duplicates & {e.episode_id for e in es.select_hard(hard)}


def test_dry_run_finds_no_cached_first_call_at_the_swept_efforts(settings: Settings) -> None:
    episodes = es.e4_episodes(None, filled_hard(), SPLITS, SITES, day_builder)[:3]
    key = es.first_call_key(episodes[0], settings, "medium")
    path = settings.cache_dir / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    assert es.dry_run(episodes, settings) == {"low": 0, "high": 0, "medium": 1}
    assert es.first_call_key(episodes[0], settings, "low") != key


def test_runner_e4_options(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--split", "test", "--intents", "e4", "--arms", "A2,A4", "--agent-effort", "low"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--split", "test", "--intents", "e4", "--arms", "A4"])  # medium is not re-run
    with pytest.raises(SystemExit):
        runner.parse_args(["--split", "test", "--intents", "e4", "--arms", "A4", "--agent-effort", "low", "--cost-cap", "6"])
    args = runner.parse_args(["--split", "test", "--intents", "e4", "--arms", "A4", "--agent-effort", "high", "--concurrency", "4"])
    assert args.concurrency == 1
    arms = StubArms()
    assert run_main(tmp_path, settings, ["--split", "test", "--intents", "e4", "--plan-only", "--run-id", "e4-plan-t"], arms, hard_intents=filled_hard()) == 0
    plan = json.loads((tmp_path / "runs" / "e4-plan-t" / "episodes_plan.json").read_text(encoding="utf-8"))
    assert plan["n_episodes"] == 60 and plan["split"] == "test-e4" and arms.calls == []
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"passed": True, "code_fingerprint": runner.code_fingerprint()}), encoding="utf-8")
    argv = ["--split", "test", "--intents", "e4", "--arms", "A4", "--agent-effort", "low", "--limit", "2", "--force"]
    assert run_main(tmp_path, settings, argv, arms, hard_intents=filled_hard(), regression_path=good) == 0 and len(arms.calls) == 2
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.name.startswith("e4-low-"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["agent_reasoning_effort"] == "low" and config["reviewer_reasoning_effort"] == "medium" and config["e4_cap_usd"] == 5.0
    assert "reviewer at medium" in capsys.readouterr().out


# --- the E4 report ---------------------------------------------------------------------------------------


def write_run(path: Path, rows: list[dict[str, Any]], plan_ids: list[str] | None = None, reasoning: int = 100) -> Path:
    (path / "traces").mkdir(parents=True)
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if plan_ids is not None:
        (path / "episodes_plan.json").write_text(json.dumps({"plan_sha256": "abc", "episodes": [{"episode_id": e} for e in plan_ids]}), encoding="utf-8")
    for r in rows:
        calls = [{"kind": "llm_call", "name": "agent", "data": {"tokens_reasoning": reasoning, "finish_reason": "stop", "cache_hit": False}},
                 {"kind": "llm_call", "name": "review", "data": {"tokens_reasoning": 7, "finish_reason": "stop", "cache_hit": True}}]
        (path / "traces" / f"A4__{r['episode_id']}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")
    return path


def a4_row(eid: str, gold: bool, exact: bool = True, cost: float = 0.01, explanation: str = "") -> dict[str, Any]:
    return {"episode_id": eid, "arm": "A4", "gold_satisfied": gold, "plan_physics_valid": True, "spec_precision": 1.0, "spec_recall": 1.0 if exact else 0.5,
            "repair_rounds": 0, "escalated": False, "fallback_used": False, "cost_usd": cost, "latency_s": 10.0,
            "tokens_in": 1000, "tokens_out": 200, "tokens_cached": 0, "explanation": explanation}


def test_e4_report_compares_each_effort_with_the_stored_medium_rows(tmp_path: Path, settings: Settings) -> None:
    from voltproof.models import Shortfall

    ids = ["site_d_cost_min_p0", "site_d_never_below_p1", "site_d_hard_impossible_p0", "site_d_hard_tonight_p1"]
    medium = [a4_row(ids[0], True), a4_row(ids[1], True), a4_row(ids[2], False, explanation="no number"), a4_row(ids[3], True)]
    main_run = write_run(tmp_path / "main", medium[:2])
    hard_run = write_run(tmp_path / "hard", medium[2:])
    low = write_run(tmp_path / "e4-low", [a4_row(ids[0], True), a4_row(ids[1], False, exact=False), a4_row(ids[2], False, explanation="short by 5.75 kWh"), a4_row(ids[3], True)], ids, reasoning=50)
    high = write_run(tmp_path / "e4-high", [a4_row(i, i != ids[2], cost=0.03, explanation="short by 6 kWh") for i in ids], ids, reasoning=900)
    target = Shortfall(path="min_soc_pct_all_day", rule="ALL_DAY_MIN", hours=[0], amount=5.75, unit="kWh", message="m")
    out = es.e4_report(low, high, tmp_path / "results", settings, main_run=main_run, hard_run=hard_run, shortfalls=lambda _: {ids[2]: ([target], 40.0)})
    tables = {t.name: t.rows for t in out["tables"]}
    by = {(r["subset"], r["effort"]): r for r in tables["e4_by_effort"]}
    assert by[("main", "low")]["goal satisfied"] == "50.0%" and by[("main", "medium")]["goal satisfied"] == "100.0%"
    assert by[("all", "high")]["agent reasoning tokens / episode"] == "900" and by[("all", "medium")]["cached calls"] == 4
    assert by[("all", "high")]["billed USD / episode"] == "0.0300"
    boot = {(r["measure"], r["comparison"]): r for r in tables["e4_bootstrap"]}
    assert boot[("goal satisfied", "low - medium")]["main"].startswith("-50.0 pts") and "(1 of 2 episodes differ)" in boot[("goal satisfied", "low - medium")]["main"]
    assert boot[("goal satisfied", "high - medium")]["all"].startswith("+0.0 pts")
    assert {r["effort"]: r["shortfall stated"] for r in tables["e4_impossible"]} == {"low": "1/1", "medium": "0/1", "high": "1/1"}
    text = (tmp_path / "results" / "e4" / "REPORT.md").read_text(encoding="utf-8")
    assert "never pooled" in text and "indicative only" in text and "token USD" in text and out["figure"].is_file()
