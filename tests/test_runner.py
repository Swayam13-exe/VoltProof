"""VoltProof Bench runner: plan construction, resume, budget refusal, stops, error policy, pilot.

Offline: synthetic sites (cold store, clinic, household) and an injected day builder, so no data
file and no held-out site is touched. LLM arms are stubbed; the baseline arms run for real.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.fakes_core import make_day
from tests.fakes_data import DYNAMIC_TEMPLATE, FLAT_TEMPLATE, TOU_TEMPLATE, make_site
from voltproof.agents.loop import LLMTotals, Trace
from voltproof.agents.tools import ArmRun
from voltproof.bench import runner
from voltproof.bench.episode import build_test_episodes, dev_episodes
from voltproof.bench.intents import LANGUAGES, gold_spec, load_intents
from voltproof.config import Settings
from voltproof.data.days import load_splits
from voltproof.data.tariff import build_tariff
from voltproof.llm.client import LLMAccountError
from voltproof.models import GridSpec

STORE = make_site("syn_store", site_type="commercial_cold_store", tariff_template=TOU_TEMPLATE)
CLINIC = make_site("syn_clinic", site_type="health_clinic", tariff_template=FLAT_TEMPLATE).model_copy(
    update={"grid": GridSpec(import_max_kw=50, export_max_kw=0, export_allowed=False)}
)
HOME = make_site("syn_home", site_type="household", tariff_template=DYNAMIC_TEMPLATE)
SITES = {s.site_id: s for s in (STORE, CLINIC, HOME)}
DATES = {"syn_store": ["2025-01-10", "2025-03-04"], "syn_clinic": ["2025-02-01", "2025-05-05"], "syn_home": ["2025-04-02", "2025-06-30"]}
SPLITS = {"test": [{"site_id": s, "date": d} for s, ds in DATES.items() for d in ds], "dev": []}
ENGLISH_PER_DAY = {"syn_store": 12, "syn_clinic": 10, "syn_home": 12}


def day_builder(site, day_date: date):
    day = make_day(pv=[0.0] * 8 + [3.0] * 8 + [0.0] * 8, load=2.0, critical=1.0, price=[5.0] * 12 + [10.0] * 12)
    tariff = build_tariff(site.tariff_template, day_date, seed=2026, stream=site.site_id)
    return day.model_copy(update={"date": day_date, "site_id": site.site_id}), tariff


class StubArms:
    """Stands in for the LLM arms. `plan(arm, call_index)` returns a stop_reason or raises."""

    def __init__(self, plan=None) -> None:
        self.plan = plan or (lambda arm, i: "finalized")
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, arm: str, llm: Any, episode, tags: dict[str, Any]) -> ArmRun:
        with self.lock:
            index = len(self.calls)
            self.calls.append((arm, episode.episode_id))
        outcome = self.plan(arm, index)
        if isinstance(outcome, BaseException):
            raise outcome
        ok = outcome in ("finalized", "final_answer", "llm_error:LLMRequestRejected") and outcome == "finalized"
        trace = Trace()
        trace.add("final", outcome)
        totals = LLMTotals(tokens_in=1000, tokens_out=200, tokens_cached=400, cost_usd=0.002, calls=1)
        return ArmRun(
            arm, [0.0] * 24 if ok else None, None, None, "finalized" if ok else "fallback", not ok, outcome, "stub", trace, totals, 1, 0
        )


def run_main(tmp_path: Path, settings: Settings, argv: list[str], arms: StubArms, **kwargs: Any) -> int:
    return runner.main(
        argv,
        llm=object(),
        settings=settings,
        splits=kwargs.pop("splits", SPLITS),
        sites=SITES,
        day_builder=day_builder,
        runs_dir=tmp_path / "runs",
        run_arm=arms,
        progress=False,
        **kwargs,
    )


def rows(tmp_path: Path, run_id: str, name: str = "episodes.jsonl") -> list[dict[str, Any]]:
    return runner.read_jsonl(tmp_path / "runs" / run_id / name)


# --- plan construction ------------------------------------------------------------------------


def test_test_plan_is_deterministic_with_the_expected_counts() -> None:
    intents = load_intents()
    first = build_test_episodes(intents, SPLITS, SITES, day_builder)
    second = build_test_episodes(intents, SPLITS, SITES, day_builder)
    plan_a, plan_b = runner.plan_document(first, "test", 2026), runner.plan_document(second, "test", 2026)
    assert plan_a == plan_b
    english = [e for e in first if e.goal.language == "en" and e.goal.goal_id.startswith(f"{e.goal.template_id}_p")]
    multilingual = [e for e in first if e not in english]
    assert len(english) == sum(2 * n for n in ENGLISH_PER_DAY.values()) == 68
    for site_id, per_day in ENGLISH_PER_DAY.items():
        assert sum(e.site.site_id == site_id for e in english) == 2 * per_day
    assert len(multilingual) == 6 * 5 * 3 == 90 and plan_a["n_episodes"] == 158
    assert {e.goal.language for e in multilingual} == {"en", *LANGUAGES}
    for site_id in SITES:  # each template uses both phrasings on each site
        for template in {e.goal.template_id for e in english if e.site.site_id == site_id}:
            phrasings = {e.goal.goal_id[-1] for e in english if e.site.site_id == site_id and e.goal.template_id == template}
            assert phrasings == {"0", "1"}
        ml_days = {e.day.date for e in multilingual if e.site.site_id == site_id}
        assert len(ml_days) == 1  # one seeded day per site
    assert runner.plan_document(build_test_episodes(intents, SPLITS, SITES, day_builder, seed=7), "test", 7)["plan_sha256"] != plan_a["plan_sha256"]


def test_held_out_templates_appear_only_in_the_test_set() -> None:
    intents = load_intents()
    held = {t.id for t in intents.templates if t.held_out}
    test = build_test_episodes(intents, SPLITS, SITES, day_builder)
    assert held <= {e.goal.template_id for e in test}
    dev = dev_episodes(intents, load_splits(), day_builder)
    assert dev and not {e.goal.template_id for e in dev} & held and not any(e.site.held_out for e in dev)


def test_multilingual_episodes_carry_language_and_the_template_gold() -> None:
    intents = load_intents()
    for episode in build_test_episodes(intents, SPLITS, SITES, day_builder):
        if episode.goal.goal_id.startswith("ml_"):
            template = intents.template(episode.goal.template_id)
            assert episode.goal.gold == gold_spec(template, episode.site, episode.tariff)
            assert episode.episode_id.endswith(f"_{episode.goal.language}")


# --- runs, resume, refusal ----------------------------------------------------------------------


def test_run_writes_results_and_resume_skips_finished_jobs(tmp_path: Path, settings: Settings) -> None:
    arms = StubArms()
    argv = ["--split", "test", "--arms", "A0,A1,A4", "--limit", "3", "--run-id", "test-r1", "--force"]
    assert run_main(tmp_path, settings, argv, arms) == runner.EXIT_OK
    done = rows(tmp_path, "test-r1")
    assert len(done) == 3 * 3 and len(arms.calls) == 3  # A0 + A1 rows per episode, one A4 job each
    assert (tmp_path / "runs" / "test-r1" / "episodes_plan.json").is_file()
    path = tmp_path / "runs" / "test-r1" / "episodes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    removed = next(line for line in lines if json.loads(line)["arm"] == "A4")
    path.write_text("\n".join(line for line in lines if line != removed) + "\n" + removed[:40], encoding="utf-8")  # crash mid-line
    again = StubArms()
    assert run_main(tmp_path, settings, argv + ["--resume"], again) == runner.EXIT_OK
    assert again.calls == [("A4", json.loads(removed)["episode_id"])]
    keys = [(r["episode_id"], r["arm"]) for r in rows(tmp_path, "test-r1")]
    assert len(keys) == len(set(keys)) == 9


def test_resume_refuses_a_changed_fingerprint(tmp_path: Path, settings: Settings) -> None:
    argv = ["--split", "test", "--arms", "A0", "--limit", "1", "--run-id", "test-r2", "--force"]
    assert run_main(tmp_path, settings, argv, StubArms()) == 0
    config_path = tmp_path / "runs" / "test-r2" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps({**config, "fingerprint": "changed"}), encoding="utf-8")
    assert run_main(tmp_path, settings, argv + ["--resume"], StubArms()) == runner.EXIT_REFUSED


def write_pilot(tmp_path: Path, usd_per_episode: float) -> None:
    pilot = tmp_path / "runs" / "pilot-x"
    pilot.mkdir(parents=True)
    arm = {"mean_token_usd": usd_per_episode, "max_token_usd": usd_per_episode}
    (pilot / "pilot_summary.json").write_text(json.dumps({"run_id": "pilot-x", "arms": {"A2": arm, "A3": arm, "A4": arm}}), encoding="utf-8")


def test_test_run_needs_a_pilot_projection_within_budget(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--split", "test", "--arms", "A4", "--limit", "5"]
    assert run_main(tmp_path, settings, argv, StubArms()) == runner.EXIT_REFUSED  # no pilot summary
    assert "no pilot summary" in capsys.readouterr().out
    write_pilot(tmp_path, 12.0)  # 5 x $12 = $60 > $85 - $30
    arms = StubArms()
    assert run_main(tmp_path, settings, argv, arms) == runner.EXIT_REFUSED and arms.calls == []
    assert "REFUSED" in capsys.readouterr().out
    assert run_main(tmp_path, settings, argv + ["--force"], arms) == runner.EXIT_OK and len(arms.calls) == 5


def test_projection_within_budget_runs(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    write_pilot(tmp_path, 1.0)  # 5 x $1 = $5 <= $55
    arms = StubArms()
    assert run_main(tmp_path, settings, ["--split", "test", "--arms", "A4", "--limit", "5"], arms) == 0 and len(arms.calls) == 5
    assert "projected $5.00" in capsys.readouterr().out


def test_budget_arithmetic() -> None:
    per_arm = {"A2": {"mean_token_usd": 0.02}, "A4": {"mean_token_usd": 0.03}}
    assert runner.projection(per_arm, {"A2": 10, "A3": 0, "A4": 4}) == pytest.approx(0.32)
    with pytest.raises(ValueError, match="A3"):
        runner.projection(per_arm, {"A3": 1})


# --- stops and the error policy ----------------------------------------------------------------


@pytest.mark.parametrize("concurrency", [1, 2])
def test_fatal_error_stops_and_saves(tmp_path: Path, settings: Settings, concurrency: int, capsys: pytest.CaptureFixture[str]) -> None:
    arms = StubArms(lambda arm, i: LLMAccountError("402", status_code=402) if i == 2 else "finalized")
    argv = ["--split", "test", "--arms", "A4", "--limit", "8", "--run-id", "test-f", "--force", "--concurrency", str(concurrency)]
    assert run_main(tmp_path, settings, argv, arms) == runner.EXIT_FATAL
    assert "STOPPED: LLMAccountError (HTTP 402)" in capsys.readouterr().out
    saved = rows(tmp_path, "test-f")
    assert len(saved) >= 2 and len(arms.calls) <= 2 + concurrency  # nothing new submitted after the error


def test_budget_exceeded_stops_and_does_not_record_the_job(tmp_path: Path, settings: Settings) -> None:
    arms = StubArms(lambda arm, i: "budget_exceeded" if i == 1 else "finalized")
    argv = ["--split", "test", "--arms", "A4", "--limit", "5", "--run-id", "test-b", "--force", "--concurrency", "1"]
    assert run_main(tmp_path, settings, argv, arms) == runner.EXIT_BUDGET
    assert [r["episode_id"] for r in rows(tmp_path, "test-b")] == [arms.calls[0][1]]


def test_infrastructure_errors_are_retried_then_recorded_as_fallback(tmp_path: Path, settings: Settings) -> None:
    argv = ["--split", "test", "--arms", "A4", "--limit", "1", "--run-id", "test-i", "--force"]
    infra = StubArms(lambda arm, i: runner.INFRA_STOP)
    assert run_main(tmp_path, settings, argv, infra) == 0
    assert rows(tmp_path, "test-i") == [] and len(rows(tmp_path, "test-i", "errors.jsonl")) == 1
    assert run_main(tmp_path, settings, argv + ["--resume"], infra) == 0
    assert rows(tmp_path, "test-i") == [] and len(rows(tmp_path, "test-i", "errors.jsonl")) == 2
    assert run_main(tmp_path, settings, argv + ["--resume"], infra) == 0
    (result,) = rows(tmp_path, "test-i")
    assert result["infra_retries"] == 2 and result["fallback_used"] and result["stop_reason"] == runner.INFRA_STOP
    assert len(infra.calls) == 3


def test_rejected_requests_are_outcomes_at_once(tmp_path: Path, settings: Settings) -> None:
    arms = StubArms(lambda arm, i: "llm_error:LLMRequestRejected")
    assert run_main(tmp_path, settings, ["--split", "test", "--arms", "A2", "--limit", "1", "--run-id", "test-q", "--force"], arms) == 0
    (result,) = rows(tmp_path, "test-q")
    assert result["fallback_used"] and result["infra_retries"] is None
    assert rows(tmp_path, "test-q", "errors.jsonl") == []


def test_concurrency_writes_every_result_once_with_separate_traces(tmp_path: Path, settings: Settings) -> None:
    arms = StubArms()
    argv = ["--split", "test", "--arms", "A2,A3,A4", "--limit", "4", "--run-id", "test-c", "--force", "--concurrency", "2"]
    assert run_main(tmp_path, settings, argv, arms) == 0
    keys = [(r["episode_id"], r["arm"]) for r in rows(tmp_path, "test-c")]
    assert len(keys) == len(set(keys)) == 12
    assert len(list((tmp_path / "runs" / "test-c" / "traces").glob("*.jsonl"))) == 12


def test_plan_only_writes_the_plan_without_any_llm_call(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    arms = StubArms()
    assert run_main(tmp_path, settings, ["--split", "test", "--plan-only", "--run-id", "test-plan"], arms) == 0
    plan = json.loads((tmp_path / "runs" / "test-plan" / "episodes_plan.json").read_text(encoding="utf-8"))
    assert plan["n_episodes"] == 158 and arms.calls == []
    assert "by language:" in capsys.readouterr().out
    assert runner.test_plan_size(tmp_path / "runs") == (158, "test-plan")


# --- pilot ------------------------------------------------------------------------------------


def test_pilot_uses_three_dev_episodes_per_dev_site() -> None:
    splits = load_splits()
    episodes = runner.pilot_episodes(load_intents(), splits, day_builder)
    dev_sites = {e["site_id"] for e in splits["dev"]}
    assert len(episodes) == 3 * len(dev_sites)
    for site_id in dev_sites:
        mine = [e for e in episodes if e.site.site_id == site_id]
        assert len(mine) == 3 and len({e.goal.template_id for e in mine}) == 3 and len({e.day.date for e in mine}) == 3
    assert not any(e.site.held_out or e.goal.held_out for e in episodes)


def test_pilot_run_summarizes_and_projects(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    arms = StubArms()
    assert run_main(tmp_path, settings, ["--pilot", "--run-id", "pilot-t"], arms, splits=load_splits()) == 0
    assert {arm for arm, _ in arms.calls} == {"A2", "A3", "A4"} and len(arms.calls) == 18
    summary = json.loads((tmp_path / "runs" / "pilot-t" / "pilot_summary.json").read_text(encoding="utf-8"))
    expected = (600 * 1.25 + 400 * 0.15 + 200 * 4.25) / 1e6  # token-priced, cache-aware
    assert summary["arms"]["A4"]["mean_token_usd"] == pytest.approx(expected)
    assert summary["test_episodes"] == runner.NOMINAL_TEST_EPISODES
    assert summary["projected_usd"] == pytest.approx(3 * expected * runner.NOMINAL_TEST_EPISODES)
    out = capsys.readouterr().out
    assert "PILOT (dev split only)" in out and "Projected test cost" in out and "allowed after the $30 reserve" in out


# --- E2: the hard goal set (SPEC §6.15) ---------------------------------------------------------------


def filled_hard_intents():
    from voltproof.bench.intents_hard import load_hard_intents

    intents = load_hard_intents()
    goals = [g.model_copy(update={lang: f"{g.en} [{lang}]" for lang in LANGUAGES}) for g in intents.multilingual]
    return intents.model_copy(update={"multilingual": goals})


def write_regression(path: Path, passed: bool = True, fingerprint: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"passed": passed, "code_fingerprint": fingerprint or runner.code_fingerprint()}), encoding="utf-8")
    return path


def test_hard_plan_only_works_with_pending_translations(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    arms = StubArms()
    assert run_main(tmp_path, settings, ["--split", "test", "--intents", "hard", "--plan-only", "--run-id", "e2-plan"], arms) == 0
    plan = json.loads((tmp_path / "runs" / "e2-plan" / "episodes_plan.json").read_text(encoding="utf-8"))
    assert plan["split"] == "test-hard" and plan["n_episodes"] == 56 + 45 and arms.calls == []
    assert plan["counts"]["set"] == {"english": 56, "multilingual": 45}
    assert sum(e["goal"] == "TRANSLATION_PENDING" for e in plan["episodes"]) == 36
    config = json.loads((tmp_path / "runs" / "e2-plan" / "config.json").read_text(encoding="utf-8"))
    assert config["args"]["intents"] == "hard" and len(config["intents_hard_sha256"]) == 64


def test_hard_run_refuses_pending_translations_and_a_missing_or_stale_regression_check(tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--split", "test", "--intents", "hard", "--arms", "A4", "--limit", "2", "--force"]
    arms = StubArms()
    assert run_main(tmp_path, settings, argv, arms) == runner.EXIT_REFUSED and "TRANSLATION_PENDING" in capsys.readouterr().out
    hard = filled_hard_intents()
    missing = tmp_path / "none.json"
    assert run_main(tmp_path, settings, argv, arms, hard_intents=hard, regression_path=missing) == runner.EXIT_REFUSED
    assert "no regression check" in capsys.readouterr().out
    stale = write_regression(tmp_path / "stale.json", fingerprint="0" * 64)
    assert run_main(tmp_path, settings, argv, arms, hard_intents=hard, regression_path=stale) == runner.EXIT_REFUSED
    failed = write_regression(tmp_path / "failed.json", passed=False)
    assert run_main(tmp_path, settings, argv, arms, hard_intents=hard, regression_path=failed) == runner.EXIT_REFUSED
    assert arms.calls == []
    good = write_regression(tmp_path / "good.json")
    assert run_main(tmp_path, settings, argv, arms, hard_intents=hard, regression_path=good) == runner.EXIT_OK and len(arms.calls) == 2
    assert all(eid.split("_", 3)[-1].startswith(("hard_", "ml_hard_")) or "_hard_" in eid for _, eid in arms.calls)
    assert any(p.name.startswith("e2-") for p in (tmp_path / "runs").iterdir())


def test_hard_intents_need_the_test_split() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--intents", "hard", "--pilot"])
