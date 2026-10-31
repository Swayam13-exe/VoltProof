"""Recorded A4 runs turned into precomputed replay files (SPEC §6.13): `web/replays/*.json`.

Each example chip is one recorded run of the frozen A4 (frozen test run or E2): finalized, gold met,
spec exactly matching gold, reviewer approved, no repairs. Building a replay:
1. rebuild the run's episodes (they must hash to the run's plan) and read the A4 trace and row
2. recover the accepted plan by replaying the trace's set_checklist / solve_plan calls on a fresh
   PlanBook (every replayed tool output must equal the trace exactly)
3. check that the recovered plan reproduces the benchmark row (realized_cost within 1e-9,
   gold_satisfied, unserved_critical_kwh within 1e-9) through the benchmark's own evaluate
4. build the page view with `view_model` (no LLM, no weather files needed afterwards)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from voltproof.api.view import view_model
from voltproof.bench.counterfactual import read_trace, recover_a4, rows_by_key, run_episodes
from voltproof.bench.episode import Episode, evaluate
from voltproof.config import PROJECT_ROOT
from voltproof.core.metrics import spec_accuracy

REPLAY_DIR = PROJECT_ROOT / "web" / "replays"
RUNS_DIR = PROJECT_ROOT / "runs"
MAIN_RUN = "test-20261030-181315"
HARD_RUN = "e2-20261031-130234"
REPRODUCE_TOL = 1e-9


@dataclass(frozen=True)
class Chip:
    chip_id: str
    run: str
    episode_id: str
    kind: str  # english | hard | multilingual


CHIPS: tuple[Chip, ...] = (
    Chip("ahmedabad-outage-en", MAIN_RUN, "ahmedabad_cold_store_2025-01-12_outage_critical_p1", "english"),
    Chip("ahmedabad-three-part-en", HARD_RUN, "ahmedabad_cold_store_2025-01-12_hard_compound3_p1", "hard"),
    Chip("ahmedabad-outage-hi", MAIN_RUN, "ahmedabad_cold_store_2025-12-17_ml_outage_hi", "multilingual"),
    Chip("ahmedabad-tonight-es", HARD_RUN, "ahmedabad_cold_store_2025-01-12_ml_hard_tonight_es", "multilingual"),
    Chip("nairobi-outage-reserve-en", MAIN_RUN, "nairobi_clinic_2025-11-09_outage_then_reserve_p0", "english"),
    Chip("nairobi-noon-midnight-en", HARD_RUN, "nairobi_clinic_2025-03-28_hard_noon_midnight_p0", "hard"),
    Chip("nairobi-storm-full-pt", MAIN_RUN, "nairobi_clinic_2025-11-09_ml_storm_full_pt", "multilingual"),
    Chip("nairobi-tonight-de", HARD_RUN, "nairobi_clinic_2025-03-28_ml_hard_tonight_de", "multilingual"),
    Chip("munich-peak-only-en", MAIN_RUN, "munich_home_2025-02-19_peak_only_discharge_p0", "english"),
    Chip("munich-distractor-en", HARD_RUN, "munich_home_2025-02-19_hard_distractor_p0", "hard"),
    Chip("munich-import-cap-es", MAIN_RUN, "munich_home_2025-02-19_ml_import_cap_es", "multilingual"),
    Chip("munich-relative-time-hi", HARD_RUN, "munich_home_2025-08-30_ml_hard_relative_time_hi", "multilingual"),
)


class ReplayCheckFailed(ValueError):
    """A replay does not reproduce the benchmark row it was recorded from."""


def check_reproduces(episode: Episode, run: Any, row: Mapping[str, Any]) -> None:
    """The recovered plan, evaluated exactly as the benchmark evaluated it, must give the stored row."""
    result = evaluate(episode, run)
    problems = []
    if abs(result.realized_cost - row["realized_cost"]) > REPRODUCE_TOL:
        problems.append(f"realized_cost {result.realized_cost} vs {row['realized_cost']}")
    if result.gold_satisfied != row["gold_satisfied"]:
        problems.append(f"gold_satisfied {result.gold_satisfied} vs {row['gold_satisfied']}")
    if abs(result.unserved_critical_kwh - row["unserved_critical_kwh"]) > REPRODUCE_TOL:
        problems.append(f"unserved_critical_kwh {result.unserved_critical_kwh} vs {row['unserved_critical_kwh']}")
    if problems:
        raise ReplayCheckFailed(f"{episode.episode_id}: " + "; ".join(problems))


def replay_meta(run_dir: Path, row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.get("settings") or {}
    return {
        "source_run": run_dir.name,
        "episode_id": row["episode_id"],
        "model": settings.get("llm_model"),
        "reasoning_effort": settings.get("llm_reasoning_effort"),
        "llm_cost_usd": row.get("cost_usd"),
        "latency_s": row.get("latency_s"),
        "tokens_in": row.get("tokens_in"),
        "tokens_out": row.get("tokens_out"),
        "plan_source": row.get("plan_source"),
        "benchmark_gold_satisfied": row.get("gold_satisfied"),
        "benchmark_realized_cost": row.get("realized_cost"),
    }


def build_replay(episode: Episode, run_dir: Path, row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    events = read_trace(run_dir, "A4", episode.episode_id)
    plan = recover_a4(episode, events)
    if plan.battery_kw is None or plan.spec is None:
        raise ReplayCheckFailed(f"{episode.episode_id}: the recorded run has no accepted plan")
    check_reproduces(episode, plan.arm_run(), row)
    matches_gold = spec_accuracy(plan.spec, episode.goal.gold) == (1.0, 1.0)
    view = view_model(
        kind="replay",
        site=episode.site,
        day=episode.day,
        tariff=episode.tariff,
        goal=episode.goal.text,
        language=episode.goal.language,
        events=events,
        logged_battery_kw=plan.battery_kw,
        spec=plan.spec,
        meta=replay_meta(run_dir, row, config),
        spec_matches_gold=matches_gold,
    )
    if matches_gold and plan.spec.grid_outages == episode.goal.gold.grid_outages:
        shown = view["same_day"]["agent"]["realized_cost"]
        if abs(shown - round(row["realized_cost"], 2)) > 0.005 + 1e-9:
            raise ReplayCheckFailed(f"{episode.episode_id}: the page would show {shown}, the benchmark has {row['realized_cost']}")
    return view


def index_entry(chip: Chip, view: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": chip.chip_id,
        "file": f"{chip.chip_id}.json",
        "site_id": view["site"]["site_id"],
        "site_name": view["site"]["name"],
        "kind": chip.kind,
        "language": view["goal"]["language"],
        "goal": view["goal"]["text"],
        "source_run": chip.run,
        "episode_id": chip.episode_id,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def build_all(chips: Sequence[Chip] = CHIPS, runs_dir: Path = RUNS_DIR, out_dir: Path = REPLAY_DIR) -> list[dict[str, Any]]:
    """Build and write every chip's replay and the index; returns the index entries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for run in dict.fromkeys(c.run for c in chips):
        run_dir = runs_dir / run
        episodes = {e.episode_id: e for e in run_episodes(run_dir)}
        rows = rows_by_key(run_dir)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for chip in (c for c in chips if c.run == run):
            view = build_replay(episodes[chip.episode_id], run_dir, rows[(chip.episode_id, "A4")], config)
            (out_dir / f"{chip.chip_id}.json").write_text(dumps(view), encoding="utf-8")
            entries.append(index_entry(chip, view))
    order = {c.chip_id: i for i, c in enumerate(chips)}
    entries.sort(key=lambda e: order[e["id"]])
    (out_dir / "index.json").write_text(dumps({"schema_version": 1, "chips": entries}), encoding="utf-8")
    return entries
