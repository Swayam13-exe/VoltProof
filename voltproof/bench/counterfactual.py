"""Plan recovery from a finished run, the regression check, and the E1 outage counterfactual (SPEC §6.15).

A run stores metrics, not setpoints, so each arm's executed plan is recovered:
- A0, A1, A5, A6: recomputed (deterministic), exactly as `episode.baseline_results` does
- A2: the one LLM reply in the trace, parsed with `parse_direct`, declarations mapped with
  `controller_limits`; a parse failure (or no reply) is the A0 fallback
- A3: every simulate_plan / accepted finalize call replayed on a fresh SimBook, then run_a3's
  selection: the last submitted schedule if physics-valid with its declared export limit, else the
  cheapest valid simulation, else A0
- A4: set_checklist / solve_plan replayed in trace order on a fresh PlanBook; every replayed output
  must equal the traced tool result exactly (the re-solve reproduces the traced plan); the plan is
  the one the trace's selection event names, else A0

Regression check: every row of the run re-evaluated through the frozen code paths
(`baseline_results`, `evaluate`) must reproduce realized_cost, realized_total_cost (|Δ| ≤ 1e-9) and
gold_satisfied. E1: the outage-template episodes re-executed with `outage_override=False`.

PowerShell, from the repo root:
    python -m voltproof.bench.counterfactual regression
    python -m voltproof.bench.counterfactual e1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from voltproof.agents.declarations import controller_limits
from voltproof.agents.llm_direct import parse_direct
from voltproof.agents.loop import LLMTotals, Trace
from voltproof.agents.tools import (
    ArmRun,
    ChecklistArgs,
    PlanBook,
    ScheduleFinalizeArgs,
    SimBook,
    SimulateArgs,
    SolvePlanArgs,
    simulate_schedule,
)
from voltproof.baselines.optimizer_arms import a1_plan, a5_plan, a6_plan
from voltproof.baselines.rule_based import a0_setpoints
from voltproof.bench.episode import Episode, baseline_results, build_test_episodes, evaluate
from voltproof.bench.intents import load_intents
from voltproof.bench.runner import ALL_ARMS, RUNS_DIR, code_fingerprint, plan_document, read_jsonl
from voltproof.config import PROJECT_ROOT
from voltproof.core.executor import execute
from voltproof.core.metrics import gold_check
from voltproof.data import GLOBAL_SEED
from voltproof.data.days import load_splits
from voltproof.data.sites import SITES
from voltproof.models import TOL, ExecutionResult, IntentSpec

FROZEN_RUN = RUNS_DIR / "test-20261030-181315"
RESULTS_DIR = PROJECT_ROOT / "results"
GATE_TOL = 1e-9
OUTAGE_TEMPLATES = ("outage_critical", "outage_then_reserve")
LLM_ARMS = ("A2", "A3", "A4")

Event = Mapping[str, Any]


class RecoveryError(ValueError):
    """The trace cannot be replayed into the plan the run executed."""


@dataclass(frozen=True)
class RecoveredPlan:
    """What one arm executed. `battery_kw=None` means the A0 fallback."""

    arm: str
    battery_kw: list[float] | None
    plan_source: str
    plan_basis: str = "forecast"
    export_limit_kw: list[float] | None = None
    import_limit_kw: list[float] | None = None
    protected_soc_kwh: list[float] | None = None
    spec: IntentSpec | None = None
    replayed_calls: int = 0  # A4: solve_plan / set_checklist outputs that matched the trace

    def arm_run(self) -> ArmRun:
        """The plan as the harness's ArmRun, so `episode.evaluate` scores it exactly as the run did."""
        return ArmRun(
            arm=self.arm,
            battery_kw=self.battery_kw,
            export_limit_kw=self.export_limit_kw,
            protected_soc_kwh=self.protected_soc_kwh,
            plan_source=self.plan_source,
            fallback_used=self.battery_kw is None,
            stop_reason="recovered",
            explanation=None,
            trace=Trace(),
            totals=LLMTotals(),
            tool_calls=0,
            repair_turns=0,
            spec=self.spec,
            import_limit_kw=self.import_limit_kw,
        )


# --- traces --------------------------------------------------------------------------------------


def trace_path(run_dir: Path, arm: str, episode_id: str) -> Path:
    return run_dir / "traces" / f"{arm}__{episode_id}.jsonl"


def read_trace(run_dir: Path, arm: str, episode_id: str) -> list[dict[str, Any]]:
    path = trace_path(run_dir, arm, episode_id)
    if not path.is_file():
        raise RecoveryError(f"no trace {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tool_calls(events: Sequence[Event]) -> list[tuple[str, Any, Any]]:
    """(name, args, result) of every tool call, results paired by call id in order."""
    pairs: list[tuple[str, Any, Any]] = []
    pending: list[tuple[str, str | None, Any]] = []
    for event in events:
        if event["kind"] == "tool_call":
            pending.append((event["name"], event.get("call_id"), event["data"].get("args")))
        elif event["kind"] == "tool_result":
            index = next((i for i, (name, call_id, _) in enumerate(pending) if name == event["name"] and call_id == event.get("call_id")), None)
            if index is None:
                raise RecoveryError(f"tool result {event['name']} without its call")
            name, _, args = pending.pop(index)
            pairs.append((name, args, event["data"].get("result")))
    return pairs


def selection_event(events: Sequence[Event]) -> Event:
    found = [e for e in events if e["kind"] == "final" and e["name"] == "selection"]
    if not found:
        raise RecoveryError("trace has no selection event")
    return found[-1]


def _valid_args(model: type[BaseModel], args: Any) -> BaseModel | None:
    if not isinstance(args, dict):
        return None
    try:
        return model.model_validate(args)
    except ValidationError:
        return None


def _normalized(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


# --- recovery per arm ----------------------------------------------------------------------------


def recover_a2(episode: Episode, events: Sequence[Event]) -> RecoveredPlan:
    calls = [e for e in events if e["kind"] == "llm_call"]
    if not calls:
        return RecoveredPlan("A2", None, "fallback")
    plan, _ = parse_direct(calls[0]["data"].get("content"))
    if plan is None:
        return RecoveredPlan("A2", None, "fallback")
    limits = controller_limits(plan.battery_kw, plan.no_export_hours, plan.reserve, episode.day, episode.site, plan.import_caps)
    return RecoveredPlan(
        "A2",
        list(plan.battery_kw),
        "finalized",
        export_limit_kw=limits.export_limit_kw,
        import_limit_kw=limits.import_limit_kw,
        protected_soc_kwh=limits.protected_soc_kwh,
    )


def recover_a3(episode: Episode, events: Sequence[Event]) -> RecoveredPlan:
    ctx = episode.context()
    sims = SimBook(ctx)
    submitted: list[ScheduleFinalizeArgs] = []
    for name, args, result in tool_calls(events):
        has_result = isinstance(result, dict) and "error" not in result
        if name == "simulate_plan" and has_result:
            parsed = _valid_args(SimulateArgs, args)
            if parsed is None:
                raise RecoveryError("a traced simulation has arguments the tool would have refused")
            replayed = sims.simulate(parsed)
            if replayed["sim_id"] != result.get("sim_id"):
                raise RecoveryError(f"simulation id {replayed['sim_id']} replayed as {result.get('sim_id')}")
        elif name == "finalize" and has_result and result.get("accepted"):
            parsed = _valid_args(ScheduleFinalizeArgs, args)
            if parsed is None:
                raise RecoveryError("an accepted finalize has arguments the tool would have refused")
            submitted.append(parsed)
    if submitted:
        final = submitted[-1]
        limits = controller_limits(final.battery_kw, final.no_export_hours, final.reserve, episode.day, episode.site, final.import_caps)
        if simulate_schedule(final.battery_kw, ctx, limits.export_limit_kw).valid:
            return RecoveredPlan(
                "A3",
                list(final.battery_kw),
                "finalized",
                export_limit_kw=limits.export_limit_kw,
                import_limit_kw=limits.import_limit_kw,
                protected_soc_kwh=limits.protected_soc_kwh,
            )
    best = sims.cheapest_valid()
    if best is not None:
        return RecoveredPlan("A3", list(best.battery_kw), "log_physics_valid")
    return RecoveredPlan("A3", None, "fallback")


def recover_a4(episode: Episode, events: Sequence[Event]) -> RecoveredPlan:
    book = PlanBook(episode.context())
    replayed = 0
    for name, args, result in tool_calls(events):
        if name not in ("set_checklist", "solve_plan"):
            continue
        model = ChecklistArgs if name == "set_checklist" else SolvePlanArgs
        parsed = _valid_args(model, args)
        if parsed is None:  # the loop refused the arguments before the tool ran
            continue
        try:
            output = book.set_checklist(parsed) if name == "set_checklist" else book.solve(parsed)
        except Exception as exc:  # the loop returns tool exceptions to the model the same way
            output = {"error": f"{type(exc).__name__}: {exc}"}
        if _normalized(output) != _normalized(result):
            raise RecoveryError(f"replayed {name} differs from the trace")
        replayed += 1
    selection = selection_event(events)
    plan_id = selection["data"].get("plan_id")
    source = str(selection.get("summary") or "").removeprefix("plan source: ")
    last = book.last()
    if plan_id is None:
        return RecoveredPlan("A4", None, source, spec=last.spec if last else None, replayed_calls=replayed)
    record = book.records.get(plan_id)
    if record is None:
        raise RecoveryError(f"selected plan {plan_id} was not replayed")
    plan = record.plan
    return RecoveredPlan(
        "A4",
        list(plan.battery_kw),
        source,
        export_limit_kw=plan.export_limit_kw,
        import_limit_kw=plan.import_limit_kw,
        protected_soc_kwh=plan.protected_soc_kwh,
        spec=record.spec,
        replayed_calls=replayed,
    )


RECOVER: Mapping[str, Callable[[Episode, Sequence[Event]], RecoveredPlan]] = {"A2": recover_a2, "A3": recover_a3, "A4": recover_a4}


def baseline_plan(arm: str, episode: Episode) -> RecoveredPlan:
    site, day, gold = episode.site, episode.day, episode.goal.gold
    if arm == "A0":
        return RecoveredPlan("A0", [float(b) for b in a0_setpoints(day, site, "actual")], "baseline", plan_basis="actual")
    plan = {"A1": lambda: a1_plan(day, site), "A5": lambda: a5_plan(day, site, gold), "A6": lambda: a6_plan(day, site, gold)}[arm]()
    return RecoveredPlan(
        arm,
        list(plan.battery_kw),
        "baseline",
        plan_basis=plan.use,
        export_limit_kw=plan.export_limit_kw,
        import_limit_kw=plan.import_limit_kw,
        protected_soc_kwh=plan.protected_soc_kwh,
        spec=plan.spec,
    )


def recover(arm: str, episode: Episode, run_dir: Path) -> RecoveredPlan:
    if arm in RECOVER:
        return RECOVER[arm](episode, read_trace(run_dir, arm, episode.episode_id))
    return baseline_plan(arm, episode)


def execute_recovered(plan: RecoveredPlan, episode: Episode, *, outage_override: bool = True) -> ExecutionResult:
    """Execute a recovered plan with the same arguments the run used; a fallback runs A0."""
    site, day, gold = episode.site, episode.day, episode.goal.gold
    if plan.battery_kw is None or plan.arm == "A0":
        return execute(a0_setpoints(day, site, "actual"), day, site, gold.grid_outages, outage_override=outage_override)
    return execute(
        plan.battery_kw,
        day,
        site,
        gold.grid_outages,
        export_limit_kw=plan.export_limit_kw,
        plan_basis=plan.plan_basis,  # type: ignore[arg-type]
        protected_soc_kwh=plan.protected_soc_kwh,
        import_limit_kw=plan.import_limit_kw,
        outage_override=outage_override,
    )


# --- the run's episodes --------------------------------------------------------------------------


def run_episodes(run_dir: Path) -> list[Episode]:
    """The run's test episodes, rebuilt; refuses unless they hash to the run's plan."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    episodes = build_test_episodes(load_intents(), load_splits(), SITES, seed=config["args"].get("seed", GLOBAL_SEED))
    digest = plan_document(episodes, "test", config["args"].get("seed", GLOBAL_SEED))["plan_sha256"]
    if digest != config["plan_sha256"]:
        raise RecoveryError(f"rebuilt episodes hash to {digest[:12]}, the run's plan is {config['plan_sha256'][:12]}")
    return episodes


def rows_by_key(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["episode_id"], r["arm"]): r for r in read_jsonl(run_dir / "episodes.jsonl")}


# --- regression check ----------------------------------------------------------------------------

GATE_FIELDS = ("realized_cost", "realized_total_cost", "gold_satisfied")
INFO_FIELDS = ("plan_physics_valid", "unserved_critical_kwh", "spec_precision", "spec_recall", "plan_source")


def _differs(field: str, stored: Any, recomputed: Any) -> float | bool:
    """|Δ| for numbers (inf when only one side is None), inequality for the rest."""
    if isinstance(stored, bool) or isinstance(recomputed, bool) or isinstance(stored, str) or isinstance(recomputed, str):
        return stored != recomputed
    if stored is None or recomputed is None:
        return 0.0 if stored is recomputed else float("inf")
    return abs(float(stored) - float(recomputed))


def regression_check(run_dir: Path, episodes: Sequence[Episode] | None = None, progress: bool = True) -> dict[str, Any]:
    episodes = list(episodes) if episodes is not None else run_episodes(run_dir)
    stored = rows_by_key(run_dir)
    max_abs = {f: 0.0 for f in ("realized_cost", "realized_total_cost", "unserved_critical_kwh", "spec_precision", "spec_recall")}
    gate_mismatches: list[dict[str, Any]] = []
    info_mismatches: list[dict[str, Any]] = []
    recovery_errors: list[dict[str, str]] = []
    checked, replayed = 0, 0
    for episode in tqdm(episodes, desc="regression", unit="episode", disable=not progress):
        recomputed = {r.arm: r.model_dump(mode="json") for r in baseline_results(episode)}
        for arm in LLM_ARMS:
            try:
                plan = recover(arm, episode, run_dir)
            except RecoveryError as exc:
                recovery_errors.append({"episode_id": episode.episode_id, "arm": arm, "error": str(exc)})
                continue
            replayed += plan.replayed_calls
            recomputed[arm] = evaluate(episode, plan.arm_run()).model_dump(mode="json")
        for arm in ALL_ARMS:
            row, new = stored.get((episode.episode_id, arm)), recomputed.get(arm)
            if row is None or new is None:
                continue
            checked += 1
            for field in GATE_FIELDS + INFO_FIELDS:
                delta = _differs(field, row.get(field), new.get(field))
                if field in max_abs and isinstance(delta, float):
                    max_abs[field] = max(max_abs[field], delta)
                bad = delta > GATE_TOL if isinstance(delta, float) else bool(delta)
                if bad:
                    entry = {"episode_id": episode.episode_id, "arm": arm, "field": field, "stored": row.get(field), "recomputed": new.get(field)}
                    (gate_mismatches if field in GATE_FIELDS else info_mismatches).append(entry)
    missing = len(stored) - checked
    passed = not gate_mismatches and not recovery_errors and missing == 0
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "code_fingerprint": code_fingerprint(),
        "tolerance": GATE_TOL,
        "rows_stored": len(stored),
        "rows_checked": checked,
        "rows_not_checked": missing,
        "passed": passed,
        "gate_fields": list(GATE_FIELDS),
        "max_abs_diff": max_abs,
        "gate_mismatches": gate_mismatches,
        "info_mismatches": info_mismatches,
        "recovery_errors": recovery_errors,
        "a4_replayed_tool_outputs_matching_trace": replayed,
    }


# --- E1 --------------------------------------------------------------------------------------------

E1_CAPTION = (
    "E1 counterfactual: only the executor's deficit-covering outage override is removed; in outage hours "
    "the battery follows its clipped setpoint and still stores surplus PV within its limits (the no-wasted-solar "
    "rule every arm gets outside outages), and any deficit goes unserved, non-critical load first. Planned "
    "charging above the PV surplus during an outage is a deficit, as in the verifier. This is a conservative "
    "estimate of what the plans alone would have lost."
)


def e1_rows(run_dir: Path, episodes: Sequence[Episode] | None = None, progress: bool = True) -> list[dict[str, Any]]:
    """One row per outage-template episode and arm: unserved energy with and without the override."""
    episodes = [e for e in (episodes if episodes is not None else run_episodes(run_dir)) if e.goal.template_id in OUTAGE_TEMPLATES]
    stored = rows_by_key(run_dir)
    rows = []
    for episode in tqdm(episodes, desc="E1", unit="episode", disable=not progress):
        for arm in ALL_ARMS:
            frozen = stored.get((episode.episode_id, arm))
            if frozen is None:
                continue
            plan = recover(arm, episode, run_dir)
            with_override = execute_recovered(plan, episode)
            if abs(with_override.unserved_critical_kwh - frozen["unserved_critical_kwh"]) > GATE_TOL:
                raise RecoveryError(f"{arm} {episode.episode_id}: the recovered plan does not reproduce the run's unserved critical load")
            without = execute_recovered(plan, episode, outage_override=False)
            satisfied, _ = gold_check(without, episode.goal.gold, episode.site)
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "site_id": episode.site.site_id,
                    "template_id": episode.goal.template_id,
                    "language": episode.goal.language,
                    "arm": arm,
                    "plan_source": frozen.get("plan_source"),
                    "plan_physics_valid": frozen["plan_physics_valid"],
                    "overrides": frozen["overrides"],
                    "unserved_critical_kwh": with_override.unserved_critical_kwh,
                    "unserved_noncritical_kwh": with_override.unserved_noncritical_kwh,
                    "gold_satisfied": frozen["gold_satisfied"],
                    "cf_unserved_critical_kwh": without.unserved_critical_kwh,
                    "cf_unserved_noncritical_kwh": without.unserved_noncritical_kwh,
                    "cf_gold_satisfied": satisfied,
                }
            )
    return rows


def e1_summary(rows: Iterable[Mapping[str, Any]], group: Sequence[str] = ("arm",)) -> list[dict[str, Any]]:
    """Per group: episodes, unserved kWh with / without the override, episodes with any critical loss."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[g] for g in group), []).append(row)
    order = {arm: i for i, arm in enumerate(ALL_ARMS)}
    out = []
    for key in sorted(groups, key=lambda k: tuple(order[v] if g == "arm" else v for g, v in zip(group, k))):
        mine = groups[key]
        out.append(
            {
                **dict(zip(group, key)),
                "episodes": len(mine),
                "unserved_critical_kwh": sum(r["unserved_critical_kwh"] for r in mine),
                "episodes_any_critical_loss": sum(r["unserved_critical_kwh"] > TOL for r in mine),
                "unserved_noncritical_kwh": sum(r["unserved_noncritical_kwh"] for r in mine),
                "cf_unserved_critical_kwh": sum(r["cf_unserved_critical_kwh"] for r in mine),
                "cf_episodes_any_critical_loss": sum(r["cf_unserved_critical_kwh"] > TOL for r in mine),
                "cf_unserved_noncritical_kwh": sum(r["cf_unserved_noncritical_kwh"] for r in mine),
                "cf_gold_satisfied_pct": 100.0 * sum(bool(r["cf_gold_satisfied"]) for r in mine) / len(mine),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        if not rows:
            return
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_e1(rows: Sequence[Mapping[str, Any]], out_dir: Path, run_id: str) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "caption": E1_CAPTION,
        "templates": list(OUTAGE_TEMPLATES),
        "per_arm": e1_summary(rows),
        "per_site_arm": e1_summary(rows, ("site_id", "arm")),
    }
    write_csv(out_dir / "e1_episodes.csv", rows)
    write_csv(out_dir / "e1_per_arm.csv", summary["per_arm"])
    write_csv(out_dir / "e1_per_site_arm.csv", summary["per_site_arm"])
    (out_dir / "e1_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


# --- CLI -------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voltproof.bench.counterfactual", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("regression", "e1"))
    parser.add_argument("--run", type=Path, default=FROZEN_RUN, help="finished run directory (read only)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.command == "regression":
        report = regression_check(args.run)
        path = args.out / "test-run" / "regression_check.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
        print(
            f"regression {'PASSED' if report['passed'] else 'FAILED'}: {report['rows_checked']}/{report['rows_stored']} rows, "
            f"{len(report['gate_mismatches'])} gate mismatches, {len(report['recovery_errors'])} recovery errors, "
            f"{len(report['info_mismatches'])} info mismatches; max abs diff {report['max_abs_diff']}; "
            f"A4 replayed tool outputs matching the trace: {report['a4_replayed_tool_outputs_matching_trace']}\n-> {path}"
        )
        return 0 if report["passed"] else 1
    summary = write_e1(e1_rows(args.run), args.out / "e1", args.run.name)
    for row in summary["per_arm"]:
        print(
            f"{row['arm']}: {row['episodes']} ep | override: crit {row['unserved_critical_kwh']:.3f} kWh ({row['episodes_any_critical_loss']} ep) | "
            f"no override: crit {row['cf_unserved_critical_kwh']:.3f} kWh ({row['cf_episodes_any_critical_loss']} ep), "
            f"non-crit {row['cf_unserved_noncritical_kwh']:.3f} kWh, gold {row['cf_gold_satisfied_pct']:.1f}%"
        )
    print(f"-> {args.out / 'e1'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
