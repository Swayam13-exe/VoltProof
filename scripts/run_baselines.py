"""Run the non-LLM arms A0, A1, A5, A6 on the DEV days and check the P2 gate (SPEC §8).

PowerShell, from the repo root:
    python scripts\\run_baselines.py

Dev days only (held-out sites are refused). Three hand-written gold specs per day. Every arm is
executed on actual data facing the spec's outages. Gate:
- no arm's realized_total_cost (A6's executed total included) is below the oracle LP bound
  (A6's LP objective on actual data, the true lower bound)
- on the outage spec, A5 serves all critical energy whenever A6 does (the outage is feasible)
- every optimizer solve takes < 1 s
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltproof.baselines.optimizer_arms import a1_plan, a5_plan, a6_plan  # noqa: E402
from voltproof.baselines.rule_based import a0_setpoints  # noqa: E402
from voltproof.core.executor import execute  # noqa: E402
from voltproof.core.metrics import episode_result, oracle_bound, oracle_gate, trajectory_total_cost  # noqa: E402
from voltproof.core.rules import penalty_scale  # noqa: E402
from voltproof.core.verifier import verify  # noqa: E402
from voltproof.data.days import build_day, dev_days, load_splits  # noqa: E402
from voltproof.data.sites import SITES  # noqa: E402
from voltproof.data.weather import processed_csv_path, read_processed_csv  # noqa: E402
from voltproof.models import DayData, EpisodeResult, ExecutionResult, IntentSpec, Site  # noqa: E402

SPECS = {
    "no_goal": IntentSpec(),
    "evening_reserve_50pct_18_22": IntentSpec.model_validate(
        {"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]}
    ),
    "outage_14_17": IntentSpec.model_validate({"grid_outages": [{"start_hour": 14, "end_hour": 17}]}),
}
ARMS = ("A0", "A1", "A5", "A6")
MAX_SOLVE_S = 1.0


def run_case(
    site: Site, day: DayData, spec_name: str, gold: IntentSpec
) -> tuple[list[EpisodeResult], dict[str, ExecutionResult], float, list[str]]:
    """All four arms on one site-day-spec: their results, executions, the oracle bound and any gate failures."""
    runs = {}
    a0_actual = a0_setpoints(day, site, "actual")
    a0_check = verify(a0_setpoints(day, site, "forecast"), day, site, gold, physics_only=True)
    runs["A0"] = (execute(a0_actual, day, site, gold.grid_outages), a0_check, None)  # no protected SOC: no top-up
    plans = {"A1": a1_plan(day, site), "A5": a5_plan(day, site, gold), "A6": a6_plan(day, site, gold)}
    for arm, plan in plans.items():
        check = verify(plan.battery_kw, day, site, gold, physics_only=True, export_limit_kw=plan.export_limit_kw)
        execution = execute(
            plan.battery_kw,
            day,
            site,
            gold.grid_outages,
            plan.export_limit_kw,
            plan_basis=plan.use,
            protected_soc_kwh=plan.protected_soc_kwh,
            import_limit_kw=plan.import_limit_kw,
        )
        runs[arm] = (execution, check, plan.solve_time_s)
    bound = oracle_bound(plans["A6"])

    costs = {arm: trajectory_total_cost(execution, gold, day, site) for arm, (execution, _, _) in runs.items()}
    results = [
        episode_result(
            arm=arm,
            spec_name=spec_name,
            day=day,
            site=site,
            gold_spec=gold,
            execution=execution,
            plan_check=check,
            a0_money=costs["A0"].net_money,
            a0_total=costs["A0"].total,
            oracle_total=bound,
            solve_time_s=solve_time,
        )
        for arm, (execution, check, solve_time) in runs.items()
    ]

    failures = []
    p_hat = penalty_scale(day.import_price_per_kwh)
    below_bound = oracle_gate([(arm, costs[arm].total) for arm in ARMS], bound, p_hat)
    if below_bound:
        failures.append(f"total cost below the oracle LP bound for {below_bound}")
    by_arm = {r.arm: r for r in results}
    if gold.grid_outages and by_arm["A6"].unserved_critical_kwh <= 1e-6 and by_arm["A5"].unserved_critical_kwh > 1e-6:
        failures.append(f"A5 left {by_arm['A5'].unserved_critical_kwh:.3f} kWh of critical load unserved on a feasible outage")
    slow = [r.arm for r in results if r.solve_time_s is not None and r.solve_time_s >= MAX_SOLVE_S]
    if slow:
        failures.append(f"solves >= {MAX_SOLVE_S} s: {slow}")
    return results, {arm: execution for arm, (execution, _, _) in runs.items()}, bound, failures


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "   n/a"
    return f"{0.0 if abs(value) < 0.05 else value:+6.1f}"  # no "-0.0" for numerical noise


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    sites = list(SITES.values())
    cases = dev_days(load_splits(), sites)
    frames = {}
    all_failures: list[str] = []
    max_solve = 0.0
    for site, day_date in cases:
        if site.site_id not in frames:
            frames[site.site_id] = read_processed_csv(processed_csv_path(site.site_id), site.timezone)
        day = build_day(site, day_date, frames[site.site_id])
        for spec_name, gold in SPECS.items():
            results, executions, bound, failures = run_case(site, day, spec_name, gold)
            print(f"\n{site.site_id}  {day_date}  {spec_name}   (money in {site.currency}; net of battery energy change)")
            print(f"  oracle LP bound (A6 objective on actual data + V*E0): {bound:.2f}")
            print("  arm   realized_cost   total_cost   savings%   gap%   unserved_crit_kWh  gold  overrides  topups  solve_ms")
            for r in results:
                solve = "      -" if r.solve_time_s is None else f"{r.solve_time_s * 1000:7.1f}"
                max_solve = max(max_solve, r.solve_time_s or 0.0)
                topups = len(executions[r.arm].goal_topup_hours)
                print(
                    f"  {r.arm:<4} {r.realized_cost:14.2f} {r.realized_total_cost:12.2f}   {_fmt_pct(r.savings_vs_rule_pct)}  "
                    f"{_fmt_pct(r.gap_to_oracle_pct)}   {r.unserved_critical_kwh:12.3f}       "
                    f"{'yes' if r.gold_satisfied else 'NO ':<4}  {r.overrides:5d}   {topups:5d}   {solve}"
                    + ("" if r.gold_satisfied else f"   violated: {','.join(r.gold_violations)}")
                )
            for failure in failures:
                print(f"  GATE FAIL: {failure}")
            all_failures.extend(f"{site.site_id} {day_date} {spec_name}: {f}" for f in failures)

    print(f"\n{len(cases)} dev days x {len(SPECS)} specs x {len(ARMS)} arms; slowest solve {max_solve * 1000:.1f} ms")
    if all_failures:
        print("GATE: FAIL")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1
    print("GATE: PASS (no arm below the oracle LP bound; A5 serves critical load on feasible outages; all solves < 1 s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
