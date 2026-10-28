"""Episode metrics (SPEC §6.7).

Two costs per episode, both in site currency and both net of battery energy change, i.e. plus
V·(E0 − e_23) with V = mean import price · η_d (SPEC §6.4), so an arm that ends the day with an
emptier battery does not look cheaper than one that keeps the energy for tomorrow:
- realized_cost:       money (grid + degradation) + battery energy change
- realized_total_cost: realized_cost + the optimizer's penalties on the EXECUTED trajectory:
                       1000p̂ per unserved critical kWh, 100p̂ per unserved non-critical kWh,
                       1e-4·p̂ per curtailed kWh, 20p̂ per kWh of gold-goal shortfall (SOC goals once
                       per goal at their worst hour + 0.2p̂ per kWh per hour, flow goals summed over
                       hours; throughput 5p̂)
Any executed trajectory is a feasible point of the oracle's LP (same physics, slacks = shortfalls),
so the oracle bound = A6's LP objective on actual data + V·E0 is a true lower bound on every arm's
realized_total_cost. The oracle gate and gap_to_oracle_pct use that bound (SPEC §6.7); A6's own
executed total is reported separately and can sit slightly above it when the executor's rules adjust
its setpoints. savings_vs_rule_pct uses realized_cost. Percentages are currency-free.

gold_satisfied checks the executor's REALIZED SOC and flows against the gold spec (never a
re-simulation of setpoints), plus: no unserved critical energy during gold outage windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from voltproof.core.rules import (
    CRITICAL_WEIGHT,
    CURTAIL_WEIGHT,
    NONCRITICAL_WEIGHT,
    Trajectory,
    goal_penalty,
    penalty_scale,
    physics_message,
    spec_violations,
    terminal_value_per_kwh,
)
from voltproof.core.spec import SpecItem, outage_hours, soft_rows, spec_items
from voltproof.models import TOL, DayData, EpisodeResult, ExecutionResult, IntentSpec, Plan, Site, VerifyResult, Violation

MIN_BASE_FRAC_OF_P_HAT = 0.01  # a base cost under 1 % of one kWh at the day's top price is "near zero"
PCT_TOLERANCE = 1.0
KW_TOLERANCE = 0.1
THROUGHPUT_ABS_TOLERANCE = 0.1
THROUGHPUT_REL_TOLERANCE = 0.01


@dataclass(frozen=True)
class TotalCost:
    money: float  # grid + degradation
    energy_change: float  # V·(E0 − e_23): positive when the day ends with less stored energy
    unserved_penalty: float
    curtail_penalty: float
    goal_penalty: float

    @property
    def net_money(self) -> float:
        """realized_cost: money net of battery energy change."""
        return self.money + self.energy_change

    @property
    def total(self) -> float:
        """realized_total_cost."""
        return self.net_money + self.unserved_penalty + self.curtail_penalty + self.goal_penalty


def executed_trajectory(execution: ExecutionResult) -> Trajectory:
    flows = execution.flows
    return Trajectory(
        initial_soc_kwh=execution.initial_soc_kwh,
        soc_kwh=np.asarray(execution.soc_kwh),
        charge_kw=np.asarray(flows.charge_kw),
        discharge_kw=np.asarray(flows.discharge_kw),
        grid_import_kw=np.asarray(flows.grid_import_kw),
        grid_export_kw=np.asarray(flows.grid_export_kw),
    )


def trajectory_total_cost(execution: ExecutionResult, gold_spec: IntentSpec, day: DayData, site: Site) -> TotalCost:
    p_hat = penalty_scale(day.import_price_per_kwh)
    flows = execution.flows
    value = terminal_value_per_kwh(day.import_price_per_kwh, site.battery.eta_discharge)
    return TotalCost(
        money=execution.realized_cost,
        energy_change=value * (execution.initial_soc_kwh - execution.soc_kwh[-1]),
        unserved_penalty=p_hat
        * (CRITICAL_WEIGHT * sum(flows.unserved_critical_kw) + NONCRITICAL_WEIGHT * sum(flows.unserved_noncritical_kw)),
        curtail_penalty=p_hat * CURTAIL_WEIGHT * sum(flows.curtail_kw),
        goal_penalty=goal_penalty(soft_rows(gold_spec, site), executed_trajectory(execution), p_hat),
    )


def oracle_bound(a6_plan: Plan) -> float:
    """The oracle's LP objective on actual data, in realized_total_cost units (+ the constant V·E0)."""
    if a6_plan.use != "actual":
        raise ValueError("the oracle bound needs the A6 plan solved on actual data")
    return a6_plan.objective + a6_plan.terminal_value_per_kwh * a6_plan.initial_soc_kwh


def gold_check(execution: ExecutionResult, gold_spec: IntentSpec, site: Site) -> tuple[bool, list[Violation]]:
    """Gold goals checked on the executed trajectory, plus critical load served during gold outages."""
    violations = spec_violations(soft_rows(gold_spec, site), executed_trajectory(execution))
    for t in sorted(outage_hours(gold_spec)):
        unserved = execution.flows.unserved_critical_kw[t]
        if unserved > TOL:
            violations.append(
                physics_message(
                    "OUTAGE_DEFICIT",
                    t,
                    unserved,
                    0.0,
                    "{v} kWh of critical load went unserved during the announced outage (allowed {limit} kWh)",
                    "kWh",
                )
            )
    return not violations, violations


def pct_of_base(difference: float, base: float, p_hat: float) -> float | None:
    """difference/|base|·100, or None when |base| is near zero (< 1 % of a kWh at p̂)."""
    if abs(base) < MIN_BASE_FRAC_OF_P_HAT * p_hat:
        return None
    return difference / abs(base) * 100.0


def critical_served_pct(execution: ExecutionResult, day: DayData) -> float:
    demand = float(sum(day.critical_load_kw))
    if demand <= TOL:
        return 100.0
    return 100.0 * (demand - execution.unserved_critical_kwh) / demand


def episode_result(
    *,
    arm: str,
    spec_name: str,
    day: DayData,
    site: Site,
    gold_spec: IntentSpec,
    execution: ExecutionResult,
    plan_check: VerifyResult,
    a0_money: float,
    a0_total: float,
    oracle_total: float,
    solve_time_s: float | None = None,
) -> EpisodeResult:
    """`oracle_total` is the oracle bound (`oracle_bound(a6_plan)`), not A6's executed total."""
    p_hat = penalty_scale(day.import_price_per_kwh)
    cost = trajectory_total_cost(execution, gold_spec, day, site)
    total = cost.total
    satisfied, violations = gold_check(execution, gold_spec, site)
    return EpisodeResult(
        site_id=site.site_id,
        date=day.date,
        arm=arm,
        spec_name=spec_name,
        currency=site.currency,
        realized_cost=cost.net_money,
        realized_total_cost=total,
        savings_vs_rule_pct=pct_of_base(a0_money - cost.net_money, a0_money, p_hat),
        gap_to_oracle_pct=pct_of_base(total - oracle_total, a0_total, p_hat),
        plan_physics_valid=plan_check.valid,
        gold_satisfied=satisfied,
        gold_violations=sorted({v.rule for v in violations}),
        unserved_critical_kwh=execution.unserved_critical_kwh,
        critical_served_pct=critical_served_pct(execution, day),
        overrides=execution.overrides,
        solve_time_s=solve_time_s,
    )


def spec_accuracy(agent: IntentSpec, gold: IntentSpec) -> tuple[float, float]:
    """(precision, recall) of agent spec items against gold items (SPEC §6.7).

    Items match on kind and exact hours; numbers within ±1 percentage point (SOC %), ±0.1 kW
    (import caps) or max(±0.1 kWh, ±1 %) (throughput). Each gold item matches at most once.
    """
    agent_items, gold_items = spec_items(agent), spec_items(gold)
    unmatched = list(gold_items)
    hits = 0
    for item in agent_items:
        match = next((g for g in unmatched if _items_match(item, g)), None)
        if match is not None:
            unmatched.remove(match)
            hits += 1
    precision = hits / len(agent_items) if agent_items else 1.0
    recall = hits / len(gold_items) if gold_items else 1.0
    return precision, recall


def _items_match(a: SpecItem, b: SpecItem) -> bool:
    if (a.kind, a.start, a.end) != (b.kind, b.start, b.end):
        return False
    if a.value is None or b.value is None:
        return a.value is None and b.value is None
    if a.kind == "import_cap":
        return abs(a.value - b.value) <= KW_TOLERANCE + 1e-9
    if a.kind == "max_throughput":
        return abs(a.value - b.value) <= max(THROUGHPUT_ABS_TOLERANCE, THROUGHPUT_REL_TOLERANCE * abs(b.value)) + 1e-9
    return abs(a.value - b.value) <= PCT_TOLERANCE + 1e-9


def oracle_gate(totals: Sequence[tuple[str, float]], oracle_total: float, p_hat: float) -> list[str]:
    """Arms whose total cost is below the oracle bound by more than numerical noise (should be empty;
    A6's own executed total is included in `totals`)."""
    return [arm for arm, total in totals if total < oracle_total - 1e-6 * max(1.0, p_hat)]
