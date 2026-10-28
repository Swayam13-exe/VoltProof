"""Penalty weights, message formatting and spec-rule checks on a trajectory (SPEC §6.4–6.7).

The same checks run on the verifier's simulated trajectory and on the executor's realized one,
and `row_shortfalls` measures exactly what the optimizer's slack variables measure.

Penalty weights (SPEC §6.4), all multiples of p̂ = the day's maximum import price, encode the
priority order critical load > non-critical load > goals > saving cycles > money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from voltproof.core.spec import INITIAL, SoftRow
from voltproof.models import TOL, RuleCode, Violation

FloatArray = npt.NDArray[np.float64]

CRITICAL_WEIGHT = 1000.0
NONCRITICAL_WEIGHT = 100.0
GOAL_WEIGHT = 20.0
HOURLY_SOC_WEIGHT = 0.2  # per kWh of SOC-goal shortfall per hour, so a missed goal still recovers quickly
THROUGHPUT_WEIGHT = 5.0
CURTAIL_WEIGHT = 1e-4


def penalty_scale(import_price_per_kwh: npt.ArrayLike) -> float:
    """p̂ = the day's maximum import price (1.0 if every price is zero, so penalties stay positive)."""
    p_hat = float(np.max(import_price_per_kwh))
    return p_hat if p_hat > 0 else 1.0


def terminal_value_per_kwh(import_price_per_kwh: npt.ArrayLike, eta_discharge: float) -> float:
    """V = mean import price · η_d: what one kWh left in the battery at midnight will save later (SPEC §6.4)."""
    return float(np.mean(import_price_per_kwh)) * eta_discharge


def row_weight(row: SoftRow) -> float:
    return THROUGHPUT_WEIGHT if row.rule == "THROUGHPUT" else GOAL_WEIGHT


def fmt_pair(value: float, limit: float, decimals: int = 6) -> tuple[str, str]:
    """Format value and limit with 6 decimals, adding decimals if they would print identically."""
    for places in range(decimals, 13):
        a, b = f"{value:.{places}f}", f"{limit:.{places}f}"
        if a != b or value == limit:
            return a, b
    return f"{value!r}", f"{limit!r}"


@dataclass(frozen=True)
class Trajectory:
    """Battery and grid trajectory of one day, from a simulation or a real execution."""

    initial_soc_kwh: float
    soc_kwh: FloatArray
    charge_kw: FloatArray
    discharge_kw: FloatArray
    grid_import_kw: FloatArray
    grid_export_kw: FloatArray


def row_value(row: SoftRow, trajectory: Trajectory) -> float:
    if row.kind == "soc_min":
        return trajectory.initial_soc_kwh if row.hour == INITIAL else float(trajectory.soc_kwh[row.hour])
    if row.kind == "import_max":
        return float(trajectory.grid_import_kw[row.hour])
    if row.kind == "export_max":
        return float(trajectory.grid_export_kw[row.hour])
    if row.kind == "discharge_max":
        return float(trajectory.discharge_kw[row.hour])
    return float(np.sum(trajectory.charge_kw) + np.sum(trajectory.discharge_kw))


def row_shortfall(row: SoftRow, trajectory: Trajectory) -> float:
    """How far the trajectory misses the row (≥ 0), in the row's kWh/kW unit: the LP slack."""
    value = row_value(row, trajectory)
    if row.kind == "soc_min":
        return max(0.0, row.bound - value)
    return max(0.0, value - row.bound)


def row_slack_keys(row: SoftRow) -> list[tuple[str, float]]:
    """The slacks a row is attached to, with their weights (multiples of p̂), SPEC §6.4:
    SOC goals: the goal's shared worst-hour slack (20) AND the row's own hourly slack (0.2);
    flow goals: one slack per hour (20); throughput: one slack (5)."""
    keys = [(row.slack_key, row_weight(row))]
    if row.kind == "soc_min":
        keys.append((f"{row.path}@{row.hour}", HOURLY_SOC_WEIGHT))
    return keys


def slack_values(rows: list[SoftRow], trajectory: Trajectory) -> dict[str, tuple[float, float]]:
    """(weight, value) of every slack for this trajectory: each slack takes the worst shortfall among
    the rows attached to it. Exactly what the optimizer's slack variables measure."""
    slacks: dict[str, tuple[float, float]] = {}
    for row in rows:
        shortfall = row_shortfall(row, trajectory)
        for key, weight in row_slack_keys(row):
            slacks[key] = (weight, max(slacks.get(key, (weight, 0.0))[1], shortfall))
    return slacks


def goal_penalty(rows: list[SoftRow], trajectory: Trajectory, p_hat: float) -> float:
    """Σ weight·p̂·slack: SOC goals 20p̂·worst-hour shortfall + 0.2p̂·Σ hourly shortfalls; flow goals
    20p̂ per kWh of excess, hour by hour; throughput 5p̂ per kWh."""
    return float(sum(weight * p_hat * value for weight, value in slack_values(rows, trajectory).values()))


def spec_violations(rows: list[SoftRow], trajectory: Trajectory) -> list[Violation]:
    """One Violation per soft row the trajectory misses by more than TOL."""
    violations = []
    for row in rows:
        if row_shortfall(row, trajectory) > TOL:
            value = row_value(row, trajectory)
            violations.append(
                Violation(hour=row.hour, rule=row.rule, value=value, limit=row.bound, message=_row_message(row, value))
            )
    return violations


def _row_message(row: SoftRow, value: float) -> str:
    v, limit = fmt_pair(value, row.bound)
    where = "initial state of charge" if row.hour == INITIAL else f"hour {row.hour}"
    if row.kind == "soc_min":
        return (
            f"{row.rule} at {where}: state of charge {v} kWh is below the required {limit} kWh "
            f"({row.path})"
        )
    if row.kind == "import_max":
        return f"{row.rule} at hour {row.hour}: grid import {v} kW exceeds the cap of {limit} kW ({row.path})"
    if row.kind == "export_max":
        return f"{row.rule} at hour {row.hour}: grid export {v} kW in a no-export window (limit {limit} kW, {row.path})"
    if row.kind == "discharge_max":
        return (
            f"{row.rule} at hour {row.hour}: battery discharge {v} kW outside the allowed discharge windows "
            f"(limit {limit} kW)"
        )
    return f"{row.rule}: battery throughput {v} kWh exceeds the cap of {limit} kWh ({row.path})"


def physics_message(rule: RuleCode, hour: int | None, value: float, limit: float, text: str, unit: str) -> Violation:
    """A Violation whose message states the rule, hour, value and limit with 6 decimals."""
    v, lim = fmt_pair(value, limit)
    where = f" at hour {hour}" if hour is not None else ""
    return Violation(hour=hour, rule=rule, value=value, limit=limit, message=f"{rule}{where}: {text.format(v=v, limit=lim, unit=unit)}")
