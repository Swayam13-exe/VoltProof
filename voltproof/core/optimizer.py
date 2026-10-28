"""Linear-programming battery planner (SPEC §6.4), solved with scipy's HiGHS.

Variables, block-major (`x[block * 24 + t]`), then one slack per soft-goal row:
    c  charge kW (AC)            [0, P_c]
    d  discharge kW (AC)         [0, P_d]
    gp grid import kW            [0, G_imp], fixed to 0 in spec outage hours
    gm grid export kW            [0, G_exp] (0 if export not allowed), fixed to 0 in outage hours
    k  PV curtailment kW         [0, PV_t]
    u  unserved non-critical kW  [0, L_t − Lc_t]
    v  unserved critical kW      [0, Lc_t]
    e  SOC kWh at end of hour    [E_min, E_max]
    s  one slack per slack key   [0, ∞)

Equalities: balance  −k + gp + d + u + v − c − gm = L − PV;
            SOC      e_t − e_{t−1} − η_c·c_t + d_t/η_d = 0  (e_{−1} = E0).
Soft rows come from `core.spec.soft_rows`; `core.rules.row_slack_keys` attaches them to slacks
(SPEC §6.4): each SOC goal has one shared slack (its worst-hour shortfall, e_τ + S ≥ R for every τ in
its window, 20p̂) plus one small slack per hour (e_τ + s_τ ≥ R, 0.2p̂) so a missed goal still recovers
quickly; each flow goal one slack per hour (20p̂); throughput one slack (5p̂).

Objective (site currency), with p̂ = the day's maximum import price:
    Σ_t [p⁺gp − p⁻gm + β(c + d) + 100p̂·u + 1000p̂·v + 1e-4·p̂·k] + Σ weight·p̂·slack
    − V·e_23, with V = mean_t(p⁺_t)·η_d: energy left at midnight is worth what it will save later,
      so draining the starting charge is not free (and flat prices leave the battery idle).

Degradation: β = degradation_cost_per_kwh, charged per kWh of AC throughput in BOTH directions,
so delivering 1 kWh through a full cycle costs β·(1 + 1/(η_c·η_d)).
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

from voltproof.core.rules import (
    CRITICAL_WEIGHT,
    CURTAIL_WEIGHT,
    NONCRITICAL_WEIGHT,
    Trajectory,
    penalty_scale,
    row_shortfall,
    row_slack_keys,
    terminal_value_per_kwh,
)
from voltproof.core.spec import INITIAL, SoftRow, no_export_hours, outage_hours, soft_rows
from voltproof.models import HOURS, TOL, DayData, Flows, IntentSpec, Plan, Shortfall, Site

FloatArray = npt.NDArray[np.float64]
BLOCKS = ("c", "d", "gp", "gm", "k", "u", "v", "e")
OFF = {name: i * HOURS for i, name in enumerate(BLOCKS)}
N_CORE = len(BLOCKS) * HOURS
HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
SHORTFALL_UNITS = {"soc_min": "kWh", "throughput_max": "kWh", "import_max": "kW", "export_max": "kW", "discharge_max": "kW"}


class OptimizerError(RuntimeError):
    """HiGHS did not return an optimal solution (should not happen: the LP is always feasible)."""


def day_inputs(day: DayData, use: Literal["forecast", "actual"]) -> dict[str, FloatArray]:
    if use not in ("forecast", "actual"):
        raise ValueError(f"use must be 'forecast' or 'actual', got {use!r}")
    pv = day.pv_forecast_kw if use == "forecast" else day.pv_actual_kw
    load = day.load_forecast_kw if use == "forecast" else day.load_actual_kw
    return {
        "pv": np.asarray(pv, dtype=float),
        "load": np.asarray(load, dtype=float),
        "critical": np.asarray(day.critical_load_kw, dtype=float),
        "p_import": np.asarray(day.import_price_per_kwh, dtype=float),
        "p_export": np.asarray(day.export_price_per_kwh, dtype=float),
    }


def default_export_limits(site: Site, outages: set[int], no_export: set[int] = frozenset()) -> FloatArray:
    """Export limit per hour: the site limit, or 0 when export is not allowed, the grid is out, or in no-export windows."""
    limit = site.grid.export_max_kw if site.grid.export_allowed else 0.0
    return np.array([0.0 if t in outages or t in no_export else limit for t in range(HOURS)])


def solve(day: DayData, site: Site, spec: IntentSpec | None = None, use: Literal["forecast", "actual"] = "forecast") -> Plan:
    """Cheapest plan for `day` that respects `spec` as far as physically possible."""
    spec = spec or IntentSpec()
    data = day_inputs(day, use)
    bad = np.flatnonzero(data["p_export"] > data["p_import"] + TOL)
    if bad.size:
        raise ValueError(f"export price exceeds import price at hours {bad.tolist()}: the LP would arbitrage without limit")
    rows = soft_rows(spec, site)
    # One LP inequality per (row, slack) pair: an SOC row appears twice, once with its goal's shared
    # worst-hour slack and once with its own hourly slack (core.rules.row_slack_keys).
    attached = [(row, key, weight) for row in rows for key, weight in row_slack_keys(row)]
    slack_col = {key: N_CORE + i for i, key in enumerate(dict.fromkeys(key for _, key, _ in attached))}
    a_eq, b_eq = _equality_matrix(data, site, N_CORE + len(slack_col))
    a_ub, b_ub = _inequality_matrix(attached, site, slack_col) if attached else (None, None)
    started = time.perf_counter()
    result = linprog(
        _objective(data, site, attached, slack_col),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=_bounds(data, site, spec, len(slack_col)),
        method="highs",
        options=HIGHS_OPTIONS,
    )
    solve_time = time.perf_counter() - started
    if result.status != 0:
        raise OptimizerError(f"HiGHS status {result.status}: {result.message}")
    objective = float(result.fun) * penalty_scale(data["p_import"])
    return _plan(day, site, spec, use, data, rows, np.asarray(result.x, dtype=float), objective, solve_time)


def _objective(
    data: dict[str, FloatArray], site: Site, attached: list[tuple[SoftRow, str, float]], slack_col: dict[str, int]
) -> FloatArray:
    """Objective coefficients divided by p̂, so the LP HiGHS sees is the same in every currency."""
    p_hat = penalty_scale(data["p_import"])
    beta = lp_degradation_cost(site, p_hat)
    cost = np.zeros(N_CORE + len(slack_col))
    cost[_block("c")] = beta
    cost[_block("d")] = beta
    cost[_block("gp")] = data["p_import"]
    cost[_block("gm")] = -data["p_export"]
    cost[_block("k")] = CURTAIL_WEIGHT * p_hat
    cost[_block("u")] = NONCRITICAL_WEIGHT * p_hat
    cost[_block("v")] = CRITICAL_WEIGHT * p_hat
    for _, key, weight in attached:
        cost[slack_col[key]] = weight * p_hat
    cost[OFF["e"] + HOURS - 1] = -terminal_value_per_kwh(data["p_import"], site.battery.eta_discharge)
    return cost / p_hat


def lp_degradation_cost(site: Site, p_hat: float) -> float:
    """β used in the LP: the site's β, floored at the curtailment weight (1e-4·p̂).

    Without the floor, a site with a near-zero β could charge and discharge in the same hour to
    burn surplus PV through battery losses more cheaply than curtailing it; such a plan cannot be
    expressed as one net setpoint. With β ≥ 1e-4·p̂, dissipating 1 kWh this way costs at least
    1e-4·p̂·(1 + η²)/(1 − η²) > 1e-4·p̂, so curtailing is always cheaper. Reported costs use the real β.
    """
    return max(site.battery.degradation_cost_per_kwh, CURTAIL_WEIGHT * p_hat)


def _equality_matrix(data: dict[str, FloatArray], site: Site, n: int) -> tuple[FloatArray, FloatArray]:
    battery = site.battery
    a = np.zeros((2 * HOURS, n))  # slack columns stay zero
    b = np.zeros(2 * HOURS)
    for t in range(HOURS):
        a[t, OFF["k"] + t] = -1.0
        a[t, OFF["gp"] + t] = 1.0
        a[t, OFF["d"] + t] = 1.0
        a[t, OFF["u"] + t] = 1.0
        a[t, OFF["v"] + t] = 1.0
        a[t, OFF["c"] + t] = -1.0
        a[t, OFF["gm"] + t] = -1.0
        b[t] = data["load"][t] - data["pv"][t]
        row = HOURS + t
        a[row, OFF["e"] + t] = 1.0
        a[row, OFF["c"] + t] = -battery.eta_charge
        a[row, OFF["d"] + t] = 1.0 / battery.eta_discharge
        if t > 0:
            a[row, OFF["e"] + t - 1] = -1.0
        else:
            b[row] = battery.initial_soc_frac * battery.capacity_kwh
    return a, b


def _inequality_matrix(
    attached: list[tuple[SoftRow, str, float]], site: Site, slack_col: dict[str, int]
) -> tuple[FloatArray, FloatArray]:
    initial = site.battery.initial_soc_frac * site.battery.capacity_kwh
    a = np.zeros((len(attached), N_CORE + len(slack_col)))
    b = np.zeros(len(attached))
    for j, (row, key, _) in enumerate(attached):
        a[j, slack_col[key]] = -1.0  # every row: (expression) − s ≤ bound, s shared per slack key
        if row.kind == "soc_min":  # −e_τ − s ≤ −bound
            if row.hour == INITIAL:
                b[j] = initial - row.bound
            else:
                a[j, OFF["e"] + row.hour] = -1.0
                b[j] = -row.bound
        elif row.kind == "import_max":
            a[j, OFF["gp"] + row.hour] = 1.0
            b[j] = row.bound
        elif row.kind == "export_max":
            a[j, OFF["gm"] + row.hour] = 1.0
        elif row.kind == "discharge_max":
            a[j, OFF["d"] + row.hour] = 1.0
        else:  # throughput_max
            a[j, OFF["c"] : OFF["c"] + HOURS] = 1.0
            a[j, OFF["d"] : OFF["d"] + HOURS] = 1.0
            b[j] = row.bound
    return a, b


def _bounds(data: dict[str, FloatArray], site: Site, spec: IntentSpec, n_slack: int) -> list[tuple[float, float | None]]:
    battery, grid = site.battery, site.grid
    outages = outage_hours(spec)
    export_max = grid.export_max_kw if grid.export_allowed else 0.0
    bounds: list[tuple[float, float | None]] = []
    bounds += [(0.0, battery.p_charge_max_kw)] * HOURS
    bounds += [(0.0, battery.p_discharge_max_kw)] * HOURS
    bounds += [(0.0, 0.0 if t in outages else grid.import_max_kw) for t in range(HOURS)]
    bounds += [(0.0, 0.0 if t in outages else export_max) for t in range(HOURS)]
    bounds += [(0.0, max(0.0, float(pv))) for pv in data["pv"]]
    bounds += [(0.0, max(0.0, float(load - crit))) for load, crit in zip(data["load"], data["critical"])]
    bounds += [(0.0, max(0.0, float(crit))) for crit in data["critical"]]
    bounds += [(battery.soc_min_frac * battery.capacity_kwh, battery.soc_max_frac * battery.capacity_kwh)] * HOURS
    bounds += [(0.0, None)] * n_slack
    return bounds


def _plan(
    day: DayData,
    site: Site,
    spec: IntentSpec,
    use: Literal["forecast", "actual"],
    data: dict[str, FloatArray],
    rows: list[SoftRow],
    x: FloatArray,
    objective: float,
    solve_time: float,
) -> Plan:
    v = {name: np.maximum(x[OFF[name] : OFF[name] + HOURS], 0.0) for name in BLOCKS}
    soc = x[OFF["e"] : OFF["e"] + HOURS]
    initial = site.battery.initial_soc_frac * site.battery.capacity_kwh
    trajectory = Trajectory(initial, soc, v["c"], v["d"], v["gp"], v["gm"])
    beta = site.battery.degradation_cost_per_kwh
    energy_cost = float(
        data["p_import"] @ v["gp"] - data["p_export"] @ v["gm"] + beta * (v["c"].sum() + v["d"].sum())
    )
    value_per_kwh = terminal_value_per_kwh(data["p_import"], site.battery.eta_discharge)
    terminal_value = value_per_kwh * float(soc[-1])
    simultaneous = [t for t in range(HOURS) if min(v["c"][t], v["d"][t]) > TOL]
    status = "optimal" if not simultaneous else f"optimal; simultaneous charge and discharge at hours {simultaneous}"
    export_limit = default_export_limits(site, outage_hours(spec), no_export_hours(spec))
    export_limit = np.where(v["gm"] > TOL, np.maximum(export_limit, v["gm"]), export_limit)
    return Plan(
        site_id=day.site_id,
        date=day.date,
        use=use,
        spec=spec,
        battery_kw=(v["c"] - v["d"]).tolist(),
        export_limit_kw=export_limit.tolist(),
        flows=Flows(
            charge_kw=v["c"].tolist(),
            discharge_kw=v["d"].tolist(),
            grid_import_kw=v["gp"].tolist(),
            grid_export_kw=v["gm"].tolist(),
            curtail_kw=v["k"].tolist(),
            unserved_noncritical_kw=v["u"].tolist(),
            unserved_critical_kw=v["v"].tolist(),
        ),
        soc_kwh=soc.tolist(),
        initial_soc_kwh=initial,
        protected_soc_kwh=_protected_soc(rows, soc, v["c"], site),
        energy_cost=energy_cost,
        penalty_cost=objective - energy_cost + terminal_value,
        terminal_value_per_kwh=value_per_kwh,
        terminal_value=terminal_value,
        objective=objective,
        shortfalls=_shortfalls(rows, trajectory),
        status=status,
        solve_time_s=solve_time,
        simultaneous_hours=simultaneous,
    )


def _protected_soc(rows: list[SoftRow], soc: FloatArray, charge: FloatArray, site: Site) -> list[float] | None:
    """SOC levels the executor may defend by importing (SPEC §6.6 goal protection), or None.

    For every SOC goal with required level R and first constrained hour τ0, protect
    min(planned SOC, required level), where the required level is
    - R at the goal's own hours
    - R along the lead-in: from the start of the plan's last charging run before τ0 up to τ0 (the
      charging the plan counted on, e.g. from PV, so a top-up happens where the plan meant to charge)
    - R − P_c·η_c·(τ0 − t) at any earlier hour t (the least SOC from which full-power charging can
      still reach R by τ0), so a top-up is always still possible in time
    Never above the plan's own SOC; 0 where nothing needs protecting.
    """
    goals: dict[str, list[SoftRow]] = {}
    for row in rows:
        if row.kind == "soc_min":
            goals.setdefault(row.path, []).append(row)
    if not goals:
        return None
    step = site.battery.p_charge_max_kw * site.battery.eta_charge
    protected = np.zeros(HOURS)
    for goal_rows in goals.values():
        hours = sorted(row.hour for row in goal_rows if row.hour is not None and row.hour >= 0)
        if not hours:
            continue
        bound, first = goal_rows[0].bound, hours[0]
        required = np.zeros(HOURS)
        required[hours] = bound
        charging = [t for t in range(first) if charge[t] > TOL]
        if charging:
            start = charging[-1]
            while start - 1 >= 0 and charge[start - 1] > TOL:
                start -= 1
            required[start:first] = bound
        for t in range(first):
            required[t] = max(required[t], bound - step * (first - t))
        protected = np.maximum(protected, np.minimum(soc, required))
    return np.maximum(protected, 0.0).tolist()


def _shortfalls(rows: list[SoftRow], trajectory: Trajectory) -> list[Shortfall]:
    """Each goal the plan misses by more than TOL: its worst shortfall and the hours it falls short."""
    grouped: dict[str, list[tuple[SoftRow, float]]] = {}
    for row in rows:
        missed = row_shortfall(row, trajectory)
        if missed > TOL:
            grouped.setdefault(row.path, []).append((row, missed))
    out = []
    for path, missed in grouped.items():
        rule, kind = missed[0][0].rule, missed[0][0].kind
        unit = SHORTFALL_UNITS[kind]
        hours = [row.hour for row, _ in missed if row.hour is not None]
        amount = max(amount for _, amount in missed)
        where = f" at hours {hours}" if hours else ""
        out.append(
            Shortfall(
                path=path,
                rule=rule,
                hours=hours,
                amount=amount,
                unit=unit,
                message=f"{rule} ({path}) cannot be fully met: short by up to {amount:.6f} {unit}{where}",
            )
        )
    return out


def _block(name: str) -> slice:
    return slice(OFF[name], OFF[name] + HOURS)
