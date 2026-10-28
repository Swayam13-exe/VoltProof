"""Executor (SPEC §6.6): replays a plan on what actually happened.

Hour by hour on ACTUAL PV and load, the same rules for every arm:
1. the planned setpoint is clipped to the power limits and the SOC headroom (recorded as clipping)
2. during a true outage the grid is gone, so the battery overrides its setpoint as a safety measure,
   critical load first across the whole outage window:
   - it always covers the hour's critical deficit (as far as power and SOC allow)
   - it covers the non-critical deficit only from energy beyond what the critical load needs for the
     rest of that outage window (estimated from expected PV, which the site controller knows)
   - it stores any PV surplus it can; the rest is curtailed
   A plan that already covers the deficit is followed unchanged.
3. otherwise:
   a. grid charging only as intended: the battery may import to charge at most
      max(0, planned_charge − expected_surplus), expected_surplus = max(0, expected PV − expected load);
      further planned charging must come from actual PV surplus (recorded in grid_charge_limited_hours)
      goal protection: a plan that carries protected_soc_kwh (the optimizer sets it from the spec's
      SOC goals) may import beyond that cap, only as much as needed to end the hour at or above the
      protected level, within power and import limits (recorded in goal_topup_hours); plans without
      it get no top-up
   b. import limit: if the hour would import more than the grid allows, first reduce charging, then
      discharge within limits, and only then shed load, non-critical first (counted as an override)
   c. no wasted solar: PV surplus that would be curtailed (export not allowed or at its limit) is
      charged into the battery within power and SOC limits (recorded in solar_absorbed_hours)
   d. the grid imports the rest of any deficit and exports surplus up to the hour's export limit;
      anything left is curtailed

"Expected" PV and load are the data the plan was made with: the forecast for every arm except the
oracle, which plans on actual data (`plan_basis="actual"`).

realized_cost = Σ p⁺·import − p⁻·export + β·Σ|b| (grid + degradation, the optimizer's definition).
True outages are facts of the world (from the gold spec), the same for every arm.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from voltproof.core.optimizer import default_export_limits
from voltproof.models import HOURS, TOL, DayData, ExecutionResult, Flows, HourWindow, Site


def execute(
    battery_kw: Sequence[float],
    day: DayData,
    site: Site,
    true_outages: Sequence[HourWindow] = (),
    export_limit_kw: Sequence[float] | None = None,
    plan_basis: Literal["forecast", "actual"] = "forecast",
    protected_soc_kwh: Sequence[float] | None = None,
) -> ExecutionResult:
    planned = np.asarray(list(battery_kw), dtype=float)
    if planned.shape != (HOURS,) or not np.all(np.isfinite(planned)):
        raise ValueError(f"execute needs {HOURS} finite setpoints")
    battery, grid = site.battery, site.grid
    outages = {hour for window in true_outages for hour in window.hours()}
    limits = default_export_limits(site, outages) if export_limit_kw is None else np.asarray(export_limit_kw, dtype=float)
    limits = np.where([t in outages for t in range(HOURS)], 0.0, np.maximum(limits, 0.0))

    pv = np.asarray(day.pv_actual_kw, dtype=float)
    load = np.asarray(day.load_actual_kw, dtype=float)
    critical = np.asarray(day.critical_load_kw, dtype=float)
    expected_pv = np.asarray(day.pv_forecast_kw if plan_basis == "forecast" else day.pv_actual_kw, dtype=float)
    expected_load = np.asarray(day.load_forecast_kw if plan_basis == "forecast" else day.load_actual_kw, dtype=float)
    expected_surplus = np.maximum(0.0, expected_pv - expected_load)
    e_min, e_max = battery.soc_min_frac * battery.capacity_kwh, battery.soc_max_frac * battery.capacity_kwh
    initial = battery.initial_soc_frac * battery.capacity_kwh

    protected = None if protected_soc_kwh is None else np.asarray(protected_soc_kwh, dtype=float)
    applied = np.zeros(HOURS)
    soc = np.zeros(HOURS)
    f = {name: np.zeros(HOURS) for name in ("import", "export", "curtail", "unserved_nc", "unserved_c")}
    clipped, overridden, grid_limited, absorbed, topped_up = [], [], [], [], []
    e = initial
    for t in range(HOURS):
        max_charge = max(0.0, min(battery.p_charge_max_kw, (e_max - e) / battery.eta_charge))
        max_discharge = max(0.0, min(battery.p_discharge_max_kw, (e - e_min) * battery.eta_discharge))
        b = min(max(planned[t], -max_discharge), max_charge)
        if abs(b - planned[t]) > TOL:
            clipped.append(t)
        if t in outages:
            residual = load[t] + b - pv[t]
            future_need = _future_critical_need(t, outages, critical, expected_pv)
            b_new = _outage_setpoint(b, residual, load[t], critical[t], e, e_min, max_discharge, max_charge, future_need, battery.eta_discharge)
            if abs(b_new - b) > TOL:
                overridden.append(t)
            b = b_new
        else:
            if b > 0:  # (a) grid charging only as intended
                allowed = max(0.0, pv[t] - load[t]) + max(0.0, b - expected_surplus[t])
                if b > allowed + TOL:
                    b = allowed
                    grid_limited.append(t)
            if protected is not None:  # goal protection: keep realized SOC at the plan's protected level
                b_new = _topup_setpoint(b, e, protected[t], max_charge, grid.import_max_kw - load[t] + pv[t], battery)
                if b_new > b + TOL:
                    b = b_new
                    topped_up.append(t)
            residual = load[t] + b - pv[t]
            if residual > grid.import_max_kw + TOL:  # (b) import limit: charging, then discharge, then shed
                excess = residual - grid.import_max_kw
                cut = min(max(b, 0.0), excess)
                extra = min(excess - cut, b - cut + max_discharge)
                if cut + extra > TOL:
                    overridden.append(t)
                b -= cut + extra
            elif -residual > limits[t] + TOL and b < max_charge - TOL:  # (c) no wasted solar
                b = min(b + (-residual - limits[t]), max_charge)
                absorbed.append(t)
        residual = load[t] + b - pv[t]
        if residual > 0:
            f["import"][t] = 0.0 if t in outages else min(residual, grid.import_max_kw)
            _shed(t, residual - f["import"][t], load, critical, f)
        else:
            f["export"][t] = min(-residual, limits[t])
            f["curtail"][t] = -residual - f["export"][t]
        charge, discharge = max(b, 0.0), max(-b, 0.0)
        e = min(max(e + battery.eta_charge * charge - discharge / battery.eta_discharge, e_min), e_max)
        applied[t], soc[t] = b, e

    charge_kw, discharge_kw = np.maximum(applied, 0.0), np.maximum(-applied, 0.0)
    grid_cost = float(np.dot(day.import_price_per_kwh, f["import"]) - np.dot(day.export_price_per_kwh, f["export"]))
    degradation = battery.degradation_cost_per_kwh * float(charge_kw.sum() + discharge_kw.sum())
    return ExecutionResult(
        battery_kw=applied.tolist(),
        export_limit_kw=limits.tolist(),
        flows=Flows(
            charge_kw=charge_kw.tolist(),
            discharge_kw=discharge_kw.tolist(),
            grid_import_kw=f["import"].tolist(),
            grid_export_kw=f["export"].tolist(),
            curtail_kw=f["curtail"].tolist(),
            unserved_noncritical_kw=f["unserved_nc"].tolist(),
            unserved_critical_kw=f["unserved_c"].tolist(),
        ),
        soc_kwh=soc.tolist(),
        initial_soc_kwh=initial,
        realized_cost=grid_cost + degradation,
        grid_cost=grid_cost,
        degradation_cost=degradation,
        unserved_noncritical_kwh=float(f["unserved_nc"].sum()),
        unserved_critical_kwh=float(f["unserved_c"].sum()),
        overrides=len(overridden),
        override_hours=overridden,
        clipped_hours=clipped,
        grid_charge_limited_hours=grid_limited,
        goal_topup_hours=topped_up,
        solar_absorbed_hours=absorbed,
        outage_hours=sorted(outages),
    )


def _topup_setpoint(b: float, e: float, protected: float, max_charge: float, import_headroom: float, battery) -> float:
    """Raise the setpoint just enough that SOC ends the hour at `protected`, within the charge power,
    SOC headroom and the grid import limit (import_headroom = G_imp − load + PV: the largest b the
    grid can supply). Never lowers b."""
    if protected <= 0:
        return b
    end_soc = e + battery.eta_charge * max(b, 0.0) - max(-b, 0.0) / battery.eta_discharge
    if end_soc >= protected - TOL:
        return b
    needed = (protected - e) / battery.eta_charge if protected >= e else (protected - e) * battery.eta_discharge
    return max(b, min(needed, max_charge, import_headroom))


def _future_critical_need(t: int, outages: set[int], critical: np.ndarray, expected_pv: np.ndarray) -> float:
    """Critical energy (kWh delivered) still needed after hour t in the same contiguous outage window."""
    need = 0.0
    tau = t + 1
    while tau in outages and tau < HOURS:
        need += max(0.0, critical[tau] - expected_pv[tau])
        tau += 1
    return need


def _outage_setpoint(
    b: float,
    residual: float,
    load: float,
    critical: float,
    e: float,
    e_min: float,
    max_discharge: float,
    max_charge: float,
    future_need: float,
    eta_discharge: float,
) -> float:
    """Battery setpoint for an outage hour: balance surplus, cover the critical deficit, then
    the non-critical deficit only with energy the rest of the outage's critical load won't need."""
    if residual <= 0:
        return min(b - residual, max_charge)  # store the surplus (b − residual balances the hour)
    unserved_nc = min(residual, max(0.0, load - critical))
    unserved_c = residual - unserved_nc
    b = b - min(unserved_c, b + max_discharge)  # critical first
    spare = max(0.0, (e - e_min) * eta_discharge - future_need - max(0.0, -b))
    return b - min(unserved_nc, spare, b + max_discharge)


def _shed(t: int, deficit: float, load: np.ndarray, critical: np.ndarray, f: dict[str, np.ndarray]) -> None:
    """Leave `deficit` kW unserved: non-critical load first, then critical."""
    noncritical = max(0.0, load[t] - critical[t])
    f["unserved_nc"][t] = min(deficit, noncritical)
    f["unserved_c"][t] = deficit - f["unserved_nc"][t]
