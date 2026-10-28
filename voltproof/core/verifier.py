"""Plan verifier (SPEC §6.5): trusts nothing but the 24 battery setpoints.

It simulates the day from the setpoints as given (never clipped) and reports every breach with a
stable rule code and a plain message (values to 6 decimals, never printed equal to their limit):
- SOC:      e_t = e_{t−1} + η_c·max(b_t, 0) − max(−b_t, 0)/η_d
- residual: r_t = L_t + b_t − PV_t; r > 0 → import if the grid is up (up to its limit), else unserved;
            r < 0 → export up to the hour's export limit, the rest is curtailed (not a violation)

Physics rules use the spec's outages (grid availability) even with `physics_only=True`; that flag
skips only the goal rules (RESERVE … THROUGHPUT), as A3's simulate tool needs.
Cost on the given data: grid import − export revenue + degradation β·Σ|b| (same as the optimizer).
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np

from voltproof.core.optimizer import day_inputs, default_export_limits
from voltproof.core.rules import Trajectory, physics_message, spec_violations
from voltproof.core.spec import outage_hours, soft_rows
from voltproof.models import HOURS, TOL, DayData, Flows, IntentSpec, Site, VerifyResult, Violation

MAX_LISTED = 20


def verify(
    battery_kw: Sequence[float],
    day: DayData,
    site: Site,
    spec: IntentSpec | None = None,
    use: Literal["forecast", "actual"] = "forecast",
    *,
    physics_only: bool = False,
    export_limit_kw: Sequence[float] | None = None,
) -> VerifyResult:
    battery = site.battery
    initial = battery.initial_soc_frac * battery.capacity_kwh
    setpoints = list(battery_kw)
    early = _input_violations(setpoints)
    if early:
        return VerifyResult(
            valid=False,
            use=use,
            physics_only=physics_only,
            violations=early[:MAX_LISTED],
            total_violations=len(early),
            initial_soc_kwh=initial,
        )

    data = day_inputs(day, use)
    b = np.asarray(setpoints, dtype=float)
    outages = outage_hours(spec)
    # A plan that carries an inverter export limit (optimizer plans) passes it; battery-only plans
    # (A0, LLM-direct) get the site's own limit, so exporting in a no-export window is flagged.
    limits = (
        np.asarray(export_limit_kw, dtype=float)
        if export_limit_kw is not None
        else default_export_limits(site, outages)
    )
    violations: list[Violation] = []

    charge, discharge = np.maximum(b, 0.0), np.maximum(-b, 0.0)
    soc = initial + np.cumsum(battery.eta_charge * charge - discharge / battery.eta_discharge)
    e_min, e_max = battery.soc_min_frac * battery.capacity_kwh, battery.soc_max_frac * battery.capacity_kwh
    flows = {name: np.zeros(HOURS) for name in ("import", "export", "curtail", "unserved_nc", "unserved_c")}

    for t in range(HOURS):
        if charge[t] > battery.p_charge_max_kw + TOL:
            violations.append(
                physics_message("CHARGE_LIMIT", t, charge[t], battery.p_charge_max_kw, "charge setpoint {v} kW exceeds the maximum charge power {limit} kW", "kW")
            )
        if discharge[t] > battery.p_discharge_max_kw + TOL:
            violations.append(
                physics_message("DISCHARGE_LIMIT", t, discharge[t], battery.p_discharge_max_kw, "discharge setpoint {v} kW exceeds the maximum discharge power {limit} kW", "kW")
            )
        if soc[t] < e_min - TOL:
            violations.append(
                physics_message("SOC_MIN", t, soc[t], e_min, "state of charge {v} kWh at the end of the hour is below the minimum {limit} kWh", "kWh")
            )
        if soc[t] > e_max + TOL:
            violations.append(
                physics_message("SOC_MAX", t, soc[t], e_max, "state of charge {v} kWh at the end of the hour is above the maximum {limit} kWh", "kWh")
            )
        violations.extend(_dispatch(t, data, b[t], t in outages, site, limits[t], flows))

    trajectory = Trajectory(initial, soc, charge, discharge, flows["import"], flows["export"])
    if spec is not None and not physics_only:
        violations.extend(spec_violations(soft_rows(spec, site), trajectory))
    violations.sort(key=lambda v: (v.hour if v.hour is not None else -2))

    grid_cost = float(data["p_import"] @ flows["import"] - data["p_export"] @ flows["export"])
    degradation = battery.degradation_cost_per_kwh * float(charge.sum() + discharge.sum())
    return VerifyResult(
        valid=not violations,
        use=use,
        physics_only=physics_only,
        violations=violations[:MAX_LISTED],
        total_violations=len(violations),
        soc_kwh=soc.tolist(),
        initial_soc_kwh=initial,
        flows=Flows(
            charge_kw=charge.tolist(),
            discharge_kw=discharge.tolist(),
            grid_import_kw=flows["import"].tolist(),
            grid_export_kw=flows["export"].tolist(),
            curtail_kw=flows["curtail"].tolist(),
            unserved_noncritical_kw=flows["unserved_nc"].tolist(),
            unserved_critical_kw=flows["unserved_c"].tolist(),
        ),
        grid_cost=grid_cost,
        degradation_cost=degradation,
        energy_cost=grid_cost + degradation,
        unserved_noncritical_kwh=float(flows["unserved_nc"].sum()),
        unserved_critical_kwh=float(flows["unserved_c"].sum()),
    )


def _input_violations(setpoints: list[float]) -> list[Violation]:
    if len(setpoints) != HOURS:
        return [physics_message("LENGTH", None, float(len(setpoints)), float(HOURS), "the plan has {v} setpoints; it needs exactly {limit}", "")]
    out = []
    for t, value in enumerate(setpoints):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            out.append(
                Violation(hour=t, rule="NOT_FINITE", value=float("nan"), limit=0.0, message=f"NOT_FINITE at hour {t}: setpoint {value!r} is not a finite number")
            )
    return out


def _dispatch(
    t: int,
    data: dict[str, np.ndarray],
    b: float,
    outage: bool,
    site: Site,
    export_limit: float,
    flows: dict[str, np.ndarray],
) -> list[Violation]:
    """Grid dispatch for hour t; fills `flows` and returns physics violations."""
    out: list[Violation] = []
    residual = data["load"][t] + b - data["pv"][t]
    if residual > 0:
        if outage:
            unserved = residual
            if residual > TOL:
                out.append(
                    physics_message("OUTAGE_DEFICIT", t, residual, 0.0, "during the grid outage, PV and battery leave a deficit of {v} kW (allowed {limit} kW)", "kW")
                )
        else:
            flows["import"][t] = min(residual, site.grid.import_max_kw)
            unserved = residual - flows["import"][t]
            if residual > site.grid.import_max_kw + TOL:
                out.append(
                    physics_message("IMPORT_LIMIT", t, residual, site.grid.import_max_kw, "required grid import {v} kW exceeds the grid import limit {limit} kW", "kW")
                )
        noncritical_load = max(0.0, data["load"][t] - data["critical"][t])
        flows["unserved_nc"][t] = min(unserved, noncritical_load)
        flows["unserved_c"][t] = unserved - flows["unserved_nc"][t]
        if unserved > TOL and not outage:
            out.append(
                physics_message(
                    "UNSERVED_LOAD",
                    t,
                    unserved,
                    0.0,
                    f"{{v}} kWh of load cannot be served, {flows['unserved_c'][t]:.6f} kWh of it critical (allowed {{limit}} kWh)",
                    "kWh",
                )
            )
    else:
        surplus = -residual
        flows["export"][t] = min(surplus, max(0.0, export_limit))
        flows["curtail"][t] = surplus - flows["export"][t]
    return out
