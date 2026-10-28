"""A0: the rule-based self-consumption battery people use today (SPEC §6.10).

Each hour, using only that hour's PV and load (causal, no forecast):
- PV surplus → charge with it, within the power limit and the SOC headroom
- deficit    → discharge to cover it while SOC is above 20 % (floor = max(SOC min, 20 %))
- never charges from the grid; ignores prices and goals (it does not know about outages; during a
  real outage the executor's safety override takes over, see core.executor)

For execution A0 runs on ACTUAL PV and load: a reactive controller needs no forecast. Its
forecast setpoints are used only to judge its plan's physical validity before the day.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from voltproof.core.optimizer import day_inputs
from voltproof.models import HOURS, DayData, Site

FloatArray = npt.NDArray[np.float64]
SOC_FLOOR_FRAC = 0.20


def rule_based_setpoints(pv_kw: npt.ArrayLike, load_kw: npt.ArrayLike, site: Site) -> FloatArray:
    battery = site.battery
    pv, load = np.asarray(pv_kw, dtype=float), np.asarray(load_kw, dtype=float)
    e_max = battery.soc_max_frac * battery.capacity_kwh
    floor = max(battery.soc_min_frac, SOC_FLOOR_FRAC) * battery.capacity_kwh
    e = battery.initial_soc_frac * battery.capacity_kwh
    setpoints = np.zeros(HOURS)
    for t in range(HOURS):
        net = pv[t] - load[t]
        if net > 0:
            b = min(net, battery.p_charge_max_kw, max(0.0, (e_max - e) / battery.eta_charge))
        else:
            b = -min(-net, battery.p_discharge_max_kw, max(0.0, (e - floor) * battery.eta_discharge))
        e += battery.eta_charge * max(b, 0.0) - max(-b, 0.0) / battery.eta_discharge
        setpoints[t] = b
    return setpoints


def a0_setpoints(day: DayData, site: Site, use: Literal["forecast", "actual"] = "actual") -> FloatArray:
    data = day_inputs(day, use)
    return rule_based_setpoints(data["pv"], data["load"], site)
