"""A2/A3 declarations → the same controller levers the optimizer gives A4's plans (fairness).

An LLM that writes its own schedule may also declare `no_export_hours` (windows with no export),
`reserve` (windows with a minimum SOC) and `import_caps` (windows with a grid import cap). They are
mapped with the optimizer's own helpers:
- export_limit_kw   = `default_export_limits(site, no outages, no_export)`: 0 in declared hours
- import_limit_kw   = `default_import_limits(site, caps)`: the declared cap in its hours
- protected_soc_kwh = `protected_soc(...)` on the declared reserve rows and the schedule's own
                      simulated SOC and charging, so the executor may top up from the grid exactly
                      as it does for optimizer plans, and never above the planned SOC
Invalid declarations are dropped (and reported); the schedule itself is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from voltproof.core.optimizer import default_export_limits, default_import_limits, protected_soc
from voltproof.core.spec import format_errors, no_export_hours, parse_spec, soft_rows
from voltproof.core.verifier import verify
from voltproof.models import DayData, IntentSpec, Site


class DeclWindow(BaseModel):
    """Loosely typed on purpose: range checks happen in `controller_limits`, so a bad window is
    dropped instead of rejecting the whole schedule."""

    model_config = ConfigDict(extra="forbid")
    start_hour: int
    end_hour: int


class DeclReserve(DeclWindow):
    min_soc_pct: float


class DeclImportCap(DeclWindow):
    max_kw: float


@dataclass(frozen=True)
class ControllerLimits:
    export_limit_kw: list[float]
    import_limit_kw: list[float]
    protected_soc_kwh: list[float] | None
    spec: IntentSpec  # the declarations as a spec (empty when none or invalid)
    errors: list[str]


def controller_limits(
    battery_kw: Sequence[float],
    no_export: Sequence[DeclWindow],
    reserve: Sequence[DeclReserve],
    day: DayData,
    site: Site,
    import_caps: Sequence[DeclImportCap] = (),
) -> ControllerLimits:
    kept: dict[str, list[dict[str, float]]] = {"no_export_windows": [], "reserve_windows": [], "import_caps": []}
    messages = []
    declared = (("no_export_windows", no_export), ("reserve_windows", reserve), ("import_caps", import_caps))
    for field_name, windows in declared:
        for window in windows:
            _, errors = parse_spec({field_name: [window.model_dump()]}, site)
            if errors:
                messages.append(f"dropped {window.model_dump()}: {format_errors(errors)}")
            else:
                kept[field_name].append(window.model_dump())
    spec = IntentSpec.model_validate(kept)
    export_limit = default_export_limits(site, set(), no_export_hours(spec))
    protected = None
    if spec.reserve_windows:
        sim = verify(list(battery_kw), day, site, None, "forecast", physics_only=True)
        if sim.soc_kwh is not None and sim.flows is not None:
            protected = protected_soc(soft_rows(spec, site), np.asarray(sim.soc_kwh), np.asarray(sim.flows.charge_kw), site)
    return ControllerLimits(
        export_limit_kw=export_limit.tolist(),
        import_limit_kw=default_import_limits(site, spec).tolist(),
        protected_soc_kwh=protected,
        spec=spec,
        errors=messages,
    )
