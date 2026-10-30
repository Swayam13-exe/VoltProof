"""A2/A3 declarations map to the optimizer's own export-limit and protected-SOC helpers."""

from __future__ import annotations

import numpy as np

from tests.fakes_llm import agent_day, agent_site
from voltproof.agents.declarations import DeclImportCap, DeclReserve, DeclWindow, controller_limits
from voltproof.core.optimizer import default_export_limits, default_import_limits, protected_soc
from voltproof.core.spec import soft_rows
from voltproof.core.verifier import verify
from voltproof.models import IntentSpec

SCHEDULE = [0.0] * 8 + [1.0] * 4 + [0.0] * 6 + [-1.0] * 4 + [0.0] * 2


def test_no_export_hours_zero_the_export_limit_exactly_there() -> None:
    site, day = agent_site(), agent_day()
    limits = controller_limits(SCHEDULE, [DeclWindow(start_hour=10, end_hour=14)], [], day, site)
    assert limits.export_limit_kw == default_export_limits(site, set(), {10, 11, 12, 13}).tolist()
    assert limits.export_limit_kw[10:14] == [0.0] * 4 and limits.export_limit_kw[9] == 3.0
    assert limits.protected_soc_kwh is None and limits.errors == []


def test_reserve_maps_to_protected_soc_like_the_optimizer() -> None:
    site, day = agent_site(), agent_day()
    reserve = DeclReserve(start_hour=18, end_hour=22, min_soc_pct=40)
    limits = controller_limits(SCHEDULE, [], [reserve], day, site)
    sim = verify(SCHEDULE, day, site, None, physics_only=True)
    spec = IntentSpec.model_validate({"reserve_windows": [reserve.model_dump()]})
    expected = protected_soc(soft_rows(spec, site), np.asarray(sim.soc_kwh), np.asarray(sim.flows.charge_kw), site)
    assert limits.protected_soc_kwh == expected
    assert np.all(np.asarray(limits.protected_soc_kwh) <= np.asarray(sim.soc_kwh) + 1e-9)
    assert max(limits.protected_soc_kwh[17:22]) > 0


def test_no_declarations_give_default_limits() -> None:
    site, day = agent_site(), agent_day()
    limits = controller_limits(SCHEDULE, [], [], day, site)
    assert limits.export_limit_kw == default_export_limits(site, set()).tolist() and limits.protected_soc_kwh is None


def test_invalid_declarations_are_dropped_and_reported() -> None:
    site, day = agent_site(), agent_day()
    limits = controller_limits(
        SCHEDULE,
        [DeclWindow(start_hour=16, end_hour=12), DeclWindow(start_hour=10, end_hour=12)],
        [DeclReserve(start_hour=18, end_hour=22, min_soc_pct=150)],
        day,
        site,
    )
    assert len(limits.errors) == 2
    assert [w.start_hour for w in limits.spec.no_export_windows] == [10] and limits.spec.reserve_windows == []
    assert limits.protected_soc_kwh is None


def test_import_caps_map_to_the_optimizers_import_limits() -> None:
    site, day = agent_site(), agent_day()
    cap = DeclImportCap(start_hour=13, end_hour=17, max_kw=1.5)
    limits = controller_limits(SCHEDULE, [], [], day, site, [cap, DeclImportCap(start_hour=5, end_hour=3, max_kw=1)])
    expected = default_import_limits(site, IntentSpec.model_validate({"import_caps": [cap.model_dump()]}))
    assert limits.import_limit_kw == expected.tolist()
    assert limits.import_limit_kw[13:17] == [1.5] * 4 and limits.import_limit_kw[12] == site.grid.import_max_kw
    assert len(limits.errors) == 1  # the reversed window is dropped
    assert controller_limits(SCHEDULE, [], [], day, site).import_limit_kw == [site.grid.import_max_kw] * 24
