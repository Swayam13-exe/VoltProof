"""Optimizer arms (SPEC §6.10). The gold spec is an argument here; intents arrive in P3.

- A1 cost-only:          optimizer, empty spec, forecast (knows nothing of the goal, not even outages)
- A5 perfect translator: optimizer, gold spec, forecast (what A4 would do with a perfect spec)
- A6 oracle:             optimizer, gold spec, actual data (the upper bound)
"""

from __future__ import annotations

from voltproof.core.optimizer import solve
from voltproof.models import DayData, IntentSpec, Plan, Site


def a1_plan(day: DayData, site: Site) -> Plan:
    return solve(day, site, IntentSpec(), use="forecast")


def a5_plan(day: DayData, site: Site, gold_spec: IntentSpec) -> Plan:
    return solve(day, site, gold_spec, use="forecast")


def a6_plan(day: DayData, site: Site, gold_spec: IntentSpec) -> Plan:
    return solve(day, site, gold_spec, use="actual")
