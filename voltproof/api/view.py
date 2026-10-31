"""The page's view of one A4 run (SPEC §6.13), shared by recorded replays and live runs.

Built only from the frozen code, imported and called unchanged:
- the plan shown is the accepted plan from the plan log; `plan_view` re-solves the finalized spec
  only to get its hourly series and asserts the re-solved setpoints equal the logged ones exactly
- the same-day comparison executes that plan and the rule-based battery (A0) on the same-day weather
  analysis, with the spec's outages, and checks both against the spec (for a typed goal the agent's
  spec is all there is; benchmark scores use hand-written gold specs)
- the trace keeps every step the page shows (tool calls and results, gate attempts, the reviewer's
  verdict) and leaves out model reasoning text and raw model replies

Every string from the model stays a plain string: the page renders it as text, never as HTML.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np

from voltproof.baselines.rule_based import a0_setpoints
from voltproof.core.executor import execute
from voltproof.core.metrics import critical_served_pct, gold_check, trajectory_total_cost
from voltproof.core.optimizer import solve
from voltproof.core.spec import INITIAL, hours_to_windows, no_export_hours, outage_hours, soft_rows
from voltproof.core.verifier import verify
from voltproof.models import HOURS, DayData, ExecutionResult, IntentSpec, Plan, Site, Tariff

SCHEMA_VERSION = 1
REPLAY_LABEL = "replay (recorded run)"
LIVE_LABEL = "live run"
AGENT_SPEC_NOTE = "Checked against the goals as the agent understood them; benchmark scores use hand-written gold specs."
GOLD_MATCH_NOTE = "Checked against the agent's spec, which matched the benchmark's hand-written gold spec for this goal."
TRACE_DROP = ("reasoning", "content")  # model reasoning text and raw replies are not shown

Event = Mapping[str, Any]


class LoggedPlanMismatch(ValueError):
    """The re-solved plan differs from the accepted plan in the plan log."""


def _r(values: Any, digits: int) -> Any:
    """Round for output (a list, a number or None); -0.0 becomes 0.0."""
    if values is None:
        return None
    if isinstance(values, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in values]
    value = round(float(values), digits)
    return 0.0 if value == 0 else value


def _event(event: Event) -> dict[str, Any]:
    return dict(event) if isinstance(event, Mapping) else event.model_dump(mode="json")  # type: ignore[union-attr]


# --- trace -------------------------------------------------------------------------------------------


def trace_view(events: Sequence[Event]) -> list[dict[str, Any]]:
    """The steps the page shows, in order; model reasoning and raw replies left out."""
    out = []
    for raw in map(_event, events):
        data = dict(raw.get("data") or {})
        if raw["kind"] == "llm_call":
            data = {k: v for k, v in data.items() if k not in TRACE_DROP}
        out.append({"kind": raw["kind"], "turn": raw.get("turn"), "name": raw.get("name"), "summary": raw.get("summary"), "data": data})
    return out


def tool_calls(events: Sequence[Event], name: str) -> list[dict[str, Any]]:
    return [dict(e["data"].get("args") or {}) for e in map(_event, events) if e["kind"] == "tool_call" and e.get("name") == name and isinstance(e["data"].get("args"), dict)]


def selected_plan_id(events: Sequence[Event]) -> str | None:
    selections = [e for e in map(_event, events) if e["kind"] == "final" and e.get("name") == "selection"]
    return selections[-1]["data"].get("plan_id") if selections else None


def accepted_finalize(events: Sequence[Event]) -> dict[str, Any]:
    """The arguments of the last finalize call on the selected plan ({} if there is none)."""
    plan_id = selected_plan_id(events)
    calls = [args for args in tool_calls(events, "finalize") if args.get("plan_id") == plan_id]
    return calls[-1] if calls else {}


def gate_view(events: Sequence[Event]) -> dict[str, Any]:
    """Every finalize attempt with the checks it failed, the outcome and the reviewer's verdicts."""
    attempts, outcome, reviews = [], None, []
    for e in map(_event, events):
        data = e.get("data") or {}
        if e["kind"] == "gate" and e.get("name") == "finalize":
            attempts.append(
                {
                    "attempt": data.get("attempt"),
                    "final": data.get("final"),
                    "plan_id": data.get("plan_id"),
                    "failed_checks": [{"check": f.get("check"), "message": f.get("message")} for f in data.get("failed_checks", [])],
                }
            )
        elif e["kind"] == "gate":
            outcome = {"decision": e.get("name"), "message": e.get("summary")}
        elif e["kind"] == "review":
            reviews.append({"approved": data.get("approved"), "issues": data.get("issues", [])})
    return {"attempts": attempts, "outcome": outcome, "reviews": reviews, "repair_rounds": max(0, len(attempts) - 1)}


# --- plan ----------------------------------------------------------------------------------------------


def goal_overlays(spec: IntentSpec, site: Site) -> dict[str, Any]:
    """What the chart shades: the minimum SOC % each hour must end at (the initial SOC separately),
    and the hours of outages, no-export windows, import caps and discharge limits."""
    capacity = site.battery.capacity_kwh
    required: list[float | None] = [None] * HOURS
    initial_required: float | None = None
    for row in soft_rows(spec, site):
        if row.kind != "soc_min":
            continue
        pct = 100.0 * row.bound / capacity
        if row.hour == INITIAL:
            initial_required = max(initial_required or 0.0, pct)
        else:
            required[row.hour] = max(required[row.hour] or 0.0, pct)
    by_path: dict[str, dict[str, Any]] = {}
    for row in soft_rows(spec, site):
        if row.kind == "soc_min":
            entry = by_path.setdefault(row.path, {"rule": row.rule, "path": row.path, "min_soc_pct": _r(100.0 * row.bound / capacity, 1), "hours": []})
            entry["hours"].append(row.hour)
    goals = list(by_path.values())
    caps: list[float | None] = [None] * HOURS
    for cap in spec.import_caps:
        for hour in cap.hours():
            caps[hour] = cap.max_kw if caps[hour] is None else min(caps[hour], cap.max_kw)  # type: ignore[type-var]
    allowed = None if spec.discharge_only_windows is None else sorted({h for w in spec.discharge_only_windows for h in w.hours()})
    return {
        "required_soc_pct": _r(required, 1),
        "initial_required_soc_pct": _r(initial_required, 1),
        "soc_goals": goals,
        "outage_hours": sorted(outage_hours(spec)),
        "no_export_hours": sorted(no_export_hours(spec)),
        "import_cap_kw": _r(caps, 3),
        "discharge_allowed_hours": allowed,
        "throughput_cap_kwh": _r(spec.max_throughput_kwh, 3),
    }


def logged_plan(spec: IntentSpec, day: DayData, site: Site, logged_battery_kw: Sequence[float]) -> Plan:
    """Re-solve the finalized spec on the forecast (as PlanBook.solve does) and require the setpoints
    to equal the accepted plan in the log exactly."""
    plan = solve(day, site, spec, use="forecast")
    if [float(v) for v in plan.battery_kw] != [float(v) for v in logged_battery_kw]:
        raise LoggedPlanMismatch("the re-solved plan differs from the accepted plan in the plan log")
    return plan


def plan_view(plan: Plan, day: DayData, site: Site, tariff: Tariff) -> dict[str, Any]:
    capacity = site.battery.capacity_kwh
    spec = plan.spec
    check = verify(plan.battery_kw, day, site, spec, "forecast", export_limit_kw=plan.export_limit_kw)
    flows = plan.flows
    return {
        "battery_kw": _r(plan.battery_kw, 3),
        "soc_pct": _r([100.0 * e / capacity for e in plan.soc_kwh], 1),
        "initial_soc_pct": _r(100.0 * plan.initial_soc_kwh / capacity, 1),
        "charge_kw": _r(flows.charge_kw, 3),
        "discharge_kw": _r(flows.discharge_kw, 3),
        "grid_import_kw": _r(flows.grid_import_kw, 3),
        "grid_export_kw": _r(flows.grid_export_kw, 3),
        "curtail_kw": _r(flows.curtail_kw, 3),
        "pv_forecast_kw": _r(day.pv_forecast_kw, 3),
        "load_forecast_kw": _r(day.load_forecast_kw, 3),
        "critical_load_kw": _r(day.critical_load_kw, 3),
        "import_price_per_kwh": _r(day.import_price_per_kwh, 4),
        "export_price_per_kwh": _r(day.export_price_per_kwh, 4),
        "tariff_periods": {name: [[a, b] for a, b in hours_to_windows(list(hours))] for name, hours in sorted(tariff.periods.items())},
        "goals": goal_overlays(spec, site),
        "energy_cost": _r(plan.energy_cost, 2),
        "shortfalls": [{"path": s.path, "rule": s.rule, "hours": s.hours, "amount": _r(s.amount, 3), "unit": s.unit, "message": s.message} for s in plan.shortfalls],
        "verified": check.valid,
        "violations": [v.message for v in check.violations],
    }


# --- same-day comparison against the rule-based battery ---------------------------------------------------


def _side(execution: ExecutionResult, spec: IntentSpec, day: DayData, site: Site) -> dict[str, Any]:
    capacity = site.battery.capacity_kwh
    satisfied, violations = gold_check(execution, spec, site)
    return {
        "soc_pct": _r([100.0 * e / capacity for e in execution.soc_kwh], 1),
        "battery_kw": _r(execution.battery_kw, 3),
        "grid_import_kw": _r(execution.flows.grid_import_kw, 3),
        "grid_export_kw": _r(execution.flows.grid_export_kw, 3),
        "realized_cost": _r(trajectory_total_cost(execution, spec, day, site).net_money, 2),
        "unserved_critical_kwh": _r(execution.unserved_critical_kwh, 3),
        "unserved_noncritical_kwh": _r(execution.unserved_noncritical_kwh, 3),
        "critical_served_pct": _r(critical_served_pct(execution, day), 1),
        "goals_met": satisfied,
        "goals_missed": sorted({v.rule for v in violations}),
    }


def same_day_view(plan: Plan, day: DayData, site: Site) -> dict[str, Any]:
    """Execute the plan and the rule-based battery on the same-day weather analysis, facing the spec's
    outages, and check both against the spec."""
    spec = plan.spec
    outages = spec.grid_outages
    agent = execute(
        plan.battery_kw,
        day,
        site,
        outages,
        export_limit_kw=plan.export_limit_kw,
        plan_basis="forecast",
        protected_soc_kwh=plan.protected_soc_kwh,
        import_limit_kw=plan.import_limit_kw,
    )
    rule = execute(a0_setpoints(day, site, "actual"), day, site, outages)
    agent_view, rule_view = _side(agent, spec, day, site), _side(rule, spec, day, site)
    return {
        "pv_actual_kw": _r(day.pv_actual_kw, 3),
        "load_actual_kw": _r(day.load_actual_kw, 3),
        "agent": agent_view,
        "rule_based": rule_view,
        "cost_difference": _r(rule_view["realized_cost"] - agent_view["realized_cost"], 2),
    }


# --- the whole view ---------------------------------------------------------------------------------------


def site_card(site: Site, tariff: Tariff) -> dict[str, Any]:
    return {
        "site_id": site.site_id,
        "name": site.name,
        "site_type": site.site_type,
        "country": site.country,
        "lat": site.lat,
        "lon": site.lon,
        "timezone": site.timezone,
        "currency": site.currency,
        "tariff_kind": tariff.kind,
        "tariff_name": tariff.name,
        "pv_kwp": site.pv.capacity_kwp,
        "battery_kwh": site.battery.capacity_kwh,
        "battery_kw": site.battery.p_charge_max_kw,
        "import_max_kw": site.grid.import_max_kw,
        "export_allowed": site.grid.export_allowed,
        "critical_load_name": site.critical_load_name,
        "critical_load_kw": site.critical_load_kw,
    }


def day_card(day: DayData) -> dict[str, Any]:
    w = day.weather
    return {
        "date": day.date.isoformat(),
        "pv_forecast_kwh": _r(w.pv_forecast_kwh, 1),
        "pv_actual_kwh": _r(w.pv_actual_kwh, 1),
        "clear_sky_index_forecast": _r(w.clear_sky_index_forecast, 2),
        "clear_sky_index_actual": _r(w.clear_sky_index_actual, 2),
    }


def view_model(
    *,
    kind: Literal["replay", "live"],
    site: Site,
    day: DayData,
    tariff: Tariff,
    goal: str,
    language: str | None,
    events: Sequence[Event],
    logged_battery_kw: Sequence[float] | None,
    spec: IntentSpec | None,
    meta: Mapping[str, Any],
    spec_matches_gold: bool = False,
) -> dict[str, Any]:
    """The page payload. `logged_battery_kw` and `spec` are the accepted plan from the plan log
    (None when the run ended without a plan: the page then shows the trace and the fallback)."""
    finalize = accepted_finalize(events)
    checklists = tool_calls(events, "set_checklist")
    view: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "label": REPLAY_LABEL if kind == "replay" else LIVE_LABEL,
        "checked_against": GOLD_MATCH_NOTE if spec_matches_gold else AGENT_SPEC_NOTE,
        "meta": dict(meta),
        "site": site_card(site, tariff),
        "day": day_card(day),
        "goal": {"text": goal, "language": language},
        "checklist": list(checklists[0].get("items", [])) if checklists else [],
        "spec": spec.model_dump(mode="json", exclude_defaults=True, exclude={"assumptions"}) if spec is not None else None,
        "assumptions": list(finalize.get("assumptions", [])),
        "spec_assumptions": list(spec.assumptions) if spec is not None else [],
        "explanation": finalize.get("explanation"),
        "coverage": list(finalize.get("coverage", [])),
        "gate": gate_view(events),
        "trace": trace_view(events),
        "plan": None,
        "same_day": None,
    }
    if spec is not None and logged_battery_kw is not None:
        plan = logged_plan(spec, day, site, logged_battery_kw)
        view["plan"] = plan_view(plan, day, site, tariff)
        view["same_day"] = same_day_view(plan, day, site)
    return view
