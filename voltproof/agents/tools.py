"""Agent tools (SPEC §6.9): schemas from pydantic models, compact JSON outputs.

The info tools read only the site, the FORECAST parts of DayData and the day's tariff, never
actuals. `PlanBook` (A4) validates a spec, solves it on the forecast, verifies the plan and stores it
under a deterministic plan_id; `SimBook` (A3) runs the physics-only verifier on a raw schedule.
Rounding: kW/kWh 3 decimals, prices 4, money 2, percentages 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from voltproof.agents.declarations import DeclImportCap, DeclReserve, DeclWindow
from voltproof.agents.loop import LLMTotals, Tool, ToolOutcome, Trace, args_schema
from voltproof.core.optimizer import solve
from voltproof.core.spec import INITIAL, hours_to_windows, outage_hours, parse_spec, soft_rows
from voltproof.core.verifier import verify
from voltproof.llm.prompts import GLOSSARY
from voltproof.models import HOURS, DayData, IntentSpec, Plan, Site, Tariff, VerifyResult

PHYSICS_RULES = frozenset(
    {"LENGTH", "NOT_FINITE", "CHARGE_LIMIT", "DISCHARGE_LIMIT", "SOC_MIN", "SOC_MAX", "IMPORT_LIMIT", "UNSERVED_LOAD", "OUTAGE_DEFICIT"}
)
MAX_LISTED_VIOLATIONS = 10
SOC_REPORT_HOURS = tuple(range(2, HOURS, 3))  # every 3rd hour, ending at hour 23
SPEC_FIELD_DESCRIPTIONS = {
    "objective": 'always "min_cost"',
    "grid_outages": "hours with no grid",
    "reserve_windows": "keep SOC >= min_soc_pct during [start_hour, end_hour)",
    "soc_targets": "SOC >= min_soc_pct at the end of hour by_hour-1",
    "min_soc_pct_all_day": "SOC >= this at the end of every hour",
    "end_of_day_min_soc_pct": "SOC at the end of hour 23",
    "import_caps": "grid import <= max_kw during [start_hour, end_hour)",
    "no_export_windows": "no export to the grid during these windows",
    "discharge_only_windows": "if set, the battery may discharge only in these windows",
    "max_throughput_kwh": "cap on total kWh charged + discharged",
    "assumptions": "interpretations you made",
}


@dataclass(frozen=True)
class EpisodeContext:
    """Everything an agent may learn about one episode. Actual PV and load stay inside `day` and are
    never read by a tool."""

    episode_id: str
    site: Site
    day: DayData
    tariff: Tariff
    goal: str


@dataclass
class ArmRun:
    """What an LLM arm hands to the harness. `battery_kw=None` means the A0 fallback runs."""

    arm: str
    battery_kw: list[float] | None
    export_limit_kw: list[float] | None
    protected_soc_kwh: list[float] | None
    plan_source: str
    fallback_used: bool
    stop_reason: str
    explanation: str | None
    trace: Trace
    totals: LLMTotals
    tool_calls: int
    repair_turns: int
    spec: IntentSpec | None = None
    metrics: dict[str, Any] = field(default_factory=dict)  # extra EpisodeResult fields (A4 gate)
    import_limit_kw: list[float] | None = None  # None: the site's import limit every hour


# --- argument models -----------------------------------------------------------------------


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


ChecklistItem = Annotated[str, StringConstraints(min_length=1, max_length=120)]


class ChecklistArgs(_Args):
    items: list[ChecklistItem] = Field(
        min_length=1, max_length=8, description="what the goal requires, one requirement per item, in plain words"
    )


class SolvePlanArgs(_Args):
    spec: dict[str, Any]  # validated by parse_spec so every error comes back at once; schema: IntentSpec


class CoverageEntry(_Args):
    item: int = Field(ge=1)
    maps_to: str = Field(
        description='a spec field of the plan, e.g. "objective", "grid_outages[0]", "min_soc_pct_all_day", or "assumptions[i]"'
    )


class FinalizeArgs(_Args):
    plan_id: str
    explanation: str = Field(max_length=2000)
    assumptions: list[str]
    coverage: list[CoverageEntry] = Field(description="one entry per checklist item")


Schedule = Annotated[list[float], Field(min_length=HOURS, max_length=HOURS, description="24 hourly setpoints in kW, > 0 charges, < 0 discharges")]


class SimulateArgs(_Args):
    battery_kw: Schedule


class ScheduleFinalizeArgs(_Args):
    battery_kw: Schedule
    explanation: str = Field(max_length=2000)
    no_export_hours: list[DeclWindow] = Field(default_factory=list, description="windows when the site must not export")
    reserve: list[DeclReserve] = Field(default_factory=list, description="windows with a minimum SOC to protect")
    import_caps: list[DeclImportCap] = Field(default_factory=list, description="windows with a grid import cap in kW")


def intent_spec_schema() -> dict[str, Any]:
    schema = args_schema(IntentSpec)
    for name, description in SPEC_FIELD_DESCRIPTIONS.items():
        schema["properties"][name]["description"] = description
    return schema


def solve_plan_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"spec": intent_spec_schema()},
        "required": ["spec"],
        "additionalProperties": False,
    }


# --- info tools ----------------------------------------------------------------------------


def site_info(ctx: EpisodeContext, *, glossary: bool = True) -> dict[str, Any]:
    site, battery, grid = ctx.site, ctx.site.battery, ctx.site.grid
    info: dict[str, Any] = {
        "name": site.name,
        "site_type": site.site_type,
        "country": site.country,
        "lat": site.lat,
        "lon": site.lon,
        "timezone": site.timezone,
        "currency": site.currency,
        "pv_capacity_kwp": site.pv.capacity_kwp,
        "battery": {
            "capacity_kwh": battery.capacity_kwh,
            "min_soc_pct": _pct(battery.soc_min_frac * 100),
            "max_soc_pct": _pct(battery.soc_max_frac * 100),
            "initial_soc_pct": _pct(battery.initial_soc_frac * 100),
            "max_charge_kw": battery.p_charge_max_kw,
            "max_discharge_kw": battery.p_discharge_max_kw,
            "charge_efficiency": battery.eta_charge,
            "discharge_efficiency": battery.eta_discharge,
            "degradation_cost_per_kwh": battery.degradation_cost_per_kwh,
        },
        "grid": {"import_max_kw": grid.import_max_kw, "export_allowed": grid.export_allowed, "export_max_kw": grid.export_max_kw},
        "critical_load": {"name": site.critical_load_name, "kw": site.critical_load_kw},
    }
    if glossary:
        info["glossary"] = list(GLOSSARY)
    return info


def forecast_info(ctx: EpisodeContext) -> dict[str, Any]:
    day = ctx.day
    return {
        "note": "forecast for the planned day; actual weather and load will differ",
        "pv_kw": _kw_list(day.pv_forecast_kw),
        "load_kw": _kw_list(day.load_forecast_kw),
        "critical_kw": _kw_list(day.critical_load_kw),
        "pv_kwh_total": _kw(sum(day.pv_forecast_kw)),
        "load_kwh_total": _kw(sum(day.load_forecast_kw)),
    }


def tariff_info(ctx: EpisodeContext) -> dict[str, Any]:
    tariff = ctx.tariff
    return {
        "name": tariff.name,
        "kind": tariff.kind,
        "currency": tariff.currency,
        "import_price_per_kwh": [_price(p) for p in ctx.day.import_price_per_kwh],
        "export_price_per_kwh": [_price(p) for p in ctx.day.export_price_per_kwh],
        "periods": {name: [list(w) for w in hours_to_windows(hours)] for name, hours in sorted(tariff.periods.items())},
    }


def info_tools(ctx: EpisodeContext) -> list[Tool]:
    return [
        Tool(
            "get_site_info",
            "Site type, location, timezone, currency, battery, grid, the critical load and the conventions glossary.",
            NoArgs,
            lambda _args: site_info(ctx),
        ),
        Tool(
            "get_forecast",
            "Hourly PV, load and critical-load forecast for the planned day (kW, hours 0-23 local).",
            NoArgs,
            lambda _args: forecast_info(ctx),
        ),
        Tool(
            "get_tariff",
            "Hourly import and export prices, currency, tariff kind and the named periods for the day.",
            NoArgs,
            lambda _args: tariff_info(ctx),
        ),
    ]


# --- A4: checklist and solve_plan ----------------------------------------------------------


@dataclass(frozen=True)
class PlanRecord:
    plan_id: str
    plan: Plan
    check: VerifyResult  # with the spec, on the forecast
    physics_valid: bool

    @property
    def spec(self) -> IntentSpec:
        return self.plan.spec

    @property
    def fully_verified(self) -> bool:
        return self.check.valid


class PlanBook:
    """A4's checklist and every plan it solved, by plan_id."""

    def __init__(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.checklist: list[str] | None = None
        self.records: dict[str, PlanRecord] = {}

    def set_checklist(self, args: ChecklistArgs) -> dict[str, Any]:
        if self.records:
            return {"error": "the checklist is fixed once you have solved a plan; map its items in finalize instead"}
        self.checklist = list(args.items)
        return {"checklist": [{"item": i, "text": text} for i, text in enumerate(self.checklist, start=1)]}

    def solve(self, args: SolvePlanArgs) -> dict[str, Any]:
        if self.checklist is None:
            return {"error": "call set_checklist first: list what the goal requires before solving"}
        spec, errors = parse_spec(args.spec, self.ctx.site)
        if spec is None:
            return {"error": "invalid spec", "spec_errors": [error.message for error in errors]}
        ctx = self.ctx
        plan = solve(ctx.day, ctx.site, spec, use="forecast")
        check = verify(plan.battery_kw, ctx.day, ctx.site, spec, "forecast", export_limit_kw=plan.export_limit_kw)
        physics = verify(plan.battery_kw, ctx.day, ctx.site, spec, "forecast", physics_only=True, export_limit_kw=plan.export_limit_kw)
        plan_id = f"p{len(self.records) + 1}"
        self.records[plan_id] = PlanRecord(plan_id, plan, check, physics.valid)
        return plan_summary(plan_id, plan, check, ctx.site)

    def cheapest(self, predicate: Callable[[PlanRecord], bool]) -> PlanRecord | None:
        candidates = [record for record in self.records.values() if predicate(record)]
        return min(candidates, key=lambda r: r.plan.energy_cost) if candidates else None

    def last(self) -> PlanRecord | None:
        return list(self.records.values())[-1] if self.records else None


def plan_summary(plan_id: str, plan: Plan, check: VerifyResult, site: Site) -> dict[str, Any]:
    capacity = site.battery.capacity_kwh
    flows = plan.flows
    soc = plan.soc_kwh
    report: dict[str, float] = {"start": _pct(plan.initial_soc_kwh / capacity * 100)}
    for hour in soc_report_hours(plan.spec, site):
        report[str(hour)] = _pct(soc[hour] / capacity * 100)
    low_hour = min(range(HOURS), key=lambda t: soc[t])
    return {
        "plan_id": plan_id,
        "valid": check.valid,
        "total_violations": check.total_violations,
        "violations": [v.message for v in check.violations[:MAX_LISTED_VIOLATIONS]],
        "shortfalls": [shortfall_entry(s, capacity) for s in plan.shortfalls],
        "energy_cost": _money(plan.energy_cost),
        "currency": site.currency,
        "grid_import_kwh": _kw(sum(flows.grid_import_kw)),
        "grid_export_kwh": _kw(sum(flows.grid_export_kw)),
        "curtailed_kwh": _kw(sum(flows.curtail_kw)),
        "unserved_critical_kwh": _kw(sum(flows.unserved_critical_kw)),
        "unserved_noncritical_kwh": _kw(sum(flows.unserved_noncritical_kw)),
        "soc_pct_end_of_hour": report,
        "soc_pct_lowest": {"hour": low_hour, "pct": _pct(soc[low_hour] / capacity * 100)},
    }


def shortfall_entry(shortfall: Any, capacity_kwh: float) -> dict[str, Any]:
    entry: dict[str, Any] = {"goal": shortfall.path, "rule": shortfall.rule}
    if shortfall.unit == "kWh":
        entry["short_by_kwh"] = _kw(shortfall.amount)
        entry["short_by_pct"] = _pct(shortfall.amount / capacity_kwh * 100)
    else:
        entry["short_by_kw"] = _kw(shortfall.amount)
    entry["hours"] = shortfall.hours
    entry["message"] = shortfall.message
    return entry


def soc_report_hours(spec: IntentSpec, site: Site) -> list[int]:
    """Every 3rd hour plus the hours the spec's SOC goals and outages constrain (the all-day
    minimum is covered by `soc_pct_lowest`)."""
    hours = set(SOC_REPORT_HOURS)
    for row in soft_rows(spec, site):
        if row.kind == "soc_min" and row.rule != "ALL_DAY_MIN" and row.hour is not None and row.hour != INITIAL:
            hours.add(row.hour)
    for hour in outage_hours(spec):
        hours.update(h for h in (hour - 1, hour) if h >= 0)
    return sorted(hours)


def a4_tools(ctx: EpisodeContext, book: PlanBook, finalize: Callable[[FinalizeArgs], dict[str, Any] | ToolOutcome]) -> list[Tool]:
    return [
        Tool(
            "set_checklist",
            "Your first step: the list of what the operator's goal requires, in plain words.",
            ChecklistArgs,
            book.set_checklist,
        ),
        *info_tools(ctx),
        Tool(
            "solve_plan",
            "Validate the spec, compute the cheapest plan that meets it on the forecast, verify it, and store it under a plan_id.",
            SolvePlanArgs,
            book.solve,
            parameters=solve_plan_parameters(),
        ),
        Tool(
            "finalize",
            "Submit plan_id, explanation, assumptions and checklist coverage. Runs mandatory checks; the only way to end.",
            FinalizeArgs,
            finalize,
        ),
    ]


# --- A3: simulate_plan ---------------------------------------------------------------------


@dataclass(frozen=True)
class SimRecord:
    sim_id: str
    battery_kw: list[float]
    check: VerifyResult


def simulate_schedule(battery_kw: list[float], ctx: EpisodeContext, export_limit_kw: list[float] | None = None) -> VerifyResult:
    """Physics-only check of a raw schedule on the forecast (no goal rules, no outages)."""
    return verify(battery_kw, ctx.day, ctx.site, None, "forecast", physics_only=True, export_limit_kw=export_limit_kw)


class SimBook:
    def __init__(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.records: list[SimRecord] = []

    def simulate(self, args: SimulateArgs) -> dict[str, Any]:
        check = simulate_schedule(args.battery_kw, self.ctx)
        sim_id = f"s{len(self.records) + 1}"
        self.records.append(SimRecord(sim_id, list(args.battery_kw), check))
        return sim_summary(sim_id, check, self.ctx.site)

    def cheapest_valid(self) -> SimRecord | None:
        valid = [r for r in self.records if r.check.valid]
        return min(valid, key=lambda r: r.check.energy_cost or 0.0) if valid else None


def sim_summary(sim_id: str, check: VerifyResult, site: Site) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sim_id": sim_id,
        "valid": check.valid,
        "total_violations": check.total_violations,
        "violations": [v.message for v in check.violations[:MAX_LISTED_VIOLATIONS]],
    }
    if check.soc_kwh is None or check.flows is None:
        return out
    flows = check.flows
    out.update(
        {
            "soc_pct_end_of_hour": [_pct(e / site.battery.capacity_kwh * 100) for e in check.soc_kwh],
            "energy_cost": _money(check.energy_cost or 0.0),
            "currency": site.currency,
            "grid_import_kwh": _kw(sum(flows.grid_import_kw)),
            "grid_export_kwh": _kw(sum(flows.grid_export_kw)),
            "curtailed_kwh": _kw(sum(flows.curtail_kw)),
            "unserved_critical_kwh": _kw(sum(flows.unserved_critical_kw)),
            "unserved_noncritical_kwh": _kw(sum(flows.unserved_noncritical_kw)),
        }
    )
    return out


def a3_tools(ctx: EpisodeContext, sims: SimBook, finalize: Callable[[ScheduleFinalizeArgs], dict[str, Any] | ToolOutcome]) -> list[Tool]:
    return [
        *info_tools(ctx),
        Tool(
            "simulate_plan",
            "Physics simulation of 24 battery setpoints on the forecast: violations, SOC at the end of every hour, cost. Does not check the goal.",
            SimulateArgs,
            sims.simulate,
        ),
        Tool(
            "finalize",
            "Submit the final 24 setpoints and the explanation, with optional no_export_hours, reserve and import_caps declarations. Ends the episode.",
            ScheduleFinalizeArgs,
            finalize,
        ),
    ]


# --- rounding --------------------------------------------------------------------------------


def _kw(value: float) -> float:
    return _clean(round(float(value), 3))


def _kw_list(values: list[float]) -> list[float]:
    return [_kw(v) for v in values]


def _price(value: float) -> float:
    return _clean(round(float(value), 4))


def _money(value: float) -> float:
    return _clean(round(float(value), 2))


def _pct(value: float) -> float:
    return _clean(round(float(value), 1))


def _clean(value: float) -> float:
    return 0.0 if value == 0 else value  # no "-0.0" in outputs
