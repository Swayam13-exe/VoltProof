"""Pydantic v2 data models (SPEC §6.1). P1 subset: equipment, tariffs, sites and DayData.

Every 24-value array is checked for length and finiteness. Money fields carry no
currency code in their names; the currency lives on the tariff (SPEC §3).
"""

from __future__ import annotations

import math
from datetime import date as Date
from typing import Annotated, Any, Literal, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

HOURS = 24
TOL = 1e-6


def _check_array24(values: list[float]) -> list[float]:
    if len(values) != HOURS:
        raise ValueError(f"must have {HOURS} hourly values, got {len(values)}")
    bad = [hour for hour, value in enumerate(values) if not math.isfinite(value)]
    if bad:
        raise ValueError(f"values must be finite; non-finite at hours {bad}")
    return values


Array24 = Annotated[list[float], AfterValidator(_check_array24)]
Hour = Annotated[int, Field(ge=0, le=23)]
SiteType = Literal["commercial_cold_store", "health_clinic", "household"]
TariffKind = Literal["time_of_use", "hourly_dynamic", "flat"]
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HourWindow(_Model):
    """Half-open hour range [start_hour, end_hour) in site-local time."""

    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24)

    @model_validator(mode="after")
    def _ordered(self) -> HourWindow:
        if self.start_hour >= self.end_hour:
            raise ValueError(f"start_hour={self.start_hour} must be < end_hour={self.end_hour}")
        return self

    def hours(self) -> list[int]:
        return list(range(self.start_hour, self.end_hour))


def window_hours(windows: list[HourWindow]) -> list[int]:
    """Sorted hours covered by `windows`; raises if two windows overlap."""
    covered: list[int] = []
    for window in windows:
        overlap = set(covered) & set(window.hours())
        if overlap:
            raise ValueError(f"windows overlap at hours {sorted(overlap)}")
        covered.extend(window.hours())
    return sorted(covered)


class PVSpec(_Model):
    capacity_kwp: float = Field(gt=0)
    performance_ratio: float = Field(default=0.80, gt=0, le=1)
    temp_coeff_per_c: float = Field(default=-0.004, ge=-0.02, le=0)
    noct_c: float = Field(default=45.0, gt=20, lt=80)


class BatterySpec(_Model):
    capacity_kwh: float = Field(gt=0)
    soc_min_frac: float = Field(ge=0, lt=1)
    soc_max_frac: float = Field(gt=0, le=1)
    p_charge_max_kw: float = Field(gt=0)
    p_discharge_max_kw: float = Field(gt=0)
    eta_charge: float = Field(gt=0, le=1)
    eta_discharge: float = Field(gt=0, le=1)
    initial_soc_frac: float = Field(ge=0, le=1)
    degradation_cost_per_kwh: float = Field(ge=0)

    @model_validator(mode="after")
    def _soc_bounds(self) -> BatterySpec:
        if self.soc_min_frac >= self.soc_max_frac:
            raise ValueError(f"soc_min_frac={self.soc_min_frac} must be < soc_max_frac={self.soc_max_frac}")
        if not self.soc_min_frac <= self.initial_soc_frac <= self.soc_max_frac:
            raise ValueError(
                f"initial_soc_frac={self.initial_soc_frac} must lie in "
                f"[{self.soc_min_frac}, {self.soc_max_frac}]"
            )
        return self


class GridSpec(_Model):
    import_max_kw: float = Field(gt=0)
    export_max_kw: float = Field(ge=0)
    export_allowed: bool

    @model_validator(mode="after")
    def _export_consistent(self) -> GridSpec:
        if not self.export_allowed and self.export_max_kw != 0:
            raise ValueError("export_max_kw must be 0 when export_allowed is false")
        return self


class Tariff(_Model):
    """Prices for one day. `periods` maps a name (e.g. peak, cheap) to its local hours."""

    name: str
    kind: TariffKind
    currency: Currency
    import_price_per_kwh: Array24
    export_price_per_kwh: Array24
    periods: dict[str, list[Hour]] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("periods")
    @classmethod
    def _disjoint_periods(cls, periods: dict[str, list[int]]) -> dict[str, list[int]]:
        seen: dict[int, str] = {}
        for name, hours in periods.items():
            if len(set(hours)) != len(hours):
                raise ValueError(f"period {name!r} repeats an hour")
            for hour in hours:
                if hour in seen:
                    raise ValueError(f"hour {hour} is in both {seen[hour]!r} and {name!r}")
                seen[hour] = name
        return periods


class _TemplateBase(_Model):
    name: str
    currency: Currency
    export_price_per_kwh: float = Field(ge=0)
    notes: str = ""


class TimeOfUseTemplate(_TemplateBase):
    """Fixed prices by named window: price = base · multiplier[period]; uncovered hours are `default_period`."""

    kind: Literal["time_of_use"] = "time_of_use"
    base_price_per_kwh: float = Field(gt=0)
    default_period: str
    windows: dict[str, list[HourWindow]]
    multipliers: dict[str, float]

    @model_validator(mode="after")
    def _consistent(self) -> TimeOfUseTemplate:
        if set(self.windows) != set(self.multipliers):
            raise ValueError("windows and multipliers must name the same periods")
        if self.default_period in self.windows:
            raise ValueError(f"default_period {self.default_period!r} must not have windows")
        if any(m <= 0 for m in self.multipliers.values()):
            raise ValueError("multipliers must be > 0")
        window_hours([w for windows in self.windows.values() for w in windows])
        return self


class FlatTemplate(_TemplateBase):
    kind: Literal["flat"] = "flat"
    price_per_kwh: float = Field(gt=0)


class HourlyDynamicTemplate(_TemplateBase):
    """A price per hour per day: typical shape, solar dip on sunny days, seeded day-to-day variation."""

    kind: Literal["hourly_dynamic"] = "hourly_dynamic"
    mean_price_per_kwh: float = Field(gt=0)
    shape: Array24  # relative daily shape; normalized to mean 1 when used
    solar_dip_windows: list[HourWindow] = Field(default_factory=list)
    solar_dip_frac: float = Field(default=0.0, ge=0, lt=1)
    day_sigma_frac: float = Field(default=0.0, ge=0, lt=1)
    hour_sigma_frac: float = Field(default=0.0, ge=0, lt=1)
    floor_frac: float = Field(default=0.2, gt=0, le=1)
    cheap_hours: int = Field(default=6, ge=1, le=12)
    peak_hours: int = Field(default=4, ge=1, le=12)

    @field_validator("shape")
    @classmethod
    def _positive_shape(cls, shape: list[float]) -> list[float]:
        if min(shape) <= 0:
            raise ValueError("shape values must be > 0")
        return shape


TariffTemplate = Annotated[
    Union[TimeOfUseTemplate, FlatTemplate, HourlyDynamicTemplate], Field(discriminator="kind")
]


class LoadParams(_Model):
    """Synthetic load parameters (non-critical part; the critical load is on the Site).

    Activity is a 0–1 schedule from `activity_windows` on weekdays, scaled on weekends,
    plus optional weekend-only windows. The temperature term is cooling or heating
    depending on the site type (see `voltproof.data.load`).
    """

    base_kw: float = Field(ge=0)
    activity_kw: float = Field(ge=0)
    activity_windows: list[HourWindow]
    saturday_factor: float = Field(ge=0)
    sunday_factor: float = Field(ge=0)
    weekend_extra_windows: list[HourWindow] = Field(default_factory=list)
    weekend_extra_factor: float = Field(default=0.0, ge=0)
    cooling_kw_per_c: float = Field(default=0.0, ge=0)
    cooling_threshold_c: float = 24.0
    cooling_activity_factor: float = Field(default=1.0, ge=1)
    heating_kw_per_c: float = Field(default=0.0, ge=0)
    heating_threshold_c: float = 15.0
    noise_sigma_frac: float = Field(default=0.08, ge=0, lt=1)
    ar1_phi: float = Field(default=0.7, ge=0, lt=1)

    @model_validator(mode="after")
    def _windows_valid(self) -> LoadParams:
        window_hours(self.activity_windows)
        window_hours(self.weekend_extra_windows)
        return self


class OutageProfile(_Model):
    """How often grid outages are realistic at a site; used to generate goal scenarios, not real data."""

    outages_per_month: float = Field(ge=0)
    duration_hours_min: int = Field(ge=1, le=24)
    duration_hours_max: int = Field(ge=1, le=24)
    notes: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> OutageProfile:
        if self.duration_hours_min > self.duration_hours_max:
            raise ValueError("duration_hours_min must be <= duration_hours_max")
        return self


class Site(_Model):
    site_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    site_type: SiteType
    country: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timezone: str
    pv: PVSpec
    battery: BatterySpec
    grid: GridSpec
    tariff_template: TariffTemplate
    critical_load_name: str
    critical_load_kw: float = Field(gt=0)
    load_params: LoadParams
    outage_profile: OutageProfile
    held_out: bool = False  # True: test split only, never used in development

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone {value!r}") from exc
        return value

    @property
    def currency(self) -> str:
        return self.tariff_template.currency


class WeatherSummary(_Model):
    """Daily weather and PV forecast quality. 'Actual' is the model's same-day analysis, not a measurement."""

    ghi_forecast_kwh_per_m2: float = Field(ge=0)
    ghi_actual_kwh_per_m2: float = Field(ge=0)
    clear_sky_index_forecast: float = Field(ge=0)
    clear_sky_index_actual: float = Field(ge=0)
    cloud_cover_forecast_pct: float = Field(ge=0, le=100)
    cloud_cover_actual_pct: float = Field(ge=0, le=100)
    temp_forecast_mean_c: float
    temp_actual_mean_c: float
    pv_forecast_kwh: float = Field(ge=0)
    pv_actual_kwh: float = Field(ge=0)
    pv_error_kwh: float  # forecast − actual
    pv_hourly_mae_kw: float = Field(ge=0)
    pv_hourly_bias_kw: float  # mean(forecast − actual)


Pct = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
RuleCode = Literal[
    "LENGTH",
    "NOT_FINITE",
    "CHARGE_LIMIT",
    "DISCHARGE_LIMIT",
    "SOC_MIN",
    "SOC_MAX",
    "IMPORT_LIMIT",
    "UNSERVED_LOAD",
    "OUTAGE_DEFICIT",
    "RESERVE",
    "SOC_TARGET",
    "ALL_DAY_MIN",
    "END_SOC",
    "IMPORT_CAP",
    "NO_EXPORT",
    "DISCHARGE_WINDOW",
    "THROUGHPUT",
]


class ReserveWindow(HourWindow):
    """Keep SOC ≥ min_soc_pct of capacity during [start_hour, end_hour) (SPEC §3 window semantics)."""

    min_soc_pct: Pct


class SocTarget(_Model):
    """SOC ≥ min_soc_pct of capacity by hour `by_hour`, i.e. at the end of hour by_hour − 1."""

    by_hour: int = Field(ge=1, le=24)
    min_soc_pct: Pct


class ImportCap(HourWindow):
    max_kw: float = Field(ge=0, allow_inf_nan=False)


class IntentSpec(_Model):
    """The language-independent goal: what the optimizer must respect (SPEC §6.3)."""

    objective: Literal["min_cost"] = "min_cost"
    grid_outages: list[HourWindow] = Field(default_factory=list)
    reserve_windows: list[ReserveWindow] = Field(default_factory=list)
    soc_targets: list[SocTarget] = Field(default_factory=list)
    min_soc_pct_all_day: Pct | None = None
    end_of_day_min_soc_pct: Pct | None = None
    import_caps: list[ImportCap] = Field(default_factory=list)
    no_export_windows: list[HourWindow] = Field(default_factory=list)
    discharge_only_windows: list[HourWindow] | None = None
    max_throughput_kwh: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    assumptions: list[str] = Field(default_factory=list)


class Shortfall(_Model):
    """A soft goal the plan cannot fully meet; `amount` is the largest per-hour shortfall."""

    path: str  # e.g. "reserve_windows[0]"
    rule: RuleCode
    hours: list[int]  # −1 = the initial state of charge
    amount: float = Field(ge=0)
    unit: Literal["kWh", "kW"]
    message: str


class Flows(_Model):
    """Hourly energy flows (kW over a 1 h step, so also kWh)."""

    charge_kw: Array24
    discharge_kw: Array24
    grid_import_kw: Array24
    grid_export_kw: Array24
    curtail_kw: Array24
    unserved_noncritical_kw: Array24
    unserved_critical_kw: Array24


class Plan(_Model):
    site_id: str
    date: Date
    use: Literal["forecast", "actual"]
    spec: IntentSpec
    battery_kw: Array24  # > 0 charges, < 0 discharges
    export_limit_kw: Array24  # inverter export limit per hour (0 in no-export windows and outages)
    # import limit per hour for the executor: spec import caps in their hours, the site limit elsewhere
    import_limit_kw: Array24 | None = None
    flows: Flows
    soc_kwh: Array24  # at the END of each hour
    initial_soc_kwh: float
    # SOC the executor may keep by importing (goal protection, SPEC §6.6); None: no SOC goals, no top-up
    protected_soc_kwh: Array24 | None = None
    energy_cost: float  # grid + degradation, site currency
    penalty_cost: float  # unserved load, curtailment and goal penalties
    terminal_value_per_kwh: float  # V = mean import price · η_d
    terminal_value: float  # V · SOC at the end of hour 23
    objective: float  # energy_cost + penalty_cost − terminal_value
    shortfalls: list[Shortfall]
    status: str
    solve_time_s: float = Field(ge=0)
    simultaneous_hours: list[int] = Field(default_factory=list)

    @property
    def unserved_critical_kwh(self) -> float:
        return float(sum(self.flows.unserved_critical_kw))

    @property
    def unserved_noncritical_kwh(self) -> float:
        return float(sum(self.flows.unserved_noncritical_kw))


class Violation(_Model):
    hour: int | None  # None: not tied to an hour; −1: the initial state of charge
    rule: RuleCode
    value: float
    limit: float
    message: str


class VerifyResult(_Model):
    valid: bool
    use: Literal["forecast", "actual"]
    physics_only: bool
    violations: list[Violation]  # at most 20
    total_violations: int = Field(ge=0)
    soc_kwh: Array24 | None = None
    initial_soc_kwh: float
    flows: Flows | None = None
    grid_cost: float | None = None
    degradation_cost: float | None = None
    energy_cost: float | None = None
    unserved_noncritical_kwh: float | None = None
    unserved_critical_kwh: float | None = None


class ExecutionResult(_Model):
    battery_kw: Array24  # setpoints actually applied after clipping and overrides
    export_limit_kw: Array24
    flows: Flows
    soc_kwh: Array24
    initial_soc_kwh: float
    realized_cost: float  # grid + degradation, site currency
    grid_cost: float
    degradation_cost: float
    unserved_noncritical_kwh: float = Field(ge=0)
    unserved_critical_kwh: float = Field(ge=0)
    overrides: int = Field(ge=0)  # outage safety overrides + import-limit overrides
    override_hours: list[int]
    clipped_hours: list[int]  # setpoint clipped to power / SOC limits
    grid_charge_limited_hours: list[int]  # planned charging the plan expected from PV, not taken from the grid
    goal_topup_hours: list[int]  # extra grid charging to keep realized SOC at the plan's protected level
    import_cap_hours: list[int] = Field(default_factory=list)  # hours an import cap changed the setpoint
    solar_absorbed_hours: list[int]  # PV surplus charged instead of curtailed
    outage_hours: list[int]


class ReviewIssue(_Model):
    kind: Literal["missing", "wrong", "unneeded", "malformed"]  # malformed: the reviewer's output, not the spec
    detail: str


class EpisodeResult(_Model):
    """One arm on one site-day-goal. Money in site currency; percentages are currency-free."""

    site_id: str
    date: Date
    arm: str
    spec_name: str
    currency: Currency
    realized_cost: float
    realized_total_cost: float
    savings_vs_rule_pct: float | None
    gap_to_oracle_pct: float | None
    plan_physics_valid: bool
    gold_satisfied: bool
    gold_violations: list[RuleCode]
    unserved_critical_kwh: float
    critical_served_pct: float
    overrides: int
    solve_time_s: float | None = None
    spec_precision: float | None = None
    spec_recall: float | None = None
    language: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_cached: int | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    tool_calls: int | None = None
    repair_turns: int | None = None
    fallback_used: bool = False
    # LLM arms (SPEC §6.7); A4 fills the gate fields
    episode_id: str | None = None
    stop_reason: str | None = None
    plan_source: str | None = None  # finalized | log | fallback | escalated_with_plan | ...
    checklist_coverage: float | None = None
    dropped_requirements: list[str] | None = None
    explanation_grounded: bool | None = None
    ungrounded_numbers: list[str] | None = None
    review_approved: bool | None = None
    review_issues: list[ReviewIssue] | None = None
    review_disputed: bool | None = None
    repair_rounds: int | None = None
    escalated: bool | None = None
    escalation: Literal["with_plan", "fallback"] | None = None
    unknown_tool_calls: int | None = None
    import_cap_hours: int | None = None


TraceKind = Literal["llm_call", "tool_call", "tool_result", "gate", "review", "final"]


class TraceEvent(_Model):
    """One step of an agent episode, for logs and the UI (SPEC §6.9)."""

    kind: TraceKind
    turn: int = Field(ge=0)
    name: str | None = None  # tool name, gate check or arm
    call_id: str | None = None
    summary: str | None = None  # ≤ 200 characters
    data: dict[str, Any] = Field(default_factory=dict)


class DayData(_Model):
    site_id: str
    date: Date
    pv_forecast_kw: Array24
    pv_actual_kw: Array24
    load_forecast_kw: Array24
    load_actual_kw: Array24
    critical_load_kw: Array24
    import_price_per_kwh: Array24
    export_price_per_kwh: Array24
    weather: WeatherSummary

    @model_validator(mode="after")
    def _physical(self) -> DayData:
        for name in ("pv_forecast_kw", "pv_actual_kw", "critical_load_kw"):
            if min(getattr(self, name)) < -TOL:
                raise ValueError(f"{name} must be >= 0")
        for name in ("load_forecast_kw", "load_actual_kw"):
            load = getattr(self, name)
            for hour, (total, critical) in enumerate(zip(load, self.critical_load_kw)):
                if total < critical - TOL:
                    raise ValueError(
                        f"{name}[{hour}]={total:.6f} is below critical_load_kw[{hour}]={critical:.6f}"
                    )
        return self
