"""Pydantic v2 data models (SPEC §6.1). P1 subset: equipment, tariffs, sites and DayData.

Every 24-value array is checked for length and finiteness. Money fields carry no
currency code in their names; the currency lives on the tariff (SPEC §3).
"""

from __future__ import annotations

import math
from datetime import date as Date
from typing import Annotated, Literal, Union
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
