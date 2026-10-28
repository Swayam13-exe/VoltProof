from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from tests.fakes_data import make_site
from voltproof.models import (
    BatterySpec,
    DayData,
    GridSpec,
    HourWindow,
    Tariff,
    TimeOfUseTemplate,
    WeatherSummary,
    window_hours,
)

SUMMARY = WeatherSummary(
    ghi_forecast_kwh_per_m2=5.0,
    ghi_actual_kwh_per_m2=4.5,
    clear_sky_index_forecast=0.8,
    clear_sky_index_actual=0.7,
    cloud_cover_forecast_pct=20,
    cloud_cover_actual_pct=30,
    temp_forecast_mean_c=25,
    temp_actual_mean_c=24,
    pv_forecast_kwh=50,
    pv_actual_kwh=45,
    pv_error_kwh=5,
    pv_hourly_mae_kw=0.4,
    pv_hourly_bias_kw=0.2,
)


def day_kwargs(**overrides):
    base = dict(
        site_id="test_site",
        date=date(2024, 6, 1),
        pv_forecast_kw=[0.0] * 24,
        pv_actual_kw=[0.0] * 24,
        load_forecast_kw=[5.0] * 24,
        load_actual_kw=[5.0] * 24,
        critical_load_kw=[2.0] * 24,
        import_price_per_kwh=[10.0] * 24,
        export_price_per_kwh=[1.0] * 24,
        weather=SUMMARY,
    )
    base.update(overrides)
    return base


def test_valid_day_data() -> None:
    day = DayData(**day_kwargs())
    assert len(day.load_actual_kw) == 24 and day.weather.pv_error_kwh == 5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pv_actual_kw", [0.0] * 23, "must have 24 hourly values, got 23"),
        ("load_forecast_kw", [5.0] * 23 + [float("nan")], "non-finite at hours [23]"),
        ("import_price_per_kwh", [float("inf")] + [1.0] * 23, "non-finite at hours [0]"),
        ("load_actual_kw", [1.0] * 24, "below critical_load_kw[0]"),
        ("pv_forecast_kw", [-1.0] + [0.0] * 23, "pv_forecast_kw must be >= 0"),
    ],
)
def test_day_data_rejects_bad_arrays(field: str, value: list[float], message: str) -> None:
    with pytest.raises(ValidationError) as info:
        DayData(**day_kwargs(**{field: value}))
    assert message in str(info.value)


def test_extra_fields_forbidden_and_models_frozen() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DayData(**day_kwargs(unexpected=1))
    day = DayData(**day_kwargs())
    with pytest.raises(ValidationError):
        day.site_id = "other"  # type: ignore[misc]


def test_hour_window_and_overlap() -> None:
    assert HourWindow(start_hour=22, end_hour=24).hours() == [22, 23]
    with pytest.raises(ValidationError, match="must be <"):
        HourWindow(start_hour=5, end_hour=5)
    with pytest.raises(ValueError, match="overlap at hours \\[6\\]"):
        window_hours([HourWindow(start_hour=4, end_hour=7), HourWindow(start_hour=6, end_hour=8)])


def test_battery_soc_bounds() -> None:
    kwargs = dict(
        capacity_kwh=10,
        soc_min_frac=0.1,
        soc_max_frac=1.0,
        p_charge_max_kw=5,
        p_discharge_max_kw=5,
        eta_charge=0.95,
        eta_discharge=0.95,
        initial_soc_frac=0.5,
        degradation_cost_per_kwh=0.1,
    )
    BatterySpec(**kwargs)
    with pytest.raises(ValidationError, match="initial_soc_frac"):
        BatterySpec(**{**kwargs, "initial_soc_frac": 0.05})
    with pytest.raises(ValidationError, match="soc_min_frac"):
        BatterySpec(**{**kwargs, "soc_min_frac": 0.9, "soc_max_frac": 0.8, "initial_soc_frac": 0.85})


def test_grid_export_consistency() -> None:
    with pytest.raises(ValidationError, match="export_max_kw must be 0"):
        GridSpec(import_max_kw=10, export_max_kw=5, export_allowed=False)


def test_tariff_periods_must_be_disjoint_and_currency_iso() -> None:
    kwargs = dict(name="t", kind="flat", currency="XTS", import_price_per_kwh=[1.0] * 24, export_price_per_kwh=[0.0] * 24)
    with pytest.raises(ValidationError, match="hour 3 is in both"):
        Tariff(**kwargs, periods={"a": [1, 2, 3], "b": [3, 4]})
    with pytest.raises(ValidationError):
        Tariff(**{**kwargs, "currency": "rupees"})


def test_time_of_use_template_validation() -> None:
    with pytest.raises(ValidationError, match="same periods"):
        TimeOfUseTemplate(
            name="t",
            currency="XTS",
            base_price_per_kwh=1,
            default_period="normal",
            windows={"peak": [HourWindow(start_hour=1, end_hour=2)]},
            multipliers={},
            export_price_per_kwh=0,
        )


def test_site_rejects_unknown_timezone() -> None:
    with pytest.raises(ValidationError, match="unknown IANA timezone"):
        make_site(timezone="Mars/Olympus_Mons")
