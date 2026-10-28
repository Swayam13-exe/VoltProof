from __future__ import annotations

import json

from tests.fakes_core import simple_site
from voltproof.core.spec import parse_spec, soft_rows, spec_items, window_soc_indices
from voltproof.models import IntentSpec


def messages(data, site=None) -> list[str]:
    spec, errors = parse_spec(data, site)
    assert spec is None
    return [e.message for e in errors]


def test_valid_spec_and_defaults() -> None:
    spec, errors = parse_spec({"grid_outages": [{"start_hour": 14, "end_hour": 17}]})
    assert errors == [] and spec is not None
    assert spec.objective == "min_cost" and spec.reserve_windows == [] and spec.discharge_only_windows is None


def test_range_message_has_path_value_and_rule() -> None:
    assert messages({"reserve_windows": [{"start_hour": 18, "end_hour": 25, "min_soc_pct": 50}]}) == [
        "reserve_windows[0].end_hour=25 must be <= 24"
    ]


def test_all_errors_reported_at_once() -> None:
    errors = messages(
        {
            "reserve_windows": [{"start_hour": -1, "end_hour": 25, "min_soc_pct": 150}],
            "soc_targets": [{"by_hour": 0, "min_soc_pct": 95}],
            "import_caps": [{"start_hour": 3, "end_hour": 4, "max_kw": -2}],
            "max_throughput_kwh": -5,
            "colour": "blue",
        }
    )
    assert "reserve_windows[0].start_hour=-1 must be >= 0" in errors
    assert "reserve_windows[0].end_hour=25 must be <= 24" in errors
    assert "reserve_windows[0].min_soc_pct=150 must be <= 100" in errors
    assert "soc_targets[0].by_hour=0 must be >= 1" in errors
    assert "import_caps[0].max_kw=-2 must be >= 0" in errors
    assert "max_throughput_kwh=-5 must be >= 0" in errors
    assert "colour='blue' is not a recognised field" in errors
    assert len(errors) == 7


def test_window_order_message() -> None:
    assert messages({"grid_outages": [{"start_hour": 17, "end_hour": 14}]}) == [
        "grid_outages[0]: start_hour=17 must be < end_hour=14"
    ]


def test_type_and_missing_messages() -> None:
    errors = messages({"soc_targets": [{"min_soc_pct": "lots"}], "objective": "max_profit"})
    assert "soc_targets[0].by_hour: is required" in errors
    assert "soc_targets[0].min_soc_pct='lots' must be a number" in errors
    assert any(e.startswith("objective='max_profit' must be 'min_cost'") for e in errors)


def test_nan_and_infinity_rejected() -> None:
    errors = messages({"min_soc_pct_all_day": float("nan"), "max_throughput_kwh": float("inf")})
    assert any("min_soc_pct_all_day" in e and "finite" in e for e in errors)
    assert any("max_throughput_kwh" in e and "finite" in e for e in errors)


def test_site_aware_soc_limit() -> None:
    site = simple_site().model_copy(
        update={"battery": simple_site().battery.model_copy(update={"soc_max_frac": 0.9})}
    )
    errors = messages({"end_of_day_min_soc_pct": 95, "soc_targets": [{"by_hour": 18, "min_soc_pct": 92}]}, site)
    assert errors == [
        "soc_targets[0].min_soc_pct=92.0 must be <= 90 (the battery's maximum state of charge, %)",
        "end_of_day_min_soc_pct=95.0 must be <= 90 (the battery's maximum state of charge, %)",
    ]


def test_json_schema_round_trip() -> None:
    schema = IntentSpec.model_json_schema()
    assert schema["additionalProperties"] is False and "reserve_windows" in schema["properties"]
    spec = IntentSpec.model_validate({"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]})
    assert IntentSpec.model_validate(json.loads(spec.model_dump_json())) == spec


def test_window_semantics() -> None:
    assert window_soc_indices(18, 22) == [17, 18, 19, 20, 21]
    assert window_soc_indices(0, 2) == [-1, 0, 1]  # −1 = initial SOC


def test_soft_rows_cover_every_goal() -> None:
    site = simple_site(capacity_kwh=20)
    spec = IntentSpec.model_validate(
        {
            "reserve_windows": [{"start_hour": 18, "end_hour": 20, "min_soc_pct": 50}],
            "soc_targets": [{"by_hour": 6, "min_soc_pct": 95}],
            "min_soc_pct_all_day": 20,
            "end_of_day_min_soc_pct": 30,
            "import_caps": [{"start_hour": 12, "end_hour": 14, "max_kw": 3}],
            "no_export_windows": [{"start_hour": 11, "end_hour": 13}],
            "discharge_only_windows": [{"start_hour": 17, "end_hour": 23}],
            "max_throughput_kwh": 15,
        }
    )
    rows = soft_rows(spec, site)
    by_rule: dict[str, list] = {}
    for row in rows:
        by_rule.setdefault(row.rule, []).append(row)
    assert [(r.hour, r.bound) for r in by_rule["RESERVE"]] == [(17, 10.0), (18, 10.0), (19, 10.0)]
    assert [(r.hour, r.bound) for r in by_rule["SOC_TARGET"]] == [(5, 19.0)]
    assert len(by_rule["ALL_DAY_MIN"]) == 24 and by_rule["END_SOC"][0].hour == 23
    assert [r.hour for r in by_rule["IMPORT_CAP"]] == [12, 13]
    assert [r.hour for r in by_rule["NO_EXPORT"]] == [11, 12]
    assert len(by_rule["DISCHARGE_WINDOW"]) == 24 - 6
    assert by_rule["THROUGHPUT"][0].bound == 15


def test_spec_items_merge_windows() -> None:
    split = IntentSpec.model_validate({"grid_outages": [{"start_hour": 15, "end_hour": 17}, {"start_hour": 14, "end_hour": 15}]})
    whole = IntentSpec.model_validate({"grid_outages": [{"start_hour": 14, "end_hour": 17}]})
    assert spec_items(split) == spec_items(whole)
    different = IntentSpec.model_validate(
        {
            "reserve_windows": [
                {"start_hour": 16, "end_hour": 18, "min_soc_pct": 50},
                {"start_hour": 18, "end_hour": 20, "min_soc_pct": 60},
            ]
        }
    )
    assert len(spec_items(different)) == 2  # different values are not merged
