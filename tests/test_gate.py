"""A4 finalize checks as units: number extraction, grounding, verified plan, coverage."""

from __future__ import annotations

from tests.fakes_llm import agent_site, make_context
from voltproof.agents.gate import (
    NumberPool,
    build_pool,
    check_coverage,
    check_verified_plan,
    extract_numbers,
    grounding_message,
    shortfall_stated,
    ungrounded_numbers,
)
from voltproof.agents.tools import ChecklistArgs, CoverageEntry, PlanBook, PlanRecord, SolvePlanArgs
from voltproof.models import IntentSpec, Shortfall


def texts(text: str) -> list[str]:
    return [n.text for n in extract_numbers(text)]


def test_extract_numbers_times_units_and_separators() -> None:
    numbers = extract_numbers("Grid off 14:00-17:00, reserve 50% from 6 PM; cost 1,385.42 and 2,5 kWh; plan p2")
    by_text = {n.text: n for n in numbers}
    assert by_text["14:00"].hour == 14 and by_text["17:00"].hour == 17 and by_text["6 PM"].hour == 18
    assert by_text["50%"].candidates == (50.0,)
    assert by_text["1,385.42"].candidates == (1385.42,)
    assert by_text["2,5 kWh"].candidates == (2.5,)
    assert "2" not in by_text  # "p2" is an identifier, not a number
    assert extract_numbers("12 AM")[0].hour == 0 and extract_numbers("12 p.m.")[0].hour == 12
    assert set(extract_numbers("2,500")[0].candidates) == {2.5, 2500.0}


def test_non_ascii_digits_are_read() -> None:
    (number,) = extract_numbers("शाम ६ बजे")
    assert number.candidates == (6.0,)


def test_grounding_uses_tool_outputs_goal_and_spec() -> None:
    spec = IntentSpec.model_validate({"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]})
    pool = build_pool([{"energy_cost": 1385.42, "soc_pct_end_of_hour": {"17": 52.0}}], "keep half from 6 PM", spec)
    assert ungrounded_numbers("Cost about 1,385 with 52% at 17:00, reserve 50% from 18:00 to 22:00 (6 PM).", pool) == []
    assert ungrounded_numbers("It saves 42.3 kWh and ends at 25:00.", pool) == ["42.3 kWh", "25:00"]
    assert grounding_message(["42.3 kWh", "18%"]) == (
        'these numbers in the explanation do not match any tool output in this episode or the goal: "42.3 kWh", "18%". '
        "Quote numbers as the tools reported them (rounding within 1% is fine; hours exactly), or remove them."
    )


def test_one_percent_tolerance() -> None:
    pool = NumberPool({100.0})
    assert not ungrounded_numbers("99.1", pool) and ungrounded_numbers("98.9", pool) == ["98.9"]


def shortfall(amount: float) -> Shortfall:
    return Shortfall(path="soc_targets[0]", rule="SOC_TARGET", hours=[0], amount=amount, unit="kWh", message="m")


def test_shortfall_stated_within_5_percent_or_whole_unit() -> None:
    capacity = 10.0
    assert shortfall_stated(shortfall(3.6), extract_numbers("short by 3.5 kWh"), capacity)  # within 5 %
    assert shortfall_stated(shortfall(3.6), extract_numbers("about 4 kWh short"), capacity)  # rounded to whole
    assert shortfall_stated(shortfall(3.6), extract_numbers("36% short"), capacity)  # percent of capacity
    assert not shortfall_stated(shortfall(3.6), extract_numbers("short by 3 kWh"), capacity)
    assert not shortfall_stated(shortfall(0.3), extract_numbers("0 kWh"), capacity)  # rounding to 0 does not count


def infeasible_record() -> PlanRecord:
    book = PlanBook(make_context(site=agent_site(power_kw=1.0)))
    book.set_checklist(ChecklistArgs(items=["full by 1 AM"]))
    book.solve(SolvePlanArgs(spec={"soc_targets": [{"by_hour": 1, "min_soc_pct": 100}]}))
    return book.records["p1"]


def test_verified_plan_messages() -> None:
    assert check_verified_plan("p9", None, "", 10.0) == [
        'plan_id "p9" does not exist. Call solve_plan with your spec and finalize with the plan_id it returns.'
    ]
    record = infeasible_record()
    (message,) = check_verified_plan("p1", record, "The battery will be charged as far as possible.", 10.0)
    assert message == (
        "plan p1 cannot fully meet 1 goal and the explanation does not say by how much: SOC_TARGET (soc_targets[0]) "
        "short by up to 4.000000 kWh (40.000000% of capacity) at hours [0]. Say in the explanation what falls short and "
        "by how much (quote the shortfall from solve_plan), or fix the spec if you misread the goal."
    )
    assert check_verified_plan("p1", record, "It will be 4 kWh short of full at 1 AM.", 10.0) == []


def test_physics_invalid_plan_is_not_verified() -> None:
    record = infeasible_record()
    broken = PlanRecord(record.plan_id, record.plan, record.check, physics_valid=False)
    (message,) = check_verified_plan("p1", broken, "x", 10.0)
    assert message.startswith("plan p1 failed physics verification") and message.endswith("Change the spec and call solve_plan again.")


SPEC = IntentSpec.model_validate(
    {"grid_outages": [{"start_hour": 14, "end_hour": 17}], "min_soc_pct_all_day": 20}
)
CHECKLIST = ["grid off 14:00-17:00", "keep the fridge running", "keep the bill low"]


def entries(*pairs: tuple[int, str]) -> list[CoverageEntry]:
    return [CoverageEntry(item=i, maps_to=ref) for i, ref in pairs]


def test_full_coverage() -> None:
    result = check_coverage(CHECKLIST, entries((1, "grid_outages[0]"), (2, "assumptions[0]"), (3, "objective")), SPEC, ["critical first"], "p1")
    assert result.messages == [] and result.dropped == [] and result.coverage == 1.0


def test_unmapped_item_message() -> None:
    result = check_coverage(CHECKLIST, entries((1, "grid_outages[0]"), (2, "min_soc_pct_all_day")), SPEC, [], "p2")
    assert result.messages == [
        'checklist item 3 ("keep the bill low") is not mapped. Map every item to a spec field of plan p2 '
        '(e.g. "objective", "grid_outages[0]") or to one of your assumptions ("assumptions[0]").'
    ]
    assert result.dropped == ["keep the bill low"] and result.coverage == 2 / 3


def test_bad_references() -> None:
    result = check_coverage(
        CHECKLIST + ["a", "b"],
        entries((1, "reserve_windows[1]"), (2, "end_of_day_min_soc_pct"), (3, "assumptions[4]"), (4, "foo"), (5, "objective"), (7, "objective")),
        SPEC,
        ["x", "y"],
        "p2",
    )
    assert result.messages == [
        "coverage refers to checklist item 7, which does not exist (items are 1-5).",
        'checklist item 1 maps to "reserve_windows[1]", but plan p2\'s spec has no reserve_windows[1].',
        "checklist item 2 maps to \"end_of_day_min_soc_pct\", but end_of_day_min_soc_pct is not set in plan p2's spec.",
        'checklist item 3 maps to "assumptions[4]", but there is no assumptions[4] (you passed 2 assumptions).',
        'checklist item 4 maps to "foo", but "foo" is not a spec field or assumptions[i].',
    ]
    assert result.coverage == 1 / 5


def test_any_valid_entry_maps_an_item() -> None:
    result = check_coverage(["x"], entries((1, "foo"), (1, "spec.grid_outages")), SPEC, [], "p1")
    assert result.messages == [] and result.coverage == 1.0
