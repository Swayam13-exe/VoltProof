"""The vision feasibility probe (SPEC §6.12): sample bills, request shape and scoring. Offline, no LLM call."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "vision_samples"


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vision_probe", ROOT / "scripts" / "vision_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vision_probe"] = module
    spec.loader.exec_module(module)
    return module


def truth(name: str) -> dict:
    return json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def test_samples_come_in_clean_and_photo_pairs_with_valid_ground_truth(probe: ModuleType) -> None:
    names = ("bill_1_tou_inr", "bill_2_flat_kes", "bill_3_tou_eur")
    assert [f.name for f in probe.sample_files()] == [f"{n}.png" for n in names] + [f"{n}_photo.jpg" for n in names]
    assert {truth(n)["currency"] for n in names} == {"INR", "KES", "EUR"}
    for name in names:
        t = truth(name)
        assert probe.hourly(t["periods"]) is not None  # covers all 24 hours once
        assert t["meta"]["utility"].endswith("(fictional)")
        assert probe.truth_for(SAMPLES / f"{name}_photo.jpg") == t
        assert all((SAMPLES / f).stat().st_size < 300_000 for f in (f"{name}.png", f"{name}_photo.jpg"))
    assert truth("bill_2_flat_kes")["export_rate"] is None and truth("bill_2_flat_kes")["tariff_kind"] == "flat"


def test_request_is_an_openai_style_image_message(probe: ModuleType) -> None:
    messages = probe.messages_for(SAMPLES / "bill_1_tou_inr_photo.jpg")
    assert [m["role"] for m in messages] == ["system", "user"]
    text_part, image_part = messages[1]["content"]
    assert text_part == {"type": "text", "text": probe.INSTRUCTIONS}
    assert image_part["type"] == "image_url" and image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert probe.messages_for(SAMPLES / "bill_1_tou_inr.png")[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_perfect_answer_scores_all_correct(probe: ModuleType) -> None:
    for name in ("bill_1_tou_inr", "bill_2_flat_kes", "bill_3_tou_eur"):
        t = truth(name)
        scored = probe.score({k: t[k] for k in probe.FIELDS}, t)
        assert scored["all_correct"] and scored["hours_correct"] == 24 and scored["wrong_hours"] == []


def test_score_catches_the_mistakes_the_confirm_step_must_catch(probe: ModuleType) -> None:
    t = truth("bill_1_tou_inr")
    good = {k: t[k] for k in probe.FIELDS}

    wrong_rate = copy.deepcopy(good)
    wrong_rate["periods"][2]["import_rate"] = 9.0  # 18:00-22:00 read as 9.00 instead of 9.90
    scored = probe.score(wrong_rate, t)
    assert not scored["fields"]["periods"] and scored["wrong_hours"] == [18, 19, 20, 21] and not scored["boundary_off_by_one"]

    off_by_one = copy.deepcopy(good)
    off_by_one["periods"][0]["end_hour"], off_by_one["periods"][1]["start_hour"] = 7, 7  # night ends at 07:00, not 06:00
    scored = probe.score(off_by_one, t)
    assert scored["wrong_hours"] == [6] and scored["boundary_off_by_one"] and scored["periods_well_formed"]

    wrong_currency = dict(good, currency="USD")
    assert probe.score(wrong_currency, t)["fields"]["currency"] is False

    cents = copy.deepcopy(good)
    for p in cents["periods"]:
        p["import_rate"] *= 100  # ct instead of the main currency unit
    assert probe.score(cents, t)["hours_correct"] == 0

    gap = copy.deepcopy(good)
    del gap["periods"][3]  # 22:00-24:00 missing
    assert not probe.score(gap, t)["periods_well_formed"]
    overlap = copy.deepcopy(good)
    overlap["periods"][1]["start_hour"] = 5
    assert not probe.score(overlap, t)["periods_well_formed"]

    assert probe.score(dict(good, export_rate=None), t)["fields"]["export_rate"] is False
    assert probe.score(dict(good, fixed_charge=45.0), t)["fields"]["fixed_charge"] is False
    kes = truth("bill_2_flat_kes")
    assert probe.score({k: kes[k] for k in probe.FIELDS} | {"export_rate": 0.0}, kes)["fields"]["export_rate"] is False  # "no export" is null, not 0


def test_parse_separates_strict_json_from_fenced_and_invalid(probe: ModuleType) -> None:
    assert probe.parse('{"a": 1}') == ({"a": 1}, "strict")
    assert probe.parse('```json\n{"a": 1}\n```') == ({"a": 1}, "fenced")
    assert probe.parse('Here you go: {"a": 1} hope it helps')[1] == "fenced"
    assert probe.parse("not json") == (None, "invalid") and probe.parse(None) == (None, "invalid")
