"""The bill reader's evaluation set and scoring (scripts/vision_eval.py). Offline, no LLM call."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from voltproof.vision.validate import check_bill

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ev() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vision_eval", ROOT / "scripts" / "vision_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vision_eval"] = module
    spec.loader.exec_module(module)
    return module


def test_ten_bills_each_with_a_truth_and_a_stated_expectation(ev: ModuleType) -> None:
    files = ev.bill_files()
    assert len(files) == 10 and all(f.stat().st_size < 300_000 for f in files)
    truths = [ev.truth_for(f) for f in files]
    assert sum(t["expect"] == "read" for t in truths) == 7 and sum(t["expect"] == "refuse" for t in truths) == 3
    assert all(t["site_currency"] in ev.REFERENCE for t in truths)
    for t in truths:
        if t["expect"] == "read":
            assert len(ev.truth_hourly(t["tariff"])) == 24 and all(r > 0 for r in ev.truth_hourly(t["tariff"]))


def test_the_truths_score_as_expected(ev: ModuleType) -> None:
    """A model that answered exactly the truth: every 'read' bill is read correctly and every 'refuse'
    bill is refused (by the half-hour or currency validator, or as unreadable)."""
    for f in ev.bill_files():
        t = ev.truth_for(f)
        check = check_bill(t["tariff"] or {"unreadable": True}, t["site_currency"], ev.reference_price(t["site_currency"]))
        assert ev.outcome(t, check)[0] == ("read correctly" if t["expect"] == "read" else "correctly refused"), f.name


def test_wrong_answers_score_as_wrong(ev: ModuleType) -> None:
    files = {f.stem: f for f in ev.bill_files()}
    read = ev.truth_for(files["h07_three_period"])
    shifted = json.loads(json.dumps(read["tariff"]))
    shifted["periods"][2]["start_hour"], shifted["periods"][1]["end_hour"] = 17, 17  # peak an hour late
    assert ev.outcome(read, check_bill(shifted, "INR", 7.0)) == ("wrong", "accepted with wrong hourly rates")
    assert ev.outcome(read, check_bill({"unreadable": True}, "INR", 7.0)) == ("wrong", "model said unreadable")
    refuse = ev.truth_for(files["h04_wrong_currency"])
    as_rupees = {**refuse["tariff"], "currency": "INR"}
    assert ev.outcome(refuse, check_bill(as_rupees, "INR", 7.0))[0] == "wrong"


def test_the_recorded_run_is_one_pass_under_the_cap(ev: ModuleType) -> None:
    results = json.loads((ROOT / "results" / "vision_eval" / "results.json").read_text(encoding="utf-8"))
    assert results["cost_usd"] <= ev.CAP_USD and len(results["bills"]) == 10
    assert sum(results["counts"].values()) == 10
    assert sum(r["cost_usd"] for r in results["bills"] if "cost_usd" in r) == pytest.approx(results["cost_usd"], abs=0.001)
