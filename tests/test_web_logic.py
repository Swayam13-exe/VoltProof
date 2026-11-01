"""The page's pure helpers (web/app.js), run in Node on committed replays: missed goals in plain words,
the cost line against the rule-based battery and the "Other load kept on" line. Skipped where Node is
not installed."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from voltproof.api.app import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
REPLAYS = WEB_DIR / "replays"
SHED_DAY = "ahmedabad-outage-hi.json"  # Ahmedabad 2025-12-17, grid off 14-17: the rule-based battery sheds 41.12 kWh

HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");  // node -e: argv[1] is the first argument
const slice = (from, to) => src.slice(src.indexOf(from), src.indexOf(to));
const code = slice("function money(", "function status(") + slice("function hourText(", "function stat(") + slice("function reasonText(", "async function fallBack(");
const helpers = new Function(code + "; return { costLine, missedText, otherLoadLine, reasonText, liveNote };")();
const out = {};
if (process.argv[2] === "--notes") {
  const cases = JSON.parse(process.argv[3]);
  console.log(JSON.stringify({ reasons: cases.reasons.map(([r, b]) => helpers.reasonText(r, b)), notes: cases.notes.map((s) => helpers.liveNote(s)) }));
  process.exit(0);
}
for (const file of process.argv.slice(2)) {
  const view = JSON.parse(fs.readFileSync(file, "utf8"));
  out[file.split(/[\\/]/).pop()] = {
    cost: helpers.costLine(view.same_day, view.site.currency, view.spec, "default"),
    other: helpers.otherLoadLine(view.same_day),
  };
}
console.log(JSON.stringify(out));
"""


def run(*replays: str | Path) -> dict:
    paths = [str(r if isinstance(r, Path) else REPLAYS / r) for r in replays]
    result = subprocess.run([NODE, "-e", HARNESS, str(WEB_DIR / "app.js"), *paths], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def variant(tmp_path: Path, name: str, change: Callable[[dict[str, Any]], None]) -> Path:
    """A copy of the shed-load replay with its same-day comparison changed."""
    view = json.loads((REPLAYS / SHED_DAY).read_text(encoding="utf-8"))
    change(view["same_day"])
    path = tmp_path / name
    path.write_text(json.dumps(view), encoding="utf-8")
    return path


def test_cost_line_cases() -> None:
    lines = run("ahmedabad-outage-en.json", "nairobi-noon-midnight-en.json")
    cheaper = lines["ahmedabad-outage-en.json"]["cost"]
    assert cheaper["tone"] == "ok" and cheaper["value"].endswith("cheaper") and cheaper["explain"] == "default"
    met_goal = lines["nairobi-noon-midnight-en.json"]["cost"]  # VoltProof costs more; the rule-based battery missed goals
    assert met_goal["tone"] == "neutral" and met_goal["value"].endswith("more expensive")
    assert met_goal["explain"].startswith("Costs more because it met the goal; the rule-based battery was cheaper only by missing: battery reserve: at least 50% from 12:00 to midnight")
    assert "charge target: at least 95% by 12:00" in met_goal["explain"]


def test_cost_line_explains_load_the_rule_based_battery_switched_off(tmp_path: Path) -> None:
    def missed(sd: dict) -> None:
        sd["rule_based"].update(goals_met=False, goals_missed=["END_SOC"])

    def no_shed(sd: dict) -> None:
        sd["rule_based"].update(non_critical_shed_kwh=0.0, non_critical_shed_outage_kwh=0.0)

    def both_shed(sd: dict) -> None:
        sd["agent"].update(non_critical_shed_kwh=5.0, non_critical_shed_outage_kwh=5.0)

    def small_gap(sd: dict) -> None:
        sd["agent"].update(non_critical_shed_kwh=41.05, non_critical_shed_outage_kwh=41.05)

    files = [variant(tmp_path, f"{name}.json", change) for name, change in (("missed", missed), ("no_shed", no_shed), ("both_shed", both_shed), ("small_gap", small_gap))]
    lines = run(SHED_DAY, *files)
    shed = lines[SHED_DAY]["cost"]  # costs ₹322.29 more, both met every goal, the rule-based battery shed 41.12 kWh
    assert shed["tone"] == "neutral" and shed["value"].endswith("more expensive")
    assert re.fullmatch(r"Costs more because it kept the whole site running; the rule-based battery switched off 41[.,]12 kWh of other load during the outage\.", shed["explain"])
    both = lines["missed.json"]["cost"]  # the rule-based battery also missed a goal: both reasons, one sentence
    assert both["tone"] == "neutral" and both["explain"].count(".") <= 2  # the only dots: the decimal and the end
    assert both["explain"].startswith("Costs more because it met the goal and kept the whole site running; the rule-based battery missed: end-of-day level")
    assert re.search(r", and switched off 41[.,]12 kWh of other load during the outage\.$", both["explain"])
    assert lines["no_shed.json"]["cost"]["tone"] == "fail" and lines["no_shed.json"]["cost"]["explain"] == "default"  # no reason: stays red
    partial = lines["both_shed.json"]["cost"]
    assert partial["tone"] == "neutral" and "kept more of the site running" in partial["explain"]
    assert lines["small_gap.json"]["cost"]["tone"] == "fail"  # under 0.1 kWh more shed is not a reason


def test_other_load_line(tmp_path: Path) -> None:
    both = variant(tmp_path, "both_shed.json", lambda sd: sd["agent"].update(non_critical_shed_kwh=5.0, non_critical_shed_outage_kwh=5.0))
    lines = run(SHED_DAY, "ahmedabad-outage-en.json", both)
    kept = lines[SHED_DAY]["other"]
    assert kept["value"] == "✓ all of it" and kept["tone"] == "ok"
    assert re.fullmatch(r"Rule-based battery switched off 41[.,]12 kWh of other load during the outage\.", kept["explain"])
    assert lines["ahmedabad-outage-en.json"]["other"] is None  # neither side switched anything off: no line
    partial = lines["both_shed.json"]["other"]
    assert partial["value"] == "all but 5 kWh" and partial["tone"] == "neutral"


def test_limit_reasons_and_the_live_note() -> None:
    cases = {
        "reasons": [["hourly_cap", {"retry_after_s": 1500}], ["hourly_cap", None], ["rate_limit", {"retry_after_s": 61}]],
        "notes": [
            {"live": True, "per_visitor_limit": False, "runs_left_this_hour": 37},
            {"live": True, "per_visitor_limit": True, "runs_left_this_hour": 9},
            {"live": True, "runs_left_this_hour": 9},  # an older server without the field: per visitor
            {"live": False, "reason": "live_mode_off"},
            None,
        ],
    }
    result = subprocess.run([NODE, "-e", HARNESS, str(WEB_DIR / "app.js"), "--notes", json.dumps(cases)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["reasons"] == [
        "this hour's live runs for all visitors are used up; try again in about 25 min",
        "this hour's live runs for all visitors are used up",
        "you have used this hour's 10 live runs; try again in about 2 min",
    ]
    shared, mine, old, off, unreachable = out["notes"]
    assert shared == "Live AI on: 37 live runs left this hour (shared by all visitors)."
    assert mine == old == "Live AI on: 9 live runs left for you this hour."
    assert off.startswith("Live AI unavailable (the live AI is switched off here)") and unreachable.startswith("Live AI not available here")
