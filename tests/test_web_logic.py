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


# --- the landing page's pure functions (web/landing.js) ---------------------------------------------------

LANDING = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const code = src.slice(src.indexOf("/* --- pure: begin"), src.indexOf("/* --- pure: end"));
const f = new Function(code + "; return { HERO_PAIRS, HERO_EXAMPLE_MS, HERO_STEPS, heroPicks, heroFrame, heroStart, specCodes, hourWindows, sourceLine, leadingNumber, parseShare, parseDelta, rowFor, armNames };")();
const index = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const cases = JSON.parse(process.argv[3]);
const out = {
  constants: { pairs: f.HERO_PAIRS, exampleMs: f.HERO_EXAMPLE_MS, steps: f.HERO_STEPS },
  picks: f.heroPicks(index.chips).map((c) => (c ? c.id : null)),
  missing: f.heroPicks(index.chips, [["nairobi_clinic", "en"], ["nowhere", "xx"]]).map((c) => (c ? c.id : null)),
  frames: cases.frames.map(([t, n]) => f.heroFrame(t, n)),
  starts: [0, 1, 2].map(f.heroStart),
  codes: cases.specs.map(f.specCodes),
  windows: f.hourWindows([14, 9, 10, 11, 10]),
  sources: [f.sourceLine("results/report/summary.json", "goal_satisfaction"), f.sourceLine("web/replays/x.json")],
  leading: ["98.8% (257/260)", "17.8%", "+8.5 pts [+5.0, +11.9]", "n/a"].map(f.leadingNumber),
  shares: ["98.8% (257/260)", "100.0%", "n/a"].map(f.parseShare),
  deltas: ["+8.5 pts [+5.0, +11.9]", "-1.2 pts [-3.0, +0.4]", "x"].map(f.parseDelta),
  names: f.armNames([{ arm: "A2", name: "LLM writes schedule" }, { arm: "A4", name: "VoltProof" }]),
  row: f.rowFor([{ arm: "A2", v: 1 }, { arm: "A4", v: 2 }], "arm", "A4"),
};
console.log(JSON.stringify(out));
"""


def landing(cases: dict) -> dict:
    result = subprocess.run([NODE, "-e", LANDING, str(WEB_DIR / "landing.js"), str(REPLAYS / "index.json"), json.dumps(cases)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_hero_plays_nairobi_english_then_ahmedabad_hindi_then_the_held_out_site_in_spanish() -> None:
    """The third example is the held-out household site; it is named only through landing.js and the
    replay index, never in tests (test_days guards that)."""
    out = landing({"frames": [], "specs": []})
    pairs = out["constants"]["pairs"]
    assert pairs[:2] == [["nairobi_clinic", "en"], ["ahmedabad_cold_store", "hi"]] and pairs[2][1] == "es"
    chips = json.loads((REPLAYS / "index.json").read_text(encoding="utf-8"))["chips"]
    third = [c for c in chips if [c["site_id"], c["language"]] == pairs[2]]
    assert len(third) == 1 and third[0]["site"]["site_type"] == "household"
    assert out["picks"] == ["nairobi-outage-reserve-en", "ahmedabad-outage-hi", third[0]["id"]]  # an everyday goal before a hard one
    assert out["missing"] == ["nairobi-outage-reserve-en", None]  # a missing pair is skipped, never replaced
    for pick in out["picks"]:
        view = json.loads((REPLAYS / f"{pick}.json").read_text(encoding="utf-8"))
        assert view["same_day"]["agent"]["goals_met"] and view["plan"]["verified"]


def test_hero_timing_and_rotation() -> None:
    ms = 9000
    frames = [[0, 3], [1799, 3], [1800, 3], [3200, 3], [5600, 3], [8999, 3], [9000, 3], [2 * ms + 4000, 3], [3 * ms, 3], [3 * ms + 1800, 3], [-1, 3], [100, 0]]
    out = landing({"frames": frames, "specs": []})
    assert out["constants"]["exampleMs"] == ms and [s[1] for s in out["constants"]["steps"]] == [0, 1800, 3200, 5600]
    got = [(f["index"], f["step"]) if f else None for f in out["frames"]]
    assert got == [(0, "goal"), (0, "goal"), (0, "rules"), (0, "chart"), (0, "result"), (0, "result"), (1, "goal"), (2, "chart"), (0, "goal"), (0, "rules"), (2, "result"), None]
    assert out["frames"][0]["progress"] == 0 and abs(out["frames"][5]["progress"] - 8999 / ms) < 1e-9
    assert out["starts"] == [0, ms, 2 * ms]  # clicking line i jumps to the start of example i


def test_rule_codes_windows_and_source_lines() -> None:
    specs = [{"grid_outages": [{"start_hour": 9, "end_hour": 12}], "reserve_windows": [{}]}, {"import_caps": [{}], "no_export_windows": []}, {"end_of_day_min_soc_pct": 0}, {}, None]
    out = landing({"frames": [], "specs": specs})
    assert out["codes"] == [["OUTAGE_DEFICIT", "RESERVE"], ["IMPORT_CAP"], ["END_SOC"], [], []]
    assert out["windows"] == [[9, 12], [14, 15]]
    assert out["sources"] == ["Source: results/report/summary.json · goal_satisfaction", "Source: web/replays/x.json"]
    assert out["leading"] == ["98.8%", "17.8%", "+8.5", None]
    assert out["shares"] == [{"pct": 98.8, "num": 257, "den": 260}, {"pct": 100, "num": None, "den": None}, None]
    assert out["deltas"] == [{"value": "+8.5", "low": "+5.0", "high": "+11.9"}, {"value": "-1.2", "low": "-3.0", "high": "+0.4"}, None]
    assert out["names"] == {"A2": "LLM writes schedule", "A4": "VoltProof"} and out["row"] == {"arm": "A4", "v": 2}


def test_the_numbers_the_landing_page_shows_parse_from_the_committed_results() -> None:
    """The Problem and Evidence numbers: every string the page reads exists in the files and parses."""
    main = json.loads((WEB_DIR.parent / "results" / "report" / "summary.json").read_text(encoding="utf-8"))["tables"]
    e2 = json.loads((WEB_DIR.parent / "results" / "e2" / "summary.json").read_text(encoding="utf-8"))["tables"]
    names = {r["arm"]: r["name"] for r in main["run_summary"]}
    assert names["A2"] == "LLM writes schedule" and names["A4"] == "VoltProof"
    outage = {r["arm"]: r for r in main["outage_safety"]}
    assert outage["A2"]["plan physics valid"] == "17.8%" and outage["A4"]["plan physics valid"] == "100.0%" and outage["A2"]["episodes"] == 45
    satisfied = {r["arm"]: r["overall"] for r in main["goal_satisfaction"]}
    assert satisfied["A4"] == "98.8% (257/260)"
    assert {r["comparison"]: r["overall"] for r in main["goal_satisfaction_bootstrap"]}["A4 - A2"] == "+8.5 pts [+5.0, +11.9]"
    assert {r["arm"]: r["overall"] for r in e2["goal_satisfaction"]}["A4"] == "94.1% (95/101)"
    assert [r["exact match"] for r in main["a4_spec_by_language"] if r["group"].startswith("multilingual:")] == ["100.0%"] * 5


def test_every_page_script_parses() -> None:
    """The other tests load only slices of the scripts; a syntax error elsewhere (say, a comment closed
    early by a glob in its text) would stop the whole page, so check every script parses."""
    for name in ("app.js", "charts.js", "landing.js"):
        result = subprocess.run([NODE, "--check", str(WEB_DIR / name)], capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"{name}: {result.stderr}"
