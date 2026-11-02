"""The page's pure helpers (web/app.js and web/landing.js), run in Node on committed replays: missed goals
in plain words, the cost line against the rule-based battery, the "Other load kept on" line, and the
landing page's day player, rule drawing and dot counts. Skipped where Node is not installed."""

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


def test_cost_line_says_same_cost_when_the_difference_rounds_to_zero(tmp_path: Path) -> None:
    chips = json.loads((REPLAYS / "index.json").read_text(encoding="utf-8"))["chips"]
    same = [c["file"] for c in chips if round(abs(replay(c["id"])["same_day"]["cost_difference"]) * 100) == 0]
    assert same  # at least one recorded day costs the same as the rule-based battery
    tiny = variant(tmp_path, "tiny.json", lambda sd: sd.update(cost_difference=-0.004))
    cent = variant(tmp_path, "cent.json", lambda sd: sd.update(cost_difference=0.01))
    lines = run(*same, tiny, cent)
    for name in [*same, "tiny.json"]:
        cost = lines[name]["cost"]
        assert cost["value"] == "Same cost as the rule-based battery" and cost["tone"] == "neutral" and cost["same"] is True, name
    assert lines["cent.json"]["cost"]["value"].endswith("cheaper") and "same" not in lines["cent.json"]["cost"]


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


# --- the landing page's pure functions (web/landing.js), on the committed replays -------------------------

LANDING = r"""
const fs = require("fs");
const path = require("path");
const src = fs.readFileSync(process.argv[1], "utf8");
const code = src.slice(src.indexOf("/* --- pure: begin"), src.indexOf("/* --- pure: end"));
const f = new Function(code + "; return { HERO_PAIRS, HOUR_MS, DAY_MS, SITE_MS, heroPicks, heroClock, heroTime, hourFlows, specCodes, hourWindows, timelineRules, ruleResults, focusHour, dotCount, sourceLine, leadingNumber, parseShare, parseDelta, rowFor, armNames, hourText };")();
const dir = process.argv[2];
const index = JSON.parse(fs.readFileSync(path.join(dir, "index.json"), "utf8"));
const views = Object.fromEntries(index.chips.map((c) => [c.id, JSON.parse(fs.readFileSync(path.join(dir, c.file), "utf8"))]));
const out = {};
for (const [key, expr] of Object.entries(JSON.parse(process.argv[3]))) out[key] = new Function("f", "views", "index", `return (${expr});`)(f, views, index);
console.log(JSON.stringify(out));
"""


def landing(**exprs: str) -> dict:
    """Evaluate JS expressions over landing.js's pure functions `f`, the replays `views` and `index`."""
    result = subprocess.run([NODE, "-e", LANDING, str(WEB_DIR / "landing.js"), str(REPLAYS), json.dumps(exprs)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def replay(chip_id: str) -> dict:
    return json.loads((REPLAYS / f"{chip_id}.json").read_text(encoding="utf-8"))


def hero_ids() -> list[str]:
    return landing(picks="f.heroPicks(index.chips).map((c) => c.id)")["picks"]


def test_hero_plays_nairobi_english_then_ahmedabad_hindi_then_the_held_out_site_in_spanish() -> None:
    """The third day is the held-out household site; it is named only through landing.js and the
    replay index, never in tests (test_days guards that)."""
    out = landing(pairs="f.HERO_PAIRS", picks="f.heroPicks(index.chips).map((c) => c && c.id)", missing='f.heroPicks(index.chips, [["nairobi_clinic", "en"], ["nowhere", "xx"]]).map((c) => c && c.id)')
    pairs = out["pairs"]
    assert pairs[:2] == [["nairobi_clinic", "en"], ["ahmedabad_cold_store", "hi"]] and pairs[2][1] == "es"
    chips = json.loads((REPLAYS / "index.json").read_text(encoding="utf-8"))["chips"]
    third = [c for c in chips if [c["site_id"], c["language"]] == pairs[2]]
    assert len(third) == 1 and third[0]["site"]["site_type"] == "household"
    assert out["picks"] == ["nairobi-outage-reserve-en", "ahmedabad-outage-hi", third[0]["id"]]  # an everyday goal before a hard one
    assert out["missing"] == ["nairobi-outage-reserve-en", None]  # a missing pair is skipped, never replaced
    for pick in out["picks"]:
        assert replay(pick)["same_day"]["agent"]["goals_met"] and replay(pick)["plan"]["verified"]


def test_translated_goals_carry_their_english_source() -> None:
    chips = json.loads((REPLAYS / "index.json").read_text(encoding="utf-8"))["chips"]
    for chip in chips:
        goal = replay(chip["id"])["goal"]
        if goal["language"] == "en":
            assert goal["english"] is None
        else:
            assert goal["english"] and goal["english"] != goal["text"], chip["id"]
    hindi, spanish = (replay(i)["goal"] for i in hero_ids()[1:])
    assert hindi["english"].startswith("The grid will be off from 2 PM to 5 PM")
    cap = replay(hero_ids()[2])["spec"]["import_caps"][0]["max_kw"]
    assert f"{cap:g} kW" in spanish["text"] and f"{cap:g} kW" in spanish["english"]  # filled with the same value


def test_hero_clock_steps_through_the_hours_and_back() -> None:
    out = landing(
        consts="[f.HOUR_MS, f.DAY_MS, f.SITE_MS]",
        frames="[0, 349, 350, 23 * 350, 8400, 9999, 10000, 2 * 10000 + 15 * 350, 30000, -1].map((t) => { const c = f.heroClock(t, 3); return [c.index, c.hour]; })",
        none="f.heroClock(100, 0)",
        times="[[0, 0], [1, 15], [2, 24]].map(([i, h]) => f.heroTime(i, h))",
        roundtrip="[0, 1, 2].every((i) => [...Array(25).keys()].every((h) => { const c = f.heroClock(f.heroTime(i, h), 3); return c.index === i && c.hour === h; }))",
        hours="[0, 9, 23].map(f.hourText)",
    )
    assert out["consts"] == [350, 8400, 10000]
    assert out["frames"] == [[0, 0], [0, 0], [0, 1], [0, 23], [0, 24], [0, 24], [1, 0], [2, 15], [0, 0], [2, 24]]
    assert out["none"] is None and out["times"] == [0, 15250, 28400] and out["roundtrip"] is True
    assert out["hours"] == ["00:00", "09:00", "23:00"]


def test_the_playhead_drives_the_diagram_with_the_replay_values() -> None:
    ids = hero_ids()
    out = landing(flows=f"{json.dumps(ids)}.map((id) => [...Array(25).keys()].map((h) => f.hourFlows(views[id], h)))")
    for chip_id, flows in zip(ids, out["flows"]):
        view = replay(chip_id)
        sd, run_, plan = view["same_day"], view["same_day"]["agent"], view["plan"]
        for hour in range(24):
            got = flows[hour]
            assert got["pv"] == sd["pv_actual_kw"][hour] and got["load"] == sd["load_actual_kw"][hour]
            assert got["gridImport"] == run_["grid_import_kw"][hour] and got["gridExport"] == run_["grid_export_kw"][hour]
            assert got["charge"] - got["discharge"] == pytest.approx(run_["battery_kw"][hour])
            assert got["critical"] == plan["critical_load_kw"][hour]
            assert got["socEnd"] == run_["soc_pct"][hour] and got["socStart"] == (plan["initial_soc_pct"] if hour == 0 else run_["soc_pct"][hour - 1])
            assert got["outage"] == (hour in plan["goals"]["outage_hours"])
            assert abs(got["balance"]) < 0.01, (chip_id, hour, got["balance"])  # energy in = energy out
            assert 0 <= got["curtail"] <= got["pv"] + 1e-9 and got["fullLoadServed"] is True
            if got["outage"]:
                assert got["gridImport"] == 0 and got["gridExport"] == 0
        assert flows[24]["end"] is True and flows[24]["socEnd"] == run_["soc_pct"][23]


def test_rules_are_drawn_on_the_timeline_and_checked() -> None:
    nairobi, ahmedabad, third = hero_ids()
    three = json.dumps([nairobi, ahmedabad, third])
    out = landing(
        rules=f"{three}.map((id) => f.timelineRules(views[id]))",
        focus=f"{three}.map((id) => f.focusHour(views[id]))",
        results=f'[f.ruleResults(views["{nairobi}"].spec, []), f.ruleResults(views["{nairobi}"].spec, ["RESERVE"]), f.ruleResults({{no_export_windows: [{{}}]}}, [])]',
        codes="[{grid_outages: [{start_hour: 9, end_hour: 12}], reserve_windows: [{}]}, {import_caps: [{}], no_export_windows: []}, {end_of_day_min_soc_pct: 0}, {}, null].map(f.specCodes)",
        windows="f.hourWindows([14, 9, 10, 11, 10])",
    )
    n, a, t = out["rules"]
    required = replay(nairobi)["plan"]["goals"]["required_soc_pct"]
    assert n["outages"] == [[9, 12]] and n["importCap"] == [] and n["listed"] == []
    assert [f["hour"] for f in n["floor"]] == [h for h, v in enumerate(required) if v is not None] and {f["pct"] for f in n["floor"]} == {40}
    assert a["outages"] == [[14, 17]] and a["floor"] == [] and a["importCap"] == []
    assert t["outages"] == [] and [c["hour"] for c in t["importCap"]] == [13, 14, 15, 16] and {c["kw"] for c in t["importCap"]} == {7}
    assert out["focus"] == [10, 15, 14]  # mid-outage, mid-outage, middle of the import cap
    assert out["results"][0] == [{"code": "OUTAGE_DEFICIT", "met": True}, {"code": "RESERVE", "met": True}]
    assert out["results"][1][1] == {"code": "RESERVE", "met": False}
    assert out["results"][2] == [{"code": "NO_EXPORT", "met": True}]  # listed in words, not drawn
    assert out["codes"] == [["OUTAGE_DEFICIT", "RESERVE"], ["IMPORT_CAP"], ["END_SOC"], [], []]
    assert out["windows"] == [[9, 12], [14, 15]]


def test_dot_grid_counts_reproduce_the_printed_rates_and_the_episode_records() -> None:
    main = json.loads((WEB_DIR.parent / "results" / "report" / "summary.json").read_text(encoding="utf-8"))["tables"]
    outage = {r["arm"]: r for r in main["outage_safety"]}
    cases = [[outage["A2"]["plan physics valid"], outage["A2"]["episodes"]], [outage["A4"]["plan physics valid"], outage["A4"]["episodes"]], ["12.3%", 45], ["50%", 0]]
    out = landing(counts=f"{json.dumps(cases)}.map(([rate, n]) => f.dotCount(rate, n))")
    assert out["counts"] == [{"passed": 8, "failed": 37}, {"passed": 45, "failed": 0}, None, None]
    # the same counts straight from the per-episode records of the frozen test run
    rows = [json.loads(line) for line in (WEB_DIR.parent / "results" / "test-run" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    outage_rows = [r for r in rows if r["spec_name"] in ("outage_critical", "outage_then_reserve")]
    for arm, expected in (("A2", 8), ("A4", 45)):
        mine = [r for r in outage_rows if r["arm"] == arm]
        assert len(mine) == outage[arm]["episodes"] == 45 and sum(r["plan_physics_valid"] for r in mine) == expected


def test_source_lines_and_parsers() -> None:
    out = landing(
        sources='[f.sourceLine("results/report/summary.json", "goal_satisfaction"), f.sourceLine("web/replays/x.json")]',
        leading='["98.8% (257/260)", "17.8%", "+8.5 pts [+5.0, +11.9]", "n/a"].map(f.leadingNumber)',
        shares='["98.8% (257/260)", "100.0%", "n/a"].map(f.parseShare)',
        deltas='["+8.5 pts [+5.0, +11.9]", "-1.2 pts [-3.0, +0.4]", "x"].map(f.parseDelta)',
        names='f.armNames([{arm: "A2", name: "LLM writes schedule"}, {arm: "A4", name: "VoltProof"}])',
        row='f.rowFor([{arm: "A2", v: 1}, {arm: "A4", v: 2}], "arm", "A4")',
    )
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
    for name in ("theme.js", "app.js", "charts.js", "landing.js"):
        result = subprocess.run([NODE, "--check", str(WEB_DIR / name)], capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"{name}: {result.stderr}"
