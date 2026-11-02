"""The page's pure helpers (web/app.js and web/landing.js), run in Node on committed replays: missed goals
in plain words, the cost line against the rule-based battery, the "Other load kept on" line, and the
landing page's day player, rule drawing and dot counts. Skipped where Node is not installed."""

from __future__ import annotations

import json
import math
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
  console.log(JSON.stringify({ reasons: cases.reasons.map(([r, b, l]) => helpers.reasonText(r, b, l)), notes: cases.notes.map((s) => helpers.liveNote(s)) }));
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
        "reasons": [["hourly_cap", {"retry_after_s": 1500}], ["hourly_cap", None], ["rate_limit", {"retry_after_s": 61}, {"per_visitor_per_hour": 10}], ["rate_limit", None, None]],
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
        "you have used this hour's 10 live runs; try again in about 2 min",  # the limit comes from /api/status
        "you have used this hour's live runs",
    ]
    shared, mine, old, off, unreachable = out["notes"]
    assert shared == "Live AI on: 37 live runs left this hour (shared by all visitors)."
    assert mine == old == "Live AI on: 9 live runs left for you this hour."
    assert off.startswith("Live AI unavailable (the live AI is switched off here)") and unreachable.startswith("Live AI not available here")


# --- pure functions of the day player (web/dayplayer.js), the landing page (web/landing.js) and Try it ----

PURE = r"""
const fs = require("fs");
const path = require("path");
const read = (name) => fs.readFileSync(path.join(process.argv[1], name), "utf8");
const block = (src, from, to) => src.slice(src.indexOf(from), src.indexOf(to));
const day = block(read("dayplayer.js"), "/* --- pure: begin", "/* --- pure: end");
const landing = block(read("landing.js"), "/* --- pure: begin", "/* --- pure: end");
const app = read("app.js");
const tryCode = block(app, "function money(", "function status(") + block(app, "function hourText(", "function stat(") + block(app, "function reasonText(", "async function fallBack(");
const D = new Function(day + "; return { DAY_MS, HOLD_MS, SITE_MS, playClock, playTime, hourFlows, specCodes, hourWindows, ruleBands, ruleResults, focusHour, shortSiteName, hourText };")();
const L = new Function(landing + "; return { HERO_PAIRS, heroPicks, dotCount, sourceLine, leadingNumber, parseShare, parseDelta, rowFor, armNames };")();
const T = new Function("VPDay", tryCode + "; return { PIPELINE_STAGES, recordedFor, recordedTiming, runButtonState, pipelineStart, pipelineStep, pipelineFrames, traceEvents, approvalState, comparisonRows, csvRows, csvText, costLine };")(D);
const dir = process.argv[2];
const index = JSON.parse(fs.readFileSync(path.join(dir, "index.json"), "utf8"));
const views = Object.fromEntries(index.chips.map((c) => [c.id, JSON.parse(fs.readFileSync(path.join(dir, c.file), "utf8"))]));
const out = {};
for (const [key, expr] of Object.entries(JSON.parse(process.argv[3]))) out[key] = new Function("D", "L", "T", "views", "index", `return (${expr});`)(D, L, T, views, index);
console.log(JSON.stringify(out));
"""


def pure(**exprs: str) -> dict:
    """Evaluate JS expressions over the pure functions: D (day player), L (landing), T (Try it), the
    committed replays `views` and their `index`."""
    result = subprocess.run([NODE, "-e", PURE, str(WEB_DIR), str(REPLAYS), json.dumps(exprs)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def replay(chip_id: str) -> dict:
    return json.loads((REPLAYS / f"{chip_id}.json").read_text(encoding="utf-8"))


def chips() -> list[dict]:
    return json.loads((REPLAYS / "index.json").read_text(encoding="utf-8"))["chips"]


def hero_ids() -> list[str]:
    return pure(picks="L.heroPicks(index.chips).map((c) => c.id)")["picks"]


def test_hero_plays_nairobi_english_then_ahmedabad_hindi_then_the_held_out_site_in_spanish() -> None:
    """The third day is the held-out household site; it is named only through landing.js and the
    replay index, never in tests (test_days guards that)."""
    out = pure(pairs="L.HERO_PAIRS", picks="L.heroPicks(index.chips).map((c) => c && c.id)", missing='L.heroPicks(index.chips, [["nairobi_clinic", "en"], ["nowhere", "xx"]]).map((c) => c && c.id)')
    pairs = out["pairs"]
    assert pairs[:2] == [["nairobi_clinic", "en"], ["ahmedabad_cold_store", "hi"]] and pairs[2][1] == "es"
    third = [c for c in chips() if [c["site_id"], c["language"]] == pairs[2]]
    assert len(third) == 1 and third[0]["site"]["site_type"] == "household"
    assert out["picks"] == ["nairobi-outage-reserve-en", "ahmedabad-outage-hi", third[0]["id"]]  # an everyday goal before a hard one
    assert out["missing"] == ["nairobi-outage-reserve-en", None]  # a missing pair is skipped, never replaced
    for pick in out["picks"]:
        assert replay(pick)["same_day"]["agent"]["goals_met"] and replay(pick)["plan"]["verified"]


def test_short_site_names_for_the_phone_buttons() -> None:
    ids = hero_ids()
    names = [replay(i)["site"]["name"] for i in ids]
    out = pure(short=f"{json.dumps(names)}.map(D.shortSiteName)")
    assert out["short"][:2] == ["Nairobi", "Ahmedabad"]
    assert out["short"][2] == re.findall(r"\b[A-Z][\w-]*", names[2])[-1] and out["short"][2] in names[2]


def test_translated_goals_carry_their_english_source() -> None:
    for chip in chips():
        goal = replay(chip["id"])["goal"]
        if goal["language"] == "en":
            assert goal["english"] is None
        else:
            assert goal["english"] and goal["english"] != goal["text"], chip["id"]
    hindi, spanish = (replay(i)["goal"] for i in hero_ids()[1:])
    assert hindi["english"].startswith("The grid will be off from 2 PM to 5 PM")
    cap = replay(hero_ids()[2])["spec"]["import_caps"][0]["max_kw"]
    assert f"{cap:g} kW" in spanish["text"] and f"{cap:g} kW" in spanish["english"]  # filled with the same value


def test_play_clock_paces_14_s_per_day_then_holds_3_s_on_the_checked_rules() -> None:
    out = pure(
        consts="[D.DAY_MS, D.HOLD_MS, D.SITE_MS]",
        loop="[0, 583, 584, 13999, 14000, 16999, 17000, 2 * 17000 + 7000, 51000, -1].map((t) => { const c = D.playClock(t, 3); return [c.index, c.hour]; })",
        once="[0, 7000, 13999, 14000, 99999].map((t) => { const c = D.playClock(t, 1, false); return [c.index, c.hour, c.done]; })",
        none="D.playClock(100, 0)",
        roundtrip="[0, 1, 2].every((i) => [...Array(25).keys()].every((h) => { const c = D.playClock(D.playTime(i, h), 3); return c.index === i && c.hour === h; }))",
        hours="[0, 9, 23].map(D.hourText)",
    )
    assert out["consts"] == [14000, 3000, 17000]
    assert out["loop"] == [[0, 0], [0, 0], [0, 1], [0, 23], [0, 24], [0, 24], [1, 0], [2, 12], [0, 0], [2, 24]]
    assert out["once"] == [[0, 0, False], [0, 12, False], [0, 23, False], [0, 24, True], [0, 24, True]]  # Try it plays once, ends on the rules
    assert out["none"] is None and out["roundtrip"] is True and out["hours"] == ["00:00", "09:00", "23:00"]


def test_the_playhead_drives_the_diagram_with_the_replay_values() -> None:
    ids = hero_ids()
    out = pure(flows=f"{json.dumps(ids)}.map((id) => [...Array(25).keys()].map((h) => D.hourFlows(views[id], h)))")
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


def test_rules_are_drawn_exactly_where_the_checker_tests_them() -> None:
    """For every rule in the three hero replays, the timeline's x-range equals what the checker checks:
    a charge floor at the instants soc[i] is tested (the end of hour i, i.e. (i+1):00; the initial
    charge at 0:00), an outage or import cap over the hours its rows cover."""
    from voltproof.core.spec import INITIAL, soft_rows
    from voltproof.data.sites import SITES
    from voltproof.models import IntentSpec

    ids = hero_ids()
    out = pure(bands=f"{json.dumps(ids)}.map((id) => D.ruleBands(views[id]))")
    checked_any = {"floor": 0, "cap": 0, "outage": 0}
    for chip_id, bands in zip(ids, out["bands"]):
        view = replay(chip_id)
        spec = IntentSpec.model_validate(view["spec"])
        rows = soft_rows(spec, SITES[view["site"]["site_id"]])
        capacity = SITES[view["site"]["site_id"]].battery.capacity_kwh
        want_floor: dict[int, float] = {}
        for row in rows:
            if row.kind == "soc_min":
                instant = 0 if row.hour == INITIAL else row.hour + 1
                want_floor[instant] = max(want_floor.get(instant, 0.0), round(100.0 * row.bound / capacity, 1))
        drawn_floor = {t: f["pct"] for f in bands["floors"] for t in range(f["from"], f["to"] + 1)}
        assert drawn_floor == want_floor, chip_id
        want_cap = {row.hour for row in rows if row.kind == "import_max"}
        assert {h for c in bands["caps"] for h in range(c["from"], c["to"])} == want_cap, chip_id
        want_outage = {h for w in spec.grid_outages for h in w.hours()}
        assert {h for o in bands["outages"] for h in range(o["from"], o["to"])} == want_outage, chip_id
        checked_any["floor"] += len(want_floor)
        checked_any["cap"] += len(want_cap)
        checked_any["outage"] += len(want_outage)
    assert all(checked_any.values()), checked_any  # each kind actually occurs in the hero days
    nairobi = out["bands"][0]
    assert nairobi["floors"] == [{"kind": "floor", "from": 12, "to": 20, "pct": 40}]  # "12:00-20:00", not 11:00


def test_rule_results_focus_hour_and_listed_rules() -> None:
    nairobi, ahmedabad, third = hero_ids()
    three = json.dumps([nairobi, ahmedabad, third])
    out = pure(
        focus=f"{three}.map((id) => D.focusHour(views[id]))",
        results=f'[D.ruleResults(views["{nairobi}"].spec, []), D.ruleResults(views["{nairobi}"].spec, ["RESERVE"]), D.ruleResults({{no_export_windows: [{{}}]}}, [])]',
        listed='D.ruleBands({plan: {goals: {initial_required_soc_pct: null, required_soc_pct: Array(24).fill(null), import_cap_kw: Array(24).fill(null), outage_hours: []}}, spec: {no_export_windows: [{}]}}).listed',
        codes="[{grid_outages: [{start_hour: 9, end_hour: 12}], reserve_windows: [{}]}, {import_caps: [{}], no_export_windows: []}, {end_of_day_min_soc_pct: 0}, {}, null].map(D.specCodes)",
        windows="D.hourWindows([14, 9, 10, 11, 10])",
    )
    assert out["focus"] == [10, 15, 14]  # mid-outage, mid-outage, inside the import cap
    assert out["results"][0] == [{"code": "OUTAGE_DEFICIT", "met": True}, {"code": "RESERVE", "met": True}]
    assert out["results"][1][1] == {"code": "RESERVE", "met": False}
    assert out["results"][2] == [{"code": "NO_EXPORT", "met": True}] and out["listed"] == ["NO_EXPORT"]  # named in words, not drawn
    assert out["codes"] == [["OUTAGE_DEFICIT", "RESERVE"], ["IMPORT_CAP"], ["END_SOC"], [], []]
    assert out["windows"] == [[9, 12], [14, 15]]


def test_dot_grid_counts_reproduce_the_printed_rates_and_the_episode_records() -> None:
    main = json.loads((WEB_DIR.parent / "results" / "report" / "summary.json").read_text(encoding="utf-8"))["tables"]
    outage = {r["arm"]: r for r in main["outage_safety"]}
    cases = [[outage["A2"]["plan physics valid"], outage["A2"]["episodes"]], [outage["A4"]["plan physics valid"], outage["A4"]["episodes"]], ["12.3%", 45], ["50%", 0]]
    out = pure(counts=f"{json.dumps(cases)}.map(([rate, n]) => L.dotCount(rate, n))")
    assert out["counts"] == [{"passed": 8, "failed": 37}, {"passed": 45, "failed": 0}, None, None]
    rows = [json.loads(line) for line in (WEB_DIR.parent / "results" / "test-run" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    outage_rows = [r for r in rows if r["spec_name"] in ("outage_critical", "outage_then_reserve")]
    for arm, expected in (("A2", 8), ("A4", 45)):
        mine = [r for r in outage_rows if r["arm"] == arm]
        assert len(mine) == outage[arm]["episodes"] == 45 and sum(r["plan_physics_valid"] for r in mine) == expected


def test_source_lines_and_parsers() -> None:
    out = pure(
        sources='[L.sourceLine("results/report/summary.json", "goal_satisfaction"), L.sourceLine("web/replays/x.json")]',
        leading='["98.8% (257/260)", "17.8%", "+8.5 pts [+5.0, +11.9]", "n/a"].map(L.leadingNumber)',
        shares='["98.8% (257/260)", "100.0%", "n/a"].map(L.parseShare)',
        deltas='["+8.5 pts [+5.0, +11.9]", "-1.2 pts [-3.0, +0.4]", "x"].map(L.parseDelta)',
        names='L.armNames([{arm: "A2", name: "LLM writes schedule"}, {arm: "A4", name: "VoltProof"}])',
        row='L.rowFor([{arm: "A2", v: 1}, {arm: "A4", v: 2}], "arm", "A4")',
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


# --- Try it ----------------------------------------------------------------------------------------------


def test_the_main_button_offers_exactly_one_thing() -> None:
    chip = next(c for c in chips() if c["id"] == "nairobi-outage-reserve-en")
    other_day = next(c for c in chips() if c["site_id"] == chip["site_id"] and c["date"] != chip["date"])
    site, day = json.dumps(chip["site_id"]), json.dumps(chip["date"])
    rec = f"T.recordedFor(index.chips, {site}, {day})"
    live_on = "{live: true, runs_left_this_hour: 7, per_visitor_limit: true}"
    timing = "T.recordedTiming(index.chips)"
    out = pure(
        empty=f'T.runButtonState("   ", {rec}, {live_on}, {timing})',
        recorded=f"T.runButtonState({json.dumps('  ' + chip['goal'] + ' ')}, {rec}, null, {timing})",
        edited=f"T.runButtonState({json.dumps(chip['goal'] + ' please')}, {rec}, {live_on}, {timing})",
        untimed=f"T.runButtonState({json.dumps(chip['goal'] + ' please')}, {rec}, {live_on}, null)",
        other_day=f"T.runButtonState({json.dumps(other_day['goal'])}, {rec}, {live_on}, {timing})",
        shared=f'T.runButtonState("Keep 50% tonight", {rec}, {{live: true, runs_left_this_hour: 30, per_visitor_limit: false}}, null)',
        off=f'T.runButtonState("Keep 50% tonight", {rec}, {{live: false, reason: "live_mode_off"}}, null)',
        unreachable=f'T.runButtonState("Keep 50% tonight", {rec}, null, null)',
        used_up=f'T.runButtonState("Keep 50% tonight", {rec}, {{live: true, runs_left_this_hour: 0}}, null)',
        timing=timing,
    )
    assert out["empty"] == {"mode": "empty", "label": "Plan my day", "disabled": True, "hint": "Type a goal above or pick a recorded one."}
    assert out["recorded"]["mode"] == "recorded" and out["recorded"]["label"] == "Play recorded run (free)" and out["recorded"]["chip"]["id"] == chip["id"]
    assert not out["recorded"]["disabled"]  # a recorded goal plays even with live runs off
    assert out["edited"]["mode"] == "live" and out["edited"]["label"] == "Run live (7 left this hour)" and not out["edited"]["disabled"]
    # no hand-typed duration: the hint quotes the recorded runs' own latency, from the index
    latencies = [c["latency_s"] for c in chips()]
    assert out["timing"] == {"min": math.floor(min(latencies)), "max": math.ceil(max(latencies)), "runs": len(latencies)}
    assert f"The {len(latencies)} recorded runs took {math.floor(min(latencies))}-{math.ceil(max(latencies))} s (web/replays/index.json)." in out["edited"]["hint"]
    assert out["untimed"]["hint"] == "Runs the AI on your goal now."
    assert out["other_day"]["mode"] == "live"  # a recorded goal of another day is typed text here
    assert out["shared"]["label"] == "Run live (30 left this hour)" and "shared by all visitors" in out["shared"]["hint"]
    for key in ("off", "unreachable", "used_up"):
        assert out[key]["mode"] == "unavailable" and out[key]["disabled"] is True and out[key]["label"] == "Live runs unavailable: play a recorded run"
    assert "the live AI is switched off here" in out["off"]["hint"] and out["off"]["hint"].endswith("Pick a recorded goal above.")


STAGES = ["rules", "optimizer", "checks", "review", "accepted", "replay"]


def test_recorded_traces_run_in_the_order_the_strip_and_the_diagram_show() -> None:
    """SPEC §6.10: the reviewer runs only after the checks pass. In every recorded trace the plan is
    solved, then finalize runs the checks, then the reviewer, then the plan is accepted."""
    for chip in chips():
        trace = replay(chip["id"])["trace"]
        at = lambda kind, name=None: next(i for i, e in enumerate(trace) if e["kind"] == kind and (name is None or e["name"] == name))
        solved = at("tool_result", "solve_plan")
        checks = at("tool_call", "finalize")
        review = at("review")
        accepted = next(i for i, e in enumerate(trace) if e["kind"] == "tool_result" and e["name"] == "finalize" and e["data"]["result"].get("accepted"))
        assert solved < checks < review < accepted, chip["id"]
    out = pure(
        stages="T.PIPELINE_STAGES.map(([k]) => k)",
        who="T.PIPELINE_STAGES.map(([, , , w]) => w)",
        order='index.chips.map((c) => { const done = []; for (const f of T.pipelineFrames([...T.traceEvents(views[c.id]), {kind: "end", data: {status: "done"}}, {kind: "replayed", data: {ok: true}}])) for (const [k] of T.PIPELINE_STAGES) if (f.state[k] === "done" && !done.includes(k)) done.push(k); return done; })',
    )
    assert out["stages"] == STAGES and out["who"] == ["ai", "tool", "tool", "ai", "tool", "tool"]
    for order in out["order"]:  # every recorded run finishes its stages in exactly the order shown
        assert order == STAGES


def test_how_it_works_shows_the_same_stages_in_the_same_order() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    how = html[html.index('<section id="how"'):html.index('<section id="try"')]
    names = re.findall(r'<span class="pipe-name">([^<]+)</span>', how)
    stages = re.findall(r'<li class="pipe-node (ai|tool)" data-stage="([a-z]+)">', how)
    labels = pure(labels="T.PIPELINE_STAGES.map(([k, label, , who]) => [k, label, who])")["labels"]
    assert names == ["Your goal", *[label for _, label, _ in labels]]
    assert [(k, who) for who, k in stages] == [(k, who) for k, _, who in labels]
    assert "The AI writes the rules. Tools make and check the plan. A second AI can only say no." in how
    assert "goes back to the AI" in how and 'id="repair-count"' in how  # the repair limit is filled from MAX_REPAIRS


def test_pipeline_states_repairs_and_pacing() -> None:
    out = pure(
        repair='(() => { const seen = []; for (const f of T.pipelineFrames([{kind: "tool_call", name: "set_checklist", data: {}}, {kind: "tool_call", name: "solve_plan", data: {}}, {kind: "tool_result", name: "solve_plan", data: {result: {plan_id: "p1", valid: false}}}, {kind: "tool_call", name: "finalize", data: {}}, {kind: "tool_result", name: "finalize", data: {result: {accepted: false, failed_checks: [{check: "verified_plan"}]}}}, {kind: "tool_call", name: "solve_plan", data: {}}, {kind: "tool_result", name: "solve_plan", data: {result: {plan_id: "p2"}}}, {kind: "tool_call", name: "finalize", data: {}}, {kind: "review", data: {approved: true}}, {kind: "tool_result", name: "finalize", data: {result: {accepted: true}}}])) seen.push(f.state); return seen; })()',
        rejected='(() => { let s = T.pipelineStart(); s = T.pipelineStep(s, {kind: "tool_call", name: "finalize", data: {}}); s = T.pipelineStep(s, {kind: "review", data: {approved: false}}); return T.pipelineStep(s, {kind: "tool_result", name: "finalize", data: {name: "finalize", result: {accepted: false, failed_checks: ["review"]}}}); })()',
        solve_error='T.pipelineStep(T.pipelineStep(T.pipelineStart(), {kind: "tool_call", name: "solve_plan", data: {}}), {kind: "tool_result", name: "solve_plan", data: {result: {error: "invalid spec"}}})',
        unavailable='T.pipelineStep(T.pipelineStart(), {kind: "end", data: {status: "unavailable", reason: "budget"}}).replay',
        totals='index.chips.map((c) => T.pipelineFrames([...T.traceEvents(views[c.id]), {kind: "end", data: {status: "done"}}, {kind: "replayed", data: {ok: true}}]).reduce((t, f) => t + f.delayMs, 0))',
        still='T.pipelineFrames([...T.traceEvents(views["ahmedabad-outage-hi"]), {kind: "end", data: {status: "done"}}, {kind: "replayed", data: {ok: true}}], 0, 0).reduce((t, f) => t + f.delayMs, 0)',
    )
    failed = out["repair"][4]  # the failed check: back to the AI, one repair
    assert (failed["rules"], failed["checks"], failed["repairs"], failed["failed"]) == ("active", "fail", 1, ["verified_plan"])
    final = out["repair"][-1]
    assert all(final[k] == "done" for k in STAGES[:5]) and final["repairs"] == 1
    rejected = out["rejected"]  # a rejection by the reviewer: the checks had passed, the review failed
    assert (rejected["checks"], rejected["review"], rejected["repairs"], rejected["rules"]) == ("done", "fail", 1, "active")
    assert out["solve_error"]["optimizer"] == "fail" and out["solve_error"]["rules"] == "active"
    assert out["unavailable"] == "fail"
    assert all(4000 <= t <= 5000 for t in out["totals"]), out["totals"]  # about 0.7 s a stage, 4-5 s in all
    assert out["still"] == 0  # reduced motion: all at once


def test_operator_approval_states() -> None:
    out = pure(
        waiting='T.approvalState(views["ahmedabad-outage-hi"], false)',
        approved='T.approvalState(views["ahmedabad-outage-hi"], true)',
        unverified='T.approvalState({plan: {verified: false}}, false)',
        none="T.approvalState({plan: null}, false)",
    )
    assert out["waiting"] == {"show": True, "state": "waiting", "text": "Verified - waiting for your approval", "button": "Approve plan"}
    assert out["approved"] == {"show": True, "state": "approved", "text": "Approved by you. Marked as sent to the battery (demo: no device is controlled).", "button": None}
    assert out["unverified"]["state"] == "unverified" and out["unverified"]["button"] is None
    assert out["none"] == {"show": False}
    app = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "approvalBlock(view)" in app and "text.focus()" in app  # the confirmation takes keyboard focus
    assert 'emptyResult(true);' in app  # a new run clears the last result and its approval


def test_csv_rows_are_the_executed_plan_hour_by_hour() -> None:
    view = replay("ahmedabad-outage-hi")
    out = pure(rows='T.csvRows(views["ahmedabad-outage-hi"])', text='T.csvText([["a", "b,c"], [1, "say \\"x\\""]])')
    rows = out["rows"]
    assert rows[0] == ["hour", "solar kW", "load kW", "battery kW (+ charging, - discharging)", "battery % (end of hour)", "grid import kW", "grid export kW"]
    assert len(rows) == 25
    sd, run_ = view["same_day"], view["same_day"]["agent"]
    for h in range(24):
        assert rows[h + 1] == [f"{h}:00", sd["pv_actual_kw"][h], sd["load_actual_kw"][h], run_["battery_kw"][h], run_["soc_pct"][h], run_["grid_import_kw"][h], run_["grid_export_kw"][h]]
    assert out["text"] == 'a,"b,c"\r\n1,"say ""x"""\r\n'


def test_comparison_panel_on_the_ahmedabad_hindi_day() -> None:
    out = pure(c='T.comparisonRows(views["ahmedabad-outage-hi"], D.specCodes(views["ahmedabad-outage-hi"].spec))')["c"]
    assert out["rules"] == [{"label": "critical load during the outage: from 14:00 to 17:00", "agent": True, "rule": True}]
    cost, other, critical = out["facts"]
    assert cost["label"] == "Cost on the same-day weather" and re.search(r"2[,.]?419[.,]51", cost["agent"]) and re.search(r"2[,.]?097[.,]22", cost["rule"])
    assert other["label"] == "Other load switched off" and other["agent"] == "0 kWh" and re.fullmatch(r"41[.,]12 kWh", other["rule"])
    assert critical["agent"] == "100%" and critical["rule"] == "100%" and "cold room" in critical["label"]
    assert re.fullmatch(r"₹\s?322[.,]29 more expensive", out["cost"]["value"]) and out["cost"]["tone"] == "neutral"
    assert "kept the whole site running" in out["cost"]["explain"]


def test_try_it_result_hides_the_goal_and_the_cost_line_the_hero_keeps_them() -> None:
    app, landing_js, day = ((WEB_DIR / n).read_text(encoding="utf-8") for n in ("app.js", "landing.js", "dayplayer.js"))
    assert '{ id: "try", loop: false, showGoal: false, cost: "none" }' in app
    assert '{ id: "hero", switcher: true, loop: true, live: $("day-live"), cost: "muted" }' in landing_js
    assert 'showGoal ? make("div", { class: "day-goal" }' in day and 'cost === "none" ? null' in day


def test_try_it_order_goal_chips_button_strip() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    tool = html[html.index('<section id="try"'):html.index('<section id="evidence"')]
    order = [tool.index(marker) for marker in ('id="goal"', "Or pick a recorded goal", 'id="chips"', 'id="run-button"', 'id="try-pipeline"', 'id="conversation"')]
    assert order == sorted(order)
    assert 'class="button primary"' in tool[tool.index('id="run-button"') - 60:tool.index('id="run-button"') + 60]  # the hero's button style
    assert not re.search(r">\s*\d\.\s", tool)  # no numbered "1. 2. 3." labels
    app = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    fill = app[app.index("function fillFromChip("):app.index("/* --- missed goals")]
    assert "markChips();" in fill and "renderChips();" not in fill  # a chip click does not rebuild the list (no jump)


def test_one_day_player_component_serves_the_hero_and_try_it() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    order = re.findall(r'<script[^>]*src="([^"]+)"', html)
    assert order.index("dayplayer.js") < order.index("app.js") < order.index("landing.js")
    app, landing_js, day = ((WEB_DIR / n).read_text(encoding="utf-8") for n in ("app.js", "landing.js", "dayplayer.js"))
    assert "VPDay.mount(" in app and "VPDay.mount(" in landing_js and "function mount(" in day
    for name in ("function flowDiagram", "function timeline(", "function hourFlows", "function ruleBands"):
        assert name in day and name not in app and name not in landing_js, name  # drawn once, in dayplayer.js


def test_every_page_script_parses() -> None:
    """The other tests load only slices of the scripts; a syntax error elsewhere (say, a comment closed
    early by a glob in its text) would stop the whole page, so check every script parses."""
    for name in ("theme.js", "app.js", "charts.js", "dayplayer.js", "landing.js"):
        result = subprocess.run([NODE, "--check", str(WEB_DIR / name)], capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"{name}: {result.stderr}"
