"""The page's pure helpers (web/app.js), run in Node on committed replays: missed goals in plain words
and the cost line against the rule-based battery. Skipped where Node is not installed."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from voltproof.api.app import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");  // node -e: argv[1] is the first argument
const slice = (from, to) => src.slice(src.indexOf(from), src.indexOf(to));
const code = slice("function money(", "function status(") + slice("function hourText(", "function stat(");
const helpers = new Function(code + "; return { costLine, missedText };")();
const out = {};
for (const file of process.argv.slice(2)) {
  const view = JSON.parse(fs.readFileSync(file, "utf8"));
  out[file.split(/[\\/]/).pop()] = helpers.costLine(view.same_day, view.site.currency, view.spec, "default");
}
console.log(JSON.stringify(out));
"""


def run(*replays: str) -> dict:
    paths = [str(WEB_DIR / "replays" / name) for name in replays]
    result = subprocess.run([NODE, "-e", HARNESS, str(WEB_DIR / "app.js"), *paths], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cost_line_three_cases() -> None:
    lines = run("ahmedabad-outage-en.json", "nairobi-noon-midnight-en.json", "ahmedabad-outage-hi.json")
    cheaper = lines["ahmedabad-outage-en.json"]
    assert cheaper["tone"] == "ok" and cheaper["value"].endswith("cheaper") and cheaper["explain"] == "default"
    met_goal = lines["nairobi-noon-midnight-en.json"]  # VoltProof costs more; the rule-based battery missed goals
    assert met_goal["tone"] == "neutral" and met_goal["value"].endswith("more expensive")
    assert met_goal["explain"].startswith("Costs more because it met the goal; the rule-based battery was cheaper only by missing: battery reserve: at least 50% from 12:00 to midnight")
    assert "charge target: at least 95% by 12:00" in met_goal["explain"]
    both_met = lines["ahmedabad-outage-hi.json"]  # costs more and both met the goals: stays red
    assert both_met["tone"] == "fail" and both_met["value"].endswith("more expensive") and both_met["explain"] == "default"
