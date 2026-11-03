"""Evaluate the bill reader on ten harder bills (SPEC §6.12). Dev tool; one pass only.

PowerShell, from the repo root:
    python scripts\\make_vision_samples.py --hard   # draws data/vision_eval/ (no LLM call)
    python scripts\\vision_eval.py --dry-run        # the plan and the worst-case cost, no call
    python scripts\\vision_eval.py                  # ONE pass: ten real model calls

Each bill goes through exactly what the app does with an upload: sanitize_image (re-encoded JPEG, no
metadata, at most 1600 px), read_bill (one uncached call, purpose "bill"), then check_bill against a
site in the bill's `site_currency`. The plausibility reference for each currency is the average hourly
import rate of that currency's sample bill in data/vision_samples/ (it only adds warnings).

Outcomes per bill:
- expect "read":   read correctly = the check passes and the 24 hourly rates, the kind and the export
                   rate equal the truth; anything else is wrong
- expect "refuse": correctly refused = the check blocks it (the model said unreadable, or a validator
                   blocked it); a passing check is wrong

The guard is tightened to this run's cap: ledger total + $0.15. Calls are tagged {"run": "vision-eval"}.
Stops at the first fatal error or budget refusal. It refuses to run again once results exist: the
point is one honest pass with no prompt tuning. Writes results/vision_eval/results.json and README.md.
Never reads the key.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltproof.config import ConfigError, get_settings  # noqa: E402
from voltproof.llm.client import BudgetExceeded, LLMClient, LLMError, LLMFatalError, read_ledger_totals, worst_case_cost_usd  # noqa: E402
from voltproof.vision.extract import image_messages, read_bill  # noqa: E402
from voltproof.vision.image import OUTPUT_MIME, sanitize_image  # noqa: E402
from voltproof.vision.validate import check_bill  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BILLS = ROOT / "data" / "vision_eval"
SAMPLES = ROOT / "data" / "vision_samples"
OUT = ROOT / "results" / "vision_eval"
CAP_USD = 0.15
TAGS = {"run": "vision-eval"}
MIME = {".png": "image/png", ".jpg": "image/jpeg"}
REFERENCE = {"INR": "bill_1_tou_inr", "KES": "bill_2_flat_kes", "EUR": "bill_3_tou_eur"}
RATE_TOL = 0.0005


def bill_files() -> list[Path]:
    return sorted(p for p in BILLS.iterdir() if p.suffix in MIME)


def truth_for(image: Path) -> dict[str, Any]:
    return json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))


def reference_price(currency: str) -> float:
    """The average hourly import rate of this currency's sample bill (for the plausibility warning)."""
    truth = json.loads((SAMPLES / f"{REFERENCE[currency]}.json").read_text(encoding="utf-8"))
    hourly = [0.0] * 24
    for p in truth["periods"]:
        for hour in range(p["start_hour"], p["end_hour"]):
            hourly[hour] = p["import_rate"]
    return sum(hourly) / 24


def truth_hourly(tariff: Mapping[str, Any]) -> list[float]:
    hourly = [0.0] * 24
    for p in tariff["periods"]:
        for hour in range(int(p["start_hour"]), int(p["end_hour"])):
            hourly[hour] = float(p["import_rate"])
    return hourly


def close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= RATE_TOL


def outcome(truth: Mapping[str, Any], check: Mapping[str, Any]) -> tuple[str, str]:
    """("read correctly" | "correctly refused" | "wrong", why)."""
    if truth["expect"] == "refuse":
        if check["ok"]:
            return "wrong", "accepted a bill that should be refused"
        return "correctly refused", "model said unreadable" if check["unreadable"] else "validator: " + " | ".join(check["errors"])
    if not check["ok"]:
        return "wrong", ("model said unreadable" if check["unreadable"] else "blocked: " + " | ".join(check["errors"]))
    want = truth["tariff"]
    bad = []
    if any(not close(g, w) for g, w in zip(check["hourly"], truth_hourly(want))):
        bad.append("hourly rates")
    if check["kind"] != want["tariff_kind"]:
        bad.append("kind")
    if not close(check["export_rate"], want["export_rate"]):
        bad.append("export rate")
    return ("read correctly", "") if not bad else ("wrong", "accepted with wrong " + ", ".join(bad))


def readme(results: Mapping[str, Any]) -> str:
    rows = results["bills"]
    counts = results["counts"]
    lines = [
        "# Bill reader: evaluation on ten harder bills",
        "",
        f"One pass on {results['started']} (UTC), model `{results['model']}`, reasoning effort `{results['reasoning_effort']}`. "
        "No prompt tuning and no re-run: these are the first and only results. Bills: `data/vision_eval/` "
        "(drawn by `python scripts/make_vision_samples.py --hard`); script: `scripts/vision_eval.py`; raw answers: `results.json`.",
        "",
        "Each bill went through the app's own path for an upload: `sanitize_image` → `read_bill` (one uncached call) → `check_bill` "
        "against a site in the bill's `site_currency`.",
        "",
        "## Counts",
        "",
        "| outcome | bills |",
        "|---|---|",
        f"| read correctly | {counts['read correctly']} |",
        f"| correctly refused | {counts['correctly refused']} |",
        f"| wrong | {counts['wrong']} |",
        f"| not run (error or budget stop) | {counts['not run']} |",
        "",
        f"Expected: {sum(1 for r in rows if r['expect'] == 'read')} to read, {sum(1 for r in rows if r['expect'] == 'refuse')} to refuse.",
        "",
        f"Cost: ${results['cost_usd']:.4f} by the ledger (cap ${CAP_USD:.2f}); per-call costs in `results.json`.",
        "",
        "## Per bill",
        "",
        "| bill | tests | expect | outcome | details | cost | latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cost = f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "-"
        latency = f"{r['latency_s']} s" if r.get("latency_s") is not None else "-"
        details = (r.get("why") or r.get("error") or "").replace("|", "/")
        lines.append(f"| {r['file']} | {r['tests']} | {r['expect']} | {r['outcome']} | {details} | {cost} | {latency} |")
    failures = [r for r in rows if r["outcome"] in ("wrong", "not run")]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    for r in failures:
        lines.append(f"- **{r['file']}** ({r['tests']}): {r.get('why') or r.get('error')}. Model answer: `{json.dumps(r.get('answer'), ensure_ascii=False)}`")
    lines += ["", "Warnings (\"check this rate\") are shown to the user but do not block; they are listed per bill in `results.json`.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the worst-case cost, no call")
    args = parser.parse_args(argv)
    files = bill_files()
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    spent = read_ledger_totals(settings.ledger_path).total_usd
    guarded = dataclasses.replace(settings, budget_usd_cap=spent + CAP_USD)
    cleaned = {f: sanitize_image(f.read_bytes(), MIME[f.suffix]) for f in files}
    worst = [worst_case_cost_usd(image_messages(cleaned[f], OUTPUT_MIME), None, None, guarded) for f in files]
    print(f"{len(files)} bills; model {settings.llm_model}, effort {settings.llm_reasoning_effort}; ledger ${spent:.4f}; cap for this run ${CAP_USD:.2f}; guard reserves up to ${max(worst):.4f} per call")
    if args.dry_run:
        for f in files:
            t = truth_for(f)
            print(f"  {f.name:<26} expect {t['expect']:<7} site {t['site_currency']}  ({t['why']})")
        return 0
    if (OUT / "results.json").is_file():
        print("results/vision_eval/results.json exists: this evaluation runs once. Not running again.", file=sys.stderr)
        return 5
    client = LLMClient(guarded)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows: list[dict[str, Any]] = []
    stopped = None
    for f in files:
        truth = truth_for(f)
        row: dict[str, Any] = {"file": f.name, "tests": truth["why"], "expect": truth["expect"], "site_currency": truth["site_currency"]}
        if stopped:
            rows.append(row | {"outcome": "not run", "error": stopped})
            continue
        try:
            read = read_bill(client, cleaned[f], OUTPUT_MIME, tags=TAGS)
        except (LLMFatalError, BudgetExceeded) as exc:
            stopped = f"{type(exc).__name__}: {exc}"
            rows.append(row | {"outcome": "not run", "error": stopped})
            print(f"{f.name}: STOPPED, {stopped}")
            continue
        except LLMError as exc:
            rows.append(row | {"outcome": "not run", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{f.name}: ERROR, {type(exc).__name__}")
            continue
        check = check_bill(read["answer"], truth["site_currency"], reference_price(truth["site_currency"]))
        result, why = outcome(truth, check)
        row |= {"outcome": result, "why": why, "answer": read["answer"], "json": read["json"], "errors": check["errors"], "warnings": check["warnings"],
                "fixed_charge_read": check["fixed_charge"], "fixed_charge_truth": (truth["tariff"] or {}).get("fixed_charge"), "cost_usd": round(read["cost_usd"], 5), "latency_s": round(read["latency_s"], 2)}
        rows.append(row)
        print(f"{f.name}: {result}{f' ({why})' if why else ''}; ${read['cost_usd']:.4f}, {read['latency_s']:.1f} s")
    cost = read_ledger_totals(settings.ledger_path).total_usd - spent
    counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in ("read correctly", "correctly refused", "wrong", "not run")}
    results = {"started": started, "model": settings.llm_model, "reasoning_effort": settings.llm_reasoning_effort, "cap_usd": CAP_USD, "cost_usd": round(cost, 5), "counts": counts, "bills": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(readme(results), encoding="utf-8")
    print(f"counts {counts}; cost ${cost:.4f} (cap ${CAP_USD:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
