"""Feasibility probe: can Muse Spark Open read a tariff off a bill image? (SPEC §6.12, dev tool only)

PowerShell, from the repo root:
    python scripts\\vision_probe.py --dry-run              # prints the projected cost and the budget, no call
    python scripts\\vision_probe.py --files bill_1_tou_inr.png
    python scripts\\vision_probe.py                         # every sample, clean and photo-like

Sends each image in data/vision_samples/ (see scripts/make_vision_samples.py) as an OpenAI-style
`image_url` data-URI content part through the existing LLMClient, so the ledger, cache, budget guard and
key handling are the project's own. The guard is tightened to this probe's own cap: ledger total + $0.50.
Calls are tagged {"run": "vision-probe"}. Stops at the first fatal error or budget refusal. Results go to
results/vision_probe/results.json (the model's raw answers and the scores). Never reads the key.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltproof.config import ConfigError, Settings, get_settings  # noqa: E402
from voltproof.vision.extract import INSTRUCTIONS, SYSTEM, image_messages, parse_reply  # noqa: E402,F401  (the prompt lives there)
from voltproof.llm.client import (  # noqa: E402
    LLMClient,
    LLMError,
    LLMFatalError,
    read_ledger_totals,
    worst_case_cost_usd,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "vision_samples"
RESULTS = ROOT / "results" / "vision_probe" / "results.json"
PROBE_CAP_USD = 0.50
EXPECTED_USD_PER_CALL = 0.02  # a rough guess for the projection only; the guard and the ledger use real numbers
TAGS = {"run": "vision-probe"}
MIME = {".png": "image/png", ".jpg": "image/jpeg"}
FIELDS = ("currency", "tariff_kind", "periods", "export_rate", "fixed_charge")
RATE_TOL, MONEY_TOL = 0.0005, 0.005

# --- the request ------------------------------------------------------------------------------------------


def messages_for(image: Path) -> list[dict[str, Any]]:
    return image_messages(image.read_bytes(), MIME[image.suffix])


def sample_files() -> list[Path]:
    return sorted(SAMPLES.glob("bill_*.png")) + sorted(SAMPLES.glob("bill_*_photo.jpg"))


def truth_for(image: Path) -> dict[str, Any]:
    name = image.stem.removesuffix("_photo")
    return json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))


# --- scoring ----------------------------------------------------------------------------------------------


def hourly(periods: Any) -> list[float | None] | None:
    """Rate for each of the 24 hours, or None if the periods are malformed, overlap or leave gaps."""
    if not isinstance(periods, list) or not periods:
        return None
    rates: list[float | None] = [None] * 24
    for p in periods:
        try:
            start, end, rate = int(p["start_hour"]), int(p["end_hour"]), float(p["import_rate"])
        except (TypeError, KeyError, ValueError):
            return None
        if not 0 <= start < end <= 24:
            return None
        for hour in range(start, end):
            if rates[hour] is not None:
                return None
            rates[hour] = rate
    return rates if None not in rates else None


def close(a: Any, b: Any, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def score(answer: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
    want = hourly(truth["periods"])
    got = hourly(answer.get("periods"))
    wrong_hours = list(range(24)) if got is None else [h for h in range(24) if not close(got[h], want[h], RATE_TOL)]
    boundaries = lambda rates: {h for h in range(1, 24) if rates[h] != rates[h - 1]} if rates else set()
    off_by_one = bool(wrong_hours) and got is not None and all(any(abs(b - t) == 1 for t in boundaries(want)) for b in boundaries(got) - boundaries(want)) and boundaries(got) != boundaries(want)
    fields = {
        "currency": str(answer.get("currency", "")).strip().upper() == truth["currency"],
        "tariff_kind": answer.get("tariff_kind") == truth["tariff_kind"],
        "periods": got is not None and not wrong_hours,
        "export_rate": close(answer.get("export_rate"), truth["export_rate"], RATE_TOL),
        "fixed_charge": close(answer.get("fixed_charge"), truth["fixed_charge"], MONEY_TOL),
    }
    return {
        "fields": fields,
        "all_correct": all(fields.values()),
        "hours_correct": 24 - len(wrong_hours),
        "wrong_hours": wrong_hours,
        "boundary_off_by_one": off_by_one,
        "periods_well_formed": got is not None,
    }


parse = parse_reply  # the probe's name for it


# --- the run ----------------------------------------------------------------------------------------------


def capped(settings: Settings) -> tuple[Settings, float]:
    spent = read_ledger_totals(settings.ledger_path).total_usd
    return dataclasses.replace(settings, budget_usd_cap=spent + PROBE_CAP_USD), spent


def load_results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}


def save_results(results: Mapping[str, Any]) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="*", help="sample file names (default: every sample)")
    parser.add_argument("--dry-run", action="store_true", help="print the projected cost and stop")
    args = parser.parse_args(argv)
    files = [SAMPLES / name for name in args.files] if args.files else sample_files()
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    guarded, spent = capped(settings)
    worst = [worst_case_cost_usd(messages_for(f), None, None, guarded) for f in files]
    print(f"model {settings.llm_model}, reasoning effort {settings.llm_reasoning_effort}")
    print(f"ledger total ${spent:.4f} of the ${settings.budget_usd_cap:.2f} project cap (${settings.budget_usd_cap - spent:.2f} left); probe cap ${PROBE_CAP_USD:.2f}")
    print(f"{len(files)} calls; projected about ${EXPECTED_USD_PER_CALL * len(files):.2f} (guess ${EXPECTED_USD_PER_CALL:.2f} each); the guard reserves up to ${max(worst):.3f} per call (base64 counted as text)")
    if args.dry_run:
        return 0
    client = LLMClient(guarded)
    results = load_results()
    for image in files:
        truth = truth_for(image)
        entry: dict[str, Any] = {"file": image.name, "image_kb": round(image.stat().st_size / 1024, 1)}
        try:
            response = client.chat(messages_for(image), purpose="vision-probe", tags=TAGS)
        except LLMFatalError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            results[image.name] = entry
            save_results(results)
            print(f"{image.name}: STOPPED, {entry['error']}")
            return 3
        except LLMError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            results[image.name] = entry
            save_results(results)
            print(f"{image.name}: ERROR, {entry['error']}")
            if "BudgetExceeded" in entry["error"]:
                return 4
            continue
        answer, how = parse(response.content)
        usage = response.usage
        entry |= {
            "reply": response.content,
            "json": how,
            "latency_s": round(response.latency_s, 2),
            "cost_usd": round(response.cost_usd, 5),
            "tokens_in": usage.prompt,
            "tokens_out": usage.completion,
            "tokens_reasoning": usage.reasoning,
            "finish_reason": response.finish_reason,
            "cache_hit": response.cache_hit,
            "answer": answer,
            "score": score(answer, truth) if isinstance(answer, dict) else None,
        }
        results[image.name] = entry
        save_results(results)
        s = entry["score"]
        verdict = "no usable JSON" if s is None else f"all correct: {s['all_correct']}, hours {s['hours_correct']}/24, fields {''.join('Y' if v else 'n' for v in s['fields'].values())}"
        print(f"{image.name}: json={how}, {verdict}; ${entry['cost_usd']:.4f}, {entry['latency_s']} s, {usage.prompt} in / {usage.completion} out ({usage.reasoning} reasoning)")
    print(f"probe spend this run: ${read_ledger_totals(settings.ledger_path).total_usd - spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
