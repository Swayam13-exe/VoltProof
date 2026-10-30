"""A4's finalize checks (SPEC §6.9), each a small function returning its failure messages.

1. verified plan (hard):        the plan exists, passes the physics rules, and every optimizer
                                shortfall is stated in the explanation (quoted within 5 % relative,
                                or rounded to the nearest whole unit)
2. checklist coverage (hard):   every checklist item maps to a field of the plan's spec or to an
                                assumption
3. grounded explanation (soft): every number in the explanation matches a number from this
                                episode's tool outputs, the goal or the finalized spec (1 % relative;
                                times of day must match an hour exactly)
4. reviewer (soft):             a separate LLM call with no tools and a fresh context judges whether
                                the spec captures the goal
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from voltproof.agents.loop import STOP_RUN_ERRORS, ChatLLM, parse_json_object, to_json
from voltproof.agents.tools import CoverageEntry, EpisodeContext, PlanRecord
from voltproof.core.spec import hours_to_windows
from voltproof.llm.client import BudgetExceeded, LLMError, LLMResponse
from voltproof.llm.prompts import REVIEWER_SYSTEM
from voltproof.models import IntentSpec, ReviewIssue, Shortfall

GROUND_REL_TOL = 0.01
SHORTFALL_REL_TOL = 0.05
LIST_FIELDS = ("grid_outages", "reserve_windows", "soc_targets", "import_caps", "no_export_windows", "discharge_only_windows")
SCALAR_FIELDS = ("min_soc_pct_all_day", "end_of_day_min_soc_pct", "max_throughput_kwh")
_REF_RE = re.compile(r"^(?:spec\.)?([a-z_]+)(?:\[(\d+)\])?$")
_TIME_HM_RE = re.compile(r"(?<![\w.,:])(\d{1,2}):(\d{2})(?!\d)")
_TIME_AMPM_RE = re.compile(r"(?<![\w.,])(\d{1,2})(?::(\d{2}))?\s?([ap])\.?\s?m\b\.?", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)(\s?(?:%|kWh|kW|h\b))?")

REVIEW_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "spec_review",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["approve", "issues"],
            "properties": {
                "approve": {"type": "boolean"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "detail"],
                        "properties": {
                            "kind": {"type": "string", "enum": ["missing", "wrong", "unneeded"]},
                            "detail": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


# --- numbers in text ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Number:
    text: str
    candidates: tuple[float, ...]  # every plausible reading ("2,500" may be 2.5 or 2500)
    hour: int | None = None  # set for times of day ("14:00", "2 PM")


def normalize_digits(text: str) -> str:
    """Map every Unicode decimal digit (e.g. Devanagari) to ASCII."""
    return "".join(
        str(unicodedata.decimal(ch)) if not ch.isascii() and unicodedata.decimal(ch, None) is not None else ch for ch in text
    )


def extract_numbers(text: str) -> list[Number]:
    text = normalize_digits(text)
    found: list[tuple[int, Number]] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    for match in _TIME_HM_RE.finditer(text):
        hour, minute = int(match.group(1)), int(match.group(2))
        if minute < 60:  # an impossible hour (25:00) stays one time token, and is ungrounded
            taken.append(match.span())
            found.append((match.start(), Number(match.group(0), (float(hour),), hour)))
    for match in _TIME_AMPM_RE.finditer(text):
        hour = int(match.group(1))
        if 1 <= hour <= 12 and free(match.span()):
            hour24 = hour % 12 + (12 if match.group(3).lower() == "p" else 0)
            taken.append(match.span())
            found.append((match.start(), Number(match.group(0).strip(), (float(hour24),), hour24)))
    for match in _NUMBER_RE.finditer(text):
        if free(match.span(1)):
            readings = _readings(match.group(1))
            if readings:
                found.append((match.start(), Number(match.group(1) + (match.group(2) or ""), readings)))
    return [number for _, number in sorted(found, key=lambda pair: pair[0])]


def _readings(raw: str) -> tuple[float, ...]:
    """Plausible values of a digit string with '.'/',' separators, in either convention."""
    dots, commas = raw.count("."), raw.count(",")
    if not dots and not commas:
        return (float(raw),)
    if dots and commas:  # the last separator is the decimal one
        decimal = "." if raw.rfind(".") > raw.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        return _floats(raw.replace(thousands, "").replace(decimal, "."))
    sep, count = (".", dots) if dots else (",", commas)
    head, _, tail = raw.rpartition(sep)
    if count > 1:  # 1.234.567 or 1,234,567
        return _floats(raw.replace(sep, ""))
    if len(tail) == 3:  # 2,500 / 2.500: decimal or thousands
        return _floats(f"{head}.{tail}", head + tail)
    return _floats(f"{head}.{tail}")


def _floats(*texts: str) -> tuple[float, ...]:
    out = []
    for text in texts:
        try:
            out.append(float(text))
        except ValueError:
            continue
    return tuple(dict.fromkeys(out))


@dataclass
class NumberPool:
    values: set[float] = field(default_factory=set)

    @property
    def hours(self) -> set[int]:
        return {int(v) for v in self.values if v == int(v) and 0 <= v <= 24}

    def add_text(self, text: str) -> None:
        for number in extract_numbers(text):
            self.values.update(number.candidates)

    def add(self, obj: Any) -> None:
        """Every number in a JSON-like object: values, numeric keys and numbers inside strings."""
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            self.values.add(abs(float(obj)))
        elif isinstance(obj, str):
            self.add_text(obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                self.add_text(str(key))
                self.add(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self.add(item)

    def grounds(self, number: Number) -> bool:
        if number.hour is not None:
            return number.hour in self.hours
        return any(_close(c, v) for c in number.candidates for v in self.values)


def _close(value: float, reference: float, rel_tol: float = GROUND_REL_TOL) -> bool:
    return value == reference or abs(value - reference) <= rel_tol * abs(reference)


def build_pool(tool_outputs: Iterable[Any], goal: str, spec: IntentSpec | None) -> NumberPool:
    pool = NumberPool()
    for output in tool_outputs:
        pool.add(output)
    pool.add_text(goal)
    if spec is not None:
        pool.add(spec.model_dump(mode="json", exclude={"assumptions"}))
    return pool


def ungrounded_numbers(explanation: str, pool: NumberPool) -> list[str]:
    return [number.text for number in extract_numbers(explanation) if not pool.grounds(number)]


# --- check 1: verified plan ------------------------------------------------------------------


def check_verified_plan(plan_id: str, record: PlanRecord | None, explanation: str, capacity_kwh: float) -> list[str]:
    if record is None:
        return [f'plan_id "{plan_id}" does not exist. Call solve_plan with your spec and finalize with the plan_id it returns.']
    if not record.physics_valid:
        physics = [v.message for v in record.check.violations if v.rule not in _GOAL_RULES][:2]
        return [
            f"plan {plan_id} failed physics verification ({record.check.total_violations} violations), "
            f"e.g. {'; '.join(physics)}. Change the spec and call solve_plan again."
        ]
    numbers = extract_numbers(explanation)
    unstated = [s for s in record.plan.shortfalls if not shortfall_stated(s, numbers, capacity_kwh)]
    if not unstated:
        return []
    listed = "; ".join(_shortfall_text(s, capacity_kwh) for s in unstated)
    noun = "goal" if len(unstated) == 1 else "goals"
    return [
        f"plan {plan_id} cannot fully meet {len(unstated)} {noun} and the explanation does not say by how much: {listed}. "
        "Say in the explanation what falls short and by how much (quote the shortfall from solve_plan), "
        "or fix the spec if you misread the goal."
    ]


_GOAL_RULES = frozenset({"RESERVE", "SOC_TARGET", "ALL_DAY_MIN", "END_SOC", "IMPORT_CAP", "NO_EXPORT", "DISCHARGE_WINDOW", "THROUGHPUT"})


def shortfall_amounts(shortfall: Shortfall, capacity_kwh: float) -> list[float]:
    if shortfall.unit == "kWh":
        return [shortfall.amount, shortfall.amount / capacity_kwh * 100.0]
    return [shortfall.amount]


def shortfall_stated(shortfall: Shortfall, numbers: Sequence[Number], capacity_kwh: float) -> bool:
    """Quoted within 5 % relative, or as the amount rounded to the nearest whole unit (≥ 1)."""
    for target in shortfall_amounts(shortfall, capacity_kwh):
        whole = round(target)
        for number in numbers:
            for value in number.candidates:
                if _close(value, target, SHORTFALL_REL_TOL) or (whole >= 1 and value == whole):
                    return True
    return False


def _shortfall_text(shortfall: Shortfall, capacity_kwh: float) -> str:
    where = f" at hours {shortfall.hours}" if shortfall.hours else ""
    if shortfall.unit == "kWh":
        pct = shortfall.amount / capacity_kwh * 100.0
        amount = f"{shortfall.amount:.6f} kWh ({pct:.6f}% of capacity)"
    else:
        amount = f"{shortfall.amount:.6f} kW"
    return f"{shortfall.rule} ({shortfall.path}) short by up to {amount}{where}"


# --- check 2: checklist coverage -------------------------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    messages: list[str]
    dropped: list[str]  # texts of unmapped checklist items
    coverage: float  # mapped / total


def check_coverage(
    checklist: Sequence[str], coverage: Sequence[CoverageEntry], spec: IntentSpec, assumptions: Sequence[str], plan_id: str
) -> CoverageResult:
    messages: list[str] = []
    n = len(checklist)
    reasons: dict[int, str] = {}
    mapped: set[int] = set()
    for entry in coverage:
        if not 1 <= entry.item <= n:
            messages.append(f"coverage refers to checklist item {entry.item}, which does not exist (items are 1-{n}).")
            continue
        problem = _ref_problem(entry.maps_to, spec, assumptions, plan_id)
        if problem is None:
            mapped.add(entry.item)
        else:
            reasons.setdefault(entry.item, problem)
    dropped = []
    for item in range(1, n + 1):
        if item in mapped:
            continue
        text = checklist[item - 1]
        dropped.append(text)
        if item in reasons:
            messages.append(f"checklist item {item} maps to {reasons[item]}")
        else:
            messages.append(
                f'checklist item {item} ("{text}") is not mapped. Map every item to a spec field of plan {plan_id} '
                '(e.g. "objective", "grid_outages[0]") or to one of your assumptions ("assumptions[0]").'
            )
    return CoverageResult(messages=messages, dropped=dropped, coverage=len(mapped) / n if n else 1.0)


def _ref_problem(ref: str, spec: IntentSpec, assumptions: Sequence[str], plan_id: str) -> str | None:
    """None if `ref` names something present; else the rest of the coverage message."""
    shown = f'"{ref}"'
    match = _REF_RE.match(ref.strip())
    if match is None:
        return f'{shown}, but {shown} is not a spec field or assumptions[i].'
    name, index = match.group(1), match.group(2)
    if name == "objective" and index is None:
        return None
    if name == "assumptions":
        if index is not None and int(index) < len(assumptions):
            return None
        return f"{shown}, but there is no assumptions[{index}] (you passed {len(assumptions)} assumptions)."
    if name in SCALAR_FIELDS and index is None:
        return None if getattr(spec, name) is not None else f"{shown}, but {name} is not set in plan {plan_id}'s spec."
    if name in LIST_FIELDS:
        items = getattr(spec, name) or []
        if (index is None and items) or (index is not None and int(index) < len(items)):
            return None
        target = name if index is None else f"{name}[{index}]"
        return f"{shown}, but plan {plan_id}'s spec has no {target}."
    return f'{shown}, but {shown} is not a spec field or assumptions[i].'


# --- check 4: reviewer -----------------------------------------------------------------------


class _VerdictIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["missing", "wrong", "unneeded"]
    detail: str


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    approve: bool
    issues: list[_VerdictIssue] = []


@dataclass(frozen=True)
class ReviewOutcome:
    approved: bool | None  # None: no usable verdict (malformed output or LLM error); never blocks
    issues: list[ReviewIssue]
    raw: str | None
    response: LLMResponse | None


def site_summary(ctx: EpisodeContext) -> str:
    periods = "; ".join(
        f"{name} " + " ".join(f"[{a},{b})" for a, b in hours_to_windows(hours)) for name, hours in sorted(ctx.tariff.periods.items())
    )
    return (
        f"Site: {ctx.site.site_type}; critical load: {ctx.site.critical_load_name}; "
        f"tariff: {ctx.tariff.kind}; named periods: {periods or 'none'}"
    )


def review_assumptions(spec: IntentSpec, assumptions: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*spec.assumptions, *assumptions]))


def review_messages(ctx: EpisodeContext, spec: IntentSpec, assumptions: Sequence[str]) -> list[dict[str, str]]:
    constraints = spec.model_dump(mode="json", exclude_defaults=True, exclude={"assumptions"})
    spec_text = to_json(constraints) if constraints else "{} (no constraints: cost minimization only)"
    lines = "\n".join(f"- {a}" for a in assumptions) or "- (none)"
    user = f"Operator's goal:\n{ctx.goal}\n\n{site_summary(ctx)}\n\nSpec:\n{spec_text}\n\nAssumptions:\n{lines}"
    return [{"role": "system", "content": REVIEWER_SYSTEM}, {"role": "user", "content": user}]


def parse_review(content: str | None) -> ReviewVerdict | None:
    try:
        return ReviewVerdict.model_validate(parse_json_object(content))
    except (ValueError, ValidationError):
        return None


def run_review(
    llm: ChatLLM, ctx: EpisodeContext, spec: IntentSpec, assumptions: Sequence[str], tags: dict[str, Any]
) -> ReviewOutcome:
    """One reviewer call (purpose="review"). Fatal errors and BudgetExceeded propagate; any other LLM
    error or unusable output gives approved=None."""
    messages = review_messages(ctx, spec, review_assumptions(spec, assumptions))
    try:
        response = llm.chat(messages, response_format=REVIEW_FORMAT, purpose="review", tags=tags)
    except (BudgetExceeded, *STOP_RUN_ERRORS):
        raise
    except LLMError as exc:
        return ReviewOutcome(None, [ReviewIssue(kind="malformed", detail=f"reviewer unavailable: {type(exc).__name__}")], None, None)
    verdict = parse_review(response.content)
    if verdict is None:
        detail = "reviewer output could not be parsed: " + json.dumps((response.content or "")[:120], ensure_ascii=False)
        return ReviewOutcome(None, [ReviewIssue(kind="malformed", detail=detail)], response.content, response)
    issues = [ReviewIssue(kind=i.kind, detail=i.detail) for i in verdict.issues]
    if not verdict.approve and not issues:
        issues = [ReviewIssue(kind="wrong", detail="the reviewer did not approve but gave no details")]
    return ReviewOutcome(verdict.approve, issues, response.content, response)


def review_message(outcome: ReviewOutcome) -> str:
    listed = "; ".join(f"[{issue.kind}] {issue.detail}" for issue in outcome.issues)
    return (
        f"the independent reviewer did not approve the spec: {listed}. If you agree, fix the spec (solve_plan again) "
        "or your assumptions; if you disagree, keep the spec and state your reading in assumptions."
    )


def grounding_message(ungrounded: Sequence[str]) -> str:
    listed = ", ".join(f'"{text}"' for text in ungrounded)
    return (
        f"these numbers in the explanation do not match any tool output in this episode or the goal: {listed}. "
        "Quote numbers as the tools reported them (rounding within 1% is fine; hours exactly), or remove them."
    )
