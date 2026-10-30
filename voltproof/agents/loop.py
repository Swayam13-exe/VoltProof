"""Shared tool-calling loop and trace (SPEC §6.9).

`run_tool_loop` drives one agent episode: it sends the conversation, runs the tool calls of each
assistant turn in the order given (several per turn = parallel calls, at most `max_tool_calls`
run), answers every call with one `role="tool"` message, and stops on a stopping tool result
(an accepted finalize), a text answer, the turn cap, or `BudgetExceeded`. Tool errors, unknown
tools and bad arguments go back to the model as `{"error": ...}`; fatal client errors (auth,
credits, persistent rate limits, missing key) propagate so the whole run stops.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ValidationError

from voltproof.llm.client import BudgetExceeded, LLMError, LLMFatalError, LLMResponse, MissingAPIKey
from voltproof.llm.prompts import FINALIZE_NUDGE, LAST_TURN_NOTE
from voltproof.models import TraceEvent, TraceKind

STOP_RUN_ERRORS: tuple[type[Exception], ...] = (LLMFatalError, MissingAPIKey)
SUMMARY_CHARS = 200
_FENCE_RE = re.compile(r"\{.*\}", re.DOTALL)


class ChatLLM(Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: Mapping[str, Any] | None = None,
        *,
        purpose: str,
        tags: Mapping[str, Any] | None = None,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class ToolOutcome:
    """A tool result that may end the episode (an accepted or escalated finalize)."""

    result: dict[str, Any]
    stop: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any], dict[str, Any] | ToolOutcome]
    parameters: dict[str, Any] | None = None  # overrides the schema generated from args_model

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters if self.parameters is not None else args_schema(self.args_model),
            },
        }


@dataclass
class LLMTotals:
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0

    def add(self, response: LLMResponse) -> None:
        self.tokens_in += response.usage.prompt
        self.tokens_out += response.usage.completion
        self.tokens_cached += response.usage.cached
        self.cost_usd += response.cost_usd
        self.latency_s += response.latency_s
        self.calls += 1


@dataclass
class Trace:
    """Ordered TraceEvents of one episode; `turn` is the current LLM turn (0 before the first)."""

    events: list[TraceEvent] = field(default_factory=list)
    turn: int = 0

    def add(
        self,
        kind: TraceKind,
        name: str | None = None,
        *,
        call_id: str | None = None,
        summary: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(kind=kind, turn=self.turn, name=name, call_id=call_id, summary=summary, data=data or {})
        )


@dataclass
class LoopResult:
    messages: list[dict[str, Any]]
    trace: Trace
    totals: LLMTotals
    turns: int
    tool_calls: int
    unknown_tool_calls: int
    stop_reason: str  # finalized | final_answer | max_turns | budget_exceeded | llm_error:<Class>
    final_text: str | None = None


def run_tool_loop(
    llm: ChatLLM,
    system: str,
    user: str,
    tools: Sequence[Tool],
    *,
    purpose: str,
    tags: Mapping[str, Any] | None = None,
    max_tool_calls: int = 8,
    max_turns: int = 10,
    require_tool_end: bool = False,
    trace: Trace | None = None,
    totals: LLMTotals | None = None,
) -> LoopResult:
    """Run the agent until it stops (see module docstring). `trace` and `totals` may be shared with
    the caller, so harness events (gate, reviewer) interleave with the loop's own."""
    trace = trace if trace is not None else Trace()
    totals = totals if totals is not None else LLMTotals()
    by_name = {tool.name: tool for tool in tools}
    schemas = [tool.schema() for tool in tools]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    counts = {"tool_calls": 0, "unknown": 0}
    stop_reason, final_text, nudged, turn = "max_turns", None, False, 0

    for turn in range(1, max_turns + 1):
        trace.turn = turn
        if require_tool_end and turn == max_turns and max_turns > 1:
            messages.append({"role": "user", "content": LAST_TURN_NOTE})
        try:
            response = llm.chat(messages, tools=schemas, tool_choice="auto", purpose=purpose, tags={**(tags or {}), "turn": turn})
        except STOP_RUN_ERRORS:
            raise
        except BudgetExceeded:
            stop_reason = "budget_exceeded"
            break
        except LLMError as exc:
            stop_reason = f"llm_error:{type(exc).__name__}"
            trace.add("final", "llm_error", summary=_short(str(exc)))
            break
        totals.add(response)
        trace.add("llm_call", purpose, summary=_short(response.content or ""), data=call_data(response))

        if not response.tool_calls:
            messages.append({"role": "assistant", "content": response.content or ""})
            if require_tool_end and not nudged and turn < max_turns:
                messages.append({"role": "user", "content": FINALIZE_NUDGE})
                nudged = True
                continue
            stop_reason, final_text = "final_answer", response.content
            break

        calls = [(call.id or f"call_{turn}_{i}", call.name, call.arguments_json) for i, call in enumerate(response.tool_calls)]
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                    for call_id, name, arguments in calls
                ],
            }
        )
        stop = False
        try:
            for index, (call_id, name, arguments) in enumerate(calls):
                if index >= max_tool_calls:
                    result = {"error": f"too many tool calls in one turn: at most {max_tool_calls} are run per turn; this one was skipped"}
                else:
                    counts["tool_calls"] += 1
                    result, stops = _run_call(by_name, name, arguments, call_id, trace, counts)
                    stop = stop or stops
                messages.append({"role": "tool", "tool_call_id": call_id, "content": to_json(result)})
        except BudgetExceeded:  # raised by a tool that calls the LLM (the A4 reviewer)
            stop_reason = "budget_exceeded"
            break
        if stop:
            stop_reason = "finalized"
            break

    trace.add("final", stop_reason, data={"turns": turn, "tool_calls": counts["tool_calls"]})
    return LoopResult(
        messages=messages,
        trace=trace,
        totals=totals,
        turns=turn,
        tool_calls=counts["tool_calls"],
        unknown_tool_calls=counts["unknown"],
        stop_reason=stop_reason,
        final_text=final_text,
    )


def _run_call(
    by_name: Mapping[str, Tool], name: str, arguments: str, call_id: str, trace: Trace, counts: dict[str, int]
) -> tuple[dict[str, Any], bool]:
    """Validate and run one tool call; every failure becomes an `{"error": ...}` result."""
    args_data = _loads(arguments)
    trace.add("tool_call", name, call_id=call_id, data={"args": args_data if isinstance(args_data, dict) else arguments})
    tool = by_name.get(name)
    stop = False
    if tool is None:
        counts["unknown"] += 1
        result: dict[str, Any] = {"error": f"unknown tool {name!r}. Available tools: {', '.join(by_name)}"}
    elif not isinstance(args_data, dict):
        result = {"error": f"arguments are not valid JSON: {args_data}" if isinstance(args_data, str) else "arguments must be a JSON object"}
    else:
        try:
            args = tool.args_model.model_validate(args_data)
        except ValidationError as exc:
            result = {"error": "invalid arguments: " + "; ".join(_validation_message(e) for e in exc.errors())}
        else:
            try:
                outcome = tool.run(args)
            except (BudgetExceeded, *STOP_RUN_ERRORS):
                raise
            except Exception as exc:  # any tool failure goes back to the model, never crashes the episode
                outcome = {"error": f"{type(exc).__name__}: {exc}"}
            if isinstance(outcome, ToolOutcome):
                result, stop = outcome.result, outcome.stop
            else:
                result = outcome
    trace.add("tool_result", name, call_id=call_id, summary=_short(to_json(result)), data={"result": result})
    return result, stop


def _loads(arguments: str) -> Any:
    """Parsed arguments, or the parse error text (a str) when they are not JSON."""
    try:
        return json.loads(arguments or "{}")
    except ValueError as exc:
        return str(exc)


def _validation_message(error: Mapping[str, Any]) -> str:
    path = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.get("loc", ())).lstrip(".")
    return f"{path or 'arguments'}: {error.get('msg', 'is invalid')}"


def call_data(response: LLMResponse) -> dict[str, Any]:
    return {
        "tokens_in": response.usage.prompt,
        "tokens_out": response.usage.completion,
        "tokens_cached": response.usage.cached,
        "tokens_reasoning": response.usage.reasoning,
        "cost_usd": response.cost_usd,
        "cache_hit": response.cache_hit,
        "finish_reason": response.finish_reason,
        "latency_s": round(response.latency_s, 3),
        "content": response.content,
        "reasoning": response.reasoning,
    }


def _short(text: str) -> str:
    return text if len(text) <= SUMMARY_CHARS else text[: SUMMARY_CHARS - 1] + "…"


def to_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


def parse_json_object(text: str | None) -> Any:
    """JSON from a model reply: the whole text, else the outermost {...} (e.g. inside a code fence).
    Raises ValueError when neither parses."""
    if text is None:
        raise ValueError("empty reply")
    try:
        return json.loads(text)
    except ValueError:
        match = _FENCE_RE.search(text)
        if match is None:
            raise ValueError("no JSON object in the reply") from None
        return json.loads(match.group(0))


def args_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema with every $ref inlined and titles removed (smaller, and no
    dependence on the endpoint resolving references). Class docstrings are developer notes, so
    model-level descriptions are dropped; Field descriptions are kept."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    schema.pop("description", None)
    return _strip(_inline(schema, defs))


def _inline(node: Any, defs: Mapping[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
            target.pop("description", None)
            extra = {key: value for key, value in node.items() if key != "$ref"}
            return _inline({**target, **extra}, defs)
        return {key: _inline(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    return node


def _strip(node: Any, parent_key: str | None = None) -> Any:
    if isinstance(node, dict):
        # "properties" maps field names to schemas: a field could be called "title", so keep its keys
        return {
            key: _strip(value, key)
            for key, value in node.items()
            if not (key == "title" and parent_key != "properties" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [_strip(item) for item in node]
    return node
