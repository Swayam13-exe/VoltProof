"""Shared tool loop (SPEC §6.9) with a scripted fake LLM: no network."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from tests.fakes_llm import ScriptedLLM, call, reply
from voltproof.agents.loop import Tool, ToolOutcome, args_schema, parse_json_object, run_tool_loop
from voltproof.agents.tools import FinalizeArgs, solve_plan_parameters
from voltproof.llm.client import BudgetExceeded, LLMAccountError, LLMUnavailable, MissingAPIKey
from voltproof.llm.prompts import FINALIZE_NUDGE, LAST_TURN_NOTE


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int


def echo_tools(log: list[Any] | None = None) -> list[Tool]:
    log = log if log is not None else []

    def echo(args: EchoArgs) -> dict[str, Any]:
        log.append(args.x)
        return {"x": args.x}

    def boom(_args: BaseModel) -> dict[str, Any]:
        raise RuntimeError("boom")

    def done(_args: BaseModel) -> ToolOutcome:
        return ToolOutcome({"accepted": True}, stop=True)

    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    return [Tool("echo", "echo x", EchoArgs, echo), Tool("boom", "fails", NoArgs, boom), Tool("finalize", "ends", NoArgs, done)]


def tool_messages(llm: ScriptedLLM, index: int) -> list[dict[str, Any]]:
    return [m for m in llm.calls[index]["messages"] if m["role"] == "tool"]


def test_parallel_calls_run_in_order_with_matching_ids() -> None:
    log: list[int] = []
    llm = ScriptedLLM(reply(None, [call("echo", {"x": 1}, "a"), call("echo", {"x": 2}, "b")]), reply("done"))
    result = run_tool_loop(llm, "sys", "user", echo_tools(log), purpose="t")
    assert log == [1, 2]
    second = llm.calls[1]["messages"]
    assistant = second[2]
    assert assistant["role"] == "assistant" and assistant["content"] == ""  # None becomes ""
    assert [c["id"] for c in assistant["tool_calls"]] == ["a", "b"]
    tools = tool_messages(llm, 1)
    assert [m["tool_call_id"] for m in tools] == ["a", "b"]
    assert [json.loads(m["content"]) for m in tools] == [{"x": 1}, {"x": 2}]
    assert result.stop_reason == "final_answer" and result.tool_calls == 2 and result.turns == 2


def test_tool_errors_bad_json_and_bad_args_go_back_to_the_model() -> None:
    llm = ScriptedLLM(
        reply(None, [call("boom", {}, "a"), call("echo", "{not json", "b"), call("echo", {"x": "abc"}, "c"), call("echo", [1], "d")]),
        reply("ok"),
    )
    run_tool_loop(llm, "sys", "user", echo_tools(), purpose="t")
    results = [json.loads(m["content"]) for m in tool_messages(llm, 1)]
    assert results[0] == {"error": "RuntimeError: boom"}
    assert results[1]["error"].startswith("arguments are not valid JSON")
    assert results[2]["error"].startswith("invalid arguments: x:")
    assert results[3] == {"error": "arguments must be a JSON object"}


def test_unknown_tool_is_rejected_and_counted() -> None:
    llm = ScriptedLLM(reply(None, [call("delete_everything", {}, "a"), call("echo", {"x": 1}, "b")]), reply("ok"))
    result = run_tool_loop(llm, "sys", "user", echo_tools(), purpose="t")
    first = json.loads(tool_messages(llm, 1)[0]["content"])
    assert first["error"].startswith("unknown tool 'delete_everything'. Available tools: echo, boom, finalize")
    assert result.unknown_tool_calls == 1


def test_at_most_max_tool_calls_run_per_turn() -> None:
    log: list[int] = []
    calls = [call("echo", {"x": i}, f"c{i}") for i in range(10)]
    llm = ScriptedLLM(reply(None, calls), reply("ok"))
    result = run_tool_loop(llm, "sys", "user", echo_tools(log), purpose="t", max_tool_calls=8)
    assert log == list(range(8)) and result.tool_calls == 8
    tools = tool_messages(llm, 1)
    assert len(tools) == 10  # every call id still gets an answer
    assert "at most 8 are run per turn" in json.loads(tools[9]["content"])["error"]


def test_turn_cap_stops_and_warns_on_the_last_turn() -> None:
    llm = ScriptedLLM(*[reply(None, [call("echo", {"x": i}, f"c{i}")]) for i in range(3)])
    result = run_tool_loop(llm, "sys", "user", echo_tools(), purpose="t", max_turns=3, require_tool_end=True)
    assert result.stop_reason == "max_turns" and result.turns == 3 and len(llm.calls) == 3
    assert llm.calls[2]["messages"][-1] == {"role": "user", "content": LAST_TURN_NOTE}
    assert all(m.get("content") != LAST_TURN_NOTE for m in llm.calls[1]["messages"])


def test_text_answer_is_nudged_once_when_a_tool_must_end_the_episode() -> None:
    llm = ScriptedLLM(reply("I think we are done"), reply("still talking"))
    result = run_tool_loop(llm, "sys", "user", echo_tools(), purpose="t", require_tool_end=True)
    assert llm.calls[1]["messages"][-1] == {"role": "user", "content": FINALIZE_NUDGE}
    assert result.stop_reason == "final_answer" and result.final_text == "still talking"


def test_stopping_tool_ends_the_loop_after_the_turn() -> None:
    log: list[int] = []
    llm = ScriptedLLM(reply(None, [call("finalize", {}, "a"), call("echo", {"x": 5}, "b")]))
    result = run_tool_loop(llm, "sys", "user", echo_tools(log), purpose="t")
    assert result.stop_reason == "finalized" and log == [5] and len(llm.calls) == 1


def test_budget_exceeded_stops_the_loop() -> None:
    llm = ScriptedLLM(reply(None, [call("echo", {"x": 1})]), BudgetExceeded("cap"))
    result = run_tool_loop(llm, "sys", "user", echo_tools(), purpose="t")
    assert result.stop_reason == "budget_exceeded" and result.turns == 2


def test_budget_exceeded_inside_a_tool_stops_the_loop() -> None:
    class NoArgs(BaseModel):
        pass

    def reviewer(_args: Any) -> dict[str, Any]:
        raise BudgetExceeded("cap")

    llm = ScriptedLLM(reply(None, [call("review", {})]))
    result = run_tool_loop(llm, "sys", "user", [Tool("review", "x", NoArgs, reviewer)], purpose="t")
    assert result.stop_reason == "budget_exceeded"


@pytest.mark.parametrize("error", [LLMAccountError("credits", status_code=402), MissingAPIKey("no key")])
def test_fatal_errors_propagate(error: Exception) -> None:
    with pytest.raises(type(error)):
        run_tool_loop(ScriptedLLM(error), "sys", "user", echo_tools(), purpose="t")


def test_non_fatal_llm_error_ends_the_episode() -> None:
    result = run_tool_loop(ScriptedLLM(LLMUnavailable("503", status_code=503)), "sys", "user", echo_tools(), purpose="t")
    assert result.stop_reason == "llm_error:LLMUnavailable"


def test_trace_events_and_totals() -> None:
    llm = ScriptedLLM(reply(None, [call("echo", {"x": 1})], prompt=1000, completion=200, cost_usd=0.002), reply("ok", prompt=1200, completion=10, cost_usd=0.001))
    result = run_tool_loop(llm, "sys", "user", echo_tools(), purpose="agent", tags={"arm": "T"})
    kinds = [e.kind for e in result.trace.events]
    assert kinds == ["llm_call", "tool_call", "tool_result", "llm_call", "final"]
    assert result.trace.events[1].data["args"] == {"x": 1} and result.trace.events[2].data["result"] == {"x": 1}
    assert result.totals.tokens_in == 2200 and result.totals.tokens_out == 210
    assert result.totals.cost_usd == pytest.approx(0.003) and result.totals.calls == 2
    assert llm.calls[0]["tags"] == {"arm": "T", "turn": 1} and llm.calls[0]["purpose"] == "agent"
    assert llm.calls[0]["tool_choice"] == "auto" and llm.calls[0]["tools"][0]["function"]["name"] == "echo"


def test_schemas_have_no_refs_titles_or_docstrings() -> None:
    for schema in (args_schema(FinalizeArgs), solve_plan_parameters()):
        text = json.dumps(schema)
        assert "$ref" not in text and '"title"' not in text and "SPEC" not in text


def test_parse_json_object_accepts_fenced_json() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
