"""Independent reviewer: separate context, no tools, structured JSON, malformed output handling."""

from __future__ import annotations

import pytest

from tests.fakes_llm import ScriptedLLM, make_context, reply
from voltproof.agents.gate import REVIEW_FORMAT, parse_review, review_messages, run_review, site_summary
from voltproof.llm.client import BudgetExceeded, LLMAccountError, LLMUnavailable
from voltproof.llm.prompts import REVIEWER_SYSTEM
from voltproof.models import IntentSpec

SPEC = IntentSpec.model_validate({"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}], "assumptions": ["'half' read as 50%"]})


def test_parse_review_variants() -> None:
    assert parse_review('{"approve": true, "issues": []}').approve is True
    fenced = parse_review('Here:\n```json\n{"approve": false, "issues": [{"kind": "missing", "detail": "no outage"}]}\n```')
    assert fenced.approve is False and fenced.issues[0].kind == "missing"
    assert parse_review("I approve!") is None
    assert parse_review('{"approve": false, "issues": [{"kind": "style", "detail": "x"}]}') is None
    assert parse_review(None) is None


def test_reviewer_call_is_fresh_toolless_and_structured() -> None:
    llm = ScriptedLLM(review=[reply('{"approve": true, "issues": []}')])
    ctx = make_context("Keep half the battery from 6 PM to 10 PM.")
    outcome = run_review(llm, ctx, SPEC, ["evening = 18-22"], {"arm": "A4"})
    assert outcome.approved is True and outcome.issues == []
    (sent,) = llm.calls
    assert sent["purpose"] == "review" and sent["tools"] is None and sent["response_format"] == REVIEW_FORMAT
    assert [m["role"] for m in sent["messages"]] == ["system", "user"] and sent["messages"][0]["content"] == REVIEWER_SYSTEM
    user = sent["messages"][1]["content"]
    assert ctx.goal in user and site_summary(ctx) in user and '"reserve_windows"' in user
    assert "- 'half' read as 50%" in user and "- evening = 18-22" in user and "assumptions\":" not in user


def test_site_summary_lists_type_critical_load_and_periods() -> None:
    assert site_summary(make_context()) == (
        "Site: commercial_cold_store; critical load: test fridge; tariff: time_of_use; named periods: cheap [0,6); normal [6,17) [21,24); peak [17,21)"
    )


def test_empty_spec_is_described() -> None:
    assert "{} (no constraints: cost minimization only)" in review_messages(make_context(), IntentSpec(), [])[1]["content"]


def test_disapproval_issues_are_kept() -> None:
    llm = ScriptedLLM(review=[reply('{"approve": false, "issues": [{"kind": "wrong", "detail": "window should end at 22"}]}')])
    outcome = run_review(llm, make_context(), SPEC, [], {})
    assert outcome.approved is False and outcome.issues[0].kind == "wrong"


def test_disapproval_without_issues_gets_a_generic_issue() -> None:
    outcome = run_review(ScriptedLLM(review=[reply('{"approve": false, "issues": []}')]), make_context(), SPEC, [], {})
    assert outcome.approved is False and outcome.issues[0].detail == "the reviewer did not approve but gave no details"


def test_malformed_output_gives_no_verdict() -> None:
    outcome = run_review(ScriptedLLM(review=[reply("Looks fine to me.")]), make_context(), SPEC, [], {})
    assert outcome.approved is None and outcome.issues[0].kind == "malformed" and outcome.raw == "Looks fine to me."


def test_llm_error_gives_no_verdict() -> None:
    outcome = run_review(ScriptedLLM(review=[LLMUnavailable("503")]), make_context(), SPEC, [], {})
    assert outcome.approved is None and "LLMUnavailable" in outcome.issues[0].detail


@pytest.mark.parametrize("error", [BudgetExceeded("cap"), LLMAccountError("402", status_code=402)])
def test_budget_and_fatal_errors_propagate(error: Exception) -> None:
    with pytest.raises(type(error)):
        run_review(ScriptedLLM(review=[error]), make_context(), SPEC, [], {})
