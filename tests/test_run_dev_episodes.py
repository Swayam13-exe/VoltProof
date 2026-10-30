"""scripts/run_dev_episodes.py stops at once on fatal LLM errors and on the budget cap (fake SDK)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.fakes import FakeSDK, make_client, make_response, status_error
from voltproof.config import Settings
from voltproof.llm.client import MAX_ATTEMPTS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dev_episodes.py"


@pytest.fixture
def script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_dev_episodes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_dev_episodes"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RUNS_DIR", tmp_path / "runs")
    yield module
    sys.modules.pop("run_dev_episodes", None)


@pytest.mark.parametrize("status", [401, 402, 403])
def test_stops_on_auth_or_credit_errors(script: ModuleType, settings: Settings, capsys: pytest.CaptureFixture[str], status: int) -> None:
    sdk = FakeSDK(status_error(status), *[make_response() for _ in range(5)])
    code = script.main(["--arms", "A2,A4", "--n", "2"], llm=make_client(settings, sdk), settings=settings)
    out = capsys.readouterr().out
    assert code == script.EXIT_FATAL
    assert f"STOPPED: LLMAccountError (HTTP {status})" in out and "No further LLM calls were made" in out
    assert len(sdk.calls) == 1


def test_stops_on_a_persistent_rate_limit(script: ModuleType, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    sdk = FakeSDK(*[status_error(429) for _ in range(MAX_ATTEMPTS)], make_response())
    code = script.main(["--arms", "A4", "--n", "2"], llm=make_client(settings, sdk), settings=settings)
    assert code == script.EXIT_FATAL and "STOPPED: LLMRateLimitPersisted (HTTP 429)" in capsys.readouterr().out
    assert len(sdk.calls) == MAX_ATTEMPTS


def test_stops_cleanly_on_the_budget_cap(script: ModuleType, settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    tiny = dataclasses.replace(settings, budget_usd_cap=1e-6, demo_budget_usd_cap=1e-6)
    sdk = FakeSDK(make_response())
    code = script.main(["--arms", "A4", "--n", "2"], llm=make_client(tiny, sdk), settings=tiny)
    assert code == script.EXIT_BUDGET and "STOPPED: BudgetExceeded" in capsys.readouterr().out
    assert sdk.calls == []


def test_dry_run_makes_no_call(script: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    class NoLLM:
        def chat(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("dry run must not call the LLM")

    assert script.main(["--arms", "A4", "--n", "1", "--dry-run"], llm=NoLLM()) == 0
    out = capsys.readouterr().out
    assert "selected 1 for arms A4" in out and "Gold feasibility" in out
