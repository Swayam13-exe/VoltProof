"""Shared fixtures. Nothing here touches the network or the real `.env`."""

from __future__ import annotations

import dataclasses

import pytest

import voltproof.config as config
from voltproof.config import Settings, load_settings


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """No test may read the real .env or settings exported in the developer's shell."""
    monkeypatch.setattr(config, "DOTENV_PATH", tmp_path_factory.mktemp("no-dotenv") / "absent.env")
    for name in config.ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Defaults from .env.example, no key, cache and ledger under tmp_path."""
    return dataclasses.replace(
        load_settings({}),
        cache_dir=tmp_path / "cache" / "llm",
        ledger_path=tmp_path / "runs" / "ledger.jsonl",
    )
