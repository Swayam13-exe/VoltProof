"""Prompts are site-agnostic; prompts, goals and fixtures never mention the held-out site."""

from __future__ import annotations

import re
from pathlib import Path

from voltproof.bench.intents import INTENTS_PATH
from voltproof.data.sites import SITES
from voltproof.llm import prompts

FIXTURES = Path(__file__).parent / "fixtures"


def held_out_markers() -> list[str]:
    markers = []
    for site in SITES.values():
        if site.held_out:
            markers += [site.site_id, site.name, site.country, site.timezone, site.name.split()[0]]
    assert markers, "expected at least one held-out site"
    return markers


def site_markers() -> list[str]:
    return [m for site in SITES.values() for m in (site.site_id, site.name, site.country, site.timezone, site.currency, site.critical_load_name)]


def test_prompts_name_no_site_country_currency_or_timezone() -> None:
    text = "\n".join(prompts.ALL_PROMPTS)
    for marker in site_markers():
        assert not re.search(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE), marker


def test_no_held_out_site_in_prompts_intents_or_fixtures() -> None:
    sources = {"prompts": "\n".join(prompts.ALL_PROMPTS), "intents.yaml": INTENTS_PATH.read_text(encoding="utf-8")}
    for path in FIXTURES.rglob("*"):
        if path.is_file():
            sources[str(path)] = path.read_text(encoding="utf-8", errors="replace")
    for name, text in sources.items():
        for marker in held_out_markers():
            assert marker.lower() not in text.lower(), f"{marker!r} found in {name}"


def test_prompts_are_fixed_text() -> None:
    assert prompts.CONVENTIONS in prompts.A4_SYSTEM and prompts.CONVENTIONS in prompts.A3_SYSTEM and prompts.CONVENTIONS in prompts.A2_SYSTEM
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", "\n".join(prompts.ALL_PROMPTS))  # no dates: identical runs hit the cache
