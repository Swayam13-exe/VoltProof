"""Data layer: sites, weather, PV, load, tariffs and days (SPEC §6.2)."""

from __future__ import annotations

import zlib
from datetime import date

import numpy as np

GLOBAL_SEED = 2026


def day_rng(seed: int, stream: str, day: date) -> np.random.Generator:
    """Independent, reproducible random stream per (seed, stream name, date)."""
    return np.random.default_rng([seed, zlib.crc32(stream.encode("utf-8")), day.toordinal()])
