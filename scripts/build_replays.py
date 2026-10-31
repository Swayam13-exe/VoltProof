"""Build the page's precomputed replays from recorded A4 runs (SPEC §6.13): web/replays/*.json.

PowerShell, from the repo root:
    python scripts\\build_replays.py

No LLM call. Reads the recorded runs in runs/ (frozen test run, E2) and the committed weather CSVs;
each replay must reproduce its benchmark row (realized cost, goal met, unserved critical load) or
the build stops. See voltproof/api/replays.py for the chips and the checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltproof.api.replays import REPLAY_DIR, build_all  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    entries = build_all()
    for entry in entries:
        size = (REPLAY_DIR / entry["file"]).stat().st_size
        print(f"{entry['file']:<34} {size / 1024:6.1f} KB  {entry['site_id']:<22} {entry['language']}  {entry['goal'][:70]}")
    print(f"{len(entries)} replays and index.json in {REPLAY_DIR}; every one reproduces its benchmark row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
