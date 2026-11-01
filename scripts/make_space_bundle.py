"""Build a clean folder for a Hugging Face Docker Space: exactly the files the image needs (SPEC §6.14).

PowerShell, from the repo root:
    python scripts\\make_space_bundle.py ..\\VoltProof-space

The folder gets the Dockerfile, the run-time files it copies (the same list as the Dockerfile and
.dockerignore) and a Space README.md with the Space front matter (the repo's own README is not
touched). It refuses .env files, *.key files, .cache and runs/ anywhere in the bundle, and prints the
bundle size and the 10 largest files; files over 10 MB need git-lfs/xet on the Space. The output
folder must be new or empty and must lie outside the repo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# what the Dockerfile copies (keep in step with Dockerfile and .dockerignore)
INCLUDE = (
    "Dockerfile",
    "requirements.txt",
    "LICENSE",
    "voltproof",
    "web",
    "data/splits.json",
    "data/processed",
    "results/report/summary.json",
    "results/e2/summary.json",
    "results/e3/summary.json",
    "results/e4/summary.json",
)
SKIP_PARTS = ("__pycache__",)  # build leftovers: left out silently
SKIP_SUFFIXES = (".pyc",)
FORBIDDEN_DIRS = (".cache", "runs", ".git", ".venv", "tests")
LARGE_BYTES = 10 * 1024 * 1024

SPACE_README = """---
title: VoltProof AI
sdk: docker
app_port: 7860
license: mit
---

VoltProof AI: tell a solar-and-battery site what you need in plain words; an AI writes the rules, an optimizer plans, and a physics check verifies the plan.
Built with Muse Spark Open (Meta), served by Baseten. Weather data by Open-Meteo.com (CC BY 4.0). Sites, tariffs and loads are illustrative.
"""


class ForbiddenFile(ValueError):
    """A file that must never reach the Space (a secret or local state)."""


def forbidden(relative: Path) -> bool:
    name = relative.name
    return name == ".env" or name.startswith(".env.") or name.endswith(".key") or any(part in FORBIDDEN_DIRS for part in relative.parts)


def collect(root: Path = ROOT, include: tuple[str, ...] = INCLUDE) -> list[Path]:
    """Every file to bundle, relative to `root`; raises ForbiddenFile on a secret or local state."""
    files: list[Path] = []
    for entry in include:
        path = root / entry
        if not path.exists():
            raise FileNotFoundError(f"{entry} is missing (the Dockerfile copies it)")
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if any(part in SKIP_PARTS for part in relative.parts) or relative.suffix in SKIP_SUFFIXES:
                continue
            if forbidden(relative):
                raise ForbiddenFile(f"refusing to bundle {relative.as_posix()}")
            files.append(relative)
    return files


def build(out: Path, root: Path = ROOT) -> list[Path]:
    out = out.resolve()
    if out == root or root in out.parents:
        raise ValueError("the bundle folder must lie outside the repo")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    files = collect(root)
    for relative in files:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    (out / "README.md").write_text(SPACE_README, encoding="utf-8", newline="\n")
    return files + [Path("README.md")]


def report(out: Path, files: list[Path]) -> str:
    sizes = sorted(((out / f).stat().st_size, f.as_posix()) for f in files)
    total = sum(size for size, _ in sizes)
    lines = [f"{len(files)} files, {total / 1024 / 1024:.2f} MB in {out}", "largest:"]
    lines += [f"  {size / 1024 / 1024:7.2f} MB  {name}" for size, name in reversed(sizes[-10:])]
    large = [name for size, name in sizes if size > LARGE_BYTES]
    lines.append(f"over 10 MB (need git-lfs/xet): {', '.join(large) if large else 'none'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/make_space_bundle.py", description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", type=Path, default=ROOT.parent / "VoltProof-space", help="new or empty folder outside the repo")
    args = parser.parse_args(argv)
    try:
        files = build(args.out)
    except (ForbiddenFile, ValueError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(report(args.out.resolve(), files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
