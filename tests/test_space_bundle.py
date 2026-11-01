"""scripts/make_space_bundle.py and the Dockerfile: the Space gets exactly the run-time files, never a
secret or local state (SPEC §6.14). Offline; no Docker needed."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_space_bundle.py"


@pytest.fixture(scope="module")
def bundle() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_space_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_space_bundle"] = module
    spec.loader.exec_module(module)
    return module


def dockerfile_sources() -> list[str]:
    sources = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^COPY\s+(?:--\S+\s+)*(.+)\s+\S+$", line.strip())
        if match:
            sources += match.group(1).split()
    return [s.rstrip("/") for s in sources]


def test_bundle_has_the_dockerfile_the_space_readme_and_every_copied_path(bundle: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "space"
    files = {f.as_posix() for f in bundle.build(out)}
    assert {"Dockerfile", "README.md", "requirements.txt", "web/index.html", "web/replays/index.json", "voltproof/api/app.py", "data/splits.json"} <= files
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: VoltProof AI\nsdk: docker\napp_port: 7860\nlicense: mit\n---\n")
    assert (out / "Dockerfile").read_bytes() == (ROOT / "Dockerfile").read_bytes()
    for source in dockerfile_sources():  # everything the image copies is in the bundle
        assert source in files or any(f.startswith(f"{source}/") for f in files), source
    assert not any(part in {".cache", "runs", "tests", "__pycache__", ".git", ".venv"} for f in files for part in Path(f).parts)
    assert not any(Path(f).name.startswith(".env") or f.endswith((".key", ".pyc")) for f in files)
    assert (ROOT / "README.md").read_text(encoding="utf-8") != readme  # the repo README is untouched


def test_bundle_refuses_secrets_and_local_state(bundle: ModuleType, tmp_path: Path) -> None:
    for bad in ("voltproof/.env", "web/.env.local", "voltproof/api/server.key", "web/runs/ledger.jsonl", "voltproof/.cache/llm/x.json"):
        root = tmp_path / bad.replace("/", "_")
        (root / "voltproof").mkdir(parents=True)
        (root / "voltproof" / "ok.py").write_text("", encoding="utf-8")
        target = root / bad
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
        (root / "web").mkdir(exist_ok=True)
        with pytest.raises(bundle.ForbiddenFile, match=re.escape(bad)):
            bundle.collect(root, ("voltproof", "web"))


def test_bundle_must_be_outside_the_repo_and_empty(bundle: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repo"):
        bundle.build(ROOT / "space-bundle")
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        bundle.build(full)
    assert bundle.main([str(full)]) == 1


def test_dockerfile_runs_as_uid_1000_on_port_and_never_copies_secrets() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.startswith("#") and "FROM python:3.10-slim" in text and "--uid 1000" in text and "\nUSER app\n" in text
    assert "PORT=7860" in text and "${PORT}" in text and "--factory" in text and "--no-access-log" in text and "HEALTHCHECK" in text
    assert "LIVE_MODE=false" in text and "OPENBLAS_NUM_THREADS=1" in text
    for source in dockerfile_sources():
        assert not any(bad in source for bad in (".env", ".cache", "runs", "tests", "data/raw", ".git")), source
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ignore[ignore.index("*") if "*" in ignore else 0] == "*"  # allow-list: everything else is out
    for needed in dockerfile_sources():
        assert f"!{needed}" in ignore or f"!{needed}/" in ignore, needed
