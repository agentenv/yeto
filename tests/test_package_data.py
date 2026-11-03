"""Non-Python files the package reads at runtime ship in the wheel."""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Source-checkout-only files: docs, and the TMS preload binary that the SSH
# harness takes from the source tree it attests.
SOURCE_ONLY_SUFFIXES = {".md", ".so"}


def test_every_runtime_data_file_matches_package_data():
    package_data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    missing = []
    for path in sorted((REPO / "yeto").rglob("*")):
        if not path.is_file() or path.suffix in {".py", ".pyc"} | SOURCE_ONLY_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO)
        shipped = False
        for package, globs in package_data.items():
            base = REPO / package.replace(".", "/")
            if base in path.parents:
                inner = path.relative_to(base).as_posix()
                shipped = shipped or any(fnmatch.fnmatch(inner, g) and inner.count("/") == g.count("/") for g in globs)
        if not shipped:
            missing.append(rel.as_posix())
    assert missing == []
