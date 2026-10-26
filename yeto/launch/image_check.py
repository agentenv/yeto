"""Start check of a yeto engine image from Python (S26 E0.4).

A Docker image runs ``docker/image-assert/entrypoint.sh`` before its
command. Modal does not run the image entrypoint, so the Harbor-layer images
(:mod:`yeto.rl.harness.harbor.image_layer`) are checked from yeto's own entry
points through :func:`check_image_at_start`: the same stdlib checker, against
the lock the image build wrote. Outside a yeto image (no lock file: CPU
tests, a developer machine) the check does nothing.
"""

from __future__ import annotations

import importlib.util
import json
import os
from functools import cache
from pathlib import Path

IMAGE_DIR = Path("/opt/yeto/image")
LOCK_ENV = "YETO_IMAGE_LOCK"


class ImageVersionMismatch(RuntimeError):
    """The running image differs from its version lock; the run must not start."""


def _checker(path: Path):
    spec = importlib.util.spec_from_file_location("yeto_image_assert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_image(lock_path: Path, checker_path: Path) -> list[dict]:
    """Rows of the check; raise :class:`ImageVersionMismatch` naming every failed row."""
    checker = _checker(checker_path)
    lock = json.loads(lock_path.read_text())
    report = checker.run(lock, commits_file=checker_path.parent / "commits.json")
    if report.failed:
        lines = "\n".join(f"  {row['check']}: {row['detail']}" for row in report.failed)
        raise ImageVersionMismatch(
            f"image {lock.get('image')} does not match its version lock {lock_path} "
            f"({len(report.failed)} failed check(s)):\n{lines}\n"
            "Rebuild the image, or review and update docker/harbor-layer/versions.json.")
    return report.rows


@cache
def check_image_at_start() -> tuple[dict, ...]:
    """Check the image once per process; empty when the process does not run in a yeto image."""
    lock_path = Path(os.environ.get(LOCK_ENV) or IMAGE_DIR / "versions.lock.json")
    checker_path = IMAGE_DIR / "assert_image_versions.py"
    if not lock_path.is_file():
        return ()
    if not checker_path.is_file():
        raise ImageVersionMismatch(f"{lock_path} exists but the checker {checker_path} is missing")
    return tuple(check_image(lock_path, checker_path))
