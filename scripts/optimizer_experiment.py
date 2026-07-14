#!/usr/bin/env python3
"""CLI wrapper for the sibling :mod:`yeto.optimizer_harness` checkout."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yeto.optimizer_harness as _optimizer_harness  # noqa: E402


_EXPECTED_HARNESS = (REPO_ROOT / "yeto" / "optimizer_harness.py").resolve()
_ACTUAL_HARNESS = Path(_optimizer_harness.__file__).resolve()
if _ACTUAL_HARNESS != _EXPECTED_HARNESS:
    raise RuntimeError(
        "refusing optimizer experiment controller module drift: "
        f"loaded {_ACTUAL_HARNESS}, expected {_EXPECTED_HARNESS}"
    )

main = _optimizer_harness.main


if __name__ == "__main__":
    raise SystemExit(main())
