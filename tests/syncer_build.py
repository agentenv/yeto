"""The debug syncer binary, built once per pytest process.

A cold ``cargo build`` of the yeto-syncer crate takes 8-40 s on a CI runner
(rust-cache restores the dependencies, never the workspace crate itself). When
the first test that needs the binary paid for that build in its setup, a test
with a short timeout failed with every case of its module (the module fixture's
error is cached). conftest.py calls ``build_syncer`` before the first test of a
session that selects a syncer test, so the build counts against no test's
timeout; later calls return the cached path without running cargo again.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "syncer/target/debug/yeto-syncer"
# Bounds the build on its own (it runs outside every test's timeout).
BUILD_TIMEOUT_S = 900

_lock = threading.Lock()
_built: Path | None = None


def build_syncer() -> Path:
    """Return the debug syncer binary, running ``cargo build -q`` on the first call."""
    global _built
    with _lock:
        if _built is None:
            # Output captured: the pre-build runs outside pytest's capture, and
            # cargo's warnings would fill every shard's log.
            proc = subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", capture_output=True, text=True,
                                  timeout=BUILD_TIMEOUT_S)
            if proc.returncode != 0:
                raise RuntimeError(f"cargo build of the syncer failed ({proc.returncode}):\n{proc.stderr[-8000:]}")
            assert BINARY.is_file(), BINARY
            _built = BINARY
        return _built


def uses_syncer_build(item) -> bool:
    """Whether a collected test needs the binary: it requests a ``syncer_binary``
    fixture, or its module calls this ``build_syncer`` (imported directly or via
    another test module)."""
    if "syncer_binary" in getattr(item, "fixturenames", ()):
        return True
    module = getattr(item, "module", None)
    return module is not None and getattr(module, "build_syncer", None) is build_syncer
