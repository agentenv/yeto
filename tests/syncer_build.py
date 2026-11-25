"""The debug syncer binary, built once per pytest process.

A cold ``cargo build`` of the yeto-syncer crate takes 8-40 s on a CI runner
(rust-cache restores the dependencies, never the workspace crate itself). When
the first test that needs the binary paid for that build in its setup, a test
with a short timeout failed with every case of its module (the module fixture's
error is cached). conftest.py calls ``build_syncer`` before the first test of a
session that selects a syncer test, so the build counts against no test's
timeout; later calls return the cached path without running cargo again.

Under an isolated HOME (goldens, baselines) cargo runs with the caller's own
toolchain: conftest.py applies :data:`TOOLCHAIN` (``yeto.launch.cargo_toolchain``)
to the environment before any test runs, so every ``cargo build`` of a test
module inherits it. With no toolchain at all, a binary at least as new as the
crate's sources is used as built, and a test whose binary cannot be had skips
with the reason (:func:`skip_reason`); it never fails for want of a toolchain.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from yeto.launch import cargo_toolchain

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "syncer/target/debug/yeto-syncer"
# Bounds the build on its own (it runs outside every test's timeout).
BUILD_TIMEOUT_S = 900

_lock = threading.Lock()
_built: Path | None = None

# Resolved once, at conftest import (before any test module computes its CARGO).
TOOLCHAIN = cargo_toolchain.resolve()


def binary_is_current(binary: Path = BINARY, crate: Path = ROOT / "syncer") -> bool:
    """Whether ``binary`` exists and is at least as new as every source of the crate."""
    if not binary.is_file():
        return False
    sources = [crate / "Cargo.toml", crate / "Cargo.lock", *(crate / "src").rglob("*")]
    newest = max((p.stat().st_mtime for p in sources if p.is_file()), default=0.0)
    return binary.stat().st_mtime >= newest


def skip_reason(item) -> str | None:
    """Why a collected test that needs a syncer binary cannot have one (None: it can).
    With no toolchain, a module that calls :func:`build_syncer` runs on a current
    binary; every other syncer test (its module runs cargo itself) skips."""
    if TOOLCHAIN.ok or not needs_cargo(item):
        return None
    module = getattr(item, "module", None)
    if module is not None and getattr(module, "build_syncer", None) is build_syncer and binary_is_current():
        return None
    return f"needs cargo to build the syncer: {TOOLCHAIN.reason}"


def needs_cargo(item) -> bool:
    """A syncer test (:func:`uses_syncer_build`), or one marked to skip without cargo."""
    return uses_syncer_build(item) or any("cargo" in str(m.kwargs.get("reason", ""))
                                          for m in item.iter_markers("skipif"))


def build_syncer() -> Path:
    """Return the debug syncer binary, running ``cargo build -q`` on the first call."""
    global _built
    with _lock:
        if _built is None and not TOOLCHAIN.ok:
            if not binary_is_current():
                raise RuntimeError(f"no syncer binary: {TOOLCHAIN.reason}, and {BINARY} is missing or older "
                                   "than the crate's sources")
            _built = BINARY  # built before, by a toolchain this environment cannot reach
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
