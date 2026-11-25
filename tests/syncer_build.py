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

The path returned is this process's own copy of the binary, not
``syncer/target/debug/yeto-syncer``: every ``cargo build``, a fresh one too,
deletes and rewrites that file, so a syncer another pytest process (a parallel
session, an xdist worker) starts while a module fixture here rebuilds fails to
exec or is killed by the OS before it prints anything (exit -9).
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from yeto.launch import cargo_toolchain

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "syncer/target/debug/yeto-syncer"
# Per-process copies, under the checkout so the orphan sweep (syncer_reaper) still sees them.
COPIES = ROOT / "syncer/target/test-bin"
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


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def remove_copies() -> None:
    """Remove this process's copies (conftest calls it at session end: the session leaves
    through ``os._exit``, which skips atexit, tests/hard_exit.py)."""
    shutil.rmtree(COPIES / str(os.getpid()), ignore_errors=True)


def private_copy(binary: Path = BINARY, timeout: float = 60.0) -> Path:
    """``binary`` copied to a path of this process only (``COPIES/<pid>/<n>/``), written
    under a temporary name and renamed into place, so no other process's ``cargo build``
    ever rewrites the file a test executes. Removed by :func:`remove_copies` or at exit; a
    dead process's copies are removed by the next process that makes one."""
    own = COPIES / str(os.getpid())
    if not own.exists():
        for stale in COPIES.glob("*") if COPIES.exists() else ():
            if stale.name.isdigit() and not _alive(int(stale.name)):
                shutil.rmtree(stale, ignore_errors=True)
        atexit.register(remove_copies)
    target = own / str(len(list(own.glob("*"))) if own.exists() else 0) / binary.name
    target.parent.mkdir(parents=True)
    staging = target.with_name(f".{binary.name}.partial")
    deadline = time.monotonic() + timeout
    while True:  # another process's cargo build may have deleted the file a moment ago
        try:
            shutil.copy2(binary, staging)
            break
        except FileNotFoundError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    os.replace(staging, target)
    return target


def build_syncer() -> Path:
    """Return this process's copy of the debug syncer binary, running ``cargo build -q``
    on the first call."""
    global _built
    with _lock:
        if _built is not None and not _built.is_file():
            _built = None  # this process's copy was removed: copy (and build) again
        if _built is None and not TOOLCHAIN.ok:
            if not binary_is_current():
                raise RuntimeError(f"no syncer binary: {TOOLCHAIN.reason}, and {BINARY} is missing or older "
                                   "than the crate's sources")
            _built = private_copy()  # built before, by a toolchain this environment cannot reach
        if _built is None:
            # Output captured: the pre-build runs outside pytest's capture, and
            # cargo's warnings would fill every shard's log.
            proc = subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", capture_output=True, text=True,
                                  timeout=BUILD_TIMEOUT_S)
            if proc.returncode != 0:
                raise RuntimeError(f"cargo build of the syncer failed ({proc.returncode}):\n{proc.stderr[-8000:]}")
            _built = private_copy()
        return _built


def uses_syncer_build(item) -> bool:
    """Whether a collected test needs the binary: it requests a ``syncer_binary``
    fixture, or its module calls this ``build_syncer`` (imported directly or via
    another test module)."""
    if "syncer_binary" in getattr(item, "fixturenames", ()):
        return True
    module = getattr(item, "module", None)
    return module is not None and getattr(module, "build_syncer", None) is build_syncer
