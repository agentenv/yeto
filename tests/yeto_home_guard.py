"""Keep the test session out of the developer's real ``~/.yeto``.

The run registry, the resource and cost ledgers, the ``run up`` state, the NC pool
state and the ssh control sockets all live under ``~/.yeto``, and most of them are
resolved once, at import (``runs.RUNS_DIR``, ``up.STATE_ROOT``): a test that only
sets ``HOME`` after the import still writes the real files. A later ``yeto cleanup``,
``yeto runs`` or dashboard on that machine then reads fake leases and fake spend.

:func:`isolate` (called by conftest.py before any ``yeto`` module is imported) points
every state directory at one temporary root through the override each path already
reads. HOME itself is left alone: the syncer tests build with the caller's own
rustup/cargo (tests/syncer_build.py).

:func:`install_guard` then watches the real ``~/.yeto`` two ways:

* in this process, an audit hook refuses (``PermissionError``) every write, create,
  rename or delete under it and records the test that tried; any such attempt fails
  the session (:func:`session_report`), also when the code under test swallowed the
  error;
* for child processes, a snapshot of every entry's mtime and size at the start is
  compared at the end. New or changed entries are reported as a warning only: another
  process on this machine (a real run, another checkout's test session) may have
  written them.
"""

from __future__ import annotations

import os
import pwd
import shutil
import sys
import tempfile
from pathlib import Path

# Each state path and the override it reads (the default stays ~/.yeto/<...>).
STATE_ENV = {
    "YETO_RUNS_DIR": "runs",  # yeto.runs.RUNS_DIR: registry, resources.jsonl, cost-events.jsonl
    "YETO_STATE_DIR": "rl-up",  # yeto.rl.up.STATE_ROOT, dashboard sources/breakers
    "YETO_NC_STATE_DIR": "nc",  # yeto.launch.nc.tunnel.state_dir: pool / sky-cluster / modal records
    "YETO_SSH_CONTROL_DIR": "ssh",  # yeto.launch.ssh_transport control sockets
    "YETO_SANDBOX_DIR": "sandbox",  # yeto.sandbox.state_root: ``yeto sandbox prepare`` stage
    "YETO_ISLANDS_LOG_DIR": "islands",  # yeto.rl.trainer_worker.LocalLaunchOps launch logs
    "YETO_SSH_RUNS_DIR": "ssh-runs",  # yeto.rl.ssh_harness.prepare plans
    "YETO_VERDA_IDS_PATH": "verda-instance-ids.json",  # yeto.sky_patches.verda
}
# Points into one real run's ledger when inherited from an operator shell.
CLEARED_ENV = ("YETO_RESOURCE_LEDGER",)
ROOT_ENV = "YETO_TEST_STATE_ROOT"  # set by the first session; children and nested runs reuse it
# Restored around every test (conftest's _StoreEnvGuard): code under test that writes
# os.environ itself (``run up`` exports its config's env) leaks no state path to later tests.
RESTORED_ENV = (*STATE_ENV, *CLEARED_ENV, ROOT_ENV)

_WRITE_EVENTS = frozenset({"open", "os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.truncate",
                           "os.utime", "os.chmod", "os.symlink", "os.link", "shutil.rmtree"})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
WARN_LIST_MAX = 30

_state: dict = {"root": None, "owned": False, "guarded": (), "snapshot": {}, "violations": [],
                "test": "<collection>", "installed": False}


def isolate() -> Path:
    """Point every ~/.yeto state path at a temporary root; return it. Idempotent: a nested
    in-process run, or a child pytest that inherited the variables, keeps the same root."""
    if _state["root"] is not None:
        return _state["root"]
    inherited = os.environ.get(ROOT_ENV)
    if inherited and Path(inherited).is_dir():
        root = Path(inherited)
    else:
        root = Path(tempfile.mkdtemp(prefix="yeto-test-state-"))
        _state["owned"] = True
        os.environ[ROOT_ENV] = str(root)
    for name, sub in STATE_ENV.items():
        os.environ[name] = str(root / sub)
    for name in CLEARED_ENV:
        os.environ.pop(name, None)
    _state["root"] = root
    return root


def _real_roots() -> tuple[str, ...]:
    homes = {os.environ.get("HOME") or "", pwd.getpwuid(os.getuid()).pw_dir}
    roots = []
    for home in sorted(h for h in homes if h):
        root = os.path.join(os.path.abspath(home), ".yeto")
        if _state["root"] is not None and os.path.abspath(str(_state["root"])).startswith(root + os.sep):
            continue  # an isolated root under it (TMPDIR in HOME) is ours to write
        roots.append(root)
    return tuple(roots)


def _snapshot(roots) -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames + filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                # a directory's mtime moves with every entry made in it: the entries say what
                snap[path] = (0, 0) if name in dirnames else (st.st_mtime_ns, st.st_size)
    return snap


def _guarded(path) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    full = os.path.abspath(text)
    for root in _state["guarded"]:
        if full == root or full.startswith(root + os.sep):
            return full
    return None


def _is_write(event: str, args: tuple) -> bool:
    if event != "open":
        return True
    mode, flags = (args[1], args[2]) if len(args) >= 3 else (args[1] if len(args) > 1 else None, 0)
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _audit(event: str, args: tuple) -> None:
    if event not in _WRITE_EVENTS or not _state["guarded"]:
        return
    targets = (args[0], args[1]) if event in ("os.rename", "os.symlink", "os.link") else (args[0],)
    for target in targets:
        path = _guarded(target)
        if path is None or not _is_write(event, args):
            continue
        if event == "os.mkdir" and os.path.isdir(path):
            continue  # mkdir(exist_ok=True) of what is there writes nothing
        _state["violations"].append((_state["test"], event, path))
        raise PermissionError(f"test isolation: {event} under the real ~/.yeto refused: {path} "
                              f"(tests/yeto_home_guard.py; point the path at tmp_path or its env override)")


def install_guard() -> None:
    """Snapshot the real ~/.yeto and start refusing in-process writes to it (once per process)."""
    if _state["installed"]:
        return
    _state["installed"] = True
    _state["guarded"] = _real_roots()
    _state["snapshot"] = _snapshot(_state["guarded"])
    sys.addaudithook(_audit)


def set_current_test(nodeid: str) -> None:
    _state["test"] = nodeid


def session_report() -> tuple[list[str], list[str]]:
    """(errors, warnings) for the end of the session."""
    errors = [f"{test}: {event} {path}" for test, event, path in _state["violations"]]
    before, after = _state["snapshot"], _snapshot(_state["guarded"])
    new = sorted(p for p in after if p not in before)
    changed = sorted(p for p in after if p in before and after[p] != before[p])
    warnings = [f"new {p}" for p in new] + [f"changed {p}" for p in changed]
    return errors, warnings


def remove_root() -> None:
    """Delete the temporary root this process made (not one it inherited)."""
    root = _state["root"]
    if _state["owned"] and root is not None:
        shutil.rmtree(root, ignore_errors=True)
