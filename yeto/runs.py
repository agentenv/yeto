"""Local run registry for detached yeto runs. Stdlib only.

Each run lives in ``~/.yeto/runs/<name>/`` (override with $YETO_RUNS_DIR):

    meta.json      -- launch args, worker pid, cluster names, state,
                      started/finished timestamps, exit code
    launcher.log   -- combined stdout/stderr of the background worker

A run's name is its ``--cluster-prefix``. Writers go through
``update_run`` which does a locked read-modify-write, and every meta write
is an atomic replace, so the parent CLI and the worker can update
different fields concurrently.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

RUNS_DIR = Path(os.environ.get("YETO_RUNS_DIR") or "~/.yeto/runs").expanduser()

# Run states recorded in meta.json. A live worker pid overrides the
# recorded state for display purposes (see the status command).
PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
DOWN = "DOWN"


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def log_path(name: str) -> Path:
    return run_dir(name) / "launcher.log"


def _meta_path(name: str) -> Path:
    return run_dir(name) / "meta.json"


def is_alive(pid) -> bool:
    """True if `pid` refers to a live process (signal-0 probe)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (OverflowError, ValueError, OSError):
        return False
    return True


def _write_meta(name: str, meta: dict) -> None:
    """Atomically replace meta.json (write temp file in-dir, then rename)."""
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".meta-", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp, _meta_path(name))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextmanager
def _locked(name: str):
    """Exclusive advisory lock for read-modify-write of one run's meta."""
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def create_run(name: str, args_dict: dict) -> dict:
    """Create (or reset, for a reused name) the registry entry for a run.

    Truncates launcher.log so a reused name starts with a fresh stream.
    """
    meta = {
        "name": name,
        "args": dict(args_dict),
        "pid": None,
        "clusters": [],
        "state": PENDING,
        "started_at": time.time(),
        "finished_at": None,
        "exit_code": None,
    }
    with _locked(name):
        _write_meta(name, meta)
        log_path(name).write_text("")
    return meta


def load_run(name: str) -> dict | None:
    """Return the run's meta dict, or None if unknown/corrupt."""
    try:
        return json.loads(_meta_path(name).read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def list_runs() -> list[dict]:
    """All known runs, most recently started first."""
    if not RUNS_DIR.is_dir():
        return []
    metas = []
    for child in RUNS_DIR.iterdir():
        meta = load_run(child.name)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda m: m.get("started_at") or 0, reverse=True)
    return metas


def update_run(name: str, **fields) -> dict:
    """Locked read-modify-write of selected meta fields."""
    with _locked(name):
        meta = load_run(name) or {"name": name}
        meta.update(fields)
        _write_meta(name, meta)
    return meta
