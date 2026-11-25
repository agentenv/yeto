"""The resource reaper: a detached process per run that outlives the launcher.

``yeto launch`` starts it in its own session next to the worker. It polls the
worker pid; once the worker is gone - a normal exit, an exception, SIGTERM, or
SIGKILL that skipped every ``finally`` - it releases whatever the run's
resource ledger still lists as open (:func:`.cleanup.cleanup_run`, exact ids
only). Runs launched with ``--keep`` and head-controller runs (the controller
lives on the head VM) are left alone. One reaper per run: a second exits.

The island launch of a fleet launched island by island (``--rl-fleet-role island``)
is supervised by its fleet's trainer worker, which relaunches it under the same run
name; that launch adopts the island still running (its records in the run store).
Its reaper waits :data:`ADOPT_GRACE_S` for that worker first and then watches it (the
new worker's own reaper found this one's lock); only an island no launch took back
within the grace is released.
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Callable

POLL_S = 15.0
ADOPT_GRACE_S = 600.0  # a fleet island launch: how long its relaunch has to take the island back


def _newer_worker(run: str, worker_pid: int, alive: Callable[[int], bool]) -> int | None:
    from .. import runs

    pid = (runs.load_run(run) or {}).get("pid")
    return int(pid) if pid not in (None, worker_pid) and alive(int(pid)) else None


def reap(run: str, worker_pid: int, *, alive: Callable[[int], bool] | None = None,
         cleanup: Callable[[str], bool] | None = None, sleep: Callable[[float], None] = time.sleep,
         poll_s: float = POLL_S, log=print, adopt_grace_s: float = ADOPT_GRACE_S) -> int:
    from .. import runs
    from .cleanup import cleanup_run
    from .ledger import ResourceLedger

    alive = alive or runs.is_alive
    cleanup = cleanup or cleanup_run
    d = runs.run_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    lock = open(d / ".reaper.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0  # another reaper watches this run
    try:
        while True:
            while alive(worker_pid):
                sleep(poll_s)
            meta = runs.load_run(run) or {}
            args = meta.get("args") or {}
            if args.get("keep"):
                log(f"[reaper] {run}: worker {worker_pid} gone; --keep: resources left up")
                return 0
            if meta.get("controller") == "head":
                return 0
            # a fleet island launch whose worker died without a result (killed): its relaunch adopts it
            fleet_island = args.get("rl_fleet_role") == "island" and meta.get("state") in (runs.PENDING, runs.RUNNING)
            newer = _newer_worker(run, worker_pid, alive)
            if newer is not None and not fleet_island:
                return 0  # the run was relaunched under the same name by a newer worker
            if newer is None and fleet_island:
                log(f"[reaper] {run}: worker {worker_pid} gone; a fleet island launch: its trainer worker "
                    f"relaunches it and the new launch adopts the running island; waiting {adopt_grace_s:.0f}s "
                    "before anything is released")
                for _ in range(max(1, int(-(-adopt_grace_s // poll_s)))):
                    sleep(poll_s)
                    newer = _newer_worker(run, worker_pid, alive)
                    if newer is not None:
                        break
            if newer is not None:  # a fleet island launch taken back: watch its new worker
                log(f"[reaper] {run}: worker {newer} took the island back; watching it")
                worker_pid = newer
                continue
            break
        led = ResourceLedger.for_run(run)
        open_res = [r for r in led.open_resources()
                    if not (r.kind == "modal_app" and r.attrs.get("shared"))]
        if not open_res:
            return 0
        log(f"[reaper] {run}: worker {worker_pid} gone with {len(open_res)} open resource(s) in "
            f"{led.path}; releasing them")
        ok = cleanup(run)
        if not ok:
            log(f"[reaper] {run}: not everything confirmed released; run `yeto cleanup --run {run}`")
            try:
                runs.update_run(run, state=runs.TEARDOWN_INCOMPLETE)
            except OSError:
                pass
        return 0 if ok else 1
    finally:
        try:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        except OSError:
            pass


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(reap(sys.argv[1], int(sys.argv[2])))
