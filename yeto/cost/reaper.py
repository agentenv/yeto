"""The resource reaper: a detached process per run that outlives the launcher.

``yeto launch`` starts it in its own session next to the worker. It polls the
worker pid; once the worker is gone - a normal exit, an exception, SIGTERM, or
SIGKILL that skipped every ``finally`` - it releases whatever the run's
resource ledger still lists as open (:func:`.cleanup.cleanup_run`, exact ids
only). Runs launched with ``--keep`` and head-controller runs (the controller
lives on the head VM) are left alone. One reaper per run: a second exits.
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Callable

POLL_S = 15.0


def reap(run: str, worker_pid: int, *, alive: Callable[[int], bool] | None = None,
         cleanup: Callable[[str], bool] | None = None, sleep: Callable[[float], None] = time.sleep,
         poll_s: float = POLL_S, log=print) -> int:
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
        while alive(worker_pid):
            sleep(poll_s)
        meta = runs.load_run(run) or {}
        args = meta.get("args") or {}
        if args.get("keep"):
            log(f"[reaper] {run}: worker {worker_pid} gone; --keep: resources left up")
            return 0
        if meta.get("controller") == "head":
            return 0
        if meta.get("pid") not in (None, worker_pid) and alive(int(meta["pid"])):
            return 0  # the run was relaunched under the same name by a newer worker
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
