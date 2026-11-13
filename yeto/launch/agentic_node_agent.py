"""Node-side loops of the agentic node setup (:mod:`yeto.launch.agentic_setup`).

Standard library only: these run in a helper container of the sampler image on
the sampler node, with the launcher's yeto tree mounted at ``/yeto``.

* ``tunnel``: keeps one ``ssh -N -R`` reverse forward up (the records proxy of
  the sampler, reachable from the sandboxes on the sandbox host). Every connect
  and every drop is a JSON line on the events file (``rl_agentic_tunnel``,
  ``state`` ``up`` / ``down``); a drop is followed by a reconnect after a
  backoff that doubles while connections keep failing fast and resets after a
  connection that stayed up.

Image pre-pulls and the pi build of the sandbox images belong to the sandbox
host side (``agentic.envs``), not to this node setup.

Usage: ``python -m yeto.launch.agentic_node_agent tunnel --events F -- ssh ...``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

TUNNEL_EVENT = "rl_agentic_tunnel"


def append_event(path: str | Path, record: dict[str, Any], *, clock: Callable[[], float] = time.time) -> None:
    """Best effort: the events file is bookkeeping. A failed write (full disk,
    unmounted volume) goes to stderr instead of ending the loop, which would
    drop a working forward and restart the container over and over."""
    line = json.dumps({"ts": round(clock(), 3), **record}, sort_keys=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"agentic_node_agent: events file {path} not written ({exc}): {line}", file=sys.stderr, flush=True)


# -- tunnel --------------------------------------------------------------------

def tunnel_loop(ssh_argv: Sequence[str], events: str | Path, *, settle_s: float = 5.0, retry_s: float = 2.0,
                max_backoff_s: float = 60.0, stable_s: float = 60.0, max_attempts: int | None = None,
                popen: Callable[..., Any] = subprocess.Popen, sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic) -> int:
    """Run ``ssh_argv`` (an ``ssh -N -o ExitOnForwardFailure=yes -R ...``) forever.

    The forward counts as up once ssh is still running ``settle_s`` after the
    start (``ExitOnForwardFailure`` makes a refused forward exit at once). On
    exit: a ``down`` event with the exit code and the tail of ssh's stderr, a
    backoff sleep, the next attempt. ``max_attempts`` bounds the loop (tests)."""
    attempt, backoff = 0, retry_s
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        started = clock()
        proc = popen(list(ssh_argv), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        up = False
        try:
            proc.wait(timeout=settle_s)
        except subprocess.TimeoutExpired:
            up = True
            append_event(events, {"event": TUNNEL_EVENT, "state": "up", "attempt": attempt})
            proc.wait()
        err = ""
        try:
            err = (proc.stderr.read() if proc.stderr else "") or ""
        except (OSError, ValueError):
            pass
        lived = clock() - started
        append_event(events, {"event": TUNNEL_EVENT, "state": "down", "attempt": attempt, "was_up": up,
                              "exit": proc.returncode, "lived_s": round(lived, 1),
                              "stderr": err.strip()[-300:]})
        backoff = retry_s if lived >= stable_s else min(max_backoff_s, backoff * 2 if attempt > 1 else retry_s)
        sleep(backoff)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m yeto.launch.agentic_node_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tunnel")
    t.add_argument("--events", required=True)
    t.add_argument("--settle-s", type=float, default=5.0)
    t.add_argument("--retry-s", type=float, default=2.0)
    t.add_argument("--max-backoff-s", type=float, default=60.0)
    t.add_argument("ssh", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    argv_ssh = args.ssh[1:] if args.ssh[:1] == ["--"] else args.ssh
    if not argv_ssh:
        p.error("tunnel needs the ssh command after --")
    return tunnel_loop(argv_ssh, args.events, settle_s=args.settle_s, retry_s=args.retry_s,
                       max_backoff_s=args.max_backoff_s)


if __name__ == "__main__":
    sys.exit(main())
