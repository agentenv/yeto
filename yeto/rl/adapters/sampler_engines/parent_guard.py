"""Run an engine server command and end its process group when the sampler dies.

``popen_spawn`` (:mod:`.server`) starts every engine server in its own session, so a
reclaim's SIGINT to the container and a stop of the sampler's process group do not
reach it. The cost: a sampler killed with SIGKILL (a crash, an OOM kill, a process
group KILL after the stop grace) left the server and its workers running on the GPUs,
and the next sampler start failed on memory or two engines shared the cards.

This guard is the session leader in between: ``python -I parent_guard.py --parent
<sampler pid> -- <server argv>``. It runs the server as its child, forwards SIGTERM to
it (the engine's ``stop()``), exits with the server's exit status (a server ended by a
signal ends the guard by the same signal), and checks once per ``--poll-s`` that its
parent is still the sampler. Once the parent is gone (the guard was re-parented), it
sends SIGTERM to its whole process group (server, engine core, TP workers), waits up
to ``--grace-s``, then SIGKILLs the group, itself included.

Polling the parent works whatever thread of the sampler started the server (Linux's
``PR_SET_PDEATHSIG`` fires when the creating *thread* exits, which would end a server
a short-lived reload thread started) and on macOS. Standard library only: it runs as
a script by path, with ``-I``, under whatever ``PYTHONPATH`` the engine needs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _parse(argv: list[str]) -> tuple[int, float, float, list[str]]:
    if "--" not in argv:
        raise SystemExit("usage: parent_guard.py --parent PID [--grace-s S] [--poll-s S] -- CMD ...")
    i = argv.index("--")
    opts, cmd = argv[:i], argv[i + 1:]
    values = {"--parent": None, "--grace-s": "20", "--poll-s": "1"}
    it = iter(opts)
    for key in it:
        if key not in values:
            raise SystemExit(f"parent_guard.py: unknown option {key}")
        values[key] = next(it, None)
    if values["--parent"] is None or not cmd:
        raise SystemExit("parent_guard.py: --parent PID and a command are required")
    return int(values["--parent"]), float(values["--grace-s"]), float(values["--poll-s"]), cmd


def _end_group(child: subprocess.Popen, grace_s: float) -> None:
    """The sampler is gone: TERM the group, KILL what is left after ``grace_s``. The
    group is only signalled when this process leads it (``start_new_session``);
    otherwise only the child is ended (never the caller's group)."""
    leader = os.getpgrp() == os.getpid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        if leader:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        else:
            child.terminate()
    except OSError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and child.poll() is None:
        time.sleep(0.2)
    if leader:
        os.killpg(os.getpgrp(), signal.SIGKILL)  # workers outlive their parent's exit; this ends us too
    elif child.poll() is None:
        child.kill()


def main(argv: list[str] | None = None) -> int:
    parent, grace_s, poll_s, cmd = _parse(list(sys.argv[1:] if argv is None else argv))
    child = subprocess.Popen(cmd)  # noqa: S603 - the engine's own argv

    def forward(signum, _frame):
        try:
            child.send_signal(signum)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, forward)
    while True:
        try:
            code = child.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            pass
        if os.getppid() != parent:
            print(f"[parent-guard] sampler {parent} is gone: ending the engine server {child.pid} and its "
                  "process group", file=sys.stderr, flush=True)
            _end_group(child, grace_s)
            code = child.wait()
            break
    if code < 0:  # the server died of a signal: so does the guard (the caller reads -N)
        signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
        time.sleep(1)
    return code


if __name__ == "__main__":
    sys.exit(main())
