"""Mirror a remote learner tape over SSH for the live dashboard.

``ssh <target> tail -c +<offset+1> -F <remote>`` streams the remote tape
from exactly the byte the local mirror ends at, so a reconnect neither
loses nor repeats records (the local mirror only ever holds remote bytes,
hence its size *is* the resume offset). Connection loss is backed off
exponentially and recorded as ``dashboard_source_lost`` /
``dashboard_source_restored`` events in a sidecar ``<mirror>.source.jsonl``
next to the mirror (never inside it, which would break the offset).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

_TAIL_CHUNK = 64 * 1024  # bytes read per step while looking for the last complete line


def ssh_tail_command(target: str, remote: str, offset: int) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", target,
            f"tail -c +{offset + 1} -F {shlex.quote(remote)}"]


class SshTapeMirror:
    def __init__(self, target: str, remote: str, local: str | Path, *, island: str | None = None,
                 command: Callable[[int], list[str]] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
                 backoff_initial: float = 1.0, backoff_max: float = 30.0):
        self.target = target
        self.remote = remote
        self.local = Path(local)
        self.sidecar = self.local.with_name(self.local.name + ".source.jsonl")
        self.island = island
        self.command = command or (lambda off: ssh_tail_command(target, remote, off))
        self.sleep = sleep
        self.clock = clock
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.lost = False
        self.connections = 0

    def offset(self) -> int:
        """Bytes mirrored so far, trimmed back to the last complete line."""
        if not self.local.exists():
            return 0
        # scan backwards in chunks: a multi-day mirror is far too large to read whole
        with open(self.local, "r+b") as fh:
            size = fh.seek(0, 2)
            pos, end = size, 0
            while pos > 0:
                step = min(_TAIL_CHUNK, pos)
                pos -= step
                fh.seek(pos)
                nl = fh.read(step).rfind(b"\n")
                if nl >= 0:
                    end = pos + nl + 1
                    break
            if end != size:  # torn tail from a killed stream: drop it, re-fetch
                fh.truncate(end)
        return end

    def _event(self, event: str, **fields) -> None:
        rec = {"event": event, "time_unix": self.clock(), "source": f"{self.target}:{self.remote}", **fields}
        if self.island is not None:
            rec["island_id"] = self.island
        with open(self.sidecar, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run_once(self) -> int:
        """One connection: stream until it ends. Returns bytes mirrored."""
        self.local.parent.mkdir(parents=True, exist_ok=True)
        off = self.offset()
        self.connections += 1
        got = 0
        proc = subprocess.Popen(self.command(off), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Drain stderr while stdout streams: a chatty ssh would otherwise fill
        # the stderr pipe and block, stalling the stdout stream for good.
        tail = bytearray()

        def drain() -> None:
            assert proc.stderr is not None
            for chunk in iter(lambda: proc.stderr.read(65536), b""):
                tail.extend(chunk)
                del tail[:-4096]

        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()
        try:
            with open(self.local, "ab") as out:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if not line.endswith(b"\n"):
                        break  # torn last line of a dying stream; refetched next time
                    out.write(line)
                    out.flush()
                    got += len(line)
                    if self.lost:
                        self.lost = False
                        self._event("dashboard_source_restored", offset=off)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            drainer.join(5)
        err = bytes(tail).decode("utf-8", "replace").strip()
        if not self.lost:
            from yeto.rl.engine.init_from_release import redact_text  # noqa: PLC0415
            self.lost = True
            self._event("dashboard_source_lost", error=redact_text(err or f"stream ended rc={proc.returncode}")[:300],
                        offset=off + got)
        return got

    def run_forever(self, stop: threading.Event | None = None, max_connections: int | None = None) -> None:
        delay = self.backoff_initial
        while not (stop is not None and stop.is_set()):
            got = self.run_once()
            if max_connections is not None and self.connections >= max_connections:
                return
            delay = self.backoff_initial if got else min(self.backoff_max, delay * 2)
            self.sleep(delay)
