"""Copy an island's event tape into the run store, so the dashboard can read every
island of a run from one place (``yeto dashboard serve --store``).

An island's tape is a local JSONL file (a sampler on a leased node writes it on
that node's disk). :class:`TapeUploader` appends nothing in the store: every
``interval_s`` it writes the complete lines added since the last upload as one
new object ``tapes/<island>/<incarnation>/<seq>.jsonl`` (seq zero-padded, so a
reader lists them in order). A replacement node is a new incarnation and simply
adds its own prefix; nothing has to be started per node.

Observation only: an upload failure is printed and retried with the next chunk;
it never stops or slows the island (the thread is a daemon). The final drain at
``stop`` uploads every complete line still on the tape, retrying a failed chunk
under the same key (a retried chunk overwrites, never duplicates), and is bounded
in wall time (:data:`FINAL_FLUSH_ENV`): one store write may block for minutes
(botocore: a 60 s read timeout times 9 attempts). A SIGINT / SIGTERM that arrives
during that drain on the main thread (a cluster teardown racing the last upload)
is held and delivered again once the drain ends, so the last chunk is not cut
off mid-request.

The syncer's tape goes the same way (:func:`syncer_uploaders`): ``tapes/syncer/`` and
``tapes/syncer-shard<i>/`` for its shards, so ``--store`` alone shows the syncer even when
it runs in a container or on a VM the dashboard cannot reach. The uploader only reads the
file the syncer appends to, on its own thread (or process: ``follow``); it is never on the
syncer's merge path. What it has not uploaded yet stays on the local tape, read back in
bounded chunks (:data:`MAX_CHUNK_BYTES`), so nothing queues up in memory and nothing is
dropped; a failed write is counted (``failures``) and logged with the error type and the
key only, never the store credentials.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

TAPES_PREFIX = "tapes"
INTERVAL_ENV = "YETO_RL_TAPE_UPLOAD_S"
DEFAULT_INTERVAL_S = 30.0
FINAL_FLUSH_ENV = "YETO_RL_TAPE_FINAL_FLUSH_S"
DEFAULT_FINAL_FLUSH_S = 30.0
FINAL_RETRY_WAIT_S = 1.0
HELD_SIGNALS = ("SIGINT", "SIGTERM")
MAX_CHUNK_BYTES = 8 << 20
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_part(name: Any) -> str:
    """One key component: no ``/``, no empty string."""
    return _SAFE.sub("_", str(name)).strip("._") or "_"


def chunk_key(island: str, incarnation: str, seq: int, prefix: str = TAPES_PREFIX) -> str:
    """``<prefix>/<island>/<incarnation>/<seq>.jsonl`` (``prefix``: ``tapes`` for the event
    tapes, ``fleet/log`` for the launcher's ``fleet.jsonl``, ``dashboard.fleet_store``)."""
    return f"{prefix}/{safe_part(island)}/{safe_part(incarnation)}/{seq:08d}.jsonl"


def final_flush_s(environ: Any = None) -> float:
    """Wall-time budget of the final drain at :meth:`TapeUploader.stop`
    (:data:`FINAL_FLUSH_ENV`, else :data:`DEFAULT_FINAL_FLUSH_S`; 0 = no drain)."""
    env = os.environ if environ is None else environ
    raw = env.get(FINAL_FLUSH_ENV)
    value = float(raw) if raw not in (None, "") else DEFAULT_FINAL_FLUSH_S
    return max(0.0, value)


@contextlib.contextmanager
def held_signals(names: Any = HELD_SIGNALS, *, log: Callable[[str], None] = print, signal_mod: Any = signal):
    """Hold ``names`` (SIGINT / SIGTERM) for the block and deliver each one that
    arrived again after it, with the previous handlers back in place. Only the main
    thread can install a handler: elsewhere this is a no-op."""
    if threading.current_thread() is not threading.main_thread():
        yield []
        return
    arrived: list[int] = []
    previous: dict[int, Any] = {}

    def hold(num, _frame):
        arrived.append(num)
        log(f"[tape-upload] signal {num} held until the final tape upload ends")

    for name in names:
        num = getattr(signal_mod, name, None)
        if num is None:
            continue
        try:
            previous[num] = signal_mod.signal(num, hold)
        except (ValueError, OSError):  # not settable here
            continue
    try:
        yield arrived
    finally:
        for num, handler in previous.items():
            signal_mod.signal(num, handler)
        for num in dict.fromkeys(arrived):  # once each, in arrival order
            signal_mod.raise_signal(num)


def upload_interval_s(spec_value: Any = None, environ: Any = None) -> float | None:
    """Seconds between uploads: the env wins, then the spec value, else the default;
    0 (or a negative value) turns the upload off."""
    env = os.environ if environ is None else environ
    raw = env.get(INTERVAL_ENV)
    value = float(raw) if raw not in (None, "") else (
        float(spec_value) if spec_value is not None else DEFAULT_INTERVAL_S)
    return value if value > 0 else None


class TapeUploader:
    def __init__(self, objects: Any, tape: str | os.PathLike[str], *, island: str, incarnation: str,
                 interval_s: float = DEFAULT_INTERVAL_S, log: Callable[[str], None] = print,
                 prefix: str = TAPES_PREFIX) -> None:
        self.objects = objects
        self.prefix = prefix
        self.tape = Path(tape)
        self.island = str(island)
        self.incarnation = str(incarnation)
        self.interval_s = float(interval_s)
        self.log = log
        self.offset = 0
        self.seq = 0
        self.failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def flush(self) -> int:
        """Upload the complete lines written since the last upload; returns their bytes."""
        with self._lock:
            try:
                size = self.tape.stat().st_size
            except OSError:
                return 0
            if size <= self.offset:
                return 0
            try:  # the tape can vanish / rotate between stat and open
                with open(self.tape, "rb") as fh:
                    fh.seek(self.offset)
                    data = fh.read(min(size - self.offset, MAX_CHUNK_BYTES))
            except OSError as error:
                self.failures += 1
                self.log(f"[tape-upload] {self.tape.name}: read {type(error).__name__}: {error}"[:300])
                return 0
            end = data.rfind(b"\n")
            if end < 0:
                return 0  # a torn last line waits for its newline
            data = data[: end + 1]
            try:
                self.objects.write(chunk_key(self.island, self.incarnation, self.seq, self.prefix), data)
            except Exception as error:  # noqa: BLE001 - observation must never stop the island
                self.failures += 1
                self.log(f"[tape-upload] {self.tape.name}: {type(error).__name__}: {error}"[:300])
                return 0
            self.offset += len(data)
            self.seq += 1
            return len(data)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            while self.flush() >= MAX_CHUNK_BYTES - 1:  # catch up a large backlog
                pass

    def start(self) -> "TapeUploader":
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tape-upload")
        self._thread.start()
        return self

    def pending_bytes(self) -> int:
        """Bytes on the local tape not uploaded yet (0 when the tape is unreadable)."""
        try:
            return max(0, self.tape.stat().st_size - self.offset)
        except OSError:
            return 0

    def _drain(self, deadline: float, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        """Upload every complete line still on the tape, chunk by chunk; a failed
        write is retried (same key) until ``deadline``."""
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, deadline - clock()))
        while clock() < deadline:
            failures = self.failures
            if self.flush() > 0:
                continue
            if self.failures == failures:
                return  # caught up (or only a torn last line is left)
            sleep(min(FINAL_RETRY_WAIT_S, max(0.0, deadline - clock())))

    def stop(self, timeout_s: float | None = None, *, clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep) -> bool:
        """Stop the thread and drain the tape into the store within ``timeout_s``
        (default :func:`final_flush_s`). The drain runs on its own daemon thread, so a
        write that hangs past the budget is left behind, not waited for; SIGINT /
        SIGTERM arriving meanwhile are held (:func:`held_signals`). Returns True when
        the store has every complete line of the tape."""
        self._stop.set()
        budget = final_flush_s() if timeout_s is None else max(0.0, float(timeout_s))
        deadline = clock() + budget
        done = threading.Event()

        def run() -> None:
            try:
                self._drain(deadline, clock, sleep)
            finally:
                done.set()

        drain = threading.Thread(target=run, daemon=True, name="tape-upload-final")
        with held_signals(log=self.log):
            drain.start()
            done.wait(budget + FINAL_RETRY_WAIT_S)
        left = self.pending_bytes()
        complete = done.is_set() and self._complete(left)
        if not complete:
            self.log(f"[tape-upload] {self.tape.name}: final upload incomplete after {budget:.0f} s: "
                     f"{left} byte(s) not in the store (the local tape keeps them)")
        return complete

    def _complete(self, left: int) -> bool:
        """Nothing but a torn last line (no newline yet) is left on the tape."""
        if left == 0:
            return True
        try:
            with open(self.tape, "rb") as fh:
                fh.seek(self.offset)
                return b"\n" not in fh.read(min(left, MAX_CHUNK_BYTES))
        except OSError:
            return False


def store_objects(store: Any) -> Any:
    """The object API of a checkpoint store: its ``objects`` (S3 / OSS) or the
    store directory itself (local and shared directories, tests)."""
    objects = getattr(store, "objects", None)
    if objects is not None:
        return objects
    from .object_store import DirectoryObjectStore  # noqa: PLC0415

    return DirectoryObjectStore(store.root)


def event_key(island: str, incarnation: str, when_ms: int) -> str:
    """Key of one pushed event batch (``seq`` = epoch milliseconds: 13 digits, ordered)."""
    return f"{TAPES_PREFIX}/{safe_part(island)}/{safe_part(incarnation)}/{int(when_ms):013d}.jsonl"


def trainer_uploader(event_tape: Any, *, learner_id: Any = 0, environ: Any = None,
                     log: Callable[[str], None] = print) -> "TapeUploader | None":
    """The trainer island's uploader (``tapes/trainer/<incarnation>/``): on when the island
    uses the s3 store (``YETO_RL_STORE_S3_URI``) and :data:`INTERVAL_ENV` is not 0. Modal
    containers have no inbound address: the dashboard reads the trainer from the store."""
    env = os.environ if environ is None else environ
    uri = env.get("YETO_RL_STORE_S3_URI")
    interval = upload_interval_s(None, env)
    if not uri or not event_tape or interval is None:
        return None
    import time  # noqa: PLC0415

    from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

    import socket  # noqa: PLC0415

    inc = f"l{learner_id}-{socket.gethostname()}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    up = TapeUploader(head_objects(uri, environ=env), Path(os.path.expanduser(str(event_tape))),
                      island="trainer", incarnation=inc, interval_s=interval, log=log)
    return up.start()


SYNCER_ISLAND = "syncer"


def syncer_island(index: int) -> str:
    """Store name of syncer shard ``index``: ``syncer`` for shard 0, ``syncer-shard<i>`` else."""
    return SYNCER_ISLAND if index == 0 else f"{SYNCER_ISLAND}-shard{index}"


def shard_tape(tape: str | os.PathLike[str], index: int) -> Path:
    """Shard ``index``'s tape next to the syncer's (the rule of ``launch.syncer._shard_path``:
    ``<stem>.shard<i><suffix>`` for i > 0)."""
    path = Path(os.path.expanduser(str(tape)))
    return path if index == 0 else path.with_name(f"{path.stem}.shard{index}{path.suffix}")


def syncer_uploaders(objects: Any, tape: str | os.PathLike[str], *, shards: int = 1, incarnation: str,
                     interval_s: float = DEFAULT_INTERVAL_S, from_end: bool = True,
                     log: Callable[[str], None] = print) -> list[TapeUploader]:
    """Started uploaders of the syncer's tape and of each shard's (a tape that does not exist
    yet is picked up once it does). ``from_end``: start at the tape's current end, because a
    reused host keeps an earlier job's tape, which that job uploaded."""
    out = []
    for index in range(max(1, int(shards))):
        path = shard_tape(tape, index)
        up = TapeUploader(objects, path, island=syncer_island(index), incarnation=incarnation,
                          interval_s=interval_s, log=log)
        if from_end:
            with contextlib.suppress(OSError):
                up.offset = path.stat().st_size
        out.append(up.start())
    return out


def syncer_incarnation() -> str:
    """This syncer process's store incarnation: host and UTC start time (a restarted syncer
    adds its own prefix next to the previous one)."""
    import socket  # noqa: PLC0415

    return f"{socket.gethostname()}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _follow(args: Any) -> int:
    """Upload the syncer tape(s) until SIGTERM / SIGINT, then drain (the sidecar of a syncer
    the launcher does not run itself, e.g. in a container)."""
    from yeto.dashboard.store_tapes import open_objects  # noqa: PLC0415

    interval = upload_interval_s(args.interval)
    if interval is None:
        print("[tape-upload] upload interval 0: nothing to do", flush=True)
        return 0
    done = threading.Event()
    for name in HELD_SIGNALS:
        signal.signal(getattr(signal, name), lambda *_: done.set())
    ups = syncer_uploaders(open_objects(args.store), args.tape, shards=args.shards,
                           incarnation=args.incarnation or syncer_incarnation(), interval_s=interval,
                           from_end=args.from_end)
    print(f"[tape-upload] following {args.tape} ({args.shards} shard(s)) every {interval:.0f} s", flush=True)
    done.wait()
    return 0 if all([up.stop() for up in ups]) else 1


def main(argv: Any = None) -> int:
    """``event --store URI --island NAME [--incarnation I]``: one JSON event per stdin line
    into ``tapes/<island>/<incarnation>/<ms>.jsonl`` (the NC pool's ``event_command``).
    ``follow --tape PATH --store URI [--shards N]``: the syncer tape uploader as a process."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m yeto.rl.engine.tape_upload")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("event", help="push the stdin event lines into the run store")
    ev.add_argument("--store", required=True)
    ev.add_argument("--island", required=True)
    ev.add_argument("--incarnation", default="controller")
    fo = sub.add_parser("follow", help="upload the syncer tape (and its shards' tapes) until stopped")
    fo.add_argument("--tape", required=True, help="the syncer's --event-tape")
    fo.add_argument("--store", required=True, help="the run store (s3://, oss:// or a directory)")
    fo.add_argument("--shards", type=int, default=1, help="--syncer-shards of the run")
    fo.add_argument("--incarnation", default=None, help="default: host and UTC start time")
    fo.add_argument("--interval", type=float, default=None, help=f"seconds between uploads ({INTERVAL_ENV} wins)")
    fo.add_argument("--from-end", action="store_true", help="skip what the tape holds already")
    args = parser.parse_args(argv)
    if args.cmd == "follow":
        return _follow(args)
    data = b"".join(line if line.endswith(b"\n") else line + b"\n"
                    for line in sys.stdin.buffer.read().splitlines() if line.strip())
    if not data:
        return 0
    from yeto.dashboard.store_tapes import open_objects  # noqa: PLC0415

    open_objects(args.store).write(event_key(args.island, args.incarnation, int(time.time() * 1000)), data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
