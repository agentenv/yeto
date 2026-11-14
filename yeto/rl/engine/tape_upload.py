"""Copy an island's event tape into the run store, so the dashboard can read every
island of a run from one place (``yeto dashboard serve --store``).

An island's tape is a local JSONL file (a sampler on a leased node writes it on
that node's disk). :class:`TapeUploader` appends nothing in the store: every
``interval_s`` it writes the complete lines added since the last upload as one
new object ``tapes/<island>/<incarnation>/<seq>.jsonl`` (seq zero-padded, so a
reader lists them in order). A replacement node is a new incarnation and simply
adds its own prefix; nothing has to be started per node.

Observation only: an upload failure is printed and retried with the next chunk;
it never stops or slows the island (the thread is a daemon and the final flush
at ``stop`` is bounded by one write).
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

TAPES_PREFIX = "tapes"
INTERVAL_ENV = "YETO_RL_TAPE_UPLOAD_S"
DEFAULT_INTERVAL_S = 30.0
MAX_CHUNK_BYTES = 8 << 20
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_part(name: Any) -> str:
    """One key component: no ``/``, no empty string."""
    return _SAFE.sub("_", str(name)).strip("._") or "_"


def chunk_key(island: str, incarnation: str, seq: int) -> str:
    return f"{TAPES_PREFIX}/{safe_part(island)}/{safe_part(incarnation)}/{seq:08d}.jsonl"


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
                 interval_s: float = DEFAULT_INTERVAL_S, log: Callable[[str], None] = print) -> None:
        self.objects = objects
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
                self.objects.write(chunk_key(self.island, self.incarnation, self.seq), data)
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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
        self.flush()


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


def main(argv: Any = None) -> int:
    """``event --store URI --island NAME [--incarnation I]``: one JSON event per stdin line
    into ``tapes/<island>/<incarnation>/<ms>.jsonl`` (the NC pool's ``event_command``)."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m yeto.rl.engine.tape_upload")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("event", help="push the stdin event lines into the run store")
    ev.add_argument("--store", required=True)
    ev.add_argument("--island", required=True)
    ev.add_argument("--incarnation", default="controller")
    args = parser.parse_args(argv)
    data = b"".join(line if line.endswith(b"\n") else line + b"\n"
                    for line in sys.stdin.buffer.read().splitlines() if line.strip())
    if not data:
        return 0
    from yeto.dashboard.store_tapes import open_objects  # noqa: PLC0415

    open_objects(args.store).write(event_key(args.island, args.incarnation, int(time.time() * 1000)), data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
