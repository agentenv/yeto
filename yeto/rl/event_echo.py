"""Event tapes carried over a log stream: ``YETO_RL_EVENT <json>`` lines.

One format with the INFRA driver echo (``driver._ECHO_EVENTS``): the json is
the tape record itself -- ``{"island_id", "time_unix", **event}``, one line,
``sort_keys=True``, ``separators=(",", ":")``. The prefix is searched inside
the line (the launcher prepends ``[cluster]``); records are de-duplicated on
their raw json and kept in stream order, so a reconnecting stream that
replays lines is harmless.

``python -m yeto.rl.event_echo LOG OUT`` rebuilds a tape from a saved log (the
successor of rl-infra-spec's ``extract_tape.py``; it writes the raw tape
lines, i.e. byte-identical to the island's file).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterable
from pathlib import Path

PREFIX = "YETO_RL_EVENT "
# Set (to "1") by the learner for islands whose tape travels over the log
# stream; inherited by the island's Ray workers (entry.connect_island_ray).
ECHO_ENV = "YETO_RL_ECHO_EVENTS"
_WRITE_LOCK = threading.Lock()


def _emit(text: str) -> None:
    """One write per record, then flush.

    A single ``write`` of a line up to PIPE_BUF (4096 bytes on Linux) to a
    pipe is atomic, so records of several processes (learner + Ray workers)
    sharing a stdout pipe do not interleave. A longer record may interleave
    with another process's output; the collector then counts it as a
    discarded malformed line (and a missing record makes the tape incomplete:
    fail closed, never a silently wrong tape).
    """

    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def echo_enabled() -> bool:
    return os.environ.get(ECHO_ENV) == "1"


def enable_echo() -> None:
    os.environ[ECHO_ENV] = "1"


def encode(record: dict) -> str:
    """The tape line of ``record`` (the one encoding every writer uses)."""

    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def append_record(path, record: dict) -> str:
    """Append one record to a tape -- the single low-level tape writer.

    With echo enabled the identical line is printed as ``YETO_RL_EVENT
    <line>`` right after the write, so the rebuilt tape equals the file.
    """

    line = encode(record)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if echo_enabled():
            _emit(PREFIX + line)
    return line


def echo_record(record: dict) -> None:
    """Echo a record that has no tape file in this process (e.g. a Ray worker)."""

    if echo_enabled():
        _emit(PREFIX + encode(record))


FINALIZED_EVENT = "rl_learner_finalized"
MAX_PENDING_CHARS = 1 << 20  # bound on a held, unterminated tape line
REQUIRED_KEYS = ("island_id", "time_unix", "event")
INVALID = object()  # prefix present but not a tape record (e.g. truncated line)


def format_record(record: dict) -> str:
    # Exactly the tape writer's encoding (append_record).
    return PREFIX + encode(record)


def parse_line(line):
    """Raw json of an echoed tape record in ``line``; None without the
    prefix; :data:`INVALID` for a prefixed line that is not a tape record
    (not json, not an object, or missing island_id/time_unix/event)."""

    text = str(line)
    at = text.find(PREFIX)
    if at < 0:
        return None
    raw = text[at + len(PREFIX):].strip()
    try:
        record = json.loads(raw)
    except ValueError:
        return INVALID
    if not isinstance(record, dict) or any(k not in record for k in REQUIRED_KEYS):
        return INVALID
    return raw


class TapeCollector:
    """Appends each new echoed record of a log stream to ``path``."""

    def __init__(self, path, *, fresh: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and (self.path.exists() or self.incomplete_marker.exists()):
            raise FileExistsError(
                f"event tape {self.path} already exists; refusing to mix runs "
                "(pick another --cluster-prefix or remove it)"
            )
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        # A log stream chunk is not a line: Modal log entries may carry several
        # lines, or end in the middle of one. The unterminated tail of a chunk
        # that is not yet a record waits here for the rest.
        self._pending = ""
        self.count = 0
        self.discarded = 0
        self.finalized = False
        self.closed = False

    @property
    def incomplete_marker(self) -> Path:
        return self.path.with_name(self.path.name + ".incomplete")

    def feed(self, chunk) -> None:
        """Feed one log-stream chunk: split it into lines, join a line cut across
        chunks, and record every echoed tape record."""
        with self._lock:
            if self.closed:
                return  # after teardown nothing is written any more
            segments = str(chunk).split("\n")
            complete, tail = segments[:-1], segments[-1]
            if self._pending:
                pending, self._pending = self._pending, ""
                first = complete[0] if complete else tail
                joined = pending + first
                if parse_line(joined) not in (None, INVALID):
                    if complete:
                        complete[0] = joined
                    else:
                        tail = joined
                else:
                    self._record_line(pending)  # the held tail was not a cut record
            for segment in complete:
                self._record_line(segment)
            if tail:
                if parse_line(tail) is INVALID:
                    self._pending = tail[-MAX_PENDING_CHARS:] if len(tail) > MAX_PENDING_CHARS else tail
                else:
                    self._record_line(tail)

    def _record_line(self, line: str) -> None:
        raw = parse_line(line)
        if raw is None:
            return
        if raw is INVALID:
            self.discarded += 1
            return
        if raw in self._seen:
            return
        self._seen.add(raw)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(raw + "\n")
        self.count += 1
        if json.loads(raw).get("event") == FINALIZED_EVENT:
            self.finalized = True

    def recover_from_file(self, tape_file) -> bool:
        """Complete the tape from the island's own tape file (raw tape lines, the
        source the echo mirrors) when the log stream did not deliver the
        records: every record not yet seen is appended in file order; an
        incomplete marker is removed once the finalized record is in. Works on
        a closed collector (it is closed again afterwards). True if finalized.
        Lines that are not tape records count as discarded (fail closed)."""
        text = Path(tape_file).read_text(encoding="utf-8")
        with self._lock:
            was_closed, self.closed = self.closed, False
            try:
                for line in text.splitlines():
                    if line.strip():
                        self._record_line(PREFIX + line)
            finally:
                self.closed = was_closed
            if self.finalized and self.incomplete_marker.exists():
                self.incomplete_marker.unlink()
            return self.finalized

    def close(self) -> bool:
        """Stop writing; mark an unfinalized tape incomplete. True if complete."""

        with self._lock:
            if self._pending:  # a cut record that never got its rest
                pending, self._pending = self._pending, ""
                self._record_line(pending)
            self.closed = True
            if not self.finalized:
                self.incomplete_marker.write_text(
                    f"no {FINALIZED_EVENT} event received; {self.count} record(s), "
                    f"{self.discarded} discarded prefixed line(s)\n",
                    encoding="utf-8",
                )
            return self.finalized


def tape_is_complete(path) -> bool:
    """A rebuilt/island tape is complete when it holds the finalized event and
    carries no ``.incomplete`` marker."""

    path = Path(path)
    if path.with_name(path.name + ".incomplete").exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("event") == FINALIZED_EVENT:
                return True
        except ValueError:
            continue
    return False


def extract(lines: Iterable, out) -> int:
    collector = TapeCollector(out)
    for line in lines:
        collector.feed(line)
    collector.close()
    if collector.discarded:
        print(f"discarded {collector.discarded} malformed prefixed line(s)", file=sys.stderr)
    return collector.count


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0], errors="replace") as handle:
        print(extract(handle, argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
