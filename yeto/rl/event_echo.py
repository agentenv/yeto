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
import sys
import threading
from collections.abc import Iterable
from pathlib import Path

PREFIX = "YETO_RL_EVENT "


FINALIZED_EVENT = "rl_learner_finalized"
REQUIRED_KEYS = ("island_id", "time_unix", "event")
INVALID = object()  # prefix present but not a tape record (e.g. truncated line)


def format_record(record: dict) -> str:
    # Exactly the tape writer's encoding (yeto.rl.miles._append_rl_event).
    return PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"))


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
        self.count = 0
        self.discarded = 0
        self.finalized = False
        self.closed = False

    @property
    def incomplete_marker(self) -> Path:
        return self.path.with_name(self.path.name + ".incomplete")

    def feed(self, line) -> None:
        raw = parse_line(line)
        if raw is None:
            return
        with self._lock:
            if self.closed:
                return  # after teardown nothing is written any more
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

    def close(self) -> bool:
        """Stop writing; mark an unfinalized tape incomplete. True if complete."""

        with self._lock:
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
