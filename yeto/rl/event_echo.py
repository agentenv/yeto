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


def format_record(record: dict) -> str:
    return PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def parse_line(line) -> str | None:
    """The raw json of an echoed record in ``line``, or None."""

    text = str(line)
    at = text.find(PREFIX)
    if at < 0:
        return None
    raw = text[at + len(PREFIX):].strip()
    try:
        json.loads(raw)
    except ValueError:
        return None
    return raw


class TapeCollector:
    """Appends each new echoed record of a log stream to ``path``."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self.count = 0

    def feed(self, line) -> None:
        raw = parse_line(line)
        if raw is None:
            return
        with self._lock:
            if raw in self._seen:
                return
            self._seen.add(raw)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(raw + "\n")
            self.count += 1


def extract(lines: Iterable, out) -> int:
    collector = TapeCollector(out)
    for line in lines:
        collector.feed(line)
    return collector.count


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0], errors="replace") as handle:
        print(extract(handle, argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
