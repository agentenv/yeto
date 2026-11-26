"""Island stragglers of an island pool: a running island none of whose pushes was merged in
N merge cycles (``island_straggler``).

The syncer already keeps such an island from holding the others: a pool round whose quorum
deadline passes merges the pushes that arrived, the silent island is recorded absent and
excused from the next rounds, and once it delivers again it is caught up and merged again.
What nobody said so far is that an island has stopped contributing: it heartbeats, its
launch is up, and the fleet controller sees nothing wrong. This module watches the syncer's
merges and tells the controller, which writes the event and decides what to do (wait, the
default, or a replacement the island's provider offers).

* **What is counted.** The syncer's tape (its ``outer_step`` records; ``pool_join`` for the
  start of an island). A merge cycle is one outer step of every fragment: with P fragments
  (the distinct ``fragment`` ids merged so far, 1 for a non-fragment pool) an island is a
  straggler once ``N x P`` merges in a row went by without a push of it in their weights.
  A push the syncer refused, or one still in flight, is not merged and does not count.
* **When counting starts.** At the island's first ``pool_join`` after it was (re)launched:
  the time a launch takes to come up and join is not counted. A later re-JOIN of the same
  incarnation (a lease that expired) does not restart the count. A merged push of the island
  resets it.
* **Who is watched.** The islands the controller holds running (not draining, not
  recovering): it passes their learner ids with a token of the incarnation; a new token
  is a relaunch and starts the count again.
* **Signals.** One :class:`Straggler` when an island crosses the threshold, and one
  ``cleared`` signal when a push of it is merged again (or when it was relaunched while
  flagged). Nothing in between: the controller writes one event per stretch.

Sources (:func:`tape_source`): the syncer's tape file on this host (a head-mode syncer), or
its copy in the run store (``tapes/syncer/<incarnation>/<seq>.jsonl``, uploaded every 30 s
by ``yeto.rl.engine.tape_upload``). A source that does not read is skipped for that poll;
nothing here ever stops or slows the fleet.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVENT = "island_straggler"
CLEARED_EVENT = "island_straggler_cleared"
DEFAULT_CYCLES = 3
SYNCER_TAPES = "tapes/syncer/"
STORE_POLL_S = 30.0  # the syncer's uploads are 30 s apart (tape_upload.DEFAULT_INTERVAL_S)
MAX_READ_BYTES = 8 << 20  # one poll reads at most this much of a local tape
QUIET_POLLS = 40  # polls (20 min at 30 s) without a syncer record before that is said, once

WAIT = "wait"  # the syncer merges without it; it is caught up once it delivers again
REPLACE = "replace"  # the island's provider replaces it (its own policy decides)
ACTIONS = (WAIT, REPLACE)


def straggler_cycles(args: Any) -> int:
    """``--rl-island-straggler-cycles`` (N >= 0; 0 turns the watch off), default
    :data:`DEFAULT_CYCLES`."""
    value = getattr(args, "rl_island_straggler_cycles", None)
    if value is None:
        return DEFAULT_CYCLES
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"--rl-island-straggler-cycles must be a whole number >= 0 (got {value!r})")
    return int(value)


@dataclass(frozen=True)
class Straggler:
    """One signal about island ``island`` (its learner id)."""

    island: int
    cleared: bool  # False: it became a straggler; True: a push of it is merged again (or relaunched)
    missed_cycles: int  # whole merge cycles since its last merged push (or its join)
    merges: int  # merges since then
    fragments: int  # P
    cycles: int  # N, the threshold
    newest_step: int | None  # the newest merged step the syncer recorded
    last_merged_step: int | None  # the newest step a push of the island was merged into
    absent: bool  # the newest merge listed it absent (expected, not answered)
    reason: str = ""

    def fields(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("island")
        out.pop("cleared")
        return out


@dataclass
class _Island:
    joined: bool = False
    misses: int = 0
    last_merged_step: int | None = None
    absent: bool = False
    flagged: bool = False
    token: Any = None


@dataclass
class StragglerTracker:
    """Folds syncer tape records (:meth:`feed`) and answers :meth:`check` (module doc)."""

    cycles: int = DEFAULT_CYCLES
    islands: dict[int, _Island] = field(default_factory=dict)
    fragments: set[int] = field(default_factory=set)
    newest_step: int | None = None

    def _island(self, k: int) -> _Island:
        return self.islands.setdefault(int(k), _Island())

    @property
    def per_cycle(self) -> int:
        return max(1, len(self.fragments))

    def feed(self, record: Mapping[str, Any]) -> None:
        kind = record.get("kind")
        if kind == "pool_join":
            island = _int(record.get("island_id"))
            if island is not None and not self._island(island).joined:
                state = self._island(island)
                state.joined, state.misses, state.absent = True, 0, False
            return
        if kind != "outer_step":
            return
        step, fragment = _int(record.get("step")), _int(record.get("fragment"))
        if fragment is not None:
            self.fragments.add(fragment)
        if step is not None:
            self.newest_step = step if self.newest_step is None else max(self.newest_step, step)
        merged = {_int(entry[0]) for entry in _entries(record.get("weights")) + _entries(record.get("raw_weights"))}
        absent = {_int(i) for i in record.get("absent") or () if _int(i) is not None}
        for k, state in self.islands.items():
            if k in merged:
                state.misses, state.absent = 0, False
                state.last_merged_step = step if step is not None else state.last_merged_step
                state.joined = True  # a merged push: it is in the pool whatever the tape showed of its join
            elif state.joined:
                state.misses += 1
                state.absent = k in absent
        for k in merged - set(self.islands) - {None}:
            self.islands[k] = _Island(joined=True, last_merged_step=step)

    def check(self, running: Mapping[int, Any]) -> list[Straggler]:
        """The signals for the islands held running (learner id -> incarnation token)."""
        out: list[Straggler] = []
        threshold = self.cycles * self.per_cycle
        for k, token in running.items():
            state = self._island(int(k))
            if state.token != token:
                if state.token is not None:  # relaunched: its count starts at its next join
                    if state.flagged:
                        out.append(self._signal(int(k), state, cleared=True, reason="relaunched"))
                    state.joined, state.misses, state.absent, state.flagged = False, 0, False, False
                state.token = token
            if self.cycles <= 0 or not state.joined:
                continue
            if not state.flagged and state.misses >= threshold:
                state.flagged = True
                out.append(self._signal(int(k), state, cleared=False,
                                        reason=f"no push merged in {state.misses // self.per_cycle} merge cycle(s) "
                                        f"({state.misses} merges over {self.per_cycle} fragment(s))"))
            elif state.flagged and state.misses < threshold:
                state.flagged = False
                out.append(self._signal(int(k), state, cleared=True, reason="a push of it is merged again"))
        for k, state in self.islands.items():
            if k not in running:
                state.flagged = False  # not running (recovering, draining, ended): no stretch to close
        return out

    def _signal(self, k: int, state: _Island, *, cleared: bool, reason: str) -> Straggler:
        return Straggler(island=k, cleared=cleared, missed_cycles=state.misses // self.per_cycle,
                         merges=state.misses, fragments=self.per_cycle, cycles=self.cycles,
                         newest_step=self.newest_step, last_merged_step=state.last_merged_step,
                         absent=state.absent, reason=reason)


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entries(value: Any) -> list:
    return [e for e in value or () if isinstance(e, (list, tuple)) and e]


# -- the syncer's tape ---------------------------------------------------------------------

class FileTape:
    """The syncer's tape file on this host: the complete lines added since the last read."""

    def __init__(self, path: str | os.PathLike[str], *, offset: int = 0) -> None:
        self.path = Path(os.path.expanduser(str(path)))
        self.offset = int(offset)  # a reused host's tape: what an earlier syncer job wrote is not read
        self.label = str(self.path)

    def read(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.offset:  # rewritten from the start: read it again
            self.offset = 0
        if size == self.offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read(MAX_READ_BYTES)
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        return _records(data[: end + 1])


class StoreTape:
    """The syncer's tape in the run store (``tapes/syncer/<incarnation>/<seq>.jsonl``): the
    chunks not read yet, incarnations in the order they started, each chunk once."""

    def __init__(self, objects: Any, *, interval_s: float = STORE_POLL_S,
                 clock: Callable[[], float] | None = None) -> None:
        import time

        self.objects = objects
        self.interval_s = float(interval_s)
        self.clock = clock or time.monotonic
        self.cursors: dict[str, str] = {}  # incarnation -> its last chunk read
        self.label = f"{getattr(objects, 'uri', 'store')}/{SYNCER_TAPES}"
        self._next: float | None = None

    def read(self) -> list[dict[str, Any]]:
        from yeto.dashboard.store_tapes import _launch_ordered  # noqa: PLC0415

        now = self.clock()
        if self._next is not None and now < self._next:
            return []
        self._next = now + self.interval_s
        keys = []
        for key in self.objects.list(SYNCER_TAPES):
            parts = str(key)[len(SYNCER_TAPES):].split("/")
            if len(parts) == 2 and parts[1].endswith(".jsonl") and key > self.cursors.get(parts[0], ""):
                keys.append(str(key))
        out: list[dict[str, Any]] = []
        for key in _launch_ordered(keys, "tapes/"):
            data = self.objects.read(key)
            self.cursors[key[len(SYNCER_TAPES):].split("/")[0]] = key
            if data:
                out += _records(data if isinstance(data, bytes) else str(data).encode())
        return out


def _records(data: bytes) -> list[dict[str, Any]]:
    out = []
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


class StragglerWatch:
    """A tracker fed from one source: :meth:`poll` reads what the syncer added and returns
    the signals for the islands held running. A read that fails is logged once per stretch
    of failures and tried again at the next poll."""

    def __init__(self, source: Any, *, cycles: int = DEFAULT_CYCLES,
                 log: Callable[[str], None] | None = None) -> None:
        import sys

        self.source = source
        self.tracker = StragglerTracker(cycles=int(cycles))
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._failing = False
        self._empty_polls = 0  # polls with islands running and no syncer record read yet

    def poll(self, running: Mapping[int, Any]) -> list[Straggler]:
        try:
            records = self.source.read()
        except Exception as error:  # noqa: BLE001 - a watch, never the end of supervision
            if not self._failing:
                self.log(f"[launcher] island straggler watch: the syncer tape {self.source.label} does not read "
                         f"({type(error).__name__}: {error}); tried again next poll"[:400])
            self._failing = True
            records = []
        else:
            self._failing = False
        for record in records:
            self.tracker.feed(record)
        if records:
            self._empty_polls = -1
        elif running and self._empty_polls >= 0:
            self._empty_polls += 1
            if self._empty_polls == QUIET_POLLS:
                self.log(f"[launcher] island straggler watch: no syncer record read from {self.source.label} in "
                         f"{QUIET_POLLS} polls; no straggler is reported until the syncer's tape is there")
        return self.tracker.check(running)


def tape_source(*, local_tape: str | os.PathLike[str] | None = None, local_offset: int = 0,
                objects: Any = None) -> Any:
    """The syncer tape this controller can read: the local file of a syncer on this host,
    else the run store's copy; None when neither is there."""
    if local_tape:
        return FileTape(local_tape, offset=local_offset)
    if objects is not None:
        return StoreTape(objects)
    return None


def watch_for(args: Any, *, local_tape: str | os.PathLike[str] | None = None, local_offset: int = 0,
              objects: Any = None, log: Callable[[str], None] | None = None) -> StragglerWatch | None:
    """The launcher's watch of an island pool (``--rl-island-straggler-cycles``, default 3;
    0 turns it off): None without an island pool, with the watch off, or with no syncer tape
    to read from here (said once)."""
    import sys

    from .syncer import island_pool

    if getattr(args, "training_mode", "sft") != "rl" or not island_pool(args):
        return None
    cycles = straggler_cycles(args)
    if cycles == 0:
        return None
    source = tape_source(local_tape=local_tape, local_offset=local_offset, objects=objects)
    say = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if source is None:
        say("[launcher] island straggler watch off: no syncer tape to read here (a syncer on this host, or "
            "the run store's tapes/syncer/)")
        return None
    say(f"[launcher] island straggler watch: an island none of whose pushes is merged in {cycles} merge "
        f"cycle(s) is reported ({EVENT}); syncer tape {source.label}")
    return StragglerWatch(source, cycles=cycles, log=log)


def signal_fields(signal: Straggler, action: str | None) -> dict[str, Any]:
    """The event's and the island record's fields of ``signal``."""
    out = signal.fields()
    if action is not None:
        out["action"] = action
    return out
