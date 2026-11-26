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
* **Only this launch's history.** The watch reads only records stamped (``time_unix``) at or
  after its own start, and the run store's copy from the tail of the newest syncer
  incarnation it finds at its first read: a relaunched or moved island's new launch never
  counts the merges its earlier launches missed.
* **When counting starts.** At the island's first ``pool_join`` after the watch started, or
  after it left running (recovering, draining; a relaunch is a new incarnation token), or at
  a ``pool_join`` of a new generation or syncer term (a relaunch the controller did not
  see). What the syncer records after the island left running is its next incarnation's: a
  join written before the controller holds the new incarnation running is counted. A later re-JOIN of
  the same generation (a lease that expired) does not restart the count. A merged push of
  the island resets it. An island adopted by a restarted launcher has joined before the
  watch started: it is counted from its first merged push after that.
* **Who is watched.** The islands the controller holds running (not draining, not
  recovering): it passes their learner ids with a token of the incarnation.
* **Signals.** One :class:`Straggler` when an island crosses the threshold, and one
  ``cleared`` signal when a push of it is merged again, when it rejoined as a new generation,
  when it was relaunched, or when it stopped running (recovering, draining, ended): the
  controller writes one event per stretch and closes each one.

Sources (:func:`tape_source`): the syncer's tape file on this host (a head-mode syncer), or
its copy in the run store (``tapes/syncer/<incarnation>/<seq>.jsonl``, uploaded every 30 s
by ``yeto.rl.engine.tape_upload``). A store read costs at most :data:`STORE_LISTS_PER_POLL`
listings of one incarnation's prefix after its last chunk read and :data:`STORE_GETS_PER_POLL`
chunk reads, and starts no new request after :data:`STORE_POLL_BUDGET_S`. Finding the syncer's
incarnations costs one listing page per incarnation, at the first read, every
:data:`STORE_DISCOVER_POLLS` reads and while none is known; the first read also lists the
newest incarnation once, for its tail. A source that does not read is skipped for that poll; nothing
here ever stops or slows the fleet. When nothing can be watched (no tape to read, the watch
turned off) or the tape stays silent for :data:`QUIET_POLLS` reads, the controller is given
one notice (``island_straggler_unavailable``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVENT = "island_straggler"
CLEARED_EVENT = "island_straggler_cleared"
UNAVAILABLE_EVENT = "island_straggler_unavailable"
DEFAULT_CYCLES = 3
SYNCER_TAPES = "tapes/syncer/"
STORE_POLL_S = 30.0  # the syncer's uploads are 30 s apart (tape_upload.DEFAULT_INTERVAL_S)
STORE_GETS_PER_POLL = 16  # chunk reads per poll; the rest waits for the next poll
STORE_LISTS_PER_POLL = 4  # listings of the incarnation being read per poll
STORE_DISCOVER_POLLS = 10  # a full listing of tapes/syncer/ (a restarted syncer) every this many reads
STORE_POLL_BUDGET_S = 5.0  # no new store request after this long in one poll
MAX_READ_BYTES = 8 << 20  # one poll reads at most this much of a local tape
QUIET_POLLS = 10  # reads with islands running and no syncer record yet before that is said, once

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
    cleared: bool  # False: it became a straggler; True: its stretch ended (``reason`` says how)
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
    generation: tuple | None = None  # (syncer_epoch, generation) of the join counted from
    since: float | None = None  # wall time it last left running: older records are another incarnation's
    rejoined: str | None = None  # a flagged island joined again as a new generation: why its stretch ends
    away: bool = False  # not running now (recovering, draining, ended): its count restarts at its next join
    was_joined: bool = False  # whether it was in the pool when it left running


def _time(record: Mapping[str, Any]) -> float | None:
    value = record.get("time_unix")
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StragglerTracker:
    """Folds syncer tape records (:meth:`feed`) and answers :meth:`check` (module doc).
    ``since``: the watch's start (wall time); a record stamped before it is ignored."""

    cycles: int = DEFAULT_CYCLES
    since: float | None = None
    clock: Callable[[], float] = time.time
    islands: dict[int, _Island] = field(default_factory=dict)
    fragments: set[int] = field(default_factory=set)
    newest_step: int | None = None

    def _island(self, k: int) -> _Island:
        return self.islands.setdefault(int(k), _Island())

    @property
    def per_cycle(self) -> int:
        return max(1, len(self.fragments))

    @staticmethod
    def _before(t: float | None, since: float | None) -> bool:
        return t is not None and since is not None and t < since

    def feed(self, record: Mapping[str, Any]) -> None:
        kind = record.get("kind")
        t = _time(record)
        if kind not in ("pool_join", "outer_step") or self._before(t, self.since):
            return
        if kind == "pool_join":
            island = _int(record.get("island_id"))
            if island is None:
                return
            state = self._island(island)
            if self._before(t, state.since):
                return  # an earlier incarnation's join, read late
            generation = (record.get("syncer_epoch"), record.get("generation"))
            if not state.joined or generation != state.generation:
                if state.joined and state.flagged:
                    state.rejoined = f"rejoined (generation {generation[1]}, syncer term {generation[0]})"
                state.joined, state.misses, state.absent, state.generation = True, 0, False, generation
            return
        step, fragment = _int(record.get("step")), _int(record.get("fragment"))
        if fragment is not None:
            self.fragments.add(fragment)
        if step is not None:
            self.newest_step = step if self.newest_step is None else max(self.newest_step, step)
        merged = {_int(entry[0]) for entry in _entries(record.get("weights")) + _entries(record.get("raw_weights"))}
        absent = {_int(i) for i in record.get("absent") or () if _int(i) is not None}
        for k, state in self.islands.items():
            if self._before(t, state.since):
                continue  # merged before its relaunch: not this incarnation's
            if k in merged:
                state.misses, state.absent = 0, False
                state.last_merged_step = step if step is not None else state.last_merged_step
                state.joined = True  # a merged push: it is in the pool (an adopted island's join came earlier)
            elif state.joined:
                state.misses += 1
                state.absent = k in absent
        for k in merged - set(self.islands) - {None}:
            self.islands[k] = _Island(joined=True, last_merged_step=step)

    def check(self, running: Mapping[int, Any]) -> list[Straggler]:
        """The signals for the islands held running (learner id -> incarnation token), and
        the end of the stretch of a flagged island that is not running any more."""
        out: list[Straggler] = []
        threshold = self.cycles * self.per_cycle
        for k, state in self.islands.items():
            if k in running or state.token is None or state.away:
                continue
            # it left running: what the syncer records from now on is its next incarnation's (a
            # join of it may come before its controller holds it running again)
            if state.flagged:
                out.append(self._signal(k, state, cleared=True, reason="not running"))
            was_joined = state.joined
            self._restart(state)
            state.away, state.was_joined = True, was_joined
        for k, token in running.items():
            state = self._island(int(k))
            if state.away and state.token == token:  # the same incarnation back: counted from now
                state.joined = state.joined or state.was_joined
            elif state.token != token and state.token is not None and not state.away:
                # relaunched between two checks: its count starts at its next join
                if state.flagged:
                    out.append(self._signal(int(k), state, cleared=True, reason="relaunched"))
                self._restart(state)
            state.token, state.away = token, False
            if state.rejoined is not None:
                if state.flagged:
                    out.append(self._signal(int(k), state, cleared=True, reason=state.rejoined))
                state.flagged, state.rejoined = False, None
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
            if k not in running and state.flagged:  # never held running here
                state.flagged = False
                out.append(self._signal(k, state, cleared=True, reason="not running"))
        return out

    def _restart(self, state: _Island) -> None:
        state.joined, state.misses, state.absent, state.flagged = False, 0, False, False
        state.generation, state.rejoined, state.since = None, None, self.clock()

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
        self.error: str | None = None
        self.polled = True  # every call reads

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


def _incarnation_order(incarnation: str) -> tuple:
    from yeto.dashboard.store_tapes import launch_started

    started = launch_started(incarnation)
    return (0 if started is None else 1, started or 0.0, incarnation)


class StoreTape:
    """The syncer's tape in the run store (``tapes/syncer/<incarnation>/<seq>.jsonl``).

    The first read finds the newest syncer incarnation and starts at its tail: what the
    syncer wrote before this watch is not read. From then on each read lists that
    incarnation's prefix after its last chunk read (``start_after``) and reads its new chunks
    in order, at most :data:`STORE_GETS_PER_POLL` (the rest at the next read). A newer
    incarnation (a restarted syncer, found by a full listing every
    :data:`STORE_DISCOVER_POLLS` reads) is read from its start once the current one has no
    chunk left. A chunk read moves the cursor past it, so a request that fails keeps what was
    read before it and is tried again at the next read (``error`` says why); no chunk is
    read twice or skipped."""

    def __init__(self, objects: Any, *, interval_s: float = STORE_POLL_S,
                 clock: Callable[[], float] | None = None, gets_per_poll: int = STORE_GETS_PER_POLL,
                 discover_polls: int = STORE_DISCOVER_POLLS, budget_s: float = STORE_POLL_BUDGET_S) -> None:
        self.objects = objects
        self.interval_s = float(interval_s)
        self.clock = clock or time.monotonic
        self.gets_per_poll, self.discover_polls, self.budget_s = int(gets_per_poll), int(discover_polls), budget_s
        self.queue: list[str] = []  # the incarnations to read, the current one first
        self.known: set[str] = set()
        self.cursors: dict[str, str] = {}  # incarnation -> its last chunk read
        self.label = f"{getattr(objects, 'uri', 'store')}/{SYNCER_TAPES}"
        self.error: str | None = None
        self.polled = False  # whether the last read() asked the store (False within the interval)
        self.reads = 0
        self.requests = 0  # listings and chunk reads so far
        self._next: float | None = None

    def _incarnations(self) -> set[str]:
        """The syncer incarnations under ``tapes/syncer/``: one listing page per incarnation
        (a page that ends inside one jumps past the rest of its chunks)."""
        page = getattr(self.objects, "list_page", None)
        if not callable(page):
            self.requests += 1
            return {str(k)[len(SYNCER_TAPES):].split("/")[0] for k in self.objects.list(SYNCER_TAPES)
                    if len(str(k)[len(SYNCER_TAPES):].split("/")) == 2}
        found: set[str] = set()
        cursor = ""
        while True:
            self.requests += 1
            keys, last = page(SYNCER_TAPES, start_after=cursor, limit=1000)
            found |= {str(k)[len(SYNCER_TAPES):].split("/")[0] for k in keys
                      if len(str(k)[len(SYNCER_TAPES):].split("/")) == 2}
            if last is None or not keys:
                return found
            cursor = f"{SYNCER_TAPES}{str(last)[len(SYNCER_TAPES):].split('/')[0]}/~"  # past its chunks

    def _discover(self, *, first: bool) -> None:
        new = sorted((inc for inc in self._incarnations() if inc not in self.known), key=_incarnation_order)
        if first and new:
            newest = new[-1]
            self.known.update(new)
            self.queue = [newest]
            self.requests += 1
            chunks = [str(k) for k in self.objects.list(f"{SYNCER_TAPES}{newest}/") if str(k).endswith(".jsonl")]
            self.cursors[newest] = max(chunks, default="")  # its tail: what it wrote before the watch is not read
            return
        newer = [inc for inc in new if not self.queue or _incarnation_order(inc) > _incarnation_order(self.queue[-1])]
        self.known.update(new)  # an older one found late (its syncer ran before this watch): not read
        self.queue += newer

    def read(self) -> list[dict[str, Any]]:
        now = self.clock()
        self.polled = False
        if self._next is not None and now < self._next:
            return []
        self._next = now + self.interval_s
        self.polled, self.error = True, None
        started, out, gets, lists = self.clock(), [], 0, 0
        first = self.reads == 0
        self.reads += 1
        try:
            if first or not self.queue or self.reads % self.discover_polls == 0:
                self._discover(first=first)
                if first:
                    return []
            while self.queue and gets < self.gets_per_poll and lists < STORE_LISTS_PER_POLL \
                    and self.clock() - started < self.budget_s:
                inc = self.queue[0]
                lists += 1
                self.requests += 1
                keys = sorted(str(k) for k in self.objects.list(f"{SYNCER_TAPES}{inc}/",
                                                                start_after=self.cursors.get(inc, ""))
                              if str(k).endswith(".jsonl") and str(k) > self.cursors.get(inc, ""))
                if not keys:
                    if len(self.queue) > 1:
                        self.queue.pop(0)  # a newer syncer incarnation took over: read it from its start
                        continue
                    break
                for key in keys:
                    if gets >= self.gets_per_poll or self.clock() - started >= self.budget_s:
                        break
                    gets += 1
                    self.requests += 1
                    data = self.objects.read(key)
                    self.cursors[inc] = key
                    if data:
                        out += _records(data if isinstance(data, bytes) else str(data).encode())
                else:
                    continue
                break
        except Exception as error:  # noqa: BLE001 - what was read is kept; the rest at the next read
            self.error = f"{type(error).__name__}: {error}"[:300]
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
    of failures and tried again at the next poll. :meth:`notices` hands the controller, once,
    why nothing is reported (the source stayed silent for :data:`QUIET_POLLS` reads)."""

    def __init__(self, source: Any, *, cycles: int = DEFAULT_CYCLES, since: float | None = None,
                 log: Callable[[str], None] | None = None, clock: Callable[[], float] = time.time) -> None:
        import sys

        self.source = source
        self.tracker = StragglerTracker(cycles=int(cycles), since=since, clock=clock)
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._failing = False
        self._quiet = 0  # reads with islands running and no syncer record yet; -1 once one was read
        self._notices: list[str] = []

    def poll(self, running: Mapping[int, Any]) -> list[Straggler]:
        try:
            records = self.source.read()
            error = getattr(self.source, "error", None)
        except Exception as raised:  # noqa: BLE001 - a watch, never the end of supervision
            records, error = [], f"{type(raised).__name__}: {raised}"
        if error is not None:
            if not self._failing:
                self.log(f"[launcher] island straggler watch: the syncer tape {self.source.label} does not read "
                         f"({error}); tried again next poll"[:400])
            self._failing = True
        else:
            self._failing = False
        for record in records:
            self.tracker.feed(record)
        if records:
            self._quiet = -1
        elif running and self._quiet >= 0 and getattr(self.source, "polled", True):
            self._quiet += 1
            if self._quiet == QUIET_POLLS:
                why = (f"no syncer record read from {self.source.label} in {QUIET_POLLS} reads; no straggler is "
                       "reported until the syncer's tape is there")
                self.log(f"[launcher] WARNING: island straggler watch: {why}")
                self._notices.append(why)
        return self.tracker.check(running)

    def notices(self) -> list[str]:
        out, self._notices = self._notices, []
        return out


class UnavailableWatch:
    """An island pool whose stragglers cannot be watched from here: no signal ever, and one
    notice saying why (the controller writes it as ``island_straggler_unavailable``)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self._notices = [reason]

    def poll(self, running: Mapping[int, Any]) -> list[Straggler]:
        return []

    def notices(self) -> list[str]:
        out, self._notices = self._notices, []
        return out


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
              objects: Any = None, log: Callable[[str], None] | None = None,
              no_source: str | None = None) -> StragglerWatch | UnavailableWatch | None:
    """The launcher's watch of an island pool (``--rl-island-straggler-cycles``, default 3;
    0 turns it off): None without an island pool. With the watch off, or with no syncer
    tape to read from here (``no_source``: why), an :class:`UnavailableWatch` that says so
    once. The watch counts only what the syncer records from now on."""
    import sys

    from .syncer import island_pool

    if getattr(args, "training_mode", "sft") != "rl" or not island_pool(args):
        return None
    say = log or (lambda m: print(m, file=sys.stderr, flush=True))
    cycles = straggler_cycles(args)
    if cycles == 0:
        say("[launcher] island straggler watch off (--rl-island-straggler-cycles 0)")
        return UnavailableWatch("turned off (--rl-island-straggler-cycles 0)")
    source = tape_source(local_tape=local_tape, local_offset=local_offset, objects=objects)
    if source is None:
        why = no_source or "no syncer tape to read here (a syncer on this host, or the run store's tapes/syncer/)"
        say(f"[launcher] island straggler watch off: {why}")
        return UnavailableWatch(why)
    say(f"[launcher] island straggler watch: an island none of whose pushes is merged in {cycles} merge "
        f"cycle(s) is reported ({EVENT}); syncer tape {source.label}")
    return StragglerWatch(source, cycles=cycles, since=time.time(), log=log)


def signal_fields(signal: Straggler, action: str | None) -> dict[str, Any]:
    """The event's and the island record's fields of ``signal``."""
    out = signal.fields()
    if action is not None:
        out["action"] = action
    return out
