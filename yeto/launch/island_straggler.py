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
* **Only this launch's history.** Each source starts at the tape's end when the watch starts
  (the local file at its offset; the run store's copy at the tail of every syncer incarnation
  there): a relaunched or moved island's new launch never counts the merges its earlier
  launches missed, whatever the hosts' clocks say. Only an incarnation the store lists later
  is read from its start, without its records stamped more than :data:`CLOCK_SKEW_S` before
  the watch's start (a listing that missed it at the first read).
* **When counting starts.** At the island's first ``pool_join`` read by this watch, or at the
  first merge that lists it ``absent`` (the syncer expects it: an island a restarted launcher
  adopted joined before this watch), or at its first merged push. Only its controller resets
  the count: when the island leaves running (recovering, draining, stopped) it starts again
  once it runs again: at once for the same incarnation; for a new one (a relaunch) from its
  join. A join read while the controller holds the island away from running (recovering)
  counts; a relaunch the controller holds running again by its next poll, never seen away,
  is counted from the first join read after that poll. A ``pool_join``
  of a reconnect (each connection is a new generation) or of a syncer that took over (a new
  term) keeps the count and the mark. A merged push of the island resets the count.
* **A new launch catches up first.** A launch its controller relaunched (a spare node, a
  move) or a new island whose ``pool_join`` this watch read is not counted until its first
  merged push, for at most ``grace_s`` (the controller's recover timeout) on the controller's
  clock; past that it is counted from then on. An adopted island (no join read) has no grace,
  and a reconnect's join neither starts nor extends one.
* **Who is watched.** The islands the controller holds running (not draining, not
  recovering): it passes their learner ids with a token of the incarnation.
* **Signals.** One :class:`Straggler` when an island crosses the threshold, and one
  ``cleared`` signal when a push of it is merged again, when it was relaunched, or when it
  stopped running (recovering, draining, ended): the controller writes one event per stretch
  and closes each one.

Sources (:func:`tape_source`): the syncer's tape file on this host (a head-mode syncer), or
its copy in the run store (``tapes/syncer/<incarnation>/<seq>.jsonl``, uploaded every 30 s
by ``yeto.rl.engine.tape_upload``). A store read costs one listing of each known incarnation's
prefix after its last chunk read and at most :data:`STORE_GETS_PER_POLL` chunk reads, and
starts no new request after :data:`STORE_POLL_BUDGET_S`. Finding the syncer's incarnations
costs one listing page per incarnation, at the first read, every :data:`STORE_DISCOVER_POLLS`
reads and while none is known; the first read also lists each incarnation once, for its
tail. A source that does not read is skipped
for that poll; nothing here ever stops or slows the fleet. When nothing can be watched (no tape
to read, the watch turned off), or the tape goes silent for :func:`quiet_window_s` after its
first record about this launch's islands, the controller is given one notice
(``island_straggler_unavailable``), and one more when records are read again
(``island_straggler_available``).
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
AVAILABLE_EVENT = "island_straggler_available"  # the syncer's records are read again after a silent stretch
DEFAULT_CYCLES = 3
SYNCER_TAPES = "tapes/syncer/"
STORE_POLL_S = 30.0  # the syncer's uploads are 30 s apart (tape_upload.DEFAULT_INTERVAL_S)
STORE_GETS_PER_POLL = 16  # chunk reads per poll; the rest waits for the next poll
STORE_DISCOVER_POLLS = 10  # a full listing of tapes/syncer/ (a restarted syncer) every this many reads
STORE_POLL_BUDGET_S = 5.0  # no new store request after this long in one poll
MAX_READ_BYTES = 8 << 20  # one poll reads at most this much of a local tape
QUIET_MIN_S = 300.0  # the shortest silence of the syncer's tape, after its first record about this launch, that is said
CLOCK_SKEW_S = 600.0  # the launcher's and the syncer host's clocks may differ by this much

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
    away: bool = False  # not running now (recovering, draining, ended): its count restarts
    was_joined: bool = False  # whether it was in the pool when it left running
    joined_away: bool = False  # a pool_join read while it was away (its next incarnation's, or a reconnect)
    pushed: bool = False  # a push of this launch was merged
    join_read: bool = False  # a pool_join of this launch was read (a new island; an adopted one has none)
    grace_until: float | None = None  # a new launch catching up: not counted until then (controller clock)
    graced: bool = False  # its grace ran out before its first merged push


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
    Which records are this launch's is its source's business (each one starts at the tape's
    end); no clock is read here."""

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
        if kind not in ("pool_join", "outer_step"):
            return
        if kind == "pool_join":
            island = _int(record.get("island_id"))
            if island is None:
                return
            # in the pool; a reconnect (a new connection generation) or a new syncer term keeps
            # its count and its mark: only its controller's relaunch or stop restarts them
            state = self._island(island)
            state.joined, state.join_read = True, True
            state.joined_away = state.joined_away or state.away
            return
        step, fragment = _int(record.get("step")), _int(record.get("fragment"))
        if fragment is not None:
            self.fragments.add(fragment)
        if step is not None:
            self.newest_step = step if self.newest_step is None else max(self.newest_step, step)
        merged = {_int(entry[0]) for entry in _entries(record.get("weights")) + _entries(record.get("raw_weights"))}
        absent = {_int(i) for i in record.get("absent") or () if _int(i) is not None}
        for k in (merged | absent) - set(self.islands) - {None}:
            self.islands[k] = _Island()
        for k, state in self.islands.items():
            if k in merged:
                state.misses, state.absent = 0, False
                state.last_merged_step = step if step is not None else state.last_merged_step
                state.joined = True  # a merged push: it is in the pool
                state.pushed, state.grace_until, state.graced = True, None, False  # its catch-up is over
            elif state.joined or k in absent:
                # listed absent: the syncer expects it, so it is in the pool even when its join
                # came before this watch (an island adopted by a restarted launcher)
                state.joined = True
                state.absent = k in absent
                if state.grace_until is None:  # a new launch still catching up is not counted
                    state.misses += 1

    def check(self, running: Mapping[int, Any], *, now: float | None = None,
              grace_s: float = 0.0) -> list[Straggler]:
        """The signals for the islands held running (learner id -> incarnation token), and
        the end of the stretch of a flagged island that is not running any more. ``now``:
        the controller's clock; ``grace_s``: how long a new launch may catch up before its
        first merged push without being counted (its controller's recover timeout; 0: none)."""
        out: list[Straggler] = []
        threshold = self.cycles * self.per_cycle
        for k, state in self.islands.items():
            if k in running or state.token is None or state.away:
                continue
            # it left running: its count restarts when it is running again
            if state.flagged:
                out.append(self._signal(k, state, cleared=True, reason="not running"))
            was_joined = state.joined
            self._restart(state)
            state.away, state.was_joined, state.joined_away = True, was_joined, False
        for k, token in running.items():
            state = self._island(int(k))
            relaunched = state.token is not None and state.token != token
            if state.away:
                # back: the same incarnation keeps its place in the pool; a new one (relaunched)
                # is in it once it joined, which may come before its controller holds it running
                state.joined = state.joined_away or (state.token == token and state.was_joined)
                state.misses, state.absent = 0, False
            elif state.token is not None and state.token != token:
                # relaunched between two checks: its count starts at its next join
                if state.flagged:
                    out.append(self._signal(int(k), state, cleared=True, reason="relaunched"))
                self._restart(state)
            state.token, state.away, state.joined_away = token, False, False
            # a new launch catches up before its first push: a relaunch (a spare node too), or a
            # new island whose join this watch read (an adopted island joined before the watch).
            # One grace per launch: a reconnect's join neither starts nor extends it
            if ((relaunched or state.join_read) and not state.pushed and not state.graced
                    and state.grace_until is None and now is not None and grace_s > 0):
                state.grace_until, state.misses = now + grace_s, 0
            if state.grace_until is not None:
                if now is None or now < state.grace_until:
                    continue  # still catching up: not counted
                # no push merged within its recover timeout: counted from now on
                state.grace_until, state.graced, state.misses = None, True, 0
            if self.cycles <= 0 or not state.joined:
                continue
            if not state.flagged and state.misses >= threshold:
                state.flagged = True
                out.append(self._signal(int(k), state, cleared=False,
                                        reason=f"no push merged in {state.misses // self.per_cycle} merge cycle(s) "
                                        f"({state.misses} merges over {self.per_cycle} fragment(s))"
                                        + (" after its catch-up grace (the recover timeout) ran out with no push "
                                           "of its launch merged" if state.graced else "")))
            elif state.flagged and state.misses < threshold:
                state.flagged = False
                out.append(self._signal(int(k), state, cleared=True, reason="a push of it is merged again"))
        for k, state in self.islands.items():
            if k not in running and state.flagged:  # never held running here
                state.flagged = False
                out.append(self._signal(k, state, cleared=True, reason="not running"))
        return out

    @staticmethod
    def _restart(state: _Island) -> None:
        state.joined, state.misses, state.absent, state.flagged = False, 0, False, False
        state.pushed, state.join_read, state.grace_until, state.graced = False, False, None, False

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


class StoreTape:
    """The syncer's tape in the run store (``tapes/syncer/<incarnation>/<seq>.jsonl``).

    No incarnation is picked as "the" syncer: a restarted syncer may resume an earlier step,
    keep its term, or write only its own records (round opens, connections) at first. The
    first read lists every syncer incarnation and keeps each one's tail (its last chunk): what
    the syncers wrote before this watch is not read. Until that worked (a listing failed)
    every read tries it again and reads nothing else. From then on each read lists every known
    incarnation after its last chunk read (``start_after``; the one that grew most recently
    first, then the one listed longest ago) and reads the new chunks of those that grew, at most
    :data:`STORE_GETS_PER_POLL` (the rest at the next read). An incarnation found later (by a
    full listing every :data:`STORE_DISCOVER_POLLS` reads) did not exist at the first read: it
    is read from its start, without its records stamped more than :data:`CLOCK_SKEW_S` before
    ``since`` (a listing that missed it at the first read). A chunk read moves the cursor past
    it, so a request that fails keeps what was read before it and is tried again at the next
    read (``error`` says why); no chunk is read twice or skipped."""

    def __init__(self, objects: Any, *, interval_s: float = STORE_POLL_S,
                 clock: Callable[[], float] | None = None, gets_per_poll: int = STORE_GETS_PER_POLL,
                 discover_polls: int = STORE_DISCOVER_POLLS, budget_s: float = STORE_POLL_BUDGET_S,
                 since: float | None = None) -> None:
        self.objects = objects
        self.interval_s = float(interval_s)
        self.clock = clock or time.monotonic
        self.gets_per_poll, self.discover_polls, self.budget_s = int(gets_per_poll), int(discover_polls), budget_s
        self.since = time.time() if since is None else float(since)  # the watch's start (wall time)
        self.cursors: dict[str, str] = {}  # incarnation -> its last chunk read ("" before its first)
        self.fresh: set[str] = set()  # incarnations read from their start (found after the first read)
        self.grew: dict[str, int] = {}  # incarnation -> the read that last found a new chunk of it
        self.listed: dict[str, int] = {}  # incarnation -> the read that last listed it
        self.label = f"{getattr(objects, 'uri', 'store')}/{SYNCER_TAPES}"
        self.error: str | None = None
        self.polled = False  # whether the last read() asked the store (False within the interval)
        self.tailed = False  # the first read's tails are established
        self.reads = 0
        self.requests = 0  # listings and chunk reads so far
        self._next: float | None = None

    @property
    def known(self) -> set[str]:
        return set(self.cursors)

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

    def _chunks(self, inc: str, after: str = "") -> list[str]:
        self.requests += 1
        return sorted(str(k) for k in self.objects.list(f"{SYNCER_TAPES}{inc}/", start_after=after)
                      if str(k).endswith(".jsonl") and str(k) > after)

    def _tail(self) -> None:
        """The first read: every incarnation's tail. Committed only when every listing worked
        (else tried again at the next read)."""
        cursors = {}
        for inc in sorted(self._incarnations()):
            chunks = self._chunks(inc)
            cursors[inc] = chunks[-1] if chunks else ""
        self.cursors.update(cursors)
        self.tailed = True

    def _discover(self) -> None:
        for inc in sorted(self._incarnations() - set(self.cursors)):  # a known one keeps its cursor
            self.cursors[inc] = ""  # not there at the first read: read from its start
            self.fresh.add(inc)

    def _records_of(self, inc: str, data: Any) -> list[dict[str, Any]]:
        records = _records(data if isinstance(data, bytes) else str(data).encode())
        if inc not in self.fresh:
            return records
        floor = self.since - CLOCK_SKEW_S
        return [r for r in records if (_time(r) is None or _time(r) >= floor)]

    def read(self) -> list[dict[str, Any]]:
        now = self.clock()
        self.polled = False
        if self._next is not None and now < self._next:
            return []
        self._next = now + self.interval_s
        self.polled, self.error = True, None
        started, out, gets = self.clock(), [], 0
        self.reads += 1
        try:
            if not self.tailed:
                self._tail()
                return []
            if not self.cursors or self.reads % self.discover_polls == 0:
                self._discover()
            order = sorted(self.cursors, key=lambda i: (-self.grew.get(i, -1), self.listed.get(i, -1), i))
            for inc in order:
                if gets >= self.gets_per_poll or self.clock() - started >= self.budget_s:
                    break
                self.listed[inc] = self.reads
                keys = self._chunks(inc, self.cursors[inc])
                if keys:
                    self.grew[inc] = self.reads
                for key in keys:
                    if gets >= self.gets_per_poll or self.clock() - started >= self.budget_s:
                        break
                    gets += 1
                    self.requests += 1
                    data = self.objects.read(key)
                    self.cursors[inc] = key
                    if data:
                        out += self._records_of(inc, data)
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
    of failures and tried again at the next poll. :meth:`notices` hands the controller
    ``(event, reason)`` pairs to write: once the syncer recorded something about this launch's
    islands (a join, or a merge that merged or expected one), ``quiet_s`` (:func:`quiet_window_s`)
    without any record read is said once (:data:`UNAVAILABLE_EVENT`), and the next record read
    takes it back (:data:`AVAILABLE_EVENT`). A fleet whose trainers start slowly is never said
    to be unavailable: nothing is counted before that first record."""

    def __init__(self, source: Any, *, cycles: int = DEFAULT_CYCLES, quiet_s: float = QUIET_MIN_S,
                 log: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        import sys

        self.source = source
        self.tracker = StragglerTracker(cycles=int(cycles))
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.quiet_s, self.clock = float(quiet_s), clock
        self._failing = False
        self._seen = False  # a record about one of this launch's islands was read
        self._last: float | None = None  # when a record was last read, once seen
        self._said = False  # the silence was said (UNAVAILABLE_EVENT) and not taken back yet
        self._notices: list[tuple[str, str]] = []

    @staticmethod
    def _about(record: Mapping[str, Any], islands: Any) -> bool:
        if record.get("kind") == "pool_join":
            return _int(record.get("island_id")) in islands
        if record.get("kind") == "outer_step":
            named = {_int(e[0]) for e in _entries(record.get("weights")) + _entries(record.get("raw_weights"))}
            named |= {_int(i) for i in record.get("absent") or ()}
            return bool(named & set(islands))
        return False

    def poll(self, running: Mapping[int, Any], *, now: float | None = None,
             grace_s: float = 0.0) -> list[Straggler]:
        """Read the tape and judge the islands held running (:meth:`StragglerTracker.check`)."""
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
        islands = {int(k) for k in running} | {k for k, s in self.tracker.islands.items() if s.token is not None}
        self._seen = self._seen or any(self._about(r, islands) for r in records)
        now_q = self.clock()
        if records and self._seen:
            self._last = now_q
            if self._said:
                self._said = False
                self._notices.append((AVAILABLE_EVENT, f"syncer records read again from {self.source.label}"))
        elif (self._seen and running and getattr(self.source, "polled", True) and self._last is not None
              and now_q - self._last >= self.quiet_s and not self._said):
            self._said = True
            why = (f"no syncer record read from {self.source.label} in {self.quiet_s:g} s; no straggler is "
                   "reported until the syncer's tape is read again")
            self.log(f"[launcher] WARNING: island straggler watch: {why}")
            self._notices.append((UNAVAILABLE_EVENT, why))
        return self.tracker.check(running, now=now, grace_s=grace_s)

    def notices(self) -> list[tuple[str, str]]:
        out, self._notices = self._notices, []
        return out


class UnavailableWatch:
    """An island pool whose stragglers cannot be watched from here: no signal ever, and one
    notice saying why (the controller writes it as ``island_straggler_unavailable``)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self._notices = [(UNAVAILABLE_EVENT, reason)]

    def poll(self, running: Mapping[int, Any], **_judge: Any) -> list[Straggler]:
        return []

    def notices(self) -> list[tuple[str, str]]:
        out, self._notices = self._notices, []
        return out


def tape_source(*, local_tape: str | os.PathLike[str] | None = None, local_offset: int = 0,
                objects: Any = None, since: float | None = None) -> Any:
    """The syncer tape this controller can read: the local file of a syncer on this host,
    else the run store's copy; None when neither is there."""
    if local_tape:
        return FileTape(local_tape, offset=local_offset)
    if objects is not None:
        return StoreTape(objects, since=since)
    return None


def quiet_window_s(args: Any, *, upload_s: float = 0.0) -> float:
    """How long the syncer's tape may be silent before that is said: two of its longest rounds
    (a round ends at its soft deadline, else at its quorum timeout: ``--rl-elastic-quorum-timeout-s``
    or the syncer's default) and two upload intervals of its store copy (``upload_s``; 0 for a
    tape on this host), at least :data:`QUIET_MIN_S`."""
    from .syncer import SYNCER_DEFAULT_QUORUM_TIMEOUT_S, _rl_quorum_timeout_s

    soft = getattr(args, "rl_sync_soft_deadline_s", None)
    round_s = float(soft) if soft else float(_rl_quorum_timeout_s(args) or SYNCER_DEFAULT_QUORUM_TIMEOUT_S)
    return max(QUIET_MIN_S, 2.0 * round_s + 2.0 * max(0.0, float(upload_s or 0.0)))


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
    source = tape_source(local_tape=local_tape, local_offset=local_offset, objects=objects, since=time.time())
    if source is None:
        why = no_source or "no syncer tape to read here (a syncer on this host, or the run store's tapes/syncer/)"
        say(f"[launcher] island straggler watch off: {why}")
        return UnavailableWatch(why)
    say(f"[launcher] island straggler watch: an island none of whose pushes is merged in {cycles} merge "
        f"cycle(s) is reported ({EVENT}); syncer tape {source.label}")
    upload_s = 0.0
    if isinstance(source, StoreTape):
        from ..rl.engine.tape_upload import DEFAULT_INTERVAL_S, upload_interval_s

        upload_s = upload_interval_s() or DEFAULT_INTERVAL_S
    return StragglerWatch(source, cycles=cycles, quiet_s=quiet_window_s(args, upload_s=upload_s), log=log)


def signal_fields(signal: Straggler, action: str | None) -> dict[str, Any]:
    """The event's and the island record's fields of ``signal``."""
    out = signal.fields()
    if action is not None:
        out["action"] = action
    return out
