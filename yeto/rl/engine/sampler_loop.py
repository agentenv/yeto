"""The sampler island's loop (rl-trainer-start-stop).

A sampler island generates rounds and writes their groups to the durable sample
buffer (the engine's rollout function does the write). Before each round it:

1. loads the newest policy release, when there is one (:class:`PolicyFollower`);
2. waits while the backlog (groups in the buffer the trainer has not read, as of
   its last release) is at the cap. A stopped trainer releases nothing, so the
   sampler keeps the last version and stops at the cap; a newer release (the
   trainer runs again) lifts the wait.

With ``require_release`` (a LoRA sampler: its engines serve no adapter until
the first release is loaded) step 2 also waits until a release is loaded.

A load hook may refuse a release without stopping the island
(:class:`~.policy_release.ReleaseNotLoaded`: a scheme the engine does not serve,
a failed post-load check, a delta release whose rebuilt files fail their digest
check). The loop writes ``rl_sampler_release_not_loaded`` and goes on; with
``ready`` (the engine sampler: False while the engine holds
weights that failed a check) it waits instead of generating, so no sample of
unchecked weights reaches the buffer.

Staleness: ``rl_sampler_round`` carries the version the round generated with,
``version_lag`` (trainer versions between the newest release seen and that
version, at the round start) and the follower's release counters
(:class:`~.policy_release.FollowerCounters`). The round runs inside
``follower.generating()``: a refresh during it loads nothing unless the engines
update their weights mid-generation.

Updates during a generation (``follower.live_update``): while a round
generates, a background thread polls the store every ``live_poll_s`` and
loads each new release into the running engine (the adapter pauses the
generation, loads the weights and continues). Only that thread calls the
follower during the round; it stops before the round's events are written.
``rl_sampler_policy_loaded`` then carries ``in_flight`` True. A load whose
post-load checks fail during the round sets :attr:`SamplerLoop.unchecked_in_flight`
(the round's caller drops its groups); an error other than
:class:`~.policy_release.ReleaseNotLoaded` stops the island after the round.

``rl_sampler_policy_loaded`` carries the release latency fields: the trainer's
``released_at``, the follower's ``seen_at`` (both wall clock, of two hosts) and
``follower_timings`` (:attr:`~.policy_release.PolicyFollower.last_timings`),
and ``deferred_s`` (how long the in-flight rule deferred it) and ``forced``
True when the follower's deferral bound (``max_defer_s``) loaded it.

Store reads of the loop are polls: the newest release (``LATEST``), the
buffer's own index (the backlog, the round's group count). One that fails is no
verdict and never ends the island: the served weights stay
(``rl_sampler_release_not_loaded``, reason ``latest_unread``; a ``LATEST`` that is
there but is no release record, ``damaged``, ends the island after
``max_damaged_latest_reads`` reads in a row: the trainer rewrites it with every
release, so one that stays damaged is not an outage), a round is not
admitted while the backlog is unknown, and ``rl_sampler_store_read_failed`` /
``rl_sampler_store_read_ok`` mark the outage on the tape; the next poll reads
again. Only a failed group write ends a round.

After the last round it calls ``generate.flush()`` when the round has one (a
streaming agentic round writes the groups still in flight) and marks the data end, so the trainer stops after reading
the rest. ``num_rollouts`` None: no last round; the loop generates until
``should_stop`` (a full-parameter synced group's sampler stops when its trainers
ended, ``sample_roles`` module doc) and marks no data end. The loop is
engine-neutral: ``generate(rollout_id)`` runs one round.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from .merged_release import fastest_lane_read
from .object_store import is_fatal_store_error
from .policy_release import LATEST_UNREAD, PolicyFollower, ReleaseNotLoaded
from .sample_buffer import SampleBufferError, buffer_dir, read_read_position
from .task_dispense import DispenseEnded, DispenseFencedEnd

DEFAULT_POLL_S = 5.0
READ_FAILED_EVENT = "rl_sampler_store_read_failed"  # a store poll of the loop failed (first of an outage)
READ_OK_EVENT = "rl_sampler_store_read_ok"  # ... and the first one that worked again
DEFAULT_MAX_DAMAGED_LATEST_READS = 120  # reads in a row of a LATEST that does not parse (10 min at poll_s 5)
# the longest one poll of the loop may keep failing before the island ends on it: past the
# read outage it rides out (DEFAULT_WRITE_DEFER_MAX_S, 10 min), well short of "forever"
DEFAULT_MAX_READ_OUTAGE_S = 1800.0


class StoreReadOutage(RuntimeError):
    """A poll of the loop failed for longer than ``max_read_outage_s``."""


def heartbeat_read(beat: Mapping[str, Any], seen: Any, shard: str | None, read: int) -> int:
    """The read position to count the backlog from, given the trainer's heartbeat
    ``beat`` (``sample_buffer.read_read_position``), the sampler's last seen release
    ``seen`` and the release's position ``read``.

    A heartbeat that names the trainer's newest release (``release``) is the
    trainer's current position when it belongs to the lineage the sampler follows:
    the same release (seq and policy hash) or a newer one written after ``seen``.
    It then replaces the release's position, also when it is lower (a trainer
    resumed from an earlier checkpoint reads again from there). One from another
    run or from before ``seen`` is ignored. An old-format heartbeat (no
    ``release``) can only move the position forward."""
    if shard is not None:
        if shard not in beat["buffer_shards"]:
            return read
        beat_read = int(beat["buffer_shards"][shard])
    else:
        beat_read = int(beat["buffer_selected"])
    if "release" not in beat:
        return max(read, beat_read)
    release = beat["release"]
    if seen is None:
        return beat_read if release is None else max(read, beat_read)
    if release is None:
        return read
    if int(release["seq"]) == int(seen.seq):
        return beat_read if str(release["policy_hash"]) == str(seen.policy_hash) else read
    released_at = float(getattr(seen, "released_at", 0.0) or 0.0)
    if int(release["seq"]) > int(seen.seq) and float(beat.get("time_unix") or 0.0) >= released_at:
        return beat_read
    return read


WAIT_HEARTBEAT_S = 60.0


class SamplerLoop:
    def __init__(self, *, follower: PolicyFollower, buffer: Any,
                 generate: Callable[[int], None], num_rollouts: int | None,
                 max_backlog_groups: int, emit: Callable[..., None],
                 poll_s: float = DEFAULT_POLL_S, sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False,
                 min_round_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 require_release: bool = False,
                 loaded_fields: Callable[[], dict[str, Any]] = dict,
                 ready: Callable[[], bool] = lambda: True,
                 live_poll_s: float | None = None) -> None:
        if int(max_backlog_groups) < 1:
            raise ValueError("max_backlog_groups must be >= 1")
        self.follower = follower
        self.buffer = buffer
        self.generate = generate
        self.num_rollouts = None if num_rollouts is None else int(num_rollouts)
        self.max_backlog_groups = int(max_backlog_groups)
        self.emit = emit
        self.poll_s = float(poll_s)
        self.sleep = sleep
        self.should_stop = should_stop
        self.min_round_s = None if min_round_s is None else float(min_round_s)
        self.clock = clock
        self.require_release = bool(require_release)
        self.loaded_fields = loaded_fields  # extra fields of rl_sampler_policy_loaded
        self.ready = ready  # False: the engine may not generate (failed post-load check)
        self._not_loaded: tuple[Any, ...] | None = None  # (seq, reason, fields) last reported
        self.live_poll_s = self.poll_s if live_poll_s is None else float(live_poll_s)
        self.unchecked_in_flight = False  # this round generated on weights that failed a check
        # while it waits (no trainer reading: backlog at the cap, or no release yet) the
        # sampler writes rl_sampler_waiting every heartbeat_s: its event tape keeps moving,
        # so the NC pool's stall check (elastic stall_s) never takes a paused sampler for a
        # crashed one and restarts or replaces it while the trainer is away
        self.heartbeat_s = WAIT_HEARTBEAT_S
        self.read_failures = 0  # store polls of the loop that failed
        self.max_damaged_latest_reads = DEFAULT_MAX_DAMAGED_LATEST_READS
        self._damaged_latest = 0  # in a row
        self._read_failing: dict[str, tuple[float, int]] = {}  # what -> (since, failures) of an outage
        self.max_read_outage_s: float | None = DEFAULT_MAX_READ_OUTAGE_S  # None: no bound
        self._latest_unread_since: float | None = None

    def _outage_bound(self, what: str, since: float, failures: int, error: BaseException) -> None:
        """Raise when ``what`` has failed for longer than ``max_read_outage_s``."""
        bound = getattr(self, "max_read_outage_s", DEFAULT_MAX_READ_OUTAGE_S)
        if bound is not None and self.clock() - since > bound:
            raise StoreReadOutage(f"store read {what!r} failed {failures} times over "
                                  f"{self.clock() - since:.0f} s (max_read_outage_s {bound:.0f}): "
                                  f"{type(error).__name__}: {error}"[:500]) from error

    def _read(self, what: str, read: Callable[[], Any], default: Any = None) -> Any:
        """``read()``, a poll of the store (module doc): ``default`` when it fails.
        A verdict about the buffer's content (``SampleBufferError``) is not a failed
        read and is raised."""
        try:
            value = read()
        except SampleBufferError:
            raise
        except Exception as error:  # noqa: BLE001 - unknown this time: the next poll reads again
            if is_fatal_store_error(error):  # refused credentials, no such bucket: no outage
                raise
            self.read_failures += 1
            since, failures = self._read_failing.get(what, (self.clock(), 0))
            self._read_failing[what] = (since, failures + 1)
            if not failures:
                self.emit(READ_FAILED_EVENT, what=what, error=f"{type(error).__name__}: {error}"[:300])
            self._outage_bound(what, since, failures + 1, error)
            return default
        if what in self._read_failing:
            since, failures = self._read_failing.pop(what)
            self.emit(READ_OK_EVENT, what=what, failed_reads=failures,
                      outage_s=round(self.clock() - since, 1))
        return value

    def backlog(self) -> int:
        """Groups in the buffer after the trainer's read position of its last release
        (0 before any release: no trainer has read anything yet). A sampler that
        writes its own shard (``buffer.shard``) counts its shard only, from the
        trainer's read position in that shard (``buffer_shards``). The trainer's read
        heartbeat (``sample-buffer/read-position.json``) refines it: see
        :func:`heartbeat_read`; a missing, broken or foreign heartbeat leaves the
        release's position."""
        self.buffer.refresh()
        pending = getattr(self.buffer, "pending_groups", None)
        if pending is not None:  # the relay: groups sent and not yet taken by the trainer
            return max(0, int(pending()))
        seen = self.follower.last_seen
        shard = getattr(self.buffer, "shard", None)
        if seen is None:
            read = 0
        elif shard is not None:
            read = int((seen.buffer_shards or {}).get(shard, 0))
        else:
            read = 0 if seen.buffer_selected is None else int(seen.buffer_selected)
        root = buffer_dir(self.buffer)
        beat = None if root is None else read_read_position(root)
        if beat is not None:
            read = heartbeat_read(beat, seen, shard, read)
        # synced trainers' lanes: count from the furthest one, so a stopped trainer's
        # unread lane cannot hold the cap (merged_release.fastest_lane_read)
        lanes = fastest_lane_read(self.buffer, shard)
        if lanes is not None:
            read = max(read, lanes)
        groups_from = getattr(self.buffer, "groups_from", None)
        if shard is not None and callable(groups_from):
            # ``read`` is a shard position; the shard's ownership records hold positions too
            return int(groups_from(read))
        return max(0, len(self.buffer) - read)

    def _load_newest(self, *, in_flight: bool = False) -> None:
        try:
            loaded = self.follower.refresh()
        except ReleaseNotLoaded as error:
            unread = error.reason == LATEST_UNREAD
            if not unread:
                self._latest_unread_since = None
            self.read_failures += int(unread)
            self._damaged_latest = self._damaged_latest + 1 if unread and error.fields.get("damaged") else 0
            if self._damaged_latest >= self.max_damaged_latest_reads:
                raise
            if unread:  # the served weights stay, but not for ever on a store that never answers
                if getattr(self, "_latest_unread_since", None) is None:
                    self._latest_unread_since = self.clock()
                self._outage_bound("latest", self._latest_unread_since, self.follower.latest_read_failures,
                                   error)
            seen = self.follower.last_seen
            mark = (None if seen is None else seen.seq, error.reason, tuple(sorted(error.fields.items())))
            if mark != self._not_loaded:  # one event per release, cause and fields, not per poll
                self._not_loaded = mark
                self.emit("rl_sampler_release_not_loaded", seq=mark[0], reason=error.reason,
                          detail=str(error)[:300], policy_version=self.follower.version,
                          **error.fields)
            return
        self._damaged_latest = 0
        self._latest_unread_since = None
        if self._not_loaded is not None and self._not_loaded[1] == LATEST_UNREAD:
            self._not_loaded = None  # read again: a later outage is reported anew
        if loaded is not None:
            self.emit("rl_sampler_policy_loaded", seq=loaded.seq, policy_version=loaded.version,
                      policy_hash=loaded.policy_hash, buffer_selected=loaded.buffer_selected,
                      weights=loaded.weights, sha256=loaded.sha256,
                      released_at=loaded.released_at, seen_at=self.follower.seen_at,
                      follower_timings=self.follower.last_timings,
                      **({"in_flight": True} if in_flight else {}),
                      **({"forced": True} if getattr(self.follower, "last_forced", False) else {}),
                      **({} if getattr(self.follower, "last_deferred_s", None) is None
                         else {"deferred_s": self.follower.last_deferred_s}),
                      **self.loaded_fields())

    @contextmanager
    def _live_refresh(self) -> Iterator[None]:
        """Load new releases while the round generates (module doc)."""
        self.unchecked_in_flight = False
        if not self.follower.live_update:
            yield
            return
        done = threading.Event()
        errors: list[BaseException] = []

        def poll() -> None:
            while not done.wait(self.live_poll_s):
                try:
                    self._load_newest(in_flight=True)
                except BaseException as error:  # noqa: BLE001 - re-raised on the loop thread
                    errors.append(error)
                    return
                if not self.ready():
                    self.unchecked_in_flight = True

        thread = threading.Thread(target=poll, name="sampler-live-refresh", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()
        if errors:
            raise errors[0]

    def _admit(self, rollout_id: int) -> bool:
        """True when the round may start; False when asked to stop while waiting."""
        waiting_since = None
        waiting_policy_since = None
        unread_since = None
        beat = [time.monotonic()]

        def heartbeat(reason: str, since: float, **fields: Any) -> None:
            now = time.monotonic()
            if now - beat[0] >= self.heartbeat_s:
                beat[0] = now
                self.emit("rl_sampler_waiting", rollout_id=rollout_id, reason=reason,
                          waited_s=round(now - since, 1), policy_version=self.follower.version, **fields)

        while True:
            self._load_newest()
            behind = getattr(self.follower.releases, "behind", None)  # spot samplers
            on_old_weights = behind is not None and behind(
                None if self.follower.release is None else self.follower.version)
            if not self.ready() or (self.require_release and self.follower.release is None) or on_old_weights:
                if waiting_policy_since is None:
                    waiting_policy_since = time.monotonic()
                    self.emit("rl_sampler_waiting_policy", rollout_id=rollout_id)
                heartbeat("policy", waiting_policy_since)
                if self.should_stop():
                    return False
                self.sleep(self.poll_s)
                continue
            if waiting_policy_since is not None:
                self.emit("rl_sampler_policy_ready", rollout_id=rollout_id,
                          waited_s=round(time.monotonic() - waiting_policy_since, 3),
                          policy_version=self.follower.version)
                waiting_policy_since = None
            backlog = self._read("backlog", self.backlog)
            if backlog is None:  # the buffer was not read: no round starts on an unknown backlog
                if unread_since is None:
                    unread_since = time.monotonic()
                heartbeat("store_unread", unread_since)
                if self.should_stop():
                    return False
                self.sleep(self.poll_s)
                continue
            if backlog < self.max_backlog_groups:
                if waiting_since is not None:
                    self.emit("rl_sampler_admitted", rollout_id=rollout_id, backlog_groups=backlog,
                              waited_s=round(time.monotonic() - waiting_since, 3),
                              policy_version=self.follower.version)
                return True
            if waiting_since is None:
                waiting_since = time.monotonic()
                self.emit("rl_sampler_backlog_full", rollout_id=rollout_id, backlog_groups=backlog,
                          max_backlog_groups=self.max_backlog_groups,
                          policy_version=self.follower.version)
            heartbeat("backlog_full", waiting_since, backlog_groups=backlog)
            if self.should_stop():
                return False
            self.sleep(self.poll_s)

    def _throttle(self, started: float, *, last: bool) -> None:
        """Keep ``min_round_s`` between two round starts (not after the last round)."""
        if self.min_round_s is None or last:
            return
        rest = self.min_round_s - (self.clock() - started)
        if rest > 0:
            self.emit("rl_sampler_throttle", sleep_s=round(rest, 3))
            self.sleep(rest)

    def run(self, start: int = 0) -> int:
        """Generate rounds ``start .. num_rollouts - 1`` (no end for None: until
        ``should_stop``); returns the rounds generated."""
        done = self.rounds_done = 0
        rollout_ids = (itertools.count(int(start)) if self.num_rollouts is None
                       else range(int(start), self.num_rollouts))
        for rollout_id in rollout_ids:
            if self.should_stop() or not self._admit(rollout_id):
                self.emit("rl_sampler_stopped", rollout_id=rollout_id, rounds=done)
                return done
            before = len(self.buffer)
            started = self.clock()
            version, lag = self.follower.version, self.follower.version_lag()
            try:
                with self.follower.generating(), self._live_refresh():
                    self.generate(rollout_id)
            except DispenseEnded as error:
                if isinstance(error, DispenseFencedEnd):  # fenced: the data did not end
                    raise
                # the task dispenser settled every task: a normal end, not a failure
                print(f"[engine-sampler] data ended: {error}", flush=True)
                self.emit("rl_sampler_data_ended", rollout_id=rollout_id, rounds=done, detail=str(error)[:200])
                break
            self._read("buffer", self.buffer.refresh)  # the round's groups are written; this only counts them
            generate_s = self.clock() - started
            self.emit("rl_sampler_round", rollout_id=rollout_id, policy_version=version,
                      version_lag=lag, releases=self.follower.counters.to_dict(),
                      groups=len(self.buffer) - before, buffer_end=len(self.buffer),
                      generate_s=round(generate_s, 3))
            done += 1
            self.rounds_done = done
            self._throttle(started, last=rollout_id == (self.num_rollouts or 0) - 1)
        flush = getattr(self.generate, "flush", None)
        if callable(flush):  # groups of a streaming round still in flight
            with self.follower.generating(), self._live_refresh():
                flush()
            self._read("buffer", self.buffer.refresh)
        self.buffer.mark_data_end()
        self.emit("rl_sampler_data_end", rounds=done, buffer_end=len(self.buffer))
        return done
