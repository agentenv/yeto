"""The sampler island's loop (rl-trainer-start-stop L4.6).

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

``rl_sampler_policy_loaded`` carries the latency fields of S26 E5: the trainer's
``released_at``, the follower's ``seen_at`` (both wall clock, of two hosts) and
``follower_timings`` (:attr:`~.policy_release.PolicyFollower.last_timings`).

After the last round it marks the data end, so the trainer stops after reading
the rest. ``num_rollouts`` None: no last round; the loop generates until
``should_stop`` (a full-parameter synced group's sampler stops when its trainers
ended, ``sample_roles`` module doc) and marks no data end. The loop is
engine-neutral: ``generate(rollout_id)`` runs one round.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .policy_release import PolicyFollower, ReleaseNotLoaded

DEFAULT_POLL_S = 5.0


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

    def backlog(self) -> int:
        """Groups in the buffer after the trainer's read position of its last release
        (0 before any release: no trainer has read anything yet). A sampler that
        writes its own shard (``buffer.shard``) counts its shard only, from the
        trainer's read position in that shard (``buffer_shards``)."""
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
        return max(0, len(self.buffer) - read)

    def _load_newest(self, *, in_flight: bool = False) -> None:
        try:
            loaded = self.follower.refresh()
        except ReleaseNotLoaded as error:
            seen = self.follower.last_seen
            mark = (None if seen is None else seen.seq, error.reason, tuple(sorted(error.fields.items())))
            if mark != self._not_loaded:  # one event per release, cause and fields, not per poll
                self._not_loaded = mark
                self.emit("rl_sampler_release_not_loaded", seq=mark[0], reason=error.reason,
                          detail=str(error)[:300], policy_version=self.follower.version,
                          **error.fields)
            return
        if loaded is not None:
            self.emit("rl_sampler_policy_loaded", seq=loaded.seq, policy_version=loaded.version,
                      policy_hash=loaded.policy_hash, buffer_selected=loaded.buffer_selected,
                      weights=loaded.weights, sha256=loaded.sha256,
                      released_at=loaded.released_at, seen_at=self.follower.seen_at,
                      follower_timings=self.follower.last_timings,
                      **({"in_flight": True} if in_flight else {}), **self.loaded_fields())

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
        while True:
            self._load_newest()
            behind = getattr(self.follower.releases, "behind", None)  # S29 D9 (spot samplers)
            on_old_weights = behind is not None and behind(
                None if self.follower.release is None else self.follower.version)
            if not self.ready() or (self.require_release and self.follower.release is None) or on_old_weights:
                if waiting_policy_since is None:
                    waiting_policy_since = time.monotonic()
                    self.emit("rl_sampler_waiting_policy", rollout_id=rollout_id)
                if self.should_stop():
                    return False
                self.sleep(self.poll_s)
                continue
            if waiting_policy_since is not None:
                self.emit("rl_sampler_policy_ready", rollout_id=rollout_id,
                          waited_s=round(time.monotonic() - waiting_policy_since, 3),
                          policy_version=self.follower.version)
                waiting_policy_since = None
            backlog = self.backlog()
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
        done = 0
        rollout_ids = (itertools.count(int(start)) if self.num_rollouts is None
                       else range(int(start), self.num_rollouts))
        for rollout_id in rollout_ids:
            if self.should_stop() or not self._admit(rollout_id):
                self.emit("rl_sampler_stopped", rollout_id=rollout_id, rounds=done)
                return done
            before = len(self.buffer)
            started = self.clock()
            version, lag = self.follower.version, self.follower.version_lag()
            with self.follower.generating(), self._live_refresh():
                self.generate(rollout_id)
            self.buffer.refresh()
            generate_s = self.clock() - started
            self.emit("rl_sampler_round", rollout_id=rollout_id, policy_version=version,
                      version_lag=lag, releases=self.follower.counters.to_dict(),
                      groups=len(self.buffer) - before, buffer_end=len(self.buffer),
                      generate_s=round(generate_s, 3))
            done += 1
            self._throttle(started, last=rollout_id == (self.num_rollouts or 0) - 1)
        self.buffer.mark_data_end()
        self.emit("rl_sampler_data_end", rounds=done, buffer_end=len(self.buffer))
        return done
