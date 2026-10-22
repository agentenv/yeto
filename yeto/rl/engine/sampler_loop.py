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
a failed post-load check). The loop writes ``rl_sampler_release_not_loaded`` and
goes on; with ``ready`` (the engine sampler: False while the engine holds
weights that failed a check) it waits instead of generating, so no sample of
unchecked weights reaches the buffer.

Staleness: ``rl_sampler_round`` carries the version the round generated with,
``version_lag`` (trainer versions between the newest release seen and that
version, at the round start) and the follower's release counters
(:class:`~.policy_release.FollowerCounters`). The round runs inside
``follower.generating()``: a refresh during it loads nothing unless the engines
update their weights mid-generation.

After the last round it marks the data end, so the trainer stops after reading
the rest. The loop is engine-neutral: ``generate(rollout_id)`` runs one round.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .policy_release import PolicyFollower, ReleaseNotLoaded

DEFAULT_POLL_S = 5.0


class SamplerLoop:
    def __init__(self, *, follower: PolicyFollower, buffer: Any,
                 generate: Callable[[int], None], num_rollouts: int,
                 max_backlog_groups: int, emit: Callable[..., None],
                 poll_s: float = DEFAULT_POLL_S, sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False,
                 min_round_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 require_release: bool = False,
                 loaded_fields: Callable[[], dict[str, Any]] = dict,
                 ready: Callable[[], bool] = lambda: True) -> None:
        if int(max_backlog_groups) < 1:
            raise ValueError("max_backlog_groups must be >= 1")
        self.follower = follower
        self.buffer = buffer
        self.generate = generate
        self.num_rollouts = int(num_rollouts)
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
        self._not_loaded: tuple[int, str] | None = None  # (seq, reason) last reported

    def backlog(self) -> int:
        """Groups in the buffer after the trainer's read position of its last release
        (0 before any release: no trainer has read anything yet). A sampler that
        writes its own shard (``buffer.shard``) counts its shard only, from the
        trainer's read position in that shard (``buffer_shards``)."""
        self.buffer.refresh()
        seen = self.follower.last_seen
        shard = getattr(self.buffer, "shard", None)
        if seen is None:
            read = 0
        elif shard is not None:
            read = int((seen.buffer_shards or {}).get(shard, 0))
        else:
            read = 0 if seen.buffer_selected is None else int(seen.buffer_selected)
        return max(0, len(self.buffer) - read)

    def _load_newest(self) -> None:
        try:
            loaded = self.follower.refresh()
        except ReleaseNotLoaded as error:
            seen = self.follower.last_seen
            mark = (None if seen is None else seen.seq, error.reason)
            if mark != self._not_loaded:  # one event per release and cause, not per poll
                self._not_loaded = mark
                self.emit("rl_sampler_release_not_loaded", seq=mark[0], reason=error.reason,
                          detail=str(error)[:300], policy_version=self.follower.version)
            return
        if loaded is not None:
            self.emit("rl_sampler_policy_loaded", seq=loaded.seq, policy_version=loaded.version,
                      policy_hash=loaded.policy_hash, buffer_selected=loaded.buffer_selected,
                      weights=loaded.weights, sha256=loaded.sha256, **self.loaded_fields())

    def _admit(self, rollout_id: int) -> bool:
        """True when the round may start; False when asked to stop while waiting."""
        waiting_since = None
        waiting_policy_since = None
        while True:
            self._load_newest()
            if not self.ready() or (self.require_release and self.follower.release is None):
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
        """Generate rounds ``start .. num_rollouts - 1``; returns the rounds generated."""
        done = 0
        for rollout_id in range(int(start), self.num_rollouts):
            if self.should_stop() or not self._admit(rollout_id):
                self.emit("rl_sampler_stopped", rollout_id=rollout_id, rounds=done)
                return done
            before = len(self.buffer)
            started = self.clock()
            version, lag = self.follower.version, self.follower.version_lag()
            with self.follower.generating():
                self.generate(rollout_id)
            self.buffer.refresh()
            self.emit("rl_sampler_round", rollout_id=rollout_id, policy_version=version,
                      version_lag=lag, releases=self.follower.counters.to_dict(),
                      groups=len(self.buffer) - before, buffer_end=len(self.buffer))
            done += 1
            self._throttle(started, last=rollout_id == self.num_rollouts - 1)
        self.buffer.mark_data_end()
        self.emit("rl_sampler_data_end", rounds=done, buffer_end=len(self.buffer))
        return done
