"""The sampler island's loop (rl-trainer-start-stop L4.6).

A sampler island generates rounds and writes their groups to the durable sample
buffer (the engine's rollout function does the write). Before each round it:

1. loads the newest policy release, when there is one (:class:`PolicyFollower`);
2. waits while the backlog (groups in the buffer the trainer has not read, as of
   its last release) is at the cap. A stopped trainer releases nothing, so the
   sampler keeps the last version and stops at the cap; a newer release (the
   trainer runs again) lifts the wait.

After the last round it marks the data end, so the trainer stops after reading
the rest. The loop is engine-neutral: ``generate(rollout_id)`` runs one round.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .policy_release import PolicyFollower

DEFAULT_POLL_S = 5.0


class SamplerLoop:
    def __init__(self, *, follower: PolicyFollower, buffer: Any,
                 generate: Callable[[int], None], num_rollouts: int,
                 max_backlog_groups: int, emit: Callable[..., None],
                 poll_s: float = DEFAULT_POLL_S, sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False) -> None:
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

    def backlog(self) -> int:
        """Groups in the buffer after the trainer's read position of its last release
        (0 before any release: no trainer has read anything yet)."""
        self.buffer.refresh()
        seen = self.follower.last_seen
        read = 0 if seen is None or seen.buffer_selected is None else int(seen.buffer_selected)
        return max(0, len(self.buffer) - read)

    def _load_newest(self) -> None:
        loaded = self.follower.refresh()
        if loaded is not None:
            self.emit("rl_sampler_policy_loaded", policy_version=loaded.version,
                      policy_hash=loaded.policy_hash, buffer_selected=loaded.buffer_selected)

    def _admit(self, rollout_id: int) -> bool:
        """True when the round may start; False when asked to stop while waiting."""
        waiting_since = None
        while True:
            self._load_newest()
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

    def run(self, start: int = 0) -> int:
        """Generate rounds ``start .. num_rollouts - 1``; returns the rounds generated."""
        done = 0
        for rollout_id in range(int(start), self.num_rollouts):
            if self.should_stop() or not self._admit(rollout_id):
                self.emit("rl_sampler_stopped", rollout_id=rollout_id, rounds=done)
                return done
            before = len(self.buffer)
            self.generate(rollout_id)
            self.buffer.refresh()
            self.emit("rl_sampler_round", rollout_id=rollout_id,
                      policy_version=self.follower.version,
                      groups=len(self.buffer) - before, buffer_end=len(self.buffer))
            done += 1
        self.buffer.mark_data_end()
        self.emit("rl_sampler_data_end", rounds=done, buffer_end=len(self.buffer))
        return done
