"""One full-parameter synchronization for every algorithm (P0-D8).

An algorithm declares its trainable roles (GRPO: ``actor``; SAO/PPO:
``actor`` and ``critic``).  Each role owns one :class:`RoleStream`: one syncer
process, one chunked full-parameter cut, staged fragment broadcasts and
retained answers to syncer PULL permits.  :class:`RolePairing` is the only
coordinator above the streams.  Its rules hold for any number of roles, so
one role needs no special case:

1. every role shares one schedule (learner, roster, fragments, budget, H,
   pipeline depth, training contract, syncer profile);
2. a logical fragment submission is pushed only when every role has a
   candidate for it (paired retention);
3. a fragment broadcast is applied only when every role has staged it
   (paired apply), and all roles must then hold the same fragment versions;
4. when one role fails, every role is closed (fail-stop) with one reason;
5. a PULL seen by some roles but not by the others fails after the roles'
   wait timeout.

The dense strict sweep is a restricted profile of the same rules
(:attr:`FullParameterSyncProfile.sweep`): one role, H=1, every fragment of
one local step in flight together, and broadcasts applied only as a complete
sweep.

The asynchronous full-parameter rounds (S27 P2) are the
streaming profile with :attr:`FullParameterSyncProfile.partial_quorum`: the
syncer may close a fragment round with a quorum K below the island count.
Rule 3 then also covers broadcasts this island did not contribute to (late,
absent, or a push the syncer refused): every role overwrites its fragment with
the global value (alpha = 0) in version order, and a retained submission for
that round is released.  A re-JOIN catch-up (or any broadcast after which
an older version of that fragment can no longer arrive) jumps the fragment
straight to the delivered version (also alpha = 0).

Policy versions under the asynchronous rounds:

* the island's *policy version* stays its local round counter: boundary r
  publishes version r + 1, and the published hash covers every fragment the
  island overwrote at that boundary.  Every sample is stamped with the local
  version (and hash) of the policy that generated it, so the behaviour
  log-probabilities that truncated importance sampling divides by always come
  from the exact weights that sampled, whether or not a merged broadcast
  overwrote a fragment between two local versions;
* ``--rl-max-policy-age N`` counts local publications: a sample may lag at
  most N local versions.  One local version may contain merged broadcasts as
  well as the island's own step; the round event reports how many fragments
  the island took from the syncer since its previous boundary
  (``sync/overwritten_fragments``);
* the *merged version* (:func:`merged_version`) is the number of merges every
  fragment has received: one per outer round of all P fragments, the same
  unit as the outer version of the LoRA syncer.  The session exposes it as
  ``outer_version`` (and sets ``IslandDriver.outer_version``), so synced
  trainers release it to sampler islands and gate buffered samples on it.

This module is engine-neutral; it never imports an engine adapter.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, Self

from yeto.protocol import PartialMessageGenerationLost
from yeto.rl.core import parse_policy_snapshot_token
from yeto.rl.pool_client import LeaveReason
from yeto.rl.trajectory_evidence import bind_round_policy
from yeto.wan_streams import upload_window

from .bridges import RoundCutHooks
from .driver import SyncBoundary, SyncStart
from .sync_timing import SyncTiming, unix_now, upload_fields

if TYPE_CHECKING:
    from .driver import IslandDriver

LOCKSTEP_FIELDS = (
    "learner_id",
    "learner_generation",
    "learner_generations",
    "total_fragment_steps",
    "expected_fragments",
    "local_horizon",
    "pipeline_depth",
    "training_contract_hash",
    "syncer_profile_hash",
)

LogicalKey = tuple[int, int, int]  # (global step, fragment id, base version)


def fragment_rounds(fragment_id: int, version: int, fragments: int) -> int:
    """Merged rounds behind ``version`` of ``fragment_id``: its versions are the
    global steps ``fragment_id + 1``, ``fragment_id + 1 + P``, ..."""

    return 0 if version <= 0 else (version - 1 - fragment_id) // fragments + 1


def merged_version(versions: Sequence[int]) -> int:
    """Merges every fragment of ``versions`` (one per fragment) has received."""

    fragments = len(versions)
    return min(fragment_rounds(f, v, fragments) for f, v in enumerate(versions)) if versions else 0


class RoleStream(Protocol):
    """One trainable role's streaming session (implemented per engine)."""

    config: Any
    terminal_submitted: bool
    final_acknowledged: bool

    @property
    def fragment_versions(self) -> tuple[int, ...]: ...
    @property
    def content_hash(self) -> str: ...
    @property
    def finalizing(self) -> bool: ...

    async def initialize(self, group: Any) -> None: ...
    async def complete_local_round(self, *, rollout_id: int, trajectories: Any) -> Any: ...
    def resident_state(self, policy_version: int) -> Any: ...
    def ready_submission_candidates(self) -> tuple[Any, ...]: ...
    def retain_candidate(self, candidate: Any) -> None: ...
    def release_candidate(self, candidate: Any) -> None: ...
    def push_candidate(self, candidate: Any) -> Any: ...
    async def stage_available_broadcasts(self) -> tuple[tuple[int, int], ...]: ...
    @property
    def staged_fragment_versions(self) -> tuple[tuple[int, int], ...]: ...
    def has_staged_broadcast(self, fragment_id: int, version: int) -> bool: ...
    async def apply_staged_broadcast(self, fragment_id: int, version: int) -> tuple[int, ...]: ...
    # Asynchronous rounds (profile.partial_quorum) only: monotonic_broadcasts
    #   (the stream never delivers an older version of a fragment after a
    #   newer one) lets apply_staged_broadcast(..., catch_up=True) overwrite a
    #   fragment with a version that skips older ones (re-JOIN catch-up).
    # S26 E3 sync/train overlap (profile.overlap) only:
    #   complete_local_round(..., hold_submission=True) keeps the round base and
    #   the previous submission; apply_staged_broadcast(..., rebase=True) applies
    #   G + (current - base) with G as the anchor; finish_rebase() makes the
    #   rebased cut the submission; mark_round_base() pins the next round base.
    # Partial quorum only: apply_staged_broadcast(..., catch_up=True) jumps a
    # fragment to a version whose base it never held (a re-JOIN catch-up).
    # ``monotonic_broadcasts`` (optional, default False): the stream never
    # delivers a version of a fragment older than one it already delivered,
    # so a staged jump with no older staged version is final.
    async def wait_and_apply_final(self) -> None: ...
    def acknowledge_final(self) -> None: ...
    def discard_detached_submission_cut(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class FullParameterSyncProfile:
    """The schedule every role of one island shares."""

    roles: tuple[str, ...]
    total_fragment_steps: int
    fragments: int
    local_horizon: int
    pipeline_depth: int
    wait_timeout: float
    poll_seconds: float
    sweep: bool = False
    # S26 E3: train round k+1 while sweep k syncs; rebase at the next boundary.
    overlap: bool = False
    # S27 P2: the syncer closes rounds with quorum K <= islands, so a fragment
    # broadcast may arrive without this island's submission (overwrite it).
    partial_quorum: bool = False
    # S27 G-AG lead cap K (streaming only; None = off, the old behaviour): an
    # island trains at most K - 1 local rounds on top of an unmerged push of
    # its own; K = 1 starts no round while one is unmerged (RolePairing.wait_for_lead).
    max_local_rounds_ahead: int | None = None

    def __post_init__(self) -> None:
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("full-parameter sync needs distinct trainable roles")
        if self.roles[0] != "actor":
            raise ValueError("the actor is the published role and must come first")
        if self.sweep and (self.roles != ("actor",) or self.local_horizon != 1):
            raise ValueError("a dense strict sweep has one actor role and H=1")
        if self.overlap and not self.sweep:
            raise ValueError("sync/train overlap needs the dense strict sweep")
        if self.partial_quorum and self.sweep:
            raise ValueError("a partial quorum needs the streaming profile, not the strict sweep")
        if self.max_local_rounds_ahead is not None:
            if type(self.max_local_rounds_ahead) is not int or self.max_local_rounds_ahead < 1:
                raise ValueError("max_local_rounds_ahead must be an integer >= 1")
            if self.sweep or self.overlap:
                raise ValueError("a local-round lead cap needs the streaming profile (no sweep, no overlap)")

    @property
    def total_rounds(self) -> int:
        """Local rounds of a sweep profile (one sweep per round)."""
        return self.total_fragment_steps // self.fragments

    @property
    def effective_pipeline_depth(self) -> int:
        if self.sweep:
            return self.fragments
        return min(self.pipeline_depth, self.fragments)

    @classmethod
    def from_configs(
        cls, configs: Mapping[str, Any], *, sweep: bool = False,
        max_local_rounds_ahead: int | None = None,
    ) -> FullParameterSyncProfile:
        """Profile of role-stream configs that pass :func:`require_lockstep`."""

        require_lockstep(tuple(configs.values()))
        first = next(iter(configs.values()))
        return cls(
            roles=tuple(configs),
            total_fragment_steps=first.total_fragment_steps,
            fragments=first.expected_fragments,
            local_horizon=first.local_horizon,
            pipeline_depth=first.pipeline_depth,
            wait_timeout=min(config.wait_timeout for config in configs.values()),
            poll_seconds=min(config.poll_seconds for config in configs.values()),
            sweep=sweep,
            overlap=bool(getattr(first, "overlap", False)),
            partial_quorum=bool(getattr(first, "partial_quorum", False)),
            max_local_rounds_ahead=max_local_rounds_ahead,
        )


def require_lockstep(configs: Sequence[Any]) -> None:
    """Rule 1: every role shares one schedule and owns its own syncer."""

    first = configs[0]
    if len({config.syncer_addr for config in configs}) != len(configs):
        raise ValueError("full-parameter roles require distinct syncer ports")
    for config in configs[1:]:
        if any(getattr(config, name) != getattr(first, name) for name in LOCKSTEP_FIELDS) or (
            getattr(config, "partial_quorum", False) != getattr(first, "partial_quorum", False)
        ):
            raise ValueError("full-parameter role streams must use one lockstep schedule")


@dataclass
class PairedSubmission:
    """Retained per-role operands of one logical submission in flight."""

    candidates: dict[str, Any]
    submissions: dict[str, Any] = field(default_factory=dict)
    retained_at: float = field(default_factory=time.monotonic)
    # The island's local round (RolePairing.local_round) when it retained the
    # pair: the round whose cut the pair pushes (lead cap).
    local_round: int = 0
    # Background push (S27 G-AP): the worker job pushing this pair, until
    # the pairing harvests it.
    job: _PushJob | None = None

    @property
    def logical_key(self) -> LogicalKey:
        return next(iter(self.candidates.values())).logical_key

    @property
    def complete(self) -> bool:
        return set(self.submissions) == set(self.candidates)


@dataclass(frozen=True)
class RoleProgress:
    """What one coordinator step did for one role."""

    submissions: tuple[Any, ...] = ()
    applied_fragments: tuple[int, ...] = ()

    def extend(self, other: RoleProgress) -> RoleProgress:
        return RoleProgress(
            self.submissions + other.submissions,
            tuple(dict.fromkeys(self.applied_fragments + other.applied_fragments)),
        )


class RolePairing:
    """The pairing rule over every trainable role of one island."""

    def __init__(self, streams: Mapping[str, RoleStream], profile: FullParameterSyncProfile) -> None:
        if tuple(streams) != profile.roles:
            raise ValueError("role streams differ from the sync profile roles")
        self.streams = dict(streams)
        self.profile = profile
        self.inflight: dict[LogicalKey, PairedSubmission] = {}
        self.journal: list[dict[str, Any]] = []
        # Fragments whose last round this island can submit (or that closed
        # without it); the island stops training once it covers all of them.
        self._final_rounds_ready: set[int] = set()
        self.terminal_seen = False
        self.failure: BaseException | None = None
        self._pull_skew_since: dict[LogicalKey, float] = {}
        # Partial quorum: broadcasts applied without this island's submission
        # (catch-ups included), and the rounds catch-up jumps skipped.
        self.overwritten_rounds = 0
        self.caught_up_rounds = 0
        self.caught_up_fragments = 0  # catch-up jumps (one per fragment and jump)
        # Terminal phase: last rounds closed without a trainer apply (_retires).
        self.retired_rounds = 0
        # Lead cap: local rounds this island completed (set by the session at
        # each boundary) and what its waits did.
        self.local_round = 0
        self.lead_waits = 0
        self.lead_wait_seconds = 0.0
        self.lead_wait_escapes = 0
        # Background pushes (S27 G-AP), decided at the first submit (a role
        # knows whether it can encode ahead once it is initialized); None
        # keeps every push on this thread.
        self._pusher: _BackgroundPusher | None = None
        self._pusher_decided = False
        # Pairs released while their push job still ran: their operands stay
        # pinned until the job is harvested.
        self._releasing: list[PairedSubmission] = []
        # Submissions harvested since the last submit_ready (its return value).
        self._harvested: dict[str, list[Any]] = {role: [] for role in self.streams}

    @property
    def pushes_in_background(self) -> bool:
        """Whether pushes run on worker threads (:class:`_BackgroundPusher`)."""
        return self._pusher is not None

    @property
    def push_jobs_running(self) -> int:
        """Push jobs submitted and not yet harvested."""
        return sum(pair.job is not None for pair in (*self.inflight.values(), *self._releasing))

    def push_job_counters(self) -> dict[str, float]:
        """Cumulative background push counters (empty without background pushes)."""
        pusher = self._pusher
        if pusher is None:
            return {}
        return {"submitted": pusher.submitted, "harvested": pusher.harvested,
                "busy_seconds": pusher.busy_seconds, "running": self.push_jobs_running}

    # -- fail-stop ---------------------------------------------------------

    def fail_stop(self, error: BaseException) -> None:
        """Rule 4: close every role once; keep the first failure."""

        if self.failure is None:
            self.failure = error
        for note in self.close_all():
            error.add_note(note)

    def close_all(self) -> list[str]:
        """Close every role even when one close raises; one note per failure.

        Background pushes not started yet are cancelled; a running one stops
        at its next enqueue once its client is closed (the enqueue rechecks
        every 0.5 s), and is awaited for at most PUSH_SHUTDOWN_WAIT_S."""

        running = self._pusher.stop() if self._pusher is not None else []
        notes = []
        for role, stream in self.streams.items():
            try:
                stream.close()
            except Exception as close_error:  # noqa: BLE001 - every role must close
                notes.append(f"{role} close failed: {type(close_error).__name__}: {close_error}")
        if running:
            _done, pending = wait_for_futures(running, timeout=PUSH_SHUTDOWN_WAIT_S)
            if pending:
                notes.append(f"{len(pending)} background push(es) still running at close")
        return notes

    async def gather_roles(self, make: Any) -> dict[str, Any]:
        """Run ``make(role, stream)`` for every role; fail-stop all on any error."""

        tasks = {role: asyncio.create_task(make(role, stream)) for role, stream in self.streams.items()}
        try:
            results = await asyncio.gather(*tasks.values())
        except BaseException as error:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            self.fail_stop(error)
            raise
        return dict(zip(tasks, results, strict=True))

    @property
    def finalizing(self) -> bool:
        return all(stream.finalizing for stream in self.streams.values())

    @property
    def any_finalizing(self) -> bool:
        return self.terminal_seen or any(stream.finalizing for stream in self.streams.values())

    def fragment_versions(self) -> tuple[int, ...]:
        """Rule 3: every role holds the same committed fragment cut."""

        versions = {role: stream.fragment_versions for role, stream in self.streams.items()}
        if len(set(versions.values())) != 1:
            error = RuntimeError(f"role fragment cuts diverged: {versions}")
            self.fail_stop(error)
            raise error
        return next(iter(versions.values()))

    # -- submission (rules 2 and 5) -----------------------------------------

    def submit_ready(self) -> dict[str, RoleProgress]:
        """Nonblockingly fill the paired pipeline from currently visible PULLs."""

        if self.failure is not None:
            raise RuntimeError("full-parameter pairing is fail-stopped") from self.failure
        produced: dict[str, list[Any]] = {role: [] for role in self.streams}
        try:
            if not self._pusher_decided:
                self._pusher = _BackgroundPusher.for_streams(self.streams, self.profile)
                self._pusher_decided = True
            self._harvest_pushes()
            for role, submissions in self._harvested.items():
                produced[role].extend(submissions)
                submissions.clear()
            by_role = {
                role: _candidate_map(stream.ready_submission_candidates(), role=role)
                for role, stream in self.streams.items()
            }
            now = time.monotonic()
            self._note_final_rounds(by_role)
            for key, pair in tuple(self.inflight.items()):
                observed = {role: by_role[role].pop(key, None) for role in self.streams}
                if pair.job is not None:
                    # A worker is pushing it: the PULL stays pending until the
                    # push is harvested, and is no replay.
                    continue
                self._push_once(pair, observed, produced)
            capacity = self.profile.effective_pipeline_depth - len(self.inflight)
            shared = sorted(set.intersection(*(set(keys) for keys in by_role.values())))
            if self.profile.partial_quorum:
                # A round whose broadcast is staged or applied closed without
                # this island: pushing would only upload a refused delta.
                shared = [key for key in shared if not self._round_closed(key[1], key[0])]
            planned = [(key, {role: by_role[role].pop(key) for role in self.streams})
                       for key in shared[: max(0, capacity)]]
            if self._pusher is not None:
                for key, candidates in planned:
                    pair = self._new_pair(candidates)
                    self._retain(pair)
                    self.inflight[key] = pair
                    self._push_in_background(pair, dict(pair.candidates))
            else:
                with _PushPrefetch(self.streams, planned) as prefetch:
                    for key, candidates in planned:
                        pair = self._new_pair(candidates)
                        self._retain(pair)
                        self.inflight[key] = pair
                        self._push_once(pair, dict(pair.candidates), produced, prepared=prefetch.take(key))
            self._check_timeouts(by_role, now)
        except BaseException as error:
            self.fail_stop(error)
            raise
        return {role: RoleProgress(submissions=tuple(values)) for role, values in produced.items()}

    async def prepare_submissions(self) -> None:
        """Materialize the deferred fragments the next :meth:`submit_ready`
        would push (S27 G-AA deferred capture).

        The plan repeats ``submit_ready``'s choice (shared, not closed, within
        the free pipeline capacity) over candidates that include deferred
        fragments.  A PULL that arrives after this plan stays pending until a
        later call, exactly as if it had arrived after ``submit_ready``."""

        if self.failure is not None or not all(
            callable(getattr(stream, "materialize_submission", None)) for stream in self.streams.values()
        ):
            return
        try:
            by_role = {
                role: _candidate_map(stream.ready_submission_candidates(include_deferred=True), role=role)
                for role, stream in self.streams.items()
            }
            capacity = self.profile.effective_pipeline_depth - len(self.inflight)
            shared = sorted(set.intersection(*(set(keys) for keys in by_role.values())) - set(self.inflight))
            if self.profile.partial_quorum:
                shared = [key for key in shared if not self._round_closed(key[1], key[0])]
            fragment_ids = [key[1] for key in shared[: max(0, capacity)]]
            if fragment_ids:
                await asyncio.gather(*(stream.materialize_submission(fragment_ids)
                                       for stream in self.streams.values()))
        except BaseException as error:
            self.fail_stop(error)
            raise

    def _note_final_rounds(self, by_role: Mapping[str, Mapping[LogicalKey, Any]]) -> None:
        """Set ``terminal_seen`` once every fragment's last round is ready.

        Fragment ``f`` is last merged at step ``total - P + 1 + f``. With one
        syncer process PULLs arrive in step order, so a ready terminal-step
        PULL implied every other last round was ready too. A sharded head
        runs each shard's steps independently: the terminal PULL of one shard
        can arrive while another shard still has to ask for an earlier last
        round. Stopping local training then leaves no round to answer that
        PULL and both sides wait on each other. So training stops only when
        each fragment's last round is ready here or already closed.
        """

        total, fragments = self.profile.total_fragment_steps, self.profile.fragments
        first_last = total - fragments + 1
        for keys in by_role.values():
            self._final_rounds_ready.update(key[1] for key in keys if key[0] >= first_last)
        for fragment_id in range(fragments):
            if fragment_id not in self._final_rounds_ready and self._round_closed(
                fragment_id, first_last + fragment_id
            ):
                self._final_rounds_ready.add(fragment_id)
        if len(self._final_rounds_ready) == fragments:
            self.terminal_seen = True

    def _new_pair(self, candidates: dict[str, Any]) -> PairedSubmission:
        first = next(iter(candidates.values()))
        for candidate in candidates.values():
            if (
                candidate.logical_key != first.logical_key
                or candidate.local_step != first.local_step
                or candidate.c_steps != first.c_steps
                or candidate.c_tokens != first.c_tokens
                or candidate.c_steps < self.profile.local_horizon
            ):
                raise RuntimeError("role retained cuts are not one H-qualified round")
        return PairedSubmission(candidates, local_round=self.local_round)

    def _retain(self, pair: PairedSubmission) -> None:
        retained: list[str] = []
        try:
            for role, candidate in pair.candidates.items():
                self.streams[role].retain_candidate(candidate)
                retained.append(role)
        except BaseException:
            for role in retained:
                self.streams[role].release_candidate(pair.candidates[role])
            raise

    def _release(self, pair: PairedSubmission) -> None:
        """Unpin a pair's operands; with its push job still running, only
        once the job is harvested (a job not yet started is cancelled)."""
        job = pair.job
        if job is not None:
            if job.future.cancel():
                pair.job = None
            else:
                self._releasing.append(pair)
                return
        for role, candidate in pair.candidates.items():
            self.streams[role].release_candidate(candidate)

    def _push_once(
        self,
        pair: PairedSubmission,
        observed: Mapping[str, Any | None],
        produced: dict[str, list[Any]],
        prepared: Mapping[str, Any] | None = None,
    ) -> None:
        replays = {}
        for role in self.streams:
            replayed = _retained_replay(pair.candidates[role], observed.get(role))
            if replayed is None:
                continue
            pair.candidates[role] = replayed
            # A PULL visible after a successful enqueue proves that generation
            # did not commit it: replay the exact retained bytes.
            pair.submissions.pop(role, None)
            replays[role] = replayed
        if self._pusher is not None:
            if replays:
                self._push_in_background(pair, replays)
            return
        for role, replayed in replays.items():
            stream = self.streams[role]
            ahead = None if prepared is None else prepared.get(role)
            try:
                submission = (stream.push_candidate(replayed) if ahead is None
                              else stream.push_candidate(replayed, ahead))
            except PartialMessageGenerationLost:
                submission = None
            if submission is not None:
                pair.submissions[role] = submission
                produced[role].append(submission)
                # The wait for this round's broadcast starts at its last push:
                # a replay answers a reopened round (new attempt).
                pair.retained_at = time.monotonic()

    # -- background pushes (S27 G-AP) -------------------------------------------

    def _push_in_background(self, pair: PairedSubmission, candidates: Mapping[str, Any]) -> None:
        """Hand ``candidates`` (role -> candidate of ``pair``) to a worker.

        Roles whose permit is no longer pending are skipped, as a synchronous
        push would return no submission for them."""
        pushable = {role: candidate for role, candidate in candidates.items()
                    if self.streams[role].pushable(candidate)}
        if pushable:
            pair.job = _PushJob(self._pusher.submit(pushable), pushable)

    def _harvest_pushes(self) -> None:
        """Record every finished push job on this thread.

        A queued push becomes the pair's submission (and the wait for its
        broadcast starts when its last chunk was queued); a push that was not
        queued leaves the pair waiting for its replayed PULL.  A pair released
        while its job ran is unpinned now.  A worker exception propagates."""
        pusher = self._pusher
        if pusher is None:
            return
        live = {id(pair) for pair in self.inflight.values()}
        for pair in [pair for pair in (*self.inflight.values(), *self._releasing)
                     if pair.job is not None and pair.job.future.done()]:
            job, pair.job = pair.job, None
            results = job.future.result()
            pusher.harvested += 1
            for role, sent in results.items():
                stream = self.streams[role]
                stream.note_push_timing(job.candidates[role], sent.timings)
                pusher.busy_seconds += sent.timings.get("prepare_s", 0.0) + sent.timings.get("push_s", 0.0)
                if not sent.queued:
                    continue
                submission = stream.note_pushed(job.candidates[role], sent.link_generation)
                if id(pair) in live:
                    pair.submissions[role] = submission
                    self._harvested[role].append(submission)
                    pair.retained_at = sent.finished_at
        for pair in [pair for pair in self._releasing if pair.job is None]:
            self._releasing.remove(pair)
            self._release(pair)

    async def _drain_pushes(self) -> None:
        """Wait (at most the wait timeout) for every push job, then harvest."""
        futures = [pair.job.future for pair in (*self.inflight.values(), *self._releasing)
                   if pair.job is not None]
        if futures:
            _done, pending = await asyncio.to_thread(
                wait_for_futures, futures, timeout=self.profile.wait_timeout)
            if pending:
                raise TimeoutError(f"{len(pending)} background push(es) did not finish before the wait timeout")
        self._harvest_pushes()

    def _check_timeouts(self, by_role: Mapping[str, Mapping[LogicalKey, Any]], now: float) -> None:
        seen = set().union(*(set(keys) for keys in by_role.values()))
        everywhere = set.intersection(*(set(keys) for keys in by_role.values()))
        unmatched = seen - everywhere
        self._pull_skew_since = {key: self._pull_skew_since.get(key, now) for key in unmatched}
        timeout = self.profile.wait_timeout
        if any(now - started >= timeout for started in self._pull_skew_since.values()):
            raise TimeoutError("timed out waiting for a peer role's matching PULL")
        expired = [(key, pair) for key, pair in self.inflight.items() if now - pair.retained_at >= timeout]
        if not expired:
            return
        # A broadcast that arrived while this island trained, captured or
        # pushed closes its round; it is applied at the next commit, so it
        # is not a missing broadcast.  Stage what has arrived, then decide.
        self._stage_arrived_broadcasts()
        for (step, fragment_id, _base), pair in expired:
            if self._round_closed(fragment_id, step):
                continue
            if pair.complete:
                raise TimeoutError("timed out waiting for a paired WAN broadcast")
            raise TimeoutError("timed out with a one-sided paired submission in flight")

    def _stage_arrived_broadcasts(self) -> None:
        """Stage arrived broadcasts now (no apply) on streams that can do it
        synchronously (``stage_available_broadcasts_now``)."""
        for stream in self.streams.values():
            stage = getattr(stream, "stage_available_broadcasts_now", None)
            if stage is not None:
                stage()

    # -- apply (rule 3) ------------------------------------------------------

    def _round_closed(self, fragment_id: int, version: int) -> bool:
        return all(
            stream.fragment_versions[fragment_id] >= version
            or stream.has_staged_broadcast(fragment_id, version)
            for stream in self.streams.values()
        )

    async def commit_staged_broadcasts(self) -> dict[str, RoleProgress]:
        """Apply matched role broadcasts without waiting for WAN I/O."""

        applied: dict[str, list[int]] = {role: [] for role in self.streams}
        try:
            # A finished background push makes its pair complete: harvest it
            # first, so its broadcast is the paired apply, not an overwrite.
            self._harvest_pushes()
            await asyncio.gather(*(stream.stage_available_broadcasts() for stream in self.streams.values()))
            if self.profile.partial_quorum:
                await self._commit_partial_quorum(applied)
                return {role: RoleProgress(applied_fragments=tuple(dict.fromkeys(v)))
                        for role, v in applied.items()}
            known = {
                (pair.logical_key[1], pair.logical_key[0]) for pair in self.inflight.values()
            }
            staged = set().union(*(set(stream.staged_fragment_versions) for stream in self.streams.values()))
            if staged - known:
                raise RuntimeError("role syncer broadcast has no retained paired submission")
            ready = [
                (key, pair)
                for key, pair in sorted(self.inflight.items())
                if key[0] != self.profile.total_fragment_steps
                and all(stream.has_staged_broadcast(key[1], key[0]) for stream in self.streams.values())
            ]
            if self.profile.sweep and ready and not self._is_whole_sweep([key for key, _ in ready]):
                return {role: RoleProgress() for role in self.streams}
            for key, pair in ready:
                await self._apply_pair(key, pair, applied)
        except BaseException as error:
            self.fail_stop(error)
            raise
        return {role: RoleProgress(applied_fragments=tuple(dict.fromkeys(v))) for role, v in applied.items()}

    async def _commit_partial_quorum(self, applied: dict[str, list[int]]) -> None:
        """Apply every broadcast all roles staged, oldest version first.

        A broadcast with a complete retained submission is the normal paired
        apply.  Any other broadcast closed its round without this island
        (absent, late, a refused or lost push): every role overwrites the
        fragment (alpha = 0) and a retained pair for that round is released.

        A broadcast whose base version this island does not hold waits while
        an older version may still arrive.  When no older version can arrive
        (every role's stream delivers broadcasts in version order per
        fragment, and none is staged), it is a catch-up: the syncer re-sends
        each fragment's newest value after a re-JOIN or reconnect, and the
        stream already dropped any older version.  Every role then jumps the
        fragment to that version, and every retained pair of that fragment up
        to it is released."""

        fragments = self.profile.fragments
        staged = set.intersection(
            *(set(stream.staged_fragment_versions) for stream in self.streams.values())
        )
        monotonic = all(getattr(stream, "monotonic_broadcasts", False) for stream in self.streams.values())
        # One trainer apply per batch of distinct fragments (S27 G-AA).  The
        # order of decisions is unchanged: a fragment seen twice first flushes
        # the batch, so its second version is judged against the applied one.
        batch: list[_PlannedApply] | None = [] if self._batches_applies() else None
        for fragment_id, version in sorted(staged, key=lambda item: (item[1], item[0])):
            if version == self.profile.total_fragment_steps:
                continue  # the terminal cut arrives as FINAL
            if batch and any(planned.fragment_id == fragment_id for planned in batch):
                await self._flush_applies(batch, applied)
            base = max(0, version - fragments)
            held = {stream.fragment_versions[fragment_id] for stream in self.streams.values()}
            if len(held) != 1:
                continue
            (current,) = held
            if current == base:
                await self._overwrite_or_apply_pair(fragment_id, version, applied, catch_up=False, batch=batch)
            elif current < base and monotonic and not any(
                f == fragment_id and v < version for f, v in staged
            ):
                await self._overwrite_or_apply_pair(fragment_id, version, applied, catch_up=True, batch=batch)
                self.caught_up_fragments += 1
                self.caught_up_rounds += fragment_rounds(fragment_id, base, fragments) - (
                    fragment_rounds(fragment_id, current, fragments))
        if batch:
            await self._flush_applies(batch, applied)

    def _batches_applies(self) -> bool:
        """Every role applies several fragments at once
        (``apply_staged_broadcasts``); ``YETO_FULL_PARAMETER_BATCH_APPLY=0``
        keeps one apply per fragment."""

        return (not self.profile.overlap
                and os.environ.get(BATCH_APPLY_ENV, "1") != "0"
                and all(callable(getattr(stream, "apply_staged_broadcasts", None))
                        for stream in self.streams.values()))

    async def _flush_applies(self, batch: list[_PlannedApply], applied: dict[str, list[int]]) -> None:
        """Apply every planned fragment in one call per role, then do each
        fragment's bookkeeping exactly as its own apply would."""

        items = [(planned.fragment_id, planned.version, planned.catch_up) for planned in batch]
        results = await asyncio.gather(*(stream.apply_staged_broadcasts(items) for stream in self.streams.values()))
        for role, fragment_ids in zip(self.streams, results, strict=True):
            applied[role].extend(fragment_ids)
        for planned in batch:
            for role in self.streams:
                if self.streams[role].fragment_versions[planned.fragment_id] != planned.version:
                    raise RuntimeError("batched apply committed mismatched versions")
            if planned.pair is not None:
                self._journal(planned.pair)
                self.inflight.pop(planned.key)
                continue
            for key, pair in planned.closed:
                self._release(pair)
                self.inflight.pop(key)
            self.overwritten_rounds += 1
        batch.clear()

    def _retires(self, version: int) -> bool:
        """Terminal phase: close ``version`` without a trainer apply.

        Once ``any_finalizing`` holds the island trains no more, and a
        fragment's last round (``version >= total - P + 1``) is followed only
        by the FINAL cut, which overwrites every fragment.  Nothing reads the
        trainer's value of that fragment in between, so its apply cannot
        change the final parameters; streams record the round as closed
        (``retire_staged_broadcast``) instead.  Every role must support it."""

        return (
            self.profile.partial_quorum
            and self.any_finalizing
            and version >= self.profile.total_fragment_steps - self.profile.fragments + 1
            and all(getattr(stream, "retires_terminal_broadcasts", False)
                    for stream in self.streams.values())
        )

    async def _overwrite_or_apply_pair(
        self, fragment_id: int, version: int, applied: dict[str, list[int]], *, catch_up: bool,
        batch: list[_PlannedApply] | None = None,
    ) -> None:
        closed = [(key, pair) for key, pair in sorted(self.inflight.items())
                  if key[1] == fragment_id and key[0] <= version]
        exact = next((entry for entry in closed if entry[0][0] == version), None)
        retire = self._retires(version)
        paired = not catch_up and exact is not None and exact[1].complete
        if batch is not None and not retire:
            # The paired apply and the overwrite differ only in bookkeeping.
            batch.append(_PlannedApply(fragment_id, version, catch_up,
                                       key=exact[0] if paired else None,
                                       pair=exact[1] if paired else None,
                                       closed=() if paired else tuple(closed)))
            return
        if paired:
            await self._apply_pair(exact[0], exact[1], applied, retire=retire)
            return
        if retire:
            self.retired_rounds += 1
            calls = (stream.retire_staged_broadcast(fragment_id, version, catch_up=catch_up)
                     for stream in self.streams.values())
        elif catch_up:
            calls = (stream.apply_staged_broadcast(fragment_id, version, catch_up=True)
                     for stream in self.streams.values())
        else:
            calls = (stream.apply_staged_broadcast(fragment_id, version) for stream in self.streams.values())
        results = await asyncio.gather(*calls)
        for role, fragment_ids in zip(self.streams, results, strict=True):
            applied[role].extend(fragment_ids)
            if self.streams[role].fragment_versions[fragment_id] != version:
                raise RuntimeError("overwrite committed mismatched versions")
        for key, pair in closed:
            self._release(pair)
            self.inflight.pop(key)
        self.overwritten_rounds += 1

    def _is_whole_sweep(self, ready: list[LogicalKey]) -> bool:
        """Strict sweep: ``ready`` is all P fragments of one local step.

        Counting only the keys already in flight is not enough: while the
        syncer has permitted part of a sweep, the in-flight keys of that local
        step are themselves partial."""

        local_step = self.inflight[ready[0]].candidates["actor"].local_step
        sweep = [key for key, pair in self.inflight.items()
                 if pair.candidates["actor"].local_step == local_step]
        return (len(sweep) == self.profile.fragments
                and sorted(ready) == sorted(sweep)
                and {key[1] for key in sweep} == set(range(self.profile.fragments)))

    async def _apply_pair(
        self, key: LogicalKey, pair: PairedSubmission, applied: dict[str, list[int]],
        *, retire: bool = False,
    ) -> None:
        step, fragment_id, _base = key
        if not pair.complete:
            raise RuntimeError("broadcast arrived before every paired PUSH was retained")
        if retire:
            self.retired_rounds += 1
            calls = (stream.retire_staged_broadcast(fragment_id, step) for stream in self.streams.values())
        elif self.profile.overlap:
            calls = (stream.apply_staged_broadcast(fragment_id, step, rebase=True)
                     for stream in self.streams.values())
        else:
            calls = (stream.apply_staged_broadcast(fragment_id, step) for stream in self.streams.values())
        results = await asyncio.gather(*calls)
        for role, fragments in zip(self.streams, results, strict=True):
            applied[role].extend(fragments)
            if self.streams[role].fragment_versions[fragment_id] != step:
                raise RuntimeError("paired apply committed mismatched versions")
        self._journal(pair)
        self.inflight.pop(key)

    def _journal(self, pair: PairedSubmission) -> None:
        self.journal.append(dict(pair.submissions))
        self._release(pair)

    # -- local-round lead cap (S27 G-AG) -----------------------------------------

    def lead_blockers(self) -> list[PairedSubmission]:
        """In-flight pairs that stop the next local round under the lead cap K.

        At a boundary after local round ``r`` the next round is ``r + 1``.  A
        pair retained at local round ``p`` (its cut is round ``p``) blocks when
        ``p <= r + 1 - K``: training round ``r + 1`` would put ``K`` or more
        rounds on top of that unmerged push.  With a fixed roster a merge needs
        every island's push, so with K = 1 an island starts a round only after
        every island pushed everything it pushed: it stays at most one local
        round ahead of the slowest island.  A larger K allows K - 1 rounds on
        top of an unmerged push; that bounds the island's c_steps per push,
        not its absolute lead (pushes made one boundary late can repeat)."""

        limit = self.profile.max_local_rounds_ahead
        if limit is None:
            return []
        return [pair for pair in self.inflight.values() if pair.local_round <= self.local_round + 1 - limit]

    async def wait_for_lead(self, serve: Any = None) -> float:
        """Lead cap: block at this boundary until :meth:`lead_blockers` is empty.

        The wait keeps the island's sync duties: every poll materializes and
        pushes PULLs it has candidates for (from this boundary's cut), applies
        arrived broadcasts, and runs ``serve`` (peer snapshots).  It stops at
        once when the terminal phase starts (``any_finalizing``): the terminal
        barrier then waits instead.

        No new timer: each pair's wait timeout keeps running inside
        :meth:`submit_ready`, so a broadcast that never comes still fails
        there.  The lead wait gives up (and the island trains on, the old
        behaviour) once the oldest blocking pair has waited half of
        ``wait_timeout`` since its last push, before that pair could time out.
        This also breaks a theoretical deadlock where each island waits for a
        push the other can make only after training (possible only when two
        broadcasts reach the two islands in different orders).

        Returns the seconds this boundary waited."""

        blockers = self.lead_blockers()
        if not blockers or self.any_finalizing or self.failure is not None:
            return 0.0
        started = time.monotonic()
        poll = max(self.profile.poll_seconds, LEAD_WAIT_POLL_S)
        escaped = False
        oldest = min(pair.local_round for pair in blockers)
        print(f"[yeto-full-parameter] lead wait: local round {self.local_round}, "
              f"{len(blockers)} pair(s) pushed at local round {oldest}.. are unmerged "
              f"(lead cap {self.profile.max_local_rounds_ahead})", file=sys.stderr, flush=True)
        while True:
            await asyncio.sleep(poll)
            await self.prepare_submissions()
            self.submit_ready()
            await self.commit_staged_broadcasts()
            if serve is not None:
                await serve()
            blockers = self.lead_blockers()
            if not blockers or self.any_finalizing:
                break
            give_up = min(pair.retained_at for pair in blockers) + self.profile.wait_timeout / 2
            if time.monotonic() >= give_up:
                escaped = True
                break
        waited = time.monotonic() - started
        self.lead_waits += 1
        self.lead_wait_seconds += waited
        self.lead_wait_escapes += int(escaped)
        print(f"[yeto-full-parameter] lead wait: local round {self.local_round} waited {waited:.1f}s"
              + (" and gave up at half the wait timeout (trains on)" if escaped else ""),
              file=sys.stderr, flush=True)
        return waited

    # -- terminal barrier ------------------------------------------------------

    async def finish_without_more_training(self) -> dict[str, RoleProgress]:
        """Reach every role's final manifest without another train step."""

        progress = {role: RoleProgress() for role in self.streams}
        deadline = time.monotonic() + self.profile.wait_timeout
        while not self.finalizing:
            await self.prepare_submissions()
            submitted = self.submit_ready()
            committed = await self.commit_staged_broadcasts()
            progress = {
                role: progress[role].extend(submitted[role]).extend(committed[role]) for role in self.streams
            }
            if self.finalizing:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("paired terminal barrier could not reach every final cut")
            await asyncio.sleep(self.profile.poll_seconds)
        try:
            await self._drain_pushes()
            await asyncio.gather(*(stream.wait_and_apply_final() for stream in self.streams.values()))
            self.fragment_versions()
            for key, pair in sorted(self.inflight.items()):
                if pair.complete:
                    self._journal(pair)
                elif self.profile.partial_quorum:
                    self._release(pair)  # the final cut superseded it
                else:
                    raise RuntimeError("finalization retained an incomplete paired submission")
                self.inflight.pop(key)
        except BaseException as error:
            self.fail_stop(error)
            raise
        return progress

    def acknowledge_final(self) -> bool:
        """ACK every role's final manifest; True when every ACK landed."""

        errors: list[BaseException] = []
        for stream in self.streams.values():
            try:
                stream.acknowledge_final()
            except Exception as error:  # noqa: BLE001 - collected and re-raised below
                errors.append(error)
        if errors:
            failure = RuntimeError(f"terminal ACK publication failed for {len(errors)} role(s)")
            for error in errors:
                failure.add_note(f"{type(error).__name__}: {error}")
            raise failure from errors[0]
        return all(stream.final_acknowledged for stream in self.streams.values())


def _candidate_map(candidates: Sequence[Any], *, role: str) -> dict[LogicalKey, Any]:
    result: dict[LogicalKey, Any] = {}
    for candidate in candidates:
        if candidate.logical_key in result:
            raise RuntimeError(f"{role} returned duplicate logical PULLs")
        result[candidate.logical_key] = candidate
    return result


BATCH_APPLY_ENV = "YETO_FULL_PARAMETER_BATCH_APPLY"
# S27 G-AG: threads that compute the next pushes' deltas and wire encodings
# while the current one is enqueued (1 = compute each push in turn, as before).
PUSH_WORKERS_ENV = "YETO_FULL_PARAMETER_PUSH_WORKERS"
DEFAULT_PUSH_WORKERS = 3
_PUSH_POOL: list[Any] = []  # (workers, ThreadPoolExecutor), created on first use


def push_workers() -> int:
    try:
        return max(1, int(os.environ.get(PUSH_WORKERS_ENV, DEFAULT_PUSH_WORKERS)))
    except ValueError:
        return 1


def _push_pool(workers: int) -> Any:
    if not _PUSH_POOL or _PUSH_POOL[0][0] != workers:
        from concurrent.futures import ThreadPoolExecutor

        _PUSH_POOL[:] = [(workers, ThreadPoolExecutor(max_workers=workers,
                                                      thread_name_prefix="yeto-push-prepare"))]
    return _PUSH_POOL[0][1]


class _PushPrefetch:
    """Compute the deltas and wire encodings of the next planned pushes on
    worker threads (S27 G-AG) while the pushing thread enqueues them in plan
    order.

    Only when every role can (``encodes_ahead``) and
    ``YETO_FULL_PARAMETER_PUSH_WORKERS`` > 1; otherwise :meth:`take` returns
    None and each push computes its bytes itself, as before.  The bytes are
    the same either way: a prepared push reads the same retained cut and
    anchor, and the codec is deterministic (no stochastic rounding, no error
    feedback).  At most ``workers`` pushes are computed ahead; on exit every
    unfinished computation is cancelled or awaited, so no worker outlives the
    submit call."""

    def __init__(self, streams: Mapping[str, Any], planned: Sequence[tuple[LogicalKey, Mapping[str, Any]]]):
        self.streams = streams
        self.workers = push_workers()
        self.enabled = len(planned) > 1 and self.workers > 1 and all(
            getattr(stream, "encodes_ahead", False) and callable(getattr(stream, "prepare_push", None))
            for stream in streams.values())
        self.queue = list(planned)
        self.futures: dict[LogicalKey, dict[str, Any]] = {}

    def __enter__(self) -> Self:
        if self.enabled:
            self._fill()
        return self

    def _fill(self) -> None:
        pool = _push_pool(self.workers)
        while self.queue and len(self.futures) < self.workers:
            key, candidates = self.queue.pop(0)
            self.futures[key] = {role: pool.submit(self.streams[role].prepare_push, candidate)
                                 for role, candidate in candidates.items()}

    def take(self, key: LogicalKey) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        futures = self.futures.pop(key, None)
        if futures is None:
            return None
        try:
            return {role: future.result() for role, future in futures.items()}
        finally:
            self._fill()

    def __exit__(self, *exc: object) -> None:
        for futures in self.futures.values():
            for future in futures.values():
                future.cancel()
        for futures in self.futures.values():
            for future in futures.values():
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:  # noqa: BLE001, S110 - the push that needed it already failed
                        pass
        self.futures.clear()
        self.queue.clear()
# Lead cap: poll interval of the boundary wait (the wait lasts a peer's round).
LEAD_WAIT_POLL_S = 0.05

# S27 G-AP background pushes: "0" keeps every push on the coordinator thread.
BACKGROUND_PUSH_ENV = "YETO_FULL_PARAMETER_BACKGROUND_PUSH"
# CPU priority (nice increment) of the background push threads: training,
# whose optimizer may run on the CPU, keeps the cores it needs.
PUSH_NICE_ENV = "YETO_FULL_PARAMETER_PUSH_NICE"
DEFAULT_PUSH_NICE = 10
# Seconds close_all waits for running background pushes after closing roles.
PUSH_SHUTDOWN_WAIT_S = 30.0


@dataclass(frozen=True)
class _PushJob:
    """One pair's background push: the candidates it pushes and its future
    (role -> ``SentPush``)."""

    future: Future
    candidates: dict[str, Any]


def _lower_thread_priority(nice: int) -> None:
    """Raise this thread's nice value (Linux applies it per thread)."""
    if nice <= 0 or not hasattr(os, "setpriority"):
        return
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except OSError:  # pragma: no cover - a platform without per-thread priority
        pass


def push_nice() -> int:
    try:
        return max(0, min(19, int(os.environ.get(PUSH_NICE_ENV, DEFAULT_PUSH_NICE))))
    except ValueError:
        return DEFAULT_PUSH_NICE


class _BackgroundPusher:
    """Push preparation and upload on worker threads (S27 G-AP).

    The synchronous push made the island wait at every boundary while it
    computed each pushed fragment's delta and wire encoding and queued the
    bytes (s27-gr-j: 150-250 s of 450-600 s rounds on a 30B model).  A job
    does that for one pair (``prepare_push`` + ``send_prepared`` for every
    role) while the island trains on.  The pairing keeps every table: it
    submits jobs, keeps a pair's operands pinned while its job runs, and
    records results (``note_pushed``) only when it harvests a finished job on
    its own thread.  The bytes are those of a synchronous push: the job reads
    the same retained cut and anchor, and only deterministic codecs encode
    ahead.

    Used for the asynchronous rounds (``partial_quorum``) when every role can
    encode ahead and offers the background methods;
    ``YETO_FULL_PARAMETER_BACKGROUND_PUSH=0`` keeps the synchronous push.
    ``YETO_FULL_PARAMETER_PUSH_WORKERS`` sets the worker count (default 3)."""

    METHODS = ("pushable", "prepare_push", "send_prepared", "note_pushed", "note_push_timing")

    def __init__(self, streams: Mapping[str, Any], workers: int, nice: int) -> None:
        self.streams = streams
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yeto-push-bg",
                                           initializer=_lower_thread_priority, initargs=(nice,))
        self.workers = workers
        self.submitted = 0
        self.harvested = 0
        self.busy_seconds = 0.0
        self._futures: list[Future] = []

    @classmethod
    def for_streams(cls, streams: Mapping[str, Any], profile: FullParameterSyncProfile) -> _BackgroundPusher | None:
        if (
            not profile.partial_quorum
            or profile.sweep
            or profile.overlap
            or os.environ.get(BACKGROUND_PUSH_ENV, "1") == "0"
            or not all(getattr(stream, "encodes_ahead", False)
                       and all(callable(getattr(stream, name, None)) for name in cls.METHODS)
                       for stream in streams.values())
        ):
            return None
        return cls(streams, push_workers(), push_nice())

    def submit(self, candidates: Mapping[str, Any]) -> Future:
        future = self.executor.submit(self._push, dict(candidates))
        self.submitted += 1
        self._futures = [value for value in self._futures if not value.done()] + [future]
        return future

    def _push(self, candidates: dict[str, Any]) -> dict[str, Any]:
        sent = {}
        for role, candidate in candidates.items():
            stream = self.streams[role]
            sent[role] = stream.send_prepared(candidate, stream.prepare_push(candidate))
        return sent

    def stop(self) -> list[Future]:
        """Cancel jobs not started; return the running ones."""
        self.executor.shutdown(wait=False, cancel_futures=True)
        return [future for future in self._futures if not future.done()]


@dataclass(frozen=True)
class _PlannedApply:
    """One broadcast of a batched partial-quorum apply and its bookkeeping:
    a paired round (``key``/``pair``, journaled) or an overwrite (``closed``
    pairs released)."""

    fragment_id: int
    version: int
    catch_up: bool
    key: LogicalKey | None = None
    pair: PairedSubmission | None = None
    closed: tuple[tuple[LogicalKey, PairedSubmission], ...] = ()


def _retained_replay(retained: Any, observed: Any | None) -> Any | None:
    """Consume only a newly observed permit; keep the retained operands."""

    if observed is None:
        return None
    if (
        observed.logical_key != retained.logical_key
        or observed.permit.round_attempt < retained.permit.round_attempt
    ):
        raise RuntimeError("reconnect replay changed logical identity")
    return replace(retained, permit=observed.permit)




class _PhaseClock:
    """Seconds between consecutive marks, by the name of the part that ended."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._last = time.monotonic()

    def mark(self, name: str) -> None:
        now = time.monotonic()
        self.seconds[name] = self.seconds.get(name, 0.0) + now - self._last
        self._last = now


class FullParameterSync(RoundCutHooks):
    """The ports ``SyncSession`` for full-parameter training (every algorithm).

    The driver trains one local round, then :meth:`boundary` records it on
    every role, submits and applies through :class:`RolePairing`, and returns
    the actor's cut as a trainer-resident handle that the driver publishes
    with its normal publication check.  With ``--rl-checkpoint-store`` the island
    also writes periodic round cuts (:class:`RoundCutHooks`) for offline
    evaluation; the syncer stays authoritative for the policy.  A strict sweep profile blocks in the
    boundary until the whole sweep is applied; a streaming profile never waits
    for WAN I/O.

    ``run`` executes a coroutine on the engine's event loop; ``groups`` maps a
    role to its trainer group; ``trajectories(rollout_id)`` returns the
    immutable trajectory evidence of the round the trainer just consumed.
    """

    # S27 G-ED: the local round counter starts at 0 in every process (start()),
    # so a ledger restored from an earlier process's island cut does not apply
    # (yeto.rl.engine.ledger.island_cut_ledger).
    local_rounds_restart = True

    def __init__(
        self,
        pairing: RolePairing,
        *,
        groups: Mapping[str, Any],
        run: Any,
        trajectories: Any,
    ) -> None:
        if set(groups) != set(pairing.streams):
            raise ValueError("every trainable role needs exactly one trainer group")
        self.pairing = pairing
        self.groups = dict(groups)
        self.run = run
        self.trajectories = trajectories
        self.terminal = False
        self.finished = False
        self.closed = False
        self._driver: IslandDriver | None = None
        self._timing = SyncTiming(None, None)  # L5.6, set by each boundary
        # Overlap: fragment versions before the sweep still in flight.
        self._pending_sweep: tuple[int, ...] | None = None
        # S26 E3b: actor data-stream counters at the previous boundary.
        self._upload_before: tuple | None = None
        # Partial quorum: overwritten rounds at the previous boundary.
        self._overwritten_before = 0
        self._lead_wait_s = 0.0  # lead cap: this boundary's wait
        self._push_timing_before: dict[str, float] | None = None
        # S27 G-AP: seconds of each part of this boundary, and the background
        # push counters at the previous boundary.
        self._phase_seconds: dict[str, float] = {}
        self._push_jobs_before: dict[str, float] = {}
        # S27 task assignment: this process joined a pool that was already running
        # (fragments newer than version 0, or a peer recovery), so the island ran
        # before; the driver then continues at its recorded data position
        # (``data_position`` module doc). Set by ``start``.
        self.joined_running_pool = False

    @property
    def outer_version(self) -> int | None:
        """The merged version (module doc) of the asynchronous rounds; None
        for the strict sweep, whose policy version is already global."""

        if not self.pairing.profile.partial_quorum:
            return None
        return merged_version(self.pairing.fragment_versions())

    @property
    def actor(self) -> RoleStream:
        return self.pairing.streams["actor"]

    def start(self, driver: IslandDriver) -> SyncStart:
        self._driver = driver
        self.run(self.pairing.gather_roles(lambda role, stream: stream.initialize(self.groups[role])))
        self._upload_before = self._stream_snapshots()
        outcome = getattr(self.actor, "recovery_outcome", None)
        if outcome is not None:
            driver.emit("rl_peer_recovery", path=outcome.path, donor=outcome.donor, t_s=outcome.t_s,
                        done_step=outcome.done_step, steps=outcome.steps, replayed=outcome.replayed,
                        refused_sources=outcome.refused_sources, seconds=outcome.seconds,
                        versions=list(outcome.versions))
        self.joined_running_pool = outcome is not None or any(
            int(version) > 0 for version in self.pairing.fragment_versions())
        self._publish_outer_version(driver)
        # the local round counter starts at 0 in every process (module doc)
        return SyncStart(state=self.actor.resident_state(0), rollout_id=0)

    def _publish_outer_version(self, driver: IslandDriver) -> None:
        """The driver's publish event and the synced-trainer releases read it."""

        if self.pairing.profile.partial_quorum:
            driver.outer_version = self.outer_version

    def boundary(self, driver: IslandDriver, *, rollout_id: int, stats: Any) -> SyncBoundary:
        del stats
        self._timing = SyncTiming(None, None)
        self._lead_wait_s = 0.0
        self._phase_seconds = {}
        self.pairing.local_round = rollout_id + 1
        self.terminal = self.run(self._boundary(rollout_id))
        versions = self.pairing.fragment_versions()
        state = self.actor.resident_state(rollout_id + 1)
        self._publish_outer_version(driver)
        driver.emit(
            "rl_full_parameter_round",
            policy_version=rollout_id + 1,
            terminal=self.terminal,
            **{
                "sync/fragment_versions": list(versions),
                "sync/global_policy_hash": state.policy_tensor_hash(),
                "sync/inflight_pairs": len(self.pairing.inflight),
                **(self._partial_quorum_fields(versions) if self.pairing.profile.partial_quorum else {}),
                **self._lead_fields(),
                **self._push_timing_fields(),
                **self._boundary_fields(),
                **self._timing.fields(),
                **self._upload_fields(),
            },
        )
        return SyncBoundary(state=state, stop=self.terminal)

    def _partial_quorum_fields(self, versions: tuple[int, ...]) -> dict[str, int]:
        pairing = self.pairing
        overwritten = pairing.overwritten_rounds - self._overwritten_before
        self._overwritten_before = pairing.overwritten_rounds
        return {
            "sync/overwritten_rounds": pairing.overwritten_rounds,
            "sync/overwritten_fragments": overwritten,
            "sync/caught_up_rounds": pairing.caught_up_rounds,
            "sync/caught_up_fragments": pairing.caught_up_fragments,
            "sync/retired_rounds": pairing.retired_rounds,
            "sync/outer_version": merged_version(versions),
        }

    def _lead_fields(self) -> dict[str, object]:
        pairing = self.pairing
        if pairing.profile.max_local_rounds_ahead is None:
            return {}
        return {
            "sync/max_local_rounds_ahead": pairing.profile.max_local_rounds_ahead,
            "sync/lead_wait_s": round(self._lead_wait_s, 3),
            "sync/lead_waits": pairing.lead_waits,
            "sync/lead_wait_escapes": pairing.lead_wait_escapes,
        }

    def _boundary_fields(self) -> dict[str, object]:
        """S27 G-AP: where this boundary's time went (``sync/boundary_<part>_s``:
        capture = record + capture of the local round, apply, materialize,
        submit, apply_after_submit, wait = peer snapshots, lead cap and sweep
        waits) and the background pushes (submitted / harvested this
        boundary, still running, and the worker seconds they took)."""
        fields: dict[str, object] = {f"sync/boundary_{name}_s": round(value, 3)
                                     for name, value in self._phase_seconds.items()}
        # S27 G-AY: only on a round whose capture read no master (the policy
        # hash of that round is a label, not a content hash).
        if getattr(self.actor, "last_capture_lazy", False) is True:
            fields["sync/capture_lazy"] = True
        counters = self.pairing.push_job_counters()
        fields["sync/push_background"] = bool(counters)
        if counters:
            before, self._push_jobs_before = self._push_jobs_before, counters
            fields.update({
                "sync/push_jobs_submitted": int(counters["submitted"] - before.get("submitted", 0)),
                "sync/push_jobs_harvested": int(counters["harvested"] - before.get("harvested", 0)),
                "sync/push_jobs_running": int(counters["running"]),
                "sync/push_worker_s": round(counters["busy_seconds"] - before.get("busy_seconds", 0.0), 3),
            })
        return fields

    def _push_timing_fields(self) -> dict[str, object]:
        """The actor's push time split of this boundary (S27 G-AG): delta
        computation, error feedback, wire encoding and enqueue, summed over
        the fragments it pushed; streams without the counters add nothing."""
        totals = getattr(self.actor, "push_timing_totals", None)
        if not isinstance(totals, dict):
            return {}
        before, self._push_timing_before = self._push_timing_before or {}, dict(totals)
        return {f"sync/push_{name}": round(value - before.get(name, 0.0), 3) if name != "count"
                else int(value - before.get(name, 0)) for name, value in totals.items()}

    def _stream_snapshots(self) -> tuple | None:
        snapshots = getattr(getattr(self.actor, "client", None), "stream_snapshots", None)
        return snapshots() if callable(snapshots) else None

    def _upload_fields(self) -> dict[str, object]:
        """Actor upload rates of this boundary and the wait after the upload."""
        after = self._stream_snapshots()
        before, self._upload_before = self._upload_before, after
        if after is None or before is None:
            return {}
        client = self.actor.client
        return upload_fields(
            upload_window(before, after),
            drained_ts=client.upload_drained_at,
            timing=self._timing,
            stalls=client.send_progress().stalls,
        )

    async def _boundary(self, rollout_id: int) -> bool:
        if self.pairing.profile.overlap:
            return await self._overlap_boundary(rollout_id)
        pairing = self.pairing
        trajectories = self._round_trajectories(rollout_id)
        phases = _PhaseClock()
        await pairing.gather_roles(
            lambda _role, stream: stream.complete_local_round(
                rollout_id=rollout_id, trajectories=trajectories
            )
        )
        phases.mark("capture")
        before = pairing.fragment_versions()
        if pairing.profile.partial_quorum:
            # Rounds that closed while this island trained are applied first,
            # so the island never uploads a delta the syncer would refuse.
            await pairing.commit_staged_broadcasts()
            phases.mark("apply")
        await pairing.prepare_submissions()
        phases.mark("materialize")
        pairing.submit_ready()
        phases.mark("submit")
        submit_ts = unix_now()
        if not pairing.pushes_in_background:
            # A synchronous submit took long enough for more broadcasts to
            # arrive; with background pushes the next boundary applies them
            # (one trainer apply per boundary: an apply has whole-model costs).
            await pairing.commit_staged_broadcasts()
            phases.mark("apply_after_submit")
        await self._serve_peer_snapshots()
        if pairing.profile.max_local_rounds_ahead is not None:
            self._lead_wait_s = await pairing.wait_for_lead(self._serve_peer_snapshots)
        if pairing.profile.sweep:
            await self._wait_for_sweep(before)
            # Broadcasts are staged in the background, so the receive time
            # is not separated: the wait includes it.
            self._timing = SyncTiming(submit_ts, unix_now())
        else:
            self._timing = SyncTiming(submit_ts, None)  # streaming: no global wait
        phases.mark("wait")
        self._phase_seconds = phases.seconds
        if pairing.any_finalizing:
            await pairing.finish_without_more_training()
            return True
        for stream in pairing.streams.values():
            stream.discard_detached_submission_cut()
        return False

    def _round_trajectories(self, rollout_id: int) -> Any:
        """The trajectory evidence of round ``rollout_id``. Rollout ahead (driver
        ``rollout_ahead``): a batch generated by the previous version is bound to
        the policy the driver published for this round, the trainer's start state;
        a batch generated by the round's own policy (round 0) already names it."""
        evidence = self.trajectories(rollout_id)
        driver = self._driver
        if driver is None or not getattr(driver, "rollout_ahead", 0):
            return evidence
        version, policy_hash = parse_policy_snapshot_token(driver.expected_token)
        if version != rollout_id:
            raise RuntimeError(f"round {rollout_id} trains from published policy v{version}")
        if evidence.behavior_policy_hash == policy_hash:
            return evidence
        return bind_round_policy(evidence, policy_hash)

    async def _serve_peer_snapshots(self) -> None:
        """S27 E-3: a donor answers snapshot requests here, after this
        boundary's broadcasts are applied and before the next local round (a
        safe point). Streams without peer recovery have no such method."""
        served = 0
        for stream in self.pairing.streams.values():
            serve = getattr(stream, "serve_peer_snapshots", None)
            if serve is not None:
                served += await serve()
        if served and self._driver is not None:
            self._driver.emit("rl_peer_snapshot_served", count=served)

    async def _overlap_boundary(self, rollout_id: int) -> bool:
        """Streaming-DiLoCo overlap (S26 E3; one more outer step of staleness).

        Round k's delta (``current - base``) is pushed at boundary k and the
        island trains round k+1 at once. At boundary k+1 the island first
        waits for sweep k (``G_k``), then rebases its trainer in place:
        ``theta <- G_k + (theta_local_{k+1} - base_k)`` with ``G_k`` as the
        anchor, so the next push is exactly round k+1's progress and round
        k's delta is never counted twice. The terminal round waits for the
        final cut like the strict sweep."""

        pairing = self.pairing
        trajectories = self._round_trajectories(rollout_id)
        hold = self._pending_sweep is not None
        await pairing.gather_roles(
            lambda _role, stream: stream.complete_local_round(
                rollout_id=rollout_id, trajectories=trajectories, hold_submission=hold
            )
        )
        wait_ts = unix_now()
        if hold:
            await self._wait_for_sweep(self._pending_sweep)
            self._pending_sweep = None
            if not pairing.any_finalizing:
                for stream in pairing.streams.values():
                    stream.finish_rebase()
        if pairing.any_finalizing:
            await pairing.finish_without_more_training()
            return True
        before = pairing.fragment_versions()
        if rollout_id + 1 >= pairing.profile.total_rounds:
            # Terminal round: nothing trains after it, so wait for the cut.
            pairing.submit_ready()
            submit_ts = unix_now()
            await pairing.commit_staged_broadcasts()
            await self._wait_for_sweep(before)
            self._timing = SyncTiming(submit_ts, unix_now())
            await pairing.finish_without_more_training()
            return True
        await self._push_whole_sweep(before)
        self._pending_sweep = before
        for stream in pairing.streams.values():
            stream.mark_round_base()
        # This boundary blocked on the previous sweep (wait + rebase) and on
        # handing every fragment of this sweep to the wire.
        self._timing = SyncTiming(wait_ts, unix_now())
        return False

    async def _push_whole_sweep(self, before: tuple[int, ...]) -> None:
        """Overlap: submit every fragment of the sweep after ``before`` before
        training on. A PULL that arrives later would otherwise wait for the
        next boundary, a whole local round, and stall the sweep. The syncer
        sends all P PULLs at once with ``--pipeline P`` (required by the
        launcher), so this waits only for their arrival."""

        pairing = self.pairing
        fragments = pairing.profile.fragments
        steps = {version + fragments if version else fragment_id + 1
                 for fragment_id, version in enumerate(before)}
        deadline = time.monotonic() + pairing.profile.wait_timeout

        def pushed() -> bool:
            return steps <= {key[0] for key, pair in pairing.inflight.items() if pair.complete}

        pairing.submit_ready()
        while not pushed():
            if time.monotonic() >= deadline:
                error = TimeoutError("overlap could not submit the whole sweep before the wait timeout")
                pairing.fail_stop(error)
                raise error
            await asyncio.sleep(pairing.profile.poll_seconds)
            pairing.submit_ready()

    async def _wait_for_sweep(self, before: tuple[int, ...]) -> None:
        """Strict sweep: block until all P fragments of this local step are
        applied, i.e. every fragment version moved past ``before``.  An empty
        in-flight set alone proves nothing: the syncer may not have permitted
        the rest of the sweep yet."""

        pairing = self.pairing
        deadline = time.monotonic() + pairing.profile.wait_timeout

        def applied() -> bool:
            return all(now > then for now, then in zip(pairing.fragment_versions(), before, strict=True))

        while not applied() and not pairing.any_finalizing:
            if time.monotonic() >= deadline:
                error = TimeoutError("dense sweep was not applied before the wait timeout")
                pairing.fail_stop(error)
                raise error
            await asyncio.sleep(pairing.profile.poll_seconds)
            pairing.submit_ready()
            await pairing.commit_staged_broadcasts()

    def is_final_round(self, driver: IslandDriver, *, rollout_id: int) -> bool:
        del driver, rollout_id
        return self.terminal

    def published(self, driver: IslandDriver, *, rollout_id: int, policy_hash: str) -> None:
        del rollout_id, policy_hash
        if self.terminal:
            self.finished = self.pairing.acknowledge_final()
            # S27 G-AY: only when the islands ACK the final manifest without
            # downloading the final parameters (they stay with the syncer).
            final_cut = getattr(getattr(self.actor, "config", None), "final_cut", "download")
            driver.emit("rl_full_parameter_final_ack", acknowledged=self.finished,
                        **({"sync/final_cut": final_cut} if final_cut != "download" else {}))

    def finish(self, driver: IslandDriver) -> None:
        del driver
        if not self.finished:
            raise RuntimeError("full-parameter sync stopped before every role acknowledged its final cut")

    def close(self, reason: int = LeaveReason.FINISHED) -> None:
        """Close every role once.  Full-parameter syncers run in legacy mode,
        which has no LEAVE message, so the P5 reason code is only recorded."""

        if self.closed:
            return
        self.closed = True
        failures = self.pairing.close_all()
        if self._driver is not None:
            self._driver.emit(
                "rl_sync_closed",
                **{"sync/leave_reason": int(reason),
                   "sync/leave_reason_name": LeaveReason(reason).name,
                   "sync/finished": self.finished,
                   **({"sync/close_failures": failures} if failures else {})},
            )
        if failures:
            raise RuntimeError("full-parameter role close failed: " + "; ".join(failures))


class FullParameterPolicyState:
    """``PolicyState`` of a full-parameter island: the actor stream's cut.

    Tensors never leave the trainer (P0-D1).  Applying a cut happens only
    inside the role streams, so :meth:`apply` refuses.
    """

    def __init__(self, sync: FullParameterSync) -> None:
        self._sync = sync
        self.policy_version = 0

    @property
    def _actor(self) -> RoleStream:
        return self._sync.actor

    @property
    def layout_hash(self) -> str:
        adapter = getattr(self._actor, "adapter", None)
        if adapter is None:
            raise RuntimeError("full-parameter layout requested before the sync started")
        return adapter.layout.layout_hash

    def export(self, *, policy_version: int | None = None) -> Any:
        return self._actor.resident_state(self.policy_version if policy_version is None else policy_version)

    def export_digest(self, *, policy_version: int | None = None) -> Any:
        return self.export(policy_version=policy_version)

    def apply(self, state: Any, *, optimizer: str, local_step: int) -> None:
        raise RuntimeError("full-parameter cuts are applied only through the role streams")
