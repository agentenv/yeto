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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from yeto.protocol import PartialMessageGenerationLost
from yeto.rl.pool_client import LeaveReason
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
        cls, configs: Mapping[str, Any], *, sweep: bool = False
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

    # -- fail-stop ---------------------------------------------------------

    def fail_stop(self, error: BaseException) -> None:
        """Rule 4: close every role once; keep the first failure."""

        if self.failure is None:
            self.failure = error
        for note in self.close_all():
            error.add_note(note)

    def close_all(self) -> list[str]:
        """Close every role even when one close raises; one note per failure."""

        notes = []
        for role, stream in self.streams.items():
            try:
                stream.close()
            except Exception as close_error:  # noqa: BLE001 - every role must close
                notes.append(f"{role} close failed: {type(close_error).__name__}: {close_error}")
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
            by_role = {
                role: _candidate_map(stream.ready_submission_candidates(), role=role)
                for role, stream in self.streams.items()
            }
            now = time.monotonic()
            self._note_final_rounds(by_role)
            for key, pair in tuple(self.inflight.items()):
                observed = {role: by_role[role].pop(key, None) for role in self.streams}
                self._push_once(pair, observed, produced)
            capacity = self.profile.effective_pipeline_depth - len(self.inflight)
            shared = sorted(set.intersection(*(set(keys) for keys in by_role.values())))
            if self.profile.partial_quorum:
                # A round whose broadcast is staged or applied closed without
                # this island: pushing would only upload a refused delta.
                shared = [key for key in shared if not self._round_closed(key[1], key[0])]
            for key in shared[: max(0, capacity)]:
                pair = self._new_pair({role: by_role[role].pop(key) for role in self.streams})
                self._retain(pair)
                self.inflight[key] = pair
                self._push_once(pair, dict(pair.candidates), produced)
            self._check_timeouts(by_role, now)
        except BaseException as error:
            self.fail_stop(error)
            raise
        return {role: RoleProgress(submissions=tuple(values)) for role, values in produced.items()}

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
        return PairedSubmission(candidates)

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
        for role, candidate in pair.candidates.items():
            self.streams[role].release_candidate(candidate)

    def _push_once(
        self,
        pair: PairedSubmission,
        observed: Mapping[str, Any | None],
        produced: dict[str, list[Any]],
    ) -> None:
        for role, stream in self.streams.items():
            replayed = _retained_replay(pair.candidates[role], observed.get(role))
            if replayed is None:
                continue
            pair.candidates[role] = replayed
            # A PULL visible after a successful enqueue proves that generation
            # did not commit it: replay the exact retained bytes.
            pair.submissions.pop(role, None)
            try:
                submission = stream.push_candidate(replayed)
            except PartialMessageGenerationLost:
                submission = None
            if submission is not None:
                pair.submissions[role] = submission
                produced[role].append(submission)
                # The wait for this round's broadcast starts at its last push:
                # a replay answers a reopened round (new attempt).
                pair.retained_at = time.monotonic()

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
        for fragment_id, version in sorted(staged, key=lambda item: (item[1], item[0])):
            if version == self.profile.total_fragment_steps:
                continue  # the terminal cut arrives as FINAL
            base = max(0, version - fragments)
            held = {stream.fragment_versions[fragment_id] for stream in self.streams.values()}
            if len(held) != 1:
                continue
            (current,) = held
            if current == base:
                await self._overwrite_or_apply_pair(fragment_id, version, applied, catch_up=False)
            elif current < base and monotonic and not any(
                f == fragment_id and v < version for f, v in staged
            ):
                await self._overwrite_or_apply_pair(fragment_id, version, applied, catch_up=True)
                self.caught_up_fragments += 1
                self.caught_up_rounds += fragment_rounds(fragment_id, base, fragments) - (
                    fragment_rounds(fragment_id, current, fragments))

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
        self, fragment_id: int, version: int, applied: dict[str, list[int]], *, catch_up: bool
    ) -> None:
        closed = [(key, pair) for key, pair in sorted(self.inflight.items())
                  if key[1] == fragment_id and key[0] <= version]
        exact = next((entry for entry in closed if entry[0][0] == version), None)
        retire = self._retires(version)
        if not catch_up and exact is not None and exact[1].complete:
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

    # -- terminal barrier ------------------------------------------------------

    async def finish_without_more_training(self) -> dict[str, RoleProgress]:
        """Reach every role's final manifest without another train step."""

        progress = {role: RoleProgress() for role in self.streams}
        deadline = time.monotonic() + self.profile.wait_timeout
        while not self.finalizing:
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
        self._publish_outer_version(driver)
        return SyncStart(state=self.actor.resident_state(0), rollout_id=0)

    def _publish_outer_version(self, driver: IslandDriver) -> None:
        """The driver's publish event and the synced-trainer releases read it."""

        if self.pairing.profile.partial_quorum:
            driver.outer_version = self.outer_version

    def boundary(self, driver: IslandDriver, *, rollout_id: int, stats: Any) -> SyncBoundary:
        del stats
        self._timing = SyncTiming(None, None)
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
        trajectories = self.trajectories(rollout_id)
        await pairing.gather_roles(
            lambda _role, stream: stream.complete_local_round(
                rollout_id=rollout_id, trajectories=trajectories
            )
        )
        before = pairing.fragment_versions()
        if pairing.profile.partial_quorum:
            # Rounds that closed while this island trained are applied first,
            # so the island never uploads a delta the syncer would refuse.
            await pairing.commit_staged_broadcasts()
        pairing.submit_ready()
        submit_ts = unix_now()
        await pairing.commit_staged_broadcasts()
        await self._serve_peer_snapshots()
        if pairing.profile.sweep:
            await self._wait_for_sweep(before)
            # Broadcasts are staged in the background, so the receive time
            # is not separated: the wait includes it.
            self._timing = SyncTiming(submit_ts, unix_now())
        else:
            self._timing = SyncTiming(submit_ts, None)  # streaming: no global wait
        if pairing.any_finalizing:
            await pairing.finish_without_more_training()
            return True
        for stream in pairing.streams.values():
            stream.discard_detached_submission_cut()
        return False

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
        trajectories = self.trajectories(rollout_id)
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
            driver.emit("rl_full_parameter_final_ack", acknowledged=self.finished)

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
