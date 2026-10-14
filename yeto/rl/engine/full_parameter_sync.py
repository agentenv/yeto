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
sweep.  This module is engine-neutral; it never imports an engine adapter.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from yeto.protocol import PartialMessageGenerationLost

from .driver import SyncBoundary, SyncStart

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

    def __post_init__(self) -> None:
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("full-parameter sync needs distinct trainable roles")
        if self.roles[0] != "actor":
            raise ValueError("the actor is the published role and must come first")
        if self.sweep and (self.roles != ("actor",) or self.local_horizon != 1):
            raise ValueError("a dense strict sweep has one actor role and H=1")

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
        )


def require_lockstep(configs: Sequence[Any]) -> None:
    """Rule 1: every role shares one schedule and owns its own syncer."""

    first = configs[0]
    if len({config.syncer_addr for config in configs}) != len(configs):
        raise ValueError("full-parameter roles require distinct syncer ports")
    for config in configs[1:]:
        if any(getattr(config, name) != getattr(first, name) for name in LOCKSTEP_FIELDS):
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
        self.terminal_seen = False
        self.failure: BaseException | None = None
        self._pull_skew_since: dict[LogicalKey, float] = {}

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
            if any(key[0] == self.profile.total_fragment_steps for keys in by_role.values() for key in keys):
                self.terminal_seen = True
            for key, pair in tuple(self.inflight.items()):
                observed = {role: by_role[role].pop(key, None) for role in self.streams}
                self._push_once(pair, observed, produced)
            capacity = self.profile.effective_pipeline_depth - len(self.inflight)
            shared = sorted(set.intersection(*(set(keys) for keys in by_role.values())))
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

    def _check_timeouts(self, by_role: Mapping[str, Mapping[LogicalKey, Any]], now: float) -> None:
        seen = set().union(*(set(keys) for keys in by_role.values()))
        everywhere = set.intersection(*(set(keys) for keys in by_role.values()))
        unmatched = seen - everywhere
        self._pull_skew_since = {key: self._pull_skew_since.get(key, now) for key in unmatched}
        timeout = self.profile.wait_timeout
        if any(now - started >= timeout for started in self._pull_skew_since.values()):
            raise TimeoutError("timed out waiting for a peer role's matching PULL")
        for pair in self.inflight.values():
            if now - pair.retained_at < timeout:
                continue
            if pair.complete:
                raise TimeoutError("timed out waiting for a paired WAN broadcast")
            raise TimeoutError("timed out with a one-sided paired submission in flight")

    # -- apply (rule 3) ------------------------------------------------------

    async def commit_staged_broadcasts(self) -> dict[str, RoleProgress]:
        """Apply matched role broadcasts without waiting for WAN I/O."""

        applied: dict[str, list[int]] = {role: [] for role in self.streams}
        try:
            await asyncio.gather(*(stream.stage_available_broadcasts() for stream in self.streams.values()))
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
        self, key: LogicalKey, pair: PairedSubmission, applied: dict[str, list[int]]
    ) -> None:
        step, fragment_id, _base = key
        if not pair.complete:
            raise RuntimeError("broadcast arrived before every paired PUSH was retained")
        results = await asyncio.gather(
            *(stream.apply_staged_broadcast(fragment_id, step) for stream in self.streams.values())
        )
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
                if not pair.complete:
                    raise RuntimeError("finalization retained an incomplete paired submission")
                self._journal(pair)
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


# LEAVE reason of a clean finish (P5 code 0, FINISHED).  P5 owns the full code
# table; this session only forwards the code it is closed with.
# TODO(P5 rl-island-graceful-leave): once yeto/rl/engine/leave_reason.py lands,
# replace this constant with LeaveReason.FINISHED (an IntEnum equal to 0).
REASON_FINISHED = 0


class FullParameterSync:
    """The ports ``SyncSession`` for full-parameter training (every algorithm).

    The driver trains one local round, then :meth:`boundary` records it on
    every role, submits and applies through :class:`RolePairing`, and returns
    the actor's cut as a trainer-resident handle that the driver publishes
    with its normal publication check.  A strict sweep profile blocks in the
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

    @property
    def actor(self) -> RoleStream:
        return self.pairing.streams["actor"]

    def start(self, driver: IslandDriver) -> SyncStart:
        self._driver = driver
        self.run(self.pairing.gather_roles(lambda role, stream: stream.initialize(self.groups[role])))
        return SyncStart(state=self.actor.resident_state(0), rollout_id=0)

    def boundary(self, driver: IslandDriver, *, rollout_id: int, stats: Any) -> SyncBoundary:
        del stats
        self.terminal = self.run(self._boundary(rollout_id))
        versions = self.pairing.fragment_versions()
        state = self.actor.resident_state(rollout_id + 1)
        driver.emit(
            "rl_full_parameter_round",
            policy_version=rollout_id + 1,
            terminal=self.terminal,
            **{
                "sync/fragment_versions": list(versions),
                "sync/global_policy_hash": state.policy_tensor_hash(),
                "sync/inflight_pairs": len(self.pairing.inflight),
            },
        )
        return SyncBoundary(state=state, stop=self.terminal)

    async def _boundary(self, rollout_id: int) -> bool:
        pairing = self.pairing
        trajectories = self.trajectories(rollout_id)
        await pairing.gather_roles(
            lambda _role, stream: stream.complete_local_round(
                rollout_id=rollout_id, trajectories=trajectories
            )
        )
        before = pairing.fragment_versions()
        pairing.submit_ready()
        await pairing.commit_staged_broadcasts()
        if pairing.profile.sweep:
            await self._wait_for_sweep(before)
        if pairing.any_finalizing:
            await pairing.finish_without_more_training()
            return True
        for stream in pairing.streams.values():
            stream.discard_detached_submission_cut()
        return False

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

    def close(self, reason: int = REASON_FINISHED) -> None:
        """Close every role once.  Full-parameter syncers run in legacy mode,
        which has no LEAVE message, so the reason is only recorded."""

        if self.closed:
            return
        self.closed = True
        failures = self.pairing.close_all()
        if self._driver is not None:
            self._driver.emit(
                "rl_sync_closed",
                **{"sync/leave_reason": int(reason), "sync/finished": self.finished,
                   **({"sync/close_failures": failures} if failures else {})},
            )
        if failures:
            raise RuntimeError("full-parameter role close failed: " + "; ".join(failures))
