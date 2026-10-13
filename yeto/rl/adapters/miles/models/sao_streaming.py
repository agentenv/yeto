"""Full-parameter SAO over role-isolated streaming DiLoCo sessions.

The actor and critic deliberately use independent syncer processes.  Miles
represents each role as a complete, contiguous full-parameter cut, while one
syncer process owns exactly one layout and session.  Paired fragment updates
may remain in flight across multiple local SAO rounds; received role updates
are staged and are applied only as matching actor/critic logical pairs at a
Miles train/publication-safe boundary.  Only actor content is ever published
to rollout workers.

The pairing guarantee is fail-stop and publication-atomic, not crash-atomic
across the two independent syncers.  A permanent one-role failure or durable
checkpoint skew closes both streams and requires operator recovery; this
module does not claim a two-phase commit or durable two-role rollback.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from yeto.protocol import (
    PartialMessageGenerationLost,
)
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleFragmentSubmission,
    RoleRoundResult,
    RoleStreamConfig,
    RoleSubmissionCandidate,
)
from yeto.rl.trajectory_evidence import (
    TrajectoryBatchEvidence,
    read_trajectory_batch_evidence,
    trajectory_batch_evidence_path,
)

@dataclass(frozen=True)
class MilesSaoStreamingConfig:
    """Actor and critic streams that form one externally visible SAO learner."""

    actor: RoleStreamConfig
    critic: RoleStreamConfig

    def __post_init__(self) -> None:
        if self.actor.role != "actor" or self.critic.role != "critic":
            raise ValueError("SAO streaming configuration must contain both roles")
        if self.actor.syncer_addr == self.critic.syncer_addr:
            raise ValueError("SAO actor and critic require distinct syncer ports")
        lockstep = (
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
        if any(
            getattr(self.actor, name) != getattr(self.critic, name) for name in lockstep
        ):
            raise ValueError(
                "SAO actor and critic streams must use one lockstep schedule"
            )


@dataclass
class _PairedSubmission:
    """Retained actor/critic operands for one independently in-flight pair."""

    actor: RoleSubmissionCandidate
    critic: RoleSubmissionCandidate
    actor_submission: RoleFragmentSubmission | None = None
    critic_submission: RoleFragmentSubmission | None = None
    retained_at: float = field(default_factory=time.monotonic)

    @property
    def logical_key(self) -> tuple[int, int, int]:
        return self.actor.logical_key

    @property
    def complete(self) -> bool:
        return self.actor_submission is not None and self.critic_submission is not None


@dataclass(frozen=True)
class _PendingPublication:
    rollout_id: int
    actor_content_hash: str
    actor_fragment_versions: tuple[int, ...]
    critic_fragment_versions: tuple[int, ...]
    actor_result: RoleRoundResult
    critic_result: RoleRoundResult
    terminal: bool


class MilesSaoStreamingPolicySync:
    """Nonblocking paired-stream callback with fail-stop cross-role safety.

    Pair submission and WAN completion overlap later local SAO rounds up to
    ``pipeline_depth``.  Broadcasts are staged per role, then matching logical
    actor/critic fragments are applied at a Miles callback boundary.  The two
    applies cannot be made crash-atomic across independent syncers/processes;
    any apply skew or permanent role failure therefore fail-stops both roles.
    """

    supports_critic = True
    requires_exact_publication_info = True

    def __init__(self, args: Any, config: MilesSaoStreamingConfig) -> None:
        self.args = args
        self.config = config
        self.actor = MilesFullParameterRoleStream(config.actor)
        self.critic = MilesFullParameterRoleStream(config.critic)
        self.actor_model = None
        self.critic_model = None
        self.rollout_manager = None
        self.initial_publication_pending = False
        self.pending_publication: _PendingPublication | None = None
        self.published_rollout_id = 0
        self.published_actor_hash: str | None = None
        self.finished = False
        self._inflight_pairs: dict[tuple[int, int, int], _PairedSubmission] = {}
        self._paired_journal: list[
            tuple[RoleFragmentSubmission, RoleFragmentSubmission]
        ] = []
        self._pull_skew_since: dict[tuple[int, int, int], float] = {}
        self._terminal_seen = False
        self._paired_failure: BaseException | None = None

    async def initialize(self, *, actor_model, critic_model, rollout_manager) -> None:
        if os.environ.get("MILES_EXPERIMENTAL_FT_TRAINER", "0") != "0":
            raise RuntimeError("SAO full-parameter streaming requires Miles v1 actors")
        if self.actor_model is not None:
            raise RuntimeError("SAO streaming synchronizer initialized twice")
        if int(getattr(self.args, "start_rollout_id", 0)) != 0:
            raise RuntimeError("SAO streaming currently requires a version-zero start")
        if getattr(self.args, "yeto_rl_trajectory_evidence_dir", None) is None:
            raise RuntimeError("SAO streaming requires immutable trajectory evidence")
        await self.actor.initialize(actor_model)
        try:
            await self.critic.initialize(critic_model)
        except BaseException:
            self.actor.close()
            raise
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.rollout_manager = rollout_manager
        self.initial_publication_pending = True
        self.published_rollout_id = 0
        self.args.start_rollout_id = 0

    async def after_local_train(
        self,
        *,
        rollout_id: int,
        actor_model,
        critic_model,
        rollout_data,
    ) -> bool:
        del rollout_data
        if (
            actor_model is not self.actor_model
            or critic_model is not self.critic_model
            or self.initial_publication_pending
            or self.pending_publication is not None
            or self.finished
            or rollout_id != self.published_rollout_id
            or self.published_actor_hash != self.actor.content_hash
        ):
            raise RuntimeError("Miles called SAO streaming outside its safe boundary")
        from yeto.rl.adapters.miles.publish import get_current_published_policy_identity

        if get_current_published_policy_identity(
            self.args,
            expected_policy_version=rollout_id,
        ) != (rollout_id, self.published_actor_hash):
            raise RuntimeError("SAO rollout does not bind the published actor")
        trajectories = self._trajectory_evidence(rollout_id)
        actor_result, critic_result = await self._complete_local_rounds(
            rollout_id=rollout_id,
            trajectories=trajectories,
        )
        actor_submissions, critic_submissions = await self._submit_paired_ready()
        actor_applied, critic_applied = await self._commit_staged_paired_broadcasts()
        actor_result = self._extend_result(
            actor_result,
            submissions=actor_submissions,
            applied=actor_applied,
        )
        critic_result = self._extend_result(
            critic_result,
            submissions=critic_submissions,
            applied=critic_applied,
        )
        terminal = (
            self.actor.finalizing or self.critic.finalizing or self._terminal_seen
        )
        if terminal:
            (
                actor_result,
                critic_result,
            ) = await self._finish_paired_without_more_training(
                actor_result,
                critic_result,
            )
        else:
            self.actor.discard_detached_submission_cut()
            self.critic.discard_detached_submission_cut()
        actor_versions = self.actor.fragment_versions
        critic_versions = self.critic.fragment_versions
        if actor_versions != critic_versions:
            error = RuntimeError(
                "SAO actor and critic committed fragment cuts diverged"
            )
            self._fail_stop(error)
            raise error
        pending = _PendingPublication(
            rollout_id=rollout_id + 1,
            actor_content_hash=self.actor.content_hash,
            actor_fragment_versions=actor_versions,
            critic_fragment_versions=critic_versions,
            actor_result=actor_result,
            critic_result=critic_result,
            terminal=terminal,
        )
        self.pending_publication = pending
        self._record_round(pending)
        return terminal

    async def _complete_local_rounds(
        self,
        *,
        rollout_id: int,
        trajectories: TrajectoryBatchEvidence,
    ) -> tuple[RoleRoundResult, RoleRoundResult]:
        """Cancel the peer callback and fail-stop both streams on role failure."""

        tasks = (
            asyncio.create_task(
                self.actor.complete_local_round(
                    rollout_id=rollout_id,
                    trajectories=trajectories,
                )
            ),
            asyncio.create_task(
                self.critic.complete_local_round(
                    rollout_id=rollout_id,
                    trajectories=trajectories,
                )
            ),
        )
        try:
            actor_result, critic_result = await asyncio.gather(*tasks)
            return actor_result, critic_result
        except BaseException as error:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._fail_stop(error)
            raise

    def _fail_stop(self, error: BaseException) -> None:
        if self._paired_failure is None:
            self._paired_failure = error
        for stream in (self.actor, self.critic):
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as close_error:  # noqa: BLE001 - fail-stop cleanup
                    error.add_note(
                        f"{getattr(stream, 'role', 'role')} close failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )

    @staticmethod
    def _extend_result(
        result: RoleRoundResult,
        *,
        submissions: Sequence[RoleFragmentSubmission] = (),
        applied: Sequence[int] = (),
    ) -> RoleRoundResult:
        merged_submissions = result.submissions + tuple(submissions)
        return RoleRoundResult(
            result.role,
            result.receipt,
            tuple(dict.fromkeys(result.applied_fragments + tuple(applied))),
            merged_submissions,
            result.terminal_submitted,
        )

    def _new_pair(
        self,
        actor: RoleSubmissionCandidate,
        critic: RoleSubmissionCandidate,
    ) -> _PairedSubmission:
        if (
            actor.logical_key != critic.logical_key
            or actor.local_step != critic.local_step
            or actor.c_steps != critic.c_steps
            or actor.c_tokens != critic.c_tokens
            or actor.c_steps < self.config.actor.local_horizon
            or critic.c_steps < self.config.critic.local_horizon
        ):
            raise RuntimeError(
                "SAO actor and critic retained cuts are not one H-qualified round"
            )
        return _PairedSubmission(actor, critic)

    @staticmethod
    def _same_candidate(
        retained: RoleSubmissionCandidate,
        replayed: RoleSubmissionCandidate,
    ) -> bool:
        return (
            retained.logical_key == replayed.logical_key
            and replayed.permit.round_attempt >= retained.permit.round_attempt
            and retained.local_step == replayed.local_step
            and retained.c_steps == replayed.c_steps
            and retained.c_tokens == replayed.c_tokens
            and retained.payload_bytes == replayed.payload_bytes
            and retained.submission_cut is replayed.submission_cut
            and retained.anchor is replayed.anchor
        )

    @property
    def _effective_pipeline_depth(self) -> int:
        return min(
            self.config.actor.pipeline_depth,
            self.config.actor.expected_fragments,
        )

    @staticmethod
    def _candidate_map(
        candidates: Sequence[RoleSubmissionCandidate],
        *,
        role: str,
    ) -> dict[tuple[int, int, int], RoleSubmissionCandidate]:
        result = {}
        for candidate in candidates:
            if candidate.logical_key in result:
                raise RuntimeError(f"SAO {role} returned duplicate logical PULLs")
            result[candidate.logical_key] = candidate
        return result

    def _retain_pair(self, pair: _PairedSubmission) -> None:
        retained_actor = False
        try:
            retain = getattr(self.actor, "retain_candidate", None)
            if callable(retain):
                retain(pair.actor)
            retained_actor = True
            retain = getattr(self.critic, "retain_candidate", None)
            if callable(retain):
                retain(pair.critic)
        except BaseException:
            if retained_actor:
                release = getattr(self.actor, "release_candidate", None)
                if callable(release):
                    release(pair.actor)
            raise

    def _release_pair(self, pair: _PairedSubmission) -> None:
        for stream, candidate in (
            (self.actor, pair.actor),
            (self.critic, pair.critic),
        ):
            release = getattr(stream, "release_candidate", None)
            if callable(release):
                release(candidate)

    @staticmethod
    def _retained_replay(
        retained: RoleSubmissionCandidate,
        observed: RoleSubmissionCandidate | None,
    ) -> RoleSubmissionCandidate | None:
        if observed is None:
            return None
        if (
            observed.logical_key != retained.logical_key
            or observed.permit.round_attempt < retained.permit.round_attempt
        ):
            raise RuntimeError("SAO reconnect replay changed logical identity")
        # The current model may have advanced through more local rounds while
        # this PUSH was in flight.  Only the new permit is consumed: bytes,
        # counters, cut, and anchor remain the exact retained operands.
        return replace(retained, permit=observed.permit)

    def _push_pair_once(
        self,
        pair: _PairedSubmission,
        *,
        actor_observed: RoleSubmissionCandidate | None,
        critic_observed: RoleSubmissionCandidate | None,
    ) -> tuple[RoleFragmentSubmission | None, RoleFragmentSubmission | None]:
        produced: list[RoleFragmentSubmission | None] = []
        for role_name, stream, observed in (
            ("actor", self.actor, actor_observed),
            ("critic", self.critic, critic_observed),
        ):
            retained = getattr(pair, role_name)
            replayed = self._retained_replay(retained, observed)
            if replayed is None:
                produced.append(None)
                continue
            setattr(pair, role_name, replayed)
            # A PULL visible after a previous successful enqueue proves that
            # generation did not commit it.  Replay the exact retained bytes
            # under the newly observed attempt.
            setattr(pair, f"{role_name}_submission", None)
            try:
                submission = stream.push_candidate(replayed)
            except PartialMessageGenerationLost:
                submission = None
            if submission is not None:
                setattr(pair, f"{role_name}_submission", submission)
            produced.append(submission)
        return produced[0], produced[1]

    async def _submit_paired_ready(
        self,
    ) -> tuple[tuple[RoleFragmentSubmission, ...], tuple[RoleFragmentSubmission, ...]]:
        """Nonblockingly fill the paired pipeline from currently visible PULLs."""

        if self._paired_failure is not None:
            raise RuntimeError(
                "SAO paired submission coordinator is fail-stopped"
            ) from (self._paired_failure)
        actor_submissions: list[RoleFragmentSubmission] = []
        critic_submissions: list[RoleFragmentSubmission] = []
        try:
            actor_by_key = self._candidate_map(
                self.actor.ready_submission_candidates(),
                role="actor",
            )
            critic_by_key = self._candidate_map(
                self.critic.ready_submission_candidates(),
                role="critic",
            )
            now = time.monotonic()
            total = self.config.actor.total_fragment_steps
            if any(key[0] == total for key in actor_by_key | critic_by_key):
                self._terminal_seen = True

            # Existing pairs consume only a replayed permit.  Their original
            # reference-backed operands stay pinned even after later local
            # training boundaries replace the role's current cut.
            for key, pair in tuple(self._inflight_pairs.items()):
                actor_submission, critic_submission = self._push_pair_once(
                    pair,
                    actor_observed=actor_by_key.pop(key, None),
                    critic_observed=critic_by_key.pop(key, None),
                )
                if actor_submission is not None:
                    actor_submissions.append(actor_submission)
                if critic_submission is not None:
                    critic_submissions.append(critic_submission)

            capacity = self._effective_pipeline_depth - len(self._inflight_pairs)
            shared = sorted(actor_by_key.keys() & critic_by_key.keys())
            for key in shared[: max(0, capacity)]:
                pair = self._new_pair(actor_by_key.pop(key), critic_by_key.pop(key))
                self._retain_pair(pair)
                self._inflight_pairs[key] = pair
                actor_submission, critic_submission = self._push_pair_once(
                    pair,
                    actor_observed=pair.actor,
                    critic_observed=pair.critic,
                )
                if actor_submission is not None:
                    actor_submissions.append(actor_submission)
                if critic_submission is not None:
                    critic_submissions.append(critic_submission)

            unmatched = set(actor_by_key) ^ set(critic_by_key)
            self._pull_skew_since = {
                key: self._pull_skew_since.get(key, now) for key in unmatched
            }
            timeout = min(
                self.config.actor.wait_timeout,
                self.config.critic.wait_timeout,
            )
            if any(
                now - started >= timeout for started in self._pull_skew_since.values()
            ):
                raise TimeoutError(
                    "SAO timed out waiting for a peer role's matching PULL"
                )
            for pair in self._inflight_pairs.values():
                if now - pair.retained_at < timeout:
                    continue
                if pair.complete:
                    raise TimeoutError(
                        "SAO timed out waiting for a paired WAN broadcast"
                    )
                raise TimeoutError(
                    "SAO timed out with a one-sided paired submission in flight"
                )
            return tuple(actor_submissions), tuple(critic_submissions)
        except BaseException as error:
            self._fail_stop(error)
            raise

    async def _commit_staged_paired_broadcasts(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Apply currently matched role broadcasts without waiting for WAN I/O."""

        actor_applied: list[int] = []
        critic_applied: list[int] = []
        try:
            await asyncio.gather(
                self.actor.stage_available_broadcasts(),
                self.critic.stage_available_broadcasts(),
            )
            known = {
                (pair.actor.permit.fragment_id, pair.actor.permit.global_step)
                for pair in self._inflight_pairs.values()
            }
            actor_staged = set(self.actor.staged_fragment_versions)
            critic_staged = set(self.critic.staged_fragment_versions)
            unknown = (actor_staged | critic_staged) - known
            if unknown:
                raise RuntimeError(
                    "SAO role syncer broadcast has no retained paired submission"
                )

            for key, pair in sorted(self._inflight_pairs.items()):
                step, fragment_id, _base = key
                if step == self.config.actor.total_fragment_steps:
                    continue
                actor_ready = self.actor.has_staged_broadcast(fragment_id, step)
                critic_ready = self.critic.has_staged_broadcast(fragment_id, step)
                if not (actor_ready and critic_ready):
                    continue
                if not pair.complete:
                    raise RuntimeError(
                        "SAO broadcast arrived before both paired PUSHes were retained"
                    )
                actor_update, critic_update = await asyncio.gather(
                    self.actor.apply_staged_broadcast(fragment_id, step),
                    self.critic.apply_staged_broadcast(fragment_id, step),
                )
                actor_applied.extend(actor_update)
                critic_applied.extend(critic_update)
                if (
                    self.actor.fragment_versions[fragment_id] != step
                    or self.critic.fragment_versions[fragment_id] != step
                ):
                    raise RuntimeError("SAO paired apply committed mismatched versions")
                assert pair.actor_submission is not None
                assert pair.critic_submission is not None
                self._paired_journal.append(
                    (pair.actor_submission, pair.critic_submission)
                )
                self._release_pair(pair)
                self._inflight_pairs.pop(key)
            return (
                tuple(dict.fromkeys(actor_applied)),
                tuple(dict.fromkeys(critic_applied)),
            )
        except BaseException as error:
            self._fail_stop(error)
            raise

    async def _finish_paired_without_more_training(
        self,
        actor_result: RoleRoundResult,
        critic_result: RoleRoundResult,
    ) -> tuple[RoleRoundResult, RoleRoundResult]:
        """Reach both final manifests without manufacturing another train step."""

        deadline = time.monotonic() + min(
            self.config.actor.wait_timeout,
            self.config.critic.wait_timeout,
        )
        while not (self.actor.finalizing and self.critic.finalizing):
            actor_submissions, critic_submissions = await self._submit_paired_ready()
            (
                actor_applied,
                critic_applied,
            ) = await self._commit_staged_paired_broadcasts()
            actor_result = self._extend_result(
                actor_result,
                submissions=actor_submissions,
                applied=actor_applied,
            )
            critic_result = self._extend_result(
                critic_result,
                submissions=critic_submissions,
                applied=critic_applied,
            )
            if self.actor.finalizing and self.critic.finalizing:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "SAO paired terminal barrier could not reach both final cuts"
                )
            await asyncio.sleep(
                min(
                    self.config.actor.poll_seconds,
                    self.config.critic.poll_seconds,
                )
            )
        try:
            await asyncio.gather(
                self.actor.wait_and_apply_final(),
                self.critic.wait_and_apply_final(),
            )
            if self.actor.fragment_versions != self.critic.fragment_versions:
                raise RuntimeError("SAO actor and critic final manifests diverged")
            for key, pair in sorted(self._inflight_pairs.items()):
                if not pair.complete:
                    raise RuntimeError(
                        "SAO finalization retained an incomplete paired submission"
                    )
                assert pair.actor_submission is not None
                assert pair.critic_submission is not None
                self._paired_journal.append(
                    (pair.actor_submission, pair.critic_submission)
                )
                self._release_pair(pair)
                self._inflight_pairs.pop(key)
        except BaseException as error:
            self._fail_stop(error)
            raise
        return (
            RoleRoundResult(
                actor_result.role,
                actor_result.receipt,
                actor_result.applied_fragments,
                actor_result.submissions,
                bool(
                    getattr(
                        self.actor,
                        "terminal_submitted",
                        actor_result.terminal_submitted,
                    )
                ),
            ),
            RoleRoundResult(
                critic_result.role,
                critic_result.receipt,
                critic_result.applied_fragments,
                critic_result.submissions,
                bool(
                    getattr(
                        self.critic,
                        "terminal_submitted",
                        critic_result.terminal_submitted,
                    )
                ),
            ),
        )

    async def after_inference_publication(
        self,
        *,
        rollout_id: int | None,
        actor_model,
        publication_info=None,
    ) -> None:
        if actor_model is not self.actor_model:
            raise RuntimeError("SAO publication used the wrong actor group")
        if rollout_id is None:
            if (
                not self.initial_publication_pending
                or self.pending_publication is not None
            ):
                raise RuntimeError("unexpected initial SAO inference publication")
            await self._publish_identity(
                0,
                self.actor.content_hash,
                publication_info=publication_info,
            )
            self.published_actor_hash = self.actor.content_hash
            self.initial_publication_pending = False
            self._record_publication(terminal=False)
            return
        pending = self.pending_publication
        if pending is None or rollout_id + 1 != pending.rollout_id:
            raise RuntimeError("SAO inference publication has no matching trainer cut")
        await self._publish_identity(
            pending.rollout_id,
            pending.actor_content_hash,
            publication_info=publication_info,
        )
        self.published_rollout_id = pending.rollout_id
        self.published_actor_hash = pending.actor_content_hash
        if pending.terminal:
            failures = await asyncio.gather(
                self._acknowledge_role(self.actor),
                self._acknowledge_role(self.critic),
                return_exceptions=True,
            )
            errors = [value for value in failures if isinstance(value, BaseException)]
            self.finished = bool(
                not errors
                and self.actor.final_acknowledged
                and self.critic.final_acknowledged
            )
            if errors:
                failure = RuntimeError(
                    f"SAO terminal ACK publication failed for {len(errors)} role(s)"
                )
                for error in errors:
                    failure.add_note(f"{type(error).__name__}: {error}")
                raise failure from errors[0]
        self.pending_publication = None
        self._record_publication(terminal=pending.terminal)

    @staticmethod
    async def _acknowledge_role(stream: MilesFullParameterRoleStream) -> None:
        stream.acknowledge_final()

    async def _publish_identity(
        self,
        rollout_id: int,
        policy_hash: str,
        *,
        publication_info,
    ) -> None:
        token = f"yeto:{rollout_id}:{policy_hash}"
        if publication_info is None:
            raise RuntimeError(
                "SAO streaming requires exact publication worker evidence"
            )
        engines: Sequence[Any] = tuple(getattr(publication_info, "rollout_engines", ()))
        if not engines:
            raise RuntimeError(
                "SAO streaming publication evidence has no inference worker"
            )
        await asyncio.gather(
            *(engine.update_weight_version.remote(token) for engine in engines)
        )
        identity = await self.rollout_manager.set_external_policy_identity.remote(
            rollout_id,
            policy_hash,
        )
        if identity != (rollout_id, policy_hash):
            raise RuntimeError(
                "rollout process installed the wrong SAO policy identity"
            )
        from yeto.rl.adapters.miles.publish import set_current_published_policy_identity

        set_current_published_policy_identity(
            self.args,
            policy_version=rollout_id,
            policy_hash=policy_hash,
        )
        self.args.yeto_rl_policy_token = token

    def _trajectory_evidence(self, rollout_id: int) -> TrajectoryBatchEvidence:
        directory = getattr(self.args, "yeto_rl_trajectory_evidence_dir", None)
        if type(directory) is not str:
            raise RuntimeError("SAO trajectory evidence directory is unavailable")
        path = trajectory_batch_evidence_path(directory, rollout_id)
        if not path.exists():
            raise RuntimeError("Miles did not persist this SAO rollout evidence")
        evidence = read_trajectory_batch_evidence(path)
        if evidence.rollout_id != rollout_id:
            raise RuntimeError("SAO trajectory evidence rollout identity changed")
        return evidence

    def _record_round(self, pending: _PendingPublication) -> None:
        self._append_event(
            {
                "event": "rl_sao_streaming_round",
                "rollout_id": pending.rollout_id - 1,
                "target_rollout_id": pending.rollout_id,
                "actor_policy_hash": pending.actor_content_hash,
                "actor_fragment_versions": list(pending.actor_fragment_versions),
                "critic_fragment_versions": list(pending.critic_fragment_versions),
                "actor_applied_fragments": list(pending.actor_result.applied_fragments),
                "critic_applied_fragments": list(
                    pending.critic_result.applied_fragments
                ),
                "actor_submitted_fragments": [
                    value.fragment_id for value in pending.actor_result.submissions
                ],
                "critic_submitted_fragments": [
                    value.fragment_id for value in pending.critic_result.submissions
                ],
                "terminal": pending.terminal,
            }
        )

    def _record_publication(self, *, terminal: bool) -> None:
        self._append_event(
            {
                "event": "rl_sao_streaming_publication",
                "rollout_id": self.published_rollout_id,
                "actor_policy_hash": self.published_actor_hash,
                "actor_fragment_versions": list(self.actor.fragment_versions),
                "critic_fragment_versions": list(self.critic.fragment_versions),
                "terminal": terminal,
            }
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        from yeto.rl.adapters.miles.legacy.engine import _append_rl_event

        _append_rl_event(self.args, event)

    async def finalize(self) -> None:
        if (
            not self.finished
            or self.initial_publication_pending
            or self.pending_publication is not None
            or not self.actor.final_acknowledged
            or not self.critic.final_acknowledged
        ):
            raise RuntimeError("Miles stopped before SAO streaming finalization")
        self.actor.close()
        self.critic.close()


def create_miles_sao_streaming_sync(args) -> MilesSaoStreamingPolicySync:
    config = getattr(args, "yeto_rl_sao_streaming_config", None)
    if not isinstance(config, MilesSaoStreamingConfig):
        raise TypeError("Miles SAO streaming configuration is missing")
    return MilesSaoStreamingPolicySync(args, config)
