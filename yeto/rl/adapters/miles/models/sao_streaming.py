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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleRoundResult,
    RoleStreamConfig,
)
from yeto.rl.engine.full_parameter_sync import (
    FullParameterSyncProfile,
    RoleProgress,
    RolePairing,
    require_lockstep,
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
        require_lockstep((self.actor, self.critic))


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
    """Miles callback for SAO's actor and critic full-parameter role streams.

    The engine-neutral :class:`RolePairing` owns submission, paired apply and
    fail-stop.  This class binds it to Miles' training callbacks and publishes
    only actor content to rollout workers.  The two role applies cannot be
    crash-atomic across independent syncers; any apply skew or permanent role
    failure therefore fail-stops both roles.
    """

    supports_critic = True
    requires_exact_publication_info = True

    def __init__(self, args: Any, config: MilesSaoStreamingConfig) -> None:
        self.args = args
        self.config = config
        self.actor = MilesFullParameterRoleStream(config.actor)
        self.critic = MilesFullParameterRoleStream(config.critic)
        self.pairing: RolePairing | None = None
        self.actor_model = None
        self.critic_model = None
        self.rollout_manager = None
        self.initial_publication_pending = False
        self.pending_publication: _PendingPublication | None = None
        self.published_rollout_id = 0
        self.published_actor_hash: str | None = None
        self.finished = False

    def _new_pairing(self) -> RolePairing:
        profile = FullParameterSyncProfile.from_configs(
            {"actor": self.config.actor, "critic": self.config.critic}
        )
        return RolePairing({"actor": self.actor, "critic": self.critic}, profile)

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
        self.pairing = self._new_pairing()
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
        pairing = self.pairing
        if (
            pairing is None
            or actor_model is not self.actor_model
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
        results = await pairing.gather_roles(
            lambda _role, stream: stream.complete_local_round(
                rollout_id=rollout_id, trajectories=trajectories
            )
        )
        progress = _merge(pairing.submit_ready(), await pairing.commit_staged_broadcasts())
        terminal = pairing.any_finalizing
        if terminal:
            progress = _merge(progress, await pairing.finish_without_more_training())
        else:
            for stream in pairing.streams.values():
                stream.discard_detached_submission_cut()
        results = {
            role: _with_progress(result, progress[role], pairing.streams[role])
            for role, result in results.items()
        }
        versions = pairing.fragment_versions()
        pending = _PendingPublication(
            rollout_id=rollout_id + 1,
            actor_content_hash=self.actor.content_hash,
            actor_fragment_versions=versions,
            critic_fragment_versions=versions,
            actor_result=results["actor"],
            critic_result=results["critic"],
            terminal=terminal,
        )
        self.pending_publication = pending
        self._record_round(pending)
        return terminal

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
            assert self.pairing is not None
            self.finished = self.pairing.acknowledge_final()
        self.pending_publication = None
        self._record_publication(terminal=pending.terminal)

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


def _merge(
    left: Mapping[str, RoleProgress], right: Mapping[str, RoleProgress]
) -> dict[str, RoleProgress]:
    return {role: left[role].extend(right[role]) for role in left}


def _with_progress(
    result: RoleRoundResult, progress: RoleProgress, stream: Any
) -> RoleRoundResult:
    return RoleRoundResult(
        result.role,
        result.receipt,
        tuple(dict.fromkeys(result.applied_fragments + progress.applied_fragments)),
        result.submissions + progress.submissions,
        bool(getattr(stream, "terminal_submitted", result.terminal_submitted)),
    )


def create_miles_sao_streaming_sync(args) -> MilesSaoStreamingPolicySync:
    config = getattr(args, "yeto_rl_sao_streaming_config", None)
    if not isinstance(config, MilesSaoStreamingConfig):
        raise TypeError("Miles SAO streaming configuration is missing")
    return MilesSaoStreamingPolicySync(args, config)
