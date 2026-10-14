"""Miles wiring of the ports full-parameter sync (P0-D8).

``island_entry`` stores one :class:`FullParameterSyncConfig` on the Miles
args; :func:`build_full_parameter_sync` turns it into the engine-neutral
:class:`~yeto.rl.engine.full_parameter_sync.FullParameterSync` over one
:class:`MilesFullParameterRoleStream` per trainable role.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from yeto.rl.adapters.miles.full_parameter.role_stream import (
    MilesFullParameterRoleStream,
    RoleStreamConfig,
)
from yeto.rl.engine.full_parameter_sync import (
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)
from yeto.rl.engine.policy_digest import TrainerResidentState
from yeto.rl.trajectory_evidence import (
    TrajectoryBatchEvidence,
    read_trajectory_batch_evidence,
    trajectory_batch_evidence_path,
)

CONFIG_ATTRIBUTE = "yeto_rl_full_parameter_config"


@dataclass(frozen=True)
class FullParameterSyncConfig:
    """Every trainable role's stream config, actor first, plus the profile kind."""

    roles: tuple[RoleStreamConfig, ...]
    sweep: bool = False

    def __post_init__(self) -> None:
        names = tuple(config.role for config in self.roles)
        FullParameterSyncProfile.from_configs(dict(zip(names, self.roles, strict=True)), sweep=self.sweep)

    def by_role(self) -> dict[str, RoleStreamConfig]:
        return {config.role: config for config in self.roles}


def read_round_evidence(args: Any, rollout_id: int) -> TrajectoryBatchEvidence:
    """The immutable trajectory evidence Miles persisted for ``rollout_id``."""

    directory = getattr(args, "yeto_rl_trajectory_evidence_dir", None)
    if type(directory) is not str:
        raise RuntimeError("Miles trajectory evidence directory is unavailable")
    path = trajectory_batch_evidence_path(directory, rollout_id)
    if not path.exists():
        raise RuntimeError("Miles did not persist this rollout's trajectory evidence")
    evidence = read_trajectory_batch_evidence(path)
    if evidence.rollout_id != rollout_id:
        raise RuntimeError("trajectory evidence rollout identity changed")
    return evidence


def build_full_parameter_sync(
    miles_args: Any, *, runner: Any, groups: Mapping[str, Any]
) -> FullParameterSync:
    config = getattr(miles_args, CONFIG_ATTRIBUTE, None)
    if not isinstance(config, FullParameterSyncConfig):
        raise TypeError("Miles full-parameter sync configuration is missing")
    configs = config.by_role()
    streams = {role: MilesFullParameterRoleStream(stream) for role, stream in configs.items()}
    pairing = RolePairing(streams, FullParameterSyncProfile.from_configs(configs, sweep=config.sweep))
    return FullParameterSync(
        pairing,
        groups=groups,
        run=runner.run,
        trajectories=lambda rollout_id: read_round_evidence(miles_args, rollout_id),
    )


class MilesFullParameterPolicyState:
    """``PolicyState`` of a full-parameter island: the actor stream's cut.

    Tensors never leave the trainer (P0-D1).  Applying a cut happens only
    inside the role streams, so :meth:`apply` refuses.
    """

    def __init__(self, sync: FullParameterSync) -> None:
        self._sync = sync
        self.policy_version = 0

    @property
    def _actor(self) -> MilesFullParameterRoleStream:
        return self._sync.actor  # type: ignore[return-value]

    @property
    def layout_hash(self) -> str:
        if self._actor.adapter is None:
            raise RuntimeError("full-parameter layout requested before the sync started")
        return self._actor.adapter.layout.layout_hash

    def export(self, *, policy_version: int | None = None) -> TrainerResidentState:
        return self._actor.resident_state(self.policy_version if policy_version is None else policy_version)

    def export_digest(self, *, policy_version: int | None = None) -> TrainerResidentState:
        return self.export(policy_version=policy_version)

    def apply(self, state: Any, *, optimizer: str, local_step: int) -> None:
        raise RuntimeError("full-parameter cuts are applied only through the role streams")
