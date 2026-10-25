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
    FullParameterPolicyState,
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)
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
    # S27 G-AG lead cap (``--sweep-max-local-rounds-ahead``); None = off.
    # Island-local: not part of the session contract or the syncer profile.
    max_local_rounds_ahead: int | None = None

    def __post_init__(self) -> None:
        self.profile()

    def profile(self) -> FullParameterSyncProfile:
        return FullParameterSyncProfile.from_configs(
            self.by_role(), sweep=self.sweep, max_local_rounds_ahead=self.max_local_rounds_ahead)

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
    pairing = RolePairing(streams, config.profile())
    return FullParameterSync(
        pairing,
        groups=groups,
        run=runner.run,
        trajectories=lambda rollout_id: read_round_evidence(miles_args, rollout_id),
    )


class MilesFullParameterPolicyState(FullParameterPolicyState):
    """``PolicyState`` of a Miles full-parameter island: the actor stream's cut
    (:class:`~yeto.rl.engine.full_parameter_sync.FullParameterPolicyState`)."""
