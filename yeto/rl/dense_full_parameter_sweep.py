"""Materialized correctness backend for atomic dense-DiLoCo policy sweeps.

The generic wire state machine owns reconnect/replay and sequential fragment
commit. This module adds the canonical :class:`ParameterCut` validation used
by unit oracles and small deterministic replay. Production Miles uses the
same wire machine with reference-backed Ray fragments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ..protocol import DTYPE_F32, FinalManifest, SyncerClient
from ..tensor_io import unpack_fragment
from .dense_sweep_wire import (
    DenseFragmentSubmission,
    DenseSweepClient,
    DenseSweepConfig,
    DenseSweepWire,
    PendingDenseWirePolicy,
)
from .local_learner import (
    DenseTrainerUpdate,
    ParameterCut,
    ParameterLayout,
    _validate_dense_update,
    _validate_parameter_cut,
    dense_sweep_session_contract_hash,
    parameter_cut_from_fragment_flats,
)


@dataclass(frozen=True)
class PendingDensePolicy:
    """One complete authoritative cut awaiting safe-boundary model apply."""

    cut: ParameterCut
    fragment_versions: tuple[int, ...]
    submissions: tuple[DenseFragmentSubmission, ...]
    terminal_manifest: FinalManifest | None
    _wire_pending: PendingDenseWirePolicy = field(repr=False, compare=False)

    @property
    def terminal(self) -> bool:
        return self.terminal_manifest is not None


class DenseFullParameterSweep:
    """Validate materialized H=1 updates around one generic dense wire sweep."""

    def __init__(
        self,
        layout: ParameterLayout,
        initial: ParameterCut,
        config: DenseSweepConfig,
        *,
        client: DenseSweepClient | None = None,
        learner_generations: Mapping[int, int] | None = None,
    ) -> None:
        _validate_parameter_cut(layout, initial)
        if initial.policy_version != 0:
            raise ValueError("dense sweep currently requires a version-zero start")
        self.layout = layout
        self.config = config
        self.initial = initial
        if client is None:
            if learner_generations is None:
                raise ValueError("dense syncer sessions require the complete learner roster")
            if learner_generations.get(config.learner_id) != config.learner_generation:
                raise ValueError("dense learner generation does not match the fixed roster")
            client = SyncerClient(
                config.syncer_addr,
                config.learner_id,
                layout.fragments,
                dtype=DTYPE_F32,
                num_streams=config.wan_streams,
                max_reconnects=None,
                session_contract_hash=dense_sweep_session_contract_hash(
                    layout,
                    policy_rounds=config.policy_rounds,
                    learner_generations=learner_generations,
                ),
            )
        self.wire = DenseSweepWire(layout.fragments, config, client=client)
        self.current: ParameterCut | None = None
        self.pending: PendingDensePolicy | None = None

    @property
    def num_fragments(self) -> int:
        return self.wire.num_fragments

    @property
    def total_fragment_steps(self) -> int:
        return self.wire.total_fragment_steps

    def start(self) -> ParameterCut:
        payloads = self.wire.start(
            {
                fragment.fragment_id: (
                    lambda fragment=fragment: (_tensor_bytes(fragment.flat),)
                )
                for fragment in self.initial.fragments
            }
        )
        authoritative = parameter_cut_from_fragment_flats(
            self.layout,
            policy_version=0,
            fragments={
                fragment_id: unpack_fragment(
                    self.layout.fragments.fragments[fragment_id],
                    payload,
                    DTYPE_F32,
                )
                for fragment_id, payload in enumerate(payloads)
            },
        )
        if authoritative.policy_hash != self.initial.policy_hash:
            raise RuntimeError("syncer version-zero cut differs from the checkpoint")
        self.current = authoritative
        return authoritative

    def exchange(self, update: DenseTrainerUpdate) -> PendingDensePolicy:
        """Submit a complete local cut, but do not advance until Miles applies."""

        if self.pending is not None:
            raise RuntimeError("previous dense policy is not committed")
        if self.current is None:
            raise RuntimeError("dense sweep has no committed policy")
        _validate_dense_update(self.layout, update)
        manifest = update.manifest
        receipt = update.receipt
        base_policy_version = self.current.policy_version
        target_policy_version = base_policy_version + 1
        if (
            manifest.learner_id != self.config.learner_id
            or manifest.learner_generation != self.config.learner_generation
            or manifest.base_policy_version != base_policy_version
            or manifest.target_policy_version != target_policy_version
            or manifest.parameter_layout_hash != self.layout.layout_hash
            or receipt.learner_id != self.config.learner_id
            or receipt.learner_generation != self.config.learner_generation
            or receipt.base_policy_version != base_policy_version
            or receipt.base_policy_hash != self.current.policy_hash
            or receipt.optimizer_steps != 1
            or not receipt.optimizer_step_succeeded
        ):
            raise ValueError("dense trainer update does not bind this H=1 round")
        wire_pending = self.wire.exchange(
            base_policy_version=base_policy_version,
            trained_tokens=receipt.trained_tokens,
            sweep_update_id=manifest.payload_hash,
            delta_parts={
                fragment.fragment_id: (
                    lambda fragment=fragment: (
                        _tensor_bytes(fragment.target_minus_base),
                    )
                )
                for fragment in update.fragments
            },
        )
        cut = parameter_cut_from_fragment_flats(
            self.layout,
            policy_version=wire_pending.policy_version,
            fragments={
                fragment_id: unpack_fragment(
                    self.layout.fragments.fragments[fragment_id],
                    payload,
                    DTYPE_F32,
                )
                for fragment_id, payload in enumerate(wire_pending.payloads)
            },
        )
        self.pending = PendingDensePolicy(
            cut,
            wire_pending.fragment_versions,
            wire_pending.submissions,
            wire_pending.terminal_manifest,
            wire_pending,
        )
        return self.pending

    def commit_applied(self, pending: PendingDensePolicy) -> ParameterCut:
        if self.pending is None or pending is not self.pending:
            raise RuntimeError("dense policy commit does not match the pending cut")
        if (
            self.current is None
            or pending.cut.policy_version != self.current.policy_version + 1
        ):
            raise RuntimeError("dense policy commit is not monotonic")
        self.wire.commit_applied(pending._wire_pending)
        self.current = pending.cut
        self.pending = None
        return self.current

    def close(self) -> None:
        self.wire.close()


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError("dense sweep tensor is not canonical CPU FP32")
    return memoryview(value.numpy()).cast("B")
