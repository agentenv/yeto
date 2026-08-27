"""Pipelined fragment synchronization for Miles RL safe boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..protocol import DTYPE_F32, FinalManifest, PullRequest, SyncerClient
from ..tensor_io import (
    apply_fragment,
    fragment_flat,
    pack_fragment,
    pack_tensor,
    unpack_fragment,
)
from .core import (
    CanonicalLoraState,
    CanonicalTensorSpec,
    build_rl_fragment_layout,
    canonical_state,
)


@dataclass(frozen=True)
class DecoupledBridgeConfig:
    syncer_addr: tuple[str, int]
    learner_id: int
    total_fragment_steps: int
    num_fragments: int
    pipeline: int
    local_horizon: int
    expected_specs: tuple[CanonicalTensorSpec, ...]
    base_model_revision: str
    lora_config_hash: str
    canonical_layout_hash: str
    wan_streams: int = 4
    learner_budget_steps: int | None = None

    def __post_init__(self) -> None:
        if self.num_fragments < 2:
            raise ValueError("decoupled RL requires at least 2 fragments")
        if not 1 <= self.pipeline <= self.num_fragments:
            raise ValueError("decoupled RL pipeline must be between 1 and fragments")
        if (
            self.total_fragment_steps < self.num_fragments
            or self.total_fragment_steps % self.num_fragments
        ):
            raise ValueError("decoupled RL requires complete fragment sweeps")
        if self.local_horizon < 2:
            raise ValueError("decoupled RL local horizon must be at least 2")
        if self.learner_budget_steps is not None and self.learner_budget_steps < 1:
            raise ValueError("decoupled RL learner budget must be positive")


@dataclass(frozen=True)
class BroadcastReceipt:
    fragment_id: int
    version: int
    anchor: torch.Tensor = field(repr=False, compare=False)
    payload_bytes: int
    queue_seconds: float


@dataclass(frozen=True)
class BroadcastBatch:
    state: CanonicalLoraState
    broadcasts: tuple[BroadcastReceipt, ...]

    @property
    def fragment_ids(self) -> tuple[int, ...]:
        return tuple(value.fragment_id for value in self.broadcasts)

    @property
    def bytes_received(self) -> int:
        return sum(value.payload_bytes for value in self.broadcasts)


@dataclass(frozen=True)
class InitialCut:
    state: CanonicalLoraState
    fragment_versions: tuple[int, ...]


@dataclass(frozen=True)
class FragmentSubmission:
    fragment_id: int
    global_step: int
    round_attempt: int
    base_version: int
    c_steps: int
    c_tokens: int
    delta_l2_norm: float
    payload_bytes: int
    pull_to_push_seconds: float


@dataclass(frozen=True)
class BudgetConsolidation:
    manifest: FinalManifest
    state: CanonicalLoraState
    submissions: tuple[FragmentSubmission, ...]
    bytes_received: int


class DecoupledRlBridge:
    """Own one island's raw anchors, permits, and fragment protocol client."""

    def __init__(
        self,
        initial: CanonicalLoraState,
        config: DecoupledBridgeConfig,
    ) -> None:
        self.config = config
        self.initial = canonical_state(
            0,
            initial.tensors,
            base_model_revision=config.base_model_revision,
            lora_config_hash=config.lora_config_hash,
            layout_hash=config.canonical_layout_hash,
            expected_specs=config.expected_specs,
        )
        self.layout = build_rl_fragment_layout(
            self.initial.specs,
            config.num_fragments,
        )
        self.client = SyncerClient(
            config.syncer_addr,
            config.learner_id,
            self.layout,
            dtype=DTYPE_F32,
            num_streams=config.wan_streams,
            max_reconnects=(None if config.learner_budget_steps is not None else 0),
        )
        count = self.layout.num_fragments
        self._anchors: list[torch.Tensor | None] = [None] * count
        self._fragment_versions: list[int | None] = [None] * count
        self._steps_at_anchor = [0] * count
        self._tokens_at_anchor = [0] * count
        self._pending: dict[int, PullRequest] = {}
        self._answered: set[tuple[int, int]] = set()
        self.startup_final_manifest: FinalManifest | None = None
        self.final_payload_bytes_received = 0

    @property
    def fragment_versions(self) -> tuple[int, ...]:
        if any(version is None for version in self._fragment_versions):
            raise RuntimeError("decoupled RL fragment cut is not initialized")
        return tuple(int(version) for version in self._fragment_versions)

    @property
    def steps_at_anchor(self) -> tuple[int, ...]:
        return tuple(self._steps_at_anchor)

    @property
    def tokens_at_anchor(self) -> tuple[int, ...]:
        return tuple(self._tokens_at_anchor)

    @property
    def finalizing(self) -> bool:
        if self.client.finalizing.is_set():
            return True
        expected = tuple(
            range(
                self.config.total_fragment_steps - self.layout.num_fragments + 1,
                self.config.total_fragment_steps + 1,
            )
        )
        return (
            all(version is not None for version in self._fragment_versions)
            and tuple(self._fragment_versions) == expected
        )

    def start(self) -> None:
        self.client.start()
        if self.config.learner_id == 0:
            params = dict(self.initial.tensors)
            for fragment_id, fragment in enumerate(self.layout.fragments):
                self.client.send_init(
                    fragment_id,
                    pack_fragment(fragment, params, DTYPE_F32),
                )

    def wait_for_initial_cut(
        self,
        *,
        optimizer_steps: int,
        action_tokens: int,
    ) -> InitialCut:
        state = self.initial
        versions: list[int | None] = [None] * self.layout.num_fragments
        while any(version is None for version in versions):
            self.client.check_health()
            if self.client.finalizing.is_set():
                return self._recover_startup_final_cut(
                    policy_version=optimizer_steps,
                )
            batch = self._drain_broadcast_updates(state, versions)
            state = batch.state
            for update in batch.broadcasts:
                versions[update.fragment_id] = update.version
            self._queue_pulls()
            if self.client.finalizing.is_set():
                return self._recover_startup_final_cut(
                    policy_version=optimizer_steps,
                )
            if any(version is None for version in versions):
                time.sleep(0.05)
        return InitialCut(state, tuple(int(version) for version in versions))

    def _recover_startup_final_cut(
        self,
        *,
        policy_version: int,
    ) -> InitialCut:
        manifest, state = self.wait_for_final_cut(policy_version=policy_version)
        self.startup_final_manifest = manifest
        return InitialCut(state, manifest.versions)

    def commit_initial_cut(
        self,
        cut: InitialCut,
        *,
        optimizer_steps: int,
        action_tokens: int,
    ) -> None:
        if optimizer_steps < 0 or action_tokens < 0:
            raise ValueError("decoupled RL progress must be non-negative")
        if any(version is not None for version in self._fragment_versions):
            raise RuntimeError("decoupled RL initial cut is already committed")
        if len(cut.fragment_versions) != self.layout.num_fragments:
            raise RuntimeError("decoupled RL initial cut is incomplete")
        state = self._canonical(cut.state)
        params = dict(state.tensors)
        for fragment_id, fragment in enumerate(self.layout.fragments):
            version = cut.fragment_versions[fragment_id]
            self._validate_fragment_version(fragment_id, version)
            self._anchors[fragment_id] = fragment_flat(fragment, params).cpu().clone()
            self._fragment_versions[fragment_id] = version
            self._steps_at_anchor[fragment_id] = optimizer_steps
            self._tokens_at_anchor[fragment_id] = action_tokens

    def drain_broadcasts(
        self,
        local: CanonicalLoraState,
        *,
        optimizer_steps: int,
        action_tokens: int,
    ) -> BroadcastBatch:
        if optimizer_steps < 0 or action_tokens < 0:
            raise ValueError("decoupled RL progress must be non-negative")
        return self._drain_broadcast_updates(local, self._fragment_versions)

    def _drain_broadcast_updates(
        self,
        local: CanonicalLoraState,
        versions: list[int | None],
    ) -> BroadcastBatch:
        self.client.check_health()
        state = self._canonical(local)
        params = dict(state.tensors)
        staged_versions = list(versions)
        broadcasts = []
        drained_at = time.monotonic()
        updates = sorted(
            self.client.drain_updates(),
            key=lambda update: (update.fragment_id, update.version),
        )
        for update in updates:
            fragment_id = update.fragment_id
            self._validate_fragment_version(fragment_id, update.version)
            previous = staged_versions[fragment_id]
            if previous is not None and update.version <= previous:
                continue
            fragment = self.layout.fragments[fragment_id]
            flat = unpack_fragment(fragment, update.data, DTYPE_F32)
            apply_fragment(fragment, flat, params)
            staged_versions[fragment_id] = update.version
            broadcasts.append(
                BroadcastReceipt(
                    fragment_id,
                    update.version,
                    flat.clone(),
                    len(update.data),
                    max(0.0, drained_at - update.received_at),
                )
            )
        return BroadcastBatch(
            self._canonical_from_tensors(state.policy_version, params),
            tuple(broadcasts),
        )

    def commit_broadcasts(
        self,
        batch: BroadcastBatch,
        *,
        optimizer_steps: int,
        action_tokens: int,
    ) -> None:
        if optimizer_steps < 0 or action_tokens < 0:
            raise ValueError("decoupled RL progress must be non-negative")
        for update in batch.broadcasts:
            previous = self._fragment_versions[update.fragment_id]
            if previous is not None and update.version <= previous:
                raise RuntimeError("decoupled RL broadcast batch is no longer monotonic")
            self._anchors[update.fragment_id] = update.anchor.clone()
            self._fragment_versions[update.fragment_id] = update.version
            self._steps_at_anchor[update.fragment_id] = optimizer_steps
            self._tokens_at_anchor[update.fragment_id] = action_tokens

    def submit_ready(
        self,
        local: CanonicalLoraState,
        *,
        optimizer_steps: int,
        action_tokens: int,
    ) -> tuple[FragmentSubmission, ...]:
        self.client.check_health()
        if self.finalizing:
            return ()
        state = self._canonical(local)
        self._queue_pulls()
        submissions = []
        for fragment_id in sorted(self._pending):
            permit = self._pending[fragment_id]
            version = self._fragment_versions[fragment_id]
            expected_base = max(0, permit.global_step - self.layout.num_fragments)
            if version != expected_base:
                continue
            c_steps = optimizer_steps - self._steps_at_anchor[fragment_id]
            if c_steps < self.config.local_horizon:
                continue
            c_tokens = action_tokens - self._tokens_at_anchor[fragment_id]
            if c_tokens < 0:
                raise RuntimeError("decoupled RL action-token counter moved backwards")
            anchor = self._anchors[fragment_id]
            if anchor is None:
                continue
            fragment = self.layout.fragments[fragment_id]
            delta = fragment_flat(fragment, dict(state.tensors)).cpu() - anchor
            payload = pack_tensor(delta, DTYPE_F32)
            self.client.push_fragment(
                fragment_id,
                permit.global_step,
                permit.round_attempt,
                version,
                optimizer_steps,
                c_steps,
                c_tokens,
                payload,
            )
            self._answered.add((permit.global_step, permit.round_attempt))
            self._pending.pop(fragment_id)
            pull_to_push_seconds = time.monotonic() - permit.received_at
            submissions.append(
                FragmentSubmission(
                    fragment_id,
                    permit.global_step,
                    permit.round_attempt,
                    version,
                    c_steps,
                    c_tokens,
                    float(delta.norm().item()),
                    len(payload),
                    pull_to_push_seconds,
                )
            )
        return tuple(submissions)

    def wait_for_final_cut(
        self,
        *,
        policy_version: int,
    ) -> tuple[FinalManifest, CanonicalLoraState]:
        manifest, fragments = self.client.wait_for_final_fragments()
        self.final_payload_bytes_received = sum(
            len(fragment.data) for fragment in fragments
        )
        expected_versions = tuple(
            range(
                self.config.total_fragment_steps - self.layout.num_fragments + 1,
                self.config.total_fragment_steps + 1,
            )
        )
        if (
            manifest.global_step != self.config.total_fragment_steps
            or manifest.versions != expected_versions
        ):
            raise RuntimeError(
                "decoupled RL final cut is not a complete fragment sweep"
            )
        return manifest, self._assemble_final(policy_version, manifest, fragments)

    def consolidate_budget(
        self,
        frozen: CanonicalLoraState,
        *,
        optimizer_steps: int,
        action_tokens: int,
    ) -> BudgetConsolidation:
        target = self.config.learner_budget_steps
        if target is None or optimizer_steps != target or action_tokens < 0:
            raise RuntimeError("decoupled RL reached an invalid learner budget")
        state = self._canonical(frozen)
        generation = self.client.send_budget_done(target)
        self.client.wait_for_budget_restart(generation)
        bases: dict[int, tuple[int, torch.Tensor]] = {}
        completed: set[int] = set()
        submissions = []
        bytes_received = 0
        first_step = None
        while len(completed) < self.layout.num_fragments:
            deadline = time.monotonic() + self.client.finalization_timeout
            permit = None
            permit_received = None
            while True:
                self.client.check_health()
                for update in sorted(
                    self.client.drain_updates(),
                    key=lambda value: (value.fragment_id, value.version),
                ):
                    self._validate_fragment_version(
                        update.fragment_id,
                        update.version,
                    )
                    bytes_received += len(update.data)
                    if update.fragment_id in completed:
                        continue
                    fragment = self.layout.fragments[update.fragment_id]
                    bases[update.fragment_id] = (
                        update.version,
                        unpack_fragment(fragment, update.data, DTYPE_F32),
                    )
                pulls = self.client.drain_pulls()
                if pulls:
                    if permit is not None or len(pulls) != 1:
                        raise RuntimeError(
                            "decoupled budget consolidation received concurrent pulls"
                        )
                    permit = pulls[0]
                    self._validate_permit(permit)
                    permit_received = permit.received_at
                    if permit.fragment_id in completed:
                        raise RuntimeError(
                            "decoupled budget consolidation repeated a fragment"
                        )
                if permit is not None and permit.fragment_id in bases:
                    base_version = bases[permit.fragment_id][0]
                    expected_base = max(
                        0,
                        permit.global_step - self.layout.num_fragments,
                    )
                    if base_version > expected_base:
                        raise RuntimeError(
                            "decoupled budget consolidation received a stale PULL"
                        )
                    if base_version == expected_base:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError("decoupled budget consolidation timed out")
                time.sleep(0.01)

            fragment_id = permit.fragment_id
            if permit_received is None:
                raise RuntimeError("decoupled budget consolidation lost its PULL time")
            if first_step is None:
                first_step = permit.global_step
            base_version, base = bases.pop(fragment_id)
            fragment = self.layout.fragments[fragment_id]
            delta = fragment_flat(fragment, dict(state.tensors)).cpu() - base
            payload = pack_tensor(delta, DTYPE_F32)
            self.client.push_fragment(
                fragment_id,
                permit.global_step,
                permit.round_attempt,
                base_version,
                target,
                target,
                action_tokens,
                payload,
            )
            submissions.append(
                FragmentSubmission(
                    fragment_id,
                    permit.global_step,
                    permit.round_attempt,
                    base_version,
                    target,
                    action_tokens,
                    float(delta.norm().item()),
                    len(payload),
                    time.monotonic() - permit_received,
                )
            )
            completed.add(fragment_id)

        manifest, fragments = self.client.wait_for_final_fragments()
        bytes_received += sum(len(fragment.data) for fragment in fragments)
        if first_step is None:
            raise RuntimeError("decoupled budget consolidation completed no fragments")
        expected_steps = set(range(first_step, first_step + self.layout.num_fragments))
        if (
            manifest.global_step != first_step + self.layout.num_fragments - 1
            or set(manifest.versions) != expected_steps
            or any(
                version > 0 and (version - 1) % self.layout.num_fragments != fragment_id
                for fragment_id, version in enumerate(manifest.versions)
            )
        ):
            raise RuntimeError("decoupled budget consolidation produced an invalid cut")
        return BudgetConsolidation(
            manifest,
            self._assemble_final(optimizer_steps, manifest, fragments),
            tuple(submissions),
            bytes_received,
        )

    def _assemble_final(
        self,
        policy_version: int,
        manifest: FinalManifest,
        fragments,
    ) -> CanonicalLoraState:
        if [fragment.fragment_id for fragment in fragments] != list(
            range(self.layout.num_fragments)
        ):
            raise RuntimeError("decoupled RL final cut is missing fragments")
        params = dict(self.initial.tensors)
        for fragment in fragments:
            if fragment.version != manifest.versions[fragment.fragment_id]:
                raise RuntimeError("decoupled RL final fragment version mismatch")
            layout_fragment = self.layout.fragments[fragment.fragment_id]
            flat = unpack_fragment(layout_fragment, fragment.data, DTYPE_F32)
            apply_fragment(layout_fragment, flat, params)
        return self._canonical_from_tensors(policy_version, params)

    def acknowledge_finalization(self, manifest: FinalManifest) -> None:
        self.client.acknowledge_finalization(manifest)

    def close(self) -> None:
        self.client.close()

    def _queue_pulls(self) -> None:
        for permit in self.client.drain_pulls():
            self._validate_permit(permit)
            key = (permit.global_step, permit.round_attempt)
            if key in self._answered:
                continue
            previous = self._pending.get(permit.fragment_id)
            if previous is not None and previous != permit:
                raise RuntimeError("decoupled RL received conflicting fragment permits")
            if previous is None:
                self._pending[permit.fragment_id] = permit

    def _validate_permit(self, permit: PullRequest) -> None:
        if (
            not 0 <= permit.fragment_id < self.layout.num_fragments
            or permit.global_step < 1
            or permit.global_step > self.config.total_fragment_steps
            or (permit.global_step - 1) % self.layout.num_fragments
            != permit.fragment_id
            or permit.round_attempt < 1
        ):
            raise RuntimeError("decoupled RL received an invalid PULL permit")

    def _validate_fragment_version(self, fragment_id: int, version: int) -> None:
        if (
            not 0 <= fragment_id < self.layout.num_fragments
            or version < 0
            or version > self.config.total_fragment_steps
            or (
                version > 0 and (version - 1) % self.layout.num_fragments != fragment_id
            )
        ):
            raise RuntimeError("decoupled RL received an invalid fragment version")

    def _canonical(self, state: CanonicalLoraState) -> CanonicalLoraState:
        if (
            state.base_model_revision,
            state.lora_config_hash,
            state.layout_hash,
        ) != (
            self.config.base_model_revision,
            self.config.lora_config_hash,
            self.config.canonical_layout_hash,
        ):
            raise RuntimeError("decoupled RL canonical LoRA identity changed")
        return self._canonical_from_tensors(state.policy_version, dict(state.tensors))

    def _canonical_from_tensors(
        self,
        policy_version: int,
        tensors: dict[str, torch.Tensor],
    ) -> CanonicalLoraState:
        return canonical_state(
            policy_version,
            tensors,
            base_model_revision=self.config.base_model_revision,
            lora_config_hash=self.config.lora_config_hash,
            layout_hash=self.config.canonical_layout_hash,
            expected_specs=self.config.expected_specs,
        )
