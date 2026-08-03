"""Yeto-owned strict global-policy boundary around an island RL runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from ..protocol import DTYPE_F32, PullRequest, SyncerClient
from ..tensor_io import pack_tensor, unpack_fragment
from .cache import CachedResult, ResultCache
from .core import (
    CanonicalLoraState,
    CanonicalTensorSpec,
    PolicyIdentity,
    build_avg_layout,
    canonical_state,
    flat_tensor,
    policy_delta,
    tensors_from_flat,
)


@dataclass(frozen=True)
class LocalRoundResult:
    groups: int
    samples_per_group: int
    optimizer_steps: int
    rollout_identities: frozenset[PolicyIdentity]
    rollout_seconds: float = 0.0
    train_seconds: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


class IslandRuntime(Protocol):
    def initialize(self) -> Mapping[str, torch.Tensor]: ...

    def cancel_or_drain_rollouts(self) -> None: ...

    def apply_global_policy(self, state: CanonicalLoraState) -> None: ...

    def run_local_round(
        self,
        policy_identity: PolicyIdentity,
        *,
        groups: int,
        samples_per_group: int,
        optimizer_steps: int,
    ) -> LocalRoundResult: ...

    def export_local_policy(self) -> Mapping[str, torch.Tensor]: ...

    def read_trainer_policy_identity(self) -> PolicyIdentity: ...

    def read_rollout_policy_identity(self) -> PolicyIdentity: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class BridgeConfig:
    syncer_addr: tuple[str, int]
    learner_id: int
    run_manifest_sha256: str
    groups_per_round: int
    samples_per_group: int
    local_optimizer_steps: int
    cache_dir: Path
    run_id: str
    event_tape: Path
    wan_streams: int = 4
    round_timeout_s: float = 0.0
    expected_specs: tuple[CanonicalTensorSpec, ...] | None = None
    expected_layout_fingerprint: str | None = None
    audit_dir: Path | None = None

    def __post_init__(self) -> None:
        if min(
            self.groups_per_round,
            self.samples_per_group,
            self.local_optimizer_steps,
        ) <= 0:
            raise ValueError("RL G/K/U values must be positive")
        if not self.run_id:
            raise ValueError("RL run ID cannot be empty")


class StrictRlBridge:
    def __init__(self, runtime: IslandRuntime, config: BridgeConfig) -> None:
        self.runtime = runtime
        self.config = config
        initial = canonical_state(
            0,
            runtime.initialize(),
            expected_specs=config.expected_specs,
            expected_layout_fingerprint=config.expected_layout_fingerprint,
        )
        self.specs = initial.specs
        self.layout = build_avg_layout(self.specs)
        self.initial = initial
        timeout = config.round_timeout_s if config.round_timeout_s > 0 else math.inf
        self.client = SyncerClient(
            config.syncer_addr,
            config.learner_id,
            self.layout,
            dtype=DTYPE_F32,
            num_streams=config.wan_streams,
            finalization_timeout=timeout,
        )
        self.cache = ResultCache(
            config.cache_dir,
            run_manifest_sha256=config.run_manifest_sha256,
            learner_id=config.learner_id,
            layout_fingerprint=initial.layout_fingerprint,
        )
        self.current: CanonicalLoraState | None = None
        self.permits: dict[int, PullRequest] = {}

    def _append_event(self, event: Mapping[str, Any]) -> None:
        path = Path(self.config.event_tape).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _f32_bytes(value: torch.Tensor) -> bytes:
        tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(tensor).all().item():
            raise ValueError("RL audit tensor contains NaN or Inf")
        return tensor.numpy().astype("<f4", copy=False).tobytes()

    @staticmethod
    def _replace_bytes(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _record_round_audit(
        self,
        *,
        base: CanonicalLoraState,
        cached: CachedResult,
    ) -> None:
        """Persist opt-in f32 inputs for an independent strict-average oracle."""

        if self.config.audit_dir is None:
            return
        directory = Path(self.config.audit_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"round-{cached.target_step:08d}"
        base_bytes = self._f32_bytes(flat_tensor(base.tensors, base.specs))
        delta_bytes = self._f32_bytes(cached.delta)
        metadata = {
            "schema": 1,
            "run_manifest_sha256": self.config.run_manifest_sha256,
            "learner_id": self.config.learner_id,
            "base_version": base.policy_version,
            "base_policy_hash": base.policy_hash,
            "target_step": cached.target_step,
            "layout_fingerprint": base.layout_fingerprint,
            "numel": cached.delta.numel(),
            "base_f32_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "delta_sha256": cached.delta_sha256,
        }
        files = {
            directory / f"{stem}.base.f32": base_bytes,
            directory / f"{stem}.delta.f32": delta_bytes,
            directory / f"{stem}.json": (
                json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            ),
        }
        for path, data in files.items():
            if path.exists():
                if path.read_bytes() != data:
                    raise RuntimeError(f"RL audit record changed at {path}")
                continue
            self._replace_bytes(path, data)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _record_committed_round(
        self,
        previous: CanonicalLoraState,
        committed: CanonicalLoraState,
        trainer: PolicyIdentity,
        rollout: PolicyIdentity,
        received_at_ns: int,
    ) -> None:
        cached = self.cache.load(
            base_identity=previous.identity,
            target_step=committed.policy_version,
            expected_numel=sum(spec.numel for spec in self.specs),
        )
        if cached is None:
            return
        first_push = cached.first_push_unix_ns or received_at_ns
        self._append_event(
            {
                "schema": 1,
                "event": "rl_island_round",
                "run_id": self.config.run_id,
                "run_manifest_sha256": self.config.run_manifest_sha256,
                "learner_id": self.config.learner_id,
                "base_version": previous.policy_version,
                "base_policy_hash": previous.policy_hash,
                "committed_version": committed.policy_version,
                "committed_policy_hash": committed.policy_hash,
                **cached.stats,
                "delta_sha256": cached.delta_sha256,
                "trainer_applied_identity": {
                    "version": trainer.version,
                    "policy_hash": trainer.policy_hash,
                },
                "rollout_applied_identity": {
                    "version": rollout.version,
                    "policy_hash": rollout.policy_hash,
                },
                "sync_wait_seconds": max(
                    0.0, (received_at_ns - first_push) / 1_000_000_000
                ),
                "cache_resend_count": max(0, cached.push_attempts - 1),
            }
        )

    def run(self) -> CanonicalLoraState:
        try:
            self.client.start()
            if self.config.learner_id == 0:
                self.client.send_init(
                    0,
                    pack_tensor(flat_tensor(self.initial.tensors), DTYPE_F32),
                )
            while True:
                self.client.check_health()
                if self.client.finalizing.is_set():
                    return self._finalize()
                progressed = self._drain_messages()
                permit = self._ready_permit()
                if permit is not None:
                    self._run_or_resend(permit)
                    progressed = True
                if not progressed:
                    time.sleep(0.05)
        finally:
            try:
                self.runtime.shutdown()
            finally:
                self.client.close()

    def _drain_messages(self) -> bool:
        progressed = False
        for update in self.client.drain_updates():
            progressed = True
            received_at_ns = time.time_ns()
            if update.fragment_id != 0:
                raise RuntimeError("strict RL received a nonzero fragment")
            if self.current is not None and update.version != self.current.policy_version + 1:
                raise RuntimeError(
                    "strict RL policy jumped from "
                    f"{self.current.policy_version} to {update.version}"
                )
            flat = unpack_fragment(self.layout.fragments[0], update.data, DTYPE_F32)
            state = canonical_state(
                update.version,
                tensors_from_flat(flat, self.specs),
                expected_specs=self.specs,
                expected_layout_fingerprint=self.initial.layout_fingerprint,
            )
            trainer, rollout = self._apply_and_verify(state)
            previous = self.current
            if previous is not None:
                self._record_committed_round(
                    previous, state, trainer, rollout, received_at_ns
                )
            self.current = state
            self.cache.clear_if_committed(state.policy_version)
            self.permits = {
                step: permit for step, permit in self.permits.items() if step > update.version
            }
        for permit in self.client.drain_pulls():
            progressed = True
            if permit.fragment_id != 0 or permit.round_attempt != 1:
                raise RuntimeError("strict RL received an invalid PULL permit")
            if permit.global_step <= (self.current.policy_version if self.current else -1):
                continue
            previous = self.permits.get(permit.global_step)
            if previous is not None and previous != permit:
                raise RuntimeError("strict RL received conflicting PULL permits")
            self.permits[permit.global_step] = permit
        return progressed

    def _ready_permit(self) -> PullRequest | None:
        if self.current is None:
            return None
        return self.permits.pop(self.current.policy_version + 1, None)

    def _run_or_resend(self, permit: PullRequest) -> None:
        base = self.current
        if base is None or permit.global_step != base.policy_version + 1:
            raise RuntimeError("strict RL attempted a round without its exact base policy")
        expected_numel = sum(spec.numel for spec in self.specs)
        cached = self.cache.load(
            base_identity=base.identity,
            target_step=permit.global_step,
            expected_numel=expected_numel,
        )
        if cached is None:
            result = self.runtime.run_local_round(
                base.identity,
                groups=self.config.groups_per_round,
                samples_per_group=self.config.samples_per_group,
                optimizer_steps=self.config.local_optimizer_steps,
            )
            expected_identity = frozenset({base.identity})
            if (
                result.groups != self.config.groups_per_round
                or result.samples_per_group != self.config.samples_per_group
                or result.optimizer_steps != self.config.local_optimizer_steps
                or result.rollout_identities != expected_identity
                or any(
                    not math.isfinite(duration) or duration < 0
                    for duration in (result.rollout_seconds, result.train_seconds)
                )
            ):
                raise RuntimeError(
                    "island runtime violated the fixed G/K/U or policy identity contract"
                )
            local = canonical_state(
                base.policy_version,
                self.runtime.export_local_policy(),
                expected_specs=self.specs,
                expected_layout_fingerprint=base.layout_fingerprint,
            )
            delta = policy_delta(local, base)
            stats = {
                "groups": result.groups,
                "samples_per_group": result.samples_per_group,
                "trajectories": result.groups * result.samples_per_group,
                "optimizer_steps": result.optimizer_steps,
                "rollout_identity_set": [
                    {
                        "version": identity.version,
                        "policy_hash": identity.policy_hash,
                    }
                    for identity in sorted(
                        result.rollout_identities,
                        key=lambda identity: (identity.version, identity.policy_hash),
                    )
                ],
                "local_policy_hash": local.policy_hash,
                "delta_norm": torch.linalg.vector_norm(delta.double()).item(),
                "rollout_seconds": result.rollout_seconds,
                "train_seconds": result.train_seconds,
                "runtime_stats": result.stats,
            }
            cached = self.cache.save(
                base_identity=base.identity,
                target_step=permit.global_step,
                delta=delta,
                stats=stats,
            )
        self._record_round_audit(base=base, cached=cached)
        cached = self.cache.record_push(cached)
        self.client.push_fragment(
            0,
            permit.global_step,
            permit.round_attempt,
            base.policy_version,
            permit.global_step,
            1,
            1,
            pack_tensor(cached.delta, DTYPE_F32),
        )

    def _apply_and_verify(
        self, state: CanonicalLoraState
    ) -> tuple[PolicyIdentity, PolicyIdentity]:
        self.runtime.cancel_or_drain_rollouts()
        self.runtime.apply_global_policy(state)
        trainer = self.runtime.read_trainer_policy_identity()
        rollout = self.runtime.read_rollout_policy_identity()
        if trainer != state.identity or rollout != state.identity:
            raise RuntimeError(
                f"global policy apply mismatch: expected {state.identity}, "
                f"trainer={trainer}, rollout={rollout}"
            )
        return trainer, rollout

    def _finalize(self) -> CanonicalLoraState:
        timeout = self.config.round_timeout_s if self.config.round_timeout_s > 0 else None
        manifest, fragments = self.client.wait_for_final_fragments(timeout=timeout)
        if len(fragments) != 1 or manifest.versions != (manifest.global_step,):
            raise RuntimeError("strict RL received an inconsistent final manifest")
        flat = unpack_fragment(self.layout.fragments[0], fragments[0].data, DTYPE_F32)
        final = canonical_state(
            manifest.global_step,
            tensors_from_flat(flat, self.specs),
            expected_specs=self.specs,
            expected_layout_fingerprint=self.initial.layout_fingerprint,
        )
        received_at_ns = time.time_ns()
        trainer, rollout = self._apply_and_verify(final)
        if self.current is not None and final.policy_version == self.current.policy_version + 1:
            self._record_committed_round(
                self.current, final, trainer, rollout, received_at_ns
            )
        self.cache.clear()
        acknowledged_generation = -1
        while not self.client.shutdown.is_set():
            self.client.check_health()
            generation = self.client.connection_generation
            if generation != acknowledged_generation:
                replayed, _ = self.client.wait_for_final_fragments(timeout=timeout)
                if replayed != manifest:
                    raise RuntimeError("final manifest changed while awaiting shutdown")
                self.client.acknowledge_finalization(replayed, timeout=timeout)
                acknowledged_generation = generation
            self.client.shutdown.wait(0.1)
        return final
