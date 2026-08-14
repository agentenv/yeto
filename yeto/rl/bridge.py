"""One fixed-roster RL island's PULL/local/PUSH/BCAST loop."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import torch

from ..protocol import DTYPE_F32, FinalManifest, PullRequest, SyncerClient
from ..tensor_io import pack_tensor, unpack_fragment
from .core import (
    CanonicalLoraState,
    CanonicalTensorSpec,
    LocalRoundStats,
    StrictRlInvariantError,
    bounded_tensor_groups,
    build_avg_layout,
    canonical_state,
    canonical_state_from_owned_tensors,
    canonical_state_from_validated_owned_tensors,
    flat_tensor,
    policy_delta,
    tensors_from_flat_owned,
)


class IslandRuntime(Protocol):
    def initialize(self) -> CanonicalLoraState: ...

    def apply_global_policy(self, state: CanonicalLoraState) -> None: ...

    def run_local_round(
        self,
        *,
        expected_policy_version: int,
        groups: int,
        samples_per_group: int,
        optimizer_steps: int,
    ) -> LocalRoundStats: ...

    def export_local_policy(self) -> CanonicalLoraState: ...

    def record_local_round(self, stats: LocalRoundStats) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class BridgeConfig:
    syncer_addr: tuple[str, int]
    learner_id: int
    global_rounds: int
    groups_per_round: int
    samples_per_group: int
    local_optimizer_steps: int
    expected_specs: tuple[CanonicalTensorSpec, ...]
    base_model_revision: str
    lora_config_hash: str
    layout_hash: str
    event_tape: str
    audit_dir: str | None = None
    wan_streams: int = 4


def _write_round_audit(
    directory: str | Path,
    *,
    learner_id: int,
    target_step: int,
    base: CanonicalLoraState,
    delta,
) -> None:
    """Atomically retain the f32 inputs needed by the SSH acceptance oracle."""

    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"round-{target_step:08d}"
    def write_f32(path: Path, tensors) -> str:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as handle:
                for tensor in tensors:
                    value = tensor.detach().to(
                        device="cpu", dtype=torch.float32
                    ).contiguous()
                    if not torch.isfinite(value).all().item():
                        raise ValueError(f"audit tensor {path.name!r} contains NaN or Inf")
                    raw = memoryview(value.numpy()).cast("B")
                    for offset in range(0, len(raw), 64 * 1024 * 1024):
                        chunk = raw[offset : offset + 64 * 1024 * 1024]
                        handle.write(chunk)
                        digest.update(chunk)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return digest.hexdigest()

    base_path = root / f"{stem}.base.f32"
    delta_path = root / f"{stem}.delta.f32"
    base_sha256 = write_f32(
        base_path,
        (base.tensors[spec.name] for spec in base.specs),
    )
    delta_sha256 = write_f32(delta_path, (delta,))
    metadata = {
        "schema": 1,
        "learner_id": learner_id,
        "base_version": base.policy_version,
        "target_step": target_step,
        "base_model_revision": base.base_model_revision,
        "lora_config_hash": base.lora_config_hash,
        "layout_hash": base.layout_hash,
        "specs": [asdict(spec) for spec in base.specs],
        "base_f32_sha256": base_sha256,
        "delta_f32_sha256": delta_sha256,
    }
    path = root / f"{stem}.json"
    payload = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _StreamingRoundAudit:
    """Write the strict acceptance audit while bounded delta groups are live."""

    def __init__(
        self,
        directory: str | Path,
        *,
        learner_id: int,
        target_step: int,
        base: CanonicalLoraState,
    ) -> None:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        stem = f"round-{target_step:08d}"
        self.learner_id = learner_id
        self.target_step = target_step
        self.base_version = base.policy_version
        self.base_model_revision = base.base_model_revision
        self.lora_config_hash = base.lora_config_hash
        self.layout_hash = base.layout_hash
        self.specs = base.specs
        self.base_path = root / f"{stem}.base.f32"
        self.delta_path = root / f"{stem}.delta.f32"
        self.metadata_path = root / f"{stem}.json"
        suffix = f".tmp-{os.getpid()}"
        self.base_temporary = self.base_path.with_name(
            f".{self.base_path.name}{suffix}"
        )
        self.delta_temporary = self.delta_path.with_name(
            f".{self.delta_path.name}{suffix}"
        )
        self.base_digest = hashlib.sha256()
        self.delta_digest = hashlib.sha256()
        self.base_handle = self.base_temporary.open("wb")
        self.delta_handle = self.delta_temporary.open("wb")
        self.closed = False

    @staticmethod
    def _raw(tensor: torch.Tensor) -> memoryview:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError("audit tensor contains NaN or Inf")
        return memoryview(value.numpy()).cast("B")

    def write_group(
        self,
        specs: tuple[CanonicalTensorSpec, ...],
        delta: torch.Tensor,
        base_tensors: dict[str, torch.Tensor],
    ) -> None:
        for spec in specs:
            raw = self._raw(base_tensors[spec.name])
            self.base_handle.write(raw)
            self.base_digest.update(raw)
        raw = self._raw(delta)
        self.delta_handle.write(raw)
        self.delta_digest.update(raw)

    def finish(self) -> None:
        self.base_handle.close()
        self.delta_handle.close()
        os.replace(self.base_temporary, self.base_path)
        os.replace(self.delta_temporary, self.delta_path)
        metadata = {
            "schema": 1,
            "learner_id": self.learner_id,
            "base_version": self.base_version,
            "target_step": self.target_step,
            "base_model_revision": self.base_model_revision,
            "lora_config_hash": self.lora_config_hash,
            "layout_hash": self.layout_hash,
            "specs": [asdict(spec) for spec in self.specs],
            "base_f32_sha256": self.base_digest.hexdigest(),
            "delta_f32_sha256": self.delta_digest.hexdigest(),
        }
        payload = (
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        temporary = self.metadata_path.with_name(
            f".{self.metadata_path.name}.tmp-{os.getpid()}"
        )
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, self.metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
        self.closed = True

    def abort(self) -> None:
        if not self.base_handle.closed:
            self.base_handle.close()
        if not self.delta_handle.closed:
            self.delta_handle.close()
        self.base_temporary.unlink(missing_ok=True)
        self.delta_temporary.unlink(missing_ok=True)


class StrictRlBridge:
    def __init__(self, runtime: IslandRuntime, config: BridgeConfig) -> None:
        self.runtime = runtime
        self.config = config
        initialized = runtime.initialize()
        if (
            initialized.base_model_revision,
            initialized.lora_config_hash,
            initialized.layout_hash,
        ) != (
            config.base_model_revision,
            config.lora_config_hash,
            config.layout_hash,
        ):
            raise StrictRlInvariantError(
                "layout_hash_mismatch",
                "Miles initialized a different canonical LoRA identity",
            )
        normalize = (
            canonical_state_from_owned_tensors
            if getattr(initialized, "_owns_tensor_storage", False)
            else canonical_state
        )
        self.initial: CanonicalLoraState | None = normalize(
            0,
            initialized.tensors,
            base_model_revision=initialized.base_model_revision,
            lora_config_hash=initialized.lora_config_hash,
            layout_hash=initialized.layout_hash,
            expected_specs=config.expected_specs,
        )
        self.specs = self.initial.specs
        self.layout = build_avg_layout(self.specs)
        self.client = SyncerClient(
            config.syncer_addr,
            config.learner_id,
            self.layout,
            dtype=DTYPE_F32,
            num_streams=config.wan_streams,
            # A dead syncer connection makes this island exit. The launcher
            # restarts the same logical ID, which reapplies the committed cut
            # and recomputes any uncommitted local result.
            max_reconnects=0,
        )
        self.current: CanonicalLoraState | None = None
        # Keep the committed version after releasing the model-sized tensor
        # storage.  Strict-AVG may wait for the next cut with no local copy of
        # the base policy resident on the head process.
        self.current_version: int | None = None
        self._terminal_manifest: FinalManifest | None = None
        self._terminal_policy: CanonicalLoraState | None = None
        self.permits: dict[int, PullRequest] = {}
        self.pushed_step: int | None = None

    def run(self) -> CanonicalLoraState:
        try:
            self.start()
            while True:
                self.client.check_health()
                if self.client.finalizing.is_set():
                    return self._finalize()
                progressed = self._drain_messages()
                permit = self._ready_permit()
                if permit is not None:
                    self._run_round(permit)
                    progressed = True
                if not progressed:
                    time.sleep(0.05)
        except StrictRlInvariantError as error:
            self._append_event(
                {
                    "event": "rl_strict_failure",
                    "metric": error.metric,
                    "value": 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[yeto-rl-strict-failure] {error.metric}: {error}",
                file=sys.stderr,
                flush=True,
            )
            raise
        finally:
            try:
                self.runtime.shutdown()
            finally:
                self.client.close()

    def start(self) -> None:
        self.client.start()
        if self.config.learner_id == 0:
            if self.initial is None:
                raise RuntimeError("RL initial policy was released before INIT_PARAMS")
            self.client.send_init(
                0,
                pack_tensor(
                    flat_tensor(self.initial.tensors, self.specs),
                    DTYPE_F32,
                ),
            )

    def wait_for_global_policy(self, version: int) -> CanonicalLoraState:
        while (
            self._committed_version() is None
            or self._committed_version() < version
        ):
            self.client.check_health()
            if self.client.finalizing.is_set():
                manifest, final = self._terminal_state()
                if manifest.global_step != version:
                    raise RuntimeError(
                        "RL terminal policy differs from expected version "
                        f"{version}"
                    )
                self._install_policy(final)
                break
            self._drain_messages()
            if (
                self._committed_version() is None
                or self._committed_version() < version
            ):
                time.sleep(0.05)
        if self.current_version != version or self.current is None:
            raise RuntimeError(
                f"RL policy jumped past expected version {version}"
            )
        return self.current

    def wait_for_initial_policy(self) -> CanonicalLoraState:
        while self.current is None:
            self.client.check_health()
            if self.client.finalizing.is_set():
                _, final = self._terminal_state()
                self._install_policy(final)
                return self.current
            self._drain_messages()
            if self.current is None:
                time.sleep(0.05)
        return self.current

    def wait_for_round(self) -> PullRequest:
        while True:
            self.client.check_health()
            self._drain_messages()
            permit = self._ready_permit()
            if permit is not None:
                return permit
            time.sleep(0.05)

    def _drain_messages(self) -> bool:
        progressed = False
        for update in self.client.drain_updates():
            progressed = True
            if update.fragment_id != 0:
                raise RuntimeError("RL received a nonzero fragment")
            if self.current_version is not None:
                if update.version < self.current_version:
                    continue
                if update.version == self.current_version:
                    continue
                if update.version != self.current_version + 1:
                    raise RuntimeError(
                        f"RL policy jumped from {self.current_version} "
                        f"to {update.version}"
                    )
            state = self._state_from_payload(update.version, update.data)
            self._install_policy(state)

        for permit in self.client.drain_pulls():
            progressed = True
            if permit.fragment_id != 0 or permit.round_attempt != 1:
                raise RuntimeError("RL received an invalid PULL permit")
            if permit.global_step > self.config.global_rounds:
                raise RuntimeError("RL received a PULL beyond configured rounds")
            current_version = (
                self.current_version if self.current_version is not None else -1
            )
            if permit.global_step <= current_version:
                continue
            previous = self.permits.get(permit.global_step)
            if previous is not None and previous != permit:
                raise RuntimeError("RL received conflicting PULL permits")
            self.permits[permit.global_step] = permit
        return progressed

    def _install_policy(self, state: CanonicalLoraState) -> None:
        """Install one authoritative cut without cloning its wire storage."""

        self.runtime.apply_global_policy(state)
        self.current = state
        self.current_version = state.policy_version
        if state.policy_version == 0:
            # INIT_PARAMS is committed and its rebroadcast is now the
            # authoritative base. The private startup copy is dead state.
            self.initial = None
        self.pushed_step = None
        self.permits = {
            step: permit
            for step, permit in self.permits.items()
            if step > state.policy_version
        }

    def _committed_version(self) -> int | None:
        # A few embedders/tests seed ``current`` directly rather than through
        # the receive loop. Adopt that exact state's version once; releases
        # thereafter deliberately keep the scalar while dropping the tensors.
        if self.current_version is None and self.current is not None:
            self.current_version = self.current.policy_version
        return self.current_version

    def release_current(self, expected_version: int) -> None:
        """Release the exact base before receiving the next model-sized cut."""

        if (
            self.current is None
            or self._committed_version() != expected_version
            or self.current.policy_version != expected_version
        ):
            raise RuntimeError(
                "RL attempted to release a policy other than its committed base"
            )
        self.current = None

    def _state_from_payload(self, version: int, data: bytes) -> CanonicalLoraState:
        try:
            flat = unpack_fragment(self.layout.fragments[0], data, DTYPE_F32)
            return canonical_state_from_validated_owned_tensors(
                version,
                tensors_from_flat_owned(flat, self.specs),
                base_model_revision=self.config.base_model_revision,
                lora_config_hash=self.config.lora_config_hash,
                layout_hash=self.config.layout_hash,
                expected_specs=self.specs,
            )
        except (TypeError, ValueError) as error:
            metric = (
                "nonfinite_delta_count"
                if "NaN or Inf" in str(error)
                else "layout_hash_mismatch"
            )
            raise StrictRlInvariantError(metric, str(error)) from error

    def _ready_permit(self) -> PullRequest | None:
        if self.current is None:
            return None
        current_version = self._committed_version()
        if current_version is None:
            return None
        target = current_version + 1
        if self.pushed_step == target:
            self.permits.pop(target, None)
            return None
        return self.permits.pop(target, None)

    def _run_round(self, permit: PullRequest) -> None:
        base = self.current
        if base is None or permit.global_step != base.policy_version + 1:
            raise RuntimeError("RL attempted a local round without its exact base")
        stats = self.runtime.run_local_round(
            expected_policy_version=base.policy_version,
            groups=self.config.groups_per_round,
            samples_per_group=self.config.samples_per_group,
            optimizer_steps=self.config.local_optimizer_steps,
        )
        if (
            stats.island_id != self.config.learner_id
            or stats.base_policy_version != base.policy_version
            or stats.local_round_id != permit.global_step
        ):
            raise RuntimeError("Miles returned LocalRoundStats for a different round")

        self.submit_local_state(
            permit,
            base,
            self.runtime.export_local_policy(),
            stats,
        )

    def export_tensor_groups(
        self,
    ) -> tuple[tuple[CanonicalTensorSpec, ...], ...]:
        return bounded_tensor_groups(self.specs)

    def _validate_submission(
        self,
        permit: PullRequest,
        base: CanonicalLoraState,
        stats: LocalRoundStats,
    ) -> None:
        if (
            self.current is None
            or self._committed_version() is None
            or base is not self.current
            or base.policy_version != self.current_version
        ):
            raise RuntimeError("RL attempted to submit without its exact base")
        if (
            permit.global_step != base.policy_version + 1
            or stats.island_id != self.config.learner_id
            or stats.base_policy_version != base.policy_version
            or stats.local_round_id != permit.global_step
        ):
            raise RuntimeError("Miles returned LocalRoundStats for a different round")

    def _record_submission(
        self,
        stats: LocalRoundStats,
        *,
        payload_bytes: int,
    ) -> None:
        self.runtime.record_local_round(stats)
        self._append_event(
            {
                "event": "rl_local_round",
                **asdict(stats),
                "rl/active_groups": stats.active_groups,
                "rl/completed_groups": stats.completed_groups,
                "rl/cancelled_groups": stats.cancelled_groups,
                "rl/completed_trajectories": stats.completed_trajectories,
                "rl/action_tokens": stats.action_tokens,
                "rl/tool_wait_seconds": stats.tool_wait_seconds,
                "rl/reward_mean": stats.reward_mean,
                "rl/reward_std": stats.reward_std,
                "rl/rollout_seconds": stats.rollout_seconds,
                "rl/group_p50_seconds": stats.group_p50_seconds,
                "rl/group_p95_seconds": stats.group_p95_seconds,
                "rl/group_p99_seconds": stats.group_p99_seconds,
                "rl/zero_variance_group_ratio": stats.zero_variance_group_ratio,
                "rl/global_policy_version": stats.base_policy_version,
                "rl/rollout_policy_version": stats.base_policy_version,
                "rl/mixed_version_group_count": 0,
                "rl/local_delta_norm": stats.delta_l2_norm,
                "rl/current_vs_rollout_kl": stats.mean_kl,
                "rl/ess_ratio": stats.ess_ratio,
                "rl/clip_fraction": stats.clip_fraction,
                "sync/bytes_sent": 48 + payload_bytes,
            }
        )

    def submit_local_state(
        self,
        permit: PullRequest,
        base: CanonicalLoraState,
        local: CanonicalLoraState,
        stats: LocalRoundStats,
    ) -> LocalRoundStats:
        self._validate_submission(permit, base, stats)
        try:
            normalize = (
                canonical_state_from_owned_tensors
                if getattr(local, "_owns_tensor_storage", False)
                else canonical_state
            )
            local = normalize(
                local.policy_version,
                local.tensors,
                base_model_revision=local.base_model_revision,
                lora_config_hash=local.lora_config_hash,
                layout_hash=local.layout_hash,
                expected_specs=self.specs,
            )
            delta = policy_delta(local, base)
        except ValueError as error:
            message = str(error)
            if "NaN or Inf" in message:
                metric = "nonfinite_delta_count"
            elif any(
                value in message
                for value in ("layout", "names, shapes, or dtypes")
            ):
                metric = "layout_hash_mismatch"
            else:
                raise
            raise StrictRlInvariantError(metric, message) from error
        stats = replace(stats, delta_l2_norm=float(delta.norm().item()))
        if self.config.audit_dir is not None:
            _write_round_audit(
                self.config.audit_dir,
                learner_id=self.config.learner_id,
                target_step=permit.global_step,
                base=base,
                delta=delta,
            )
        payload = pack_tensor(delta, DTYPE_F32)
        self._record_submission(stats, payload_bytes=len(payload))
        self.client.push_fragment(
            0,
            permit.global_step,
            permit.round_attempt,
            base.policy_version,
            permit.global_step,
            1,
            1,
            payload,
        )
        self.pushed_step = permit.global_step
        return stats

    def submit_chunked_local_state(
        self,
        permit: PullRequest,
        base: CanonicalLoraState,
        exported,
        stats: LocalRoundStats,
        *,
        before_last_enqueue: Callable[[], None] | None = None,
    ) -> LocalRoundStats:
        """Consume an owner-sharded export as bounded strict-AVG delta parts."""

        self._validate_submission(permit, base, stats)
        expected_names = tuple(spec.name for spec in self.specs)
        if (
            getattr(exported, "_yeto_chunked_export", None)
            != "owner-sharded-v1"
            or exported.policy_version != base.policy_version
            or tuple(exported.expected_names) != expected_names
        ):
            discard = getattr(exported, "discard", None)
            if callable(discard):
                discard()
            raise StrictRlInvariantError(
                "layout_hash_mismatch",
                "chunked local policy identity or canonical tensor order changed",
            )

        audit = (
            None
            if self.config.audit_dir is None
            else _StreamingRoundAudit(
                self.config.audit_dir,
                learner_id=self.config.learner_id,
                target_step=permit.global_step,
                base=base,
            )
        )
        groups = self.export_tensor_groups()
        if not isinstance(base.tensors, dict):
            exported.discard()
            raise StrictRlInvariantError(
                "layout_hash_mismatch",
                "strict chunk base does not expose consumable owned storage",
            )
        base_tensors = base.tensors
        norm_squares = [0.0]
        complete = [False]

        def delta_parts():
            try:
                for group in groups:
                    local_tensors = exported.take_tensors(group)
                    delta = torch.empty(
                        sum(spec.numel for spec in group),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    offset = 0
                    for spec in group:
                        local_tensor = local_tensors[spec.name]
                        base_tensor = base.tensors[spec.name]
                        if (
                            tuple(local_tensor.shape) != spec.shape
                            or tuple(base_tensor.shape) != spec.shape
                        ):
                            raise ValueError(
                                f"canonical tensor {spec.name!r} changed shape"
                            )
                        target = delta[
                            offset : offset + spec.numel
                        ].view(spec.shape)
                        torch.sub(local_tensor, base_tensor, out=target)
                        offset += spec.numel
                    del local_tensor, base_tensor, target, local_tensors
                    if not torch.isfinite(delta).all().item():
                        raise ValueError("local LoRA delta contains NaN or Inf")
                    group_norm = float(torch.linalg.vector_norm(delta).item())
                    norm_squares[0] += group_norm * group_norm
                    if audit is not None:
                        audit.write_group(group, delta, base_tensors)
                    # This group is now represented by the queued delta. Drop
                    # its base tensors before exposing the delta bytes to the
                    # sender. In particular, the final group is gone before
                    # the last PUSH envelope can make the terminal FINAL
                    # allocation observable at the learner.
                    for spec in group:
                        del base_tensors[spec.name]
                    raw = memoryview(delta.numpy()).cast("B")
                    try:
                        yield raw
                    finally:
                        raw.release()
                        del delta
                exported.finish()
                complete[0] = True
            except BaseException:
                exported.discard()
                raise

        payload_bytes = sum(spec.numel for spec in self.specs) * 4
        try:
            sent = self.client.push_fragment_parts(
                0,
                permit.global_step,
                permit.round_attempt,
                base.policy_version,
                permit.global_step,
                1,
                1,
                delta_parts(),
                before_last_enqueue=before_last_enqueue,
            )
            if sent is not True or not complete[0]:
                raise RuntimeError(
                    "streamed PUSH_FRAGMENT was not completely queued"
                )
            if audit is not None:
                audit.finish()
        except (TypeError, ValueError) as error:
            exported.discard()
            if audit is not None:
                audit.abort()
            message = str(error)
            metric = (
                "nonfinite_delta_count"
                if "NaN or Inf" in message or "non-finite" in message
                else "layout_hash_mismatch"
            )
            raise StrictRlInvariantError(metric, message) from error
        except BaseException:
            exported.discard()
            if audit is not None:
                audit.abort()
            raise

        stats = replace(
            stats,
            delta_l2_norm=norm_squares[0] ** 0.5,
        )
        self._record_submission(stats, payload_bytes=payload_bytes)
        self.pushed_step = permit.global_step
        return stats

    def finalize(self) -> CanonicalLoraState:
        while not self.client.finalizing.is_set():
            self.client.check_health()
            self._drain_messages()
            if not self.client.finalizing.is_set():
                time.sleep(0.05)
        return self._finalize()

    def _finalize(self) -> CanonicalLoraState:
        manifest, final = self._terminal_state()
        if self.current is not final or self.current_version != final.policy_version:
            self._install_policy(final)
        self.client.acknowledge_finalization(manifest)
        return final

    def _terminal_state(self) -> tuple[FinalManifest, CanonicalLoraState]:
        if self._terminal_manifest is not None:
            if self._terminal_policy is None:
                raise RuntimeError("RL terminal policy cache is incomplete")
            return self._terminal_manifest, self._terminal_policy
        manifest, fragments = self.client.wait_for_final_fragments()
        if (
            manifest.global_step != self.config.global_rounds
            or manifest.versions != (manifest.global_step,)
            or len(fragments) != 1
        ):
            raise RuntimeError("RL received an inconsistent final checkpoint cut")
        final = self._state_from_payload(
            manifest.global_step,
            fragments[0].data,
        )
        self._terminal_manifest = manifest
        self._terminal_policy = final
        return manifest, final

    def _append_event(self, event: dict) -> None:
        path = Path(self.config.event_tape).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "island_id": self.config.learner_id,
            "time_unix": time.time(),
            **event,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            )
