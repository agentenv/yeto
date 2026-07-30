"""One fixed-roster RL island's PULL/local/PUSH/BCAST loop."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from ..protocol import DTYPE_F32, PullRequest, SyncerClient
from ..tensor_io import pack_tensor, unpack_fragment
from .core import (
    CanonicalLoraState,
    CanonicalTensorSpec,
    LocalRoundStats,
    StrictRlInvariantError,
    build_avg_layout,
    canonical_state,
    flat_tensor,
    policy_delta,
    tensors_from_flat,
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
    wan_streams: int = 4


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
        self.initial = canonical_state(
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
        self.permits: dict[int, PullRequest] = {}
        self.pushed_step: int | None = None

    def run(self) -> CanonicalLoraState:
        try:
            self.client.start()
            if self.config.learner_id == 0:
                self.client.send_init(
                    0,
                    pack_tensor(
                        flat_tensor(self.initial.tensors, self.specs),
                        DTYPE_F32,
                    ),
                )
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

    def _drain_messages(self) -> bool:
        progressed = False
        for update in self.client.drain_updates():
            progressed = True
            if update.fragment_id != 0:
                raise RuntimeError("RL received a nonzero fragment")
            if self.current is not None:
                if update.version < self.current.policy_version:
                    continue
                if update.version == self.current.policy_version:
                    continue
                if update.version != self.current.policy_version + 1:
                    raise RuntimeError(
                        f"RL policy jumped from {self.current.policy_version} "
                        f"to {update.version}"
                    )
            state = self._state_from_payload(update.version, update.data)
            self.runtime.apply_global_policy(state)
            self.current = state
            self.pushed_step = None
            self.permits = {
                step: permit
                for step, permit in self.permits.items()
                if step > state.policy_version
            }

        for permit in self.client.drain_pulls():
            progressed = True
            if permit.fragment_id != 0 or permit.round_attempt != 1:
                raise RuntimeError("RL received an invalid PULL permit")
            if permit.global_step > self.config.global_rounds:
                raise RuntimeError("RL received a PULL beyond configured rounds")
            current_version = self.current.policy_version if self.current else -1
            if permit.global_step <= current_version:
                continue
            previous = self.permits.get(permit.global_step)
            if previous is not None and previous != permit:
                raise RuntimeError("RL received conflicting PULL permits")
            self.permits[permit.global_step] = permit
        return progressed

    def _state_from_payload(self, version: int, data: bytes) -> CanonicalLoraState:
        try:
            flat = unpack_fragment(self.layout.fragments[0], data, DTYPE_F32)
            return canonical_state(
                version,
                tensors_from_flat(flat, self.specs),
                base_model_revision=self.config.base_model_revision,
                lora_config_hash=self.config.lora_config_hash,
                layout_hash=self.config.layout_hash,
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
        target = self.current.policy_version + 1
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

        try:
            exported = self.runtime.export_local_policy()
            local = canonical_state(
                exported.policy_version,
                exported.tensors,
                base_model_revision=exported.base_model_revision,
                lora_config_hash=exported.lora_config_hash,
                layout_hash=exported.layout_hash,
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
        self.runtime.record_local_round(stats)
        payload = pack_tensor(delta, DTYPE_F32)
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
                "sync/bytes_sent": 48 + len(payload),
            }
        )
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

    def _finalize(self) -> CanonicalLoraState:
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
        self.runtime.apply_global_policy(final)
        self.client.acknowledge_finalization(manifest)
        return final

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
