"""Outer-sync sessions for :class:`yeto.rl.engine.driver.IslandDriver`.

The strict-avg and decoupled protocol state machines are reused unchanged:
``yeto.rl.bridge.StrictRlBridge`` and ``yeto.rl.decoupled.DecoupledRlBridge``.
What this module replaces is ``yeto.rl.miles.MilesPolicySync`` /
``DecoupledMilesPolicySync``: every call into Miles internals (actor
``apply_trainable_state``/``export_trainable_state``, SGLang
``update_weight_version``) goes through the ``PolicyState`` port (via
``driver.apply_policy``) and the ``Publisher`` port (via ``driver.publish``).

Island progress checkpoints keep the legacy on-disk formats (strict schema 3,
decoupled schema 4) by importing the legacy helpers from ``yeto.rl.miles``;
they never contain LoRA tensors or optimizer state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from yeto.rl import miles as legacy
from yeto.rl.bridge import BridgeConfig, StrictRlBridge
from yeto.rl.core import (
    CanonicalLoraState,
    LocalRoundStats,
    PolicySnapshot,
    canonical_state,
    policy_hash,
    policy_tensor_hash,
)

from .pause_audit import PAUSABLE_PHASE
from .driver import IslandDriver, SyncBoundary, SyncStart
from .trainable_state import TrainableState


def _lora(state: TrainableState) -> CanonicalLoraState:
    return state.to_lora()


def _at_version(state: CanonicalLoraState, version: int) -> CanonicalLoraState:
    return canonical_state(
        version,
        state.tensors,
        base_model_revision=state.base_model_revision,
        lora_config_hash=state.lora_config_hash,
        layout_hash=state.layout_hash,
        expected_specs=state.specs,
    )


# ---------------------------------------------------------------------------
# No outer sync (single island smoke; task 4.1)
# ---------------------------------------------------------------------------
class LocalOnlySync:
    """Train-only island: the local result is the next policy."""

    def __init__(self, num_rollout: int) -> None:
        if num_rollout < 1:
            raise ValueError("num_rollout must be positive")
        self.num_rollout = num_rollout

    def start(self, driver: IslandDriver) -> SyncStart:
        state = _at_version(_lora(driver.export_local()), 0)
        return SyncStart(TrainableState.from_lora(state), 0)

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        return rollout_id + 1 >= self.num_rollout

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        local = _at_version(_lora(driver.export_local()), rollout_id + 1)
        driver.emit("rl_local_round", **asdict(stats))
        return SyncBoundary(
            TrainableState.from_lora(local), stop=rollout_id + 1 >= self.num_rollout
        )

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        pass

    def finish(self, driver) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Strict-avg
# ---------------------------------------------------------------------------
class StrictIslandProgress:
    """Strict island progress in the legacy schema-3 file (no LoRA/optimizer).

    The rollout process may already have written this round's record together
    with its completed-group queue (legacy ``_save_completed_groups``); that
    record is kept. Otherwise the driver writes this generation's metrics with
    an empty queue, replacing any record a killed learner left for the same
    round (as legacy's recovery rewrites it).
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.path = Path(args.yeto_rl_completed_groups_path).expanduser()

    def _load(self) -> dict | None:
        if not self.path.is_file():
            return None
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception:  # the local queue is disposable, global state is not
            return None
        return payload if isinstance(payload, dict) else None

    def after_generate(self, *, rollout_id, policy_token, metrics) -> None:
        payload = self._load()
        if (
            payload is not None
            and payload.get("schema_version") == legacy._ISLAND_CHECKPOINT_SCHEMA
            and payload.get("policy_version") == rollout_id
            and payload.get("config") == legacy._island_checkpoint_config(self.args)
            and isinstance(payload.get("rollout_metrics"), Mapping)
            and payload["rollout_metrics"]
            and payload.get("completed_groups")
        ):
            return
        legacy._atomic_save_island_checkpoint(
            self.path,
            {
                "schema_version": legacy._ISLAND_CHECKPOINT_SCHEMA,
                "config": legacy._island_checkpoint_config(self.args),
                "local_round_id": rollout_id + 1,
                "policy_version": rollout_id,
                "rollout_metrics": {k: float(v) for k, v in metrics.items()},
                "local_round_stats": None,
                "completed_groups": [],
            },
        )

    def commit_round(self, stats: LocalRoundStats) -> None:
        # Same contract as legacy ``_BridgeRuntime.record_local_round``.
        legacy._BridgeRuntime(None, self.args).record_local_round(stats)


class _PortsStrictRuntime:
    """``IslandRuntime`` for ``StrictRlBridge``; applies happen via the driver."""

    def __init__(self, initial: CanonicalLoraState, progress: StrictIslandProgress | None):
        self.initial = initial
        self.progress = progress

    def initialize(self) -> CanonicalLoraState:
        if self.initial is None:
            raise RuntimeError("initial policy was already consumed")
        initial, self.initial = self.initial, None
        return initial

    def apply_global_policy(self, _state: CanonicalLoraState) -> None:
        pass

    def record_local_round(self, stats: LocalRoundStats) -> None:
        if self.progress is not None:
            self.progress.commit_round(stats)

    def shutdown(self) -> None:
        pass


class StrictAvgSync:
    """Strict-avg at the safe boundary: export+PUSH, wait v+1, reset-apply."""

    def __init__(
        self,
        config: BridgeConfig,
        *,
        progress: StrictIslandProgress | None = None,
        client_factory: Callable[[StrictRlBridge], Any] | None = None,
    ) -> None:
        self.config = config
        self.progress = progress
        self.client_factory = client_factory
        self.bridge: StrictRlBridge | None = None
        self.current: CanonicalLoraState | None = None
        self.permit = None

    def _apply(self, driver: IslandDriver, state: CanonicalLoraState) -> TrainableState:
        return driver.apply_policy(
            TrainableState.from_lora(state),
            optimizer="reset",
            # Scheduler progress is counted in optimizer steps, not rounds:
            # every strict round runs ``local_optimizer_steps`` steps.
            local_step=state.policy_version * self.config.local_optimizer_steps,
        )

    def start(self, driver: IslandDriver) -> SyncStart:
        initial = _lora(driver.export_local())
        self.bridge = StrictRlBridge(_PortsStrictRuntime(initial, self.progress), self.config)
        if self.client_factory is not None:
            self.bridge.client.close()
            self.bridge.client = self.client_factory(self.bridge)
        del initial
        self.bridge.start()
        self.current = self.bridge.wait_for_initial_policy()
        state = self._apply(driver, self.current)
        version = self.current.policy_version
        finished = version >= self.config.global_rounds
        if not finished:
            self.permit = self.bridge.wait_for_round()
        return SyncStart(state, version, finished)

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        # Strict: local round ``rollout_id + 1`` is the last iff it reaches
        # global_rounds (fix-decoupled-lr-schedule D4).
        return rollout_id + 1 >= self.config.global_rounds

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        if self.current is None or self.permit is None:
            raise RuntimeError("strict sync called outside an active round")
        if rollout_id != self.current.policy_version:
            raise RuntimeError("rollout ID differs from the global policy version")
        driver.phase("export_push", rollout_id=rollout_id)
        # The strict delta is taken against the exact base version.
        local = _at_version(_lora(driver.export_local()), rollout_id)
        self.bridge.submit_local_state(self.permit, self.current, local, stats)
        self.bridge.release_current(rollout_id)
        self.current = None
        driver.phase("wait_global", policy_version=rollout_id + 1)
        current = self.bridge.wait_for_global_policy(rollout_id + 1)
        state = self._apply(driver, current)
        self.current = current
        stop = current.policy_version >= self.config.global_rounds
        self.permit = None if stop else self.bridge.wait_for_round()
        return SyncBoundary(state, stop)

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        pass

    def outer_phase(self, driver, *, rollout_id: int) -> str:
        """3.8: pausable only while the next PULL permit is held and the syncer
        has not started finalizing (pause_audit.md: strict row)."""
        client = getattr(self.bridge, "client", None)
        finalizing = getattr(client, "finalizing", None)
        if finalizing is not None and finalizing.is_set():
            return "finalizing"
        if self.current is None:
            return "in-boundary"
        if self.permit is None:
            return "stop-round"
        return PAUSABLE_PHASE

    def finish(self, driver) -> None:
        if self.bridge is None or self.current is None:
            raise RuntimeError("finalized an uninitialized strict sync")
        final = self.bridge.finalize()
        if policy_hash(final) != policy_hash(self.current):
            raise RuntimeError("final policy differs from the committed global policy")

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.client.close()


# ---------------------------------------------------------------------------
# Decoupled
# ---------------------------------------------------------------------------
class DecoupledIslandProgress:
    """Decoupled island progress in the legacy schema-4 file."""

    def __init__(self, args: Any) -> None:
        self.args = args

    def after_generate(self, *, rollout_id, policy_token, metrics) -> None:
        payload = legacy._load_decoupled_checkpoint(self.args)
        if (
            payload is None
            or payload.get("next_rollout_id") != rollout_id
            or payload.get("policy_token") != policy_token
        ):
            raise RuntimeError("decoupled RL progress changed during rollout")
        if payload.get("rollout_metrics"):
            return  # written by the rollout process with its group queue
        legacy._save_decoupled_checkpoint(
            self.args,
            snapshot=PolicySnapshot(
                rollout_id, tuple(payload["fragment_versions"]), payload["policy_hash"]
            ),
            optimizer_steps=payload["optimizer_steps"],
            action_tokens=payload["action_tokens"],
            rollout_metrics=metrics,
            local_round_stats=payload.get("local_round_stats"),
            completed_groups=payload["completed_groups"],
        )


class DecoupledSync:
    """Run-until-stop decoupled fragments through PolicyState/Publisher.

    Mirrors ``DecoupledMilesPolicySync`` step for step; ``_record_local_round``
    is the legacy implementation (called unbound, it only needs
    ``_append_event`` and ``snapshot``).
    """

    def __init__(
        self,
        args: Any,
        *,
        bridge_factory: Callable[..., Any] | None = None,
        client_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.args = args
        self.bridge_factory = bridge_factory
        self.client_factory = client_factory
        self.bridge = None
        self.current: CanonicalLoraState | None = None
        self.snapshot: PolicySnapshot | None = None
        self.optimizer_steps = 0
        self.action_tokens = 0
        self.finished = False
        self._driver: IslandDriver | None = None

    # legacy-compatible helpers ------------------------------------------
    def _append_event(self, event: dict[str, Any]) -> None:
        self._driver.events.append(event)

    _record_local_round = legacy.DecoupledMilesPolicySync._record_local_round
    _record_final_payload = legacy.DecoupledMilesPolicySync._record_final_payload
    _save_progress = legacy.DecoupledMilesPolicySync._save_progress

    def _apply(self, state: CanonicalLoraState, *, reset: bool) -> CanonicalLoraState:
        self._driver.apply_policy(
            TrainableState.from_lora(state),
            optimizer="reset" if reset else "preserve",
            local_step=self.optimizer_steps,
        )
        return state

    def _snapshot_state(self, snapshot: PolicySnapshot) -> TrainableState:
        self._driver.emit(
            "rl_policy_snapshot",
            **{
                "rl/rollout_id": snapshot.rollout_id,
                "rl/policy_token": snapshot.token,
                "rl/policy_hash": snapshot.policy_hash,
                "rl/fragment_versions": list(snapshot.fragment_versions),
                "rl/canonical_layout_hash": self.current.layout_hash,
                "rl/sync_layout_fingerprint": self.args.yeto_rl_sync_layout_fingerprint,
                "rl/mixed_version_group_count": 0,
            },
        )
        return TrainableState.from_lora(_at_version(self.current, snapshot.rollout_id))

    # session -------------------------------------------------------------
    def start(self, driver: IslandDriver) -> SyncStart:
        from yeto.rl import decoupled

        self._driver = driver
        args = self.args
        initial = _lora(driver.export_local())
        adapter = getattr(args, "yeto_rl_initial_adapter", None)
        adapter_sha = getattr(args, "yeto_rl_initial_adapter_sha256", None)
        if (adapter is None) != (adapter_sha is None):
            raise ValueError(
                "decoupled RL initial adapter path and SHA256 must be set together"
            )
        parent_hash = None
        if adapter is not None:
            from yeto.rl.initial_adapter import load_initial_adapter

            initial = load_initial_adapter(
                adapter, adapter_sha, model=args.yeto_rl_model, expected=initial
            )
            parent_hash = policy_tensor_hash(initial)
        checkpoint_error = None
        try:
            checkpoint = legacy._load_decoupled_checkpoint(args)
        except RuntimeError as error:
            checkpoint, checkpoint_error = None, error
        if checkpoint is not None:
            self.optimizer_steps = checkpoint["optimizer_steps"]
            self.action_tokens = checkpoint["action_tokens"]

        factory = self.bridge_factory or decoupled.DecoupledRlBridge
        self.bridge = factory(initial, args.yeto_rl_bridge_config)
        if self.client_factory is not None:
            self.bridge.client.close()
            self.bridge.client = self.client_factory(self.bridge)
        self.bridge.start()
        cut = self.bridge.wait_for_initial_cut(
            optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        versions = cut.fragment_versions
        if any(versions) and checkpoint is None:
            detail = "missing" if checkpoint_error is None else str(checkpoint_error)
            raise RuntimeError(
                f"nonzero decoupled RL cut has no valid island checkpoint: {detail}"
            )
        if (
            parent_hash is not None
            and not any(versions)
            and policy_tensor_hash(cut.state) != parent_hash
        ):
            raise RuntimeError(
                "decoupled RL initial adapter policy differs from version-zero cut"
            )
        if parent_hash is not None and checkpoint is None:
            self._append_event(
                {
                    "event": "rl_initial_adapter",
                    "parent_adapter_sha256": adapter_sha,
                    "parent_policy_hash": parent_hash,
                }
            )
        if checkpoint is not None and any(
            current < saved for current, saved in zip(versions, checkpoint["fragment_versions"])
        ):
            raise RuntimeError("decoupled RL syncer cut predates island checkpoint")
        # Restart: the authoritative cut is applied with an optimizer reset.
        self.current = self._apply(_at_version(cut.state, self.optimizer_steps), reset=True)
        self.bridge.commit_initial_cut(
            cut, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        self.snapshot = PolicySnapshot.create(self.optimizer_steps, self.current, versions)
        self._save_progress(self.snapshot, stats=None)
        startup_manifest = getattr(self.bridge, "startup_final_manifest", None)
        if startup_manifest is not None:
            self.bridge.acknowledge_finalization(startup_manifest)
            self._record_final_payload(self.bridge.final_payload_bytes_received)
            self.finished = True
        return SyncStart(
            self._snapshot_state(self.snapshot), self.snapshot.rollout_id, self.finished
        )

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        if (
            self.snapshot is None
            or rollout_id != self.snapshot.rollout_id
            or policy_hash != self.snapshot.policy_hash
        ):
            raise RuntimeError("published policy differs from the decoupled snapshot")
        self.args.yeto_rl_policy_token = self.snapshot.token

    def _commit_final_cut(self, manifest, final, *, stats, final_payload_bytes_received=0):
        self.current = self._apply(final, reset=False)
        self.snapshot = PolicySnapshot.create(
            final.policy_version, self.current, manifest.versions
        )
        self._save_progress(self.snapshot, stats=stats)
        self.bridge.acknowledge_finalization(manifest)
        if final_payload_bytes_received:
            self._record_final_payload(final_payload_bytes_received)
        self.finished = True
        return SyncBoundary(self._snapshot_state(self.snapshot), stop=True)

    def _finish(self, *, policy_version, stats) -> SyncBoundary:
        manifest, final = self.bridge.wait_for_final_cut(policy_version=policy_version)
        return self._commit_final_cut(
            manifest,
            final,
            stats=stats,
            final_payload_bytes_received=self.bridge.final_payload_bytes_received,
        )

    def outer_phase(self, driver, *, rollout_id: int) -> str:
        # Decoupled is not pause-certified (pause_audit); report finalization
        # anyway so the veto reason is the precise one.
        if self.bridge is not None and self.bridge.finalizing:
            return "finalizing"
        return PAUSABLE_PHASE

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        # Decoupled: final once the syncer's final cut is known (finalizing),
        # or this round exhausts the island's optional step budget (D4).
        if self.bridge is not None and self.bridge.finalizing:
            return True
        budget = getattr(self.args, "yeto_rl_learner_budget_steps", None)
        return budget is not None and budget == self.optimizer_steps + 1

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        import time

        started = time.monotonic()
        result = self._boundary(driver, rollout_id=rollout_id, stats=stats)
        self._append_event(
            {
                "event": "rl_sync_hook",
                "rl/rollout_id": rollout_id,
                "sync/hook_seconds": time.monotonic() - started,
                "sync/remote_quorum_wait_seconds": 0.0,
                "sync/finalization": result.stop,
            }
        )
        return result

    def _boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        if (
            self.current is None
            or self.snapshot is None
            or rollout_id != self.snapshot.rollout_id
            or self.args.yeto_rl_policy_token != self.snapshot.token
        ):
            raise RuntimeError("decoupled synchronization outside its policy snapshot")
        next_rollout_id = rollout_id + 1
        local = _at_version(_lora(driver.export_local()), next_rollout_id)
        self.optimizer_steps += 1
        self.action_tokens += stats.action_tokens
        if self.optimizer_steps != next_rollout_id:
            raise RuntimeError("decoupled RL optimizer progress diverged from rollout ID")
        if self.bridge.finalizing:
            self._record_local_round(stats, rollout_id=rollout_id)
            return self._finish(policy_version=next_rollout_id, stats=stats)
        if getattr(self.args, "yeto_rl_learner_budget_steps", None) == self.optimizer_steps:
            consolidation = self.bridge.consolidate_budget(
                local, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
            )
            stats = replace(
                stats,
                delta_l2_norm=math.sqrt(
                    sum(s.delta_l2_norm**2 for s in consolidation.submissions)
                ),
            )
            self._record_local_round(
                stats,
                rollout_id=rollout_id,
                submissions=consolidation.submissions,
                additional_payload_bytes_received=consolidation.bytes_received,
            )
            return self._commit_final_cut(consolidation.manifest, consolidation.state, stats=stats)

        driver.phase("drain_bcast", rollout_id=rollout_id)
        batch = self.bridge.drain_broadcasts(
            local, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        current = batch.state
        if batch.fragment_ids:
            current = self._apply(current, reset=False)
            self.bridge.commit_broadcasts(
                batch, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
            )
        driver.phase("drain_pull", rollout_id=rollout_id)
        submissions = self.bridge.submit_ready(
            current, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        stats = replace(
            stats,
            delta_l2_norm=math.sqrt(sum(s.delta_l2_norm**2 for s in submissions)),
        )
        self._record_local_round(
            stats, rollout_id=rollout_id, batch=batch, submissions=submissions
        )
        if self.bridge.finalizing or any(
            s.global_step == self.args.yeto_rl_total_fragment_steps for s in submissions
        ):
            return self._finish(policy_version=next_rollout_id, stats=stats)
        self.current = current
        self.snapshot = PolicySnapshot.create(
            next_rollout_id, current, self.bridge.fragment_versions
        )
        self._save_progress(self.snapshot, stats=stats)
        return SyncBoundary(self._snapshot_state(self.snapshot), stop=False)

    def finish(self, driver) -> None:
        if not self.finished:
            raise RuntimeError("stopped before decoupled RL finalization")

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
