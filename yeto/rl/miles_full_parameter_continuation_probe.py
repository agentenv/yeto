"""Two-step SFT hardware gate for the full-parameter DiLoCo boundary."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_learner import (
    ComponentIdentity,
    ParameterCut,
    advance_parameter_cut_version,
)
from .miles_full_parameter import MilesFullParameterAdapter
from .miles_full_parameter_probe import (
    _hardware_identity,
    _positive_int_environment,
    _source_identity,
    _write_private_json,
)

_EVIDENCE_ENV = "YETO_FULL_PARAMETER_CONTINUATION_EVIDENCE"
_MODEL_REVISION_ENV = "YETO_FULL_PARAMETER_MODEL_REVISION"
_CONFIG_HASH_ENV = "YETO_FULL_PARAMETER_CONFIG_HASH"
_FRAGMENT_COUNT_ENV = "YETO_FULL_PARAMETER_FRAGMENT_COUNT"
_CONVERSION_MANIFEST_ENV = "YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256"
_IMAGE_DIGEST_ENV = "YETO_MILES_IMAGE_DIGEST"
_YETO_SOURCE_ENV = "YETO_FULL_PARAMETER_YETO_SOURCE_ROOT"
_MILES_SOURCE_ENV = "YETO_FULL_PARAMETER_MILES_SOURCE_ROOT"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _changed_fragment_count(base: ParameterCut, target: ParameterCut) -> int:
    if len(base.fragments) != len(target.fragments):
        raise RuntimeError("full-parameter continuation changed fragment count")
    return sum(
        left.payload_hash != right.payload_hash
        for left, right in zip(base.fragments, target.fragments, strict=True)
    )


def _closed_int(value: Any, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        raise RuntimeError(f"invalid optimizer proof field: {name}")
    return value


def _optimizer_proof_payload(
    states,
    *,
    policy_version: int,
    local_step_generation: int,
    rollout_id: int,
    scheduler_num_steps: int,
) -> tuple[dict[str, object], ...]:
    if not states:
        raise RuntimeError("Miles returned no optimizer-state proofs")
    payloads = []
    identities = set()
    for state in states:
        topology = getattr(state, "topology", None)
        shard_id = getattr(topology, "shard_id", None)
        if not isinstance(shard_id, str) or not shard_id or shard_id in identities:
            raise RuntimeError("Miles optimizer-state topology is malformed")
        identities.add(shard_id)
        if (
            getattr(state, "role", None) != "actor"
            or getattr(state, "installed_policy_version", None) != policy_version
            or getattr(state, "local_step_generation", None) != local_step_generation
            or getattr(state, "last_rollout_id", None) != rollout_id
            or getattr(state, "scheduler_num_steps", None) != scheduler_num_steps
        ):
            raise RuntimeError("Miles optimizer-state progress is stale")
        selected_hash = getattr(state, "selected_state_sha256", None)
        selected_name = getattr(state, "selected_wire_name", None)
        if (
            not isinstance(selected_hash, str)
            or not _SHA256.fullmatch(selected_hash)
            or not isinstance(selected_name, str)
            or "::" not in selected_name
        ):
            raise RuntimeError("Miles optimizer-state fingerprint is malformed")
        payloads.append(
            {
                "shard_id": shard_id,
                "selected_wire_name": selected_name,
                "selected_state_sha256": selected_hash,
                "populated_parameter_count": _closed_int(
                    getattr(state, "populated_parameter_count", None),
                    "populated_parameter_count",
                    positive=True,
                ),
                "optimizer_state_tensor_count": _closed_int(
                    getattr(state, "optimizer_state_tensor_count", None),
                    "optimizer_state_tensor_count",
                    positive=True,
                ),
                "optimizer_state_scalar_count": _closed_int(
                    getattr(state, "optimizer_state_scalar_count", None),
                    "optimizer_state_scalar_count",
                    positive=True,
                ),
                "model_master_parameter_count": _closed_int(
                    getattr(state, "model_master_parameter_count", None),
                    "model_master_parameter_count",
                    positive=True,
                ),
            }
        )
    return tuple(sorted(payloads, key=lambda payload: payload["shard_id"]))


def _require_moments_preserved(
    before: tuple[dict[str, object], ...],
    after: tuple[dict[str, object], ...],
) -> None:
    if before != after:
        raise RuntimeError("full-parameter apply changed Adam optimizer state")


def _receipt_payload(receipts, *, base: int, generation: int, rollout: int):
    if not receipts:
        raise RuntimeError("Miles returned no local-step receipts")
    payloads = []
    identities = set()
    for receipt in receipts:
        topology = getattr(receipt, "topology", None)
        shard_id = getattr(topology, "shard_id", None)
        if not isinstance(shard_id, str) or not shard_id or shard_id in identities:
            raise RuntimeError("Miles local-step topology is malformed")
        identities.add(shard_id)
        if (
            getattr(receipt, "role", None) != "actor"
            or getattr(receipt, "base_policy_version", None) != base
            or getattr(receipt, "local_step_generation", None) != generation
            or getattr(receipt, "rollout_id", None) != rollout
        ):
            raise RuntimeError("Miles local-step receipt is stale")
        optimizer_steps = _closed_int(
            getattr(receipt, "optimizer_steps", None),
            "optimizer_steps",
            positive=True,
        )
        scheduler_start = _closed_int(
            getattr(receipt, "scheduler_start_steps", None),
            "scheduler_start_steps",
        )
        scheduler_end = _closed_int(
            getattr(receipt, "scheduler_end_steps", None),
            "scheduler_end_steps",
            positive=True,
        )
        if scheduler_end <= scheduler_start:
            raise RuntimeError("Miles local-step scheduler did not advance")
        payloads.append(
            {
                "shard_id": shard_id,
                "optimizer_steps": optimizer_steps,
                "scheduler_start_steps": scheduler_start,
                "scheduler_end_steps": scheduler_end,
            }
        )
    return tuple(sorted(payloads, key=lambda payload: payload["shard_id"]))


class MilesFullParameterContinuationProbeSync:
    """Prove two local steps around a moment-preserving global policy apply."""

    def __init__(self, args) -> None:
        self.args = args
        evidence = os.environ.get(_EVIDENCE_ENV)
        if evidence is None:
            raise RuntimeError(f"{_EVIDENCE_ENV} is required")
        self.evidence_path = Path(evidence)
        self.component = ComponentIdentity(
            "actor",
            os.environ.get(_MODEL_REVISION_ENV, ""),
            os.environ.get(_CONFIG_HASH_ENV, ""),
        )
        self.num_fragments = _positive_int_environment(_FRAGMENT_COUNT_ENV)
        self.conversion_manifest_sha256 = os.environ.get(
            _CONVERSION_MANIFEST_ENV,
            "",
        )
        self.image_digest = os.environ.get(_IMAGE_DIGEST_ENV, "")
        if not _SHA256.fullmatch(self.conversion_manifest_sha256):
            raise RuntimeError("conversion manifest hash is malformed")
        if not _IMAGE_DIGEST.fullmatch(self.image_digest):
            raise RuntimeError("Miles image digest is malformed")
        self.yeto_source_root = Path(os.environ.get(_YETO_SOURCE_ENV, ""))
        self.miles_source_root = Path(os.environ.get(_MILES_SOURCE_ENV, ""))
        self.actor_model = None
        self.adapter: MilesFullParameterAdapter | None = None
        self.anchor: ParameterCut | None = None
        self.applied: ParameterCut | None = None
        self.first_moments: tuple[dict[str, object], ...] | None = None
        self.first_after_moments: tuple[dict[str, object], ...] | None = None
        self.first_receipts: tuple[dict[str, object], ...] | None = None
        self.evidence: dict[str, object] | None = None

    async def initialize(self, *, actor_model, rollout_manager) -> None:
        del rollout_manager
        if (
            self.args.start_rollout_id != 0
            or self.args.num_rollout != 2
            or self.args.num_steps_per_rollout != 1
            or self.args.global_batch_size != 1
            or not self.args.debug_train_only
            or self.args.loss_type != "sft_loss"
        ):
            raise RuntimeError(
                "continuation probe requires the exact two-step SFT profile"
            )
        self.actor_model = actor_model
        self.adapter, self.anchor = await MilesFullParameterAdapter.capture_initial(
            actor_model,
            policy_version=0,
            algorithm="grpo",
            components=(self.component,),
            num_fragments=self.num_fragments,
        )
        self.yeto_source = _source_identity(self.yeto_source_root)
        self.miles_source = _source_identity(self.miles_source_root)
        self.hardware = _hardware_identity()

    async def after_local_train(
        self,
        *,
        rollout_id,
        actor_model,
        rollout_data,
    ) -> bool:
        if (
            actor_model is not self.actor_model
            or rollout_data is None
            or self.adapter is None
            or self.anchor is None
        ):
            raise RuntimeError("continuation probe ran outside its actor boundary")
        if rollout_id == 0 and self.applied is None:
            receipts = await actor_model.record_full_parameter_local_step(
                base_policy_version=0,
                rollout_id=0,
            )
            self.first_receipts = _receipt_payload(
                receipts,
                base=0,
                generation=1,
                rollout=0,
            )
            before = _optimizer_proof_payload(
                await actor_model.full_parameter_optimizer_states(),
                policy_version=0,
                local_step_generation=1,
                rollout_id=0,
                scheduler_num_steps=1,
            )
            local = await self.adapter.capture(
                actor_model,
                policy_version=0,
                local_step_generation=1,
            )
            changed = _changed_fragment_count(self.anchor, local)
            if changed < 1:
                raise RuntimeError("first SFT optimizer step changed no parameters")
            target = advance_parameter_cut_version(
                self.adapter.layout,
                local,
                target_policy_version=1,
            )
            await self.adapter.apply(actor_model, target)
            after = _optimizer_proof_payload(
                await actor_model.full_parameter_optimizer_states(),
                policy_version=1,
                local_step_generation=0,
                rollout_id=0,
                scheduler_num_steps=1,
            )
            _require_moments_preserved(before, after)
            self.first_moments = before
            self.first_after_moments = after
            self.applied = target
            self.first_local_hash = local.policy_hash
            self.first_changed_fragments = changed
            return False

        if (
            rollout_id != 1
            or self.applied is None
            or self.first_moments is None
            or self.first_after_moments is None
        ):
            raise RuntimeError("continuation probe rollout order changed")
        receipts = await actor_model.record_full_parameter_local_step(
            base_policy_version=1,
            rollout_id=1,
        )
        second_receipts = _receipt_payload(
            receipts,
            base=1,
            generation=1,
            rollout=1,
        )
        before = _optimizer_proof_payload(
            await actor_model.full_parameter_optimizer_states(),
            policy_version=1,
            local_step_generation=1,
            rollout_id=1,
            scheduler_num_steps=2,
        )
        local = await self.adapter.capture(
            actor_model,
            policy_version=1,
            local_step_generation=1,
        )
        changed = _changed_fragment_count(self.applied, local)
        if changed < 1:
            raise RuntimeError("second SFT optimizer step changed no parameters")
        target = advance_parameter_cut_version(
            self.adapter.layout,
            local,
            target_policy_version=2,
        )
        await self.adapter.apply(actor_model, target)
        after = _optimizer_proof_payload(
            await actor_model.full_parameter_optimizer_states(),
            policy_version=2,
            local_step_generation=0,
            rollout_id=1,
            scheduler_num_steps=2,
        )
        _require_moments_preserved(before, after)
        self.evidence = {
            "schema": "yeto-miles-full-parameter-continuation-probe-v1",
            "observed_utc": datetime.now(UTC).isoformat(),
            "algorithm": self.adapter.layout.algorithm,
            "model_revision": self.component.model_revision,
            "model_config_sha256": self.component.config_hash,
            "conversion_manifest_sha256": self.conversion_manifest_sha256,
            "miles_image_digest": self.image_digest,
            "yeto_source": self.yeto_source,
            "miles_source": self.miles_source,
            "hardware": self.hardware,
            "parameter_layout_hash": self.adapter.layout.layout_hash,
            "parameter_tensor_count": self.adapter.expected_parameter_tensor_count,
            "parameter_scalar_count": self.adapter.expected_parameter_scalar_count,
            "fragment_count": self.adapter.layout.fragments.num_fragments,
            "initial_policy_hash": self.anchor.policy_hash,
            "first_local_policy_hash": self.first_local_hash,
            "first_global_policy_hash": self.applied.policy_hash,
            "second_local_policy_hash": local.policy_hash,
            "final_global_policy_hash": target.policy_hash,
            "first_changed_fragment_count": self.first_changed_fragments,
            "second_changed_fragment_count": changed,
            "first_local_step_receipts": self.first_receipts,
            "second_local_step_receipts": second_receipts,
            "optimizer_state_proof_scope": (
                "selected_parameter_per_topology_plus_cardinalities"
            ),
            "first_optimizer_state_before_apply": self.first_moments,
            "first_optimizer_state_after_apply": self.first_after_moments,
            "second_optimizer_state_before_apply": before,
            "second_optimizer_state_after_apply": after,
            "optimizer_state_preserved_across_apply": True,
            "next_step_after_global_apply_verified": True,
            "final_policy_version": 2,
        }
        self.applied = target
        return True

    async def finalize(self) -> None:
        if self.evidence is None:
            raise RuntimeError("full-parameter continuation probe did not complete")
        _write_private_json(self.evidence_path, self.evidence)


def create_full_parameter_continuation_probe(
    args,
) -> MilesFullParameterContinuationProbeSync:
    """Miles ``--external-policy-sync-path`` factory for the two-step gate."""

    return MilesFullParameterContinuationProbeSync(args)
