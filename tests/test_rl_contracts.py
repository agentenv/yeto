from __future__ import annotations

from dataclasses import replace

import pytest

from yeto.rl.contracts import (
    InferencePublicationManifest,
    LocalStepReceipt,
    TrainerUpdateManifest,
    TrajectoryEnvelope,
)

HASH = "a" * 64


def test_trajectory_envelope_binds_policy_reward_cleanup_and_tokens():
    envelope = TrajectoryEnvelope(
        trajectory_id="trajectory-1",
        task_id="task-1",
        prompt_group_id="group-1",
        sample_index=0,
        behavior_policy_version=4,
        behavior_policy_hash=HASH,
        token_ids=(10, 11, 12),
        response_token_count=2,
        behavior_logprobs_hash="b" * 64,
        reward=1.0,
        reward_contract_hash="c" * 64,
        cleanup_evidence_hash="d" * 64,
    )

    assert envelope.behavior_policy_version == 4
    with pytest.raises(ValueError, match="token IDs"):
        replace(envelope, token_ids=())
    with pytest.raises(ValueError, match="response-token count"):
        replace(envelope, response_token_count=4)
    with pytest.raises(ValueError, match="response-token count"):
        replace(envelope, response_token_count=0)
    with pytest.raises(ValueError, match="finite"):
        replace(envelope, reward=float("nan"))


def test_local_step_receipt_rejects_duplicate_credit_and_false_progress():
    receipt = LocalStepReceipt(
        algorithm="grpo",
        learner_id=1,
        learner_generation=2,
        base_policy_version=3,
        base_policy_hash=HASH,
        input_batch_hash="b" * 64,
        trajectory_ids=("trajectory-1", "trajectory-2"),
        trained_tokens=128,
        optimizer_steps=1,
        optimizer_step_succeeded=True,
        parameter_layout_hash="c" * 64,
    )

    with pytest.raises(ValueError, match="duplicate"):
        replace(receipt, trajectory_ids=("trajectory-1", "trajectory-1"))
    with pytest.raises(ValueError, match="cannot claim"):
        replace(receipt, optimizer_step_succeeded=False)


def test_dense_and_pulseloco_manifests_have_disjoint_state_contracts():
    dense = TrainerUpdateManifest(
        exchange_mode="dense",
        learner_id=1,
        learner_generation=2,
        base_policy_version=3,
        target_policy_version=4,
        parameter_layout_hash=HASH,
        payload_hash="b" * 64,
        payload_bytes=64,
        fragment_count=2,
        complete=True,
    )

    with pytest.raises(ValueError, match="cannot carry PULSE"):
        replace(dense, error_feedback_version=0)
    pulse = replace(
        dense,
        exchange_mode="pulseloco",
        error_feedback_version=0,
        threshold_contract_hash="c" * 64,
    )
    assert pulse.error_feedback_version == 0


def test_full_and_pulsesync_publications_enforce_exact_base_semantics():
    full = InferencePublicationManifest(
        publication_mode="full",
        base_policy_version=None,
        target_policy_version=4,
        target_policy_hash=HASH,
        target_manifest_hash="b" * 64,
        payload_hash="c" * 64,
        payload_bytes=64,
        complete=True,
    )

    with pytest.raises(ValueError, match="cannot depend"):
        replace(full, base_policy_version=3)
    pulse = replace(full, publication_mode="pulsesync", base_policy_version=3)
    assert pulse.target_policy_version == 4
