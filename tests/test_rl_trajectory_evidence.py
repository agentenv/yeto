import json
from types import SimpleNamespace

import pytest

from yeto.rl.trajectory_evidence import (
    build_trajectory_batch_evidence,
    read_trajectory_batch_evidence,
    write_trajectory_batch_evidence,
)
from yeto_miles_secrlenv import reward


def _sample(*, group_index, index, value, status="completed"):
    outcome = {
        "schema": 1,
        "status": "completed",
        "episode_id": f"episode-{index}",
        "task_id": f"CVE-2026-{index:04d}",
        "reward": value,
        "passed": value == 1.0,
        "class": "flaky",
        reward.INFRASTRUCTURE_503_RETRIES_KEY: 0,
    }
    return SimpleNamespace(
        group_index=group_index,
        index=index,
        status=SimpleNamespace(value=status),
        metadata={
            reward.OUTCOME_KEY: outcome,
            reward.MAC_KEY: reward.sign_outcome(outcome),
            "untrusted_arbitrary_payload": {"must": "not be persisted"},
        },
        reward=value,
        tokens=[101, 102 + index, 103 + index],
        response_length=2,
        rollout_log_probs=[-0.25, -0.5],
    )


def test_authenticated_batch_round_trips_without_arbitrary_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    groups = [
        [_sample(group_index=11, index=0, value=0.0)],
        [_sample(group_index=19, index=1, value=1.0, status="truncated")],
    ]
    evidence = build_trajectory_batch_evidence(
        groups,
        rollout_id=3,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )

    assert evidence.trajectory_ids == tuple(
        envelope.trajectory_id for envelope in evidence.envelopes
    )
    assert evidence.trained_tokens == 4
    assert [item.response_token_count for item in evidence.envelopes] == [2, 2]
    assert [item.task_id for item in evidence.envelopes] == [
        "CVE-2026-0000",
        "CVE-2026-0001",
    ]
    assert all(item.behavior_policy_version == 3 for item in evidence.envelopes)

    root = tmp_path / "private"
    path = write_trajectory_batch_evidence(root, evidence)
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_trajectory_batch_evidence(path) == evidence
    assert write_trajectory_batch_evidence(root, evidence) == path
    raw = path.read_text()
    assert "untrusted_arbitrary_payload" not in raw
    assert "episode-" not in raw


def test_forged_reward_or_nonterminal_status_fails_before_evidence(monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    sample = _sample(group_index=0, index=0, value=1.0)
    sample.metadata[reward.OUTCOME_KEY]["reward"] = 0.0
    with pytest.raises(reward.UntrustedOutcome, match="signature mismatch"):
        build_trajectory_batch_evidence(
            [[sample]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
        )

    sample = _sample(group_index=0, index=0, value=1.0, status="aborted")
    with pytest.raises(ValueError, match="non-trainable"):
        build_trajectory_batch_evidence(
            [[sample]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
        )


def test_tampered_persisted_batch_hash_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    evidence = build_trajectory_batch_evidence(
        [[_sample(group_index=0, index=0, value=1.0)]],
        rollout_id=0,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    value = json.loads(path.read_text())
    value["envelopes"][0]["reward"] = 0.0
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    with pytest.raises(ValueError, match="batch hash changed"):
        read_trajectory_batch_evidence(path)


def test_tampered_persisted_trained_token_aggregate_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    evidence = build_trajectory_batch_evidence(
        [[_sample(group_index=0, index=0, value=1.0)]],
        rollout_id=0,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    value = json.loads(path.read_text())
    value["trained_tokens"] += 1
    path.write_text(json.dumps(value))
    path.chmod(0o600)

    with pytest.raises(ValueError, match="trained-token aggregate changed"):
        read_trajectory_batch_evidence(path)


def test_response_token_count_is_authenticated_by_batch_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    evidence = build_trajectory_batch_evidence(
        [[_sample(group_index=0, index=0, value=1.0)]],
        rollout_id=0,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    value = json.loads(path.read_text())
    value["envelopes"][0]["response_token_count"] = 1
    value["trained_tokens"] = 1
    path.write_text(json.dumps(value))
    path.chmod(0o600)

    with pytest.raises(ValueError, match="batch hash changed"):
        read_trajectory_batch_evidence(path)


def test_existing_public_evidence_root_is_rejected_without_chmod(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    evidence = build_trajectory_batch_evidence(
        [[_sample(group_index=0, index=0, value=1.0)]],
        rollout_id=0,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )
    with pytest.raises(RuntimeError, match="root is not private"):
        write_trajectory_batch_evidence(root, evidence)
    assert root.stat().st_mode & 0o777 == 0o755
