import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl import trajectory_evidence
from yeto.rl import secrlenv_outcome as reward
from yeto.rl.trajectory_evidence import (
    build_trajectory_batch_evidence,
    read_trajectory_batch_evidence,
    write_trajectory_batch_evidence,
)


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
            "untrusted_arbitrary_payload": {"must": "not be persisted"},
        },
        reward=value,
        tokens=[101, 102 + index, 103 + index],
        response_length=2,
        rollout_log_probs=[-0.25, -0.5],
    )


def test_validated_batch_round_trips_without_arbitrary_metadata(tmp_path):
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
    assert evidence.schema_version == 3
    assert evidence.evidence_kind == "secrlenv"

    root = tmp_path / "private"
    path = write_trajectory_batch_evidence(root, evidence)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["schema"] == 3
    assert read_trajectory_batch_evidence(path) == evidence
    assert write_trajectory_batch_evidence(root, evidence) == path
    raw = path.read_text()
    assert "untrusted_arbitrary_payload" not in raw
    assert "episode-" not in raw


def test_invalid_reward_or_nonterminal_status_fails_before_evidence():
    sample = _sample(group_index=0, index=0, value=1.0)
    sample.metadata[reward.OUTCOME_KEY]["reward"] = 1.5
    with pytest.raises(reward.UntrustedOutcome, match="outside"):
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


def test_tampered_persisted_batch_hash_is_rejected(tmp_path):
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


def test_tampered_persisted_trained_token_aggregate_is_rejected(tmp_path):
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


def test_tampered_response_token_count_is_rejected(tmp_path):
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

    with pytest.raises(ValueError, match="exceeds its response length|batch hash changed"):
        read_trajectory_batch_evidence(path)


def test_existing_public_evidence_root_is_rejected_without_chmod(tmp_path):
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


def _batch():
    return build_trajectory_batch_evidence(
        [[_sample(group_index=0, index=0, value=1.0)]],
        rollout_id=0,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )


def test_outcome_hashes_ignore_legacy_mac_field_and_reward_key_env(monkeypatch):
    plain = _batch()
    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "s" * 48)
    legacy = _sample(group_index=0, index=0, value=1.0)
    legacy.metadata["secrlenv_trusted_outcome_hmac"] = "0" * 64
    with_mac = build_trajectory_batch_evidence(
        [[legacy]],
        rollout_id=0,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
    )
    assert with_mac == plain


_FIXTURES = Path(__file__).parent / "fixtures" / "trajectory_evidence"


def _private_copy(tmp_path: Path, name: str) -> Path:
    """Readers require a private file; git does not keep the mode."""

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "rollout-00000000.json"
    shutil.copyfile(_FIXTURES / name, path)
    path.chmod(0o600)
    return path


@pytest.mark.parametrize(
    ("name", "schema_version", "evidence_kind", "envelopes"),
    [
        ("schema1-secrlenv.json", 1, "secrlenv", 2),
        ("schema2-tbench.json", 2, "terminal-bench-2.1", 2),
    ],
)
def test_signed_era_evidence_written_by_old_code_is_still_readable(
    tmp_path, name, schema_version, evidence_kind, envelopes
):
    # Fixtures were written by commit f2db57f8 (signed rewards); see
    # tests/fixtures/trajectory_evidence/generate_signed_era.py.
    path = _private_copy(tmp_path, name)
    raw = json.loads(path.read_text())

    evidence = read_trajectory_batch_evidence(path)
    assert evidence.schema_version == schema_version
    assert evidence.evidence_kind == evidence_kind
    assert len(evidence.envelopes) == envelopes
    assert evidence.input_batch_hash == raw["input_batch_hash"]

    raw["envelopes"][0]["reward"] = 1.0 - raw["envelopes"][0]["reward"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="batch hash changed"):
        read_trajectory_batch_evidence(path)


def test_unknown_schema_is_rejected(tmp_path):
    path = write_trajectory_batch_evidence(tmp_path / "private", _batch())
    value = json.loads(path.read_text())
    value["schema"] = 5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="schema is unsupported"):
        read_trajectory_batch_evidence(path)


def test_schema_3_records_every_layered_field():
    item = _batch().envelopes[0]
    assert set(asdict(item)) == set(trajectory_evidence._LAYERED_ENVELOPE_FIELDS)
    assert item.active_token_count == item.response_token_count


def _reward_function_sample(*, group_index, index, value, prompt="2+2?"):
    """A sample scored by an ordinary reward function (e.g. GSM8K): no grader
    verdict in the metadata, the reward is ``sample.reward``."""

    return SimpleNamespace(
        group_index=group_index,
        index=index,
        status=SimpleNamespace(value="completed"),
        metadata={},
        prompt=[{"role": "user", "content": prompt}],
        reward=value,
        tokens=[101, 102 + index, 103 + index],
        response_length=2,
        rollout_log_probs=[-0.25, -0.5],
    )


def test_reward_function_evidence_binds_the_reward_function_output(tmp_path):
    groups = [
        [
            _reward_function_sample(group_index=4, index=0, value=1.0),
            _reward_function_sample(group_index=4, index=1, value=0.0),
        ]
    ]
    evidence = build_trajectory_batch_evidence(
        groups,
        rollout_id=2,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
        evidence_kind="reward-function",
        behavior_policy_of=lambda _segments: (2, "a" * 64),
        max_policy_age=0,
    )

    assert evidence.schema_version == 4
    assert evidence.evidence_kind == "reward-function"
    assert [item.reward for item in evidence.envelopes] == [1.0, 0.0]
    assert [item.behavior_policy_version for item in evidence.envelopes] == [2, 2]
    # one prompt group shares one task identity
    assert len({item.task_id for item in evidence.envelopes}) == 1
    assert evidence.envelopes[0].task_id.startswith("prompt-sha256:")
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    assert read_trajectory_batch_evidence(path) == evidence


def test_reward_function_evidence_hash_changes_with_the_reward():
    def build(value):
        return build_trajectory_batch_evidence(
            [[_reward_function_sample(group_index=0, index=0, value=value)]],
            rollout_id=1,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="reward-function",
        )

    assert build(1.0).input_batch_hash != build(0.0).input_batch_hash


@pytest.mark.parametrize("value", (None, True, float("nan"), {"score": 1.0}))
def test_reward_function_evidence_requires_a_finite_scalar_reward(value):
    with pytest.raises(ValueError, match="finite scalar reward"):
        build_trajectory_batch_evidence(
            [[_reward_function_sample(group_index=0, index=0, value=value)]],
            rollout_id=1,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="reward-function",
        )


def test_grader_verdict_cannot_be_downgraded_to_reward_function_evidence():
    # A SecRLEnv sample keeps its trusted-grader evidence kind.
    with pytest.raises(ValueError, match="grader verdict"):
        build_trajectory_batch_evidence(
            [[_sample(group_index=0, index=0, value=1.0)]],
            rollout_id=1,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="reward-function",
        )


def test_secrlenv_evidence_still_rejects_a_sample_without_a_verdict():
    with pytest.raises(reward.UntrustedOutcome):
        build_trajectory_batch_evidence(
            [[_reward_function_sample(group_index=0, index=0, value=1.0)]],
            rollout_id=1,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="secrlenv",
        )
