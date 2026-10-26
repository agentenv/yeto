from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from yeto.rl import tbench_outcome
from yeto.rl.trajectory_evidence import (
    build_trajectory_batch_evidence,
    read_trajectory_batch_evidence,
    write_trajectory_batch_evidence,
)


def _outcome_metadata(*, sample_id: str = "train:fix-git:r0") -> dict:
    return {
        "reward": 1.0,
        "exit_status": "completed",
        **tbench_outcome.build_outcome_metadata(
            task_id="fix-git",
            sample_id=sample_id,
            episode_id="openenv-0123456789abcdef",
            status="completed",
            reward=1.0,
            verifier=tbench_outcome.TEST_SH_VERIFIER,
            testsh_rc=0,
        ),
    }


def _segment(index: int, *, mask: list[int], metadata: dict | None = None):
    values = copy.deepcopy(metadata or _outcome_metadata())
    values.update(
        {
            "compaction_schema_version": 1,
            "compaction_trajectory_id": "session-0123456789abcdef",
            "compaction_context_window": index // 2,
            "compaction_context_budget": 8192,
            "compaction_segment_index": index,
            "compaction_segment_type": "execution" if index % 2 == 0 else "summary",
        }
    )
    return SimpleNamespace(
        group_index=3,
        index=7,
        status=SimpleNamespace(value="completed"),
        metadata=values,
        reward=1.0,
        tokens=[101, 200 + index, 210 + index, 220 + index],
        response_length=3,
        loss_mask=mask,
        rollout_log_probs=[-0.1 - index, -0.2 - index, -0.3 - index],
    )


def test_tbench_compaction_round_trips_composite_identity_and_active_tokens(tmp_path):
    segments = [
        _segment(0, mask=[1, 0, 1]),
        _segment(1, mask=[0, 1, 0]),
        _segment(2, mask=[1, 1, 1]),
    ]
    evidence = build_trajectory_batch_evidence(
        [[segments]],
        rollout_id=5,
        behavior_policy_hash="a" * 64,
        reward_contract_hash="b" * 64,
        evidence_kind="terminal-bench-2.1",
    )

    assert evidence.schema_version == 3
    assert evidence.evidence_kind == "terminal-bench-2.1"
    assert evidence.trained_tokens == 6
    assert [item.sample_index for item in evidence.envelopes] == [7, 7, 7]
    assert [item.compaction_segment_index for item in evidence.envelopes] == [0, 1, 2]
    assert [item.compaction_context_budget for item in evidence.envelopes] == [8192] * 3
    assert [item.active_token_count for item in evidence.envelopes] == [2, 1, 3]
    assert len(set(evidence.trajectory_ids)) == 3
    assert all(item.loss_mask_hash for item in evidence.envelopes)
    assert all(item.active_token_ids_hash for item in evidence.envelopes)
    assert all(item.active_behavior_logprobs_hash for item in evidence.envelopes)

    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    assert json.loads(path.read_text())["schema"] == 3
    assert read_trajectory_batch_evidence(path) == evidence


def test_tbench_compaction_rejects_inconsistent_outcome_and_segment_identity_drift():
    segments = [_segment(index, mask=[1, 1, 1]) for index in range(3)]
    changed = copy.deepcopy(segments)
    changed[1].metadata[tbench_outcome.OUTCOME_KEY]["reward"] = 0.0
    with pytest.raises(tbench_outcome.UntrustedTBenchOutcome):
        build_trajectory_batch_evidence(
            [[changed]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="terminal-bench-2.1",
        )

    segments[1].metadata["compaction_segment_index"] = 2
    with pytest.raises(ValueError, match="indexes are not contiguous"):
        build_trajectory_batch_evidence(
            [[segments]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="terminal-bench-2.1",
        )

    segments = [_segment(index, mask=[1, 1, 1]) for index in range(3)]
    segments[1].metadata["compaction_context_budget"] = 4096
    with pytest.raises(ValueError, match="context budget changed"):
        build_trajectory_batch_evidence(
            [[segments]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="terminal-bench-2.1",
        )


def test_tbench_active_token_contract_is_fail_closed():
    segment = _segment(0, mask=[1, 1, 1])
    segment.loss_mask = [1, 2, 0]
    with pytest.raises(ValueError, match="loss mask is malformed"):
        build_trajectory_batch_evidence(
            [[[segment]]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="terminal-bench-2.1",
        )

    segment = _segment(0, mask=[1, 1, 1])
    segment.rollout_log_probs = None
    with pytest.raises(ValueError, match="no behavior log probabilities"):
        build_trajectory_batch_evidence(
            [[[segment]]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="terminal-bench-2.1",
        )


def test_explicit_evidence_kind_rejects_cross_benchmark_metadata():
    segment = _segment(0, mask=[1, 1, 1])
    segment.metadata["secrlenv_trusted_outcome"] = {}
    with pytest.raises(ValueError, match="mixes benchmark evidence kinds"):
        build_trajectory_batch_evidence(
            [[[segment]]],
            rollout_id=0,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            evidence_kind="terminal-bench-2.1",
        )


ROUND_HASH = "a" * 64
OLDER_HASH = "c" * 64


def _two_trajectory_batch(behavior_policy_of=None):
    first = _segment(0, mask=[1, 1, 1])
    second = copy.deepcopy(_segment(0, mask=[1, 1, 1], metadata=_outcome_metadata(sample_id="train:fix-git:r1")))
    for sample, index in ((first, 7), (second, 8)):
        sample.metadata = {k: v for k, v in sample.metadata.items() if not k.startswith("compaction_")}
        sample.index = index
    return build_trajectory_batch_evidence(
        [[first, second]],
        rollout_id=5,
        behavior_policy_hash=ROUND_HASH,
        reward_contract_hash="b" * 64,
        evidence_kind="terminal-bench-2.1",
        behavior_policy_of=behavior_policy_of,
        **({"max_policy_age": 1} if behavior_policy_of else {}),
    )


def test_schema4_records_every_trajectory_policy_and_round_trips(tmp_path):
    """P0-S: a trajectory continued from an older policy carries that policy."""
    policies = iter(((4, OLDER_HASH), (5, ROUND_HASH)))
    evidence = _two_trajectory_batch(lambda segments: next(policies))

    assert evidence.schema_version == 4
    assert [(e.behavior_policy_version, e.behavior_policy_hash) for e in evidence.envelopes] == [
        (4, OLDER_HASH), (5, ROUND_HASH)]
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    assert read_trajectory_batch_evidence(path) == evidence
    # the batch hash covers the per-trajectory policies
    strict = _two_trajectory_batch()
    assert strict.schema_version == 3
    assert strict.input_batch_hash != evidence.input_batch_hash


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        ((6, ROUND_HASH), "newer than its round"),
        ((5, OLDER_HASH), "other weights"),
        ((3, OLDER_HASH), "policy-age limit"),
    ),
)
def test_schema4_refuses_a_policy_that_does_not_fit_the_round(policy, message):
    with pytest.raises(ValueError, match=message):
        _two_trajectory_batch(lambda segments: policy)


def test_schema4_tampered_trajectory_policy_is_rejected(tmp_path):
    evidence = _two_trajectory_batch(lambda segments: (4, OLDER_HASH))
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    value = json.loads(path.read_text())
    value["envelopes"][0]["behavior_policy_hash"] = "d" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="batch hash changed"):
        read_trajectory_batch_evidence(path)


def test_schema4_records_its_policy_age_limit(tmp_path):
    evidence = _two_trajectory_batch(lambda segments: (4, OLDER_HASH))
    assert evidence.max_policy_age == 1
    path = write_trajectory_batch_evidence(tmp_path / "private", evidence)
    value = json.loads(path.read_text())
    value["max_policy_age"] = 0  # a tighter limit the trajectories do not meet
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="policy-age limit"):
        read_trajectory_batch_evidence(path)
