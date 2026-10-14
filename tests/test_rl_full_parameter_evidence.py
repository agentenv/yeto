"""Ports rollout hook writes the full-parameter trajectory evidence (P0-1.7b.3).

The ``record_trained_groups`` hook is the one writer: it binds the kept batch,
after the final loss masks, to the policy token the driver published for the
round.  ``rollout_legacy_port.generate_rollout`` (still the ports
``--rollout-function-path``) no longer writes evidence; a second write of the
same round would fail the exclusive create or bind the pre-filter batch.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl import trajectory_evidence
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.full_parameter import evidence

POLICY_HASH = "a" * 64
REWARD_HASH = "b" * 64


@pytest.fixture
def written(monkeypatch):
    calls = []

    def build(groups, **fields):
        calls.append((groups, fields))
        return ("evidence", fields["rollout_id"])

    monkeypatch.setattr(trajectory_evidence, "build_trajectory_batch_evidence", build)
    monkeypatch.setattr(
        trajectory_evidence,
        "write_trajectory_batch_evidence",
        lambda directory, value: calls.append(("write", directory, value)),
    )
    return calls


def _args(tmp_path, **overrides):
    values = {
        "yeto_rl_trajectory_evidence_dir": str(tmp_path / "evidence"),
        "yeto_rl_trajectory_evidence_kind": "secrlenv",
        "yeto_rl_trajectory_evidence_schema_version": 1,
        "yeto_rl_reward_sha256": REWARD_HASH,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_published_token_binds_round_and_behavior_policy(tmp_path, written):
    groups = [["sample"]]

    evidence.persist_published_round_evidence(
        _args(tmp_path), groups, f"yeto:7:{POLICY_HASH}"
    )

    (built_groups, fields), write = written
    assert built_groups is groups
    assert fields == {
        "rollout_id": 7,
        "behavior_policy_hash": POLICY_HASH,
        "reward_contract_hash": REWARD_HASH,
        "evidence_kind": "secrlenv",
        "schema_version": 1,
    }
    assert write == ("write", str(tmp_path / "evidence"), ("evidence", 7))


def _kept_batch():
    return [[SimpleNamespace(group_index=0, index=0, tokens=[1], response_length=1,
                             loss_mask=[1], metadata={}, status=None)]]


def _hook_round(tmp_path, monkeypatch, rollout_id: int) -> None:
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{tmp_path}")
    hook.put_policy_token(f"yeto:{rollout_id}:{POLICY_HASH}", f"dir:{tmp_path}")
    monkeypatch.setattr("yeto.rl.algos.sample_filters.apply_sample_filters", lambda *_a: None)


def test_without_an_evidence_dir_the_hook_writes_nothing(tmp_path, monkeypatch, written):
    _hook_round(tmp_path, monkeypatch, 2)
    hook.record_trained_groups(SimpleNamespace(), _kept_batch())
    assert written == []


@pytest.mark.parametrize(
    ("overrides", "token", "message"),
    (
        ({}, None, "published policy token"),
        ({}, "yeto:7", "invalid policy snapshot token"),
        ({"yeto_rl_trajectory_evidence_kind": "gsm8k"}, f"yeto:7:{POLICY_HASH}", "kind"),
        ({"yeto_rl_trajectory_evidence_schema_version": 3}, f"yeto:7:{POLICY_HASH}", "schema"),
    ),
)
def test_unbound_context_fails_closed(tmp_path, written, overrides, token, message):
    with pytest.raises((RuntimeError, ValueError), match=message):
        evidence.persist_published_round_evidence(_args(tmp_path, **overrides), [["s"]], token)
    assert written == []


def test_record_trained_groups_writes_the_kept_batch_once(tmp_path, monkeypatch, written):
    _hook_round(tmp_path, monkeypatch, 2)
    kept = _kept_batch()

    hook.record_trained_groups(_args(tmp_path), kept)

    (built_groups, fields), write = written
    assert built_groups is kept
    assert (fields["rollout_id"], fields["behavior_policy_hash"]) == (2, POLICY_HASH)
    assert write == ("write", str(tmp_path / "evidence"), ("evidence", 2))


def test_the_hook_is_the_only_evidence_writer():
    """No rollout function writes round evidence beside the hook (one per round)."""

    root = Path(evidence.__file__).resolve().parents[5]
    writers = sorted(
        str(path.relative_to(root))
        for path in (root / "yeto").rglob("*.py")
        if "write_trajectory_batch_evidence(" in path.read_text(encoding="utf-8")
        or "persist_published_round_evidence(" in path.read_text(encoding="utf-8")
    )
    assert writers == [
        "yeto/rl/adapters/miles/full_parameter/evidence.py",
        "yeto/rl/adapters/miles/rollout_meta_hook.py",
        "yeto/rl/trajectory_evidence.py",
    ]
