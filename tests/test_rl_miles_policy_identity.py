from types import SimpleNamespace

import pytest

from yeto.rl import miles

POLICY_HASH = "a" * 64


def test_trajectory_evidence_requires_exact_current_published_policy_identity(
    tmp_path,
):
    args = SimpleNamespace(
        yeto_rl_trajectory_evidence_dir=str(tmp_path),
        yeto_rl_policy_hash=POLICY_HASH,
    )

    with pytest.raises(RuntimeError, match="no current published policy identity"):
        miles._persist_trajectory_evidence(args, 3, ())

    miles.set_current_published_policy_identity(
        args,
        policy_version=3,
        policy_hash=POLICY_HASH,
    )
    assert miles.get_current_published_policy_identity(
        args,
        expected_policy_version=3,
    ) == (3, POLICY_HASH)
    with pytest.raises(RuntimeError, match="does not match the rollout"):
        miles._persist_trajectory_evidence(args, 4, ())


def test_published_policy_identity_cannot_change_at_a_version_or_regress():
    args = SimpleNamespace()
    miles.set_current_published_policy_identity(
        args,
        policy_version=3,
        policy_hash=POLICY_HASH,
    )

    with pytest.raises(RuntimeError, match="hash changed"):
        miles.set_current_published_policy_identity(
            args,
            policy_version=3,
            policy_hash="b" * 64,
        )
    with pytest.raises(RuntimeError, match="move backwards"):
        miles.set_current_published_policy_identity(
            args,
            policy_version=2,
            policy_hash=POLICY_HASH,
        )
