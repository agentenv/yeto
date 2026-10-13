from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.legacy import engine as miles
from yeto.rl.adapters.miles.models import rollout_legacy_port as rollout_port
from yeto.rl.adapters.miles import publish

POLICY_HASH = "a" * 64


def test_trajectory_evidence_requires_exact_current_published_policy_identity(
    tmp_path,
):
    args = SimpleNamespace(
        yeto_rl_trajectory_evidence_dir=str(tmp_path),
        yeto_rl_policy_hash=POLICY_HASH,
    )

    with pytest.raises(RuntimeError, match="no current published policy identity"):
        rollout_port._persist_trajectory_evidence(args, 3, ())

    publish.set_current_published_policy_identity(
        args,
        policy_version=3,
        policy_hash=POLICY_HASH,
    )
    assert args.yeto_rl_policy_token == f"yeto:3:{POLICY_HASH}"
    assert rollout_port._policy_token_for_rollout(
        SimpleNamespace(
            yeto_rl_sync_preset="decoupled",
            yeto_rl_policy_token=args.yeto_rl_policy_token,
        ),
        3,
    ) == f"yeto:3:{POLICY_HASH}"
    assert publish.get_current_published_policy_identity(
        args,
        expected_policy_version=3,
    ) == (3, POLICY_HASH)
    with pytest.raises(RuntimeError, match="does not match the rollout"):
        rollout_port._persist_trajectory_evidence(args, 4, ())


def test_published_policy_identity_cannot_change_at_a_version_or_regress():
    args = SimpleNamespace()
    publish.set_current_published_policy_identity(
        args,
        policy_version=3,
        policy_hash=POLICY_HASH,
    )
    original_token = args.yeto_rl_policy_token

    with pytest.raises(RuntimeError, match="hash changed"):
        publish.set_current_published_policy_identity(
            args,
            policy_version=3,
            policy_hash="b" * 64,
        )
    with pytest.raises(RuntimeError, match="move backwards"):
        publish.set_current_published_policy_identity(
            args,
            policy_version=2,
            policy_hash=POLICY_HASH,
        )
    assert args.yeto_rl_policy_token == original_token
