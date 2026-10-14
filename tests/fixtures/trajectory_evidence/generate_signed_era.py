"""Regenerate the signed-era (schema 1 and 2) trajectory evidence fixtures.

These files must be produced by the code from before reward signing was
removed (commit f2db57f8), not by the current tree:

    git archive f2db57f8 yeto | tar -x -C /tmp/yeto-f2db57f8
    PYTHONPATH=/tmp/yeto-f2db57f8 python tests/fixtures/trajectory_evidence/generate_signed_era.py

The HMAC key below is a fixed test value; it is not used by any reader.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["SECRLENV_REWARD_HMAC_KEY"] = "s" * 48
os.environ["TBENCH_REWARD_HMAC_KEY"] = "t" * 48
os.environ.pop("TBENCH_REWARD_HMAC_KEY_FILE", None)

from yeto.rl import tbench_outcome  # noqa: E402
from yeto.rl.harness.codex import reward  # noqa: E402
from yeto.rl.trajectory_evidence import (  # noqa: E402
    build_trajectory_batch_evidence,
    write_trajectory_batch_evidence,
)

HERE = Path(__file__).resolve().parent


def _secrlenv_sample(index: int, value: float) -> SimpleNamespace:
    outcome = {
        "schema": 1,
        "status": "completed",
        "episode_id": f"episode-{index}",
        "task_id": f"CVE-2026-{index:04d}",
        "reward": value,
        "passed": value == 1.0,
        "class": None,
        reward.INFRASTRUCTURE_503_RETRIES_KEY: 0,
    }
    return SimpleNamespace(
        group_index=index,
        index=index,
        status=SimpleNamespace(value="completed"),
        metadata={reward.OUTCOME_KEY: outcome, reward.MAC_KEY: reward.sign_outcome(outcome)},
        reward=value,
        tokens=[101, 102 + index, 103 + index],
        response_length=2,
        rollout_log_probs=[-0.25, -0.5],
    )


def _tbench_sample(index: int, value: float) -> SimpleNamespace:
    metadata = tbench_outcome.build_signed_metadata(
        task_id=f"fix-git-{index}",
        sample_id=f"train:fix-git-{index}:r0",
        episode_id=f"openenv-{index:016x}",
        status="completed",
        reward=value,
        verifier=tbench_outcome.TEST_SH_VERIFIER,
        testsh_rc=0 if value else 1,
    )
    return SimpleNamespace(
        group_index=index,
        index=index,
        status=SimpleNamespace(value="completed"),
        metadata={"reward": value, **metadata},
        reward=value,
        tokens=[101, 200 + index, 210 + index, 220 + index],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.1, -0.2, -0.3],
    )


def _write(name: str, evidence) -> None:
    with tempfile.TemporaryDirectory() as root:
        path = write_trajectory_batch_evidence(Path(root) / "private", evidence)
        shutil.copyfile(path, HERE / name)


def main() -> None:
    _write(
        "schema1-secrlenv.json",
        build_trajectory_batch_evidence(
            [[_secrlenv_sample(0, 0.0)], [_secrlenv_sample(1, 1.0)]],
            rollout_id=3,
            behavior_policy_hash="a" * 64,
            reward_contract_hash="b" * 64,
            schema_version=1,
        ),
    )
    _write(
        "schema2-tbench.json",
        build_trajectory_batch_evidence(
            [[_tbench_sample(0, 1.0)], [_tbench_sample(1, 0.0)]],
            rollout_id=4,
            behavior_policy_hash="c" * 64,
            reward_contract_hash="d" * 64,
            evidence_kind="terminal-bench-2.1",
        ),
    )


if __name__ == "__main__":
    main()
