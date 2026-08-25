from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yeto.rl.m1_online_comparison import (
    ARMS,
    COMPARISON_SCHEMA,
    QWEN_MODEL,
    QWEN_REVISION,
    RUN_SCHEMA,
    SEEDS,
    ComparisonError,
    study_contract_sha256,
    verify_comparison,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _digest(payload)


def _artifact(root: Path, arm: str, seed: int, name: str) -> dict[str, str]:
    relative = f"artifacts/{arm}-{seed}-{name}.json"
    payload = _canonical({"arm": arm, "name": name, "seed": seed})
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": _digest(payload)}


def _study_manifest() -> dict:
    def hash_of(name: str) -> str:
        return _digest(f"contract:{name}")

    return {
        "schema": COMPARISON_SCHEMA,
        "study_id": "qwen35-full-parameter-m1-paired",
        "seeds": list(SEEDS),
        "arms": {
            "centralized": {
                "kind": "centralized_native_full_parameter",
                "islands": 1,
                "learners": 1,
                "trainer_gpus": 4,
                "inference_gpus": 4,
                "total_gpus": 8,
                "inference_workers": 4,
                "local_optimizer_steps_per_policy_round": 1,
            },
            "dense_two_island": {
                "kind": "dense_two_island_full_parameter",
                "islands": 2,
                "learners": 2,
                "trainer_gpus": 4,
                "inference_gpus": 4,
                "total_gpus": 8,
                "inference_workers": 4,
                "local_optimizer_steps_per_policy_round": 1,
            },
        },
        "inputs": {
            "model": {
                "logical_id": QWEN_MODEL,
                "revision": QWEN_REVISION,
                "initial_manifest_sha256": hash_of("initial-manifest"),
                "initial_policy_hash": hash_of("initial-policy"),
                "parameter_layout_hash": hash_of("layout"),
            },
            "tokenizer": {
                "logical_id": QWEN_MODEL,
                "revision": QWEN_REVISION,
                "contract_sha256": hash_of("tokenizer"),
            },
            "data": {
                "task_pack_sha256": hash_of("task-pack"),
                "train_sha256": hash_of("train"),
                "heldout_sha256": hash_of("heldout"),
                "prompt_schedule_sha256": hash_of("prompt-schedule"),
            },
            "reward": {
                "source_sha256": hash_of("reward-source"),
                "contract_sha256": hash_of("reward-contract"),
            },
            "sampling": {
                "contract_sha256": hash_of("sampling"),
                "samples_per_group": 3,
                "max_response_tokens": 256,
            },
            "optimizer": {
                "name": "adam",
                "contract_sha256": hash_of("optimizer"),
            },
            "implementation": {
                "container_image_id": f"sha256:{hash_of('container')}",
                "yeto_source_sha256": hash_of("yeto"),
                "miles_source_sha256": hash_of("miles"),
            },
            "evaluation": {
                "contract_sha256": hash_of("evaluation"),
                "prompt_count": 2,
                "samples_per_prompt": 1,
            },
        },
        "budget": {
            "policy_rounds": 2,
            "accepted_groups": 4,
            "accepted_trajectories": 12,
            "trained_tokens": 120,
            "centralized_optimizer_steps": 2,
            "dense_local_optimizer_steps": 4,
            "trainer_gpus": 4,
            "inference_gpus": 4,
            "total_gpus": 8,
        },
        "non_inferiority": {
            "method": "paired_all_seeds_and_mean_improvement",
            "primary_metric": "heldout_pass_at_1_improvement",
            "pass_at_1_absolute_margin": 0.05,
            "result_absolute_margin": 0.05,
        },
        "runs": [],
    }


def _evaluation(
    manifest: dict,
    *,
    seed: int,
    arm: str,
    version: int,
    phase: str,
    result: float,
    pass_at_1: float,
    evidence_sha256: str,
) -> dict:
    return {
        "policy_version": version,
        "dataset_sha256": manifest["inputs"]["data"]["heldout_sha256"],
        "evaluation_contract_sha256": manifest["inputs"]["evaluation"][
            "contract_sha256"
        ],
        "generation_seed_manifest_sha256": _digest(f"eval-seeds:{seed}:{phase}"),
        "sample_manifest_sha256": _digest(
            f"eval-samples:{seed}:{phase}:{'shared' if phase == 'v0' else arm}"
        ),
        "sample_count": 2,
        "result": result,
        "pass_at_1": pass_at_1,
        "evidence_sha256": evidence_sha256,
    }


def _run_record(root: Path, manifest: dict, arm: str, seed: int) -> dict:
    initial_hash = manifest["inputs"]["model"]["initial_policy_hash"]
    final_hash = _digest(f"final-policy:{arm}:{seed}")
    learners = 1 if arm == "centralized" else 2
    local_steps = 1
    rounds = []
    for round_id in range(2):
        learner_rows = []
        for learner_id in range(learners):
            learner_rows.append(
                {
                    "learner_id": learner_id,
                    "accepted_groups": 2 // learners,
                    "accepted_trajectories": 6 // learners,
                    "trained_tokens": 60 // learners,
                    "optimizer_steps": local_steps,
                    "mean_kl": 0.01 * (round_id + 1),
                    "local_update_l2_norm": 0.08 + learner_id * 0.01,
                }
            )
        rounds.append(
            {
                "base_policy_version": round_id,
                "target_policy_version": round_id + 1,
                "prompt_group_manifest_sha256": _digest(
                    f"prompt-groups:{seed}:{round_id}"
                ),
                "generation_seed_manifest_sha256": _digest(
                    f"rollout-seeds:{seed}:{round_id}"
                ),
                "accepted_groups": 2,
                "accepted_trajectories": 6,
                "trained_tokens": 60,
                "optimizer_steps": learners,
                "mean_kl": 0.01 * (round_id + 1),
                "global_update_l2_norm": 0.1 + round_id * 0.01,
                "learners": learner_rows,
            }
        )
    version_one_hash = _digest(f"policy-v1:{arm}:{seed}")
    publications = []
    for version, policy_hash in enumerate((initial_hash, version_one_hash, final_hash)):
        publications.append(
            {
                "policy_version": version,
                "policy_hash": policy_hash,
                "inference_worker_ids": [0, 1, 2, 3],
                "receipt_manifest_sha256": _digest(
                    f"publication:{arm}:{seed}:{version}"
                ),
            }
        )
    central = arm == "centralized"
    artifacts = {
        name: _artifact(root, arm, seed, name)
        for name in (
            "run_manifest",
            "trajectory_accounting",
            "optimizer_receipts",
            "publication_receipts",
            "v0_heldout",
            "final_heldout",
            "final_policy_manifest",
        )
    }
    return {
        "schema": RUN_SCHEMA,
        "study_contract_sha256": study_contract_sha256(manifest),
        "arm": arm,
        "seed": seed,
        "artifacts": artifacts,
        "initial_policy_hash": initial_hash,
        "final_policy_hash": final_hash,
        "accounting": {
            "initial_policy_version": 0,
            "final_policy_version": 2,
            "policy_rounds": 2,
            "accepted_groups": 4,
            "accepted_trajectories": 12,
            "trained_tokens": 120,
            "optimizer_steps": 2 if central else 4,
        },
        "rounds": rounds,
        "publications": publications,
        "evaluations": {
            "v0": _evaluation(
                manifest,
                seed=seed,
                arm=arm,
                version=0,
                phase="v0",
                result=0.25,
                pass_at_1=0.2,
                evidence_sha256=artifacts["v0_heldout"]["sha256"],
            ),
            "final": _evaluation(
                manifest,
                seed=seed,
                arm=arm,
                version=2,
                phase="final",
                result=0.5 if central else 0.48,
                pass_at_1=0.55 if central else 0.52,
                evidence_sha256=artifacts["final_heldout"]["sha256"],
            ),
        },
    }


def _comparison(tmp_path: Path) -> Path:
    manifest = _study_manifest()
    for seed in SEEDS:
        for arm in ARMS:
            run = _run_record(tmp_path, manifest, arm, seed)
            relative = f"runs/{arm}-{seed}.json"
            digest = _write_json(tmp_path / relative, run)
            manifest["runs"].append(
                {"arm": arm, "seed": seed, "path": relative, "sha256": digest}
            )
    path = tmp_path / "comparison.json"
    _write_json(path, manifest)
    return path


def _mutate_run(comparison: Path, arm: str, seed: int, mutate) -> None:
    manifest = json.loads(comparison.read_text())
    binding = next(
        row for row in manifest["runs"] if row["arm"] == arm and row["seed"] == seed
    )
    run_path = comparison.parent / binding["path"]
    run = json.loads(run_path.read_text())
    mutate(run)
    binding["sha256"] = _write_json(run_path, run)
    _write_json(comparison, manifest)


def test_verifies_closed_paired_online_comparison(tmp_path: Path) -> None:
    result = verify_comparison(_comparison(tmp_path))

    assert result["accounting_verified"] is True
    assert result["paired_prompt_and_evaluation_seeds_verified"] is True
    assert result["non_inferiority"] == {
        "method": "paired_all_seeds_and_mean_improvement",
        "primary_metric": "heldout_pass_at_1_improvement",
        "passed": True,
        "conclusion": "non_inferior",
    }
    assert result["pass_at_1"]["mean_paired_improvement_difference"] == pytest.approx(
        -0.03
    )


def test_rejects_unpaired_prompt_schedule_even_with_rehashed_record(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)
    _mutate_run(
        comparison,
        "dense_two_island",
        17,
        lambda run: run["rounds"][0].__setitem__(
            "prompt_group_manifest_sha256", _digest("different-prompt-schedule")
        ),
    )

    with pytest.raises(ComparisonError, match="different prompt-group schedules"):
        verify_comparison(comparison)


def test_valid_negative_result_is_not_reported_as_non_inferior(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)

    def lower_dense_result(run: dict) -> None:
        run["evaluations"]["final"]["pass_at_1"] = 0.1
        run["evaluations"]["final"]["result"] = 0.1

    _mutate_run(comparison, "dense_two_island", 29, lower_dense_result)
    result = verify_comparison(comparison)

    assert result["non_inferiority"]["passed"] is False
    assert result["non_inferiority"]["conclusion"] == "not_demonstrated"
    assert result["pass_at_1"]["per_seed"][1]["within_margin"] is False


def test_rejects_artifact_path_through_intermediate_symlink(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    manifest = json.loads(comparison.read_text())
    binding = manifest["runs"][0]
    run_path = comparison.parent / binding["path"]
    run = json.loads(run_path.read_text())
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "artifact.json"
    target.write_text("{}\n")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    run["artifacts"]["run_manifest"] = {
        "path": "linked/artifact.json",
        "sha256": _digest(target.read_bytes()),
    }
    binding["sha256"] = _write_json(run_path, run)
    _write_json(comparison, manifest)

    with pytest.raises(ComparisonError, match="traverses a symlink"):
        verify_comparison(comparison)
