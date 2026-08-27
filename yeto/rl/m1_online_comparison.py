"""Verify the closed Milestone-1 full-parameter online comparison.

This module intentionally does not produce benchmark data.  It verifies six
already-produced Qwen3.5 records: a same-compute centralized Miles arm and a
two-island dense arm for seeds 17, 29, and 43.  The study contract is hashed
before run records are loaded, so a result cannot silently change the model,
data, reward, sampling, optimizer, budget, or non-inferiority margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
from pathlib import Path, PurePosixPath
from typing import Any

COMPARISON_SCHEMA = "yeto-qwen35-m1-online-comparison/v1"
RUN_SCHEMA = "yeto-qwen35-m1-online-arm-result/v1"
VERIFIED_SCHEMA = "yeto-qwen35-m1-online-comparison-result/v1"
QWEN_MODEL = "Qwen/Qwen3.5-4B"
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEEDS = (17, 29, 43)
ARMS = ("centralized", "dense_two_island")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STUDY_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


class ComparisonError(RuntimeError):
    """The proposed M1 comparison is incomplete, unpaired, or malformed."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ComparisonError(f"duplicate JSON key: {name!r}")
        result[name] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, name: str, *, maximum: int) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ComparisonError(f"missing {name}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ComparisonError(f"{name} is not a regular non-symlink file: {path}")
    if not 0 < info.st_size <= maximum:
        raise ComparisonError(f"{name} has an invalid size: {path}")


def _load_canonical(path: Path, name: str) -> dict[str, Any]:
    _regular(path, name, maximum=_MAX_JSON_BYTES)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid {name}: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ComparisonError(f"{name} is not canonical JSON: {path}")
    return value


def _object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ComparisonError(
            f"{name} keys differ: expected={sorted(keys)!r} actual={actual!r}"
        )
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ComparisonError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or float(value) < minimum
    ):
        raise ComparisonError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _signed_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ComparisonError(f"{name} must be finite")
    return float(value)


def _metric(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result > 1.0:
        raise ComparisonError(f"{name} must be in [0, 1]")
    return result


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ComparisonError(f"{name} must be a lowercase SHA256")
    return value


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _resolve_binding(root: Path, binding: Any, name: str) -> Path:
    row = _object(binding, {"path", "sha256"}, name)
    relative = row["path"]
    if not isinstance(relative, str):
        raise ComparisonError(f"{name}.path must be a relative POSIX path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise ComparisonError(f"{name}.path is unsafe")
    path = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ComparisonError(f"{name}.path traverses a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"{name}.path escapes the comparison directory") from exc
    _regular(path, name, maximum=_MAX_ARTIFACT_BYTES)
    if _sha256(path) != _hash(row["sha256"], f"{name}.sha256"):
        raise ComparisonError(f"{name} SHA256 differs")
    return path


def _validate_arm(name: str, value: Any) -> dict[str, Any]:
    arm = _object(
        value,
        {
            "kind",
            "islands",
            "learners",
            "trainer_gpus",
            "inference_gpus",
            "total_gpus",
            "inference_workers",
            "local_optimizer_steps_per_policy_round",
        },
        f"arms.{name}",
    )
    expected_kind = {
        "centralized": "centralized_native_full_parameter",
        "dense_two_island": "dense_two_island_full_parameter",
    }[name]
    expected_count = 1 if name == "centralized" else 2
    if arm["kind"] != expected_kind:
        raise ComparisonError(f"arms.{name}.kind differs")
    if (
        _integer(arm["islands"], f"arms.{name}.islands", minimum=1) != expected_count
        or _integer(arm["learners"], f"arms.{name}.learners", minimum=1)
        != expected_count
    ):
        raise ComparisonError(f"arms.{name} has the wrong learner topology")
    trainer = _integer(arm["trainer_gpus"], f"arms.{name}.trainer_gpus", minimum=1)
    inference = _integer(
        arm["inference_gpus"], f"arms.{name}.inference_gpus", minimum=1
    )
    total = _integer(arm["total_gpus"], f"arms.{name}.total_gpus", minimum=1)
    workers = _integer(
        arm["inference_workers"], f"arms.{name}.inference_workers", minimum=1
    )
    local_steps = _integer(
        arm["local_optimizer_steps_per_policy_round"],
        f"arms.{name}.local_optimizer_steps_per_policy_round",
        minimum=1,
    )
    if trainer + inference != total or workers > inference:
        raise ComparisonError(f"arms.{name} GPU/worker accounting differs")
    if (trainer, inference, total, workers) != (4, 4, 8, 4):
        raise ComparisonError(
            f"arms.{name} must use the fixed 4-trainer/4-inference GPU budget"
        )
    if name == "dense_two_island" and trainer % 2:
        raise ComparisonError("dense trainer GPU count must divide two islands")
    if name == "dense_two_island" and inference % 2:
        raise ComparisonError("dense inference GPU count must divide two islands")
    if local_steps != 1:
        raise ComparisonError(
            f"arms.{name} must use H=1 (one local optimizer step per policy round)"
        )
    return arm


def _validate_inputs(value: Any) -> dict[str, Any]:
    inputs = _object(
        value,
        {
            "model",
            "tokenizer",
            "data",
            "reward",
            "sampling",
            "optimizer",
            "implementation",
            "evaluation",
        },
        "inputs",
    )
    model = _object(
        inputs["model"],
        {
            "logical_id",
            "revision",
            "initial_manifest_sha256",
            "initial_policy_hash",
            "parameter_layout_hash",
        },
        "inputs.model",
    )
    if model["logical_id"] != QWEN_MODEL or model["revision"] != QWEN_REVISION:
        raise ComparisonError("comparison is not the pinned Qwen3.5-4B model")
    for key in (
        "initial_manifest_sha256",
        "initial_policy_hash",
        "parameter_layout_hash",
    ):
        _hash(model[key], f"inputs.model.{key}")

    tokenizer = _object(
        inputs["tokenizer"],
        {"logical_id", "revision", "contract_sha256"},
        "inputs.tokenizer",
    )
    if tokenizer["logical_id"] != QWEN_MODEL or tokenizer["revision"] != QWEN_REVISION:
        raise ComparisonError("tokenizer differs from the pinned model revision")
    _hash(tokenizer["contract_sha256"], "inputs.tokenizer.contract_sha256")

    for section, keys in (
        (
            "data",
            {
                "task_pack_sha256",
                "train_sha256",
                "heldout_sha256",
                "prompt_schedule_sha256",
            },
        ),
        ("reward", {"source_sha256", "contract_sha256"}),
    ):
        row = _object(inputs[section], keys, f"inputs.{section}")
        for key in keys:
            _hash(row[key], f"inputs.{section}.{key}")

    sampling = _object(
        inputs["sampling"],
        {"contract_sha256", "samples_per_group", "max_response_tokens"},
        "inputs.sampling",
    )
    _hash(sampling["contract_sha256"], "inputs.sampling.contract_sha256")
    _integer(
        sampling["samples_per_group"],
        "inputs.sampling.samples_per_group",
        minimum=2,
    )
    _integer(
        sampling["max_response_tokens"],
        "inputs.sampling.max_response_tokens",
        minimum=1,
    )

    optimizer = _object(
        inputs["optimizer"],
        {"name", "contract_sha256"},
        "inputs.optimizer",
    )
    if optimizer["name"] != "adam":
        raise ComparisonError("M1 comparison requires the same Adam contract")
    _hash(optimizer["contract_sha256"], "inputs.optimizer.contract_sha256")

    implementation = _object(
        inputs["implementation"],
        {"container_image_id", "yeto_source_sha256", "miles_source_sha256"},
        "inputs.implementation",
    )
    if (
        not isinstance(implementation["container_image_id"], str)
        or _IMAGE_ID.fullmatch(implementation["container_image_id"]) is None
    ):
        raise ComparisonError("inputs.implementation.container_image_id is malformed")
    _hash(
        implementation["yeto_source_sha256"], "inputs.implementation.yeto_source_sha256"
    )
    _hash(
        implementation["miles_source_sha256"],
        "inputs.implementation.miles_source_sha256",
    )

    evaluation = _object(
        inputs["evaluation"],
        {"contract_sha256", "prompt_count", "samples_per_prompt"},
        "inputs.evaluation",
    )
    _hash(evaluation["contract_sha256"], "inputs.evaluation.contract_sha256")
    _integer(evaluation["prompt_count"], "inputs.evaluation.prompt_count", minimum=1)
    _integer(
        evaluation["samples_per_prompt"],
        "inputs.evaluation.samples_per_prompt",
        minimum=1,
    )
    return inputs


def _validate_budget(value: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    budget = _object(
        value,
        {
            "policy_rounds",
            "accepted_groups",
            "accepted_trajectories",
            "trained_tokens",
            "centralized_optimizer_steps",
            "dense_local_optimizer_steps",
            "trainer_gpus",
            "inference_gpus",
            "total_gpus",
        },
        "budget",
    )
    for name in budget:
        _integer(budget[name], f"budget.{name}", minimum=1)
    if budget["trainer_gpus"] + budget["inference_gpus"] != budget["total_gpus"]:
        raise ComparisonError("budget GPU accounting differs")
    if (
        budget["trainer_gpus"],
        budget["inference_gpus"],
        budget["total_gpus"],
    ) != (4, 4, 8):
        raise ComparisonError("comparison requires the fixed eight-GPU M1 budget")
    if (
        budget["accepted_groups"] * inputs["sampling"]["samples_per_group"]
        != budget["accepted_trajectories"]
    ):
        raise ComparisonError("trajectory budget differs from group size")
    if budget["trained_tokens"] < budget["accepted_trajectories"]:
        raise ComparisonError("trained-token budget is too small")
    return budget


def _validate_non_inferiority(value: Any) -> dict[str, Any]:
    gate = _object(
        value,
        {
            "method",
            "primary_metric",
            "pass_at_1_absolute_margin",
            "result_absolute_margin",
        },
        "non_inferiority",
    )
    if gate["method"] != "paired_all_seeds_and_mean_improvement":
        raise ComparisonError("non-inferiority method differs")
    if gate["primary_metric"] != "heldout_pass_at_1_improvement":
        raise ComparisonError("non-inferiority primary metric differs")
    _metric(
        gate["pass_at_1_absolute_margin"],
        "non_inferiority.pass_at_1_absolute_margin",
    )
    _metric(
        gate["result_absolute_margin"],
        "non_inferiority.result_absolute_margin",
    )
    return gate


def study_contract_sha256(manifest: dict[str, Any]) -> str:
    """Hash every field that must be declared before run records are mixed."""

    payload = {
        key: manifest[key]
        for key in (
            "schema",
            "study_id",
            "seeds",
            "arms",
            "inputs",
            "budget",
            "non_inferiority",
        )
    }
    return hashlib.sha256(
        b"yeto-qwen35-m1-online-study-v1\0" + _canonical(payload)
    ).hexdigest()


def _validate_artifacts(root: Path, value: Any, arm: str, seed: int) -> None:
    keys = {
        "run_manifest",
        "trajectory_accounting",
        "optimizer_receipts",
        "publication_receipts",
        "v0_heldout",
        "final_heldout",
        "final_policy_manifest",
    }
    artifacts = _object(value, keys, f"run[{arm},{seed}].artifacts")
    for name in sorted(keys):
        _resolve_binding(root, artifacts[name], f"run[{arm},{seed}].artifacts.{name}")


def _validate_learner_rounds(
    value: Any,
    *,
    arm_name: str,
    seed: int,
    round_id: int,
    arm: dict[str, Any],
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != arm["learners"]:
        raise ComparisonError(
            f"run[{arm_name},{seed}].rounds[{round_id}] learner roster differs"
        )
    learners = []
    for index, item in enumerate(value):
        learner = _object(
            item,
            {
                "learner_id",
                "accepted_groups",
                "accepted_trajectories",
                "trained_tokens",
                "optimizer_steps",
                "mean_kl",
                "local_update_l2_norm",
            },
            f"run[{arm_name},{seed}].rounds[{round_id}].learners[{index}]",
        )
        if learner["learner_id"] != index:
            raise ComparisonError("round learner IDs are not the fixed roster")
        for key in (
            "accepted_groups",
            "accepted_trajectories",
            "trained_tokens",
            "optimizer_steps",
        ):
            _integer(
                learner[key],
                f"run[{arm_name},{seed}].rounds[{round_id}].learners[{index}].{key}",
                minimum=1,
            )
        if (
            learner["accepted_groups"] * sampling["samples_per_group"]
            != learner["accepted_trajectories"]
        ):
            raise ComparisonError("learner trajectory count differs from group size")
        if learner["trained_tokens"] < learner["accepted_trajectories"]:
            raise ComparisonError("learner trained-token accounting is too small")
        _signed_finite(learner["mean_kl"], "learner mean_kl")
        _finite(learner["local_update_l2_norm"], "learner local_update_l2_norm")
        learners.append(learner)
    return learners


def _validate_rounds(
    value: Any,
    *,
    arm_name: str,
    seed: int,
    arm: dict[str, Any],
    budget: dict[str, Any],
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != budget["policy_rounds"]:
        raise ComparisonError(f"run[{arm_name},{seed}] round count differs")
    rounds = []
    for index, item in enumerate(value):
        row = _object(
            item,
            {
                "base_policy_version",
                "target_policy_version",
                "prompt_group_manifest_sha256",
                "generation_seed_manifest_sha256",
                "accepted_groups",
                "accepted_trajectories",
                "trained_tokens",
                "optimizer_steps",
                "mean_kl",
                "global_update_l2_norm",
                "learners",
            },
            f"run[{arm_name},{seed}].rounds[{index}]",
        )
        if (
            row["base_policy_version"] != index
            or row["target_policy_version"] != index + 1
        ):
            raise ComparisonError("policy rounds are not contiguous from version zero")
        _hash(row["prompt_group_manifest_sha256"], "round prompt manifest")
        _hash(row["generation_seed_manifest_sha256"], "round generation seeds")
        for key in (
            "accepted_groups",
            "accepted_trajectories",
            "trained_tokens",
            "optimizer_steps",
        ):
            _integer(row[key], f"round.{key}", minimum=1)
        if (
            row["accepted_groups"] * sampling["samples_per_group"]
            != row["accepted_trajectories"]
        ):
            raise ComparisonError("round trajectory count differs from group size")
        learners = _validate_learner_rounds(
            row["learners"],
            arm_name=arm_name,
            seed=seed,
            round_id=index,
            arm=arm,
            sampling=sampling,
        )
        for key in (
            "accepted_groups",
            "accepted_trajectories",
            "trained_tokens",
            "optimizer_steps",
        ):
            if sum(learner[key] for learner in learners) != row[key]:
                raise ComparisonError(f"round {key} does not reconcile across learners")
        expected_steps = arm["learners"] * arm["local_optimizer_steps_per_policy_round"]
        if row["optimizer_steps"] != expected_steps:
            raise ComparisonError("round optimizer-step budget differs")
        weighted_kl = (
            sum(learner["mean_kl"] * learner["trained_tokens"] for learner in learners)
            / row["trained_tokens"]
        )
        mean_kl = _signed_finite(row["mean_kl"], "round mean_kl")
        if not _same_float(mean_kl, weighted_kl):
            raise ComparisonError("round KL does not match trained-token weighting")
        _finite(row["global_update_l2_norm"], "round global_update_l2_norm")
        rounds.append(row)
    expected_totals = {
        "accepted_groups": budget["accepted_groups"],
        "accepted_trajectories": budget["accepted_trajectories"],
        "trained_tokens": budget["trained_tokens"],
        "optimizer_steps": budget[
            "centralized_optimizer_steps"
            if arm_name == "centralized"
            else "dense_local_optimizer_steps"
        ],
    }
    for key, expected in expected_totals.items():
        if sum(row[key] for row in rounds) != expected:
            raise ComparisonError(f"run[{arm_name},{seed}] {key} budget differs")
    if not any(row["global_update_l2_norm"] > 0 for row in rounds):
        raise ComparisonError(f"run[{arm_name},{seed}] produced no global update")
    for learner_id in range(arm["learners"]):
        if not any(
            row["learners"][learner_id]["local_update_l2_norm"] > 0 for row in rounds
        ):
            raise ComparisonError(
                f"run[{arm_name},{seed}] learner {learner_id} produced no update"
            )
    return rounds


def _validate_publications(
    value: Any,
    *,
    arm_name: str,
    seed: int,
    arm: dict[str, Any],
    budget: dict[str, Any],
    initial_policy_hash: str,
    final_policy_hash: str,
) -> list[dict[str, Any]]:
    count = budget["policy_rounds"] + 1
    if not isinstance(value, list) or len(value) != count:
        raise ComparisonError(f"run[{arm_name},{seed}] publication count differs")
    expected_workers = list(range(arm["inference_workers"]))
    publications = []
    for version, item in enumerate(value):
        row = _object(
            item,
            {
                "policy_version",
                "policy_hash",
                "inference_worker_ids",
                "receipt_manifest_sha256",
            },
            f"run[{arm_name},{seed}].publications[{version}]",
        )
        if row["policy_version"] != version:
            raise ComparisonError("published policy versions are not contiguous")
        _hash(row["policy_hash"], "publication policy_hash")
        _hash(row["receipt_manifest_sha256"], "publication receipt manifest")
        if row["inference_worker_ids"] != expected_workers:
            raise ComparisonError("publication receipt worker roster differs")
        publications.append(row)
    if publications[0]["policy_hash"] != initial_policy_hash:
        raise ComparisonError("version-zero publication differs from the initial model")
    if publications[-1]["policy_hash"] != final_policy_hash:
        raise ComparisonError("final publication differs from the final policy")
    return publications


def _validate_evaluation(
    value: Any,
    *,
    arm_name: str,
    seed: int,
    phase: str,
    version: int,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    row = _object(
        value,
        {
            "policy_version",
            "dataset_sha256",
            "evaluation_contract_sha256",
            "generation_seed_manifest_sha256",
            "sample_manifest_sha256",
            "sample_count",
            "result",
            "pass_at_1",
            "evidence_sha256",
        },
        f"run[{arm_name},{seed}].evaluations.{phase}",
    )
    if row["policy_version"] != version:
        raise ComparisonError(f"{phase} evaluation used the wrong policy version")
    if row["dataset_sha256"] != inputs["data"]["heldout_sha256"]:
        raise ComparisonError(f"{phase} evaluation used the wrong heldout split")
    if row["evaluation_contract_sha256"] != inputs["evaluation"]["contract_sha256"]:
        raise ComparisonError(f"{phase} evaluation contract differs")
    for key in (
        "generation_seed_manifest_sha256",
        "sample_manifest_sha256",
        "evidence_sha256",
    ):
        _hash(row[key], f"evaluation.{key}")
    expected_samples = (
        inputs["evaluation"]["prompt_count"]
        * inputs["evaluation"]["samples_per_prompt"]
    )
    if row["sample_count"] != expected_samples:
        raise ComparisonError(f"{phase} evaluation sample count differs")
    _metric(row["result"], f"evaluation.{phase}.result")
    _metric(row["pass_at_1"], f"evaluation.{phase}.pass_at_1")
    return row


def _validate_run(
    root: Path,
    value: dict[str, Any],
    *,
    expected_arm: str,
    expected_seed: int,
    contract_hash: str,
    arms: dict[str, dict[str, Any]],
    inputs: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    run = _object(
        value,
        {
            "schema",
            "study_contract_sha256",
            "arm",
            "seed",
            "artifacts",
            "initial_policy_hash",
            "final_policy_hash",
            "accounting",
            "rounds",
            "publications",
            "evaluations",
        },
        f"run[{expected_arm},{expected_seed}]",
    )
    if (
        run["schema"] != RUN_SCHEMA
        or run["study_contract_sha256"] != contract_hash
        or run["arm"] != expected_arm
        or run["seed"] != expected_seed
    ):
        raise ComparisonError(f"run[{expected_arm},{expected_seed}] identity differs")
    _validate_artifacts(root, run["artifacts"], expected_arm, expected_seed)
    initial_hash = _hash(run["initial_policy_hash"], "run.initial_policy_hash")
    final_hash = _hash(run["final_policy_hash"], "run.final_policy_hash")
    if initial_hash != inputs["model"]["initial_policy_hash"]:
        raise ComparisonError("run initial policy differs from the study contract")
    if final_hash == initial_hash:
        raise ComparisonError("run final policy is unchanged")

    accounting = _object(
        run["accounting"],
        {
            "initial_policy_version",
            "final_policy_version",
            "policy_rounds",
            "accepted_groups",
            "accepted_trajectories",
            "trained_tokens",
            "optimizer_steps",
        },
        f"run[{expected_arm},{expected_seed}].accounting",
    )
    expected_accounting = {
        "initial_policy_version": 0,
        "final_policy_version": budget["policy_rounds"],
        "policy_rounds": budget["policy_rounds"],
        "accepted_groups": budget["accepted_groups"],
        "accepted_trajectories": budget["accepted_trajectories"],
        "trained_tokens": budget["trained_tokens"],
        "optimizer_steps": budget[
            "centralized_optimizer_steps"
            if expected_arm == "centralized"
            else "dense_local_optimizer_steps"
        ],
    }
    if accounting != expected_accounting:
        raise ComparisonError(f"run[{expected_arm},{expected_seed}] accounting differs")

    rounds = _validate_rounds(
        run["rounds"],
        arm_name=expected_arm,
        seed=expected_seed,
        arm=arms[expected_arm],
        budget=budget,
        sampling=inputs["sampling"],
    )
    publications = _validate_publications(
        run["publications"],
        arm_name=expected_arm,
        seed=expected_seed,
        arm=arms[expected_arm],
        budget=budget,
        initial_policy_hash=initial_hash,
        final_policy_hash=final_hash,
    )
    evaluations = _object(
        run["evaluations"],
        {"v0", "final"},
        f"run[{expected_arm},{expected_seed}].evaluations",
    )
    v0 = _validate_evaluation(
        evaluations["v0"],
        arm_name=expected_arm,
        seed=expected_seed,
        phase="v0",
        version=0,
        inputs=inputs,
    )
    final = _validate_evaluation(
        evaluations["final"],
        arm_name=expected_arm,
        seed=expected_seed,
        phase="final",
        version=budget["policy_rounds"],
        inputs=inputs,
    )
    if v0["evidence_sha256"] != run["artifacts"]["v0_heldout"]["sha256"]:
        raise ComparisonError("version-zero evaluation evidence binding differs")
    if final["evidence_sha256"] != run["artifacts"]["final_heldout"]["sha256"]:
        raise ComparisonError("final evaluation evidence binding differs")
    return {
        **run,
        "rounds": rounds,
        "publications": publications,
        "evaluations": {"v0": v0, "final": final},
    }


def _paired_metric(
    records: dict[tuple[str, int], dict[str, Any]],
    *,
    field: str,
    margin: float,
) -> dict[str, Any]:
    rows = []
    differences = []
    for seed in SEEDS:
        central = records[("centralized", seed)]["evaluations"]
        dense = records[("dense_two_island", seed)]["evaluations"]
        central_improvement = central["final"][field] - central["v0"][field]
        dense_improvement = dense["final"][field] - dense["v0"][field]
        difference = dense_improvement - central_improvement
        differences.append(difference)
        rows.append(
            {
                "seed": seed,
                "centralized_v0": central["v0"][field],
                "centralized_final": central["final"][field],
                "centralized_improvement": central_improvement,
                "dense_v0": dense["v0"][field],
                "dense_final": dense["final"][field],
                "dense_improvement": dense_improvement,
                "paired_improvement_difference": difference,
                "within_margin": difference + 1e-12 >= -margin,
            }
        )
    mean = statistics.fmean(differences)
    passed = all(row["within_margin"] for row in rows) and mean + 1e-12 >= -margin
    return {
        "absolute_margin": margin,
        "per_seed": rows,
        "mean_paired_improvement_difference": mean,
        "minimum_paired_improvement_difference": min(differences),
        "passed": passed,
    }


def verify_comparison(path: str | Path) -> dict[str, Any]:
    """Validate one closed study and return its deterministic gate result."""

    comparison_path = Path(path).expanduser().resolve()
    manifest = _load_canonical(comparison_path, "comparison manifest")
    _object(
        manifest,
        {
            "schema",
            "study_id",
            "seeds",
            "arms",
            "inputs",
            "budget",
            "non_inferiority",
            "runs",
        },
        "comparison manifest",
    )
    if manifest["schema"] != COMPARISON_SCHEMA:
        raise ComparisonError("comparison schema differs")
    if (
        not isinstance(manifest["study_id"], str)
        or _STUDY_ID.fullmatch(manifest["study_id"]) is None
    ):
        raise ComparisonError("study_id is malformed")
    if manifest["seeds"] != list(SEEDS):
        raise ComparisonError("comparison requires exact seeds 17,29,43")
    arms_value = _object(manifest["arms"], set(ARMS), "arms")
    arms = {name: _validate_arm(name, arms_value[name]) for name in ARMS}
    central, dense = arms["centralized"], arms["dense_two_island"]
    for key in ("trainer_gpus", "inference_gpus", "total_gpus", "inference_workers"):
        if central[key] != dense[key]:
            raise ComparisonError(f"same-compute arm {key} differs")
    inputs = _validate_inputs(manifest["inputs"])
    budget = _validate_budget(manifest["budget"], inputs)
    for key in ("trainer_gpus", "inference_gpus", "total_gpus"):
        if budget[key] != central[key]:
            raise ComparisonError(f"budget.{key} differs from arm topology")
    expected_central_steps = (
        budget["policy_rounds"]
        * central["learners"]
        * central["local_optimizer_steps_per_policy_round"]
    )
    expected_dense_steps = (
        budget["policy_rounds"]
        * dense["learners"]
        * dense["local_optimizer_steps_per_policy_round"]
    )
    if (
        budget["centralized_optimizer_steps"] != expected_central_steps
        or budget["dense_local_optimizer_steps"] != expected_dense_steps
    ):
        raise ComparisonError("declared per-arm optimizer-step budgets differ")
    gate = _validate_non_inferiority(manifest["non_inferiority"])
    contract_hash = study_contract_sha256(manifest)

    run_bindings = manifest["runs"]
    if not isinstance(run_bindings, list) or len(run_bindings) != len(SEEDS) * len(
        ARMS
    ):
        raise ComparisonError("comparison requires exactly six run bindings")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    root = comparison_path.parent.resolve()
    for index, binding in enumerate(run_bindings):
        row = _object(binding, {"arm", "seed", "path", "sha256"}, f"runs[{index}]")
        key = (row["arm"], row["seed"])
        if key not in {(arm, seed) for seed in SEEDS for arm in ARMS} or key in records:
            raise ComparisonError(f"run binding identity differs: {key!r}")
        run_path = _resolve_binding(
            root, {"path": row["path"], "sha256": row["sha256"]}, f"runs[{index}]"
        )
        run = _load_canonical(run_path, f"run record {key!r}")
        records[key] = _validate_run(
            root,
            run,
            expected_arm=key[0],
            expected_seed=key[1],
            contract_hash=contract_hash,
            arms=arms,
            inputs=inputs,
            budget=budget,
        )
    expected_keys = {(arm, seed) for seed in SEEDS for arm in ARMS}
    if set(records) != expected_keys:
        raise ComparisonError("comparison run matrix is incomplete")

    for seed in SEEDS:
        central_run = records[("centralized", seed)]
        dense_run = records[("dense_two_island", seed)]
        for round_id in range(budget["policy_rounds"]):
            for field, description in (
                ("prompt_group_manifest_sha256", "prompt-group schedules"),
                ("generation_seed_manifest_sha256", "rollout generation seeds"),
            ):
                if (
                    central_run["rounds"][round_id][field]
                    != dense_run["rounds"][round_id][field]
                ):
                    raise ComparisonError(f"paired arms used different {description}")
        central_v0 = central_run["evaluations"]["v0"]
        dense_v0 = dense_run["evaluations"]["v0"]
        for key in ("generation_seed_manifest_sha256", "sample_manifest_sha256"):
            if central_v0[key] != dense_v0[key]:
                raise ComparisonError("version-zero heldout evaluation is not paired")
        for key in ("result", "pass_at_1"):
            if not _same_float(central_v0[key], dense_v0[key]):
                raise ComparisonError("version-zero heldout metrics differ")
        if (
            central_run["evaluations"]["final"]["generation_seed_manifest_sha256"]
            != dense_run["evaluations"]["final"]["generation_seed_manifest_sha256"]
        ):
            raise ComparisonError("final heldout generation seeds are not paired")

    pass_at_1 = _paired_metric(
        records,
        field="pass_at_1",
        margin=float(gate["pass_at_1_absolute_margin"]),
    )
    result = _paired_metric(
        records,
        field="result",
        margin=float(gate["result_absolute_margin"]),
    )
    passed = pass_at_1["passed"] and result["passed"]
    return {
        "schema": VERIFIED_SCHEMA,
        "study_id": manifest["study_id"],
        "comparison_manifest_sha256": _sha256(comparison_path),
        "study_contract_sha256": contract_hash,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "accounting_verified": True,
        "paired_prompt_and_evaluation_seeds_verified": True,
        "pass_at_1": pass_at_1,
        "result": result,
        "non_inferiority": {
            "method": gate["method"],
            "primary_metric": gate["primary_metric"],
            "passed": passed,
            "conclusion": "non_inferior" if passed else "not_demonstrated",
        },
    }


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ComparisonError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(_canonical(value))
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_comparison(args.comparison)
        if args.output is not None:
            _write_private(args.output, result)
    except ComparisonError as exc:
        raise SystemExit(f"M1_COMPARISON_INVALID: {exc}") from exc
    print(_canonical(result).decode(), end="")
    return 0 if result["non_inferiority"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
