#!/usr/bin/env python3
"""Build the immutable Terminal-Bench 2.1 SAO/DiLoCo run inputs.

This tool only prepares data and a launch contract.  It never starts Docker,
Ray, a model server, or a training process.  The resulting plan intentionally
records that value pretraining consumes all four baseline trajectories from
all tasks, including the actor-evaluation half, while actor RL sees only the
44-task training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tomllib

from yeto.rl import CODEX_OPENENV_AGENT, CODEX_OPENENV_IDENTITY_ENV
from yeto.rl.codex_backend import QWEN35_08B_MODEL, QWEN35_08B_REVISION

SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"
SPLIT_DOMAIN = b"yeto-tbench21-sao-split-v1\0"
ROLLOUTS_PER_TASK = 4
EPISODE_TIMEOUT_SECONDS = 1800
ISLANDS = 8
TARGET_CONCURRENCY = 304
PER_ISLAND_CONCURRENCY = 38
SGLANG_MEM_FRACTION_STATIC = 0.15
SGLANG_MAX_TOTAL_TOKENS = 393216
SGLANG_MAX_MAMBA_CACHE_SIZE = 256
MAX_SEQ_LEN = 8192
COMPACTION_TRIGGER_TOKENS = 6144
COMPACTION_SUMMARY_MAX_TOKENS = 1024
MAX_COMPACTIONS = 3
ROLLOUT_SEED_BASE = 82621
MIN_H200_MEMORY_BYTES = 120 * 1024**3
SYSTEM_MESSAGE = (
    "The authenticated Terminal-Bench daemon supplies the immutable task "
    "instruction. Use the Codex terminal tools to solve it and submit exactly once."
)


class PlanError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")


def _git_revision(root: Path) -> str | None:
    try:
        value = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _load_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    if not tasks_dir.is_dir() or tasks_dir.is_symlink():
        raise PlanError("tasks directory must be a real directory")
    tasks: list[dict[str, Any]] = []
    for task_dir in sorted(tasks_dir.iterdir(), key=lambda path: path.name):
        if not task_dir.is_dir() or task_dir.is_symlink():
            continue
        task_toml = task_dir / "task.toml"
        instruction = task_dir / "instruction.md"
        dockerfile = task_dir / "environment" / "Dockerfile"
        if (
            not task_toml.is_file()
            or not instruction.is_file()
            or not dockerfile.is_file()
        ):
            raise PlanError(f"task {task_dir.name!r} is incomplete")
        try:
            raw = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise PlanError(f"task {task_dir.name!r} has invalid task.toml") from error
        metadata = raw.get("metadata", {})
        environment = raw.get("environment", {})
        task = raw.get("task", {})
        if not all(isinstance(value, dict) for value in (metadata, environment, task)):
            raise PlanError(f"task {task_dir.name!r} metadata is malformed")
        docker_image = environment.get("docker_image")
        if not isinstance(docker_image, str) or not docker_image:
            raise PlanError(f"task {task_dir.name!r} has no Docker image identity")
        category = metadata.get("category", "unknown")
        if not isinstance(category, str) or not category:
            raise PlanError(f"task {task_dir.name!r} has no category")
        tasks.append(
            {
                "task_id": task_dir.name,
                "task_name": task.get("name"),
                "category": category,
                "docker_image": docker_image,
                "cpus": environment.get("cpus"),
                "memory_mb": environment.get("memory_mb"),
                "storage_mb": environment.get("storage_mb"),
                "task_toml_sha256": _sha256_file(task_toml),
                "instruction_sha256": _sha256_file(instruction),
                "dockerfile_sha256": _sha256_file(dockerfile),
            }
        )
    if len(tasks) != 89:
        raise PlanError(
            f"Terminal-Bench 2.1 must contain exactly 89 tasks, got {len(tasks)}"
        )
    ids = [task["task_id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise PlanError("Terminal-Bench task IDs are not unique")
    return tasks


def _split_tasks(
    tasks: list[dict[str, Any]], seed: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return an exact deterministic 44-train/45-eval partition."""

    if not seed:
        raise PlanError("split seed must not be empty")

    def key(task: dict[str, Any]) -> tuple[bytes, str]:
        payload = (
            SPLIT_DOMAIN
            + seed.encode("utf-8")
            + b"\0"
            + task["task_id"].encode("utf-8")
        )
        return hashlib.sha256(payload).digest(), task["task_id"]

    ranked = sorted(tasks, key=key)
    return ranked[:44], ranked[44:]


def _rollout_rows(tasks: list[dict[str, Any]], *, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        for replica in range(ROLLOUTS_PER_TASK):
            island_id = (task_index * ROLLOUTS_PER_TASK + replica) % ISLANDS
            rollout_seed = ROLLOUT_SEED_BASE + island_id
            sample_id = f"{phase}:{task['task_id']}:r{replica}"
            rows.append(
                {
                    "messages": [{"role": "system", "content": SYSTEM_MESSAGE}],
                    "metadata": {
                        "task_id": task["task_id"],
                        "prompt_tier": "l2",
                        "max_seq_len": MAX_SEQ_LEN,
                        "split": phase,
                        "rollout_replica": replica,
                        "sample_id": sample_id,
                        "island_id": island_id,
                        "rollout_seed": rollout_seed,
                        "episode_timeout_seconds": EPISODE_TIMEOUT_SECONDS,
                    },
                }
            )
    return rows


def _island_contract(island_id: int, run_root: str) -> dict[str, Any]:
    base = 32000 + island_id * 100
    return {
        "island_id": island_id,
        "physical_gpu_index": island_id,
        "cuda_visible_devices": str(island_id),
        "rollout_seed": ROLLOUT_SEED_BASE + island_id,
        "run_dir": f"{run_root}/island-{island_id}",
        "entrypoint": "/root/miles/tools/probes/train_sao_streaming_secrlenv.py",
        "runtime_contracts": {
            "sao_context": f"{run_root}/island-{island_id}/sao-context.json",
            "streaming_context": f"{run_root}/island-{island_id}/sao-streaming-context.json",
            "actor_syncer_port": 29400,
            "critic_syncer_port": 29401,
        },
        "openenv_agent_contract": {
            "custom_agent_function_path": CODEX_OPENENV_AGENT,
            "custom_rm_path": "openenv_generate.reward_func",
            "dynamic_sampling_filter_path": (
                "openenv_generate.check_terminal_bench_episode"
            ),
            "input_key": "messages",
            "tito_model": "qwen35",
            "codex_backend_profile": "qwen35_08b",
            "model": QWEN35_08B_MODEL,
            "model_revision": QWEN35_08B_REVISION,
            "max_seq_len": MAX_SEQ_LEN,
            "must_not_enter_secrlenv_retry_path": True,
        },
        "ray_gcs_port": 26000 + island_id,
        "ports": {
            "session_server": 31000 + island_id,
            "rollout_engine_base": base,
            "sglang_router": base + 20,
            "sglang_router_prometheus": base + 21,
            "train_master_base": base + 40,
        },
        "environment": {
            "SECRLENV_MAX_ROLLOUT_TIME_SECONDS": str(EPISODE_TIMEOUT_SECONDS),
            "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(EPISODE_TIMEOUT_SECONDS),
            "YETO_CODEX_COMPACTION_ENABLED": "1",
            "YETO_CODEX_COMPACTION_TRIGGER_TOKENS": str(COMPACTION_TRIGGER_TOKENS),
            "YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS": str(
                COMPACTION_SUMMARY_MAX_TOKENS
            ),
            "YETO_CODEX_MAX_COMPACTIONS": str(MAX_COMPACTIONS),
            **CODEX_OPENENV_IDENTITY_ENV,
            "NCCL_NVLS_ENABLE": "0",
            "RAY_DEDUP_LOGS": "0",
        },
        "miles_topology_args": [
            "--sao-one-gpu-island",
            "--sao-compaction",
            "--input-key",
            "messages",
            "--colocate",
            "--num-gpus-per-node",
            "1",
            "--actor-num-nodes",
            "1",
            "--actor-num-gpus-per-node",
            "1",
            "--critic-num-nodes",
            "1",
            "--critic-num-gpus-per-node",
            "1",
            "--rollout-num-gpus",
            "1",
            "--rollout-num-gpus-per-engine",
            "1",
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
            "--context-parallel-size",
            "1",
            "--expert-model-parallel-size",
            "1",
            "--sglang-server-concurrency",
            str(PER_ISLAND_CONCURRENCY),
            "--sglang-max-running-requests",
            str(PER_ISLAND_CONCURRENCY),
            "--async-max-concurrent-samples",
            str(PER_ISLAND_CONCURRENCY),
            "--sglang-mem-fraction-static",
            str(SGLANG_MEM_FRACTION_STATIC),
            "--sglang-max-total-tokens",
            str(SGLANG_MAX_TOTAL_TOKENS),
            "--sglang-max-mamba-cache-size",
            str(SGLANG_MAX_MAMBA_CACHE_SIZE),
        ],
        "ray_fractional_gpu": {"actor": 0.4, "critic": 0.4, "sglang": 0.2},
    }


def _category_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(task["category"] for task in tasks).items()))


def build_plan(tasks_dir: Path, output_dir: Path, *, seed: str, run_root: str) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise PlanError("output directory must be fresh")
    if not os.path.isabs(run_root) or ".." in Path(run_root).parts:
        raise PlanError("run root must be an absolute safe path")

    tasks = _load_tasks(tasks_dir)
    train_tasks, eval_tasks = _split_tasks(tasks, seed)
    baseline_tasks = sorted(tasks, key=lambda task: task["task_id"])
    rows_by_phase = {
        "baseline": _rollout_rows(baseline_tasks, phase="baseline"),
        "train": _rollout_rows(train_tasks, phase="train"),
        "eval": _rollout_rows(eval_tasks, phase="eval"),
    }
    if [len(rows_by_phase[name]) for name in ("baseline", "train", "eval")] != [
        356,
        176,
        180,
    ]:
        raise AssertionError("Terminal-Bench rollout cardinality changed")

    output_dir.mkdir(mode=0o755, parents=True)
    files: dict[str, str] = {}
    for phase, rows in rows_by_phase.items():
        phase_dir = output_dir / phase
        phase_dir.mkdir()
        all_path = phase_dir / "all.jsonl"
        _write_jsonl(all_path, rows)
        files[str(all_path.relative_to(output_dir))] = _sha256_file(all_path)
        for island_id in range(ISLANDS):
            island_path = phase_dir / f"island-{island_id}.jsonl"
            _write_jsonl(
                island_path,
                (row for row in rows if row["metadata"]["island_id"] == island_id),
            )
            files[str(island_path.relative_to(output_dir))] = _sha256_file(island_path)

    baseline_ids = [row["metadata"]["sample_id"] for row in rows_by_phase["baseline"]]
    value_selection = {
        "schema": "yeto.sao-value-pretraining-selection.v1",
        "source_phase": "baseline",
        "selection": "all",
        "sample_count": len(baseline_ids),
        "task_count": len(tasks),
        "rollouts_per_task": ROLLOUTS_PER_TASK,
        "baseline_sample_ids": baseline_ids,
        "critic_pretraining_exposes_actor_eval_tasks": True,
        "actor_eval_task_count_exposed_to_critic": len(eval_tasks),
        "actor_eval_trajectory_count_exposed_to_critic": len(eval_tasks)
        * ROLLOUTS_PER_TASK,
    }
    value_path = output_dir / "value-pretraining-selection.json"
    _write_json(value_path, value_selection)
    files[value_path.name] = _sha256_file(value_path)

    task_contracts = {
        task["task_id"]: {key: value for key, value in task.items() if key != "task_id"}
        for task in sorted(tasks, key=lambda item: item["task_id"])
    }
    inventory_sha256 = _sha256_bytes(_canonical_bytes(task_contracts))
    split_payload = {
        "seed": seed,
        "algorithm": "sha256-domain-ranked-first-44",
        "train_task_ids": [task["task_id"] for task in train_tasks],
        "eval_task_ids": [task["task_id"] for task in eval_tasks],
    }
    split_sha256 = _sha256_bytes(_canonical_bytes(split_payload))

    manifest = {
        "schema": SCHEMA,
        "terminal_bench": {
            "version": "2.1",
            "revision": _git_revision(tasks_dir.parent),
            "task_count": len(tasks),
            "task_inventory_sha256": inventory_sha256,
            "task_contracts": task_contracts,
        },
        "split": {
            **split_payload,
            "sha256": split_sha256,
            "train_task_count": len(train_tasks),
            "eval_task_count": len(eval_tasks),
            "train_category_counts": _category_counts(train_tasks),
            "eval_category_counts": _category_counts(eval_tasks),
        },
        "rollouts": {
            "per_task": ROLLOUTS_PER_TASK,
            "baseline": 356,
            "train": 176,
            "eval": 180,
            "episode_timeout_seconds": EPISODE_TIMEOUT_SECONDS,
            "target_concurrency": TARGET_CONCURRENCY,
            "per_island_concurrency": PER_ISLAND_CONCURRENCY,
            "seed_base": ROLLOUT_SEED_BASE,
            "per_island_seeds": [
                ROLLOUT_SEED_BASE + island_id for island_id in range(ISLANDS)
            ],
            "baseline_achievable_concurrency": TARGET_CONCURRENCY,
            "train_cardinality_limited_concurrency": 176,
            "eval_cardinality_limited_concurrency": 180,
        },
        "value_pretraining": {
            "selection_path": value_path.name,
            "selection_sha256": files[value_path.name],
            "uses_all_baseline_trajectories": True,
            "sample_count": 356,
            "critic_pretraining_exposes_actor_eval_tasks": True,
        },
        "compaction": {
            "enabled": True,
            "trainer_objective": "sao",
            "max_seq_len": MAX_SEQ_LEN,
            "trigger_semantics": "consumed-context-tokens-at-least",
            "trigger_tokens": COMPACTION_TRIGGER_TOKENS,
            "summary_max_tokens": COMPACTION_SUMMARY_MAX_TOKENS,
            "max_compactions_per_episode": MAX_COMPACTIONS,
        },
        "topology": {
            "islands": ISLANDS,
            "physical_gpus": ISLANDS,
            "one_physical_gpu_per_island": True,
            "model": QWEN35_08B_MODEL,
            "model_revision": QWEN35_08B_REVISION,
            "minimum_gpu_memory_bytes": MIN_H200_MEMORY_BYTES,
            "cross_island_collectives": False,
            "synchronization": "dual-role streaming DiLoCo",
            "syncer_sessions": {"actor_port": 29400, "critic_port": 29401},
            "island_contracts": [
                _island_contract(island_id, run_root) for island_id in range(ISLANDS)
            ],
        },
        "required_gates": {
            "codex_harness": True,
            "compaction": True,
            "all_task_images_available": True,
            "openenv_tbench_max_active_episodes_at_least": TARGET_CONCURRENCY,
            "openenv_compaction_trajectory_evidence_v2": True,
            "terminal_bench_signed_outcome_hmac_key_file": True,
            "direct_codex_qwen35_08b_preflight": True,
            "host_open_file_limit_at_least": 65536,
            "fresh_per_island_ray_clusters": True,
            "actor_and_critic_streaming_syncers_healthy": True,
            "layout_attestation_per_role": True,
        },
        "launch_readiness": {
            "ready": False,
            "blocking_issues": [
                {
                    "code": "task-images-not-attested",
                    "detail": (
                        "All 89 immutable Terminal-Bench task images must be present "
                        "and capacity-preflighted before the baseline or online phase."
                    ),
                },
                {
                    "code": "openenv-capacity-not-attested",
                    "detail": (
                        "The trusted OpenEnv service has not yet attested at least 304 "
                        "simultaneous Terminal-Bench episodes on this host."
                    ),
                },
                {
                    "code": "one-gpu-island-runtime-not-smoked",
                    "detail": (
                        "Actor, critic, and SGLang fractional placement is validated "
                        "fail-closed but still needs a physical one-GPU memory/runtime "
                        "smoke before the eight-island launch."
                    ),
                },
                {
                    "code": "final-layout-attestations-not-generated",
                    "detail": (
                        "Each island still needs its final actor/critic layout probe "
                        "and hash-bound streaming runtime after fragment count P is "
                        "discovered."
                    ),
                },
            ],
        },
        "integration_notes": [
            {
                "code": "streaming-entrypoint-retains-legacy-cli-name",
                "blocking": False,
                "detail": (
                    "The retained entrypoint and flag names mention SecRLEnv for "
                    "compatibility, but the v2 base context binds the benchmark as "
                    "terminal-bench-2.1 without synthetic SecRLEnv fields. Direct "
                    "preflight attests the exact Codex/OpenEnv Qwen3.5-0.8B adapter "
                    "before Miles parses arguments or allocates a GPU."
                ),
            }
        ],
        "files": dict(sorted(files.items())),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", default="tbench21-sao-20260826")
    parser.add_argument("--run-root", default="/data/sft/tbench21-sao-diloco-full")
    args = parser.parse_args(argv)
    manifest = build_plan(
        args.tasks_dir.resolve(),
        args.output_dir.resolve(),
        seed=args.split_seed,
        run_root=args.run_root,
    )
    print(manifest)


if __name__ == "__main__":
    main()
