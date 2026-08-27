from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from yeto.rl import CODEX_OPENENV_IDENTITY_ENV, SECRLENV_AGENTS

TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "probes"
    / "build_tbench21_sao_diloco_plan.py"
)
SPEC = importlib.util.spec_from_file_location("build_tbench21_sao_diloco_plan", TOOL)
assert SPEC is not None and SPEC.loader is not None
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


def _task_tree(root: Path, count: int = 89) -> Path:
    tasks = root / "tasks"
    for index in range(count):
        task = tasks / f"task-{index:03d}"
        (task / "environment").mkdir(parents=True)
        (task / "instruction.md").write_text("untrusted task text\n")
        (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
        (task / "task.toml").write_text(
            f"""schema_version = "1.1"
[task]
name = "terminal-bench/task-{index:03d}"
[metadata]
category = "category-{index % 7}"
[environment]
docker_image = "example/task-{index:03d}:fixed"
cpus = 1
memory_mb = 2048
storage_mb = 10240
"""
        )
    return tasks


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_builds_exact_full_suite_plan(tmp_path):
    tasks = _task_tree(tmp_path / "source")
    output = tmp_path / "plan"

    manifest_path = plan.build_plan(
        tasks,
        output,
        seed="fixed-seed",
        run_root="/run/tbench21",
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema"] == plan.SCHEMA
    assert manifest["split"]["train_task_count"] == 44
    assert manifest["split"]["eval_task_count"] == 45
    train_ids = set(manifest["split"]["train_task_ids"])
    eval_ids = set(manifest["split"]["eval_task_ids"])
    assert not train_ids & eval_ids
    assert len(train_ids | eval_ids) == 89
    assert manifest["rollouts"] == {
        "per_task": 4,
        "baseline": 356,
        "train": 176,
        "eval": 180,
        "episode_timeout_seconds": 1800,
        "target_concurrency": 304,
        "per_island_concurrency": 38,
        "seed_base": 82621,
        "per_island_seeds": list(range(82621, 82629)),
        "baseline_achievable_concurrency": 304,
        "train_cardinality_limited_concurrency": 176,
        "eval_cardinality_limited_concurrency": 180,
    }

    baseline = _jsonl(output / "baseline" / "all.jsonl")
    train = _jsonl(output / "train" / "all.jsonl")
    evaluation = _jsonl(output / "eval" / "all.jsonl")
    assert (len(baseline), len(train), len(evaluation)) == (356, 176, 180)
    assert {row["metadata"]["task_id"] for row in train} == train_ids
    assert {row["metadata"]["task_id"] for row in evaluation} == eval_ids
    assert all(row["metadata"]["episode_timeout_seconds"] == 1800 for row in baseline)
    assert all(
        row["metadata"]["rollout_seed"]
        == 82621 + row["metadata"]["island_id"]
        for row in baseline + train + evaluation
    )
    assert all(
        len([row for row in baseline if row["metadata"]["task_id"] == task_id]) == 4
        for task_id in train_ids | eval_ids
    )

    assert [len(_jsonl(output / "train" / f"island-{i}.jsonl")) for i in range(8)] == [
        22
    ] * 8
    baseline_counts = [
        len(_jsonl(output / "baseline" / f"island-{i}.jsonl")) for i in range(8)
    ]
    assert sum(baseline_counts) == 356
    assert max(baseline_counts) - min(baseline_counts) == 1

    value = json.loads((output / "value-pretraining-selection.json").read_text())
    assert value["selection"] == "all"
    assert value["sample_count"] == 356
    assert value["actor_eval_task_count_exposed_to_critic"] == 45
    assert value["actor_eval_trajectory_count_exposed_to_critic"] == 180
    assert value["critic_pretraining_exposes_actor_eval_tasks"] is True
    assert manifest["compaction"] == {
        "enabled": True,
        "trainer_objective": "sao",
        "max_seq_len": 8192,
        "trigger_semantics": "consumed-context-tokens-at-least",
        "trigger_tokens": 6144,
        "summary_max_tokens": 1024,
        "max_compactions_per_episode": 3,
    }

    topology = manifest["topology"]
    assert topology["islands"] == topology["physical_gpus"] == 8
    assert topology["one_physical_gpu_per_island"] is True
    assert topology["cross_island_collectives"] is False
    for island in topology["island_contracts"]:
        assert island["rollout_seed"] == 82621 + island["island_id"]
        assert island["entrypoint"].endswith("train_sao_streaming_secrlenv.py")
        assert island["runtime_contracts"]["actor_syncer_port"] == 29400
        assert island["runtime_contracts"]["critic_syncer_port"] == 29401
        assert island["openenv_agent_contract"] == {
            "custom_agent_function_path": (
                "codex_openenv_subprocess_agent_function.run"
            ),
            "custom_rm_path": "openenv_generate.reward_func",
            "dynamic_sampling_filter_path": (
                "openenv_generate.check_terminal_bench_episode"
            ),
            "input_key": "messages",
            "tito_model": "qwen35",
            "codex_backend_profile": "qwen35_08b",
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "max_seq_len": 8192,
            "must_not_enter_secrlenv_retry_path": True,
        }
        assert (
            island["openenv_agent_contract"]["custom_agent_function_path"]
            not in SECRLENV_AGENTS
        )
        assert island["ray_fractional_gpu"] == {
            "actor": 0.4,
            "critic": 0.4,
            "sglang": 0.2,
        }
        args = island["miles_topology_args"]
        assert "--sao-one-gpu-island" in args
        assert "--sao-compaction" in args
        assert args[args.index("--input-key") + 1] == "messages"
        assert "--colocate" in args
        assert args[args.index("--sglang-mem-fraction-static") + 1] == "0.15"
        assert args[args.index("--sglang-max-total-tokens") + 1] == "393216"
        assert args[args.index("--sglang-max-mamba-cache-size") + 1] == "256"
        environment = island["environment"]
        assert environment["SECRLENV_MAX_ROLLOUT_TIME_SECONDS"] == "1800"
        assert environment["OPENENV_MAX_ROLLOUT_TIME_SECONDS"] == "1800"
        assert environment["YETO_CODEX_COMPACTION_ENABLED"] == "1"
        assert environment["YETO_CODEX_COMPACTION_TRIGGER_TOKENS"] == "6144"
        assert environment["YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS"] == "1024"
        assert environment["YETO_CODEX_MAX_COMPACTIONS"] == "3"
        assert all(
            environment[name] == value
            for name, value in CODEX_OPENENV_IDENTITY_ENV.items()
        )
        assert "PYTORCH_CUDA_ALLOC_CONF" not in environment
    assert (
        manifest["required_gates"]["openenv_tbench_max_active_episodes_at_least"] == 304
    )
    assert (
        manifest["required_gates"]["openenv_compaction_trajectory_evidence_v2"] is True
    )
    assert (
        manifest["required_gates"]["terminal_bench_signed_outcome_hmac_key_file"]
        is True
    )
    assert manifest["required_gates"]["direct_codex_qwen35_08b_preflight"] is True
    assert manifest["launch_readiness"]["ready"] is False
    assert [
        issue["code"] for issue in manifest["launch_readiness"]["blocking_issues"]
    ] == [
        "task-images-not-attested",
        "openenv-capacity-not-attested",
        "one-gpu-island-runtime-not-smoked",
        "final-layout-attestations-not-generated",
    ]
    assert manifest["integration_notes"] == [
        {
            "code": "streaming-entrypoint-retains-legacy-cli-name",
            "blocking": False,
            "detail": manifest["integration_notes"][0]["detail"],
        }
    ]


def test_split_is_reproducible(tmp_path):
    tasks = _task_tree(tmp_path / "source")
    first = plan.build_plan(
        tasks,
        tmp_path / "first",
        seed="fixed-seed",
        run_root="/run/one",
    )
    second = plan.build_plan(
        tasks,
        tmp_path / "second",
        seed="fixed-seed",
        run_root="/run/two",
    )
    one = json.loads(first.read_text())["split"]
    two = json.loads(second.read_text())["split"]
    assert one == two


def test_rejects_incomplete_suite(tmp_path):
    tasks = _task_tree(tmp_path / "source", count=88)
    with pytest.raises(plan.PlanError, match="exactly 89"):
        plan.build_plan(
            tasks,
            tmp_path / "output",
            seed="fixed-seed",
            run_root="/run/tbench21",
        )
