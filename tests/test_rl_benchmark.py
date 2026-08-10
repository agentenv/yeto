"""Pure-logic tests for the Miles RL benchmark harness."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_miles_rl", ROOT / "scripts" / "benchmark_rl.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _args(**overrides):
    values = {
        "global_rounds": 3,
        "groups_per_island": 4,
        "samples_per_group": 2,
        "optimizer_steps": 1,
        "gpus_per_island": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_four_arms_have_equal_hardware_and_trajectory_budgets():
    arms = benchmark.select_arms("2", gpus_per_island=2, groups_per_island=4)

    assert [arm.name for arm in arms] == [
        "native-miles-m2",
        "yeto-single-m2",
        "yeto-federated-m2",
        "yeto-decoupled-m2",
    ]
    assert [
        (arm.islands, arm.gpus_per_island, arm.groups_per_round) for arm in arms
    ] == [
        (1, 4, 8),
        (1, 4, 8),
        (2, 2, 4),
        (2, 2, 4),
    ]

    budgets = [benchmark.workload(arm, rounds=3, samples_per_group=2) for arm in arms]
    assert {budget["total_gpus"] for budget in budgets} == {4}
    assert {budget["prompt_groups"] for budget in budgets} == {24}
    assert {budget["trajectories"] for budget in budgets} == {48}
    assert {budget["groups_per_gpu_per_round"] for budget in budgets} == {2.0}


def test_arm_selection_rejects_duplicates_and_nonpositive_values():
    with pytest.raises(ValueError, match="at least one"):
        benchmark.select_arms("", 1, 1)
    with pytest.raises(ValueError, match="duplicates"):
        benchmark.select_arms("2,2", 1, 1)
    with pytest.raises(ValueError, match="positive"):
        benchmark.select_arms("0", 1, 1)


def test_native_only_selection_keeps_the_direct_miles_arm():
    arms = benchmark.select_arms(
        "2",
        gpus_per_island=2,
        groups_per_island=4,
        kinds=benchmark.parse_arm_kinds("native"),
    )

    assert [arm.name for arm in arms] == ["native-miles-m2"]


def test_arm_kind_selection_rejects_unknown_or_duplicate_values():
    with pytest.raises(ValueError, match="unknown"):
        benchmark.parse_arm_kinds("native,old-direct")
    with pytest.raises(ValueError, match="duplicates"):
        benchmark.parse_arm_kinds("native,native")


def test_decoupled_arm_requires_the_fixed_fragment_contract():
    with pytest.raises(ValueError, match="at least 2 fragments"):
        benchmark.select_arms("2", 1, 1, kinds=("decoupled",), fragments=1)
    with pytest.raises(ValueError, match="pipeline"):
        benchmark.select_arms(
            "2", 1, 1, kinds=("decoupled",), fragments=4, pipeline=5
        )
    with pytest.raises(ValueError, match="local horizon"):
        benchmark.select_arms(
            "2", 1, 1, kinds=("decoupled",), fragments=4, local_horizon=1
        )


def test_workload_validation_requires_equal_per_rank_batch():
    benchmark.validate_workload(_args())
    with pytest.raises(ValueError, match="one optimizer step"):
        benchmark.validate_workload(_args(optimizer_steps=2))
    with pytest.raises(ValueError, match="divisible"):
        benchmark.validate_workload(_args(groups_per_island=3, samples_per_group=3))


def test_local_ray_cluster_uses_rays_short_default_temp_root(monkeypatch):
    calls = {}

    def init(**kwargs):
        calls["init"] = kwargs
        return SimpleNamespace(address_info={"address": "local"})

    ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=init,
        shutdown=lambda: calls.setdefault("shutdown", True),
    )
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setitem(sys.modules, "ray", ray)

    with benchmark.local_ray_cluster(8) as address:
        assert address == "local"

    assert "_temp_dir" not in calls["init"]
    assert calls["init"]["include_dashboard"] is True
    assert calls["shutdown"] is True


def test_gpu_samples_report_time_weighted_activity_and_utilization():
    summary = benchmark.summarize_gpu_samples(
        [
            (10.0, {"gpu-a": 50.0, "gpu-b": 0.0}),
            (12.0, {"gpu-a": 100.0, "gpu-b": 50.0}),
        ],
        started=10.0,
        ended=14.0,
        expected_gpus=2,
    )

    assert summary["gpu_active_seconds"] == pytest.approx(6.0)
    assert summary["gpu_active_fraction"] == pytest.approx(0.75)
    assert summary["mean_gpu_utilization"] == pytest.approx(50.0)
    assert summary["min_gpu_utilization"] == pytest.approx(25.0)


def test_expert_parallel_default_is_fixed_across_all_arms():
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
        ]
    )

    assert args.expert_parallel == 1


def test_chat_template_kwargs_are_forwarded_to_training_workers(tmp_path):
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
            "--apply-chat-template-kwargs",
            '{"enable_thinking": false}',
        ]
    )
    args._active_seed = 17
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text("{}\n", encoding="utf-8")
    worker = benchmark.WorkerSpec(0, 1, 1, prompt, False)

    payload = benchmark.worker_payload(
        args,
        worker,
        arm=benchmark.select_arms("1", 1, 1, kinds=("native",))[0],
        run_dir=tmp_path,
        model_path=tmp_path / "model",
        syncer=None,
        reward_sha256="c" * 64,
    )

    assert payload["arguments"]["apply_chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_prompt_streams_are_round_major_and_exactly_paired():
    rows = [
        {
            "messages": [{"role": "user", "content": f"prompt {index}"}],
            "label": str(index),
        }
        for index in range(8)
    ]

    streams = benchmark.paired_prompt_streams(rows, islands=2, groups=2, rounds=2)

    assert streams.combined_ids == tuple(range(8))
    assert streams.island_ids == ((0, 1, 4, 5), (2, 3, 6, 7))
    for round_id in range(2):
        combined = streams.combined_ids[round_id * 4 : (round_id + 1) * 4]
        federated = tuple(
            prompt_id
            for island in streams.island_ids
            for prompt_id in island[round_id * 2 : (round_id + 1) * 2]
        )
        assert federated == combined


def test_prompt_streams_cycle_deterministically_without_touching_eval_rows():
    rows = [
        {"messages": [{"role": "user", "content": value}]} for value in ("a", "b", "c")
    ]
    streams = benchmark.paired_prompt_streams(rows, islands=2, groups=2, rounds=2)
    assert streams.combined_ids == (0, 1, 2, 0, 1, 2, 0, 1)


def test_worker_specs_keep_native_outside_yeto_and_partition_federated_prompts(
    tmp_path,
):
    native, single, federated, decoupled = benchmark.select_arms("2", 2, 4)
    combined = tmp_path / "combined.jsonl"
    islands = (tmp_path / "island-0.jsonl", tmp_path / "island-1.jsonl")

    native_specs = benchmark.worker_specs(native, combined, islands)
    single_specs = benchmark.worker_specs(single, combined, islands)
    federated_specs = benchmark.worker_specs(federated, combined, islands)
    decoupled_specs = benchmark.worker_specs(decoupled, combined, islands)

    assert len(native_specs) == len(single_specs) == 1
    assert native_specs[0].policy_sync is False
    assert single_specs[0].policy_sync is True
    assert native_specs[0].gpus == single_specs[0].gpus == 4
    assert native_specs[0].groups_per_round == single_specs[0].groups_per_round == 8
    assert [spec.prompt_path for spec in federated_specs] == list(islands)
    assert [spec.learner_id for spec in federated_specs] == [0, 1]
    assert all(spec.policy_sync for spec in federated_specs)
    assert all(
        spec.gpus == 2 and spec.groups_per_round == 4 for spec in federated_specs
    )
    assert decoupled_specs == federated_specs


def test_federated_workers_use_disjoint_miles_host_ports(tmp_path):
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
        ]
    )
    args._active_seed = 17
    arm = benchmark.select_arms("2", 2, 4)[2]
    combined = tmp_path / "combined.jsonl"
    island_paths = (tmp_path / "island-0.jsonl", tmp_path / "island-1.jsonl")
    for path in island_paths:
        path.write_text("{}\n", encoding="utf-8")
    workers = benchmark.worker_specs(arm, combined, island_paths)

    payloads = [
        benchmark.worker_payload(
            args,
            worker,
            arm=arm,
            run_dir=tmp_path,
            model_path=tmp_path / "model",
            syncer="127.0.0.1:30000",
            reward_sha256="c" * 64,
        )
        for worker in workers
    ]

    assert [
        payload["arguments"]["rollout_engine_base_port"] for payload in payloads
    ] == [21100, 22100]
    assert [payload["arguments"]["sglang_router_port"] for payload in payloads] == [
        21000,
        22000,
    ]
    assert [
        payload["arguments"]["sglang_router_prometheus_port"] for payload in payloads
    ] == [21001, 22001]
    assert [payload["arguments"]["train_master_base_port"] for payload in payloads] == [
        21002,
        22002,
    ]
    assert {payload["arguments"]["expert_parallel"] for payload in payloads} == {1}


def test_reward_summary_uses_standard_pass_at_k_estimator():
    result = benchmark.summarize_rewards(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        pass_ks=(1, 4),
        threshold=0.0,
    )

    assert result["reward_mean"] == pytest.approx(0.125)
    assert result["pass_at_k"] == {"1": pytest.approx(0.125), "4": 0.5}
    with pytest.raises(ValueError, match="cannot exceed"):
        benchmark.summarize_rewards([[1.0, 0.0]], pass_ks=(3,), threshold=0.0)


def test_resume_keys_cover_every_seed_m_and_arm():
    arms = benchmark.select_arms("2,4", 1, 2)
    keys = benchmark.expected_record_keys((17, 29), arms)

    assert len(keys) == 16
    assert ("native-miles-m2", 2, 17) in keys
    assert ("yeto-federated-m4", 4, 29) in keys
    assert ("yeto-decoupled-m4", 4, 29) in keys
    benchmark.validate_result_records(
        [
            {"arm": name, "m": m, "seed": seed}
            for name, m, seed in sorted(keys, key=str)
        ],
        keys,
    )
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.validate_result_records(
            [
                {"arm": "native-miles-m2", "m": 2, "seed": 17},
                {"arm": "native-miles-m2", "m": 2, "seed": 17},
            ],
            keys,
        )


def test_resume_identity_survives_json_round_trip(monkeypatch):
    monkeypatch.setattr(
        "yeto.benchmark_resume.implementation_fingerprint",
        lambda _root, _paths: "d" * 64,
    )
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
        ]
    )
    args.eval_samples_per_prompt = args.samples_per_group
    args.reward_sha256 = "c" * 64
    arms = benchmark.select_arms(
        args.islands, args.gpus_per_island, args.groups_per_island
    )
    benchmark.validate_args(args, arms, check_runtime=False)

    identity = benchmark._resume_identity(args, arms)

    assert json.loads(json.dumps(identity)) == identity
    assert identity["arguments"]["reward_sha256"] == "c" * 64
    assert identity["implementation_sha256"] == "d" * 64


def test_resume_identity_fingerprints_all_yeto_sources_and_syncer_binary(
    monkeypatch,
):
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
        ]
    )
    args.eval_samples_per_prompt = args.samples_per_group
    args.reward_sha256 = "c" * 64
    arms = benchmark.select_arms(
        args.islands,
        args.gpus_per_island,
        args.groups_per_island,
    )
    observed = {}

    def fingerprint(root, paths):
        observed["root"] = root
        observed["paths"] = tuple(paths)
        return "d" * 64

    monkeypatch.setattr(
        "yeto.benchmark_resume.implementation_fingerprint",
        fingerprint,
    )

    identity = benchmark._resume_identity(args, arms)

    assert identity["implementation_sha256"] == "d" * 64
    assert benchmark.REPO_ROOT / "yeto" in observed["paths"]
    assert benchmark.SYNCER_BIN in observed["paths"]


def test_report_deltas_distinguish_yeto_contract_from_federation():
    records = [
        {"arm": "native-miles-m2", "m": 2, "seed": 17, "eval": {"reward_mean": 0.5}},
        {"arm": "yeto-single-m2", "m": 2, "seed": 17, "eval": {"reward_mean": 0.4}},
        {"arm": "yeto-federated-m2", "m": 2, "seed": 17, "eval": {"reward_mean": 0.3}},
        {"arm": "yeto-decoupled-m2", "m": 2, "seed": 17, "eval": {"reward_mean": 0.35}},
    ]

    annotated = {row["arm"]: row for row in benchmark.annotate_deltas(records)}

    assert annotated["native-miles-m2"]["delta_vs_native"] is None
    assert annotated["yeto-single-m2"]["delta_vs_native"] == pytest.approx(-0.1)
    assert annotated["yeto-federated-m2"]["delta_vs_native"] == pytest.approx(-0.2)
    assert annotated["yeto-federated-m2"]["delta_vs_single"] == pytest.approx(-0.1)
    assert annotated["yeto-decoupled-m2"]["delta_vs_native"] == pytest.approx(-0.15)
    assert annotated["yeto-decoupled-m2"]["delta_vs_strict"] == pytest.approx(0.05)


def test_aggregate_report_keeps_the_three_comparisons_in_order(tmp_path):
    records = []
    for arm, kind, reward, sync in (
        ("native-miles-m2", "native", 0.5, None),
        (
            "yeto-single-m2",
            "single",
            0.4,
            {"mean_kl": 0.1, "mean_sync_ms": 2.0, "sync_bytes_sent": 100},
        ),
        (
            "yeto-federated-m2",
            "federated",
            0.3,
            {
                "mean_kl": 0.2,
                "mean_sync_ms": 3.0,
                "sync_bytes_sent": 200,
                "mean_pull_to_push_s": 0.5,
                "mean_bcast_queue_s": 1.5,
            },
        ),
    ):
        records.append(
            {
                "arm": arm,
                "kind": kind,
                "m": 2,
                "seed": 17,
                "total_gpus": 4,
                "train_wall_s": 10.0,
                "artifact_s": None if kind == "native" else 2.0,
                "artifact_ready_s": 10.0 if kind == "native" else 12.0,
                "gpu_hours": 1.0,
                "estimated_cost": None,
                "training": {"trajectories": 8, "action_tokens": 32},
                "sync": sync,
                "eval": {
                    "reward_mean": reward,
                    "pass_at_k": {"1": reward},
                    "wall_s": 1.0,
                },
            }
        )

    aggregates = benchmark.aggregate_records(records)

    assert [row["arm"] for row in aggregates] == [
        "native-miles-m2",
        "yeto-single-m2",
        "yeto-federated-m2",
    ]
    assert aggregates[1]["delta_vs_native"] == pytest.approx(-0.1)
    assert aggregates[2]["delta_vs_single"] == pytest.approx(-0.1)
    assert aggregates[0]["artifact_ready_s"] == 10.0
    assert aggregates[1]["artifact_ready_s"] == 12.0
    assert aggregates[2]["mean_pull_to_push_s"] == pytest.approx(0.5)

    args = SimpleNamespace(
        model="org/model",
        global_rounds=1,
        seeds="17",
        eval_prompts=1,
        eval_samples_per_prompt=1,
        report_dir=tmp_path,
    )
    benchmark.write_report(args, records)
    report = (tmp_path / "report.md").read_text()
    assert "native-miles-m2" in report
    assert "artifact-ready s" in report


def test_dry_run_does_not_import_ray_or_materialize_data(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ray", None)
    result = benchmark.main(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
            "--islands",
            "2",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "native-miles-m2" in output
    assert "yeto-single-m2" in output
    assert "yeto-federated-m2" in output
    plan = json.loads(output.split("PLAN_JSON ", 1)[1])
    assert plan["fairness"]["same_total_gpus"]
    assert plan["fairness"]["same_expert_parallel"]
    assert plan["expert_parallel"] == 1


def test_dry_run_can_select_only_the_current_native_miles_arm(capsys):
    assert benchmark.main(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "yeto.tasks.cybergym.reward:score",
            "--arms",
            "native",
            "--islands",
            "2",
            "--dry-run",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "native-miles-m2" in output
    assert "yeto-single-m2" not in output
    assert "yeto-federated-m2" not in output


def test_dry_run_works_when_script_is_invoked_outside_repo(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_rl.py"),
        "--model",
        "org/model",
        "--model-revision",
        "a" * 40,
        "--data",
        "org/data",
        "--data-revision",
        "b" * 40,
        "--reward-function",
        "pkg.reward:score",
        "--dry-run",
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PLAN_JSON" in result.stdout


def test_benchmark_rejects_mutable_local_model_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            str(model),
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
        ]
    )
    args.eval_samples_per_prompt = args.samples_per_group
    arms = benchmark.select_arms(
        args.islands,
        args.gpus_per_island,
        args.groups_per_island,
    )

    with pytest.raises(ValueError, match="local model"):
        benchmark.validate_args(args, arms, check_runtime=False)


def test_benchmark_rejects_rollout_port_ranges_outside_host_port_space():
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
            "--islands",
            "2",
            "--miles-port-base",
            "65000",
        ]
    )
    args.eval_samples_per_prompt = args.samples_per_group
    arms = benchmark.select_arms(
        args.islands,
        args.gpus_per_island,
        args.groups_per_island,
    )

    with pytest.raises(ValueError, match="Miles host port ranges"):
        benchmark.validate_args(args, arms, check_runtime=False)


def test_materialized_prompt_files_preserve_holdout_and_prompt_identity(tmp_path):
    rows = [
        {
            "messages": [{"role": "user", "content": f"prompt {index}"}],
            "label": str(index),
            "metadata": {"source_id": index},
        }
        for index in range(6)
    ]
    train, evaluation = rows[:-2], rows[-2:]
    streams = benchmark.paired_prompt_streams(train, islands=2, groups=2, rounds=1)
    combined, islands, evaluation_path = benchmark.write_prompt_files(
        streams,
        evaluation,
        tmp_path,
    )

    combined_rows = [json.loads(line) for line in combined.read_text().splitlines()]
    eval_rows = [json.loads(line) for line in evaluation_path.read_text().splitlines()]
    assert [row["metadata"]["benchmark_prompt_id"] for row in combined_rows] == [
        0,
        1,
        2,
        3,
    ]
    assert [row["label"] for row in eval_rows] == ["4", "5"]
    assert [row["metadata"]["source_id"] for row in eval_rows] == [4, 5]
    assert len(islands) == 2


def test_strict_syncer_command_is_one_fragment_exact_base_avg(tmp_path):
    arm = benchmark.select_arms("2", 2, 4)[2]
    command = benchmark.syncer_command(arm, 29400, tmp_path, rounds=3)

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--learners") == "2"
    assert value("--quorum") == "2"
    assert value("--total-steps") == "3"
    assert value("--pipeline") == "1"
    assert value("--sync-interval-steps") == "0"
    assert value("--delta-correction") == "none"
    assert value("--outer-lr") == "1"
    assert value("--outer-momentum") == "0"
    assert value("--max-base-lag") == "0"
    assert value("--learner-weight") == "equal"
    assert "--mark-final-checkpoint" not in command


def test_decoupled_syncer_commands_split_budget_cutoff_and_consolidation(tmp_path):
    arm = benchmark.select_arms(
        "2",
        2,
        4,
        kinds=("decoupled",),
        fragments=8,
        pipeline=2,
        local_horizon=4,
    )[0]

    cutoff = benchmark.syncer_command(arm, 29400, tmp_path, rounds=6)
    final = benchmark.syncer_command(
        arm,
        29400,
        tmp_path,
        rounds=6,
        resume_from_step=11,
    )

    assert cutoff[cutoff.index("--total-steps") + 1] == "48"
    assert cutoff[cutoff.index("--learner-budget-steps") + 1] == "6"
    assert cutoff[cutoff.index("--pipeline") + 1] == "2"
    assert cutoff[cutoff.index("--sync-interval-steps") + 1] == "4"
    assert cutoff[cutoff.index("--outer-lr") + 1] == "0.7"
    assert cutoff[cutoff.index("--outer-momentum") + 1] == "0.9"
    assert "--resume" not in cutoff

    assert final[final.index("--total-steps") + 1] == "19"
    assert final[final.index("--pipeline") + 1] == "1"
    assert final[final.index("--sync-interval-steps") + 1] == "0"
    assert "--learner-budget-steps" not in final
    assert "--resume" in final
    assert "--mark-final-checkpoint" in final


def test_decoupled_worker_receives_budget_and_fragment_contract(tmp_path):
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
            "--global-rounds",
            "6",
            "--fragments",
            "8",
            "--pipeline",
            "2",
            "--local-horizon",
            "4",
        ]
    )
    args._active_seed = 17
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text("{}\n", encoding="utf-8")
    arm = benchmark.select_arms(
        "2",
        1,
        1,
        kinds=("decoupled",),
        fragments=8,
        pipeline=2,
        local_horizon=4,
    )[0]
    worker = benchmark.worker_specs(arm, prompt, (prompt, prompt))[0]

    payload = benchmark.worker_payload(
        args,
        worker,
        arm=arm,
        run_dir=tmp_path,
        model_path=tmp_path / "model",
        syncer="127.0.0.1:29400",
        reward_sha256="c" * 64,
    )

    values = payload["arguments"]
    assert values["sync_preset"] == "decoupled"
    assert values["fragments"] == 8
    assert values["pipeline"] == 2
    assert values["local_horizon"] == 4
    assert values["total_fragment_steps"] == 48
    assert values["learner_budget_steps"] == 6


def test_decoupled_training_restarts_syncer_for_terminal_consolidation(
    tmp_path, monkeypatch
):
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--model-revision",
            "a" * 40,
            "--data",
            "org/data",
            "--data-revision",
            "b" * 40,
            "--reward-function",
            "pkg.reward:score",
            "--global-rounds",
            "6",
        ]
    )
    args._active_seed = 17
    args.miles_root = tmp_path
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text("{}\n", encoding="utf-8")
    arm = benchmark.select_arms("2", 1, 1, kinds=("decoupled",))[0]
    workers = benchmark.worker_specs(arm, prompt, (prompt, prompt))
    commands = []

    class Process:
        returncode = 0

        def poll(self):
            return 0

    def popen(command, **_kwargs):
        commands.append(command)
        return Process()

    @benchmark.contextmanager
    def ray_cluster(_gpus):
        yield "ray"

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "local_ray_cluster", ray_cluster)
    monkeypatch.setattr(benchmark, "_wait_for_port", lambda *_args: None)
    monkeypatch.setattr(benchmark, "_wait_for_training", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark, "_stop_process", lambda *_args: None)

    class Sampler:
        def __init__(self, expected_gpus):
            self.samples = [
                (
                    benchmark.time.monotonic(),
                    {f"gpu-{index}": 50.0 for index in range(expected_gpus)},
                )
            ]
            self.error = None

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(benchmark, "_GpuSampler", Sampler)
    monkeypatch.setattr("yeto.final_marker.read_checkpoint_global_step", lambda _path: 11)
    validated = {}
    monkeypatch.setattr(
        "yeto.budget_finalization.validate_consolidation_tape",
        lambda path, **kwargs: validated.update(path=path, **kwargs),
    )

    benchmark._run_training_processes(
        args,
        arm,
        workers,
        run_dir=tmp_path / "run",
        model_path=tmp_path / "model",
        reward_sha256="c" * 64,
    )

    syncers = [command for command in commands if command[0] == str(benchmark.SYNCER_BIN)]
    assert len(syncers) == 2
    assert "--learner-budget-steps" in syncers[0]
    assert syncers[1][syncers[1].index("--total-steps") + 1] == "19"
    assert validated == {
        "path": tmp_path / "run" / "syncer.jsonl",
        "cutoff_step": 11,
        "fragments": 8,
        "learners": 2,
        "budget_steps": 6,
    }


def test_worker_miles_extras_capture_real_rollouts_and_only_native_saves(tmp_path):
    native, single, _, _ = benchmark.select_arms("2", 2, 4)
    native_worker = benchmark.worker_specs(native, tmp_path / "all", ())[0]
    single_worker = benchmark.worker_specs(single, tmp_path / "all", ())[0]

    native_extra = benchmark.miles_extra_argv(native_worker, tmp_path / "native", 3)
    single_extra = benchmark.miles_extra_argv(single_worker, tmp_path / "single", 3)

    assert "--save-debug-rollout-data" in native_extra
    assert "--save-debug-rollout-data" in single_extra
    assert "--save" in native_extra
    assert native_extra[native_extra.index("--save-interval") + 1] == "3"
    assert "--save-hf" not in native_extra
    assert "--save-hf" not in single_extra


def test_native_miles_adapter_names_are_mapped_to_the_exact_peft_contract():
    specs = (
        SimpleNamespace(
            name="base_model.model.model.layers.0.q_proj.lora_A.weight",
            shape=(2, 4),
        ),
        SimpleNamespace(
            name="base_model.model.model.layers.0.q_proj.lora_B.weight",
            shape=(4, 2),
        ),
    )
    raw = {
        "model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
        "model.layers.0.q_proj.lora_B.weight": torch.ones(4, 2),
    }

    mapped = benchmark.canonical_native_adapter_tensors(raw, specs)

    assert tuple(mapped) == tuple(spec.name for spec in specs)
    assert all(
        torch.equal(tensor, torch.ones(spec.shape))
        for tensor, spec in zip(mapped.values(), specs)
    )

    raw.pop("model.layers.0.q_proj.lora_B.weight")
    with pytest.raises(RuntimeError, match="does not match the PEFT contract"):
        benchmark.canonical_native_adapter_tensors(raw, specs)


def test_native_miles_adapter_is_rewritten_as_standard_peft(tmp_path, monkeypatch):
    from yeto.rl import export as rl_export
    from yeto.rl.core import CanonicalTensorSpec

    specs = (
        CanonicalTensorSpec(
            "base_model.model.model.layers.0.q_proj.lora_A.weight",
            (2, 4),
            "float32",
            8,
        ),
        CanonicalTensorSpec(
            "base_model.model.model.layers.0.q_proj.lora_B.weight",
            (4, 2),
            "float32",
            8,
        ),
    )
    source = tmp_path / "miles"
    source.mkdir()
    torch.save(
        {
            "model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4),
            "model.layers.0.q_proj.lora_B.weight": torch.ones(4, 2),
        },
        source / "adapter_model.bin",
    )
    written = {}
    monkeypatch.setattr(
        rl_export, "derive_peft_lora_specs", lambda *args, **kwargs: specs
    )
    monkeypatch.setattr(
        rl_export,
        "write_peft_adapter",
        lambda state, output, **kwargs: written.update(
            state=state,
            output=output,
            kwargs=kwargs,
        ),
    )
    args = SimpleNamespace(
        model="org/model",
        model_revision="a" * 40,
        lora_r=2,
        lora_targets="attention",
        global_rounds=3,
        trust_remote_code=True,
    )

    output = benchmark.standardize_native_adapter(args, source, tmp_path / "adapter")

    assert output == tmp_path / "adapter"
    assert written["state"].policy_version == 3
    assert written["state"].specs == specs
    assert written["kwargs"] == {
        "base_model": "org/model",
        "model_revision": "a" * 40,
        "rank": 2,
    }


def test_rollout_summary_verifies_prompt_pairing_and_counts_real_work(tmp_path):
    paths = []
    for island_id, prompt_ids in enumerate(((0, 1), (2, 3))):
        path = tmp_path / f"island-{island_id}.pt"
        samples = []
        for prompt_id in prompt_ids:
            for sample_id in range(2):
                samples.append(
                    {
                        "metadata": {"benchmark_prompt_id": prompt_id},
                        "reward": float((prompt_id + sample_id) % 2),
                        "response_length": prompt_id + sample_id + 1,
                        "rollout_routed_experts": np.array([prompt_id]),
                        "status": "completed",
                    }
                )
        torch.save({"rollout_id": 0, "samples": samples}, path)
        paths.append((path,))

    summary = benchmark.summarize_rollouts(
        tuple(paths),
        expected_prompt_ids=((0, 1), (2, 3)),
        samples_per_group=2,
    )

    assert summary["prompt_groups"] == 4
    assert summary["trajectories"] == 8
    assert summary["action_tokens"] == 24
    assert summary["reward_mean"] == 0.5
    assert summary["truncated_trajectories"] == 0

    payload = torch.load(paths[1][0], weights_only=False)
    for sample in payload["samples"][:2]:
        sample["metadata"]["benchmark_prompt_id"] = 99
    torch.save(payload, paths[1][0])
    with pytest.raises(RuntimeError, match="prompt stream mismatch"):
        benchmark.summarize_rollouts(
            tuple(paths),
            expected_prompt_ids=((0, 1), (2, 3)),
            samples_per_group=2,
        )

    payload = torch.load(paths[1][0], weights_only=False)
    for sample in payload["samples"][:2]:
        sample["metadata"]["benchmark_prompt_id"] = 2
    payload["samples"][0]["status"] = "aborted"
    torch.save(payload, paths[1][0])
    with pytest.raises(RuntimeError, match="status"):
        benchmark.summarize_rollouts(
            tuple(paths),
            expected_prompt_ids=((0, 1), (2, 3)),
            samples_per_group=2,
        )


def test_yeto_event_summary_reports_decoupled_overlap_metrics(tmp_path):
    island = tmp_path / "island-0"
    island.mkdir()
    events = [
        {
            "event": "rl_local_round",
            "rollout_seconds": 2.0,
            "train_seconds": 3.0,
            "mean_kl": 0.1,
            "ess_ratio": 0.9,
            "clip_fraction": 0.2,
            "sync/fragment_payload_bytes_sent": 10,
            "sync/fragment_payload_bytes_received": 20,
        },
        {
            "event": "rl_fragment_push",
            "realized_h": 4,
            "pull_to_push_seconds": 0.5,
        },
        {"event": "rl_policy_snapshot"},
        {
            "event": "rl_policy_apply",
            "partial_fragment_apply": True,
            "sync/apply_seconds": 0.25,
        },
        {
            "event": "rl_fragment_bcast",
            "queue_seconds": 1.5,
        },
        {
            "event": "rl_sync_hook",
            "sync/hook_seconds": 0.75,
            "sync/remote_quorum_wait_seconds": 0.0,
            "sync/finalization": False,
        },
        {
            "event": "rl_sync_hook",
            "sync/hook_seconds": 1.25,
            "sync/remote_quorum_wait_seconds": 0.0,
            "sync/finalization": True,
        },
        {
            "event": "rl_final_cut",
            "sync/fragment_payload_bytes_received": 4,
        },
    ]
    (island / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (tmp_path / "syncer.jsonl").write_text(
        json.dumps({"ms": 3.0, "responders": [{"id": 0}]}) + "\n",
        encoding="utf-8",
    )

    summary = benchmark.summarize_yeto_events(tmp_path, islands=1)

    assert summary["policy_snapshots"] == 1
    assert summary["in_process_applies"] == 1
    assert summary["hook_s"] == pytest.approx(2.0)
    assert summary["finalization_s"] == pytest.approx(1.25)
    assert summary["remote_quorum_wait_s"] == 0
    assert summary["fragment_payload_bytes_received"] == 24
    assert summary["fragment_payload_traffic_bytes"] == 34
    assert summary["mean_realized_h"] == 4
    assert summary["mean_pull_to_push_s"] == pytest.approx(0.5)
    assert summary["mean_bcast_queue_s"] == pytest.approx(1.5)
    assert summary["mean_bcast_apply_s"] == pytest.approx(0.25)
    assert summary["mean_responders"] == 1


def test_yeto_event_summary_keeps_legacy_bytes_out_of_fragment_payload(tmp_path):
    island = tmp_path / "island-0"
    island.mkdir()
    (island / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "rl_local_round",
                "sync/bytes_sent": 128,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = benchmark.summarize_yeto_events(tmp_path, islands=1)

    assert summary["sync_bytes_sent"] == 128
    assert summary["fragment_payload_bytes_sent"] is None
    assert summary["fragment_payload_bytes_received"] is None
    assert summary["fragment_payload_traffic_bytes"] is None


def test_evaluation_prompt_matches_miles_rendering_and_truncates_all_token_fields(
    monkeypatch,
):
    calls = {}

    def render(
        messages,
        *,
        tokenizer,
        tools,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        calls.update(
            messages=messages,
            tokenizer=tokenizer,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return "rendered prompt"

    monkeypatch.setitem(
        sys.modules,
        "miles.utils.chat_template_utils",
        SimpleNamespace(apply_chat_template=render),
    )

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors):
            assert text == "rendered prompt"
            assert add_special_tokens is False
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 1, 1]]),
            }

    tokenizer = Tokenizer()
    row = {
        "messages": [{"role": "user", "content": "question"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    prompt, encoded = benchmark.prepare_evaluation_prompt(
        tokenizer,
        row,
        max_prompt_tokens=3,
        device="cpu",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert prompt == "rendered prompt"
    assert encoded["input_ids"].tolist() == [[2, 3, 4]]
    assert encoded["attention_mask"].tolist() == [[1, 1, 1]]
    assert encoded["token_type_ids"].tolist() == [[0, 1, 1]]
    assert calls == {
        "messages": row["messages"],
        "tokenizer": tokenizer,
        "tools": row["tools"],
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_evaluation_worker_receives_the_same_chat_template_kwargs(
    tmp_path,
    monkeypatch,
):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        reward_function="pkg.reward:score",
        samples_per_group=4,
        groups_per_island=4,
        rollout_max_response_len=768,
        seq_len=2048,
        eval_samples_per_prompt=4,
        _pass_ks=(1, 4),
        pass_threshold=0.0,
        eval_temperature=1.0,
        eval_top_p=1.0,
        eval_seed=100000,
        eval_device="cpu",
        trust_remote_code=True,
        miles_root=tmp_path,
        arm_timeout_min=1,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    def run(command, _log, **_kwargs):
        payload = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        assert payload["apply_chat_template_kwargs"] == {
            "enable_thinking": False
        }
        Path(payload["result_path"]).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(benchmark, "_run_checked", run)

    assert benchmark.evaluate_artifact(
        args,
        adapter_path=tmp_path / "adapter",
        model_path=tmp_path / "model",
        eval_path=eval_path,
        reward_sha256="c" * 64,
        run_dir=tmp_path / "run",
        seed=17,
    ) == {}


def test_evaluation_rewards_use_miles_single_sample_contract(monkeypatch):
    observed = []

    async def score(_args, sample):
        observed.append(sample)
        return float(sample)

    monkeypatch.setitem(
        sys.modules,
        "miles.rollout.rm_hub",
        SimpleNamespace(async_rm=score),
    )

    rewards = asyncio.run(benchmark.evaluate_rewards(SimpleNamespace(), [1, 2, 3]))

    assert rewards == [1.0, 2.0, 3.0]
    assert observed == [1, 2, 3]


def test_evaluation_rejects_missing_peft_adapter_keys():
    class PeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_path):
            warnings.warn(
                "Found missing adapter keys while loading the checkpoint: ['lora_A']."
            )
            return model

    with pytest.raises(RuntimeError, match="missing adapter keys"):
        benchmark.load_peft_adapter(PeftModel, object(), "adapter")


def test_generation_pad_token_keeps_valid_zero_id():
    assert (
        benchmark.generation_pad_token_id(
            SimpleNamespace(pad_token_id=0, eos_token_id=2)
        )
        == 0
    )
    assert (
        benchmark.generation_pad_token_id(
            SimpleNamespace(pad_token_id=None, eos_token_id=[2, 3])
        )
        == 2
    )


def test_worker_input_verification_rejects_prompt_or_reward_drift(
    tmp_path,
    monkeypatch,
):
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text("{}\n", encoding="utf-8")
    expected_prompt = benchmark.file_sha256(prompt)
    monkeypatch.setattr(
        "yeto.provenance.python_spec_sha256",
        lambda spec, base_dir=None: "a" * 64,
    )

    benchmark.verify_worker_inputs(
        prompt_path=prompt,
        prompt_sha256=expected_prompt,
        reward_function="pkg.reward:score",
        reward_sha256="a" * 64,
    )
    prompt.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="prompt source SHA256 mismatch"):
        benchmark.verify_worker_inputs(
            prompt_path=prompt,
            prompt_sha256=expected_prompt,
            reward_function="pkg.reward:score",
            reward_sha256="a" * 64,
        )

    with pytest.raises(RuntimeError, match="reward source SHA256 mismatch"):
        benchmark.verify_worker_inputs(
            prompt_path=prompt,
            prompt_sha256=benchmark.file_sha256(prompt),
            reward_function="pkg.reward:score",
            reward_sha256="b" * 64,
        )


def test_syncer_is_always_built_from_the_locked_source(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)

    benchmark.ensure_syncer()

    assert calls == [
        (
            ["cargo", "build", "--release", "--locked", "--quiet"],
            {"cwd": benchmark.REPO_ROOT / "syncer", "check": True},
        )
    ]


def test_gpu_drain_check_ignores_compute_apps_on_hidden_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def run(command, **_kwargs):
        query = next(value for value in command if value.startswith("--query-"))
        if query.startswith("--query-gpu"):
            stdout = "0, GPU-visible\n1, GPU-hidden\n"
        else:
            stdout = "GPU-hidden, 123, python, 30000\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", run)

    assert benchmark._visible_gpu_uuids() == {"GPU-visible"}
    benchmark.wait_for_free_gpus(timeout_s=0)
