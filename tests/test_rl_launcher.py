from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

from yeto import launcher
from yeto.adapter_lifecycle import directory_sha256
from yeto.cli import parse_args
from yeto.launcher import (
    FleetController,
    _prepare_rl_args,
    make_miles_island_task,
    syncer_command,
)
from yeto.rl import (
    MILES_COMMIT,
    MILES_PEFT_VERSION,
    MILES_REPOSITORY,
    SGLANG_COMMIT,
    SGLANG_REPOSITORY,
)
from yeto.rl import learner as rl_learner
from yeto.rl.core import CanonicalTensorSpec
from yeto.rl.decoupled import DecoupledBridgeConfig
from yeto.rl.learner import build_miles_argv
from yeto.rl.miles import verify_miles_revision


def _args(extra=()):
    return parse_args(
        [
            "--gpu",
            "aws:1xa100@us-east-1,aws:1xa100@us-west-2",
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--training-mode",
            "rl",
            "--total-steps",
            "3",
            "--rollout-batch-size",
            "4",
            "--n-samples-per-prompt",
            "2",
            "--rollout-max-response-len",
            "128",
            "--local-rl-rounds-per-sync",
            "1",
            "--reward-function",
            "pkg.reward:score",
            "--trust-remote-code",
            *extra,
        ]
    )


def test_init_rl_cli_and_strict_preset_are_the_public_contract():
    args = _args()
    _prepare_rl_args(args)
    assert args.training_mode == "rl"
    assert args.rl_runtime == "miles"
    assert args.advantage_estimator == "grpo"
    assert args.rl_sync_preset == "strict-avg"
    assert args.rl_policy_version == "strict"
    assert args.local_rl_rounds_per_sync == 1
    assert args.over_sampling_batch_size == args.rollout_batch_size
    assert (
        args.fragments,
        args.quorum,
        args.grace_ms,
        args.pipeline,
        args.sync_interval_steps,
        args.delta_correction,
        args.outer_lr,
        args.outer_momentum,
        args.merge_alpha,
        args.wire_dtype,
    ) == (1, 2, 0, 1, 0.0, "none", 1.0, 0.0, 0.0, "f32")
    assert not hasattr(args, "rl_global_rounds")


def test_variance_aware_filter_requires_and_forwards_oversampling():
    args = _args(
        [
            "--over-sampling-batch-size",
            "8",
            "--dynamic-sampling-filter-path",
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        ]
    )
    _prepare_rl_args(args)
    assert args.over_sampling_batch_size == 8
    assert args.dynamic_sampling_filter_path.endswith("check_reward_nonzero_std")

    invalid = _args(
        [
            "--dynamic-sampling-filter-path",
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        ]
    )
    with pytest.raises(ValueError, match="greater than"):
        _prepare_rl_args(invalid)


def test_bounded_variance_filter_and_large_run_safety_options():
    args = _args(
        [
            "--over-sampling-batch-size",
            "8",
            "--dynamic-sampling-filter-path",
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
            "--dynamic-sampling-max-replacements",
            "4",
            "--rl-offload-train",
            "--rl-distributed-timeout-minutes",
            "7",
        ]
    )
    _prepare_rl_args(args)
    assert args.dynamic_sampling_filter_path == (
        "yeto.rl.filters.bounded_nonzero_reward_std"
    )
    assert args.rl_offload_train is True
    assert args.rl_distributed_timeout_minutes == 7


def test_signed_secrlenv_variance_filter_accepts_a_bounded_replacement_budget():
    args = _args(
        [
            "--over-sampling-batch-size",
            "8",
            "--dynamic-sampling-filter-path",
            "yeto_miles_secrlenv.reward.check_group",
            "--dynamic-sampling-max-replacements",
            "4",
        ]
    )

    _prepare_rl_args(args)

    assert args.dynamic_sampling_filter_path == (
        "yeto_miles_secrlenv.reward.check_group"
    )
    assert args.dynamic_sampling_max_replacements == 4


def test_decoupled_rl_preset_fixes_the_fragment_outer_contract():
    args = _args(
        (
            "--rl-sync-preset",
            "decoupled",
            "--fragments",
            "4",
            "--pipeline",
            "2",
            "--local-rl-rounds-per-sync",
            "3",
        )
    )

    _prepare_rl_args(args)

    assert args.total_steps == 3
    assert args.rl_total_fragment_steps == 12
    assert (
        args.fragments,
        args.quorum,
        args.grace_ms,
        args.pipeline,
        args.sync_interval_steps,
        args.delta_correction,
        args.outer_lr,
        args.outer_momentum,
        args.merge_alpha,
        args.wire_dtype,
        args.fragment_pattern,
    ) == (4, 2, 0, 2, 3.0, "none", 0.7, 0.9, 0.0, "f32", "binpack")
    command = syncer_command(args, 2, binary="syncer")
    assert "--total-steps 12" in command


def test_decoupled_rl_attests_a_local_initial_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    digest = directory_sha256(adapter)
    args = _args(
        (
            "--rl-sync-preset",
            "decoupled",
            "--fragments",
            "2",
            "--local-rl-rounds-per-sync",
            "2",
            "--rl-initial-adapter",
            str(adapter),
            "--rl-initial-adapter-sha256",
            digest.upper(),
        )
    )

    _prepare_rl_args(args)

    assert args.rl_initial_adapter == str(adapter)
    assert args.rl_initial_adapter_sha256 == digest


def test_rl_initial_adapter_rejects_a_supplied_digest_mismatch(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    args = _args(
        (
            "--rl-sync-preset",
            "decoupled",
            "--fragments",
            "2",
            "--local-rl-rounds-per-sync",
            "2",
            "--rl-initial-adapter",
            str(adapter),
            "--rl-initial-adapter-sha256",
            "0" * 64,
        )
    )

    with pytest.raises(ValueError, match="initial adapter SHA256 mismatch"):
        _prepare_rl_args(args)


@pytest.mark.parametrize(
    "extra,match",
    [
        (("--rl-initial-adapter-sha256", "0" * 64), "requires --rl-initial-adapter"),
        (
            (
                "--rl-sync-preset",
                "decoupled",
                "--local-rl-rounds-per-sync",
                "2",
                "--rl-initial-adapter",
                "/does/not/exist",
            ),
            "local directory",
        ),
    ],
)
def test_rl_initial_adapter_rejects_incomplete_or_nonlocal_input(extra, match):
    args = _args(extra)

    with pytest.raises(ValueError, match=match):
        _prepare_rl_args(args)


def test_rl_initial_adapter_is_decoupled_only(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    args = _args(("--rl-initial-adapter", str(adapter)))

    with pytest.raises(ValueError, match="only supported with --rl-sync-preset decoupled"):
        _prepare_rl_args(args)


def test_rl_initial_adapter_is_rejected_by_sft(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    args = parse_args(
        [
            "--gpu",
            "aws:1xa100",
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--rl-sync-preset",
            "decoupled",
            "--rl-initial-adapter",
            str(adapter),
        ]
    )

    with pytest.raises(ValueError, match="requires --training-mode rl"):
        _prepare_rl_args(args)


@pytest.mark.parametrize(
    "extra,match",
    [
        (("--fragments", "1"), "at least 2 fragments"),
        (("--fragments", "4", "--pipeline", "5"), "pipeline"),
        (("--fragments", "4", "--local-rl-rounds-per-sync", "1"), "at least 2"),
        (("--fragments", "4", "--fragment-pattern", "strided"), "binpack"),
        (("--fragments", "4", "--experimental-rl-sync"), "experimental"),
    ],
)
def test_decoupled_rl_rejects_values_outside_its_fixed_contract(extra, match):
    args = _args(
        (
            "--rl-sync-preset",
            "decoupled",
            "--local-rl-rounds-per-sync",
            "3",
            *extra,
        )
    )
    with pytest.raises(ValueError, match=match):
        _prepare_rl_args(args)


def test_rl_maps_long_task_and_oversampling_options_to_miles():
    args = _args(
        (
            "--over-sampling-batch-size",
            "8",
            "--custom-generate-function-path",
            "miles.rollout.generate_hub.agentic_tool_call.generate",
            "--custom-agent-function-path",
            "pkg.agent.run",
            "--agent-max-seq-len",
            "4096",
            "--seq-len",
            "4096",
            "--use-session-server",
            "--session-server-ip",
            "127.0.0.1",
            "--session-server-port",
            "31000",
            "31002",
            "--tito-model",
            "deepseekv4",
            "--tito-allowed-append-roles",
            "tool",
            "user",
            "tool",
        )
    )
    _prepare_rl_args(args)
    assert args.over_sampling_batch_size == 8
    assert args.custom_generate_function_path.endswith("agentic_tool_call.generate")
    assert args.custom_agent_function_path == "pkg.agent.run"
    assert args.agent_max_seq_len == 4096
    assert args.use_session_server
    assert args.session_server_ip == "127.0.0.1"
    assert args.session_server_port == [31000, 31002]
    assert args.tito_model == "deepseekv4"
    assert args.tito_allowed_append_roles == ["tool", "user"]


def test_rl_parses_cybergym_worker_configuration():
    args = _args(
        (
            "--cybergym-url",
            "http://10.0.0.8:8666",
            "--cybergym-agent-id",
            "benchmark-agent",
            "--cybergym-timeout",
            "90",
        )
    )

    assert args.cybergym_url == "http://10.0.0.8:8666"
    assert args.cybergym_agent_id == "benchmark-agent"
    assert args.cybergym_timeout == 90
    assert not hasattr(args, "cybergym_api_key")


def test_default_sft_parse_is_unchanged():
    args = parse_args(
        ["--gpu", "aws:1xa100", "--model", "org/model", "--data", "org/data"]
    )
    before = (args.total_steps, args.fragments, args.outer_lr, args.wire_dtype)
    _prepare_rl_args(args)
    assert args.training_mode == "sft"
    assert (args.total_steps, args.fragments, args.outer_lr, args.wire_dtype) == before


@pytest.mark.parametrize(
    "extra,match",
    [
        (("--tuning", "full"), "lora"),
        (("--lora-r", "0"), "positive LoRA rank"),
        (("--tensor-parallel", "0"), "parallelism must be positive"),
        (("--expert-parallel", "3"), "divide every island"),
        (("--local-rl-rounds-per-sync", "2"), "requires --local"),
        (("--rl-image", "docker:example/miles:latest"), "sha256"),
        (("--learner-image", "docker:example/miles:test"), "--rl-image"),
        (("--reward-function", "bad"), "package.module:function"),
        (("--over-sampling-batch-size", "3"), "at least --rollout-batch-size"),
        (("--custom-generate-function-path", "bad"), "package.module.function"),
        (
            ("--sglang-attention-backend", "dsv4"),
            "no-sglang-deterministic-inference",
        ),
        (("--custom-agent-function-path", "pkg.agent.run"), "requires --custom-generate"),
        (
            (
                "--custom-generate-function-path",
                "pkg.agent.generate",
                "--custom-agent-function-path",
                "pkg.agent.run",
            ),
            "requires --use-session-server",
        ),
        (("--agent-max-seq-len", "4096"), "requires --custom-agent"),
        (("--session-server-ip", "127.0.0.1"), "requires --use-session-server"),
        (
            ("--use-session-server", "--session-server-port", "0"),
            "positive port",
        ),
        (
            (
                "--use-session-server",
                "--session-server-port",
                "31002",
                "31000",
            ),
            "increasing range",
        ),
        (("--cybergym-timeout", "0"), "CyberGym timeout"),
    ],
)
def test_rl_validation_rejects_out_of_contract_topology(extra, match):
    args = _args(extra)
    with pytest.raises(ValueError, match=match):
        _prepare_rl_args(args)


def test_rl_validation_rejects_diffusion_models():
    args = _args(("--model-kind", "diffusion"))
    with pytest.raises(ValueError, match="only causal language models"):
        _prepare_rl_args(args)


def test_rl_sequence_capacity_covers_the_response_cap():
    args = _args(("--seq-len", "512", "--rollout-max-response-len", "4096"))
    _prepare_rl_args(args)
    assert args.seq_len == 4096


def test_rl_accepts_chat_template_kwargs():
    args = _args(
        ("--apply-chat-template-kwargs", '{"enable_thinking": false}')
    )

    _prepare_rl_args(args)

    assert args.apply_chat_template_kwargs == {"enable_thinking": False}


def test_rl_accepts_single_and_multigpu_islands():
    single = _args(("--gpu", "aws:8xa100@us-east-1"))
    _prepare_rl_args(single)
    assert single.quorum == 1
    multi = _args(("--gpu", "aws:2x4xa100@us-east-1,gcp:8xa100"))
    _prepare_rl_args(multi)
    assert multi.quorum == 2


def test_rl_accepts_two_node_deepseek_model_parallel_island():
    args = _args(
        (
            "--gpu",
            "ssh:2x8xh200@island-0",
            "--rollout-model",
            "/data/models/deepseek-v4-flash-fp8",
            "--rollout-model-revision",
            "7eb21d27aee405755da5251f4458e9fff87c047b",
            "--rl-model-recipe",
            "deepseek-v4-flash",
            "--lora-targets",
            "attention-routed-experts",
            "--tensor-parallel",
            "8",
            "--pipeline-parallel",
            "2",
            "--expert-parallel",
            "8",
            "--rollout-num-gpus-per-engine",
            "8",
            "--sglang-tp-size",
            "8",
            "--sglang-ep-size",
            "8",
            "--sglang-attention-backend",
            "dsv4",
            "--no-sglang-deterministic-inference",
            "--sglang-page-size",
            "256",
            "--use-rollout-routing-replay",
        )
    )
    _prepare_rl_args(args)

    assert args.tensor_parallel == 8
    assert args.pipeline_parallel == 2
    assert args.expert_parallel == 8
    assert args.rollout_num_gpus_per_engine == 8
    assert args.rollout_model == "/data/models/deepseek-v4-flash-fp8"
    assert args.rollout_model_revision == (
        "7eb21d27aee405755da5251f4458e9fff87c047b"
    )
    assert args.rl_model_recipe == "deepseek-v4-flash"
    assert args.sglang_attention_backend == "dsv4"
    assert args.sglang_deterministic_inference is False


def test_rl_accepts_attested_sixteen_expert_full_deepseek_recipe():
    args = _args(
        (
            "--gpu",
            "ssh:2x8xh200@island-0",
            "--rollout-model",
            "/data/models/deepseek-v4-flash-fp8",
            "--rollout-model-revision",
            "7eb21d27aee405755da5251f4458e9fff87c047b",
            "--rl-model-recipe",
            "deepseek-v4-flash",
            "--lora-targets",
            "attention",
            "--expert-full-count",
            "16",
            "--expert-full-lr",
            "1e-6",
            "--expert-selection-sha256",
            "a" * 64,
            "--expert-selection-contract-sha256",
            "b" * 64,
            "--tensor-parallel",
            "8",
            "--pipeline-parallel",
            "2",
            "--expert-parallel",
            "8",
            "--rollout-num-gpus-per-engine",
            "8",
            "--sglang-tp-size",
            "8",
            "--sglang-ep-size",
            "8",
            "--sglang-attention-backend",
            "dsv4",
            "--no-sglang-deterministic-inference",
            "--sglang-page-size",
            "256",
        )
    )

    _prepare_rl_args(args)

    assert args.expert_full_count == 16
    assert args.pipeline_parallel == 2
    assert args.expert_full_lr == 1e-6
    assert args.expert_selection_sha256 == "a" * 64
    assert args.expert_selection_contract_sha256 == "b" * 64


@pytest.mark.parametrize(
    "extra,match",
    [
        (("--expert-full-count", "33"), "between 1 and 32"),
        (
            (
                "--expert-full-count",
                "16",
                "--expert-full-lr",
                "1e-6",
                "--expert-selection-sha256",
                "a" * 64,
                "--expert-selection-contract-sha256",
                "b" * 64,
            ),
            "only supported by --rl-model-recipe deepseek-v4-flash",
        ),
        (
            (
                "--rl-model-recipe",
                "deepseek-v4-flash",
                "--lora-targets",
                "attention-routed-experts",
                "--expert-full-count",
                "16",
                "--expert-full-lr",
                "1e-6",
                "--expert-selection-sha256",
                "a" * 64,
                "--expert-selection-contract-sha256",
                "b" * 64,
            ),
            "requires --lora-targets attention",
        ),
        (
            (
                "--rl-model-recipe",
                "deepseek-v4-flash",
                "--lora-targets",
                "attention",
                "--expert-full-count",
                "16",
                "--expert-full-lr",
                "1e-6",
                "--expert-selection-sha256",
                "A" * 64,
                "--expert-selection-contract-sha256",
                "b" * 64,
            ),
            "64 lowercase hex",
        ),
        (("--expert-selection-sha256", "a" * 64), "requires --expert-full-count"),
    ],
)
def test_rl_rejects_invalid_expert_full_contract(extra, match):
    args = _args(extra)

    with pytest.raises(ValueError, match=match):
        _prepare_rl_args(args)


def test_rl_provenance_hashes_reward_inside_synced_workdir(monkeypatch):
    args = _args()
    args._provenance = {
        "model": {"source": "huggingface", "resolved_revision": "a" * 40},
        "dataset": {"source": "huggingface", "resolved_revision": "b" * 40},
    }
    reward_path = Path(__file__).resolve()
    monkeypatch.setattr("yeto.provenance.python_spec_path", lambda *a, **k: reward_path)
    monkeypatch.setattr("yeto.provenance.python_spec_sha256", lambda *a, **k: "c" * 64)
    _prepare_rl_args(args)
    assert args.reward_sha256 == "c" * 64


def test_rl_provenance_rejects_unpinned_local_dataset():
    args = _args()
    args._provenance = {
        "model": {"source": "huggingface", "resolved_revision": "a" * 40},
        "dataset": {"source": "local", "resolved_revision": None},
    }
    with pytest.raises(ValueError, match="revision-pinned Hugging Face dataset"):
        _prepare_rl_args(args)


def test_rl_provenance_allows_local_dataset_only_for_explicit_harness(
    monkeypatch,
):
    args = _args()
    args._provenance = {
        "model": {"source": "huggingface", "resolved_revision": "a" * 40},
        "dataset": {"source": "local", "resolved_revision": None},
    }
    reward_path = Path(__file__).resolve()
    monkeypatch.setattr("yeto.provenance.python_spec_path", lambda *a, **k: reward_path)
    monkeypatch.setattr(
        "yeto.provenance.python_spec_sha256", lambda *a, **k: "c" * 64
    )

    _prepare_rl_args(args, allow_local_data=True)

    assert args.reward_sha256 == "c" * 64


def test_syncer_command_uses_init_exact_base_and_equal_weight_controls():
    args = _args()
    _prepare_rl_args(args)
    command = syncer_command(args, 2, binary="syncer")
    for value in (
        "--max-base-lag 0",
        "--learner-weight equal",
        "--learners 2",
        "--quorum 2",
        "--grace-ms 0",
        "--pipeline 1",
        "--sync-interval-steps 0",
        "--delta-correction none",
        "--outer-lr 1",
        "--outer-momentum 0",
        "--checkpoint-every 1",
        "--resume",
    ):
        assert value in command
    assert "--mark-final-checkpoint" not in command
    assert "--event-tape ~/yeto-output/yeto-tape.jsonl" in command
    assert "~/yeto-output/yeto-state.ckpt" in command


def test_experimental_rl_sync_preserves_generic_sync_settings():
    args = _args(
        (
            "--experimental-rl-sync",
            "--fragments",
            "1",
            "--quorum",
            "1",
            "--grace-ms",
            "250",
            "--pipeline",
            "1",
            "--sync-interval-steps",
            "7",
            "--delta-correction",
            "heloco",
            "--outer-lr",
            "0.4",
            "--outer-momentum",
            "0.2",
            "--merge-alpha",
            "0",
            "--wire-dtype",
            "f32",
        )
    )
    _prepare_rl_args(args)
    assert (args.fragments, args.merge_alpha, args.wire_dtype) == (1, 0.0, "f32")
    command = syncer_command(args, 2, binary="syncer")
    for value in (
        "--quorum 1",
        "--grace-ms 250",
        "--pipeline 1",
        "--sync-interval-steps 7.0",
        "--delta-correction heloco",
        "--outer-lr 0.4",
        "--outer-momentum 0.2",
    ):
        assert value in command


@pytest.mark.parametrize(
    "extra,match",
    [
        (("--fragments", "2"), "--fragments 1"),
        (("--pipeline", "2"), "--pipeline 1"),
        (("--merge-alpha", "0.5"), "--merge-alpha 0"),
        (("--wire-dtype", "bf16"), "--wire-dtype f32"),
    ],
)
def test_experimental_rl_sync_rejects_unsupported_bridge_overrides(extra, match):
    values = (
        "--experimental-rl-sync",
        "--fragments",
        "1",
        "--pipeline",
        "1",
        "--merge-alpha",
        "0",
        "--wire-dtype",
        "f32",
        *extra,
    )
    with pytest.raises(ValueError, match=match):
        _prepare_rl_args(_args(values))


def test_spot_checkpoint_storage_is_stable_and_island_scoped():
    first = launcher._rl_checkpoint_storage_name("Research_Run", 0)
    assert first == launcher._rl_checkpoint_storage_name("Research_Run", 0)
    assert first != launcher._rl_checkpoint_storage_name("Research_Run", 1)
    assert len(first) <= 63

    args = _args(("--rl-completed-groups-path", "relative.pt"))
    with pytest.raises(ValueError, match="absolute or ~/"):
        _prepare_rl_args(args)


class _Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.calls = []

    def set_resources(self, resources):
        self.calls.append("resources")
        self.resources = resources

    def set_storage_mounts(self, storage_mounts):
        self.calls.append("storage")
        self.storage_mounts = storage_mounts


class _Resources:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Storage(_Resources):
    pass


class _StorageMode:
    MOUNT = "mount"


def test_miles_task_forwards_expert_full_contract_and_runtime_environment(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task,
            Resources=_Resources,
            Storage=_Storage,
            StorageMode=_StorageMode,
        ),
    )
    args = _args(
        (
            "--gpu",
            "ssh:2x8xh200@island-0",
            "--rollout-model",
            "/data/models/deepseek-v4-flash-fp8",
            "--rollout-model-revision",
            "7eb21d27aee405755da5251f4458e9fff87c047b",
            "--rl-model-recipe",
            "deepseek-v4-flash",
            "--lora-targets",
            "attention",
            "--expert-full-count",
            "16",
            "--expert-full-lr",
            "1e-6",
            "--expert-selection-sha256",
            "a" * 64,
            "--expert-selection-contract-sha256",
            "b" * 64,
            "--tensor-parallel",
            "8",
            "--expert-parallel",
            "8",
            "--rollout-num-gpus-per-engine",
            "8",
            "--sglang-tp-size",
            "8",
            "--sglang-ep-size",
            "8",
            "--sglang-attention-backend",
            "dsv4",
            "--no-sglang-deterministic-inference",
            "--sglang-page-size",
            "256",
        )
    )
    args.model_revision = "c" * 40
    args.data_revision = "d" * 40
    args.source_sha256 = "e" * 64
    args.reward_sha256 = "f" * 64
    _prepare_rl_args(args)
    from yeto.gpu_spec import parse_gpu_spec

    task = make_miles_island_task(
        args,
        parse_gpu_spec(args.gpu)[0],
        0,
        1,
        "127.0.0.1:29400",
    )

    assert "--expert-full-count 16" in task.run
    assert "--expert-full-lr 1e-06" in task.run
    assert f"--expert-selection-sha256 {'a' * 64}" in task.run
    assert f"--expert-selection-contract-sha256 {'b' * 64}" in task.run
    assert task.envs["YETO_DSV4_EXPERT_CLONE"] == "1"
    assert task.envs["YETO_DSV4_EXPERT_FULL"] == "1"
    assert task.envs["YETO_DSV4_EXPERT_FULL_COUNT"] == "16"
    assert task.envs["YETO_DSV4_EXPERT_FULL_LR"] == "1e-06"
    assert task.envs["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == "0"
    assert "YETO_DSV4_CLONE_ONLY_LORA" not in task.envs


def test_miles_tasks_mount_the_same_attested_initial_adapter(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task,
            Resources=_Resources,
            Storage=_Storage,
            StorageMode=_StorageMode,
        ),
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    args = _args(
        (
            "--rl-sync-preset",
            "decoupled",
            "--fragments",
            "2",
            "--local-rl-rounds-per-sync",
            "2",
            "--rl-initial-adapter",
            str(adapter),
        )
    )
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    from yeto.gpu_spec import parse_gpu_spec

    tasks = [
        make_miles_island_task(args, spec, learner_id, 2, "127.0.0.1:29400")
        for learner_id, spec in enumerate(parse_gpu_spec(args.gpu))
    ]

    for task in tasks:
        assert task.file_mounts["~/yeto-rl-initial-adapter"] == str(adapter)
        assert "chmod -R a-w ~/yeto-rl-initial-adapter" in task.setup
        assert "--initial-adapter ~/yeto-rl-initial-adapter" in task.run
        assert (
            f"--initial-adapter-sha256 {directory_sha256(adapter)}" in task.run
        )


def test_miles_task_checks_out_exact_commit_and_builds_multinode_ray(monkeypatch):
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret")
    monkeypatch.setenv("CYBERGYM_REWARD_SCHEME", "shaped_v1")
    monkeypatch.setenv("CYBERGYM_REWARD_VIEW", "train")
    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task,
            Resources=_Resources,
            Storage=_Storage,
            StorageMode=_StorageMode,
        ),
    )
    args = _args(
        (
            "--gpu",
            "aws:2x4xa100@us-east-1",
            "--over-sampling-batch-size",
            "6",
            "--dynamic-sampling-filter-path",
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
            "--dynamic-sampling-max-replacements",
            "8",
            "--rl-offload-train",
            "--rl-distributed-timeout-minutes",
            "7",
            "--custom-generate-function-path",
            "miles.rollout.generate_hub.agentic_tool_call.generate",
            "--custom-agent-function-path",
            "pkg.agent.run",
            "--agent-max-seq-len",
            "4096",
            "--seq-len",
            "4096",
            "--use-session-server",
            "--session-server-ip",
            "127.0.0.1",
            "--session-server-port",
            "31000",
            "--tito-model",
            "deepseekv4",
            "--tito-allowed-append-roles",
            "tool",
            "user",
            "--cybergym-url",
            "http://10.0.0.8:8666",
            "--cybergym-agent-id",
            "benchmark-agent",
            "--cybergym-timeout",
            "90",
        )
    )
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    from yeto.gpu_spec import parse_gpu_spec

    task = make_miles_island_task(
        args,
        parse_gpu_spec(args.gpu)[0],
        0,
        1,
        "127.0.0.1:29400",
    )
    assert MILES_REPOSITORY in task.setup
    assert MILES_COMMIT in task.setup
    assert (
        f"git -C ~/miles fetch --depth 1 origin {MILES_COMMIT}"
        in task.setup
    )
    assert "MILES_BUNDLE" not in task.setup
    assert "git -C ~/miles fetch ~/sky_workdir/" not in task.setup
    assert SGLANG_REPOSITORY in task.setup
    assert SGLANG_COMMIT in task.setup
    assert "checkout --detach" in task.setup
    for duplicate_check in (
        "config --get remote.origin.url",
        "rev-parse HEAD",
        "diff --quiet",
        "pathlib.Path(miles.__file__)",
        "import megatron.bridge",
    ):
        assert duplicate_check not in task.setup
    assert f"pip install -q --no-deps -e ~/miles 'peft=={MILES_PEFT_VERSION}'" in task.setup
    assert "pip install -q --no-deps -e ~/sglang/python" in task.setup
    assert task.envs["NVTE_FLASH_ATTN"] == "0"
    assert task.envs["NVTE_FUSED_ATTN"] == "0"
    assert task.envs["NVTE_UNFUSED_ATTN"] == "1"
    assert task.envs["CYBERGYM_URL"] == "http://10.0.0.8:8666"
    assert task.envs["CYBERGYM_AGENT_ID"] == "benchmark-agent"
    assert task.envs["CYBERGYM_TIMEOUT"] == "90.0"
    assert task.envs["CYBERGYM_API_KEY"] == "test-secret"
    assert task.envs["CYBERGYM_REWARD_SCHEME"] == "shaped_v1"
    assert task.envs["CYBERGYM_REWARD_VIEW"] == "train"
    assert "python3 -m yeto.rl.learner" in task.run
    assert "$HOME/sglang/python" in task.run
    assert "--initial-adapter" not in task.run
    assert "--num-learners" not in task.run
    assert "ray start --head" in task.run
    assert 'ray start --address="$MASTER_ADDR:6379"' in task.run
    assert "--actor-num-nodes 2" in task.run
    assert "--actor-num-gpus-per-node 4" in task.run
    assert "--over-sampling-batch-size 6" in task.run
    assert "--dynamic-sampling-filter-path yeto.rl.filters.bounded_nonzero_reward_std" in task.run
    assert "--dynamic-sampling-max-replacements 8" in task.run
    assert "--rl-offload-train" in task.run
    assert "--rl-distributed-timeout-minutes 7" in task.run
    assert "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate" in task.run
    assert "--custom-agent-function-path pkg.agent.run" in task.run
    assert "--agent-max-seq-len 4096" in task.run
    assert "--use-session-server" in task.run
    assert "--session-server-ip 127.0.0.1" in task.run
    assert "--session-server-port 31000" in task.run
    assert "--tito-model deepseekv4" in task.run
    assert "--tito-allowed-append-roles tool user" in task.run
    assert task.num_nodes == 2
    assert task.resources.image_id == args.rl_image
    assert task.resources.network_tier == "best"
    assert list(task.storage_mounts) == ["~/yeto-rl"]
    storage = task.storage_mounts["~/yeto-rl"]
    assert storage.persistent is False
    assert storage.sync_on_reconstruction is True
    assert storage.mode == _StorageMode.MOUNT
    assert storage.name.endswith("-rl-0")
    assert task.calls == ["resources", "storage"]

    decoupled = _args(
        (
            "--gpu",
            "aws:2x4xa100@us-east-1",
            "--rl-sync-preset",
            "decoupled",
            "--fragments",
            "4",
            "--pipeline",
            "2",
            "--local-rl-rounds-per-sync",
            "3",
        )
    )
    decoupled.model_revision = "a" * 40
    decoupled.data_revision = "b" * 40
    decoupled.source_sha256 = "c" * 64
    decoupled.reward_sha256 = "d" * 64
    _prepare_rl_args(decoupled)
    decoupled_task = make_miles_island_task(
        decoupled,
        parse_gpu_spec(decoupled.gpu)[0],
        0,
        1,
        "127.0.0.1:29400",
    )
    for value in (
        "--sync-preset decoupled",
        "--fragments 4",
        "--pipeline 2",
        "--local-horizon 3",
        "--global-rounds 3",
        "--total-fragment-steps 12",
    ):
        assert value in decoupled_task.run

    args.spot = False
    on_demand = make_miles_island_task(
        args,
        parse_gpu_spec(args.gpu)[0],
        0,
        1,
        "127.0.0.1:29400",
    )
    assert not hasattr(on_demand, "storage_mounts")


def test_miles_runtime_requires_exact_detached_clean_checkout(tmp_path, monkeypatch):
    root = tmp_path / "miles"
    root.mkdir()
    responses = {
        ("rev-parse", "HEAD"): MILES_COMMIT,
        ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
        ("config", "--get", "remote.origin.url"): MILES_REPOSITORY,
        ("status", "--porcelain", "--untracked-files=all"): "",
    }

    def run(command, **_kwargs):
        return types.SimpleNamespace(stdout=responses[tuple(command[3:])] + "\n")

    monkeypatch.setattr("yeto.rl.miles.subprocess.run", run)
    monkeypatch.setattr(
        "yeto.rl.miles.importlib.import_module",
        lambda _name: types.SimpleNamespace(__file__=root / "miles/__init__.py"),
    )
    assert verify_miles_revision(root) == root.resolve()

    responses[("rev-parse", "--abbrev-ref", "HEAD")] = "main"
    with pytest.raises(RuntimeError, match="not detached"):
        verify_miles_revision(root)
    responses[("rev-parse", "--abbrev-ref", "HEAD")] = "HEAD"
    responses[("status", "--porcelain", "--untracked-files=all")] = "?? rogue.py"
    with pytest.raises(RuntimeError, match="not clean"):
        verify_miles_revision(root)


def test_miles_argv_uses_provider_capabilities_without_model_family_branches(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.chat_template_utils",
        types.SimpleNamespace(
            resolve_reasoning_and_tool_call_parser=lambda model: (
                ("deepseek-v4", "deepseekv4")
                if model == "deepseekv4"
                else ("qwen3", "qwen25")
            )
        ),
    )
    args = argparse.Namespace(
        seq_len=128,
        groups_per_round=4,
        samples_per_group=2,
        optimizer_steps=1,
        lora_targets="auto",
        lora_r=4,
        seed=1,
        learner_id=0,
        global_rounds=3,
        inner_lr=1e-4,
        reward_function="pkg.reward:score",
        over_sampling_batch_size=6,
        rollout_max_response_len=64,
        custom_generate_function_path="pkg.agent.generate",
        custom_agent_function_path="pkg.agent.run",
        agent_max_seq_len=96,
        use_session_server=True,
        session_server_ip="127.0.0.1",
        session_server_port=[31000, 31002],
        tito_model="qwen3",
        tito_allowed_append_roles=["tool", "user"],
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        expert_parallel=None,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    provider = argparse.Namespace(
        hidden_size=16,
        num_attention_heads=4,
        num_layers=2,
        ffn_hidden_size=32,
        num_query_groups=2,
        kv_channels=4,
        seq_length=256,
        vocab_size=64,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        position_embedding_type="yarn",
        rotary_base=10000,
        gated_linear_unit=True,
        share_embeddings_and_output_weights=False,
        add_bias_linear=False,
        add_qkv_bias=True,
        qk_layernorm=True,
        num_moe_experts=8,
        moe_ffn_hidden_size=24,
        moe_router_topk=2,
        moe_layer_freq=[0, 1],
        moe_shared_expert_intermediate_size=12,
        multi_latent_attention=True,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
    )
    argv = build_miles_argv(
        args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=provider,
        target_modules=["qkv_proj", "out_proj"],
    )
    assert "--add-qkv-bias" in argv
    assert "--qk-layernorm" in argv
    assert "--group-query-attention" not in argv
    assert argv[argv.index("--target-modules") + 1] == "qkv_proj,out_proj"
    assert argv[argv.index("--max-position-embeddings") + 1] == "256"
    assert argv[argv.index("--rotary-base") + 1] == "10000"
    assert argv[argv.index("--position-embedding-type") + 1] == "rope"
    assert argv[argv.index("--rope-type") + 1] == "yarn"
    assert argv[argv.index("--num-experts") + 1] == "8"
    assert argv[argv.index("--moe-ffn-hidden-size") + 1] == "24"
    assert argv[argv.index("--moe-router-topk") + 1] == "2"
    assert argv[argv.index("--moe-layer-freq") + 1] == "[0, 1]"
    assert argv[argv.index("--moe-shared-expert-intermediate-size") + 1] == "12"
    assert argv[argv.index("--q-lora-rank") + 1] == "8"
    assert argv[argv.index("--kv-lora-rank") + 1] == "4"
    assert "--multi-latent-attention" in argv
    assert argv[argv.index("--attention-backend") + 1] == "unfused"
    assert "--sglang-attention-backend" not in argv
    assert argv[argv.index("--lr") + 1] == "0.0001"
    assert argv[argv.index("--actor-num-nodes") + 1] == "1"
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "8"
    assert argv[argv.index("--expert-model-parallel-size") + 1] == "8"
    assert argv[argv.index("--over-sampling-batch-size") + 1] == "6"
    assert "--balance-data" in argv
    assert json.loads(argv[argv.index("--apply-chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }
    assert argv[argv.index("--rollout-max-response-len") + 1] == "64"
    assert (
        argv[argv.index("--custom-generate-function-path") + 1]
        == "pkg.agent.generate"
    )
    assert argv[argv.index("--custom-agent-function-path") + 1] == "pkg.agent.run"
    assert argv[argv.index("--max-seq-len") + 1] == "96"
    assert "--apply-chat-template" not in argv
    assert "--use-session-server" in argv
    assert argv[argv.index("--session-server-ip") + 1] == "127.0.0.1"
    port = argv.index("--session-server-port")
    assert argv[port + 1 : port + 3] == ["31000", "31002"]
    assert argv[argv.index("--tito-model") + 1] == "qwen3"
    assert argv[argv.index("--sglang-reasoning-parser") + 1] == "qwen3"
    assert argv[argv.index("--sglang-tool-call-parser") + 1] == "qwen25"
    roles = argv.index("--tito-allowed-append-roles")
    assert argv[roles + 1 : roles + 3] == ["tool", "user"]
    assert (
        argv[argv.index("--rollout-all-samples-process-path") + 1]
        == "yeto.rl.miles.queue_completed_groups"
    )
    assert (
        argv[argv.index("--external-policy-sync-path") + 1]
        == "yeto.rl.miles.create_policy_sync"
    )
    assert "--custom-megatron-init-path" not in argv
    assert "--use-distributed-optimizer" not in argv
    assert "--no-offload-train" in argv
    assert argv[argv.index("--sglang-mem-fraction-static") + 1] == "0.4"
    assert "--sglang-enable-deterministic-inference" in argv
    assert argv[argv.index("--rollout-seed") + 1] == "1"
    assert "--pin-rollout-manager-to-head" in argv
    for recipe_flag in (
        "--rollout-shuffle",
        "--rollout-temperature",
        "--eps-clip",
        "--eps-clip-high",
        "--entropy-coef",
        "--kl-coef",
        "--optimizer",
        "--min-lr",
        "--lr-decay-style",
        "--lr-warmup-iters",
        "--weight-decay",
        "--adam-beta1",
        "--adam-beta2",
        "--clip-grad",
        "--micro-batch-size",
        "--attention-dropout",
        "--hidden-dropout",
    ):
        assert recipe_flag not in argv

    parallel_values = vars(args).copy()
    parallel_values.update(
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        tensor_parallel=8,
        pipeline_parallel=2,
        expert_parallel=8,
        rollout_num_gpus_per_engine=8,
        sglang_tp_size=8,
        sglang_dp_size=1,
        sglang_ep_size=8,
        sglang_mem_fraction_static=0.5,
        sglang_attention_backend="dsv4",
        sglang_deterministic_inference=False,
        sglang_page_size=256,
        sglang_max_running_requests=8,
        sglang_chunked_prefill_size=4096,
        use_rollout_routing_replay=True,
    )
    parallel_args = argparse.Namespace(**parallel_values)
    parallel_argv = build_miles_argv(
        parallel_args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=provider,
        target_modules=["qkv_proj", "out_proj"],
    )
    assert parallel_argv[parallel_argv.index("--tensor-model-parallel-size") + 1] == "8"
    assert parallel_argv[parallel_argv.index("--pipeline-model-parallel-size") + 1] == "2"
    assert parallel_argv[parallel_argv.index("--rollout-num-gpus-per-engine") + 1] == "8"
    assert parallel_argv[parallel_argv.index("--sglang-tp-size") + 1] == "8"
    assert parallel_argv[parallel_argv.index("--sglang-ep-size") + 1] == "8"
    assert parallel_argv[parallel_argv.index("--sglang-attention-backend") + 1] == "dsv4"
    assert "--sequence-parallel" in parallel_argv
    assert "--use-rollout-routing-replay" in parallel_argv
    assert "--sglang-enable-deterministic-inference" not in parallel_argv

    uneven_provider = argparse.Namespace(**vars(provider))
    uneven_provider.num_layers = 43
    uneven_provider.moe_layer_freq = [1] * 43
    recipe_values = vars(parallel_args).copy()
    recipe_values.update(
        rl_model_recipe="deepseek-v4-flash",
        pipeline_parallel=2,
        lora_targets="attention-routed-experts",
        tito_model="deepseekv4",
    )
    recipe_args = argparse.Namespace(**recipe_values)
    uneven_argv = build_miles_argv(
        recipe_args,
        model_path="/models/deepseek-v4-flash-bf16",
        rollout_model_path="/models/deepseek-v4-flash-fp8",
        prompt_path="/prompts.jsonl",
        provider=uneven_provider,
        target_modules=["qkv_proj", "out_proj"],
    )
    assert uneven_argv[uneven_argv.index("--hf-checkpoint") + 1] == (
        "/models/deepseek-v4-flash-fp8"
    )
    assert uneven_argv[uneven_argv.index("--ref-load") + 1] == (
        "/models/deepseek-v4-flash-bf16"
    )
    assert "--load" not in uneven_argv
    assert uneven_argv[uneven_argv.index("--model-name") + 1] == "deepseekv4"
    assert uneven_argv[uneven_argv.index("--attention-backend") + 1] == "flash"
    assert uneven_argv[uneven_argv.index("--qkv-format") + 1] == "bshd"
    assert uneven_argv[uneven_argv.index("--transformer-impl") + 1] == (
        "transformer_engine"
    )
    assert "--recompute-granularity" in uneven_argv
    assert "--moe-router-freeze-gate" in uneven_argv
    assert "--freeze-e-score-correction-bias" in uneven_argv
    assert uneven_argv[
        uneven_argv.index("--decoder-first-pipeline-num-layers") + 1
    ] == "22"
    assert uneven_argv[
        uneven_argv.index("--decoder-last-pipeline-num-layers") + 1
    ] == "21"
    assert uneven_argv[uneven_argv.index("--moe-layer-freq") + 1] == "1"
    assert uneven_argv[
        uneven_argv.index("--sglang-reasoning-parser") + 1
    ] == "deepseek-v4"
    assert uneven_argv[
        uneven_argv.index("--sglang-tool-call-parser") + 1
    ] == "deepseekv4"

    bad_v4_provider = argparse.Namespace(**vars(uneven_provider))
    bad_v4_provider.moe_layer_freq = [1] * 42 + [0]
    with pytest.raises(ValueError, match="every one of its 43 layers"):
        build_miles_argv(
            recipe_args,
            model_path="/models/deepseek-v4-flash-bf16",
            rollout_model_path="/models/deepseek-v4-flash-fp8",
            prompt_path="/prompts.jsonl",
            provider=bad_v4_provider,
            target_modules=["qkv_proj", "out_proj"],
        )

    clone_recipe_args = argparse.Namespace(**recipe_values)
    clone_recipe_args.lora_targets = "attention-routed-experts"
    clone_argv = build_miles_argv(
        clone_recipe_args,
        model_path="/models/deepseek-v4-flash-bf16-e288",
        rollout_model_path="/models/deepseek-v4-flash-fp8-e288",
        prompt_path="/prompts.jsonl",
        provider=uneven_provider,
        target_modules=[
            "decoder.layers.0.self_attention.linear_q_down_proj",
            "decoder.layers.0.mlp.experts.linear_fc1",
            "decoder.layers.0.mlp.experts.linear_fc2",
        ],
    )
    assert clone_argv[clone_argv.index("--lora-type") + 1] == "lora"
    assert "--no-sglang-lora-use-virtual-experts" in clone_argv
    assert "--sglang-disable-shared-experts-fusion" in clone_argv
    assert clone_argv[clone_argv.index("--sglang-moe-runner-backend") + 1] == (
        "triton"
    )

    full_recipe_args = argparse.Namespace(**recipe_values)
    full_recipe_args.lora_targets = "attention"
    full_recipe_args.expert_full_count = 16
    full_recipe_args.expert_full_lr = 1e-6
    full_argv = build_miles_argv(
        full_recipe_args,
        model_path="/models/deepseek-v4-flash-bf16-e288",
        rollout_model_path="/models/deepseek-v4-flash-fp8-e288",
        prompt_path="/prompts.jsonl",
        provider=uneven_provider,
        target_modules=["decoder.layers.0.self_attention.linear_q_down_proj"],
    )
    assert full_argv[full_argv.index("--lora-type") + 1] == "canonical_lora"
    assert "--accumulate-allreduce-grads-in-fp32" in full_argv
    assert full_argv[full_argv.index("--optimizer") + 1] == "adam"
    assert full_argv[full_argv.index("--adam-beta1") + 1] == "0.9"
    assert full_argv[full_argv.index("--adam-beta2") + 1] == "0.98"
    assert full_argv[full_argv.index("--adam-eps") + 1] == "1e-08"
    assert full_argv[full_argv.index("--weight-decay") + 1] == "0"
    assert full_argv[full_argv.index("--clip-grad") + 1] == "1.0"
    assert full_argv[full_argv.index("--kl-coef") + 1] == "0.001"

    native_argv = build_miles_argv(
        args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=provider,
        target_modules=["qkv_proj", "out_proj"],
        yeto_policy_sync=False,
    )
    assert "--external-policy-sync-path" not in native_argv
    assert "--rollout-all-samples-process-path" not in native_argv
    assert native_argv[native_argv.index("--rollout-function-path") + 1] == (
        "miles.rollout.sglang_rollout.generate_rollout"
    )
    assert not any(value.startswith("yeto.rl.") for value in native_argv)

    paired_args = argparse.Namespace(
        **vars(args),
        rollout_seed=91,
        rollout_engine_base_port=22000,
        sglang_router_port=21900,
        sglang_router_prometheus_port=21901,
        train_master_base_port=21902,
    )
    paired_argv = build_miles_argv(
        paired_args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=provider,
        target_modules=["qkv_proj", "out_proj"],
    )
    assert paired_argv[paired_argv.index("--rollout-seed") + 1] == "91"
    assert paired_argv[
        paired_argv.index("--rollout-engine-base-port") + 1
    ] == "22000"
    assert paired_argv[paired_argv.index("--sglang-router-port") + 1] == "21900"
    assert paired_argv[
        paired_argv.index("--sglang-router-prometheus-port") + 1
    ] == "21901"
    assert paired_argv[
        paired_argv.index("--train-master-base-port") + 1
    ] == "21902"

    dense_values = vars(provider).copy()
    dense_values.pop("num_moe_experts")
    dense_provider = argparse.Namespace(**dense_values)
    dense_args = argparse.Namespace(**vars(args))
    dense_args.optimizer_steps = 1
    dense_argv = build_miles_argv(
        dense_args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=dense_provider,
        target_modules=["qkv_proj", "out_proj"],
    )
    assert dense_argv[dense_argv.index("--expert-model-parallel-size") + 1] == "1"
    assert "--qkv-format" not in dense_argv

    gdn_values = vars(dense_provider).copy()
    gdn_values["experimental_attention_variant"] = "gated_delta_net"
    gdn_argv = build_miles_argv(
        dense_args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=argparse.Namespace(**gdn_values),
        target_modules=["qkv_proj", "out_proj"],
    )
    assert gdn_argv[gdn_argv.index("--qkv-format") + 1] == "bshd"

    dense_args.expert_parallel = 2
    with pytest.raises(ValueError, match="EP>1 requires a MoE"):
        build_miles_argv(
            dense_args,
            model_path="/model",
            prompt_path="/prompts.jsonl",
            provider=dense_provider,
            target_modules=["qkv_proj", "out_proj"],
        )

    too_long = argparse.Namespace(**vars(args))
    too_long.seq_len = 257
    with pytest.raises(ValueError, match="model context limit"):
        build_miles_argv(
            too_long,
            model_path="/model",
            prompt_path="/prompts.jsonl",
            provider=provider,
            target_modules=["qkv_proj", "out_proj"],
        )


def test_miles_argv_preserves_mrope_provider_configuration():
    args = argparse.Namespace(
        seq_len=128,
        groups_per_round=4,
        samples_per_group=2,
        optimizer_steps=1,
        lora_targets="attention",
        lora_r=4,
        seed=1,
        learner_id=0,
        global_rounds=1,
        inner_lr=1e-5,
        reward_function="pkg.reward:score",
        over_sampling_batch_size=4,
        rollout_max_response_len=64,
        custom_generate_function_path=None,
        use_session_server=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        expert_parallel=None,
    )
    provider = argparse.Namespace(
        hidden_size=16,
        num_attention_heads=4,
        num_layers=2,
        ffn_hidden_size=32,
        num_query_groups=2,
        kv_channels=4,
        seq_length=256,
        vocab_size=64,
        layernorm_epsilon=1e-6,
        position_embedding_type="mrope",
        rotary_base=1000000,
        rotary_percent=0.25,
        mrope_section=[11, 11, 10],
    )

    argv = build_miles_argv(
        args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=provider,
        target_modules=["q_proj"],
    )

    assert argv[argv.index("--position-embedding-type") + 1] == "mrope"
    assert argv[argv.index("--rotary-percent") + 1] == "0.25"
    section = argv.index("--mrope-section")
    assert argv[section + 1 : section + 4] == ["11", "11", "10"]


def test_megatron_targets_follow_the_exact_peft_bridge_mappings():
    qkv = types.SimpleNamespace(
        megatron_param=(
            "language_model.decoder.layers.3.self_attention.linear_qkv.weight"
        ),
        hf_param={
            "q": "model.language_model.layers.3.self_attn.q_proj.weight",
            "k": "model.language_model.layers.3.self_attn.k_proj.weight",
            "v": "model.language_model.layers.3.self_attn.v_proj.weight",
        },
    )
    projection = types.SimpleNamespace(
        megatron_param=(
            "language_model.decoder.layers.3.self_attention.linear_proj.weight"
        ),
        hf_param="model.language_model.layers.3.self_attn.o_proj.weight",
    )
    mappings = {
        value: qkv for value in qkv.hf_param.values()
    } | {projection.hf_param: projection}
    registry = types.SimpleNamespace(
        hf_to_megatron_lookup=lambda name: mappings.get(name)
    )
    bridge = types.SimpleNamespace(
        _model_bridge=types.SimpleNamespace(mapping_registry=lambda: registry)
    )
    specs = tuple(
        types.SimpleNamespace(name=f"base_model.model.{module}.lora_{side}.weight")
        for module in (
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.3.self_attn.k_proj",
            "model.language_model.layers.3.self_attn.v_proj",
            "model.language_model.layers.3.self_attn.o_proj",
        )
        for side in ("A", "B")
    )

    assert rl_learner.megatron_adapter_targets(specs, bridge) == [
        "language_model.decoder.layers.3.self_attention.linear_k",
        "language_model.decoder.layers.3.self_attention.linear_proj",
        "language_model.decoder.layers.3.self_attention.linear_q",
        "language_model.decoder.layers.3.self_attention.linear_v",
    ]


def test_megatron_targets_reject_a_peft_module_without_a_bridge_mapping():
    bridge = types.SimpleNamespace(
        _model_bridge=types.SimpleNamespace(
            mapping_registry=lambda: types.SimpleNamespace(
                hf_to_megatron_lookup=lambda _name: None
            )
        )
    )
    specs = (
        types.SimpleNamespace(
            name="base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        ),
    )

    with pytest.raises(ValueError, match="has no Megatron-Bridge mapping"):
        rl_learner.megatron_adapter_targets(specs, bridge)


def test_megatron_targets_collapse_e288_experts_to_standard_grouped_modules():
    specs = tuple(
        types.SimpleNamespace(
            name=(
                "base_model.model.model.layers.3.mlp.experts.287."
                f"{projection}.lora_{side}.weight"
            )
        )
        for projection in ("gate_proj", "up_proj", "down_proj")
        for side in ("A", "B")
    )
    bridge = types.SimpleNamespace(
        _model_bridge=types.SimpleNamespace(
            mapping_registry=lambda: types.SimpleNamespace(
                hf_to_megatron_lookup=lambda _name: None
            )
        )
    )

    assert rl_learner.megatron_adapter_targets(
        specs,
        bridge,
        standard_grouped_experts=True,
    ) == [
        "decoder.layers.3.mlp.experts.linear_fc1",
        "decoder.layers.3.mlp.experts.linear_fc2",
    ]


def test_miles_runner_keeps_native_arm_outside_yeto_policy_sync(monkeypatch):
    captured = {}

    class Provider:
        def finalize(self):
            captured["finalized"] = True

    class Bridge:
        def to_megatron_provider(self, load_weights):
            assert load_weights is False
            return Provider()

    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(path, trust_remote_code):
            assert path == "/model"
            assert trust_remote_code is True
            return Bridge()

    async def train(args):
        captured["miles_args"] = args

    def build(*args, **kwargs):
        captured["policy_sync"] = kwargs["yeto_policy_sync"]
        return ["train.py"]

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        types.SimpleNamespace(AutoBridge=AutoBridge),
    )
    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(train=train))
    monkeypatch.setattr(rl_learner, "derive_peft_lora_specs", lambda *a, **k: ())
    monkeypatch.setattr(rl_learner, "adapter_targets", lambda specs: [])
    monkeypatch.setattr(
        rl_learner,
        "megatron_adapter_targets",
        lambda specs, bridge, **_kwargs: [],
    )
    monkeypatch.setattr(rl_learner, "build_miles_argv", build)
    monkeypatch.setattr(
        rl_learner,
        "_parse_miles_args",
        lambda argv: argparse.Namespace(argv=argv),
    )
    args = argparse.Namespace(
        trust_remote_code=True,
        lora_r=4,
        lora_targets="auto",
        learner_id=0,
    )

    rl_learner.run_miles(
        args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
        yeto_policy_sync=False,
        extra_argv=("--save-debug-rollout-data", "/rollouts/{rollout_id}.pt"),
    )

    assert captured["finalized"]
    assert captured["policy_sync"] is False
    assert captured["miles_args"].argv == [
        "train.py",
        "--save-debug-rollout-data",
        "/rollouts/{rollout_id}.pt",
    ]
    assert not hasattr(captured["miles_args"], "yeto_rl_bridge_config")


def test_miles_runner_builds_the_decoupled_runtime_contract(monkeypatch, tmp_path):
    captured = {}

    class Provider:
        def finalize(self):
            pass

    class Bridge:
        def to_megatron_provider(self, load_weights):
            assert load_weights is False
            return Provider()

    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(*_args, **_kwargs):
            return Bridge()

    async def train(args):
        captured["miles_args"] = args

    specs = tuple(
        CanonicalTensorSpec(
            f"base_model.model.layer{index}.lora_A.weight",
            (1, index + 1),
            "float32",
            index + 1,
        )
        for index in range(4)
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        types.SimpleNamespace(AutoBridge=AutoBridge),
    )
    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(train=train))
    monkeypatch.setattr(rl_learner, "derive_peft_lora_specs", lambda *a, **k: specs)
    monkeypatch.setattr(
        rl_learner,
        "megatron_adapter_targets",
        lambda *_args, **_kwargs: ["layer"],
    )
    monkeypatch.setattr(rl_learner, "build_miles_argv", lambda *a, **k: ["train.py"])
    monkeypatch.setattr(
        rl_learner,
        "_parse_miles_args",
        lambda argv: argparse.Namespace(argv=argv),
    )
    args = argparse.Namespace(
        trust_remote_code=True,
        lora_r=4,
        lora_targets="all-linear",
        learner_id=0,
        model="org/model",
        data="org/data",
        model_revision="a" * 40,
        data_revision="b" * 40,
        reward_sha256="c" * 64,
        source_sha256="d" * 64,
        completed_groups_path=str(tmp_path / "island.pt"),
        event_tape=str(tmp_path / "events.jsonl"),
        syncer="127.0.0.1:29400",
        global_rounds=3,
        sync_preset="decoupled",
        fragments=2,
        pipeline=2,
        local_horizon=4,
        total_fragment_steps=6,
        learner_budget_steps=3,
        optimizer_steps=1,
        wan_streams=0,
        audit_dir=None,
        initial_adapter="/parent-adapter",
        initial_adapter_sha256="9" * 64,
    )

    rl_learner.run_miles(
        args,
        model_path="/model",
        prompt_path="/prompts.jsonl",
    )

    miles_args = captured["miles_args"]
    assert isinstance(miles_args.yeto_rl_bridge_config, DecoupledBridgeConfig)
    assert miles_args.yeto_rl_bridge_config.num_fragments == 2
    assert miles_args.yeto_rl_bridge_config.pipeline == 2
    assert miles_args.yeto_rl_bridge_config.local_horizon == 4
    assert miles_args.yeto_rl_bridge_config.total_fragment_steps == 6
    assert miles_args.yeto_rl_bridge_config.learner_budget_steps == 3
    assert miles_args.yeto_rl_sync_preset == "decoupled"
    assert miles_args.yeto_rl_pipeline == 2
    assert miles_args.external_policy_sync_run_until_stop is True
    assert len(miles_args.yeto_rl_sync_layout_fingerprint) == 64
    assert miles_args.yeto_rl_initial_adapter == "/parent-adapter"
    assert miles_args.yeto_rl_initial_adapter_sha256 == "9" * 64


def test_miles_runner_builds_attested_attention_lora_expert_full_policy(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class Provider:
        def finalize(self):
            pass

    class Bridge:
        def to_megatron_provider(self, load_weights):
            assert load_weights is False
            return Provider()

    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(*_args, **_kwargs):
            return Bridge()

    async def train(args):
        captured["miles_args"] = args

    attention_specs = (
        CanonicalTensorSpec(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            (8, 16),
            "float32",
            128,
        ),
    )
    expert_specs = (
        CanonicalTensorSpec(
            "base_model.model.model.layers.0.mlp.experts.256.gate_proj.weight",
            (8, 16),
            "float32",
            128,
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        types.SimpleNamespace(AutoBridge=AutoBridge),
    )
    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(train=train))
    monkeypatch.setattr(
        rl_learner,
        "derive_peft_lora_specs",
        lambda *a, **k: attention_specs,
    )

    def build_expert_specs(config, **kwargs):
        captured["expert_config"] = config
        captured["expert_kwargs"] = kwargs
        return expert_specs

    monkeypatch.setattr(
        "yeto.rl.deepseek_v4_bridge.ensure_deepseek_v4_bridge",
        lambda: None,
    )
    monkeypatch.setattr(rl_learner, "expert_full_specs", build_expert_specs)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(marker="config"),
    )
    monkeypatch.setattr(rl_learner, "adapter_targets", lambda specs: ["q_proj"])
    monkeypatch.setattr(
        rl_learner,
        "megatron_adapter_targets",
        lambda specs, *_args, **_kwargs: ["linear_q"],
    )
    monkeypatch.setattr(rl_learner, "build_miles_argv", lambda *a, **k: ["train.py"])
    monkeypatch.setattr(
        rl_learner,
        "_parse_miles_args",
        lambda argv: argparse.Namespace(argv=argv),
    )
    args = argparse.Namespace(
        trust_remote_code=True,
        rl_model_recipe="deepseek-v4-flash",
        lora_r=8,
        lora_targets="attention",
        expert_full_count=16,
        expert_full_lr=1e-6,
        expert_selection_sha256="a" * 64,
        expert_selection_contract_sha256="b" * 64,
        learner_id=0,
        model="/model",
        data="/data",
        model_revision="c" * 40,
        rollout_model_revision="d" * 40,
        data_revision=None,
        reward_sha256="e" * 64,
        completed_groups_path=str(tmp_path / "island.pt"),
        event_tape=str(tmp_path / "events.jsonl"),
        syncer="127.0.0.1:29400",
        global_rounds=1,
        sync_preset="strict-avg",
        groups_per_round=1,
        samples_per_group=2,
        optimizer_steps=1,
        wan_streams=0,
        audit_dir=str(tmp_path / "audit"),
    )

    rl_learner.run_miles(
        args,
        model_path="/model",
        rollout_model_path="/rollout",
        prompt_path="/prompts.jsonl",
    )

    miles_args = captured["miles_args"]
    assert captured["expert_config"].marker == "config"
    assert captured["expert_kwargs"] == {
        "expert_count": 16,
        "expected_selection_sha256": "a" * 64,
        "expected_selection_contract_sha256": "b" * 64,
    }
    assert miles_args.yeto_rl_expected_specs == tuple(
        sorted(attention_specs + expert_specs)
    )
    for name in (
        "yeto_rl_expert_full_count",
        "yeto_rl_expert_full_lr",
        "yeto_rl_expert_selection_sha256",
        "yeto_rl_expert_selection_contract_sha256",
    ):
        assert not hasattr(miles_args, name)
    assert os.environ["YETO_DSV4_EXPERT_FULL"] == "1"
    assert os.environ["YETO_DSV4_EXPERT_FULL_COUNT"] == "16"
    assert os.environ["YETO_DSV4_EXPERT_FULL_LR"] == "1e-06"
    assert os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == "0"


def test_decoupled_runner_rejects_multiple_optimizer_steps_per_rollout(tmp_path):
    with pytest.raises(ValueError, match="one optimizer step"):
        rl_learner.run_miles(
            argparse.Namespace(sync_preset="decoupled", optimizer_steps=2),
            model_path=tmp_path,
            prompt_path=tmp_path / "prompts.jsonl",
        )


class _Status:
    def __init__(self, value):
        self.value = value

    def is_terminal(self):
        return self.value != "RUNNING"

    def __str__(self):
        return self.value


class _Ops:
    def job_status(self, cluster, job_id):
        return _Status("RUNNING" if cluster == "syncer" else "FAILED")

    def cluster_up(self, cluster):
        return True

    def relaunch(self, task, cluster):
        return None

    def down(self, cluster):
        pass

    def now(self):
        return 0.0

    def sleep(self, seconds):
        pass


def test_fixed_roster_controller_fails_instead_of_shrinking():
    controller = FleetController(
        learners={"learner-0": (object(), 1), "learner-1": (object(), 2)},
        syncer=("syncer", object(), 3),
        sky_ops=_Ops(),
        poll_interval=0,
        recover_timeout=0,
        fixed_roster=True,
    )
    with pytest.raises(RuntimeError, match="fixed-roster learner"):
        controller.run()


def test_syncer_loss_does_not_restart_fixed_roster_learners():
    downed = []
    restarted = []
    ops = _Ops()
    ops.down = downed.append
    controller = FleetController(
        learners={"learner-0": (object(), 1), "learner-1": (object(), 2)},
        syncer=None,
        sky_ops=ops,
        poll_interval=0,
        recover_timeout=10,
        syncer_probe=lambda: "syncer exited",
        syncer_restart=lambda: restarted.append(True),
        fixed_roster=True,
    )

    controller._poll_local_syncer()

    assert downed == []
    assert restarted == [True]
    assert all(
        record["state"] == "running"
        for record in controller.learners.values()
    )


def test_confirmed_local_syncer_strict_failure_is_not_restarted():
    restarted = []
    controller = FleetController(
        learners={"learner-0": (object(), 1)},
        syncer=None,
        sky_ops=_Ops(),
        poll_interval=0,
        recover_timeout=10,
        syncer_probe=lambda: launcher._RlStrictFailure(
            "rejected_stale_updates: stale base"
        ),
        syncer_restart=lambda: restarted.append(True),
        fixed_roster=True,
    )

    with pytest.raises(RuntimeError, match="strict RL syncer failed"):
        controller._poll_local_syncer()
    assert restarted == []


def test_confirmed_remote_strict_failure_is_not_relaunched():
    class StrictOps(_Ops):
        def __init__(self):
            self.relaunched = []

        def rl_strict_failure(self, cluster, job_id):
            return "[yeto-rl-strict-failure] mixed_version_group_count"

        def relaunch(self, task, cluster):
            self.relaunched.append(cluster)
            return 9

    ops = StrictOps()
    controller = FleetController(
        learners={"learner-0": (object(), 1)},
        syncer=("syncer", object(), 2),
        sky_ops=ops,
        poll_interval=0,
        recover_timeout=10,
        fixed_roster=True,
    )

    with pytest.raises(RuntimeError, match="strict RL job learner-0 failed"):
        controller._poll(controller.learners["learner-0"], is_syncer=False)
    assert ops.relaunched == []
