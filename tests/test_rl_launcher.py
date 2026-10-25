from __future__ import annotations

import argparse
import hashlib
import os
import re
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
    MILES_NEXT_COMMIT,
    MILES_NEXT_REPOSITORY,
    MILES_PEFT_VERSION,
    SGLANG_NEXT_COMMIT,
    SGLANG_NEXT_REPOSITORY,
)
from yeto.rl.adapters.miles import island_entry as rl_learner
from yeto.rl.codex_backend import (
    QWEN35_MODEL,
    QWEN35_REVISION,
)
from yeto.rl.adapters.miles.revision import miles_execution_source_sha256, verify_miles_revision


# DeepSeek V4 expert-full rides the full-parameter strict sweep (P0 1.7b.2);
# its accepted launch is in test_rl_launcher_dense_full.py.
_FULL_PARAMETER_SWEEP = (
    "--gpu", "modal:16xh200,modal:16xh200",
    "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
    "--rl-rollout-gpus", "8", "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4",
)


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
            "--rl-sync-mode",  # these tests pin the strict control
            "strict",
            *extra,
        ]
    )


def test_default_rl_engine_is_ports():
    args = parse_args(
        [
            "--gpu",
            "aws:1xa100@us-east-1",
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--training-mode",
            "rl",
            "--reward-function",
            "pkg.reward:score",
        ]
    )
    assert args.rl_engine == "ports"


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


def test_over_sampling_without_filter_is_forwarded_for_cutoff():
    """agentic-rollout-utilization 1.1: outside SecRLEnv an explicit over-sampling
    count passes straight to the engine (cut-off at the batch, no filter needed)."""
    args = _args(["--over-sampling-batch-size", "16"])
    _prepare_rl_args(args)
    assert args.over_sampling_batch_size == 16 > args.rollout_batch_size
    assert getattr(args, "dynamic_sampling_filter_path", None) is None


def test_over_sampling_below_the_batch_is_rejected():
    args = _args(["--over-sampling-batch-size", "1"])
    with pytest.raises(ValueError, match="at least --rollout-batch-size"):
        _prepare_rl_args(args)


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
    with pytest.raises(ValueError, match="causal LMs only"):
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


def test_deepseek_recipe_island_env_table(monkeypatch):
    """fix-sky-island-tp-env 4.1: the deepseek-v4-flash island task keeps
    CUDA_DEVICE_MAX_CONNECTIONS=1, carries no NVTE_*_ATTN pin (Megatron sets
    them from --attention-backend), and adds the recipe's SGLang variables."""
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
            "aws:2x8xh100@us-east-1",
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
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        assert name not in task.envs
    assert task.envs["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert task.envs["SGLANG_DSV4_FP4_EXPERTS"] == "0"
    assert task.envs["SGLANG_SKIP_CHECKPOINT_LOAD_CHECK"] == "1"
    assert "NVTE_GROUPED_LINEAR_SINGLE_PARAM" not in task.envs


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
    args = _args(_FULL_PARAMETER_SWEEP + extra)

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


@pytest.mark.parametrize(("period", "flag"), [
    (None, None),
    ("1", None),  # N = 1 is plain Nesterov: no flag, unchanged profile hash
    ("4", "--outer-delayed-nesterov-period 4"),
])
def test_outer_delayed_nesterov_period_is_forwarded_only_when_set(period, flag):
    extra = ("--experimental-rl-sync", "--fragments", "1", "--pipeline", "1", "--merge-alpha", "0",
             "--wire-dtype", "f32")
    if period is not None:
        extra += ("--outer-delayed-nesterov-period", period)
    args = _args(extra)
    _prepare_rl_args(args)
    command = syncer_command(args, 2, binary="syncer")
    assert ("--outer-delayed-nesterov-period" in command) == (flag is not None)
    if flag is not None:
        assert flag in command


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


def test_island_ray_object_store_and_spill_dir_reach_ray_start():
    import json
    import shlex

    from yeto.gpu_spec import parse_gpu_spec

    args = _args(("--gpu", "aws:2x4xa100@us-east-1",
                  "--rl-island-ray-object-store-gib", "200",
                  "--rl-island-ray-spill-dir", "/mnt/big disk/spill"))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    run = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400").run
    _assert_island_ray_is_private(run)
    store = f"--object-store-memory={200 << 30}"
    head = run[run.index("ray start --head"):].split("\n", 1)[0]
    worker = run[run.index('ray start --address="$MASTER_ADDR:6379"'):].split("&&", 1)[0]
    assert store in head and store in worker
    # The spill dir is a cluster setting: head only, created before Ray starts.
    (config,) = [word.split("=", 1)[1] for word in shlex.split(head)
                 if word.startswith("--system-config=")]
    assert "--system-config" not in worker
    spilling = json.loads(json.loads(config)["object_spilling_config"])
    assert spilling == {"type": "filesystem", "params": {"directory_path": "/mnt/big disk/spill"}}
    assert run.index("mkdir -p '/mnt/big disk/spill'") < run.index("ray start --head")

    plain = _args(("--gpu", "aws:2x4xa100@us-east-1"))
    plain.model_revision, plain.data_revision = "a" * 40, "b" * 40
    plain.source_sha256, plain.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(plain)
    run = make_miles_island_task(plain, parse_gpu_spec(plain.gpu)[0], 0, 1, "127.0.0.1:29400").run
    assert "--object-store-memory" not in run and "--system-config" not in run


def _island_run(extra):
    from yeto.gpu_spec import parse_gpu_spec

    args = _args(("--gpu", "aws:2x4xa100@us-east-1", *extra))
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    run = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400").run
    start = run.index('if [ "$SKYPILOT_NODE_RANK" = "0" ]; then\n') + len('if [ "$SKYPILOT_NODE_RANK" = "0" ]; then\n')
    head = run[start:].split("\n", 1)[0]
    worker = run[run.index('ray start --address="$MASTER_ADDR:6379"'):].split("&&", 1)[0]
    return run, head, worker


def test_island_ray_head_store_and_preallocation_reach_only_the_head():
    """S27 RAY-SEGV R1: head-only store size and RAY_* prefix on the head's ray start."""
    run, head, worker = _island_run(("--rl-island-ray-object-store-gib", "1000",
                                     "--rl-island-ray-head-object-store-gib", "700",
                                     "--rl-island-ray-head-preallocate",
                                     "--rl-island-ray-head-start-wait-s", "1800"))
    assert head.startswith("  RAY_preallocate_plasma_memory=1 RAY_raylet_start_wait_time_s=1800 "
                           "RAY_raylet_client_num_connect_attempts=1800 ray start --head ")
    assert f"--object-store-memory={700 << 30}" in head and f"--object-store-memory={1000 << 30}" not in head
    assert f"--object-store-memory={1000 << 30}" in worker
    assert "RAY_preallocate" not in worker and "RAY_raylet_" not in worker
    assert run.count("RAY_preallocate_plasma_memory") == 1

    # preallocation without a wait gets Ray's 120 s for both waits
    _, head, _ = _island_run(("--rl-island-ray-head-object-store-gib", "64", "--rl-island-ray-head-preallocate"))
    assert head.startswith("  RAY_preallocate_plasma_memory=1 RAY_raylet_start_wait_time_s=120 "
                           "RAY_raylet_client_num_connect_attempts=120 ray start --head ")
    # a head store size alone: no RAY_ prefix; workers keep Ray's default size
    _, head, worker = _island_run(("--rl-island-ray-head-object-store-gib", "64"))
    assert head.startswith("  ray start --head ") and f"--object-store-memory={64 << 30}" in head
    assert "--object-store-memory" not in worker
    # a wait alone (no preallocation)
    _, head, _ = _island_run(("--rl-island-ray-head-start-wait-s", "300"))
    assert head.startswith("  RAY_raylet_start_wait_time_s=300 RAY_raylet_client_num_connect_attempts=300 "
                           "ray start --head ")


def test_island_ray_head_flags_unset_leave_the_head_command_alone():
    run, head, _ = _island_run(())
    assert head.startswith("  ray start --head --node-ip-address=")
    assert "RAY_preallocate" not in run and "RAY_raylet_" not in run


@pytest.mark.parametrize("extra, message", [
    (("--rl-island-ray-head-preallocate",), "needs --rl-island-ray-head-object-store-gib"),
    (("--rl-island-ray-head-object-store-gib", "0"), "head-object-store-gib must be >= 1"),
    (("--rl-island-ray-object-store-gib", "64", "--rl-island-ray-head-preallocate",
      "--rl-island-ray-head-start-wait-s", "0"), "head-start-wait-s must be >= 1"),
])
def test_island_ray_head_flags_are_checked(extra, message):
    with pytest.raises(ValueError, match=message):
        _island_run(extra)


def _assert_island_ray_is_private(run: str) -> None:
    """The island's Ray must coexist with SkyPilot's runtime Ray on the node.

    A whole-machine ``ray stop`` kills SkyPilot's own Ray (port 6380), after
    which every status refresh marks the cluster INIT and the head launcher
    relaunches it forever (live-run-failures #37).  The island therefore runs
    its Ray from a private temp dir, cleans up only processes under that dir,
    and tells the learner which cluster to join explicitly instead of letting
    ``ray.init(address="auto")`` pick up SkyPilot's address file.
    """

    assert re.search(r"(^|[;\s])ray stop\b", run, re.MULTILINE) is None
    assert 'MILES_RAY_DIR="$HOME/miles-ray"' in run
    assert run.count('--temp-dir="$MILES_RAY_DIR"') == 2
    assert 'ray start --head --node-ip-address="$MASTER_ADDR"' in run
    assert 'ray start --address="$MASTER_ADDR:6379" --temp-dir="$MILES_RAY_DIR"' in run
    assert 'RAY_ADDRESS="$MASTER_ADDR:6379"' in run
    assert 'pkill -f "$MILES_RAY_DIR/"' in run
    # gcs_server/raylet sit on SIGTERM for 30 s+; the trap must escalate.
    assert 'pkill -KILL -f "$MILES_RAY_DIR/"' in run
    assert run.index('pkill -f "$MILES_RAY_DIR/"') < run.index('pkill -KILL -f "$MILES_RAY_DIR/"')
    assert "trap stop_miles_ray EXIT" in run
    # Cleanup runs before the head starts, so a stale island Ray from an
    # earlier attempt on the same node cannot block ``ray start --head``.
    assert run.index("stop_miles_ray\n") < run.index("ray start --head")


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
    assert MILES_NEXT_REPOSITORY in task.setup
    assert MILES_NEXT_COMMIT in task.setup
    assert f"git -C ~/miles fetch --depth 1 origin {MILES_NEXT_COMMIT}" in task.setup
    # An image's own Miles clone (other remote) is re-pointed at the pin
    # before fetching; the verifier rejects any other origin.
    set_url = f"git -C ~/miles remote set-url origin {MILES_NEXT_REPOSITORY}"
    assert set_url in task.setup
    assert task.setup.index(set_url) < task.setup.index("fetch --depth 1 origin")
    # Miles pins its rollout manager via Ray's state API (dashboard-served);
    # a head started with --include-dashboard=false fails at start-up.
    assert "ray start --head" in task.run and "--include-dashboard=true" in task.run
    assert "--include-dashboard=false" not in task.run
    assert SGLANG_NEXT_REPOSITORY in task.setup
    assert SGLANG_NEXT_COMMIT in task.setup
    assert "checkout --detach" in task.setup
    for duplicate_check in (
        "diff --quiet",
        "pathlib.Path(miles.__file__)",
        "import megatron.bridge",
    ):
        assert duplicate_check not in task.setup
    assert f"pip install -q --no-deps -e ~/miles 'peft=={MILES_PEFT_VERSION}'" in task.setup
    assert "pip install -q --no-deps -e ~/sglang/python" in task.setup
    # The attention backend is the learner's per-recipe choice; a pinned
    # NVTE_*_ATTN would make Megatron assert for the flash recipes.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        assert name not in task.envs
    # Megatron asserts this for TP>1; the island exports it for every run.
    assert task.envs["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert task.envs["CYBERGYM_URL"] == "http://10.0.0.8:8666"
    assert task.envs["CYBERGYM_AGENT_ID"] == "benchmark-agent"
    assert task.envs["CYBERGYM_TIMEOUT"] == "90.0"
    assert "CYBERGYM_API_KEY" not in task.envs  # secret-handling-hardening
    assert task.secrets["CYBERGYM_API_KEY"] == "test-secret"
    assert task.envs["CYBERGYM_REWARD_SCHEME"] == "shaped_v1"
    assert task.envs["CYBERGYM_REWARD_VIEW"] == "train"
    assert "python3 -m yeto.rl.adapters.miles.island_entry" in task.run
    assert "$HOME/sglang/python" in task.run
    assert "--initial-adapter" not in task.run
    assert "--num-learners" not in task.run
    assert "ray start --head" in task.run
    assert 'ray start --address="$MASTER_ADDR:6379"' in task.run
    _assert_island_ray_is_private(task.run)
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


def test_miles_runtime_accepts_only_exact_staged_source(tmp_path, monkeypatch):
    root = tmp_path / "miles-source"
    (root / "miles").mkdir(parents=True)
    (root / "miles_plugins").mkdir()
    (root / "train.py").write_text("pass\n", encoding="utf-8")
    package = root / "miles/__init__.py"
    package.write_text("VERSION = 1\n", encoding="utf-8")
    (root / "miles_plugins/__init__.py").write_text("", encoding="utf-8")
    expected = miles_execution_source_sha256(root)
    monkeypatch.setattr(
        "yeto.rl.adapters.miles.revision.importlib.import_module",
        lambda _name: types.SimpleNamespace(__file__=package),
    )
    assert (
        verify_miles_revision(root, expected_source_sha256=expected)
        == root.resolve()
    )
    package.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source mismatch"):
        verify_miles_revision(root, expected_source_sha256=expected)


def test_eval_dataset_identity_is_verified_again_inside_the_learner(tmp_path):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    expected = hashlib.sha256(prompts.read_bytes()).hexdigest()
    args = argparse.Namespace(
        data=str(prompts),
        eval_only=True,
        eval_interval=1,
        eval_data_sha256=expected,
    )

    rl_learner._verify_eval_dataset_identity(args)
    prompts.write_text('{"messages": [{"role": "user"}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        rl_learner._verify_eval_dataset_identity(args)


def _stub_full_parameter_island(monkeypatch, tmp_path, *, miles_fields=None, **overrides):
    """A dense-full ports island with Megatron, Miles parsing and the ports
    loop stubbed; ``overrides`` change the island args (e.g. the SAO preset)."""

    captured = {}

    class Provider:
        hidden_size = 16
        num_attention_heads = 4
        num_layers = 2
        ffn_hidden_size = 32
        num_query_groups = 2
        kv_channels = 4
        seq_length = 256
        vocab_size = 64
        layernorm_epsilon = 1e-6
        position_embedding_type = "rope"

        def finalize(self):
            captured["finalized"] = True

    class Bridge:
        def to_megatron_provider(self, load_weights):
            assert load_weights is False
            return Provider()

    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(*_args, **_kwargs):
            return Bridge()

    def run_ports(_args, miles_args, launch, *_rest, **_kwargs):
        captured["miles_args"] = miles_args
        captured["argv"] = list(launch.argv)
        captured["algorithm_spec_sha256"] = launch.algorithm.sha256()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full mode must not derive LoRA state")

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge",
        types.SimpleNamespace(AutoBridge=AutoBridge),
    )
    monkeypatch.setattr(rl_learner, "derive_peft_lora_specs", forbidden)
    monkeypatch.setattr(rl_learner, "adapter_targets", forbidden)
    from yeto.rl.adapters.miles import config as miles_config
    from yeto.rl.adapters.miles import state as miles_state

    def parse(launch):
        return argparse.Namespace(argv=list(launch.argv), **dict(miles_fields or {}))

    monkeypatch.setattr(miles_config, "parse_miles_args", parse)
    monkeypatch.setattr(miles_state, "require_run_plugin", lambda: None)
    monkeypatch.setattr(rl_learner, "verify_ports_algorithm", lambda *_args: None)
    monkeypatch.setattr(rl_learner, "require_ports_router_mode", lambda *_args: None)
    monkeypatch.setattr(rl_learner, "_append_ports_event", lambda *_args: None)
    monkeypatch.setattr(rl_learner, "_run_ports", run_ports)
    monkeypatch.delenv("MILES_EXPERIMENTAL_FT_TRAINER", raising=False)
    monkeypatch.delenv(
        "MILES_ALLOW_MISSING_UNQUANTIZED_WEIGHT_UPDATE_HOOKS",
        raising=False,
    )

    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
    ref_load = tmp_path / "torch-dist"
    ref_load.mkdir()
    (ref_load / "latest_checkpointed_iteration.txt").write_text(
        "release\n", encoding="utf-8"
    )
    audit = tmp_path / "audit"
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages":[]}\n', encoding="utf-8")
    args = argparse.Namespace(
        rl_engine="ports",
        rl_placement="fixed-partition",
        parameter_mode="full",
        sync_preset="full-parameter",
        model=QWEN35_MODEL,
        model_revision=QWEN35_REVISION,
        data="org/data",
        data_revision="b" * 40,
        rollout_model_revision=None,
        megatron_ref_load=str(ref_load),
        reward_sha256="c" * 64,
        source_sha256="d" * 64,
        reward_function="pkg.reward:score",
        completed_groups_path=str(tmp_path / "island.pt"),
        event_tape=str(tmp_path / "events.jsonl"),
        audit_dir=str(audit),
        trust_remote_code=True,
        learner_id=1,
        num_learners=2,
        learner_generation=2,
        syncer="127.0.0.1:29400",
        global_rounds=3,
        fragments=2,
        parameter_layout_sha256="f" * 64,
        syncer_profile_sha256="a" * 64,
        total_fragment_steps=6,
        local_horizon=1,
        optimizer_steps=1,
        groups_per_round=2,
        samples_per_group=2,
        over_sampling_batch_size=2,
        rollout_max_response_len=64,
        seq_len=128,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        tensor_parallel=2,
        pipeline_parallel=1,
        expert_parallel=None,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        sglang_tp_size=1,
        inner_lr=1e-5,
        seed=7,
        wan_streams=4,
        lora_r=None,
        lora_targets=None,
        custom_generate_function_path=None,
        custom_agent_function_path=None,
        use_session_server=True,
        session_server_ip=None,
        session_server_port=[31002, 31003],
        tito_model=None,
        tito_allowed_append_roles=None,
        rl_model_recipe="generic",
        expert_full_count=0,
        eval_only=False,
    )
    args = argparse.Namespace(**{**vars(args), **overrides})
    return captured, args, model, config, prompts, audit, ref_load


def test_miles_dense_full_parameter_argv_and_runtime_contract(
    monkeypatch,
    tmp_path,
):
    captured, args, model, config, prompts, audit, ref_load = _stub_full_parameter_island(
        monkeypatch, tmp_path
    )

    rl_learner.run_miles(
        args,
        model_path=model,
        prompt_path=prompts,
    )

    miles_args = captured["miles_args"]
    argv = miles_args.argv
    for flag in (
        "--lora-rank",
        "--lora-alpha",
        "--lora-dropout",
        "--lora-type",
        "--target-modules",
        "--lora-base-cpu-backup",
        "--sglang-max-lora-rank",
    ):
        assert flag not in argv
    assert argv[argv.index("--num-rollout") + 1] == "3"
    assert argv[argv.index("--ref-load") + 1] == str(ref_load.resolve())
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "2"
    assert argv[argv.index("--tensor-model-parallel-size") + 1] == "2"
    assert argv[argv.index("--pipeline-model-parallel-size") + 1] == "1"
    assert "--colocate" not in argv
    assert argv[argv.index("--rollout-num-gpus") + 1] == "2"
    assert argv[argv.index("--rollout-num-gpus-per-engine") + 1] == "1"
    assert argv[argv.index("--sglang-tp-size") + 1] == "1"
    assert argv[argv.index("--model-name") + 1] == "qwen3_5"
    spec = argv.index("--spec")
    assert argv[spec + 1 : spec + 3] == [
        "miles_plugins.models.qwen3_5",
        "get_qwen3_5_spec",
    ]
    assert "--apply-layernorm-1p" in argv
    assert "--attention-output-gate" in argv
    assert argv[argv.index("--attention-dropout") + 1] == "0.0"
    assert argv[argv.index("--hidden-dropout") + 1] == "0.0"
    assert argv[argv.index("--attention-backend") + 1] == "flash"
    assert argv[argv.index("--num-gpus-per-node") + 1] == "4"
    from yeto.rl.adapters.miles.full_parameter.session import CONFIG_ATTRIBUTE

    sync = getattr(miles_args, CONFIG_ATTRIBUTE)
    assert sync.sweep is True
    (dense,) = sync.roles
    assert (dense.algorithm, dense.role) == ("grpo", "actor")
    assert dense.component.model_revision == QWEN35_REVISION
    assert dense.component.config_hash == hashlib.sha256(config.read_bytes()).hexdigest()
    assert dense.syncer_addr == ("127.0.0.1", 29400)
    assert dense.learner_id == 1
    assert dense.learner_generation == 2
    assert dense.learner_generations == (0, 2)
    assert dense.total_fragment_steps == 6
    assert dense.expected_fragments == 2
    assert dense.expected_layout_hash == "f" * 64
    assert dense.syncer_profile_hash == "a" * 64
    assert (dense.local_horizon, dense.pipeline_depth) == (1, 1)
    learner_zero_argv = list(argv)
    learner_zero_argv[learner_zero_argv.index("--rollout-seed") + 1] = "7"
    session_port = learner_zero_argv.index("--session-server-port")
    learner_zero_argv[session_port + 1 : session_port + 3] = ["31000", "31001"]
    learner_zero_args = argparse.Namespace(**vars(args))
    learner_zero_args.learner_id = 0
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    assert dense.training_contract_hash == rl_learner._dense_full_training_contract_sha256(
        learner_zero_args, learner_zero_argv, model_config_sha256=config_sha256
    )
    # P3: each island may train on its own task list; the data stays out.
    other_data_args = argparse.Namespace(**vars(learner_zero_args))
    other_data_args.data, other_data_args.data_revision = "org/other", "e" * 40
    other_data_argv = list(learner_zero_argv)
    other_data_argv[other_data_argv.index("--prompt-data") + 1] = "/elsewhere/prompts.jsonl"
    assert dense.training_contract_hash == rl_learner._dense_full_training_contract_sha256(
        other_data_args, other_data_argv, model_config_sha256=config_sha256
    )
    # P3-Q3: eval runs on the eval island; an island's eval settings stay out (v6).
    eval_args = argparse.Namespace(**vars(learner_zero_args))
    eval_args.eval_interval, eval_args.eval_data = 2, "/heldout.jsonl"
    eval_argv = learner_zero_argv + [
        "--eval-prompt-data", "heldout", "/heldout.jsonl", "--eval-interval", "2",
        "--skip-eval-before-train", "--n-samples-per-eval-prompt", "4", "--log-passrate",
        "--eval-top-p", "0.9", "--eval-max-response-len", "64",
    ]
    assert dense.training_contract_hash == rl_learner._dense_full_training_contract_sha256(
        eval_args, eval_argv, model_config_sha256=config_sha256
    )
    # P4: placement flags stay in a uniform launch's contract (different GPU
    # splits refused) and leave it only with per-island placement.
    colocated_argv = list(learner_zero_argv)
    rollout = colocated_argv.index("--rollout-num-gpus")
    colocated_argv[rollout : rollout + 2] = ["--colocate"]
    colocated_argv[colocated_argv.index("--actor-num-gpus-per-node") + 1] = "4"
    colocated_argv[colocated_argv.index("--num-gpus-per-node") + 1] = "2"
    offload = colocated_argv.index("--no-offload-train")
    colocated_argv[offload] = "--offload-train"
    other_rollout_argv = list(learner_zero_argv)
    other_rollout_argv[other_rollout_argv.index("--rollout-num-gpus") + 1] = "3"
    mixed_args = argparse.Namespace(**vars(learner_zero_args), rl_per_island_placement=True)

    def contract(island_args, island_argv):
        return rl_learner._dense_full_training_contract_sha256(
            island_args, island_argv, model_config_sha256=config_sha256)

    assert contract(learner_zero_args, other_rollout_argv) != dense.training_contract_hash
    assert contract(learner_zero_args, colocated_argv) != dense.training_contract_hash
    assert contract(mixed_args, colocated_argv) == contract(mixed_args, learner_zero_argv)
    assert contract(mixed_args, learner_zero_argv) != dense.training_contract_hash
    other_engine_argv = list(learner_zero_argv)
    other_engine_argv[other_engine_argv.index("--rollout-num-gpus-per-engine") + 1] = "2"
    assert contract(mixed_args, other_engine_argv) != contract(mixed_args, learner_zero_argv)
    changed_source_args = argparse.Namespace(**vars(learner_zero_args))
    changed_source_args.source_sha256 = "f" * 64
    assert dense.training_contract_hash != rl_learner._dense_full_training_contract_sha256(
        changed_source_args, learner_zero_argv, model_config_sha256=config_sha256
    )
    assert (
        miles_args.external_policy_identity_setter_path
        == "yeto.rl.adapters.miles.publish.set_current_published_policy_identity"
    )
    evidence = Path(miles_args.yeto_rl_trajectory_evidence_dir)
    assert evidence.parent == audit
    assert evidence.is_dir() and not evidence.is_symlink()
    assert evidence.stat().st_mode & 0o077 == 0
    assert miles_args.external_policy_sync_run_until_stop is False
    assert os.environ["MILES_EXPERIMENTAL_FT_TRAINER"] == "0"
    assert "MILES_ALLOW_MISSING_UNQUANTIZED_WEIGHT_UPDATE_HOOKS" not in os.environ


def _dense_sweep_args(**overrides):
    values = dict(
        eval_only=False,
        optimizer_steps=1,
        local_horizon=1,
        global_rounds=3,
        fragments=2,
        total_fragment_steps=6,
        parameter_layout_sha256="f" * 64,
        syncer_profile_sha256="a" * 64,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"local_horizon": 2}, "requires H=1"),
        ({"total_fragment_steps": 5}, "rounds\\*fragments"),
        ({"syncer_profile_sha256": None}, "--syncer-profile-sha256"),
        ({"eval_only": True}, "eval-only"),
        # P0-S lifts this for every full-parameter preset.
        ({"rl_max_policy_age": 1}, "--rl-max-policy-age"),
    ],
)
def test_dense_full_sweep_rejections(overrides, message):
    with pytest.raises(ValueError, match=message):
        rl_learner._check_dense_full_sweep(_dense_sweep_args(**overrides), yeto_policy_sync=True)


def test_dense_full_sweep_accepts_any_generation_roster():
    # P0-D7: no generation-0 / version-0 Milestone-1 limit any more.
    rl_learner._check_dense_full_sweep(_dense_sweep_args(), yeto_policy_sync=True)


def test_dense_full_legacy_weight_hook_rejects_quantized_rollout(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3","quantization_config":{"quant_method":"fp8"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require an unquantized rollout model"):
        rl_learner._require_unquantized_full_parameter_rollout_model(model)


def test_decoupled_runner_rejects_multiple_optimizer_steps_per_rollout(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import state

    monkeypatch.setattr(state, "require_run_plugin", lambda: None)  # no Miles here
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
def test_miles_island_forwards_chat_template_kwargs(monkeypatch):
    """`--apply-chat-template-kwargs` was accepted but never reached the
    island (e.g. Qwen3 `enable_thinking: false` for short math rollouts)."""
    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode
        ),
    )
    from yeto.gpu_spec import parse_gpu_spec

    args = _args(("--apply-chat-template-kwargs", '{"enable_thinking": false}'))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    task = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")
    assert """--apply-chat-template-kwargs '{"enable_thinking":false}'""" in task.run

def test_prompt_data_uses_the_only_split_when_there_is_no_train(tmp_path, monkeypatch):
    """A Hub dataset without a `train` split (HuggingFaceH4/MATH-500 ships
    only `test`) must still load; a dataset that has `train` keeps it."""
    import datasets

    from yeto import data as yeto_data
    from yeto.rl.adapters.miles.island_entry import prepare_prompt_data

    seen = {}

    def load_rows(source, split="train", revision=None):
        seen["split"] = split
        return [{"messages": [{"role": "user", "content": "1+1?"}], "label": "2"}]

    monkeypatch.setattr(yeto_data, "load_rows", load_rows)
    monkeypatch.setattr(
        datasets, "get_dataset_split_names", lambda source, revision=None: ["test"]
    )
    prepare_prompt_data("org/only-test", "a" * 40, tmp_path / "p.jsonl")
    assert seen["split"] == "test"

    monkeypatch.setattr(
        datasets, "get_dataset_split_names", lambda source, revision=None: ["train", "test"]
    )
    prepare_prompt_data("org/other", "a" * 40, tmp_path / "q.jsonl")
    assert seen["split"] == "train"


def test_miles_island_forwards_dataset_column_flags(monkeypatch):
    """--rl-prompt-column / --rl-label-column reach the island; absent
    flags add nothing (the learner's defaults apply)."""
    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode
        ),
    )
    from yeto.gpu_spec import parse_gpu_spec

    def island(extra):
        args = _args(extra)
        args.model_revision = "a" * 40
        args.data_revision = "b" * 40
        args.source_sha256 = "c" * 64
        args.reward_sha256 = "d" * 64
        _prepare_rl_args(args)
        return make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")

    task = island(("--rl-prompt-column", "problem", "--rl-label-column", "answer"))
    assert " --rl-prompt-column problem" in task.run and " --rl-label-column answer" in task.run
    assert "--rl-prompt-column" not in island(()).run


@pytest.mark.parametrize(
    "gpu, extra, want",
    [
        ("aws:1xa10g@us-west-2", (), "8+"),
        ("nebius:1xl40s@eu-north1", (), "8+"),
        ("nebius:2x8xh100@eu-north1", ("--rollout-batch-size", "16"), "8+"),
        ("ssh:2x8xh200@island-0", ("--rollout-batch-size", "16"), None),
        ("aws:1xa10g@us-west-2", ("--learner-cpus", "16+"), "16+"),
        ("aws:1xa10g@us-west-2", ("--learner-instance-type", "g5.xlarge"), None),
    ],
)
def test_miles_island_sets_a_per_node_vcpu_floor(monkeypatch, gpu, extra, want):
    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode
        ),
    )
    args = _args(("--gpu", gpu, *extra))
    args.model_revision = "c" * 40
    args.data_revision = "d" * 40
    args.source_sha256 = "e" * 64
    args.reward_sha256 = "f" * 64
    _prepare_rl_args(args)
    from yeto.gpu_spec import parse_gpu_spec

    spec = parse_gpu_spec(args.gpu)[0]
    task = make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert task.resources.cpus == want
    assert task.resources.accelerators == spec.accelerators


def test_run_miles_carries_the_island_task_record(monkeypatch, tmp_path):
    captured, args, model, _config, prompts, _audit, _ref = _stub_full_parameter_island(
        monkeypatch, tmp_path
    )
    record = {"task_ids": ["web1"], "data_sha256": "a" * 64}
    rl_learner.run_miles(args, model_path=model, prompt_path=prompts, island_tasks=record)
    assert captured["miles_args"].yeto_rl_island_tasks == record


@pytest.mark.parametrize("flag, value", [
    ("--codex-reasoning-effort", "xhigh"),
    ("--codex-backend-profile", "qwen35_08b"),
])
def test_removed_codex_agent_flags_are_refused(flag, value):
    args = _args((flag, value, "--use-session-server", "--tito-model", "qwen35"))
    with pytest.raises(ValueError, match="signed Codex agent"):
        _prepare_rl_args(args)


def test_nebius_island_caps_the_ray_object_store_at_dev_shm(tmp_path):
    # G-DF yeto-df-6: sky's docker runtime on Nebius gives the container a 200 GB /dev/shm and Ray refused
    # a 644 GB object store at `ray start`. The size is now min(requested, 90% of /dev/shm) on the node.
    import subprocess

    from yeto.gpu_spec import parse_gpu_spec

    args = _args(("--gpu", "nebius:2x4xh100@eu-north1", "--rl-island-ray-object-store-gib", "600"))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    run = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400").run
    head = run[run.index("ray start --head"):].split("\n", 1)[0]
    worker = run[run.index('ray start --address="$MASTER_ADDR:6379"'):].split("&&", 1)[0]
    flag = f'--object-store-memory="$(r={600 << 30};'
    assert flag in head and flag in worker
    # evaluate the flag with a fake df: /dev/shm of 200e9 bytes -> 90% of it; a large /dev/shm keeps 600 GiB
    word = head[head.index('--object-store-memory="'):]
    word = word[:word.index('fi)"') + len('fi)"')]
    fake = tmp_path / "df"
    for shm, want in ((200_000_000_000, 180_000_000_000), (2_000_000_000_000, 600 << 30)):
        fake.write_text(f"#!/bin/sh\necho '     1B-blocks'\necho ' {shm}'\n")
        fake.chmod(0o755)
        out = subprocess.run(["bash", "-c", f"echo {word}"], capture_output=True, text=True,
                             env={"PATH": f"{tmp_path}:/usr/bin:/bin"}, check=True)
        assert out.stdout.strip() == f"--object-store-memory={want}"
    # other clouds keep the literal size
    aws = _args(("--gpu", "aws:2x4xa100@us-east-1", "--rl-island-ray-object-store-gib", "600"))
    aws.model_revision, aws.data_revision = "a" * 40, "b" * 40
    aws.source_sha256, aws.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(aws)
    run = make_miles_island_task(aws, parse_gpu_spec(aws.gpu)[0], 0, 1, "127.0.0.1:29400").run
    assert f"--object-store-memory={600 << 30}" in run and "/dev/shm" not in run
