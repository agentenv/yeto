from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from yeto import launcher
from yeto.cli import parse_args
from yeto.launcher import (
    FleetController,
    _prepare_rl_args,
    make_miles_island_task,
    syncer_command,
)
from yeto.rl import MILES_COMMIT, MILES_PEFT_VERSION, MILES_REPOSITORY
from yeto.rl import learner as rl_learner
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


def test_rl_maps_long_task_and_oversampling_options_to_miles():
    args = _args(
        (
            "--over-sampling-batch-size",
            "8",
            "--custom-generate-function-path",
            "pkg.agent.generate",
            "--use-session-server",
            "--session-server-ip",
            "127.0.0.1",
            "--session-server-port",
            "31000",
            "31002",
            "--tito-model",
            "qwen3",
        )
    )
    _prepare_rl_args(args)
    assert args.over_sampling_batch_size == 8
    assert args.custom_generate_function_path == "pkg.agent.generate"
    assert args.use_session_server
    assert args.session_server_ip == "127.0.0.1"
    assert args.session_server_port == [31000, 31002]
    assert args.tito_model == "qwen3"


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
        (("--tensor-parallel", "2"), "TP=PP=1"),
        (("--expert-parallel", "3"), "divide every island"),
        (("--local-rl-rounds-per-sync", "2"), "requires --local"),
        (("--rl-image", "docker:example/miles:latest"), "sha256"),
        (("--learner-image", "docker:example/miles:test"), "--rl-image"),
        (("--reward-function", "bad"), "package.module:function"),
        (("--over-sampling-batch-size", "3"), "at least --rollout-batch-size"),
        (("--custom-generate-function-path", "bad"), "package.module.function"),
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


def test_rl_accepts_single_and_multigpu_islands():
    single = _args(("--gpu", "aws:8xa100@us-east-1"))
    _prepare_rl_args(single)
    assert single.quorum == 1
    multi = _args(("--gpu", "aws:2x4xa100@us-east-1,gcp:8xa100"))
    _prepare_rl_args(multi)
    assert multi.quorum == 2


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


def test_miles_task_checks_out_exact_commit_and_builds_multinode_ray(monkeypatch):
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
            "--custom-generate-function-path",
            "pkg.agent.generate",
            "--use-session-server",
            "--session-server-ip",
            "127.0.0.1",
            "--session-server-port",
            "31000",
            "--tito-model",
            "qwen3",
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
    assert "checkout --detach" in task.setup
    for duplicate_check in (
        "config --get remote.origin.url",
        "rev-parse HEAD",
        "diff --quiet",
        "pathlib.Path(miles.__file__)",
        "import megatron.bridge",
    ):
        assert duplicate_check not in task.setup
    assert (
        f"pip install -q --no-deps -e ~/miles 'peft=={MILES_PEFT_VERSION}'\n(nohup"
        in task.setup
    )
    assert task.envs["NVTE_FLASH_ATTN"] == "0"
    assert task.envs["NVTE_FUSED_ATTN"] == "0"
    assert task.envs["NVTE_UNFUSED_ATTN"] == "1"
    assert "python3 -m yeto.rl.learner" in task.run
    assert "--num-learners" not in task.run
    assert "ray start --head" in task.run
    assert 'ray start --address="$MASTER_ADDR:6379"' in task.run
    assert "--actor-num-nodes 2" in task.run
    assert "--actor-num-gpus-per-node 4" in task.run
    assert "--over-sampling-batch-size 6" in task.run
    assert "--custom-generate-function-path pkg.agent.generate" in task.run
    assert "--use-session-server" in task.run
    assert "--session-server-ip 127.0.0.1" in task.run
    assert "--session-server-port 31000" in task.run
    assert "--tito-model qwen3" in task.run
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


def test_miles_argv_uses_provider_capabilities_without_model_family_branches():
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
        use_session_server=True,
        session_server_ip="127.0.0.1",
        session_server_port=[31000, 31002],
        tito_model="qwen3",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
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
    assert "--group-query-attention" in argv
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
    assert argv[argv.index("--rollout-max-response-len") + 1] == "64"
    assert (
        argv[argv.index("--custom-generate-function-path") + 1]
        == "pkg.agent.generate"
    )
    assert "--use-session-server" in argv
    assert argv[argv.index("--session-server-ip") + 1] == "127.0.0.1"
    port = argv.index("--session-server-port")
    assert argv[port + 1 : port + 3] == ["31000", "31002"]
    assert argv[argv.index("--tito-model") + 1] == "qwen3"
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
        lambda specs, bridge: [],
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
