import copy
import hashlib
import json
import struct
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.core import canonical_state, tensors_from_flat
from yeto.rl.export import (
    derive_peft_lora_specs,
    export_rl_checkpoint,
    specs_manifest,
)
from yeto.rl.learner import (
    _validate_miles_args,
    build_miles_argv,
    prepare_prompt_data,
)
from yeto.rl.manifest import (
    MILES_COMMIT,
    MILES_REPOSITORY,
    build_run_manifest,
    canonical_json,
    manifest_sha256,
    path_tree_sha256,
    validate_manifest,
)
from yeto.rl.miles import verify_miles_revision


def manifest_args(canonical_layout, *, model="org/model", data="org/data"):
    return SimpleNamespace(
        _provenance={
            "model": {
                "resolved_identifier": model,
                "resolved_revision": "a" * 40,
            },
            "dataset": {
                "resolved_identifier": data,
                "resolved_revision": "b" * 40,
            },
        },
        model=model,
        model_revision="a" * 40,
        data=data,
        data_revision="b" * 40,
        trust_remote_code=True,
        lora_r=2,
        lora_targets="attention",
        rl_groups_per_island_round=3,
        rl_samples_per_group=2,
        rl_local_optimizer_steps=2,
        rl_global_rounds=2,
        inner_lr=1e-5,
        seq_len=32,
        seed=7,
        reward_function="package.reward:score",
        source_sha256="c" * 64,
        learner_image="docker:radixark/miles@sha256:" + "d" * 64,
        cluster_prefix="rl-test",
        rl_canonical_layout=canonical_layout,
    )


def test_manifest_is_canonical_and_binds_the_pinned_miles_and_layout():
    layout = specs_manifest(
        canonical_state(
            0,
            {
                "base_model.model.layer.q_proj.lora_A.weight": torch.zeros(2, 4),
                "base_model.model.layer.q_proj.lora_B.weight": torch.zeros(4, 2),
            },
        ).specs
    )
    manifest = build_run_manifest(
        manifest_args(layout), learners=2, reward_sha256="e" * 64
    )
    text = canonical_json(manifest)

    assert validate_manifest(text, manifest_sha256(text)) == manifest
    assert manifest["miles"] == {
        "repository": MILES_REPOSITORY,
        "commit": MILES_COMMIT,
    }

    tampered = copy.deepcopy(manifest)
    tampered["canonical_lora"]["tensors"][0]["shape"][0] += 1
    with pytest.raises(ValueError, match="canonical LoRA"):
        validate_manifest(canonical_json(tampered))

    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"not_finite": float("nan")})


def test_local_dataset_hash_and_prompt_normalization_are_stable(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "b.jsonl").write_text("b\n", encoding="utf-8")
    (dataset / "a.jsonl").write_text("a\n", encoding="utf-8")
    first = path_tree_sha256(dataset)
    assert first == path_tree_sha256(dataset)
    (dataset / "a.jsonl").write_text("changed\n", encoding="utf-8")
    assert path_tree_sha256(dataset) != first

    rows = [
        {"prompt": "hello", "label": "ok", "id": 3},
        {
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    ]
    monkeypatch.setattr("yeto.data.load_rows", lambda source, revision=None: rows)
    output = prepare_prompt_data("ignored", None, tmp_path / "prompts.jsonl")
    normalized = [json.loads(line) for line in output.read_text().splitlines()]
    assert normalized[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert normalized[0]["metadata"]["id"] == 3
    assert normalized[1]["tools"] == rows[1]["tools"]


@pytest.mark.parametrize(
    "origin,changes,match",
    [
        ("https://github.com/other/miles", "", "origin mismatch"),
        (MILES_REPOSITORY, " M miles/file.py", "source changes"),
        (MILES_REPOSITORY, "?? scripts/override.py", "source changes"),
    ],
)
def test_miles_checkout_verification_fails_closed(
    tmp_path, monkeypatch, origin, changes, match
):
    root = tmp_path / "miles"
    package = root / "miles"
    package.mkdir(parents=True)
    module = SimpleNamespace(__file__=str(package / "__init__.py"))

    def fake_run(command, **kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            output = MILES_COMMIT
        elif command[-3:] == ["config", "--get", "remote.origin.url"]:
            output = origin
        else:
            output = changes
        return SimpleNamespace(stdout=output + ("\n" if output else ""))

    monkeypatch.setattr("yeto.rl.miles.subprocess.run", fake_run)
    monkeypatch.setattr("yeto.rl.miles.importlib.import_module", lambda name: module)
    with pytest.raises(RuntimeError, match=match):
        verify_miles_revision(root)


def test_miles_checkout_verification_accepts_only_the_pinned_import(tmp_path, monkeypatch):
    root = tmp_path / "miles"
    package = root / "miles"
    package.mkdir(parents=True)
    outputs = iter((MILES_COMMIT, MILES_REPOSITORY + ".git", ""))
    monkeypatch.setattr(
        "yeto.rl.miles.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(stdout=next(outputs) + "\n"),
    )
    monkeypatch.setattr(
        "yeto.rl.miles.importlib.import_module",
        lambda name: SimpleNamespace(__file__=str(package / "__init__.py")),
    )
    assert verify_miles_revision(root) == root.resolve()


def test_miles_argv_and_resolved_contract_are_strict(tmp_path):
    requested = SimpleNamespace(
        seq_len=32,
        groups_per_round=3,
        samples_per_group=2,
        optimizer_steps=2,
        global_rounds=2,
        cache_dir=str(tmp_path),
        lora_r=2,
        lora_targets="attention",
        inner_lr=1e-5,
        seed=7,
        learner_id=1,
        generate_function="package.generate:trajectory",
    )
    manifest = {
        "lora": {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "generation": {
            "custom_generate": {
                "callable": requested.generate_function,
                "source_sha256": "f" * 64,
            }
        },
    }
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        vocab_size=32,
        model_type="llama",
        hidden_act="silu",
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    argv = build_miles_argv(
        requested,
        manifest,
        model_path=tmp_path / "model",
        prompt_path=tmp_path / "prompts.jsonl",
        config=config,
    )
    for flag in (
        "--bf16",
        "--use-distributed-optimizer",
        "--colocate",
        "--lora-base-cpu-backup",
        "--no-load-optim",
        "--no-save-optim",
        "--swiglu",
        "--disable-bias-linear",
    ):
        assert flag in argv
    assert argv[argv.index("--global-batch-size") + 1] == "3"
    assert argv[argv.index("--advantage-estimator") + 1] == "grpo"
    assert argv[argv.index("--tensor-model-parallel-size") + 1] == "1"
    assert argv[argv.index("--rotary-base") + 1] == "10000"
    assert argv[argv.index("--custom-generate-function-path") + 1] == (
        "package.generate.trajectory"
    )

    def model_argv(candidate):
        return build_miles_argv(
            requested,
            manifest,
            model_path=tmp_path / "model",
            prompt_path=tmp_path / "prompts.jsonl",
            config=candidate,
        )

    qwen2 = copy.copy(config)
    qwen2.model_type = "qwen2"
    assert "--add-qkv-bias" in model_argv(qwen2)

    qwen3 = copy.copy(config)
    qwen3.model_type = "qwen3"
    assert "--qk-layernorm" in model_argv(qwen3)

    nested_rope = copy.copy(config)
    del nested_rope.rope_theta
    nested_rope.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    nested_argv = model_argv(nested_rope)
    assert nested_argv[nested_argv.index("--rotary-base") + 1] == "500000"

    llama31 = copy.copy(config)
    llama31.rope_scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    llama31_argv = model_argv(llama31)
    assert llama31_argv[llama31_argv.index("--rope-scaling-factor") + 1] == "8.0"

    llama32 = copy.copy(llama31)
    llama32.rope_scaling = {**llama31.rope_scaling, "factor": 32.0}
    llama32_argv = model_argv(llama32)
    assert llama32_argv[llama32_argv.index("--rope-scaling-factor") + 1] == "32.0"

    unsupported_rope = copy.copy(llama31)
    unsupported_rope.rope_scaling = {**llama31.rope_scaling, "low_freq_factor": 2.0}
    with pytest.raises(ValueError, match="Llama 3 RoPE"):
        model_argv(unsupported_rope)

    sliding = copy.copy(config)
    sliding.use_sliding_window = True
    with pytest.raises(ValueError, match="sliding-window"):
        model_argv(sliding)

    resolved = SimpleNamespace(
        train_backend="megatron",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        data_parallel_size=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        lora_type="canonical_lora",
        lora_rank=2,
        lora_alpha=2,
        lora_dropout=0.0,
        target_modules=manifest["lora"]["target_modules"],
        lora_base_cpu_backup=True,
        bf16=True,
        use_distributed_optimizer=True,
        use_critic=False,
        advantage_estimator="grpo",
        compute_advantages_and_returns=True,
        rewards_normalization=True,
        grpo_std_normalization=True,
        normalize_advantages=False,
        loss_type="policy_loss",
        eps_clip=0.2,
        eps_clip_high=0.28,
        entropy_coef=0.0,
        use_kl_loss=False,
        kl_coef=0.0,
        kl_loss_coef=0.0,
        use_tis=False,
        use_rollout_logprobs=False,
        partial_rollout=False,
        dynamic_sampling_filter_path=None,
        use_dynamic_batch_size=False,
        balance_data=False,
        rollout_batch_size=3,
        n_samples_per_prompt=2,
        num_steps_per_rollout=2,
        global_batch_size=3,
        micro_batch_size=1,
        num_rollout=2,
        over_sampling_batch_size=3,
        rollout_shuffle=True,
        rollout_seed=8,
        rollout_max_context_len=32,
        rollout_max_prompt_len=16,
        rollout_max_response_len=16,
        rollout_temperature=1.0,
        custom_rm_path="yeto.rl.reward.miles_reward",
        custom_generate_function_path="package.generate.trajectory",
        input_key="messages",
        label_key="label",
        metadata_key="metadata",
        tool_key="tools",
        apply_chat_template=True,
        optimizer="adam",
        lr=1e-5,
        min_lr=1e-5,
        lr_decay_style="constant",
        lr_warmup_iters=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        clip_grad=1.0,
        seed=7,
        no_load_optim=True,
        no_load_rng=True,
        no_save_optim=True,
        no_save_rng=True,
        finetune=True,
        multi_lora=False,
        use_fault_tolerance=False,
        colocate=True,
        offload_train=True,
        offload_rollout=True,
    )
    _validate_miles_args(resolved, requested)
    resolved.partial_rollout = True
    with pytest.raises(RuntimeError, match="partial_rollout"):
        _validate_miles_args(resolved, requested)


def test_tiny_llama_checkpoint_exports_as_standard_peft_adapter(tmp_path):
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, LlamaConfig

    model_dir = tmp_path / "model"
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=32,
    )
    config.save_pretrained(model_dir)
    specs = derive_peft_lora_specs(
        str(model_dir),
        "a" * 40,
        rank=2,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        trust_remote_code=True,
    )
    args = manifest_args(specs_manifest(specs), model=str(model_dir))
    manifest = build_run_manifest(args, learners=2, reward_sha256="e" * 64)
    text = canonical_json(manifest)
    manifest_hash = manifest_sha256(text)

    flat = torch.arange(sum(spec.numel for spec in specs), dtype=torch.float32) / 100
    state = canonical_state(1, tensors_from_flat(flat, specs), expected_specs=specs)
    checkpoint = tmp_path / "state.ckpt"
    body = bytearray()
    body.extend(struct.pack("<QI", 1, 1))
    body.extend(struct.pack("<QQ", 1, flat.numel()))
    body.extend(flat.numpy().astype("<f4", copy=False).tobytes())
    body.extend(torch.ones_like(flat).numpy().astype("<f4", copy=False).tobytes())
    body.extend(struct.pack("<I", 2))
    for learner_id in range(2):
        body.extend(struct.pack("<IQQQ", learner_id, 1, 1, 1))
    header = struct.pack("<IH", 0xD17052A1, 1)
    header += bytes.fromhex(manifest_hash)
    header += bytes.fromhex(state.layout_fingerprint)
    header += struct.pack("<I", 2)
    header += bytes.fromhex(state.policy_hash)
    checkpoint.write_bytes(header + body)
    checkpoint.with_name(checkpoint.name + ".final").write_text(
        "YETO_RL_FINAL_V1\n"
        "global_step=1\n"
        "roster_size=2\n"
        f"run_manifest_sha256={manifest_hash}\n"
        f"layout_fingerprint={state.layout_fingerprint}\n"
        f"policy_sha256={state.policy_hash}\n",
        encoding="utf-8",
    )

    output = tmp_path / "adapter"
    provenance = export_rl_checkpoint(checkpoint, text, output)
    base = AutoModelForCausalLM.from_config(config)
    loaded = PeftModel.from_pretrained(base, output)
    reread = canonical_state(1, get_peft_model_state_dict(loaded), expected_specs=specs)
    assert reread.policy_hash == state.policy_hash == provenance["policy_sha256"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == provenance["checkpoint_sha256"]
