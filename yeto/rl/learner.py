"""Entrypoint for one pinned-Miles strict RL island."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .bridge import BridgeConfig, StrictRlBridge
from .export import _transformers_model_family, specs_from_manifest
from .manifest import (
    MILES_COMMIT,
    canonical_json,
    manifest_sha256,
    path_tree_sha256,
    validate_manifest,
)
from .miles import MilesIslandRuntime, verify_miles_revision
from .reward import configure_reward_environment, validate_callable_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.learner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-revision", default=None)
    parser.add_argument("--syncer", required=True)
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--num-learners", type=int, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--reward-function", required=True)
    parser.add_argument("--reward-sha256", required=True)
    parser.add_argument("--generate-function", default=None)
    parser.add_argument("--generate-sha256", default=None)
    parser.add_argument("--global-rounds", type=int, required=True)
    parser.add_argument("--groups-per-round", type=int, required=True)
    parser.add_argument("--samples-per-group", type=int, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--round-timeout-s", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument("--lora-targets", required=True)
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _text_config(config):
    return getattr(config, "text_config", config)


def _required(config, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"pinned model config has no positive {name}")
    return value


def build_miles_argv(
    args,
    manifest: dict[str, Any],
    *,
    model_path: str | Path,
    prompt_path: str | Path,
    config,
) -> list[str]:
    """Construct the fixed synchronous Miles/Megatron configuration."""

    from .export import _reject_unsupported_config, _rope_scaling_factor, _rope_theta

    _reject_unsupported_config(config)
    text = _text_config(config)
    hidden = _required(text, "hidden_size")
    heads = _required(text, "num_attention_heads")
    layers = _required(text, "num_hidden_layers")
    ffn = _required(text, "intermediate_size")
    kv_heads = int(getattr(text, "num_key_value_heads", heads))
    head_dim = int(getattr(text, "head_dim", hidden // heads))
    rope_scaling_factor = _rope_scaling_factor(text)
    max_positions = int(getattr(text, "max_position_embeddings", args.seq_len))
    norm_epsilon = float(
        getattr(text, "rms_norm_eps", getattr(text, "layer_norm_eps", 1e-5))
    )
    targets = ",".join(manifest["lora"]["target_modules"])
    global_batch = args.groups_per_round * args.samples_per_group // args.optimizer_steps
    output = Path(args.cache_dir).expanduser() / "miles-local"
    output.mkdir(parents=True, exist_ok=True)

    values = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", str(model_path),
        "--load", str(model_path),
        "--save", str(output),
        "--megatron-to-hf-mode", "bridge",
        "--model-name", str(getattr(text, "model_type", "yeto_rl")),
        "--num-layers", str(layers),
        "--hidden-size", str(hidden),
        "--num-attention-heads", str(heads),
        "--num-query-groups", str(kv_heads),
        "--kv-channels", str(head_dim),
        "--ffn-hidden-size", str(ffn),
        "--max-position-embeddings", str(max_positions),
        "--seq-length", str(args.seq_len),
        "--normalization", "RMSNorm",
        "--norm-epsilon", str(norm_epsilon),
        "--position-embedding-type", "rope",
        "--rotary-base", str(_rope_theta(text)),
        "--vocab-size", str(_required(text, "vocab_size")),
        "--lora-rank", str(args.lora_r),
        "--lora-alpha", str(args.lora_r),
        "--lora-dropout", "0",
        "--lora-type", "canonical_lora",
        "--target-modules", targets,
        "--lora-base-cpu-backup",
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", "1",
        "--num-gpus-per-node", "1",
        "--rollout-num-gpus-per-engine", "1",
        "--colocate",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--prompt-data", str(prompt_path),
        "--input-key", "messages",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rollout-seed", str(args.seed + args.learner_id),
        "--num-rollout", str(args.global_rounds),
        "--rollout-batch-size", str(args.groups_per_round),
        "--n-samples-per-prompt", str(args.samples_per_group),
        "--num-steps-per-rollout", str(args.optimizer_steps),
        "--global-batch-size", str(global_batch),
        "--rollout-max-context-len", str(args.seq_len),
        "--rollout-max-prompt-len", str(args.seq_len // 2),
        "--rollout-max-response-len", str(args.seq_len - args.seq_len // 2),
        "--rollout-temperature", "1",
        "--rollout-function-path", "yeto.rl.miles.generate_rollout",
        "--custom-rm-path", "yeto.rl.reward.miles_reward",
        "--advantage-estimator", "grpo",
        "--eps-clip", "0.2",
        "--eps-clip-high", "0.28",
        "--entropy-coef", "0",
        "--kl-coef", "0",
        "--optimizer", "adam",
        "--lr", str(args.inner_lr),
        "--min-lr", str(args.inner_lr),
        "--lr-decay-style", "constant",
        "--lr-warmup-iters", "0",
        "--weight-decay", "0",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.999",
        "--clip-grad", "1",
        "--micro-batch-size", "1",
        "--attention-dropout", "0",
        "--hidden-dropout", "0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--use-distributed-optimizer",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--finetune",
        "--seed", str(args.seed),
        "--sglang-max-lora-rank", str(args.lora_r),
        "--sglang-mem-fraction-static", "0.5",
    ]
    if kv_heads < heads:
        values.append("--group-query-attention")
    values.append("--swiglu")
    if not bool(getattr(text, "tie_word_embeddings", False)):
        values.append("--untie-embeddings-and-output-weights")
    values.append("--disable-bias-linear")
    if text.model_type == "qwen2":
        values.append("--add-qkv-bias")
    if text.model_type == "qwen3":
        values.append("--qk-layernorm")
    if rope_scaling_factor is not None:
        values.extend(
            ("--use-rope-scaling", "--rope-scaling-factor", str(rope_scaling_factor))
        )
    custom_generate = (manifest.get("generation") or {}).get("custom_generate")
    if custom_generate is not None:
        module, _, function = custom_generate["callable"].partition(":")
        values.extend(("--custom-generate-function-path", f"{module}.{function}"))
    return values


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages", row.get("prompt", row.get("input")))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value or any(not isinstance(item, dict) for item in value):
        raise ValueError("RL prompt rows must contain messages or a string prompt/input")
    return value


def prepare_prompt_data(
    source: str,
    revision: str | None,
    output_path: str | Path,
) -> Path:
    from ..data import load_rows

    rows = load_rows(source, revision=revision)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for raw in rows:
            row = dict(raw)
            messages = _messages(row)
            metadata = {key: value for key, value in row.items() if key != "messages"}
            normalized = {
                "messages": messages,
                "label": row.get("label"),
                "metadata": metadata,
            }
            if "tools" in row:
                normalized["tools"] = row["tools"]
            handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("RL prompt dataset is empty")
    os.replace(temporary, output)
    return output


def _parse_miles_args(argv: list[str]):
    old = sys.argv
    try:
        sys.argv = argv
        from miles.utils.arguments import parse_args as parse_miles_args

        return parse_miles_args()
    finally:
        sys.argv = old


def _validate_miles_args(args, requested) -> None:
    from .manifest import target_modules

    global_batch = (
        requested.groups_per_round
        * requested.samples_per_group
        // requested.optimizer_steps
    )
    expected = {
        "train_backend": "megatron",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "rollout_num_gpus": 1,
        "rollout_num_gpus_per_engine": 1,
        "lora_type": "canonical_lora",
        "lora_rank": requested.lora_r,
        "lora_alpha": requested.lora_r,
        "lora_dropout": 0.0,
        "target_modules": list(target_modules(requested.lora_targets)),
        "lora_base_cpu_backup": True,
        "bf16": True,
        "use_distributed_optimizer": True,
        "use_critic": False,
        "advantage_estimator": "grpo",
        "compute_advantages_and_returns": True,
        "rewards_normalization": True,
        "grpo_std_normalization": requested.samples_per_group > 1,
        "normalize_advantages": False,
        "loss_type": "policy_loss",
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "entropy_coef": 0.0,
        "use_kl_loss": False,
        "kl_coef": 0.0,
        "kl_loss_coef": 0.0,
        "use_tis": False,
        "use_rollout_logprobs": False,
        "partial_rollout": False,
        "dynamic_sampling_filter_path": None,
        "use_dynamic_batch_size": False,
        "balance_data": False,
        "rollout_batch_size": requested.groups_per_round,
        "n_samples_per_prompt": requested.samples_per_group,
        "num_steps_per_rollout": requested.optimizer_steps,
        "global_batch_size": global_batch,
        "micro_batch_size": 1,
        "num_rollout": requested.global_rounds,
        "over_sampling_batch_size": requested.groups_per_round,
        "rollout_shuffle": True,
        "rollout_seed": requested.seed + requested.learner_id,
        "rollout_max_context_len": requested.seq_len,
        "rollout_max_prompt_len": requested.seq_len // 2,
        "rollout_max_response_len": requested.seq_len - requested.seq_len // 2,
        "rollout_temperature": 1.0,
        "rollout_function_path": "yeto.rl.miles.generate_rollout",
        "custom_rm_path": "yeto.rl.reward.miles_reward",
        "custom_generate_function_path": (
            getattr(requested, "generate_function", None).replace(":", ".")
            if getattr(requested, "generate_function", None)
            else None
        ),
        "input_key": "messages",
        "label_key": "label",
        "metadata_key": "metadata",
        "tool_key": "tools",
        "apply_chat_template": True,
        "optimizer": "adam",
        "lr": requested.inner_lr,
        "min_lr": requested.inner_lr,
        "lr_decay_style": "constant",
        "lr_warmup_iters": 0,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "clip_grad": 1.0,
        "seed": requested.seed,
        "no_load_optim": True,
        "no_load_rng": True,
        "no_save_optim": True,
        "no_save_rng": True,
        "finetune": True,
        "multi_lora": False,
        "use_fault_tolerance": False,
    }
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise RuntimeError(
                f"pinned Miles resolved unsupported {name}={getattr(args, name, None)!r}"
            )
    if not args.colocate or not args.offload_train or not args.offload_rollout:
        raise RuntimeError("RL v0 requires Miles colocated train/rollout offload")
    if args.num_steps_per_rollout != requested.optimizer_steps:
        raise RuntimeError("Miles changed the requested optimizer-step count")


def _syncer_address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("--syncer must be HOST:PORT")
    return host, int(port)


def _validate_runtime_contract(args, manifest: dict[str, Any]) -> None:
    base = manifest["base_model"]
    dataset = manifest["dataset"]
    lora = manifest["lora"]
    reward = manifest["reward"]
    generation = manifest["generation"]
    if args.model_revision != base["revision"]:
        raise RuntimeError("learner model revision does not match the RL manifest")
    if bool(args.trust_remote_code) != base["trust_remote_code"]:
        raise RuntimeError("learner remote-code setting does not match the RL manifest")
    if args.data_revision != dataset["revision"]:
        raise RuntimeError("learner dataset revision does not match the RL manifest")
    if (
        args.reward_function != reward["callable"]
        or args.reward_sha256 != reward["source_sha256"]
    ):
        raise RuntimeError("learner reward identity does not match the RL manifest")
    if args.source_sha256 != manifest["yeto_source_sha256"]:
        raise RuntimeError("learner Yeto source identity does not match the RL manifest")
    if args.lora_r != lora["rank"] or args.lora_targets != lora["targets"]:
        raise RuntimeError("learner LoRA settings do not match the RL manifest")
    if args.inner_lr != manifest["optimizer"]["learning_rate"]:
        raise RuntimeError("learner optimizer settings do not match the RL manifest")
    if args.seq_len != generation["max_context_length"] or args.seed != generation["seed"]:
        raise RuntimeError("learner generation settings do not match the RL manifest")
    custom_generate = generation["custom_generate"]
    if custom_generate is None:
        if args.generate_function is not None or args.generate_sha256 is not None:
            raise RuntimeError("learner generate function is absent from the RL manifest")
    elif (
        args.generate_function != custom_generate["callable"]
        or args.generate_sha256 != custom_generate["source_sha256"]
    ):
        raise RuntimeError("learner generate function does not match the RL manifest")


def main(argv=None) -> None:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    from ..provenance import verify_source_tree_sha256

    verify_source_tree_sha256(args.source_sha256)
    manifest_text = os.environ.get("YETO_RL_MANIFEST")
    if manifest_text is None:
        raise RuntimeError("YETO_RL_MANIFEST is not set")
    manifest = validate_manifest(manifest_text, args.manifest_sha256)
    if canonical_json(manifest) != manifest_text or manifest_sha256(manifest_text) != args.manifest_sha256:
        raise RuntimeError("RL manifest identity mismatch")
    if manifest["miles"]["commit"] != MILES_COMMIT:
        raise RuntimeError("RL manifest selected an unsupported Miles revision")
    if manifest["workload"] != {
        "learners": args.num_learners,
        "groups_per_island_round": args.groups_per_round,
        "samples_per_group": args.samples_per_group,
        "local_optimizer_steps": args.optimizer_steps,
        "global_rounds": args.global_rounds,
    }:
        raise RuntimeError("learner workload arguments do not match the RL manifest")
    _validate_runtime_contract(args, manifest)
    if not 0 <= args.learner_id < args.num_learners:
        raise ValueError("logical learner ID is outside the fixed roster")

    workdir = Path(__file__).resolve().parents[2]
    configure_reward_environment(args.reward_function, args.reward_sha256, workdir)
    if args.generate_function is not None:
        validate_callable_source(
            args.generate_function,
            args.generate_sha256,
            workdir,
            label="RL generate",
        )
    verify_miles_revision(args.miles_root)
    cache = Path(args.cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    base = manifest["base_model"]
    model_path = snapshot_download(
        repo_id=base["identifier"],
        revision=base["revision"],
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=bool(base.get("trust_remote_code")),
    )
    _transformers_model_family(config)
    dataset = manifest["dataset"]
    if dataset["revision"] is None:
        if path_tree_sha256(args.data) != dataset["content_sha256"]:
            raise RuntimeError("mounted RL dataset content hash does not match the manifest")
        data_source, data_revision = args.data, None
    else:
        data_source, data_revision = dataset["identifier"], dataset["revision"]
    prompt_path = prepare_prompt_data(
        data_source,
        data_revision,
        cache / "prompts.jsonl",
    )

    miles_argv = build_miles_argv(
        args,
        manifest,
        model_path=model_path,
        prompt_path=prompt_path,
        config=config,
    )
    miles_args = _parse_miles_args(miles_argv)
    miles_args.yeto_rl_trust_remote_code = args.trust_remote_code
    _validate_miles_args(miles_args, args)
    expected_specs = specs_from_manifest(manifest)
    runtime = MilesIslandRuntime(miles_args, manifest)
    bridge = StrictRlBridge(
        runtime,
        BridgeConfig(
            syncer_addr=_syncer_address(args.syncer),
            learner_id=args.learner_id,
            run_manifest_sha256=args.manifest_sha256,
            groups_per_round=args.groups_per_round,
            samples_per_group=args.samples_per_group,
            local_optimizer_steps=args.optimizer_steps,
            cache_dir=cache / "result",
            run_id=manifest["run_id"],
            event_tape=cache / "events.jsonl",
            wan_streams=args.wan_streams,
            round_timeout_s=args.round_timeout_s,
            expected_specs=expected_specs,
            expected_layout_fingerprint=manifest["canonical_lora"]["layout_fingerprint"],
        ),
    )
    final = bridge.run()
    if final.policy_version != args.global_rounds:
        raise RuntimeError(
            f"strict RL finalized at version {final.policy_version}, expected {args.global_rounds}"
        )
    print(
        f"[rl] learner {args.learner_id} finalized policy "
        f"v{final.policy_version} {final.policy_hash}"
    )


if __name__ == "__main__":
    main()
