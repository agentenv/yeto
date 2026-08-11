"""Entrypoint for one pinned-Miles RL island."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bridge import BridgeConfig
from .decoupled import DecoupledBridgeConfig
from .export import adapter_targets, derive_peft_lora_specs
from .miles import verify_miles_revision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.learner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-revision", default=None)
    parser.add_argument("--syncer", required=True)
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--reward-function", required=True)
    parser.add_argument("--reward-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--global-rounds", type=int, required=True)
    parser.add_argument(
        "--sync-preset",
        choices=["strict-avg", "decoupled"],
        default="strict-avg",
    )
    parser.add_argument("--fragments", type=int, default=1)
    parser.add_argument("--pipeline", type=int, default=1)
    parser.add_argument("--local-horizon", type=int, default=1)
    parser.add_argument("--total-fragment-steps", type=int, default=None)
    parser.add_argument("--initial-adapter", default=None)
    parser.add_argument("--initial-adapter-sha256", default=None)
    parser.add_argument("--groups-per-round", type=int, required=True)
    parser.add_argument("--samples-per-group", type=int, required=True)
    parser.add_argument("--over-sampling-batch-size", type=int, required=True)
    parser.add_argument("--dynamic-sampling-filter-path", default=None)
    parser.add_argument("--dynamic-sampling-max-replacements", type=int, default=None)
    parser.add_argument("--rl-offload-train", action="store_true")
    parser.add_argument("--rl-distributed-timeout-minutes", type=int, default=10)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--rollout-max-response-len", type=int, required=True)
    parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default=None)
    parser.add_argument("--custom-generate-function-path", default=None)
    parser.add_argument("--use-session-server", action="store_true")
    parser.add_argument("--session-server-ip", default=None)
    parser.add_argument("--session-server-port", type=int, nargs="+", default=None)
    parser.add_argument("--tito-model", default=None)
    parser.add_argument("--completed-groups-path", required=True)
    parser.add_argument("--event-tape", required=True)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--actor-num-nodes", type=int, required=True)
    parser.add_argument("--actor-num-gpus-per-node", type=int, required=True)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--expert-parallel", type=int, default=None)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        required=True,
    )
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _provider_value(provider, *names: str):
    for name in names:
        value = getattr(provider, name, None)
        if value is not None:
            return value
    raise ValueError(
        f"Megatron-Bridge provider {type(provider).__name__} lacks {names[0]}"
    )


def _positive_int(provider, *names: str) -> int:
    value = _provider_value(provider, *names)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Megatron-Bridge provider has invalid {names[0]}={value!r}")
    return value


def _text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _miles_callable(spec: str) -> str:
    module, separator, function = spec.partition(":")
    if not separator or not module or not function.isidentifier():
        raise ValueError("RL callable must be package.module:function")
    return f"{module}.{function}"


def megatron_adapter_targets(specs, bridge) -> list[str]:
    """Map the exact PEFT contract onto Bridge's Megatron module paths."""

    model_bridge = getattr(bridge, "_model_bridge", None)
    if model_bridge is None:
        raise ValueError("Megatron-Bridge does not expose its model mapping")
    registry = model_bridge.mapping_registry()
    modules = {
        spec.name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
        for spec in specs
    }
    targets = set()
    for module in modules:
        hf_weight = f"{module}.weight"
        mapping = registry.hf_to_megatron_lookup(hf_weight)
        if mapping is None:
            raise ValueError(f"PEFT module {module!r} has no Megatron-Bridge mapping")
        megatron_module = mapping.megatron_param.removesuffix(".weight")
        prefix, separator, leaf = megatron_module.rpartition(".")
        if not separator:
            raise ValueError(f"invalid Megatron adapter module {megatron_module!r}")
        if leaf in {"linear_qkv", "linear_fc1"}:
            hf_params = mapping.hf_param
            component = next(
                (
                    name
                    for name, value in hf_params.items()
                    if value == hf_weight
                ),
                None,
            ) if isinstance(hf_params, dict) else None
            allowed = {"q", "k", "v"} if leaf == "linear_qkv" else {"gate", "up"}
            if component not in allowed:
                raise ValueError(
                    f"PEFT module {module!r} cannot use canonical Megatron LoRA"
                )
            leaf = (
                f"linear_{component}"
                if leaf == "linear_qkv"
                else f"linear_fc1_{component}"
            )
        targets.add(f"{prefix}.{leaf}")
    return sorted(targets)


def build_miles_argv(
    args,
    *,
    model_path: str | Path,
    prompt_path: str | Path,
    provider,
    target_modules: list[str],
    yeto_policy_sync: bool = True,
) -> list[str]:
    """Construct Miles arguments from Bridge's actual model provider."""

    hidden = _positive_int(provider, "hidden_size")
    heads = _positive_int(provider, "num_attention_heads")
    layers = _positive_int(provider, "num_layers")
    ffn = _positive_int(provider, "ffn_hidden_size")
    query_groups = int(getattr(provider, "num_query_groups", heads))
    kv_channels = int(getattr(provider, "kv_channels", hidden // heads))
    max_positions = int(
        _provider_value(
            provider,
            "seq_length",
            "max_sequence_length",
            "max_position_embeddings",
        )
    )
    if args.seq_len > max_positions:
        raise ValueError(
            f"--seq-len {args.seq_len} exceeds model context limit {max_positions}"
        )
    vocab_size = _positive_int(provider, "vocab_size", "padded_vocab_size")
    normalization = _text(getattr(provider, "normalization", "RMSNorm"))
    epsilon = float(
        _provider_value(provider, "layernorm_epsilon", "norm_epsilon")
    )
    position_type = _text(getattr(provider, "position_embedding_type", "rope"))
    rope_type = None
    if position_type == "yarn":
        position_type, rope_type = "rope", "yarn"
    rotary_base = int(getattr(provider, "rotary_base", 10000))
    rotary_percent = float(getattr(provider, "rotary_percent", 1.0))
    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    pipeline_parallel = getattr(args, "pipeline_parallel", 1)
    if pipeline_parallel <= 0 or actor_gpus % pipeline_parallel:
        raise ValueError("pipeline parallelism must divide Miles actor world size")
    data_parallel = actor_gpus // pipeline_parallel
    is_moe = getattr(provider, "num_moe_experts", None) is not None
    expert_parallel = getattr(args, "expert_parallel", None) or (
        data_parallel if is_moe else 1
    )
    if not is_moe and expert_parallel != 1:
        raise ValueError("EP>1 requires a MoE model")
    if data_parallel % expert_parallel:
        raise ValueError("expert parallelism must divide data parallelism")
    if is_moe and expert_parallel > 1 and args.lora_targets == "all-linear":
        raise ValueError(
            "EP>1 requires replicated attention LoRA, not expert-sharded all-linear LoRA"
        )
    global_batch = (
        args.groups_per_round * args.samples_per_group // args.optimizer_steps
    )
    if global_batch % data_parallel:
        raise ValueError("Miles global batch must divide evenly across DP ranks")
    if not target_modules:
        raise ValueError("PEFT selected no LoRA target modules")
    target_value = ",".join(target_modules)

    values = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", str(model_path),
        "--load", str(model_path),
        "--megatron-to-hf-mode", "bridge",
        "--model-name", type(provider).__name__,
        "--num-layers", str(layers),
        "--hidden-size", str(hidden),
        "--num-attention-heads", str(heads),
        "--num-query-groups", str(query_groups),
        "--kv-channels", str(kv_channels),
        "--ffn-hidden-size", str(ffn),
        "--max-position-embeddings", str(max_positions),
        "--seq-length", str(args.seq_len),
        "--normalization", normalization,
        "--norm-epsilon", str(epsilon),
        "--position-embedding-type", position_type,
        "--rotary-base", str(rotary_base),
        "--rotary-percent", str(rotary_percent),
        "--vocab-size", str(vocab_size),
        "--lora-rank", str(args.lora_r),
        "--lora-alpha", str(args.lora_r),
        "--lora-dropout", "0",
        "--lora-type", "canonical_lora",
        "--target-modules", target_value,
        "--lora-base-cpu-backup",
        "--actor-num-nodes", str(args.actor_num_nodes),
        "--actor-num-gpus-per-node", str(args.actor_num_gpus_per_node),
        "--num-gpus-per-node", str(args.actor_num_gpus_per_node),
        "--rollout-num-gpus-per-engine", "1",
        "--colocate",
        (
            "--offload-train"
            if getattr(args, "rl_offload_train", False)
            else "--no-offload-train"
        ),
        "--sglang-mem-fraction-static", "0.4",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", str(pipeline_parallel),
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", str(expert_parallel),
        "--expert-tensor-parallel-size", "1",
        "--prompt-data", str(prompt_path),
        "--input-key", "messages",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--apply-chat-template",
        "--rollout-seed",
        str(getattr(args, "rollout_seed", args.seed + args.learner_id)),
        "--sglang-enable-deterministic-inference",
        "--num-rollout", str(args.global_rounds),
        "--rollout-batch-size", str(args.groups_per_round),
        "--n-samples-per-prompt", str(args.samples_per_group),
        "--over-sampling-batch-size", str(args.over_sampling_batch_size),
        "--num-steps-per-rollout", str(args.optimizer_steps),
        "--global-batch-size", str(global_batch),
        "--balance-data",
        "--rollout-max-context-len", str(args.seq_len),
        "--rollout-max-response-len", str(args.rollout_max_response_len),
        "--rollout-function-path",
        (
            "yeto.rl.miles.generate_rollout"
            if yeto_policy_sync
            else "miles.rollout.sglang_rollout.generate_rollout"
        ),
        "--custom-rm-path", _miles_callable(args.reward_function),
        "--advantage-estimator", "grpo",
        "--lr", str(args.inner_lr),
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "unfused",
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--finetune",
        "--seed", str(args.seed),
        "--sglang-max-lora-rank", str(args.lora_r),
        "--pin-rollout-manager-to-head",
    ]
    values.extend(
        (
            "--distributed-timeout-minutes",
            str(getattr(args, "rl_distributed_timeout_minutes", 10)),
        )
    )
    chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", None)
    if chat_template_kwargs:
        values.extend(
            (
                "--apply-chat-template-kwargs",
                json.dumps(chat_template_kwargs, sort_keys=True, separators=(",", ":")),
            )
        )
    if yeto_policy_sync:
        values.extend(
            (
                "--rollout-all-samples-process-path",
                "yeto.rl.miles.queue_completed_groups",
                "--external-policy-sync-path",
                "yeto.rl.miles.create_policy_sync",
            )
        )
    for flag, name in (
        ("--rollout-engine-base-port", "rollout_engine_base_port"),
        ("--sglang-router-port", "sglang_router_port"),
        ("--sglang-router-prometheus-port", "sglang_router_prometheus_port"),
        ("--train-master-base-port", "train_master_base_port"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            values.extend((flag, str(value)))
    if args.custom_generate_function_path:
        values.extend(
            (
                "--custom-generate-function-path",
                args.custom_generate_function_path,
            )
        )
    dynamic_filter = getattr(args, "dynamic_sampling_filter_path", None)
    if dynamic_filter:
        values.extend(
            (
                "--dynamic-sampling-filter-path",
                dynamic_filter,
            )
        )
    if args.use_session_server:
        values.append("--use-session-server")
        if args.session_server_ip:
            values.extend(("--session-server-ip", args.session_server_ip))
        if args.session_server_port:
            values.append("--session-server-port")
            values.extend(str(port) for port in args.session_server_port)
        if args.tito_model:
            values.extend(("--tito-model", args.tito_model))
    if query_groups < heads:
        values.append("--group-query-attention")
    if rope_type is not None:
        values.extend(("--rope-type", rope_type))
    if position_type == "mrope":
        section = _provider_value(provider, "mrope_section")
        values.append("--mrope-section")
        values.extend(str(int(value)) for value in section)
    if bool(getattr(provider, "gated_linear_unit", False)):
        values.append("--swiglu")
    if not bool(getattr(provider, "share_embeddings_and_output_weights", True)):
        values.append("--untie-embeddings-and-output-weights")
    if getattr(provider, "add_bias_linear", None) is False:
        values.append("--disable-bias-linear")
    if bool(getattr(provider, "add_qkv_bias", False)):
        values.append("--add-qkv-bias")
    if bool(getattr(provider, "qk_layernorm", False)):
        values.append("--qk-layernorm")
    if (
        _text(getattr(provider, "experimental_attention_variant", None))
        == "gated_delta_net"
    ):
        values.extend(("--qkv-format", "bshd"))
    if getattr(provider, "num_moe_experts", None) is not None:
        values.extend(
            (
                "--num-experts",
                str(_positive_int(provider, "num_moe_experts")),
                "--moe-ffn-hidden-size",
                str(_positive_int(provider, "moe_ffn_hidden_size")),
                "--moe-router-topk",
                str(_positive_int(provider, "moe_router_topk")),
                "--moe-layer-freq",
                str(_provider_value(provider, "moe_layer_freq")),
            )
        )
        shared = getattr(provider, "moe_shared_expert_intermediate_size", None)
        if shared is not None:
            values.extend(
                ("--moe-shared-expert-intermediate-size", str(int(shared)))
            )
    if bool(getattr(provider, "multi_latent_attention", False)):
        values.append("--multi-latent-attention")
        for name in (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
        ):
            value = getattr(provider, name, None)
            if value is not None:
                values.extend((f"--{name.replace('_', '-')}", str(int(value))))
    return values


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages", row.get("prompt", row.get("input")))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise ValueError("RL rows must contain messages or a string prompt/input")
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
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"messages", "metadata"}
                }
            )
            normalized = {
                "messages": _messages(row),
                "label": row.get("label"),
                "metadata": metadata,
            }
            if "tools" in row:
                normalized["tools"] = row["tools"]
            handle.write(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("RL prompt dataset is empty")
    os.replace(temporary, output)
    return output


def _parse_miles_args(argv: list[str]):
    previous = sys.argv
    try:
        sys.argv = argv
        from miles.utils.arguments import parse_args as parse_miles_args

        return parse_miles_args()
    finally:
        sys.argv = previous


def _syncer_address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("--syncer must be HOST:PORT")
    return host, int(port)


def run_miles(
    args,
    *,
    model_path: str | Path,
    prompt_path: str | Path,
    yeto_policy_sync: bool = True,
    extra_argv: Sequence[str] = (),
) -> None:
    """Run one Miles job, optionally with Yeto's external policy boundary."""

    if (
        yeto_policy_sync
        and getattr(args, "sync_preset", "strict-avg") == "decoupled"
        and args.optimizer_steps != 1
    ):
        raise ValueError("decoupled RL requires one optimizer step per rollout")

    from megatron.bridge import AutoBridge

    model_bridge = AutoBridge.from_hf_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    provider = model_bridge.to_megatron_provider(load_weights=False)
    provider.finalize()
    specs = derive_peft_lora_specs(
        model_path,
        None,
        rank=args.lora_r,
        targets=args.lora_targets,
        trust_remote_code=args.trust_remote_code,
    )
    canonical_targets = adapter_targets(specs)
    miles_targets = megatron_adapter_targets(specs, model_bridge)
    miles_argv = build_miles_argv(
        args,
        model_path=model_path,
        prompt_path=prompt_path,
        provider=provider,
        target_modules=miles_targets,
        yeto_policy_sync=yeto_policy_sync,
    )
    miles_argv.extend(extra_argv)
    miles_args = _parse_miles_args(miles_argv)

    if yeto_policy_sync:
        from ..protocol import layout_fingerprint
        from .core import (
            build_rl_fragment_layout,
            canonical_layout_hash,
            canonical_lora_config_hash,
        )

        layout_hash = canonical_layout_hash(specs)
        lora_config_hash = canonical_lora_config_hash(
            rank=args.lora_r,
            target_modules=canonical_targets,
        )
        miles_args.yeto_rl_trust_remote_code = args.trust_remote_code
        miles_args.yeto_rl_model = args.model
        miles_args.yeto_rl_data = args.data
        miles_args.yeto_rl_base_model_revision = args.model_revision
        miles_args.yeto_rl_data_revision = args.data_revision
        miles_args.yeto_rl_lora_config_hash = lora_config_hash
        miles_args.yeto_rl_layout_hash = layout_hash
        miles_args.yeto_rl_reward_sha256 = args.reward_sha256
        miles_args.yeto_rl_dynamic_sampling_max_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        miles_args.yeto_rl_completed_groups_path = args.completed_groups_path
        miles_args.yeto_rl_event_tape = args.event_tape
        miles_args.yeto_rl_learner_id = args.learner_id
        sync_preset = getattr(args, "sync_preset", "strict-avg")
        miles_args.yeto_rl_sync_preset = sync_preset
        miles_args.external_policy_sync_run_until_stop = sync_preset == "decoupled"
        if sync_preset == "decoupled":
            miles_args.yeto_rl_source_sha256 = args.source_sha256
            miles_args.yeto_rl_initial_adapter = getattr(
                args, "initial_adapter", None
            )
            miles_args.yeto_rl_initial_adapter_sha256 = getattr(
                args, "initial_adapter_sha256", None
            )
            total_fragment_steps = args.total_fragment_steps
            if total_fragment_steps != args.global_rounds * args.fragments:
                raise ValueError(
                    "decoupled RL total fragment steps must equal sweeps*fragments"
                )
            layout = build_rl_fragment_layout(specs, args.fragments)
            sync_fingerprint = layout_fingerprint(layout).hex()
            miles_args.yeto_rl_num_fragments = args.fragments
            miles_args.yeto_rl_pipeline = args.pipeline
            miles_args.yeto_rl_local_horizon = args.local_horizon
            miles_args.yeto_rl_total_sweeps = args.global_rounds
            miles_args.yeto_rl_total_fragment_steps = total_fragment_steps
            miles_args.yeto_rl_sync_layout_fingerprint = sync_fingerprint
            miles_args.yeto_rl_learner_budget_steps = getattr(
                args,
                "learner_budget_steps",
                None,
            )
            miles_args.yeto_rl_bridge_config = DecoupledBridgeConfig(
                syncer_addr=_syncer_address(args.syncer),
                learner_id=args.learner_id,
                total_fragment_steps=total_fragment_steps,
                num_fragments=args.fragments,
                pipeline=args.pipeline,
                local_horizon=args.local_horizon,
                expected_specs=specs,
                base_model_revision=args.model_revision,
                lora_config_hash=lora_config_hash,
                canonical_layout_hash=layout_hash,
                wan_streams=args.wan_streams,
                learner_budget_steps=miles_args.yeto_rl_learner_budget_steps,
            )
        else:
            miles_args.yeto_rl_bridge_config = BridgeConfig(
                syncer_addr=_syncer_address(args.syncer),
                learner_id=args.learner_id,
                global_rounds=args.global_rounds,
                groups_per_round=args.groups_per_round,
                samples_per_group=args.samples_per_group,
                local_optimizer_steps=args.optimizer_steps,
                wan_streams=args.wan_streams,
                expected_specs=specs,
                base_model_revision=args.model_revision,
                lora_config_hash=lora_config_hash,
                layout_hash=layout_hash,
                event_tape=args.event_tape,
                audit_dir=args.audit_dir,
            )

    from train import train as miles_train

    asyncio.run(miles_train(miles_args))
    print(f"[rl] learner {args.learner_id} finalized")


def main(argv=None) -> None:
    args = parse_args(argv)
    from ..provenance import (
        is_immutable_commit,
        is_local_reference,
        python_spec_sha256,
        verify_source_tree_sha256,
    )

    verify_source_tree_sha256(args.source_sha256)
    if not is_immutable_commit(args.model_revision):
        raise ValueError("RL model revision must be an immutable commit")
    if not is_local_reference(args.data) and not is_immutable_commit(
        args.data_revision
    ):
        raise ValueError("RL remote dataset revision must be an immutable commit")
    reward_sha256 = python_spec_sha256(args.reward_function)
    if reward_sha256 != args.reward_sha256.lower():
        raise ValueError(
            f"reward source SHA256 mismatch: expected {args.reward_sha256.lower()}, "
            f"got {reward_sha256}"
        )
    verify_miles_revision(args.miles_root)
    miles_root = str(Path(args.miles_root).expanduser().resolve())
    if miles_root not in sys.path:
        sys.path.insert(0, miles_root)

    from miles.utils.misc import load_function

    load_function(_miles_callable(args.reward_function))
    if args.custom_generate_function_path:
        load_function(args.custom_generate_function_path)
    from huggingface_hub import snapshot_download

    from ..models import resolve
    model = resolve(args.model)
    if is_local_reference(model):
        model_path = str(Path(model).expanduser().resolve())
    else:
        model_path = snapshot_download(repo_id=model, revision=args.model_revision)
    prompt_path = prepare_prompt_data(
        args.data,
        args.data_revision,
        "~/yeto-rl/prompts.jsonl",
    )
    run_miles(args, model_path=model_path, prompt_path=prompt_path)


if __name__ == "__main__":
    main()
