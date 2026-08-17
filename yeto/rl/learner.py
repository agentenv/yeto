"""Entrypoint for one pinned-Miles RL island."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

from .bridge import BridgeConfig
from .decoupled import DecoupledBridgeConfig
from .deepseek_v4_expert_full import expert_full_specs
from .export import adapter_targets, derive_peft_lora_specs
from .miles import verify_miles_revision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.learner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollout-model", default=None)
    parser.add_argument("--rollout-model-revision", default=None)
    parser.add_argument(
        "--rl-model-recipe",
        choices=["generic", "deepseek-v4-flash"],
        default="generic",
    )
    parser.add_argument("--expert-full-count", type=int, default=0)
    parser.add_argument("--expert-full-lr", type=float, default=1e-6)
    parser.add_argument("--expert-selection-sha256", default=None)
    parser.add_argument("--expert-selection-contract-sha256", default=None)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-revision", default=None)
    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--eval-data-sha256", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-samples-per-prompt", type=int, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-top-p", type=float, default=None)
    parser.add_argument("--eval-max-prompt-len", type=int, default=None)
    parser.add_argument("--eval-max-response-len", type=int, default=None)
    parser.add_argument("--eval-max-context-len", type=int, default=None)
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
    parser.add_argument("--custom-agent-function-path", default=None)
    parser.add_argument("--codex-harness-contract", type=json.loads, default=None)
    parser.add_argument("--agent-max-seq-len", type=int, default=None)
    parser.add_argument("--use-session-server", action="store_true")
    parser.add_argument("--session-server-ip", default=None)
    parser.add_argument("--session-server-port", type=int, nargs="+", default=None)
    parser.add_argument("--tito-model", default=None)
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        choices=["tool", "user", "system"],
        default=None,
    )
    parser.add_argument("--completed-groups-path", required=True)
    parser.add_argument("--event-tape", required=True)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--actor-num-nodes", type=int, required=True)
    parser.add_argument("--actor-num-gpus-per-node", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--expert-parallel", type=int, default=None)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--sglang-tp-size", type=int, default=None)
    parser.add_argument("--sglang-dp-size", type=int, default=None)
    parser.add_argument("--sglang-ep-size", type=int, default=None)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    parser.add_argument("--sglang-attention-backend", default=None)
    parser.add_argument(
        "--sglang-deterministic-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sglang-page-size", type=int, default=None)
    parser.add_argument("--sglang-max-running-requests", type=int, default=None)
    parser.add_argument("--sglang-chunked-prefill-size", type=int, default=None)
    parser.add_argument("--use-rollout-routing-replay", action="store_true")
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument(
        "--lora-targets",
        choices=[
            "auto",
            "attention",
            "attention-routed-experts",
            "all-linear",
        ],
        required=True,
    )
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _strict_json_file_semantics(path: Path) -> tuple[Any, ...]:
    """Parse exact, typed JSON semantics independent of object-key order."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"duplicate JSON object key: {name!r}")
            value[name] = item
        return value

    def finite_number(name: str) -> None:
        raise ValueError(f"non-finite JSON number: {name}")

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=unique_object,
            parse_constant=finite_number,
            parse_float=Decimal,
            parse_int=Decimal,
        )
    except (OSError, UnicodeError, ValueError, DecimalException) as exc:
        raise ValueError("Codex app-server schema is malformed JSON") from exc

    def typed_semantics(item: Any) -> tuple[Any, ...]:
        if item is None:
            return ("null",)
        if isinstance(item, bool):
            return ("boolean", item)
        if isinstance(item, Decimal):
            return ("number", item)
        if isinstance(item, str):
            return ("string", item)
        if isinstance(item, list):
            return ("array", tuple(typed_semantics(value) for value in item))
        if isinstance(item, dict):
            return (
                "object",
                tuple(
                    sorted(
                        (name, typed_semantics(value))
                        for name, value in item.items()
                    )
                ),
            )
        raise TypeError(f"unsupported parsed JSON value: {type(item).__name__}")

    return typed_semantics(value)


def _verify_live_codex_app_server_schema(pinned: Path, generated: Path) -> None:
    try:
        pinned_semantics = _strict_json_file_semantics(pinned)
        generated_semantics = _strict_json_file_semantics(generated)
    except ValueError as exc:
        raise ValueError("live stock Codex app-server schema drifted") from exc
    if generated_semantics != pinned_semantics:
        raise ValueError("live stock Codex app-server schema drifted")


def _preflight_codex_harness(args) -> None:
    """Fail before model allocation if the signed Codex surface has drifted."""

    from . import (
        CODEX_APP_SERVER_PROTOCOL_REVISION,
        CODEX_APP_SERVER_SCHEMA_SHA256,
        CODEX_BASE_INSTRUCTIONS_SHA256,
        CODEX_CLI_VERSION,
        CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH,
        CODEX_CONTAINER_BINARY_PATH,
        CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
        CODEX_HARNESS_AGENT,
        CODEX_HARNESS_AGENT_SHA256,
        CODEX_LINUX_BINARY_SHA256,
        CODEX_LINUX_BINARY_SIZE_BYTES,
        CODEX_LINUX_TARGET,
        CODEX_NPM_PACKAGE,
        CODEX_NPM_TARBALL_SHA256,
        CODEX_PACKAGE_MANIFEST_SHA256,
        CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256,
    )

    contract = getattr(args, "codex_harness_contract", None)
    if args.custom_agent_function_path != CODEX_HARNESS_AGENT:
        if contract is not None:
            raise ValueError(
                "--codex-harness-contract requires the signed stock Codex agent"
            )
        return
    if not isinstance(contract, dict):
        raise ValueError("the signed stock Codex agent requires its harness contract")
    required = {
        "agent_function_path",
        "agent_source_sha256",
        "controller_binary_path",
        "controller_package_manifest_path",
        "controller_app_server_schema_path",
        "bundle_binary_path",
        "bundle_package_manifest_path",
        "bundle_app_server_schema_path",
        "container_binary_path",
        "container_app_server_schema_path",
        "binary_sha256",
        "binary_size_bytes",
        "cli_version",
        "npm_package",
        "target",
        "npm_tarball_sha256",
        "package_manifest_sha256",
        "app_server_protocol_revision",
        "app_server_schema_sha256",
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
        "reasoning_effort",
        "backend",
    }
    if set(contract) != required:
        raise ValueError("stock Codex harness contract shape drifted")
    pinned = {
        "agent_function_path": CODEX_HARNESS_AGENT,
        "agent_source_sha256": CODEX_HARNESS_AGENT_SHA256,
        "container_binary_path": CODEX_CONTAINER_BINARY_PATH,
        "container_app_server_schema_path": (
            CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH
        ),
        "binary_sha256": CODEX_LINUX_BINARY_SHA256,
        "binary_size_bytes": CODEX_LINUX_BINARY_SIZE_BYTES,
        "cli_version": CODEX_CLI_VERSION,
        "npm_package": CODEX_NPM_PACKAGE,
        "target": CODEX_LINUX_TARGET,
        "npm_tarball_sha256": CODEX_NPM_TARBALL_SHA256,
        "package_manifest_sha256": CODEX_PACKAGE_MANIFEST_SHA256,
        "app_server_protocol_revision": CODEX_APP_SERVER_PROTOCOL_REVISION,
        "app_server_schema_sha256": CODEX_APP_SERVER_SCHEMA_SHA256,
        "base_instructions_sha256": CODEX_BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": (
            CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256
        ),
        "submit_tool_schema_sha256": CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        "dynamic_tools_schema_sha256": CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
        "reasoning_effort": "xhigh",
    }
    if any(contract.get(name) != expected for name, expected in pinned.items()):
        raise ValueError("stock Codex runtime identity drifted")
    expected_backend = {
        "model": "deepseekv4",
        "max_tokens": args.rollout_max_response_len,
        "reasoning_effort": "max",
        "thinking": {"type": "enabled"},
        "chat_template": "deepseekv4",
        "chat_template_kwargs": {
            "thinking_mode": "thinking",
            "reasoning_effort": "max",
            "drop_thinking": False,
        },
        "tito_allowed_append_roles": ["tool", "user"],
    }
    if (
        contract.get("backend") != expected_backend
        or args.rl_model_recipe != "deepseek-v4-flash"
        or args.tito_model != "deepseekv4"
        or args.tito_allowed_append_roles != ["tool", "user"]
        or args.apply_chat_template_kwargs
        != expected_backend["chat_template_kwargs"]
    ):
        raise ValueError("stock Codex DSV4 backend/TITO identity drifted")
    try:
        from yeto_miles_secrlenv import codex_harness_agent

        live_identity = codex_harness_agent.codex_harness_identity()
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("cannot attest the Yeto Codex harness adapter") from exc
    identity_names = (
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
    )
    if live_identity != {name: contract.get(name) for name in identity_names}:
        raise ValueError("Yeto Codex adapter identity drifted")

    from ..provenance import file_sha256

    adapter_path = Path(codex_harness_agent.__file__)
    if adapter_path.is_symlink() or not adapter_path.is_file():
        raise ValueError("Yeto Codex adapter source identity drifted")
    adapter_path = adapter_path.resolve()
    if file_sha256(adapter_path) != contract.get("agent_source_sha256"):
        raise ValueError("Yeto Codex adapter source identity drifted")
    binary = Path(CODEX_CONTAINER_BINARY_PATH)
    manifest = Path("/opt/yeto/codex/codex-package.json")
    schema = Path(CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH)
    if (
        binary.is_symlink()
        or not binary.is_file()
        or binary.stat().st_size != CODEX_LINUX_BINARY_SIZE_BYTES
        or file_sha256(binary) != CODEX_LINUX_BINARY_SHA256
        or manifest.is_symlink()
        or not manifest.is_file()
        or file_sha256(manifest) != CODEX_PACKAGE_MANIFEST_SHA256
        or schema.is_symlink()
        or not schema.is_file()
        or file_sha256(schema) != CODEX_APP_SERVER_SCHEMA_SHA256
    ):
        raise ValueError("mounted stock Codex artifact does not match its Yeto pin")
    try:
        version = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot execute the pinned stock Codex binary") from exc
    if version != CODEX_CLI_VERSION:
        raise ValueError("stock Codex executable version drifted")
    try:
        with tempfile.TemporaryDirectory(prefix="yeto-codex-schema-") as temporary:
            root = Path(temporary)
            output = root / "schema"
            subprocess.run(
                [
                    str(binary),
                    "app-server",
                    "generate-json-schema",
                    "--experimental",
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            generated_schema = (
                output / "codex_app_server_protocol.v2.schemas.json"
            )
            if (
                generated_schema.is_symlink()
                or not generated_schema.is_file()
            ):
                raise ValueError("live stock Codex app-server schema drifted")
            _verify_live_codex_app_server_schema(schema, generated_schema)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot verify the stock Codex app-server schema") from exc
    expected_env = {
        "YETO_CODEX_BINARY_PATH": CODEX_CONTAINER_BINARY_PATH,
        "YETO_CODEX_BINARY_SHA256": CODEX_LINUX_BINARY_SHA256,
        "YETO_CODEX_BINARY_SIZE_BYTES": str(CODEX_LINUX_BINARY_SIZE_BYTES),
        "YETO_CODEX_VERSION": CODEX_CLI_VERSION,
        "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION": (
            CODEX_APP_SERVER_PROTOCOL_REVISION
        ),
        "YETO_CODEX_APP_SERVER_SCHEMA_SHA256": CODEX_APP_SERVER_SCHEMA_SHA256,
        "YETO_CODEX_BASE_INSTRUCTIONS_SHA256": contract[
            "base_instructions_sha256"
        ],
        "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": contract[
            "terminal_exec_tool_schema_sha256"
        ],
        "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256": contract[
            "submit_tool_schema_sha256"
        ],
        "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256": contract[
            "dynamic_tools_schema_sha256"
        ],
        "YETO_CODEX_REASONING_EFFORT": "xhigh",
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(args.rollout_max_response_len),
        "YETO_CODEX_BACKEND_REASONING_EFFORT": "max",
        "YETO_CODEX_BACKEND_THINKING": "enabled",
        "YETO_CODEX_CHAT_TEMPLATE": "deepseekv4",
        "YETO_CODEX_CHAT_TEMPLATE_KWARGS": json.dumps(
            expected_backend["chat_template_kwargs"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES": "tool,user",
        "YETO_CODEX_HARNESS_CONTRACT_SHA256": _canonical_sha256(contract),
    }
    mismatched = [
        name for name, expected in expected_env.items() if os.getenv(name) != expected
    ]
    if mismatched:
        raise ValueError(
            "stock Codex container environment drifted: " + ", ".join(mismatched)
        )


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


_ROUTED_EXPERT_LORA_MODULE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\."
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)

_MEGATRON_LAYER_MODULE = re.compile(
    r"^(?P<prefix>(?:.*\.)?decoder\.layers\.)\d+(?P<suffix>\..+)$"
)


def _pipeline_local_target(target: str, pipeline_parallel: int) -> str:
    if pipeline_parallel <= 1:
        return target
    # Pipeline stages renumber their physical decoder layers from zero.
    # Bridge's PEFT matcher accepts ``*`` here, so retain the exact module
    # type while making the target independent of that local renumbering.
    # Canonical HF specs still define and validate global layer ownership.
    match = _MEGATRON_LAYER_MODULE.fullmatch(target)
    if match is None:
        return target
    return f"{match.group('prefix')}*{match.group('suffix')}"


def megatron_adapter_targets(
    specs,
    bridge,
    *,
    standard_grouped_experts: bool = False,
    pipeline_parallel: int = 1,
) -> list[str]:
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
        expert = _ROUTED_EXPERT_LORA_MODULE.fullmatch(module)
        if standard_grouped_experts and expert is not None:
            branch = (
                "linear_fc2"
                if expert.group("projection") == "down_proj"
                else "linear_fc1"
            )
            targets.add(
                _pipeline_local_target(
                    f"decoder.layers.{expert.group('layer')}.mlp.experts.{branch}",
                    pipeline_parallel,
                )
            )
            continue
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
        target = f"{prefix}.{leaf}"
        targets.add(_pipeline_local_target(target, pipeline_parallel))
    return sorted(targets)


def build_miles_argv(
    args,
    *,
    model_path: str | Path,
    rollout_model_path: str | Path | None = None,
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
    multi_latent_attention = bool(
        getattr(provider, "multi_latent_attention", False)
    )
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
    tensor_parallel = getattr(args, "tensor_parallel", 1)
    pipeline_parallel = getattr(args, "pipeline_parallel", 1)
    model_parallel = tensor_parallel * pipeline_parallel
    if tensor_parallel <= 0 or pipeline_parallel <= 0 or actor_gpus % model_parallel:
        raise ValueError("Miles actor world must be divisible by TP*PP")
    is_moe = getattr(provider, "num_moe_experts", None) is not None
    expert_parallel = getattr(args, "expert_parallel", None) or (
        actor_gpus if is_moe else 1
    )
    if not is_moe and expert_parallel != 1:
        raise ValueError("EP>1 requires a MoE model")
    if actor_gpus % expert_parallel:
        raise ValueError("expert parallelism must divide Miles actor world size")
    if is_moe and expert_parallel > 1 and args.lora_targets == "all-linear":
        raise ValueError(
            "EP>1 requires replicated attention LoRA, not expert-sharded all-linear LoRA"
        )
    if args.lora_targets == "attention-routed-experts" and (
        not is_moe or getattr(args, "rl_model_recipe", "generic") != "deepseek-v4-flash"
    ):
        raise ValueError(
            "attention-routed-experts is reserved for the expanded DeepSeek V4 recipe"
        )
    expert_full_count = int(getattr(args, "expert_full_count", 0) or 0)
    if not 0 <= expert_full_count <= 32:
        raise ValueError("expert-full count must be between 1 and 32 when enabled")
    expert_full = expert_full_count > 0
    if expert_full and (
        getattr(args, "rl_model_recipe", "generic") != "deepseek-v4-flash"
        or args.lora_targets != "attention"
    ):
        raise ValueError(
            "expert-full tuning requires the DeepSeek V4 recipe and attention LoRA"
        )
    data_parallel = actor_gpus // model_parallel
    global_batch = (
        args.groups_per_round * args.samples_per_group // args.optimizer_steps
    )
    if global_batch % data_parallel:
        raise ValueError("Miles global batch must divide evenly across DP ranks")
    if not target_modules:
        raise ValueError("PEFT selected no LoRA target modules")
    target_value = ",".join(target_modules)
    recipe = getattr(args, "rl_model_recipe", "generic")
    if recipe == "deepseek-v4-flash":
        expected_lora_targets = (
            "attention" if expert_full else "attention-routed-experts"
        )
        if args.lora_targets != expected_lora_targets:
            raise ValueError(
                "expanded DeepSeek V4 recipe requires "
                f"{expected_lora_targets}"
            )
        if (
            tensor_parallel != 8
            or expert_parallel != 8
            or getattr(args, "rollout_num_gpus_per_engine", 1) != 8
        ):
            raise ValueError(
                "expanded DeepSeek V4 requires TP8/EP8 pipeline stages and "
                "per-node eight-GPU rollout replicas"
            )
        if layers != 43 or not is_moe or not multi_latent_attention:
            raise ValueError(
                "DeepSeek V4 Flash recipe requires the 43-layer MoE/MLA provider"
            )
        model_name = "deepseekv4"
        training_attention_backend = "flash"
    else:
        model_name = type(provider).__name__
        training_attention_backend = "unfused"

    values = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", str(rollout_model_path or model_path),
        "--ref-load", str(model_path),
        "--megatron-to-hf-mode", "bridge",
        "--model-name", model_name,
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
        "--lora-type",
        (
            "lora"
            if args.lora_targets == "attention-routed-experts"
            else "canonical_lora"
        ),
        "--target-modules", target_value,
        "--lora-base-cpu-backup",
        "--actor-num-nodes", str(args.actor_num_nodes),
        "--actor-num-gpus-per-node", str(args.actor_num_gpus_per_node),
        "--num-gpus-per-node", str(args.actor_num_gpus_per_node),
        "--rollout-num-gpus-per-engine",
        str(getattr(args, "rollout_num_gpus_per_engine", 1)),
        "--colocate",
        (
            "--offload-train"
            if getattr(args, "rl_offload_train", False)
            else "--no-offload-train"
        ),
        "--sglang-mem-fraction-static",
        str(getattr(args, "sglang_mem_fraction_static", 0.4)),
        "--tensor-model-parallel-size", str(tensor_parallel),
        "--pipeline-model-parallel-size", str(pipeline_parallel),
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", str(expert_parallel),
        "--expert-tensor-parallel-size", "1",
        "--prompt-data", str(prompt_path),
        "--input-key", "messages",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--rollout-seed",
        str(getattr(args, "rollout_seed", args.seed + args.learner_id)),
        "--num-rollout",
        str(0 if getattr(args, "eval_only", False) else args.global_rounds),
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
        "--attention-backend", training_attention_backend,
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
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is not None:
        if not yeto_policy_sync:
            raise ValueError("Yeto evaluation requires the external policy boundary")
        if eval_interval <= 0:
            raise ValueError("evaluation interval must be positive")
        if not getattr(args, "eval_only", False) or eval_interval != 1:
            raise ValueError("SSH evaluation must be one separate eval-only run")
        eval_name = getattr(args, "eval_dataset_name", None)
        eval_samples = getattr(args, "eval_samples_per_prompt", None)
        if not eval_name or not isinstance(eval_samples, int) or eval_samples <= 0:
            raise ValueError(
                "evaluation requires a dataset name and positive sample count"
            )
        # Point Miles at the exact normalized prompt file used for training.
        # Leaving no second preparation/copy path is what makes the train/eval
        # task identity mechanically checkable.
        values.extend(
            (
                "--eval-function-path",
                "yeto.rl.miles.generate_rollout",
                "--eval-prompt-data",
                str(eval_name),
                str(prompt_path),
                "--eval-interval",
                str(eval_interval),
                "--skip-eval-before-train",
                "--n-samples-per-eval-prompt",
                str(eval_samples),
                "--log-passrate",
            )
        )
        for flag, name in (
            ("--eval-temperature", "eval_temperature"),
            ("--eval-top-p", "eval_top_p"),
            ("--eval-max-prompt-len", "eval_max_prompt_len"),
            ("--eval-max-response-len", "eval_max_response_len"),
            ("--eval-max-context-len", "eval_max_context_len"),
        ):
            value = getattr(args, name, None)
            if value is not None:
                values.extend((flag, str(value)))
    if expert_full:
        values.extend(
            (
                "--optimizer",
                "adam",
                "--adam-beta1",
                "0.9",
                "--adam-beta2",
                "0.98",
                "--adam-eps",
                str(1e-8),
                "--weight-decay",
                "0",
                "--clip-grad",
                "1.0",
                "--kl-coef",
                "0.001",
            )
        )
    if getattr(args, "sglang_deterministic_inference", True):
        values.append("--sglang-enable-deterministic-inference")
    if recipe == "deepseek-v4-flash":
        values.extend(
            (
                "--transformer-impl",
                "transformer_engine",
                "--qkv-format",
                "bshd",
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
                "--micro-batch-size",
                "1",
                "--train-memory-margin-bytes",
                str(3 * 1024**3),
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-router-freeze-gate",
                "--freeze-e-score-correction-bias",
                "--attention-dropout",
                "0.0",
                "--hidden-dropout",
                "0.0",
                "--sglang-moe-runner-backend",
                "triton",
                "--sglang-disable-shared-experts-fusion",
            )
        )
        if args.lora_targets == "attention-routed-experts":
            values.append("--no-sglang-lora-use-virtual-experts")
    if pipeline_parallel > 1 and layers % pipeline_parallel:
        middle_layers = layers // pipeline_parallel
        remainder = layers - middle_layers * pipeline_parallel
        first_layers = middle_layers + (remainder + 1) // 2
        last_layers = middle_layers + remainder // 2
        values.extend(
            (
                "--decoder-first-pipeline-num-layers",
                str(first_layers),
                "--decoder-last-pipeline-num-layers",
                str(last_layers),
            )
        )
    # Agentic session generation must receive the original message list.  A
    # pre-render here would be rendered again by the session server and breaks
    # both tool-role validation and TITO token identity.
    if not getattr(args, "custom_agent_function_path", None):
        values.append("--apply-chat-template")
    if tensor_parallel > 1:
        values.append("--sequence-parallel")
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
    for flag, name in (
        ("--sglang-tp-size", "sglang_tp_size"),
        ("--sglang-dp-size", "sglang_dp_size"),
        ("--sglang-ep-size", "sglang_ep_size"),
        ("--sglang-attention-backend", "sglang_attention_backend"),
        ("--sglang-page-size", "sglang_page_size"),
        ("--sglang-max-running-requests", "sglang_max_running_requests"),
        ("--sglang-chunked-prefill-size", "sglang_chunked_prefill_size"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            values.extend((flag, str(value)))
    if getattr(args, "use_rollout_routing_replay", False):
        values.append("--use-rollout-routing-replay")
    if args.custom_generate_function_path:
        values.extend(
            (
                "--custom-generate-function-path",
                args.custom_generate_function_path,
            )
        )
    if getattr(args, "custom_agent_function_path", None):
        agent_max_seq_len = getattr(args, "agent_max_seq_len", None) or args.seq_len
        if eval_interval is not None and getattr(
            args, "eval_max_context_len", None
        ) is not None:
            agent_max_seq_len = args.eval_max_context_len
        values.extend(
            (
                "--custom-agent-function-path",
                args.custom_agent_function_path,
                "--max-seq-len",
                str(agent_max_seq_len),
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
            from miles.utils.chat_template_utils import (
                resolve_reasoning_and_tool_call_parser,
            )

            values.extend(("--tito-model", args.tito_model))
            reasoning_parser, tool_call_parser = (
                resolve_reasoning_and_tool_call_parser(args.tito_model)
            )
            if reasoning_parser is not None:
                values.extend(
                    ("--sglang-reasoning-parser", reasoning_parser)
                )
            if tool_call_parser is not None:
                values.extend(
                    ("--sglang-tool-call-parser", tool_call_parser)
                )
        if getattr(args, "tito_allowed_append_roles", None):
            values.append("--tito-allowed-append-roles")
            values.extend(args.tito_allowed_append_roles)
    # Megatron rejects GQA together with MLA.  MLA's compressed KV geometry is
    # expressed by its own rank/head arguments even when the provider exposes
    # fewer query groups than attention heads.
    if query_groups < heads and not multi_latent_attention:
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
        moe_layer_freq = _provider_value(provider, "moe_layer_freq")
        if recipe == "deepseek-v4-flash":
            if isinstance(moe_layer_freq, (list, tuple)):
                mask = tuple(int(value) for value in moe_layer_freq)
                if len(mask) != layers or set(mask) != {1}:
                    raise ValueError(
                        "DeepSeek V4 Flash requires every one of its 43 layers "
                        "to be an MoE layer"
                    )
            elif int(moe_layer_freq) != 1:
                raise ValueError(
                    "DeepSeek V4 Flash requires moe_layer_freq=1"
                )
            # Pinned Miles' moe_freq_type parser accepts the uniform topology
            # as a scalar, not the provider's Python list string.
            moe_layer_freq = 1
        values.extend(
            (
                "--num-experts",
                str(_positive_int(provider, "num_moe_experts")),
                "--moe-ffn-hidden-size",
                str(_positive_int(provider, "moe_ffn_hidden_size")),
                "--moe-router-topk",
                str(_positive_int(provider, "moe_router_topk")),
                "--moe-layer-freq",
                str(moe_layer_freq),
            )
        )
        shared = getattr(provider, "moe_shared_expert_intermediate_size", None)
        if shared is not None:
            values.extend(
                ("--moe-shared-expert-intermediate-size", str(int(shared)))
            )
    if multi_latent_attention:
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


def _verify_eval_dataset_identity(args) -> None:
    """Verify the source bytes attested by an SSH evaluation plan."""

    if getattr(args, "eval_interval", None) is None:
        if getattr(args, "eval_only", False):
            raise ValueError("--eval-only requires evaluation configuration")
        return
    if not getattr(args, "eval_only", False):
        raise ValueError("evaluation configuration requires --eval-only")
    expected = str(getattr(args, "eval_data_sha256", "") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("evaluation dataset requires an immutable SHA256")
    source = Path(args.data).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("evaluation requires one regular local training dataset file")
    from ..provenance import file_sha256

    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(
            f"evaluation dataset SHA256 mismatch: expected {expected}, got {actual}"
        )


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
    rollout_model_path: str | Path | None = None,
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

    clone_only_lora = args.lora_targets == "attention-routed-experts"
    expert_full = int(getattr(args, "expert_full_count", 0) or 0) > 0

    def require_env(name: str, value: str) -> None:
        current = os.environ.get(name)
        if current not in (None, value):
            raise ValueError(f"{name} must be unset or {value}, got {current!r}")
        os.environ[name] = value

    if clone_only_lora or expert_full:
        require_env("YETO_DSV4_EXPERT_CLONE", "1")
    if clone_only_lora:
        require_env("YETO_DSV4_CLONE_ONLY_LORA", "1")
    if clone_only_lora or expert_full:
        fuse_wqa_wkv = os.environ.get("SGLANG_OPT_FUSE_WQA_WKV")
        if fuse_wqa_wkv not in (None, "0"):
            raise ValueError(
                "SGLANG_OPT_FUSE_WQA_WKV must be unset or 0 for V4 attention LoRA"
            )
        os.environ["SGLANG_OPT_FUSE_WQA_WKV"] = "0"
    if expert_full:
        require_env("YETO_DSV4_EXPERT_FULL", "1")
        require_env(
            "YETO_DSV4_EXPERT_FULL_COUNT",
            str(args.expert_full_count),
        )
        require_env("YETO_DSV4_EXPERT_FULL_LR", str(args.expert_full_lr))
        require_env("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")

    if getattr(args, "rl_model_recipe", "generic") == "deepseek-v4-flash":
        from .deepseek_v4_bridge import ensure_deepseek_v4_bridge

        ensure_deepseek_v4_bridge()

    from megatron.bridge import AutoBridge

    model_bridge = AutoBridge.from_hf_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    provider = model_bridge.to_megatron_provider(load_weights=False)
    provider.finalize()
    attention_specs = derive_peft_lora_specs(
        model_path,
        None,
        rank=args.lora_r,
        targets=args.lora_targets,
        trust_remote_code=args.trust_remote_code,
    )
    specs = tuple(attention_specs)
    if expert_full:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=args.trust_remote_code,
        )
        specs = tuple(
            sorted(
                specs
                + expert_full_specs(
                    config,
                    expert_count=args.expert_full_count,
                    expected_selection_sha256=args.expert_selection_sha256,
                    expected_selection_contract_sha256=(
                        args.expert_selection_contract_sha256
                    ),
                )
            )
        )
    canonical_targets = adapter_targets(attention_specs)
    miles_targets = megatron_adapter_targets(
        attention_specs,
        model_bridge,
        standard_grouped_experts=clone_only_lora,
        pipeline_parallel=getattr(args, "pipeline_parallel", 1),
    )
    miles_argv = build_miles_argv(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
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
        miles_args.yeto_rl_rollout_model_revision = (
            getattr(args, "rollout_model_revision", None) or args.model_revision
        )
        miles_args.yeto_rl_data_revision = args.data_revision
        miles_args.yeto_rl_lora_config_hash = lora_config_hash
        miles_args.yeto_rl_layout_hash = layout_hash
        miles_args.yeto_rl_clone_only_lora = clone_only_lora
        if expert_full:
            miles_args.yeto_rl_expected_specs = specs
        if clone_only_lora:
            # The Bridge exposes one representative mapping per packed expert
            # parameter.  Miles needs the sparse canonical clone set at the
            # external policy boundary and reconstructs frozen originals as zeros.
            miles_args.yeto_rl_canonical_lora_names = tuple(
                spec.name for spec in specs
            )
        miles_args.yeto_rl_reward_sha256 = args.reward_sha256
        miles_args.yeto_rl_codex_harness_contract = getattr(
            args,
            "codex_harness_contract",
            None,
        )
        miles_args.yeto_rl_dynamic_sampling_max_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        miles_args.yeto_rl_completed_groups_path = args.completed_groups_path
        miles_args.yeto_rl_event_tape = args.event_tape
        miles_args.yeto_rl_learner_id = args.learner_id
        if getattr(args, "eval_only", False):
            miles_args.yeto_rl_eval_policy_version = args.global_rounds
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
                send_initial_params=not getattr(args, "eval_only", False),
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
    _preflight_codex_harness(args)
    verify_miles_revision(args.miles_root)
    miles_root = str(Path(args.miles_root).expanduser().resolve())
    if miles_root not in sys.path:
        sys.path.insert(0, miles_root)

    from miles.utils.misc import load_function

    load_function(_miles_callable(args.reward_function))
    if args.custom_generate_function_path:
        load_function(args.custom_generate_function_path)
    if args.custom_agent_function_path:
        load_function(args.custom_agent_function_path)
    from . import CODEX_HARNESS_AGENT

    if args.custom_agent_function_path in {
        "yeto_miles_secrlenv.agent.run",
        CODEX_HARNESS_AGENT,
    }:
        from yeto_miles_secrlenv.client import require_daemon_ready

        require_daemon_ready()
        print("[rl] secrlenv episode daemon ready")
    from huggingface_hub import snapshot_download

    from ..models import resolve
    model = resolve(args.model)
    if is_local_reference(model):
        model_path = str(Path(model).expanduser().resolve())
    else:
        model_path = snapshot_download(repo_id=model, revision=args.model_revision)
    rollout_model = resolve(args.rollout_model) if args.rollout_model else model
    if rollout_model == model:
        rollout_model_path = model_path
    elif is_local_reference(rollout_model):
        rollout_model_path = str(Path(rollout_model).expanduser().resolve())
        if not Path(rollout_model_path).is_dir():
            raise ValueError("--rollout-model local directory does not exist")
    else:
        rollout_model_path = snapshot_download(
            repo_id=rollout_model,
            revision=args.rollout_model_revision,
        )
    _verify_eval_dataset_identity(args)
    prompt_path = prepare_prompt_data(
        args.data,
        args.data_revision,
        "~/yeto-rl/prompts.jsonl",
    )
    run_miles(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
        prompt_path=prompt_path,
    )


if __name__ == "__main__":
    main()
