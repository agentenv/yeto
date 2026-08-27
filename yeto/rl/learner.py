"""Entrypoint for one pinned-Miles RL island."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
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
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--eval-data-sha256", default=None)
    parser.add_argument("--eval-summary-path", default=None)
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
    parser.add_argument("--num-learners", type=int, default=1)
    parser.add_argument("--learner-generation", type=int, default=0)
    parser.add_argument("--reward-function", required=True)
    parser.add_argument("--reward-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--global-rounds", type=int, required=True)
    parser.add_argument(
        "--parameter-mode",
        choices=["lora", "full"],
        default="lora",
    )
    parser.add_argument(
        "--sync-preset",
        choices=["strict-avg", "decoupled", "dense-full"],
        default="strict-avg",
    )
    parser.add_argument("--fragments", type=int, default=1)
    parser.add_argument("--parameter-layout-sha256", default=None)
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
    parser.add_argument(
        "--secrlenv-max-infrastructure-replacements", type=int, default=None
    )
    parser.add_argument("--rl-offload-train", action="store_true")
    parser.add_argument("--rl-distributed-timeout-minutes", type=int, default=10)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--rollout-max-response-len", type=int, required=True)
    parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default=None)
    parser.add_argument("--custom-generate-function-path", default=None)
    parser.add_argument("--custom-agent-function-path", default=None)
    parser.add_argument("--codex-harness-contract", type=json.loads, default=None)
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=["xhigh"],
        default=None,
    )
    parser.add_argument("--agent-max-seq-len", type=int, default=None)
    parser.add_argument("--use-session-server", action="store_true")
    parser.add_argument("--session-server-ip", default=None)
    parser.add_argument("--session-server-port", type=int, nargs="+", default=None)
    parser.add_argument("--tito-model", default=None)
    parser.add_argument("--codex-backend-profile", default=None)
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
    parser.add_argument("--rollout-num-gpus", type=int, default=None)
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
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument(
        "--lora-targets",
        choices=[
            "auto",
            "attention",
            "attention-routed-experts",
            "all-linear",
        ],
        default=None,
    )
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument("--miles-source-sha256", default=None)
    parser.add_argument("--megatron-ref-load", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    from ..wandb_logger import add_arguments as add_wandb_arguments

    add_wandb_arguments(parser)
    args = parser.parse_args(argv)
    if args.parameter_mode == "lora" and (
        args.lora_r is None or args.lora_targets is None
    ):
        parser.error("LoRA mode requires --lora-r and --lora-targets")
    if (args.parameter_mode == "full") != (args.sync_preset == "dense-full"):
        parser.error("--parameter-mode full requires --sync-preset dense-full")
    return args


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


_DENSE_ISLAND_LOCAL_MILES_FLAGS = frozenset(
    {
        "--rollout-seed",
        "--rollout-engine-base-port",
        "--session-server-port",
        "--sglang-router-port",
        "--sglang-router-prometheus-port",
        "--train-master-base-port",
    }
)

_DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS = frozenset(
    {
        "--session-server-port",
    }
)

_DENSE_FULL_QUANTIZATION_CONFIG_KEYS = frozenset(
    {
        "compression_config",
        "modelopt_quant",
        "quantization",
        "quantization_config",
        "torchao_config",
    }
)


def _require_unquantized_dense_full_rollout_model(model_path: str | Path) -> None:
    """Enable the pinned SGLang BF16 hook fallback only for an exact safe model.

    The public SGLang pin used by the Milestone-1 gate predates the optional
    begin/end weight-update endpoints.  Miles independently rechecks its live
    server arguments before accepting a missing endpoint; this early check
    prevents Yeto from opting into that compatibility path for a quantized
    rollout model in the first place.
    """

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise ValueError("dense full-parameter rollout model has no config.json")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate rollout config key: {name!r}")
            result[name] = value
        return result

    try:
        config = json.loads(
            config_path.read_bytes(),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("dense full-parameter rollout config is malformed") from exc
    if not isinstance(config, dict):
        raise TypeError("dense full-parameter rollout config must be an object")
    configured = sorted(
        name
        for name in _DENSE_FULL_QUANTIZATION_CONFIG_KEYS
        if config.get(name) is not None
    )
    if configured:
        raise ValueError(
            "legacy SGLang weight-update hooks require an unquantized rollout "
            f"model; configured fields: {', '.join(configured)}"
        )


def _dense_full_training_contract_sha256(
    args: Any,
    miles_argv: Sequence[str],
    *,
    model_config_sha256: str,
    prompt_data_sha256: str,
) -> str:
    """Hash shared training semantics while excluding island-local routing."""

    shared_argv = []
    index = 0
    while index < len(miles_argv):
        value = miles_argv[index]
        if value in _DENSE_ISLAND_LOCAL_MILES_FLAGS:
            if index + 1 >= len(miles_argv):
                raise ValueError(f"Miles flag has no value: {value}")
            index += 1
            value_count = 0
            while index < len(miles_argv) and not miles_argv[index].startswith("--"):
                index += 1
                value_count += 1
                if value not in _DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS:
                    break
            if value_count < 1:
                raise ValueError(f"Miles flag has no value: {value}")
            continue
        shared_argv.append(value)
        index += 1
    explicit_rollout_seed = getattr(args, "rollout_seed", None)
    rollout_seed_contract = (
        {"mode": "fixed", "value": explicit_rollout_seed}
        if explicit_rollout_seed is not None
        else {"mode": "base-plus-learner-id", "base_seed": args.seed}
    )
    return _canonical_sha256(
        {
            "schema": "yeto-dense-full-training-contract-v3",
            "shared_miles_argv": shared_argv,
            "rollout_seed_contract": rollout_seed_contract,
            "model": args.model,
            "model_revision": args.model_revision.lower(),
            "model_config_sha256": model_config_sha256,
            "prompt_data_sha256": prompt_data_sha256,
            "data": args.data,
            "data_revision": args.data_revision,
            "evaluation": (
                None
                if getattr(args, "eval_interval", None) is None
                else {
                    "data": getattr(args, "eval_data", None),
                    "data_sha256": getattr(args, "eval_data_sha256", None),
                    "dataset_name": getattr(args, "eval_dataset_name", None),
                    "interval": args.eval_interval,
                    "samples_per_prompt": getattr(
                        args, "eval_samples_per_prompt", None
                    ),
                    "temperature": getattr(args, "eval_temperature", None),
                    "top_p": getattr(args, "eval_top_p", None),
                    "max_prompt_len": getattr(args, "eval_max_prompt_len", None),
                    "max_response_len": getattr(
                        args, "eval_max_response_len", None
                    ),
                    "max_context_len": getattr(args, "eval_max_context_len", None),
                }
            ),
            "yeto_source_sha256": args.source_sha256,
            "miles_source_sha256": args.miles_source_sha256,
            "reward_sha256": args.reward_sha256,
            "codex_harness_contract": getattr(
                args,
                "codex_harness_contract",
                None,
            ),
        }
    )


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


def _preflight_codex_openenv_adapter(args, profile_name: str) -> None:
    """Attest the isolated OpenEnv wrapper inside the pinned Miles source."""

    from . import CODEX_OPENENV_AGENT_MODULES, CODEX_OPENENV_IDENTITY_ENV

    if profile_name != "qwen35_08b":
        raise ValueError(
            "the Codex OpenEnv adapter requires backend profile qwen35_08b"
        )
    adapter_dir = (
        Path(args.miles_root).expanduser().resolve()
        / "examples"
        / "experimental"
        / "openenv"
    )
    for name in CODEX_OPENENV_AGENT_MODULES:
        source = adapter_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("the Codex OpenEnv adapter source is incomplete")
    adapter_root = str(adapter_dir)
    if adapter_root not in sys.path:
        sys.path.insert(0, adapter_root)
    try:
        openenv_adapter = importlib.import_module("codex_openenv_agent_function")
        subprocess_adapter = importlib.import_module(
            "codex_openenv_subprocess_agent_function"
        )
        openenv_identity = openenv_adapter.codex_openenv_harness_identity()
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("cannot attest the Codex OpenEnv adapter") from exc
    for module in (openenv_adapter, subprocess_adapter):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ValueError("the Codex OpenEnv adapter has no source identity")
        source = Path(module_file)
        if source.is_symlink() or source.resolve().parent != adapter_dir:
            raise ValueError("the Codex OpenEnv adapter resolved outside pinned Miles")
    if not callable(getattr(subprocess_adapter, "run", None)):
        raise ValueError("the Codex OpenEnv subprocess entrypoint is missing")
    if openenv_adapter._OPENENV_IDENTITY_ENV != CODEX_OPENENV_IDENTITY_ENV:
        raise ValueError("the Codex OpenEnv launch identity drifted")
    expected_openenv_identity = {
        name.removeprefix("YETO_CODEX_OPENENV_").lower(): value
        for name, value in CODEX_OPENENV_IDENTITY_ENV.items()
        if name.endswith("_SHA256")
    }
    if openenv_identity != expected_openenv_identity:
        raise ValueError("the Codex OpenEnv surface identity drifted")
    openenv_env_mismatched = [
        name
        for name, expected in CODEX_OPENENV_IDENTITY_ENV.items()
        if os.getenv(name) != expected
    ]
    if openenv_env_mismatched:
        raise ValueError(
            "Codex OpenEnv container environment drifted: "
            + ", ".join(openenv_env_mismatched)
        )


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
        CODEX_OPENENV_AGENT,
        CODEX_OPENENV_IDENTITY_ENV,
        CODEX_PACKAGE_MANIFEST_SHA256,
        CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256,
        SIGNED_CODEX_AGENTS,
    )

    contract = getattr(args, "codex_harness_contract", None)
    if args.custom_agent_function_path not in SIGNED_CODEX_AGENTS:
        if contract is not None:
            raise ValueError(
                "--codex-harness-contract requires a signed Codex agent"
            )
        return
    if not isinstance(contract, dict):
        raise ValueError("the signed Codex agent requires its harness contract")
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
    if args.custom_agent_function_path == CODEX_OPENENV_AGENT:
        required.add("openenv_identity_env")
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
    if (
        args.custom_agent_function_path == CODEX_OPENENV_AGENT
        and contract.get("openenv_identity_env") != CODEX_OPENENV_IDENTITY_ENV
    ):
        raise ValueError("stock Codex OpenEnv surface identity drifted")
    from .codex_backend import (
        stock_codex_backend_contract,
        validate_stock_codex_fields,
    )

    profile_name = getattr(args, "codex_backend_profile", None) or args.tito_model
    expected_backend = stock_codex_backend_contract(
        profile_name,
        args.rollout_max_response_len,
    )
    validate_stock_codex_fields(
        tito_model=args.tito_model,
        codex_backend_profile=profile_name,
        rl_model_recipe=args.rl_model_recipe,
        model=args.model,
        model_revision=args.model_revision,
        rollout_model=getattr(args, "rollout_model", None),
        rollout_model_revision=getattr(args, "rollout_model_revision", None),
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        tito_allowed_append_roles=args.tito_allowed_append_roles,
        codex_reasoning_effort=args.codex_reasoning_effort,
        # The signed Qwen rollout backend profile predates full-parameter
        # training and uses ``attention`` only as its legacy tuning-profile
        # discriminator.  Full mode emits no LoRA runtime flags below.
        lora_targets=(
            args.lora_targets
            if getattr(args, "parameter_mode", "lora") == "lora"
            else "attention"
        ),
        expert_full_count=getattr(args, "expert_full_count", 0),
    )
    if contract.get("backend") != expected_backend:
        raise ValueError("stock Codex backend/TITO identity drifted")
    from ..provenance import file_sha256

    # OpenEnv owns and attests its adapter inside the pinned Miles source.  The
    # legacy SecRLEnv-backed Codex adapter remains an optional integration, but
    # it must not be required for the independent Terminal-Bench path.
    if args.custom_agent_function_path != CODEX_OPENENV_AGENT:
        try:
            from yeto_miles_secrlenv import codex_harness_agent

            live_identity = codex_harness_agent.codex_harness_identity()
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise ValueError("cannot attest the Yeto Codex harness adapter") from exc
        identity_names = (
            "base_instructions_sha256",
            "terminal_exec_tool_schema_sha256",
            "submit_tool_schema_sha256",
            "dynamic_tools_schema_sha256",
        )
        if live_identity != {name: contract.get(name) for name in identity_names}:
            raise ValueError("Yeto Codex adapter identity drifted")

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
        "YETO_CODEX_BACKEND_REASONING_EFFORT": expected_backend[
            "reasoning_effort"
        ],
        "YETO_CODEX_BACKEND_THINKING": "enabled",
        "YETO_CODEX_CHAT_TEMPLATE": expected_backend["chat_template"],
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
    if args.custom_agent_function_path == CODEX_OPENENV_AGENT:
        _preflight_codex_openenv_adapter(args, profile_name)


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
    eval_prompt_path: str | Path | None = None,
    provider,
    target_modules: list[str],
    yeto_policy_sync: bool = True,
) -> list[str]:
    """Construct Miles arguments from Bridge's actual model provider."""

    parameter_mode = getattr(args, "parameter_mode", "lora")
    if parameter_mode not in {"lora", "full"}:
        raise ValueError("unsupported RL parameter mode")
    if parameter_mode == "full":
        if target_modules:
            raise ValueError("full-parameter Miles must not select LoRA targets")
    elif not target_modules:
        raise ValueError("PEFT selected no LoRA target modules")

    from .codex_backend import QWEN35_MODEL, QWEN35_REVISION

    qwen35_recipe = (
        getattr(args, "model", None) == QWEN35_MODEL
        and getattr(args, "model_revision", None) == QWEN35_REVISION
    )

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
    if parameter_mode == "full" and data_parallel != 1:
        raise ValueError("dense full-parameter GRPO requires DP=1")
    if parameter_mode == "full":
        rollout_gpus = getattr(args, "rollout_num_gpus", None)
        rollout_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None)
        if (
            args.actor_num_nodes != 1
            or type(rollout_gpus) is not int
            or rollout_gpus < 1
            or type(rollout_gpus_per_engine) is not int
            or rollout_gpus_per_engine < 1
            or rollout_gpus % rollout_gpus_per_engine
        ):
            raise ValueError(
                "Milestone-1 dense full-parameter mode requires one node and "
                "dedicated, evenly partitioned inference engines"
            )
        if qwen35_recipe and (
            rollout_gpus_per_engine != 1
            or getattr(args, "sglang_tp_size", None) not in {None, 1}
        ):
            raise ValueError(
                "pinned Qwen3.5 requires TP1 SGLang inference engines"
            )
        placement_values = [
            "--rollout-num-gpus",
            str(rollout_gpus),
            "--bridge-distributed-weight-sync",
            "--allow-missing-unquantized-weight-update-hooks",
            "--update-weight-transfer-mode",
            "broadcast",
            "--rollout-weight-version-format",
            "yeto-policy",
        ]
        visible_gpus_per_node = args.actor_num_gpus_per_node + rollout_gpus
    else:
        placement_values = ["--colocate"]
        visible_gpus_per_node = args.actor_num_gpus_per_node

    ref_load = str(model_path)
    configured_ref_load = getattr(args, "megatron_ref_load", None)
    if configured_ref_load is not None:
        configured_path = Path(configured_ref_load).expanduser()
        if not configured_path.is_absolute():
            raise ValueError("--megatron-ref-load must be an absolute local path")
        if configured_path.is_symlink() or not configured_path.is_dir():
            raise ValueError("--megatron-ref-load must be a real local directory")
        release_marker = configured_path / "latest_checkpointed_iteration.txt"
        try:
            marker = release_marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(
                "--megatron-ref-load has no readable release marker"
            ) from exc
        if release_marker.is_symlink() or marker != "release":
            raise ValueError("--megatron-ref-load is not a release checkpoint")
        ref_load = str(configured_path.resolve())
    global_batch = (
        args.groups_per_round * args.samples_per_group // args.optimizer_steps
    )
    if global_batch % data_parallel:
        raise ValueError("Miles global batch must divide evenly across DP ranks")
    target_value = ",".join(target_modules)
    recipe = getattr(args, "rl_model_recipe", "generic")
    model_recipe_values: list[str] = []
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
    elif qwen35_recipe:
        model_name = "qwen3_5"
        training_attention_backend = "flash"
        model_recipe_values = [
            "--spec",
            "miles_plugins.models.qwen3_5",
            "get_qwen3_5_spec",
            "--apply-layernorm-1p",
            "--attention-output-gate",
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
        ]
    else:
        model_name = type(provider).__name__
        training_attention_backend = "unfused"

    lora_values: list[str] = []
    if parameter_mode == "lora":
        lora_values = [
            "--lora-rank",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_r),
            "--lora-dropout",
            "0",
            "--lora-type",
            (
                "lora"
                if args.lora_targets == "attention-routed-experts"
                else "canonical_lora"
            ),
            "--target-modules",
            target_value,
            "--lora-base-cpu-backup",
        ]

    values = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", str(rollout_model_path or model_path),
        "--ref-load", ref_load,
        "--megatron-to-hf-mode", "bridge",
        "--model-name", model_name,
        *model_recipe_values,
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
        *lora_values,
        "--actor-num-nodes", str(args.actor_num_nodes),
        "--actor-num-gpus-per-node", str(args.actor_num_gpus_per_node),
        "--num-gpus-per-node", str(visible_gpus_per_node),
        "--rollout-num-gpus-per-engine",
        str(getattr(args, "rollout_num_gpus_per_engine", 1)),
        *placement_values,
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
        "--pin-rollout-manager-to-head",
    ]
    if parameter_mode == "lora":
        values.extend(("--sglang-max-lora-rank", str(args.lora_r)))
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is not None:
        if not yeto_policy_sync:
            raise ValueError("Yeto evaluation requires the external policy boundary")
        if eval_interval <= 0:
            raise ValueError("evaluation interval must be positive")
        eval_only = getattr(args, "eval_only", False)
        dense_train_eval = (
            not eval_only
            and parameter_mode == "full"
            and getattr(args, "sync_preset", None) == "dense-full"
        )
        if eval_only:
            if eval_interval != 1:
                raise ValueError("SSH evaluation must be one separate eval-only run")
            selected_eval_prompt_path = prompt_path
        elif dense_train_eval:
            if eval_interval != args.global_rounds + 1:
                raise ValueError(
                    "dense full evaluation interval must remain outside the "
                    "training-round budget"
                )
            if eval_prompt_path is None:
                raise ValueError("dense full evaluation requires heldout prompt data")
            selected_eval_prompt_path = eval_prompt_path
        else:
            raise ValueError("training-time evaluation is restricted to dense full mode")
        eval_name = getattr(args, "eval_dataset_name", None)
        eval_samples = getattr(args, "eval_samples_per_prompt", None)
        if not eval_name or not isinstance(eval_samples, int) or eval_samples <= 0:
            raise ValueError(
                "evaluation requires a dataset name and positive sample count"
            )
        # Eval-only reuses its sole normalized prompt file.  Dense training
        # instead binds a separately normalized, immutable heldout split.  Its
        # interval sits beyond the rollout budget because the dense policy hook
        # evaluates only the exact initial and terminal published policies.
        values.extend(
            (
                "--eval-function-path",
                "yeto.rl.miles.generate_rollout",
                "--eval-prompt-data",
                str(eval_name),
                str(selected_eval_prompt_path),
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
        policy_sync_path = (
            "yeto.rl.miles_full_parameter_dense."
            "create_miles_full_parameter_dense_sync"
            if parameter_mode == "full"
            else "yeto.rl.miles.create_policy_sync"
        )
        values.extend(
            (
                "--rollout-all-samples-process-path",
                "yeto.rl.miles.queue_completed_groups",
                "--external-policy-sync-path",
                policy_sync_path,
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


def _verify_eval_dataset_identity(args) -> Path | None:
    """Verify the immutable source bytes for an eval-only or dense heldout run."""

    if getattr(args, "eval_interval", None) is None:
        if getattr(args, "eval_only", False) or getattr(args, "eval_data", None):
            raise ValueError("--eval-only requires evaluation configuration")
        return None
    eval_only = getattr(args, "eval_only", False)
    dense_train_eval = (
        not eval_only
        and getattr(args, "parameter_mode", None) == "full"
        and getattr(args, "sync_preset", None) == "dense-full"
    )
    if not eval_only and not dense_train_eval:
        raise ValueError(
            "evaluation configuration requires --eval-only or dense full mode"
        )
    expected = str(getattr(args, "eval_data_sha256", "") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("evaluation dataset requires an immutable SHA256")
    source_value = args.data if eval_only else getattr(args, "eval_data", None)
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("dense full evaluation requires --eval-data")
    source = Path(source_value).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("evaluation requires one regular local dataset file")
    if dense_train_eval and source.resolve() == Path(args.data).expanduser().resolve():
        raise ValueError("dense full evaluation must use a distinct heldout dataset")
    from ..provenance import file_sha256

    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(
            f"evaluation dataset SHA256 mismatch: expected {expected}, got {actual}"
        )
    return source


def _jsonl_record_count(path: str | Path) -> int:
    source = Path(path)
    count = 0
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("evaluation prompt data contains a blank row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("evaluation prompt row is not a JSON object")
                count += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation prompt data is not canonical JSONL") from exc
    if count < 1:
        raise ValueError("evaluation prompt data is empty")
    return count


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
    eval_prompt_path: str | Path | None = None,
    yeto_policy_sync: bool = True,
    extra_argv: Sequence[str] = (),
) -> None:
    """Run one Miles job, optionally with Yeto's external policy boundary."""

    parameter_mode = getattr(args, "parameter_mode", "lora")
    sync_preset = getattr(args, "sync_preset", "strict-avg")
    dense_full = parameter_mode == "full" or sync_preset == "dense-full"
    if dense_full and not (
        parameter_mode == "full" and sync_preset == "dense-full"
    ):
        raise ValueError("dense-full sync and full parameter mode must be selected together")
    if dense_full:
        if not yeto_policy_sync:
            raise ValueError("dense full-parameter GRPO requires Yeto policy sync")
        miles_source_sha256 = getattr(args, "miles_source_sha256", None)
        if (
            type(miles_source_sha256) is not str
            or len(miles_source_sha256) != 64
            or any(value not in "0123456789abcdef" for value in miles_source_sha256)
        ):
            raise ValueError(
                "dense full-parameter GRPO requires an exact Miles source SHA256"
            )
        if (
            args.optimizer_steps != 1
            or getattr(args, "local_horizon", 1) != 1
        ):
            raise ValueError("dense full-parameter GRPO requires H=1")
        if getattr(args, "eval_only", False):
            raise ValueError("Milestone-1 dense full-parameter mode does not run eval-only")
        if (
            type(args.global_rounds) is not int
            or args.global_rounds < 1
            or type(args.fragments) is not int
            or args.fragments < 1
            or args.total_fragment_steps != args.global_rounds * args.fragments
            or not isinstance(args.parameter_layout_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", args.parameter_layout_sha256)
        ):
            raise ValueError(
                "dense full-parameter fragment budget must equal rounds*fragments"
            )
        num_learners = getattr(args, "num_learners", 1)
        generation = getattr(args, "learner_generation", 0)
        if (
            type(num_learners) is not int
            or num_learners < 1
            or type(args.learner_id) is not int
            or args.learner_id not in range(num_learners)
            or type(generation) is not int
            or generation != 0
        ):
            raise ValueError(
                "Milestone-1 dense full-parameter mode requires a fixed generation-0 roster"
            )

    if (
        yeto_policy_sync
        and sync_preset == "decoupled"
        and args.optimizer_steps != 1
    ):
        raise ValueError("decoupled RL requires one optimizer step per rollout")

    clone_only_lora = (
        parameter_mode == "lora"
        and args.lora_targets == "attention-routed-experts"
    )
    expert_full = int(getattr(args, "expert_full_count", 0) or 0) > 0
    if dense_full and expert_full:
        raise ValueError("full-parameter mode cannot also select expert-full tuning")

    def require_env(name: str, value: str) -> None:
        current = os.environ.get(name)
        if current not in (None, value):
            raise ValueError(f"{name} must be unset or {value}, got {current!r}")
        os.environ[name] = value

    if dense_full:
        _require_unquantized_dense_full_rollout_model(
            rollout_model_path or model_path
        )
        require_env("MILES_EXPERIMENTAL_FT_TRAINER", "0")

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
    attention_specs = () if dense_full else derive_peft_lora_specs(
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
    canonical_targets = [] if dense_full else adapter_targets(attention_specs)
    miles_targets = [] if dense_full else megatron_adapter_targets(
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
        eval_prompt_path=eval_prompt_path,
        provider=provider,
        target_modules=miles_targets,
        yeto_policy_sync=yeto_policy_sync,
    )
    miles_argv.extend(extra_argv)
    miles_args = _parse_miles_args(miles_argv)

    if yeto_policy_sync:
        if dense_full:
            from ..provenance import file_sha256

            model_config_path = Path(model_path) / "config.json"
            if not model_config_path.is_file():
                raise ValueError("full-parameter model has no config.json")
            model_config_hash = file_sha256(model_config_path)
            training_contract_hash = _dense_full_training_contract_sha256(
                args,
                miles_argv,
                model_config_sha256=model_config_hash,
                prompt_data_sha256=file_sha256(prompt_path),
            )
            layout_hash = model_config_hash
            lora_config_hash = _canonical_sha256(
                {
                    "schema": "yeto-full-parameter-mode-v1",
                    "model_config_sha256": model_config_hash,
                }
            )
        else:
            from .core import (
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
        miles_args.yeto_rl_parameter_mode = parameter_mode
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
        miles_args.yeto_rl_codex_backend_profile = getattr(
            args,
            "codex_backend_profile",
            None,
        )
        miles_args.yeto_rl_dynamic_sampling_max_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        miles_args.yeto_rl_secrlenv_max_infrastructure_replacements = getattr(
            args, "secrlenv_max_infrastructure_replacements", None
        )
        miles_args.yeto_rl_completed_groups_path = args.completed_groups_path
        miles_args.yeto_rl_event_tape = args.event_tape
        miles_args.yeto_rl_learner_id = args.learner_id
        # W&B reads the same namespace the rest of the yeto context rides in;
        # yeto.rl.wandb_rl starts the island's run off these.
        miles_args.wandb = getattr(args, "wandb", False)
        miles_args.wandb_project = getattr(args, "wandb_project", "yeto")
        miles_args.wandb_entity = getattr(args, "wandb_entity", None)
        miles_args.wandb_mode = getattr(args, "wandb_mode", "online")
        if getattr(args, "eval_only", False):
            miles_args.yeto_rl_eval_policy_version = args.global_rounds
        elif dense_full and getattr(args, "eval_interval", None) is not None:
            if eval_prompt_path is None:
                raise ValueError("dense full evaluation prompt path is unavailable")
            summary_path = Path(
                str(getattr(args, "eval_summary_path", "") or "")
            ).expanduser()
            if (
                not summary_path.is_absolute()
                or summary_path.exists()
                or summary_path.is_symlink()
                or not summary_path.parent.is_dir()
                or summary_path.parent.is_symlink()
            ):
                raise ValueError(
                    "dense full evaluation summary requires a fresh absolute path"
                )
            miles_args.yeto_rl_eval_policy_versions = (0, args.global_rounds)
            miles_args.yeto_rl_eval_dataset_name = args.eval_dataset_name
            miles_args.yeto_rl_eval_prompt_count = _jsonl_record_count(
                eval_prompt_path
            )
            miles_args.yeto_rl_eval_samples_per_prompt = (
                args.eval_samples_per_prompt
            )
            miles_args.yeto_rl_eval_summary_path = str(summary_path)
        miles_args.yeto_rl_sync_preset = sync_preset
        miles_args.external_policy_sync_run_until_stop = sync_preset == "decoupled"
        if dense_full:
            from .dense_sweep_wire import DenseSweepConfig
            from .local_learner import ComponentIdentity
            from .miles_full_parameter_dense import MilesDenseFullParameterConfig

            evidence_parent = (
                Path(args.audit_dir).expanduser()
                if getattr(args, "audit_dir", None)
                else Path(args.completed_groups_path).expanduser().parent
            )
            evidence_parent.mkdir(parents=True, exist_ok=True)
            evidence_directory = Path(
                tempfile.mkdtemp(
                    prefix=f"trajectory-evidence-{args.learner_id}-",
                    dir=evidence_parent,
                )
            )
            if evidence_directory.stat().st_mode & 0o077:
                raise RuntimeError("trajectory evidence directory is not private")
            learner_generations = (0,) * getattr(args, "num_learners", 1)
            miles_args.yeto_rl_model_config_sha256 = model_config_hash
            miles_args.yeto_rl_miles_source_sha256 = args.miles_source_sha256
            miles_args.yeto_rl_dense_training_contract_sha256 = (
                training_contract_hash
            )
            miles_args.external_policy_identity_setter_path = (
                "yeto.rl.miles.set_current_published_policy_identity"
            )
            miles_args.yeto_rl_trajectory_evidence_dir = str(evidence_directory)
            miles_args.yeto_rl_trajectory_evidence_kind = "secrlenv"
            miles_args.yeto_rl_trajectory_evidence_schema_version = (
                2 if bool(getattr(miles_args, "sao_compaction", False)) else 1
            )
            miles_args.yeto_rl_num_fragments = args.fragments
            miles_args.yeto_rl_total_sweeps = args.global_rounds
            miles_args.yeto_rl_total_fragment_steps = args.total_fragment_steps
            miles_args.yeto_rl_learner_generation = 0
            miles_args.yeto_rl_learner_generations = learner_generations
            miles_args.yeto_rl_dense_full_parameter_config = (
                MilesDenseFullParameterConfig(
                    component=ComponentIdentity(
                        role="actor",
                        model_revision=args.model_revision.lower(),
                        config_hash=model_config_hash,
                    ),
                    wire=DenseSweepConfig(
                        syncer_addr=_syncer_address(args.syncer),
                        learner_id=args.learner_id,
                        learner_generation=0,
                        policy_rounds=args.global_rounds,
                        wan_streams=args.wan_streams,
                    ),
                    learner_generations=learner_generations,
                    minimum_fragments=args.fragments,
                    training_contract_hash=training_contract_hash,
                    expected_layout_hash=args.parameter_layout_sha256,
                )
            )
        elif sync_preset == "decoupled":
            from ..protocol import layout_fingerprint
            from .core import build_rl_fragment_layout

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
    miles_root = str(Path(args.miles_root).expanduser().resolve())
    if miles_root not in sys.path:
        sys.path.insert(0, miles_root)
    verify_miles_revision(
        miles_root,
        expected_source_sha256=(
            args.miles_source_sha256
            if args.parameter_mode == "full"
            else None
        ),
    )
    _preflight_codex_harness(args)

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
    eval_source = _verify_eval_dataset_identity(args)
    prompt_path = prepare_prompt_data(
        args.data,
        args.data_revision,
        "~/yeto-rl/prompts.jsonl",
    )
    eval_prompt_path = None
    if eval_source is not None:
        eval_prompt_path = (
            prompt_path
            if getattr(args, "eval_only", False)
            else prepare_prompt_data(
                str(eval_source),
                None,
                "~/yeto-rl/eval-prompts.jsonl",
            )
        )
    run_miles(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
        prompt_path=prompt_path,
        eval_prompt_path=eval_prompt_path,
    )


if __name__ == "__main__":
    main()
