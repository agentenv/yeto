"""Pre-allocation attestation for direct Miles Terminal-Bench SAO launches."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from . import (
    CODEX_OPENENV_AGENT,
    CODEX_OPENENV_AGENT_MODULES,
    CODEX_OPENENV_IDENTITY_ENV,
)
from .codex_backend import QWEN35_08B_MODEL, QWEN35_08B_REVISION
from .tbench_outcome import validate_hmac_key_source

_REWARD_FUNCTION = "openenv_generate.reward_func"
_FILTER_FUNCTION = "openenv_generate.check_terminal_bench_episode"
_GENERATE_FUNCTION = "miles.rollout.generate_hub.agentic_tool_call.generate"
_COMPACTION_ENV = {
    "YETO_CODEX_COMPACTION_ENABLED": "1",
    "YETO_CODEX_COMPACTION_TRIGGER_TOKENS": "6144",
    "YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS": "1024",
    "YETO_CODEX_MAX_COMPACTIONS": "3",
}
_TIMEOUT_ENV = {
    "OPENENV_MAX_ROLLOUT_TIME_SECONDS": "1800",
    "SECRLENV_MAX_ROLLOUT_TIME_SECONDS": "1800",
}


def _direct_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--custom-generate-function-path")
    parser.add_argument("--custom-agent-function-path")
    parser.add_argument("--custom-rm-path")
    parser.add_argument("--dynamic-sampling-filter-path")
    parser.add_argument("--input-key")
    parser.add_argument("--tito-model")
    parser.add_argument("--sao-online-recipe")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--num-gpus-per-node", type=int)
    parser.add_argument("--sglang-mem-fraction-static", type=float)
    parser.add_argument("--sglang-max-total-tokens", type=int)
    parser.add_argument("--sglang-max-mamba-cache-size", type=int)
    parser.add_argument("--sao-compaction", action="store_true")
    parser.add_argument("--sao-one-gpu-island", action="store_true")
    parser.add_argument("--colocate", action="store_true")
    values, _ = parser.parse_known_args(argv)
    return values


def _attest_adapter(miles_root: Path) -> None:
    adapter_dir = miles_root / "examples" / "experimental" / "openenv"
    if miles_root.is_symlink() or not miles_root.is_dir():
        raise ValueError("the pinned Miles root is not a real directory")
    for name in CODEX_OPENENV_AGENT_MODULES:
        source = adapter_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("the Codex OpenEnv adapter source is incomplete")
    adapter_root = str(adapter_dir)
    if adapter_root not in sys.path:
        sys.path.insert(0, adapter_root)
    try:
        adapter = importlib.import_module("codex_openenv_agent_function")
        subprocess_adapter = importlib.import_module(
            "codex_openenv_subprocess_agent_function"
        )
        identity = adapter.codex_openenv_harness_identity()
        binary = adapter.stock._attest_runtime()
    except (ImportError, AttributeError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("cannot attest the Codex OpenEnv adapter") from error
    for module in (adapter, subprocess_adapter):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise TypeError("the Codex OpenEnv adapter has no source identity")
        source = Path(module_file)
        if source.is_symlink() or source.resolve().parent != adapter_dir.resolve():
            raise ValueError("the Codex OpenEnv adapter resolved outside pinned Miles")
    if not callable(getattr(subprocess_adapter, "run", None)):
        raise TypeError("the Codex OpenEnv subprocess entrypoint is missing")
    if adapter._OPENENV_IDENTITY_ENV != CODEX_OPENENV_IDENTITY_ENV:
        raise ValueError("the Codex OpenEnv launch identity drifted")
    expected_identity = {
        name.removeprefix("YETO_CODEX_OPENENV_").lower(): value
        for name, value in CODEX_OPENENV_IDENTITY_ENV.items()
        if name.endswith("_SHA256")
    }
    if identity != expected_identity:
        raise ValueError("the Codex OpenEnv surface identity drifted")
    if (
        adapter.stock._BACKEND_PROFILE.get("model_identifier")
        != QWEN35_08B_MODEL
        or adapter.stock._BACKEND_PROFILE.get("model_revision")
        != QWEN35_08B_REVISION
        or adapter.stock._BACKEND_PROFILE.get("tito_model") != "qwen35"
        or adapter.stock.BACKEND_MODEL != "qwen35"
    ):
        raise ValueError("the Codex OpenEnv qwen35_08b profile drifted")
    if not isinstance(binary, Path) or binary.is_symlink() or not binary.is_file():
        raise ValueError("the attested stock Codex executable is unavailable")


def preflight_tbench_codex_streaming(
    argv: list[str],
    *,
    sao_context: dict[str, Any],
    streaming_runtime: Any,
    miles_root: str | Path,
) -> None:
    """Fail before Miles parses arguments or creates any model/Ray actor."""

    if (
        getattr(streaming_runtime, "trajectory_evidence_kind", None)
        != "terminal-bench-2.1"
        or getattr(streaming_runtime, "trajectory_evidence_schema_version", None)
        != 2
    ):
        raise ValueError(
            "Terminal-Bench direct launch requires trajectory evidence v2"
        )
    if (
        sao_context.get("benchmark") != "terminal-bench-2.1"
        or sao_context.get("model") != QWEN35_08B_MODEL
        or sao_context.get("base_model_revision") != QWEN35_08B_REVISION
        or sao_context.get("rollout_model_revision") != QWEN35_08B_REVISION
    ):
        raise ValueError("Terminal-Bench direct launch model identity drifted")

    options = _direct_options(argv)
    expected = {
        "custom_generate_function_path": _GENERATE_FUNCTION,
        "custom_agent_function_path": CODEX_OPENENV_AGENT,
        "custom_rm_path": _REWARD_FUNCTION,
        "dynamic_sampling_filter_path": _FILTER_FUNCTION,
        "input_key": "messages",
        "tito_model": "qwen35",
        "sao_online_recipe": "coding",
        "max_seq_len": 8192,
        "num_gpus_per_node": 1,
        "sglang_mem_fraction_static": 0.15,
        "sglang_max_total_tokens": 393216,
        "sglang_max_mamba_cache_size": 256,
        "sao_compaction": True,
        "sao_one_gpu_island": True,
        "colocate": True,
    }
    mismatched = [
        name for name, value in expected.items() if getattr(options, name) != value
    ]
    if mismatched:
        raise ValueError(
            "Terminal-Bench direct Miles arguments drifted: "
            + ", ".join(mismatched)
        )

    environment = {
        **CODEX_OPENENV_IDENTITY_ENV,
        **_COMPACTION_ENV,
        **_TIMEOUT_ENV,
    }
    mismatched_environment = [
        name for name, value in environment.items() if os.getenv(name) != value
    ]
    if mismatched_environment:
        raise ValueError(
            "Terminal-Bench trusted container environment drifted: "
            + ", ".join(mismatched_environment)
        )
    validate_hmac_key_source()
    _attest_adapter(Path(miles_root).expanduser().resolve())
