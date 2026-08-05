"""Load one completed Decoupled RL adapter as a fresh phase policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from ..adapter_lifecycle import directory_sha256
from ..models import resolve
from .core import (
    CanonicalLoraState,
    canonical_lora_config_hash,
    canonical_state,
    policy_tensor_hash,
)
from .export import adapter_targets

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read initial adapter {description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"initial adapter {description} must be a JSON object")
    return value


def load_initial_adapter(
    path: str | Path,
    expected_sha256: str,
    *,
    model: str,
    expected: CanonicalLoraState,
) -> CanonicalLoraState:
    """Validate and load a final Yeto Decoupled adapter as policy version zero."""

    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError(
            "initial adapter SHA256 must be 64 lowercase hexadecimal characters"
        )
    root = Path(path).expanduser().resolve()
    actual_sha256 = directory_sha256(root)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"initial adapter SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    config = _read_object(root / "adapter_config.json", "adapter_config.json")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("initial adapter must declare peft_type=LORA")
    if str(config.get("task_type", "")).upper() != "CAUSAL_LM":
        raise ValueError("initial adapter must declare task_type=CAUSAL_LM")
    if config.get("base_model_name_or_path") != resolve(model):
        raise ValueError("initial adapter base model differs from this RL phase")
    if config.get("revision") != expected.base_model_revision:
        raise ValueError("initial adapter model revision differs from this RL phase")

    targets = config.get("target_modules")
    expected_targets = adapter_targets(expected.specs)
    if (
        not isinstance(targets, list)
        or any(not isinstance(target, str) for target in targets)
        or sorted(targets) != expected_targets
    ):
        raise ValueError("initial adapter target modules differ from this RL phase")
    rank = config.get("r")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("initial adapter has an invalid LoRA rank")
    if config.get("lora_alpha") != rank:
        raise ValueError("initial adapter LoRA alpha must equal its rank")
    if config.get("lora_dropout") != 0.0:
        raise ValueError("initial adapter LoRA dropout must be zero")
    if config.get("bias") != "none":
        raise ValueError("initial adapter LoRA bias must be none")
    for field in ("use_rslora", "fan_in_fan_out", "use_dora"):
        if config.get(field, False) is not False:
            raise ValueError(f"initial adapter {field} must be false")
    for field in ("rank_pattern", "alpha_pattern"):
        if config.get(field, {}) != {}:
            raise ValueError(f"initial adapter {field} must be empty")
    if (
        canonical_lora_config_hash(rank=rank, target_modules=targets)
        != expected.lora_config_hash
    ):
        raise ValueError("initial adapter LoRA rank differs from this RL phase")

    weights = root / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("initial adapter has no adapter_model.safetensors")
    try:
        tensors = load_file(weights, device="cpu")
    except Exception as error:
        raise ValueError("cannot read initial adapter safetensors") from error
    state = canonical_state(
        0,
        tensors,
        base_model_revision=expected.base_model_revision,
        lora_config_hash=expected.lora_config_hash,
        expected_specs=expected.specs,
    )

    provenance = _read_object(
        root / "yeto_rl_provenance.json",
        "yeto_rl_provenance.json",
    )
    if provenance.get("sync_preset") != "decoupled":
        raise ValueError("initial adapter is not a completed Decoupled RL export")
    if provenance.get("policy_hash") != policy_tensor_hash(state):
        raise ValueError(
            "initial adapter provenance policy hash does not match weights"
        )
    return state
