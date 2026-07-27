"""Export an authoritative strict-RL checkpoint as a standard PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ..protocol import layout_fingerprint
from .checkpoint import validate_rl_final_checkpoint
from .core import (
    CanonicalTensorSpec,
    build_avg_layout,
    canonical_state,
    tensors_from_flat,
)
from .manifest import canonical_json, manifest_sha256, validate_manifest


def _rope_parameters(config):
    return getattr(config, "rope_parameters", None) or getattr(
        config, "rope_scaling", None
    )


def _rope_theta(config) -> int:
    rope_parameters = _rope_parameters(config)
    value = (
        rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000))
        if isinstance(rope_parameters, Mapping)
        else getattr(config, "rope_theta", 10000)
    )
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or value <= 0
        or int(value) != value
    ):
        raise ValueError("RL v0 requires an integral positive RoPE theta")
    return int(value)


def _rope_scaling_factor(config) -> float | None:
    rope_scaling = _rope_parameters(config)
    if rope_scaling is None:
        return None
    if not isinstance(rope_scaling, Mapping):
        raise ValueError("RL v0 has an unsupported RoPE configuration")
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type == "default":
        factor = rope_scaling.get("factor", 1.0)
        if type(factor) not in {int, float} or factor != 1.0:
            raise ValueError("RL v0 has an unsupported default RoPE factor")
        return None
    factor = rope_scaling.get("factor")
    expected = {
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    if (
        getattr(config, "model_type", None) != "llama"
        or rope_type != "llama3"
        or type(factor) not in {int, float}
        or not math.isfinite(factor)
        or factor <= 0
        or any(rope_scaling.get(name) != value for name, value in expected.items())
    ):
        raise ValueError("RL v0 supports only the pinned Miles Llama 3 RoPE mapping")
    return float(factor)


def _transformers_model_family(config):
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

    families = {
        LlamaConfig: (LlamaForCausalLM, LlamaRMSNorm, LlamaMLP),
        Qwen2Config: (Qwen2ForCausalLM, Qwen2RMSNorm, Qwen2MLP),
        Qwen3Config: (Qwen3ForCausalLM, Qwen3RMSNorm, Qwen3MLP),
    }
    family = families.get(type(config))
    if family is None:
        raise ValueError("RL v0 requires an exact supported Transformers config class")
    if getattr(config, "architectures", None) != [family[0].__name__]:
        raise ValueError("RL v0 model architectures must name the exact causal-LM class")
    return family


def _validate_transformers_model(model, family) -> None:
    model_class, norm_class, mlp_class = family
    if type(model) is not model_class:
        raise ValueError("RL v0 requires the exact supported Transformers model class")
    body = getattr(model, "model", None)
    layers = list(getattr(body, "layers", ()))
    if not layers or type(getattr(body, "norm", None)) is not norm_class:
        raise ValueError("RL v0 requires the supported RMSNorm decoder architecture")
    if any(
        type(getattr(layer, "input_layernorm", None)) is not norm_class
        or type(getattr(layer, "post_attention_layernorm", None)) is not norm_class
        or type(getattr(layer, "mlp", None)) is not mlp_class
        for layer in layers
    ):
        raise ValueError("RL v0 requires the supported RMSNorm gated dense MLP")


def _reject_unsupported_config(config) -> None:
    if hasattr(config, "text_config"):
        raise ValueError("RL v0 does not support multimodal base models")
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "model_type", None) not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("RL v0 supports only dense Llama, Qwen2, and Qwen3 models")
    for name in (
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "num_experts_per_tok",
    ):
        value = getattr(text_config, name, None)
        if isinstance(value, int) and value > 0:
            raise ValueError("RL v0 does not support MoE base models")
    if getattr(config, "quantization_config", None):
        raise ValueError("RL v0 does not support quantized base models")
    if getattr(text_config, "hidden_act", None) != "silu":
        raise ValueError("RL v0 requires a SwiGLU-compatible base model")
    norm_epsilon = getattr(text_config, "rms_norm_eps", None)
    if (
        type(norm_epsilon) not in {int, float}
        or not math.isfinite(norm_epsilon)
        or norm_epsilon <= 0
    ):
        raise ValueError("RL v0 requires RMSNorm with a positive epsilon")
    if getattr(text_config, "mlp_bias", False):
        raise ValueError("RL v0 does not support MLP bias parameters")
    if text_config.model_type != "qwen2" and getattr(
        text_config, "attention_bias", False
    ):
        raise ValueError("RL v0 supports attention bias only for Qwen2")
    layer_types = getattr(text_config, "layer_types", None)
    if getattr(text_config, "use_sliding_window", False) or (
        layer_types is not None
        and any(layer_type != "full_attention" for layer_type in layer_types)
    ):
        raise ValueError("RL v0 does not support sliding-window attention")
    _rope_scaling_factor(text_config)


def derive_peft_lora_specs(
    model: str,
    revision: str | None,
    *,
    rank: int,
    target_modules: Sequence[str],
    trust_remote_code: bool = False,
) -> tuple[CanonicalTensorSpec, ...]:
    """Build the exact on-disk PEFT LoRA layout without allocating base weights."""

    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    family = _transformers_model_family(config)
    _reject_unsupported_config(config)
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
        _validate_transformers_model(base, family)
        adapter = get_peft_model(
            base,
            LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=list(target_modules),
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    state = get_peft_model_state_dict(adapter)
    specs = tuple(
        CanonicalTensorSpec(name, tuple(int(dim) for dim in tensor.shape), tensor.numel())
        for name, tensor in sorted(state.items())
    )
    if not specs:
        raise ValueError("PEFT found no LoRA target modules in the pinned base model")
    return specs


def specs_manifest(specs: Sequence[CanonicalTensorSpec]) -> dict[str, Any]:
    specs = tuple(sorted(specs, key=lambda spec: spec.name))
    fingerprint = layout_fingerprint(build_avg_layout(specs)).hex()
    return {
        "layout_fingerprint": fingerprint,
        "tensors": [
            {"name": spec.name, "shape": list(spec.shape), "numel": spec.numel}
            for spec in specs
        ],
    }


def specs_from_manifest(manifest: Mapping[str, Any]) -> tuple[CanonicalTensorSpec, ...]:
    layout = manifest.get("canonical_lora") or {}
    raw = layout.get("tensors")
    if not isinstance(raw, list):
        raise ValueError("RL manifest has no canonical LoRA tensor layout")
    specs = tuple(
        CanonicalTensorSpec(
            item["name"],
            tuple(int(dim) for dim in item["shape"]),
            int(item["numel"]),
        )
        for item in raw
    )
    expected = specs_manifest(specs)
    if expected != layout:
        raise ValueError("RL manifest canonical LoRA layout is not normalized")
    return specs


def _manifest_text(value: str | Path | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        return canonical_json(value)
    if isinstance(value, str) and value.startswith("{"):
        return value
    path = Path(value)
    return path.read_text(encoding="utf-8") if path.is_file() else str(value)


def _adapter_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    base = manifest["base_model"]
    lora = manifest["lora"]
    return {
        "base_model_name_or_path": base["identifier"],
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora["alpha"],
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora["rank"],
        "revision": base["revision"],
        "target_modules": lora["target_modules"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _verify_exported_adapter(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    policy_version: int,
    expected_policy_sha256: str,
) -> None:
    """Reload the standard adapter files and recompute their canonical hash."""

    from peft import PeftConfig
    from safetensors.torch import load_file

    output = Path(output_dir)
    specs = specs_from_manifest(manifest)
    fingerprint = manifest["canonical_lora"]["layout_fingerprint"]
    state = canonical_state(
        policy_version,
        load_file(output / "adapter_model.safetensors", device="cpu"),
        expected_specs=specs,
        expected_layout_fingerprint=fingerprint,
    )
    if state.policy_hash != expected_policy_sha256:
        raise RuntimeError("exported PEFT safetensors failed canonical hash verification")
    PeftConfig.from_pretrained(output)


def _verify_export_in_subprocess(
    output: Path,
    manifest: Mapping[str, Any],
    policy_version: int,
    expected_policy_sha256: str,
) -> None:
    payload = {
        "output": str(output),
        "manifest": manifest,
        "policy_version": policy_version,
        "policy_sha256": expected_policy_sha256,
    }
    code = (
        "import json,sys;"
        "from yeto.rl.export import _verify_exported_adapter;"
        "p=json.load(sys.stdin);"
        "_verify_exported_adapter(p['output'],p['manifest'],"
        "p['policy_version'],p['policy_sha256'])"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", code],
            input=json.dumps(payload, separators=(",", ":")),
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown verification error").strip()
        raise RuntimeError(f"clean-process PEFT verification failed: {detail}") from exc


def export_rl_checkpoint(
    checkpoint_path: str | Path,
    manifest_value: str | Path | Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate, export, reread, and return the artifact provenance."""

    from safetensors.torch import save_file

    text = _manifest_text(manifest_value)
    manifest = validate_manifest(text)
    manifest_hash = manifest_sha256(text)
    checkpoint = validate_rl_final_checkpoint(checkpoint_path, manifest_hash)
    expected_specs = specs_from_manifest(manifest)
    base = manifest["base_model"]
    lora = manifest["lora"]
    rebuilt_specs = derive_peft_lora_specs(
        base["identifier"],
        base["revision"],
        rank=lora["rank"],
        target_modules=lora["target_modules"],
        trust_remote_code=bool(base.get("trust_remote_code")),
    )
    if rebuilt_specs != expected_specs:
        raise ValueError("pinned base model no longer reconstructs the manifest LoRA layout")
    if checkpoint.roster_size != manifest["workload"]["learners"]:
        raise ValueError("RL checkpoint roster does not match the run manifest")
    if len(checkpoint.fragments) != 1 or checkpoint.versions != (checkpoint.global_step,):
        raise ValueError("RL checkpoint is not a committed single-fragment policy")

    fingerprint = layout_fingerprint(build_avg_layout(expected_specs)).hex()
    if fingerprint != checkpoint.layout_fingerprint:
        raise ValueError("RL checkpoint layout fingerprint does not match the manifest")
    flat = torch.tensor(checkpoint.fragments[0], dtype=torch.float32)
    state = canonical_state(
        checkpoint.global_step,
        tensors_from_flat(flat, expected_specs),
        expected_specs=expected_specs,
        expected_layout_fingerprint=fingerprint,
    )
    if state.policy_hash != checkpoint.policy_sha256:
        raise ValueError("RL checkpoint policy does not match the canonical PEFT layout")

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "adapter_model.safetensors"
    temporary_weights = weights_path.with_name(weights_path.name + ".tmp")
    save_file(dict(state.tensors), temporary_weights)
    os.replace(temporary_weights, weights_path)
    _atomic_text(
        output / "adapter_config.json",
        json.dumps(_adapter_config(manifest), sort_keys=True, indent=2) + "\n",
    )

    _verify_export_in_subprocess(
        output,
        manifest,
        checkpoint.global_step,
        state.policy_hash,
    )

    checkpoint_bytes = Path(checkpoint_path).read_bytes()
    provenance = {
        "schema": 1,
        "artifact_kind": "yeto-rl-peft-lora",
        "run_manifest_sha256": manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "global_step": checkpoint.global_step,
        "roster_size": checkpoint.roster_size,
        "layout_fingerprint": fingerprint,
        "policy_sha256": state.policy_hash,
        "base_model": base,
        "miles": manifest["miles"],
    }
    _atomic_text(
        output / "yeto_rl_provenance.json",
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
    )
    return provenance


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="yeto-rl-export",
        description="Export a finalized strict-RL checkpoint as a PEFT LoRA adapter.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    provenance = export_rl_checkpoint(args.checkpoint, args.manifest, args.output_dir)
    print(
        f"exported RL policy v{provenance['global_step']} "
        f"({provenance['policy_sha256']}) to {Path(args.output_dir).expanduser()}"
    )


if __name__ == "__main__":
    main()
