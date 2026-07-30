"""Export a committed RL-AVG checkpoint as a standard PEFT adapter."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ..export import parse_checkpoint
from .core import (
    CanonicalLoraState,
    CanonicalTensorSpec,
    canonical_layout_hash,
    canonical_lora_config_hash,
    canonical_state,
    tensors_from_flat,
)


def _rl_model_factory(config):
    """Use the model class declared by the pinned checkpoint when available."""

    import transformers

    for architecture in getattr(config, "architectures", None) or ():
        factory = getattr(transformers, architecture, None)
        if factory is not None:
            return factory
    return transformers.AutoModelForCausalLM


def _rl_model_from_config(config, *, trust_remote_code: bool):
    import transformers

    factory = _rl_model_factory(config)
    if factory is transformers.AutoModelForCausalLM:
        return factory.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
    return factory._from_config(config)


def target_modules(choice: str, config) -> str:
    """Reuse Yeto's model-driven public LoRA target semantics."""

    from ..learner import resolve_lora_targets

    return resolve_lora_targets(choice, config)


def derive_peft_lora_specs(
    model: str,
    revision: str | None,
    *,
    rank: int,
    targets: str | Sequence[str],
    trust_remote_code: bool = False,
) -> tuple[CanonicalTensorSpec, ...]:
    """Build PEFT's actual LoRA tensor contract without allocating weights."""

    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if isinstance(targets, str) and targets in {
        "auto",
        "attention",
        "all-linear",
    }:
        targets = target_modules(targets, config)
    with init_empty_weights():
        base = _rl_model_from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
        if isinstance(targets, str) and targets != "all-linear":
            pattern = re.compile(targets)
            targets = sorted(
                name for name, _ in base.named_modules() if pattern.fullmatch(name)
            )
        adapter = get_peft_model(
            base,
            LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=targets,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    state = get_peft_model_state_dict(adapter)
    if not state:
        raise ValueError("PEFT found no LoRA tensors for the requested model")
    return tuple(
        CanonicalTensorSpec(
            name,
            tuple(int(dim) for dim in tensor.shape),
            "float32",
            tensor.numel(),
        )
        for name, tensor in sorted(state.items())
    )


def adapter_targets(specs: Sequence[CanonicalTensorSpec]) -> list[str]:
    targets = {
        spec.name.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1]
        for spec in specs
    }
    return sorted(targets)


def write_peft_adapter(
    state: CanonicalLoraState,
    output_dir: str | Path,
    *,
    base_model: str,
    model_revision: str | None,
    rank: int,
) -> None:
    """Write the two standard PEFT adapter files."""

    from peft import LoraConfig
    from peft.utils import CONFIG_NAME
    from safetensors.torch import save_file

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "adapter_model.safetensors"
    temporary = output / "adapter_model.safetensors.tmp"
    save_file(dict(state.tensors), temporary)
    os.replace(temporary, weights)

    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=adapter_targets(state.specs),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=True,
        base_model_name_or_path=base_model,
        revision=model_revision,
    )
    with tempfile.TemporaryDirectory(dir=output) as temporary_dir:
        config.save_pretrained(temporary_dir)
        os.replace(Path(temporary_dir) / CONFIG_NAME, output / CONFIG_NAME)


def export_rl_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    model_revision: str,
    rank: int,
    lora_targets: str = "auto",
    trust_remote_code: bool = False,
) -> CanonicalLoraState:
    from ..models import resolve

    model = resolve(model)
    checkpoint = parse_checkpoint(checkpoint_path)
    if len(checkpoint.fragments) != 1:
        raise ValueError("RL checkpoint must contain exactly one fragment")
    version, params, _ = checkpoint.fragments[0]
    if version != checkpoint.global_step:
        raise ValueError("RL checkpoint fragment version is not committed")

    specs = derive_peft_lora_specs(
        model,
        model_revision,
        rank=rank,
        targets=lora_targets,
        trust_remote_code=trust_remote_code,
    )
    layout_hash = canonical_layout_hash(specs)
    if checkpoint.layout_hash is None:
        raise ValueError("RL checkpoint does not contain a canonical layout hash")
    if checkpoint.layout_hash != layout_hash:
        raise ValueError("RL checkpoint canonical layout hash does not match exporter")
    targets = adapter_targets(specs)
    state = canonical_state(
        version,
        tensors_from_flat(params, specs),
        base_model_revision=model_revision,
        lora_config_hash=canonical_lora_config_hash(
            rank=rank,
            target_modules=targets,
        ),
        layout_hash=layout_hash,
    )
    write_peft_adapter(
        state,
        output_dir,
        base_model=model,
        model_revision=model_revision,
        rank=rank,
    )
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="yeto-rl-export",
        description="Export a committed Yeto RL checkpoint as a PEFT adapter.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    state = export_rl_checkpoint(
        args.checkpoint,
        args.output_dir,
        model=args.model,
        model_revision=args.model_revision,
        rank=args.lora_r,
        lora_targets=args.lora_targets,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"exported committed RL policy v{state.policy_version} to {args.output_dir}")


if __name__ == "__main__":
    main()
