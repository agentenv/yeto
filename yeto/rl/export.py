"""Export a committed RL-AVG checkpoint as a standard PEFT adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

from ..export import parse_checkpoint, validate_against_layout
from ..tensor_io import apply_fragment
from .core import (
    CanonicalLoraState,
    CanonicalTensorSpec,
    build_rl_fragment_layout,
    canonical_layout_hash,
    canonical_lora_config_hash,
    canonical_state,
    policy_tensor_hash,
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
    sync_preset: str = "strict-avg",
    fragments: int = 1,
    pipeline: int = 1,
    local_horizon: int = 1,
    benchmark_learner_budget_steps: int | None = None,
) -> CanonicalLoraState:
    from ..models import resolve

    model = resolve(model)
    checkpoint = parse_checkpoint(checkpoint_path)
    specs = derive_peft_lora_specs(
        model,
        model_revision,
        rank=rank,
        targets=lora_targets,
        trust_remote_code=trust_remote_code,
    )
    canonical_hash = canonical_layout_hash(specs)
    if sync_preset == "strict-avg":
        if fragments != 1 or len(checkpoint.fragments) != 1:
            raise ValueError("RL checkpoint must contain exactly one fragment")
        version, params, _ = checkpoint.fragments[0]
        if version != checkpoint.global_step:
            raise ValueError("RL checkpoint fragment version is not committed")
        if checkpoint.layout_hash is None:
            raise ValueError("RL checkpoint does not contain a canonical layout hash")
        if checkpoint.layout_hash != canonical_hash:
            raise ValueError(
                "RL checkpoint canonical layout hash does not match exporter"
            )
        tensors = tensors_from_flat(params, specs)
    elif sync_preset == "decoupled":
        if fragments < 2 or not 1 <= pipeline <= fragments or local_horizon < 2:
            raise ValueError("invalid decoupled RL export configuration")
        if (
            benchmark_learner_budget_steps is not None
            and benchmark_learner_budget_steps < 1
        ):
            raise ValueError("benchmark learner budget must be positive")
        if benchmark_learner_budget_steps is None and checkpoint.global_step % fragments:
            raise ValueError("decoupled RL checkpoint is not a complete fragment sweep")
        layout = build_rl_fragment_layout(specs, fragments)
        if checkpoint.layout_hash is None:
            raise ValueError("RL checkpoint does not contain a sync layout fingerprint")
        validate_against_layout(checkpoint, layout)
        versions = tuple(version for version, _, _ in checkpoint.fragments)
        terminal_versions = set(
            range(checkpoint.global_step - fragments + 1, checkpoint.global_step + 1)
        )
        valid_versions = (
            versions
            == tuple(
                range(
                    checkpoint.global_step - fragments + 1,
                    checkpoint.global_step + 1,
                )
            )
            if benchmark_learner_budget_steps is None
            else set(versions) == terminal_versions
            and all(
                version > 0 and (version - 1) % fragments == fragment_id
                for fragment_id, version in enumerate(versions)
            )
        )
        if not valid_versions:
            raise ValueError("decoupled RL checkpoint fragment versions are incomplete")
        tensors = {
            spec.name: torch.zeros(spec.shape, dtype=torch.float32)
            for spec in specs
        }
        for fragment, (_, params, _) in zip(layout.fragments, checkpoint.fragments):
            apply_fragment(fragment, params, tensors)
        version = checkpoint.global_step
    else:
        raise ValueError(f"unknown RL sync preset {sync_preset!r}")
    targets = adapter_targets(specs)
    state = canonical_state(
        version,
        tensors,
        base_model_revision=model_revision,
        lora_config_hash=canonical_lora_config_hash(
            rank=rank,
            target_modules=targets,
        ),
        layout_hash=canonical_hash,
    )
    write_peft_adapter(
        state,
        output_dir,
        base_model=model,
        model_revision=model_revision,
        rank=rank,
    )
    if sync_preset == "decoupled":
        provenance = {
            "sync_preset": sync_preset,
            "fragments": fragments,
            "pipeline": pipeline,
            "local_horizon": local_horizon,
            "total_sweeps": (
                checkpoint.global_step // fragments
                if benchmark_learner_budget_steps is None
                else None
            ),
            "total_fragment_steps": checkpoint.global_step,
            "benchmark_learner_budget_steps": benchmark_learner_budget_steps,
            "outer_lr": 0.7,
            "outer_momentum": 0.9,
            "final_fragment_versions": [
                version for version, _, _ in checkpoint.fragments
            ],
            "policy_hash": policy_tensor_hash(state),
            "canonical_layout_hash": canonical_hash,
            "sync_layout_fingerprint": checkpoint.layout_hash,
            "checkpoint_sha256": checkpoint.sha256,
        }
        path = Path(output_dir).expanduser() / "yeto_rl_provenance.json"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(provenance, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
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
    parser.add_argument(
        "--sync-preset",
        choices=["strict-avg", "decoupled"],
        default="strict-avg",
    )
    parser.add_argument("--fragments", type=int, default=1)
    parser.add_argument("--pipeline", type=int, default=1)
    parser.add_argument("--local-horizon", type=int, default=1)
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
        sync_preset=args.sync_preset,
        fragments=args.fragments,
        pipeline=args.pipeline,
        local_horizon=args.local_horizon,
    )
    if args.sync_preset == "decoupled":
        print(
            "exported committed RL cut at outer fragment step "
            f"{state.policy_version} to {args.output_dir}"
        )
    else:
        print(f"exported committed RL policy v{state.policy_version} to {args.output_dir}")


if __name__ == "__main__":
    main()
