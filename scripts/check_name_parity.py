#!/usr/bin/env python3
"""Cross-backend layout parity check: for a given HF model, compare the
trainable adapter FQNs (and shapes) the torch/peft learner would sync
against what the MLX backend's registry reports.

The DiLoCo fragment layout is built independently on every learner from its
(name, numel) list; identical lists => identical fragments => a Mac (MLX)
and a CUDA island (torch) can merge on one syncer. Run this before any
heterogeneous run with a new architecture:

    python scripts/check_name_parity.py --model lfm25-230m [--lora-targets auto]

Exit 0 iff names AND shapes match exactly.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def peft_names(model_id: str, targets: str, r: int, alpha: int) -> dict[str, tuple]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from yeto.learner import normalize_param_name, resolve_lora_targets

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=resolve_lora_targets(targets, config),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return {
        normalize_param_name(n): tuple(p.shape)
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def mlx_names(model_id: str, targets: str, r: int, alpha: int) -> dict[str, tuple]:
    from transformers import AutoConfig

    from yeto.mlx.learner import import_mlx_lm, mlx_config_shim
    from yeto.mlx.lora import attach_lora

    mlx_lm = import_mlx_lm()

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model, _ = mlx_lm.load(model_id, model_config=mlx_config_shim(config))
    registry = attach_lora(model, config, targets, r, alpha)
    return {n: tuple(i.shape) for n, i in registry.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora-targets", default="auto")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args()

    from yeto.models import resolve

    model_id = resolve(args.model)
    print(f"[parity] loading {model_id} via torch/peft ...")
    torch_side = peft_names(model_id, args.lora_targets, args.lora_r, args.lora_alpha)
    print(f"[parity] torch/peft: {len(torch_side)} trainable tensors")
    print(f"[parity] loading {model_id} via mlx ...")
    mlx_side = mlx_names(model_id, args.lora_targets, args.lora_r, args.lora_alpha)
    print(f"[parity] mlx: {len(mlx_side)} trainable tensors")

    missing = sorted(set(torch_side) - set(mlx_side))
    extra = sorted(set(mlx_side) - set(torch_side))
    shape_diff = sorted(
        n for n in set(torch_side) & set(mlx_side) if torch_side[n] != mlx_side[n]
    )
    if missing:
        print(f"[parity] FAIL: {len(missing)} names only on torch side, e.g.:")
        for n in missing[:8]:
            print(f"    {n} {torch_side[n]}")
    if extra:
        print(f"[parity] FAIL: {len(extra)} names only on mlx side, e.g.:")
        for n in extra[:8]:
            print(f"    {n} {mlx_side[n]}")
    if shape_diff:
        print(f"[parity] FAIL: {len(shape_diff)} shape mismatches, e.g.:")
        for n in shape_diff[:8]:
            print(f"    {n}: torch {torch_side[n]} vs mlx {mlx_side[n]}")
    if missing or extra or shape_diff:
        return 1
    print(f"[parity] OK: {len(torch_side)} tensors, names and shapes identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
