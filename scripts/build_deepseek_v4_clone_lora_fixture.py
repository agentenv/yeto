#!/usr/bin/env python3
"""Build a full BF16 E288 adapter fixture directly beside the checkpoint.

The fixture preserves the production policy layout: attention and original
expert adapters are exact zero, while all 32 cloned experts receive a small
deterministic LoRA delta.  It is intended only for SGLang load/update runtime
validation, not as a training initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import torch


_EXPERT = re.compile(r"\.mlp\.experts\.(?P<expert>\d+)\.")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--b-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--magnitude", type=float, default=0.01)
    parser.add_argument(
        "--scope",
        choices=("attention-routed-experts", "routed-experts", "attention"),
        default="attention-routed-experts",
        help=(
            "Build the production attention+clone-expert policy or a physical "
            "expert-only isolation fixture."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _args()
    if args.rank <= 0 or not 0 < args.magnitude <= 0.1:
        raise ValueError("fixture rank/magnitude is outside the audited envelope")

    from peft import LoraConfig
    from safetensors.torch import save_file

    from yeto.rl.deepseek_v4_bridge import ensure_deepseek_v4_bridge
    from yeto.rl.export import adapter_targets, derive_peft_lora_specs

    ensure_deepseek_v4_bridge()
    specs = derive_peft_lora_specs(
        args.model,
        None,
        rank=args.rank,
        targets="attention-routed-experts",
        trust_remote_code=True,
    )
    if len(specs) != 74_518:
        raise RuntimeError(f"unexpected E288 LoRA layout size {len(specs)}")
    if args.scope == "routed-experts":
        specs = tuple(spec for spec in specs if _EXPERT.search(spec.name))
        if len(specs) != 74_304:
            raise RuntimeError(
                f"unexpected E288 expert-only LoRA layout size {len(specs)}"
            )
    elif args.scope == "attention":
        specs = tuple(spec for spec in specs if not _EXPERT.search(spec.name))
        if len(specs) != 214:
            raise RuntimeError(
                f"unexpected E288 attention-only LoRA layout size {len(specs)}"
            )

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"fixture output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    tensors = {}
    clone_tensors = 0
    original_tensors = 0
    attention_tensors = 0
    for spec in specs:
        value = torch.zeros(spec.shape, dtype=torch.bfloat16)
        match = _EXPERT.search(spec.name)
        if match is None:
            attention_tensors += 1
            if args.scope == "attention":
                sign = args.b_sign if ".lora_B.weight" in spec.name else 1
                value.fill_(sign * args.magnitude)
        elif int(match.group("expert")) < 256:
            original_tensors += 1
        else:
            clone_tensors += 1
            sign = args.b_sign if ".lora_B.weight" in spec.name else 1
            value.fill_(sign * args.magnitude)
        tensors[spec.name] = value

    weights = output / "adapter_model.safetensors"
    temporary_weights = output / "adapter_model.safetensors.tmp"
    save_file(tensors, temporary_weights, metadata={"format": "pt"})
    os.replace(temporary_weights, weights)
    del tensors

    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=adapter_targets(specs),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=True,
        base_model_name_or_path=args.model,
    )
    with tempfile.TemporaryDirectory(dir=output) as temporary_dir:
        config.save_pretrained(temporary_dir)
        os.replace(
            Path(temporary_dir) / "adapter_config.json",
            output / "adapter_config.json",
        )

    manifest = {
        "schema": 1,
        "purpose": "E288 clone-only SGLang runtime fixture",
        "scope": args.scope,
        "rank": args.rank,
        "b_sign": args.b_sign,
        "magnitude": args.magnitude,
        "dtype": "bfloat16",
        "tensor_count": len(specs),
        "attention_tensors": attention_tensors,
        "original_expert_tensors": original_tensors,
        "clone_expert_tensors": clone_tensors,
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": _sha256(weights),
    }
    manifest_path = output / "yeto_fixture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
