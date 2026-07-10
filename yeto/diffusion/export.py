"""Export a syncer checkpoint as a Diffusers/PEFT adapter artifact.

The syncer checkpoint stores the authoritative merged trainable tensors as
flat fragments. It does not contain a Diffusers pipeline or PEFT directory, so
this module rebuilds the same diffusion trainable layout used by the learner,
applies the checkpoint fragments, and delegates artifact saving to the normal
diffusion learner or external-adapter hooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..export import Checkpoint, parse_checkpoint, validate_against_layout
from ..fragments import FragmentLayout, build_layout
from ..tensor_io import apply_fragment
from .learner import (
    DIFFUSION_ADAPTER_METADATA_FILE,
    load_diffusion_adapter,
    load_pipeline,
    save_adapters,
    trainable_params,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="yeto-diffusion-export",
        description="Export a syncer checkpoint into a Diffusers/PEFT adapter directory.",
    )
    p.add_argument("--checkpoint", required=True, help="path to the syncer checkpoint file")
    p.add_argument("--model", required=True, help="diffusers repo id or alias from yeto.models")
    p.add_argument(
        "--diffusion-adapter",
        default=None,
        help="module:factory or file.py:factory used by the training run",
    )
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
        help="adapter placement used during training (layout must match)",
    )
    p.add_argument("--fragments", type=int, default=8, help="P used during training")
    p.add_argument(
        "--fragment-pattern",
        choices=["binpack", "strided"],
        default="binpack",
        help="fragment pattern used during training",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def _annotate_export_metadata(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    checkpoint: Checkpoint,
    layout: FragmentLayout,
    args,
) -> None:
    path = Path(output_dir).expanduser() / DIFFUSION_ADAPTER_METADATA_FILE
    if not path.exists():
        raise RuntimeError(f"diffusion artifact save did not write {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["export"] = {
        "source": "syncer-checkpoint",
        "checkpoint": Path(checkpoint_path).name,
        "global_step": checkpoint.global_step,
        "requested_fragments": args.fragments,
        "fragments": layout.num_fragments,
        "fragment_pattern": args.fragment_pattern,
        "fragment_versions": [version for version, _, _ in checkpoint.fragments],
        "ledger": {
            str(learner_id): {
                "merges": merges,
                "steps": steps,
                "units": units,
            }
            for learner_id, (merges, steps, units) in sorted(checkpoint.ledger.items())
        },
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_checkpoint(args) -> tuple[Checkpoint, FragmentLayout, dict[str, torch.Tensor]]:
    """Rebuild a diffusion trainable layout and export the merged checkpoint."""
    checkpoint = parse_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    adapter = load_diffusion_adapter(args.diffusion_adapter)

    # Export is deliberately single-process and unwrapped. The learner's
    # normalized trainable names make this layout identical to DDP/FSDP runs.
    args.shard = "ddp"
    args.loss_function = "flow_matching"
    pipe = load_pipeline(args, device, adapter)
    params = trainable_params(pipe, adapter)
    if not params:
        raise RuntimeError("no trainable diffusion parameters; check the export flags")
    layout = build_layout(
        [(name, param.numel()) for name, param in params.items()],
        args.fragments,
        args.fragment_pattern,
    )
    try:
        validate_against_layout(checkpoint, layout)
    except ValueError as exc:
        raise ValueError(
            f"{exc} Diffusion export also requires the same --lora-alpha, "
            "--lora-targets and --diffusion-adapter used during training."
        ) from exc

    for fragment, (_, flat_params, _) in zip(layout.fragments, checkpoint.fragments):
        apply_fragment(fragment, flat_params, params)

    save_adapters(
        pipe,
        args.output_dir,
        adapter,
        args=args,
        params=params,
    )
    _annotate_export_metadata(
        args.output_dir,
        args.checkpoint,
        checkpoint,
        layout,
        args,
    )
    return checkpoint, layout, params


def main(argv=None) -> None:
    args = parse_args(argv)
    checkpoint, layout, params = export_checkpoint(args)
    out = Path(args.output_dir).expanduser()

    print(f"exported diffusion checkpoint at global_step={checkpoint.global_step} to {out}")
    print(
        f"applied {layout.num_fragments} fragments "
        f"({sum(flat.numel() for _, flat, _ in checkpoint.fragments)} params) "
        f"onto {len(params)} trainable tensors ({args.tuning})"
    )
    if checkpoint.ledger:
        total_merges = sum(merges for merges, _, _ in checkpoint.ledger.values())
        total_steps = sum(steps for _, steps, _ in checkpoint.ledger.values())
        total_units = sum(units for _, _, units in checkpoint.ledger.values())
        print(
            f"ledger: {len(checkpoint.ledger)} learners, {total_merges} merges, "
            f"{total_steps} steps, {total_units} units"
        )
        for learner_id in sorted(checkpoint.ledger):
            merges, steps, units = checkpoint.ledger[learner_id]
            print(
                f"  learner {learner_id}: merges={merges} steps={steps} units={units}"
            )
    else:
        print("ledger: empty")


if __name__ == "__main__":
    main()
