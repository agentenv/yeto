"""Export a syncer checkpoint as a usable Hugging Face model/adapter.

The syncer's binary checkpoint holds the authoritative global params (plus
outer momentum and the merge ledger), so a run can be recovered even if every
learner is gone. This module has two layers:

  * :func:`parse_checkpoint` — decode the binary format written by the Rust
    syncer (``syncer/src/state.rs``) into a :class:`Checkpoint`.
  * a CLI (``python3 -m yeto.export``) that loads the base model exactly as a
    learner would, rebuilds the deterministic fragment layout over the
    trainable params, overwrites them with the checkpointed values, and saves
    the result with ``save_pretrained``.

Binary layout (all little-endian):

    magic          u32  (0xD170_5A7E)
    global_step    u64
    num_fragments  u32
    per fragment:  version u64, numel u64, numel x f32 params,
                   numel x f32 momentum
    ledger_count   u32
    per entry:     learner_id u32, merges u64, steps u64, tokens u64

The outer momentum is parsed (and validated) but not needed for export; it
only matters when the syncer itself resumes from the checkpoint.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .fragments import FragmentLayout, build_layout
from .tensor_io import apply_fragment

CKPT_MAGIC = 0xD170_5A7E


@dataclass
class Checkpoint:
    global_step: int
    # Per fragment, in layout order: (version, params, momentum) with params
    # and momentum as flat f32 tensors of the fragment's numel.
    fragments: list[tuple[int, torch.Tensor, torch.Tensor]]
    # learner_id -> (merges, steps, tokens)
    ledger: dict[int, tuple[int, int, int]]


def parse_checkpoint(path: str | Path) -> Checkpoint:
    """Decode a syncer checkpoint file, validating magic and exact length."""
    path = Path(path)
    data = path.read_bytes()
    off = 0

    def take(n: int, what: str) -> bytes:
        nonlocal off
        if off + n > len(data):
            raise ValueError(
                f"{path}: truncated checkpoint: needed {n} bytes for {what} "
                f"at offset {off}, but the file only has {len(data)} bytes"
            )
        chunk = data[off : off + n]
        off += n
        return chunk

    (magic,) = struct.unpack("<I", take(4, "magic"))
    if magic != CKPT_MAGIC:
        raise ValueError(
            f"{path}: bad checkpoint magic 0x{magic:08X} "
            f"(expected 0x{CKPT_MAGIC:08X}); not a syncer checkpoint?"
        )
    (global_step,) = struct.unpack("<Q", take(8, "global_step"))
    (num_fragments,) = struct.unpack("<I", take(4, "num_fragments"))

    fragments: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for fid in range(num_fragments):
        version, numel = struct.unpack("<QQ", take(16, f"fragment {fid} header"))
        params = _read_f32(take(4 * numel, f"fragment {fid} params"))
        momentum = _read_f32(take(4 * numel, f"fragment {fid} momentum"))
        fragments.append((version, params, momentum))

    (ledger_count,) = struct.unpack("<I", take(4, "ledger_count"))
    ledger: dict[int, tuple[int, int, int]] = {}
    for i in range(ledger_count):
        learner_id, merges, steps, tokens = struct.unpack(
            "<IQQQ", take(28, f"ledger entry {i}")
        )
        ledger[learner_id] = (merges, steps, tokens)

    if off != len(data):
        raise ValueError(
            f"{path}: {len(data) - off} trailing bytes after the checkpoint "
            f"payload (parsed {off} of {len(data)}); file corrupt or from an "
            "incompatible syncer version"
        )
    return Checkpoint(global_step, fragments, ledger)


def _read_f32(raw: bytes) -> torch.Tensor:
    # bytearray copy: torch.frombuffer would otherwise alias the read-only
    # bytes object. Checkpoints are little-endian f32, matching torch's
    # layout on every supported platform.
    return torch.frombuffer(bytearray(raw), dtype=torch.float32)


def validate_against_layout(ckpt: Checkpoint, layout: FragmentLayout) -> None:
    """Hard-error unless the checkpoint's fragments match the rebuilt layout.

    The layout is a pure function of the trainable tensor set and
    ``--fragments``, so any mismatch means the export flags (--model,
    --tuning, --lora-r, --fragments) differ from the ones the run used.
    """
    problems = []
    if len(ckpt.fragments) != layout.num_fragments:
        problems.append(
            f"checkpoint has {len(ckpt.fragments)} fragments, "
            f"rebuilt layout has {layout.num_fragments}"
        )
    for fid, (frag, (_, params, _)) in enumerate(zip(layout.fragments, ckpt.fragments)):
        if params.numel() != frag.numel:
            problems.append(
                f"fragment {fid}: checkpoint numel {params.numel()} "
                f"!= layout numel {frag.numel}"
            )
    if problems:
        raise ValueError(
            "checkpoint does not match the rebuilt fragment layout; make sure "
            "--model, --tuning, --lora-r, --fragments, --fragment-pattern and "
            "--matrix-merge match the training run: " + "; ".join(problems)
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m yeto.export",
        description="Export a syncer checkpoint into a Hugging Face "
        "model/adapter directory.",
    )
    p.add_argument("--checkpoint", required=True, help="path to the syncer checkpoint file")
    p.add_argument("--model", required=True, help="HF model id or an alias from yeto/models.py (gemma4, qwen35-9b, llama31-8b, gptoss-120b, ...)")
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
    p.add_argument(
        "--matrix-merge",
        choices=["rda", "iso"],
        default="rda",
        help="matrix merge mode used during training",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    ckpt = parse_checkpoint(args.checkpoint)

    # Imported lazily: parse_checkpoint must stay usable without transformers.
    from .learner import load_model_and_tokenizer, trainable_params

    device = torch.device(args.device)
    # Mirror the learner's model construction exactly so the trainable tensor
    # set (and therefore the fragment layout) is identical. shard="ddp" keeps
    # the single-process (non-FSDP) dtype path.
    learner_args = argparse.Namespace(
        model=args.model,
        tuning=args.tuning,
        shard="ddp",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
    )
    model, tokenizer = load_model_and_tokenizer(learner_args, device)
    params = trainable_params(model)
    layout = build_layout(
        [(n, p.numel()) for n, p in params.items()],
        args.fragments,
        args.fragment_pattern,
        matrix_merge=args.matrix_merge,
        named_shapes={n: tuple(p.shape) for n, p in params.items()},
    )

    validate_against_layout(ckpt, layout)

    for frag, (_, flat_params, _) in zip(layout.fragments, ckpt.fragments):
        apply_fragment(frag, flat_params, params)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    print(f"exported checkpoint at global_step={ckpt.global_step} to {out}")
    print(
        f"applied {layout.num_fragments} fragments "
        f"({sum(p.numel() for _, p, _ in ckpt.fragments)} params) "
        f"onto {len(params)} trainable tensors ({args.tuning})"
    )
    if ckpt.ledger:
        total_merges = sum(m for m, _, _ in ckpt.ledger.values())
        total_steps = sum(s for _, s, _ in ckpt.ledger.values())
        total_tokens = sum(t for _, _, t in ckpt.ledger.values())
        print(
            f"ledger: {len(ckpt.ledger)} learners, {total_merges} merges, "
            f"{total_steps} steps, {total_tokens} tokens"
        )
        for lid in sorted(ckpt.ledger):
            merges, steps, tokens = ckpt.ledger[lid]
            print(f"  learner {lid}: merges={merges} steps={steps} tokens={tokens}")
    else:
        print("ledger: empty")


if __name__ == "__main__":
    main()
