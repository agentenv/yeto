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
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .backends import all_backends, get_backend
from .fragments import FragmentLayout

CKPT_MAGIC = 0xD170_5A7E


@dataclass
class Checkpoint:
    global_step: int
    # Per fragment, in layout order: (version, params, momentum) with params
    # and momentum as flat f32 tensors of the fragment's numel.
    fragments: list[tuple[int, torch.Tensor, torch.Tensor]]
    # learner_id -> (merges, steps, tokens)
    ledger: dict[int, tuple[int, int, int]]
    # Optional protocol-v3 layout/task metadata appended after the v2 payload.
    layout_meta: dict | None = None


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

    layout_meta = None
    if off != len(data):
        remaining = len(data) - off
        if remaining < 4:
            raise ValueError(
                f"{path}: {remaining} trailing bytes after the checkpoint "
                f"payload (parsed {off} of {len(data)}); file corrupt or from an "
                "incompatible syncer version"
            )
        (meta_len,) = struct.unpack("<I", take(4, "layout_meta_len"))
        if off + meta_len != len(data):
            raise ValueError(
                f"{path}: trailing bytes do not form a valid layout metadata block "
                f"(declared {meta_len} bytes, have {len(data) - off})"
            )
        raw_meta = take(meta_len, "layout_meta")
        if raw_meta:
            layout_meta = json.loads(raw_meta.decode("utf-8"))
    return Checkpoint(global_step, fragments, ledger, layout_meta)


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
            "--model, --tuning, --lora-r, --fragments and --fragment-pattern "
            "match the training run: " + "; ".join(problems)
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m yeto.export",
        description="Export a syncer checkpoint into a Hugging Face "
        "model/adapter directory.",
    )
    # Shared across tasks; per-task flags come from the registered backends.
    p.add_argument("--task", choices=[b.name for b in all_backends()], default="lm")
    p.add_argument("--checkpoint", required=True, help="path to the syncer checkpoint file")
    p.add_argument("--fragments", type=int, default=8, help="P used during training")
    p.add_argument(
        "--fragment-pattern",
        choices=["binpack", "strided"],
        default="binpack",
        help="fragment pattern used during training",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    for backend in all_backends():
        backend.add_export_cli_args(p)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    return get_backend(args.task).export(args)


if __name__ == "__main__":
    main()
