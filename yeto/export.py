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
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .backends import all_backends, get_backend
from .fragments import FragmentLayout

CKPT_MAGIC = 0xD170_5A7E
PTI_CKPT_EXTENSION_MAGIC = 0x31495450  # little-endian b"PTI1"
CPLG_CKPT_EXTENSION_MAGIC = 0x314C5043  # little-endian b"CPL1"


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
        if off + meta_len > len(data):
            raise ValueError(
                f"{path}: trailing bytes do not form a valid layout metadata block "
                f"(declared {meta_len} bytes, have {len(data) - off})"
            )
        raw_meta = take(meta_len, "layout_meta")
        if raw_meta:
            layout_meta = json.loads(raw_meta.decode("utf-8"))
    # Causal direction selectors append optimizer state after the ordinary
    # export payload. Export does not consume that state, but validates and
    # skips it so the model parameters remain recoverable.
    if off != len(data):
        magic, state_count = struct.unpack(
            "<II", take(8, "causal-selector extension header")
        )
        if magic not in (PTI_CKPT_EXTENSION_MAGIC, CPLG_CKPT_EXTENSION_MAGIC):
            raise ValueError(f"{path}: unknown checkpoint extension 0x{magic:08X}")
        if state_count != num_fragments:
            selector = "PTI" if magic == PTI_CKPT_EXTENSION_MAGIC else "CPLG"
            raise ValueError(
                f"{path}: {selector} extension has {state_count} fragments, "
                f"expected {num_fragments}"
            )
        if magic == PTI_CKPT_EXTENSION_MAGIC:
            for fid, (_version, params, _momentum) in enumerate(fragments):
                numel = params.numel()
                for field in ("previous stock", "pending candidate"):
                    (count,) = struct.unpack(
                        "<Q", take(8, f"PTI fragment {fid} {field} length")
                    )
                    if count not in (0, numel):
                        raise ValueError(
                            f"{path}: PTI fragment {fid} {field} has {count} values, "
                            f"expected 0 or {numel}"
                        )
                    take(4 * count, f"PTI fragment {fid} {field}")
                (score_count,) = struct.unpack(
                    "<I", take(4, f"PTI fragment {fid} score count")
                )
                if score_count > 3:
                    raise ValueError(
                        f"{path}: PTI fragment {fid} has invalid score count "
                        f"{score_count}"
                    )
                take(4 * score_count, f"PTI fragment {fid} scores")
        else:
            for fid, (_version, params, _momentum) in enumerate(fragments):
                numel = params.numel()
                lengths: dict[str, int] = {}
                for field in (
                    "previous stock",
                    "previous tangent",
                    "pending candidate",
                ):
                    (count,) = struct.unpack(
                        "<Q", take(8, f"CPLG fragment {fid} {field} length")
                    )
                    if count not in (0, numel):
                        raise ValueError(
                            f"{path}: CPLG fragment {fid} {field} has {count} values, "
                            f"expected 0 or {numel}"
                        )
                    raw = take(4 * count, f"CPLG fragment {fid} {field}")
                    if raw and not bool(torch.isfinite(_read_f32(raw)).all()):
                        raise ValueError(
                            f"{path}: CPLG fragment {fid} {field} is non-finite"
                        )
                    lengths[field] = count

                (theta_tag,) = struct.unpack(
                    "<B", take(1, f"CPLG fragment {fid} theta tag")
                )
                if theta_tag == 0:
                    has_theta = False
                elif theta_tag == 1:
                    (theta,) = struct.unpack(
                        "<f", take(4, f"CPLG fragment {fid} theta")
                    )
                    if not math.isfinite(theta) or theta <= 0.0:
                        raise ValueError(
                            f"{path}: CPLG fragment {fid} theta is invalid"
                        )
                    has_theta = True
                else:
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} has theta tag {theta_tag}"
                    )

                (score_count,) = struct.unpack(
                    "<I", take(4, f"CPLG fragment {fid} score count")
                )
                if score_count > 3:
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} has invalid score count "
                        f"{score_count}"
                    )
                raw_scores = take(4 * score_count, f"CPLG fragment {fid} scores")
                if raw_scores and not bool(torch.isfinite(_read_f32(raw_scores)).all()):
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} scores are non-finite"
                    )

                has_stock = lengths["previous stock"] != 0
                has_tangent = lengths["previous tangent"] != 0
                has_candidate = lengths["pending candidate"] != 0
                if has_tangent != has_theta:
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} has inconsistent phase state"
                    )
                if not has_stock and (
                    has_tangent or has_theta or has_candidate or score_count
                ):
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} has phase state without "
                        "stock history"
                    )
                if not has_tangent and (has_candidate or score_count):
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} has shadow state without "
                        "a tangent"
                    )
                if not has_candidate and score_count:
                    raise ValueError(
                        f"{path}: CPLG fragment {fid} has scores without a "
                        "pending shadow"
                    )
    if off != len(data):
        raise ValueError(
            f"{path}: {len(data) - off} trailing bytes after checkpoint extensions"
        )
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
    p.add_argument(
        "--checkpoint", required=True, help="path to the syncer checkpoint file"
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
    for backend in all_backends():
        backend.add_export_cli_args(p)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    return get_backend(args.task).export(args)


if __name__ == "__main__":
    main()
