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
    optional:      32-byte semantic layout hash (new checkpoints)
    sweep-only:    marker u32 (0x5053_5750), policy_sweep_fragments u32,
                   session_contract_hash [u8; 32]

The outer momentum is parsed (and validated) but not needed for export; it
only matters when the syncer itself resumes from the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .fragments import FragmentLayout, build_layout
from .tensor_io import apply_fragment

CKPT_MAGIC = 0xD170_5A7E
POLICY_SWEEP_CKPT_MAGIC = 0x5053_5750


@dataclass
class Checkpoint:
    global_step: int
    # Per fragment, in layout order: (version, params, momentum) with params
    # and momentum as flat f32 tensors of the fragment's numel.
    fragments: list[tuple[int, torch.Tensor, torch.Tensor]]
    # learner_id -> (merges, steps, tokens)
    ledger: dict[int, tuple[int, int, int]]
    # HELLO's semantic layout fingerprint, persisted by current syncers.
    layout_hash: str | None
    # Dense-policy accounting identity. None preserves legacy per-fragment
    # accounting; when present it must cover the complete fragment layout.
    policy_sweep_fragments: int | None
    # Dense sweep semantic/profile/fixed-roster identity.
    session_contract_hash: str | None
    # Digest of the exact byte buffer decoded into this checkpoint.
    sha256: str


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

    layout_hash = None
    policy_sweep_fragments = None
    session_contract_hash = None
    trailing = len(data) - off
    if trailing in (32, 40, 72):
        layout_hash = take(32, "layout_hash").hex()
        if trailing in (40, 72):
            sweep_magic, policy_sweep_fragments = struct.unpack(
                "<II", take(8, "policy_sweep_identity")
            )
            if sweep_magic != POLICY_SWEEP_CKPT_MAGIC:
                raise ValueError(
                    f"{path}: bad policy-sweep checkpoint marker "
                    f"0x{sweep_magic:08X} (expected "
                    f"0x{POLICY_SWEEP_CKPT_MAGIC:08X})"
                )
            if policy_sweep_fragments == 0:
                raise ValueError(
                    f"{path}: policy-sweep fragment count must be positive"
                )
            if policy_sweep_fragments != num_fragments:
                raise ValueError(
                    f"{path}: policy-sweep checkpoint declares "
                    f"{policy_sweep_fragments} fragments, but the checkpoint "
                    f"contains {num_fragments}"
                )
            if trailing == 72:
                session_contract_hash = take(32, "session_contract_hash").hex()
    if off != len(data):
        raise ValueError(
            f"{path}: {len(data) - off} trailing bytes after the checkpoint "
            f"payload (parsed {off} of {len(data)}); file corrupt or from an "
            "incompatible syncer version"
        )
    return Checkpoint(
        global_step,
        fragments,
        ledger,
        layout_hash,
        policy_sweep_fragments,
        session_contract_hash,
        hashlib.sha256(data).hexdigest(),
    )


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
    if ckpt.layout_hash is not None:
        from .protocol import layout_fingerprint

        rebuilt = layout_fingerprint(layout).hex()
        if ckpt.layout_hash != rebuilt:
            problems.append(
                f"checkpoint layout hash {ckpt.layout_hash} != rebuilt {rebuilt}"
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
    p.add_argument("--model-revision", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--source-sha256", default=None, help=argparse.SUPPRESS)
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument(
        "--base-quantization", choices=["none", "nf4"], default="none"
    )
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
    from .provenance import pin_runtime_provenance, verify_source_tree_sha256

    verify_source_tree_sha256(args.source_sha256)
    pin_runtime_provenance(args, include_data=False)

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
        base_quantization=args.base_quantization,
        shard="ddp",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        loss_function="cross_entropy",
        attention_backend="auto",
        kernel_backend="native",
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
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    from .provenance import write_provenance_manifest

    write_provenance_manifest(
        out,
        args,
        artifact_kind="causal-lm-checkpoint-export",
        extra={
            "checkpoint": Path(args.checkpoint).name,
            "checkpoint_sha256": ckpt.sha256,
            "global_step": ckpt.global_step,
        },
    )

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
