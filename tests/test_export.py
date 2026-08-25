"""Tests for yeto.export checkpoint parsing (no network, no HF model)."""

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from yeto import export as export_module
from yeto.export import (
    CKPT_MAGIC,
    POLICY_SWEEP_CKPT_MAGIC,
    parse_checkpoint,
    validate_against_layout,
)
from yeto.fragments import build_layout
from yeto.tensor_io import apply_fragment


def fake_params() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    shapes = {
        "model.embed_tokens.weight": (10, 4),
        "model.layers.0.q.weight": (8, 8),
        "model.layers.0.mlp.weight": (16, 8),
        "model.layers.1.q.weight": (8, 8),
        "lm_head.weight": (10, 4),
    }
    return {n: torch.randn(s, generator=g) for n, s in shapes.items()}


LEDGER = {0: (5, 120, 240_000), 3: (2, 48, 96_000)}


def flat_fragment(frag, params):
    return torch.cat([params[n].reshape(-1).float() for n, _ in frag.tensors])


def checkpoint_bytes(global_step, blobs, ledger, magic=CKPT_MAGIC):
    """Encode (version, params, momentum) blobs in the syncer's binary format."""
    buf = bytearray()
    buf += struct.pack("<I", magic)
    buf += struct.pack("<Q", global_step)
    buf += struct.pack("<I", len(blobs))
    for version, p, m in blobs:
        buf += struct.pack("<QQ", version, p.numel())
        buf += struct.pack(f"<{p.numel()}f", *p.reshape(-1).tolist())
        buf += struct.pack(f"<{m.numel()}f", *m.reshape(-1).tolist())
    buf += struct.pack("<I", len(ledger))
    for lid, (merges, steps, tokens) in ledger.items():
        buf += struct.pack("<IQQQ", lid, merges, steps, tokens)
    return bytes(buf)


def make_checkpoint(tmp_path, params, num_fragments=4, global_step=42, magic=CKPT_MAGIC):
    layout = build_layout([(n, p.numel()) for n, p in params.items()], num_fragments)
    blobs = [
        (100 + fid, flat_fragment(frag, params), torch.full((frag.numel,), 0.5 * fid))
        for fid, frag in enumerate(layout.fragments)
    ]
    path = tmp_path / "syncer.ckpt"
    path.write_bytes(checkpoint_bytes(global_step, blobs, LEDGER, magic=magic))
    return path, layout, blobs


def test_round_trip(tmp_path):
    params = fake_params()
    path, layout, blobs = make_checkpoint(tmp_path, params)
    ckpt = parse_checkpoint(path)

    assert ckpt.global_step == 42
    assert ckpt.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert ckpt.ledger == LEDGER
    assert ckpt.layout_hash is None
    assert ckpt.policy_sweep_fragments is None
    assert ckpt.session_contract_hash is None
    assert len(ckpt.fragments) == layout.num_fragments
    for (version, p, m), (exp_version, exp_p, exp_m) in zip(ckpt.fragments, blobs):
        assert version == exp_version
        assert torch.equal(p, exp_p)
        assert torch.equal(m, exp_m)

    validate_against_layout(ckpt, layout)  # matching layout passes


def test_layout_hash_and_policy_sweep_trailers_are_backward_compatible(tmp_path):
    params = fake_params()
    path, layout, _ = make_checkpoint(tmp_path, params)
    legacy = path.read_bytes()
    layout_hash = bytes(range(32))

    path.write_bytes(legacy + layout_hash)
    hashed = parse_checkpoint(path)
    assert hashed.layout_hash == layout_hash.hex()
    assert hashed.policy_sweep_fragments is None
    assert hashed.session_contract_hash is None

    path.write_bytes(
        legacy
        + layout_hash
        + struct.pack(
            "<II", POLICY_SWEEP_CKPT_MAGIC, layout.num_fragments
        )
        + bytes(reversed(range(32)))
    )
    sweep = parse_checkpoint(path)
    assert sweep.layout_hash == layout_hash.hex()
    assert sweep.policy_sweep_fragments == layout.num_fragments
    assert sweep.session_contract_hash == bytes(reversed(range(32))).hex()


@pytest.mark.parametrize(
    ("marker", "fragments", "message"),
    [
        (0xDEADBEEF, 4, "bad policy-sweep checkpoint marker"),
        (POLICY_SWEEP_CKPT_MAGIC, 0, "fragment count must be positive"),
        (POLICY_SWEEP_CKPT_MAGIC, 3, "declares 3 fragments"),
    ],
)
def test_malformed_policy_sweep_trailer_is_rejected(
    tmp_path, marker, fragments, message
):
    path, _, _ = make_checkpoint(tmp_path, fake_params(), num_fragments=4)
    path.write_bytes(
        path.read_bytes()
        + bytes(32)
        + struct.pack("<II", marker, fragments)
    )
    with pytest.raises(ValueError, match=message):
        parse_checkpoint(path)


def test_causal_export_records_digest_of_parsed_checkpoint_bytes(monkeypatch, tmp_path):
    params = {"weight": torch.tensor([1.0, 2.0])}
    checkpoint, _, _ = make_checkpoint(
        tmp_path,
        params,
        num_fragments=1,
        global_step=11,
    )
    original_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    output_dir = tmp_path / "exported"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def save_pretrained(self, output, safe_serialization):
            assert safe_serialization is True
            Path(output, "model.marker").write_text("saved", encoding="utf-8")

    class Tokenizer:
        def save_pretrained(self, output):
            Path(output, "tokenizer.marker").write_text("saved", encoding="utf-8")

    model = Model()
    monkeypatch.setattr(
        "yeto.learner.load_model_and_tokenizer",
        lambda args, device: (model, Tokenizer()),
    )
    monkeypatch.setattr(
        "yeto.learner.trainable_params",
        lambda loaded: {"weight": loaded.weight},
    )
    real_parse = export_module.parse_checkpoint

    def parse_then_replace(path):
        parsed = real_parse(path)
        Path(path).write_bytes(b"replacement checkpoint bytes")
        return parsed

    monkeypatch.setattr(export_module, "parse_checkpoint", parse_then_replace)
    export_module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--model",
            str(model_dir),
            "--tuning",
            "full",
            "--fragments",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    record = json.loads(
        (output_dir / "yeto_provenance.json").read_text(encoding="utf-8")
    )
    assert record["artifact"]["checkpoint_sha256"] == original_digest
    assert record["artifact"]["global_step"] == 11


def test_bad_magic(tmp_path):
    path, _, _ = make_checkpoint(tmp_path, fake_params(), magic=0xDEADBEEF)
    with pytest.raises(ValueError, match="bad checkpoint magic"):
        parse_checkpoint(path)


def test_trailing_garbage(tmp_path):
    path, _, _ = make_checkpoint(tmp_path, fake_params())
    path.write_bytes(path.read_bytes() + b"\x00\x01\x02")
    with pytest.raises(ValueError, match="3 trailing bytes"):
        parse_checkpoint(path)


def test_truncated(tmp_path):
    path, _, _ = make_checkpoint(tmp_path, fake_params())
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ValueError, match="truncated"):
        parse_checkpoint(path)


def test_numel_mismatch_raises(tmp_path):
    params = fake_params()
    path, _, _ = make_checkpoint(tmp_path, params, num_fragments=4)
    ckpt = parse_checkpoint(path)
    # Layout rebuilt with different flags (e.g. wrong --fragments) mismatches.
    other = build_layout([(n, p.numel()) for n, p in params.items()], 2)
    with pytest.raises(ValueError) as exc:
        validate_against_layout(ckpt, other)
    msg = str(exc.value)
    assert "does not match" in msg
    assert "checkpoint has 4 fragments" in msg and "layout has 2" in msg


def test_numel_mismatch_lists_both_sizes(tmp_path):
    params = fake_params()
    path, layout, _ = make_checkpoint(tmp_path, params)
    ckpt = parse_checkpoint(path)
    # Same fragment count but one tensor sized differently.
    resized = [(n, p.numel()) for n, p in params.items()]
    resized[1] = (resized[1][0], resized[1][1] + 8)
    other = build_layout(resized, len(layout.fragments))
    assert other.num_fragments == layout.num_fragments
    with pytest.raises(ValueError) as exc:
        validate_against_layout(ckpt, other)
    msg = str(exc.value)
    assert "checkpoint numel" in msg and "layout numel" in msg


def test_apply_parsed_fragments_restores_params(tmp_path):
    params = fake_params()
    path, layout, _ = make_checkpoint(tmp_path, params)
    ckpt = parse_checkpoint(path)

    target = {n: torch.zeros_like(p) for n, p in params.items()}
    for frag, (_, flat, _) in zip(layout.fragments, ckpt.fragments):
        apply_fragment(frag, flat, target)
    for name in params:
        assert torch.equal(target[name], params[name]), name
