"""Diffusion syncer-checkpoint export tests without network/model downloads."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto.diffusion import export as diffusion_export
from yeto.diffusion.learner import DIFFUSION_ADAPTER_METADATA_FILE, trainable_params
from yeto.export import CKPT_MAGIC
from yeto.fragments import build_layout
from yeto.provenance import file_sha256


class TinyTrainable(torch.nn.Module):
    def __init__(self, a: int = 6, b: int = 4):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(a))
        self.lora_B = torch.nn.Parameter(torch.zeros(b))

    def save_pretrained(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "adapter.pt")


class TinyPipe:
    def __init__(self, a: int = 6, b: int = 4):
        self.transformer = TinyTrainable(a, b)


def _args(tmp_path, checkpoint, **overrides):
    values = {
        "checkpoint": str(checkpoint),
        "model": "tiny-diffusion",
        "diffusion_adapter": None,
        "tuning": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_targets": "auto",
        "fragments": 2,
        "fragment_pattern": "binpack",
        "output_dir": str(tmp_path / "exported"),
        "device": "cpu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_checkpoint(path, params, fragments=2, global_step=17, ledger=None):
    layout = build_layout(
        [(name, value.numel()) for name, value in params.items()],
        fragments,
    )
    buf = bytearray()
    buf += struct.pack("<IQI", CKPT_MAGIC, global_step, layout.num_fragments)
    for fragment_id, fragment in enumerate(layout.fragments):
        flat = torch.cat(
            [params[name].detach().reshape(-1).float() for name, _ in fragment.tensors]
        )
        momentum = torch.full_like(flat, fragment_id + 0.25)
        buf += struct.pack("<QQ", 100 + fragment_id, flat.numel())
        buf += flat.numpy().tobytes()
        buf += momentum.numpy().tobytes()
    ledger = ledger or {}
    buf += struct.pack("<I", len(ledger))
    for learner_id, (merges, steps, units) in ledger.items():
        buf += struct.pack("<IQQQ", learner_id, merges, steps, units)
    path.write_bytes(bytes(buf))
    return layout


def _target_params(pipe):
    params = trainable_params(pipe)
    return {
        name: torch.arange(value.numel(), dtype=torch.float32).reshape_as(value) + index * 10
        for index, (name, value) in enumerate(params.items())
    }


def test_generic_export_restores_and_saves_diffusion_adapter(tmp_path, monkeypatch):
    source = TinyPipe()
    expected = _target_params(source)
    checkpoint = tmp_path / "state.ckpt"
    layout = _write_checkpoint(
        checkpoint,
        expected,
        ledger={0: (3, 24, 96), 2: (2, 16, 64)},
    )
    parsed_digest = file_sha256(checkpoint)
    rebuilt = TinyPipe()

    def load_pipeline(args, device, adapter):
        assert args.shard == "ddp"
        assert device == torch.device("cpu")
        assert adapter is None
        return rebuilt

    monkeypatch.setattr(diffusion_export, "load_pipeline", load_pipeline)
    real_save = diffusion_export.save_adapters

    def save_then_replace(*args, **kwargs):
        result = real_save(*args, **kwargs)
        checkpoint.write_bytes(b"replacement checkpoint bytes")
        return result

    monkeypatch.setattr(diffusion_export, "save_adapters", save_then_replace)
    args = _args(tmp_path, checkpoint)
    parsed, exported_layout, params = diffusion_export.export_checkpoint(args)

    assert parsed.global_step == 17
    assert parsed.sha256 == parsed_digest
    assert exported_layout == layout
    for name, value in params.items():
        assert torch.equal(value, expected[name])

    saved = torch.load(Path(args.output_dir) / "transformer" / "adapter.pt")
    assert torch.equal(saved["lora_A"], expected["transformer.lora_A"])
    assert torch.equal(saved["lora_B"], expected["transformer.lora_B"])

    meta = json.loads(
        (Path(args.output_dir) / DIFFUSION_ADAPTER_METADATA_FILE).read_text()
    )
    assert meta["kind"] == "yeto.diffusion.adapter"
    assert meta["trainable_tensor_count"] == 2
    assert meta["export"] == {
        "checkpoint": "state.ckpt",
        "checkpoint_sha256": parsed_digest,
        "fragment_pattern": "binpack",
        "fragment_versions": [100, 101],
        "fragments": 2,
        "global_step": 17,
        "ledger": {
            "0": {"merges": 3, "steps": 24, "units": 96},
            "2": {"merges": 2, "steps": 16, "units": 64},
        },
        "requested_fragments": 2,
        "source": "syncer-checkpoint",
    }


def test_export_rejects_a_different_diffusion_layout(tmp_path, monkeypatch):
    source = TinyPipe()
    checkpoint = tmp_path / "state.ckpt"
    _write_checkpoint(checkpoint, _target_params(source))
    monkeypatch.setattr(
        diffusion_export,
        "load_pipeline",
        lambda args, device, adapter: TinyPipe(a=7, b=4),
    )

    with pytest.raises(ValueError, match="same --lora-alpha"):
        diffusion_export.export_checkpoint(_args(tmp_path, checkpoint))
    assert not (tmp_path / "exported" / DIFFUSION_ADAPTER_METADATA_FILE).exists()


def test_external_adapter_reconstructs_custom_params_and_save_hook(tmp_path, monkeypatch):
    class CustomPipe:
        def __init__(self):
            self.adapter_weight = torch.nn.Parameter(torch.zeros(5))

    class CustomAdapter:
        def __init__(self):
            self.calls = []

        def load_pipeline(self, args, device):
            self.calls.append(("load", args.model, str(device)))
            return CustomPipe()

        def prepare_model(self, pipe, args, device):
            self.calls.append(("prepare", args.tuning, str(device)))
            return pipe

        def trainable_params(self, pipe):
            return {"custom.adapter_weight": pipe.adapter_weight}

        def save_adapters(self, pipe, output_dir):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(pipe.adapter_weight.detach(), out / "custom.pt")

    adapter = CustomAdapter()
    adapter_source = tmp_path / "custom_adapter.py"
    adapter_source.write_text("def make_adapter():\n    raise AssertionError\n")
    adapter_spec = f"{adapter_source}:make_adapter"
    expected = {"custom.adapter_weight": torch.arange(5, dtype=torch.float32)}
    checkpoint = tmp_path / "state.ckpt"
    _write_checkpoint(checkpoint, expected, fragments=1)

    monkeypatch.setattr(
        diffusion_export,
        "load_diffusion_adapter",
        lambda spec, **kwargs: adapter if spec == adapter_spec else None,
    )
    args = _args(
        tmp_path,
        checkpoint,
        diffusion_adapter=adapter_spec,
        fragments=1,
    )
    diffusion_export.export_checkpoint(args)

    assert adapter.calls == [
        ("load", "tiny-diffusion", "cpu"),
        ("prepare", "lora", "cpu"),
    ]
    assert torch.equal(
        torch.load(Path(args.output_dir) / "custom.pt"),
        expected["custom.adapter_weight"],
    )
    meta = json.loads(
        (Path(args.output_dir) / DIFFUSION_ADAPTER_METADATA_FILE).read_text()
    )
    assert meta["diffusion_adapter"] == adapter_spec
    assert meta["trainable_tensor_count"] == 1
    assert meta["export"]["fragments"] == 1
