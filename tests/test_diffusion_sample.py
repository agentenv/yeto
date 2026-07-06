import json
from types import SimpleNamespace

import pytest

from yeto.diffusion import sample


def _metadata(**over):
    meta = {
        "kind": "yeto.diffusion.adapter",
        "schema_version": sample.DIFFUSION_ADAPTER_SCHEMA_VERSION,
        "model": "sd35",
        "resolved_model": "stabilityai/stable-diffusion-3.5-large",
        "tuning": "lora",
        "trainable_modules": ["transformer"],
    }
    meta.update(over)
    return meta


def _write_metadata(path, **over):
    path.mkdir(parents=True, exist_ok=True)
    (path / sample.DIFFUSION_ADAPTER_METADATA_FILE).write_text(
        json.dumps(_metadata(**over)),
        encoding="utf-8",
    )


def test_read_adapter_metadata_contract(tmp_path):
    _write_metadata(tmp_path)

    meta = sample.read_adapter_metadata(tmp_path)

    assert meta["kind"] == "yeto.diffusion.adapter"
    assert meta["trainable_modules"] == ["transformer"]


def test_read_adapter_metadata_rejects_wrong_kind(tmp_path):
    _write_metadata(tmp_path, kind="other")

    with pytest.raises(ValueError, match="expected kind"):
        sample.read_adapter_metadata(tmp_path)


def test_load_artifact_pipeline_uses_metadata_and_default_loader(monkeypatch, tmp_path):
    _write_metadata(tmp_path)
    calls = []

    class Pipe:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            calls.append(("eval",))

    def load_base(model_id, device, dtype):
        calls.append(("base", model_id, device, dtype))
        return Pipe()

    def load_adapters(pipe, adapter_dir, meta):
        calls.append(("adapters", adapter_dir, meta["trainable_modules"]))
        return pipe

    monkeypatch.setattr(sample, "_select_device", lambda device: "cpu")
    monkeypatch.setattr(sample, "_torch_dtype", lambda dtype, device: "float32")
    monkeypatch.setattr(sample, "_load_base_pipeline", load_base)
    monkeypatch.setattr(sample, "_load_default_adapters", load_adapters)

    pipe, meta, adapter = sample.load_artifact_pipeline(
        tmp_path,
        SimpleNamespace(model=None, diffusion_adapter=None, device=None, dtype="auto"),
    )

    assert pipe.moved_to == "cpu"
    assert adapter is None
    assert meta["resolved_model"] == "stabilityai/stable-diffusion-3.5-large"
    assert calls == [
        ("base", "stabilityai/stable-diffusion-3.5-large", "cpu", "float32"),
        ("adapters", tmp_path, ["transformer"]),
        ("eval",),
    ]


def test_pipeline_kwargs_filters_by_call_signature():
    class Pipe:
        def __call__(self, prompt, num_inference_steps=None, height=None):
            del prompt, num_inference_steps, height

    args = SimpleNamespace(
        prompt="a cat",
        num_inference_steps=7,
        guidance_scale=3.0,
        height=512,
        width=512,
        num_frames=9,
        seed=None,
    )

    assert sample._pipeline_kwargs(Pipe(), args) == {
        "prompt": "a cat",
        "num_inference_steps": 7,
        "height": 512,
    }


def test_save_single_image_output(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    out = tmp_path / "sample.png"
    saved = sample.save_sample_output({"images": [Image.new("RGB", (2, 2))]}, out)

    assert saved == [out]
    assert out.exists()


def test_save_frame_output_directory(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    out = tmp_path / "frames"
    saved = sample.save_sample_output(
        {"frames": [[Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]]},
        out,
    )

    assert [p.name for p in saved] == ["frame_000000.png", "frame_000001.png"]
    assert all(p.exists() for p in saved)
