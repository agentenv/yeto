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


def test_validate_args_modes():
    with pytest.raises(ValueError, match="--prompt and --output"):
        sample.validate_args(SimpleNamespace(data=None, prompt=None, output=None))
    with pytest.raises(ValueError, match="--output-dir"):
        sample.validate_args(SimpleNamespace(data="rows", output_dir=None, output=None))
    with pytest.raises(ValueError, match="omit --output"):
        sample.validate_args(SimpleNamespace(data="rows", output_dir="out", output="one.png"))

    sample.validate_args(SimpleNamespace(data=None, prompt="p", output="one.png"))
    sample.validate_args(SimpleNamespace(data="rows", output_dir="out", output=None))


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


def test_batch_sample_writes_manifest(monkeypatch, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    calls = []

    def run_sample(pipe, args, meta, adapter=None):
        del pipe, meta, adapter
        calls.append((args.prompt, args.seed))
        return {"images": [Image.new("RGB", (2, 2))]}

    monkeypatch.setattr(sample, "run_sample", run_sample)
    args = SimpleNamespace(
        data=[
            {"prompt": "first"},
            {"prompt": "second"},
        ],
        prompt_column="prompt",
        seed_column=None,
        max_rows=None,
        output_dir=str(tmp_path / "samples"),
        manifest=None,
        adapter_dir=str(tmp_path / "adapter"),
        seed=100,
        num_inference_steps=4,
        guidance_scale=None,
        height=8,
        width=8,
        num_frames=None,
        fps=8,
    )

    records = sample.batch_sample(None, args, _metadata())

    assert calls == [("first", 100), ("second", 101)]
    assert [record["prompt"] for record in records] == ["first", "second"]
    assert records[0]["output_paths"] == ["sample_000000.png"]
    assert records[0]["generation"]["num_inference_steps"] == 4
    assert records[0]["artifact"]["resolved_model"] == "stabilityai/stable-diffusion-3.5-large"
    manifest = tmp_path / "samples" / "samples.jsonl"
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [line["seed"] for line in lines] == [100, 101]
    assert all((tmp_path / "samples" / path).exists() for line in lines for path in line["output_paths"])


def test_batch_sample_uses_seed_column(monkeypatch, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    seen = []

    monkeypatch.setattr(
        sample,
        "run_sample",
        lambda pipe, args, meta, adapter=None: seen.append(args.seed)
        or {"images": [Image.new("RGB", (2, 2))]},
    )
    args = SimpleNamespace(
        data=[{"prompt": "p", "seed": 7}],
        prompt_column="prompt",
        seed_column="seed",
        max_rows=None,
        output_dir=str(tmp_path / "samples"),
        manifest=str(tmp_path / "manifest.jsonl"),
        adapter_dir=str(tmp_path / "adapter"),
        seed=100,
        num_inference_steps=4,
        guidance_scale=None,
        height=None,
        width=None,
        num_frames=None,
        fps=8,
    )

    records = sample.batch_sample(None, args, _metadata())

    assert seen == [7]
    assert records[0]["seed_column"] == "seed"
    assert (tmp_path / "manifest.jsonl").exists()
