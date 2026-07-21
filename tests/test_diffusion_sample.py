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
        "provenance": {
            "model": {
                "source": "huggingface",
                "requested_identifier": "sd35",
                "resolved_identifier": "stabilityai/stable-diffusion-3.5-large",
                "requested_revision": "main",
                "resolved_revision": "a" * 40,
            }
        },
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

    def load_base(model_id, device, dtype, **kwargs):
        calls.append(("base", model_id, device, dtype, kwargs))
        return Pipe()

    def load_adapters(pipe, adapter_dir, meta):
        calls.append(("adapters", adapter_dir, meta["trainable_modules"]))
        return pipe

    monkeypatch.setattr(sample, "_select_device", lambda device: "cpu")
    monkeypatch.setattr(sample, "_torch_dtype", lambda dtype, device: "float32")
    monkeypatch.setattr(sample, "_load_base_pipeline", load_base)
    monkeypatch.setattr(sample, "_load_default_adapters", load_adapters)

    args = SimpleNamespace(model=None, diffusion_adapter=None, device=None, dtype="auto")
    pipe, meta, adapter = sample.load_artifact_pipeline(tmp_path, args)

    assert pipe.moved_to == "cpu"
    assert args.model == "stabilityai/stable-diffusion-3.5-large"
    assert adapter is None
    assert meta["resolved_model"] == "stabilityai/stable-diffusion-3.5-large"
    assert calls == [
        (
            "base",
            "stabilityai/stable-diffusion-3.5-large",
            "cpu",
            "float32",
            {"revision": "a" * 40, "trust_remote_code": False},
        ),
        ("adapters", tmp_path, ["transformer"]),
        ("eval",),
    ]


def test_artifact_model_identity_mismatch_is_rejected(tmp_path):
    _write_metadata(tmp_path, resolved_model="org/different-model")

    with pytest.raises(ValueError, match="inconsistent base-model identities"):
        sample.load_artifact_pipeline(
            tmp_path,
            SimpleNamespace(
                model=None,
                model_revision=None,
                diffusion_adapter=None,
                device="cpu",
                dtype="auto",
            ),
        )


def test_remote_artifact_requires_recorded_immutable_model_commit(tmp_path):
    meta = _metadata()
    meta["provenance"]["model"]["resolved_revision"] = "main"
    _write_metadata(tmp_path, provenance=meta["provenance"])

    with pytest.raises(ValueError, match="full immutable model commit"):
        sample.load_artifact_pipeline(
            tmp_path,
            SimpleNamespace(
                model=None,
                model_revision=None,
                diffusion_adapter=None,
                device="cpu",
                dtype="auto",
            ),
        )


def test_launcher_adapter_digest_cannot_override_artifact_digest(
    monkeypatch, tmp_path
):
    adapter_dir = tmp_path / "artifact"
    _write_metadata(adapter_dir, diffusion_adapter_sha256="a" * 64)
    adapter_source = tmp_path / "adapter.py"
    adapter_source.write_text(
        "def make_adapter():\n    raise AssertionError('must not execute')\n",
        encoding="utf-8",
    )
    from yeto.provenance import file_sha256

    current_digest = file_sha256(adapter_source)
    assert current_digest != "a" * 64
    monkeypatch.setattr(
        sample,
        "_load_external_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("adapter must not load before digest agreement")
        ),
    )

    with pytest.raises(ValueError, match="does not match the training artifact"):
        sample.load_artifact_pipeline(
            adapter_dir,
            SimpleNamespace(
                model=None,
                model_revision=None,
                diffusion_adapter=f"{adapter_source}:make_adapter",
                diffusion_adapter_sha256=current_digest,
                device="cpu",
                dtype="auto",
                trust_remote_code=False,
            ),
        )


def test_runtime_adapter_factory_must_match_artifact_binding(tmp_path):
    adapter_dir = tmp_path / "artifact"
    adapter_source = tmp_path / "adapter.py"
    adapter_source.write_text(
        "def training_factory():\n    return object()\n"
        "def runtime_factory():\n    return object()\n",
        encoding="utf-8",
    )
    from yeto.provenance import file_sha256

    digest = file_sha256(adapter_source)
    _write_metadata(
        adapter_dir,
        diffusion_adapter="adapter.py:training_factory",
        diffusion_adapter_sha256=digest,
    )

    with pytest.raises(ValueError, match="spec does not match"):
        sample.load_artifact_pipeline(
            adapter_dir,
            SimpleNamespace(
                model=None,
                model_revision=None,
                diffusion_adapter=f"{adapter_source}:runtime_factory",
                diffusion_adapter_sha256=digest,
                device="cpu",
                dtype="auto",
                trust_remote_code=False,
            ),
        )


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


def test_resolve_sample_data_arg_preserves_non_string_and_non_cloud(monkeypatch):
    rows = [{"prompt": "p"}]
    assert sample.resolve_sample_data_arg(rows) is rows

    import yeto.datasource as datasource

    seen = []

    def fake_kind(data):
        seen.append(data)
        return "hf"

    monkeypatch.setattr(datasource, "kind", fake_kind)

    assert sample.resolve_sample_data_arg("owner/dataset") == "owner/dataset"
    assert seen == ["owner/dataset"]


def test_resolve_sample_data_arg_uses_existing_cloud_mount(monkeypatch, tmp_path):
    import yeto.datasource as datasource

    mounted = tmp_path / "mounted.jsonl"
    mounted.write_text('{"prompt":"p"}\n', encoding="utf-8")
    monkeypatch.setattr(datasource, "kind", lambda data: "cloud")
    monkeypatch.setattr(datasource, "learner_data_arg", lambda data: str(mounted))

    assert sample.resolve_sample_data_arg("s3://bucket/prompts.jsonl") == str(mounted)


def test_resolve_sample_data_arg_rejects_unmounted_cloud(monkeypatch, tmp_path):
    import yeto.datasource as datasource

    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(datasource, "kind", lambda data: "cloud")
    monkeypatch.setattr(datasource, "learner_data_arg", lambda data: str(missing))

    with pytest.raises(ValueError, match="cloud data source"):
        sample.resolve_sample_data_arg("s3://bucket/prompts.jsonl")


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
        diffusion_adapter="adapter.py:make_adapter",
        diffusion_adapter_sha256="d" * 64,
    )
    meta = _metadata(
        diffusion_adapter="adapter.py:make_adapter",
        diffusion_adapter_sha256="d" * 64,
    )

    records = sample.batch_sample(None, args, meta)

    assert calls == [("first", 100), ("second", 101)]
    assert [record["prompt"] for record in records] == ["first", "second"]
    assert records[0]["output_paths"] == ["sample_000000.png"]
    assert records[0]["generation"]["num_inference_steps"] == 4
    assert records[0]["artifact"]["resolved_model"] == "stabilityai/stable-diffusion-3.5-large"
    assert records[0]["artifact"]["diffusion_adapter"] == "adapter.py:make_adapter"
    assert records[0]["artifact"]["diffusion_adapter_sha256"] == "d" * 64
    assert records[0]["artifact"]["adapter_attestation_status"] == "attested"
    assert records[0]["runtime_adapter"] == {
        "spec": "adapter.py:make_adapter",
        "sha256": "d" * 64,
        "artifact_binding": "matched",
    }
    manifest = tmp_path / "samples" / "samples.jsonl"
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [line["seed"] for line in lines] == [100, 101]
    assert all((tmp_path / "samples" / path).exists() for line in lines for path in line["output_paths"])


def test_legacy_adapter_requires_override_and_manifest_marks_unbound(
    monkeypatch, tmp_path
):
    Image = pytest.importorskip("PIL.Image")
    adapter_dir = tmp_path / "artifact"
    _write_metadata(adapter_dir, diffusion_adapter="legacy.py:make_adapter")
    adapter_source = tmp_path / "legacy.py"
    adapter_source.write_text(
        "def make_adapter():\n    return object()\n",
        encoding="utf-8",
    )

    class Pipe:
        pass

    class Adapter:
        supports_pinned_model_source = True

        def load_sample_pipeline(self, *args):
            return Pipe()

    monkeypatch.setattr(sample, "_select_device", lambda device: "cpu")
    monkeypatch.setattr(sample, "_load_external_adapter", lambda *args: Adapter())

    def make_args(allow):
        return SimpleNamespace(
            adapter_dir=str(adapter_dir),
            model=None,
            model_revision=None,
            diffusion_adapter=f"{adapter_source}:make_adapter",
            diffusion_adapter_sha256=None,
            allow_unattested_legacy_adapter=allow,
            device="cpu",
            dtype="auto",
            trust_remote_code=False,
            data=[{"prompt": "legacy sample"}],
            data_revision=None,
            prompt_column="prompt",
            seed_column=None,
            max_rows=None,
            output_dir=str(tmp_path / "samples"),
            manifest=None,
            seed=4,
            num_inference_steps=4,
            guidance_scale=None,
            height=None,
            width=None,
            num_frames=None,
            fps=8,
        )

    with pytest.raises(PermissionError, match="no complete training-time adapter"):
        sample.load_artifact_pipeline(adapter_dir, make_args(False))

    args = make_args(True)
    pipe, meta, adapter = sample.load_artifact_pipeline(adapter_dir, args)
    monkeypatch.setattr(
        sample,
        "run_sample",
        lambda *args, **kwargs: {"images": [Image.new("RGB", (2, 2))]},
    )
    records = sample.batch_sample(pipe, args, meta, adapter)

    assert records[0]["artifact"]["adapter_attestation_status"] == "legacy-unattested"
    assert records[0]["artifact"]["diffusion_adapter_sha256"] is None
    assert records[0]["runtime_adapter"] == {
        "spec": f"{adapter_source}:make_adapter",
        "sha256": args.diffusion_adapter_sha256,
        "artifact_binding": "legacy-unbound",
    }
    assert len(args.diffusion_adapter_sha256) == 64


def test_model_override_manifest_separates_artifact_and_runtime_provenance(
    monkeypatch, tmp_path
):
    Image = pytest.importorskip("PIL.Image")
    adapter_dir = tmp_path / "adapter"
    _write_metadata(adapter_dir)

    class Pipe:
        def to(self, device):
            return self

        def eval(self):
            return None

    loaded = []

    def load_base(model_id, device, dtype, **kwargs):
        loaded.append((model_id, kwargs))
        return Pipe()

    monkeypatch.setattr(sample, "_select_device", lambda device: "cpu")
    monkeypatch.setattr(sample, "_torch_dtype", lambda dtype, device: "float32")
    monkeypatch.setattr(sample, "_load_base_pipeline", load_base)
    monkeypatch.setattr(sample, "_load_default_adapters", lambda pipe, *args: pipe)
    monkeypatch.setattr(
        sample,
        "run_sample",
        lambda *args, **kwargs: {"images": [Image.new("RGB", (2, 2))]},
    )

    args = SimpleNamespace(
        adapter_dir=str(adapter_dir),
        model="org/runtime-model",
        model_revision="b" * 40,
        model_requested_identifier=None,
        model_requested_revision="release",
        trust_remote_code=False,
        diffusion_adapter=None,
        device="cpu",
        dtype="auto",
        data=[{"prompt": "override sample"}],
        data_revision=None,
        prompt_column="prompt",
        seed_column=None,
        max_rows=None,
        output_dir=str(tmp_path / "samples"),
        manifest=None,
        seed=9,
        num_inference_steps=4,
        guidance_scale=None,
        height=None,
        width=None,
        num_frames=None,
        fps=8,
    )
    pipe, meta, adapter = sample.load_artifact_pipeline(adapter_dir, args)
    records = sample.batch_sample(pipe, args, meta, adapter)

    assert loaded == [("org/runtime-model", {"revision": "b" * 40, "trust_remote_code": False})]
    artifact_model = records[0]["artifact"]["provenance"]["model"]
    runtime = records[0]["runtime_provenance"]
    assert artifact_model["resolved_identifier"] == "stabilityai/stable-diffusion-3.5-large"
    assert artifact_model["resolved_revision"] == "a" * 40
    assert runtime["model"]["resolved_identifier"] == "org/runtime-model"
    assert runtime["model"]["requested_revision"] == "release"
    assert runtime["model"]["resolved_revision"] == "b" * 40
    assert runtime["trust_remote_code"] is False


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
