import argparse
import io
import json
from types import SimpleNamespace

import pytest

from yeto.gpu_spec import ClusterSpec
from yeto.launcher import make_diffusion_sample_task, make_learner_task
from yeto.models import (
    DIFFUSION_MODEL_ALIASES,
    resolve,
    resolve_model_kind,
)


_SPEC = ClusterSpec(cloud="aws", region="us-east-2", num_nodes=1, gpus_per_node=1, gpu="A100")


def _args(**over):
    base = dict(
        model="sd35",
        model_kind="auto",
        data="org/diffusion-data",
        loss_function="cross_entropy",
        train_on="assistant",
        shard="fsdp",
        tuning="lora",
        lora_r=16,
        lora_targets="auto",
        seq_len=2048,
        micro_batch_size="auto",
        grad_accum=4,
        inner_lr=3e-4,
        fragments=8,
        fragment_pattern="binpack",
        merge_alpha=0.5,
        tokenize="stream",
        stream_workers=2,
        wire_dtype="bf16",
        wan_streams=4,
        max_rows=None,
        island_backend="torch",
        expert_parallel=None,
        tensor_parallel=1,
        pipeline_parallel=1,
        learner_image=None,
        learner_cpus=None,
        learner_instance_type=None,
        spot=True,
        disk_size=512,
        retry_until_up=True,
        cache_latents=False,
        cache_text_embeds=False,
        diffusion_adapter=None,
        image_column="image",
        video_column="video",
        prompt_column="prompt",
        latent_column="latents",
        text_embeds_column="prompt_embeds",
        text_attention_mask_column="prompt_attention_mask",
        pooled_text_embeds_column="pooled_prompt_embeds",
        height=None,
        width=None,
        num_frames=None,
        bucket_by_shape=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_diffusion_aliases_resolve_and_infer_kind():
    assert {"wan22", "ltx-video", "flux", "sd35", "nava"} <= set(DIFFUSION_MODEL_ALIASES)
    assert resolve("sd35") == DIFFUSION_MODEL_ALIASES["sd35"]
    assert resolve("nava") == "baidu/NAVA"
    assert resolve_model_kind("flux") == "diffusion"
    assert resolve_model_kind("org/custom", "diffusion") == "diffusion"
    assert resolve_model_kind("org/custom") == "causal-lm"


def test_diffusion_learner_parse_cache_defaults_are_off():
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    args = learner.parse_args([
        "--model", "sd35",
        "--data", "org/ds",
        "--syncer", "none",
        "--learner-id", "0",
        "--num-learners", "1",
    ])
    assert args.cache_latents is False
    assert args.cache_text_embeds is False
    assert args.diffusion_adapter is None
    assert args.loss_function == "flow_matching"


def test_diffusion_lora_targets_are_generic_dit_names():
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    targets = learner.resolve_lora_targets("auto", "sd35")
    assert "to_q" in targets and "to_k" in targets and "to_v" in targets
    assert "ff.net.0.proj" in targets
    assert learner.resolve_lora_targets("all-linear", "sd35") == "all-linear"


def test_diffusion_dtype_matches_cuda_bf16_support(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    assert learner.diffusion_torch_dtype(SimpleNamespace(type="cpu")) is torch.float32
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert learner.diffusion_torch_dtype(SimpleNamespace(type="cuda")) is torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert learner.diffusion_torch_dtype(SimpleNamespace(type="cuda")) is torch.float16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert learner.diffusion_torch_dtype(SimpleNamespace(type="cuda")) is torch.bfloat16


def test_flow_matching_loss_counts_elements():
    torch = pytest.importorskip("torch")
    from yeto.losses import flow_matching_loss

    pred = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    loss, denom = flow_matching_loss(pred, target, torch.tensor([1, 2]))
    assert loss == 6
    assert denom == 6


def test_cached_manifest_contract_reads_tensors_and_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    torch.save(torch.zeros(1, 2, 2), tmp_path / "lat0.pt")
    rows = [
        {
            "__yeto_data_root__": str(tmp_path),
            "latents": "lat0.pt",
            "prompt_embeds": [[1.0, 2.0]],
            "pooled_prompt_embeds": [0.5, 0.25],
            "prompt_attention_mask": [1],
        },
        {
            "__yeto_data_root__": str(tmp_path),
            "latents": torch.ones(1, 2, 2),
            "prompt_embeds": [[3.0, 4.0]],
            "pooled_prompt_embeds": [0.75, 1.0],
            "prompt_attention_mask": [1],
        },
    ]
    args = _args(cache_latents=True, cache_text_embeds=True)

    latents = learner.encode_latents(None, rows, args, torch.device("cpu"), torch.float32)
    cond = learner.encode_prompt_embeds(None, rows, args, torch.device("cpu"), torch.float32)

    assert tuple(latents.latents.shape) == (2, 1, 2, 2)
    assert latents.latent_num_frames is None
    assert latents.latent_height == 2
    assert latents.latent_width == 2
    assert tuple(cond.prompt_embeds.shape) == (2, 1, 2)
    assert tuple(cond.pooled_prompt_embeds.shape) == (2, 2)
    assert tuple(cond.attention_mask.shape) == (2, 1)


def test_cached_manifest_contract_reports_missing_columns():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    args = _args(cache_latents=True, cache_text_embeds=True)
    rows = [{"prompt_embeds": [[1.0, 2.0]]}]

    with pytest.raises(KeyError, match="--cache-latents needs column 'latents'"):
        learner.encode_latents(None, rows, args, torch.device("cpu"), torch.float32)

    mixed_optional_rows = [
        {"prompt_embeds": [[1.0, 2.0]], "pooled_prompt_embeds": [1.0]},
        {"prompt_embeds": [[3.0, 4.0]]},
    ]
    with pytest.raises(KeyError, match="pooled_prompt_embeds.*some rows"):
        learner.encode_prompt_embeds(
            None,
            mixed_optional_rows,
            args,
            torch.device("cpu"),
            torch.float32,
        )


def test_diffusion_cache_metadata_records_contract(tmp_path):
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    args = _args(
        cache_latents=True,
        cache_text_embeds=True,
        data=str(tmp_path / "data.jsonl"),
    )
    learner.write_diffusion_cache_metadata(tmp_path, args, row_count=3)

    meta = learner.read_diffusion_cache_metadata(str(tmp_path / "data.jsonl"))
    assert meta["kind"] == "yeto.diffusion.cache"
    assert meta["schema_version"] == learner.DIFFUSION_CACHE_SCHEMA_VERSION
    assert meta["row_count"] == 3
    assert meta["cache"] == {"latents": True, "text_embeds": True}
    assert meta["columns"]["latents"] == "latents"
    assert meta["relative_paths"] is True

    learner.validate_diffusion_cache_metadata(meta, args)
    with pytest.raises(ValueError, match="column 'latents'"):
        learner.validate_diffusion_cache_metadata(meta, _args(cache_latents=True, latent_column="latent_tensor"))


def test_diffusion_autobatch_doubles_until_oom(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    sizes = []

    def probe(pipe, params, opt, rows, args, device, micro_batch, adapter=None):
        del pipe, params, opt, rows, args, device, adapter
        sizes.append(micro_batch)
        if micro_batch >= 8:
            raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(learner, "_probe_diffusion_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    opt = SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    args = _args(
        data=[{"latents": [0.0], "prompt_embeds": [[0.0]]}],
        cache_latents=True,
        cache_text_embeds=True,
        micro_batch_size="auto",
    )

    got = learner.resolve_diffusion_micro_batch_size(
        args,
        None,
        {},
        opt,
        SimpleNamespace(type="cuda"),
        rank=0,
        world=1,
    )

    assert got == 4
    assert sizes == [1, 2, 4, 8]


def test_diffusion_autobatch_bucket_by_shape_uses_smallest_fit(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    probes = []

    def probe(pipe, params, opt, rows, args, device, micro_batch, adapter=None):
        del pipe, params, opt, args, device, adapter
        shape = learner._shape_key(rows[0])
        probes.append((shape, micro_batch))
        if shape == (None, 128, 128) and micro_batch >= 4:
            raise torch.cuda.OutOfMemoryError("synthetic large shape")
        if micro_batch >= 8:
            raise torch.cuda.OutOfMemoryError("synthetic small shape")

    monkeypatch.setattr(learner, "_probe_diffusion_once", probe)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    opt = SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    args = _args(
        data=[
            {"height": 64, "width": 64, "latents": [0.0], "prompt_embeds": [[0.0]]},
            {"height": 64, "width": 64, "latents": [0.0], "prompt_embeds": [[0.0]]},
            {"height": 128, "width": 128, "latents": [0.0], "prompt_embeds": [[0.0]]},
        ],
        cache_latents=True,
        cache_text_embeds=True,
        bucket_by_shape=True,
        micro_batch_size="auto",
    )

    got = learner.resolve_diffusion_micro_batch_size(
        args,
        None,
        {},
        opt,
        SimpleNamespace(type="cuda"),
        rank=0,
        world=1,
    )

    assert got == 2
    assert probes == [
        ((None, 64, 64), 1),
        ((None, 64, 64), 2),
        ((None, 64, 64), 4),
        ((None, 64, 64), 8),
        ((None, 128, 128), 1),
        ((None, 128, 128), 2),
        ((None, 128, 128), 4),
    ]


def test_diffusion_autobatch_cpu_returns_one_without_probe(monkeypatch):
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    monkeypatch.setattr(
        learner,
        "_probe_diffusion_once",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    got = learner.resolve_diffusion_micro_batch_size(
        _args(micro_batch_size="auto"),
        None,
        {},
        None,
        SimpleNamespace(type="cpu"),
        rank=0,
        world=1,
    )
    assert got == 1


def test_diffusion_adapter_file_factory(tmp_path):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    del torch
    adapter_file = tmp_path / "adapter.py"
    adapter_file.write_text(
        "class Adapter:\n"
        "    marker = 'loaded'\n"
        "\n"
        "def make_adapter():\n"
        "    return Adapter()\n"
    )

    adapter = learner.load_diffusion_adapter(f"{adapter_file}:make_adapter")
    assert adapter.marker == "loaded"


def test_tiny_diffusion_learner_smoke_cached_manifest(tmp_path):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    adapter_file = tmp_path / "tiny_adapter.py"
    adapter_file.write_text(
        "import torch\n"
        "from yeto.diffusion.learner import diffusion_torch_dtype\n"
        "\n"
        "class TinyScheduler:\n"
        "    class config:\n"
        "        num_train_timesteps = 4\n"
        "        prediction_type = 'epsilon'\n"
        "\n"
        "    def add_noise(self, latents, noise, timesteps):\n"
        "        del timesteps\n"
        "        return latents + 0.1 * noise\n"
        "\n"
        "class TinyDenoiser(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.proj = torch.nn.Linear(4, 4)\n"
        "\n"
        "    def forward(self, hidden_states, timestep, encoder_hidden_states=None, return_dict=False):\n"
        "        del timestep, return_dict\n"
        "        cond = encoder_hidden_states.mean(dim=1, keepdim=True)\n"
        "        return self.proj(hidden_states + cond)\n"
        "\n"
        "class TinyPipe:\n"
        "    def __init__(self):\n"
        "        self.transformer = TinyDenoiser()\n"
        "        self.scheduler = TinyScheduler()\n"
        "        self.components = {'transformer': self.transformer}\n"
        "\n"
        "    def to(self, device):\n"
        "        dtype = diffusion_torch_dtype(torch.device(device))\n"
        "        self.transformer.to(device=device, dtype=dtype)\n"
        "        return self\n"
        "\n"
        "def make_adapter():\n"
        "    class Adapter:\n"
        "        def load_pipeline(self, args, device):\n"
        "            del args, device\n"
        "            return TinyPipe()\n"
        "    return Adapter()\n"
    )
    for i in range(2):
        torch.save(torch.full((4,), float(i)), tmp_path / f"latents_{i}.pt")
        torch.save(torch.ones(4) * (i + 1), tmp_path / f"prompt_{i}.pt")
    data = tmp_path / "data.jsonl"
    with data.open("w", encoding="utf-8") as f:
        for i in range(2):
            f.write(
                json.dumps(
                    {
                        "latents": f"latents_{i}.pt",
                        "prompt_embeds": f"prompt_{i}.pt",
                    }
                )
                + "\n"
            )
    out = tmp_path / "out"

    learner.main(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--syncer",
            "none",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
            "--tuning",
            "full",
            "--cache-latents",
            "--cache-text-embeds",
            "--diffusion-adapter",
            f"{adapter_file}:make_adapter",
            "--micro-batch-size",
            "2",
            "--stream-workers",
            "0",
            "--fragments",
            "1",
            "--max-local-steps",
            "1",
            "--output-dir",
            str(out),
        ]
    )

    state = torch.load(out / "trainable_state.pt", map_location="cpu")
    assert "transformer.proj.weight" in state
    meta = json.loads((out / learner.DIFFUSION_ADAPTER_METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["kind"] == "yeto.diffusion.adapter"
    assert meta["model"] == "tiny"
    assert meta["cache"] == {"latents": True, "text_embeds": True}
    assert meta["trainable_modules"] == ["transformer"]
    assert meta["trainable_tensor_count"] == len(state)


def test_open_image_accepts_hf_image_bytes_dict():
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buf, format="PNG")

    got = learner._open_image({"bytes": buf.getvalue(), "path": None})

    assert got.mode == "RGB"
    assert got.size == (2, 2)


def test_raw_video_encode_uses_pipeline_pack_latents(tmp_path):
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from yeto.diffusion import learner

    clip = tmp_path / "clip"
    clip.mkdir()
    for i in range(2):
        Image.new("RGB", (4, 4), color=("red" if i == 0 else "blue")).save(
            clip / f"frame_{i:03d}.png"
        )

    class TinyLatentDist:
        def __init__(self, latents):
            self.latents = latents

        def sample(self):
            return self.latents

    class TinyVAE(torch.nn.Module):
        class config:
            scaling_factor = 1.0
            shift_factor = None

        def encode(self, sample):
            latents = torch.ones(
                sample.shape[0],
                4,
                2,
                3,
                5,
                device=sample.device,
                dtype=sample.dtype,
            )
            return SimpleNamespace(latent_dist=TinyLatentDist(latents))

    class TinyTransformer(torch.nn.Module):
        config = SimpleNamespace(patch_size=1, patch_size_t=1)

    class TinyPipe:
        def __init__(self):
            self.vae = TinyVAE()
            self.transformer = TinyTransformer()
            self.seen_pack = None

        def _pack_latents(self, latents, patch_size=1, patch_size_t=1):
            self.seen_pack = (tuple(latents.shape), patch_size, patch_size_t)
            return latents.flatten(2).transpose(1, 2)

    pipe = TinyPipe()
    got = learner.encode_latents(
        pipe,
        [{"video": "clip", "__yeto_data_root__": str(tmp_path)}],
        _args(height=4, width=4),
        torch.device("cpu"),
        torch.float32,
    )

    assert tuple(got.latents.shape) == (1, 30, 4)
    assert (got.latent_num_frames, got.latent_height, got.latent_width) == (2, 3, 5)
    assert pipe.seen_pack == ((1, 4, 2, 3, 5), 1, 1)


def test_raw_image_lora_adapter_round_trip(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    Image = pytest.importorskip("PIL.Image")
    from yeto.diffusion import learner, sample

    adapter_file = tmp_path / "raw_adapter.py"
    adapter_file.write_text(
        "from types import SimpleNamespace\n"
        "import torch\n"
        "from yeto.diffusion.learner import diffusion_torch_dtype\n"
        "\n"
        "class TinyLatentDist:\n"
        "    def __init__(self, latents):\n"
        "        self.latents = latents\n"
        "    def sample(self):\n"
        "        return self.latents\n"
        "\n"
        "class TinyVAE(torch.nn.Module):\n"
        "    class config:\n"
        "        scaling_factor = 1.0\n"
        "        shift_factor = 0.0\n"
        "    def encode(self, sample):\n"
        "        flat = sample.mean(dim=(2, 3))\n"
        "        latents = torch.nn.functional.pad(flat, (0, 4 - flat.shape[1]))[:, :4]\n"
        "        return SimpleNamespace(latent_dist=TinyLatentDist(latents))\n"
        "\n"
        "class TinyDenoiser(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.to_q = torch.nn.Linear(4, 4)\n"
        "    def forward(self, hidden_states, timestep, encoder_hidden_states=None, return_dict=False):\n"
        "        del timestep, return_dict\n"
        "        cond = encoder_hidden_states\n"
        "        if cond.ndim > hidden_states.ndim:\n"
        "            cond = cond.mean(dim=1)\n"
        "        return self.to_q(hidden_states + cond)\n"
        "\n"
        "class TinyScheduler:\n"
        "    class config:\n"
        "        num_train_timesteps = 4\n"
        "        prediction_type = 'epsilon'\n"
        "    def add_noise(self, latents, noise, timesteps):\n"
        "        del timesteps\n"
        "        return latents + 0.1 * noise\n"
        "\n"
        "class TinyPipe:\n"
        "    def __init__(self):\n"
        "        self.vae = TinyVAE()\n"
        "        self.transformer = TinyDenoiser()\n"
        "        self.scheduler = TinyScheduler()\n"
        "        self.components = {'vae': self.vae, 'transformer': self.transformer}\n"
        "    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1, do_classifier_free_guidance=False):\n"
        "        del num_images_per_prompt, do_classifier_free_guidance\n"
        "        vals = [[float(len(p)), 1.0, 0.0, 0.0] for p in prompt]\n"
        "        return torch.tensor(vals, device=device or 'cpu')\n"
        "    def to(self, device):\n"
        "        dtype = diffusion_torch_dtype(torch.device(device))\n"
        "        self.vae.to(device=device, dtype=dtype)\n"
        "        self.transformer.to(device=device, dtype=dtype)\n"
        "        return self\n"
        "\n"
        "def make_adapter():\n"
        "    class Adapter:\n"
        "        def load_pipeline(self, args, device):\n"
        "            del args, device\n"
        "            return TinyPipe()\n"
        "    return Adapter()\n"
    )
    for i, color in enumerate(("red", "blue")):
        Image.new("RGB", (2, 2), color=color).save(tmp_path / f"img_{i}.png")
    data = tmp_path / "data.jsonl"
    with data.open("w", encoding="utf-8") as f:
        for i in range(2):
            f.write(json.dumps({"image": f"img_{i}.png", "prompt": f"prompt {i}"}) + "\n")
    out = tmp_path / "out"

    learner.main(
        [
            "--model",
            "tiny",
            "--data",
            str(data),
            "--syncer",
            "none",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
            "--tuning",
            "lora",
            "--lora-r",
            "2",
            "--lora-alpha",
            "4",
            "--diffusion-adapter",
            f"{adapter_file}:make_adapter",
            "--micro-batch-size",
            "2",
            "--stream-workers",
            "0",
            "--fragments",
            "1",
            "--max-local-steps",
            "1",
            "--output-dir",
            str(out),
        ]
    )

    meta = json.loads((out / learner.DIFFUSION_ADAPTER_METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["cache"] == {"latents": False, "text_embeds": False}
    assert (out / "transformer" / "adapter_config.json").exists()

    class TinySampleDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(4, 4)

        def forward(self, hidden_states, **kwargs):
            del kwargs
            return self.to_q(hidden_states)

    class TinySamplePipe:
        def __init__(self):
            self.transformer = TinySampleDenoiser()
            self.components = {"transformer": self.transformer}
            self.moved_to = None
            self.evaluated = False

        def to(self, device):
            self.moved_to = device
            self.transformer.to(device)
            return self

        def eval(self):
            self.evaluated = True

        def __call__(self, prompt, **kwargs):
            del prompt, kwargs
            return {"images": [Image.new("RGB", (1, 1), color="green")]}

    monkeypatch.setattr(sample, "_load_base_pipeline", lambda model_id, device, dtype: TinySamplePipe())
    pipe, loaded_meta, adapter = sample.load_artifact_pipeline(
        out,
        SimpleNamespace(model=None, diffusion_adapter=None, device="cpu", dtype="auto"),
    )

    assert loaded_meta["model"] == "tiny"
    assert adapter is not None
    assert hasattr(pipe.transformer, "peft_config")
    assert pipe.evaluated is True
    sampled = sample.run_sample(
        pipe,
        SimpleNamespace(
            prompt="a prompt",
            num_inference_steps=None,
            guidance_scale=None,
            height=None,
            width=None,
            num_frames=None,
            seed=None,
            device="cpu",
        ),
        loaded_meta,
        adapter,
    )
    assert sampled["images"][0].size == (1, 1)


def test_denoise_forward_filters_kwargs_by_signature():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states=None,
            pooled_projections=None,
            height=None,
            return_dict=False,
        ):
            self.seen = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_projections,
                "height": height,
                "return_dict": return_dict,
            }
            return hidden_states + 1

    pipe = argparse.Namespace(transformer=TinyDenoiser())
    noisy = learner.LatentBatch(torch.zeros(2, 3), latent_height=8, latent_width=8)
    cond = learner.TextConditioning(
        torch.ones(2, 4),
        pooled_prompt_embeds=torch.ones(2, 5),
        attention_mask=torch.ones(2, 4),
    )
    out = learner.denoise_forward(pipe, noisy, torch.tensor([1, 2]), cond, argparse.Namespace())
    assert torch.equal(out, torch.ones(2, 3))
    assert pipe.transformer.seen["height"] == 8
    assert "width" not in pipe.transformer.seen
    assert pipe.transformer.seen["return_dict"] is False


def test_denoise_forward_inspects_wrapped_base_model_signature():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden_states, timestep, encoder_hidden_states=None, return_dict=False):
            self.seen = (hidden_states, timestep, encoder_hidden_states, return_dict)
            return hidden_states + 1

    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def get_base_model(self):
            return self.base

        def forward(self, *args, **kwargs):
            return self.base(*args, **kwargs)

    base = TinyDenoiser()
    pipe = argparse.Namespace(transformer=Wrapper(base))
    noisy = learner.LatentBatch(torch.zeros(1, 2))
    cond = learner.TextConditioning(torch.ones(1, 2))

    out = learner.denoise_forward(pipe, noisy, torch.tensor([1]), cond, argparse.Namespace())

    assert torch.equal(out, torch.ones(1, 2))
    assert base.seen[2] is cond.prompt_embeds


def test_launcher_routes_diffusion_to_diffusion_learner_and_opt_in_caches():
    args = _args(cache_latents=True, cache_text_embeds=True, height=512, width=512)
    task = make_learner_task(args, _SPEC, 0, 1, "1.2.3.4:29400")
    assert "-m yeto.diffusion.learner" in task.run
    assert "--loss-function flow_matching" in task.run
    assert "--cache-latents" in task.run and "--cache-text-embeds" in task.run
    assert "--height 512" in task.run and "--width 512" in task.run
    assert "diffusers" in task.setup
    assert "--diffusion-family" not in task.run
    assert "--train-on" not in task.run and "--seq-len" not in task.run
    assert "--tokenize" not in task.run


def test_launcher_routes_video_aliases_without_model_family_flags():
    ltx = make_learner_task(_args(model="ltx-video", bucket_by_shape=True), _SPEC, 0, 1, "a:1")
    assert "--model ltx-video" in ltx.run
    assert "--diffusion-family" not in ltx.run
    assert "--bucket-by-shape" in ltx.run
    assert "--text-attention-mask-column prompt_attention_mask" in ltx.run
    wan = make_learner_task(_args(model="wan22"), _SPEC, 0, 1, "a:1")
    assert "--model wan22" in wan.run
    assert "--diffusion-family" not in wan.run


def test_launcher_passes_diffusion_adapter_hook():
    args = _args(model="nava", diffusion_adapter="my_adapter:make")
    task = make_learner_task(args, _SPEC, 0, 1, "a:1")
    assert "--diffusion-adapter my_adapter:make" in task.run
    assert "--diffusion-family" not in task.run
    assert "libsndfile1" not in task.setup


def _sample_args(tmp_path, **over):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    base = dict(
        gpu="aws:1xa100@us-east-2",
        adapter_dir=str(adapter_dir),
        output=None,
        prompt="a cat",
        data=None,
        prompt_column="prompt",
        seed_column=None,
        max_rows=None,
        model=None,
        diffusion_adapter=None,
        dtype="auto",
        num_inference_steps=30,
        guidance_scale=None,
        height=None,
        width=None,
        num_frames=None,
        seed=None,
        fps=8,
        spot=True,
        disk_size=256,
        learner_cpus=None,
        learner_instance_type=None,
        learner_image=None,
        cluster_prefix="sample",
        keep=False,
        retry_until_up=False,
        controller_poll=30,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_diffusion_sample_task_uses_yeto_sample_and_mounts_data(tmp_path):
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"prompt":"p"}\n', encoding="utf-8")
    args = _sample_args(
        tmp_path,
        prompt=None,
        data=str(data),
        max_rows=2,
        seed=123,
        diffusion_adapter="hooks:make",
    )

    task = make_diffusion_sample_task(args, _SPEC)

    assert "python3 -m yeto.diffusion.sample" in task.run
    assert "--adapter-dir" in task.run and "~/yeto-adapter" in task.run
    assert "--data" in task.run and "~/yeto-data.jsonl" in task.run
    assert "--output-dir" in task.run and "~/yeto-output" in task.run
    assert "--max-rows 2" in task.run
    assert "--seed 123" in task.run
    assert "--diffusion-adapter hooks:make" in task.run
    assert task.file_mounts["~/yeto-adapter"] == args.adapter_dir
    assert task.file_mounts["~/yeto-data.jsonl"] == str(data)
    assert "diffusers" in task.setup
