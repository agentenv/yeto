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
        diffusion_loss_weighting="none",
        diffusion_min_snr_gamma=5.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_diffusion_aliases_resolve_and_infer_kind():
    assert {
        "chroma1-base",
        "chroma1-hd",
        "cogvideox-2b",
        "cogvideox-5b",
        "cogvideox15-5b",
        "cogview4-6b",
        "flux",
        "flux-schnell",
        "flux2-dev",
        "hidream-i1-dev",
        "hidream-i1-full",
        "hunyuan-video",
        "ideogram4",
        "ltx-video",
        "ltx2-video",
        "motif-video",
        "nava",
        "qwen-image",
        "qwen-image-2512",
        "sd35",
        "wan21-t2v-1.3b",
        "wan21-t2v-14b",
        "wan22",
    } <= set(DIFFUSION_MODEL_ALIASES)
    assert resolve("sd35") == DIFFUSION_MODEL_ALIASES["sd35"]
    assert resolve("ideogram4") == "ideogram-ai/ideogram-4-nf4-diffusers"
    assert resolve("qwen-image") == "Qwen/Qwen-Image"
    assert resolve("hunyuan-video") == "hunyuanvideo-community/HunyuanVideo"
    assert resolve("nava") == "baidu/NAVA"
    assert resolve_model_kind("flux") == "diffusion"
    assert resolve_model_kind("ideogram4") == "diffusion"
    assert resolve_model_kind("qwen-image") == "diffusion"
    assert resolve_model_kind("org/custom", "diffusion") == "diffusion"
    assert resolve_model_kind("org/custom") == "causal-lm"


def test_diffusion_capability_matrix_covers_aliases():
    from yeto.diffusion.capabilities import (
        DIFFUSION_CAPABILITIES,
        VALID_CAPABILITY_STATUSES,
        aliases_by_status,
        format_capability_table,
    )

    assert set(DIFFUSION_CAPABILITIES) == set(DIFFUSION_MODEL_ALIASES)
    for alias, cap in DIFFUSION_CAPABILITIES.items():
        assert cap.status in VALID_CAPABILITY_STATUSES, alias
        assert cap.family
        assert cap.pipeline
        assert cap.modalities
        assert cap.forward_kwargs
    assert "nava" in aliases_by_status("adapter-required")
    assert "cogvideox-2b" in aliases_by_status("generic-gap")
    assert "flux" in aliases_by_status("needs-real-validation")
    assert "wan22" in aliases_by_status("needs-real-validation")
    assert "| `wan22` |" in format_capability_table(("wan22",))


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
    assert args.diffusion_loss_weighting == "none"
    assert args.diffusion_min_snr_gamma == 5.0


def test_diffusion_lora_targets_are_generic_dit_names():
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    targets = learner.resolve_lora_targets("auto", "sd35")
    assert "to_q" in targets and "to_k" in targets and "to_v" in targets
    assert "ff.net.0.proj" in targets
    assert learner.resolve_lora_targets("all-linear", "sd35") == "all-linear"


def test_diffusion_trainable_modules_skip_none_placeholders():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    pipe = SimpleNamespace(
        transformer=torch.nn.Linear(2, 2),
        transformer_2=None,
        unet="not a module",
    )

    assert learner._trainable_module_items(pipe) == [("transformer", pipe.transformer)]


def test_diffusion_multi_denoiser_routes_by_boundary_timestep():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
            self.seen = []

        def forward(self, hidden_states, timestep, encoder_hidden_states=None, return_dict=False):
            del return_dict
            self.seen.append((hidden_states.detach().clone(), timestep.detach().clone(), encoder_hidden_states))
            return (hidden_states + self.offset,)

    high = TinyDenoiser(1.0)
    low = TinyDenoiser(2.0)
    pipe = SimpleNamespace(
        transformer=high,
        transformer_2=low,
        scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
        config=SimpleNamespace(boundary_ratio=0.5),
    )
    noisy = learner.LatentBatch(torch.zeros(4, 2))
    timesteps = torch.tensor([750.0, 250.0, 600.0, 100.0])
    prompt_embeds = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    cond = learner.TextConditioning(prompt_embeds)

    out = learner.denoise_forward(pipe, noisy, timesteps, cond, _args())

    assert torch.equal(out[:, 0], torch.tensor([1.0, 2.0, 1.0, 2.0]))
    assert torch.equal(high.seen[0][1], torch.tensor([750.0, 600.0]))
    assert torch.equal(low.seen[0][1], torch.tensor([250.0, 100.0]))
    assert torch.equal(high.seen[0][2], prompt_embeds[[0, 2]])
    assert torch.equal(low.seen[0][2], prompt_embeds[[1, 3]])


def test_diffusion_multi_denoiser_uses_second_model_for_low_noise_batch():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, hidden_states, timestep, return_dict=False):
            del timestep, return_dict
            return (hidden_states + self.offset,)

    pipe = SimpleNamespace(
        transformer=TinyDenoiser(1.0),
        transformer_2=TinyDenoiser(2.0),
        scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)),
        config=SimpleNamespace(boundary_ratio=0.5),
    )

    out = learner.denoise_forward(
        pipe,
        learner.LatentBatch(torch.zeros(2, 1)),
        torch.tensor([100.0, 200.0]),
        learner.TextConditioning(None),
        _args(),
    )

    assert torch.equal(out, torch.full((2, 1), 2.0))


def test_diffusion_aligns_packed_output_with_pipeline_unpack_helper():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyPipe:
        vae_scale_factor = 1

        @staticmethod
        def _unpack_latents(latents, height, width, vae_scale_factor):
            assert (height, width, vae_scale_factor) == (2, 2, 1)
            return latents.reshape(latents.shape[0], 1, 2, 2)

    pred = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    target = torch.zeros(2, 1, 2, 2)
    noisy = learner.LatentBatch(pred, latent_height=2, latent_width=2)

    aligned_pred, aligned_target = learner._align_prediction_and_target(
        TinyPipe(),
        pred,
        target,
        noisy,
        learner.TextConditioning(None),
    )

    assert torch.equal(aligned_pred, pred.reshape(2, 1, 2, 2))
    assert aligned_target is target


def test_diffusion_aligns_extra_output_tokens_to_target_tokens():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    pred = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    target = torch.zeros(2, 3, 1)

    aligned_pred, aligned_target = learner._align_prediction_and_target(
        SimpleNamespace(),
        pred,
        target,
        learner.LatentBatch(pred),
        learner.TextConditioning(None),
    )

    assert torch.equal(aligned_pred, pred[:, :3])
    assert aligned_target is target


def test_diffusion_unpack_with_ids_crops_extra_output_tokens():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyPipe:
        @staticmethod
        def _unpack_latents_with_ids(x, x_ids):
            assert x.shape[1] == x_ids.shape[1] == 4
            return x.reshape(x.shape[0], 1, 2, 2)

    pred = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    target = torch.zeros(2, 1, 2, 2)
    x_ids = torch.zeros(2, 4, 4)

    aligned_pred, aligned_target = learner._align_prediction_and_target(
        TinyPipe(),
        pred,
        target,
        learner.LatentBatch(pred, latent_height=2, latent_width=2),
        learner.TextConditioning(None, extra={"x_ids": x_ids}),
    )

    assert torch.equal(aligned_pred, pred[:, :4].reshape(2, 1, 2, 2))
    assert aligned_target is target


def test_diffusion_adapter_base_is_marker_not_hook_provider():
    pytest.importorskip("torch")
    from yeto.diffusion.adapters import DiffusionAdapter, DiffusionAdapterProtocol
    from yeto.diffusion.adapters.nava import NavaAdapter

    marker = DiffusionAdapter()
    assert not hasattr(marker, "load_pipeline")
    assert not hasattr(marker, "training_step")
    assert hasattr(DiffusionAdapterProtocol, "load_pipeline")
    assert issubclass(NavaAdapter, DiffusionAdapter)


def test_diffusion_adapter_template_loads():
    pytest.importorskip("torch")
    from yeto.diffusion import learner
    from yeto.diffusion.adapters.base import DiffusionAdapter
    from yeto.diffusion.adapters import template

    adapter = learner.load_diffusion_adapter("yeto.diffusion.adapters.template:make_adapter")

    assert isinstance(adapter, DiffusionAdapter)
    assert hasattr(adapter, "training_step")
    assert not hasattr(adapter, "save_adapters")

    load_only = template.make_adapter(mode="load-only")
    assert hasattr(load_only, "load_pipeline")
    assert not hasattr(load_only, "training_step")
    assert not hasattr(load_only, "encode_latents")

    encoding = template.make_adapter(mode="encoding")
    assert hasattr(encoding, "encode_latents")
    assert hasattr(encoding, "encode_prompt_embeds")
    assert not hasattr(encoding, "training_step")

    sampling = template.make_adapter(mode="sampling")
    assert hasattr(sampling, "save_adapters")
    assert hasattr(sampling, "load_sample_pipeline")
    assert hasattr(sampling, "sample")


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


def test_flow_matching_loss_applies_optional_weights():
    torch = pytest.importorskip("torch")
    from yeto.losses import flow_matching_loss

    pred = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    loss, denom = flow_matching_loss(pred, target, torch.tensor([1, 2]), torch.tensor([1.0, 2.0]))

    assert loss == 9
    assert denom == 9


def test_diffusion_loss_weights_use_scheduler_sigmas():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyScheduler:
        class config:
            num_train_timesteps = 4

        sigmas = torch.tensor([1.0, 0.5])
        timesteps = torch.tensor([10.0, 20.0])

    pipe = argparse.Namespace(scheduler=TinyScheduler())
    timesteps = torch.tensor([10.0, 20.0])
    target = torch.zeros(2, 1)

    assert learner.diffusion_loss_weights(pipe, timesteps, target, _args()) is None
    sigma = learner.diffusion_loss_weights(
        pipe,
        timesteps,
        target,
        _args(diffusion_loss_weighting="sigma"),
    )
    assert torch.allclose(sigma, torch.tensor([1.0, 0.25]))
    min_snr = learner.diffusion_loss_weights(
        pipe,
        timesteps,
        target,
        _args(diffusion_loss_weighting="min-snr", diffusion_min_snr_gamma=2.0),
    )
    assert torch.allclose(min_snr, torch.tensor([1.0, 0.5]))


def test_diffusion_loss_weights_linear_uses_normalized_timesteps():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyScheduler:
        class config:
            num_train_timesteps = 4

    weights = learner.diffusion_loss_weights(
        argparse.Namespace(scheduler=TinyScheduler()),
        torch.tensor([1, 2]),
        torch.zeros(2, 1),
        _args(diffusion_loss_weighting="linear"),
    )

    assert torch.allclose(weights, torch.tensor([0.25, 0.5]))


def test_add_noise_uses_scheduler_timesteps_sigmas_and_scale_model_input(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None, dtype=None: torch.tensor([1, 2], device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.full_like(x, 2.0))

    class TinyScheduler:
        class config:
            num_train_timesteps = 4

        sigmas = torch.tensor([0.0, 0.25, 0.75, 1.0])
        timesteps = torch.tensor([1000.0, 700.0, 300.0, 0.0])

        def __init__(self):
            self.seen = None

        def scale_model_input(self, sample, timestep):
            self.seen = (sample.clone(), timestep.clone())
            return sample + 10.0

    scheduler = TinyScheduler()
    pipe = argparse.Namespace(scheduler=scheduler)
    batch = learner.LatentBatch(torch.zeros(2, 1))

    noisy, target, timesteps = learner.add_noise_and_target(pipe, batch)

    assert torch.equal(timesteps, torch.tensor([700.0, 300.0]))
    assert torch.allclose(scheduler.seen[0], torch.tensor([[0.5], [1.5]]))
    assert torch.equal(scheduler.seen[1], timesteps)
    assert torch.allclose(noisy.latents, torch.tensor([[10.5], [11.5]]))
    assert torch.allclose(target, torch.full((2, 1), 2.0))


def test_add_noise_prefers_scheduler_scale_noise(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None, dtype=None: torch.tensor([0, 2], device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)

    class TinyScheduler:
        class config:
            num_train_timesteps = 3

        timesteps = torch.tensor([10.0, 20.0, 30.0])

        def __init__(self):
            self.called = False

        def scale_noise(self, sample, timestep, noise):
            self.called = True
            return sample + noise + timestep.view(-1, 1)

        def add_noise(self, original_samples, noise, timesteps):
            del original_samples, noise, timesteps
            raise AssertionError("scale_noise should be used before add_noise")

    scheduler = TinyScheduler()
    pipe = argparse.Namespace(scheduler=scheduler)
    latents = torch.tensor([[2.0], [4.0]])

    noisy, target, timesteps = learner.add_noise_and_target(pipe, learner.LatentBatch(latents))

    assert scheduler.called is True
    assert torch.equal(timesteps, torch.tensor([10.0, 30.0]))
    assert torch.allclose(noisy.latents, torch.tensor([[13.0], [35.0]]))
    assert torch.allclose(target, torch.tensor([[-1.0], [-3.0]]))


def test_add_noise_uses_named_add_noise_and_sample_prediction_target(monkeypatch):
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None, dtype=None: torch.tensor([1, 2], device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.full_like(x, 3.0))

    class TinyScheduler:
        class config:
            num_train_timesteps = 3
            prediction_type = "sample"

        def __init__(self):
            self.seen = None

        def add_noise(self, original_samples, noise, timesteps):
            self.seen = (original_samples, noise, timesteps)
            return original_samples + 5.0 * noise

    scheduler = TinyScheduler()
    pipe = argparse.Namespace(scheduler=scheduler)
    latents = torch.tensor([[2.0], [4.0]])

    noisy, target, timesteps = learner.add_noise_and_target(pipe, learner.LatentBatch(latents))

    assert torch.equal(timesteps, torch.tensor([1, 2]))
    assert scheduler.seen[0] is latents
    assert torch.equal(scheduler.seen[1], torch.full_like(latents, 3.0))
    assert scheduler.seen[2] is timesteps
    assert torch.allclose(noisy.latents, torch.tensor([[17.0], [19.0]]))
    assert target is latents


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


def test_diffusion_cache_metadata_validates_external_contract(tmp_path):
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    args = _args(
        cache_latents=True,
        cache_text_embeds=True,
        data=str(tmp_path / "data.jsonl"),
    )
    meta_path = tmp_path / learner.DIFFUSION_CACHE_METADATA_FILE
    meta_path.write_text(
        json.dumps(
            {
                "kind": "yeto.diffusion.cache",
                "schema_version": learner.DIFFUSION_CACHE_SCHEMA_VERSION,
                "row_count": 3,
                "cache": {"latents": True, "text_embeds": True},
                "columns": {
                    "latents": "latents",
                    "prompt_embeds": "prompt_embeds",
                    "prompt_attention_mask": "prompt_attention_mask",
                    "pooled_prompt_embeds": "pooled_prompt_embeds",
                },
                "relative_paths": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

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


def test_diffusion_shape_key_uses_cli_target_over_source_metadata():
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    args = _args(height=256, width=256, num_frames=121)
    row = {"height": 720, "width": 1280, "num_frames": 300}

    assert learner._shape_key(row, args) == (121, 256, 256)


def test_streaming_diffusion_rows_bucket_uses_target_shape_over_source_metadata():
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    dataset = learner.StreamingDiffusionRows(
        [
            {"video": "a.mp4", "height": 720, "width": 1280, "num_frames": 300},
            {"video": "b.mp4", "height": 854, "width": 480, "num_frames": 125},
        ],
        learner_id=0,
        num_learners=1,
        micro_batch_size=2,
        bucket_by_shape=True,
        target_num_frames=121,
        target_height=256,
        target_width=256,
    )

    batch = next(iter(dataset))

    assert len(batch) == 2
    assert {row["video"] for row in batch} == {"a.mp4", "b.mp4"}


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


def test_nava_adapter_factory_and_row_conversion(tmp_path):
    pytest.importorskip("torch")
    from yeto.diffusion import learner
    from yeto.diffusion.adapters import nava

    adapter = learner.load_diffusion_adapter("yeto.diffusion.adapters.nava:make_adapter")
    assert isinstance(adapter, nava.NavaAdapter)

    args = SimpleNamespace(
        video_column="video",
        prompt_column="prompt",
        width=None,
        height=None,
        num_frames=120,
    )
    row = {
        "__yeto_data_root__": str(tmp_path),
        "video": "clip.mp4",
        "prompt": "a caption",
        "duration": 5.0,
        "height": 720,
        "width": 1280,
    }

    got = nava._nava_json_row(row, args)

    assert got["text_list"][0]["text"] == "a caption"
    assert got["video_info"][0]["data_path"] == str(tmp_path / "clip.mp4")
    assert got["video_info"][0]["duration"] == 5.0
    assert got["video_info"][0]["image_height"] == 720
    assert got["video_info"][0]["image_width"] == 1280

    no_duration = {
        "__yeto_data_root__": str(tmp_path),
        "video": "clip2.mp4",
        "prompt": "another caption",
    }
    assert nava._nava_json_row(no_duration, args)["video_info"][0]["duration"] == 5.0


def test_nava_adapter_direct_latent_training_step():
    torch = pytest.importorskip("torch")
    from yeto.diffusion.adapters.nava import make_adapter

    class TinyNavaPipe:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.cfg = {"data": {}}
            self.seen = None

        def forward(self, batch, global_step=None):
            self.seen = (batch, global_step)
            return self.model(torch.ones(1, 1)).sum(), {"ddpm": torch.ones(())}

    rows = [
        {
            "captions": "one",
            "audio_latents": torch.ones(3, 8),
            "video_latents": torch.ones(2, 4, 5, 6),
            "spk_embs": [],
        },
        {
            "captions": "two",
            "audio_latents": torch.ones(3, 8),
            "video_latents": torch.ones(2, 4, 5, 6),
            "spk_embs": [],
        },
    ]
    pipe = TinyNavaPipe()
    adapter = make_adapter()

    loss, denom = adapter.training_step(
        pipe,
        rows,
        SimpleNamespace(micro_batch_size=2),
        torch.device("cpu"),
        global_step=7,
    )

    batch, global_step = pipe.seen
    assert loss.ndim == 0
    assert denom.item() == 1
    assert global_step == 7
    assert batch["captions"] == ["one", "two"]
    assert batch["t_h_w_list"] == [(2, 4, 5), (2, 4, 5)]


def test_nava_adapter_raw_state_save_and_load(tmp_path):
    torch = pytest.importorskip("torch")
    from yeto.diffusion.adapters.nava import make_adapter

    adapter = make_adapter()
    pipe = SimpleNamespace(model=torch.nn.Linear(2, 2))
    with torch.no_grad():
        pipe.model.weight.fill_(3.0)
        pipe.model.bias.fill_(4.0)

    adapter.save_adapters(pipe, tmp_path)

    assert (tmp_path / "model_state.pt").exists()
    loaded = SimpleNamespace(model=torch.nn.Linear(2, 2))
    adapter.load_adapters(loaded, tmp_path, {}, SimpleNamespace())
    assert torch.allclose(loaded.model.weight, torch.full_like(loaded.model.weight, 3.0))
    assert torch.allclose(loaded.model.bias, torch.full_like(loaded.model.bias, 4.0))


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
    assert meta["loss"] == {"function": "flow_matching", "weighting": "none", "min_snr_gamma": 5.0}
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


def test_fit_video_frames_samples_and_pads_deterministically():
    pytest.importorskip("torch")
    from yeto.diffusion import learner

    frames = list(range(5))

    assert learner._fit_video_frames(frames, None) == frames
    assert learner._fit_video_frames(frames, 3) == [0, 2, 4]
    assert learner._fit_video_frames(frames, 1) == [2]
    assert learner._fit_video_frames([0, 1], 4) == [0, 1, 1, 1]


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


def test_raw_video_encode_fits_variable_frame_counts_to_target_shape(tmp_path):
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from yeto.diffusion import learner

    for name, count in (("short", 2), ("long", 5)):
        clip = tmp_path / name
        clip.mkdir()
        for i in range(count):
            Image.new("RGB", (6, 8), color=(i * 20, 0, 0)).save(clip / f"frame_{i:03d}.png")

    class TinyLatentDist:
        def __init__(self, latents):
            self.latents = latents

        def sample(self):
            return self.latents

    class TinyVAE(torch.nn.Module):
        class config:
            scaling_factor = 1.0
            shift_factor = 0.0

        def __init__(self):
            super().__init__()
            self.seen_shape = None

        def encode(self, sample):
            self.seen_shape = tuple(sample.shape)
            latents = torch.ones(
                sample.shape[0],
                4,
                sample.shape[2],
                2,
                2,
                device=sample.device,
                dtype=sample.dtype,
            )
            return SimpleNamespace(latent_dist=TinyLatentDist(latents))

    class TinyPipe:
        def __init__(self):
            self.vae = TinyVAE()

    pipe = TinyPipe()
    rows = [
        {"video": "short", "__yeto_data_root__": str(tmp_path)},
        {"video": "long", "__yeto_data_root__": str(tmp_path)},
    ]

    got = learner.encode_latents(
        pipe,
        rows,
        _args(height=4, width=4, num_frames=3),
        torch.device("cpu"),
        torch.float32,
    )

    assert pipe.vae.seen_shape == (2, 3, 3, 4, 4)
    assert tuple(got.latents.shape) == (2, 4, 3, 2, 2)
    assert (got.latent_num_frames, got.latent_height, got.latent_width) == (3, 2, 2)


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


def test_prompt_conditioning_extra_fields_are_signature_filtered():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyConfig:
        patch_size = 2

    class TinyDenoiser(torch.nn.Module):
        config = TinyConfig()

        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states=None,
            position_ids=None,
            segment_ids=None,
            indicator=None,
            return_dict=False,
        ):
            self.seen = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "position_ids": position_ids,
                "segment_ids": segment_ids,
                "indicator": indicator,
                "return_dict": return_dict,
            }
            return hidden_states + 1

    class TinyPipe:
        def __init__(self):
            self.transformer = TinyDenoiser()
            self.prompt_seen = None

        def __call__(self, prompt=None, max_sequence_length=77):
            del prompt, max_sequence_length

        def encode_prompt(self, prompt, grid_h, grid_w, max_sequence_length, device=None):
            self.prompt_seen = {
                "prompt": prompt,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "max_sequence_length": max_sequence_length,
            }
            batch = len(prompt)
            return (
                torch.ones(batch, 2, 4, device=device),
                torch.arange(2, device=device).repeat(batch, 1),
                torch.arange(2, device=device).repeat(batch, 1) + 10,
                torch.ones(batch, 2, dtype=torch.long, device=device),
            )

    pipe = TinyPipe()
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    latents = learner.LatentBatch(torch.zeros(2, 4, 16, 8), latent_height=16, latent_width=8)

    cond = learner.encode_prompt_embeds(
        pipe,
        rows,
        _args(),
        torch.device("cpu"),
        torch.float32,
        latents=latents,
    )
    cond.extra["ignored_by_forward"] = torch.tensor([1])
    noisy = learner.LatentBatch(torch.zeros(2, 3), latent_height=16, latent_width=8)
    out = learner.denoise_forward(pipe, noisy, torch.tensor([1, 2]), cond, argparse.Namespace())

    assert torch.equal(out, torch.ones(2, 3))
    assert pipe.prompt_seen["grid_h"] == 8
    assert pipe.prompt_seen["grid_w"] == 4
    assert pipe.prompt_seen["max_sequence_length"] == 77
    assert set(cond.extra) == {"position_ids", "segment_ids", "indicator", "ignored_by_forward"}
    assert torch.equal(pipe.transformer.seen["position_ids"], cond.extra["position_ids"])
    assert torch.equal(pipe.transformer.seen["segment_ids"], cond.extra["segment_ids"])
    assert torch.equal(pipe.transformer.seen["indicator"], cond.extra["indicator"])
    assert "ignored_by_forward" not in pipe.transformer.seen


def test_encode_prompt_maps_signature_prompt_variants_from_rows():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyPipe:
        def __init__(self):
            self.seen = None

        def encode_prompt(
            self,
            prompt,
            prompt_2,
            prompt_4,
            negative_prompt,
            negative_prompt_2=None,
            do_classifier_free_guidance=False,
            device=None,
        ):
            self.seen = {
                "prompt": prompt,
                "prompt_2": prompt_2,
                "prompt_4": prompt_4,
                "negative_prompt": negative_prompt,
                "negative_prompt_2": negative_prompt_2,
                "do_classifier_free_guidance": do_classifier_free_guidance,
            }
            return torch.ones(len(prompt), 2, 3, device=device)

    pipe = TinyPipe()
    rows = [
        {
            "prompt": "base a",
            "prompt_2": "second a",
            "prompt_4": "fourth a",
            "negative_prompt": "neg a",
        },
        {"prompt": "base b", "negative_prompt_2": "neg2 b"},
    ]

    cond = learner.encode_prompt_embeds(pipe, rows, _args(), torch.device("cpu"), torch.float32)

    assert tuple(cond.prompt_embeds.shape) == (2, 2, 3)
    assert pipe.seen["prompt"] == ["base a", "base b"]
    assert pipe.seen["prompt_2"] == ["second a", "base b"]
    assert pipe.seen["prompt_4"] == ["fourth a", "base b"]
    assert pipe.seen["negative_prompt"] == ["neg a", ""]
    assert pipe.seen["negative_prompt_2"] == ["", "neg2 b"]
    assert pipe.seen["do_classifier_free_guidance"] is False


def test_cached_text_embeds_include_signature_extra_columns():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states=None,
            encoder_hidden_states_t5=None,
            position_ids=None,
        ):
            del hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_t5, position_ids

    pipe = argparse.Namespace(transformer=TinyDenoiser())
    rows = [
        {
            "prompt_embeds": [[1.0, 2.0]],
            "prompt_embeds_t5": [[3.0, 4.0]],
            "position_ids": [0, 1],
        },
        {
            "prompt_embeds": [[5.0, 6.0]],
            "prompt_embeds_t5": [[7.0, 8.0]],
            "position_ids": [2, 3],
        },
    ]

    cond = learner.encode_prompt_embeds(
        pipe,
        rows,
        _args(cache_text_embeds=True),
        torch.device("cpu"),
        torch.float32,
    )

    assert tuple(cond.prompt_embeds.shape) == (2, 1, 2)
    assert set(cond.extra) == {"encoder_hidden_states_t5", "position_ids"}
    assert tuple(cond.extra["encoder_hidden_states_t5"].shape) == (2, 1, 2)
    assert tuple(cond.extra["position_ids"].shape) == (2, 2)
    assert cond.extra["position_ids"].dtype == torch.long


def test_raw_prompt_rows_include_signature_tensor_extra_columns():
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
            audio_hidden_states=None,
            audio_encoder_hidden_states=None,
            condition_mask=None,
            text_note=None,
        ):
            del timestep
            self.seen = {
                "encoder_hidden_states": encoder_hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "condition_mask": condition_mask,
                "text_note": text_note,
            }
            return hidden_states + 1

    class TinyPipe:
        def __init__(self):
            self.transformer = TinyDenoiser()

        def encode_prompt(self, prompt, device=None):
            return torch.ones(len(prompt), 2, 4, device=device)

    pipe = TinyPipe()
    rows = [
        {
            "prompt": "a",
            "audio_hidden_states": [[1.0, 2.0]],
            "audio_encoder_hidden_states": [[3.0, 4.0]],
            "condition_mask": [1, 0],
            "text_note": "not a tensor",
        },
        {
            "prompt": "b",
            "audio_hidden_states": [[5.0, 6.0]],
            "audio_encoder_hidden_states": [[7.0, 8.0]],
            "condition_mask": [0, 1],
            "text_note": "still not a tensor",
        },
    ]

    cond = learner.encode_prompt_embeds(pipe, rows, _args(), torch.device("cpu"), torch.float32)
    assert set(cond.extra) == {"audio_hidden_states", "audio_encoder_hidden_states", "condition_mask"}
    assert tuple(cond.extra["audio_hidden_states"].shape) == (2, 1, 2)
    assert tuple(cond.extra["audio_encoder_hidden_states"].shape) == (2, 1, 2)
    assert cond.extra["condition_mask"].dtype == torch.long

    out = learner.denoise_forward(
        pipe,
        learner.LatentBatch(torch.zeros(2, 3)),
        torch.tensor([1, 2]),
        cond,
        argparse.Namespace(),
    )

    assert torch.equal(out, torch.ones(2, 3))
    assert pipe.transformer.seen["audio_hidden_states"] is cond.extra["audio_hidden_states"]
    assert pipe.transformer.seen["audio_encoder_hidden_states"] is cond.extra["audio_encoder_hidden_states"]
    assert pipe.transformer.seen["condition_mask"] is cond.extra["condition_mask"]
    assert pipe.transformer.seen["text_note"] is None


def test_denoise_forward_auto_fills_token_ids_guidance_and_attention_mask():
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
            img_ids=None,
            txt_ids=None,
            guidance=None,
            attention_mask=None,
            return_dict=False,
        ):
            self.seen = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_projections,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
                "guidance": guidance,
                "attention_mask": attention_mask,
                "return_dict": return_dict,
            }
            return hidden_states + 1

    class TinyPipe:
        def __init__(self):
            self.transformer = TinyDenoiser()

        def __call__(self, prompt=None, guidance_scale=4.5):
            del prompt, guidance_scale

    pipe = TinyPipe()
    noisy = learner.LatentBatch(torch.zeros(2, 4, 8), latent_height=4, latent_width=4)
    mask = torch.ones(2, 5, dtype=torch.long)
    cond = learner.TextConditioning(
        torch.ones(2, 5, 8),
        pooled_prompt_embeds=torch.ones(2, 8),
        attention_mask=mask,
    )

    out = learner.denoise_forward(pipe, noisy, torch.tensor([1, 2]), cond, argparse.Namespace())

    assert torch.equal(out, torch.ones(2, 4, 8))
    assert torch.equal(
        pipe.transformer.seen["img_ids"],
        torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=noisy.latents.dtype),
    )
    assert tuple(pipe.transformer.seen["txt_ids"].shape) == (5, 3)
    assert torch.equal(pipe.transformer.seen["txt_ids"], torch.zeros(5, 3))
    assert torch.equal(pipe.transformer.seen["guidance"], torch.full((2,), 4.5))
    assert pipe.transformer.seen["attention_mask"] is mask


def test_denoise_forward_prefers_pipeline_image_id_helper():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden_states, timestep, img_ids=None):
            del timestep
            self.seen = img_ids
            return hidden_states

    class TinyPipe:
        def __init__(self):
            self.transformer = TinyDenoiser()

        def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
            del batch_size, height, width
            return torch.full((4, 3), 7, device=device, dtype=dtype)

    pipe = TinyPipe()
    noisy = learner.LatentBatch(torch.zeros(1, 4, 8), latent_height=4, latent_width=4)
    cond = learner.TextConditioning(torch.ones(1, 2, 8))

    learner.denoise_forward(pipe, noisy, torch.tensor([1]), cond, argparse.Namespace())

    assert torch.equal(pipe.transformer.seen, torch.full((4, 3), 7, dtype=noisy.latents.dtype))


def test_denoise_forward_auto_fills_shape_and_mask_aliases():
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
            encoder_hidden_states_t5=None,
            encoder_hidden_states_llama3=None,
            encoder_hidden_states_mask=None,
            hidden_states_masks=None,
            img_shapes=None,
            img_sizes=None,
            return_dict=False,
        ):
            self.seen = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states_t5": encoder_hidden_states_t5,
                "encoder_hidden_states_llama3": encoder_hidden_states_llama3,
                "encoder_hidden_states_mask": encoder_hidden_states_mask,
                "hidden_states_masks": hidden_states_masks,
                "img_shapes": img_shapes,
                "img_sizes": img_sizes,
                "return_dict": return_dict,
            }
            return hidden_states + 1

    pipe = argparse.Namespace(transformer=TinyDenoiser())
    noisy = learner.LatentBatch(torch.zeros(2, 12, 8), latent_num_frames=3, latent_height=4, latent_width=4)
    mask = torch.ones(2, 6, dtype=torch.long)
    cond = learner.TextConditioning(
        None,
        attention_mask=mask,
        extra={
            "encoder_hidden_states_t5": torch.ones(2, 6, 8),
            "encoder_hidden_states_llama3": torch.ones(2, 4, 8),
        },
    )

    out = learner.denoise_forward(pipe, noisy, torch.tensor([1, 2]), cond, argparse.Namespace())

    assert torch.equal(out, torch.ones(2, 12, 8))
    assert pipe.transformer.seen["encoder_hidden_states_t5"] is cond.extra["encoder_hidden_states_t5"]
    assert pipe.transformer.seen["encoder_hidden_states_llama3"] is cond.extra["encoder_hidden_states_llama3"]
    assert pipe.transformer.seen["encoder_hidden_states_mask"] is mask
    assert pipe.transformer.seen["hidden_states_masks"] is mask
    assert pipe.transformer.seen["img_shapes"] == [(3, 2, 2), (3, 2, 2)]
    assert pipe.transformer.seen["img_sizes"] == [(2, 2), (2, 2)]


def test_packed_sequence_conditioning_patchifies_image_latents():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        config = SimpleNamespace(in_channels=8, patch_size=2)

    pipe = SimpleNamespace(transformer=TinyDenoiser(), patch_size=2)
    latents = learner.LatentBatch(torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4))
    cond = learner.TextConditioning(
        torch.zeros(1, 7, 16),
        extra={"indicator": torch.tensor([[0, 3, 3, 2, 2, 2, 2]])},
    )

    aligned, mask = learner._align_latents_to_conditioning_sequence(pipe, latents, cond)

    assert tuple(aligned.latents.shape) == (1, 7, 8)
    assert torch.equal(mask, torch.tensor([[False, False, False, True, True, True, True]]))
    assert torch.equal(aligned.latents[:, :3], torch.zeros(1, 3, 8))
    expected_tokens = (
        latents.latents.reshape(1, 2, 2, 2, 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, 4, 8)
    )
    assert torch.equal(aligned.latents[:, 3:], expected_tokens)


def test_text_only_conditioning_does_not_patchify_video_latents():
    torch = pytest.importorskip("torch")
    from yeto.diffusion import learner

    class TinyDenoiser(torch.nn.Module):
        config = SimpleNamespace(in_channels=32, patch_size=(1, 2, 2))

    pipe = SimpleNamespace(transformer=TinyDenoiser())
    latents = learner.LatentBatch(torch.zeros(1, 8, 3, 8, 8))
    cond = learner.TextConditioning(torch.zeros(1, 5, 16))

    aligned, mask = learner._align_latents_to_conditioning_sequence(pipe, latents, cond)

    assert aligned is latents
    assert mask is None


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
    assert "--diffusion-loss-weighting" not in task.run
    assert "--diffusion-family" not in task.run
    assert "--train-on" not in task.run and "--seq-len" not in task.run
    assert "--tokenize" not in task.run


def test_launcher_passes_diffusion_loss_weighting_only_for_diffusion():
    task = make_learner_task(
        _args(diffusion_loss_weighting="min-snr", diffusion_min_snr_gamma=3.0),
        _SPEC,
        0,
        1,
        "a:1",
    )
    assert "--diffusion-loss-weighting min-snr" in task.run
    assert "--diffusion-min-snr-gamma 3.0" in task.run
    assert "--train-on" not in task.run
    lm_task = make_learner_task(
        _args(model="gemma4", diffusion_loss_weighting="min-snr"),
        _SPEC,
        0,
        1,
        "a:1",
    )
    assert "--diffusion-loss-weighting" not in lm_task.run
    assert "--train-on assistant" in lm_task.run


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
