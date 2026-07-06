import argparse
import json

import pytest

from yeto.gpu_spec import ClusterSpec
from yeto.launcher import make_learner_task
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
        "        self.transformer.to(device)\n"
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
