import argparse

import pytest

from yeto.gpu_spec import ClusterSpec
from yeto.launcher import make_learner_task
from yeto.models import (
    DIFFUSION_MODEL_ALIASES,
    resolve,
    resolve_diffusion_family,
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
        diffusion_family="auto",
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
        frame_rate=25,
        bucket_by_shape=False,
        nava_root=None,
        nava_config="configs/nava.yaml",
        nava_checkpoint=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_diffusion_aliases_resolve_and_infer_kind():
    assert {"wan22", "ltx-video", "flux", "sd35", "nava"} <= set(DIFFUSION_MODEL_ALIASES)
    assert resolve("sd35") == DIFFUSION_MODEL_ALIASES["sd35"]
    assert resolve_model_kind("flux") == "diffusion"
    assert resolve_diffusion_family("ltx-video") == "ltx"
    assert resolve_diffusion_family("wan22") == "wan"
    assert resolve_diffusion_family("nava") == "nava"
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
    assert args.loss_function == "flow_matching"


def test_diffusion_lora_targets_are_dit_names():
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


def test_nava_lora_patches_only_backbone_blocks():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    from yeto.diffusion.nava_lora import LoRALinear, patch_lora

    class TinyNava(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.double_blocks = nn.ModuleList([
                nn.ModuleDict({"self_attn": nn.ModuleDict({"q": nn.Linear(4, 4)})})
            ])
            self.other = nn.Linear(4, 4)

    model = TinyNava()
    cfg = patch_lora(model, r=2, alpha=4, target="auto")
    assert cfg.patched_modules == ["backbone.double_blocks.0.self_attn.q"]
    assert isinstance(model.backbone.double_blocks[0]["self_attn"]["q"], LoRALinear)
    assert not model.other.weight.requires_grad


def test_launcher_routes_diffusion_to_diffusion_learner_and_opt_in_caches():
    args = _args(cache_latents=True, cache_text_embeds=True, height=512, width=512)
    task = make_learner_task(args, _SPEC, 0, 1, "1.2.3.4:29400")
    assert "-m yeto.diffusion.learner" in task.run
    assert "--loss-function flow_matching" in task.run
    assert "--cache-latents" in task.run and "--cache-text-embeds" in task.run
    assert "--height 512" in task.run and "--width 512" in task.run
    assert "diffusers" in task.setup
    assert "--train-on" not in task.run and "--seq-len" not in task.run
    assert "--tokenize" not in task.run


def test_launcher_routes_specific_video_families():
    ltx = make_learner_task(_args(model="ltx-video", bucket_by_shape=True), _SPEC, 0, 1, "a:1")
    assert "--diffusion-family ltx" in ltx.run
    assert "--bucket-by-shape" in ltx.run
    assert "--text-attention-mask-column prompt_attention_mask" in ltx.run
    wan = make_learner_task(_args(model="wan22", frame_rate=24), _SPEC, 0, 1, "a:1")
    assert "--diffusion-family wan" in wan.run
    assert "--frame-rate 24" in wan.run


def test_launcher_mounts_nava_root(tmp_path):
    root = tmp_path / "NAVA"
    root.mkdir()
    args = _args(model="nava", nava_root=str(root), nava_checkpoint=None)
    task = make_learner_task(args, _SPEC, 0, 1, "a:1")
    assert "--diffusion-family nava" in task.run
    assert "--nava-root ~/yeto-nava" in task.run
    assert task.file_mounts["~/yeto-nava"] == str(root)
    assert "libsndfile1" in task.setup
