import argparse
from pathlib import Path
import sys
import types

import pytest

from yeto import backends
from yeto.components import component_names, get_component
from yeto.gpu_spec import ClusterSpec
from yeto.models import get_model_spec, inferred_task


class FakeTask:
    def __init__(self, name=None, setup=None, run=None, envs=None, num_nodes=1, workdir=None, file_mounts=None):
        self.name = name
        self.setup = setup
        self.run = run
        self.envs = envs
        self.num_nodes = num_nodes
        self.workdir = workdir
        self.file_mounts = file_mounts
        self.resources = None

    def set_resources(self, resources):
        self.resources = resources


class FakeResources:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def install_fake_sky(monkeypatch):
    sky = types.ModuleType("sky")
    sky.Task = FakeTask
    sky.Resources = FakeResources
    monkeypatch.setitem(sys.modules, "sky", sky)


def test_builtin_backends_are_lm_and_generic_diffusion():
    assert backends.backend_names() == ["lm", "diffusion"]
    assert backends.default_task() == "lm"
    assert component_names() == ["nava"]
    assert get_component("nava").default_lora_targets == "mmdit-all-linear"
    assert inferred_task("nava") == "diffusion"
    assert get_model_spec("nava").component == "nava"


def _diffusion_args(tmp_path, **over):
    root = tmp_path / "NAVA"
    root.mkdir()
    ckpt = tmp_path / "base.safetensors"
    ckpt.write_bytes(b"placeholder")
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    base = dict(
        task="diffusion",
        model="nava",
        component="nava",
        component_root=str(root),
        component_config="configs/nava.yaml",
        base_checkpoint=str(ckpt),
        data=str(data),
        data_format="jsonl",
        modality="text_to_av",
        adapter="lora",
        trainable_regex=None,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_targets=None,
        merge_avg_regex=r"(^|\.)(bias|norm)(\.|$)",
        init_timeout=1800.0,
        shard="fsdp",
        fragments=8,
        fragment_pattern="binpack",
        merge_alpha=0.5,
        wire_dtype="bf16",
        wan_streams=4,
        batch_size=None,
        grad_accum=None,
        lr=None,
        weight_decay=None,
        warmup_steps=None,
        max_local_steps=None,
        num_workers=None,
        io_workers=None,
        disable_ema=False,
        save_every=100,
        learner_state_dir=None,
        learner_image="docker:nava-runtime",
        learner_cpus=None,
        learner_instance_type=None,
        spot=True,
        disk_size=512,
        syncer_memory=32,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_diffusion_backend_builds_component_learner_without_nava_task_flags(monkeypatch, tmp_path):
    install_fake_sky(monkeypatch)
    args = _diffusion_args(tmp_path)
    spec = ClusterSpec(cloud="aws", region="us-west-2", num_nodes=1, gpus_per_node=1, gpu="L4")

    task = backends.get_backend("diffusion").build_learner_task(args, spec, 0, 2, "203.0.113.10:29400")

    assert "-m yeto.diffusion.learner" in task.run
    assert "--component nava" in task.run
    assert "--base-checkpoint '~/yeto-base-checkpoint'" in task.run
    assert "--lora-targets mmdit-all-linear" in task.run
    assert "--nava-" not in task.run
    assert "torch_cuda_ok" in task.setup
    assert "pre_download" not in task.setup
    assert task.file_mounts["~/yeto-component"] == str(tmp_path / "NAVA")
    assert task.file_mounts["~/yeto-base-checkpoint"] == str(tmp_path / "base.safetensors")
    assert task.resources.kwargs["image_id"] == "docker:nava-runtime"



def test_model_nava_infers_diffusion_and_runtime_defaults(monkeypatch):
    from yeto import cli

    monkeypatch.setenv("YETO_NAVA_BASE_CHECKPOINT", "nava/base.safetensors")
    args = cli.parse_args([
        "--gpu", "aws:8xa100@us-east-2,runpod:8xh100@CA",
        "--model", "nava",
        "--data", "/data/train.jsonl",
    ])
    assert args.task == "diffusion"
    assert not hasattr(args, "loss_function")

    cli._normalize_launch_args(args)
    assert args.component == "nava"
    assert args.component_config == "configs/nava.yaml"
    assert args.base_checkpoint == "nava/base.safetensors"
    assert args.learner_image is None
    assert args.lora_targets == "mmdit-all-linear"
    assert cli._validate_launch_args(args) is True


def test_diffusion_backend_can_use_installed_component_runtime(monkeypatch, tmp_path):
    install_fake_sky(monkeypatch)
    args = _diffusion_args(
        tmp_path,
        component_root=None,
        base_checkpoint="nava/base.safetensors",
        learner_image="docker:nava-runtime",
    )
    spec = ClusterSpec(cloud="aws", region="us-west-2", num_nodes=1, gpus_per_node=1, gpu="L4")

    task = backends.get_backend("diffusion").build_learner_task(args, spec, 0, 1, "203.0.113.10:29400")

    assert "--component-root" not in task.run
    assert "pip install -q -e ~/yeto-component" not in task.setup
    assert "~/yeto-component" not in (task.file_mounts or {})
    assert "--base-checkpoint nava/base.safetensors" in task.run


def test_lm_help_does_not_show_diffusion_component_flags(capsys):
    from yeto import cli

    with pytest.raises(SystemExit):
        cli.main(["launch", "--help"])
    out = capsys.readouterr().out
    assert "--component-root" not in out
    assert "--base-checkpoint" not in out
    assert "--loss-function" in out


def test_diffusion_help_shows_component_flags_not_lm_loss(capsys):
    from yeto import cli

    with pytest.raises(SystemExit):
        cli.main(["launch", "--task", "diffusion", "--help"])
    out = capsys.readouterr().out
    assert "--component-root" in out
    assert "--base-checkpoint" in out
    assert "LM fine-tuning" not in out


def test_generic_diffusion_learner_has_no_nava_runtime_imports():
    source = Path("yeto/diffusion/learner.py").read_text(encoding="utf-8")
    assert "nava_src" not in source


def test_export_model_nava_help_uses_diffusion_parser(capsys):
    from yeto import export

    with pytest.raises(SystemExit):
        export.main(["--model", "nava", "--help"])
    out = capsys.readouterr().out
    assert "--component-root" in out
    assert "--base-checkpoint" in out
