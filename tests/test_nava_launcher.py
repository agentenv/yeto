import sys
import types

import yeto.cli as cli
import yeto.launcher as launcher
from yeto.gpu_spec import parse_gpu_spec


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


def test_nava_learner_task_uses_mounted_config_and_runtime_image(monkeypatch, tmp_path):
    install_fake_sky(monkeypatch)
    nava = tmp_path / "NAVA"
    nava.mkdir()
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    args = cli.parse_args(
        [
            "--gpu", "aws:8xh100@us-west-2",
            "--task", "nava",
            "--nava-root", str(nava),
            "--nava-config", "configs/nava.yaml",
            "--nava-ckpt", "s3://bucket/NAVA.safetensors",
            "--nava-data", str(data),
            "--runtime-image", "docker:yeto-nava:cu128",
            "--nava-install-flash-attn",
        ]
    )
    spec = parse_gpu_spec(args.gpu)[0]

    task = launcher.make_nava_learner_task(args, spec, 0, 1, "203.0.113.1:29400")

    assert task.file_mounts["~/NAVA"] == str(nava)
    assert task.file_mounts["~/nava_inputs/train.jsonl"] == str(data)
    assert "--nava-config" in task.run and "~/NAVA/configs/nava.yaml" in task.run
    assert "--nava-data" in task.run and "~/nava_inputs/train.jsonl" in task.run
    assert "--nava-merge-avg-regex" in task.run
    assert "flash-attn --no-build-isolation" in task.setup
    assert task.resources.kwargs["image_id"] == "docker:yeto-nava:cu128"
