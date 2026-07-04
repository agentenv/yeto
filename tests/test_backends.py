"""The task-backend / island-engine registry: the plug-in seam that lets a new
training task or trainer register without editing the CLI, launcher, or export
dispatch. These tests avoid torch/sky (sky is stubbed where a task is built)."""

import argparse
import sys
import types

import pytest

from yeto import backends
from yeto.backends import base


# ---------------------------------------------------------------------------
# registry basics


def test_builtin_backends_and_engines_registered():
    assert backends.backend_names() == ["lm", "nava"]
    assert backends.engine_names() == ["torch", "megatron"]
    assert backends.default_task() == "lm"


def test_get_backend_defaults_to_lm_and_rejects_unknown():
    assert backends.get_backend(None).name == "lm"
    assert backends.get_backend("nava").name == "nava"
    with pytest.raises(ValueError):
        backends.get_backend("does-not-exist")
    with pytest.raises(ValueError):
        backends.get_engine("does-not-exist")


def test_auto_fleet_and_output_dir_are_backend_properties():
    lm = backends.get_backend("lm")
    nava = backends.get_backend("nava")
    assert lm.supports_auto_fleet is True and lm.output_dir == "yeto-output"
    assert nava.supports_auto_fleet is False and nava.output_dir == "yeto-nava-output"


# ---------------------------------------------------------------------------
# validation / warnings live on the backends


def _lm_args(**over):
    base_args = dict(model="gemma4", data="org/d")
    base_args.update(over)
    return argparse.Namespace(**base_args)


def test_lm_validate_requires_model_and_data():
    assert backends.get_backend("lm").validate(_lm_args()) == []
    errs = backends.get_backend("lm").validate(_lm_args(model=None, data=None))
    assert errs and "--model" in errs[0] and "--data" in errs[0]


def _nava_args(**over):
    base_args = dict(
        nava_root="/x", nava_ckpt="s3://b/c", nava_data="/d.jsonl",
        shard="ddp", nava_tuning="lora", nava_full_sync="unsupported",
        syncer_memory=32, learner_image=None,
    )
    base_args.update(over)
    return argparse.Namespace(**base_args)


def test_nava_validate_requires_inputs_and_guards_fsdp_full():
    nava = backends.get_backend("nava")
    assert nava.validate(_nava_args()) == []
    missing = nava.validate(_nava_args(nava_root=None, nava_ckpt=None))
    assert missing and "--nava-root" in missing[0]
    # FSDP with non-LoRA trainables needs the gather sync mode.
    bad = nava.validate(_nava_args(shard="fsdp", nava_tuning="full", nava_full_sync="unsupported"))
    assert bad and "gather" in bad[0]
    ok = nava.validate(_nava_args(shard="fsdp", nava_tuning="full", nava_full_sync="gather"))
    assert ok == []


def test_nava_warnings_flag_full_sync_memory_and_missing_image():
    nava = backends.get_backend("nava")
    warns = nava.warnings(_nava_args(nava_tuning="full", syncer_memory=32, learner_image=None))
    text = " ".join(warns)
    assert "syncer memory" in text and "runtime setup is heavy" in text
    # A LoRA run with an explicit image is clean.
    assert nava.warnings(_nava_args(nava_tuning="lora", learner_image="docker:x")) == []


def test_nava_head_mount_and_rewrite():
    nava = backends.get_backend("nava")
    mounts = nava.head_file_mounts(_nava_args(nava_root="/tmp/NAVA"))
    assert list(mounts) == ["~/NAVA"]
    args = _nava_args(nava_root="/tmp/NAVA")
    nava.rewrite_for_head(args)
    assert args.nava_root.endswith("/NAVA") and args.nava_root != "/tmp/NAVA"


# ---------------------------------------------------------------------------
# island-engine dispatch through the LM backend (sky stubbed)


class _FakeTask:
    def __init__(self, **k):
        self.__dict__.update(k)
        self.resources = None

    def set_resources(self, r):
        self.resources = r


def _install_fake_sky(monkeypatch):
    sky = types.ModuleType("sky")
    sky.Task = _FakeTask
    sky.Resources = lambda **k: types.SimpleNamespace(kwargs=k)
    monkeypatch.setitem(sys.modules, "sky", sky)


def _full_lm_args(**over):
    base_args = dict(
        model="gemma4", data="org/d", loss_function="cross_entropy", train_on="assistant",
        shard="fsdp", tuning="lora", lora_r=16, lora_targets="auto", seq_len=2048,
        micro_batch_size="auto", grad_accum=4, inner_lr=3e-4, fragments=8,
        fragment_pattern="binpack", merge_alpha=0.5, tokenize="stream", stream_workers=2,
        wire_dtype="q4", wan_streams=4, max_rows=None, island_backend="torch",
        expert_parallel=None, tensor_parallel=1, pipeline_parallel=1, learner_image=None,
        learner_cpus=None, learner_instance_type=None, spot=True, disk_size=512,
        retry_until_up=True,
    )
    base_args.update(over)
    return argparse.Namespace(**base_args)


def test_lm_backend_dispatches_to_the_selected_engine(monkeypatch):
    _install_fake_sky(monkeypatch)
    from yeto.gpu_spec import ClusterSpec

    spec = ClusterSpec(cloud="aws", region="us-east-2", num_nodes=1, gpus_per_node=8, gpu="B200")
    lm = backends.get_backend("lm")

    torch_task = lm.build_learner_task(_full_lm_args(island_backend="torch"), spec, 0, 1, "h:1")
    assert "-m yeto.learner" in torch_task.run and "--shard fsdp" in torch_task.run
    assert "--island-backend" not in torch_task.run

    mega_task = lm.build_learner_task(_full_lm_args(island_backend="megatron"), spec, 0, 1, "h:1")
    assert "-m yeto.megatron.learner" in mega_task.run
    assert "--expert-parallel 8" in mega_task.run and "--shard" not in mega_task.run


# ---------------------------------------------------------------------------
# extensibility: a third task/engine drops in with no core edits


def test_a_new_backend_and_engine_register_via_the_public_api():
    saved_backends = dict(base._BACKENDS)
    saved_engines = dict(base._ENGINES)
    try:
        class MyEngine(backends.IslandEngine):
            name = "myeng"

            def entrypoint(self):
                return "pkg.learner"

            def setup_steps(self, args):
                return ["echo hi"]

        class MyBackend(backends.TaskBackend):
            name = "rlhf"
            output_dir = "yeto-rlhf-output"

        backends.register_engine(MyEngine())
        backends.register_backend(MyBackend())

        assert "rlhf" in backends.backend_names()
        assert "myeng" in backends.engine_names()
        assert backends.get_backend("rlhf").output_dir == "yeto-rlhf-output"
        # The default task is unchanged by a later registration.
        assert backends.default_task() == "lm"
    finally:
        base._BACKENDS.clear()
        base._BACKENDS.update(saved_backends)
        base._ENGINES.clear()
        base._ENGINES.update(saved_engines)
