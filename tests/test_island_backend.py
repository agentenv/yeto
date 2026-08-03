"""The --island-backend selector: the torch and megatron backends share the
DiLoCo/LoRA/data flags but differ in the intra-island parallelism, entrypoint,
and setup deps."""

import argparse

import pytest

from yeto.gpu_spec import ClusterSpec
from yeto.launcher import make_learner_task


def _args(**over):
    base = dict(
        model="gemma4",
        data="org/data",
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
        wire_dtype="q4",
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
    )
    base.update(over)
    return argparse.Namespace(**base)


_SPEC = ClusterSpec(cloud="aws", region="us-east-2", num_nodes=1, gpus_per_node=8, gpu="B200")


def test_torch_backend_uses_shard_and_torch_learner():
    task = make_learner_task(_args(island_backend="torch"), _SPEC, 0, 1, "1.2.3.4:29400")
    assert "-m yeto.learner" in task.run
    assert "--shard fsdp" in task.run
    assert "--assistant-mask-mode native" in task.run
    assert "--data-format auto" in task.run
    assert "--base-quantization none" in task.run
    assert "--seed 0" in task.run
    assert "--island-backend" not in task.run
    assert "megatron-core" not in task.setup
    assert "--attention-backend auto" in task.run
    assert "--kernel-backend native" in task.run
    assert "liger-kernel" not in task.setup
    assert "peft==0.19.1" not in task.setup
    assert "flash-attn" not in task.setup


def test_torch_backend_installs_only_explicit_kernel_dependencies():
    task = make_learner_task(
        _args(
            island_backend="torch",
            attention_backend="flash-attn-2",
            kernel_backend="liger",
            shard="ddp",
        ),
        _SPEC,
        0,
        1,
        "1.2.3.4:29400",
    )
    assert "--attention-backend flash-attn-2" in task.run
    assert "--kernel-backend liger" in task.run
    assert "liger-kernel==0.8.0" in task.setup
    assert "peft==0.19.1" in task.setup
    assert "flash-attn==2.8.3" in task.setup
    assert "--no-build-isolation" in task.setup


def test_liger_launcher_rejects_non_builtin_loss():
    with pytest.raises(ValueError, match="only the built-in cross_entropy"):
        make_learner_task(
            _args(kernel_backend="liger", loss_function="pickle:loss.pkl"),
            _SPEC,
            0,
            1,
            "1.2.3.4:29400",
        )


def test_liger_launcher_rejects_unvalidated_tuning_and_sharding():
    with pytest.raises(ValueError, match="only for --tuning lora"):
        make_learner_task(
            _args(kernel_backend="liger", tuning="full", shard="ddp"),
            _SPEC,
            0,
            1,
            "1.2.3.4:29400",
        )
    with pytest.raises(ValueError, match="only for --shard ddp"):
        make_learner_task(
            _args(kernel_backend="liger", tuning="lora", shard="fsdp"),
            _SPEC,
            0,
            1,
            "1.2.3.4:29400",
        )


def test_megatron_rejects_torch_kernel_flags():
    with pytest.raises(ValueError, match="torch causal-LM island backend"):
        make_learner_task(
            _args(island_backend="megatron", attention_backend="sdpa"),
            _SPEC,
            0,
            1,
            "1.2.3.4:29400",
        )


def test_launcher_forwards_explicit_legacy_mask_mode():
    task = make_learner_task(
        _args(assistant_mask_mode="legacy"), _SPEC, 0, 1, "1.2.3.4:29400"
    )
    assert "--assistant-mask-mode legacy" in task.run


def test_launcher_forwards_data_format():
    task = make_learner_task(
        _args(data_format="sharegpt"), _SPEC, 0, 1, "1.2.3.4:29400"
    )
    assert "--data-format sharegpt" in task.run


def test_qlora_installs_and_forwards_nf4():
    task = make_learner_task(
        _args(base_quantization="nf4", shard="ddp"),
        _SPEC,
        0,
        1,
        "1.2.3.4:29400",
    )
    assert "--base-quantization nf4" in task.run
    assert "bitsandbytes>=0.46.1" in task.setup


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tuning": "full", "shard": "ddp"}, "requires --tuning lora"),
        ({"shard": "fsdp"}, "requires --shard ddp"),
        ({"shard": "ddp", "kernel_backend": "liger"}, "requires --kernel-backend native"),
        ({"shard": "ddp", "island_backend": "megatron"}, "torch causal-LM"),
    ],
)
def test_qlora_rejects_unsupported_launch_profiles(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_learner_task(
            _args(base_quantization="nf4", **overrides),
            _SPEC,
            0,
            1,
            "1.2.3.4:29400",
        )


def test_megatron_backend_swaps_entrypoint_and_runs_in_the_ngc_container():
    from yeto.launcher import MEGATRON_IMAGE, learner_image_for

    args = _args(island_backend="megatron")
    task = make_learner_task(args, _SPEC, 0, 1, "1.2.3.4:29400")
    assert "-m yeto.megatron.learner" in task.run
    assert "--island-backend megatron" in task.run
    assert "--assistant-mask-mode native" in task.run
    assert "--seed 0" in task.run
    assert "--shard" not in task.run  # megatron has its own parallelism
    # The stack lives in the NGC container, so setup does NOT install torch or
    # the megatron stack, nor RAID the NVMe (a host op).
    assert "megatron" not in task.setup
    assert "TORCH_SETUP" not in task.setup and "torch==" not in task.setup
    assert "mdadm" not in task.setup
    # The island runs inside the container image.
    assert learner_image_for(args, _SPEC) == MEGATRON_IMAGE
    assert MEGATRON_IMAGE.startswith("docker:")


def test_torch_backend_does_not_use_the_megatron_container():
    from yeto.launcher import MEGATRON_IMAGE, learner_image_for

    # A torch B200 island keeps its DLAMI pin, never the megatron container.
    img = learner_image_for(_args(island_backend="torch"), _SPEC)
    assert img != MEGATRON_IMAGE


def test_expert_parallel_defaults_to_filling_the_island():
    # 1 node x 8 GPUs, tp=1, pp=1 -> ep should fill the island (8).
    task = make_learner_task(_args(island_backend="megatron"), _SPEC, 0, 1, "a:1")
    assert "--expert-parallel 8" in task.run


def test_expert_parallel_respects_tp_pp_and_override():
    # tp=2, pp=2 over 8 GPUs -> auto ep = 8 // 4 = 2.
    task = make_learner_task(
        _args(island_backend="megatron", tensor_parallel=2, pipeline_parallel=2), _SPEC, 0, 1, "a:1"
    )
    assert "--expert-parallel 2" in task.run
    assert "--tensor-parallel 2" in task.run
    assert "--pipeline-parallel 2" in task.run
    # explicit override wins
    task2 = make_learner_task(
        _args(island_backend="megatron", expert_parallel=4), _SPEC, 0, 1, "a:1"
    )
    assert "--expert-parallel 4" in task2.run
