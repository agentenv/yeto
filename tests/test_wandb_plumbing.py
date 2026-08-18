"""--wandb plumbing: launch flag -> learner command, env, and setup.

Telemetry has to survive the trip from the submitting machine to a spot
learner in another cloud, and it must be invisible when it is off.
"""

import argparse
import json

import pytest

from yeto.gpu_spec import ClusterSpec
from yeto.launcher import SYNCER_EVENT_TAPE, LocalSyncer, make_learner_task, syncer_command

_SPEC = ClusterSpec(cloud="aws", region="us-east-2", num_nodes=1, gpus_per_node=8, gpu="B200")


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
        cluster_prefix="my-fleet",
        wandb=False,
        wandb_project="yeto",
        wandb_entity=None,
        wandb_mode="online",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _task(**over):
    return make_learner_task(_args(**over), _SPEC, 0, 1, "1.2.3.4:29400")


def test_telemetry_is_absent_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    task = _task(wandb=False)
    assert "--wandb" not in task.run
    assert "wandb" not in task.setup
    # A key in the submitter's environment must not follow the fleet around
    # unless telemetry was actually asked for.
    assert "WANDB_API_KEY" not in task.envs
    assert "YETO_RUN_GROUP" not in task.envs


def test_flags_reach_the_learner_command(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    task = _task(wandb=True, wandb_project="fleet-lab", wandb_entity="acme", wandb_mode="offline")
    assert "--wandb " in task.run
    assert "--wandb-project fleet-lab" in task.run
    assert "--wandb-entity acme" in task.run
    assert "--wandb-mode offline" in task.run
    assert "pip install -q wandb" in task.setup


def test_islands_are_told_which_fleet_they_belong_to(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    task = _task(wandb=True)
    # The group is the run's name, so every island and the syncer's tape run
    # land on one comparison view.
    assert task.envs["YETO_RUN_GROUP"] == "my-fleet"
    assert task.envs["WANDB_API_KEY"] == "secret"


def test_a_missing_key_still_launches(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    task = _task(wandb=True)
    assert "WANDB_API_KEY" not in task.envs
    assert task.envs["YETO_RUN_GROUP"] == "my-fleet"


def test_an_omitted_entity_is_not_forwarded_as_none(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    task = _task(wandb=True, wandb_entity=None)
    assert "--wandb-entity" not in task.run


def test_a_failed_install_does_not_fail_the_island(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    setup = _task(wandb=True).setup
    install = [line for line in setup.splitlines() if "pip install -q wandb" in line]
    assert install and "||" in install[0]


@pytest.mark.parametrize(
    "backend,entrypoint",
    [("torch", "yeto.learner"), ("megatron", "yeto.megatron.learner")],
)
def test_every_backend_forwards_the_flags(monkeypatch, backend, entrypoint):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    task = _task(wandb=True, island_backend=backend)
    assert f"-m {entrypoint}" in task.run
    assert "--wandb " in task.run
    assert "pip install -q wandb" in task.setup


@pytest.mark.parametrize(
    "module",
    [
        "yeto.learner",
        "yeto.diffusion.learner",
        "yeto.megatron.learner",
        "yeto.mlx.learner",
    ],
)
def test_every_learner_entrypoint_parses_the_flags(module):
    """The launcher appends the flags to the shared common_flags block, so a
    backend that forgot yeto.wandb_logger.add_arguments dies on startup."""
    import importlib

    parser = importlib.import_module(module).parse_args
    base = [
        "--model", "qwen35-9b",
        "--data", "org/chat",
        "--syncer", "1.2.3.4:29400",
        "--learner-id", "0",
        "--num-learners", "2",
    ]
    args = parser(base + ["--wandb", "--wandb-project", "p", "--wandb-mode", "offline"])
    assert (args.wandb, args.wandb_project, args.wandb_mode) == (True, "p", "offline")
    assert parser(base).wandb is False


# --------------------------------------------------------------------------
# syncer side


def _syncer_args(**over):
    base = dict(
        quorum=2,
        grace_ms=500,
        grace_gamma=1.5,
        grace_tau=0.5,
        pipeline=2,
        sync_interval_steps=24.0,
        delta_correction="none",
        total_steps=1000,
        outer_lr=0.7,
        outer_momentum=0.9,
        cluster_prefix="my-fleet",
        wandb=False,
        wandb_project="yeto",
        wandb_entity=None,
        wandb_mode="online",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_the_tape_path_is_shared_between_writer_and_reader():
    # The Rust syncer writes it and the head's forwarder tails it; a drift
    # between the two would be a silently empty syncer run.
    assert f"--event-tape {SYNCER_EVENT_TAPE}" in syncer_command(_syncer_args(), 2)
    import os

    syncer = LocalSyncer(_syncer_args(), 2)
    assert syncer.tape_file == os.path.expanduser(SYNCER_EVENT_TAPE)


def test_the_forwarder_is_not_started_without_the_flag():
    syncer = LocalSyncer(_syncer_args(wandb=False), 2)
    syncer.start_tape_forwarder(_syncer_args(wandb=False))
    assert syncer.tape_forwarder is None


def test_the_forwarder_is_skipped_when_wandb_is_unavailable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "wandb", None)  # import raises ImportError
    syncer = LocalSyncer(_syncer_args(wandb=True), 2)
    syncer.start_tape_forwarder(_syncer_args(wandb=True))
    assert syncer.tape_forwarder is None


# --------------------------------------------------------------------------
# CLI


def test_launch_parser_accepts_and_round_trips_the_flags():
    from yeto.cli import _serializable_args, parse_args

    args = parse_args(
        [
            "--model", "qwen35-9b",
            "--data", "org/chat",
            "--gpu", "aws:8xa100@us-east-2",
            "--wandb",
            "--wandb-project", "fleet-lab",
            "--wandb-mode", "offline",
        ]
    )
    assert args.wandb is True
    assert args.wandb_project == "fleet-lab"
    assert args.wandb_mode == "offline"
    # Head-controller mode replays the launch through JSON on the head VM.
    replayed = json.loads(json.dumps(_serializable_args(args)))
    assert replayed["wandb"] is True
    assert replayed["wandb_project"] == "fleet-lab"
    assert replayed["wandb_mode"] == "offline"


def test_launch_defaults_leave_telemetry_off():
    from yeto.cli import parse_args

    args = parse_args(["--model", "qwen35-9b", "--data", "org/chat"])
    assert args.wandb is False
