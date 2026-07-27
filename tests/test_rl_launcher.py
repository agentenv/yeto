import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto import cli
from yeto.gpu_spec import parse_gpu_spec
from yeto.launcher import (
    _prepare_rl_launch_args,
    _rl_fatal_marker_reason,
    _rl_final_marker_matches,
    _stage_rl_checkpoint,
    make_miles_island_task,
    syncer_command,
)
from yeto.rl.core import canonical_state
from yeto.rl.export import specs_manifest
from yeto.rl.manifest import MILES_COMMIT, MILES_REPOSITORY, validate_manifest


def rl_args():
    layout = specs_manifest(
        canonical_state(
            0,
            {
                "base_model.model.layer.q_proj.lora_A.weight": torch.zeros(2, 4),
                "base_model.model.layer.q_proj.lora_B.weight": torch.zeros(4, 2),
            },
        ).specs
    )
    return SimpleNamespace(
        training_mode="rl",
        rl_runtime="miles",
        model="org/model",
        model_kind="causal-lm",
        model_revision="a" * 40,
        data="org/data",
        data_revision="b" * 40,
        tuning="lora",
        rl_global_rounds=2,
        rl_groups_per_island_round=3,
        rl_samples_per_group=2,
        rl_local_optimizer_steps=2,
        rl_round_timeout_s=0,
        reward_function="yeto.rl.reward:miles_reward",
        trust_remote_code=True,
        seq_len=32,
        gpu="aws:1xA100@us-east-1,aws:1xA100@us-west-2",
        external_learners=0,
        tensor_parallel=None,
        pipeline_parallel=None,
        expert_parallel=None,
        learner_image="docker:radixark/miles@sha256:" + "d" * 64,
        _explicit_launch_flags=[],
        _provenance={
            "model": {
                "source": "huggingface",
                "resolved_identifier": "org/model",
                "resolved_revision": "a" * 40,
            },
            "dataset": {
                "source": "huggingface",
                "resolved_identifier": "org/data",
                "resolved_revision": "b" * 40,
            },
        },
        rl_canonical_layout=layout,
        lora_r=2,
        lora_targets="attention",
        inner_lr=1e-5,
        seed=7,
        source_sha256="c" * 64,
        cluster_prefix="rl-launch-test",
        wan_streams=2,
        learner_cpus=None,
        learner_instance_type=None,
        spot=False,
        disk_size=128,
    )


def test_rl_launch_preparation_keeps_island_roster_and_reuses_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("yeto.runs.run_dir", lambda prefix: tmp_path / prefix)
    args = rl_args()
    _prepare_rl_launch_args(args)
    manifest_text = args.rl_manifest_json
    manifest = validate_manifest(manifest_text, args.run_manifest_sha256)
    assert manifest["workload"]["learners"] == 2
    assert manifest["generation"]["custom_generate"] is None
    assert args.quorum == 2
    assert args.fragments == args.pipeline == 1
    assert args.merge_alpha == 0.0
    assert args.wire_dtype == "f32"

    _prepare_rl_launch_args(args)
    assert args.rl_manifest_json == manifest_text
    assert (tmp_path / args.cluster_prefix / "rl-manifest.json").read_text() == manifest_text

    args.inner_lr = 2e-5
    with pytest.raises(ValueError, match="persisted manifest"):
        _prepare_rl_launch_args(args)


def test_rl_launch_attests_an_optional_complete_trajectory_generator(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("yeto.runs.run_dir", lambda prefix: tmp_path / prefix)
    args = rl_args()
    args.rl_generate_function = "yeto.rl.reward:miles_reward"

    _prepare_rl_launch_args(args)

    manifest = validate_manifest(args.rl_manifest_json, args.run_manifest_sha256)
    assert manifest["generation"]["custom_generate"] == {
        "callable": args.rl_generate_function,
        "source_sha256": args.rl_generate_sha256,
    }
    assert len(args.rl_generate_sha256) == 64


def test_rl_launch_rejects_an_explicit_conflicting_sync_flag():
    args = rl_args()
    args._explicit_launch_flags = ["outer_lr"]
    args.outer_lr = 0.7
    with pytest.raises(ValueError, match="outer-lr"):
        _prepare_rl_launch_args(args)

    args = rl_args()
    args._explicit_launch_flags = ["merge_alpha"]
    args.merge_alpha = 0.5
    with pytest.raises(ValueError, match="merge-alpha"):
        _prepare_rl_launch_args(args)


def test_rl_launch_imports_the_reward_callable_before_provisioning():
    args = rl_args()
    args.reward_function = "yeto.rl.reward:_FUNCTION_ENV"
    with pytest.raises(TypeError, match="not callable"):
        _prepare_rl_launch_args(args)


def test_cli_tracks_only_explicit_strict_flags():
    parser = cli.build_parser()
    base = ["launch", "--model", "org/model", "--data", "org/data"]
    defaults = parser.parse_args(base)
    explicit = parser.parse_args(
        base + ["--outer-lr", "1", "--merge-alpha", "0", "--wire-dtype", "f32"]
    )
    assert not hasattr(defaults, "_explicit_launch_flags")
    assert set(explicit._explicit_launch_flags) == {
        "merge_alpha",
        "outer_lr",
        "wire_dtype",
    }


def test_syncer_command_contains_the_atomic_strict_rl_configuration():
    args = rl_args()
    args.quorum = 2
    args.grace_ms = 1000
    args.grace_gamma = 0.8
    args.grace_tau = 2.0
    args.pipeline = 1
    args.sync_interval_steps = 0.0
    args.delta_correction = "none"
    args.total_steps = 2
    args.outer_lr = 1.0
    args.outer_momentum = 0.0
    args.run_manifest_sha256 = "f" * 64
    command = syncer_command(args, 2)
    assert "--rl-strict-avg" in command
    assert "--checkpoint-every 1" in command
    assert "--run-manifest-sha256 " + "f" * 64 in command


def test_miles_task_checks_out_the_exact_clean_revision(monkeypatch):
    class Task:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def set_resources(self, resources):
            self.resources = resources

    fake_sky = SimpleNamespace(
        Task=Task,
        Resources=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    args = rl_args()
    args.rl_manifest_json = "{}"
    args.run_manifest_sha256 = "f" * 64
    args.reward_sha256 = "e" * 64
    task = make_miles_island_task(
        args,
        parse_gpu_spec(args.gpu)[0],
        learner_id=0,
        num_learners=2,
        syncer_addr="syncer:29400",
    )
    assert f"git clone --no-checkout {MILES_REPOSITORY}" in task.setup
    assert f'remote.origin.url)" = "{MILES_REPOSITORY}"' in task.setup
    assert f"git -C ~/miles checkout --detach {MILES_COMMIT}" in task.setup
    assert "status --porcelain --untracked-files=all" in task.setup
    assert task.envs["YETO_RL_MANIFEST_SHA256"] == "f" * 64


def test_rl_checkpoint_staging_uses_the_authoritative_syncer_copy(tmp_path, monkeypatch):
    args = SimpleNamespace(cluster_prefix="rl-run")
    assert _stage_rl_checkpoint(args, head_mode=True, syncer_cluster=None) == Path(
        "~/yeto-state.ckpt"
    ).expanduser()

    monkeypatch.setattr("yeto.runs.run_dir", lambda prefix: tmp_path / prefix)
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    checkpoint = _stage_rl_checkpoint(
        args,
        head_mode=False,
        syncer_cluster="rl-run-syncer",
    )
    assert checkpoint == tmp_path / "rl-run/yeto-state.ckpt"
    assert [call[0][2] for call in calls] == [
        "rl-run-syncer:yeto-state.ckpt",
        "rl-run-syncer:yeto-state.ckpt.final",
    ]
    assert all(call[1]["check"] for call in calls)


def test_head_installs_only_the_dependencies_needed_by_the_rl_exporter():
    assert cli.HEAD_RL_EXPORT_PIP == "pip install -q peft accelerate safetensors"


def test_launcher_terminal_markers_are_bound_to_the_rl_manifest():
    manifest = "a" * 64
    final = (
        "YETO_RL_FINAL_V1\n"
        "global_step=2\n"
        "roster_size=2\n"
        f"run_manifest_sha256={manifest}\n"
        f"layout_fingerprint={'b' * 64}\n"
        f"policy_sha256={'c' * 64}\n"
    )
    assert _rl_final_marker_matches(final, manifest)
    assert not _rl_final_marker_matches(final, "d" * 64)
    fatal = f"YETO_RL_FATAL_V1\nrun_manifest_sha256={manifest}\nround timed out\n"
    assert _rl_fatal_marker_reason(fatal, manifest) == "round timed out"
    assert "does not match" in _rl_fatal_marker_reason(fatal, "d" * 64)
