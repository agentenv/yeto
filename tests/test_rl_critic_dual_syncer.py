"""rl-algo-critic-family 4.2/4.3 (design D4 plan a): dual syncer, critic tensor
plugin, cross-channel atomic commit, round-cut critic checkpoint (CPU only)."""

from __future__ import annotations

import json

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.engine.algorithm import AlgorithmSpec

PPO = AlgorithmSpec(advantage_estimator="ppo")


# -- 4.2.1 launcher: second syncer, own port / checkpoint / tape --------------------


def _two_island_args(spec=None):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", gpu="aws:1xa100@us-east-1,aws:1xa100@us-west-2")
    args.controller = "local"
    _prepare_rl_args(args)
    if spec is not None:
        # prepare refuses an unverified critic on two islands before G3 (check_unverified_allowance);
        # the launcher plumbing is exercised on the prepared args directly.
        args.rl_algorithm_spec_json = spec.canonical_json()
        args.rl_expected_algorithm_sha256 = spec.sha256()
    return args


def test_dry_run_ppo_has_two_syncers():
    from yeto import launcher

    args = _two_island_args(PPO)
    plan = launcher.dry_run_plan(args)
    critic = plan["critic_syncer"]
    assert critic["port"] == launcher.CRITIC_SYNCER_PORT != launcher.SYNCER_PORT
    assert critic["layout"] == "critic_layout_hash"
    command = critic["command"]
    assert f"--port {launcher.CRITIC_SYNCER_PORT}" in command
    assert f"--port {launcher.SYNCER_PORT}" not in command
    assert "yeto-critic-state.ckpt" in command and "yeto-critic-tape.jsonl" in command
    assert " --learners 2 " in command
    for island in plan["island_requests"]:
        assert "--syncer $SYNCER_ADDR --critic-syncer $CRITIC_SYNCER_ADDR" in island["learner_command"]
    full = launcher.syncer_command(args, 2)
    # critic syncer backgrounded first, actor syncer stays the foreground process
    assert full.index(f"--port {launcher.CRITIC_SYNCER_PORT}") < full.index(f"--port {launcher.SYNCER_PORT}")
    assert full.count("~/yeto-output/yeto-state.ckpt") == 2  # actor: path + resume test
    assert launcher.syncer_ports(args) == [launcher.SYNCER_PORT, launcher.CRITIC_SYNCER_PORT]
    snapshot = {k: plan[k] for k in ("islands", "syncer", "outer_sync", "critic_syncer")}
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_critic_free_plan_and_syncer_unchanged():
    from yeto import launcher

    args = _two_island_args()
    plan = launcher.dry_run_plan(args)
    assert "critic_syncer" not in plan
    assert all("--critic-syncer" not in i["learner_command"] for i in plan["island_requests"])
    command = launcher.syncer_command(args, 2)
    assert str(launcher.CRITIC_SYNCER_PORT) not in command and "critic" not in command
    assert launcher.syncer_ports(args) == [launcher.SYNCER_PORT]


def test_island_env_carries_the_critic_syncer_address():
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec

    args = _two_island_args(PPO)
    task = launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 2, "10.0.0.5:29400")
    assert task.envs["CRITIC_SYNCER_ADDR"] == f"10.0.0.5:{launcher.CRITIC_SYNCER_PORT}"
    assert launcher.critic_syncer_address("$SYNCER_ADDR") == "$CRITIC_SYNCER_ADDR"

