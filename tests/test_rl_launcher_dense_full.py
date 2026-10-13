"""Launcher end-to-end wiring of --rl-sync-preset dense-full (P0-1.7b.1-Q6/Q7). No Ray.

The dense strict sweep is launched like any ports island on a fixed
partition; the syncer adds the policy-sweep profile and the launcher sends
every island the semantic profile hash that syncer serves.
"""

from __future__ import annotations

import shlex

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401

LAYOUT = "e" * 64
TWO_ISLANDS = "modal:2xh100,modal:2xh100"
DENSE = ("--rl-sync-preset", "dense-full", "--rl-placement", "fixed-partition",
         "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "4")


def _dense_plan(extra=()):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", DENSE + tuple(extra), gpu=TWO_ISLANDS)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _flag(command: str, flag: str) -> str:
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1]


def test_dense_syncer_command_is_the_strict_policy_sweep_profile():
    from yeto import launcher

    args, plan = _dense_plan()
    [actor] = plan["role_syncers"]
    command = actor["command"]
    assert _flag(command, "--policy-sweep-fragments") == "4"
    assert "--require-profile-binding" in command
    expected = {"--learners": "2", "--quorum": "2", "--grace-ms": "0", "--pipeline": "1",
                "--sync-interval-steps": "0.0", "--delta-correction": "none",
                "--total-steps": "12", "--outer-lr": "1.0", "--outer-momentum": "0.0",
                "--learner-weight": "equal", "--max-base-lag": "0", "--checkpoint-every": "1"}
    assert {flag: _flag(command, flag) for flag in expected} == expected
    # The profile the islands bind mirrors exactly this command.
    profile = launcher.rl_syncer_profile(args, 2)
    assert (profile.learners, profile.quorum, profile.total_steps) == (2, 2, 12)
    assert profile.policy_sweep_fragments == 4 and profile.require_profile_binding


def test_dense_island_command_carries_layout_roster_and_profile_hash():
    from yeto import launcher

    args, plan = _dense_plan()
    profile_hash = launcher.rl_syncer_profile(args, 2).sha256
    for island in plan["island_requests"]:
        command = island["learner_command"]
        assert _flag(command, "--sync-preset") == "dense-full"
        assert _flag(command, "--parameter-mode") == "full"
        assert _flag(command, "--parameter-layout-sha256") == LAYOUT
        assert _flag(command, "--num-learners") == "2"
        assert _flag(command, "--syncer-profile-sha256") == profile_hash
        assert _flag(command, "--fragments") == "4"
        assert _flag(command, "--total-fragment-steps") == "12"
        assert _flag(command, "--rl-placement") == "fixed-partition"
        assert "--lora-r" not in command and "--lora-targets" not in command


def test_lora_island_command_is_unchanged():
    from yeto.launcher import _prepare_rl_args, make_miles_island_task
    from yeto.gpu_spec import parse_gpu_spec

    args = _launcher_args("ports")
    _prepare_rl_args(args)
    run = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:1").run
    assert f"--lora-r {args.lora_r} --lora-targets {args.lora_targets}" in run
    assert "--syncer-profile-sha256" not in run and "--parameter-mode" not in run


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--rl-placement", "colocated"), "fixed-partition"),
        (("--rl-parameter-layout-sha256", "nope"), "rl-parameter-layout-sha256"),
        (("--local-rl-rounds-per-sync", "2"), "H=1"),
        (("--rl-island-scheduling", "elastic"), "elastic"),
    ],
)
def test_dense_full_refusals(extra, message, monkeypatch):
    from yeto.launcher import _prepare_rl_args

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", DENSE + extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


def test_layout_hash_only_with_dense_full():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-parameter-layout-sha256", LAYOUT))
    with pytest.raises(ValueError, match="only applies to --rl-sync-preset dense-full"):
        _prepare_rl_args(args)


def test_island_entry_accepts_the_dense_island_command():
    from yeto.rl.adapters.miles import island_entry

    _, plan = _dense_plan()
    command = plan["island_requests"][0]["learner_command"]
    tokens = shlex.split(command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1])
    tokens = [{"$SYNCER_ADDR": "127.0.0.1:29400", "$LEARNER_ID": "1"}.get(t, t) for t in tokens]
    parsed = island_entry.parse_args(tokens)
    assert (parsed.parameter_mode, parsed.sync_preset) == ("full", "dense-full")
    assert (parsed.num_learners, parsed.learner_id) == (2, 1)
    island_entry._check_dense_full_sweep(parsed, rl_engine="ports", yeto_policy_sync=True)
