"""Launcher end-to-end wiring of --rl-sync-preset full-parameter (P0-1.7b.1-Q6/Q7). No Ray.

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
DENSE = ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
         "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "4")


def _dense_plan(extra=(), preset="full-parameter"):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    flags = tuple(preset if value == "full-parameter" else value for value in DENSE)
    args = _launcher_args("ports", flags + tuple(extra), gpu=TWO_ISLANDS)
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
        assert _flag(command, "--sync-preset") == "full-parameter"
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
        (("--rl-sync-island-lease-s", "30"), "membership options need --rl-sync-full-parameter-rounds async"),
    ],
)
def test_dense_full_refusals(extra, message, monkeypatch):
    from yeto.launcher import _prepare_rl_args

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", DENSE + extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


def test_dense_full_is_an_alias_of_the_full_parameter_preset():
    # P0-1.7b.2-Q2: the old name still parses, on both layers, to the new one.
    args, plan = _dense_plan(preset="dense-full")
    assert args.rl_sync_preset == "full-parameter"
    command = plan["island_requests"][0]["learner_command"]
    assert _flag(command, "--sync-preset") == "full-parameter"
    from yeto.rl.engine.selection import canonical_sync_preset

    assert canonical_sync_preset("dense-full") == "full-parameter"


def test_layout_hash_only_with_dense_full():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-parameter-layout-sha256", LAYOUT))
    with pytest.raises(ValueError, match="only applies to --rl-sync-preset full-parameter"):
        _prepare_rl_args(args)


def test_island_entry_accepts_the_dense_island_command():
    from yeto.rl.adapters.miles import island_entry

    _, plan = _dense_plan()
    command = plan["island_requests"][0]["learner_command"]
    tokens = shlex.split(command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1])
    tokens = [{"$SYNCER_ADDR": "127.0.0.1:29400", "$LEARNER_ID": "1"}.get(t, t) for t in tokens]
    parsed = island_entry.parse_args(tokens)
    assert (parsed.parameter_mode, parsed.sync_preset) == ("full", "full-parameter")
    assert (parsed.num_learners, parsed.learner_id) == (2, 1)
    island_entry._check_dense_full_sweep(parsed, yeto_policy_sync=True)


@pytest.mark.parametrize(
    ("reward_function", "kind"),
    (
        ("yeto.rl.harness.codex.reward:reward_func", "secrlenv"),
        ("yeto.rl.gsm8k_reward:score", "reward-function"),
        ("yeto.rl.math_reward:reward_func", "reward-function"),
    ),
)
def test_dense_island_evidence_kind_follows_its_reward_function(reward_function, kind):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.trajectory_evidence import EVIDENCE_KINDS

    chosen = island_entry._trajectory_evidence_kind(
        SimpleNamespace(reward_function=reward_function)
    )
    assert chosen == kind
    assert chosen in EVIDENCE_KINDS


EXPERT_FULL = (
    "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
    "--rl-rollout-gpus", "8", "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "4",
    "--rl-model-recipe", "deepseek-v4-flash", "--lora-targets", "attention",
    "--expert-full-count", "4", "--expert-full-lr", "1e-06",
    "--expert-selection-sha256", "1" * 64, "--expert-selection-contract-sha256", "2" * 64,
    "--tensor-parallel", "8", "--expert-parallel", "8", "--rollout-num-gpus-per-engine", "8",
    "--sglang-tp-size", "8", "--sglang-ep-size", "8", "--sglang-attention-backend", "dsv4",
    "--sglang-page-size", "256", "--no-sglang-deterministic-inference",
)


def test_deepseek_v4_expert_full_island_is_the_lora_hybrid_on_the_dense_sweep(monkeypatch):
    # P0 1.7b.2: attention LoRA + clone experts ride the same strict sweep as dense.
    from yeto import launcher
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", EXPERT_FULL, gpu="modal:16xh200,modal:16xh200")
    args.controller = "local"
    _prepare_rl_args(args)
    plan = dry_run_plan(args)
    profile_hash = launcher.rl_syncer_profile(args, 2).sha256
    for island in plan["island_requests"]:
        command = island["learner_command"]
        assert _flag(command, "--sync-preset") == "full-parameter"
        assert "--parameter-mode" not in command
        assert _flag(command, "--lora-targets") == "attention"
        assert _flag(command, "--expert-full-count") == "4"
        assert _flag(command, "--syncer-profile-sha256") == profile_hash
    [actor] = plan["role_syncers"]
    assert _flag(actor["command"], "--policy-sweep-fragments") == "4"


    from yeto.gpu_spec import parse_gpu_spec

    task = launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 2, "127.0.0.1:1")
    assert task.envs["YETO_DSV4_EXPERT_FULL_PORTS"] == "1"
    assert "YETO_DSV4_EXPERT_FULL" not in task.envs  # the legacy runtime stays off

    tokens = shlex.split(plan["island_requests"][0]["learner_command"].split(
        "-m yeto.rl.adapters.miles.island_entry", 1)[1])
    tokens = [{"$SYNCER_ADDR": "127.0.0.1:29400", "$LEARNER_ID": "1"}.get(t, t) for t in tokens]
    from yeto.rl.adapters.miles import island_entry

    parsed = island_entry.parse_args(tokens)
    assert (parsed.parameter_mode, parsed.sync_preset, parsed.expert_full_count) == ("lora", "full-parameter", 4)
    island_entry._check_dense_full_sweep(parsed, yeto_policy_sync=True)


def test_ports_entry_takes_the_full_parameter_identity_from_run_miles(monkeypatch):
    """P0-1.11 GPU run fpprobe-20261014a: the full-parameter island has no LoRA
    rank, so ``_run_ports`` must forward the identity run_miles bound instead
    of re-deriving a LoRA hash (it raised TypeError on lora_r=None)."""
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import entry, island_entry

    _, plan = _dense_plan()
    command = plan["island_requests"][0]["learner_command"]
    tokens = shlex.split(command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1])
    tokens = [{"$SYNCER_ADDR": "127.0.0.1:29400", "$LEARNER_ID": "0"}.get(t, t) for t in tokens]
    args = island_entry.parse_args(tokens)
    assert args.lora_r is None
    args.event_tape = "/tmp/tape.jsonl"
    miles_args = SimpleNamespace(
        yeto_rl_parameter_mode="full",
        yeto_rl_lora_config_hash="c" * 64,
        yeto_rl_layout_hash="d" * 64,
    )
    seen = {}
    monkeypatch.setattr(island_entry, "apply_ports_infra_switches", lambda *a: None)
    monkeypatch.setattr(island_entry, "_run_test_debug_rollout_replay", lambda *a: False)
    monkeypatch.setattr(entry, "run_ports_island", lambda *a, **kw: seen.update(kw))
    island_entry._run_ports(args, miles_args, "launch", "algorithm", specs=(),
                            canonical_targets=(), yeto_policy_sync=True)
    assert (seen["lora_config_hash"], seen["layout_hash"]) == ("c" * 64, "d" * 64)
    assert seen["yeto_policy_sync"] is True
    assert miles_args.yeto_rl_learner_id == 0
