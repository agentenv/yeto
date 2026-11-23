"""The verl backend runs the sync presets its island has a session for, and refuses the rest.

Before ``run_config_rules.check_sync_preset``, ``yeto launch --rl-backend verl
--rl-sync-preset decoupled`` started a decoupled syncer while every verl island dropped
``--sync-preset`` (``parse_known_args``) and ran strict-avg. Decoupled LoRA now runs on
the synchronous colocated FSDP2 island (``decoupled_session``); other shapes and unknown
presets are refused by the launcher and by the island. CPU only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl import run_config_rules as rules
from yeto.rl.engine import backends

GPU = "modal:1xh200,modal:1xh200"
ISLAND = ("--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0",
          "--reward-function", "yeto.rl.gsm8k_reward:score", "--global-rounds", "2", "--groups-per-round", "1",
          "--samples-per-group", "2", "--rollout-max-response-len", "8", "--event-tape", "t",
          "--inner-lr", "1", "--seq-len", "64", "--seed", "1", "--syncer", "127.0.0.1:1")


def _prepared(*extra):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-backend", "verl", "--reward-function", "gsm8k_reward:score", *extra),
                          gpu=GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


DECOUPLED = ("--rl-sync-preset", "decoupled", "--fragments", "2", "--pipeline", "2",
             "--local-rl-rounds-per-sync", "2")


def test_the_launcher_accepts_decoupled_lora_on_verl():
    args = _prepared(*DECOUPLED)
    assert args.rl_sync_preset == "decoupled" and args.rl_total_fragment_steps == args.total_steps * 2


@pytest.mark.parametrize("extra, match", [
    (("--rl-max-policy-age", "1", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1"),
     "--rl-max-policy-age 0|--rl-placement colocated"),
    (("--tensor-parallel", "2"), "TP=PP=EP=CP=1|Megatron"),
])
def test_the_launcher_refuses_decoupled_shapes_the_island_does_not_cover(extra, match):
    with pytest.raises(ValueError, match=match):
        _prepared(*DECOUPLED, *extra)


def test_the_launcher_still_accepts_strict_avg_on_verl():
    assert _prepared().rl_sync_preset == "strict-avg"


def test_miles_has_no_such_rule():
    # The hook is the backend's own: a Miles launch is unchanged
    assert getattr(backends.module("run_config_rules", "miles"), "check_sync_preset", None) is None


@pytest.mark.parametrize("preset, ok", [
    ("strict-avg", True), ("full-parameter", True), ("dense-full", True), (None, True),
    ("decoupled", True), ("sao-streaming-full", False),
])
def test_the_rule_names_what_the_island_runs(preset, ok):
    args = SimpleNamespace(rl_sync_preset=preset, rl_single_island_no_sync=False, tuning="lora")
    if ok:
        rules.check_sync_preset(args)
    else:
        with pytest.raises(ValueError, match="verl"):
            rules.check_sync_preset(args)
    # a no-sync island has no outer sync to pick
    rules.check_sync_preset(SimpleNamespace(rl_sync_preset=preset, rl_single_island_no_sync=True))


def test_the_island_refuses_decoupled_full_parameters_before_any_work():
    with pytest.raises(SystemExit, match="needs --parameter-mode lora"):
        island_entry.main([*ISLAND, "--sync-preset", "decoupled", "--parameter-mode", "full"])


def test_the_island_refuses_an_unknown_preset_and_a_mismatched_mode():
    with pytest.raises(SystemExit, match="not run by the verl island"):
        island_entry.main([*ISLAND, "--sync-preset", "sao-streaming-full"])
    with pytest.raises(SystemExit, match="needs --parameter-mode full"):
        island_entry.main([*ISLAND, "--sync-preset", "full-parameter"])


@pytest.mark.parametrize("preset, mode", [
    (None, "lora"), ("strict-avg", "lora"), ("full-parameter", "full"), ("dense-full", "full"),
    ("decoupled", "lora"),
])
def test_the_island_accepts_what_it_runs(preset, mode):
    assert rules.island_sync_preset_problem(preset, parameter_mode=mode) is None


def test_the_island_parses_the_preset_instead_of_dropping_it():
    args, ignored = island_entry.parse_args([*ISLAND, "--sync-preset", "strict-avg"])
    assert args.sync_preset == "strict-avg" and "--sync-preset" not in ignored
