"""S27 G-EF (CPU, no Ray, no GPU): full-parameter trainer and engines colocated on one island.

A full-parameter island without outer sync (``--rl-single-island-no-sync``) may
place its engines on the trainer GPUs (``--rl-placement colocated``, no
``--rl-rollout-gpus``). The Miles adapter then runs Miles' own ``--colocate`` with
``--offload-train``; with the Harbor agent and ``--rl-max-policy-age N`` it derives
Miles' suspension between turns. A synced island, a multi-node island and an
uneven engine split stay refused; the fixed-partition layout is unchanged.
"""

from __future__ import annotations

import shlex

import pytest

from test_agentic_stage3_island_argv import _island_argv, _provider, _value
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles import config as miles_config
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import run_config
from yeto.rl.engine.run_config import full_parameter_layout_problem


def _layout(**overrides):
    kwargs = dict(data_parallel=1, actor_num_nodes=1, rollout_gpus=None, rollout_gpus_per_engine=2,
                  island_gpus=8, colocated=True, single_island_no_sync=True)
    kwargs.update(overrides)
    return full_parameter_layout_problem(**kwargs)


def test_colocated_full_parameters_need_one_island_without_sync():
    assert _layout() is None
    assert _layout(rollout_gpus=0) is None
    assert "--rl-single-island-no-sync" in _layout(single_island_no_sync=False)
    # the old default call (no colocated request) keeps its refusal text
    assert "dedicated, evenly partitioned" in full_parameter_layout_problem(
        data_parallel=1, actor_num_nodes=1, rollout_gpus=None, rollout_gpus_per_engine=1)


def test_colocated_full_parameters_refuse_multi_node_and_uneven_engines():
    assert "one node" in _layout(actor_num_nodes=2)
    assert "one node" in _layout(island_gpus_per_node=8)
    assert "must divide" in _layout(rollout_gpus_per_engine=3)
    assert "must divide" in _layout(rollout_gpus_per_engine=0)


def test_dedicated_engines_keep_the_fixed_partition_rule():
    # a colocated request with dedicated engine GPUs is the fixed-partition check
    assert _layout(rollout_gpus=8) is None
    assert "dedicated, evenly partitioned" in _layout(rollout_gpus=7, colocated=False)


def _full_island(tmp_path, *extra: str) -> list[str]:
    argv = [a if a != "lora" else "full" for a in _island_argv(tmp_path)]
    i = argv.index("--rl-max-policy-age")
    argv[i + 1] = extra[0] if extra else "1"
    return argv


def _miles_argv(tmp_path, monkeypatch, argv: list[str]) -> list[str]:
    monkeypatch.setattr(run_config, "_resolve_ref_load", lambda a, model_path: str(model_path))
    monkeypatch.setattr(miles_config, "_resolve_tito_parsers", lambda model: {"qwen3": ("qwen3", "qwen25")}[model])
    args = island_entry.parse_args(argv)
    rc = run_config.resolve_rl_run_config(
        args, model_path=str(tmp_path / "model"), rollout_model_path=None,
        prompt_path=str(tmp_path / "prompts.jsonl"), eval_prompt_path=None, provider=_provider(),
        target_modules=[], yeto_policy_sync=False)
    assert rc.parallel.colocated and rc.parallel.dedicated_rollout_gpus is None
    assert rc.parallel.visible_gpus_per_node == 2
    return list(island_entry.build_ports_launch(args, rc, ()).argv)


def test_full_parameter_colocated_island_runs_miles_colocate_with_trainer_offload(tmp_path, monkeypatch):
    argv = _miles_argv(tmp_path, monkeypatch, _full_island(tmp_path, "1"))
    assert "--colocate" in argv and "--offload-train" in argv and "--no-offload-train" not in argv
    assert "--rollout-num-gpus" not in argv and "--update-weight-transfer-mode" not in argv
    assert _value(argv, "--num-gpus-per-node") == "2" and _value(argv, "--actor-num-gpus-per-node") == "2"
    assert "--lora-rank" not in argv and "--lora-base-cpu-backup" not in argv
    # stage 2: the Harbor agent at policy age 1 is suspended between turns, not fully async
    assert "--agentic-suspend-between-turns" in argv and _value(argv, "--agentic-suspend-max-rounds") == "1"
    assert "--fully-async" not in argv and "--partial-rollout" not in argv


def test_policy_age_zero_keeps_the_serial_colocated_island(tmp_path, monkeypatch):
    argv = _miles_argv(tmp_path, monkeypatch, _full_island(tmp_path, "0"))
    assert "--colocate" in argv and "--offload-train" in argv
    assert "--agentic-suspend-between-turns" not in argv and "--fully-async" not in argv


def test_a_synced_full_parameter_island_cannot_colocate(tmp_path, monkeypatch):
    monkeypatch.setattr(run_config, "_resolve_ref_load", lambda a, model_path: str(model_path))
    args = island_entry.parse_args(_full_island(tmp_path, "1"))
    args.rl_single_island_no_sync = False  # what a synced island's run config sees
    with pytest.raises(ValueError, match="--rl-single-island-no-sync"):
        run_config.resolve_rl_run_config(
            args, model_path=str(tmp_path / "model"), rollout_model_path=None,
            prompt_path=str(tmp_path / "prompts.jsonl"), eval_prompt_path=None, provider=_provider(),
            target_modules=[], yeto_policy_sync=True)


def _plan(gpu: str, *extra: str):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", ("--tuning", "full", "--rl-single-island-no-sync",
                                    "--cluster-prefix", "gef-colo", *extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return dry_run_plan(args)


def _island_tokens(plan) -> list[str]:
    (island,) = plan["island_requests"]
    tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    return shlex.split(tail)


def test_launch_plans_one_colocated_island_on_one_node():
    plan = _plan("modal:1x2xh100", "--rollout-num-gpus-per-engine", "2")
    (island,) = plan["island_requests"]
    assert (island["num_nodes"], island["total_gpus"]) == (1, 2)
    assert island["placement"]["placement"] == "colocated"
    tokens = _island_tokens(plan)
    assert tokens[tokens.index("--parameter-mode") + 1] == "full"
    assert "--rl-placement" not in tokens and "--rollout-num-gpus" not in tokens
    assert plan["syncer"] is None and plan["outer_sync"] is False


def test_launch_keeps_the_fixed_partition_island_command():
    plan = _plan("modal:1x4xh100", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "2",
                 "--rollout-num-gpus-per-engine", "2")
    (island,) = plan["island_requests"]
    assert island["placement"]["placement"] == "fixed-partition"
    tokens = _island_tokens(plan)
    assert tokens[tokens.index("--rl-placement") + 1] == "fixed-partition"
    assert tokens[tokens.index("--rollout-num-gpus") + 1] == "2"
