"""Batch-1 GPU findings: launcher -> island learner wiring, end to end (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

BASE = ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--gpu", "aws:2xa100@us-east-1")


def _eval(tmp_path, *extra):
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"prompt": "q", "label": "a"}\n')
    return ("--rl-eval-interval", "2", "--rl-eval-data", str(heldout),
            "--rl-eval-dataset-name", "held", "--rl-eval-samples-per-prompt", "1", *extra)


def test_eval_sampling_knobs_reach_the_learner_and_the_run_config(tmp_path, monkeypatch):
    run = island_run(BASE + _eval(tmp_path, "--rl-eval-temperature", "0", "--rl-eval-top-p", "1",
                                  "--rl-eval-max-prompt-len", "256",
                                  "--rl-eval-max-response-len", "64",
                                  "--rl-eval-max-context-len", "512"), monkeypatch)
    args, _ = learner_from_run(run, tmp_path / "home")
    assert (args.eval_temperature, args.eval_top_p) == (0.0, 1.0)
    assert (args.eval_max_prompt_len, args.eval_max_response_len, args.eval_max_context_len) == (
        256, 64, 512)
    from yeto.rl.engine.run_config import _resolve_eval

    config = _resolve_eval(args, parameter_mode="lora", prompt_path="/p", eval_prompt_path="/e",
                           yeto_policy_sync=True)
    assert config.temperature == 0.0 and config.max_response_len == 64
    # the shipped heldout file is the one the learner verifies
    from yeto.rl import learner

    args.parameter_mode = "lora"
    assert learner._verify_eval_dataset_identity(args).read_text().startswith('{"prompt"')


def test_eval_knobs_need_the_eval_interval(tmp_path, monkeypatch):
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    with pytest.raises(ValueError, match="need --rl-eval-interval"):
        launcher._check_ports_infra_switches(_cli(("--rl-eval-temperature", "0")), "ports")


def test_default_run_carries_no_eval_flags(tmp_path, monkeypatch):
    args, _ = learner_from_run(island_run(BASE, monkeypatch), tmp_path / "home")
    assert args.eval_interval is None and args.eval_temperature is None


def _elastic(tmp_path):
    import json

    res = tmp_path / "res.json"
    res.write_text(json.dumps({"configs": {"c0": {"trainer": 1, "rollout": 1}}, "edges": []}))
    return ("--rl-elastic", "--rl-elastic-resources", str(res), "--rl-elastic-initial-config",
            "c0", "--rl-elastic-cells", "a")


def test_observe_and_tool_wait_board_reach_the_island_wiring(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl import learner
    from yeto.rl.engine.miles_adapter import elastic_wiring, entry
    from yeto.rl.engine.miles_adapter.elastic_wiring import LazyBoardActor

    run = island_run(BASE + _elastic(tmp_path) + ("--rl-observe-timeline",
                                                  "--rl-elastic-tool-wait-board"), monkeypatch)
    args, _ = learner_from_run(run, tmp_path / "home")
    assert args.rl_observe_timeline and args.rl_elastic_tool_wait_board
    miles_args = SimpleNamespace(yeto_rl_learner_id=0)
    learner.apply_ports_infra_switches(args, miles_args, {})
    assert miles_args.yeto_rl_observe_timeline is True
    seen = {}
    monkeypatch.setattr(elastic_wiring, "build_elastic", lambda **kw: seen.update(kw) or "W")
    entry.elastic_wiring_for(miles_args, profile="P", fingerprint="F")
    board = seen["tool_wait_board"]
    assert isinstance(board, LazyBoardActor) and board.learner_id == 0
    # lazy: the actor is created only on first use
    made = []
    board._factory = lambda lid: made.append(lid) or SimpleNamespace(snapshot="S")
    assert board.snapshot == "S" and made == [0]


def test_defaults_set_neither_observe_nor_board(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl import learner
    from yeto.rl.engine.miles_adapter import elastic_wiring, entry

    args, _ = learner_from_run(island_run(BASE + _elastic(tmp_path), monkeypatch), tmp_path / "h")
    miles_args = SimpleNamespace(yeto_rl_learner_id=0)
    learner.apply_ports_infra_switches(args, miles_args, {})
    assert not hasattr(miles_args, "yeto_rl_observe_timeline")
    seen = {}
    monkeypatch.setattr(elastic_wiring, "build_elastic", lambda **kw: seen.update(kw) or "W")
    entry.elastic_wiring_for(miles_args, profile="P", fingerprint="F")
    assert "tool_wait_board" not in seen


# ---------------------------------------------------------------- item 1: fork cells (F-R1 prep)
def test_deferred_cells_travel_from_the_cli_to_the_miles_placement_map(tmp_path, monkeypatch):
    import argparse
    import json

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    run = island_run(BASE + _elastic(tmp_path)[:-2] + ("--rl-elastic-deferred-cells", "2"),
                     monkeypatch)
    args, _ = learner_from_run(run, tmp_path / "home")
    assert args.rl_elastic_deferred_cells == 2 and args.rl_elastic_cells is None
    (base,), kwargs = _captured_args()
    merged = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                   "rollout_num_gpus": 1,
                                   "rl_elastic_deferred_cells": args.rl_elastic_deferred_cells})
    argv = mc.translate_run_config(rc.resolve_rl_run_config(merged, **kwargs), AlgorithmSpec()).argv
    pm = json.loads(argv[argv.index("--yeto-placement-map") + 1])
    assert pm["deferred_rollout_cells"] == 2 and pm["rollout"] and "trainer" in pm


def test_default_run_has_no_deferred_cells_and_no_placement_map_change(tmp_path, monkeypatch):
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    args, _ = learner_from_run(island_run(BASE + _elastic(tmp_path), monkeypatch), tmp_path / "h")
    assert args.rl_elastic_deferred_cells == 0
    (base,), kwargs = _captured_args()
    part = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                 "rollout_num_gpus": 1})
    before = mc.translate_run_config(rc.resolve_rl_run_config(part, **kwargs), AlgorithmSpec()).argv
    part.rl_elastic_deferred_cells = 0
    after = mc.translate_run_config(rc.resolve_rl_run_config(part, **kwargs), AlgorithmSpec()).argv
    assert before == after and "--yeto-placement-map" not in after


def test_declared_cells_come_from_the_fork_names():
    import asyncio
    from types import SimpleNamespace

    import pytest

    from yeto.rl.engine.miles_adapter.entry import resolve_declared_cells

    runner = SimpleNamespace(run=asyncio.run)
    names = {"inference-engine-all-0-0-00000": {}, "inference-engine-all-0-0-00001": {}}

    class Fork:
        async def describe_cells(self):
            return names

    assert resolve_declared_cells(Fork(), runner) == tuple(sorted(names))
    assert resolve_declared_cells(Fork(), runner, ("inference-engine-all-0-0-00001",)) == (
        "inference-engine-all-0-0-00001",)
    with pytest.raises(ValueError, match=r"\['c0'\] are not cells the fork declares"):
        resolve_declared_cells(Fork(), runner, ("c0",))
    with pytest.raises(ValueError, match="--rl-elastic-cells is required"):
        resolve_declared_cells(object(), runner, ())
    assert resolve_declared_cells(object(), runner, ("x",)) == ("x",)
