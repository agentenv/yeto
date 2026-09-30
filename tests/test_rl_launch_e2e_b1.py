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
def test_declared_cells_travel_from_the_cli_to_the_fork_rollout_cells(tmp_path, monkeypatch):
    import argparse
    import json

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    elastic = _elastic(tmp_path)[:-2] + ("--rl-elastic-cells", "r0,r1,r2",
                                         "--rl-elastic-declare-cells")
    run = island_run(("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
                      "--gpu", "aws:3xa100@us-east-1", "--rl-standby-gpus", "1") + elastic,
                     monkeypatch)
    args, _ = learner_from_run(run, tmp_path / "home")
    assert args.rl_elastic_declare_cells and args.rl_elastic_cells == "r0,r1,r2"
    (base,), kwargs = _captured_args()
    merged = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                   "rollout_num_gpus": 1, "rl_standby_gpus": 1,
                                   "rl_elastic_declare_cells": True,
                                   "rl_elastic_cells": args.rl_elastic_cells})
    argv = mc.translate_run_config(rc.resolve_rl_run_config(merged, **kwargs), AlgorithmSpec()).argv
    pm = json.loads(argv[argv.index("--yeto-placement-map") + 1])
    # the FR1 interface: {"rollout_cells": [{"name", "bundles", "start"}]}; nothing else extra
    assert set(pm) == {"trainer", "rollout", "standby", "rollout_cells"}
    assert pm["rollout_cells"] == [
        {"name": "r0", "bundles": pm["rollout"], "start": True},
        {"name": "r1", "bundles": pm["standby"], "start": False},
        {"name": "r2", "bundles": [], "start": False},
    ]


def test_default_run_declares_no_cells_and_keeps_the_placement_map(tmp_path, monkeypatch):
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    args, _ = learner_from_run(island_run(BASE + _elastic(tmp_path), monkeypatch), tmp_path / "h")
    assert not args.rl_elastic_declare_cells
    (base,), kwargs = _captured_args()
    part = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                 "rollout_num_gpus": 1})
    before = mc.translate_run_config(rc.resolve_rl_run_config(part, **kwargs), AlgorithmSpec()).argv
    part.rl_elastic_cells = "a"  # names alone (no declare) change nothing
    after = mc.translate_run_config(rc.resolve_rl_run_config(part, **kwargs), AlgorithmSpec()).argv
    assert before == after and "--yeto-placement-map" not in after


def test_declared_cells_resolve_yeto_names_through_the_fork_alias():
    import asyncio
    from types import SimpleNamespace

    import pytest

    from yeto.rl.engine.miles_adapter.entry import resolve_declared_cells

    runner = SimpleNamespace(run=asyncio.run)
    cells = {"inference-engine-all-0-0-00000": {"alias": "r0"},
             "inference-engine-all-0-0-00001": {"alias": "r1"}}

    class Fork:
        async def describe_cells(self):
            return cells

    assert resolve_declared_cells(Fork(), runner) == tuple(sorted(cells))
    assert resolve_declared_cells(Fork(), runner, ("r1", "r0")) == (
        "inference-engine-all-0-0-00001", "inference-engine-all-0-0-00000")
    assert resolve_declared_cells(Fork(), runner, ("inference-engine-all-0-0-00001",)) == (
        "inference-engine-all-0-0-00001",)
    with pytest.raises(ValueError, match=r"\['c0'\] are not cells the fork declares"):
        resolve_declared_cells(Fork(), runner, ("c0",))
    with pytest.raises(ValueError, match="--rl-elastic-cells is required"):
        resolve_declared_cells(object(), runner, ())
    assert resolve_declared_cells(object(), runner, ("x",)) == ("x",)


def test_declare_cells_needs_names(tmp_path):
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    res = tmp_path / "r.json"
    res.write_text('{"configs": {"c0": {"trainer": 1, "rollout": 1}}, "edges": []}')
    with pytest.raises(ValueError, match="needs --rl-elastic-cells"):
        launcher._check_ports_infra_switches(_cli((
            "--rl-placement", "fixed-partition", "--rl-elastic", "--rl-elastic-resources", str(res),
            "--rl-elastic-initial-config", "c0", "--rl-elastic-declare-cells")), "ports")


# ---------------------------------------------------------------- item 4: tool-wait workload
TOOL = ("--custom-generate-function-path", "yeto.rl.tool_wait_workload.generate",
        "--rl-test-tool-delay-s", "30")


def test_tool_workload_reaches_the_learner_and_every_ray_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.engine.miles_adapter.entry import connect_island_ray
    from yeto.rl.tool_wait_workload import TOOL_DELAY_ENV, tool_delay_s

    run = island_run(BASE + TOOL, monkeypatch)
    args, env = learner_from_run(run, tmp_path / "home")
    assert args.custom_generate_function_path == "yeto.rl.tool_wait_workload.generate"
    assert env[TOOL_DELAY_ENV] == "30.0" and tool_delay_s(env) == 30.0
    seen = {}
    ray = SimpleNamespace(init=lambda **kw: seen.update(kw), is_initialized=lambda: False)
    connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.1:6379", **env}, ray_module=ray)
    assert seen["runtime_env"]["env_vars"][TOOL_DELAY_ENV] == "30.0"


def test_tool_delay_needs_the_workload_generate(tmp_path, monkeypatch):
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    with pytest.raises(ValueError, match="needs --custom-generate-function-path"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-tool-delay-s", "30")), "ports")
    run = island_run(BASE, monkeypatch)
    assert "YETO_RL_TEST_TOOL_DELAY_S" not in run


def test_tool_call_is_counted_on_the_board_and_in_non_generation_time():
    import asyncio
    from types import SimpleNamespace

    from yeto.rl.engine.tool_wait import ToolWaitBoard, read_tool_wait
    from yeto.rl.tool_wait_workload import tool_call

    board = ToolWaitBoard()
    sample = SimpleNamespace(group_index=3, index=1, non_generation_time=0.5)
    during = []

    async def sleep(seconds):
        during.append((seconds, read_tool_wait(board).in_flight))

    clock = iter([10.0, 40.0])
    waited = asyncio.run(tool_call(sample, delay=30.0, board=board, sleep=sleep,
                                   clock=lambda: next(clock)))
    assert during == [(30.0, 1)] and waited == 30.0
    assert read_tool_wait(board).in_flight == 0 and sample.non_generation_time == 30.5


def test_workload_generate_delays_train_samples_only(monkeypatch):
    import asyncio
    import sys
    import types
    from types import SimpleNamespace

    from yeto.rl import tool_wait_workload as w

    calls = []

    async def stock(args, sample, params, evaluation=False):
        calls.append(("gen", evaluation))
        return sample

    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout",
                        types.SimpleNamespace(generate=stock))
    monkeypatch.setitem(sys.modules, "miles.rollout.base_types",
                        types.SimpleNamespace(GenerateFnOutput=lambda samples: ("out", samples)))

    async def fake_call(sample, *, delay, board):
        calls.append(("tool", delay))

    monkeypatch.setattr(w, "tool_call", fake_call)
    monkeypatch.setattr(w, "_board", lambda lid: "B")
    monkeypatch.setenv(w.TOOL_DELAY_ENV, "5")
    inp = SimpleNamespace(args=SimpleNamespace(yeto_rl_learner_id=0), sample="S",
                          sampling_params={}, evaluation=False)
    assert asyncio.run(w.generate(inp)) == ("out", "S")
    inp.evaluation = True
    asyncio.run(w.generate(inp))
    assert calls == [("tool", 5.0), ("gen", False), ("gen", True)]


# ---------------------------------------------------------------- items 5/6: injections, restart
def test_injection_and_restart_switches_reach_the_island(tmp_path, monkeypatch):
    run = island_run(BASE + _elastic(tmp_path) + (
        "--rl-elastic-state-dir", "/vol/elastic", "--rl-elastic-restart-attempts", "2",
        "--rl-test-inject-weight-override", "/vol/other-ckpt",
        "--rl-test-inject-stop-failures", "1", "--rl-test-kill-learner-at", "COMMITTED"),
        monkeypatch)
    assert "yeto_rl_restart_loop python3 -m yeto.rl.learner" in run
    args, env = learner_from_run(run, tmp_path / "home")
    assert args.rl_elastic_state_dir == "/vol/elastic"
    assert env["YETO_RL_TEST_INJECT_WEIGHT_OVERRIDE_PATH"] == "/vol/other-ckpt"
    assert env["YETO_RL_TEST_INJECT_STOP_FAILURES"] == "1"
    assert env["YETO_RL_TEST_KILL_LEARNER_AT"] == "COMMITTED"
    assert env["YETO_RL_RESTART_ATTEMPTS"] == "2"


def test_default_run_has_no_injection_and_no_restart_loop(tmp_path, monkeypatch):
    run = island_run(BASE + _elastic(tmp_path), monkeypatch)
    assert "yeto_rl_restart_loop" not in run and "YETO_RL_TEST_" not in run
    args, _ = learner_from_run(run, tmp_path / "home")
    assert args.rl_elastic_state_dir == "~/yeto-rl/elastic-state" or args.rl_elastic_state_dir.endswith(
        "yeto-rl/elastic-state")


def test_restart_loop_reruns_the_same_command_until_success(tmp_path):
    import subprocess

    from yeto.launcher import RESTART_LOOP_FN

    counter = tmp_path / "n"
    script = (RESTART_LOOP_FN + f"yeto_rl_restart_loop bash -c 'echo x >> {counter}; "
              f"[ $(wc -l < {counter}) -ge 3 ]'\n")
    ok = subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin",
                                                    "YETO_RL_RESTART_ATTEMPTS": "2"})
    assert ok.returncode == 0 and counter.read_text().count("x") == 3
    counter.unlink()
    capped = subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin",
                                                        "YETO_RL_RESTART_ATTEMPTS": "1"})
    assert capped.returncode != 0 and counter.read_text().count("x") == 2


def test_injection_switches_are_refused_without_elastic_or_restart():
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    for extra in (("--rl-test-inject-stop-failures", "1"), ("--rl-elastic-state-dir", "/v"),
                  ("--rl-test-inject-weight-override", "/c")):
        with pytest.raises(ValueError, match="need --rl-elastic"):
            launcher._check_ports_infra_switches(_cli(extra), "ports")
    with pytest.raises(ValueError, match="needs --rl-elastic-restart-attempts"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-kill-learner-at", "COMMITTED")),
                                             "ports")
    with pytest.raises(ValueError, match="must be one of"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-kill-learner-at", "NOPE")), "ports")
