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
    miles_args = SimpleNamespace(yeto_rl_learner_id=0, use_miles_router=True)
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
    miles_args = SimpleNamespace(yeto_rl_learner_id=0, use_miles_router=True)
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
        "--rl-test-inject-lora-perturb", "0.01",
        "--rl-test-inject-stop-failures", "1", "--rl-test-kill-learner-at", "COMMITTED"),
        monkeypatch)
    assert "yeto_rl_restart_loop python3 -m yeto.rl.learner" in run
    args, env = learner_from_run(run, tmp_path / "home")
    assert args.rl_elastic_state_dir == "/vol/elastic"
    assert env["YETO_RL_TEST_INJECT_LORA_PERTURB"] == "0.01"
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
                  ("--rl-test-inject-lora-perturb", "0.01")):
        with pytest.raises(ValueError, match="need --rl-elastic"):
            launcher._check_ports_infra_switches(_cli(extra), "ports")
    with pytest.raises(ValueError, match="needs --rl-elastic-restart-attempts"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-kill-learner-at", "COMMITTED")),
                                             "ports")
    with pytest.raises(ValueError, match="must be one of"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-kill-learner-at", "NOPE")), "ports")


# ---------------------------------------------------------------- item 7: A5 as two 3-GPU islands
def test_a5_three_plus_three_islands_launch_with_standby_and_elastic(tmp_path, monkeypatch):
    run = island_run(("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
                      "--rl-standby-gpus", "1",
                      "--gpu", "aws:3xa100@us-east-1,aws:3xa100@us-west-2")
                     + _elastic(tmp_path)[:-2]
                     + ("--rl-elastic-cells", "r0,r1", "--rl-elastic-declare-cells",
                        "--rl-elastic-quorum-timeout-s", "120", "--rl-elastic-pause-margin", "2.0",
                        "--rl-test-inject-start-delay-s", "150"), monkeypatch)
    args, env = learner_from_run(run, tmp_path / "home")
    assert (args.actor_num_gpus_per_node, args.rollout_num_gpus, args.rl_standby_gpus) == (1, 1, 1)
    assert args.rl_elastic and args.rl_elastic_declare_cells and args.rl_elastic_cells == "r0,r1"
    assert (args.rl_elastic_quorum_timeout_s, args.rl_elastic_pause_margin) == (120.0, 2.0)
    assert env["YETO_RL_TEST_INJECT_START_DELAY_S"] == "150.0"


# ---------------------------------------------------------------- item 10: determinism
def test_deterministic_trainer_reaches_miles_argv_env_and_ray_workers(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl import learner
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc
    from yeto.rl.engine.miles_adapter.entry import DETERMINISM_ENV, connect_island_ray

    args, _ = learner_from_run(island_run(BASE + ("--rl-deterministic-trainer",), monkeypatch),
                               tmp_path / "home")
    assert args.rl_deterministic_trainer
    environ = {}
    learner.apply_ports_infra_switches(args, SimpleNamespace(), environ)
    assert environ == DETERMINISM_ENV
    (base,), kwargs = _captured_args()
    merged = argparse.Namespace(**{**vars(base), "rl_deterministic_trainer": True})
    argv = mc.translate_run_config(rc.resolve_rl_run_config(merged, **kwargs), AlgorithmSpec()).argv
    assert "--deterministic-mode" in argv
    seen = {}
    ray = SimpleNamespace(init=lambda **kw: seen.update(kw), is_initialized=lambda: False)
    connect_island_ray(environ={"RAY_ADDRESS": "1.2.3.4:6379", **environ}, ray_module=ray)
    for key, value in DETERMINISM_ENV.items():
        assert seen["runtime_env"]["env_vars"][key] == value


def test_determinism_is_off_by_default(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl import learner
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    args, _ = learner_from_run(island_run(BASE, monkeypatch), tmp_path / "home")
    environ = {}
    learner.apply_ports_infra_switches(args, SimpleNamespace(), environ)
    assert environ == {}
    (base,), kwargs = _captured_args()
    assert "--deterministic-mode" not in mc.translate_run_config(
        rc.resolve_rl_run_config(base, **kwargs), AlgorithmSpec()).argv


# ---------------------------------------------------------------- --balance-data vs trainer edges
def _argv_from(args_ns):
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    (base,), kwargs = _captured_args()
    keys = ("rl_elastic", "rl_elastic_trainer_edges")
    merged = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                   "rollout_num_gpus": 1,
                                   **{k: getattr(args_ns, k) for k in keys}})
    return mc.translate_run_config(rc.resolve_rl_run_config(merged, **kwargs), AlgorithmSpec()).argv


def test_trainer_edges_drop_balance_data_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl import learner

    edges, _ = learner_from_run(island_run(BASE + _elastic(tmp_path) + (
        "--rl-elastic-trainer-edges",), monkeypatch), tmp_path / "a")
    plain, _ = learner_from_run(island_run(BASE + _elastic(tmp_path), monkeypatch), tmp_path / "b")
    default, _ = learner_from_run(island_run(BASE, monkeypatch), tmp_path / "c")
    assert edges.rl_elastic_trainer_edges and not plain.rl_elastic_trainer_edges
    with_edges, elastic_only, off = _argv_from(edges), _argv_from(plain), _argv_from(default)
    assert "--balance-data" not in with_edges
    assert "--balance-data" in elastic_only and "--balance-data" in off
    # elastic adds only --use-miles-router; trainer edges only drop --balance-data
    assert [a for a in elastic_only if a != "--use-miles-router"] == list(off)
    assert [a for a in elastic_only if a != "--balance-data"] == list(with_edges)
    miles_args = SimpleNamespace(yeto_rl_learner_id=0, use_miles_router=True)
    learner.apply_ports_infra_switches(edges, miles_args, {})
    assert miles_args.yeto_rl_elastic["trainer_edges"] is True


def test_trainer_edges_need_elastic():
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(_cli(("--rl-elastic-trainer-edges",)), "ports")


# ---------------------------------------------------------------- E2 cut injections (patch v1)
def test_cut_injections_rank_zero_reaches_the_island_and_ray_workers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.engine.miles_adapter.entry import connect_island_ray

    run = island_run(BASE + _elastic(tmp_path) + (
        "--rl-test-inject-cut-save-kill-rank", "0", "--rl-test-inject-cut-restore-sleep", "1:30",
        "--rl-test-inject-rebuild-fail"), monkeypatch)
    _, env = learner_from_run(run, tmp_path / "home")
    assert env["YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK"] == "0"  # rank 0 is not False
    assert env["YETO_RL_TEST_INJECT_CUT_RESTORE_SLEEP"] == "1:30"
    assert env["YETO_RL_TEST_INJECT_REBUILD_FAIL"] == "1"  # one shared rebuild-fail variable
    seen = {}
    ray = SimpleNamespace(init=lambda **kw: seen.update(kw), is_initialized=lambda: False)
    connect_island_ray(environ={"RAY_ADDRESS": "1.2.3.4:6379", **env}, ray_module=ray)
    forwarded = seen["runtime_env"]["env_vars"]
    assert forwarded["YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK"] == "0"
    assert forwarded["YETO_RL_TEST_INJECT_CUT_RESTORE_SLEEP"] == "1:30"
    assert "YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK" not in island_run(BASE + _elastic(tmp_path),
                                                                        monkeypatch)


# ---------------------------------------------------------------- --rl-lora-dropout
def test_lora_dropout_reaches_the_miles_argv_and_default_stays_zero(tmp_path, monkeypatch):
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    args, _ = learner_from_run(island_run(BASE + ("--rl-lora-dropout", "0.05"), monkeypatch),
                               tmp_path / "a")
    assert args.rl_lora_dropout == 0.05
    (base,), kwargs = _captured_args()

    def argv(**extra):
        ns = argparse.Namespace(**{**vars(base), "rl_engine": "ports", "parameter_mode": "lora",
                                   **extra})
        return mc.translate_run_config(rc.resolve_rl_run_config(ns, **kwargs), AlgorithmSpec()).argv

    if getattr(base, "parameter_mode", "lora") != "lora":
        import pytest

        pytest.skip("captured base run is not LoRA")
    on, off = argv(rl_lora_dropout=args.rl_lora_dropout), argv()
    assert on[on.index("--lora-dropout") + 1] == "0.05"
    assert off[off.index("--lora-dropout") + 1] == "0"
    assert [a if a != "0.05" else "0" for a in on] == list(off)
    default, _ = learner_from_run(island_run(BASE, monkeypatch), tmp_path / "b")
    assert default.rl_lora_dropout is None


def test_lora_dropout_range_is_checked():
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    with pytest.raises(ValueError, match="--rl-lora-dropout"):
        launcher._check_ports_infra_switches(_cli(("--rl-lora-dropout", "1.0")), "ports")
    with pytest.raises(ValueError, match="--rl-lora-dropout"):
        launcher._check_ports_infra_switches(_cli(("--rl-lora-dropout", "0.05")), "legacy")


# ---------------------------------------------------------------- Miles router under --rl-elastic
def test_elastic_runs_carry_use_miles_router_and_defaults_do_not(tmp_path, monkeypatch):
    import argparse

    from test_rl_argv_snapshot import _captured_args
    from yeto.rl.engine import run_config as rc
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    elastic, _ = learner_from_run(island_run(BASE + _elastic(tmp_path), monkeypatch), tmp_path / "a")
    default, _ = learner_from_run(island_run(BASE, monkeypatch), tmp_path / "b")
    (base,), kwargs = _captured_args()

    def argv(args_ns):
        ns = argparse.Namespace(**{**vars(base), "rl_placement": "fixed-partition",
                                   "rollout_num_gpus": 1, "rl_elastic": args_ns.rl_elastic})
        return mc.translate_run_config(rc.resolve_rl_run_config(ns, **kwargs), AlgorithmSpec()).argv

    on, off = argv(elastic), argv(default)
    assert "--use-miles-router" in on and "--use-miles-router" not in off
    assert [a for a in on if a != "--use-miles-router"] == list(off)
    plain = argparse.Namespace(**{**vars(base)})
    assert "--use-miles-router" not in mc.translate_run_config(
        rc.resolve_rl_run_config(plain, **kwargs), AlgorithmSpec()).argv


def test_elastic_wiring_refuses_missing_fork_verb_preconditions():
    from types import SimpleNamespace

    import pytest

    from yeto.rl.engine.miles_adapter import entry

    cfg = {"resources": {}, "attestation": None, "state_dir": "/s", "initial_config": "c0",
           "declared_cells": ()}
    with pytest.raises(ValueError, match="--use-miles-router"):
        entry.elastic_wiring_for(SimpleNamespace(yeto_rl_elastic=cfg, use_miles_router=False),
                                 profile=None, fingerprint="f")
    with pytest.raises(ValueError, match="rollout offload"):
        entry.elastic_wiring_for(SimpleNamespace(yeto_rl_elastic=cfg, use_miles_router=True,
                                                 offload_rollout=True), profile=None,
                                 fingerprint="f")
    entry.check_elastic_miles_args(SimpleNamespace(use_miles_router=True))


# ---------------------------------------------------------------- controller timeouts (E1-C, E1-D 4)
def test_drain_and_recovery_timeouts_reach_the_controller(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from yeto.rl import learner
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import entry

    res = tmp_path / "res.json"
    res.write_text(json.dumps({"configs": {"c0": {"trainer": 1, "rollout": 1}}, "edges": []}))
    cli = BASE + ("--rl-elastic", "--rl-elastic-resources", str(res),
                  "--rl-elastic-initial-config", "c0", "--rl-elastic-cells", "a")
    profile = ExecutionProfile(name="t", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())

    def controller(extra, home):
        args, _ = learner_from_run(island_run(cli + extra, monkeypatch), home)
        args.rl_elastic_resources = str(res)
        args.rl_elastic_state_dir = str(home / "state")
        miles_args = SimpleNamespace(yeto_rl_learner_id=0, use_miles_router=True)
        learner.apply_ports_infra_switches(args, miles_args, {})
        return entry.elastic_wiring_for(miles_args, profile=profile, fingerprint="f").controller

    ctl = controller(("--rl-elastic-drain-timeout-s", "5", "--rl-elastic-recovery-timeout-s", "60"),
                     tmp_path / "a")
    assert (ctl.timeouts.drain, ctl.timeouts.recovery) == (5.0, 60.0)
    assert ctl.timeouts.safe_point == 600.0  # others keep their defaults
    ctl.close()
    default = controller((), tmp_path / "b")
    assert (default.timeouts.drain, default.timeouts.recovery) == (120.0, 900.0)
    default.close()
    assert "--rl-elastic-drain-timeout-s" not in island_run(cli, monkeypatch)


def test_timeouts_need_elastic_and_positive_values():
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(_cli(("--rl-elastic-drain-timeout-s", "5")), "ports")
    with pytest.raises(ValueError, match="must be positive"):
        launcher._check_ports_infra_switches(_cli(("--rl-elastic-drain-timeout-s", "0")), "ports")


def test_hold_and_tool_wait_switches_reach_the_island(tmp_path, monkeypatch):
    run = island_run(BASE + _elastic(tmp_path) + (
        "--rl-elastic-tool-wait-board", "--rl-test-inject-lora-perturb", "0.01",
        "--rl-test-hold-before-check-s", "45", "--rl-test-inject-tool-wait-s", "30"), monkeypatch)
    args, env = learner_from_run(run, tmp_path / "home")
    assert env["YETO_RL_TEST_HOLD_BEFORE_CHECK_S"] == "45.0"
    assert env["YETO_RL_TEST_INJECT_TOOL_WAIT_S"] == "30.0"


def test_hold_and_tool_wait_switch_validation(tmp_path):
    import pytest

    from test_rl_engine_selection import _cli
    from yeto import launcher

    for extra in (("--rl-test-hold-before-check-s", "5", "--rl-test-inject-lora-perturb", "0.01"),
                  ("--rl-test-inject-tool-wait-s", "5", "--rl-elastic-tool-wait-board")):
        with pytest.raises(ValueError, match="need --rl-elastic"):
            launcher._check_ports_infra_switches(_cli(extra), "ports")
    with pytest.raises(ValueError, match="must be positive"):
        launcher._check_ports_infra_switches(
            _cli(_elastic(tmp_path) + ("--rl-elastic-tool-wait-board",
                                       "--rl-test-inject-tool-wait-s", "0")), "ports")
    with pytest.raises(ValueError, match="needs --rl-elastic-tool-wait-board"):
        launcher._check_ports_infra_switches(
            _cli(_elastic(tmp_path) + ("--rl-test-inject-tool-wait-s", "5")), "ports")
    with pytest.raises(ValueError, match="another --rl-test"):
        launcher._check_ports_infra_switches(
            _cli(_elastic(tmp_path) + ("--rl-test-hold-before-check-s", "5")), "ports")
