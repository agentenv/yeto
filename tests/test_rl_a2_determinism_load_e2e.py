"""A2 follow-up (trainer determinism switch reaches every rank) and A2+ (1.7 load
samples with queued/capacity/tool-wait), end to end on CPU:

real ``yeto launch`` CLI -> island learner args -> miles_args -> ports entry
(``execution_profile_for`` / ``compose_island``) -> ``IslandDriver.run`` over the
stubbed upstream of ``test_rl_engine_selection`` -> events / stdout.
No cloud, no GPU, no Ray.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

BASE = ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--gpu", "aws:2xa100@us-east-1")
ROUTER = ("10.0.0.1", 3000)
ENGINES = ("http://10.0.0.2:30000", "http://10.0.0.3:30000")


def _miles_args(args):
    """The island's miles_args fields this path reads (as Miles' parser names them)."""
    from yeto.rl import learner

    miles_args = SimpleNamespace(
        yeto_rl_learner_id=0, num_steps_per_rollout=1, offload_train=False,
        offload_rollout=False, rollout_batch_size=2, n_samples_per_prompt=2,
        eval_interval=args.eval_interval, eval_temperature=args.eval_temperature,
        rollout_temperature=1.0, eval_uses_snapshots=False,
        custom_generate_function_path=getattr(args, "custom_generate_function_path", None),
        sglang_router_ip=ROUTER[0], sglang_router_port=ROUTER[1],
    )
    learner.apply_ports_infra_switches(args, miles_args, {})
    return miles_args


class _Http:
    """Router + SGLang engine endpoints. ``load`` = (running, waiting) per engine."""

    def __init__(self, log, load=(0, 0), capacity=8):
        self.log, self.load, self.capacity = log, load, capacity

    def get(self, url):
        if url == f"http://{ROUTER[0]}:{ROUTER[1]}/worker_inflight":
            return {"inflight": {e: 0 for e in ENGINES}, "cordoned": []}
        base, _, path = url.rpartition("/")
        assert base in ENGINES, url
        if path == "get_load":
            running, waiting = self.load
            return [{"dp_rank": 0, "num_reqs": running + waiting, "num_waiting_reqs": waiting}]
        if path == "server_info":
            return {"internal_states": [{"effective_max_running_requests_per_dp": self.capacity}]}
        raise OSError(url)



def _island(tmp_path, monkeypatch, miles_args, http, *, observe=False, profile=None,
            gen_delay=0.0):
    from tests.test_rl_engine_selection import NAME, _Actor, _Controller
    from tests.test_rl_miles_adapter_rollout import Call, Sample, Span
    from yeto.rl.core import canonical_state
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape
    from yeto.rl.adapters.miles import LoopRunner
    from yeto.rl.adapters.miles import rollout as rollout_mod
    from yeto.rl.adapters.miles import rollout_meta_hook as hook
    from yeto.rl.adapters.miles.entry import (
        compose_island,
        miles_capabilities,
        with_partitioned_serial,
    )
    from yeto.rl.adapters.miles.placement import MilesPlacement, PlacementRequest
    from yeto.rl.adapters.miles.rollout import DirMetadataSource

    sink = tmp_path / "sink"
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{sink}")
    monkeypatch.setattr(rollout_mod, "_http_get_json", lambda url, timeout_s=2.0: http.get(url))
    controller, actor = _Controller(), _Actor()
    original_prepare = controller.prepare_rollout

    async def prepare(rollout_id):
        http.log.append(("prepare", rollout_id))
        await original_prepare(rollout_id)

    controller.prepare_rollout = prepare
    rollout_args = SimpleNamespace(n_samples_per_prompt=2)

    class Executor:
        async def get(self, rollout_id):
            if gen_delay:
                time.sleep(gen_delay)
            token = controller.engine.version
            groups = [[Sample(index=10 * g + i, group_index=g, rollout_id=rollout_id,
                              reward=float(i), weight_versions=[Call([Span(token)])])
                       for i in range(2)] for g in range(2)]
            hook.record_trained_groups(rollout_args, groups)
            hook.extract_rollout_metadata(rollout_args, groups)
            return SimpleNamespace(sample_indices=[s.index for g in groups for s in g])

    async def update_weights(*args, **kwargs):
        pass

    evals = []
    runner = LoopRunner()
    part = PlacementRequest("fixed-partition", 1, 1, 1)
    driver = compose_island(
        miles_args=miles_args,
        launch=SimpleNamespace(placement=part),
        algorithm=AlgorithmSpec(),
        inference_controller=controller, rollout_executor=Executor(), actor_model=actor,
        learner_id=0, base_model_revision="0" * 40, lora_config_hash="1" * 64,
        layout_hash=canonical_state(0, {NAME: torch.zeros(2, 4)}, base_model_revision="0" * 40,
                                    lora_config_hash="1" * 64).layout_hash,
        sync=LocalOnlySync(2), progress=None, metadata=DirMetadataSource(sink),
        capabilities=with_partitioned_serial(miles_capabilities("sha256:" + "0" * 64)),
        runner=runner, events=EventTape(tmp_path / "events.jsonl", 0),
        evaluate=(lambda r: (http.log.append(("eval", r)), evals.append(r), {})[2])
        if miles_args.eval_interval else None,
        eval_interval=miles_args.eval_interval,
        update_weights=update_weights,
        release_refs=lambda args, pack: None,
        flatten_checksums=lambda raw: [{"w": "x"} for _ in raw],
        placement=MilesPlacement(part, {"actor": (None, [0, 1], [0, 1]), "rollout": (None, [1], [1])},
                                 logical=True),
        profile=profile, observe=observe,
    )
    return driver, runner


def _learner(tmp_path, monkeypatch, *cli):
    args, _ = learner_from_run(island_run(BASE + cli, monkeypatch), tmp_path / "home")
    return args


# ------------------------------------------------------------------ A2 follow-up: trainer determinism
def test_cli_deterministic_trainer_reaches_every_ray_worker_with_the_te_switch(
        tmp_path, monkeypatch):
    from yeto.rl import learner
    from yeto.rl.adapters.miles.entry import DETERMINISM_ENV, connect_island_ray

    args = _learner(tmp_path, monkeypatch, "--rl-deterministic-trainer")
    environ = {}
    learner.apply_ports_infra_switches(args, SimpleNamespace(yeto_rl_learner_id=0), environ)
    assert environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert environ == DETERMINISM_ENV
    seen = {}
    ray = SimpleNamespace(init=lambda **kw: seen.update(kw), is_initialized=lambda: False)
    connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.1:6379", **environ}, ray_module=ray)
    workers = seen["runtime_env"]["env_vars"]
    assert {k: workers[k] for k in DETERMINISM_ENV} == DETERMINISM_ENV


# ------------------------------------------------------------------ A2+ / 1.7 load samples
TOOL = ("--custom-generate-function-path", "yeto.rl.tool_wait_workload.generate",
        "--rl-test-tool-delay-s", "5", "--rl-observe-timeline")


def _profile(miles_args):
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.adapters.miles import entry

    spec = AlgorithmSpec()
    return entry.execution_profile_for(
        miles_args, SimpleNamespace(placement=SimpleNamespace(kind="fixed-partition")), spec,
        yeto_policy_sync=False, expected_sha256=spec.sha256())


def _samples(tmp_path):
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()
            if '"rl_load_sample"' in line]


def test_cli_tool_workload_load_samples_carry_queued_capacity_and_tool_waits(
        tmp_path, monkeypatch):
    from yeto.rl.engine.tool_wait import ToolWaitBoard

    args = _learner(tmp_path, monkeypatch, *TOOL)
    miles_args = _miles_args(args)
    assert miles_args.yeto_rl_observe_timeline is True
    board = ToolWaitBoard()
    board.enter("g0-s0")  # a trajectory inside its tool call while generating
    driver, runner = _island(tmp_path, monkeypatch, miles_args, _Http([], load=(0, 0)),
                             observe=True, profile=_profile(miles_args), gen_delay=0.12)
    lazy = driver.rollout._load_tool_wait
    lazy._factory = lambda learner_id: board  # the island's named board actor
    driver.load_sample_interval_s = 0.02
    driver.run()
    runner.close()
    samples = _samples(tmp_path)
    assert samples
    for s in samples:
        assert (s["queued_requests"], s["running_requests"], s["engine_capacity"],
                s["tool_wait_trajectories"], s["active_requests"]) == (0, 0, 16, 1, 0)
        assert s["load_class"] == "tool-wait" and s["ready_groups"] is None


def test_cli_stock_generate_load_samples_classify_saturation_without_tool_waits(
        tmp_path, monkeypatch):
    args = _learner(tmp_path, monkeypatch, "--rl-observe-timeline")
    miles_args = _miles_args(args)
    assert miles_args.custom_generate_function_path is None
    driver, runner = _island(tmp_path, monkeypatch, miles_args,
                             _Http([], load=(8, 3), capacity=8),
                             observe=True, profile=_profile(miles_args), gen_delay=0.12)
    driver.load_sample_interval_s = 0.02
    driver.run()
    runner.close()
    samples = _samples(tmp_path)
    assert samples
    for s in samples:
        assert (s["queued_requests"], s["running_requests"], s["engine_capacity"],
                s["tool_wait_trajectories"]) == (6, 16, 16, 0)
        assert s["load_class"] == "rollout-saturated"
