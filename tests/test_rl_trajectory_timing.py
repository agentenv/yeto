"""agentic-rollout-utilization 1.3: phase clock and tool-wait wiring."""

from types import SimpleNamespace

from yeto.rl.engine.trajectory_timing import PhaseClock, phase_totals, timing_fields


def test_phase_clock_splits_generation_and_tool_and_sums_to_wall():
    ticks = iter([0.0, 2.0, 5.0, 6.0, 6.5, 9.0])
    clock = PhaseClock(lambda: next(ticks))
    clock.enter_tool()   # 2.0: generation 2.0
    clock.enter_tool()   # repeated edge ignored (no tick consumed)
    clock.exit_tool()    # 5.0: tool 3.0
    clock.enter_tool()   # 6.0: generation 1.0
    clock.exit_tool()    # 6.5: tool 0.5
    out = clock.finish()  # 9.0: generation 2.5
    assert out == {"worker_seconds": 9.0, "generation_seconds": 5.5, "tool_seconds": 3.5,
                   "turn_generation_seconds": [2.0, 1.0, 2.5],
                   "turn_tool_seconds": [3.0, 0.5]}
    totals = phase_totals({**out, "evaluate_time": 4.0})
    assert totals == {"generation_seconds": 5.5, "tool_seconds": 3.5, "judge_seconds": 4.0}
    assert totals["generation_seconds"] + totals["tool_seconds"] == out["worker_seconds"]


def test_timing_fields_pick_typed_values_from_metadata_and_agent_metrics():
    meta = {"trajectory_started_at": 1.0, "sandbox_start_seconds": "x",
            "agent_metrics": {"turn_tool_seconds": [1, 2.5], "turn_generation_seconds": [True]}}
    assert timing_fields(meta) == {"trajectory_started_at": 1.0, "turn_tool_seconds": [1.0, 2.5]}


def test_codex_openenv_agent_reads_the_island_tool_wait_board():
    from yeto.rl import CODEX_OPENENV_AGENT
    from yeto.rl.adapters.miles.elastic_wiring import LazyBoardActor
    from yeto.rl.adapters.miles.entry import load_tool_wait_source

    args = SimpleNamespace(custom_generate_function_path="miles.x.generate",
                           custom_agent_function_path=CODEX_OPENENV_AGENT, yeto_rl_learner_id=1)
    assert isinstance(load_tool_wait_source(args), LazyBoardActor)
    args.custom_agent_function_path = "other.agent.run"
    assert load_tool_wait_source(args) is None


def test_engine_kv_fields_and_round_peaks():
    from yeto.rl.adapters.miles.rollout import _engine_kv_capacity, _engine_kv_used
    from yeto.rl.engine.timeline import load_peaks, validate_load_sample

    load = [{"num_reqs": 3, "num_waiting_reqs": 1, "num_tokens": 900, "num_pending_tokens": 100},
            {"num_reqs": 1, "num_waiting_reqs": 0, "num_tokens": 50, "num_pending_tokens": 0}]
    assert _engine_kv_used(load) == 850
    assert _engine_kv_used([{"num_reqs": 1, "num_waiting_reqs": 0}]) is None  # older engine
    info = {"internal_states": [{"memory_usage": {"token_capacity": 4000}},
                                {"memory_usage": {"token_capacity": 4000}}]}
    assert _engine_kv_capacity(info) == 8000
    assert _engine_kv_capacity({"internal_states": [{}]}) is None
    assert validate_load_sample({"kv_used_tokens": 850, "kv_capacity_tokens": 8000}) == []
    events = [{"event": "rl_load_sample", "t": 1.0, "queued_requests": 2,
               "kv_used_tokens": 800, "kv_capacity_tokens": 8000},
              {"event": "rl_load_sample", "t": 2.0, "queued_requests": 5,
               "kv_used_tokens": 960, "kv_capacity_tokens": 8000},
              {"event": "rl_load_sample", "t": 9.0, "queued_requests": 50}]
    assert load_peaks(events, 0.0, 3.0) == {"peak_queued_requests": 5.0, "peak_kv_fraction": 0.12}
    assert load_peaks(events, 4.0, 5.0) == {"peak_queued_requests": None, "peak_kv_fraction": None}
