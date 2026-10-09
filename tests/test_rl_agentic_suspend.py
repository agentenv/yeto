"""agentic-rollout-utilization stage 3 (5.1-5.3) and the Miles in-flight cut wiring (3.3).

CPU only: fake session server (aiohttp), fake Codex client, fake executor.
No Ray, no Miles process, no GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from test_codex_bridge_compaction import FakeCodex, _exec, _submit
from test_rl_miles_carry_over import Call, FakePartialEngine, Sample, Span, tok
from test_secrlenv_codex_harness import _fake_miles_sse
from yeto.rl.adapters.miles import carry_over, in_flight
from yeto.rl.adapters.miles import policy_age as miles_policy_age
from yeto.rl.engine.cut import context_problems
from yeto.rl.engine.policy_age import PolicyAgeError
from yeto.rl.engine.trajectory_timing import PhaseClock
from yeto.rl.harness.codex import codex_harness_agent as harness
from yeto.rl.harness.codex import codex_openenv_generate as wrapper
from yeto.rl.harness.codex import codex_openenv_subprocess_agent_function as trusted

AGENT = "yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run"


# ------------------------------------------------------------ 5.1 the gate


class GatedSessionServer:
    """Fake Miles session server; ``on_request(n, body)`` may return an HTTP status
    to answer instead of the next completion (503 = engine aborted the turn)."""

    def __init__(self, replies, on_request=None):
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.on_request = on_request

    async def __aenter__(self):
        async def chat(request: web.Request) -> web.Response:
            body = await request.json()
            self.bodies.append(body)
            status = self.on_request(len(self.bodies), body) if self.on_request else None
            if status is not None:
                return web.json_response({"error": "upstream generation aborted"}, status=status)
            return web.Response(body=_fake_miles_sse(self.replies.pop(0)), content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/sessions/{sid}/v1/chat/completions", chat)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/sessions/S0"
        return self

    async def __aexit__(self, *_):
        await self.runner.cleanup()


def _gate_env(monkeypatch, gate, max_s="30"):
    monkeypatch.setenv("YETO_CODEX_BACKEND_MAX_TOKENS", "512")
    monkeypatch.delenv("YETO_CODEX_COMPACTION_ENABLED", raising=False)
    monkeypatch.setenv(harness.SUSPEND_GATE_ENV, str(gate))
    monkeypatch.setenv(harness.SUSPEND_MAX_SECONDS_ENV, max_s)
    monkeypatch.setattr(harness, "SUSPEND_POLL_SECONDS", 0.02)


def _bridge(url, metrics):
    return harness._ResponsesBridge(url, "solve target", {"max_tokens": 512}, metrics, max_seq_len=8000)


def test_cutoff_during_a_tool_call_parks_the_next_model_turn_until_resume(monkeypatch, tmp_path):
    gate = tmp_path / "t.gate"
    _gate_env(monkeypatch, gate)
    edges: list[str] = []
    harness.set_suspend_event_sink(edges.append)

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with GatedSessionServer([_exec(1, 500), _submit(900)]) as server:
            async with _bridge(server.url, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                assert (await codex.turn())[0] == 200  # model turn 1 -> tool call
                gate.write_text("suspended\n")  # cut-off while the tool runs
                # the tool returns and its result goes into the next request; the
                # bridge holds that request (no model call while suspended)
                pending = asyncio.ensure_future(codex.turn('{"output":"tool result"}'))
                await asyncio.sleep(0.3)
                assert not pending.done() and len(server.bodies) == 1
                gate.unlink()  # resume
                status, resp = await pending
                assert status == 200 and resp["output"][1]["name"] == "submit"
                return server, metrics

    try:
        server, metrics = asyncio.run(scenario())
    finally:
        harness.set_suspend_event_sink(None)
    # the request after resume carries the tool result written before the suspension
    assert any("tool result" in json.dumps(m) for m in server.bodies[1]["messages"])
    assert metrics.suspensions == 1 and metrics.suspended_seconds >= 0.25
    assert metrics.suspend_retried_turns == 0 and metrics.suspend_expired == 0
    assert edges == ["enter", "exit"]
    # parked time is not generation time
    assert metrics.total_generation_time < metrics.suspended_seconds


def test_a_model_turn_aborted_by_the_cutoff_is_redone_after_resume(monkeypatch, tmp_path):
    gate = tmp_path / "t.gate"
    _gate_env(monkeypatch, gate)

    def on_request(n, _body):
        if n == 2:  # the cut-off: gate closed first, then the engine aborts this turn
            gate.write_text("suspended\n")
            asyncio.get_event_loop().call_later(0.2, gate.unlink)
            return 503
        return None

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with GatedSessionServer([_exec(1, 500), _submit(900)], on_request) as server:
            async with _bridge(server.url, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                assert (await codex.turn())[0] == 200
                status, resp = await codex.turn('{"output":"r"}')
                assert status == 200 and resp["output"][1]["name"] == "submit"
                return server, metrics

    server, metrics = asyncio.run(scenario())
    assert len(server.bodies) == 3 and server.bodies[1] == server.bodies[2]  # same turn redone
    assert metrics.suspend_retried_turns == 1 and metrics.suspensions == 1


def test_a_503_without_a_suspension_is_still_an_error(monkeypatch, tmp_path):
    _gate_env(monkeypatch, tmp_path / "never.gate")

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with GatedSessionServer([_exec(1, 500)], lambda n, _b: 503 if n == 2 else None) as server:
            async with _bridge(server.url, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                assert (await codex.turn())[0] == 200
                status, _ = await codex.turn('{"output":"r"}')
                return status, bridge, metrics

    status, bridge, metrics = asyncio.run(scenario())
    assert status == 400 and "HTTP 503" in str(bridge.fatal.result())
    assert metrics.suspensions == 0


def test_a_suspension_past_the_survival_limit_discards_the_trajectory(monkeypatch, tmp_path):
    gate = tmp_path / "t.gate"
    _gate_env(monkeypatch, gate, max_s="0.2")

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with GatedSessionServer([_exec(1, 500), _submit(900)]) as server:
            async with _bridge(server.url, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                assert (await codex.turn())[0] == 200
                gate.write_text("suspended\n")
                status, _ = await codex.turn('{"output":"r"}')
                return status, bridge, metrics, server

    status, bridge, metrics, server = asyncio.run(scenario())
    assert status == 400 and isinstance(bridge.fatal.result(), harness.CodexSuspendExpired)
    assert metrics.suspend_expired == 1 and len(server.bodies) == 1  # never sampled again


def test_codex_waits_for_a_held_request_only_when_suspension_is_on(monkeypatch, tmp_path):
    monkeypatch.delenv(harness.SUSPEND_GATE_ENV, raising=False)
    assert harness._suspend_settings() == []  # default command line unchanged
    monkeypatch.setenv(harness.SUSPEND_GATE_ENV, str(tmp_path / "g"))
    monkeypatch.setenv(harness.SUSPEND_MAX_SECONDS_ENV, "600")
    assert harness._suspend_settings() == ["model_providers.miles.stream_idle_timeout_ms=1200000"]


def test_phase_clock_keeps_parked_time_out_of_generation():
    now = [0.0]
    clock = PhaseClock(lambda: now[0])
    now[0] = 2.0
    clock.enter_tool()
    now[0] = 3.0
    clock.exit_tool()
    now[0] = 4.0
    clock.enter_suspend()  # parked before model turn 2
    now[0] = 104.0
    clock.exit_suspend()
    now[0] = 106.0
    out = clock.finish()
    assert out["turn_generation_seconds"] == [2.0, 3.0] and out["turn_tool_seconds"] == [1.0]
    assert out["suspended_seconds"] == 100.0 and out["worker_seconds"] == 106.0
    assert "suspended_seconds" not in PhaseClock(lambda: 0.0).finish()  # unchanged when never parked


# ------------------------------------------------- 5.1/5.3 trusted-side hooks


def test_suspend_closes_the_gates_of_trajectories_in_flight_and_resume_opens_them(monkeypatch, tmp_path):
    monkeypatch.setattr(trusted, "_SUSPEND_DIR", None)
    monkeypatch.setattr(trusted, "_SUSPEND_STATE", {"suspended_at": None, "gates": {}})
    monkeypatch.setattr(trusted, "_INFLIGHT", {"a": {"stage": "worker"}, "b": {"stage": "verify"},
                                               "c": {"stage": "teardown"}})
    assert trusted._suspend_worker_env("x") == {}  # no suspension in use: env unchanged
    asyncio.run(trusted.resume())  # Miles calls resume at the start of every rollout
    assert trusted._suspend_worker_env("x") == {harness.SUSPEND_GATE_ENV: trusted._gate_path("x")}
    assert asyncio.run(trusted.suspend()) == 2  # teardown is not suspended
    gates = trusted._SUSPEND_STATE["gates"]
    assert set(gates) == {"a", "b"} and all(os.path.exists(p) for p in gates.values())
    # a trajectory started while suspended is parked too
    env = trusted._suspend_worker_env("late")
    assert os.path.exists(env[harness.SUSPEND_GATE_ENV])
    paths = list(trusted._SUSPEND_STATE["gates"].values())
    assert asyncio.run(trusted.resume()) == 3
    assert not any(os.path.exists(p) for p in paths) and trusted._SUSPEND_STATE["gates"] == {}


# ------------------------------------------------------- 5.2 version segments


def _traj(versions):
    calls = [{"spans": [{"version": v}]} for v in versions]
    return SimpleNamespace(metadata={}, weight_versions=calls, status="completed", loss_mask=None)


def test_codex_versions_are_recorded_as_segments_within_the_window(monkeypatch):
    monkeypatch.setattr(wrapper, "assert_sample_alignment", lambda s: None)
    ok = _traj([tok(3), tok(3), tok(4)])
    wrapper.apply_trajectory_bookkeeping(SimpleNamespace(group_index=1, index=9), [ok],
                                         expected_version=tok(3), max_policy_age=1, current_version=tok(4))
    assert ok.status == "completed" and wrapper.POLICY_AGE_KEY not in ok.metadata
    assert ok.metadata[wrapper.VERSION_SEGMENTS_KEY] == sorted({tok(3), tok(4)})
    old = _traj([tok(2), tok(4)])
    wrapper.apply_trajectory_bookkeeping(SimpleNamespace(group_index=1, index=9), [old],
                                         expected_version=tok(2), max_policy_age=1, current_version=tok(4))
    reason = old.metadata[wrapper.INFRASTRUCTURE_KEY]
    assert reason.startswith(wrapper.POLICY_AGE_EXCEEDED) and wrapper.POLICY_AGE_KEY not in old.metadata
    # limit 0: drift is still a policy-identity violation (unchanged)
    drift = _traj([tok(3), tok(4)])
    wrapper.apply_trajectory_bookkeeping(SimpleNamespace(group_index=1, index=9), [drift],
                                         expected_version=tok(3))
    assert drift.metadata[wrapper.POLICY_AGE_KEY] == 1


def test_window_problem_rules():
    assert wrapper.window_problem([tok(3)], tok(4), 1) is None
    assert "newer" in wrapper.window_problem([tok(5)], tok(4), 1)
    assert "unparsable" in wrapper.window_problem(["default"], tok(4), 1)
    assert "no generation" in wrapper.window_problem([], tok(4), 1)


# ----------------------------------------------------- policy age (launcher)


def test_miles_stage_three_accepts_the_suspendable_codex_agent_only():
    assert miles_policy_age.SUPPORT.stage == 3
    miles_policy_age.check_task(1, custom_generate=miles_policy_age.AGENTIC_GENERATE, custom_agent=AGENT)
    with pytest.raises(PolicyAgeError, match="suspended between turns"):
        miles_policy_age.check_task(1, custom_generate=miles_policy_age.AGENTIC_GENERATE,
                                    custom_agent="some.other.agent.run")
    with pytest.raises(PolicyAgeError, match="stage 3"):
        miles_policy_age.check_task(1, custom_generate="miles.rollout.generate_hub.multi_turn.generate")
    assert miles_policy_age.policy_age_argv(1, agentic=True) == (
        "--agentic-suspend-between-turns", "--agentic-suspend-max-rounds", "1")
    assert miles_policy_age.policy_age_argv(0, agentic=True) == ()


# ------------------------------------------- carry-over metrics (5.3 / 5.5)


def test_agentic_tool_tokens_are_not_cross_version_and_ratios_are_summarised():
    s = Sample(index=0, group_index=0)
    FakePartialEngine().generate(s, 3, 2, finish=False)
    s.tokens = s.tokens + [9, 9]  # tool output (no version span)
    s.response_length += 2
    s.rollout_log_probs = s.rollout_log_probs + [0.0, 0.0]
    FakePartialEngine().generate(s, 4, 1, finish=True)
    s.loss_mask = [1, 1, 0, 0, 1]
    assert carry_over.response_token_versions(s) == [3, 3, None, None, 4]
    assert carry_over.trained_token_versions(s, 4) == [3, 3, 4, 4, 4]
    out = carry_over.estimate_cross_version_truncation(
        SimpleNamespace(tis_clip_low=0.0, tis_clip=2.0), [s], 4, lambda _s: [-0.5, 0.5, 0.0, 0.0, -0.5])
    assert out["cross_version_scored_tokens"] == 2 and out["cross_version_unscored_samples"] == 0
    assert out["cross_version_truncated_fraction"] == 0.5  # ratios 1 and e (>2: truncated)
    assert out["cross_version_ratio_max"] == pytest.approx(2.718281828, rel=1e-6)


def test_suspend_fields_report_the_fork_stats_and_the_survival_cost():
    args = SimpleNamespace(rollout_suspend_stats={"suspended_groups": 3, "over_age_cancelled_groups": 1},
                           rollout_resume_stats={"resumed_groups": 2, "resumed_done_groups": 0})
    expired = SimpleNamespace(metadata={"tbench_infrastructure_error": "CodexSuspendExpired: x",
                                        "agent_metrics": {"suspensions": 1, "suspended_seconds": 600.0}})
    parked = SimpleNamespace(metadata={"agent_metrics": {"suspensions": 1, "suspended_seconds": 40.5,
                                                         "suspend_retried_turns": 1}})
    plain = SimpleNamespace(metadata={})
    out = carry_over.suspend_fields(args, [[expired, parked], [plain]])
    assert out["suspended_groups"] == 3 and out["resumed_groups"] == 2
    assert out["suspend_expired_trajectories"] == 1 and out["suspended_trajectories"] == 2
    assert out["suspended_env_seconds"] == 640.5 and out["suspend_retried_turns"] == 1
    assert args.rollout_suspend_stats is None  # consumed
    assert carry_over.suspend_fields(args, []) == {}


# ----------------------------------------- 3.3 Miles in-flight cut wiring


class MilesLikeSample(Sample):
    def to_dict(self):
        d = asdict(self)
        d["status"] = self.status.value
        return d


def _buffered_group(gi, version=3):
    group = [MilesLikeSample(index=gi * 10 + i, group_index=gi) for i in range(2)]
    for s in group:
        FakePartialEngine().generate(s, version, 3, finish=False)
        s.metadata["task_id"] = f"task-{gi}"
    return group


def test_export_and_restore_buffered_groups_and_agentic_references(monkeypatch):
    executor = SimpleNamespace(data_source=SimpleNamespace(buffer=[_buffered_group(1), _buffered_group(2)]))
    monkeypatch.setattr(in_flight, "_codex_suspension",
                        lambda: ({"t7": "/tmp/t7.gate"}, {"t7": {"stage": "worker"}}))
    exported = in_flight.export_in_flight(executor)
    assert exported["buffer_groups"] == 2 and len(exported["entries"]) == 5
    assert in_flight.check_export(exported) == []
    agentic = [e for e in exported["entries"] if e.get("session_ref")]
    assert agentic[0]["session_ref"] == "codex-suspended:t7" and agentic[0]["stage"] == "worker"
    # the cut accepts the buffer because the in-flight section carries it
    base = dict(cut_id="r000004-x", progress=SimpleNamespace(problems=lambda: []),
                algorithm=SimpleNamespace(algorithm_spec_sha256="a" * 64),
                data={"sample_offset": 0, "epoch_id": 0, "sample_group_index": 3, "sample_index": 6,
                      "buffer_length": 2},
                outer={"settled": True}, runtime={})
    entries = tuple(exported["entries"])
    ok_ledger = {"carried_over": 5, "ready_unconsumed": 0, "max_policy_age": 1}
    assert not [p for p in context_problems(**base, ledger=ok_ledger, in_flight=entries, max_policy_age=1)
                if p.startswith(("data", "in_flight", "ledger"))]
    zero = context_problems(**base, ledger={"carried_over": 0, "ready_unconsumed": 0})
    assert any("not carried by a cut" in p for p in zero)  # limit 0: unchanged refusal
    short = context_problems(**base, ledger=ok_ledger, in_flight=entries[:2], max_policy_age=1)
    assert any("carries 1" in p for p in short)

    # restore at version 4 with limit 1: group 2 already trained, agentic lost
    target = SimpleNamespace(data_source=SimpleNamespace(buffer=[], args=None))
    target.data_source.add_samples = lambda groups: target.data_source.buffer.extend(groups)
    report = in_flight.import_in_flight(target, exported["entries"], max_policy_age=1, current_version=4,
                                        completed_group_ids=["g2"], sample_from_dict=dict)
    assert report["resumed_groups"] == 1 and report["resumed"] == 2
    assert report["reasons"] == {"group_already_completed": 2, in_flight.AGENTIC_SESSION_LOST: 1}
    restored = target.data_source.buffer[0]
    assert [m["index"] for m in restored] == [10, 11] and restored[0]["metadata"]["task_id"] == "task-1"
    # too old at version 5 (limit 1): discarded
    target.data_source.buffer.clear()
    report = in_flight.import_in_flight(target, exported["entries"], max_policy_age=1, current_version=5,
                                        sample_from_dict=dict)
    assert report["resumed_groups"] == 0 and report["reasons"]["policy_age_exceeded"] == 4
    # restored under limit 0: everything discarded and reported
    report = in_flight.import_in_flight(target, exported["entries"], max_policy_age=0, current_version=4,
                                        sample_from_dict=dict)
    assert report["discarded"] == 5 and target.data_source.buffer == []


def test_export_refuses_a_buffered_token_of_unknown_version():
    group = _buffered_group(1)
    group[0].weight_versions.append(Call([Span("default", 0, len(group[0].tokens))]))
    group[0].tokens = group[0].tokens + [1]
    group[0].response_length += 1
    group[0].rollout_log_probs.append(-0.1)
    group[0].weight_versions = [Call([Span("default", len(group[0].tokens) - 4, len(group[0].tokens))])]
    with pytest.raises(in_flight.InFlightExportError, match="unknown"):
        in_flight.buffered_entries([group])


def test_cut_source_writes_the_in_flight_section_under_a_limit():
    from yeto.rl.adapters.miles.rebuild_wiring import CutSource

    exported = {"entries": in_flight.buffered_entries([_buffered_group(1)]), "buffer_groups": 1}
    state = SimpleNamespace(policy_version=4, policy_tensor_hash=lambda: "h" * 64)
    driver = SimpleNamespace(published_state=state, published_version=4, local_step=4,
                             at_safe_point=True, expected_token=tok(4),
                             profile=SimpleNamespace(max_policy_age=1))
    rollout = SimpleNamespace(data_cursor=lambda: {"sample_offset": 0, "epoch_id": 0,
                                                   "sample_group_index": 2, "sample_index": 4},
                              export_in_flight=lambda: exported)
    ledger = SimpleNamespace(cut_summary=lambda: {"carried_over": 0, "ready_unconsumed": 0,
                                                  "engine_buffer_length": 1})
    source = CutSource.__new__(CutSource)
    source.driver_ref, source.rollout, source.ledger = driver, rollout, ledger
    source.cut_root, source.backend_fingerprint, source.global_batch_size = "/tmp", "f", 4
    source.identity, source.shared_filesystem = SimpleNamespace(), True
    context = source.context("r000004-x")
    assert len(context.in_flight) == 2 and context.data["buffer_length"] == 1
    assert context.ledger["carried_over"] == 2 and context.ledger["max_policy_age"] == 1
    driver.profile.max_policy_age = 0
    assert source.context("r000004-y").in_flight == ()  # limit 0: section absent
