"""S26 E9.5: Harbor agentic trajectories suspended between rounds (CPU).

Real pieces: the turn gate, the outer layer, the gate relay over HTTP on
127.0.0.1 (aiohttp), the Miles agent function and hooks, the cut export /
restore, the verl rollout engine retry.  Fakes: Miles' session server (an
aiohttp app), the Harbor trial (an agent that calls its endpoint), verl's
rollout client.  No Ray, no Miles process, no GPU.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import test_agentic_harbor as agentic
from test_agentic_harbor import _FakeTrial, _Harness, _result, _task

from yeto.rl.agentic import AdmissionGate, HarnessConfigError, Outcome, TrialVerdict, run_agent_task
from yeto.rl.agentic.suspend import (
    SESSION_REF_PREFIX,
    TIMEOUT_AFTER_SUSPENSION_REASON,
    SuspendLedger,
    TurnGate,
    after_suspension,
    is_suspended_ref,
    suspended_references,
)
from yeto.rl.agentic.types import OUTCOME_KEY, AgentBudget, AgentTrajectory

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

miles_stubs = agentic.miles_stubs


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ gate + ledger


def test_gate_holds_waiters_and_counts_each_closure_once():
    async def go():
        gate = TurnGate()
        ledger = gate.register("t1", task_id="fix-git")
        await gate.wait_open(ledger)  # open: no wait, nothing counted
        assert ledger.metrics() == {}
        gate.close()
        assert [r["session_ref"] for r in suspended_references()] == [SESSION_REF_PREFIX + "t1"]
        waiters = [asyncio.ensure_future(gate.wait_open(ledger)) for _ in range(2)]
        await asyncio.sleep(0.05)
        assert ledger.waiting == 2 and not any(w.done() for w in waiters)
        gate.open()
        await asyncio.gather(*waiters)
        assert ledger.suspensions == 1 and ledger.suspended_seconds > 0 and ledger.waiting == 0
        assert suspended_references() == []  # open gate: nothing suspended
        gate.close()
        await asyncio.gather(gate.wait_open(ledger), _open_later(gate))
        assert ledger.suspensions == 2
        gate.unregister(ledger)
        assert all(r["trajectory_id"] != "t1" for r in suspended_references())
        gate.open()

    _run(go())


async def _open_later(gate, delay=0.05):
    await asyncio.sleep(delay)
    gate.open()


def test_duplicate_trajectory_is_refused_and_refs_are_recognised():
    gate = TurnGate()
    gate.register("t1")
    with pytest.raises(ValueError, match="already registered"):
        gate.register("t1")
    assert is_suspended_ref("agentic-suspended:t1") and is_suspended_ref("codex-suspended:t1")
    assert not is_suspended_ref(None) and not is_suspended_ref("other:t1")


def test_wall_clock_cap_does_not_count_suspended_time():
    class Parks:
        def __init__(self, gate, ledger):
            self.gate, self.ledger = gate, ledger

        async def run(self, task, endpoint):
            self.gate.close()
            opener = asyncio.ensure_future(_open_later(self.gate, 0.3))
            await self.gate.wait_open(self.ledger)  # longer than the 0.2 s cap
            await opener
            return TrialVerdict(rewards={"reward": 1.0})

    async def go(with_ledger):
        gate = TurnGate()
        ledger = gate.register("t")
        task = _task(budget=AgentBudget(wall_clock_s=0.2))
        return await run_agent_task(task, Parks(gate, ledger), "http://e/v1", AdmissionGate(1),
                                    ledger=ledger if with_ledger else None)

    assert _run(go(True)).outcome is Outcome.OK
    lost = _run(go(False))
    assert (lost.outcome, lost.reason) == (Outcome.ABORTED, "wall_clock_cap")


def test_a_time_limit_after_a_suspension_is_aborted_not_a_zero():
    ledger = SuspendLedger("t", suspensions=1, suspended_seconds=12.5, retried_turns=1)
    verdict = TrialVerdict(rewards=None, exception_type="AgentTimeoutError")
    limit = AgentTrajectory(task=_task(), outcome=Outcome.POLICY_LIMIT, reward=0.0,
                            reason="AgentTimeoutError", verdict=verdict)
    out = after_suspension(limit, ledger)
    assert out.outcome is Outcome.ABORTED and out.reason.startswith(TIMEOUT_AFTER_SUSPENSION_REASON)
    assert out.to_metadata()["agent_metrics"] == {"suspensions": 1, "suspended_seconds": 12.5,
                                                  "suspend_retried_turns": 1}
    # never suspended: the policy's own time budget stays a trainable 0
    assert after_suspension(limit, SuspendLedger("t")) is limit
    ok = AgentTrajectory(task=_task(), outcome=Outcome.OK, reward=1.0, reason="verified",
                         verdict=TrialVerdict(rewards={"reward": 1.0}))
    assert after_suspension(ok, ledger).outcome is Outcome.OK
    # an ABORTED trajectory without a verdict (wall-clock cap, lost sandbox) keeps the counters too
    lost = AgentTrajectory(task=_task(), outcome=Outcome.ABORTED, reward=0.0, reason="wall_clock_cap", verdict=None)
    assert after_suspension(lost, ledger).to_metadata()["agent_metrics"]["suspensions"] == 1


# ------------------------------------------------------------------ gate relay


class FakeSessionServer:
    """Miles' session server: ``status_for(n)`` may answer an error to call n (1-based)."""

    def __init__(self, status_for=None, delay_for=None):
        self.bodies: list[dict] = []
        self.paths: list[str] = []
        self.status_for = status_for or (lambda n: None)
        self.delay_for = delay_for or (lambda n: 0.0)

    async def __aenter__(self):
        async def handle(request):
            body = await request.json()
            self.bodies.append(body)
            self.paths.append(request.path)
            await asyncio.sleep(self.delay_for(len(self.bodies)))
            status = self.status_for(len(self.bodies))
            if status is not None:
                return web.json_response({"error": "aborted"}, status=status)
            if body.get("stream"):
                return web.Response(body=b"data: {\"n\": %d}\n\ndata: [DONE]\n\n" % len(self.bodies),
                                    content_type="text/event-stream")
            return web.json_response({"n": len(self.bodies)})

        app = web.Application()
        app.router.add_post("/sessions/{sid}/v1/{tail:.*}", handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = f"http://127.0.0.1:{self.runner.addresses[0][1]}"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


async def _post(url, body):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body) as resp:
            return resp.status, await resp.read()


def test_relay_forwards_holds_and_retries_an_aborted_turn():
    from yeto.rl.harness.gateway.gate_relay import GateRelay

    async def go():
        gate = TurnGate()
        ledger = gate.register("t1")
        # call 2 is aborted by the engines at the cut-off (503 while the gate is closed)
        server = FakeSessionServer(lambda n: 503 if n == 2 else None, lambda n: 0.1 if n == 2 else 0.0)
        async with server:
            relay = GateRelay(gate)
            url = await relay.open(f"{server.base}/sessions/s1/v1", ledger)
            assert await _post(f"{url}/chat/completions", {"m": 1}) == (200, b'{"n": 1}')
            assert server.paths == ["/sessions/s1/v1/chat/completions"]

            async def cutoff():
                await asyncio.sleep(0.05)
                gate.close()
                await asyncio.sleep(0.2)
                gate.open()

            # the call reaches the engine just before the gate closes; its 503 is retried after resume
            call = asyncio.ensure_future(_post(f"{url}/chat/completions", {"m": 2}))
            await asyncio.gather(call, cutoff())
            assert call.result() == (200, b'{"n": 3}')
            assert ledger.retried_turns == 1
            # a call that arrives at a closed gate waits for resume
            gate.close()
            held = asyncio.ensure_future(_post(f"{url}/chat/completions", {"m": 3}))
            await asyncio.sleep(0.1)
            assert len(server.bodies) == 3 and not held.done()
            gate.open()
            assert (await held)[0] == 200 and ledger.suspensions >= 1
            relay.close(url)
            status, _ = await _post(f"{url}/chat/completions", {"m": 4})
            assert status == 404
            await relay.aclose()

    _run(go())


def test_relay_passes_an_error_through_when_nothing_was_suspended():
    from yeto.rl.harness.gateway.gate_relay import GateRelay

    async def go():
        gate = TurnGate()
        async with FakeSessionServer(lambda n: 503) as server:
            relay = GateRelay(gate)
            url = await relay.open(f"{server.base}/sessions/s1/v1")
            assert (await _post(f"{url}/chat/completions", {}))[0] == 503
            assert len(server.bodies) == 1
            await relay.aclose()

    _run(go())


def test_relay_keeps_a_held_stream_alive_then_sends_the_reply():
    from yeto.rl.harness.gateway.gate_relay import GateRelay

    async def go():
        gate = TurnGate()
        async with FakeSessionServer() as server:
            relay = GateRelay(gate, keepalive_s=0.03)
            url = await relay.open(f"{server.base}/sessions/s1/v1")
            gate.close()
            call = asyncio.ensure_future(_post(f"{url}/responses", {"stream": True}))
            await asyncio.sleep(0.15)
            gate.open()
            status, body = await call
            assert status == 200 and body.startswith(b": held\n\n")
            assert body.endswith(b"data: {\"n\": 1}\n\ndata: [DONE]\n\n")
            await relay.aclose()

    _run(go())


def test_relay_refuses_a_short_path_secret():
    from yeto.rl.harness.gateway.gate_relay import GateRelay, RelayError

    with pytest.raises(RelayError, match="at least"):
        GateRelay(TurnGate(), path_secret="short")


# ------------------------------------------------------------------ Miles agent function + hooks


def test_miles_suspend_hook_fails_closed_without_the_relay(miles_stubs, monkeypatch):
    mod, _ = miles_stubs
    monkeypatch.delenv(mod.RELAY_PORT_ENV, raising=False)
    with pytest.raises(HarnessConfigError, match=mod.RELAY_PORT_ENV):
        _run(mod.suspend(None))
    with pytest.raises(HarnessConfigError, match=mod.RELAY_PORT_ENV):
        mod.check_suspend_config({})
    with pytest.raises(HarnessConfigError, match="N, N-M or 0"):
        mod.relay_settings({mod.RELAY_PORT_ENV: "http"})
    with pytest.raises(HarnessConfigError, match="not a valid"):
        mod.relay_settings({mod.RELAY_PORT_ENV: "31010-31000"})
    with pytest.raises(HarnessConfigError, match="not a range"):
        mod.relay_settings({mod.RELAY_PORT_ENV: "31000-31003", mod.RELAY_PUBLIC_BASE_ENV: "https://t"})
    assert mod.relay_settings({mod.RELAY_PORT_ENV: "31000-31003"}).ports == (31000, 31001, 31002, 31003)
    assert mod.relay_settings({mod.RELAY_PORT_ENV: "0"}).ports == (0,)


def test_miles_trial_runs_through_the_relay_and_survives_a_round_boundary(miles_stubs, monkeypatch):
    mod, _ = miles_stubs
    monkeypatch.setenv(mod.RELAY_PORT_ENV, "0")
    monkeypatch.setenv(mod.RELAY_HOST_ENV, "127.0.0.1")
    mod._turn_gates.clear()
    mod._relays.clear()
    seen = {}

    class AgentTrial(_FakeTrial):
        """Calls the model twice; Miles' cut-off falls between the two calls."""

        @classmethod
        async def create(cls, config):
            seen["endpoint"] = config.agent.env["OPENAI_BASE_URL"]
            return cls(config)

        async def run(self):
            url = seen["endpoint"]
            await _post(f"{url}/chat/completions", {"turn": 1})
            await mod.suspend(None)  # the cut-off: a tool call is running
            seen["refs"] = suspended_references()
            await asyncio.sleep(0.05)  # the tool call returns; the next model call waits
            resume = asyncio.ensure_future(_resume_later(mod))
            await _post(f"{url}/chat/completions", {"turn": 2})
            await resume
            return _result({"reward": 1.0})

    monkeypatch.setattr(mod, "HarborHarness",
                        lambda build: __import__("yeto.rl.harness.harbor", fromlist=["x"]).HarborHarness(
                            build, trial_cls=AgentTrial))

    async def go():
        async with FakeSessionServer() as server:
            out = await mod.run(f"{server.base}/sessions/s1", metadata={"instance_id": "fix-git",
                                                                         "trajectory_id": "g7"})
            for relay in list(mod._relays.values()):
                await relay.aclose()
            mod._relays.clear()
            return out, server

    out, server = _run(go())
    assert "/r/" in seen["endpoint"] and server.paths == ["/sessions/s1/v1/chat/completions"] * 2
    assert [r["task_id"] for r in seen["refs"]] == ["fix-git"]
    assert seen["refs"][0]["trajectory_id"].startswith("g7-")
    assert (out[OUTCOME_KEY], out["reward"]) == ("ok", 1.0)
    metrics = out["agent_metrics"]
    assert metrics["suspensions"] == 1 and metrics["suspended_seconds"] > 0
    assert suspended_references() == []  # the trajectory left the gate


async def _resume_later(mod, delay=0.1):
    await asyncio.sleep(delay)
    await mod.resume(None)


def test_miles_stage_three_accepts_the_harbor_agent_and_the_island_needs_the_relay(monkeypatch):
    from yeto.rl.adapters.miles import harbor_agent, policy_age
    from yeto.rl.engine.policy_age import PolicyAgeError

    policy_age.check_task(1, custom_generate=policy_age.AGENTIC_GENERATE, custom_agent=harbor_agent.AGENT_FUNCTION)
    assert policy_age.policy_age_argv(1, agentic=True) == (
        "--agentic-suspend-between-turns", "--agentic-suspend-max-rounds", "1")
    with pytest.raises(PolicyAgeError):
        policy_age.check_task(1, custom_generate=policy_age.AGENTIC_GENERATE, custom_agent="other.agent.run")
    assert harbor_agent.check_suspend_config({harbor_agent.RELAY_PORT_ENV: "31000"}).ports == (31000,)


# ------------------------------------------------------------------ cut export / restore (restart)


def test_a_suspended_trajectory_is_a_reference_in_the_cut_and_lost_on_restore():
    from yeto.rl.adapters.miles import in_flight

    gate = TurnGate()
    gate.register("t9", task_id="fix-git")
    gate.close()
    try:
        executor = SimpleNamespace(args=SimpleNamespace(), data_source=SimpleNamespace(buffer=[]))
        exported = in_flight.export_in_flight(executor)
    finally:
        gate.open()
    entries = [e for e in exported["entries"] if e["trajectory_id"] == "t9"]
    assert entries and entries[0]["session_ref"] == SESSION_REF_PREFIX + "t9"
    assert in_flight.check_export(exported) == []
    legacy = {**entries[0], "trajectory_id": "old", "session_ref": "codex-suspended:old"}
    report = in_flight.import_in_flight(SimpleNamespace(data_source=SimpleNamespace()), [entries[0], legacy],
                                        max_policy_age=1, current_version=3)
    assert report["discarded"] == 2 and report["reasons"] == {in_flight.AGENTIC_SESSION_LOST: 2}
    assert report["resumed"] == 0


def test_inflight_save_counts_harbor_references(tmp_path):
    from yeto.rl.adapters.miles import inflight_save

    entry = {"trajectory_id": "t1", "group_id": "agentic:t1", "session_ref": SESSION_REF_PREFIX + "t1"}
    out = inflight_save.write_in_flight({"entries": [entry], "buffer_groups": 0}, tmp_path, "cut")
    assert json.loads((tmp_path / "cut.json").read_text())["entries"] == [entry]
    assert (out["agentic_entries"], out["buffered_entries"]) == (1, 0)


def test_rollout_fields_count_harbor_suspensions():
    from yeto.rl.adapters.miles import carry_over

    parked = SimpleNamespace(metadata={OUTCOME_KEY: "ok", "agent_metrics": {
        "suspensions": 1, "suspended_seconds": 4.0, "suspend_retried_turns": 1}})
    expired = SimpleNamespace(metadata={OUTCOME_KEY: "aborted", "outcome_reason":
                                        f"{TIMEOUT_AFTER_SUSPENSION_REASON}: AgentTimeoutError",
                                        "agent_metrics": {"suspensions": 2, "suspended_seconds": 6.0}})
    args = SimpleNamespace(rollout_suspend_stats={"suspended_groups": 2}, rollout_resume_stats=None)
    fields = carry_over.suspend_fields(args, [[parked, expired]])
    assert fields["suspended_groups"] == 2 and fields["suspended_trajectories"] == 2
    assert fields["suspend_expired_trajectories"] == 1 and fields["suspended_env_seconds"] == 10.0
    assert fields["suspend_retried_turns"] == 1


# ------------------------------------------------------------------ verl: aborted calls retried


class _Output(SimpleNamespace):
    pass


class AbortingClient:
    def __init__(self, aborts):
        self.aborts, self.calls = aborts, 0

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        self.calls += 1
        if self.calls <= self.aborts:
            return _Output(stop_reason="aborted", token_ids=[5], log_probs=[-0.1], extra_fields={})
        return _Output(stop_reason="completed", token_ids=[7, 2], log_probs=[-0.5, -0.1],
                       extra_fields={"global_steps": 4})


def _engine(client):
    from yeto.rl.adapters.verl.rollout_engine import VerlRolloutEngine
    from yeto.rl.engine.sampler_engine import EngineIdentity

    identity = EngineIdentity(engine="verl", engine_version="test", device_family="cpu", card_model="none",
                              precision="bf16", quant_scheme="bf16", topology="tp1")
    engine = VerlRolloutEngine(max_model_len=64, identity=identity)
    engine._client = client

    async def no_sleep(_):
        return None

    engine.sleep = no_sleep
    return engine


def test_verl_engine_retries_calls_aborted_at_a_weight_update():
    from yeto.rl.engine.sampler_engine import SamplerEngineError, SamplingParams

    params = SamplingParams(max_tokens=8).check()
    engine = _engine(AbortingClient(aborts=2))
    ledger = SuspendLedger("traj")
    engine.track("traj", ledger)
    out = _run(engine._generate([1, 2], params, [2], "traj"))
    assert out.token_ids == (7, 2) and engine.pop_steps("traj") == [4]
    assert (ledger.suspensions, ledger.retried_turns) == (1, 2) and ledger.suspended_seconds > 0
    engine.untrack("traj")
    with pytest.raises(SamplerEngineError, match="aborted"):
        _run(_engine(AbortingClient(aborts=1))._generate([1], params, [2], "other"))
    limited = _engine(AbortingClient(aborts=10))
    limited.abort_retry_s = 0.0
    limited.track("traj", SuspendLedger("traj"))
    with pytest.raises(SamplerEngineError, match="aborted"):
        _run(limited._generate([1], params, [2], "traj"))


def test_verl_abort_tracking_needs_verl_rollout_engine():
    from yeto.rl.adapters.verl import rollout_engine

    recorder = SimpleNamespace(proxy=SimpleNamespace(backend=SimpleNamespace(engine=object())))
    assert rollout_engine.track_aborts(recorder, "t", SuspendLedger("t")) is False  # not verl's engine


# ------------------------------------------------------------------ CompactionRL on the Harbor path


def test_compactionrl_is_refused_on_the_harbor_path():
    from yeto.launch.harness_env import check_compaction_launch
    from yeto.rl.adapters.miles.harbor_agent import AGENT_FUNCTION
    from yeto.rl.compaction import CompactionConfig
    from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

    args = SimpleNamespace(training_mode="rl", reward_function="yeto.rl.adapters.miles.harbor_agent.reward_func",
                           rl_algorithm_spec_json=json.dumps({"advantage": {"gae_variant": "cross_segment_per_sample"}}),
                           custom_agent_function_path=AGENT_FUNCTION)
    with pytest.raises(ValueError, match="needs compacted rollouts"):
        check_compaction_launch(args, environ={})
    compacting = SimpleNamespace(backend=SimpleNamespace(compaction=CompactionConfig(context_budget=20_000)))
    with pytest.raises(HarnessConfigError, match="CompactionRL is not supported"):
        ProxyTokenRecorder(compacting)


_CHILD = """
import asyncio, socket, sys
from yeto.rl.agentic.suspend import TurnGate
from yeto.rl.harness.gateway.gate_relay import GateRelay
async def main():
    relay = GateRelay(TurnGate(), port=range({first}, {last} + 1))
    await relay.open("http://127.0.0.1:1/v1")
    print(relay.port, flush=True)
    sys.stdin.readline()  # hold the port until the parent read every child's port
    await relay.aclose()
asyncio.run(main())
"""


def _free_range(n):
    import socket

    for start in range(39000, 60000, 97):
        socks = []
        try:
            for p in range(start, start + n):
                s = socket.socket()
                s.bind(("127.0.0.1", p))
                socks.append(s)
            return start
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    pytest.skip("no free port range")


def test_rollout_processes_share_a_port_range_and_bind_distinct_ports():
    import subprocess
    import sys

    first = _free_range(3)
    code = _CHILD.format(first=first, last=first + 2)
    procs = [subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True) for _ in range(3)]
    try:
        ports = [int(p.stdout.readline()) for p in procs]
        assert sorted(ports) == [first, first + 1, first + 2]
        extra = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, input="", timeout=60)
        assert extra.returncode != 0 and "no free relay port" in extra.stderr
    finally:
        for p in procs:
            p.communicate("\n", timeout=60)


def test_relay_port_zero_binds_any_free_port():
    from yeto.rl.harness.gateway.gate_relay import GateRelay, RelayError

    async def go():
        relay = GateRelay(TurnGate(), port=0)
        url = await relay.open("http://127.0.0.1:1/v1")
        assert relay.port and f":{relay.port}/r/" in url
        await relay.aclose()

    _run(go())
    with pytest.raises(RelayError, match="fixed relay port"):
        GateRelay(TurnGate(), port=(1, 2), public_base="https://t")
