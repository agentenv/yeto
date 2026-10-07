"""G1 (real TB2 EnvironmentProvider, CPU backend) and G6(a) (tito counter mirroring).

The ``LocalProcessBackend`` stands in for a Modal sandbox: commands run in a
scratch root on this host, ``/tests`` and ``/logs/verifier`` are redirected
through ``TB2_TESTS_DIR`` / ``TB2_VERIFIER_LOGS_DIR``.  Everything above the
backend (relay wire, verifier, lease, faults, trusted-side counters) is the
production code path.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

import test_secrlenv_codex_harness as legacy_tests
from yeto.rl import tbench_outcome
from yeto.rl.engine.tool_wait import HarnessBoard
from yeto.rl.harness.codex import (
    codex_harness_agent as harness,
    codex_openenv_agent_function as adapter,
    codex_openenv_subprocess_agent_function as subprocess_agent,
    preflight,
    tb2_provider,
    tbench_reward,
)
from yeto.rl.harness.codex.environment import HttpTerminalEnvironment

KEY = "k" * 48

TEST_SH = textwrap.dedent(
    """\
    #!/bin/bash
    L=${TB2_VERIFIER_LOGS_DIR:-/logs/verifier}
    mkdir -p "$L"
    if [ -f fixed.txt ] && grep -q ok fixed.txt; then
      echo 1 > "$L/reward.txt"; exit 0
    else
      echo 0 > "$L/reward.txt"; exit 1
    fi
    """
)


def _make_task(root: Path, task_id: str = "fix-git") -> Path:
    task_dir = root / task_id
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "task.toml").write_text(
        '[environment]\ndocker_image = "alexgshaw/fix-git:20251031"\ncpus = 1\nmemory_mb = 2048\n'
        "[agent]\ntimeout_sec = 900.0\n[verifier]\ntimeout_sec = 60.0\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
    (task_dir / "tests" / "test.sh").write_text(TEST_SH)
    (task_dir / "tests" / "test_outputs.py").write_text("# fixture\n")
    return root


def _run(coro):
    return asyncio.run(coro)


def _provider(tmp_path: Path, **kw: Any) -> tb2_provider.Tb2EnvironmentProvider:
    tasks_dir = _make_task(tmp_path / "tb2")
    backend = tb2_provider.LocalProcessBackend(tmp_path / "sandboxes")
    return tb2_provider.Tb2EnvironmentProvider(backend, tasks_dir, **kw)


def _configure(monkeypatch, provider, harness_board: Any = None) -> None:
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", KEY)
    monkeypatch.setenv("YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER", "1")
    subprocess_agent.configure(provider=provider, tool_wait_board=None, harness_board=harness_board, member="m0")


def _metadata(script: list[str], **extra) -> dict[str, Any]:
    return {"task_id": "fix-git", "trajectory_id": "traj-1", "prompt": "fix the repo",
            "codex_openenv_driver": "scripted", "script": script, "expected_policy_version": "pv-7", **extra}


# ---------------------------------------------------------------- task resolution / faults


def test_resolve_task_reads_task_toml_and_dockerfile_workdir(tmp_path):
    tasks_dir = _make_task(tmp_path)
    task = tb2_provider.resolve_task("fix-git", tasks_dir)
    assert task.docker_image == "alexgshaw/fix-git:20251031" and task.workdir == "/app"
    assert task.cpus == 1 and task.memory_mb == 2048 and task.verifier_timeout_s == 60.0
    with pytest.raises(FileNotFoundError):
        tb2_provider.resolve_task("nope", tasks_dir)
    with pytest.raises(ValueError):
        tb2_provider.resolve_task("../fix-git", tasks_dir)


def test_parse_faults_accepts_matrix_entries_and_rejects_unknown_kinds():
    faults = tb2_provider.parse_faults("create_fail:0, deadline:traj-9=2, max_turns:2=2")
    assert [f.kind for f in faults] == ["create_fail", "deadline", "max_turns"]
    assert faults[0].matches(0, "x") and not faults[0].matches(1, "x")
    assert faults[1].matches(5, "traj-9") and faults[1].value == "2"
    assert tb2_provider.parse_faults("") == ()
    with pytest.raises(ValueError, match="invalid fault"):
        tb2_provider.parse_faults("explode:0")
    with pytest.raises(ValueError, match="selector"):
        tb2_provider.parse_faults("create_fail:")


# ---------------------------------------------------------------- lease lifecycle on the wire


def test_lease_relays_execute_submit_verifies_and_is_gone_after_destroy(tmp_path):
    provider = _provider(tmp_path)

    async def scenario() -> None:
        lease = await provider.acquire("fix-git", "traj-1")
        assert lease.deadline_seconds == 900.0 and await lease.describe() == "live"
        async with HttpTerminalEnvironment(lease.env_url, lease.env_token) as env:
            out = await env.execute("ep-1", "echo hello; pwd; exit 3", timeout_seconds=10, output_bytes=4096)
            assert out["exit_code"] == 3 and "hello" in out["output"] and out["output"].rstrip().endswith("/app")
            assert out["timed_out"] is False and out["truncated"] is False
            big = await env.execute("ep-1", "head -c 100 /dev/zero | tr '\\0' x", timeout_seconds=10, output_bytes=16)
            assert big["truncated"] is True and len(big["output"]) == 16
            slow = await env.execute("ep-1", "sleep 5", timeout_seconds=1, output_bytes=64)
            assert slow["timed_out"] is True
            # the verifier is not reachable before evaluate: /tests is empty
            probe = await env.execute("ep-1", "ls $TB2_TESTS_DIR 2>/dev/null | wc -l", timeout_seconds=10, output_bytes=64)
            assert probe["output"].strip() == "0"
            unknown = HttpTerminalEnvironment(lease.env_url, "wrong")
            async with unknown:
                with pytest.raises(Exception):
                    await unknown.execute("ep-1", "true", timeout_seconds=1, output_bytes=1)
            assert (await env.execute("ep-1", "echo ok > fixed.txt", timeout_seconds=10, output_bytes=64))["exit_code"] == 0
            assert (await env.submit("ep-1", {"evidence": "done"})) == {"accepted": True}
        evaluation = await lease.verifier.evaluate("ep-1")
        assert evaluation == {"passed": True, "testsh_rc": 0, "timed_out": False}
        assert lease.environment.commands[0].startswith("echo hello") and lease.environment.submitted
        await lease.destroy()
        assert await lease.describe() == "gone" and provider.destroyed == 1 and provider.live == {}
        await lease.destroy()  # idempotent
        assert provider.destroyed == 1

    _run(scenario())


def test_verifier_reports_failure_and_binds_to_its_episode(tmp_path):
    provider = _provider(tmp_path)

    async def scenario() -> None:
        lease = await provider.acquire("fix-git", "traj-2")
        evaluation = await lease.verifier.evaluate("ep-1")  # unbound: no command ran (e.g. timeout path)
        assert evaluation["passed"] is False and evaluation["testsh_rc"] == 1
        async with HttpTerminalEnvironment(lease.env_url, lease.env_token) as env:
            await env.execute("ep-1", "true", timeout_seconds=5, output_bytes=16)
            with pytest.raises(Exception):  # a second episode on the same sandbox is refused
                await env.execute("ep-2", "true", timeout_seconds=5, output_bytes=16)
        with pytest.raises(ValueError):
            await lease.verifier.evaluate("other-episode")
        await lease.destroy()

    _run(scenario())


# ---------------------------------------------------------------- full subprocess path (R-TB matrix)


def test_subprocess_run_positive_and_negative_rewards_through_the_real_relay(monkeypatch, tmp_path):
    hb = HarnessBoard()
    provider = _provider(tmp_path)
    _configure(monkeypatch, provider, harness_board=hb)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(["ls", "echo ok > fixed.txt"])))
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert reward == 1.0 and outcome["status"] == "completed" and outcome["verifier"] == tbench_outcome.TEST_SH_VERIFIER
    assert result["chains_total"] == 1 and result["chain_break_reason"] is None
    assert provider.destroyed == 1 and provider.live == {}
    snap = hb.snapshot()
    assert snap.env_live == 0 and snap.in_flight == 0 and snap.tito_session_mismatch == 0

    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(["ls", "echo nope > fixed.txt"], trajectory_id="traj-neg")))
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert reward == 0.0 and outcome["status"] == "completed" and tbench_outcome.MAC_KEY in result
    assert provider.destroyed == 2 and hb.snapshot().env_live == 0


def test_create_fail_fault_is_infrastructure_aborted_and_touches_no_sandbox(monkeypatch, tmp_path):
    hb = HarnessBoard()
    provider = _provider(tmp_path, faults=tb2_provider.parse_faults("create_fail:traj-1"))
    _configure(monkeypatch, provider, harness_board=hb)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(["ls"])))
    assert tbench_reward.INFRASTRUCTURE_KEY in result and tbench_outcome.MAC_KEY not in result
    assert "InjectedCreateFailure" in result[tbench_reward.INFRASTRUCTURE_KEY]
    sample = SimpleNamespace(metadata=result, status=None)
    assert _run(tbench_reward.reward_func(None, sample)) == 0.0 and sample.status == "ABORTED"
    assert provider.backend.created == [] and provider.fault_log == [("create_fail", "traj-1", 0)]
    assert hb.snapshot().env_live == 0 and hb.snapshot().in_flight == 0
    # ordinal selectors: the second acquire is unaffected
    ok = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(["echo ok > fixed.txt"], trajectory_id="traj-2")))
    assert tbench_outcome.verified_outcome(ok)[1] == 1.0


def test_deadline_fault_cancels_the_episode_destroys_and_drains_env_live(monkeypatch, tmp_path):
    hb = HarnessBoard()
    provider = _provider(tmp_path, faults=tb2_provider.parse_faults("deadline:0=1"))
    _configure(monkeypatch, provider, harness_board=hb)
    seen: list[int] = []

    async def scenario() -> dict[str, Any]:
        task = asyncio.create_task(subprocess_agent.run("http://miles", "p", {}, _metadata(["sleep 30"])))
        while not hb.snapshot().env_live:
            await asyncio.sleep(0.02)
        seen.append(hb.snapshot().env_live)
        return await task

    result = _run(scenario())
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert seen == [1] and reward == 0.0 and outcome["status"] == "timeout"
    assert outcome["verifier"] == tbench_outcome.TIMEOUT_VERIFIER
    assert provider.destroyed == 1 and not provider.backend.created[0].alive()
    snap = hb.snapshot()
    assert snap.env_live == 0 and snap.in_flight == 0 and snap.leases_expired_total == 0


def test_max_turns_fault_sets_the_worker_turn_budget(monkeypatch, tmp_path):
    provider = _provider(tmp_path, faults=tb2_provider.parse_faults("max_turns:traj-1=2"))
    _configure(monkeypatch, provider)
    captured: dict[str, Any] = {}
    original = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(["echo ok > fixed.txt"])))
    assert tbench_outcome.verified_outcome(result)[1] == 1.0
    assert captured["env"][tb2_provider.TURN_BUDGET_ENV] == "2"
    assert "TBENCH_REWARD_HMAC_KEY" not in captured["env"]
    # the budget is the bridge's signed turn limit
    monkeypatch.setenv(tb2_provider.TURN_BUDGET_ENV, "2")
    legacy_tests._set_bridge_env(monkeypatch)
    bridge = harness._ResponsesBridge("http://miles", "p", {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128, "stream": True, "n": 1},
                                      harness.legacy.AgentMetrics(), max_seq_len=512)
    assert bridge._max_turns == 2


# ---------------------------------------------------------------- factories (module:callable)


def test_local_provider_factory_reads_env_and_resolves_through_preflight(monkeypatch, tmp_path):
    tasks_dir = _make_task(tmp_path / "tb2")
    monkeypatch.delenv(tb2_provider.TASKS_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=tb2_provider.TASKS_DIR_ENV):
        tb2_provider.local_provider(None)
    monkeypatch.setenv(tb2_provider.TASKS_DIR_ENV, str(tasks_dir))
    monkeypatch.setenv(tb2_provider.FAULT_ENV, "create_fail:3")
    monkeypatch.setenv(tb2_provider.LEASE_SECONDS_ENV, "120")
    provider = preflight.resolve_environment_provider(
        SimpleNamespace(yeto_harness_environment_provider=None),
        {preflight.ENVIRONMENT_PROVIDER_ENV: "yeto.rl.harness.codex.tb2_provider:local_provider"},
    )
    assert isinstance(provider, tb2_provider.Tb2EnvironmentProvider)
    assert provider.lease_seconds == 120.0 and provider.faults[0].kind == "create_fail"
    monkeypatch.setattr(tb2_provider, "require_modal_client", lambda: None)  # no client in the test venv
    modal = tb2_provider.modal_provider(None)
    assert isinstance(modal.backend, tb2_provider.ModalSandboxBackend) and modal.backend.app_name == tb2_provider.DEFAULT_MODAL_APP


def test_modal_backend_maps_task_to_sandbox_create(monkeypatch, tmp_path):
    import sys
    import types

    calls: dict[str, Any] = {}

    class _Sandbox:
        object_id = "sb-1"

        def __init__(self) -> None:
            self.polls = 0

        def exec(self, *args, **kwargs):
            calls["exec"] = (args, kwargs)
            return SimpleNamespace(stdout=SimpleNamespace(read=lambda: "out\n"), stderr=SimpleNamespace(read=lambda: ""), wait=lambda: 0)

        def terminate(self):
            calls["terminated"] = True

        def poll(self):
            return None if "terminated" not in calls else 0

        @classmethod
        def create(cls, *args, **kwargs):
            calls["create"] = (args, kwargs)
            return cls()

    fake = types.ModuleType("modal")
    fake.Sandbox = _Sandbox
    fake.Image = SimpleNamespace(from_registry=lambda tag: ("image", tag))
    fake.App = SimpleNamespace(lookup=lambda name, create_if_missing: ("app", name))
    monkeypatch.setitem(sys.modules, "modal", fake)

    task = tb2_provider.resolve_task("fix-git", _make_task(tmp_path))
    backend = tb2_provider.ModalSandboxBackend(app_name="t", ttl_s=600, idle_timeout_s=60, run_id="r1")
    handle = backend.create(task, "traj-1")
    args, kwargs = calls["create"]
    assert args == ("sleep", "infinity") and kwargs["image"] == ("image", "alexgshaw/fix-git:20251031")
    assert kwargs["app"] == ("app", "t") and kwargs["timeout"] == 600 and kwargs["idle_timeout"] == 60
    assert kwargs["cpu"] == 1.0 and kwargs["memory"] == 2048 and kwargs["workdir"] == "/app"
    assert kwargs["tags"] == {"yeto-tb2-task": "fix-git", "yeto-trajectory": "traj-1", "yeto-run-id": "r1"}
    result = handle.exec("echo out", timeout_s=5)
    assert result.exit_code == 0 and result.output == "out\n" and calls["exec"][1]["workdir"] == "/app"
    assert handle.alive()
    handle.terminate()
    assert not handle.alive()


# ---------------------------------------------------------------- G6(a): tito counters


def test_bridge_counts_session_mismatch_and_retry_fork(monkeypatch):
    legacy_tests._set_bridge_env(monkeypatch)

    async def scenario() -> None:
        runner, miles_url, _requests = await legacy_tests._fake_miles([
            legacy_tests._completion("terminal.exec", '{ "command": "git status" }', "call-1", "look", 25),
        ])
        metrics = harness.legacy.AgentMetrics()
        try:
            async with harness._ResponsesBridge(
                miles_url, "fix the repo", {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128, "stream": True, "n": 1},
                metrics, max_seq_len=512,
            ) as bridge:
                headers = {"Authorization": f"Bearer {bridge.token}"}
                initial = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix the repo"}]}]
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{bridge.url}/v1/responses", json=legacy_tests._codex_body(initial), headers=headers) as r:
                        assert r.status == 200
                    assert harness.tito_counters(metrics) == {"tito_session_mismatch": 0, "tito_chain_breaks": {}}
                    # byte-identical retry -> retry_fork chain break
                    async with session.post(f"{bridge.url}/v1/responses", json=legacy_tests._codex_body(initial), headers=headers) as r:
                        assert r.status == 400
                    mutated = copy.deepcopy(initial)
                    mutated[0]["content"][0]["text"] = "fix the repo!"
                    async with session.post(f"{bridge.url}/v1/responses", json=legacy_tests._codex_body(mutated), headers=headers) as r:
                        assert r.status == 400
        finally:
            await runner.cleanup()
        assert harness.tito_counters(metrics) == {"tito_session_mismatch": 1, "tito_chain_breaks": {"retry_fork": 1}}
        assert adapter._metrics_dict(metrics)["tito_chain_breaks"] == {"retry_fork": 1}

    _run(scenario())


def test_mirror_tito_counters_uses_gateway_board_semantics():
    hb = HarnessBoard()
    fields = adapter.mirror_tito_counters({"tito_session_mismatch": 2, "tito_chain_breaks": {"retry_fork": 1, "history_rewrite": 0}}, hb)
    assert fields == {"chain_break_reason": "retry_fork"}
    snap = hb.snapshot()
    assert snap.tito_session_mismatch == 2 and snap.tito_chain_breaks == {"retry_fork": 1}
    assert adapter.mirror_tito_counters(None, hb) == {"chain_break_reason": None}
    assert adapter.mirror_tito_counters({"tito_session_mismatch": 1}, None) == {"chain_break_reason": None}
    with pytest.raises(ValueError):
        adapter.mirror_tito_counters({"tito_chain_breaks": {"made_up": 1}}, hb)


def test_mirror_tito_counters_awaits_remote_board_calls_in_order(monkeypatch, capsys):
    """S14/A19 (FINAL-REPORT-S7 §6.1, A-T3-7 leftovers): record_session_mismatch /
    record_chain_break on an actor board were fire-and-forget; each call is now
    resolved in issue order, and a remote failure never fails the trajectory."""

    class _Ref:  # stands in for ray.ObjectRef
        def __init__(self, value):
            self.value = value

    _Ref.__name__ = "ObjectRef"
    order: list[tuple[str, tuple[Any, ...]]] = []
    resolved: list[Any] = []
    monkeypatch.setattr("yeto.rl.engine.tool_wait._resolve", lambda ref: resolved.append(ref) or ref.value)

    class _RemoteBoard:
        def __init__(self):
            self.record_session_mismatch = SimpleNamespace(
                remote=lambda *a: order.append(("record_session_mismatch", a)) or _Ref(None))
            self.record_chain_break = SimpleNamespace(
                remote=lambda *a: order.append(("record_chain_break", a)) or _Ref(None))

    metrics = {"tito_session_mismatch": 2, "tito_chain_breaks": {"retry_fork": 1, "history_rewrite": 3}}
    assert adapter.mirror_tito_counters(metrics, _RemoteBoard()) == {"chain_break_reason": "retry_fork"}
    assert order == [("record_session_mismatch", (2,)), ("record_chain_break", ("retry_fork", 1)),
                     ("record_chain_break", ("history_rewrite", 3))]
    assert len(resolved) == len(order)  # every remote call was awaited, in issue order

    class _FailingBoard:
        def __init__(self):
            self.record_session_mismatch = SimpleNamespace(
                remote=lambda *a: (_ for _ in ()).throw(RuntimeError("board down")))
            self.record_chain_break = SimpleNamespace(
                remote=lambda *a: (_ for _ in ()).throw(RuntimeError("board down")))

    assert adapter.mirror_tito_counters(metrics, _FailingBoard()) == {"chain_break_reason": "retry_fork"}
    assert "harness board record_session_mismatch failed: RuntimeError: board down" in capsys.readouterr().err


def test_subprocess_rejection_counters_reach_the_board_and_abort(monkeypatch, tmp_path):
    hb = HarnessBoard()
    provider = _provider(tmp_path)
    _configure(monkeypatch, provider, harness_board=hb)

    async def rejected(job, trajectory_id, board, worker_env=None):
        error = harness.CodexHarnessError("Codex truncated or mutated episode history")
        error.metrics = {"turns": 1, "tito_session_mismatch": 1, "tito_chain_breaks": {"retry_fork": 1}}
        raise error

    monkeypatch.setattr(subprocess_agent, "_drive_worker", rejected)
    result = _run(subprocess_agent.run("http://miles", "p", {}, _metadata(["ls"])))
    assert tbench_reward.INFRASTRUCTURE_KEY in result and tbench_outcome.MAC_KEY not in result
    assert result["chains_total"] == 1 and result["chain_break_reason"] == "retry_fork"
    assert result["agent_metrics"]["tito_session_mismatch"] == 1
    sample = SimpleNamespace(metadata=result, status=None)
    assert _run(tbench_reward.reward_func(None, sample)) == 0.0 and sample.status == "ABORTED"
    snap = hb.snapshot()
    assert snap.tito_session_mismatch == 1 and snap.tito_chain_breaks == {"retry_fork": 1}
    assert snap.env_live == 0 and snap.in_flight == 0 and provider.destroyed == 1


def test_worker_error_event_carries_metrics(monkeypatch):
    import subprocess
    import sys

    monkeypatch.setenv("YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER", "1")

    async def boom(job, env, tool_wait=None):
        error = harness.CodexHarnessError("Codex retried a Responses sample")
        error.metrics = {"tito_chain_breaks": {"retry_fork": 1}}
        raise error

    code = (
        "import asyncio, json, sys\n"
        "from yeto.rl.harness.codex import codex_openenv_agent_worker as w, codex_openenv_agent_function as a\n"
        "async def boom(job, env, tool_wait=None):\n"
        "    e = a.harness.CodexHarnessError('Codex retried a Responses sample'); e.metrics = {'tito_chain_breaks': {'retry_fork': 1}}; raise e\n"
        "a.drive_untrusted = boom\n"
        "sys.exit(w.main())\n"
    )
    job = {"env_url": "http://127.0.0.1:9", "env_token": "t", "episode_id": "e", "base_url": "b", "prompt": "p", "request_kwargs": {}}
    proc = subprocess.run([sys.executable, "-c", code], input=json.dumps(job) + "\n", capture_output=True, text=True,
                          env={k: v for k, v in os.environ.items() if "HMAC" not in k})
    event = json.loads(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 1 and event["event"] == "error" and event["metrics"] == {"tito_chain_breaks": {"retry_fork": 1}}
    del boom


def test_modal_provider_fails_closed_without_an_importable_modal_client(monkeypatch):
    """A-T3-5 (codex-smoke-20261003-7): the ports image has no Modal client; the
    driver preflight must refuse instead of every rollout worker failing at acquire."""
    import sys

    monkeypatch.setitem(sys.modules, "modal", None)  # import raises ImportError
    with pytest.raises(RuntimeError, match="importable `modal` client"):
        tb2_provider.modal_provider(None)
    assert tb2_provider.InjectedCreateFailure.injected_fault is True
