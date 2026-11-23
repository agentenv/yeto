"""What a SIGKILLed sampler leaves behind: its engine server (the parent guard ends it)
and its trials' AgentENV microVMs (owner metadata, the next incarnation's sweep, a TTL
that follows the trial wall clock). Real local processes for the guard; fakes for
AgentENV (no gateway is contacted)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from yeto.rl.adapters.sampler_engines import server
from yeto.rl.harness.harbor import agentenv_labels as al


def _gone(pid: int, within_s: float = 30.0) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)
    return False


# -- the engine server dies with the sampler -----------------------------------------------

def test_a_sigkilled_sampler_no_longer_leaves_its_engine_server(tmp_path):
    """A stand-in sampler starts a stand-in server (and a worker the server forks) with
    popen_spawn, then is SIGKILLed: the guard ends the server's whole process group."""
    pids = tmp_path / "pids"
    server_code = textwrap.dedent(f"""
        import os, subprocess, sys, time
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pids)!r}, "w").write(f"{{os.getpid()}} {{worker.pid}}")
        time.sleep(120)
    """)
    sampler_code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path(server.__file__).parents[4])!r})
        from yeto.rl.adapters.sampler_engines.server import popen_spawn
        proc = popen_spawn([sys.executable, "-c", {server_code!r}, "vllm", "--port", "39999"], {{}})
        print(proc.pid, flush=True)
        time.sleep(120)
    """)
    sampler = subprocess.Popen([sys.executable, "-c", sampler_code], stdout=subprocess.PIPE, text=True)
    guard = int(sampler.stdout.readline())
    for _ in range(100):
        if pids.exists() and pids.read_text().strip():
            break
        time.sleep(0.1)
    srv, worker = map(int, pids.read_text().split())
    assert os.getpgid(srv) == guard and os.getpgid(worker) == guard  # one group, led by the guard
    # the reap-on-restart check still recognises the group leader as the engine server
    assert server.is_engine_server(server.process_cmdline(guard), "vllm", 39999)
    sampler.kill()  # SIGKILL: no handler, no teardown
    sampler.wait()
    try:
        assert _gone(srv) and _gone(worker) and _gone(guard)
    finally:
        for pid in (srv, worker, guard):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def test_the_guard_forwards_sigterm_and_mirrors_the_exit_status():
    trap = ("import signal, sys, time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(3)); "
            "print('ready', flush=True); time.sleep(60)")
    proc = server.popen_spawn([sys.executable, "-c", trap], {})
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)  # what ServerEngine.stop() sends
    assert proc.wait(timeout=30) == 3
    plain = server.popen_spawn([sys.executable, "-c", "import time; time.sleep(60)"], {})
    time.sleep(1.0)
    plain.send_signal(signal.SIGTERM)
    assert plain.wait(timeout=30) == -signal.SIGTERM  # a server ended by a signal: the guard reports it too
    assert server.popen_spawn([sys.executable, "-c", "raise SystemExit(5)"], {}).wait(timeout=30) == 5


def test_the_guard_never_signals_a_group_it_does_not_lead(tmp_path):
    """Run without a session of its own (a test, a mistaken caller): only its child is ended."""
    child_pid = tmp_path / "child"
    code = f"import os, time; open({str(child_pid)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, {str(server.PARENT_GUARD)!r}, "--parent", "1", "--poll-s", "0.2",
                          "--grace-s", "2", "--", sys.executable, "-c", {code!r}])
        time.sleep(60)
    """)])
    try:
        for _ in range(100):
            if child_pid.exists() and child_pid.read_text():
                break
            time.sleep(0.1)
        # --parent 1 is not the guard's parent: it ends its child at once, and only that
        assert _gone(int(child_pid.read_text()))
        assert parent.poll() is None  # the caller (same process group) is untouched
    finally:
        parent.kill()
        parent.wait()


# -- AgentENV microVM owner metadata and the sweep -------------------------------------------

ENV = {"YETO_SANDBOX_RUN_LABEL": "run-a", "YETO_SANDBOX_SHARD": "run-a-1", "YETO_SAMPLER_INCARNATION": "inc-new"}


def test_owner_metadata_from_the_sampler_env():
    assert al.owner_metadata(ENV) == {"yeto_run": "run-a", "yeto_shard": "run-a-1", "yeto_inc": "inc-new"}
    env = {"YETO_NC_RUN_PREFIX": "runs-b/", "YETO_SANDBOX_SHARD": "s-0", "YETO_SAMPLER_INCARNATION": "i"}
    assert al.owner_metadata(env)["yeto_run"] == "runs-b"  # the pool's run prefix when no run label is set
    assert al.owner_metadata({"YETO_SANDBOX_SHARD": "s-0"}) == {}  # no run: no owner, nothing swept


def test_the_sweep_deletes_only_this_shards_earlier_incarnations():
    rows = [
        al.Listed("old-1", {"yeto_run": "run-a", "yeto_shard": "run-a-1", "yeto_inc": "inc-old"}),
        al.Listed("old-2", {"yeto_run": "run-a", "yeto_shard": "run-a-1", "yeto_inc": "inc-older"}),
        al.Listed("mine", {"yeto_run": "run-a", "yeto_shard": "run-a-1", "yeto_inc": "inc-new"}),
        al.Listed("other-shard", {"yeto_run": "run-a", "yeto_shard": "run-a-0", "yeto_inc": "inc-old"}),
        al.Listed("other-run", {"yeto_run": "run-b", "yeto_shard": "run-a-1", "yeto_inc": "inc-old"}),
        al.Listed("no-inc", {"yeto_run": "run-a", "yeto_shard": "run-a-1"}),
        al.Listed("colleague", {"owner": "someone"}),
        al.Listed("bare", {}),
    ]

    class Client:
        def __init__(self):
            self.deleted, self.queries = [], []

        def list_sandboxes(self, metadata):
            self.queries.append(metadata)
            return rows  # a gateway that ignores the filter: every row is checked here

        def delete(self, sandbox_id):
            self.deleted.append(sandbox_id)

    client, events = Client(), []
    owner = al.owner_from_env(ENV)
    assert al.sweep_stale(client, owner, emit=lambda e, **f: events.append((e, f))) == ["old-1", "old-2"]
    assert client.deleted == ["old-1", "old-2"]
    assert client.queries == [{"yeto_run": "run-a", "yeto_shard": "run-a-1"}]
    assert events[0][0] == "rl_sandbox_orphans_removed" and events[0][1]["removed"] == 2
    # no incarnation of our own: nothing can be told apart, nothing is deleted
    assert al.sweep_stale(Client(), al.Owner("run-a", "run-a-1", "")) == []


def test_a_failed_list_or_delete_never_raises():
    class Broken:
        def list_sandboxes(self, metadata):
            raise RuntimeError("gateway down")

    assert al.sweep_stale(Broken(), al.owner_from_env(ENV)) == []

    class FailingDelete:
        def list_sandboxes(self, metadata):
            return [al.Listed("old", {"yeto_run": "run-a", "yeto_shard": "run-a-1", "yeto_inc": "x"})]

        def delete(self, sandbox_id):
            raise RuntimeError("409")

    assert al.sweep_stale(FailingDelete(), al.owner_from_env(ENV)) == []


def test_parse_listing_keeps_only_rows_with_an_id_and_metadata():
    rows = al.parse_listing([{"sandboxID": "a", "metadata": {"yeto_run": "r"}, "state": "running"},
                             {"sandboxID": "b"}, {"metadata": {}}, "junk"])
    assert [r.sandbox_id for r in rows] == ["a"]
    assert [r.sandbox_id for r in al.parse_listing({"sandboxes": [{"id": "c", "metadata": {}}]})] == ["c"]


def test_the_client_lists_with_the_metadata_filter(monkeypatch):
    from tests.test_agentenv_secure import _client, fake_e2b  # noqa: F401
    import sys as _sys
    import types

    e2b = types.ModuleType("e2b")
    e2b.Sandbox = object
    monkeypatch.setitem(_sys.modules, "e2b", e2b)
    seen = {}

    def get(url, *, params, headers, timeout, trust_env):
        seen.update(url=url, params=params, headers=headers, trust_env=trust_env)
        return httpx.Response(200, json=[{"sandboxID": "x", "metadata": {"yeto_run": "run-a"}}],
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    rows = _client().list_sandboxes({"yeto_run": "run-a", "yeto_shard": "run-a-1"})
    assert seen["url"] == "http://gw.test/sandboxes" and seen["trust_env"] is False
    assert seen["params"] == {"metadata": "yeto_run=run-a&yeto_shard=run-a-1"}
    assert seen["headers"]["X-API-Key"] and [r.sandbox_id for r in rows] == ["x"]


def test_a_created_microvm_carries_the_owner(monkeypatch):
    from yeto.rl.harness.harbor.agentenv_sandbox import AgentenvSandbox

    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    box = AgentenvSandbox(client=None, quota=None, name="trial-1")
    assert box._metadata() == {"yeto_name": "trial-1", "yeto_run": "run-a", "yeto_shard": "run-a-1",
                               "yeto_inc": "inc-new"}


def test_the_sampler_sweeps_only_on_an_agentenv_spec(monkeypatch):
    from yeto.rl.engine import engine_sampler_entry as e
    from yeto.rl.harness.harbor import agentenv_client as ac

    called = []
    monkeypatch.setattr(ac, "process_client", lambda: called.append("client") or object())
    monkeypatch.setattr(al, "sweep_stale", lambda client, owner, emit=None: called.append(owner) or ["old"])
    spec = {"agentic": {"task_defaults": {"provider": "agentenv"}}}
    assert e.sweep_agentenv_orphans(spec, dict(ENV), lambda *a, **k: None) == ["old"]
    assert called[-1] == al.Owner("run-a", "run-a-1", "inc-new")
    called.clear()
    assert e.sweep_agentenv_orphans({"agentic": {"task_defaults": {"provider": "remote-docker"}}}, dict(ENV),
                                    lambda *a, **k: None) == []
    assert e.sweep_agentenv_orphans(spec, {"YETO_SANDBOX_SHARD": "s"}, lambda *a, **k: None) == []
    assert called == []  # no client is even built without an agentenv spec and an owner


# -- the microVM TTL follows the trial wall clock ---------------------------------------------

def test_the_agentenv_ttl_is_the_wall_clock_plus_a_margin():
    from yeto.rl.harness.harbor.trial import AGENTENV_TTL_MARGIN_S, agentenv_ttl_s

    assert agentenv_ttl_s(3900) == 3900 + AGENTENV_TTL_MARGIN_S < 3 * 3600
    assert agentenv_ttl_s(7200.0) == 7200 + AGENTENV_TTL_MARGIN_S


def test_the_trial_config_passes_the_ttl_to_an_agentenv_environment(tmp_path):
    pytest.importorskip("harbor.models.trial.config", exc_type=ImportError)
    from yeto.rl.agentic.types import AgentBudget, AgentTask
    from yeto.rl.harness.harbor.trial import HarborSettings, agentenv_ttl_s, build_trial_config

    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "task.toml").write_text("[task]\n")
    task = AgentTask(task_id="t1", provider="agentenv", agent="pi", model_name="m",
                     budget=AgentBudget(agent_timeout_s=3600, wall_clock_s=3900))
    settings = HarborSettings(tasks_dir=tmp_path, turn_max_tokens=512)
    cfg = build_trial_config(task, "http://127.0.0.1:1/v1", settings)
    assert cfg.environment.kwargs["ttl_s"] == agentenv_ttl_s(3900)
    pinned = build_trial_config(task, "http://127.0.0.1:1/v1",
                                HarborSettings(tasks_dir=tmp_path, provider_kwargs={"ttl_s": 600},
                                               turn_max_tokens=512))
    assert pinned.environment.kwargs["ttl_s"] == 600  # an operator's HARBOR_ENV_KWARGS wins
