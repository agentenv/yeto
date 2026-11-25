"""A stopped Modal pool leaves no AgentENV microVMs (P5b-15), and an infrastructure retry
says why it failed (P5b-16). Fakes only: no gateway, no Modal, no GPU."""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx

from yeto.launch.pool import modal_provider as mp
from yeto.rl.harness.harbor import agentenv_labels as al


def _spec(tmp_path: Path, provider: str = "agentenv") -> dict:
    return {"event_tape": str(tmp_path / "tape" / "events.jsonl"),
            "agentic": {"max_in_flight": 4, "proxy": {"host": "127.0.0.1", "port": 18080},
                        "task_defaults": {"provider": provider}}}


def _payload(tmp_path: Path, provider: str = "agentenv", env: dict | None = None) -> dict:
    return {"spec": _spec(tmp_path, provider), "shard": "p5gs3-0", "run_prefix": "fleet-g3/", "bucket": "b",
            "env": dict(env or {})}


# -- the pool labels its microVMs with its run ---------------------------------------------

def test_an_agentenv_pool_without_a_run_label_labels_its_microvms_with_its_run_prefix(tmp_path):
    env = mp.agentic_payload(_payload(tmp_path), {"tasks_dir": "/t"})["env"]
    assert env["YETO_SANDBOX_RUN_LABEL"] == "fleet-g3" and env["HARBOR_ENV_TYPE"] == "agentenv"
    # the sampler's owner then exists (the shard and incarnation come from the sampler itself)
    owner = al.owner_from_env({**env, "YETO_SANDBOX_SHARD": "p5gs3-0", "YETO_SAMPLER_INCARNATION": "i1"})
    assert owner == al.Owner("fleet-g3", "p5gs3-0", "i1")


def test_a_label_the_operator_set_wins_and_other_providers_are_unchanged(tmp_path):
    own = mp.agentic_payload(_payload(tmp_path, env={"YETO_SANDBOX_RUN_LABEL": "mine"}), {})["env"]
    assert own["YETO_SANDBOX_RUN_LABEL"] == "mine"
    nc = mp.agentic_payload(_payload(tmp_path, env={"YETO_NC_RUN_PREFIX": "runs/x"}), {})["env"]
    assert "YETO_SANDBOX_RUN_LABEL" not in nc
    for provider in ("boat", "remote-docker"):
        env = mp.agentic_payload(_payload(tmp_path, provider), {})["env"]
        assert "YETO_SANDBOX_RUN_LABEL" not in env, provider


# -- the sweeps select only what they own ---------------------------------------------------

ROWS = [
    al.Listed("call-a", {"yeto_run": "fleet-g3", "yeto_shard": "p5gs3-0", "yeto_inc": "i1"}),
    al.Listed("call-b", {"yeto_run": "fleet-g3", "yeto_shard": "p5gs3-0", "yeto_inc": "i2"}),
    al.Listed("successor", {"yeto_run": "fleet-g3", "yeto_shard": "p5gs3-0", "yeto_inc": "i9"}),
    al.Listed("pool-shard-1", {"yeto_run": "fleet-g3", "yeto_shard": "p5gs3-1", "yeto_inc": "i5"}),
    al.Listed("other-pool", {"yeto_run": "fleet-g3", "yeto_shard": "p5gv3-0", "yeto_inc": "i1"}),
    al.Listed("prefix-pool", {"yeto_run": "fleet-g3", "yeto_shard": "p5gs3x-0", "yeto_inc": "i1"}),
    al.Listed("other-run", {"yeto_run": "fleet-g4", "yeto_shard": "p5gs3-0", "yeto_inc": "i1"}),
    al.Listed("colleague", {"owner": "someone"}),
]


class Gateway:
    def __init__(self, rows=ROWS):
        self.rows, self.deleted, self.queries = list(rows), [], []

    def list_sandboxes(self, metadata):
        self.queries.append(metadata)
        return self.rows  # a gateway that ignores the filter: every row is checked again

    def delete(self, sandbox_id):
        self.deleted.append(sandbox_id)


def test_a_call_releases_only_its_shards_microvms_of_its_own_incarnations():
    gw = Gateway()
    assert al.release_incarnations(gw, "fleet-g3", "p5gs3-0", ["i1", "i2"]) == (["call-a", "call-b"], 2)
    assert gw.queries == [{"yeto_run": "fleet-g3", "yeto_shard": "p5gs3-0"}]
    assert al.release_incarnations(Gateway(), "fleet-g3", "p5gs3-0", []) == ([], 0)  # no incarnation: nothing


def test_a_stopped_pool_releases_every_microvm_of_its_own_shards_only():
    gw = Gateway()
    deleted, found = al.release_pool(gw, "fleet-g3", "p5gs3")
    assert deleted == ["call-a", "call-b", "successor", "pool-shard-1"] and found == 4
    assert gw.queries == [{"yeto_run": "fleet-g3"}]
    assert al.release_pool(Gateway(), "", "p5gs3") == ([], 0)


def test_a_failed_list_is_reported_never_raised():
    class Down:
        def list_sandboxes(self, metadata):
            raise RuntimeError("gateway down")

    assert al.release_pool(Down(), "fleet-g3", "p5gs3") is None
    assert al.release_incarnations(Down(), "fleet-g3", "p5gs3-0", ["i1"]) is None


def test_the_release_command_uses_the_process_client_and_the_env_run(monkeypatch, capsys):
    from yeto.rl.harness.harbor import agentenv_client as ac

    gw = Gateway()
    monkeypatch.setattr(ac, "process_client", lambda: gw)
    monkeypatch.setenv("YETO_SANDBOX_RUN_LABEL", "fleet-g3")
    assert al.main(["release", "--shard", "p5gs3-0", "--inc", "i2"]) == 0
    assert gw.deleted == ["call-b"]
    assert json.loads(capsys.readouterr().out)["released"] == 1


# -- the Modal call releases its microVMs as it ends -----------------------------------------

def test_the_call_knows_the_incarnations_whose_tape_it_wrote(tmp_path):
    spec = _spec(tmp_path)
    tape = Path(spec["event_tape"]).parent
    tape.mkdir()
    (tape / "events.old.jsonl").write_text("")
    before = mp.tape_files(spec)
    (tape / "events.i1.jsonl").write_text("")
    (tape / "events.i2.jsonl").write_text("")
    assert mp.call_incarnations(spec, before) == ["i1", "i2"]


class _Running:
    """A sampler still running when the call ends (Modal cancelled the call)."""

    returncode = None
    stderr = None

    def __init__(self, spec, tape_name="events.i1.jsonl", stop_after=None):
        self.spec, self.tape_name, self.signals, self.polls, self.stop_after = spec, tape_name, [], 0, stop_after
        Path(spec["event_tape"]).parent.mkdir(parents=True, exist_ok=True)

    def poll(self):
        self.polls += 1
        if self.polls == 1:  # the sampler starts and writes its tape
            (Path(self.spec["event_tape"]).parent / self.tape_name).write_text("")
        if self.polls == self.stop_after:
            raise KeyboardInterrupt  # the call's cancel lands in the function body
        return self.returncode

    def terminate(self):
        self.signals.append("TERM")
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.signals.append("KILL")


@contextlib.contextmanager
def _forward(port):
    yield SimpleNamespace(url="https://tunnel.test")


def test_a_cancelled_call_stops_its_sampler_then_releases_its_microvms(tmp_path, monkeypatch):
    payload = mp.agentic_payload(_payload(tmp_path), {})
    proc = _Running(payload["spec"], stop_after=3)
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw["env"]))
        return SimpleNamespace(returncode=0, stdout='{"released": 2}\n')

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
    try:
        mp.rollout_sampler_body(payload, popen=lambda *a, **k: proc, objects=SimpleNamespace(write=lambda k, v: None),
                                sleep=lambda s: None, root=str(tmp_path / "rollout"), forward=_forward)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the cancel propagates")
    assert proc.signals == ["TERM"]  # no new trial starts behind the release
    [(cmd, env)] = calls
    assert cmd[1:] == ["-m", "yeto.rl.harness.harbor.agentenv_labels", "release", "--shard", "p5gs3-0",
                       "--inc", "i1"]
    assert env["YETO_SANDBOX_RUN_LABEL"] == "fleet-g3"  # the sampler's environment: its run, key and URLs


def test_a_call_of_another_provider_releases_nothing(tmp_path, monkeypatch):
    payload = mp.agentic_payload(_payload(tmp_path, "boat"), {})
    proc = _Running(payload["spec"], stop_after=3)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no release")))
    assert mp.release_call_sandboxes(payload, payload["env"], proc, set()) is None
    assert proc.signals == []


# -- the stopped pool releases what is left ---------------------------------------------------

class _Provider:
    harbor = "agentenv"

    def __init__(self, env=None):
        self.config = {"env": {"E2B_API_URL": "http://gw.test", **(env or {})}}

    def stop_app(self, why):
        return {"app": "a", "result": "stopped"}


def _ops(provider):
    ops = mp.ModalPoolOps.__new__(mp.ModalPoolOps)
    ops.cfg, ops.provider = {"run_prefix": "fleet-g3", "shard_base": "p5gs3"}, provider
    return ops


_real = mp.release_pool_sandboxes


def test_the_pool_end_releases_its_shards_microvms(monkeypatch):
    gw = Gateway()
    monkeypatch.setattr(mp, "release_pool_sandboxes",
                        lambda cfg, section, harbor: _real(cfg, section, harbor, client=gw))
    out = _ops(_Provider()).pool_ended("pool-stop")
    assert out["result"] == "stopped" and out["sandboxes"]["removed"] == 4
    assert gw.deleted == ["call-a", "call-b", "successor", "pool-shard-1"]


def test_without_a_key_here_the_pool_end_says_it_skipped(monkeypatch):
    for name in ("AGENTENV_API_KEY", "E2B_API_KEY", "AGENTENV_API_KEY_FILE"):
        monkeypatch.delenv(name, raising=False)
    out = _ops(_Provider()).pool_ended("pool-stop")
    assert out["sandboxes"] == {"run": "fleet-g3", "shard_base": "p5gs3", "result": "skipped",
                                "why": "no AgentENV key here"}
    boat = _Provider()
    boat.harbor = "boat"
    assert _ops(boat).pool_ended("pool-stop") == {"app": "a", "result": "stopped"}  # unchanged


def test_the_pool_label_follows_the_operators_run_label():
    gw = Gateway([al.Listed("x", {"yeto_run": "mine", "yeto_shard": "p5gs3-0", "yeto_inc": "i"})])
    out = _real({"run_prefix": "fleet-g3", "shard_base": "p5gs3"}, {"env": {"YETO_SANDBOX_RUN_LABEL": "mine"}},
                "agentenv", client=gw)
    assert out["removed"] == 1 and gw.deleted == ["x"]


# -- a retry says why the trial failed (P5b-16) -------------------------------------------------

def _finished(*, exc_type=None, message="", error_detail=""):
    from yeto.rl.agentic import Outcome
    from yeto.rl.agentic.types import TrialVerdict

    verdict = None if exc_type is None else TrialVerdict(rewards=None, exception_type=exc_type, exception_message=message)
    trajectory = SimpleNamespace(outcome=Outcome.ABORTED, reason=exc_type or "AgentenvError", verdict=verdict,
                                 error_detail=error_detail)
    return SimpleNamespace(trajectory=trajectory, turns=())


def test_the_retry_detail_is_short_and_carries_no_secret_or_url_token(monkeypatch):
    from yeto.rl.engine.agentic_round import retry_error

    monkeypatch.setenv("AGENTENV_API_KEY", "sk-live-0123456789abcdef")
    message = ("agentenv create failed: ConnectError: [Errno -3] Temporary failure in name resolution "
               "https://user:pw@gw.example.dev/sandboxes/abc/t0ken-in-path?sig=s3cr3t "
               "api_key=sk-live-0123456789abcdef " + "x" * 400)
    error_type, detail = retry_error(_finished(exc_type="AgentenvError", message=message), "AgentenvError")
    assert error_type == "AgentenvError"
    assert detail.startswith("agentenv create failed: ConnectError: [Errno -3] Temporary failure in name resolution")
    assert "https://gw.example.dev " in detail
    for secret in ("t0ken-in-path", "s3cr3t", "sk-live", "pw@", "user:"):
        assert secret not in detail, secret
    assert len(detail) <= 200 and "\n" not in detail


def test_a_harness_exception_without_a_verdict_still_gives_its_message():
    from yeto.rl.engine.agentic_round import retry_error

    assert retry_error(_finished(error_detail="Server disconnected\nwithout a response"), "AgentenvError") == (
        "AgentenvError", "Server disconnected without a response")
    assert retry_error(_finished(), "AgentenvError") == ("AgentenvError", "")


def test_the_runner_keeps_the_harness_exception_message():
    import asyncio

    from yeto.rl.agentic import runner

    class Harness:
        async def run(self, task, endpoint):
            raise RuntimeError("sandbox lost: Server disconnected")

    task = SimpleNamespace(task_id="t", budget=SimpleNamespace(wall_clock_s=30.0))
    verdict, reason, detail = asyncio.run(runner._run_capped(task, Harness(), "http://x"))
    assert (verdict, reason, detail) == (None, "RuntimeError", "sandbox lost: Server disconnected")
    trajectory = runner._aborted(task, reason, policy_version=None, detail=detail)
    assert trajectory.error_detail == detail and "error_detail" not in str(trajectory.to_metadata())


def test_a_transport_error_keeps_its_message_in_the_client_error(monkeypatch):
    import sys
    import types

    import pytest

    from yeto.rl.harness.harbor.agentenv_client import AgentenvError, SdkAgentenvClient

    monkeypatch.setitem(sys.modules, "e2b", types.SimpleNamespace(Sandbox=object))

    def post(url, **kw):
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    monkeypatch.setattr(httpx, "post", post)
    client = SdkAgentenvClient(api_key="k" * 20, api_url="http://gw.test", sandbox_url="http://gw.test",
                               request_timeout_s=1.0, cold_start=False, secure=True)
    with pytest.raises(AgentenvError) as raised:
        client.create(template="t", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    assert str(raised.value) == ("agentenv create failed: ConnectError: [Errno 8] nodename nor servname provided, "
                                 "or not known")
