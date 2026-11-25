"""Bug 25: an island started before its fleet's syncer process listened.

* the syncer launch records the syncer for the islands (``fleet/island_syncer.json``) only
  once the syncer process accepts connections on its own machine (the VM is up minutes
  before a syncer built on it listens);
* an island whose probe is refused (the host answers, nothing listens) waits for the
  syncer process up to a bound instead of failing in ~30 s;
* a relaunched island (peer recovery asked) whose syncer is still in initialization has
  nothing to recover: it joins as on a first launch instead of failing on "recovery did
  not receive every fragment" (an infrastructure retry that never succeeds); a recovery
  that ends without every fragment otherwise is a deterministic stop, not relaunched.
"""

from __future__ import annotations

import asyncio
import socket
import types
from types import SimpleNamespace

import pytest

from test_peer_recovery_miles import FakeAdapter, FakeClient, FakeGroup, config, stored, stubs  # noqa: F401
from yeto import recovery_protocol as rp
from yeto.launch import fleet_syncer
from yeto.rl.adapters.miles.full_parameter.role_stream import MilesFullParameterRoleStream
from yeto.rl.adapters.miles.island_entry import probe_syncers
from yeto.rl.peer_recovery import RecoveryIncomplete


class _Conn:
    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


# -- the island's probe ---------------------------------------------------------------------


def test_the_island_waits_for_a_syncer_process_that_starts_after_its_probe(capsys):
    """r7: the island probed ~33 s before the syncer listened; 10 refusals x 3 s failed it."""
    clock, calls = Clock(), []

    def connect(addr, timeout=None):
        calls.append(clock())
        if clock() < 45.0:
            raise ConnectionRefusedError(111, "Connection refused")
        return _Conn()

    args = SimpleNamespace(syncer="203.0.113.4:29400", critic_syncer=None)
    assert probe_syncers(args, connect=connect, sleep=clock.sleep, clock=clock) == [("203.0.113.4", 29400)]
    out = capsys.readouterr().out
    assert "not listening yet" in out and "reachable (attempt 16)" in out


def test_a_syncer_that_never_listens_fails_the_island_after_the_bound_naming_the_cause():
    clock = Clock()

    def connect(addr, timeout=None):
        raise ConnectionRefusedError(111, "Connection refused")

    args = SimpleNamespace(syncer="203.0.113.4:29400", critic_syncer=None)
    with pytest.raises(ConnectionError, match=r"29400 unreachable .*\(refused for 300s: the syncer host answers"):
        probe_syncers(args, connect=connect, sleep=clock.sleep, clock=clock)
    assert 300.0 <= clock() <= 303.0


def test_an_unreachable_host_still_fails_in_about_30_s():
    """A firewall (a timeout, not a refusal) keeps the fast failure the probe exists for."""
    clock = Clock()

    def connect(addr, timeout=None):
        raise TimeoutError("timed out")

    args = SimpleNamespace(syncer="203.0.113.4:29400", critic_syncer=None)
    with pytest.raises(ConnectionError, match="after 10 attempt"):
        probe_syncers(args, connect=connect, sleep=clock.sleep, clock=clock)
    assert clock() == 27.0


# -- the syncer launch records the syncer once it listens -------------------------------------


def _serve(monkeypatch, tmp_path, *, listening, ops, polls=4):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    events = []
    monkeypatch.setattr(fleet_syncer, "record", lambda *a, **k: events.append("record") or {})
    answers = iter(listening)

    def check(args, *, head_mode, syncer_cluster):
        ok = next(answers, False)
        events.append(f"listening={ok}")
        return ok

    monkeypatch.setattr(fleet_syncer, "syncer_listening", check)
    clock = Clock()
    monkeypatch.setattr(fleet_syncer.time, "monotonic", clock)
    left = [polls]

    def sleep(s):
        clock.sleep(s)
        if "record" in events:  # supervising: the run ends after a few polls
            left[0] -= 1
            if left[0] == 0:
                runs.request_stop("r")

    code = fleet_syncer.serve(types.SimpleNamespace(cluster_prefix="r"), "h:1", 2, clusters=[], head_mode=False,
                              syncer_cluster="r-syncer", syncer_job=1, ops=ops, sleep=sleep)
    return code, events, clock


class Sky:
    def __init__(self, end=None):
        self.end = end

    def job_status(self, cluster, job):
        return _S(self.end)

    def job_alive(self, cluster, job):
        return self.end is None

    def cluster_up(self, cluster):
        return True


class _S:
    def __init__(self, end):
        self.end = end

    def is_terminal(self):
        return self.end is not None

    def __str__(self):
        return f"JobStatus.{self.end or 'RUNNING'}"


def test_the_syncer_is_recorded_for_the_islands_only_once_it_listens(tmp_path, monkeypatch):
    from yeto.launch.fleet import RUN_STOPPED_EXIT

    code, events, _ = _serve(monkeypatch, tmp_path, listening=[False, False, True], ops=Sky())
    assert code == RUN_STOPPED_EXIT
    assert events[:4] == ["listening=False", "listening=False", "listening=True", "record"]


def test_a_syncer_that_never_listens_or_whose_job_failed_is_never_recorded(tmp_path, monkeypatch, capsys):
    code, events, clock = _serve(monkeypatch, tmp_path, listening=[], ops=Sky())
    assert code == 1 and "record" not in events and clock() >= fleet_syncer.LISTEN_WAIT_S
    assert "no island was started" in capsys.readouterr().err
    code, events, clock = _serve(monkeypatch, tmp_path / "b", listening=[], ops=Sky(end="FAILED"))
    assert code == 1 and "record" not in events and clock() < fleet_syncer.LISTEN_POLL_S * 2
    assert "ended JobStatus.FAILED before it listened" in capsys.readouterr().err


def test_listening_is_checked_on_the_syncers_own_machine(monkeypatch):
    """Over ssh on the syncer VM (a probe from this machine passes through a local proxy that
    accepts any connection: r7's launcher probe said reachable 4 minutes too early); on this
    host under --controller head."""
    from yeto.launch import syncer

    monkeypatch.setattr(syncer, "_head_ports", lambda args: [29400, 29401])
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return SimpleNamespace(returncode=0 if len(seen) == 1 else 1)

    assert fleet_syncer.syncer_listening(None, head_mode=False, syncer_cluster="r-syncer", run=run)
    assert not fleet_syncer.syncer_listening(None, head_mode=False, syncer_cluster="r-syncer", run=run)
    assert seen[0][:3] == ["ssh", "-o", "BatchMode=yes"] and "r-syncer" in seen[0]
    assert seen[0][-1].endswith(" 29400 29401") and "127.0.0.1" in seen[0][-1]

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    port = server.getsockname()[1]
    try:
        monkeypatch.setattr(syncer, "_head_ports", lambda args: [port])
        assert fleet_syncer.syncer_listening(None, head_mode=True, syncer_cluster=None)
    finally:
        server.close()
    assert not fleet_syncer.syncer_listening(None, head_mode=True, syncer_cluster=None)


# -- a relaunched island whose syncer has no initial cut yet ----------------------------------


class InitAdapter(FakeAdapter):
    def fragment_parts(self, initial, fragment_id, *, ray_module=None):
        return lambda: [f"init{fragment_id}"]

    def verify_initial_fragments(self, initial, authoritative, **kw):
        self.verified = [s.version for s in authoritative]


class InitClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.init_sent = False

    def send_initial_fragments(self, parts):
        self.init_sent = True


def _initializing_events():
    return [rp.RecoverySource(rp.SOURCE_DEGRADED, None, (0, 0), b"\0" * 32, rp.INITIALIZATION_REASON),
            rp.RecoveryDone(0, 0, 0, (0, 0))]


def test_a_relaunched_island_joins_as_a_first_launch_while_the_syncer_initializes(tmp_path, stubs, monkeypatch,
                                                                                    capsys):
    """r7: the first process failed before INIT; the relaunch asked for peer recovery, the
    syncer (outer version 0, no fragment) answered from initialization, and the island
    failed on "recovery did not receive every fragment", relaunched as infrastructure."""
    adapter = InitAdapter()

    async def initialize(cls, group, **_kwargs):
        return adapter, SimpleNamespace(policy_version=0, policy_hash="p0")

    from yeto.rl.adapters.miles.models.chunked_full_parameter import MilesChunkedFullParameterAdapter

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    recovering, joining = FakeClient(events=_initializing_events()), InitClient()
    stubs.clients.extend([recovering, joining])
    stream = MilesFullParameterRoleStream(config(tmp_path / "store", recover=True))
    version_zero = (stored(0, 0, "v0"), stored(1, 0, "v1"))
    monkeypatch.setattr(stream, "_wait_for_initial_fragments", lambda client, adapter, parts, local: version_zero)
    asyncio.run(stream.initialize(FakeGroup()))

    assert [kw["recover"] for kw in stubs.kwargs] == [True, False], "a second HELLO, without the recover flag"
    assert recovering.closed and not recovering.marked
    assert joining.started and joining.init_sent, "learner 0 sends INIT as on a first launch"
    assert stream.client is joining and stream.recovery_outcome is None
    assert adapter.verified == [0, 0] and stream.fragment_versions == (0, 0)
    assert "still in initialization" in capsys.readouterr().err


def test_a_recovery_that_ends_without_every_fragment_is_a_deterministic_stop(tmp_path, stubs):
    """The syncer finalizing (or at the budget cutoff) resends nothing either; a relaunch meets
    the same syncer, so the island stops for good (exit 11), not as infrastructure."""
    from yeto.launch.island_lifecycle import island_stop_decision
    from yeto.rl.engine.leave_reason import leave_reason_for
    from yeto.rl.leave_code import LeaveReason

    events = [rp.RecoverySource(rp.SOURCE_DEGRADED, None, (0, 0), b"\0" * 32,
                                "syncer is in finalization, not training"),
              rp.RecoveryDone(0, 0, 0, (0, 0))]
    stubs.clients.append(FakeClient(events=events))
    stream = MilesFullParameterRoleStream(config(tmp_path / "store", recover=True))
    with pytest.raises(RecoveryIncomplete, match="finalization") as raised:
        asyncio.run(stream.initialize(FakeGroup()))
    assert island_stop_decision(raised.value) is not None
    assert leave_reason_for(raised.value) is LeaveReason.TRAINING_FAILED


def test_the_initialization_reason_is_the_syncers_own_text():
    """The island matches the syncer's degraded reason text: both sides name it the same."""
    from pathlib import Path

    server = (Path(__file__).resolve().parents[1] / "syncer" / "src" / "server.rs").read_text()
    template = 'format!("syncer is in {phase}, not training")'
    assert template in server and '"initialization").await' in server
    assert rp.INITIALIZATION_REASON == template[len('format!("'):-len('")')].replace("{phase}", "initialization")
