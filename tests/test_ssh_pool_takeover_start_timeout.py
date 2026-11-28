"""B6: a control plane that takes an ssh sampler pool over (on-site takeover, #934) must not
time out the start of a sampler that already serves. The pool's start timeout counts from
the order; a takeover restores a member mid-start (``STARTING``, not joined) when the old
controller never saw the JOIN line (the node runtime's container log tail, the node's
tape), and then fenced and stopped a sampler that had served for 25 minutes. The run
store's node records (ready, fresh heartbeat, no exit) are the durable proof that it
serves: the takeover reads them, keeps the member, and the node count stays.

The pool is the real ``IslandPool`` over the real ``SshSamplerPoolOps`` (ssh faked) on the
store control plane; the takeover is a new ops / pool / lock from the store alone (the old
controller's local state deleted)."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from tests.test_rl_object_store import FakeS3
from tests.test_rollout_island_pool import Clock, FakeFencer, quiet_elastic
from tests.test_ssh_pool_spare import _node
from tests.test_ssh_sampler_pool import IDLE
from yeto.cloud.s3_store import S3ObjectStore
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.nc import cli
from yeto.launch.pool import store_state
from yeto.launch.pool.island_pool import READY, STARTING, IslandPool, PoolConfig
from yeto.launch.pool.status import EventLog
from yeto.node import records

START_TIMEOUT_S = 1500.0


class Node:
    """ssh to the pool's node: the sampler container runs; its JOIN line is not in what the
    probe greps (``joined``: what the probe's JOIN grep finds)."""

    def __init__(self):
        self.joined = False
        self.calls: list[str] = []

    def __call__(self, argv, **kw):
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout=b"tar" if "archive" in argv else "abc123\n", stderr="")
        cmd = argv[-1]
        self.calls.append(cmd)
        if "YETO_SSH_CARDS_BEGIN" in cmd:
            out = f"YETO_SSH_ARCH aarch64\nYETO_SSH_FREE_GIB 300\nYETO_SSH_CARDS_BEGIN\n{IDLE}\nYETO_SSH_CARDS_END rc=0\n"
        elif "run -d" in cmd:
            out = "YETO_SSH_STARTED container=x\n"
        elif "YETO_NC_PROBE" in cmd:
            out = (f"YETO_NC_PROBE status=running exit=0 joined={int(self.joined)} age=3 tw=0 tok=- run=- wq=- "
                   "vh=0 vm=-\n")
        else:
            out = "stopped\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    def stops(self) -> list[str]:
        return [c for c in self.calls if "kill --signal TERM" in c or "YETO_SSH_TERM" in c]


@pytest.fixture
def world(tmp_path, monkeypatch):
    class Pop:
        stdout = None

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(sp.subprocess, "Popen", Pop)
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    monkeypatch.setenv(sp.ssh_transport.MUX_ENV, "off")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sampler_id": "x", "store": "/s", "prompts": {"groups_per_round": 2},
                                "engine": {"factory": "f:g"}}))
    env = tmp_path / "store.env"
    env.write_text("AWS_ACCESS_KEY_ID=x\nR2_BUCKET=bkt\n")
    cfg = {"target": 1, "slots": [0], "shard_base": "smp", "spec": str(spec), "run_prefix": "runs/r1",
           "r2_env_file": str(env),
           "provider": {"kind": "ssh", "nodes": ["n0"], "workdir": "/srv/yeto/r1", "runtime": "container",
                        "container": {"engine": "podman", "image": "img"}}}
    run_store = SimpleNamespace(objects=S3ObjectStore("s3://bkt/runs/runs/r1", client=FakeS3()))
    monkeypatch.setattr(sp, "pool_fence_store", lambda cfg, **kw: run_store)
    control_objects = S3ObjectStore("s3://bkt/runs/r1", client=FakeS3())
    return SimpleNamespace(cfg=cfg, node=Node(), clock=Clock(), fencer=FakeFencer(), store=run_store,
                           control_objects=control_objects, tmp=tmp_path)


def _controller(w, name: str):
    """One control plane's ops + pool from the store alone (its own empty local state dir)."""
    state_dir = w.tmp / name
    state_dir.mkdir()
    conf = sp.parse_config(w.cfg, inventory=[_node(0)])
    assert conf.node_runtime
    ops = sp.SshSamplerPoolOps(w.cfg, conf=conf, run=w.node, state_dir=state_dir, clock=w.clock)
    ops.start_in_background = False
    ops.data_path = lambda: w.fencer
    control = store_state.open_pool_control(w.control_objects, "ssh", "smp", holder=name, ttl_s=120.0,
                                            clock=w.clock)
    ops.attach_control(control)
    doc = control.state.load()
    config = PoolConfig(target=1, accounts=[0], shard_base="smp",
                        elastic=quiet_elastic(start_timeout_s=START_TIMEOUT_S))
    pool = IslandPool.from_doc(doc, config, ops, control=control, clock=w.clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=w.clock))
    return pool, ops, state_dir


def _tick(pool, w, n: int, step: float = 60.0, beat=None) -> None:
    for _ in range(n):
        w.clock.t += step
        if beat is not None:
            beat()
        pool.tick()


def _names(pool) -> list[str]:
    return [e["event"].removeprefix("nc_elastic_") for e in pool.events.events]


def test_a_takeover_keeps_a_serving_sampler_past_the_start_timeout(world):
    w = world
    a, a_ops, a_dir = _controller(w, "ctla")
    _tick(a, w, 2)  # ordered and started; the node runtime is still booting (no node records)
    (m,) = a.live()
    assert m.state == STARTING and not m.joined
    ticket = m.ticket
    generation = int(json.loads(w.store.objects.read(records.assignment_key(ticket)))["generation"])
    snapshot_state = json.loads(json.dumps(a.snapshot()))
    assert [x["state"] for x in snapshot_state["members"]] == [STARTING]

    # the sampler joins: its node runtime writes the ready record and heartbeats to the run
    # store. The probe's JOIN grep does not see it (the line left the container log tail).
    reporter = records.NodeReporter(w.store.objects, ticket, generation)
    reporter.ready(stage="serving")
    beat = lambda: reporter.heartbeat("serving")  # noqa: E731
    beat()

    # ctla dies right after a save that still says STARTING (its state as the takeover reads it)
    w.control_objects.write(store_state.state_key("ssh"), json.dumps(dict(snapshot_state)).encode())
    shutil.rmtree(a_dir)  # the old controller's local state is gone with its host
    w.clock.t += START_TIMEOUT_S + 600  # the sampler served on; the lock's lease ran out

    b, b_ops, _ = _controller(w, "ctlb")
    assert [x.state for x in b.live()] == [STARTING] and not b.live()[0].joined
    stops_before = len(w.node.stops())
    _tick(b, w, 5, beat=beat)

    names = _names(b)
    assert "start_resumed" in names
    assert "start_timeout" not in names and "shard_fenced" not in names, names
    assert not w.fencer.fenced
    assert len(w.node.stops()) == stops_before  # its container was never stopped
    (kept,) = b.live()
    assert kept.ticket == ticket and kept.state == READY and kept.joined
    assert [t for _s, t in b_ops.provider.run_tickets()] == [ticket]  # the pool still holds its node


def test_without_serving_evidence_the_takeover_still_times_the_start_out(world):
    """The guard reads proof, not hope: a sampler whose node runtime never wrote its ready
    record is timed out as before (counted from its order)."""
    w = world
    a, _a_ops, a_dir = _controller(w, "ctla")
    _tick(a, w, 2)
    (m,) = a.live()
    generation = int(json.loads(w.store.objects.read(records.assignment_key(m.ticket)))["generation"])
    reporter = records.NodeReporter(w.store.objects, m.ticket, generation)
    beat = lambda: reporter.heartbeat("model")  # noqa: E731 - booting: heartbeats, no ready
    w.control_objects.write(store_state.state_key("ssh"), json.dumps(a.snapshot()).encode())
    shutil.rmtree(a_dir)
    w.clock.t += START_TIMEOUT_S + 600
    b, _b_ops, _ = _controller(w, "ctlb")
    _tick(b, w, 3, beat=beat)
    assert "start_timeout" in _names(b)


def test_a_store_ready_record_of_an_exited_sampler_is_no_proof(world):
    w = world
    a, a_ops, _ = _controller(w, "ctla")
    _tick(a, w, 2)
    (m,) = a.live()
    generation = int(json.loads(w.store.objects.read(records.assignment_key(m.ticket)))["generation"])
    reporter = records.NodeReporter(w.store.objects, m.ticket, generation)
    reporter.ready(stage="serving")
    assert a_ops.serving_evidence(m)
    reporter.exit(1, "crashed")
    assert not a_ops.serving_evidence(m)
