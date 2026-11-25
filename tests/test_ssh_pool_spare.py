"""Spare nodes of an ssh sampler pool (unified-sampler-pool 3.3): the inventory offers
more nodes than the target; a node lost under its sampler is fenced, and the slot moves
to a free spare node whose sampler continues the lost shard's prompts. The lost node is
not started again while a spare is free; a stop it never got is owed to its next start.

The pool is the real ``IslandPool`` over the real ``SshSamplerPoolOps`` (ssh faked), on
the local state and on the store control plane (lock, allocator, ``GuardedOps``)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_rl_object_store import FakeS3
from tests.test_rollout_island_pool import Clock, FakeFencer, quiet_elastic
from tests.test_ssh_sampler_pool import IDLE
from yeto.cloud.s3_store import S3ObjectStore
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.nc import cli
from yeto.launch.node_inventory import parse_node
from yeto.launch.pool import store_state
from yeto.launch.pool.island_pool import GONE, READY, IslandPool, PoolConfig
from yeto.launch.pool.status import EventLog

NAMES = ("n0", "n1", "n2", "n3")


def _node(i: int):
    return parse_node({"name": NAMES[i], "host": f"ops@10.0.0.{10 + i}", "device_family": "npu",
                       "cpu_arch": "aarch64", "card": "910B4", "cards_per_node": 8, "hbm_gib": 64,
                       "host_memory_gib": 1024, "runtime": "rootfs"}, index=i)


class Nodes:
    """ssh to the four nodes: each answers like a healthy node unless it is down."""

    def __init__(self):
        self.down: set[str] = set()
        self.calls: list[tuple[str, str]] = []  # (host, command)
        self.specs: dict[str, dict] = {}  # host -> the last sampler spec uploaded there

    def __call__(self, argv, **kw):
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        host, cmd = argv[-2], argv[-1]
        self.calls.append((host, cmd))
        if "/sampler/" in cmd and ".json.tmp" in cmd and kw.get("input"):
            self.specs[host] = json.loads(kw["input"])
        if host in self.down:
            return SimpleNamespace(returncode=255, stdout="", stderr=f"ssh: connect to host {host} port 22: timed out")
        if "YETO_SSH_CARDS_BEGIN" in cmd:
            out = f"YETO_SSH_ARCH aarch64\nYETO_SSH_FREE_GIB 300\nYETO_SSH_CARDS_BEGIN\n{IDLE}\nYETO_SSH_CARDS_END rc=0\n"
        elif "run -d" in cmd:
            out = "YETO_SSH_STARTED container=x\n"
        elif "YETO_NC_PROBE" in cmd:
            out = "YETO_NC_PROBE status=running exit=0 joined=1 age=3 tw=0 tok=- run=- wq=- vh=0 vm=-\n"
        else:
            out = "stopped\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    def started_on(self, host: str) -> list[str]:
        return [c for h, c in self.calls if h == host and "run -d" in c]

    def stops_on(self, host: str, shard: str) -> list[str]:
        return [c for h, c in self.calls if h == host and f"yeto-sampler-{shard}" in c and "kill --signal TERM" in c]


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
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides",
                        lambda spec, **kw: dict(spec, continues_shard=kw.get("continues_shard")))
    monkeypatch.setenv(sp.ssh_transport.MUX_ENV, "off")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sampler_id": "x", "store": "/s", "prompts": {"groups_per_round": 2},
                                "engine": {"factory": "f:g"}}))
    env = tmp_path / "store.env"
    env.write_text("AWS_ACCESS_KEY_ID=x\n")
    cfg = {"target": 2, "slots": [0, 1, 2], "shard_base": "smp", "spec": str(spec), "run_prefix": "runs/r1",
           "r2_env_file": str(env),
           "provider": {"kind": "ssh", "nodes": list(NAMES[:3]), "workdir": "/srv/yeto/r1", "runtime": "container",
                        "container": {"engine": "podman", "image": "img"}}}
    conf = sp.parse_config(cfg, inventory=[_node(i) for i in range(4)])
    nodes, clock, fencer = Nodes(), Clock(), FakeFencer()
    ops = sp.SshSamplerPoolOps(cfg, conf=conf, run=nodes, state_dir=tmp_path, clock=clock)
    ops.start_in_background = False  # the start runs inside the tick: the test reads it right after
    ops.data_path = lambda: fencer  # the work contract's fence (store): what the pool calls through ops.fence
    return SimpleNamespace(ops=ops, nodes=nodes, clock=clock, fencer=fencer, conf=conf, cfg=cfg)


def _pool(w, *, control=None, grace_s=300.0):
    config = PoolConfig(target=2, accounts=[0, 1, 2], shard_base="smp",
                        elastic=quiet_elastic(unreachable_grace_s=grace_s))
    return IslandPool.from_doc(None, config, w.ops, control=control, clock=w.clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=w.clock))


def _run(pool, w, ticks: int, step: float = 60.0) -> None:
    for _ in range(ticks):
        w.clock.t += step
        pool.tick()


def _control(clock):
    objects = S3ObjectStore("s3://bkt/runs/r1", client=FakeS3())
    return store_state.open_pool_control(objects, "ssh", "smp", clock=clock)


@pytest.mark.parametrize("plane", ["local", "store"])
def test_a_lost_node_moves_its_slot_to_the_spare(world, plane):
    w = world
    control = _control(w.clock) if plane == "store" else None
    pool = _pool(w, control=control)
    if control is not None:  # every machine / sampler / fence call runs under the confirmed lock
        assert type(pool.ops).__name__ == "GuardedOps"
    _run(pool, w, 3)
    ready = {m.shard: m for m in pool.live() if m.state == READY}
    assert len(ready) == 2
    by_node = {w.ops.provider.node_of_ticket(m.ticket).name: m for m in ready.values()}
    assert set(by_node) == {"n0", "n1"}  # n2 is the spare
    victim = by_node["n0"]
    lost_shard = victim.shard
    w.nodes.down.add("ops@10.0.0.10")  # n0 drops off the network with its sampler
    _run(pool, w, 8)

    # fenced first: the shard's work contract was fenced by the pool
    assert (lost_shard, victim.ticket) in w.fencer.fenced
    events = [e["event"] for e in pool.events.events]
    assert any(e.endswith("node_lost") for e in events)
    fence_at = next(i for i, e in enumerate(events) if e.endswith("shard_fenced"))
    order_at = max(i for i, e in enumerate(events) if e.endswith("ordered"))
    assert fence_at < order_at

    # the replacement runs on the spare and continues the lost shard's prompts
    live = [m for m in pool.live() if m.state != GONE]
    assert len(live) == 2
    successor = next(m for m in live if m.ticket != by_node["n1"].ticket)
    assert w.ops.provider.node_of_ticket(successor.ticket).name == "n2"
    assert successor.replaces_shard == lost_shard and successor.predecessor_lost
    assert successor.shard != lost_shard  # a new shard (new incarnation), never the lost one reused
    spec = w.nodes.specs["ops@10.0.0.12"]  # what the spare's sampler was started with
    assert spec["sampler_id"] == successor.shard and spec["continues_shard"] == lost_shard
    assert spec["handoff_wait_s"] == 60.0  # its predecessor is lost: it does not wait for a handoff
    assert len(w.nodes.started_on("ops@10.0.0.12")) == 1

    # never started again on the lost node: only the first start went there
    assert len(w.nodes.started_on("ops@10.0.0.10")) == 1
    rows = w.ops.provider._rows()  # noqa: SLF001
    assert rows[victim.ticket]["state"] == "released" and rows[victim.ticket]["stop_owed"]
    assert w.ops.provider._held(rows, 0, w.clock()) is not None  # noqa: SLF001 - held as lost
    if control is not None:
        assert control.allocator.issued()[-1]["shard"] == successor.shard


def test_the_held_node_is_offered_again_without_a_spare_and_its_owed_stop_goes_first(world):
    w = world
    pool = _pool(w)
    _run(pool, w, 3)
    by_node = {w.ops.provider.node_of_ticket(m.ticket).name: m for m in pool.live()}
    w.nodes.down.add("ops@10.0.0.10")
    _run(pool, w, 8)
    lost = by_node["n0"]
    p = w.ops.provider
    # n0 held, n1 and n2 busy: no spare left, so n0 is offered again (as before spares)
    assert not p.account_free(1) and not p.account_free(2) and p.account_free(0)
    # while a spare is free the held slot reads busy
    p.release(2, next(t for s, t in p.run_tickets() if s == 2))
    assert not p.account_free(0) and p.account_free(2)
    # after the hold it is a plain free slot again
    w.clock.t += w.conf.lost_node_hold_s + 1
    assert p.account_free(0)

    # n0 comes back: the next start there stops the lost sampler before its preflight
    w.nodes.down.clear()
    m = SimpleNamespace(account=0, ticket=p.order(0), shard="smp-9", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    w.ops.start(m, ["smp-9"])
    calls = [c for h, c in w.nodes.calls if h == "ops@10.0.0.10"]
    stop_at = next(i for i, c in enumerate(calls) if f"yeto-sampler-{lost.shard}" in c and "kill --signal TERM" in c)
    pre_at = max(i for i, c in enumerate(calls) if "YETO_SSH_CARDS_BEGIN" in c)
    assert stop_at < pre_at and not p.owed_stops(0)


def test_an_owed_stop_that_does_not_go_through_refuses_the_start(world):
    w = world
    p = w.ops.provider
    t = p.order(0)
    p.note_shard(t, "smp-0")
    p.mark_lost(t, stopped=False)
    p.release(0, t)
    w.nodes.down.add("ops@10.0.0.10")
    m = SimpleNamespace(account=0, ticket=p.order(0), shard="smp-1", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    with pytest.raises(sp.SshSamplerError, match="owed"):
        w.ops.start(m, ["smp-1"])
    assert p.owed_stops(0) == [(t, "smp-0")]
    assert not w.nodes.started_on("ops@10.0.0.10")


def test_a_reachable_drain_marks_nothing(world):
    w = world
    p = w.ops.provider
    m = SimpleNamespace(account=1, ticket=p.order(1), shard="smp-0", unreachable_since=None)
    p.note_shard(m.ticket, m.shard)
    w.ops.drain(m)
    row = p._rows()[m.ticket]  # noqa: SLF001
    assert row["state"] == "released" and "held_until" not in row and "stop_owed" not in row
    assert p.account_free(1)


def test_a_drain_the_pool_saw_unreachable_holds_even_when_the_stop_answers(world):
    w = world
    p = w.ops.provider
    m = SimpleNamespace(account=1, ticket=p.order(1), shard="smp-0", unreachable_since=w.clock() - 400)
    w.ops.drain(m)
    row = p._rows()[m.ticket]  # noqa: SLF001
    assert row["held_until"] > w.clock() and "stop_owed" not in row
    assert not p.account_free(1)  # slots 0 and 2 are free spares


def test_hold_config():
    base = {"kind": "ssh", "nodes": ["n0"], "workdir": "/srv/yeto/r1", "runtime": "venv",
            "venv": {"python": "/v/bin/python"}}
    assert sp.parse_config({"provider": base}, inventory=[_node(0)]).lost_node_hold_s == 1800.0
    assert sp.parse_config({"provider": dict(base, lost_node_hold_s=0)}, inventory=[_node(0)]).lost_node_hold_s == 0
    for bad in (-1, "1h", True):
        with pytest.raises(ValueError, match="lost_node_hold_s"):
            sp.parse_config({"provider": dict(base, lost_node_hold_s=bad)}, inventory=[_node(0)])


def test_hold_zero_offers_the_lost_slot_at_once_and_keeps_the_owed_stop(world):
    w = world
    p = w.ops.provider
    object.__setattr__(w.conf, "lost_node_hold_s", 0.0)
    m = SimpleNamespace(account=1, ticket=p.order(1), shard="smp-0", unreachable_since=w.clock() - 400)
    p.note_shard(m.ticket, m.shard)
    w.nodes.down.add("ops@10.0.0.11")
    w.ops.drain(m)
    assert p.account_free(1)  # spares 0 and 2 are free too: no hold
    assert p.owed_stops(1) == [(m.ticket, "smp-0")]
