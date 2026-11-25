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


# -- the store control plane: the provider's record in the run store
# (fleet/pools/<pool>/provider/<token>.json, one record per holder of the pool's lock)


def _other_host(w, tmp_path, name):
    """The same pool's ops on another control host: its own (empty) state dir, the same nodes."""
    ops = sp.SshSamplerPoolOps(w.cfg, conf=w.conf, run=w.nodes, state_dir=tmp_path / name, clock=w.clock)
    ops.start_in_background = False
    ops.data_path = lambda: w.fencer
    return ops


def _record(objects):
    """(key, document) of the record readers take: the highest token's."""
    return store_state.read_provider_record(objects, "ssh")


def _store():
    return S3ObjectStore("s3://bkt/runs/r1", client=FakeS3())


def test_a_controller_that_takes_the_pool_over_keeps_the_hold_and_the_owed_stop(world, tmp_path, monkeypatch):
    """Controller A loses n0 under its sampler (the stop does not reach it) and dies without
    releasing its lock. Controller B on another host takes the pool over once A's lease ran
    out: it reads A's tickets, the hold on n0 and the owed stop from the store, its next start
    on n0 stops the lost sampler first, and A (fenced) writes nothing more."""
    from yeto.launch.pool import controller as rc  # noqa: PLC0415
    from yeto.launch.pool.store_lock import LockLost  # noqa: PLC0415

    w = world
    real_open = store_state.open_pool_control
    monkeypatch.setattr(store_state, "open_pool_control", lambda *a, **k: real_open(*a, clock=w.clock, **k))
    objects = _store()
    config = PoolConfig(target=2, accounts=[0, 1, 2], shard_base="smp", index_limit=64,
                        elastic=quiet_elastic(unreachable_grace_s=300.0))
    events = EventLog(log=lambda _m: None, clock=w.clock)
    a = rc._store_pool(w.cfg, config, w.ops, objects, "ssh", tmp_path / "a-state.json", events, None, None)  # noqa: SLF001
    assert w.ops.provider._store is not None  # noqa: SLF001 - attached before the pool's first tick
    a.clock = w.clock
    _run(a, w, 3)
    by_node = {w.ops.provider.node_of_ticket(m.ticket).name: m for m in a.live()}
    lost = by_node["n0"]
    w.nodes.down.add("ops@10.0.0.10")
    _run(a, w, 8)
    key, record = _record(objects)
    assert key == store_state.provider_record_key("ssh", a.control.lock.token)
    assert record["tickets"][lost.ticket]["stop_owed"] and record["control"]["token"] == a.control.lock.token
    held_until = record["tickets"][lost.ticket]["held_until"]
    assert held_until > w.clock()

    # A dies (no release, no detach); its lease runs out; B on another host claims the pool
    w.clock.t += store_state.DEFAULT_TTL_S + 60
    b_ops = _other_host(w, tmp_path, "host-b")
    assert not (tmp_path / "host-b" / "ssh-pool-r1.json").exists()
    b = rc._store_pool(w.cfg, config, b_ops, objects, "ssh", tmp_path / "host-b" / "state.json", events, None,  # noqa: SLF001
                       None)
    b.clock = w.clock
    assert b.control.lock.token > a.control.lock.token
    pb = b_ops.provider
    assert pb._rows() == w.ops.provider._rows()  # noqa: SLF001 - every ticket, hold and owed stop
    assert pb.owed_stops(0) == [(lost.ticket, lost.shard)]
    assert pb._held(pb._rows(), 0, w.clock()) == held_until  # noqa: SLF001
    assert pb.imported_from is None  # the store's record, not this host's (absent) file
    assert _record(objects)[0] == store_state.provider_record_key("ssh", b.control.lock.token)  # written at once
    assert "provider_state_imported" not in [e["event"].rsplit("_elastic_", 1)[-1] for e in events.events]

    # A is fenced: its provider refuses to write (no hold or ticket of A's lands after B's claim)
    with pytest.raises(LockLost):
        w.ops.provider.release(0, by_node["n1"].ticket)
    assert _record(objects)[1]["tickets"] == pb._rows()  # noqa: SLF001

    # n0 is back: B's next start there stops A's lost sampler before its preflight
    w.nodes.down.clear()
    w.nodes.calls.clear()
    m = SimpleNamespace(account=0, ticket=pb.order(0), shard="smp-9", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    b_ops.start(m, ["smp-9"])
    calls = [c for h, c in w.nodes.calls if h == "ops@10.0.0.10"]
    stop_at = next(i for i, c in enumerate(calls) if f"yeto-sampler-{lost.shard}" in c and "kill --signal TERM" in c)
    pre_at = max(i for i, c in enumerate(calls) if "YETO_SSH_CARDS_BEGIN" in c)
    assert stop_at < pre_at and not pb.owed_stops(0)
    assert not _record(objects)[1]["tickets"][lost.ticket].get("stop_owed")
    b.control.close()


def test_a_late_write_of_the_fenced_holder_replaces_nothing(world, tmp_path):
    """A's guard passed, then A stalled past its lease while B claimed the pool and read the
    record: A's write lands in A's own record, never over B's (readers take B's)."""
    w = world
    objects = _store()
    a_control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    w.ops.attach_control(a_control)
    pa = w.ops.provider
    t = pa.order(1)
    pa.note_shard(t, "smp-0")
    w.clock.t += store_state.DEFAULT_TTL_S + 60  # A's lease ran out
    b_control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    b_ops = _other_host(w, tmp_path, "host-b")
    b_ops.attach_control(b_control)
    b_rows = b_ops.provider._rows()  # noqa: SLF001
    a_control.lock.guard = lambda: None  # the stall: A's guard had passed before B's claim
    pa.mark_lost(t, stopped=False)  # A's late write
    a_key = store_state.provider_record_key("ssh", a_control.lock.token)
    assert json.loads(objects.read(a_key))["tickets"][t]["stop_owed"]  # it landed, in A's own record
    key, doc = _record(objects)
    assert key == store_state.provider_record_key("ssh", b_control.lock.token) and doc["tickets"] == b_rows
    b_control.close()


def test_the_local_record_is_imported_once_and_then_only_mirrored(world, tmp_path, monkeypatch):
    from yeto.launch.pool import controller as rc  # noqa: PLC0415

    w = world
    p = w.ops.provider
    t = p.order(0)  # the local plane: the host's file
    p.note_shard(t, "smp-0")
    p.mark_lost(t, stopped=False)
    local = json.loads(p.state_path.read_text())
    real_open = store_state.open_pool_control
    monkeypatch.setattr(store_state, "open_pool_control", lambda *a, **k: real_open(*a, clock=w.clock, **k))
    objects = _store()
    config = PoolConfig(target=2, accounts=[0, 1, 2], shard_base="smp", index_limit=64,
                        elastic=quiet_elastic(unreachable_grace_s=300.0))
    events = EventLog(log=lambda _m: None, clock=w.clock)
    pool = rc._store_pool(w.cfg, config, w.ops, objects, "ssh", tmp_path / "a-state.json", events, None, None)  # noqa: SLF001
    key, stored = _record(objects)
    assert key == store_state.provider_record_key("ssh", pool.control.lock.token)
    assert stored["tickets"] == local and stored["imported_from"] == str(p.state_path)
    assert p.owed_stops(0) == [(t, "smp-0")]
    imported = [e for e in events.events if e["event"].endswith("provider_state_imported")]
    assert len(imported) == 1 and imported[0]["source"] == str(p.state_path) and imported[0]["key"] == key

    # from now on the store wins: a changed local file is never read again, only overwritten
    p.state_path.write_text(json.dumps({}))
    pool.control.close()
    again = _other_host(w, tmp_path, "host-a-again")
    again.provider.state_path.parent.mkdir(parents=True, exist_ok=True)
    again.provider.state_path.write_text(json.dumps({"stale": {"slot": 1, "state": "active"}}))
    control = real_open(objects, "ssh", "smp", clock=w.clock)
    assert again.attach_control(control)["source"] == "store"
    assert again.provider._rows() == local  # noqa: SLF001
    again.provider.stop_done(t)
    mirror = json.loads(again.provider.state_path.read_text())
    assert mirror[t] == {k: v for k, v in local[t].items() if k != "stop_owed"} and "stale" not in mirror
    control.close()


def test_an_attach_without_a_local_file_writes_an_empty_record(world, tmp_path):
    """No local file and no record: the record is written at the attach, so a later attach
    on another host never imports that host's (stale) file."""
    w = world
    objects = _store()
    control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    assert w.ops.attach_control(control) == {"source": "new"}
    assert _record(objects)[1]["tickets"] == {}
    control.close()
    other = _other_host(w, tmp_path, "host-b")
    other.provider.state_path.parent.mkdir(parents=True, exist_ok=True)
    other.provider.state_path.write_text(json.dumps({"stale": {"slot": 1, "state": "active"}}))
    control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    assert other.attach_control(control)["source"] == "store"
    assert other.provider._rows() == {} and other.provider.account_free(1)  # noqa: SLF001
    control.close()


def test_the_first_versions_single_record_is_read_and_records_are_pruned(world, tmp_path):
    w = world
    objects = _store()
    old = {"tickets": {"r1-n0-1": {"slot": 0, "node": "n0", "state": "released", "shard": "smp-0",
                                   "stop_owed": True, "held_until": w.clock() + 900}}}
    objects.write(store_state.provider_key("ssh"), json.dumps(old).encode())
    tokens = []
    for i in range(3):
        control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
        ops = _other_host(w, tmp_path, f"host-{i}")
        assert ops.attach_control(control)["source"] == "store"
        assert ops.provider.owed_stops(0) == [("r1-n0-1", "smp-0")]
        tokens.append(control.lock.token)
        control.close()
    assert objects.read(store_state.provider_key("ssh")) is None  # pruned once a record replaced it
    assert store_state.provider_tokens(objects, "ssh") == tokens[-2:]  # the newest and the one before


def test_a_released_lock_writes_nothing(world):
    from yeto.launch.pool.store_lock import LockReleased  # noqa: PLC0415

    w = world
    objects = _store()
    control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    w.ops.attach_control(control)
    p = w.ops.provider
    t = p.order(1)
    control.close()  # a detach: the next holder drives the pool
    with pytest.raises(LockReleased):
        p.mark_lost(t, stopped=False)
    assert "stop_owed" not in _record(objects)[1]["tickets"][t]
    assert "stop_owed" not in p._rows()[t]  # noqa: SLF001 - nor in memory


def test_a_store_listing_blip_inside_the_lease_does_not_fail_a_save(world):
    """The save runs StoreLock.guard: a store that does not answer one LIST, inside this
    holder's lease, does not fail the order (a higher token would)."""
    w = world
    objects = _store()
    control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    w.ops.attach_control(control)
    control.lock.renew()
    real_list = objects.list
    blips = []

    def flaky(prefix):
        if prefix.endswith("/lock/") and not blips:
            blips.append(prefix)
            raise OSError("store timeout")
        return real_list(prefix)

    objects.list = flaky
    t = w.ops.provider.order(1)
    assert blips and t in _record(objects)[1]["tickets"]
    control.close()


@pytest.mark.parametrize("raw", [b'{"something": "else"}', b"{broken"])
def test_an_unreadable_store_record_refuses_the_attach(world, raw):
    w = world
    objects = _store()
    control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    objects.write(store_state.provider_record_key("ssh", control.lock.token - 1 or 1), raw)
    with pytest.raises(ValueError):
        w.ops.attach_control(control)
    assert w.ops.provider._store is None  # noqa: SLF001
    control.close()


def test_records_are_ordered_by_the_token_as_a_number():
    """The lock's fencing token is an integer (the highest + 1, created once): the record of
    token 100000000 (9 digits) is newer than 99999999's although it sorts first as a string."""
    objects = _store()
    for token, owed in ((99999999, "old"), (100000000, "new")):
        doc = {"tickets": {owed: {"slot": 0, "state": "released", "shard": "smp-0", "stop_owed": True}}}
        objects.write(store_state.provider_record_key("ssh", token), json.dumps(doc).encode())
    assert sorted(f"{t:08d}" for t in (99999999, 100000000)) == ["100000000", "99999999"]  # as strings
    assert store_state.provider_tokens(objects, "ssh") == [99999999, 100000000]
    key, doc = _record(objects)
    assert key.endswith("/100000000.json") and list(doc["tickets"]) == ["new"]
    assert list(store_state.read_provider_record(objects, "ssh", upto=99999999)[1]["tickets"]) == ["old"]


def test_an_owed_stop_the_fenced_holder_writes_late_is_run_by_the_next_holder(world, tmp_path):
    """A's guard passed, then A stalled past its lease; B claimed the pool and read A's record
    (smp-0 released, nothing owed); then A's write landed: the stop of smp-0 on n1 never got
    there (owed). B's next save merges it, and B's next start on n1 stops smp-0 first: no
    sampler is left on n1 billing with nobody to stop it. Once done it stays done."""
    w = world
    objects = _store()
    a_control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    w.ops.attach_control(a_control)
    pa = w.ops.provider
    t = pa.order(1)
    pa.note_shard(t, "smp-0")
    pa.release(1, t)
    w.clock.t += store_state.DEFAULT_TTL_S + 60  # A's lease ran out
    b_control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    b_ops = _other_host(w, tmp_path, "host-b")
    b_ops.attach_control(b_control)
    pb = b_ops.provider
    assert pb.owed_stops(1) == []  # what B read: nothing owed
    a_control.lock.guard = lambda: None  # the stall: A's guard had passed before B's claim
    pa.mark_lost(t, stopped=False)  # A's late write: the stop of smp-0 did not reach n1
    assert pb.owed_stops(1) == []  # B has not saved since

    other = pb.order(0)  # any save of B merges A's record
    assert pb.owed_stops(1) == [(t, "smp-0")] and pb._held(pb._rows(), 1, w.clock()) is not None  # noqa: SLF001
    assert _record(objects)[1]["tickets"][t]["stop_owed"]  # in B's record too
    m = SimpleNamespace(account=1, ticket=pb.order(1), shard="smp-7", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    w.nodes.calls.clear()
    b_ops.start(m, ["smp-7"])
    calls = [c for h, c in w.nodes.calls if h == "ops@10.0.0.11"]
    stop_at = next(i for i, c in enumerate(calls) if "yeto-sampler-smp-0" in c and "kill --signal TERM" in c)
    pre_at = max(i for i, c in enumerate(calls) if "YETO_SSH_CARDS_BEGIN" in c)
    assert stop_at < pre_at and pb.owed_stops(1) == []
    pb.release(0, other)  # another save: A's record still owes it, B saw it done
    assert pb.owed_stops(1) == [] and t in _record(objects)[1]["stops_done"]
    # the next holder after B does not bring it back either
    b_control.close()
    c_control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    c_ops = _other_host(w, tmp_path, "host-c")
    c_ops.attach_control(c_control)
    assert c_ops.provider.owed_stops(1) == []
    c_control.close()


def test_the_merge_takes_the_later_hold_and_keeps_this_holders_own_fields(world, tmp_path):
    w = world
    objects = _store()
    old = {"tickets": {"r1-n2-1": {"slot": 2, "node": "n2", "state": "released", "shard": "smp-3",
                                   "held_until": w.clock() + 900, "lost_at": w.clock()},
                       "r1-n0-1": {"slot": 0, "node": "n0", "state": "released", "shard": "smp-1"}}}
    objects.write(store_state.provider_record_key("ssh", 1), json.dumps(old).encode())
    objects.write(store_state.provider_record_key("ssh", 2), json.dumps(
        {"tickets": {"r1-n0-1": {"slot": 0, "node": "n0", "state": "active", "shard": "smp-1"}}}).encode())
    # token 3 claims: it reads record 2 (the newest), its saves merge record 2 (its predecessor)
    for token in (1, 2):
        objects.write(f"fleet/pools/ssh/lock/{token:08d}.json", json.dumps({"token": token}).encode())
        objects.write(f"fleet/pools/ssh/lock/{token:08d}.lease", json.dumps(
            {"token": token, "until": 0.0, "released": True}).encode())
    control = store_state.open_pool_control(objects, "ssh", "smp", clock=w.clock)
    assert control.lock.token == 3
    w.ops.attach_control(control)
    rows = w.ops.provider._rows()  # noqa: SLF001
    assert rows["r1-n0-1"]["state"] == "active"  # record 2's own state
    assert "r1-n2-1" not in rows  # record 1 is not the predecessor's newest: pruned, never merged
    control.close()
