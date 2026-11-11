"""IslandPool over fake providers: provider-neutral pool policy, MultiProvider,
ProviderPoolOps, the provider registry and the RolloutController."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.launch.pool import controller as rc
from yeto.launch.pool import provider as providers
from yeto.launch.pool.island_pool import GONE, READY, IslandPool, PoolConfig
from yeto.launch.pool.provider import MultiProvider, ProviderPoolOps
from yeto.launch.pool.status import ElasticConfig, EventLog, LeaseStatus, Probe


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeProvider:
    """Machines in a dict; ``lease_s`` gives each a lease end (a lease-bounded provider)."""

    def __init__(self, name: str, clock: Clock, *, lease_s: float | None = None):
        self.name, self.clock, self.lease_s = name, clock, lease_s
        self.machines: dict[str, dict] = {}
        self.fail_orders = 0
        self.orders: list = []
        self.released: list[str] = []
        self.foreign: list[tuple] = []  # open machines of this run the pool does not know of

    def order(self, slot):
        if self.fail_orders:
            self.fail_orders -= 1
            raise RuntimeError(f"{self.name}: no capacity")
        ticket = f"{self.name}-t{len(self.orders)}"
        self.orders.append(slot)
        until = None if self.lease_s is None else self.clock() + self.lease_s
        self.machines[ticket] = {"slot": slot, "state": "active", "lease_until": until, "end_reason": None}
        return ticket

    def status(self, slot, ticket):
        m = self.machines[ticket]
        return LeaseStatus(m["state"], lease_until=m["lease_until"], end_reason=m["end_reason"])

    def release(self, slot, ticket):
        m = self.machines[ticket]
        if m["state"] != "ended":
            m["state"], m["end_reason"] = "ended", "released"
        self.released.append(ticket)
        return {"ended": True}

    def node(self, slot, ticket):
        return {"provider": self.name, "ticket": ticket}

    def balance(self):
        return None

    def account_free(self, slot):
        return not any(m["slot"] == slot and m["state"] != "ended" for m in self.machines.values())

    def run_tickets(self):
        own = [(m["slot"], t) for t, m in self.machines.items() if m["state"] != "ended"]
        return own + list(self.foreign)

    def evict(self, ticket, reason="preempted"):
        self.machines[ticket].update(state="ended", end_reason=reason)


class FakeRoute:
    def __init__(self):
        self.started, self.stopped, self.restarted = [], [], []
        self.joined: set[str] = set()

    def start(self, member, node, shards):
        self.started.append((member.ticket, member.shard, member.replaces_shard, node["provider"]))
        self.joined.add(member.ticket)

    def restart(self, member, node, shards):
        self.restarted.append(member.ticket)

    def probe(self, member):
        return Probe(node="ok", sampler="running", joined=member.ticket in self.joined)

    def stop(self, member):
        self.stopped.append(member.ticket)


class FakeFencer:
    def __init__(self):
        self.fenced: list[tuple[str, str]] = []

    def fence_shard(self, shard, by):
        self.fenced.append((shard, by))

    def fence_incarnation(self, shard, by):
        self.fenced.append((shard, f"incarnation:{by}"))


class DrainingFencer(FakeFencer):
    """The work contract with drain requests: a shard answers once ``idle`` holds it."""

    def __init__(self):
        super().__init__()
        self.requests: dict[str, str] = {}
        self.idle: set[str] = set()
        self.cancelled: list[str] = []

    def request_drain(self, shard, seq):
        self.requests[shard] = seq
        return seq

    def drain_answered(self, shard, seq):
        return self.requests.get(shard) == seq and shard in self.idle

    def cancel_drain(self, shard):
        self.requests.pop(shard, None)
        self.cancelled.append(shard)


def quiet_elastic(**kw) -> ElasticConfig:
    return ElasticConfig(engine_stall_s=None, metrics_port=None, **kw)


def make_pool(provider, slots, *, target, route=None, fencer=None, clock, **cfg):
    route = route or FakeRoute()
    fencer = fencer or FakeFencer()
    ops = ProviderPoolOps(provider, route if not isinstance(provider, MultiProvider)
                          else {name: route for name in provider.providers}, fencer)
    config = PoolConfig(target=target, accounts=list(slots), shard_base="smp",
                        elastic=cfg.pop("elastic", quiet_elastic()), **cfg)
    pool = IslandPool(config, ops, clock=clock, log=lambda _m: None,
                      events=EventLog(log=lambda _m: None, clock=clock))
    return pool, route, fencer


def ticks(pool, n=3, clock=None, step=60.0):
    for _ in range(n):
        pool.tick()
        if clock is not None:
            clock.t += step


def names(pool):
    return [e["event"].removeprefix("nc_elastic_") for e in pool.events.events]


# -- two providers in one pool --------------------------------------------------


def two_providers(clock, **kw):
    a, b = FakeProvider("nc", clock, **kw), FakeProvider("daytona", clock)
    multi = MultiProvider({"nc": a, "daytona": b}, {"nc": [1], "daytona": [1, 2]})
    return a, b, multi


def test_mixed_providers_fill_target_across_both():
    clock = Clock()
    a, b, multi = two_providers(clock)
    assert multi.slots() == ["nc:1", "daytona:1", "daytona:2"]
    pool, route, _ = make_pool(multi, multi.slots(), target=2, clock=clock)
    ticks(pool, 3, clock)
    live = pool.live()
    assert [m.account for m in live] == ["nc:1", "daytona:1"]
    assert all(m.state == READY for m in live)
    assert a.orders == [1] and b.orders == [1]
    assert {s[3] for s in route.started} == {"nc", "daytona"}


def test_mixed_evict_fences_releases_and_refills_with_continues_shard():
    clock = Clock()
    a, b, multi = two_providers(clock)
    pool, route, fencer = make_pool(multi, multi.slots(), target=2, clock=clock)
    ticks(pool, 3, clock)
    victim = next(m for m in pool.live() if m.account == "daytona:1")
    b.evict(victim.ticket)
    ticks(pool, 3, clock)
    assert (victim.shard, victim.ticket) in fencer.fenced  # fenced before the drain
    assert victim.state == GONE and victim.ticket in b.released
    new = [m for m in pool.live() if m.ticket not in (victim.ticket,) and m.account.startswith("daytona")]
    assert len(new) == 1 and new[0].replaces_shard == victim.shard and new[0].state == READY
    ev = names(pool)
    assert ev.index("evicted") < ev.index("shard_fenced") < ev.index("released")
    assert pool.pending == []


# -- lease-end replacement (make before break) ------------------------------------


def test_lease_end_replacement_on_another_slot_then_old_drained():
    clock = Clock()
    nc = FakeProvider("nc", clock, lease_s=4 * 3600)
    pool, route, _ = make_pool(nc, [1, 4], target=1, clock=clock, replace_lead_s=3600, end_margin_s=600)
    ticks(pool, 3, clock)
    old = pool.live()[0]
    assert old.state == READY and old.lease_until is not None
    clock.t = old.lease_until - 3000  # inside replace_lead_s
    ticks(pool, 1, clock, step=10)
    successor = next(m for m in pool.live() if m.replaces == old.ticket)
    assert successor.account == 4 and old.replaced_by == successor.ticket
    ticks(pool, 3, clock, step=10)
    assert successor.state == READY and successor.replaces_shard == old.shard
    assert old.state == GONE and old.ticket in nc.released
    assert len(pool.live()) == 1


# -- scale 1 -> 2 -> 1 ---------------------------------------------------------------


def write_scale(path: Path, target: int, seq: int, retire=()):
    path.write_text(json.dumps({"run": "r1", "target": target, "seq": seq, "retire": list(retire)}))


def test_scale_one_two_one(tmp_path):
    clock = Clock()
    a, b, multi = two_providers(clock)
    pool, route, fencer = make_pool(multi, multi.slots(), target=1, clock=clock)
    pool.scale_path, pool.scale_run = tmp_path / "scale.json", "r1"
    ticks(pool, 3, clock)
    assert len(pool.live()) == 1
    write_scale(pool.scale_path, 2, 1)
    ticks(pool, 3, clock)
    assert len(pool.live()) == 2 and all(m.state == READY for m in pool.live())
    second = pool.live()[1]
    write_scale(pool.scale_path, 1, 2, retire=[second.ticket])
    ticks(pool, 2, clock)
    assert [m.ticket for m in pool.live()] != [second.ticket] and len(pool.live()) == 1
    assert second.state == GONE and (second.shard, second.ticket) in fencer.fenced
    assert "scaled_down" in names(pool) and "target_changed" in names(pool)


def test_scale_down_drains_through_the_work_contract_before_release(tmp_path):
    clock = Clock()
    a, b, multi = two_providers(clock)
    fencer = DrainingFencer()
    pool, route, _ = make_pool(multi, multi.slots(), target=2, clock=clock, fencer=fencer)
    pool.scale_path, pool.scale_run = tmp_path / "scale.json", "r1"
    ticks(pool, 3, clock)
    victim = pool.live()[1]
    write_scale(pool.scale_path, 1, 1, retire=[victim.ticket])
    ticks(pool, 2, clock)
    assert victim.shard in fencer.requests and victim.retiring_since is not None
    assert victim.state != GONE and victim.ticket not in b.released  # trials in flight finish first
    fencer.idle.add(victim.shard)
    ticks(pool, 2, clock)
    assert victim.state == GONE and victim.ticket in b.released and len(pool.live()) == 1
    assert "scale_draining" in names(pool)


def test_a_machine_released_by_a_person_lowers_the_target_and_is_not_refilled(tmp_path):
    clock = Clock()
    a, b, multi = two_providers(clock)
    pool, *_ = make_pool(multi, multi.slots(), target=2, clock=clock)
    pool.scale_path, pool.scale_run = tmp_path / "scale.json", "r1"
    ticks(pool, 3, clock)
    gone = pool.live()[1]
    b.machines[gone.ticket].update(state="ended", end_reason="released")  # the provider console
    ticks(pool, 3, clock)
    assert "manual_release_detected" in names(pool)
    assert pool.config.target == 1 and len(pool.live()) == 1 and b.orders == [1]
    assert json.loads(pool.scale_path.read_text())["target"] == 1  # outlives a restart


def test_a_paused_member_is_not_retired_when_its_sampler_crashes(tmp_path):
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, route, _ = make_pool(nc, [1, 2], target=1, clock=clock)
    pool.scale_path, pool.scale_run = tmp_path / "scale.json", "r1"
    ticks(pool, 3, clock)
    m = pool.live()[0]
    pool.scale_path.write_text(json.dumps({"run": "r1", "target": 1, "seq": 1, "paused": [m.ticket]}))
    route.joined.discard(m.ticket)
    route.probe = lambda member: Probe(node="ok", sampler="exited", exit_code=1, joined=False)
    ticks(pool, 3, clock)
    assert m.paused and m.state != GONE and nc.orders == [1] and route.restarted == []
    assert "manual_pause" in names(pool)


def test_scale_file_of_another_run_is_ignored(tmp_path):
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1, 2], target=1, clock=clock)
    pool.scale_path, pool.scale_run = tmp_path / "scale.json", "other"
    write_scale(pool.scale_path, 2, 1)
    ticks(pool, 3, clock)
    assert pool.config.target == 1 and len(pool.live()) == 1


# -- adoption ------------------------------------------------------------------------


def test_adopts_an_unowned_machine_of_the_run_instead_of_ordering():
    clock = Clock()
    a, b, multi = two_providers(clock)
    b.machines["daytona-orphan"] = {"slot": 2, "state": "active", "lease_until": None, "end_reason": None}
    pool, route, _ = make_pool(multi, multi.slots(), target=1, clock=clock)
    ticks(pool, 3, clock)
    assert [m.ticket for m in pool.live()] == ["daytona-orphan"]
    assert pool.live()[0].account == "daytona:2" and pool.live()[0].state == READY
    assert a.orders == [] and b.orders == []
    assert "adopted" in names(pool)


def test_adopt_refused_past_target_is_drained():
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1, 2], target=1, clock=clock)
    ticks(pool, 3, clock)
    nc.machines["nc-extra"] = {"slot": 2, "state": "active", "lease_until": None, "end_reason": None}
    ticks(pool, 1, clock)
    assert "adopt_refused" in names(pool) and "nc-extra" in nc.released


# -- breaker and budget ---------------------------------------------------------------


def test_breaker_opens_after_consecutive_order_failures():
    clock = Clock()
    nc = FakeProvider("nc", clock)
    nc.fail_orders = 3
    pool, *_ = make_pool(nc, [1], target=1, clock=clock, elastic=quiet_elastic(breaker_failures=3))
    ticks(pool, 3, clock, step=1)
    assert names(pool).count("order_failed") == 3 and "breaker_open" in names(pool)
    ticks(pool, 2, clock, step=1)
    assert nc.orders == []  # cooling down
    clock.t += 10_000
    ticks(pool, 3, clock)
    assert nc.orders == [1] and pool.live()[0].state == READY


def test_external_budget_gate_refuses_orders():
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1], target=1, clock=clock)
    pool.gate = lambda: "run budget spent"
    ticks(pool, 2, clock)
    assert nc.orders == [] and any(e.get("why") == "run budget spent" for e in pool.events.events)


# -- the controller --------------------------------------------------------------------


class FakeFleetLog:
    def __init__(self):
        self.rows = []

    def event(self, kind, name, **fields):
        self.rows.append((kind, name, fields))


class SyncThread:
    """Runs nothing: the tests tick the controller themselves."""

    def __init__(self, target, name, daemon):
        self.target = target

    def start(self):
        pass

    def join(self, timeout=None):
        pass


def test_controller_stop_flag_drains_every_pool_and_forwards_events(tmp_path):
    clock = Clock()
    a, b = FakeProvider("nc", clock), FakeProvider("daytona", clock)
    pa, *_ = make_pool(a, [1], target=1, clock=clock)
    pb, *_ = make_pool(b, [1], target=1, clock=clock)
    log = FakeFleetLog()
    flag = tmp_path / "STOP"
    ctl = rc.RolloutController({"nc": rc.PoolSlot(pa, state_path=tmp_path / "a.json"), "daytona": pb},
                               stop_flag=flag, fleet_log=log, thread_cls=SyncThread, log=lambda _m: None,
                               sleep=lambda _s: None)
    ctl.start()
    assert ctl in rc.active()
    for _ in range(3):
        ctl.tick()
        clock.t += 60
    assert all(m.state == READY for p in ctl.pools.values() for m in p.live())
    assert json.loads((tmp_path / "a.json").read_text())["members"]
    flag.write_text("stop")
    ctl.tick()
    assert ctl.stopped and not pa.live() and not pb.live()
    assert a.released and b.released and ctl not in rc.active()
    kinds = {f["pool_event"] for k, _n, f in log.rows if k == "rollout_event"}
    assert {"nc_elastic_member_ready", "nc_elastic_pool_stopped"} <= kinds
    assert ctl.status()["pools"]["nc"]["stopped"]


def test_controller_budget_gates_every_pool():
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1], target=1, clock=clock)
    ctl = rc.RolloutController({"p": pool}, budget=lambda: "CostGuard: run budget reached",
                               thread_cls=SyncThread, log=lambda _m: None)
    ctl.tick()
    assert nc.orders == []
    ctl.stop("test")


def test_controller_thread_runs_ticks_and_stops():
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1], target=1, clock=clock)
    ctl = rc.RolloutController({"p": pool}, interval=0.01, log=lambda _m: None, sleep=lambda _s: None)
    ctl.start()
    import time

    deadline = time.time() + 5
    while time.time() < deadline and not (pool.live() and pool.live()[0].state == READY):
        time.sleep(0.01)
    assert pool.live()[0].state == READY
    left = ctl.stop("done")
    assert left == {"p": []} and nc.released and ctl.check() is None


def test_fleet_controller_drains_rollout_when_the_fleet_ends():
    from yeto.launch.fleet import FleetController

    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1], target=1, clock=clock)
    ctl = rc.RolloutController({"p": pool}, thread_cls=SyncThread, log=lambda _m: None, sleep=lambda _s: None)
    ticks(pool, 3, clock)

    class Ops:
        def sleep(self, _s):
            pass

        def now(self):
            return 0.0

    fleet = FleetController(learners={}, syncer=None, sky_ops=Ops(), poll_interval=0, recover_timeout=0,
                            rollout=ctl)
    with pytest.raises(RuntimeError):  # no learner: "all learners abandoned"
        fleet.run()
    assert ctl.stopped == "run ended: RuntimeError" and not pool.live() and nc.released


# -- registry ----------------------------------------------------------------------------


def test_registry_builtin_kinds_and_ledger_dispatch():
    assert providers.kinds() == ["daytona", "nc"]
    assert providers.is_national({"accounts": [1]}) and not providers.is_national({"provider": {"kind": "daytona"}})
    assert providers.pool_owned_ledger_kinds() == frozenset({"nc_lease"})
    with pytest.raises(ValueError, match="'nc' and 'daytona' are built in"):
        providers.load({"provider": {"kind": "nope"}}, "r1")


def test_registry_custom_kind_and_factory_override():
    made = []
    entry = providers.register(providers.ProviderKind(
        "fake-test", lambda cfg, run, section: made.append((run, section)) or "P"))
    try:
        assert providers.load({"provider": {"kind": "fake-test", "x": 1}}, "r1") == "P"
        assert made == [("r1", {"x": 1})]
        assert providers.load({"provider": {"kind": "fake-test"}}, "r2",
                              factories={"fake-test": lambda c, r, s: "Q"}) == "Q"
        with pytest.raises(ValueError, match="already registered"):
            providers.register(entry)
    finally:
        providers._REGISTRY.pop("fake-test")  # noqa: SLF001


def test_multi_provider_rejects_unknown_slots():
    clock = Clock()
    multi = MultiProvider({"nc": FakeProvider("nc", clock)}, {"nc": [1]})
    with pytest.raises(KeyError):
        multi.order("sky:1")
    with pytest.raises(ValueError):
        MultiProvider({"a:b": FakeProvider("x", clock)}, {})


def test_cleanup_dispatches_nc_lease_through_the_registry(tmp_path, monkeypatch):
    from yeto.cost import cleanup
    from yeto.cost.ledger import ResourceLedger

    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    ledger.created("nc_lease", "T1", "smp-0", account=4)

    class Ops:
        calls = []

        def nc_release(self, account, ticket):
            self.calls.append((account, ticket))
            return {"ended": True}

    ops = Ops()
    actions = cleanup.cleanup(ledger, ops, dry_run=True)
    assert actions[0].action == "would release NC ticket T1 (account 4)" and ops.calls == []
    actions = cleanup.cleanup(ledger, ops)
    assert ops.calls == [(4, "T1")] and actions[0].ok and not ledger.open_resources()


# -- a pool.json in the controller (the subprocess pool loop's files and lock) ----------


def test_nc_pool_of_a_pool_json_runs_in_the_controller_and_shares_the_lock(tmp_path, monkeypatch):
    import subprocess

    from tests.test_nc_elastic import Clock as NcClock
    from tests.test_nc_elastic import Cloud
    from tests.test_nc_pool_run import _cfg
    from yeto.launch.nc import cli, pool_run

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4")
    monkeypatch.setenv("NC_MANAGE", "true")
    monkeypatch.setenv("NC_QUERY", "true")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    cfg, path = _cfg(tmp_path)
    clock = NcClock()
    cloud = Cloud(clock)
    slot = rc.nc_pool(cfg, path, ops=cloud)
    slot.pool.clock = clock
    assert pool_run.lock_holder(pool_run.lock_path(cfg))["role"] == "controller"
    with pytest.raises(SystemExit, match="another instance holds"):
        pool_run.run_pool(cfg, path, interval=0)
    ctl = rc.RolloutController({"nc": slot}, thread_cls=SyncThread, log=lambda _m: None, sleep=lambda _s: None)
    for _ in range(40):
        ctl.tick()
        clock.t += 120
    assert slot.pool.live() and slot.pool.live()[0].state == READY
    assert json.loads(cli.pool_state_path(cfg).read_text())["members"]
    assert pool_run.pool_process(cfg)["alive"]  # pool-status / pool-scale see the loop
    pool_run.stop_path(cfg).write_text("stop")  # what pool-stop writes
    ctl.tick()
    assert slot.pool.stopped == "pool-stop requested" and not slot.pool.live()
    left = ctl.stop("test end")
    assert left == {"nc": []}
    assert pool_run.lock_holder(pool_run.lock_path(cfg)) is None
    assert not pool_run.pid_path(cfg).exists() and not pool_run.stop_path(cfg).exists()


# -- data paths ----------------------------------------------------------------------------


def test_r2_data_path_fences_the_lease_and_retires_the_sampler(monkeypatch):
    from yeto.launch.nc import elastic
    from yeto.launch.pool.data_path import R2DataPath
    from yeto.rl.engine import store_dispense

    calls = []
    monkeypatch.setattr(elastic, "fence_shard", lambda store, shard, by: calls.append(("fence", shard, by)))
    monkeypatch.setattr(store_dispense, "retire",
                        lambda objects, ns, holder, incarnation=None, reason="": calls.append(("retire", ns, holder)))
    path = R2DataPath(lambda: "records", lambda: ("objects", "dispense"))
    path.fence_shard("smp-1", by="t1")
    assert calls == [("fence", "smp-1", "t1"), ("retire", "dispense", "smp-1")]
    calls.clear()
    R2DataPath("records", None).fence_shard("smp-2", by="t2")  # no dispenser: the lease fence only
    assert calls == [("fence", "smp-2", "t2")]


def test_syncer_data_path_drains_then_retires_in_the_island_pool(tmp_path):
    from yeto.launch.pool.data_path import SyncerDataPath

    status: dict[int, str] = {}
    calls = []

    def retire_member(address, holder, incarnation, *, reason, key, timeout_s, mode):
        calls.append((mode, holder))
        if mode == "drain":
            status.setdefault(holder, "draining")
        return {"status": status.get(holder, "retired") if mode == "drain" else mode}

    path = SyncerDataPath(("syncer.invalid", 1), retire_member=retire_member, holder_id=lambda shard: hash(shard))
    clock = Clock()
    nc = FakeProvider("nc", clock)
    pool, *_ = make_pool(nc, [1, 2], target=2, clock=clock, fencer=path)
    pool.scale_path, pool.scale_run = tmp_path / "scale.json", "r1"
    ticks(pool, 3, clock)
    victim = pool.live()[1]
    write_scale(pool.scale_path, 1, 1, retire=[victim.ticket])
    ticks(pool, 2, clock)
    assert ("drain", hash(victim.shard)) in calls and victim.state != GONE
    status[hash(victim.shard)] = "drained"
    ticks(pool, 2, clock)
    assert victim.state == GONE and ("retire", hash(victim.shard)) in calls
    assert calls.index(("retire", hash(victim.shard))) > calls.index(("drain", hash(victim.shard)))
