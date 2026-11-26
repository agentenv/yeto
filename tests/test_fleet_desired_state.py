"""The fleet's desired state and its reconcile (local store, fake providers; no cloud):
conditional generations, pools added / removed / rescaled / reconfigured while the run
trains, ``yeto run scale`` and ``yeto fleet apply`` on the store control plane."""

from __future__ import annotations

import json
import threading

import pytest

from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute, quiet_elastic
from yeto.launch.pool import controller as rc
from yeto.launch.pool import desired as dz
from yeto.launch.pool import store_state
from yeto.launch.pool.fleet_reconcile import FleetReconciler, ReconcileHooks
from yeto.launch.pool.island_pool import GONE, READY, IslandPool, PoolConfig
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.launch.pool.status import EventLog
from yeto.rl.engine import member_table as mt
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


def _member(name, target, **pool):
    return {"kind": "sampler_pool", "name": name, "pool": {"provider": "nc", "target": target, **pool}}


# -- the document and its generations --------------------------------------------------

def test_concurrent_changes_from_one_generation_the_later_is_refused(tmp_path):
    store = dz.DesiredStore(DirectoryObjectStore(tmp_path))
    first = store.apply({"members": [_member("nc", 2)]}, expect=0)
    assert first["generation"] == 1 and store.read()["generation"] == 1
    a = json.loads(json.dumps(store.read()))
    b = json.loads(json.dumps(store.read()))
    a["members"][0]["pool"]["target"] = 3
    b["members"][0]["pool"]["target"] = 1
    assert store.apply(a, expect=1)["generation"] == 2
    with pytest.raises(dz.DesiredConflict, match="generation 2, not 1"):
        store.apply(b, expect=1)
    assert dz.pool_target(dz.sampler_pools(store.read())["nc"]) == 3
    assert json.loads((tmp_path / dz.DESIRED_KEY).read_text())["generation"] == 2  # the convenience copy


def test_racing_writers_get_one_generation_each(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    dz.DesiredStore(objects).apply({"members": [_member("nc", 0)]}, expect=0)
    won, lost = [], []

    def add_one():
        try:
            won.append(dz.DesiredStore(objects).apply({"members": [_member("nc", 5)]}, expect=1)["generation"])
        except dz.DesiredConflict:
            lost.append(1)

    threads = [threading.Thread(target=add_one) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert won == [2] and len(lost) == 5


def test_relative_updates_retry_on_a_conflict(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    store, other = dz.DesiredStore(objects), dz.DesiredStore(objects)
    store.apply({"members": [_member("nc", 1)]}, expect=0)
    races = [1]

    def plus_one(doc):
        if races:  # another writer lands between this read and this write
            races.pop()
            other.update(lambda d: d["members"][0]["pool"].update(target=4))
        doc["members"][0]["pool"]["target"] += 1

    with pytest.raises(dz.DesiredConflict):
        store.update(plus_one)  # one attempt: refused
    races.append(1)
    assert store.update(plus_one, attempts=3)["members"][0]["pool"]["target"] == 5  # read again: 4 + 1


def test_the_schema_reserves_trainer_members():
    doc = dz.validate({"members": [_member("nc", 1),
                                   {"kind": "trainer", "name": "island-1", "learner_id": 1,
                                    "source": {"kind": "modal", "gpu": "l40s"}, "shape": {"nodes": 1}}]})
    assert list(dz.trainers(doc)) == ["island-1"] and list(dz.sampler_pools(doc)) == ["nc"]
    for bad, why in (({"members": [{"kind": "lora", "name": "x"}]}, "kind"),
                     ({"members": [{"kind": "trainer", "name": "t"}]}, "learner_id"),
                     ({"members": [_member("a", -1)]}, "target"),
                     ({"members": [_member("a", 1), _member("a", 2)]}, "two"),
                     ({"schema": "other/v9", "members": []}, "schema")):
        with pytest.raises(dz.DesiredError, match=why):
            dz.validate(bad)


def test_from_run_config_names_the_pools_like_run_up():
    doc = dz.from_run_config({"run": "r", "samplers": [{"name": "a", "provider": "nc", "config": "p.json",
                                                         "required": False}, {"provider": "modal"}]})
    assert [(m["name"], m["required"]) for m in doc["members"]] == [("a", False), ("modal", True)]
    assert dz.from_run_config({"pool": {"config": "p.json"}})["members"][0]["name"] == "pool"


# -- the reconcile ---------------------------------------------------------------------

class _ManualThread:
    def __init__(self, target=None, args=(), name=None, daemon=None):
        pass

    def start(self):
        pass

    def is_alive(self):
        return True

    def join(self, timeout=None):
        pass


class LeaveFencer(FakeFencer):
    """The work contract over the samplers' member records in the run store (R2DataPath)."""

    def __init__(self, objects, clock, *, lease=True):
        super().__init__()
        from yeto.launch.nc.elastic import ObjectRecords  # noqa: PLC0415
        from yeto.launch.pool.data_path import R2DataPath  # noqa: PLC0415

        self.path = R2DataPath(ObjectRecords(objects))
        self.clock, self.lease = clock, lease

    def fence_shard(self, shard, by):
        from yeto.launch.nc import elastic  # noqa: PLC0415

        super().fence_shard(shard, by)
        if self.lease:  # the pool fence's lease, with its time (False: the store keeps no fence time)
            return elastic.fence_shard(self.path.records, shard, by=by, clock=self.clock)
        return None

    def exit_evidence(self, shard):
        return self.path.exit_evidence(shard)

    def write_leave(self, shard, reason, **detail):
        return self.path.write_leave(shard, reason, **detail)


class Fleet:
    """A controller with a reconciler over pools on fake providers and one local store
    (the run store and the trainer's buffer store alike)."""

    def __init__(self, tmp_path, members, *, slots=3, data_path=False, leave_records=False, fence_lease=True):
        self.objects = DirectoryObjectStore(tmp_path / "store")
        self.data_path = data_path  # the pools' data path writes the store (in-shard end records)
        self.leave_records = leave_records  # the fencer reads and writes the samplers' member records
        self.fence_lease = fence_lease
        self.clock = Clock()
        self.store = dz.DesiredStore(self.objects)
        self.slots_n = slots
        self.providers: dict[str, FakeProvider] = {}
        self.routes: dict[str, list[FakeRoute]] = {}
        self.fencers: dict[str, FakeFencer] = {}
        self.ready_low: dict[str, int] = {}
        doc = self.store.apply({"members": members}, expect=0)
        pools = {name: self.build(name, m) for name, m in dz.sampler_pools(doc).items()}
        self.reconciler = FleetReconciler(self.store, ReconcileHooks(add=self.build, reconfigure=self.reconfigure),
                                          poll_s=0, log=lambda _m: None)
        self.reconciler.start_from(doc)
        self.c = rc.RolloutController(pools, interval=60, log=lambda _m: None, thread_cls=_ManualThread,
                                      reconciler=self.reconciler, drain_wait_s=0, sleep=lambda _s: None)

    def build(self, name, member):
        control = store_state.open_pool_control(self.objects, name, name, clock=self.clock)
        control.members = mt.MemberTableWriter(self.objects, name, name)
        self.providers[name] = FakeProvider(name, self.clock)
        self.routes[name] = [FakeRoute()]
        self.fencers[name] = LeaveFencer(self.objects, self.clock, lease=self.fence_lease) if self.leave_records \
            else FakeFencer()
        ops = ProviderPoolOps(self.providers[name], self.routes[name][-1], self.fencers[name])
        if self.data_path:
            from yeto.launch.nc.elastic import ObjectRecords  # noqa: PLC0415
            from yeto.launch.pool.data_path import R2DataPath  # noqa: PLC0415

            path = R2DataPath(ObjectRecords(self.objects))
            ops.data_path = lambda: path
        config = PoolConfig(target=dz.pool_target(member) or 0, accounts=list(range(self.slots_n)), shard_base=name,
                            elastic=quiet_elastic())
        return IslandPool.from_doc(None, config, ops, control=control, clock=self.clock, log=lambda _m: None,
                                   events=EventLog(log=lambda _m: None, clock=self.clock))

    def reconfigure(self, name, slot, member):
        route = FakeRoute()
        self.routes[name].append(route)
        from yeto.launch.pool.island_pool import GuardedOps  # noqa: PLC0415

        slot.pool.ops = GuardedOps(ProviderPoolOps(self.providers[name], route, self.fencers[name]),
                                   slot.pool._guard_store)  # noqa: SLF001

    def restart_pool(self, name):
        """Detach the controller, then build the pool again from its store state (a controller
        restarted, or one that adopts the pool): no tick of the old pool object runs any more."""
        old = self.c.slots[name].pool
        self.c.detach("test restart")
        if not old.control.lock.released:
            old.control.lock.release()  # as a detach of a pool loop of its own does
        doc = store_state.read_state(self.objects, name)
        control = store_state.open_pool_control(self.objects, name, name, clock=self.clock)
        control.members = mt.MemberTableWriter(self.objects, name, name)
        ops = ProviderPoolOps(self.providers[name], self.routes[name][-1], self.fencers[name])
        config = PoolConfig(target=0, accounts=list(range(self.slots_n)), shard_base=name, elastic=quiet_elastic())
        return IslandPool.from_doc(doc, config, ops, control=control, clock=self.clock, log=lambda _m: None,
                                   events=EventLog(log=lambda _m: None, clock=self.clock))

    def change(self, fn):
        self.store.update(fn)

    def ticks(self, n=3, watch=None):
        for _ in range(n):
            self.c.tick()
            self.clock.t += 30
            for name, slot in self.c.slots.items():
                ready = len([m for m in slot.pool.live() if m.state == READY])
                self.ready_low[name] = min(self.ready_low.get(name, ready), ready)

    def shards(self, name):
        return sorted(m.shard for m in self.c.slots[name].pool.live())


def test_a_pool_added_while_the_run_trains_is_consumed_without_a_trainer_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(mt.MEMBER_TABLE_ENV, "1")
    monkeypatch.setenv(mt.MEMBER_LANES_ENV, "*")  # run up sets it for a single trainer (its store is its lane)
    f = Fleet(tmp_path, [_member("nc", 1)])
    f.ticks()
    trainer = FileSampleBuffer(f.objects, batch_size=1, merge_shards=True, expected_shards=["nc-0"])
    trainer._members.refresh_s = 0  # noqa: SLF001
    f.change(lambda d: d["members"].append(_member("modal", 2)))
    f.ticks()
    assert f.shards("modal") == ["modal-0", "modal-1"] and f.shards("nc") == ["nc-0"]
    assert any(e["event"].endswith("pool_added") for e in f.c.slots["modal"].pool.events.events)
    trainer.refresh()
    assert {"modal-0", "modal-1"} <= set(trainer._expected_shards)  # noqa: SLF001 - no restart, no foreign warning


def test_a_target_change_in_the_desired_state_scales_the_pool(tmp_path):
    f = Fleet(tmp_path, [_member("nc", 1)])
    f.ticks()
    f.change(lambda d: d["members"][0]["pool"].update(target=3))
    f.ticks()
    assert len(f.shards("nc")) == 3
    f.c.slots["nc"].pool.set_target(2)  # the pool's own change (a manual release)...
    f.ticks()
    assert f.c.slots["nc"].pool.config.target == 2  # ...is not undone: only a new generation is applied


def test_a_removed_pool_drains_retires_its_shards_and_releases_every_machine(tmp_path, monkeypatch):
    monkeypatch.setenv(mt.MEMBER_TABLE_ENV, "1")
    f = Fleet(tmp_path, [_member("nc", 1), _member("old", 2)], data_path=True)
    f.ticks()
    old_pool = f.c.slots["old"].pool
    control = old_pool.control
    control.allocator.allocate(token=control.lock.token, base="old")  # a number issued, never ordered: old-2
    shards = f.shards("old")
    assert shards == ["old-0", "old-1"]
    f.change(lambda d: d["members"].pop(1))
    f.ticks(6)
    assert "old" not in f.c.slots and old_pool.stopped
    prov = f.providers["old"]
    assert sorted(prov.released) == sorted(prov.machines) and len(prov.orders) == 2  # all released, no extra order
    assert {s for s, _ in f.fencers["old"].fenced} == set(shards)  # fenced: their claims went back
    for shard in ("old-0", "old-1", "old-2"):
        assert control.allocator.ended(shard) is not None  # never issued again
    table = json.loads(f.objects.read(mt.table_key("old")))
    assert set(table["retired"]) == {"old-0", "old-1", "old-2"}
    from yeto.rl.engine.sample_buffer import shard_index as _index  # noqa: PLC0415

    # REV #758: the never ordered number got its in-shard data end before its end record
    assert FileSampleBuffer(f.objects, batch_size=1, index=_index("old-2")).data_ended
    applied = f.store.read_applied()
    assert applied["generation"] == 2 and applied["removing"] == []  # the removal is recorded as done
    # the trainer's data end does not wait for the removed pool's shards
    trainer = FileSampleBuffer(f.objects, batch_size=1, merge_shards=True,
                               expected_shards=["old-0", "old-1", "old-2", "nc-0"])
    from yeto.rl.engine.sample_buffer import shard_index  # noqa: PLC0415

    FileSampleBuffer(f.objects, batch_size=1, index=shard_index("nc-0")).mark_data_end()
    trainer.refresh()
    assert trainer.data_ended
    assert len(f.c.slots["nc"].pool.live()) == 1  # the other pool never noticed


@pytest.mark.parametrize("spare", [True, False])
def test_a_config_change_rolls_the_pool_one_member_at_a_time(tmp_path, spare):
    f = Fleet(tmp_path, [_member("nc", 4, spec="v1.json")], slots=5 if spare else 4)
    f.ticks(3)
    old = set(m.ticket for m in f.c.slots["nc"].pool.live())
    assert len(old) == 4
    f.ready_low.clear()
    f.change(lambda d: d["members"][0]["pool"].update(spec="v2.json"))
    f.ticks(30)
    pool = f.c.slots["nc"].pool
    live = pool.live()
    assert len(live) == 4 and not (set(m.ticket for m in live) & old)  # every member replaced
    new_route = f.routes["nc"][-1]
    assert sorted(t for t, *_ in new_route.started) == sorted(m.ticket for m in live)  # all on the new config
    assert f.ready_low["nc"] >= 3  # never more than one short
    assert not pool.roll_tickets and any(e["event"].endswith("roll_done") for e in pool.events.events)
    replaced = [m for m in pool.members if m.state == GONE]
    assert len(replaced) == 4


def test_a_lost_lock_while_removing_detaches_and_ends_nothing(tmp_path):
    """REV #758: another controller took the removed pool's lock: this one detaches it (no
    drain, no end record) and keeps the removal in fleet/applied.json for the new holder."""
    f = Fleet(tmp_path, [_member("nc", 1), _member("old", 2)])
    f.ticks()
    old_pool = f.c.slots["old"].pool
    f.change(lambda d: d["members"].pop(1))
    f.clock.t += 601
    store_state.open_pool_control(f.objects, "old", "old", clock=f.clock)  # the lease ran out: taken over
    f.ticks(4)
    assert old_pool.detached and not old_pool.stopped
    assert f.providers["old"].released == []  # nothing drained from here
    assert all(old_pool.control.allocator.ended(s) is None for s in ("old-0", "old-1"))
    assert "old" in f.reconciler.removing
    assert [m["name"] for m in f.store.read_applied()["removing"]] == ["old"]
    assert len(f.c.slots["nc"].pool.live()) == 1


def test_a_removal_waits_for_a_busy_tick_lock_and_never_blocks_the_reconcile(tmp_path):
    f = Fleet(tmp_path, [_member("nc", 1), _member("old", 1)])
    f.ticks()
    f.change(lambda d: d["members"][1]["pool"].update(spec="v2.json"))
    lock = f.c.slots["old"].tick_lock
    lock.acquire()  # a tick of "old" in flight
    try:
        f.reconciler.step(f.c, force=True)  # returns at once: the new config waits for the lock
        assert "old" in f.reconciler.pending_rolls
    finally:
        lock.release()
    f.reconciler.step(f.c)
    assert "old" not in f.reconciler.pending_rolls and f.c.slots["old"].pool.roll_tickets


def test_a_pool_whose_build_fails_is_reported_and_the_others_go_on(tmp_path):
    f = Fleet(tmp_path, [_member("nc", 1)])
    f.reconciler.hooks.add = lambda name, member: (_ for _ in ()).throw(SystemExit("another instance holds it"))
    f.change(lambda d: d["members"].append(_member("bad", 1)))
    f.ticks()
    assert "bad" not in f.c.slots and len(f.shards("nc")) == 1 and f.reconciler.applied == 2


# -- run scale / fleet apply on the store control plane ---------------------------------

def _store_run(tmp_path, monkeypatch, objects):
    from tests.test_rollout_entry import _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up.Run, "_objects", lambda self: objects)
    monkeypatch.setattr(up.Run, "budget_problem", lambda self: None)
    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process", rollout={"control_plane": "store"})
    path = tmp_path / "run.store.json"
    path.write_text(json.dumps(cfg))
    return up, cfg, str(path)


def test_run_scale_writes_the_desired_state(tmp_path, monkeypatch):
    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    run = up.Run(cfg)
    assert run.scale(3, pool="nc") == 0
    doc = dz.DesiredStore(objects).read()
    assert doc["generation"] == 1  # seeded from the run config with the scale in one generation
    assert dz.pool_target(dz.sampler_pools(doc)["nc"]) == 3
    assert run.scale(None, remove=["t9"], pool="nc") == 0
    nc = dz.sampler_pools(dz.DesiredStore(objects).read())["nc"]
    assert nc["retire"] == ["t9"] and dz.pool_target(nc) == 2


def test_run_up_store_mode_drives_the_desired_pools(tmp_path, monkeypatch):
    from tests.test_rollout_entry import _NoThread  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    clock = Clock()
    ops = {name: ProviderPoolOps(FakeProvider(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert c.reconciler is not None and c.reconciler.applied == 1 and set(c.slots) == {"nc", "modal"}
        c.tick()
        store = dz.DesiredStore(objects)
        store.update(lambda d: d["members"].pop())  # remove "modal"
        c.reconciler.poll_s = 0
        for _ in range(4):
            c.tick()
            clock.t += 30
        assert set(c.slots) == {"nc"}
    finally:
        c.stop("test end")


def test_a_pool_the_reconcile_removes_is_retired_into_the_run_spend(tmp_path, monkeypatch):
    """The reconciler's removal hook (the end of a removal) retires the pool through run up:
    it leaves the pools run up drives and is written to the state's retired_pools, so its
    billed hours stay in run.budget (Run.retired_pool_spend), here and in the next up."""
    from tests.test_rollout_entry import _NoThread  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    clock = Clock()
    ops = {name: ProviderPoolOps(FakeProvider(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    run = up.Run(cfg)
    c = run.start_controller(ops=ops, thread_cls=_NoThread)
    try:
        c.tick()
        modal = next(r for r in run.pools if r.name == "modal")
        dz.DesiredStore(objects).update(lambda d: d["members"].pop())  # remove "modal"
        c.reconciler.poll_s = 0
        for _ in range(4):
            c.tick()
            clock.t += 30
        assert set(c.slots) == {"nc"} and [r.name for r in run.pools] == ["nc"]
        (row,) = run.state["retired_pools"]
        assert row["name"] == "modal" and row["shard_base"] == str(modal.load()["shard_base"])
        assert [r.name for r, _at in run.retired_pools()] == ["modal"] and len(run.retired_pool_spend()) == 1
    finally:
        c.stop("test end")
    again = up.Run(cfg)  # the next up: the row is read from the state file
    assert [r["name"] for r in again.state["retired_pools"]] == ["modal"]


def test_desired_changes_made_while_detached_are_applied_by_the_next_controller(tmp_path, monkeypatch):
    """REV #758: the applied generation is kept in the store; a controller started after a detach
    applies what changed meanwhile: a removed pool is drained and retired, a target is applied."""
    from tests.test_rollout_entry import _NoThread  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(providers[name], FakeRoute(), FakeFencer()) for name in providers}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    for _ in range(3):
        c.tick()
        clock.t += 30
    assert all(p.live() for p in c.pools.values())
    c.detach("test")
    store = dz.DesiredStore(objects)
    assert store.read_applied()["generation"] == 1

    def change(d):
        d["members"] = [m for m in d["members"] if m["name"] != "modal"]
        d["members"][0]["pool"]["target"] = 2

    store.update(change)  # generation 2, written while no controller runs
    run = up.Run(cfg)
    again = run.start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert set(again.slots) == {"nc", "modal"}  # the applied pools: modal is still billed
        assert again.reconciler.applied == 1
        again.reconciler.poll_s = 0
        for _ in range(8):
            again.tick()
            clock.t += 30
        assert set(again.slots) == {"nc"} and len(again.pools["nc"].live()) == 2
        assert sorted(providers["modal"].released) == sorted(providers["modal"].machines)
        assert store.read_applied()["generation"] == 2 and store.read_applied()["removing"] == []
        assert [r.name for r in run.pools] == ["nc"]
    finally:
        again.stop("test end")


def test_pools_added_and_removed_while_the_run_trains_call_their_provider_in_the_run_env(tmp_path, monkeypatch):
    """#785 with the reconcile: a pool ``yeto fleet apply`` adds orders under the run config's
    env overlay, and the end of a removal releases under it, like the pools run up started."""
    import os  # noqa: PLC0415

    from tests.test_rollout_entry import _NoThread  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    monkeypatch.setenv("AWS_PROFILE", "shell")
    cfg = dict(cfg, env={"AWS_PROFILE": "run"})
    clock = Clock()
    seen: dict[str, list[tuple[str, str | None]]] = {"nc": [], "modal": []}

    def seeing(name):
        class Seeing(FakeProvider):
            def order(self, slot):
                seen[name].append(("order", os.environ.get("AWS_PROFILE")))
                return super().order(slot)

            def release(self, slot, ticket):
                seen[name].append(("release", os.environ.get("AWS_PROFILE")))
                return super().release(slot, ticket)

        return Seeing(name, clock)

    providers = {name: seeing(name) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(providers[name], FakeRoute(), FakeFencer()) for name in providers}
    store = dz.DesiredStore(objects)
    full = dz.from_run_config(cfg)
    modal = next(m for m in full["members"] if m["name"] == "modal")
    store.apply({**full, "members": [m for m in full["members"] if m["name"] != "modal"]}, expect=0)
    run = up.Run(cfg)
    c = run.start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert set(c.slots) == {"nc"}
        c.reconciler.poll_s = 0
        store.update(lambda d: d["members"].append(modal))
        for _ in range(3):
            c.tick()
            clock.t += 30
        assert set(c.slots) == {"nc", "modal"} and c.slots["modal"].environ == run.pool_environ()
        assert os.environ.get("AWS_PROFILE") == "shell"  # outside a pool call: the shell's
        assert [w for w, _ in seen["modal"]] and all(env == "run" for _, env in seen["modal"]), seen
        store.update(lambda d: d["members"].remove(modal))
        for _ in range(8):
            c.tick()
            clock.t += 30
        assert set(c.slots) == {"nc"}
        assert sorted(providers["modal"].released) == sorted(providers["modal"].machines)
        assert ("release", "run") in seen["modal"] and all(env == "run" for _, env in seen["modal"]), seen
    finally:
        c.stop("test end")
    assert all(env == "run" for env_list in seen.values() for _, env in env_list), seen
    assert os.environ.get("AWS_PROFILE") == "shell"


def test_the_end_of_a_removal_waits_while_another_env_overlay_is_active(tmp_path):
    """The end of a removal (its releases) never runs under another overlay's values: it
    is skipped like a tick and asked again once the other overlay leaves."""
    from yeto.launch.pool.env_scope import scoped_environ  # noqa: PLC0415

    f = Fleet(tmp_path, [_member("nc", 1), _member("modal", 1)])
    f.ticks()
    slot = f.c.slots["modal"]
    slot.environ = {"AWS_PROFILE": "run"}
    f.change(lambda d: d["members"].pop())  # remove "modal"
    for _ in range(10):  # the reconcile hands the end of the removal to the pool's worker
        f.c.reconciler.step(f.c)
        if slot.pending_finish is not None:
            break
        f.ticks(1)
    assert slot.pending_finish is not None
    released = list(f.providers["modal"].released)
    with scoped_environ({"AWS_PROFILE": "other"}):
        f.c._tick_slot("modal", slot)  # noqa: SLF001 - the worker's tick of this pool
        assert slot.pending_finish is not None and f.providers["modal"].released == released
        assert "modal" in f.c.slots and f.c.check() is None
        # bounded: skipped N times in a row, the pool is reported (a CRITICAL event, check(),
        # which the run layer takes as run.controller_exit; run down releases what is left)
        for _ in range(rc.ENV_CONFLICT_TICKS_N - 1):
            f.c._tick_slot("modal", slot)  # noqa: SLF001
        assert f.providers["modal"].released == released
        assert "modal" in (f.c.check() or "")
        assert any(e["event"].endswith("env_overlay_conflict") and e.get("critical")
                   for e in slot.pool.events.events)
    f.ticks(6)
    assert set(f.c.slots) == {"nc"} and f.c.check() is None
    assert sorted(f.providers["modal"].released) == sorted(f.providers["modal"].machines)


def _env_run(tmp_path, monkeypatch, attach=None):
    """A store-mode run with a run config env over the shell's, and pools on providers
    that record the environment their calls see."""
    import os  # noqa: PLC0415

    from tests.test_rollout_entry import _NoThread  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    monkeypatch.setenv("AWS_PROFILE", "shell")
    cfg = dict(cfg, env={"AWS_PROFILE": "run"})
    clock = Clock()
    seen: list[str | None] = []

    class Seeing(FakeProvider):
        def balance(self):
            seen.append(os.environ.get("AWS_PROFILE"))
            return super().balance()

    class Ops(ProviderPoolOps):
        def attach_control(self, control):
            if attach is not None:
                attach.append(control)

    ops = {name: Ops(Seeing(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    run = up.Run(cfg)
    c = run.start_controller(ops=ops, thread_cls=_NoThread)
    c.reconciler.poll_s = 0
    return up, run, c, dz.DesiredStore(objects), clock, seen


def _usd(d, value):
    next(m for m in d["members"] if m["name"] == "modal")["pool"]["usd_per_h"] = value


def test_a_reconfigured_pool_calls_its_new_provider_in_the_run_env_from_any_thread(tmp_path, monkeypatch):
    """REV #783: the new ops of a pool `yeto fleet apply` reconfigured run every call under the
    run's env overlay, a background start or release on another thread included."""
    import os  # noqa: PLC0415

    attached: list = []
    _up, _run, c, store, clock, seen = _env_run(tmp_path, monkeypatch, attach=attached)
    try:
        c.tick()
        old_ops = c.slots["modal"].pool.ops
        attached.clear()
        store.update(lambda d: _usd(d, 60))
        c.tick()
        pool = c.slots["modal"].pool
        assert pool.ops is not old_ops and pool.roll_tickets  # reconfigured: rolling
        assert attached == [pool.control]  # the provider's own record comes from the store, as at start
        seen.clear()
        t = threading.Thread(target=pool.ops.balance)  # outside any pool scope, like a background release
        t.start()
        t.join()
        assert seen == ["run"] and os.environ.get("AWS_PROFILE") == "shell"
    finally:
        c.stop("test end")


def test_a_reconfigure_a_sky_start_check_refuses_keeps_the_old_config(tmp_path, monkeypatch):
    """#786's start check runs for a reconfigured pool too, in the run's env overlay as at
    its start (AWS_PROFILE is set only in run.json); its refusal (a SystemExit) is a refused
    reconfigure, never the end of the controller."""
    import os  # noqa: PLC0415

    from yeto.launch.nc import pool_run  # noqa: PLC0415

    _up, _run, c, store, clock, seen = _env_run(tmp_path, monkeypatch)
    try:
        c.tick()
        old_ops, old_config = c.slots["modal"].pool.ops, c.slots["modal"].pool.config

        def refuse(cfg):  # REV #814: refuses only with the run's AWS_PROFILE, set in run.json alone
            if os.environ.get("AWS_PROFILE") == "run":
                raise SystemExit(f"pool {cfg.get('shard_base')}: refused: run AWS settings sky would use")

        monkeypatch.setattr(pool_run, "refuse_provider_start", refuse)
        store.update(lambda d: _usd(d, 61))
        c.tick()
        pool = c.slots["modal"].pool
        assert pool.ops is old_ops and pool.config is old_config and not pool.roll_tickets
        assert any(e["event"].endswith("reconfigure_refused") and "sky would use" in e.get("error", "")
                   for e in pool.events.events)
        assert c.check() is None and set(c.slots) == {"nc", "modal"}
    finally:
        c.stop("test end")


def test_a_reconfigure_waits_for_the_background_calls_of_the_old_ops(tmp_path):
    """REV #814: a background start or release submitted through the old ops (an ssh
    provider's record under the same lock token) is never still running when the new ops
    are swapped in: the reconfigure waits for a later pass."""
    from concurrent.futures import Future  # noqa: PLC0415

    f = Fleet(tmp_path, [_member("nc", 1)])
    f.ticks()
    pool = f.c.slots["nc"].pool
    running = Future()
    pool._starts["t-bg"] = running  # noqa: SLF001
    f.change(lambda d: dz.sampler_pools(d)["nc"]["pool"].update(image="new"))
    f.c.reconciler.step(f.c)
    assert len(f.routes["nc"]) == 1 and not pool.roll_tickets and "nc" in f.c.reconciler.pending_rolls
    from yeto.launch.pool import fleet_reconcile as fr  # noqa: PLC0415

    t = [0.0]
    f.c.reconciler.clock = lambda: t[0]
    f.c.reconciler._held_since["nc"] = 0.0  # noqa: SLF001 - held since t=0
    t[0] = fr.RECONFIGURE_WAIT_ALERT_S - 1
    f.c.reconciler.step(f.c)
    waiting = lambda: [e for e in pool.events.events if e["event"].endswith("reconfigure_waiting")]  # noqa: E731
    assert not waiting()
    for _ in range(2):  # bounded: reported once, still never swapped while the call runs
        t[0] += 60
        f.c.reconciler.step(f.c)
    assert len(waiting()) == 1 and waiting()[0]["tickets"] == ["t-bg"] and len(f.routes["nc"]) == 1
    running.set_result(None)
    f.c.reconciler.step(f.c)
    assert len(f.routes["nc"]) == 2 and pool.roll_tickets and "nc" not in f.c.reconciler.pending_rolls
    pool._starts.pop("t-bg")  # noqa: SLF001


def test_a_roll_never_replaces_a_member_that_is_already_going(tmp_path):
    """#790: a member draining away (a move to another pool lowers the target by one and
    drains it) is not picked by a rolling replacement; the next old member is."""
    f = Fleet(tmp_path, [_member("nc", 2)])
    f.ticks(4)
    pool = f.c.slots["nc"].pool
    going, other = sorted(pool.live(), key=lambda m: m.ticket)
    assert going.state == READY and other.state == READY
    pool.start_roll("test")
    going.retiring_since = f.clock()
    pool.config.target = 1
    pool._roll()  # noqa: SLF001
    steps = [e for e in pool.events.events if e["event"].endswith("roll_step")]
    assert steps and steps[-1]["ticket"] == other.ticket


def _joined(f, pool_name, **fields):
    """The sampler of the pool's one member joined (its member record in the run store)."""
    from yeto.launch.nc.elastic import ObjectRecords  # noqa: PLC0415
    from yeto.rl.engine import sampler_resume as sr  # noqa: PLC0415

    (m,) = f.c.slots[pool_name].pool.live()
    member = sr.Member(sampler=m.shard, incarnation=f"inc-{m.ticket}", joined_at=1.0, resume={}, **fields)
    sr.write_member(ObjectRecords(f.objects), member)
    return m, member


def _record(f, member):
    from yeto.rl.engine import sampler_resume as sr  # noqa: PLC0415

    return json.loads(f.objects.read(sr.member_key(member)))


def _until_waiting(f, pool, ticket, n=10):
    for _ in range(n):
        if ticket in pool._leave_wait:  # noqa: SLF001
            return f.clock()
        f.ticks(1)
    raise AssertionError("the member was never fenced into its leave wait")


def test_a_removed_pools_sampler_that_writes_no_leave_gets_one_from_the_pool(tmp_path):
    """INT2 r1: a removed pool released its machine in the tick it fenced it, before the
    sampler saw the fence: no leave in its member record (the dashboard showed it stale).
    The pool now waits for the leave, then writes it (POOL_REMOVED) and releases."""
    from yeto.launch.pool.island_pool import POOL_REMOVED_LEAVE  # noqa: PLC0415

    f = Fleet(tmp_path, [_member("nc", 1), _member("modal", 1)], leave_records=True)
    f.ticks()
    m, member = _joined(f, "modal")
    pool = f.c.slots["modal"].pool
    f.change(lambda d: d["members"].pop())  # remove "modal"
    began = _until_waiting(f, pool, m.ticket)
    assert m.shard in [s for s, _ in f.fencers["modal"].fenced]  # fenced first
    while f.clock() < began + pool.removal_leave_wait_s:
        assert f.providers["modal"].released == [] and _record(f, member)["leave_reason"] is None
        f.ticks(1)
    f.ticks(6)
    assert set(f.c.slots) == {"nc"} and f.providers["modal"].released == [m.ticket]
    rec = _record(f, member)
    assert rec["leave_reason"] == POOL_REMOVED_LEAVE and rec["left_at"] and rec["detail"]["left_by"] == "pool"
    assert any(e["event"].endswith("leave_written") for e in pool.events.events)


def test_a_removed_pools_sampler_that_leaves_on_its_fence_is_released_at_once(tmp_path):
    f = Fleet(tmp_path, [_member("nc", 1), _member("modal", 1)], leave_records=True)
    f.ticks()
    m, member = _joined(f, "modal")
    pool = f.c.slots["modal"].pool
    f.change(lambda d: d["members"].pop())
    _until_waiting(f, pool, m.ticket)
    f.ticks(2)  # no leave yet: still not released
    assert f.providers["modal"].released == []
    from yeto.launch.nc.elastic import ObjectRecords  # noqa: PLC0415
    from yeto.rl.engine import sampler_resume as sr  # noqa: PLC0415

    member.left_at, member.leave_reason = 2.0, "FENCED"  # the sampler saw its fence
    sr.write_member(ObjectRecords(f.objects), member)
    f.ticks(2)  # the leave seen: released (in the background: started, then confirmed)
    assert f.providers["modal"].released == [m.ticket]
    assert _record(f, member)["leave_reason"] == "FENCED"
    assert not any(e["event"].endswith("leave_written") for e in pool.events.events)


def test_a_controller_restarted_while_a_removed_pools_member_waits_still_writes_its_leave(tmp_path):
    """Coordinator, INT2's stale member: the wait is not only in memory. A pool rebuilt from
    the store mid-removal (a restart, an adopt after a detach) still waits for the leave of
    a member it fenced, until that fence's time (the store's pool fence) plus the wait, and
    then writes it before the release."""
    from yeto.launch.pool.island_pool import DRAINING, POOL_REMOVED_LEAVE  # noqa: PLC0415

    f = Fleet(tmp_path, [_member("nc", 1), _member("modal", 1)], leave_records=True)
    f.ticks()
    m, member = _joined(f, "modal")
    f.change(lambda d: d["members"].pop())
    _until_waiting(f, f.c.slots["modal"].pool, m.ticket)
    fenced_at = f.fencers["modal"].exit_evidence(m.shard)["fenced_at"]
    pool = f.restart_pool("modal")
    assert pool.removing and not pool._leave_wait  # noqa: SLF001 - restored from the store, nothing in memory
    assert [x.state for x in pool.members if x.ticket == m.ticket] == [DRAINING]
    while f.clock() < fenced_at + pool.removal_leave_wait_s:
        pool.tick()
        assert f.providers["modal"].released == [] and _record(f, member)["leave_reason"] is None
        f.clock.t += 30
    pool.tick()  # the wait counted from the fence, not from the restart: over, so written and released
    assert f.providers["modal"].released == [m.ticket]
    rec = _record(f, member)
    assert rec["leave_reason"] == POOL_REMOVED_LEAVE and rec["detail"]["left_by"] == "pool"


def test_a_restart_without_the_fences_time_waits_the_whole_wait_from_the_restart(tmp_path):
    """The fence's time not in the store (no pool fence lease): the wait counts from the
    restart, neither a release at once nor a wait without end."""
    from yeto.launch.pool.island_pool import POOL_REMOVED_LEAVE  # noqa: PLC0415

    f = Fleet(tmp_path, [_member("nc", 1), _member("modal", 1)], leave_records=True, fence_lease=False)
    f.ticks()
    m, member = _joined(f, "modal")
    f.change(lambda d: d["members"].pop())
    _until_waiting(f, f.c.slots["modal"].pool, m.ticket)
    assert "fenced_at" not in f.fencers["modal"].exit_evidence(m.shard)
    pool = f.restart_pool("modal")
    restarted = f.clock()
    while f.clock() < restarted + pool.removal_leave_wait_s:
        pool.tick()
        assert f.providers["modal"].released == [] and _record(f, member)["leave_reason"] is None
        f.clock.t += 30
    pool.tick()
    assert f.providers["modal"].released == [m.ticket] and _record(f, member)["leave_reason"] == POOL_REMOVED_LEAVE


def test_a_scale_down_outside_a_removal_releases_without_a_leave_wait(tmp_path):
    """Only a removed pool waits: a target change to 0 releases as before (the customer path)."""
    f = Fleet(tmp_path, [_member("nc", 1), _member("modal", 1)], leave_records=True)
    f.ticks()
    m, member = _joined(f, "modal")
    pool = f.c.slots["modal"].pool
    f.change(lambda d: dz.sampler_pools(d)["modal"]["pool"].update(target=0))
    for _ in range(4):
        f.ticks(1)
        assert m.ticket not in pool._leave_wait  # noqa: SLF001
    assert f.providers["modal"].released == [m.ticket] and _record(f, member)["leave_reason"] is None


def test_run_down_stops_every_pool_the_fleet_drives(tmp_path, monkeypatch, capsys):
    """REV #758: run down on the store control plane stops a pool `yeto fleet apply` added and one
    still being removed, and reports a store pool with live members that no config names."""
    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    store = dz.DesiredStore(objects)
    seed = store.apply(dz.from_run_config(cfg), expect=0)
    modal = dz.sampler_pools(seed)["modal"]
    added = {**modal, "name": "modal2", "pool": {**modal["pool"], "shard_base": "mod2"}}
    store.apply({"members": [m for m in seed["members"] if m["name"] != "modal"] + [added]}, expect=1)
    store.write_applied(1, dz.sampler_pools(seed), {})  # the controller still drives "modal" (being removed)
    objects.write(store_state.state_key("ghost"), json.dumps(
        {"members": [{"ticket": "g1", "state": "ready"}]}).encode())
    run = up.Run(cfg)
    calls = []
    monkeypatch.setattr(run, "_cmd", lambda argv, **kw: calls.append([str(a) for a in argv]) or 0)
    monkeypatch.setattr(run, "lock_for_down", lambda *a, **kw: True)
    monkeypatch.setattr(run, "sandbox_hosts", lambda: [])
    assert run.down() is False  # the ghost pool's live member is reported, not silently left
    out = capsys.readouterr().out
    assert "pool ghost of the store holds live member(s) ['g1']" in out
    assert "NOT verified (pool ghost (store state only: 1 live member(s) not stopped))" in out
    stopped = [a[a.index("--config") + 1] for a in calls if "pool-stop" in a]
    assert sorted(ref.name for ref in run.pools) == ["modal", "modal2", "nc"]
    assert sorted(stopped) == sorted(ref.config for ref in run.pools)


def test_fleet_apply_refuses_a_scale_up_the_run_would_refuse(tmp_path, monkeypatch, capsys):
    from yeto.cli import build_parser  # noqa: PLC0415
    from yeto.launch.pool import fleet_cli  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, path = _store_run(tmp_path, monkeypatch, objects)
    store = dz.DesiredStore(objects)
    seed = store.apply(dz.from_run_config(cfg), expect=0)
    monkeypatch.setattr(up.Run, "scale_refusal",
                        lambda self, wanted, i=0: "budget" if self.pools[i].name == "nc" and wanted > 1 else None)
    doc = json.loads(json.dumps(seed))
    doc["members"][0]["pool"]["target"] = 5
    doc_path = tmp_path / "d.json"
    doc_path.write_text(json.dumps(doc))
    args = build_parser().parse_args(["fleet", "apply", "--config", path, "--file", str(doc_path)])
    assert fleet_cli.main(args) == 4 and store.generation() == 1
    assert "refused: pool nc target 5: budget" in capsys.readouterr().err


def test_fleet_apply_refuses_a_stale_generation_and_dry_runs(tmp_path, monkeypatch, capsys):
    from yeto.cli import build_parser  # noqa: PLC0415
    from yeto.launch.pool import fleet_cli  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, path = _store_run(tmp_path, monkeypatch, objects)
    store = dz.DesiredStore(objects)
    store.apply(dz.from_run_config(cfg), expect=0)
    new = json.loads(json.dumps(store.read()))
    new["members"].append(_member("extra", 1))
    new["members"] = [m for m in new["members"] if m["name"] != "modal"]
    doc_path = tmp_path / "desired.json"
    doc_path.write_text(json.dumps(new))
    parse = build_parser().parse_args
    assert fleet_cli.main(parse(["fleet", "apply", "--config", path, "--file", str(doc_path), "--dry-run"])) == 0
    out = capsys.readouterr().out
    assert "add pool extra (target 1)" in out and "remove pool modal" in out and store.generation() == 1
    assert fleet_cli.main(parse(["fleet", "apply", "--config", path, "--file", str(doc_path),
                                 "--generation", "7"])) == 3
    assert fleet_cli.main(parse(["fleet", "apply", "--config", path, "--file", str(doc_path),
                                 "--generation", "1"])) == 0
    assert store.generation() == 2
    assert fleet_cli.main(parse(["fleet", "show", "--config", path])) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{"):])["generation"] == 2


def test_the_docs_commands_and_example_run_in_dry_run(tmp_path, monkeypatch, capsys):
    """docs/FLEET_CONTROL_PLANE.md: its desired state example is valid, every `yeto fleet` /
    `yeto run scale` command parses, and show / apply --dry-run run on a store-mode run."""
    import re  # noqa: PLC0415
    import shlex  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from yeto.cli import build_parser  # noqa: PLC0415
    from yeto.launch.pool import fleet_cli  # noqa: PLC0415

    text = (Path(__file__).resolve().parents[1] / "docs" / "FLEET_CONTROL_PLANE.md").read_text()
    example = next(b for b in re.findall(r"```json\n(.*?)```", text, re.S) if dz.SCHEMA in b)
    doc = dz.validate(json.loads(example))
    assert set(dz.trainers(doc)) == {"island-1"} and set(dz.sampler_pools(doc)) == {"nc", "modal"}
    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, path = _store_run(tmp_path, monkeypatch, objects)
    store = dz.DesiredStore(objects)
    store.apply(dz.from_run_config(cfg), expect=0)
    store.apply(store.read(), expect=1)
    store.apply(store.read(), expect=2)  # generation 3, as in the example
    desired = tmp_path / "desired.json"
    desired.write_text(example)
    parser = build_parser()
    commands = [line.split("#")[0].strip() for block in re.findall(r"```bash\n(.*?)```", text, re.S)
                for line in block.splitlines() if line.startswith(("yeto fleet", "yeto run scale", "yeto run detach"))]
    assert len(commands) >= 8
    for line in commands:
        argv = shlex.split(line.split(">")[0].replace("run.json", path).replace("desired.json", str(desired))
                           .replace("<ticket>", "t1"))[1:]
        args = parser.parse_args(argv)
        if args.command == "fleet" and (args.fleet_command == "show" or args.dry_run):
            assert fleet_cli.main(args) == 0, line
    out = capsys.readouterr().out
    assert "fleet apply: dry run: generation 4 would be written" in out


def test_run_scale_without_the_store_control_plane_is_byte_for_byte_as_before(tmp_path, monkeypatch, capsys):
    """The default control plane (customer path): run scale spawns the same pool-scale commands,
    returns the same codes and prints the same lines as before the desired state (the golden was
    captured on main before it), and never reads the store."""
    import re  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from tests.test_rollout_entry import _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    golden = json.loads((Path(__file__).parent / "fixtures" / "run_scale_local_control_plane.json").read_text())
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up.Run, "budget_problem", lambda self: None)
    monkeypatch.setattr(up.Run, "_objects", lambda self: (_ for _ in ()).throw(AssertionError("store read")))
    run_cfg, _, _ = _two_pools(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(list(argv[1:]))
                        or subprocess.CompletedProcess(argv, 0))
    r = up.Run(run_cfg)
    capsys.readouterr()
    rcs = [r.scale(3, pool="nc"), r.scale(None, add=1, pool="modal"),
           r.scale(None, remove=["t1"], pool="nc", mode="drain"), r.scale(2, pool="nc", dry_run=True), r.scale(5)]
    out = re.sub(r"\d{4}-\d\d-\d\dT[\d:]+Z", "<T>", capsys.readouterr().out).replace(str(tmp_path), "<T>")
    assert rcs == golden["rcs"]
    assert [[str(a).replace(str(tmp_path), "<T>") for a in c] for c in calls] == golden["calls"]
    assert out == golden["out"]


def test_run_scale_branches_without_the_store_control_plane_match_main(tmp_path):
    """REV #758: more branches of the default control plane's run scale (retire, force-static,
    source, provider, ticket lookup, an unknown pool, rc passthrough, a single `pool` run,
    state.scales), against a golden captured on main before the desired state."""
    from pathlib import Path  # noqa: PLC0415

    from tests.fleet_run_scale_branches import run_scale_branches  # noqa: PLC0415

    golden = json.loads((Path(__file__).parent / "fixtures" / "run_scale_local_control_plane_branches.json")
                        .read_text())
    assert run_scale_branches(tmp_path) == golden


def test_a_controller_taken_over_never_writes_the_applied_state_back(tmp_path):
    """REV #758: fleet/applied.json has one writer; a controller that started later claims it,
    and the earlier one (still running) applies and writes nothing more."""
    f = Fleet(tmp_path, [_member("nc", 1)])
    f.ticks()
    old = f.reconciler
    assert f.store.read_applied()["writer"] == old.writer
    new = FleetReconciler(f.store, ReconcileHooks(), poll_s=0, log=lambda _m: None)
    new.start_from(f.store.read(), applied=f.store.read_applied())
    new._persist()  # noqa: SLF001 - its first pass claims it
    assert f.store.read_applied()["writer"] == new.writer
    f.change(lambda d: d["members"][0]["pool"].update(target=3))
    f.ticks()
    assert old.fenced and "written by" in old.fenced
    assert len(f.shards("nc")) == 1  # the fenced controller did not apply generation 2
    assert f.store.read_applied()["writer"] == new.writer and f.store.read_applied()["generation"] == 1
    with pytest.raises(dz.AppliedFenced):
        f.store.write_applied(0, {}, {}, writer=new.writer)  # never an older generation


def test_two_controllers_claiming_the_applied_state_at_once_one_wins(tmp_path):
    """REV follow-up: the writer fence of fleet/applied.json is a conditional write. Two
    controllers that start at once both read the same applied state before either writes;
    exactly one claims it, the other is fenced (a read-then-write lets both win)."""
    import threading  # noqa: PLC0415

    f = Fleet(tmp_path, [_member("nc", 1)])
    f.ticks()
    seen = f.store.read_applied()
    barrier = threading.Barrier(2, timeout=5)
    real = dz.DesiredStore.read_applied

    def racing_read(store):  # both claimers read before either writes
        doc = real(store)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return doc

    claimers = []
    for _ in range(2):
        r = FleetReconciler(dz.DesiredStore(f.objects), ReconcileHooks(), poll_s=0, log=lambda _m: None)
        r.start_from(f.store.read(), applied=seen)
        r.store.read_applied = racing_read.__get__(r.store)
        claimers.append(r)
    threads = [threading.Thread(target=r._persist) for r in claimers]  # noqa: SLF001
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    fenced = [r for r in claimers if r.fenced]
    winners = [r for r in claimers if not r.fenced]
    assert len(winners) == 1 and len(fenced) == 1
    assert real(f.store)["writer"] == winners[0].writer


def test_a_store_without_atomic_create_refuses_to_fence_the_applied_state(tmp_path):
    """REV #783: a check-then-write create (the OSS relay, an endpoint refusing If-None-Match)
    cannot tell two claimers apart: refused, and the reconciler applies nothing more."""
    objects = DirectoryObjectStore(tmp_path)
    objects.atomic_create = False
    store = dz.DesiredStore(objects, log=lambda _m: None)
    with pytest.raises(dz.AppliedUnsafe):
        store.write_applied(1, {}, {}, writer="a", claim=True)
    r = FleetReconciler(store, ReconcileHooks(), poll_s=0, log=lambda _m: None)
    r.start_from({"generation": 1, "members": []})
    r._persist()  # noqa: SLF001
    assert r.fenced and "atomic" in r.fenced


def test_the_applied_copy_of_the_older_format_is_still_read(tmp_path):
    """REV #783: a run that wrote only fleet/applied.json (before the records): its pools being
    removed are still read, so they are still driven after the upgrade."""
    objects = DirectoryObjectStore(tmp_path)
    old = {"schema": dz.APPLIED_SCHEMA, "generation": 3, "writer": "old",
           "members": [_member("nc", 1)], "removing": [_member("gone", 2)]}
    objects.write(dz.APPLIED_KEY, json.dumps(old).encode())
    store = dz.DesiredStore(objects, log=lambda _m: None)
    assert [m["name"] for m in store.read_applied()["removing"]] == ["gone"]
    assert store.write_applied(4, {}, {}, writer="new", claim=True) == 1
    assert store.read_applied()["writer"] == "new"


def test_a_failed_copy_write_is_logged_and_the_record_still_holds(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    real = objects.write

    def write(key, data):
        if key == dz.APPLIED_KEY:
            raise ConnectionError("copy write failed")
        return real(key, data)

    objects.write = write
    logged = []
    store = dz.DesiredStore(objects, log=logged.append)
    assert store.write_applied(1, {}, {}, writer="a", claim=True) == 1
    assert logged and "WARNING" in logged[0] and "copy" in logged[0]
    assert store.read_applied()["seq"] == 1 and store.read_applied()["writer"] == "a"


def test_a_lagging_listing_never_fences_the_only_controller(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    store = dz.DesiredStore(objects, log=lambda _m: None)
    store.write_applied(1, {}, {}, writer="a", claim=True)
    store.write_applied(1, {}, {}, writer="a")
    real = objects.list
    objects.list = lambda prefix, **kw: [k for k in real(prefix, **kw) if not k.endswith("00000002.json")]
    assert store.read_applied()["seq"] == 2  # the copy says 2 although the listing lags
    assert store.write_applied(2, {}, {}, writer="a") == 3


def test_a_store_that_does_not_list_retries_the_applied_write(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    store = dz.DesiredStore(objects, log=lambda _m: None)
    r = FleetReconciler(store, ReconcileHooks(), poll_s=0, log=lambda _m: None)
    r.start_from({"generation": 1, "members": []})
    real = objects.list
    objects.list = lambda prefix, **kw: (_ for _ in ()).throw(ConnectionError("list failed"))
    r._persist()  # noqa: SLF001
    assert not r.fenced and r._dirty  # noqa: SLF001 - not written, tried again
    objects.list = real
    r._persist()  # noqa: SLF001
    assert not r._dirty and store.read_applied()["writer"] == r.writer  # noqa: SLF001


def test_old_applied_records_are_pruned(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    store = dz.DesiredStore(objects, log=lambda _m: None)
    for i in range(dz.APPLIED_KEEP + 5):
        store.write_applied(1, {}, {}, writer="a", claim=i == 0)
    seqs = store._applied_seqs()  # noqa: SLF001
    assert len(seqs) == dz.APPLIED_KEEP and seqs[-1] == dz.APPLIED_KEEP + 5


def test_a_pruned_record_number_created_again_by_a_stale_writer_is_fenced(tmp_path):
    """REV #783 (ABA): A read record 1 and stalled; B took over and wrote past APPLIED_KEEP, so
    the early numbers were pruned. A's create of record 2 then succeeds: A is fenced anyway,
    its record withdrawn, and the copy never goes back."""
    objects = DirectoryObjectStore(tmp_path)
    a = dz.DesiredStore(objects, log=lambda _m: None)
    b = dz.DesiredStore(objects, log=lambda _m: None)
    a.write_applied(1, {}, {}, writer="a", claim=True)
    stale = a.read_applied()
    for i in range(dz.APPLIED_KEEP + 3):
        b.write_applied(2, {}, {}, writer="b", claim=i == 0)
    top = b.read_applied()["seq"]
    assert 2 not in b._applied_seqs()  # noqa: SLF001 - pruned
    a.read_applied = lambda: stale  # A woke up with what it read before it stalled
    with pytest.raises(dz.AppliedFenced, match="past 2"):
        a.write_applied(1, {}, {}, writer="a")
    assert 2 not in b._applied_seqs()  # noqa: SLF001 - withdrawn
    copy = json.loads(objects.read(dz.APPLIED_KEY))
    assert copy["seq"] == top and copy["writer"] == "b"
    assert b.read_applied()["writer"] == "b"
