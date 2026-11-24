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


class Fleet:
    """A controller with a reconciler over pools on fake providers and one local store
    (the run store and the trainer's buffer store alike)."""

    def __init__(self, tmp_path, members, *, slots=3, data_path=False):
        self.objects = DirectoryObjectStore(tmp_path / "store")
        self.data_path = data_path  # the pools' data path writes the store (in-shard end records)
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
        self.fencers[name] = FakeFencer()
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
