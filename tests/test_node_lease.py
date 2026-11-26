"""Node leases across runs (yeto.launch.node_lease; own-cluster-fleet-stability 3.1-3.2):
the inventory's top-level ``lease_store``, the lease (claim, renew, release, takeover after
it ran out) on a FakeS3 store, and both node pickers that check it: the ssh sampler pool
and the ssh trainer island (spec render, start, watch, release). An inventory without
``lease_store`` reads no store and behaves as on main."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from types import SimpleNamespace

import pytest

from tests.test_rl_object_store import FakeS3
from tests.test_rl_ssh_resident import NAMES, Clock, FakeCluster, b300, make_ops, spec
from tests.test_ssh_sampler_pool import npu_node, pool_cfg, provider_section
from yeto.cloud.s3_store import S3ObjectStore
from yeto.launch import node_lease, ssh_sampler_pool as sp
from yeto.launch.node_inventory import InventoryError, NodeList, lease_store_of, parse_inventory, parse_profiles
from yeto.launch.node_lease import (
    NO_LEASE_HINT,
    TAKEOVER_EVENT,
    NodeLeased,
    NodeLeaseError,
    NodeLeases,
    pick_nodes,
)
from yeto.launch.pool.island_pool import REFUSED_ERRORS, error_is
from yeto.launch.ssh_trainer_island import SshIslandError, inventory_trainer_nodes

STORE = "s3://site/leases"


def _objects(client: FakeS3 | None = None) -> S3ObjectStore:
    return S3ObjectStore(STORE, client=client or FakeS3())


def _leases(run, objects, clock, **kw):
    events = kw.pop("events", None)
    on_event = (lambda name, **d: events.append((name, d))) if events is not None else None
    return NodeLeases(STORE, run, objects=objects, clock=clock, on_event=on_event, **kw)


def _raw(i, **kw):
    return {"name": f"b300-{i}", "host": f"ops@192.0.2.{10 + i}", "device_family": "cuda", "cpu_arch": "x86_64",
            "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, **kw}


@pytest.fixture(autouse=True)
def _fresh_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(node_lease, "_hinted", False)
    from yeto.hw import card_tools

    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])  # the fake cluster's cards
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))


# -- 3.1 the inventory field -----------------------------------------------------------

def test_an_inventory_without_lease_store_reads_as_before():
    nodes = parse_inventory({"nodes": [_raw(0), _raw(1)]})
    assert isinstance(nodes, list) and isinstance(nodes, NodeList) and lease_store_of(nodes) is None
    assert [n.name for n in nodes] == ["b300-0", "b300-1"]
    assert lease_store_of(parse_inventory([_raw(0)])) is None  # a bare node list
    assert lease_store_of([object()]) is None  # a plain list (a test's, a caller's)


@pytest.mark.parametrize("value, want", [("s3://site/leases/", "s3://site/leases"), ("oss://b/p", "oss://b/p"),
                                         ("/srv/shared/leases", "/srv/shared/leases"),
                                         ("file:///srv/leases", "file:///srv/leases")])
def test_lease_store_is_kept_on_the_node_list(value, want):
    nodes = parse_inventory({"lease_store": value, "nodes": [_raw(0)]})
    assert lease_store_of(nodes) == want
    assert lease_store_of(parse_profiles({"lease_store": value, "nodes": [_raw(0)]})) == want


@pytest.mark.parametrize("value", [3, "", "relative/dir", "s3://", "file://relative", "/a/../b", ["s3://x"]])
def test_a_bad_lease_store_is_refused(value):
    with pytest.raises(InventoryError, match="lease_store"):
        parse_inventory({"lease_store": value, "nodes": [_raw(0)]})


def test_the_unknown_field_message_names_lease_store():
    with pytest.raises(InventoryError, match=r"known: \['images', 'lease_store', 'node_health', 'nodes'\]"):
        parse_inventory({"lease": "s3://x", "nodes": [_raw(0)]})


def test_load_inventory_reads_lease_store(tmp_path):
    from yeto.launch.node_inventory import load_inventory, load_profiles

    path = tmp_path / "inv.yaml"
    path.write_text("lease_store: s3://site/leases\nnodes:\n"
                    "  - {name: n0, host: ops@192.0.2.1, device_family: cuda, cpu_arch: x86_64, card: B300,\n"
                    "     cards_per_node: 8, hbm_gib: 288, host_memory_gib: 2048}\n")
    assert lease_store_of(load_inventory(path)) == "s3://site/leases"
    assert lease_store_of(load_profiles(path)) == "s3://site/leases"


# -- 3.1 the lease on a FakeS3 store ---------------------------------------------------

def test_two_runs_racing_for_one_node_get_one_lease():
    client, clock = FakeS3(), Clock()
    won, start = [], threading.Barrier(8)

    def claim(i):
        leases = _leases(f"run-{i}", _objects(client), clock)
        start.wait()
        if leases.acquire("b300-0").ok:
            won.append(i)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(won) == 1
    assert _leases("x", _objects(client), clock).holder("b300-0") == f"run-{won[0]}"


def test_a_held_node_is_refused_with_its_holder_and_renewed_by_its_holder():
    objects, clock = _objects(), Clock()
    a, b = _leases("run-a", objects, clock), _leases("run-b", objects, clock)
    got = a.acquire("b300-0")
    assert got.ok and got.holder == "run-a" and not got.took_over
    refused = b.acquire("b300-0")
    assert not refused.ok and refused.holder == "run-a"
    clock.t += 500
    assert a.renew("b300-0")
    clock.t += 500  # 1000 s after the claim, 500 s after the renewal: still held
    assert b.acquire("b300-0").holder == "run-a" and b.holder("b300-0") == "run-a"
    assert a.acquire("b300-0").ok  # held already: renewed, no second token
    assert StoreLockTokens(objects, "b300-0") == [1]


def StoreLockTokens(objects, node):  # noqa: N802 - a helper read like a constant
    from yeto.launch.pool.store_lock import StoreLock

    return StoreLock.tokens(objects, NodeLeases.prefix(node))


def test_a_release_frees_the_node_at_once():
    objects, clock = _objects(), Clock()
    a, b = _leases("run-a", objects, clock), _leases("run-b", objects, clock)
    a.acquire("b300-0")
    a.release("b300-0")
    assert a.held() == [] and b.holder("b300-0") is None
    got = b.acquire("b300-0")
    assert got.ok and not got.took_over  # a released lease is no takeover
    assert objects.list("nodes/b300-0/events/") == []


def test_a_lease_run_out_is_taken_over_with_an_event_and_fences_the_old_holder():
    objects, clock, events = _objects(), Clock(), []
    a = _leases("run-a", objects, clock, events=[])
    b = _leases("run-b", objects, clock, events=events)
    a.acquire("b300-0")
    clock.t += 599
    assert not b.acquire("b300-0").ok
    clock.t += 2  # run A's control plane and node process are gone: nobody renewed
    got = b.acquire("b300-0")
    assert got.ok and got.took_over and got.previous == "run-a" and got.holder == "run-b"
    keys = objects.list("nodes/b300-0/events/")
    assert keys == ["nodes/b300-0/events/00000002-takeover.json"]
    doc = json.loads(objects.read(keys[0]))
    assert doc["event"] == TAKEOVER_EVENT and doc["previous"] == "run-a" and doc["holder"] == "run-b"
    assert events == [(TAKEOVER_EVENT, {"node": "b300-0", "holder": "run-b", "previous": "run-a", "token": 2,
                                        "recorded": True})]
    assert not a.renew("b300-0")  # the old holder learns it lost the node
    assert a.lost() == {"b300-0": "run-b"} and a.held() == []


def test_a_restarted_launcher_of_the_same_run_takes_its_running_lease_back():
    objects, clock = _objects(), Clock()
    first = _leases("run-a", objects, clock)
    first.acquire("b300-0")
    again = _leases("run-a", objects, clock)  # the same run, a new process
    got = again.acquire("b300-0")
    assert got.ok and not got.took_over
    assert StoreLockTokens(objects, "b300-0") == [1, 2]
    assert not first.renew_due(force=True)  # the old process: its lease is the run's still (not lost)
    assert first.lost() == {}


def test_renew_due_renews_only_after_a_quarter_of_the_lease():
    objects, clock = _objects(), Clock()
    a = _leases("run-a", objects, clock)
    a.acquire("b300-0")
    writes = len(objects.client.calls)
    clock.t += 100
    a.renew_due()
    assert len(objects.client.calls) == writes  # 100 s < 150 s: no store call
    clock.t += 60
    assert a.renew_due() == []
    assert len(objects.client.calls) > writes


def test_a_store_without_atomic_create_is_refused():
    objects = _objects(FakeS3(conditional=False))
    objects.create("probe", b"{}")  # learns the endpoint has no If-None-Match
    with pytest.raises(NodeLeaseError, match="atomic"):
        _leases("run-a", objects, Clock()).acquire("b300-0")


def test_a_directory_store_works_for_a_shared_file_system(tmp_path):
    clock = Clock()
    a = NodeLeases(str(tmp_path / "leases"), "run-a", clock=clock)
    b = NodeLeases(f"file://{tmp_path / 'leases'}", "run-b", clock=clock)
    assert a.acquire("n0").ok and not b.acquire("n0").ok
    assert (tmp_path / "leases" / "nodes" / "n0" / "lease" / "00000001.json").is_file()


def test_the_keeper_renews_in_the_background():
    objects, clock = _objects(), Clock()
    a = _leases("run-a", objects, clock, ttl_s=4)
    a.acquire("b300-0")
    clock.t += 3
    a.start_keeper(interval_s=0.01)
    try:
        import time

        deadline = time.time() + 5
        while time.time() < deadline and a._locks["b300-0"].renewed_at != clock.t:  # noqa: SLF001
            time.sleep(0.01)
    finally:
        a.stop_keeper()
    clock.t += 3  # 6 s after the claim, 3 s after the keeper's renewal
    assert _leases("run-b", objects, clock).holder("b300-0") == "run-a"


# -- pick_nodes: skip a held node, wait for a free one ---------------------------------

def test_pick_skips_held_nodes_and_keeps_inventory_order():
    objects, clock = _objects(), Clock()
    _leases("run-a", objects, clock).acquire("b300-1")
    b = _leases("run-b", objects, clock)
    assert pick_nodes(["b300-0", "b300-1", "b300-2"], 2, b) == ["b300-0", "b300-2"]
    assert b.held() == ["b300-0", "b300-2"]


def test_pick_prefers_the_nodes_this_run_holds():
    objects, clock = _objects(), Clock()
    _leases("run-b", objects, clock).acquire("b300-2")  # an earlier launcher of run B
    again = _leases("run-b", objects, clock)
    assert pick_nodes(["b300-0", "b300-1", "b300-2"], 1, again) == ["b300-2"]


def test_pick_reports_the_holders_and_waits_until_a_node_is_free():
    objects, clock, lines = _objects(), Clock(), []
    a = _leases("run-a", objects, clock)
    a.acquire("b300-0")
    a.acquire("b300-1")
    b = _leases("run-b", objects, clock)
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            a.release("b300-1")

    assert pick_nodes(["b300-0", "b300-1"], 1, b, sleep=sleep, log=lines.append) == ["b300-1"]
    assert len(sleeps) == 3
    assert lines == ["[node-lease] 1 node(s) needed, 0 free; held: b300-0 by run run-a, b300-1 by run run-a; "
                     "waiting (the leases are read every 30s)"]  # once: the holders did not change
    assert "b300-0" not in b.held()


def test_pick_waits_after_a_lost_race_instead_of_spinning():
    objects, clock = _objects(), Clock()
    b = _leases("run-b", objects, clock)
    real = b.acquire
    rival = _leases("run-a", objects, clock)
    raced = []

    def acquire(node):  # run A claims the node between run B's read and its claim
        if not raced:
            raced.append(node)
            rival.acquire(node)
        return real(node)

    b.acquire = acquire
    sleeps, lines = [], []

    def sleep(s):
        sleeps.append(s)
        rival.release("b300-0")

    assert pick_nodes(["b300-0"], 1, b, sleep=sleep, log=lines.append) == ["b300-0"]
    assert sleeps == [30.0] and lines == ["[node-lease] 1 node(s) needed, 0 free; held: b300-0 by run run-a; "
                                          "waiting (the leases are read every 30s)"]


def test_pick_gives_up_after_wait_s_naming_the_holders():
    objects, clock = _objects(), Clock()
    _leases("run-a", objects, clock).acquire("b300-0")
    t = [0.0]

    def sleep(s):
        t[0] += s

    with pytest.raises(NodeLeased, match="held: b300-0 by run run-a") as caught:
        pick_nodes(["b300-0"], 1, _leases("run-b", objects, clock), wait_s=60, sleep=sleep,
                   monotonic=lambda: t[0], log=lambda m: None)
    assert caught.value.holders == {"b300-0": "run-a"}


def test_no_lease_hint_is_shared_and_said_once(caplog):
    out = []
    assert node_lease.no_lease_hint(emit=out.append) == NO_LEASE_HINT
    assert node_lease.no_lease_hint(emit=out.append) is None
    assert out == [NO_LEASE_HINT]


def test_no_lease_hint_without_emit_goes_to_the_log_only(caplog, capsys):
    with caplog.at_level(logging.DEBUG, logger="yeto.launch.node_lease"):
        node_lease.no_lease_hint()
    assert caplog.messages == [NO_LEASE_HINT]
    assert [r.levelno for r in caplog.records] == [logging.DEBUG]  # an INFO log (the customer's) shows nothing
    assert capsys.readouterr() == ("", "")


# -- 3.2 the ssh sampler pool ----------------------------------------------------------

def _pool_conf(n=2, lease_store=STORE, **section):
    nodes = NodeList([npu_node(name=f"node-{c}", host=f"node-{c}") for c in "ab"[:n]], lease_store=lease_store)
    return sp.parse_config({"provider": provider_section(nodes=[f"node-{c}" for c in "ab"[:n]], **section)},
                           inventory=nodes)


def _provider(tmp_path, objects, clock, *, run="pool-b", n=2, lease_store=STORE):
    prov = sp.SshNodeProvider(run, _pool_conf(n, lease_store), slots=list(range(n)),
                              state_path=tmp_path / f"{run}.json", clock=clock)
    if lease_store is not None:
        assert prov.leases.run_id == f"{run}/pool"  # the holder: the pool's run tag and part
        prov._leases = NodeLeases(STORE, f"{run}/pool", objects=objects, clock=clock,  # noqa: SLF001
                                  on_event=prov._lease_event)  # noqa: SLF001
    events = []
    prov.on_event = lambda name, **d: events.append((name, d))
    return prov, events


def test_a_pool_without_lease_store_never_reads_a_store(tmp_path):
    prov, _ = _provider(tmp_path, None, Clock(), lease_store=None)
    assert prov.conf.lease_store is None and prov.leases is None
    assert prov.account_free(0) and prov.account_free(1)
    t = prov.order(0)
    assert prov.status(0, t).state == "active"
    prov.release(0, t)
    assert prov._leases is None  # noqa: SLF001 - nothing was opened


def test_the_pool_reads_lease_store_from_the_inventory_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "load_profiles", lambda _p: NodeList([npu_node()], lease_store=STORE))
    assert sp.parse_config(pool_cfg(tmp_path, inventory="inv.yaml")).lease_store == STORE
    monkeypatch.setattr(sp, "load_profiles", lambda _p: [npu_node()])
    assert sp.parse_config(pool_cfg(tmp_path, inventory="inv.yaml")).lease_store is None


def test_the_pool_skips_a_node_another_run_holds(tmp_path):
    objects, clock = _objects(), Clock()
    _leases("trainer-a", objects, clock).acquire("node-a")
    prov, events = _provider(tmp_path, objects, clock)
    assert not prov.account_free(0) and prov.account_free(1)
    assert not prov.account_free(0)  # reported once per change
    assert events == [("node_leased_elsewhere", {"node": "node-a", "holder": "trainer-a"})]
    t = prov.order(1)
    assert prov.leases.held() == ["node-b"]
    assert _leases("x", objects, clock).holder("node-b") == "pool-b/pool"
    with pytest.raises(NodeLeased, match="node-a is held by run trainer-a") as caught:
        prov.order(0)
    assert error_is(caught.value, REFUSED_ERRORS)  # the pool tries the next slot, no provisioning failure
    assert sorted(prov._rows()) == [t]  # noqa: SLF001 - no ticket for the refused order


def test_a_pool_with_no_free_node_offers_none_until_the_holder_leaves(tmp_path):
    objects, clock = _objects(), Clock()
    other = _leases("trainer-a", objects, clock)
    other.acquire("node-a")
    other.acquire("node-b")
    prov, events = _provider(tmp_path, objects, clock)
    assert not prov.account_free(0) and not prov.account_free(1)
    assert {d["node"]: d["holder"] for _n, d in events} == {"node-a": "trainer-a", "node-b": "trainer-a"}
    other.release("node-b")
    clock.t += sp.LEASE_READ_CACHE_S
    assert prov.account_free(1) and not prov.account_free(0)


def test_a_lost_node_hold_does_not_offer_a_node_leased_elsewhere(tmp_path):
    objects, clock = _objects(), Clock()
    prov, _ = _provider(tmp_path, objects, clock)
    t = prov.order(0)
    prov.mark_lost(t, stopped=True)
    prov.release(0, t)
    _leases("trainer-a", objects, clock).acquire("node-b")  # the only spare is another run's
    assert prov.account_free(0)  # no free spare: the held node is offered (as without leases)


def test_ticket_reads_renew_the_pool_lease_and_a_release_gives_it_back(tmp_path):
    objects, clock = _objects(), Clock()
    prov, _ = _provider(tmp_path, objects, clock)
    t = prov.order(0)
    for _ in range(8):  # 8 x 100 s of ticket reads: the lease stays the pool's past its 600 s
        clock.t += 100
        assert prov.status(0, t).state == "active"
    assert _leases("x", objects, clock).holder("node-a") == "pool-b/pool"
    prov.release(0, t)
    assert prov.leases.held() == [] and _leases("x", objects, clock).holder("node-a") is None


def test_a_node_with_an_owed_stop_keeps_its_lease_until_the_stop_went_through(tmp_path):
    objects, clock = _objects(), Clock()
    prov, _ = _provider(tmp_path, objects, clock)
    t = prov.order(0)
    prov.note_shard(t, "s-0")
    prov.mark_lost(t, stopped=False)
    prov.release(0, t)
    assert prov.leases.held() == ["node-a"]  # its sampler may still run there
    prov.stop_done(t)
    assert prov.leases.held() == []


def test_a_member_whose_lease_another_run_took_ends_lost(tmp_path):
    objects, clock = _objects(), Clock()
    prov, _ = _provider(tmp_path, objects, clock)
    t = prov.order(0)
    clock.t += 700  # the pool controller stalled past the lease; another run took the node
    assert _leases("trainer-a", objects, clock).acquire("node-a").took_over
    status = prov.status(0, t)
    assert status.state == "ended" and status.end_reason == "lost"
    assert status.detail == "node node-a: its lease is held by run trainer-a now"


def test_pool_dry_run_says_the_shared_hint_once_without_lease_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "load_profiles", lambda _p: [npu_node()])
    cfg = pool_cfg(tmp_path, inventory="inv.yaml")
    assert sp.lease_notes(cfg) == [NO_LEASE_HINT]
    assert sp.lease_notes(cfg) == []  # once per process
    monkeypatch.setattr(sp, "load_profiles", lambda _p: NodeList([npu_node()], lease_store=STORE))
    assert sp.lease_notes(cfg) == [f"node leases: lease_store {STORE}: a node another run holds is skipped; an "
                                   "order takes the node's lease first (no SSH or store call in a dry run)"]


def test_pool_dry_run_prints_the_hint_as_a_note(tmp_path, monkeypatch, capsys):
    from yeto.launch.nc import pool_run

    monkeypatch.setattr(sp, "load_profiles", lambda _p: [npu_node()])
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    pool_run.dry_run(pool_cfg(tmp_path, inventory="inv.yaml"), "pool.json")
    out = capsys.readouterr().out.splitlines()
    assert out.count(f"note: {NO_LEASE_HINT}") == 1


# -- 3.2 the ssh trainer island --------------------------------------------------------

def _island(n=2):
    return SimpleNamespace(num_nodes=n, gpus_per_node=8, gpu="B300")


def _args(nodes, **kw):
    return argparse.Namespace(rl_node_inventory_nodes=nodes, cluster_prefix="run-b", **kw)


def test_trainer_nodes_without_lease_store_are_as_on_main():
    nodes = NodeList([b300(i) for i in range(3)])
    with pytest.raises(SshIslandError, match="has 2 node\\(s\\); the inventory lists 3"):
        inventory_trainer_nodes(_args(nodes), _island(2))
    assert [n.name for n in inventory_trainer_nodes(_args(nodes[:2]), _island(2))] == NAMES[:2]
    from yeto.launch.ssh_trainer_island import trainer_leases

    assert trainer_leases(_args(nodes[:2])) is None


def test_trainer_nodes_with_lease_store_skip_a_held_node():
    objects, clock = _objects(), Clock()
    nodes = NodeList([b300(i) for i in range(3)], lease_store=STORE)
    assert [n.name for n in inventory_trainer_nodes(_args(nodes), _island(2))] == NAMES[:2]  # no leases: no store
    _leases("run-a", objects, clock).acquire("b300-0")
    mine = _leases("run-b/trainer", objects, clock)
    picked = inventory_trainer_nodes(_args(nodes), _island(2), leases=mine)
    assert [n.name for n in picked] == ["b300-1", "b300-2"] and mine.held() == ["b300-1", "b300-2"]
    with pytest.raises(SshIslandError, match="held: b300-0 by run run-a"):  # a second island: no free node
        inventory_trainer_nodes(_args(nodes), _island(1), leases=mine, wait_s=0)
    with pytest.raises(SshIslandError, match="has 4 node\\(s\\); the inventory lists 3"):
        inventory_trainer_nodes(_args(nodes), _island(4))


def test_a_dry_run_leases_nothing_and_logs_the_hint_only(caplog, capsys):
    from yeto.launch.ssh_trainer_island import trainer_lease_hint, trainer_leases

    assert trainer_leases(_args(NodeList([b300(0)], lease_store=STORE), dry_run=True)) is None
    with caplog.at_level(logging.DEBUG, logger="yeto.launch.node_lease"):
        trainer_lease_hint(_args(NodeList([b300(0)])))
        trainer_lease_hint(_args(NodeList([b300(0)])))
    assert caplog.messages == [NO_LEASE_HINT]
    assert capsys.readouterr() == ("", "")


def test_a_start_on_a_node_another_run_took_is_refused(tmp_path):
    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    for name in NAMES:
        mine.acquire(name)
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec(leases=mine))
    assert ops.start("isl") == 0
    assert str(ops.job_status("isl", 0)) == "RUNNING"
    clock.t += 700  # the launcher hung past the lease; run A took b300-1
    assert _leases("run-a", objects, clock).acquire("b300-1").took_over
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert "node lease lost: b300-1 is held by run run-a" in ops.islands["isl"].reason
    assert cluster.count("kill", 0) == 3  # every node released: run A may start there
    assert ops.relaunch(None, "isl") is None  # no start on another run's node; the controller retries
    assert cluster.count("start") == 3


def test_watch_ticks_renew_the_island_leases_and_down_releases_them(tmp_path):
    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    ops = make_ops(tmp_path, FakeCluster(NAMES), clock=clock)
    ops.add("isl", spec(leases=mine))
    ops.start("isl")  # the start takes every node's lease
    assert mine.held() == NAMES
    for _ in range(10):
        clock.t += 100
        assert str(ops.job_status("isl", 0)) == "RUNNING"
    assert all(_leases("x", objects, clock).holder(n) == "run1/trainer" for n in NAMES)
    assert ops.down("isl")
    assert mine.held() == [] and all(_leases("x", objects, clock).holder(n) is None for n in NAMES)


def test_an_island_without_leases_watches_as_before(tmp_path):
    ops = make_ops(tmp_path, FakeCluster(NAMES))
    ops.add("isl", spec())
    ops.start("isl")
    assert ops.islands["isl"].islands.lease_lost() is None
    assert str(ops.job_status("isl", 0)) == "RUNNING" and ops.down("isl")
