"""Node leases across runs (yeto.launch.node_lease; own-cluster-fleet-stability 3.1-3.3):
the inventory's top-level ``lease_store``, the lease (claim, renew, release, takeover after
it ran out) on a FakeS3 store, and both node pickers that check it: the ssh sampler pool
and the ssh trainer island (spec render, start, watch, release). An inventory without
``lease_store`` reads no store and behaves as on main."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path
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
    kw.setdefault("monotonic", clock)  # the renewal cadence on the test's clock too
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
    from yeto import runs
    from yeto.launch import ssh_trainer_island

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")  # the run dirs (lease identity, picks)
    monkeypatch.setattr(ssh_trainer_island, "_TRAINER_LEASES", {})


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
                     "waiting 0s so far (the leases are read every 30s)"]  # once: the holders did not change
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
                                          "waiting 0s so far (the leases are read every 30s)"]


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
        prov._leases = NodeLeases(STORE, f"{run}/pool", objects=objects, clock=clock, monotonic=clock,  # noqa: SLF001
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


# -- review: H1 one LIST per claim ------------------------------------------------------

class _Interleaved:
    """An object store whose first LIST under ``prefix`` runs ``between()`` right after it
    answered: another claimer acts between this claimer's LIST and its create."""

    def __init__(self, objects, prefix, between):
        self._objects, self._prefix, self._between, self.lists = objects, prefix, between, 0

    def __getattr__(self, name):
        return getattr(self._objects, name)

    def _listed(self, prefix):
        if prefix.startswith(self._prefix):
            self.lists += 1
            if self.lists == 1:
                self._between()

    def list(self, prefix, **kw):
        out = self._objects.list(prefix, **kw)
        self._listed(prefix)
        return out

    def list_meta(self, prefix, **kw):  # the S3 listing with object times (node leases read it)
        out = self._objects.list_meta(prefix, **kw)
        self._listed(prefix)
        return out


def test_storelock_acquire_judges_and_creates_on_one_listing():
    from yeto.launch.pool.store_lock import StoreLock

    objects, clock, won = _objects(), Clock(), {}
    seen = _Interleaved(objects, "L", lambda: won.setdefault("b", StoreLock.acquire(objects, "L", holder="b",
                                                                                     clock=clock)))
    a = StoreLock.acquire(seen, "L", holder="a", clock=clock)
    assert won["b"] is not None and won["b"].token == 1
    assert a is None  # A listed before B claimed: it raced B for token 1 and lost, never took token 2
    assert StoreLock.tokens(objects, "L") == [1]
    assert seen.lists == 1  # the claim itself listed once (the renew of a won claim lists again)


def test_node_lease_claim_interleaved_after_its_list_leaves_one_holder():
    objects, clock, won = _objects(), Clock(), {}
    b = _leases("run-b", objects, clock)
    seen = _Interleaved(objects, "nodes/b300-0/lease", lambda: won.setdefault("b", b.acquire("b300-0")))
    a = _leases("run-a", seen, clock)
    got = a.acquire("b300-0")
    assert won["b"].ok and not got.ok and got.holder == "run-b"
    assert StoreLockTokens(objects, "b300-0") == [1]


def test_storelock_reclaim_is_off_by_default():
    from yeto.launch.pool.store_lock import StoreLock

    objects, clock = _objects(), Clock()
    assert StoreLock.acquire(objects, "L", holder="same", clock=clock) is not None
    assert StoreLock.acquire(objects, "L", holder="same", clock=clock) is None  # the fleet pool lock: as before
    assert StoreLock.acquire(objects, "L", holder="same", clock=clock, reclaim=True).token == 2


# -- review: H2 the launch identity ----------------------------------------------------

def test_two_runs_of_one_name_with_different_identities_do_not_share_a_node():
    objects, clock = _objects(), Clock()
    first = NodeLeases(STORE, "yeto/trainer", identity="aaaa", objects=objects, clock=clock)
    other = NodeLeases(STORE, "yeto/trainer", identity="bbbb", objects=objects, clock=clock)
    assert first.acquire("b300-0").ok and first.holder_id == "yeto/trainer@aaaa"
    got = other.acquire("b300-0")
    assert not got.ok and got.holder == "yeto/trainer@aaaa"
    again = NodeLeases(STORE, "yeto/trainer", identity="aaaa", objects=objects, clock=clock)  # its restart
    assert again.acquire("b300-0").ok


def test_the_identity_is_made_once_and_read_back(tmp_path):
    path = tmp_path / "run" / "node-leases.json"
    one = node_lease.local_identity(path)
    assert node_lease.local_identity(path) == one and json.loads(path.read_text())["identity"] == one
    assert node_lease.local_identity(tmp_path / "other.json") != one
    objects = _objects()
    assert node_lease.store_identity(objects, "fleet/pools/p/id.json") == node_lease.store_identity(
        objects, "fleet/pools/p/id.json")


def test_trainer_leases_hold_as_run_and_identity_and_a_restart_keeps_it(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti

    nodes = NodeList([b300(0)], lease_store=str(tmp_path / "leases"))
    args = _args(nodes)
    first = sti.trainer_leases(args)
    try:
        assert first.holder_id.startswith("run-b/trainer@") and sti.trainer_leases(args) is first
        monkeypatch.setattr(sti, "_TRAINER_LEASES", {})  # a new launcher process of the run
        again = sti.trainer_leases(args)
        assert again is not first and again.holder_id == first.holder_id
    finally:
        first.stop_keeper()
        sti._TRAINER_LEASES.get((str(tmp_path / "leases"), first.holder_id), first).stop_keeper()


def test_the_pool_identity_lives_with_its_state_or_in_the_run_store(tmp_path):
    prov, _ = _provider(tmp_path, None, Clock())
    prov._leases = None  # noqa: SLF001 - the real one
    local = prov.leases.holder_id
    assert local.startswith("pool-b/pool@") and (tmp_path / "pool-b.lease-identity.json").is_file()
    stored, _ = _provider(tmp_path / "x", None, Clock())
    stored._leases = None  # noqa: SLF001
    objects = _objects()
    stored._lease_identity = node_lease.store_identity(objects, "fleet/pools/p/node-lease-identity.json")  # noqa: SLF001
    assert stored.leases.identity == node_lease.store_identity(objects, "fleet/pools/p/node-lease-identity.json")


# -- review: H3a the probe before a takeover ---------------------------------------------

def _expired(objects, clock, holder="run-a"):
    _leases(holder, objects, clock).acquire("b300-0")
    clock.t += 601


def test_a_takeover_is_refused_while_the_previous_run_still_runs_there():
    objects, clock, events = _objects(), Clock(), []
    _expired(objects, clock)
    asked = []

    def probe(node, previous):
        asked.append((node, previous))
        return f"node {node} still runs run {previous} (container:yeto-run-a-isl-i0-n0)"

    b = _leases("run-b", objects, clock, events=events, probe=probe)
    got = b.acquire("b300-0")
    assert not got.ok and got.holder == "run-a" and got.blocked.startswith("node b300-0 still runs run run-a")
    assert asked == [("b300-0", "run-a")] and b.held() == []
    assert [e[0] for e in events] == [node_lease.BLOCKED_EVENT]
    doc = json.loads(objects.read("nodes/b300-0/events/00000002-blocked.json"))
    assert doc["event"] == node_lease.BLOCKED_EVENT and doc["previous"] == "run-a"
    assert StoreLockTokens(objects, "b300-0") == [1]  # nothing claimed


def test_a_takeover_goes_ahead_once_the_probe_finds_nothing():
    objects, clock = _objects(), Clock()
    _expired(objects, clock)
    b = _leases("run-b", objects, clock, probe=lambda node, previous: None)
    assert b.acquire("b300-0").took_over


def test_no_probe_on_a_free_or_own_lease():
    objects, clock = _objects(), Clock()
    asked = []
    a = _leases("run-a", objects, clock, probe=lambda n, p: asked.append(n) or "busy")
    assert a.acquire("b300-0").ok  # no lease yet
    clock.t += 601
    again = _leases("run-a", objects, clock, probe=lambda n, p: asked.append(n) or "busy")
    assert again.acquire("b300-0").ok  # its own lease that ran out: no probe
    assert asked == []


def test_a_released_lease_is_still_probed_before_another_run_takes_the_node():
    """L3: a release (or a cleanup's force_release) does not prove the node is empty:
    the next run of another holder asks the node first; a free node is no takeover."""
    objects, clock, events = _objects(), Clock(), []
    a = _leases("run-a", objects, clock)
    assert a.acquire("b300-0").ok
    a.release("b300-0")
    asked = []
    busy = _leases("run-c", objects, clock, probe=lambda n, p: asked.append(p) or "node b300-0 still runs run run-a")
    got = busy.acquire("b300-0")
    assert not got.ok and asked == ["run-a"] and "still runs" in got.blocked
    free = _leases("run-c", objects, clock, probe=lambda n, p: None, events=events)
    got = free.acquire("b300-0")
    assert got.ok and not got.took_over and got.previous is None
    assert [n for n, _ in events if n == node_lease.TAKEOVER_EVENT] == []


@pytest.mark.parametrize("out, rc, want", [
    ("yeto-lease-probe found=container:yeto-run-a-isl-i0-n0\nyeto-lease-probe done\n", 0, "still runs run run-a"),
    ("yeto-lease-probe done\n", 0, None),
    ("", 255, "not confirmed free of run run-a"),
    ("yeto-lease-probe found=sampler:4242\n", 1, "still runs run run-a"),
])
def test_ssh_probe_reads_the_script_output(out, rc, want):
    probe = node_lease.ssh_probe(lambda node, script: SimpleNamespace(stdout=out, returncode=rc))
    why = probe(b300(0), "run-a/trainer@abc")
    assert (why is None) if want is None else (want in why)


FAKE_ENGINE = r"""#!{python}
import json, os, sys
engine = os.path.basename(sys.argv[0])
state = os.path.join(os.environ["FAKE_STATE_DIR"], engine + ("-root" if os.environ.get("FAKE_AS_ROOT") else "") + ".json")
rows = json.load(open(state)) if os.path.exists(state) else []
args = sys.argv[1:]
if args[:1] == ["run"]:
    name = args[args.index("--name") + 1]
    labels = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")
    rows = [r for r in rows if r["name"] != name] + [{{"name": name, "labels": labels}}]
    json.dump(rows, open(state, "w"))
    print("c0ffee")
elif args[:1] == ["ps"]:
    if os.environ.get("FAKE_ENGINE_FAIL") == engine:
        sys.exit(125)
    want = [args[i + 1].split("=", 1)[1] for i, a in enumerate(args) if a == "--filter"]
    for r in rows:
        ok = True
        for w in want:
            key, _, value = w.partition("=")
            ok = ok and key in r["labels"] and (not value or r["labels"][key] == value)
        if ok:
            print(r["name"])
"""

FAKE_SUDO = """#!/bin/sh
[ "$1" = -n ] && shift
[ -n "$FAKE_SUDO_DENY" ] && exit 1
[ "$1" = true ] && exit 0
FAKE_AS_ROOT=1 exec "$@"
"""


def _node_bin(tmp_path, *, sudo=True):
    """A node's PATH: a fake podman that records what the real start command runs (its
    ``--name`` and ``--label`` flags) and answers ``ps --filter label=... --format``;
    a fake ``sudo -n`` whose containers are a separate (rootful) set (no ``sudo``: one
    that always refuses). A fake ``docker`` with no containers shadows a real one."""
    import os
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for engine in ("podman", "docker"):
        (bin_dir / engine).write_text(FAKE_ENGINE.format(python=sys.executable))
        (bin_dir / engine).chmod(0o755)
    (bin_dir / "sudo").write_text(FAKE_SUDO if sudo else "#!/bin/sh\nexit 1\n")
    (bin_dir / "sudo").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path / "home"),
            "FAKE_STATE_DIR": str(tmp_path)}


def _started_container(tmp_path, monkeypatch, env, *, lease_store=STORE, sudo=False, run_prefix="f09"):
    """Start one sampler through the real pool start (``SshSamplerPoolOps.start``) and run
    its ``run -d`` command on the fake node: the container the probe sees is exactly the
    one the pool's argv makes. Returns the pool's run id."""
    import subprocess

    from tests.test_ssh_sampler_pool import FakeRun
    from yeto.launch.nc import cli

    class Pop:
        stdout = None

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    with monkeypatch.context() as mp:  # subprocess.Popen is the fake only while the pool starts
        mp.setattr(sp.subprocess, "Popen", Pop)
        mp.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
        mp.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
        section = {"container": {"engine": "podman", "image": "img", "sudo": sudo}}
        cfg = dict(pool_cfg(tmp_path, **section), run_prefix=run_prefix)
        c = sp.parse_config({"provider": provider_section(**section)},
                            inventory=NodeList([npu_node()], lease_store=lease_store))
        run = FakeRun()
        (tmp_path / run_prefix).mkdir(exist_ok=True)
        ops = sp.SshSamplerPoolOps(cfg, conf=c, run=run, state_dir=tmp_path / run_prefix)
        if lease_store is not None:
            ops.provider._leases = NodeLeases(STORE, f"{ops.provider.run_id}/pool", objects=_objects(),  # noqa: SLF001
                                              clock=Clock())
        m = SimpleNamespace(account=0, ticket=ops.order(0), shard=f"{run_prefix}-0", data_slot=None,
                            replaces_shard=None, predecessor_lost=False)
        ops.start(m, [m.shard])
    cmd = next(c[0][-1] for c in run.calls if c[0][0] == "ssh" and "run -d" in c[0][-1])
    if lease_store is None:
        assert "--label" not in cmd and node_lease.RUN_LABEL_ENV not in cmd  # the argv of main
    else:
        assert f"--label yeto.run={ops.provider.run_id}" in cmd
        assert f"{node_lease.RUN_LABEL_ENV}={ops.provider.run_id}" in cmd  # the node runtime labels its own
    done = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "YETO_SSH_STARTED" in done.stdout, done.stderr
    return ops.provider.run_id


def _probe(run, env, *, sudo=False):
    import subprocess

    done = subprocess.run(["bash", "-c", node_lease.still_runs_script(run, sudo=sudo)], capture_output=True,
                          text=True, env=env)
    return node_lease.parse_probe(done.stdout)


def test_the_probe_finds_a_leased_pools_container_by_its_run_label(tmp_path, monkeypatch):
    env = _node_bin(tmp_path)
    run = _started_container(tmp_path, monkeypatch, env)
    assert _probe(run, env) == (True, ["container:yeto-sampler-f09-0"])
    assert _probe("run-z", env) == (True, [])  # labelled: another run's, not this one's


def test_the_probe_finds_a_rootful_container_through_sudo(tmp_path, monkeypatch):
    env = _node_bin(tmp_path)
    run = _started_container(tmp_path, monkeypatch, env, sudo=True)
    assert not (tmp_path / "podman.json").exists() and (tmp_path / "podman-root.json").exists()  # rootful
    assert _probe(run, env, sudo=True) == (True, ["container:yeto-sampler-f09-0"])
    assert _probe(run, env) == (True, ["container:yeto-sampler-f09-0"])  # sudo -n works: tried anyway
    denied = {**env, "FAKE_SUDO_DENY": "1"}
    # needed and not usable: no engine is confirmed free of rootful containers
    assert _probe(run, denied, sudo=True) == (True, ["unconfirmed:sudo-docker", "unconfirmed:sudo-podman"])
    probe = node_lease.ssh_probe(lambda node, script: __import__("subprocess").run(
        ["bash", "-c", script], capture_output=True, text=True, env=denied), sudo=True)
    assert "not confirmed free" in probe("node-a", f"{run}/pool@id")


def test_a_failed_engine_ps_refuses_the_takeover(tmp_path, monkeypatch):
    env = _node_bin(tmp_path, sudo=False)
    run = _started_container(tmp_path, monkeypatch, env)
    failing = {**env, "FAKE_ENGINE_FAIL": "podman"}
    assert _probe(run, failing) == (True, ["unconfirmed:podman"])
    import subprocess

    objects, clock, events = _objects(), Clock(), []
    a = _leases(f"{run}/pool", objects, clock)
    assert a.acquire("node-a").ok
    clock.t += 601  # the pool controller is gone; its lease ran out
    probe = node_lease.ssh_probe(lambda node, script: subprocess.run(["bash", "-c", script], capture_output=True,
                                                                     text=True, env=failing))
    b = _leases("trainer-x", objects, clock, probe=probe, events=events)
    got = b.acquire("node-a")
    assert not got.ok and "not confirmed free" in got.blocked and "unconfirmed:podman" in got.blocked
    assert [name for name, _ in events] == [node_lease.BLOCKED_EVENT]


def test_an_unlabelled_sampler_container_is_possibly_the_runs(tmp_path, monkeypatch):
    """A sampler container started without a label (before labels, or a pool with no
    lease_store) may be the old run's: not confirmed free, the takeover is refused."""
    env = _node_bin(tmp_path, sudo=False)
    _started_container(tmp_path, monkeypatch, env, lease_store=None)
    assert _probe("pool-old", env) == (True, ["unconfirmed:container:yeto-sampler-f09-0"])
    probe = node_lease.ssh_probe(lambda node, script: __import__("subprocess").run(
        ["bash", "-c", script], capture_output=True, text=True, env=env))
    assert "not confirmed free of run pool-old" in probe("node-a", "pool-old/pool@id")


def test_the_probe_finds_trainer_process_groups_and_venv_samplers(tmp_path):
    import os
    import subprocess
    import time

    from yeto.launch.ssh_trainer_island import STATE_DIR

    env = _node_bin(tmp_path, sudo=False)
    rows = [{"name": "yeto-run-a-isl-i0-n0", "labels": {}}, {"name": "yeto-run-ab-x", "labels": {}}]
    (tmp_path / "docker.json").write_text(json.dumps(rows))
    home = Path(env["HOME"])
    gone = subprocess.Popen(["true"], start_new_session=True)
    gone.wait()  # a process group that ended
    group = subprocess.Popen(["sleep", "30"], start_new_session=True)
    sampler = subprocess.Popen(["bash", "-c", "exec -a yeto-ssh-sampler:run-a:s3:t3 sleep 30"])
    try:
        for island, pgid, rc in (("isl-i0", group.pid, False), ("isl-i1", group.pid, True),
                                 ("isl-i2", gone.pid, False), ("isl-i3", 1, False)):
            d = home / STATE_DIR / "run-a" / island
            d.mkdir(parents=True)
            (d / "pgid").write_text(str(pgid))
            if rc:
                (d / "rc").write_text("0")
        for _ in range(50):  # the exec -a has happened
            if "yeto-ssh-sampler:run-a:s3" in subprocess.run(["ps", "-o", "args=", "-p", str(sampler.pid)],
                                                             capture_output=True, text=True).stdout:
                break
            time.sleep(0.05)
        found = _probe("run-a", env)
        other = _probe("run-z", env)
    finally:
        group.kill()
        sampler.kill()
        group.wait()
        sampler.wait()
    assert found == (True, sorted(["container:yeto-run-a-isl-i0-n0", "process:isl-i0", f"sampler:{sampler.pid}"]))
    assert other == (True, [])  # nothing of another run; the probe does not find itself
    assert os.environ.get("FAKE_STATE_DIR") is None


def test_the_node_runtime_labels_its_containers_only_when_asked(tmp_path):
    from tests.test_ssh_sampler_pool import conf
    from yeto.node import launch as node_launch

    profile = sp.pool_profile(npu_node(), conf(runtime="container"))
    base = dict(name="yeto-sampler-a", image="img", env={}, python="python3", spec_path="/s")
    plain = node_launch.container_argv(profile, **base)
    assert "--label" not in plain  # the argv of main
    labelled = node_launch.container_argv(profile, **base, labels=node_lease.run_labels("pool-b"))
    assert labelled[labelled.index("--label") + 1] == "yeto.run=pool-b"
    assert [a for a in labelled if a not in ("--label", "yeto.run=pool-b")] == plain


def test_a_node_with_an_owed_stop_is_not_taken_over_by_another_run(tmp_path):
    """L4: the pool keeps the lease of a node whose sampler stop is owed; when that lease
    runs out (the pool controller is gone), another run's probe finds the sampler there."""
    objects, clock = _objects(), Clock()
    prov, _ = _provider(tmp_path, objects, clock)
    t = prov.order(0)
    prov.note_shard(t, "s-0")
    prov.mark_lost(t, stopped=False)
    prov.release(0, t)
    assert prov.leases.held() == ["node-a"]
    clock.t += 601  # the pool controller died; nobody renewed
    node_says = "yeto-lease-probe found=sampler:4242\nyeto-lease-probe done\n"
    other = _leases("trainer-a", objects, clock,
                    probe=node_lease.ssh_probe(lambda node, script: SimpleNamespace(stdout=node_says, returncode=0)))
    got = other.acquire("node-a")
    assert not got.ok and "node node-a still runs run pool-b" in got.blocked


# -- review: H3b kept leases ----------------------------------------------------------

def test_kept_leases_never_run_out_until_cleanup_releases_them():
    objects, clock = _objects(), Clock()
    a = _leases("run-a", objects, clock)
    a.acquire("b300-0")
    assert a.keep_all() == ["b300-0"]
    clock.t += 10 ** 6
    b = _leases("run-b", objects, clock, probe=lambda n, p: None)
    assert b.holder("b300-0") == "run-a" and not b.acquire("b300-0").ok
    raw = objects.read("nodes/b300-0/lease/00000001.lease").decode()
    lease = json.loads(raw)
    assert lease["kept"] is True and lease["until"] is None and "Infinity" not in raw  # L4: plain JSON
    assert not b.force_release("b300-0", holder="run-x")  # not that holder: untouched
    assert b.force_release("b300-0", holder="run-a")
    got = b.acquire("b300-0")
    assert got.ok and not got.took_over  # a released lease: no takeover


def test_a_kept_lease_written_with_infinity_reads_as_kept():
    """L4: a lease an earlier build kept (``until`` Infinity) still reads as held."""
    objects, clock = _objects(), Clock()
    a = _leases("run-a", objects, clock)
    a.acquire("b300-0")
    key = "nodes/b300-0/lease/00000001.lease"
    objects.write(key, json.dumps({"token": 1, "holder": "run-a", "until": float("inf"), "released": False,
                                   "kept": True}).encode())
    clock.t += 10 ** 6
    b = _leases("run-b", objects, clock, probe=lambda n, p: None)
    assert b.holder("b300-0") == "run-a" and not b.acquire("b300-0").ok
    from yeto.launch.pool.store_lock import StoreLock

    assert StoreLock.holder_at(objects, "nodes/b300-0/lease", 1, now=clock.t)["holder"] == "run-a"
    objects.write(key, json.dumps({"token": 1, "holder": "run-a", "until": None, "released": False,
                                   "kept": True}).encode())
    assert StoreLock.holder_at(objects, "nodes/b300-0/lease", 1, now=clock.t)["holder"] == "run-a"


def test_force_release_ends_only_kept_or_run_out_leases_unless_forced():
    """L3: cleanup ends a kept lease or one that ran out; a live lease needs force."""
    objects, clock = _objects(), Clock()
    a = _leases("run-a", objects, clock)
    a.acquire("b300-0")
    cleanup = _leases("cleanup", objects, clock)
    assert not cleanup.force_release("b300-0", holder="run-a")  # live: renewed by its launcher
    assert cleanup.holder("b300-0") == "run-a"
    assert cleanup.force_release("b300-0", holder="run-a", force=True)
    assert cleanup.holder("b300-0") is None
    a.acquire("b300-1")
    clock.t += 601
    assert cleanup.force_release("b300-1", holder="run-a")  # ran out: ended for the next run


def test_cleanup_ends_only_the_rows_own_holders_lease(tmp_path, monkeypatch):
    """L3: ``yeto cleanup`` of run 1 never ends the lease of the node now held or kept by
    another run (the ``holder=`` of the ledger row)."""
    from yeto.launch import ssh_trainer_ledger

    objects, clock = _objects(), Clock()
    monkeypatch.setattr(node_lease.NodeLeases, "objects", property(lambda self: objects))
    run1 = _leases("run1/trainer", objects, clock)
    run1.acquire("b300-0")
    run1.acquire("b300-1")
    run1.keep_all()
    other = _leases("run2/trainer", objects, clock)
    run1_b300_1 = node_lease.NodeLeases(STORE, "x", objects=objects, clock=clock)
    assert run1_b300_1.force_release("b300-1", holder=run1.holder_id)  # run 1's node 1 freed earlier
    assert other.acquire("b300-1").ok  # run 2 took it
    other.keep_all()
    nodes = [SimpleNamespace(name="b300-0"), SimpleNamespace(name="b300-1")]
    note = ssh_trainer_ledger.release_node_leases({"lease_store": STORE, "lease_holder": run1.holder_id}, nodes)
    assert note == "; node leases released on ['b300-0']"
    assert other.holder("b300-1") == other.holder_id  # run 2's kept lease is untouched


def test_a_closing_launcher_keeps_the_leases_of_islands_it_did_not_release(tmp_path):
    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    ops = make_ops(tmp_path, FakeCluster(NAMES))
    ops.add("isl", spec(leases=mine))
    ops.start("isl")
    ops.close()  # --keep: the island runs on
    clock.t += 10 ** 6
    assert all(_leases("x", objects, clock).holder(n) == "run1/trainer" for n in NAMES)


def test_a_closing_launcher_has_nothing_to_keep_after_a_confirmed_down(tmp_path):
    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    ops = make_ops(tmp_path, FakeCluster(NAMES))
    ops.add("isl", spec(leases=mine))
    ops.start("isl")
    assert ops.down("isl")
    ops.close()
    assert all(_leases("x", objects, clock).holder(n) is None for n in NAMES)


def test_the_ledger_row_carries_the_lease_and_cleanup_releases_it(tmp_path):
    from yeto.cost.ledger import ResourceLedger
    from yeto.launch import ssh_trainer_ledger

    store = str(tmp_path / "leases")
    mine = NodeLeases(store, "run1/trainer", identity="abc")
    ops = make_ops(tmp_path, FakeCluster(NAMES))
    ops.add("isl", spec(leases=mine))
    ops.start("isl")
    mine.keep_all()
    (row,) = ResourceLedger(tmp_path / "ledger.jsonl").open_resources()
    assert row.attrs["lease_store"] == store and row.attrs["lease_holder"] == "run1/trainer@abc"
    ok, action, _ = ssh_trainer_ledger.cleanup_release(row, ssh=SimpleNamespace(
        run=lambda node, script, timeout: SimpleNamespace(returncode=0)))
    assert ok and action.endswith(f"node leases released on {NAMES}")
    assert all(NodeLeases(store, "x").holder(n) is None for n in NAMES)


def test_a_ledger_row_without_a_lease_is_cleaned_as_before():
    from yeto.launch import ssh_trainer_ledger

    assert ssh_trainer_ledger.release_node_leases({"run_id": "r"}, []) == ""


# -- review: M1 a create that learns the store is not atomic -----------------------------

def test_a_store_found_non_atomic_by_the_claim_itself_is_refused():
    objects = _objects(FakeS3(conditional=False))  # no probe before: the claim's create learns it
    with pytest.raises(NodeLeaseError, match="atomic"):
        _leases("run-a", objects, Clock()).acquire("b300-0")


# -- review: M2 the trainer's wait -----------------------------------------------------

def test_pick_reports_every_five_minutes_with_the_time_waited_and_an_event():
    objects, clock, events, lines = _objects(), Clock(), [], []
    _leases("run-a", objects, clock).acquire("b300-0")
    b = _leases("run-b", objects, clock, events=events)
    t = [0.0]

    def sleep(s):
        t[0] += s

    with pytest.raises(NodeLeased, match="after 900s"):
        pick_nodes(["b300-0"], 1, b, wait_s=900, sleep=sleep, monotonic=lambda: t[0], log=lines.append)
    assert [line.split("waiting ")[1].split("s so far")[0] for line in lines] == ["0", "300", "600"]
    assert [d["waited_s"] for name, d in events if name == node_lease.WAITING_EVENT] == [0, 300, 600]


def test_pick_stops_waiting_at_the_stop_flag():
    objects, clock = _objects(), Clock()
    _leases("run-a", objects, clock).acquire("b300-0")
    calls = []
    with pytest.raises(NodeLeased, match="stopped waiting: STOP"):
        pick_nodes(["b300-0"], 1, _leases("run-b", objects, clock), sleep=calls.append, log=lambda m: None,
                   stop=lambda: "STOP" if calls else None)
    assert calls == [30.0]


def _held_elsewhere(objects, clock, *names):
    other = _leases("run-a", objects, clock)
    for n in names:
        other.acquire(n)
    return other


def test_the_trainer_wait_has_a_limit_and_honours_the_runs_stop_flag():
    from yeto import runs

    objects, clock = _objects(), Clock()
    _held_elsewhere(objects, clock, "b300-0")
    nodes = NodeList([b300(0)], lease_store=STORE)
    with pytest.raises(SshIslandError, match="after 0s"):
        inventory_trainer_nodes(_args(nodes, rl_node_lease_wait_s=0.0), _island(1),
                                leases=_leases("run-b/trainer", objects, clock), name="isl")
    runs.request_stop("run-b")
    with pytest.raises(SshIslandError, match="STOP flag"):
        inventory_trainer_nodes(_args(nodes), _island(1), leases=_leases("run-b/trainer", objects, clock),
                                name="isl")


def test_the_launch_flag_has_the_documented_default():
    from yeto.launch.ssh_trainer_island import DEFAULT_NODE_LEASE_WAIT_S
    from yeto.rl.engine import run_flags

    flag = next(f for f in run_flags.rl_flags() if "--rl-node-lease-wait-s" in f.options)
    assert flag.kwargs["default"] == DEFAULT_NODE_LEASE_WAIT_S == 3600.0


# -- review: M3 a restarted island stays on its journal's nodes --------------------------

def _journal(name, incarnations=("0",)):
    from yeto import runs

    path = runs.run_dir("run-b") / "ssh-islands" / f"{name}-jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: f"ssh:{name}-i{k}" for k in incarnations}))


def test_a_restart_with_a_journal_takes_the_recorded_nodes_or_refuses():
    from yeto.launch import ssh_trainer_island as sti

    objects, clock = _objects(), Clock()
    nodes = NodeList([b300(i) for i in range(4)], lease_store=STORE)
    first = _leases("run-b/trainer@one", objects, clock)
    assert [n.name for n in inventory_trainer_nodes(_args(nodes), _island(2), leases=first, name="isl")] == NAMES[:2]
    assert json.loads(sti._lease_record_path(_args(nodes)).read_text())["islands"] == {"isl": NAMES[:2]}
    _journal("isl")
    first.release_all()  # the launcher died long ago: its leases are gone
    _held_elsewhere(objects, clock, "b300-1")
    restarted = _leases("run-b/trainer@one", objects, clock)
    with pytest.raises(SshIslandError, match="not moving the island to other nodes"):
        inventory_trainer_nodes(_args(nodes), _island(2), leases=restarted, name="isl")
    assert restarted.held() == []  # nothing kept from the refused round


def test_a_restart_with_a_journal_gets_the_same_nodes_even_when_others_are_free_first():
    objects, clock = _objects(), Clock()
    nodes = NodeList([b300(i) for i in range(4)], lease_store=STORE)
    _held_elsewhere(objects, clock, "b300-0")
    first = _leases("run-b/trainer@one", objects, clock)
    assert [n.name for n in inventory_trainer_nodes(_args(nodes), _island(2), leases=first,
                                                    name="isl")] == ["b300-1", "b300-2"]
    _journal("isl")
    first.release_all()
    for n in ("b300-0",):
        _leases("x", objects, clock).force_release(n)  # b300-0 is free now: the island still stays put
    again = _leases("run-b/trainer@one", objects, clock)
    assert [n.name for n in inventory_trainer_nodes(_args(nodes), _island(2), leases=again,
                                                    name="isl")] == ["b300-1", "b300-2"]


def test_one_pick_per_island_and_process():
    objects, clock = _objects(), Clock()
    nodes = NodeList([b300(i) for i in range(4)], lease_store=STORE)
    mine = _leases("run-b/trainer@one", objects, clock)
    a = inventory_trainer_nodes(_args(nodes), _island(2), leases=mine, name="isl-a")
    assert inventory_trainer_nodes(_args(nodes), _island(2), leases=mine, name="isl-a") == a  # the gate, the render
    b = inventory_trainer_nodes(_args(nodes), _island(2), leases=mine, name="isl-b")
    assert {n.name for n in a}.isdisjoint(n.name for n in b)


# -- review: M4 release, gate, dry run --------------------------------------------------

def test_the_schedulers_island_frees_its_nodes_once_a_release_is_confirmed(tmp_path):
    from yeto.launch.ssh_trainer_island import SshTrainerIslands

    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    islands = SshTrainerIslands(spec(leases=mine), name="isl", jobs_path=tmp_path / "jobs.json",
                                ops=FakeCluster(NAMES), scheduled=True, tape=False)
    job = islands.launch(0, None)
    assert mine.held() == NAMES
    assert islands.release(job) and mine.held() == NAMES  # M-A: held between incarnations
    islands.launch(1, None)
    assert islands.release_incarnation(1) and mine.held() == NAMES
    islands.finish_leases()  # the run's loop finished, every incarnation confirmed stopped
    assert mine.held() == [] and all(mine.holder(n) is None for n in NAMES)


def test_another_run_cannot_take_the_nodes_between_two_incarnations(tmp_path):
    """M-A: between the scheduler's incarnations (a cut, a preemption) the nodes stay the
    run's: the keeper renews them, another run's pick waits and its claim is refused."""
    from yeto.launch.ssh_trainer_island import SshTrainerIslands

    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    islands = SshTrainerIslands(spec(leases=mine), name="isl", jobs_path=tmp_path / "jobs.json",
                                ops=FakeCluster(NAMES), scheduled=True, tape=False)
    assert islands.release(islands.launch(0, None))
    for _ in range(4):  # the keeper's rounds while no incarnation runs
        clock.t += mine.renew_every_s
        assert mine.renew_due() == []
    other = _leases("run2/trainer", objects, clock, probe=lambda n, p: None)
    assert all(not other.acquire(n).ok for n in NAMES)
    with pytest.raises(NodeLeased):
        node_lease.pick_nodes(NAMES, 3, other, wait_s=0, sleep=pytest.fail, log=lambda line: None)
    islands.launch(1, None)  # the run's own next incarnation starts on them
    assert sorted(n.name for n in islands.spec.nodes) == NAMES


def test_a_finished_run_keeps_the_leases_of_an_incarnation_not_confirmed_stopped(tmp_path):
    from yeto.launch.ssh_trainer_island import SshTrainerIslands

    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    islands = SshTrainerIslands(spec(leases=mine), name="isl", jobs_path=tmp_path / "jobs.json",
                                ops=FakeCluster(NAMES), scheduled=True, tape=False)
    islands.launch(0, None)  # never released
    islands.finish_leases()
    lease = json.loads(objects.read("nodes/b300-0/lease/00000001.lease"))
    assert lease["kept"] is True  # held for yeto cleanup


def test_the_ssh_binding_ends_the_leases_when_the_loop_finishes(tmp_path):
    from tests.test_rl_oss_store import _objects as store_objects
    from yeto.launch.ssh_trainer_island import SshTrainerIslands, ssh_binding
    from yeto.launch.trainer_island import TrainerIsland

    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    islands = SshTrainerIslands(spec(leases=mine), name="isl", jobs_path=tmp_path / "jobs.json",
                                ops=FakeCluster(NAMES), scheduled=True, tape=False)
    args = SimpleNamespace(rollout_batch_size=2, rl_checkpoint_store="oss://bkt/runs/a",
                           rl_trainer_command_poll_seconds=None, tuning="full")
    island = TrainerIsland(name="isl", cloud="ssh", ssh=spec(leases=mine))
    binding = ssh_binding(args, island, journal_dir=tmp_path, files=store_objects(), islands=islands)
    assert islands.release(islands.launch(0, None)) and mine.held() == NAMES
    binding.on_finished()
    assert mine.held() == []
    plain = SshTrainerIslands(spec(), name="isl", jobs_path=tmp_path / "j2.json", ops=FakeCluster(NAMES),
                              scheduled=True, tape=False)
    assert ssh_binding(args, TrainerIsland(name="isl", cloud="ssh", ssh=spec()), journal_dir=tmp_path,
                       files=store_objects(), islands=plain).on_finished is None  # no lease_store: as main


def test_a_resident_relaunch_keeps_its_nodes_leased(tmp_path):
    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    cluster = FakeCluster(NAMES)
    ops = make_ops(tmp_path, cluster)
    ops.add("isl", spec(leases=mine))
    ops.start("isl")
    cluster.exit("b300-1", 0, 3)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert mine.held() == NAMES  # released incarnation, nodes still the run's
    assert ops.relaunch(None, "isl") == 1 and mine.held() == NAMES


def test_the_gate_checks_the_leased_nodes_only_with_lease_store(monkeypatch):
    from yeto.launch.preflight import gate

    monkeypatch.setattr(gate, "resolve_nodes", lambda entry, inventory: ["as-before"])
    plain = _args(NodeList([b300(i) for i in range(2)]), gpu="ssh:2x8xB300", dry_run=False)
    assert gate.gate_nodes(plain, "ssh:2x8xB300", "inv.yaml") == ["as-before"]
    objects, clock = _objects(), Clock()
    _held_elsewhere(objects, clock, "b300-0")
    leased = _args(NodeList([b300(i) for i in range(3)], lease_store=STORE), gpu="ssh:2x8xB300", dry_run=False)
    mine = _leases("run-b/trainer@one", objects, clock)
    from yeto.launch import ssh_trainer_island as sti

    monkeypatch.setattr(sti, "trainer_leases", lambda args: mine)
    assert [n.name for n in gate.gate_nodes(leased, "ssh:2x8xB300", "inv.yaml")] == ["b300-1", "b300-2"]
    assert mine.held() == ["b300-1", "b300-2"]


def test_the_dry_run_names_what_it_checked_only_with_lease_store():
    from yeto.launch.ssh_trainer_island import trainer_lease_note

    assert trainer_lease_note(_args(NodeList([b300(0), b300(1)])), 2) is None
    note = trainer_lease_note(_args(NodeList([b300(i) for i in range(3)], lease_store=STORE)), 2)
    assert note == {"lease_store": STORE, "note": "dry run: no lease taken; checked first 2 of 3 candidate trainer "
                                                  "node(s); a launch picks the island's nodes by their leases"}


# -- review: M5 the trainer dry run's hint ---------------------------------------------

def test_the_trainer_dry_run_says_the_hint_only_for_an_inventory_with_new_fields(caplog, capsys):
    from yeto.launch.ssh_trainer_island import trainer_lease_hint

    grouped = [SimpleNamespace(name="b300-0", fabric_group="A")]
    with caplog.at_level(logging.INFO, logger="yeto.launch.node_lease"):
        trainer_lease_hint(_args(grouped))
    assert capsys.readouterr().err == f"[launcher] note: {NO_LEASE_HINT}\n" and caplog.messages == []


def test_the_customer_inventory_declares_no_new_field():
    import yaml

    from yeto.launch.node_lease import declares_new_fields

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "examples" / "k31"
                          / "inventory.b300-6x8-baremetal.yaml").read_text())
    assert "lease_store" not in raw and not any("fabric_group" in n for n in raw["nodes"])
    assert not declares_new_fields([SimpleNamespace(name=n["name"]) for n in raw["nodes"]])


# -- review: L1 expiry by the store's clock --------------------------------------------

class _StoreTimes:
    """An object store whose listing gives each object's write time and the store's time
    (``list_meta``, as S3's LastModified and Date): ``store_t`` is the store's clock."""

    def __init__(self, objects):
        self._objects, self.store_t, self.written = objects, 5_000.0, {}

    def __getattr__(self, name):
        return getattr(self._objects, name)

    def write(self, key, data):
        self.written[key] = self.store_t
        return self._objects.write(key, data)

    def create(self, key, data):
        made = self._objects.create(key, data)
        if made:
            self.written[key] = self.store_t
        return made

    def list_meta(self, prefix, *, require_date=False):
        assert require_date  # the lease never takes an object time for the store's clock
        return [(k, self.written.get(k, 0.0)) for k in self._objects.list(prefix)], self.store_t


def test_expiry_follows_the_store_clock_not_the_hosts():
    objects = _StoreTimes(_objects())
    behind, reader = Clock(), Clock()
    behind.t, reader.t = 1.0, 10.0 ** 9
    a = _leases("run-a", objects, behind)  # the holder's clock is far behind
    a.acquire("b300-0")
    # the reader's clock is far ahead: by the hosts' clocks the lease ran out long ago
    b = _leases("run-b", objects, reader)
    assert b.holder("b300-0") == "run-a"
    objects.store_t += 599
    assert not b.acquire("b300-0").ok
    objects.store_t += 2  # 601 s of the store's time without a renewal
    assert b.holder("b300-0") is None and b.acquire("b300-0").took_over


def test_without_store_times_the_local_clock_decides():
    objects, clock = _objects(), Clock()
    _leases("run-a", objects, clock).acquire("b300-0")
    clock.t += 601
    assert _leases("run-b", objects, clock).holder("b300-0") is None


# -- review: L3 a store outage past the lease -------------------------------------------

def test_a_store_outage_past_the_lease_marks_the_node_unconfirmed_with_one_alert():
    objects, clock, events = _objects(), Clock(), []
    a = _leases("run-a", objects, clock, events=events)
    a.acquire("b300-0")
    lock = a._locks["b300-0"]  # noqa: SLF001
    real = lock.renew
    lock.renew = lambda: (_ for _ in ()).throw(OSError("store down"))
    clock.t += 300
    assert a.renew_due() == [] and a.unconfirmed() == {}  # inside the lease: nothing yet
    clock.t += 301
    assert a.renew_due() == ["b300-0"] and a.renew_due() == ["b300-0"]
    assert list(a.unconfirmed()) == ["b300-0"] and a.held() == ["b300-0"]
    assert [(n, d.get("level")) for n, d in events] == [(node_lease.UNCONFIRMED_EVENT, "alert")]  # once
    lock.renew = real
    assert a.renew_due() == [] and a.unconfirmed() == {}
    assert events[-1][0] == "node_lease_confirmed"


def test_an_unconfirmed_island_lease_is_reported_not_treated_as_lost(tmp_path):
    from yeto.launch.ssh_trainer_island import SshTrainerIslands

    objects, clock = _objects(), Clock()
    mine = _leases("run1/trainer", objects, clock)
    islands = SshTrainerIslands(spec(leases=mine), name="isl", jobs_path=tmp_path / "j.json", ops=FakeCluster(NAMES),
                                scheduled=False, tape=False)
    islands.launch(0, None)
    for lock in mine._locks.values():  # noqa: SLF001
        lock.renew = lambda: (_ for _ in ()).throw(OSError("store down"))
    clock.t += 700
    assert islands.lease_lost() is None and sorted(mine.unconfirmed()) == NAMES


# -- review: L4 a store error on an order -----------------------------------------------

def test_a_store_error_on_an_order_is_a_refusal_with_an_event(tmp_path):
    objects, clock = _objects(), Clock()
    prov, events = _provider(tmp_path, objects, clock)

    def broken(node):
        raise OSError("store down")

    prov.leases.acquire = broken
    with pytest.raises(NodeLeased, match="not confirmed") as caught:
        prov.order(0)
    assert error_is(caught.value, REFUSED_ERRORS)  # no provisioning failure for the pool's breaker
    assert events[-1][0] == "node_lease_store_error" and events[-1][1]["node"] == "node-a"
    assert prov._rows() == {}  # noqa: SLF001


def test_storelock_reclaim_takes_back_only_the_same_holders_lease():
    from yeto.launch.pool.store_lock import StoreLock

    objects, clock = _objects(), Clock()
    assert StoreLock.acquire(objects, "L", holder="a", clock=clock) is not None
    assert StoreLock.acquire(objects, "L", holder="b", clock=clock, reclaim=True) is None


def test_a_kept_lease_holds_on_the_store_clock_too():
    objects = _StoreTimes(_objects())
    a = _leases("run-a", objects, Clock())
    a.acquire("b300-0")
    a.keep_all()
    objects.store_t += 10 ** 6  # by the store's clock the lease file was written long ago
    assert _leases("run-b", objects, Clock()).holder("b300-0") == "run-a"


# -- the placement order (CONTRACT: fabric_groups.pick_placement, #892) ------------------

class PlacementError(RuntimeError):
    """As fabric_groups.PlacementError: pick_nodes tells it by class name."""


def _by_group(candidates, need, *, excluded=()):
    """A group placement like fabric_groups.pick_placement: one group (the name's letter)
    with ``need`` usable nodes, group A first; the result carries ``.nodes``."""
    groups: dict[str, list[str]] = {}
    for n in candidates:
        if n not in excluded:
            groups.setdefault(n.split("-")[1][0], []).append(n)
    for group in sorted(groups):
        if len(groups[group]) >= need:
            return SimpleNamespace(group=group, nodes=sorted(groups[group])[:need])
    raise PlacementError(f"no group has {need} usable node(s)")


def test_a_claim_lost_inside_a_group_places_again_without_that_node():
    objects, clock = _objects(), Clock()
    mine, rival = _leases("run-b", objects, clock), _leases("run-a", objects, clock)
    real = mine.acquire

    def acquire(node):  # run A claims g-a2 between run B's read and its claim
        if node == "g-a2":
            rival.acquire("g-a2")
        return real(node)

    mine.acquire = acquire
    nodes = ["g-a1", "g-a2", "g-b1", "g-b2"]
    assert pick_nodes(nodes, 2, mine, place=_by_group, sleep=lambda s: pytest.fail("no wait")) == ["g-b1", "g-b2"]
    assert mine.held() == ["g-b1", "g-b2"]  # g-a1 given back: never a node of two groups


def test_held_and_excluded_nodes_are_left_out_before_the_placement():
    objects, clock = _objects(), Clock()
    _held_elsewhere(objects, clock, "g-a1")
    seen = []

    def place(candidates, need, *, excluded=()):
        seen.append((list(candidates), list(excluded)))
        return _by_group(candidates, need, excluded=excluded)

    got = pick_nodes(["g-a1", "g-a2", "g-a3", "g-a4", "g-b1"], 2, _leases("run-b", objects, clock), place=place,
                     excluded={"g-a3"}, sleep=lambda s: pytest.fail("no wait"))  # e.g. unhealthy or quarantined
    assert seen == [(["g-a2", "g-a4", "g-b1"], ["g-a3"])]  # g-a1 another run's, g-a3 the caller's: not offered
    assert got == ["g-a2", "g-a4"]


def test_no_group_fits_waits_like_too_few_free_nodes():
    objects, clock = _objects(), Clock()
    _held_elsewhere(objects, clock, "g-a1")
    t = [0.0]
    with pytest.raises(NodeLeased, match="after 60s"):
        pick_nodes(["g-a1", "g-a2", "g-b1"], 2, _leases("run-b", objects, clock), place=_by_group, wait_s=60,
                   sleep=lambda s: t.__setitem__(0, t[0] + s), monotonic=lambda: t[0], log=lambda m: None)


def test_a_placement_bug_is_not_taken_for_a_wait():
    objects, clock = _objects(), Clock()

    def broken(candidates, need, *, excluded=()):
        raise KeyError("fabric_group")

    with pytest.raises(KeyError):
        pick_nodes(["g-a1"], 1, _leases("run-b", objects, clock), place=broken)


def test_the_default_placement_is_main_order_and_honours_excluded():
    objects, clock = _objects(), Clock()
    assert pick_nodes(["b300-0", "b300-1", "b300-2"], 2, _leases("run-b", objects, clock),
                      excluded=["b300-0"]) == ["b300-1", "b300-2"]
    assert node_lease.in_order_placement(["x", "y", "z"], 2) == ["x", "y"]
    with pytest.raises(Exception, match="1 usable node"):
        node_lease.in_order_placement(["x", "y"], 2, excluded=["y"])


# -- review 2: L1 same run under another identity -----------------------------------------

def test_the_same_run_under_another_identity_is_an_error_not_a_wait(tmp_path):
    objects, clock = _objects(), Clock()
    first = NodeLeases(STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    assert first.acquire("b300-0").ok
    moved = NodeLeases(STORE, "run1/trainer", identity="bbbb", objects=objects, clock=clock, monotonic=clock,
                       identity_path="/runs/run1/node-leases.json")
    with pytest.raises(NodeLeased) as caught:
        node_lease.pick_nodes(["b300-0", "b300-1"], 2, moved, wait_s=3600, sleep=pytest.fail, log=lambda line: None)
    text = str(caught.value)
    assert "/runs/run1/node-leases.json" in text and "yeto cleanup --run run1" in text
    other = NodeLeases(STORE, "run2/trainer", identity="cccc", objects=objects, clock=clock, monotonic=clock)
    assert other.same_run_conflict(first.holder_id) is None  # another run: it waits as before
    assert first.same_run_conflict(first.holder_id) is None
    excluded = node_lease.pick_nodes(["b300-0", "b300-1"], 1, moved, excluded=["b300-0"], wait_s=0,
                                     sleep=pytest.fail, log=lambda line: None)
    assert excluded == ["b300-1"]  # a node left out anyway is no conflict


def test_a_journal_pinned_island_names_the_identity_file(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti

    objects, clock = _objects(), Clock()
    first = NodeLeases(STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    assert first.acquire("b300-0").ok
    moved = NodeLeases(STORE, "run1/trainer", identity="bbbb", objects=objects, clock=clock, monotonic=clock,
                       identity_path="/runs/run1/node-leases.json")
    monkeypatch.setattr(sti, "_lease_record", lambda args: {"islands": {"isl": ["b300-0"]}})
    monkeypatch.setattr(sti, "_journal_incarnation", lambda args, name: 1)
    trainers = [SimpleNamespace(name=n) for n in ("b300-0", "b300-1")]
    with pytest.raises(SshIslandError, match="node-leases.json.*yeto cleanup --run run1"):
        sti.lease_trainer_nodes(SimpleNamespace(cluster_prefix="run1"), trainers, 1, moved, name="isl", wait_s=0)


def test_the_pool_under_another_identity_fails_the_order(tmp_path):
    objects, clock = _objects(), Clock()
    first, _ = _provider(tmp_path / "a", objects, clock)
    assert first.order(0)
    again, events = _provider(tmp_path / "b", objects, clock)
    again._leases = NodeLeases(STORE, "pool-b/pool", identity="other", objects=objects, clock=clock,  # noqa: SLF001
                               monotonic=clock, identity_path="/state/ssh-pool.lease-identity.json")
    with pytest.raises(sp.SshSamplerError, match="lease-identity.json.*yeto cleanup --run pool-b"):
        again.order(0)
    assert [n for n, _ in events] == ["node_lease_same_run"]
