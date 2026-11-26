"""Spare nodes of a resident ssh: trainer island (yeto.launch.island_spares,
own-cluster-fleet-stability 4.1, 4.2 and 4.4) over the fake SSH cluster of
test_rl_ssh_resident: when a node is lost, how a spare is picked, the replacement and its
journal, waiting for a spare, the replacement limit, and an inventory without spares
behaving as before.

The interfaces of groups 1 and 2 (``yeto.hw.node_health``, ``Node.fabric_group``) are faked
here by their agreed signatures (CONTRACT.zh.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_rl_ssh_resident import (
    Clock,
    FakeCluster,
    _SkySide,
    _SyncThread,
    _cards,  # noqa: F401 - autouse: card checks and the run's resource ledger
    b300,
    make_ops,
    spec,
    wait_for,
)
from yeto.cost.ledger import ResourceLedger
from yeto.launch import island_spares as isp
from yeto.launch.fleet import ABANDONED, RECOVERING, RUNNING, FleetController, RelaunchBlocked
from yeto.launch.island_spares import IslandSpares, NodeRecords
from yeto.launch.ssh_resident import SshIslandOps
from yeto.modal_runner import RoutingOps
from yeto.rl.engine.object_store import DirectoryObjectStore

NAMES = ["b300-0", "b300-1", "b300-2"]
HEALTH_MARK = "yeto-fake-health"


# ------------------------------------------------------------------ fakes of groups 1 and 2
def grouped(node, group):
    """``Node.fabric_group`` (group 2) on a node of this tree, where the field is not there yet."""
    object.__setattr__(node, "fabric_group", group)
    return node


def spare(i: int, group: str | None = None):
    node = b300(i, roles=(isp.SPARE_ROLE,))
    return grouped(node, group) if group is not None else node


@dataclass
class FakeHealthResult:
    checked: bool
    ok_single: bool | None
    ok_multi: bool | None
    reasons: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)


class FakeChecker:
    """``node_health.Checker`` by its agreed signature: one script, a pure parse."""

    def script(self, hcas):
        return f"echo {HEALTH_MARK} {','.join(hcas)}"

    def parse(self, text, hcas, prev):
        bad = "bad" in text
        return FakeHealthResult(True, not bad, not bad, ["XID 79 on GPU 3"] if bad else [])


class FakeTracker:
    """``node_health.HealthTracker`` by its agreed signature: lost after two bad in a row."""

    def __init__(self):
        self.bad = 0

    def update(self, result):
        if not result.checked:
            return "read_failed"
        if result.ok_multi:
            self.bad = 0
            return "ok"
        self.bad += 1
        return "lost" if self.bad >= 2 else "bad_once"


class Cluster(FakeCluster):
    """The fake cluster with the health checker's script and nodes whose health is bad."""

    def __init__(self, names, **kw):
        super().__init__(names, **kw)
        self.sick: set[str] = set()
        self.still_runs: dict[str, list[str]] = {}  # node -> containers the lease probe finds

    def run(self, node, script, *, timeout=300.0):
        if node.name in self.unreachable:  # what was asked of a node that cannot answer
            self.calls.append(("unanswered", node.name, "kill" if "kill -TERM" in script else "other"))
        from yeto.launch.node_lease import PROBE_MARK

        if PROBE_MARK in script:  # node_lease.ssh_probe: what of a run still runs on the node
            import subprocess

            self.calls.append(("lease-probe", node.name))
            if node.name in self.unreachable:
                return subprocess.CompletedProcess([], 255, stdout="", stderr="timed out")
            found = "".join(f"{PROBE_MARK} found=container:{c}\n" for c in self.still_runs.get(node.name, ()))
            return subprocess.CompletedProcess([], 0, stdout=f"{found}{PROBE_MARK} done\n", stderr="")
        if HEALTH_MARK in script:
            import subprocess

            self.calls.append(("health", node.name))
            if node.name in self.unreachable:
                return subprocess.CompletedProcess([], 255, stdout="", stderr="timed out")
            return subprocess.CompletedProcess([], 0, stdout="bad\n" if node.name in self.sick else "ok\n",
                                               stderr="")
        return super().run(node, script, timeout=timeout)

    def lose(self, name):
        """The node loses power: no SSH answer, its streams end."""
        self.unreachable.add(name)
        self.drop(name)


@pytest.fixture(autouse=True)
def _health(monkeypatch):
    monkeypatch.setattr(isp, "_checker_for", lambda family, **kw: FakeChecker())


# node_health.Reading by its agreed fields (verdict, result.facts, degraded)
READING = SimpleNamespace(verdict="lost", result=FakeHealthResult(True, False, False, ["XID 79 on GPU 3"]),
                          degraded=False)


@dataclass
class Lease:
    ok: bool
    holder: str | None
    took_over: bool = False
    previous: str | None = None
    blocked: bool = False


class FakeLeases:
    """``node_lease.NodeLeases`` by its agreed signature."""

    def __init__(self, run_id="run1/trainer", held=None):
        self.run_id = run_id
        self.holder_id = f"{run_id}@launcher-a"  # holders are <run>/<part>@<identity>
        self.holders = dict(held or {})
        self.blocked: set[str] = set()

    def holder(self, node):
        return self.holders.get(node.name)

    def acquire(self, node):
        if node.name in self.blocked:  # a run out lease whose run may still run there
            return Lease(False, None, previous="old-run/trainer@x", blocked=True)
        if self.holders.get(node.name, self.holder_id) != self.holder_id:
            return Lease(False, self.holders[node.name])
        self.holders[node.name] = self.holder_id
        return Lease(True, self.holder_id)

    def release(self, node):
        self.holders.pop(node.name, None)

    def force_release(self, node, *, holder=None, force=False):
        """As NodeLeases: a kept lease ends at once, a live one only with ``force``."""
        if node.name not in self.holders or (holder is not None and self.holders[node.name] != holder):
            return False  # only the given holder's lease
        if node.name not in getattr(self, "kept", []) and not force:
            return False  # a live lease
        self.holders.pop(node.name)
        self.forced = [*getattr(self, "forced", []), (node.name, force)]
        return True

    def held(self):
        return [n for n, h in self.holders.items() if h == self.holder_id and n not in getattr(self, "kept", [])]

    def renew(self, node):
        pass

    store_uri = "s3://leases/site"  # the island's own start uses these (spec.leases)

    def renew_due(self):
        return []

    def lost(self):
        return {}

    def release_all(self, nodes=None):
        for n in nodes or ():
            if self.holders.get(n.name) == self.holder_id:
                self.holders.pop(n.name)

    def keep_all(self, nodes=None):
        names = [n.name for n in nodes or ()]
        self.kept = [*getattr(self, "kept", []), *(n for n in names if n not in getattr(self, "kept", []))]
        return names


def join_stops(spares, timeout=5.0):
    """Wait for the background stops of lost nodes that still answer."""
    for thread in spares.stop_threads:
        thread.join(timeout)
        assert not thread.is_alive()


def islands_spec(n=3, groups=None):
    s = spec(n)
    for i, node in enumerate(s.nodes):
        if groups is not None:
            grouped(node, groups[i])
    return s


def make_spares(tmp_path, island_spec, extra, **kw):
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    resume = kw.pop("resume", lambda: {"source": "release", "step": 7, "ref": "r00000007-v7"})
    return IslandSpares(pool=(*island_spec.nodes, *extra), records=records, resume_probe=resume, **kw), store


def lost_island(tmp_path, *, extra, rank=1, groups=None, clock=None, **kw):
    """A 3-node island with spares ``extra`` whose rank ``rank`` node lost power."""
    cluster = Cluster([*NAMES, *[n.name for n in extra]])
    clock = clock or Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    s = islands_spec(groups=groups)
    spares, store = make_spares(tmp_path, s, extra, **kw)
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    cluster.lose(NAMES[rank])
    assert wait_for(lambda: ops.islands["isl"].job.procs[rank].poll() is not None)
    for _ in range(2):
        status = str(ops.job_status("isl", 0))
    assert status == "FAILED"
    return cluster, ops, spares, store, clock


# ------------------------------------------------------------------ 4.1: when a node is lost
def test_an_ssh_hiccup_while_the_stream_flows_is_no_loss(tmp_path):
    cluster, clock = Cluster([*NAMES, "b300-3"]), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=60.0, unreachable_polls=3)
    s = islands_spec()
    spares, store = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    cluster.unreachable.add("b300-1")  # SSH checks fail, its log stream still flows
    for _ in range(3):
        clock.t += 61
        status = str(ops.job_status("isl", 0))
    assert status == "FAILED"  # the island fails as it always did (relaunched on its nodes)
    assert "does not answer over SSH" in ops.islands["isl"].reason
    assert spares.lost == {} and ops.island_events("isl") == []
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None
    assert json.loads(store.read("fleet/nodes/b300-1/claim.json"))["island"] == "isl"  # still its node
    assert store.read("fleet/nodes/b300-3/claim.json") is None
    for _ in range(5):  # its stop is not confirmed and its stream flows: 2 x 3 checks before a loss,
        assert ops.relaunch(None, "isl") is None  # counted from the relaunch (not the watch's 3)
    assert spares.lost == {}
    cluster.unreachable.clear()
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == NAMES  # the same nodes


def test_a_stream_that_ended_and_no_ssh_answer_is_a_lost_node(tmp_path):
    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)])
    (event,) = ops.island_events("isl")
    assert event["kind"] == isp.NODE_LOST and event["node"] == "b300-1" and event["rank"] == 1
    assert "does not answer over SSH" in event["reason"] and "its stream ended" in event["reason"]
    assert ("unanswered", "b300-1", "kill") not in cluster.calls  # the lost node is not waited for
    assert cluster.count("kill", 0) == 2  # the other nodes are stopped
    q = json.loads(store.read("fleet/nodes/b300-1/quarantine.json"))
    assert q["island"] == "isl" and q["owed"] == [{"run_id": "run1", "island": "isl", "incarnation": 0, "rank": 1,
                                                   "container": "yeto-run1-isl-i0-n1"}]
    assert store.read("fleet/nodes/b300-1/claim.json") is None  # the island gave its claim up
    # the incarnation's row is closed; the lost node owes its stop in a row of its own (its price only)
    (row,) = ResourceLedger(tmp_path / "ledger.jsonl").open_resources()
    assert row.id == "run1/isl-i0-owed-b300-1" and row.attrs["ranks"] == [1]
    assert [n["name"] for n in row.attrs["nodes"]] == ["b300-1"] and row.attrs["usd_per_hour"] == 8 * 2.0
    from yeto.launch.ssh_trainer_ledger import cleanup_release

    cluster.unreachable.discard("b300-1")  # back: a cleanup stops the owed incarnation at its rank
    scripts, real = [], cluster.run
    cluster.run = lambda node, script, **kw: (scripts.append((node.name, script)), real(node, script, **kw))[1]
    ok, _action, _ = cleanup_release(row, ssh=cluster)
    assert ok and ("kill", "b300-1", 0) in cluster.calls
    assert [n for n, sc in scripts if "yeto-run1-isl-i0-n1" in sc] == ["b300-1"]


def test_a_node_whose_stream_ended_is_lost_after_the_relaunch_checks(tmp_path):
    """The island failed for another reason while a node's stream ended; that node is judged
    at the relaunch: no answer ``unreachable_polls`` times in a row makes it lost."""
    cluster, clock = Cluster([*NAMES, "b300-3"]), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    cluster.exit("b300-0", 0, 1)
    cluster.lose("b300-2")
    assert wait_for(lambda: all(ops.islands["isl"].job.procs[r].poll() is not None for r in (0, 2)))
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert spares.lost == {}
    assert ops.relaunch(None, "isl") is None  # one check without an answer
    assert spares.lost == {}
    assert ops.relaunch(None, "isl") == 1  # the second: lost, replaced
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == ["b300-0", "b300-1", "b300-3"]


def test_health_bad_twice_in_a_row_is_a_lost_node(tmp_path):
    cluster, clock = Cluster([*NAMES, "b300-3"]), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec, tracker, checker = ops.islands["isl"], FakeTracker(), FakeChecker()
    cluster.sick.add("b300-2")
    verdicts = []
    for _ in range(2):  # the resident tick of group 1: the checker's verdict, then the hook on lost
        done = cluster.run(s.nodes[2], checker.script(isp.hcas_of(s.nodes[2])))
        verdict = tracker.update(checker.parse(done.stdout, [], None))
        verdicts.append(verdict)
        if verdict == "lost":
            result = checker.parse(done.stdout, [], None)
            ops._node_health_lost(rec, 2, result.reasons, SimpleNamespace(verdict="lost", result=result,
                                                                         degraded=False))
    assert verdicts == ["bad_once", "lost"]
    assert str(ops.job_status("isl", 0)) == "FAILED" and "node health bad in 2 checks in a row: XID 79" in rec.reason
    (event,) = ops.island_events("isl")
    assert event["kind"] == isp.NODE_LOST and event["node"] == "b300-2"
    join_stops(spares)
    assert cluster.count("kill", 0) == 3  # it still answers: stopped now (in the background)
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == ["b300-0", "b300-1", "b300-3"]


def test_without_spares_a_health_loss_relaunches_on_the_same_nodes(tmp_path):
    cluster = Cluster(NAMES)
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    ops.add("isl", islands_spec())
    ops.start("isl")
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], READING)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert ops.island_events("isl") == [] and ops.spares_view("isl") is None
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == NAMES


# ------------------------------------------------------------------ the replacement
@pytest.mark.parametrize("rank", [1, 0])
def test_a_spare_takes_the_lost_nodes_place_in_the_same_layout(tmp_path, rank):
    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)], rank=rank)
    ops.island_events("isl")
    assert ops.relaunch(None, "isl") == 1
    new = list(NAMES)
    new[rank] = "b300-3"
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == new
    rec = ops.islands["isl"]
    assert [n.name for n in rec.spec.nodes] == new and rec.spec.run_script == "echo island"
    script0 = cluster.runs[new[0]][1]["script"]
    # node 0's address is the rendezvous: a lost rank 0 hands it to the spare
    assert f"export SKYPILOT_NODE_IPS='{rec.spec.addresses()[0]}" in script0
    assert rec.spec.addresses()[rank] == "198.51.100.13"
    (event,) = ops.island_events("isl")
    assert event == {"kind": isp.NODE_REPLACED, "island": "isl", "rank": rank, "lost": NAMES[rank],
                     "spare": "b300-3", "incarnation": 1, "reason": event["reason"], "resume_step": 7,
                     "resume": {"source": "release", "step": 7, "ref": "r00000007-v7"}, "group": None}
    assert json.loads(store.read("fleet/nodes/b300-3/claim.json"))["island"] == "isl"
    journal = json.loads((tmp_path / "runs/run1/ssh-islands/isl-spares.json").read_text())
    assert journal["nodes"] == {"0": NAMES, "1": new}
    assert [r["spare"] for r in journal["replacements"]] == ["b300-3"]
    view = ops.spares_view("isl")
    assert view["nodes"] == new and view["lost"] == [] and view["replacements"] == 1
    assert view["waiting_spare"] is False and view["spare_nodes"] == [NAMES[rank]]
    assert str(ops.job_status("isl", 1)) == "RUNNING"
    # the new incarnation's ledger row names its own nodes (a cleanup stops it there)
    rows = {r.id: r for r in ResourceLedger(tmp_path / "ledger.jsonl").open_resources()}
    assert [n["name"] for n in rows["run1/isl-i1"].attrs["nodes"]] == new


def test_a_restarted_launcher_follows_the_island_onto_its_spare(tmp_path):
    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    rec = ops.islands["isl"]
    rec.lock.close()
    for proc in rec.job.procs:
        proc.terminate()
    again = make_ops(tmp_path, cluster, health_every_s=0)
    fresh, _ = make_spares(tmp_path, islands_spec(), [spare(3)])
    again.add("isl", islands_spec(), spares=fresh)  # the inventory's island nodes, as at the start
    assert [n.name for n in again.islands["isl"].spec.nodes] == ["b300-0", "b300-3", "b300-2"]
    assert again.start("isl") == 1  # reattached on the spare, no second start
    assert cluster.count("start", 1) == 3 and cluster.count("follow", 1) == 3
    assert ("follow", "b300-3", 1) in cluster.calls and ("status", "b300-1", 1) not in cluster.calls


def test_a_spare_whose_start_fails_is_quarantined_and_the_next_one_is_taken(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)])
    real = cluster.run

    def run(node, script, *, timeout=300.0):
        if node.name == "b300-3" and script.startswith("docker pull"):
            import subprocess

            return subprocess.CompletedProcess([], 1, stdout="", stderr="no space left on device")
        return real(node, script, timeout=timeout)

    cluster.run = run
    assert ops.relaunch(None, "isl") is None
    assert "its start as a spare failed" in json.loads(store.read("fleet/nodes/b300-3/quarantine.json"))["reason"]
    assert [n.name for n in ops.islands["isl"].spec.nodes] == NAMES  # back on the old nodes
    assert store.read("fleet/nodes/b300-3/claim.json") is None
    clock.t += 61
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == ["b300-0", "b300-4", "b300-2"]


# ------------------------------------------------------------------ 4.2: which spare
def test_a_spare_of_another_fabric_group_is_never_picked(tmp_path):
    groups = ["A", "A", "A"]
    cluster, ops, spares, store, clock = lost_island(
        tmp_path, extra=[spare(3, "B"), spare(4, "A")], groups=groups)
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == ["b300-0", "b300-4", "b300-2"]
    assert ("health", "b300-3") not in cluster.calls  # not even checked
    assert ops.island_events("isl")[-1]["group"] == "A"


def test_only_a_spare_of_another_group_is_waited_on(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3, "B")], groups=["A", "A", "A"])
    assert ops.relaunch(None, "isl") is None
    waiting = [e for e in ops.island_events("isl") if e["kind"] == isp.WAITING_SPARE]
    assert waiting and waiting[0]["reason"].startswith("fabric group A has no free healthy spare node")
    assert "b300-3: fabric group B" in waiting[0]["reason"]
    assert ops.waiting_spare("isl")


def test_quarantined_held_leased_and_unhealthy_spares_are_skipped(tmp_path):
    leases = FakeLeases(held={"b300-5": "other-run/trainer@b"})
    extra = [spare(3), spare(4), spare(5), spare(6), spare(7), spare(8)]
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=extra, leases=leases)
    spares.records.quarantine("b300-3", island="isl", reason="XID 79 earlier")
    leases.blocked.add("b300-7")  # its earlier run may still hold the cards: not taken over
    assert spares.records.claim("b300-4", "isl-other") is None  # another island of this run holds it
    cluster.sick.add("b300-6")
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == ["b300-0", "b300-8", "b300-2"]
    assert leases.holders["b300-8"] == "run1/trainer@launcher-a"
    assert "b300-7" not in leases.holders and spares.records.holder("b300-7") is None  # claim given back
    # a quarantined, held or leased spare is skipped before any SSH to it; the sick one was checked
    assert {c[1] for c in cluster.calls if c[0] == "health"} == {"b300-6", "b300-7", "b300-8"}


def test_a_released_node_can_be_picked_again(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])  # no other spare: the lost node waits
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    lost = ops.islands["isl"].spec.nodes[1]
    cluster.unreachable.discard("b300-1")  # repaired
    clock.t += 61
    assert ops.relaunch(None, "isl") is None  # still quarantined: not picked
    leases = FakeLeases(held={"b300-1": "run1/trainer@launcher-a"})
    leases.keep_all(nodes=[lost])  # kept while it was quarantined
    ok, message = isp.release_node(spares.records, lost, cluster, kill_grace_s=1, leases=leases)
    assert ok, message
    assert "b300-1" not in leases.holders  # force_release once its owed stop ran
    assert ("kill", "b300-1", 0) in cluster.calls  # the stop it owed ran first
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None
    clock.t += 121
    assert ops.relaunch(None, "isl") == 1
    assert [c[1] for c in cluster.calls if c[0] == "start" and c[2] == 1] == NAMES


def test_release_keeps_the_quarantine_while_the_owed_stop_fails(tmp_path):
    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)])
    ok, message = isp.release_node(spares.records, ops.islands["isl"].spec.nodes[1], cluster, kill_grace_s=1)
    assert not ok and "rc 255" in message
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None


def test_two_islands_of_a_run_never_take_one_spare(tmp_path):
    records = NodeRecords(DirectoryObjectStore(tmp_path / "store"), run_id="run1", log=lambda m: None)
    assert records.claim("b300-3", "isl-a") is None
    assert records.claim("b300-3", "isl-b") == "isl-a"
    assert records.claim("b300-3", "isl-a") is None  # its own claim again
    records.unclaim("b300-3", "isl-b")  # not its claim: kept
    assert records.holder("b300-3") == "isl-a"


def test_the_pick_goes_by_node_name(tmp_path):
    extra = [spare(6), spare(4), spare(5)]
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=extra)
    assert ops.relaunch(None, "isl") == 1
    assert ("start", "b300-4", 1) in cluster.calls


def test_spares_must_fit_the_island():
    island = spec(2).nodes
    with pytest.raises(isp.SpareError, match="card_per_node|cards_per_node 4"):
        isp.check_spares(island, [b300(3, roles=(isp.SPARE_ROLE,), cards_per_node=4)])
    with pytest.raises(isp.SpareError, match="roles trainer and trainer-spare"):
        isp.check_spares(island, [b300(3, roles=("trainer", isp.SPARE_ROLE))])
    isp.check_spares(island, [b300(3, roles=(isp.SPARE_ROLE,))])
    assert isp.spare_nodes([*island, b300(3, roles=(isp.SPARE_ROLE,))]) == [b300(3, roles=(isp.SPARE_ROLE,))]


# ------------------------------------------------------------------ 4.4: waiting, limit, the fleet
def test_without_a_free_spare_the_island_waits_with_a_backoff(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    spares.records.claim("b300-3", "isl-other")
    assert ops.relaunch(None, "isl") is None
    events = ops.island_events("isl")
    assert [e["kind"] for e in events] == [isp.NODE_LOST, isp.WAITING_SPARE]
    assert events[1]["reason"].startswith("the inventory has no free healthy spare node (b300-1: quarantined")
    assert "; b300-3: held by island isl-other); placement: the island needs 3 node(s); 2 of 4 candidate(s) are free" \
        in events[1]["reason"], events[1]["reason"]
    assert events[1]["next_check_s"] == 60.0
    waits = []
    for _ in range(6):  # each check after the backoff moves the next one further out
        before, due = clock.t, spares.next_check
        for _ in range(200):
            assert ops.relaunch(None, "isl") is None
            if spares.next_check != due:
                break
            clock.t += 10
        waits.append(clock.t - before)
    assert waits == [60, 120, 240, 480, 900, 900]
    assert ops.island_events("isl") == []  # the same reason: one event, not one per check
    spares.records.unclaim("b300-3", "isl-other")  # the other island is done with it
    clock.t += 900
    assert ops.relaunch(None, "isl") == 1
    assert not ops.waiting_spare("isl")


def test_the_replacement_limit_stops_the_island(tmp_path):
    extra = [spare(i) for i in range(3, 8)]
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=extra)
    for k in range(3):
        assert ops.relaunch(None, "isl") == k + 1
        victim = ops.islands["isl"].spec.nodes[1].name
        cluster.lose(victim)
        assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
        for _ in range(2):
            ops.job_status("isl", k + 1)
    ops.island_events("isl")
    with pytest.raises(RelaunchBlocked, match=r"3 node replacement\(s\) within 24 h \(max 3\)"):
        ops.relaunch(None, "isl")
    (event,) = ops.island_events("isl")
    assert event["kind"] == isp.REPLACEMENT_LIMIT
    assert [(r["lost"], r["spare"]) for r in event["replaced"]] == [
        ("b300-1", "b300-3"), ("b300-3", "b300-4"), ("b300-4", "b300-5")]
    # a day later the window has room again (the count is kept on the wall clock)
    ops.wall = lambda: 1e10
    spares.limited = False
    assert ops.relaunch(None, "isl") == 4


def _controller(ops, names=("isl",), **kw):
    return FleetController(learners={n: (ops.islands[n].spec, 0) for n in names}, syncer=None,
                           sky_ops=RoutingOps(_SkySide(), None, ssh_ops=ops), poll_interval=30.0,
                           recover_timeout=kw.pop("recover_timeout", 600.0), thread_cls=_SyncThread,
                           fixed_roster=kw.pop("fixed_roster", False), **kw)


class FleetLog:
    def __init__(self):
        self.rows = []

    def event(self, kind, name, **fields):
        self.rows.append((kind, name, fields))

    def maybe_tick(self):
        pass

    def cost_tick(self):
        pass


def test_the_fleet_waits_past_its_recover_timeout_for_a_spare_then_replaces(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    spares.records.claim("b300-3", "isl-other")
    log = FleetLog()
    ctl = _controller(ops, recover_timeout=600.0, fleet_log=log)
    ctl.ops.sky_ops.t = 0.0
    ctl.ops.now = lambda: clock.t
    rec = ctl.learners["isl"]
    ctl._poll_recorded(rec)  # FAILED -> recovering, first relaunch: waiting
    assert rec["state"] == RECOVERING
    for _ in range(40):  # 40 polls of 30 s: far past the 600 s recover timeout
        clock.t += 30
        ctl._poll_recorded(rec)
    assert rec["state"] == RECOVERING  # not given up while it waits for a spare
    kinds = [k for k, _, _ in log.rows]
    assert "island_node_lost" in kinds and "island_waiting_spare" in kinds
    spares.records.unclaim("b300-3", "isl-other")
    for _ in range(40):
        clock.t += 30
        ctl._poll_recorded(rec)
        if rec["state"] == RUNNING:
            break
    assert rec["state"] == RUNNING and rec["job_id"] == 1
    replaced = [f for k, _, f in log.rows if k == "island_node_replaced"]
    assert replaced and replaced[0]["lost"] == "b300-1" and replaced[0]["spare"] == "b300-3"
    assert replaced[0]["resume_step"] == 7


def test_past_the_limit_the_island_is_given_up_and_the_fleet_goes_on(tmp_path):
    """Two islands of one launch: the one that keeps losing nodes is given up at its
    replacement limit (abandoned: run up's fleet.min_islands judges its record); the
    other island runs on."""
    cluster = Cluster([*NAMES, "c-0", "c-1", *[f"b300-{i}" for i in range(3, 8)]])
    clock = Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=1)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(i) for i in range(3, 8)], max_replacements=1)
    ops.add("isl", s, spares=spares)
    other = spec(2)
    other = type(other)(**{**other.__dict__, "nodes": tuple(b300(20 + i) for i in range(2))})
    cluster.runs.update({n.name: {} for n in other.nodes})
    cluster.streams.update({n.name: [] for n in other.nodes})
    ops.add("other", other)
    ops.start("isl")
    ops.start("other")
    log = FleetLog()
    ctl = _controller(ops, names=("isl", "other"), recover_timeout=600.0, fleet_log=log)
    ctl.ops.now = lambda: clock.t
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(3):
        for rec in ctl.learners.values():
            ctl._poll_recorded(rec)
    assert ctl.learners["isl"]["job_id"] == 1  # the first loss: replaced
    cluster.lose("b300-3")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(3):
        clock.t += 30
        for rec in ctl.learners.values():
            ctl._poll_recorded(rec)
    assert ctl.learners["isl"]["state"] == ABANDONED
    assert "ABANDONED" in ctl.learners["isl"]["exit"]
    assert ctl.learners["other"]["state"] == RUNNING and str(ops.job_status("other", 0)) == "RUNNING"
    assert cluster.count("start", 1) == 3 and ("start", "b300-20", 1) not in cluster.calls  # never relaunched
    limit = [f for k, n, f in log.rows if k == "island_replacement_limit"]
    assert limit and [r["spare"] for r in limit[0]["replaced"]] == ["b300-3"]


# ------------------------------------------------------------------ 4.4: no spare node, as before
def test_an_inventory_without_spares_loses_a_node_as_before(tmp_path):
    """The 6 x 8 B300 K3.1 layout takes every node: no spare. A lost node fails the island,
    its stop is waited for on every node (the lost one too), no new start runs while it does
    not answer, nothing is written under fleet/nodes, and the fleet gives the island up at
    its recover timeout."""
    cluster, clock = Cluster(NAMES), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    ops.add("isl", islands_spec())
    ops.start("isl")
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    assert "rank 1 (b300-1) does not answer over SSH (2 checks in a row)" in ops.islands["isl"].reason
    assert cluster.count("kill", 0) == 2 and ("unanswered", "b300-1", "kill") in cluster.calls  # asked too
    assert ops.relaunch(None, "isl") is None and cluster.count("start", 1) == 0
    assert ops.island_events("isl") == [] and ops.spares_view("isl") is None and not ops.waiting_spare("isl")
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "runs/run1/ssh-islands/isl-spares.json").exists()
    log = FleetLog()
    ctl = _controller(ops, recover_timeout=600.0, fleet_log=log)
    ctl.ops.now = lambda: clock.t
    rec = ctl.learners["isl"]
    for _ in range(30):
        clock.t += 30
        ctl._poll_recorded(rec)
    assert rec["state"] == ABANDONED
    assert not [k for k, _, _ in log.rows if k.startswith(("island_node", "island_waiting", "island_replacement"))]


def test_the_dry_run_names_spares_only_when_the_inventory_has_them(tmp_path):
    from yeto.launch.ssh_resident import dry_run_record

    s = spec(2)
    args = SimpleNamespace(rl_node_inventory_nodes=list(s.nodes), tensor_parallel=1, pipeline_parallel=1,
                           context_parallel=1, training_mode="rl", rl_ssh_trainer_workdir="/opt/yeto")
    assert "spares" not in dry_run_record(args, s, "isl")
    args.rl_node_inventory_nodes = [*s.nodes, spare(3, "A")]
    view = dry_run_record(args, s, "isl")["spares"]
    assert view["nodes"] == ["b300-3"] and view["fabric_groups"]["b300-3"] == "A"
    assert view["waiting_spare_s"] == [60.0, 900.0] and view["max_replacements"] == {"count": 3, "window_h": 24.0}


def test_spares_from_args_is_none_without_spare_nodes(tmp_path):
    s = spec(2)
    args = SimpleNamespace(rl_node_inventory_nodes=list(s.nodes), cluster_prefix="run1", rl_checkpoint_store=None)
    assert isp.spares_from_args(args, s) is None
    from dataclasses import replace

    args.rl_node_inventory_nodes = [*s.nodes, replace(b300(3, roles=(isp.SPARE_ROLE,), address=None),
                                                      host="ops@spare-alias")]
    with pytest.raises(isp.SpareError, match="no rendezvous address"):
        isp.spares_from_args(args, s)


def test_latest_checkpoint_step_reads_the_round_cut_then_the_release(tmp_path):
    store = DirectoryObjectStore(tmp_path / "s")
    assert isp.latest_checkpoint_step(store) is None
    store.write("policy-releases/LATEST.json", json.dumps({"seq": 3, "version": 12, "path": "r3-v12"}).encode())
    assert isp.latest_checkpoint_step(store) == {"source": "release", "step": 12, "ref": "r3-v12",
                                                 "key": "policy-releases/LATEST.json"}
    store.write("LATEST", json.dumps({"seq": 1, "cut_id": "r000009-i2"}).encode())
    assert isp.latest_checkpoint_step(store) == {"source": "round_cut", "step": 9, "ref": "r000009-i2",
                                                 "key": "LATEST"}


def test_the_health_hook_is_safe_in_or_out_of_the_lock_and_once_per_loss(tmp_path):
    """The node health watch may call the hook from inside job_status's watch (the ops lock
    held) or outside it, on the tick the node turned lost or again later: no deadlock, one
    loss, one failure, one stop."""
    import threading

    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    rec.health = {2: SimpleNamespace(record=lambda: {"notes": ["xid 79 since the last check"]})}
    with ops._lock:  # as from inside the watch
        ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], READING)
    assert "notes: xid 79 since the last check" in rec.reason
    done = threading.Event()
    worker = threading.Thread(target=lambda: (ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], READING),
                                              done.set()))
    worker.start()  # outside the lock, from another thread, a later tick
    assert done.wait(5.0)
    assert [e["kind"] for e in ops.island_events("isl")] == [isp.NODE_LOST]
    join_stops(spares)
    assert cluster.count("kill", 0) == 3  # the lost node once (in the background), then the others
    assert str(ops.job_status("isl", 0)) == "FAILED"
    ops._node_health_lost(rec, 9, ["no such rank"], READING)  # a rank of another layout: ignored
    assert ops.relaunch(None, "isl") == 1


def _inventory(tmp_path, nodes):
    import yaml

    rows = []
    for n in nodes:
        rows.append({"name": n.name, "host": n.host, "device_family": n.device_family, "cpu_arch": n.cpu_arch,
                     "card": n.card, "cards_per_node": n.cards_per_node, "hbm_gib": n.hbm_gib,
                     "host_memory_gib": n.host_memory_gib, "fabric": n.fabric, "address": n.address,
                     "runtime": n.runtime, "image": n.image, "docker_args": list(n.docker_args),
                     **({"roles": list(n.roles)} if n.roles else {})})
    path = tmp_path / "inventory.yaml"
    path.write_text(yaml.safe_dump({"nodes": rows}))
    return str(path)


def test_release_node_checks_the_node_runs_its_owed_stops_and_removes_its_record(tmp_path):
    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)])
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    common = dict(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster, objects=store,
                  kill_grace_s=1, say=lines.append)
    assert isp.release_node_main(**common, dry_run=True) == 0
    assert any("owes the stop of isl incarnation 0 (rank 1, container yeto-run1-isl-i0-n1)" in ln for ln in lines)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None
    assert isp.release_node_main(**common) == 1  # still unreachable: the check fails, the record stays
    assert "stays quarantined: unreachable" in lines[-1]
    cluster.unreachable.discard("b300-1")
    cluster.sick.add("b300-1")
    assert isp.release_node_main(**common) == 1 and "unhealthy" in lines[-1]
    cluster.sick.clear()
    leases = FakeLeases(held={"b300-1": "run1/trainer@launcher-a"})
    assert isp.release_node_main(**common, leases=leases) == 0
    assert ("kill", "b300-1", 0) in cluster.calls and store.read("fleet/nodes/b300-1/quarantine.json") is None
    assert "b300-1" not in leases.holders and "released (1 owed stop(s) run)" in lines[-1]
    assert isp.release_node_main(**common) == 0 and "is not quarantined" in lines[-1]
    assert isp.release_node_main(**{**common, "name": "nope"}) == 2


def test_a_node_whose_ssh_fails_is_unreachable_whatever_the_checker_reads():
    cluster = Cluster(["b300-3"])
    cluster.unreachable.add("b300-3")
    assert isp.node_check(spare(3), cluster, multi=True) == (False, "unreachable (rc 255)")
    cluster.unreachable.clear()
    assert isp.node_check(spare(3), cluster, multi=True) == (True, "healthy")


def test_a_spare_of_the_reserved_unavailable_group_never_joins_a_multi_node_island(tmp_path):
    cluster, ops, spares, store, clock = lost_island(
        tmp_path, extra=[spare(3, "unavailable"), spare(4, "A")], groups=["A", "A", "A"])
    spares.records.claim("b300-4", "isl-other")
    assert ops.relaunch(None, "isl") is None  # only the unavailable one is left: waiting
    reason = [e for e in ops.island_events("isl") if e["kind"] == isp.WAITING_SPARE][0]["reason"]
    assert "b300-3: fabric group unavailable (never in a multi-node island)" in reason
    assert ("health", "b300-3") not in cluster.calls


def test_a_spare_whose_claim_is_lost_to_another_island_is_excluded_and_the_next_one_taken(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)])
    real = spares.records.claim

    def racing(name, island):  # another island of the run claims b300-3 between the read and the claim
        if name == "b300-3":
            real(name, "isl-other")
        return real(name, island)

    spares.records.claim = racing
    assert ops.relaunch(None, "isl") == 1
    assert ("start", "b300-4", 1) in cluster.calls
    assert spares.records.holder("b300-3") == "isl-other"


def test_the_pick_asks_pick_placement_and_excludes_a_failed_candidate(tmp_path, monkeypatch):
    """The real fabric_groups.pick_placement: the island's nodes that stay are ``required``
    (need: the island's size), every node that is out is in ``excluded`` with its reason
    (the lost node quarantined), and a spare whose health check fails is added to it, with
    the check's reason, before the pick asks again."""
    from yeto.launch import fabric_groups

    asked = []
    real = fabric_groups.pick_placement

    def pick_placement(candidates, need, *, excluded=(), required=(), inventory=None):
        asked.append(([n.name for n in candidates], need, dict(excluded), [n.name for n in required]))
        return real(candidates, need, excluded=excluded, required=required, inventory=inventory)

    monkeypatch.setattr(fabric_groups, "pick_placement", pick_placement)
    cluster, ops, spares, store, clock = lost_island(
        tmp_path, extra=[spare(3, "A"), spare(4, "A"), spare(5, "B")], groups=["A", "A", "A"])
    cluster.sick.add("b300-3")
    assert ops.relaunch(None, "isl") == 1
    assert ("start", "b300-4", 1) in cluster.calls
    stay = ["b300-0", "b300-2"]
    pool = [*stay, "b300-1", "b300-3", "b300-4"]  # b300-5 of group B is no candidate
    assert all(a[0] == pool and a[1] == 3 and a[3] == stay for a in asked)
    first, second = asked[0][2], asked[1][2]
    assert list(first) == ["b300-1"] and first["b300-1"].startswith("quarantined (rank 1 (b300-1)")
    assert sorted(second) == ["b300-1", "b300-3"] and second["b300-3"] == "unhealthy: XID 79 on GPU 3"
    assert len(asked) == 4  # the checks first (outside the ops lock), then the pick under it: the same order
    assert started(cluster, 1) == ["b300-0", "b300-4", "b300-2"]  # the nodes that stay, unchanged


def started(cluster, k):
    return [c[1] for c in cluster.calls if c[0] == "start" and c[2] == k]


def test_a_replacement_keeps_the_nodes_that_stay_when_another_group_has_more_free_nodes(tmp_path):
    extra = [spare(3, "A"), *(spare(i, "B") for i in range(4, 8))]
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=extra, groups=["A", "A", "A"])
    assert ops.relaunch(None, "isl") == 1
    assert started(cluster, 1) == ["b300-0", "b300-3", "b300-2"]  # not moved to group B's four free nodes


def test_a_placement_error_is_the_wait_reason_and_other_errors_are_raised(tmp_path, monkeypatch):
    from yeto.launch import fabric_groups

    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3, "B")], groups=["A", "A", "A"])
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    reason = spares.wait_reason
    assert "placement: the island needs 3 healthy nodes of one fabric group" in reason, reason
    a0, a1, a2 = grouped(b300(0), "A"), grouped(b300(1), "A"), spare(2, "A")
    with pytest.raises(TypeError):  # one string as excluded: the real function's TypeError is raised
        fabric_groups.pick_placement([a0, a1, a2], 3, excluded="b300-1")

    def broken(candidates, need, *, excluded=(), required=(), inventory=None):
        raise TypeError("a bug inside pick_placement")

    monkeypatch.setattr(fabric_groups, "pick_placement", broken)
    with pytest.raises(TypeError, match="a bug inside"):
        isp._pick_placement([a2], 3, required=[a0, a1])


def test_a_placement_that_moves_a_node_that_stays_is_no_placement(monkeypatch):
    from yeto.launch import fabric_groups

    a0, a1, a2, a3 = grouped(b300(0), "A"), grouped(b300(1), "A"), spare(2, "A"), spare(3, "A")
    assert isp._pick_placement([a2, a3], 3, required=[a0, a1])[0].name == "b300-2"
    node, why = isp._pick_placement([a2, a3], 3, required=[a0, a1], excluded={"b300-1": "quarantined"})
    assert node is None and "required node(s) excluded: b300-1: quarantined" in why
    monkeypatch.setattr(fabric_groups, "pick_placement", lambda candidates, need, *, excluded=(), required=(),
                        inventory=None: SimpleNamespace(group="A", nodes=(a0, a2, a3)))  # drops b300-1
    node, why = isp._pick_placement([a2, a3], 3, required=[a0, a1])
    assert node is None and "drops a node that stays" in why


def test_the_release_node_command_parses_its_flags_and_runs_release_node_main(tmp_path):
    import argparse

    from yeto import cluster_release_node as crn

    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)])
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    parser = argparse.ArgumentParser(prog="yeto cluster")
    crn.add_parser(parser.add_subparsers(dest="cluster_command"))
    args = parser.parse_args(["release-node", "b300-1", "--inventory", inv, "--store", str(tmp_path / "store"),
                              "--dry-run", "--kill-grace-s", "5"])
    assert (args.cluster_command, args.name, args.dry_run, args.force, args.kill_grace_s) == (
        "release-node", "b300-1", True, False, 5)
    lines = []
    assert crn.main(args, ssh=cluster, objects=store, say=lines.append) == 0
    assert "dry run: would check b300-1, run 1 stop(s)" in lines[-1]
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None
    args = parser.parse_args(["release-node", "b300-9", "--inventory", inv, "--store", str(tmp_path / "store")])
    assert crn.main(args, ssh=cluster, objects=store, say=lines.append) == 2  # not in the inventory


def test_yeto_cluster_release_node_runs_from_the_cluster_command(tmp_path, capsys):
    import argparse

    from yeto import cluster_cli

    cluster, ops, spares, store, _ = lost_island(tmp_path, extra=[spare(3)])
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    parser = argparse.ArgumentParser(prog="yeto")
    cluster_cli.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["cluster", "release-node", "b300-1", "--inventory", inv, "--store",
                              str(tmp_path / "store"), "--dry-run"])
    assert cluster_cli.main(args) == 0  # the documented dry run: no SSH, nothing removed
    out = capsys.readouterr().out
    assert "node b300-1: quarantined by island isl of run run1" in out
    assert "dry run: would check b300-1, run 1 stop(s) and remove fleet/nodes/b300-1/quarantine.json" in out
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None
    usage = argparse.ArgumentParser(prog="yeto")
    cluster_cli.add_parser(usage.add_subparsers(dest="command"))
    with pytest.raises(SystemExit):
        usage.parse_args(["cluster", "--help"])
    assert "{probe,check,release-node}" in capsys.readouterr().out
