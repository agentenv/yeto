"""A resident ssh: island relaunched on the same nodes (own-cluster 4.5 review): its node checks kept over
the relaunch (``reset: true``, the old job's times) must not leave a healthy running island reading
"health unread (stale)" for good. The first check of the new job writes the record again
(``FleetController._health_key`` counts ``reset``), so the record's checks are the new job's."""

from __future__ import annotations

import json

from test_dashboard_island_spares import cards, reducer, smoke
from test_node_health_wiring import HealthCluster, _island, _tick, _wait
from test_rl_ssh_resident import NAMES, _controller
from test_rl_ssh_resident import _cards  # noqa: F401 - autouse

from yeto.dashboard import fleet_islands
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.fleet import FleetController
from yeto.rl.engine.object_store import DirectoryObjectStore


def test_a_relaunch_on_the_same_nodes_reads_healthy_once_it_runs(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    wall = [2_000_000_000.0]
    for w in ops.islands["isl"].health.values():
        w.clock = lambda: wall[0]
    objects = DirectoryObjectStore(str(tmp_path / "store"))
    store = lifecycle.IslandStateStore(objects, controller={"host": "t"}, clock=lambda: wall[0])
    recorder = lifecycle.FleetRecorder(store, {"isl": lifecycle.IslandRef(island=1, source="ssh")})
    ctl = _controller(ops, recorder=recorder)
    rec = ctl.learners["isl"]
    ctl._record(rec, reason="launched")
    for _ in range(2):  # two ok checks
        wall[0] += 61
        assert _tick(ops, clock) == "RUNNING"
        ctl._health_events(rec)
    # rank 2's trainer exits 1 (not a health loss): the island fails, its checks stop
    cluster.exit("b300-2", 0, 1)
    assert _wait(lambda: ops.islands["isl"].job.procs[2].poll() is not None)
    wall[0] += 61
    assert _tick(ops, clock) == "FAILED"
    ctl._record(rec, lifecycle.RECOVERING, reason="rank 2 exited 1")
    wall[0] += 240  # backoff, preflight and the container start on the same nodes
    for w in ops.islands["isl"].health.values():
        w.clock = lambda: wall[0]
    assert ops.relaunch(None, "isl") == 1
    ctl._record(rec, lifecycle.RUNNING, reason="relaunched")  # the record keeps the old job's checks (reset)
    kept = json.loads(objects.read(lifecycle.state_key(1)))["health"]["nodes"]
    assert all(n.get("reset") for n in kept.values())
    for _ in range(60):  # it runs: ok checks every minute, an hour long
        wall[0] += 61
        clock.t += 61
        assert str(ops.job_status("isl", 1)) == "RUNNING"
        ctl._health_events(rec)
    doc = json.loads(objects.read(lifecycle.state_key(1)))
    assert doc["state"] == "running"
    assert not any(n.get("reset") for n in doc["health"]["nodes"].values())  # the new job's checks
    assert all(n["verdict"] == "ok" for n in doc["health"]["nodes"].values())
    r = reducer()
    assert r.load_fleet_state(fleet_islands.read_fleet_state(objects))
    c = cards(r)["1"]
    assert c["island_now"]["state"] == "running" and "node_health" not in c
    out = smoke(tmp_path / "page", r)
    assert not any("stale" in t for t in out["texts"]) and "stale" not in out["nodes"]


def test_the_health_key_counts_a_kept_check():
    ok = {"verdict": "ok", "reasons": [], "at": 1.0}
    key = FleetController._health_key
    assert key({"nodes": {"a": ok}}) != key({"nodes": {"a": {**ok, "reset": True}}})
    assert key({"nodes": {"a": ok}}) == key({"nodes": {"a": {**ok, "at": 99.0}}})  # times alone: no rewrite
