"""rl-multinode-island tasks 2.2-2.5: local multi-process Ray rehearsal (CPU only).

Two Ray nodes on one machine (``run_sim.sh``), each with 4 fake GPU slots and a
``yeto_node:<k>`` resource label standing in for the node identity (same IP, so
Miles' (ip, gpu) sort is emulated by sorting on the label). Cases:

  pg_blocks   PACK placement group -> node-blocked logical bundles asserted
  cells_bind  cells cut per node; bind_members refuses a cross-node target
  node_loss   worker node killed -> RECOVERY_REQUIRED; restart refuses recovery
              while the node is missing, recovers once it is back
  teardown    per-instance teardown confirmation against the live Ray nodes
  gpu_pool    Q6: a fake nvidia-smi per node (Ray task pinned to the node) -> baseline
              bound, same pool accepted, replaced worker refused, accepted with
              --rl-elastic-accept-rebind (old->new mapping journaled)
  mixed_pp2   C7 (Q2/Q3): the 2x2 cfg tests/multinode_gpu/resources-2x2.json T2R1S1 ->
              trainer_layout (PP2 trainer n0:0 + n1:0, rollout cell n0:1 on the
              trainer's node, standby n1:1) -> PlacementRequest(bundle_map) ->
              StartupBundles(placement_map) over a real 4-bundle PG pinned 2 per
              node; TP2 across nodes and a non-rectangular trainer are refused
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401 - before ray: avoids numpy's double CPU-dispatcher init in this venv
import ray

ADDRESS = os.environ.get("YETO_SIM_ADDRESS", "127.0.0.1:6379")
HEAD_DIR = os.environ.get("YETO_SIM_HEAD_DIR", "/tmp/yeto-s1-ray-h")
WORKER_DIR = os.environ.get("YETO_SIM_WORKER_DIR", "/tmp/yeto-s1-ray-w")
G = 4
RAY_BIN = os.environ.get("YETO_SIM_RAY", "ray")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def labelled_nodes() -> dict[str, dict]:
    """node id -> {label, gpus} for alive nodes carrying a yeto_node label."""
    out = {}
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        res = n.get("Resources") or {}
        labels = [k for k in res if k.startswith("yeto_node:")]
        if labels:
            out[n["NodeID"]] = {"label": labels[0], "gpus": int(res.get("GPU", 0))}
    return out


def probe() -> dict[str, int]:
    """Same shape as entry._ray_alive_nodes, restricted to the simulated nodes."""
    return {nid: info["gpus"] for nid, info in labelled_nodes().items()}


def wait_nodes(n: int, timeout: float = 90) -> dict:
    t = time.time()
    while time.time() - t < timeout:
        nodes = labelled_nodes()
        if len(nodes) == n:
            return nodes
        time.sleep(1)
    raise TimeoutError(f"expected {n} labelled nodes, have {labelled_nodes()}")


def kill_worker():
    subprocess.run(["pkill", "-f", WORKER_DIR + "/"], check=False)
    for _ in range(10):
        if subprocess.run(["pgrep", "-f", WORKER_DIR + "/"], capture_output=True).returncode:
            return
        time.sleep(1)
    subprocess.run(["pkill", "-KILL", "-f", WORKER_DIR + "/"], check=False)


def start_worker():
    subprocess.run([RAY_BIN, "start", f"--address={ADDRESS}", f"--num-gpus={G}", "--num-cpus=4",
                    '--resources={"yeto_node:1": 1}', f"--temp-dir={WORKER_DIR}"],
                   check=True, capture_output=True, timeout=120)


@dataclass
class _Info:
    """Stand-in for Miles' PlacementGroupInfo (positional (pg, bundles, gpus))."""
    pg: object
    pg_reordered_bundle_indices: list
    pg_reordered_gpu_ids: list


# ------------------------------------------------------------------ 2.2
def startup_bundles(shuffle=False):
    from yeto.rl.adapters.miles.bundles import StartupBundles

    pg = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * (2 * G), strategy="PACK")
    ray.get(pg.ready(), timeout=120)
    table = ray.util.placement_group_table(pg)
    b2n = table["bundles_to_node_id"]
    nodes = labelled_nodes()
    # Miles sorts bundles by (node ip, gpu id); same ip here -> sort by label, bundle
    order = sorted(range(2 * G), key=lambda b: (nodes[b2n[b]]["label"], b))
    if shuffle:
        order = order[::2] + order[1::2]
    pool_gpus = tuple(f"p{i}" for i in range(2 * G))
    view = _Info(pg, order, [b % G for b in order])
    # D3 head pin (2026-10-03 ruling): a multi-node StartupBundles needs the Ray head's node id; block 0 must be it
    head = next(nid for nid, v in nodes.items() if v["label"] == "yeto_node:0")
    sb = StartupBundles(pool_gpus=pool_gpus, views={"actor": view}, placement_map=None,
                        gpus_per_node=G, head_node=head,
                        node_resolver=lambda pg_, b: ray.util.placement_group_table(pg_)["bundles_to_node_id"][b])
    return pg, sb, table, nodes, order


def case_pg_blocks():
    from yeto.rl.adapters.miles.bundles import BundleMapError
    from yeto.rl.engine.multinode import Topology

    pg, sb, table, nodes, order = startup_bundles()
    topo = Topology(2, G)
    log("placement group bundles_to_node_id:", {b: nodes[n]["label"] for b, n in table["bundles_to_node_id"].items()})
    rows = []
    for p, gpu in enumerate(sb.pool_gpus):
        node, local = topo.slot_of(p)
        rows.append((p, gpu, node, local, nodes[sb.node_of(gpu)]["label"], order[p]))
        assert nodes[sb.node_of(gpu)]["label"] == f"yeto_node:{node}", rows[-1]
    log("logical bundle p -> (node, local) [label, physical bundle]:")
    for r in rows:
        log(f"  p={r[0]} {r[1]} -> (n{r[2]}, gpu{r[3]})  {r[4]}  bundle#{r[5]}")
    log("node_blocks:", [nodes[n]["label"] for n in sb.node_blocks])
    ray.util.remove_placement_group(pg)
    try:
        startup_bundles(shuffle=True)
    except BundleMapError as exc:
        log("shuffled order rejected (fail closed):", exc)
    else:
        raise AssertionError("shuffled order was accepted")
    log("PASS pg_blocks")


# ------------------------------------------------------------------ 2.3
def case_cells_bind():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import test_rl_e3_wiring_e1 as e3

    from yeto.rl.adapters.miles.placement import PlacementRequest
    from yeto.rl.adapters.miles.rollout import MembershipPlanError, MilesRolloutPool

    req = PlacementRequest("fixed-partition", trainer_gpus=4, rollout_gpus=2, gpus_per_engine=2,
                           standby_gpus=2, gpus_per_node=G, model_parallel=2,
                           rollout_cell_names=("c0", "c1", "c2"))
    pm = req.placement_map_arg
    log("placement map:", json.dumps(pm))
    for c in pm["rollout_cells"]:
        assert len({b // G for b in c["bundles"]}) <= 1, c
    assert pm["rollout_cells"][0]["bundles"] == [4, 5] and pm["rollout_cells"][1]["bundles"] == [6, 7]
    pg, sb, table, nodes, order = startup_bundles()
    manager = e3.FakeManager()
    pool = MilesRolloutPool(
        inference_controller=e3.FakeController(running=("c0",)), rollout_executor=None, metadata=None,
        expected_policy=lambda: (0, "h"), runner=SimpleNamespace(run=asyncio.run),
        declared_cells=("c0", "c1", "c2"), worker_manager=manager, bundles=sb, gpus_per_engine=2)
    pool.members = lambda: frozenset({"engine:c0"})
    try:
        pool.bind_members(frozenset({"engine:c1"}), ("p3", "p4"))
    except MembershipPlanError as exc:
        log("cross-node bind refused:", exc)
    else:
        raise AssertionError("cross-node bind accepted")
    assert not manager.calls
    view = pool.bind_members(frozenset({"engine:c1"}), ("p6", "p7"))
    log("same-node bind ok:", view, [c[0] for c in manager.calls])
    assert [c[0] for c in manager.calls] == ["view", "rebind"]
    ray.util.remove_placement_group(pg)
    log("PASS cells_bind")


# ------------------------------------------------------------------ 2.4
def case_node_loss():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_rl_reconfig_recovery import _island

    from yeto.rl.engine.journal import read_journal

    tmp = Path(tempfile.mkdtemp(prefix="yeto-s1-sim-"))
    kw = {"topology": (2, G), "node_probe": probe}
    driver, ctl, fork, *_ = _island(tmp, controller_kw=kw)
    assert ctl.check_nodes() is None, ctl.recovery_required
    log("2 nodes alive, check_nodes -> None; topology record:",
        [r for r in read_journal(tmp / "state/reconfig") if r["kind"] == "topology"][0]["alive"])
    t0 = time.time()
    kill_worker()
    log("worker killed (pkill by temp dir)")
    err = None
    while time.time() - t0 < 120:
        err = ctl.check_nodes()
        if err:
            break
        time.sleep(1)
    assert err and "node_lost" in err, err
    log(f"node_lost detected after {time.time() - t0:.1f}s:", err)
    assert ctl.inspect().health == "RECOVERY_REQUIRED" and not ctl.admission_open
    lost = [r for r in read_journal(tmp / "state/reconfig") if r["kind"] == "node_lost"]
    assert lost and len(lost[0]["alive"]) == 1, lost
    ctl.close()
    # restart while the node is still missing: no differential recovery
    _d2, ctl2, fork2, *_ = _island(tmp, controller_kw=kw)
    assert ctl2.inspect().health == "RECOVERY_REQUIRED", ctl2.inspect()
    assert not [c for c in fork2.calls if c[0] in ("restore", "start", "stop")], fork2.calls
    log("restart with 1/2 nodes: RECOVERY_REQUIRED without fork membership calls:", ctl2.recovery_required)
    ctl2.close()
    start_worker()
    wait_nodes(2)
    # Q4 a): RECOVERY_REQUIRED is a journal terminal state; the node coming back does
    # not revive the island (manual rebuild), while a fresh island on the same two
    # nodes (new node id for the worker) opens RUNNING.
    _d3, ctl3, *_ = _island(tmp, controller_kw=kw)
    assert ctl3.inspect().health == "RECOVERY_REQUIRED", ctl3.inspect()
    log("worker back (new node id): the journal keeps RECOVERY_REQUIRED (manual rebuild, Q4 a)")
    ctl3.close()
    fresh = Path(tempfile.mkdtemp(prefix="yeto-s1-sim-fresh-"))
    _d4, ctl4, *_ = _island(fresh, controller_kw=kw)
    assert ctl4.inspect().health == "RUNNING" and ctl4.check_nodes() is None, ctl4.inspect()
    log("fresh island on the rebuilt 2 nodes: RUNNING; alive =",
        [v["label"] for v in labelled_nodes().values()])
    ctl4.close()
    log("PASS node_loss")


# ------------------------------------------------------------------ 2.5
def case_teardown():
    from yeto.launcher import terminate_and_verify

    worker_ids = lambda: [nid for nid, v in labelled_nodes().items() if v["label"] == "yeto_node:1"]  # noqa: E731
    assert worker_ids()

    def down_kills_worker():
        kill_worker()
        time.sleep(3)

    ok = terminate_and_verify(None, "sim-island", probe=worker_ids, down=down_kills_worker,
                              sleep_fn=lambda s: time.sleep(min(s, 5)), attempts=6, num_nodes=2)
    log("down kills the node instance -> confirmed:", ok)
    assert ok is True
    start_worker()
    wait_nodes(2)
    ok = terminate_and_verify(None, "sim-island", probe=worker_ids, down=lambda: None,
                              sleep_fn=lambda s: None, attempts=1, num_nodes=2)
    log("down leaves the instance alive -> unconfirmed:", ok)
    assert ok is False
    log("PASS teardown")


# ------------------------------------------------------------------ Q6 gpu_pool
FAKE_SMI = """#!/usr/bin/env bash
# fake nvidia-smi --query-gpu=index,uuid --format=csv,noheader: G rows, uuids from $YETO_SIM_GPU_TAG
for i in $(seq 0 $((${YETO_SIM_G:-4} - 1))); do echo "$i, GPU-${YETO_SIM_GPU_TAG}-$i"; done
"""


def gpu_probe_with(tags: dict[str, str]):
    """entry.reconcile_gpu_pool_preflight ``gpu_probe``: one Ray task per labelled node
    (hard node affinity, like entry._ray_gpu_uuids), each running the fake nvidia-smi with
    the node's tag (``{label: tag}``), rows in logical order (yeto_node:0 first)."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    smi = Path(tempfile.mkdtemp(prefix="yeto-s1-smi-")) / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)

    @ray.remote(num_cpus=0)
    def _smi(tag: str) -> list[tuple[int, str]]:
        env = dict(os.environ, YETO_SIM_GPU_TAG=tag, YETO_SIM_G=str(G))
        out = subprocess.run([str(smi), "--query-gpu=index,uuid", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=60, env=env).stdout
        return [(int(a.strip()), b.strip()) for a, b in
                (line.split(",", 1) for line in out.splitlines() if line.strip())]

    def probe(topology):
        nodes = sorted(labelled_nodes().items(), key=lambda kv: kv[1]["label"])
        assert len(nodes) == topology.nodes, nodes
        refs = [_smi.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False))
                .remote(tags[info["label"]]) for nid, info in nodes]
        return [list(r) for r in ray.get(refs, timeout=120)]

    return probe


def case_gpu_pool():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_rl_reconfig_recovery import _ctl

    from yeto.rl.engine.journal import read_journal
    from yeto.rl.adapters.miles import entry

    tmp = Path(tempfile.mkdtemp(prefix="yeto-s1-sim-gpu-"))
    topology = SimpleNamespace(nodes=2, gpus_per_node=G)
    miles_args = SimpleNamespace(yeto_rl_event_tape=str(tmp / "events.jsonl"), yeto_rl_learner_id=0,
                                 yeto_rl_elastic={"resources": {"nodes": 2, "gpus_per_node": G}})
    pools = lambda: [r for r in read_journal(tmp / "state/reconfig") if r["kind"] == "gpu_pool"]  # noqa: E731
    probe = gpu_probe_with({"yeto_node:0": "h", "yeto_node:1": "w"})
    # v2 ruling (c) stale re-entry: incarnation markers go to a private dir (never the shared default);
    # the marker pid is this process, so markers stay "live" until removed (= `yeto down`)
    mdir = str(tmp / "markers")
    hooks = dict(incarnation_probe=lambda topo, flat: entry._marker_rows(flat, mdir),
                 marker_writer=lambda topo, observed, inc: entry._write_markers(
                     [u for node in observed for u in node], inc, os.getpid(), mdir))
    _orig_preflight = entry.reconcile_gpu_pool_preflight

    def preflight(*a, **kw):
        return _orig_preflight(*a, **{**hooks, **kw})
    ctl = _ctl(tmp / "state", {"t": 1000.0})
    res = preflight(SimpleNamespace(controller=ctl), topology, miles_args, gpu_probe=probe)
    assert res.ok and not res.rebind and pools()[-1]["source"] == "none", res
    log("incarnation 1 (cfg without uuids): baseline bound =", pools()[-1]["uuids"])
    ctl.close()
    ctl = _ctl(tmp / "state", {"t": 1000.5})
    try:
        preflight(SimpleNamespace(controller=ctl), topology, miles_args, gpu_probe=probe)
    except RuntimeError as exc:
        assert "yeto down" in str(exc), exc
        log("incarnation 1b (old incarnation still live): refused ->", str(exc)[:100])
    else:
        raise AssertionError("a new incarnation re-entered GPUs still held by a live old incarnation")
    ctl.close()
    shutil.rmtree(mdir)  # `yeto down`: the old incarnation's processes and markers are gone
    ctl = _ctl(tmp / "state", {"t": 1001.0})
    res = preflight(SimpleNamespace(controller=ctl), topology, miles_args, gpu_probe=probe)
    assert res.ok and not res.rebind and pools()[-1]["source"] == "journal", res
    log("incarnation 2 (same GPUs): accepted against the journal baseline")
    ctl.close()
    shutil.rmtree(mdir, ignore_errors=True)  # clean `yeto down` between incarnations
    shutil.rmtree(mdir, ignore_errors=True)  # each incarnation is brought down before the next
    replaced = gpu_probe_with({"yeto_node:0": "h", "yeto_node:1": "w2"})  # the worker machine changed
    ctl = _ctl(tmp / "state", {"t": 1002.0})
    try:
        preflight(SimpleNamespace(controller=ctl), topology, miles_args, gpu_probe=replaced)
    except RuntimeError as exc:
        assert "gpu_pool" in str(exc) and "--rl-elastic-accept-rebind" in str(exc), exc
        log("incarnation 3 (worker replaced): refused ->", str(exc)[:120])
    else:
        raise AssertionError("replaced worker GPUs were accepted without --rl-elastic-accept-rebind")
    assert not pools()[-1]["accepted"] and len(pools()[-1]["diffs"]) == G
    ctl.close()
    shutil.rmtree(mdir, ignore_errors=True)  # clean `yeto down` between incarnations
    shutil.rmtree(mdir, ignore_errors=True)  # each incarnation is brought down before the next
    ctl = _ctl(tmp / "state", {"t": 1003.0})
    assert ctl.recovery_required is None, ctl.recovery_required  # restartable: not a journal terminal
    miles_args.yeto_rl_elastic["accept_rebind"] = True
    res = preflight(SimpleNamespace(controller=ctl), topology, miles_args, gpu_probe=replaced)
    assert res.ok and res.rebind and len(res.mapping) == G, res
    assert pools()[-1]["rebind"] and pools()[-1]["uuids"][1][0] == "GPU-w2-0", pools()[-1]
    log("incarnation 4 (--rl-elastic-accept-rebind): rebind journaled, mapping =", res.mapping)
    ctl.close()
    shutil.rmtree(mdir, ignore_errors=True)  # clean `yeto down` between incarnations
    shutil.rmtree(mdir, ignore_errors=True)  # each incarnation is brought down before the next
    log("PASS gpu_pool")


# ------------------------------------------------------------------ C7 mixed_pp2 (Q2 + Q3)
def case_mixed_pp2():
    from yeto.rl.elastic_benchmark.capabilities import parse_configs
    from yeto.rl.adapters.miles.bundles import ROLE_VIEWS, BundleMapError, StartupBundles
    from yeto.rl.adapters.miles.placement import PlacementRequest
    from yeto.rl.engine.multinode import Topology, TopologyError, node_placement_rejection, trainer_layout

    cfg_path = Path(__file__).resolve().parents[1] / "multinode_gpu" / "resources-2x2.json"
    configs = parse_configs(json.loads(cfg_path.read_text()))
    slots = configs["T2R1S1"].placement_slots
    topo = Topology(2, 2)
    log("cfg T2R1S1 placement slots:", slots)
    # launcher side (rl_island_layout): shape + bundle map from the cfg placement
    nodes_, per_node, bundle_map = trainer_layout(slots, topo)
    assert (nodes_, per_node) == (2, 1), (nodes_, per_node)
    assert bundle_map == {"trainer": (0, 2), "rollout": (1,), "standby": (3,)}, bundle_map
    log("trainer_layout -> --actor-num-nodes 2 --actor-num-gpus-per-node 1, bundle map", bundle_map)
    # learner side rule set: tp1 pp2 (node_parallel 1, dense group 2) accepts the cross-node PP trainer
    req = PlacementRequest("fixed-partition", trainer_gpus=2, rollout_gpus=1, gpus_per_engine=1, standby_gpus=1,
                           gpus_per_node=2, model_parallel=2, node_parallel=1, bundle_map=bundle_map,
                           rollout_cell_names=("c0", "c1"))
    assert req.trainer_shape() == (2, 1), req.trainer_shape()
    pm = req.placement_map_arg
    assert pm["rollout_cells"] == [{"name": "c0", "bundles": [1], "start": True},
                                   {"name": "c1", "bundles": [3], "start": False}], pm
    log("placement map (cells cut per node; c1 = standby n1:1 for the m3 up edge):", json.dumps(pm))
    # EP2 with tp1 pp1 (M2) is accepted on the same slots; TP2 across n0:0/n1:0 is refused
    assert node_placement_rejection(slots, node_parallel=1, expert_parallel=2, gpus_per_engine=1) is None
    why = node_placement_rejection(slots, node_parallel=2, gpus_per_engine=1)
    assert why and "spans nodes" in why, why
    log("tp*cp = 2 on the same slots refused:", why)
    try:
        trainer_layout({"trainer": [(0, 0), (0, 1), (1, 0)], "rollout": [[(1, 1)]], "standby": []}, topo)
    except TopologyError as exc:
        log("non-rectangular T3 refused:", exc)
    else:
        raise AssertionError("non-rectangular trainer accepted")
    # a real placement group: 4 bundles, two pinned to each simulated node by its label
    pg = ray.util.placement_group([{"GPU": 1, "CPU": 1, "yeto_node:0": 0.01}] * 2
                                  + [{"GPU": 1, "CPU": 1, "yeto_node:1": 0.01}] * 2, strategy="PACK")
    ray.get(pg.ready(), timeout=120)
    b2n = ray.util.placement_group_table(pg)["bundles_to_node_id"]
    nodes = labelled_nodes()
    order = sorted(range(4), key=lambda b: (nodes[b2n[b]]["label"], b))  # Miles' (ip, gpu) sort
    head = next(nid for nid, v in nodes.items() if v["label"] == "yeto_node:0")
    worker = next(nid for nid, v in nodes.items() if v["label"] == "yeto_node:1")
    roles = {k: list(pm[k]) for k in ("trainer", "rollout", "standby")}
    views = {ROLE_VIEWS[r]: _Info(pg, [order[p] for p in ps], [order[p] % 2 for p in ps]) for r, ps in roles.items()}
    resolver = lambda pg_, b: ray.util.placement_group_table(pg_)["bundles_to_node_id"][b]  # noqa: E731
    sb = StartupBundles(pool_gpus=tuple(f"p{i}" for i in range(4)), views=views, placement_map=roles,
                        gpus_per_node=2, node_resolver=resolver, head_node=head)
    table = {p: nodes[sb.node_of(f"p{p}")]["label"] for p in range(4)}
    log("logical bundle -> node:", table)
    assert [nodes[n]["label"] for n in sb.node_blocks] == ["yeto_node:0", "yeto_node:1"]
    assert sb.node_of("p0") != sb.node_of("p2"), "PP stages must sit on two nodes"
    assert sb.node_of("p0") == sb.node_of("p1"), "rollout cell c0 shares n0 with trainer rank 0 (Q2 mixed)"
    assert not sb.same_node(["p1", "p3"]) and sb.same_node(["p2", "p3"])
    try:
        StartupBundles(pool_gpus=tuple(f"p{i}" for i in range(4)), views=views, placement_map=roles,
                       gpus_per_node=2, node_resolver=resolver, head_node=worker)
    except BundleMapError as exc:
        log("block 0 != head refused (D3 head pin):", exc)
    else:
        raise AssertionError("head pin accepted the worker as node 0")
    ray.util.remove_placement_group(pg)
    log("PASS mixed_pp2")


CASES = {"pg_blocks": case_pg_blocks, "cells_bind": case_cells_bind, "node_loss": case_node_loss,
         "teardown": case_teardown, "gpu_pool": case_gpu_pool, "mixed_pp2": case_mixed_pp2}

if __name__ == "__main__":
    case = sys.argv[1]
    ray.init(address=ADDRESS, include_dashboard=False, log_to_driver=False,
             _temp_dir=HEAD_DIR, namespace=f"sim-{case}")
    try:
        nodes = wait_nodes(2)
        log("nodes:", {v["label"]: v["gpus"] for v in nodes.values()})
        CASES[case]()
    finally:
        ray.shutdown()
