"""Node health where it runs (``yeto.hw.node_health``, tasks 1.2 / 1.3 of own-cluster-fleet-stability).

* the node runtime checks the node while the sampler runs and the sampler's heartbeat
  carries the check (``health``); the pool reads it from the run store, or from the node
  in the ssh pool's own probe; two bad checks in a row are a lost node (fence, then a
  replacement on a spare; the slot is held), one is not, a read failure is one event;
* a resident ssh: trainer island checks every node in the SSH call that asks for its
  state; two bad checks in a row of one node fail the island as a lost node
  (``_node_health_lost``), each check is in the island's record and its events reach
  the fleet's log;
* a device family without a checker (Ascend here) runs nothing more and writes the same
  heartbeat, probe script, status script and island record as before.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from test_cluster_preflight import ibstat_text  # tests/ is on sys.path
from test_nc_elastic import Clock as PoolClock
from test_nc_elastic import Cloud, _names, _pool, _ready_member, _run
from test_node_health import HCA, node_text, xid_line, xid_text
from test_rl_ssh_resident import NAMES, FakeCluster, _controller, b300, make_ops
from test_rl_ssh_resident import _cards  # noqa: F401 - autouse: no card tool, a resource ledger in tmp
from test_rl_ssh_resident import Clock as IslandClock
from test_ssh_sampler_pool import FakeRun, pool_cfg

from yeto.hw import node_health
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch import ssh_trainer_island
from yeto.launch.node_inventory import NodePaths, NodeProfile, parse_node
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY
from yeto.launch.ssh_trainer_island import (
    NODE_STATE_MARK,
    SshTrainerSpec,
    container_name,
    remote_paths,
    status_script,
)
from yeto.node import records
from yeto.node.launch import HEALTH_ENV
from yeto.node.report import SamplerNodeReport
from yeto.node.runtime import NodeRuntime
from yeto.rl.engine.object_store import DirectoryObjectStore

XID_79 = node_text(xid=(0, xid_text(xid_line(4000.0, 5, 79))))
XID_79_AGAIN = node_text(xid=(0, xid_text(xid_line(4000.0, 5, 79), xid_line(4100.0, 5, 79))))
TIMED_OUT = node_text(gpus=(124, ""))
PORT_DOWN = node_text(ibstat=(0, ibstat_text(down=(3,))))
ACTIVE = el.LeaseStatus("active", lease_until=None)
CFG = el.ElasticConfig(unreachable_grace_s=300)


# ------------------------------------------------------------------ the node runtime
def _profile(tmp_path, family="cuda", **over):
    card = {"cuda": "B300", "npu": "910B4"}[family]
    values = dict(name="box0", host="", device_family=family, cpu_arch="x86_64", card=card, cards_per_node=8,
                  hbm_gib=288.0 if family == "cuda" else 64.0, host_memory_gib=0.0, container="process",
                  paths=NodePaths(work=str(tmp_path / "work"), models=str(tmp_path / "models")),
                  env={"NCCL_IB_HCA": HCA})
    values.update(over)
    return NodeProfile(**values)


def _runtime(tmp_path, family="cuda", texts=(), default=None, **kw):
    objects = DirectoryObjectStore(tmp_path / "store")
    code = records.CodeRef(sha="abc", key="node-code/abc.tar", sha256="")
    assignment, _ = records.assign(objects, ticket="t-h", shard="s0", spec={"sampler_id": "s0"}, code=code)
    feed = list(texts)
    runs = []

    def run(script):
        runs.append(script)
        return feed.pop(0) if feed else (default or node_text())

    rt = NodeRuntime(objects, assignment, _profile(tmp_path, family), store_uri=None, environ={"PATH": "/usr/bin"},
                     skip_preflight=True, health_run=run, log=lambda line: None, **kw)
    return rt, objects, runs


class _Sampler:
    """The sampler as the node runtime's child: its heartbeats (yeto.node.report) from the
    environment the runtime gave it, one per health check the test runs."""

    def __init__(self, rt, objects, steps):
        self.rt, self.objects, self.steps = rt, objects, steps
        self.env: dict = {}
        self.verdicts: list = []

    def __call__(self, argv, env, **kw):
        self.env = dict(env)
        return self

    def wait(self):
        report = SamplerNodeReport(records.NodeReporter(self.objects, self.env["YETO_NODE_TICKET"], 1,
                                                        base={"sampler": "s0"}),
                                   health_file=self.env.get(HEALTH_ENV), metrics=lambda port: {})
        path = self.env.get(HEALTH_ENV)
        if path is not None:  # the runtime's first check runs as the sampler starts
            assert _wait(lambda: Path(path).exists())
            self.rt._stop_health()  # the test runs the next checks itself, one per step
        for step in range(self.steps + 1):
            if step and path is not None:
                self.rt.check_health(path)
            report._beat()
            status = records.node_status(self.objects, "t-h")
            probe = records.probe_of(status)
            self.verdicts.append((None if probe.health is None else probe.health["verdict"],
                                  el.classify(ACTIVE, probe, now=0.0, unreachable_since=None, cfg=CFG,
                                              joined=True).kind))
        return 0


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_one_bad_check_is_not_lost_two_in_a_row_are_through_the_store(tmp_path):
    rt, objects, runs = _runtime(tmp_path, texts=[node_text(), XID_79, TIMED_OUT, XID_79_AGAIN])
    plan = rt.plan(str(tmp_path / "work" / "yeto"))
    assert plan.env[HEALTH_ENV] == f"{tmp_path}/work/sampler/s0.health.json" == plan.health_host_path
    sampler = _Sampler(rt, objects, steps=3)
    rt.popen = sampler
    assert rt.start(plan) == 0
    # ok; one bad check (XID 79): not lost; a read failure counts neither way; the second bad check: lost
    assert sampler.verdicts == [("ok", el.HEALTHY), ("bad_once", el.HEALTHY), ("read_failed", el.HEALTHY),
                                ("lost", el.LOST)]
    beat = records.node_status(objects, "t-h").heartbeat
    assert beat["health"]["verdict"] == "lost" and beat["health"]["reasons"] == ["GPU 5 XID 79"]
    assert beat["health"]["ticket"] == "t-h" and beat["health"]["generation"] == 1
    verdict = el.classify(ACTIVE, records.probe_of(records.node_status(objects, "t-h")), now=0.0,
                          unreachable_since=None, cfg=CFG, joined=True)
    assert verdict.detail == "node health: GPU 5 XID 79"
    assert all("nvidia-smi --query-gpu" in script for script in runs) and len(runs) == 4


def test_a_sampler_ignores_its_ports_and_the_runtime_stops_checking_with_the_sampler(tmp_path):
    rt, objects, runs = _runtime(tmp_path, default=PORT_DOWN, health_every_s=0.01)
    plan = rt.plan(str(tmp_path / "work" / "yeto"))
    done = threading.Event()

    class Child:
        def __init__(self, argv, env, **kw):
            pass

        def wait(self):
            assert _wait(lambda: len(runs) >= 3)
            done.set()
            return 0

    rt.popen = Child
    assert rt.start(plan) == 0 and done.is_set()
    doc = json.loads(Path(plan.health_host_path).read_text())
    assert doc["verdict"] == "ok" and doc["reasons"] == [] and doc["notes"] == ["mlx5_3 port DOWN"]
    assert doc["facts"]["ports"]["mlx5_3"] == "Down 400" and doc["multi"] is False
    count = len(runs)
    time.sleep(0.1)
    assert len(runs) == count and rt._health_thread is None  # no check after the sampler ended


def test_a_stale_check_of_an_earlier_start_is_removed_before_the_sampler_starts(tmp_path):
    rt, objects, runs = _runtime(tmp_path, texts=[node_text()])
    plan = rt.plan(str(tmp_path / "work" / "yeto"))
    Path(plan.health_host_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plan.health_host_path).write_text(json.dumps({"verdict": "lost", "reasons": ["old"]}))
    seen = {}

    class Child:
        def __init__(self, argv, env, **kw):
            path = Path(env[HEALTH_ENV])
            seen["at_start"] = path.read_text() if path.exists() and "old" in path.read_text() else None

        def wait(self):
            return 0

    rt.popen = Child
    rt.start(plan)
    assert seen["at_start"] is None


def test_without_a_checker_the_runtime_runs_nothing_and_the_heartbeat_is_the_same(tmp_path):
    rt, objects, runs = _runtime(tmp_path, family="npu")
    plan = rt.plan(str(tmp_path / "work" / "yeto"))
    assert not rt.health_on and HEALTH_ENV not in plan.env and plan.health_host_path is None
    sampler = _Sampler(rt, objects, steps=0)
    rt.popen = sampler
    assert rt.start(plan) == 0 and runs == [] and not (tmp_path / "work" / "sampler").exists()
    beat = records.node_status(objects, "t-h").heartbeat
    assert "health" not in beat and records.probe_of(records.node_status(objects, "t-h")).health is None
    # the same heartbeat as a report that was never given a health file (the code of before)
    plain = SamplerNodeReport(records.NodeReporter(objects, "t-plain", 1, base={"sampler": "s0"},
                                                   clock=lambda: 5.0), metrics=lambda port: {}, clock=lambda: 5.0)
    off = SamplerNodeReport(records.NodeReporter(objects, "t-off", 1, base={"sampler": "s0"}, clock=lambda: 5.0),
                            metrics=lambda port: {}, clock=lambda: 5.0, health_file=None)
    plain._beat()
    off._beat()
    a = json.loads(objects.read(records.node_key("t-plain", "heartbeat", 1)))
    b = json.loads(objects.read(records.node_key("t-off", "heartbeat", 1)))
    assert {k: v for k, v in a.items() if k != "ticket"} == {k: v for k, v in b.items() if k != "ticket"}
    off_cli, _, _ = _runtime(tmp_path / "off", health_every_s=0)
    assert not off_cli.health_on and HEALTH_ENV not in off_cli.plan(str(tmp_path / "y")).env


def test_the_report_reads_the_health_file_from_its_environment(tmp_path):
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"verdict": "bad_once", "reasons": ["GPU 1 XID 48"]}))
    env = {"YETO_NODE_TICKET": "t9", "YETO_NODE_GENERATION": "2", HEALTH_ENV: str(path)}
    report = SamplerNodeReport.from_env(env, SimpleNamespace(root=tmp_path / "store"), {"sampler_id": "s9"})
    assert report.health_file == str(path) and report.health()["verdict"] == "bad_once"
    path.write_text("{torn")
    assert report.health() is None
    assert SamplerNodeReport.from_env({"YETO_NODE_TICKET": "t9"}, SimpleNamespace(root=tmp_path / "store"),
                                      {"sampler_id": "s9"}).health_file is None


# ------------------------------------------------------------------ the pool
def test_the_pool_fences_and_replaces_a_node_whose_health_reads_lost_once_and_logs_a_read_failure():
    clock = PoolClock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    orders = len(cloud.kinds("order"))
    probe = cloud.probe
    health = {"verdict": "ok", "reasons": []}
    cloud.probe = lambda member: replace(probe(member), health=dict(health)) if member.ticket == m.ticket \
        else probe(member)
    health.update(verdict="bad_once", reasons=["GPU 5 XID 79"])
    _run(pool, clock, clock.t + 120)
    assert m.state == READY and "node_lost" not in _names(pool)
    health.update(verdict="read_failed", reasons=[f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"])
    _run(pool, clock, clock.t + 180)
    assert _names(pool).count("node_health_read_failed") == 1 and m.state == READY
    health.update(verdict="ok", reasons=[])
    _run(pool, clock, clock.t + 60)
    health.update(verdict="read_failed", reasons=[f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"])
    _run(pool, clock, clock.t + 60)
    assert _names(pool).count("node_health_read_failed") == 2  # again after a check read in between
    health.update(verdict="lost", reasons=["GPU 5 XID 79"])
    pool.tick()
    names = _names(pool)
    assert m.state == GONE and names.index("node_health") < names.index("node_lost")
    event = next(e for e in pool.events.events if e["event"].endswith("node_health"))
    assert event["reasons"] == ["GPU 5 XID 79"] and event["ticket"] == m.ticket
    lost = next(e for e in pool.events.events if e["event"].endswith("node_lost"))
    assert lost["detail"] == "node health: GPU 5 XID 79"
    assert cloud.kinds("fence") and len(cloud.kinds("order")) > orders  # fenced, then a replacement


def _ssh_ops(tmp_path, monkeypatch, run, *, family="cuda", node_extra=None, **section):
    monkeypatch.setattr(sp.subprocess, "Popen", lambda *a, **k: SimpleNamespace(stdout=None, wait=lambda: 0))
    node = parse_node({"name": "node-a", "host": "ops@node-a", "device_family": family, "cpu_arch": "aarch64",
                       "card": "B300" if family == "cuda" else "910B4", "cards_per_node": 8,
                       "hbm_gib": 288 if family == "cuda" else 64, "host_memory_gib": 1024,
                       "image": "img", **(node_extra or {})}, index=0)
    cfg = pool_cfg(tmp_path, **section)
    ops = sp.SshSamplerPoolOps(cfg, conf=sp.parse_config(cfg, inventory=[node]), run=run, state_dir=tmp_path)
    t = ops.order(0)
    return ops, SimpleNamespace(account=0, ticket=t, shard="f09-0", data_slot=None, replaces_shard=None,
                                predecessor_lost=False, unreachable_since=None)


class HealthRun(FakeRun):
    """FakeRun whose probe also prints the node runtime's health file (or none)."""

    def __init__(self):
        super().__init__()
        self.health: dict | None = None

    def __call__(self, argv, **kw):
        out = super().__call__(argv, **kw)
        cmd = argv[-1] if argv and argv[0] == "ssh" else ""
        if "YETO_NC_PROBE" in cmd and sp.HEALTH_MARK in cmd:
            body = "" if self.health is None else json.dumps(self.health)
            out = SimpleNamespace(returncode=0, stdout=out.stdout + f"{sp.HEALTH_MARK} {body}\n", stderr="")
        return out


def test_the_ssh_pool_probe_reads_the_check_and_holds_the_slot_of_a_lost_node(tmp_path, monkeypatch):
    run = HealthRun()
    ops, m = _ssh_ops(tmp_path, monkeypatch, run, node_runtime=True)
    ops._health_generation[m.ticket] = 3
    assert ops.probe(m).health is None  # no check written yet
    probe_cmd = next(c[0][-1] for c in run.calls if c[0][0] == "ssh" and "YETO_NC_PROBE" in c[0][-1])
    assert f"{tmp_path}/" not in probe_cmd and "/srv/yeto/run0/sampler/f09-0.health.json" in probe_cmd
    run.health = {"ticket": m.ticket, "generation": 2, "verdict": "lost", "reasons": ["old"]}
    assert ops.probe(m).health is None  # a check of an earlier assignment of the ticket
    run.health = {"ticket": m.ticket, "generation": 3, "verdict": "bad_once", "reasons": ["GPU 5 XID 79"]}
    assert ops.probe(m).health["verdict"] == "bad_once" and m.ticket not in ops._health_lost
    run.health["verdict"] = "lost"
    probe = ops.probe(m)
    assert el.health_lost(probe) == ["GPU 5 XID 79"] and m.ticket in ops._health_lost
    held = []
    monkeypatch.setattr(ops.provider, "mark_lost", lambda ticket, stopped: held.append((ticket, stopped)))
    monkeypatch.setattr(ops, "logs", lambda *a, **k: "")
    ops.drain(m)
    assert held == [(m.ticket, True)] and m.ticket not in ops._health_lost  # stopped there, its slot held


def test_the_ssh_pool_probe_is_unchanged_without_a_checker_or_the_node_runtime(tmp_path, monkeypatch):
    scripts = {}
    for label, family, section in (("npu", "npu", {"node_runtime": True}),
                                   ("no-runtime", "cuda", {"node_runtime": False}),
                                   ("cuda", "cuda", {"node_runtime": True})):
        run = HealthRun()
        (tmp_path / label).mkdir()
        ops, m = _ssh_ops(tmp_path / label, monkeypatch, run, family=family, **section)
        assert ops.probe(m).health is None
        scripts[label] = next(c[0][-1] for c in run.calls if c[0][0] == "ssh" and "YETO_NC_PROBE" in c[0][-1])
    assert sp.HEALTH_MARK not in scripts["npu"] and sp.HEALTH_MARK not in scripts["no-runtime"]
    assert scripts["npu"] == scripts["no-runtime"]
    assert scripts["cuda"].startswith(scripts["npu"] + "; ") and sp.HEALTH_MARK in scripts["cuda"]


def test_the_health_probe_command_prints_one_line_for_a_file_and_for_none(tmp_path):
    import subprocess  # noqa: PLC0415

    path = tmp_path / "s.health.json"
    cmd = sp.health_probe_command(str(path))
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    assert out == f"{sp.HEALTH_MARK} \n" and sp.parse_health_line(out) is None
    path.write_text(json.dumps({"verdict": "ok", "reasons": []}, indent=1) + "\n")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
    assert sp.parse_health_line(out) == {"verdict": "ok", "reasons": []}


# ------------------------------------------------------------------ the resident ssh: trainer island
class HealthCluster(FakeCluster):
    """FakeCluster whose nodes answer the health check that rides on the status call."""

    def __init__(self, names, **kw):
        super().__init__(names, **kw)
        self.health = {n: [] for n in names}  # per node: the outputs of its next checks
        self.status_scripts: list[str] = []
        self.stuck: dict[str, bool] = {}  # node -> its status call runs past the timeout (True: the state printed)
        self.timeouts: list[float] = []

    def run(self, node, script, *, timeout=300.0):
        if "npu-smi" in script and NODE_STATE_MARK not in script:  # an Ascend node's card check
            return super().run(node, "uname -m", timeout=timeout)
        if NODE_STATE_MARK in script:
            self.timeouts.append(timeout)
        if NODE_STATE_MARK in script and node.name in self.stuck:
            done = super().run(node, script, timeout=timeout)
            self.status_scripts.append(script)
            printed = done.stdout.encode() if self.stuck[node.name] else None
            raise subprocess.TimeoutExpired(["ssh"], timeout, output=printed)
        done = super().run(node, script, timeout=timeout)
        if NODE_STATE_MARK in script:
            self.status_scripts.append(script)
            if "nvidia-smi --query-gpu" in script and node.name not in self.unreachable:
                queue = self.health[node.name]
                done.stdout += queue.pop(0) if queue else node_text()
        return done


def hca_spec() -> SshTrainerSpec:
    """The resident tests' 3-node island, its nodes' NCCL_IB_HCA as in the customer inventory."""
    return SshTrainerSpec(nodes=tuple(b300(i, env={"NCCL_IB_HCA": HCA}) for i in range(3)), run_id="run1",
                          workdir="/opt/yeto", run_script="echo island", env={"YETO_X": "1"})


def _island(tmp_path, cluster, **kw):
    clock = IslandClock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=60.0, **kw)
    ops.add("isl", hca_spec())
    ops.start("isl")
    return ops, clock, []


def _tick(ops, clock):
    clock.t += 61
    return str(ops.job_status("isl", 0))


def test_a_trainer_node_bad_twice_in_a_row_is_a_lost_node_once_is_not(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    cluster.health["b300-1"] = [node_text(), PORT_DOWN, node_text(), PORT_DOWN, PORT_DOWN]
    assert _tick(ops, clock) == "RUNNING"
    assert _tick(ops, clock) == "RUNNING"  # one bad check
    assert ops.island_health("isl")["nodes"]["b300-1"]["verdict"] == "bad_once"
    assert ops.island_health("isl")["verdict"] == "bad_once"
    assert _tick(ops, clock) == "RUNNING" and _tick(ops, clock) == "RUNNING"
    assert _tick(ops, clock) == "FAILED"  # the second bad check in a row
    rec = ops.islands["isl"]
    assert rec.reason == "rank 1 (b300-1) node health bad in 2 checks in a row: mlx5_3 port DOWN"
    assert rec.health_lost == {1: ["mlx5_3 port DOWN"]} and cluster.count("kill", 0) == 3
    decision = ops.island_decision("isl", 0)
    assert decision.cause == "lost" and decision.relaunch
    health = ops.island_health("isl")
    assert health["verdict"] == "lost" and health["nodes"]["b300-1"]["reasons"] == ["mlx5_3 port DOWN"]
    assert health["nodes"]["b300-0"]["verdict"] == "ok" and health["nodes"]["b300-1"]["multi"] is True
    [event] = ops.take_health_events("isl")
    assert event == {"event": "node_health", "node": "b300-1", "rank": 1, "incarnation": 0, "verdict": "lost",
                     "reasons": ["mlx5_3 port DOWN"]}
    assert ops.take_health_events("isl") == []
    # every check rode on the status call: one SSH call per node and check
    assert all("nvidia-smi --query-gpu" in s for s in cluster.status_scripts)
    assert cluster.count("status") == len(cluster.status_scripts)


def test_a_trainer_read_failure_is_one_event_and_no_loss(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    cluster.health["b300-2"] = [TIMED_OUT, TIMED_OUT, node_text(), TIMED_OUT]
    for _ in range(4):
        assert _tick(ops, clock) == "RUNNING"
    events = ops.take_health_events("isl")
    assert [e["event"] for e in events] == ["node_health_read_failed", "node_health_read_failed"]
    assert events[0]["reasons"] == [f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"] and events[0]["node"] == "b300-2"
    # an SSH call that does not answer is the unreachable count's, not a health read failure
    cluster.unreachable.add("b300-0")
    assert _tick(ops, clock) == "RUNNING"
    cluster.unreachable.discard("b300-0")
    assert _tick(ops, clock) == "RUNNING"
    assert ops.take_health_events("isl") == [] and ops.islands["isl"].unreachable[0] == 0


def test_the_node_health_lost_hook_runs_once_outside_the_lock_with_the_reading(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    calls = []

    def hook(rec, rank, reasons, reading):
        free = ops._lock.acquire(blocking=False)  # a plain Lock: held by this thread it would not be free
        if free:
            ops._lock.release()
        calls.append((rec.name, rank, reasons, free, reading.verdict, reading.result.facts["ports"]["mlx5_3"]))

    ops._node_health_lost = hook
    cluster.health["b300-0"] = [node_text(), PORT_DOWN, PORT_DOWN, PORT_DOWN, TIMED_OUT, PORT_DOWN]
    for _ in range(6):
        assert _tick(ops, clock) == "RUNNING"  # the hook decides; the default fails the island
    # once: when the streak reached two (not again while it stays lost, nor after a read failure)
    assert calls == [("isl", 0, ["mlx5_3 port DOWN"], True, "lost", "Down 400")]


def test_a_trainer_island_without_a_checker_asks_the_same_status_and_records_no_health(tmp_path):
    npu = tuple(b300(i, device_family="npu", card="910B4", hbm_gib=64.0, docker_args=("--device=/dev/davinci0",),
                     fabric="tcp") for i in range(3))
    cluster = HealthCluster(NAMES)
    ops = make_ops(tmp_path, cluster, clock=IslandClock(), health_every_s=60.0)
    ops.add("isl", SshTrainerSpec(nodes=npu, run_id="run1", workdir="/opt/yeto", run_script="echo island"))
    ops.start("isl")
    ops.clock.t += 61
    assert str(ops.job_status("isl", 0)) == "RUNNING"
    paths = remote_paths("run1", "isl", 0)
    want = [status_script(paths, container=container_name("run1", "isl", 0, r)) for r in range(3)]
    assert cluster.status_scripts == want  # byte for byte the status call of before
    assert ops.island_health("isl") is None and ops.take_health_events("isl") == []
    off = make_ops(tmp_path / "off", HealthCluster(NAMES), clock=IslandClock(), health_every_s=0.0)
    off.add("isl", hca_spec())
    assert off.islands["isl"].health == {}


def test_the_fleet_writes_the_island_health_into_its_record_and_the_events_into_its_log(tmp_path):
    from yeto.launch import island_lifecycle as lifecycle  # noqa: PLC0415

    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    cluster.health["b300-1"] = [PORT_DOWN] * 50

    class Objects:
        def __init__(self):
            self.docs = {}

        def read(self, key):
            return self.docs.get(key)

        def write(self, key, data):
            self.docs[key] = data

    class Log:
        def __init__(self):
            self.events = []

        def event(self, kind, name, **fields):
            self.events.append((kind, name, fields))

        def maybe_tick(self):
            pass

        def cost_tick(self):
            pass

    objects, log = Objects(), Log()
    store = lifecycle.IslandStateStore(objects, controller={"host": "test"})
    recorder = lifecycle.FleetRecorder(store, {"isl": lifecycle.IslandRef(island=0, source="ssh")})
    ctl = _controller(ops, recorder=recorder, fleet_log=log)
    rec = ctl.learners["isl"]
    ctl._record(rec, reason="launched")
    assert "health" not in json.loads(objects.docs[lifecycle.state_key(0)])  # nothing checked yet
    ops.clock.t += 61
    ctl._poll_recorded(rec)  # one bad check: the record shows it
    doc = json.loads(objects.docs[lifecycle.state_key(0)])
    assert doc["health"]["nodes"]["b300-1"]["verdict"] == "bad_once" and doc["state"] == "running"
    ops.clock.t += 61
    ctl._poll_recorded(rec)  # the second: the node is lost, the island recovers
    doc = json.loads(objects.docs[lifecycle.state_key(0)])
    assert doc["health"]["verdict"] == "lost" and doc["health"]["nodes"]["b300-1"]["reasons"] == ["mlx5_3 port DOWN"]
    kinds = [k for k, _, _ in log.events]
    assert "node_health" in kinds
    _, name, fields = next(e for e in log.events if e[0] == "node_health")
    assert name == "isl" and fields["node"] == "b300-1" and fields["reasons"] == ["mlx5_3 port DOWN"]


# ------------------------------------------------------------------ review: bounds, relaunch, switches
KILLED = f"{node_health.TIMEOUT_MARK} {node_health.HEALTH_LIMIT_S}\n"  # bounded(): the check killed on the node


def test_a_stuck_health_check_keeps_the_island_running_and_is_a_read_failure(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    # killed on the node past its limit: the status line came, the check is a read failure
    cluster.health["b300-1"] = [KILLED]
    assert _tick(ops, clock) == "RUNNING"
    [event] = ops.take_health_events("isl")
    assert event["event"] == "node_health_read_failed" and event["node"] == "b300-1"
    assert event["reasons"] == [f"node health check timed out after {node_health.HEALTH_LIMIT_S}s (killed)"]
    # the SSH call itself ran past its timeout after the status line: a read failure, not an unreachable node
    cluster.stuck["b300-2"] = True
    for _ in range(4):
        assert _tick(ops, clock) == "RUNNING"
    rec = ops.islands["isl"]
    assert rec.unreachable.get(2, 0) == 0 and rec.verdict is None
    events = ops.take_health_events("isl")
    assert [e["event"] for e in events] == ["node_health_read_failed"] and events[0]["node"] == "b300-2"
    assert events[0]["reasons"] == ["node health check cut off: its SSH call ran past its timeout"]
    # nothing printed before the timeout: the node does not answer (the unreachable count's)
    cluster.stuck["b300-2"] = False
    assert _tick(ops, clock) == "RUNNING" and rec.unreachable[2] == 1
    assert set(cluster.timeouts) == {60.0 + ssh_trainer_island.NODE_CHECK_MARGIN_S}
    assert node_health.HEALTH_LIMIT_S < 60.0 + ssh_trainer_island.NODE_CHECK_MARGIN_S


def test_a_relaunch_on_the_same_nodes_starts_a_new_streak_and_keeps_the_record(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    cluster.health["b300-1"] = [node_text(), PORT_DOWN, PORT_DOWN]
    for want in ("RUNNING", "RUNNING", "FAILED"):
        assert _tick(ops, clock) == want
    assert ops.island_decision("isl", 0).cause == "lost"
    assert ops.relaunch(None, "isl") == 1
    rec = ops.islands["isl"]
    kept = ops.island_health("isl")["nodes"]["b300-1"]
    assert kept["verdict"] == "lost" and kept["reset"] is True  # the record stays until the next check
    assert rec.health[1].prev["ports"]  # the baselines stay
    cluster.health["b300-1"] = [PORT_DOWN]
    clock.t += 61
    assert str(ops.job_status("isl", 1)) == "RUNNING"  # a new job: one bad check is not lost
    assert ops.island_health("isl")["nodes"]["b300-1"]["verdict"] == "bad_once"
    # a failure of the new incarnation that is not the node's health: not a lost node
    cluster.exit("b300-2", 1, 1)
    assert _wait(lambda: rec.job.procs[2].poll() is not None)
    assert str(ops.job_status("isl", 1)) == "FAILED"
    assert ops.island_decision("isl", 1).cause != "lost" and rec.health_lost == {}


def test_a_single_node_island_does_not_judge_its_ports(tmp_path):
    cluster = HealthCluster(NAMES)
    clock = IslandClock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=60.0)
    ops.add("one", SshTrainerSpec(nodes=(b300(0, env={"NCCL_IB_HCA": HCA}),), run_id="run1", workdir="/opt/yeto",
                                  run_script="echo island"))
    ops.start("one")
    cluster.health["b300-0"] = [PORT_DOWN] * 4
    for _ in range(4):
        clock.t += 61
        assert str(ops.job_status("one", 0)) == "RUNNING"
    doc = ops.island_health("one")["nodes"]["b300-0"]
    assert doc["verdict"] == "ok" and doc["multi"] is False and doc["notes"] == ["mlx5_3 port DOWN"]
    fast = make_ops(tmp_path / "fast", HealthCluster(NAMES), clock=IslandClock(), health_every_s=20.0)
    fast.add("isl", hca_spec())
    assert {w.every_s for w in fast.islands["isl"].health.values()} == {20.0}  # a reader's staleness period


def test_inventory_node_health_off_turns_the_trainer_check_off_per_node(tmp_path):
    nodes = (b300(0, env={"NCCL_IB_HCA": HCA}), replace(b300(1, env={"NCCL_IB_HCA": HCA}), node_health="off"),
             b300(2, env={"NCCL_IB_HCA": HCA}))
    cluster = HealthCluster(NAMES)
    ops = make_ops(tmp_path, cluster, clock=IslandClock(), health_every_s=60.0)
    ops.add("isl", SshTrainerSpec(nodes=nodes, run_id="run1", workdir="/opt/yeto", run_script="echo island"))
    assert sorted(ops.islands["isl"].health) == [0, 2]
    ops.start("isl")
    ops.clock.t += 61
    assert str(ops.job_status("isl", 0)) == "RUNNING"
    by_node = dict(zip(NAMES, cluster.status_scripts))
    assert "nvidia-smi" not in by_node["b300-1"] and "nvidia-smi" in by_node["b300-0"]
    assert sorted(ops.island_health("isl")["nodes"]) == ["b300-0", "b300-2"]


def test_the_dry_run_says_node_health_only_when_the_inventory_declares_it(caplog):
    import logging  # noqa: PLC0415

    from yeto.launch.ssh_resident import node_health_summary  # noqa: PLC0415

    plain = SshTrainerSpec(nodes=tuple(b300(i) for i in range(2)), run_id="r", workdir="/w", run_script="x")
    with caplog.at_level(logging.DEBUG, logger="yeto.launch.ssh_resident"):
        assert node_health_summary(plain) is None
    assert "node health: on (cuda) (not declared in the inventory)" in caplog.text
    auto = SshTrainerSpec(nodes=tuple(replace(b300(i), node_health="auto") for i in range(2)), run_id="r",
                          workdir="/w", run_script="x")
    assert node_health_summary(auto) == "on (cuda)"
    off = SshTrainerSpec(nodes=tuple(replace(b300(i), node_health="off") for i in range(2)), run_id="r",
                         workdir="/w", run_script="x")
    assert node_health_summary(off) == "off"
    some = SshTrainerSpec(nodes=(b300(0), replace(b300(1), node_health="off")), run_id="r", workdir="/w",
                          run_script="x")
    assert node_health_summary(some) == "on (cuda), off on ['b300-1']"
    npu = SshTrainerSpec(nodes=tuple(replace(b300(i, device_family="npu", card="910B4", hbm_gib=64.0),
                                             node_health="auto") for i in range(2)), run_id="r", workdir="/w",
                         run_script="x")
    assert node_health_summary(npu) == "off"  # no checker for the family


def test_a_multi_node_island_says_once_what_its_check_cannot_judge(tmp_path):
    cluster = HealthCluster(NAMES)
    clock = IslandClock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=60.0)
    ops.add("isl", SshTrainerSpec(nodes=tuple(b300(i, fabric="tcp") for i in range(3)), run_id="run1",
                                  workdir="/opt/yeto", run_script="echo island"))
    ops.start("isl")
    cluster.health["b300-0"] = [node_text(xid=(0, xid_text(source="none")))] * 3
    for _ in range(3):
        clock.t += 61
        assert str(ops.job_status("isl", 0)) == "RUNNING"
    events = [e for e in ops.take_health_events("isl") if e["event"] == node_health.EVENT_DEGRADED]
    assert len(events) == 3  # once per node
    first = next(e for e in events if e["node"] == "b300-0")
    assert first["reasons"] == ["XID unread (no dmesg / journalctl -k access)",
                                "no NCCL_IB_HCA: the ports are not judged"]


def test_without_nccl_ib_hca_the_island_judges_the_ports_of_its_fabric_as_the_preflight(tmp_path):
    from test_cluster_preflight import ibstat_text as ib  # noqa: PLC0415

    cluster = HealthCluster(NAMES)
    clock = IslandClock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=60.0)
    ops.add("isl", SshTrainerSpec(nodes=tuple(b300(i) for i in range(3)), run_id="run1", workdir="/opt/yeto",
                                  run_script="echo island"))  # fabric ib, no NCCL_IB_HCA
    ops.start("isl")
    all_down = node_text(ibstat=(0, ib(down=tuple(range(8)))))
    cluster.health["b300-1"] = [node_text(), PORT_DOWN, PORT_DOWN, all_down, all_down]
    for _ in range(4):  # one port down: NCCL uses the others (a warning in the preflight)
        assert _tick(ops, clock) == "RUNNING"
    assert _tick(ops, clock) == "FAILED"  # none active, twice in a row
    # its ports are judged: nothing says they are not
    assert not any(e["event"] == node_health.EVENT_DEGRADED for e in ops.take_health_events("isl"))
    assert ops.islands["isl"].health_lost[1][0].startswith("no active ib port (mlx5_0 DOWN")


def _cuda_start(tmp_path, monkeypatch, *, node_extra=None, **section):
    """An ssh pool start on a cuda node (the node runtime): its ssh commands and the profile it wrote."""
    from test_node_route_ssh import NodeRun  # noqa: PLC0415

    from yeto.launch.nc import cli  # noqa: PLC0415

    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    run = NodeRun()
    ops, m = _ssh_ops(tmp_path, monkeypatch, run, node_extra=node_extra, **section)
    monkeypatch.setattr(ops, "preflight", lambda node: {})
    ops.start(m, ["f09-0"])
    cmds = [c[0][-1] for c in run.calls if c[0][0] == "ssh"]
    profile = json.loads(next(c[1]["input"] for c in run.calls if c[0][0] == "ssh" and "profile.json" in c[0][-1]))
    ops.probe(m)
    probe = [c[0][-1] for c in run.calls if c[0][0] == "ssh" and "YETO_NC_PROBE" in c[0][-1]][-1]
    return ops, m, cmds, profile, probe


def test_the_ssh_pool_clears_an_old_check_at_start_and_node_health_off_turns_it_off(tmp_path, monkeypatch):
    health = sp.health_path("/srv/yeto/run0", "f09-0")
    (tmp_path / "on").mkdir()
    _, _, cmds, profile, probe = _cuda_start(tmp_path / "on", monkeypatch)
    start = next(c for c in cmds if "-m yeto.node run" in c)
    assert f"rm -f {health}" in cmds and cmds.index(f"rm -f {health}") < cmds.index(start)
    assert "--health-every-s" not in start and "node_health" not in profile and sp.HEALTH_MARK in probe
    for label, kw in (("provider", {"node_health": "off"}), ("node", {"node_extra": {"node_health": "off"}})):
        (tmp_path / label).mkdir()
        ops, _, cmds, profile, probe = _cuda_start(tmp_path / label, monkeypatch, **kw)
        start = next(c for c in cmds if "-m yeto.node run" in c)
        assert "--health-every-s 0" in start and profile["node_health"] == "off", label
        assert f"rm -f {health}" not in cmds and sp.HEALTH_MARK not in probe, label
    # provider.node_health wins over the node's
    (tmp_path / "wins").mkdir()
    _, _, cmds, profile, probe = _cuda_start(tmp_path / "wins", monkeypatch, node_health="auto",
                                             node_extra={"node_health": "off"})
    assert profile["node_health"] == "auto" and sp.HEALTH_MARK in probe
    assert "--health-every-s" not in next(c for c in cmds if "-m yeto.node run" in c)
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="node_health must be one of"):
        sp.parse_config(pool_cfg(tmp_path, node_health="sometimes"), inventory=[parse_node(
            {"name": "node-a", "host": "h", "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300",
             "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 1}, index=0)])


def test_the_ssh_pool_reads_only_its_own_ticket_and_a_stale_check_is_a_read_failure(tmp_path, monkeypatch):
    run = HealthRun()
    ops, m = _ssh_ops(tmp_path, monkeypatch, run, node_runtime=True)
    ops._health_generation[m.ticket] = 1
    now = time.time()
    run.health = {"ticket": "another", "generation": 1, "verdict": "lost", "reasons": ["x"], "at": now}
    assert ops.probe(m).health is None and m.ticket not in ops._health_lost
    run.health = {"ticket": m.ticket, "generation": 1, "verdict": "lost", "reasons": ["GPU 5 XID 79"],
                  "at": now - 4 * 60, "every_s": 60}
    probe = ops.probe(m)
    assert probe.health["verdict"] == "read_failed" and probe.health["stale"] is True
    assert m.ticket not in ops._health_lost and el.health_lost(probe) is None
    stale = replace(probe, health=run.health)
    assert el.health_lost(stale) == ["GPU 5 XID 79"] and el.health_lost(stale, now=now) is None
    assert el.classify(ACTIVE, stale, now=now, unreachable_since=None, cfg=CFG, joined=True).kind == el.HEALTHY
    run.health["at"] = now
    assert ops.probe(m).health["verdict"] == "lost" and m.ticket in ops._health_lost


def test_the_pool_logs_a_stale_check_and_what_a_node_cannot_judge_once():
    clock = PoolClock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    probe = cloud.probe
    health = {"verdict": "ok", "reasons": [], "at": clock.t, "every_s": 60,
              "degraded": ["XID unread (no dmesg / journalctl -k access)"]}
    cloud.probe = lambda member: replace(probe(member), health=dict(health)) if member.ticket == m.ticket \
        else probe(member)
    _run(pool, clock, clock.t + 180)
    assert _names(pool).count("node_health_degraded") == 1
    event = next(e for e in pool.events.events if e["event"].endswith("node_health_degraded"))
    assert event["reasons"] == ["XID unread (no dmesg / journalctl -k access)"] and event["ticket"] == m.ticket
    health.update(verdict="lost", reasons=["GPU 5 XID 79"], at=clock.t - 1000)  # stopped checking: stale
    _run(pool, clock, clock.t + 60)
    assert m.state == READY and "node_lost" not in _names(pool)
    assert _names(pool).count("node_health_read_failed") == 1
    stale = next(e for e in pool.events.events if e["event"].endswith("node_health_read_failed"))
    assert stale["reasons"][0].startswith("health check stale: the last one ")


def test_inventory_node_health_parses_at_the_top_and_per_node():
    import pytest  # noqa: PLC0415

    from yeto.launch.node_inventory import InventoryError, parse_inventory  # noqa: PLC0415

    def entry(name, **kw):
        return {"name": name, "host": f"ops@{name}", "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300",
                "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 1024, **kw}

    assert [n.node_health for n in parse_inventory({"nodes": [entry("a"), entry("b")]})] == [None, None]
    nodes = parse_inventory({"node_health": "off", "nodes": [entry("a"), entry("b", node_health="auto")]})
    assert [n.node_health for n in nodes] == ["off", "auto"]
    assert parse_inventory({"nodes": [entry("a", node_health=False)]})[0].node_health == "off"  # YAML's bare off
    with pytest.raises(InventoryError, match="node_health must be one of"):
        parse_inventory({"nodes": [entry("a", node_health="later")]})
    # YAML's bare on is true; on means auto
    assert parse_inventory({"node_health": True, "nodes": [entry("a")]})[0].node_health == "auto"
    assert parse_inventory({"nodes": [entry("a", node_health="on")]})[0].node_health == "auto"
    with pytest.raises(InventoryError, match="inventory: node_health"):
        parse_inventory({"node_health": "yes please", "nodes": [entry("a")]})


def test_the_pool_dry_run_says_node_health_only_when_declared(tmp_path, capsys):
    node = parse_node({"name": "node-a", "host": "ops@node-a", "device_family": "cuda", "cpu_arch": "x86_64",
                       "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 1}, index=0)
    assert sp.health_summary(sp.parse_config(pool_cfg(tmp_path), inventory=[node])) is None
    assert sp.health_summary(sp.parse_config(pool_cfg(tmp_path, node_health="auto"), inventory=[node])) == "on (cuda)"
    assert sp.health_summary(sp.parse_config(pool_cfg(tmp_path, node_health="off"), inventory=[node])) == "off"
    old = sp.parse_config(pool_cfg(tmp_path, node_health="auto", node_runtime=False), inventory=[node])
    assert sp.health_summary(old) == "off"  # the old start runs no node runtime: no check


def test_a_node_runtime_whose_profile_says_off_checks_nothing(tmp_path):
    rt, _, runs = _runtime(tmp_path)
    assert rt.health_on
    objects = DirectoryObjectStore(tmp_path / "s2")
    code = records.CodeRef(sha="abc", key="node-code/abc.tar", sha256="")
    assignment, _ = records.assign(objects, ticket="t-o", shard="s0", spec={"sampler_id": "s0"}, code=code)
    off = NodeRuntime(objects, assignment, _profile(tmp_path, node_health="off"), store_uri=None,
                      environ={"PATH": "/usr/bin"}, skip_preflight=True, log=lambda line: None)
    assert not off.health_on and HEALTH_ENV not in off.plan(str(tmp_path / "y")).env
    from yeto.node.route import runtime_command  # noqa: PLC0415

    line = runtime_command("python3", "t", "/s", env_file=None, token_file="/t", health_every_s=0)
    assert line.endswith("--skip-preflight --health-every-s 0")
    assert "--health-every-s" not in runtime_command("python3", "t", "/s", env_file=None, token_file="/t")


def test_the_default_hook_leaves_an_island_that_ended_meanwhile_as_it_ended(tmp_path):
    cluster = HealthCluster(NAMES)
    ops, clock, _ = _island(tmp_path, cluster)
    cluster.health["b300-1"] = [node_text(), PORT_DOWN, PORT_DOWN]
    for _ in range(2):
        assert _tick(ops, clock) == "RUNNING"
    rec = ops.islands["isl"]
    real = ops._node_health_lost
    queued = []
    ops._node_health_lost = lambda *args: queued.append(args)  # hold the call: the island ends first
    assert _tick(ops, clock) == "RUNNING" and len(queued) == 1
    ops._fail(rec, "rank 2 (b300-2) exited 1")  # it ended another way between the poll and the hook
    real(*queued[0])
    assert rec.reason == "rank 2 (b300-2) exited 1" and rec.health_lost == {}  # the hook does not rewrite it
    assert ops.island_decision("isl", 0).cause != "lost"


def test_a_failed_removal_of_the_old_check_does_not_stop_the_start(tmp_path, monkeypatch):
    real = sp.SshSamplerPoolOps.ssh

    def ssh(self, node, command, **kw):
        if command.startswith("rm -f ") and command.endswith(".health.json"):
            raise subprocess.TimeoutExpired(["ssh"], 60)
        return real(self, node, command, **kw)

    monkeypatch.setattr(sp.SshSamplerPoolOps, "ssh", ssh)
    _, _, cmds, _, probe = _cuda_start(tmp_path, monkeypatch)
    assert any("-m yeto.node run" in c for c in cmds) and sp.HEALTH_MARK in probe
