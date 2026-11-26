"""Where no node health check runs, what the launcher writes is main's, byte for byte.

A device family without a checker (Ascend here), a node whose inventory says
``node_health: off`` and a pool whose provider says ``node_health: off`` ask the same
status call, run the same probe and start, write the same heartbeat, plan environment,
island record and dry run as main did before node health (``yeto.hw.node_health``).

``fixtures/node_health_main/samples.json`` was recorded by :func:`capture` on main (the
commit before node health), with the code of main on the import path; it is not
regenerated from this tree. ``off`` is what this tree adds to turn the check off (main
knew no ``node_health`` key, so it was recorded with none).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FIXTURE = Path(__file__).parent / "fixtures" / "node_health_main" / "samples.json"
OFF = {"node_health": "off"}


class _Patch:
    """monkeypatch without pytest (the samples are recorded outside a test run too)."""

    def __init__(self):
        self.undo: list[tuple[Any, str, Any]] = []

    def setattr(self, target, name, value):
        self.undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def close(self):
        for target, name, value in reversed(self.undo):
            setattr(target, name, value)


TIMES = ("at", "since", "updated", "time", "created_at")


def _times(value: Any) -> Any:
    """Wall-clock stamps as ``<t>`` (the samples are of what is written, not of when)."""
    if isinstance(value, dict):
        return {k: "<t>" if k in TIMES and isinstance(v, (int, float)) else _times(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_times(v) for v in value]
    return value


def _norm(value: Any, root: Path, ticket: str | None = None) -> Any:
    text = json.dumps(_times(value), sort_keys=True).replace(str(root), "<T>").replace(str(Path.home()), "<HOME>")
    if ticket:
        text = text.replace(ticket, "<TICKET>")
    return json.loads(text)


# ------------------------------------------------------------------ trainer island (ssh:)
def _trainer(root: Path, family: str, node_kw: dict) -> dict[str, Any]:
    import os  # noqa: PLC0415

    from test_rl_ssh_resident import NAMES, FakeCluster, _controller, b300, make_ops  # noqa: PLC0415
    from test_rl_ssh_resident import Clock as IslandClock  # noqa: PLC0415

    from yeto.hw import card_tools  # noqa: PLC0415
    from yeto.launch import island_lifecycle as lifecycle  # noqa: PLC0415
    from yeto.launch.ssh_trainer_island import NODE_STATE_MARK, SshTrainerSpec  # noqa: PLC0415

    class Cluster(FakeCluster):
        def __init__(self, names):
            super().__init__(names)
            self.status_scripts: list[str] = []

        def run(self, node, script, *, timeout=300.0):
            if "npu-smi" in script and NODE_STATE_MARK not in script:  # an Ascend node's card check
                return super().run(node, "uname -m", timeout=timeout)
            if NODE_STATE_MARK in script:
                self.status_scripts.append(script)
            return super().run(node, script, timeout=timeout)

    patch = _Patch()
    patch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    old_ledger = os.environ.get("YETO_RESOURCE_LEDGER")
    os.environ["YETO_RESOURCE_LEDGER"] = str(root / "ledger.jsonl")
    try:
        npu = dict(device_family="npu", card="910B4", hbm_gib=64.0, docker_args=("--device=/dev/davinci0",),
                   fabric="tcp") if family == "npu" else {}
        nodes = tuple(b300(i, **npu, **node_kw) for i in range(3))
        spec = SshTrainerSpec(nodes=nodes, run_id="run1", workdir="/opt/yeto", run_script="echo island",
                              env={"YETO_X": "1"})
        cluster = Cluster(NAMES)
        ops = make_ops(root, cluster, clock=IslandClock())
        ops.add("isl", spec)
        ops.start("isl")

        class Objects:
            def __init__(self):
                self.docs = {}

            def read(self, key):
                return self.docs.get(key)

            def write(self, key, data):
                self.docs[key] = data

        objects = Objects()
        store = lifecycle.IslandStateStore(objects, controller={"host": "test"})
        recorder = lifecycle.FleetRecorder(store, {"isl": lifecycle.IslandRef(island=0, source="ssh")})
        ctl = _controller(ops, recorder=recorder)
        rec = ctl.learners["isl"]
        ctl._record(rec, reason="launched")
        for _ in range(2):
            ops.clock.t += 61
            ctl._poll_recorded(rec)
        record = json.loads(objects.docs[lifecycle.state_key(0)])
        return _norm({"status_scripts": cluster.status_scripts, "record": record}, root)
    finally:
        patch.close()
        if old_ledger is None:
            os.environ.pop("YETO_RESOURCE_LEDGER", None)
        else:
            os.environ["YETO_RESOURCE_LEDGER"] = old_ledger


def _dry_run(family: str) -> dict[str, Any]:
    from test_rl_ssh_resident import b300  # noqa: PLC0415

    from yeto.launch.ssh_resident import dry_run_record  # noqa: PLC0415
    from yeto.launch.ssh_trainer_island import SshTrainerSpec  # noqa: PLC0415

    npu = dict(device_family="npu", card="910B4", hbm_gib=64.0, docker_args=("--device=/dev/davinci0",),
               fabric="tcp") if family == "npu" else {}
    spec = SshTrainerSpec(nodes=tuple(b300(i, **npu) for i in range(2)), run_id="run1", workdir="/opt/yeto",
                          run_script="echo hi", env={"NCCL_IB_HCA": "mlx5_0,mlx5_3"})
    from yeto import runs  # noqa: PLC0415

    state = runs.run_dir("run1").parent  # the runs directory (~/.yeto/runs, a temporary one under the tests)
    return _norm(dry_run_record(SimpleNamespace(tensor_parallel=8, pipeline_parallel=2), spec, "isl"), state)


# ------------------------------------------------------------------ ssh sampler pool
def _pool(root: Path, family: str, section: dict) -> dict[str, Any]:
    from test_node_route_ssh import NodeRun  # noqa: PLC0415
    from test_ssh_sampler_pool import pool_cfg  # noqa: PLC0415

    from yeto.launch import ssh_sampler_pool as sp  # noqa: PLC0415
    from yeto.launch.nc import cli  # noqa: PLC0415
    from yeto.launch.node_inventory import parse_node  # noqa: PLC0415

    class NoArchive:
        stdout = None

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    patch = _Patch()
    patch.setattr(sp.subprocess, "Popen", NoArchive)
    patch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    patch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    try:
        node = parse_node({"name": "node-a", "host": "ops@node-a", "device_family": family, "cpu_arch": "aarch64",
                           "card": "B300" if family == "cuda" else "910B4", "cards_per_node": 8,
                           "hbm_gib": 288 if family == "cuda" else 64, "host_memory_gib": 1024, "image": "img"},
                          index=0)
        cfg = pool_cfg(root)
        cfg["provider"].update(section)
        run = NodeRun()
        ops = sp.SshSamplerPoolOps(cfg, conf=sp.parse_config(cfg, inventory=[node]), run=run, state_dir=root)
        ticket = ops.order(0)
        m = SimpleNamespace(account=0, ticket=ticket, shard="f09-0", data_slot=None, replaces_shard=None,
                            predecessor_lost=False, unreachable_since=None)
        if family == "npu":  # the fake node answers an Ascend card check (a cuda start: the probe only)
            ops.start(m, ["f09-0"])
        start = [c[0][-1] for c in run.calls if c[0][0] == "ssh"]
        ops.probe(m)
        probe = [c[0][-1] for c in run.calls if c[0][0] == "ssh" and "YETO_NC_PROBE" in c[0][-1]]
        return _norm({"start": start, "probe": probe}, root, ticket)
    finally:
        patch.close()


# ------------------------------------------------------------------ node runtime (Ascend)
def _runtime(root: Path) -> dict[str, Any]:
    from yeto.launch.node_inventory import NodePaths, NodeProfile  # noqa: PLC0415
    from yeto.node import records  # noqa: PLC0415
    from yeto.node.report import SamplerNodeReport  # noqa: PLC0415
    from yeto.node.runtime import NodeRuntime  # noqa: PLC0415
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

    objects = DirectoryObjectStore(root / "store")
    code = records.CodeRef(sha="abc", key="node-code/abc.tar", sha256="")
    assignment, _ = records.assign(objects, ticket="t-h", shard="s0", spec={"sampler_id": "s0"}, code=code)
    profile = NodeProfile(name="box0", host="", device_family="npu", cpu_arch="x86_64", card="910B4",
                          cards_per_node=8, hbm_gib=64.0, host_memory_gib=0.0, container="process",
                          paths=NodePaths(work=str(root / "work"), models=str(root / "models")),
                          env={"NCCL_IB_HCA": "mlx5_0"})
    rt = NodeRuntime(objects, assignment, profile, store_uri=None, environ={"PATH": "/usr/bin"},
                     skip_preflight=True, log=lambda line: None)
    plan = rt.plan(str(root / "work" / "yeto"))
    seen: dict[str, Any] = {}

    class Child:
        def __init__(self, argv, env, **kw):
            seen["env"] = dict(env)

        def wait(self):
            report = SamplerNodeReport.from_env(seen["env"], SimpleNamespace(root=root / "store"), {"sampler_id": "s0"})
            report.metrics = lambda port: {}
            report.clock = report.reporter.clock = lambda: 5.0
            report._beat()
            return 0

    rt.popen = Child
    rt.start(plan)
    beat = dict(records.node_status(objects, "t-h").heartbeat)
    beat.update(host="<host>", pid=0)
    return _norm({"plan_env": dict(plan.env), "heartbeat": beat,
                  "work": sorted(str(p.relative_to(root)) for p in (root / "work").rglob("*"))}, root)


def capture(root: Path, off: dict | None = None) -> dict[str, Any]:
    """The samples; ``off``: what turns the node health check off (``{}`` on main)."""
    off = dict(off or {})
    out: dict[str, Any] = {}
    for label, run in (("trainer_npu", lambda d: _trainer(d, "npu", {})),
                       ("trainer_cuda_off", lambda d: _trainer(d, "cuda", off)),
                       ("pool_npu", lambda d: _pool(d, "npu", {})),
                       ("pool_cuda_runtime_false", lambda d: _pool(d, "cuda", {"node_runtime": False})),
                       ("runtime_npu", _runtime)):
        d = root / label
        d.mkdir(parents=True)
        out[label] = run(d)
    (root / "pool_cuda_off").mkdir()
    out["pool_cuda_off_probe"] = _pool(root / "pool_cuda_off", "cuda", off)["probe"]
    out["dry_run_npu"], out["dry_run_cuda"] = _dry_run("npu"), _dry_run("cuda")
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def record(path: Path = FIXTURE) -> None:
    """Write the samples (run on main: ``PYTHONPATH=<main tree>:<main tree>/tests``)."""
    with tempfile.TemporaryDirectory() as tmp:
        samples = capture(Path(tmp).resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=1, sort_keys=True) + "\n")


def test_without_a_check_every_sample_is_mains(tmp_path):
    want = json.loads(FIXTURE.read_text())
    got = capture(tmp_path.resolve(), OFF)
    assert sorted(got) == sorted(want)
    for key in want:
        assert got[key] == want[key], key


def test_the_samples_hold_what_they_say():
    """The recording is of the paths it names (not of an empty run)."""
    want = json.loads(FIXTURE.read_text())
    assert len(want["trainer_npu"]["status_scripts"]) == 6 and want["trainer_npu"]["record"]["state"] == "running"
    assert len(want["trainer_cuda_off"]["status_scripts"]) == 6
    assert all("nvidia-smi" not in s for s in want["trainer_cuda_off"]["status_scripts"])
    assert any("-m yeto.node run" in c for c in want["pool_npu"]["start"])
    assert want["pool_cuda_runtime_false"]["probe"] and want["pool_cuda_runtime_false"]["start"] == []
    assert want["pool_npu"]["probe"] and want["pool_cuda_off_probe"]
    assert want["runtime_npu"]["heartbeat"]["ticket"] == "t-h" and "health" not in want["runtime_npu"]["heartbeat"]
    assert "YETO_NODE_HEALTH_FILE" not in want["runtime_npu"]["plan_env"]
    assert "node_health" not in want["dry_run_npu"] and "node_health" not in want["dry_run_cuda"]

