"""Stale-incarnation trial containers: labels, sweep, and the gate's room."""

from __future__ import annotations

import json

from yeto.rl.engine import store_dispense as sd
from yeto.sandbox.docker import DockerHost
from yeto.sandbox.envs import parse_envs
from yeto.sandbox.gate import SandboxGate
from yeto.sandbox.trial_labels import Owner, owner_from_env, sweep_stale, trial_labels

KEYS = ("ai.yeto.run", "ai.yeto.shard", "ai.yeto.inc", "com.docker.compose.project")


class Daemon:
    """A fake docker CLI: containers {id: labels}; records removals."""

    def __init__(self, containers):
        self.containers = dict(containers)
        self.removed: list[str] = []
        self.net_rm: list[str] = []

    def __call__(self, argv, env, stdin):
        args = list(argv[1:])
        if args[0] == "ps":
            filters = [args[i + 1] for i, a in enumerate(args) if a == "--filter"]
            fmt = args[args.index("--format") + 1]
            keys = [part.split('"')[1] for part in fmt.split("\t")[1:]]
            lines = []
            for cid, labels in self.containers.items():
                if all(labels.get(f[6:].split("=", 1)[0]) == f[6:].split("=", 1)[1] for f in filters):
                    lines.append("\t".join([cid, *(labels.get(k, "") for k in keys)]))
            return 0, "\n".join(lines) + "\n", ""
        if args[:2] == ["rm", "-f"]:
            for cid in args[2:]:
                self.removed.append(cid)
                self.containers.pop(cid, None)
            return 0, "", ""
        if args[:2] == ["network", "ls"]:
            project = args[-1].split("=", 2)[2]
            return 0, f"net-{project}\n", ""
        if args[:2] == ["network", "rm"]:
            self.net_rm += args[2:]
            return 0, "", ""
        return 1, "", "unexpected"


def _c(run, shard, inc, project):
    labels = {"com.docker.compose.project": project}
    for k, v in (("ai.yeto.run", run), ("ai.yeto.shard", shard), ("ai.yeto.inc", inc)):
        if v:
            labels[k] = v
    return labels


def _host():
    return Daemon({
        "old1": _c("r1", "s0", "i-old", "p1"),
        "old2": _c("r1", "s0", "i-old2", "p2"),
        "cur": _c("r1", "s0", "i-new", "p3"),
        "othershard": _c("r1", "s1", "i-old", "p4"),
        "otherrun": _c("r2", "s0", "i-old", "p5"),
        "nolabel": {},
        "noinc": _c("r1", "s0", "", "p6"),
    })


def test_trial_labels_carry_run_shard_incarnation_and_trial():
    env = {"YETO_SANDBOX_RUN_LABEL": "r1", "YETO_SANDBOX_SHARD": "s0", "YETO_SAMPLER_INCARNATION": "i1"}
    assert trial_labels("cg__abc", env) == {"ai.yeto.run": "r1", "ai.yeto.shard": "s0",
                                            "ai.yeto.inc": "i1", "ai.yeto.trial": "cg__abc"}
    assert trial_labels("x", {}) == {}
    assert owner_from_env(env) == Owner("r1", "s0", "i1")
    assert owner_from_env({"YETO_SANDBOX_RUN_LABEL": "r1"}) is None


def test_sweep_removes_only_earlier_incarnations_of_this_shard():
    daemon = _host()
    n = sweep_stale(DockerHost("ssh://h1", runner=daemon), Owner("r1", "s0", "i-new"))
    assert n == 2 and sorted(daemon.removed) == ["old1", "old2"]
    assert sorted(daemon.net_rm) == ["net-p1", "net-p2"]
    assert {"cur", "othershard", "otherrun", "nolabel", "noinc"} <= set(daemon.containers)


def test_sweep_without_incarnation_touches_nothing():
    daemon = _host()
    assert sweep_stale(DockerHost("ssh://h1", runner=daemon), Owner("r1", "s0", "")) == 0
    assert daemon.removed == []


def _gate(daemon, owner, max_sandboxes=8):
    cfg = parse_envs({"run": "r1", "hosts": [{"name": "h1", "docker_host": "ssh://h1",
                                              "max_sandboxes": max_sandboxes}]})
    return SandboxGate(None, cfg, host=cfg.host("h1"), group_size=1,
                       docker=DockerHost("ssh://h1", runner=daemon), ttl_s=0, owner=owner)


def test_gate_removes_orphans_and_reports_breakdown():
    daemon = _host()  # 7 running, 2 of them stale orphans of shard s0
    gate = _gate(daemon, Owner("r1", "s0", "i-new"), max_sandboxes=7)
    assert gate.free_groups() == 2  # the orphans no longer hold room
    assert sorted(daemon.removed) == ["old1", "old2"]
    probe = gate.last_probe
    assert probe["host_running"] == 7 and probe["stale"] == 2 and probe["removed"] == 2
    assert probe["foreign"] == 2 and probe["ours"] == 5 and probe["shard"] == 2
    assert gate.free_groups() == 2 and len(daemon.removed) == 2  # nothing more to remove


def test_gate_without_owner_counts_everything_and_removes_nothing():
    daemon = _host()
    gate = _gate(daemon, None, max_sandboxes=7)
    gate.docker.count_running = lambda: len(daemon.containers)
    assert gate.free_groups() == 0
    assert daemon.removed == [] and gate.last_probe == {"host_running": 7, "free": 0}


def test_recover_sweeps_earlier_incarnations(tmp_path):
    daemon = _host()
    store = sd.DirObjects(tmp_path / "store")
    rows = [{"text": "q0", "label": "", "metadata": {"task_id": "t0"}}]
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    gate = _gate(daemon, Owner("r1", "s0", "i-new"))
    gate.view = lambda: None
    p = sd.StoreDispensedPrompts.open(path, {"backend": "store", "namespace": "ns", "scan_s": 1, "max_scan_s": 1},
                                      store=store, holder="s0", incarnation="i-new", input_key="text",
                                      label_key="label", groups_per_round=1, renew=False,
                                      sleep=lambda s: None, gate=gate)
    events = []
    p.emit = lambda e, **f: events.append((e, f))
    p.recover()
    assert sorted(daemon.removed) == ["old1", "old2"]
    assert ("rl_sandbox_orphans_removed", "removed") in [(e, next(iter(f.keys() - {"backend", "namespace"}), None)) for e, f in events]


def test_sweep_shard_removes_every_incarnation_of_that_shard_only():
    from yeto.sandbox.trial_labels import sweep_shard

    daemon = _host()
    assert sweep_shard(DockerHost("ssh://h1", runner=daemon), "r1", "s1") == 1
    assert daemon.removed == ["othershard"] and daemon.net_rm == ["net-p4"]
    assert sweep_shard(DockerHost("ssh://h1", runner=daemon), "", "s0") == 0


def test_gate_sweep_shard_refuses_own_shard_and_needs_owner():
    import pytest

    daemon = _host()
    gate = _gate(daemon, Owner("r1", "s2", "i-new"))
    assert gate.sweep_shard("s0") == 4  # old1, old2, cur, noinc: all incarnations of the predecessor
    assert "otherrun" in daemon.containers and "othershard" in daemon.containers
    with pytest.raises(ValueError):
        gate.sweep_shard("s2")
    assert _gate(_host(), None).sweep_shard("s0") == 0


class _Lease:
    def __init__(self, records):
        self.records = records


def test_replacement_sweeps_predecessor_only_when_left_or_fenced(monkeypatch):
    from yeto.rl.engine import engine_sampler_entry as e
    from yeto.rl.engine import incarnation_lease

    leases = {}
    monkeypatch.setattr(incarnation_lease, "read_lease", lambda store, sampler: leases.get(sampler))

    class Prompts:
        def __init__(self):
            self.daemon = _host()
            self.gate = _gate(self.daemon, Owner("r1", "s2", "i-new"))

    events = []
    emit = lambda ev, **f: events.append((ev, f))  # noqa: E731
    p = Prompts()  # predecessor still running, lease its own: untouched
    leases["s0"] = {"incarnation": "i-old"}
    assert e.sweep_predecessor(p, None, "s0", False, emit) == 0 and p.daemon.removed == []
    leases["s0"] = {"incarnation": "pool-fence:ticket1"}  # evicted: a pool fence holds the lease
    assert e.sweep_predecessor(p, None, "s0", False, emit) == 4
    assert events[-1] == ("rl_sandbox_orphans_removed",
                          {"removed": 4, "shard": "s0", "why": "predecessor_fenced"})
    p = Prompts()
    leases.clear()
    assert e.sweep_predecessor(p, None, "s0", True, emit) == 4  # predecessor left (handoff)
    assert events[-1][1]["why"] == "predecessor_left"
    assert e.sweep_predecessor(object(), None, "s0", True, emit) == 0  # no gate


# -- yeto sandbox teardown / run down: this run's trial containers and their networks ----

class JsonDaemon(Daemon):
    """``Daemon`` that also answers the ``{{json .}}`` listing and inspect of the sidecar teardown."""

    def __call__(self, argv, env, stdin):
        args = list(argv[1:])
        if args[0] == "ps" and args[args.index("--format") + 1] == "{{json .}}":
            filters = [args[i + 1] for i, a in enumerate(args) if a == "--filter"]
            rows = [cid for cid, labels in self.containers.items()
                    if all(labels.get(f[6:].split("=", 1)[0]) == f[6:].split("=", 1)[1] for f in filters)]
            return 0, "".join(json.dumps({"ID": cid}) + "\n" for cid in rows), ""
        if args[:2] == ["container", "inspect"]:
            labels = self.containers.get(args[2])
            return (0, json.dumps([{"Config": {"Labels": labels}}]), "") if labels is not None else (1, "[]", "no")
        if args[0] == "images":
            return 0, "", ""
        return super().__call__(argv, env, stdin)


def _teardown_world(tmp_path):
    from yeto.sandbox.docker import LABEL_RUN

    daemon = JsonDaemon({
        "side": {LABEL_RUN: "r1"},
        "t1": _c("r1", "s0", "i1", "p1"),
        "t2": _c("r1", "s1", "i2", "p2"),
        "other": _c("r2", "s0", "i1", "p9"),
        "foreign": {},
    })
    cfg = parse_envs({"run": "r1", "hosts": [{"name": "h1", "docker_host": "ssh://u@h1"}], "tasksets": []},
                     base_dir=tmp_path)
    return daemon, cfg


def test_teardown_removes_this_runs_trial_containers_and_their_networks(tmp_path):
    """Report: after pool-stop 20 trial containers and 20 compose networks of the run
    (``ai.yeto.run``) stayed on the sandbox host; teardown removed only the sidecars."""
    from yeto.sandbox import ops

    daemon, cfg = _teardown_world(tmp_path)

    class Store:
        def read(self, key):
            return None

        def write(self, key, data):
            pass

    out = ops.teardown(cfg, Store(), factory=lambda host: DockerHost(host.docker_host, runner=daemon))
    assert set(out["h1"]["trial_containers"]) == {"t1", "t2"} and out["h1"]["trial_networks"] == ["p1", "p2"]
    assert set(daemon.containers) == {"other", "foreign"}  # another run's and an unlabelled one stay
    assert sorted(daemon.net_rm) == ["net-p1", "net-p2"]


def test_teardown_dry_run_names_the_trial_containers_and_removes_nothing(tmp_path):
    from yeto.sandbox import ops

    daemon, cfg = _teardown_world(tmp_path)
    out = ops.teardown(cfg, None, factory=lambda host: DockerHost(host.docker_host, runner=daemon), dry_run=True)
    assert set(out["h1"]["trial_containers"]) == {"t1", "t2"} and len(daemon.containers) == 5 and not daemon.net_rm


def test_run_down_sweeps_trial_networks_too(monkeypatch):
    """``yeto rl down`` with ``sandbox.docker_host``: the trial containers and their networks."""
    from yeto.rl import up
    from yeto.sandbox import docker as docker_mod

    daemon = Daemon({"t1": _c("r1", "s0", "i1", "p1"), "other": _c("r2", "s0", "i1", "p9")})
    monkeypatch.setattr(docker_mod, "subprocess_runner", lambda timeout_s=0: daemon)
    run = up.Run.__new__(up.Run)
    run.run, run.env = "r1", {"DOCKER_CONTEXT": "x"}
    run._sweep_trials("ssh://u@h1")
    assert set(daemon.containers) == {"other"} and daemon.net_rm == ["net-p1"]


def test_run_down_counts_stopped_containers_and_unreachable_hosts(monkeypatch):
    """``yeto rl down`` verification: an exited trial container left on a host and a
    host that did not answer both make the down unverified."""
    from yeto.rl import up

    run = up.Run.__new__(up.Run)
    run.run, run.env = "r1", {}
    seen = []

    def ps(argv, **k):
        seen.append(argv[4])
        if argv[2] == "ssh://u@h2":
            raise up.subprocess.TimeoutExpired("docker", 60)
        out = "exited1\n" if argv[4] == "-aq" and argv[2] == "ssh://u@h1" else ""
        return up.subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(up.subprocess, "run", ps)
    left, unreachable = run.left_trial_containers(["ssh://u@h0", "ssh://u@h1", "ssh://u@h2"])
    assert left == 1 and unreachable == ["ssh://u@h2"] and set(seen) == {"-aq"}
