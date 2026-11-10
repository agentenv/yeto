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
