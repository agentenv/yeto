"""``yeto sandbox`` (prepare / status / validate / teardown) on a fake docker
daemon, and the store dispenser's sandbox gate (readiness, capacity, affinity)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.task_dispense import DispenseEnded
from yeto.sandbox import ops
from yeto.sandbox import ready as ready_mod
from yeto.sandbox.docker import LABEL_RUN, DockerHost
from yeto.sandbox.envs import EnvsError, parse_envs
from yeto.sandbox.gate import SandboxGate, gate_from_spec


class FakeDaemon:
    """Enough of ``docker`` (and ssh curl/df, rsync) for the sandbox operations."""

    def __init__(self, *, bridge="10.9.0.1", root="/srv/docker", remote=(), broken=()) -> None:
        self.bridge, self.root = bridge, root
        self.images: dict[str, str] = {}
        self.containers: dict[str, dict] = {}
        self.remote = set(remote)  # pullable refs
        self.broken = set(broken)  # refs whose pull and build fail
        self.calls: list[list[str]] = []
        self.n = 0

    def add_container(self, name, image, *, labels=None, running=True, port=None, cport=9111, host_ip=None):
        self.n += 1
        cid = f"c{self.n:04d}"
        self.containers[cid] = {"Id": cid, "Name": "/" + name, "Config": {"Image": image, "Labels": dict(labels or {})},
                                "State": {"Running": running},
                                "HostConfig": {"PortBindings": {f"{cport}/tcp": [{"HostPort": str(port),
                                                                                  **({"HostIp": host_ip} if host_ip else {})}]}
                                               if port else {}},
                                "_port": port}
        return cid

    def _find(self, ref):
        for cid, c in self.containers.items():
            if cid == ref or c["Name"] == "/" + ref:
                return cid
        return None

    def mutations(self):
        return [c for c in self.calls if c[:2] in (["docker", "pull"], ["docker", "build"], ["docker", "run"],
                                                    ["docker", "rm"], ["docker", "rmi"])]

    def __call__(self, argv, env, stdin):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "ssh":
            cmd = argv[-1]
            if cmd.startswith("curl"):
                return 0, "", ""
            if cmd.startswith("df"):
                return 0, f"/dev/md0 1000000000 400000000 600000000 40% {self.root}\n", ""
            return 1, "", "?"
        if argv[0] == "rsync":
            return 0, "", ""
        a = argv[1:]
        if a[0] == "info":
            return 0, json.dumps({"DockerRootDir": self.root, "ServerVersion": "29", "SecurityOptions": []}), ""
        if a[:2] == ["network", "inspect"]:
            return 0, f"{self.bridge} ", ""
        if a[:2] == ["image", "inspect"]:
            ref = a[-1]
            return (0, self.images[ref] + "\n", "") if ref in self.images else (1, "", "No such image")
        if a[0] == "pull":
            if a[1] in self.remote and a[1] not in self.broken:
                self.images[a[1]] = "sha256:" + str(abs(hash(a[1])))[:12]
                return 0, "", ""
            return 1, "", "pull access denied"
        if a[0] == "build":
            tag = a[a.index("-t") + 1]
            if tag in self.broken:
                return 1, "", "build failed"
            if a[-1] == "-":
                base = stdin.decode().split("\n", 1)[0].split()[1]
                if base not in self.images:
                    return 1, "", f"no base {base}"
            self.images[tag] = "sha256:b" + str(abs(hash(tag)))[:11]
            return 0, "", ""
        if a[0] == "ps":
            if a[1:2] == ["-q"]:
                return 0, "".join(f"{cid}\n" for cid, c in self.containers.items() if c["State"]["Running"]), ""
            all_ = "-a" in a
            filters = [a[i + 1] for i, x in enumerate(a) if x == "--filter"]
            lines = []
            for cid, c in self.containers.items():
                if not all_ and not c["State"]["Running"]:
                    continue
                ok = True
                for f in filters:
                    k, v = f.split("=", 1)[1].split("=", 1)
                    ok &= c["Config"]["Labels"].get(k) == v
                if ok:
                    ports = f"{self.bridge}:{c['_port']}->9111/tcp" if c["_port"] else ""
                    lines.append(json.dumps({"ID": cid, "Names": c["Name"][1:], "Ports": ports}))
            return 0, "".join(line + "\n" for line in lines), ""
        if a[:2] == ["container", "inspect"]:
            cid = self._find(a[2])
            return (0, json.dumps([self.containers[cid]]), "") if cid else (1, "[]", "No such container")
        if a[0] == "run":
            name = a[a.index("--name") + 1]
            pub = a[a.index("-p") + 1]
            labels = dict(x.split("=", 1) for i, x in enumerate(a) if i and a[i - 1] == "--label")
            image = a[-1]
            if image not in self.images:
                return 1, "", "no image"
            if self._find(name):
                return 1, "", "name in use"
            cid = self.add_container(name, image, labels=labels, port=int(pub.split(":")[1]))
            return 0, cid + "\n", ""
        if a[:2] == ["rm", "-f"]:
            for ref in a[2:]:
                cid = self._find(ref)
                if cid:
                    del self.containers[cid]
            return 0, "", ""
        if a[0] == "images":
            return 0, "", ""
        if a[0] == "rmi":
            return 0, "", ""
        return 1, "", f"fake docker: unsupported {a}"


def _task(root: Path, tid: str, *, image=None, dockerfile=True, sidecar=None, test="echo ok"):
    d = root / tid
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir()
    env = f"docker_image = '{image}'\n" if image else ""
    (d / "task.toml").write_text(f"schema_version = \"1.1\"\n[task]\nname = '{tid}'\n[environment]\n{env}cpus = 1\n")
    if dockerfile:
        (d / "environment").mkdir()
        (d / "environment" / "Dockerfile").write_text("FROM base:1\n")
    (d / "tests" / "test.sh").write_text(test + "\n")
    (d / "solution" / "solve.sh").write_text("true\n")
    if sidecar:
        (d / "yeto-sidecars.json").write_text(json.dumps(sidecar))
    return d


@pytest.fixture
def world(tmp_path):
    tasks = tmp_path / "tasks"
    # cybergym-like: built agent image + a verification sidecar the tests reach over the bridge
    _task(tasks, "cg_1", image="sbx-cg:cg-1", test="curl {{yeto.sidecar.ts.url}}/verify",
          sidecar={"sidecars": [{"name": "ts", "image": "ts:cg-1", "build": "server", "port": 9111,
                                 "health": "/health", "env": {"AUTH_TOKEN": "x"}}],
                   "prefetch": ["runner:cg-1-vul"]})
    (tasks / "cg_1" / "server").mkdir()
    _task(tasks, "cg_2", image="sbx-cg:cg-2", test="curl {{yeto.sidecar.ts.url}}",
          sidecar={"sidecars": [{"name": "ts", "image": "ts:cg-2", "build": "server", "port": 9111,
                                 "health": "/health"}]})
    (tasks / "cg_2" / "server").mkdir()
    # TB2.1-like: a Docker Hub image per task, no sidecar, pi baked in
    tb = tmp_path / "tb"
    _task(tb, "tb_a", image="hub/tb:a")
    _task(tb, "tb_b", image="hub/tb:b", dockerfile=False)
    section = {"run": "r1", "hosts": [{"name": "h1", "docker_host": "ssh://u@h1", "ssh": "u@h1",
                                       "max_sandboxes": 10, "port_range": [20000, 20010]}],
               "tasksets": [{"name": "cg", "tasks_dir": str(tasks), "agent_base": {"image": "base:1"}},
                            {"name": "tb", "tasks_dir": str(tb), "image_source": "pull", "bake_runtime": "pi",
                             "image_repo": "sbx-tb"}]}
    daemon = FakeDaemon(remote={"base:1", "runner:cg-1-vul", "hub/tb:a", "hub/tb:b"})
    daemon.add_container("someone-else", "x:1", port=20000)  # a foreign container on our port range
    store = sd.DirObjects(tmp_path / "store")
    return {"cfg": parse_envs(section), "daemon": daemon, "store": store, "tmp": tmp_path,
            "factory": lambda h: DockerHost(h.docker_host, ssh=h.ssh, runner=daemon), "section": section}


def _prepare(w, **kw):
    return ops.prepare(w["cfg"], w["store"], stage_dir=w["tmp"] / "stage", factory=w["factory"],
                       sleep=lambda s: None, **kw)


def test_prepare_builds_pulls_bakes_starts_and_renders(world):
    report = _prepare(world)
    assert report["ok"], report
    d = world["daemon"]
    assert {"sbx-cg:cg-1", "ts:cg-1", "runner:cg-1-vul", "hub/tb:a"} <= set(d.images)
    baked = [i for i in d.images if i.startswith("sbx-tb:tb_a-pi-")]
    assert baked, d.images
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["hosts"]["h1"]["bridge_ip"] == "10.9.0.1" and doc["hosts"]["h1"]["data_root"] == "/srv/docker"
    e = doc["tasks"]["cg_1"]["hosts"]["h1"]
    assert e["prepared"] and e["digest"].startswith("sha256:") and e["validation"] is None
    port = e["sidecars"]["ts"]["host_port"]
    assert port != 20000 and 20000 < port <= 20010  # the foreign container's port is skipped
    staged = Path(e["staged"])
    assert (staged / "tests" / "test.sh").read_text().strip() == f"curl http://10.9.0.1:{port}/verify"
    assert baked[0] in (Path(doc["tasks"]["tb_a"]["hosts"]["h1"]["staged"]) / "task.toml").read_text()
    assert doc["tasks"]["tb_a"]["portable"] and not doc["tasks"]["cg_1"]["portable"]
    sidecar = [c for c in d.containers.values() if c["Config"]["Labels"].get(LABEL_RUN) == "r1"]
    assert len(sidecar) == 2


def test_prepare_is_idempotent(world):
    _prepare(world)
    d = world["daemon"]
    first = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    before = len(d.mutations())
    report = _prepare(world)
    assert report["ok"]
    assert len(d.mutations()) == before  # nothing pulled, built, started or removed again
    again = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    for tid in ("cg_1", "cg_2"):
        assert (again["tasks"][tid]["hosts"]["h1"]["sidecars"]["ts"]["host_port"]
                == first["tasks"][tid]["hosts"]["h1"]["sidecars"]["ts"]["host_port"])


def test_prepare_keeps_validation_of_unchanged_image(world):
    _prepare(world)
    ready_mod.merge_write(world["store"], "sandbox/ready.json", "r1",
                          tasks={"cg_1": {"hosts": {"h1": {"validation": {"passed": True}}}}})
    _prepare(world)
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["tasks"]["cg_1"]["hosts"]["h1"]["validation"] == {"passed": True}


def test_partial_failure_records_the_task_and_prepares_the_rest(world):
    world["daemon"].broken = {"sbx-cg:cg-2"}
    report = _prepare(world)
    assert not report["ok"]
    assert [f["task_id"] for f in report["failed"]] == ["cg_2"]
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["tasks"]["cg_2"]["hosts"]["h1"]["prepared"] is False
    assert "build failed" in doc["tasks"]["cg_2"]["hosts"]["h1"]["error"]
    assert all(doc["tasks"][t]["hosts"]["h1"]["prepared"] for t in ("cg_1", "tb_a", "tb_b"))
    world["daemon"].broken = set()
    assert _prepare(world)["ok"]  # the re-run finishes only what was missing


def test_unreachable_host_fails_its_tasks_only(world):
    section = dict(world["section"], hosts=world["section"]["hosts"] + [{"name": "h2", "docker_host": "ssh://u@h2"}])
    cfg = parse_envs(section)
    good = world["daemon"]

    def factory(h):
        if h.name == "h2":
            return DockerHost(h.docker_host, runner=lambda a, e, s: (255, "", "ssh: connect refused"))
        return DockerHost(h.docker_host, ssh=h.ssh, runner=good)

    report = ops.prepare(cfg, world["store"], stage_dir=world["tmp"] / "s", factory=factory, sleep=lambda s: None)
    assert report["hosts"]["h1"]["failed"] == 0 and report["hosts"]["h2"]["prepared"] == 0
    assert "probe failed" in report["host_facts"]["h2"]["error"]


def test_existing_foreign_container_is_adopted_not_touched(world):
    d = world["daemon"]
    d.images["ts:cg-1"] = "sha256:ts1"
    cid = d.add_container("cybergym-ts-cg-1", "ts:cg-1", port=19100)
    (world["tmp"] / "tasks" / "cg_1" / "yeto-sidecars.json").write_text(json.dumps(
        {"sidecars": [{"name": "ts", "image": "ts:cg-1", "port": 9111, "host_port": 19100,
                       "container_name": "cybergym-ts-cg-1", "health": "/health"}]}))
    world["cfg"] = parse_envs(world["section"])
    assert _prepare(world)["ok"]
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    sc = doc["tasks"]["cg_1"]["hosts"]["h1"]["sidecars"]["ts"]
    assert sc["adopted"] and sc["host_port"] == 19100 and sc["container"] == "cybergym-ts-cg-1"
    assert cid in d.containers and d.containers[cid]["Config"]["Labels"] == {}
    assert not [c for c in d.calls if c[:2] == ["docker", "rm"]]


def test_foreign_container_on_a_stale_bridge_address_is_an_error_and_left_alone(world):
    # the daemon's bridge moved (172.17.0.1 -> 10.9.0.1): the old binding publishes nothing
    d = world["daemon"]
    d.images["ts:cg-1"] = "sha256:ts1"
    cid = d.add_container("cybergym-ts-cg-1", "ts:cg-1", port=19100, host_ip="172.17.0.1")
    (world["tmp"] / "tasks" / "cg_1" / "yeto-sidecars.json").write_text(json.dumps(
        {"sidecars": [{"name": "ts", "image": "ts:cg-1", "port": 9111, "host_port": 19100,
                       "container_name": "cybergym-ts-cg-1", "health": "/health"}]}))
    world["cfg"] = parse_envs(world["section"])
    report = _prepare(world)
    assert [f["task_id"] for f in report["failed"]] == ["cg_1"]
    assert "172.17.0.1" in str(report["failed"]) and "10.9.0.1" in str(report["failed"])
    assert cid in d.containers and not [c for c in d.calls if c[:2] == ["docker", "rm"]]


def test_declared_health_without_ssh_fails_instead_of_passing(world):
    world["factory"] = lambda h: DockerHost(h.docker_host, runner=world["daemon"])  # no hosts[].ssh
    report = _prepare(world)
    failed = {f["task_id"]: f["error"] for f in report["failed"]}
    assert set(failed) == {"cg_1", "cg_2"} and "hosts[].ssh" in failed["cg_1"]
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["tasks"]["tb_a"]["hosts"]["h1"]["prepared"]  # no sidecar health declared: unchanged


def test_unknown_bridge_address_fails_the_whole_host(world):
    world["daemon"].bridge = ""
    report = _prepare(world)
    assert report["hosts"]["h1"]["prepared"] == 0 and not report["ok"]
    assert "bridge" in report["host_facts"]["h1"]["error"]
    assert not [c for c in world["daemon"].calls if c[:2] == ["docker", "run"]]


def test_own_running_sidecar_on_a_stale_bridge_address_is_not_reused(world):
    _prepare(world)
    d = world["daemon"]
    ours = [c for c in d.containers.values() if c["Config"]["Labels"].get(LABEL_RUN) == "r1"]
    for c in ours:  # the daemon's bridge moved after the first prepare
        for b in c["HostConfig"]["PortBindings"]["9111/tcp"]:
            b["HostIp"] = "172.17.0.1"
    report = _prepare(world)
    failed = {f["task_id"]: f["error"] for f in report["failed"]}
    assert set(failed) == {"cg_1", "cg_2"}
    assert "172.17.0.1" in failed["cg_1"] and "docker rm" in failed["cg_1"] and "10.9.0.1" in failed["cg_1"]


def test_foreign_container_on_the_bridge_address_is_adopted(world):
    d = world["daemon"]
    d.images["ts:cg-1"] = "sha256:ts1"
    d.add_container("cybergym-ts-cg-1", "ts:cg-1", port=19100, host_ip="10.9.0.1")
    (world["tmp"] / "tasks" / "cg_1" / "yeto-sidecars.json").write_text(json.dumps(
        {"sidecars": [{"name": "ts", "image": "ts:cg-1", "port": 9111, "host_port": 19100,
                       "container_name": "cybergym-ts-cg-1", "health": "/health"}]}))
    world["cfg"] = parse_envs(world["section"])
    assert _prepare(world)["ok"]


def test_foreign_container_with_another_image_is_an_error_and_left_alone(world):
    d = world["daemon"]
    cid = d.add_container("cybergym-ts-cg-1", "other:1", port=19100)
    (world["tmp"] / "tasks" / "cg_1" / "yeto-sidecars.json").write_text(json.dumps(
        {"sidecars": [{"name": "ts", "image": "ts:cg-1", "port": 9111, "container_name": "cybergym-ts-cg-1"}]}))
    world["cfg"] = parse_envs(world["section"])
    report = _prepare(world)
    assert [f["task_id"] for f in report["failed"]] == ["cg_1"]
    assert cid in d.containers


def test_teardown_removes_only_this_runs_containers(world):
    d = world["daemon"]
    foreign = [d.add_container(f"cybergym-ts-{i}", "ts:x") for i in range(88)]
    other_run = d.add_container("yeto-other-x", "ts:x", labels={LABEL_RUN: "r2"})
    _prepare(world)
    ours = {cid for cid, c in d.containers.items() if c["Config"]["Labels"].get(LABEL_RUN) == "r1"}
    assert len(ours) == 2
    out = ops.teardown(world["cfg"], world["store"], factory=world["factory"])
    assert set(out["h1"]["containers"]) == ours
    assert all(cid in d.containers for cid in foreign + [other_run])
    assert not ours & set(d.containers)
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["tasks"]["cg_1"]["hosts"]["h1"]["prepared"] is False  # its sidecar is gone


def test_teardown_dry_run_removes_nothing(world):
    _prepare(world)
    n = len(world["daemon"].containers)
    out = ops.teardown(world["cfg"], world["store"], factory=world["factory"], dry_run=True)
    assert len(out["h1"]["containers"]) == 2 and len(world["daemon"].containers) == n


def test_status_reports_host(world):
    _prepare(world)
    st = ops.status(world["cfg"], world["store"], factory=world["factory"])["hosts"]["h1"]
    assert st["containers_running"] == 3 and st["containers_ours"] == 2
    assert st["images_ready"] == "4/4" and st["sidecars_running"] == "2/2"
    assert st["disk"]["free_kb"] == 600000000
    assert st["dispense"] == {"ready": 0, "pending": 4, "failed": 0}  # validation required


def test_validate_marks_pass_fail_and_infra(world):
    _prepare(world)
    scores = {("cg_1", "oracle"): (1.0, None), ("cg_1", "nop"): (0.0, None),
              ("cg_2", "oracle"): (0.0, None), ("cg_2", "nop"): (0.0, None),
              ("tb_a", "oracle"): (None, "EnvironmentStartTimeoutError: x"), ("tb_a", "nop"): (0.0, None),
              ("tb_b", "oracle"): (1.0, None), ("tb_b", "nop"): (1.0, None)}
    seen = []

    def trial(task, host, root, agent):
        seen.append((task.task_id, agent))
        assert (root / task.task_id / "task.toml").is_file()  # the host's rendered copy
        return scores[(task.task_id, agent)]

    report = ops.validate(world["cfg"], world["store"], trial=trial)
    assert report["passed"] == ["cg_1"]
    by = {r["task_id"]: r for r in report["results"]}
    assert by["cg_2"]["passed"] is False and by["cg_2"]["reason"] == "oracle_reward=0.0"
    assert by["tb_b"]["reason"] == "nop_reward=1.0"
    assert by["tb_a"]["passed"] is None and by["tb_a"]["reason"].startswith("infra")
    seen.clear()
    ops.validate(world["cfg"], world["store"], trial=trial)
    assert {t for t, _ in seen} == {"tb_a"}  # only the infra failure is checked again
    view = ready_mod.ReadyView(ready_mod.read_manifest(world["store"], "sandbox/ready.json"), host="h1")
    assert view.classify("cg_1")[0] == "ready" and view.classify("cg_2")[0] == "failed"
    assert view.classify("tb_a")[0] == "pending"


def test_envs_rejects_unknown_keys_and_bad_run():
    with pytest.raises(EnvsError):
        parse_envs({"run": "r", "hostz": []})
    with pytest.raises(EnvsError):
        parse_envs({"run": "bad run!"})


# ------------------------------------------------------------- dispenser gate


def _manifest(store, tasks, *, key="sandbox/ready.json"):
    """tasks: {tid: {host: passed | 'prepared' | False(validation failed)}}."""
    doc = ready_mod.empty("r1")
    for tid, hosts in tasks.items():
        entry = {"portable": True, "hosts": {}}
        for h, state in hosts.items():
            if state == "prepared":
                entry["hosts"][h] = {"prepared": True, "validation": None}
            else:
                entry["hosts"][h] = {"prepared": True,
                                     "validation": {"passed": state, "reason": None if state else "oracle_reward=0.0"}}
        doc["tasks"][tid] = entry
    store.write(key, json.dumps(doc).encode())


class FixedRoom:
    def __init__(self, n):
        self.n = n

    def count_running(self):
        return self.n


def _prompts(tmp_path, store, gate, n=4, groups=2):
    rows = [{"text": f"q{i}", "label": "", "metadata": {"task_id": f"t{i}"}} for i in range(n)]
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return sd.StoreDispensedPrompts.open(path, {"backend": "store", "namespace": "ns", "scan_s": 1, "max_scan_s": 1},
                                         store=store, holder="s0", incarnation="i0", input_key="text",
                                         label_key="label", groups_per_round=groups, renew=False,
                                         sleep=lambda s: None, gate=gate)


def _gate(store, *, host="h1", max_sandboxes=None, running=0, group_size=1, affinity="strict"):
    cfg = parse_envs({"run": "r1", "affinity": affinity,
                      "hosts": [{"name": "h1", "docker_host": "ssh://h1", "max_sandboxes": max_sandboxes},
                                {"name": "h2", "docker_host": "ssh://h2"}]})
    return SandboxGate(store, cfg, host=cfg.host(host) if host else None, group_size=group_size,
                       docker=FixedRoom(running), ttl_s=0)


def _claimed(store):
    keys, _ = store.list_meta("ns/claims/")
    return sorted(k for k, _ in keys)


def test_only_ready_tasks_are_claimed_and_later_ready_ones_join(tmp_path):
    store = sd.DirObjects(tmp_path / "store")
    _manifest(store, {"t0": {"h1": "prepared"}, "t1": {"h1": True}, "t2": {"h1": True}})
    p = _prompts(tmp_path, store, _gate(store))
    got = p.take(4)
    assert sorted(p.slots.row(s) for s in got) == [1, 2]
    p.written(0, got)
    _manifest(store, {"t0": {"h1": True}, "t1": {"h1": True}, "t2": {"h1": True}, "t3": {"h1": True}})
    got = p.take(4)
    assert sorted(p.slots.row(s) for s in got) == [0, 3]


def test_validation_failure_is_dropped_with_reason(tmp_path):
    store = sd.DirObjects(tmp_path / "store")
    _manifest(store, {"t0": {"h1": False}, "t1": {"h1": True}, "t2": {"h1": True}, "t3": {"h1": True}})
    p = _prompts(tmp_path, store, _gate(store))
    got = p.take(4)
    assert sorted(p.slots.row(s) for s in got) == [1, 2, 3]
    keys, _ = store.list_meta("ns/done/")
    dropped = [k for k, _ in keys if k.endswith(".dropped")]
    assert len(dropped) == 1
    body = json.loads(store.read(dropped[0]))
    assert body["reason"].startswith("validate_failed: h1:oracle_reward=0.0") and body["task_id"] == "t0"
    p.written(0, got)
    with pytest.raises(DispenseEnded):
        p.take(4)


def test_no_claim_when_the_host_is_full(tmp_path):
    store = sd.DirObjects(tmp_path / "store")
    _manifest(store, {f"t{i}": {"h1": True} for i in range(4)})
    gate = _gate(store, max_sandboxes=10, running=10, group_size=2)
    p = _prompts(tmp_path, store, gate)
    events = []
    p.emit = lambda e, **f: events.append(e)
    calls = {"n": 0}

    def sleep(_s):  # the host frees room for one group of 2 after two waits
        calls["n"] += 1
        if calls["n"] == 2:
            gate.docker.n = 8

    p._sleep = sleep
    got = p.take(2)
    assert len(got) == 1 and calls["n"] >= 2
    assert len(_claimed(store)) == 1  # nothing was claimed while the host was full
    assert "rl_dispense_wait_capacity" in events


def test_process_cap_limits_the_claim(tmp_path):
    from yeto.rl.harness.harbor.capacity import SandboxLimits

    store = sd.DirObjects(tmp_path / "store")
    _manifest(store, {f"t{i}": {"h1": True} for i in range(4)})
    gate = _gate(store, group_size=2)
    gate.provider, gate._limits = "remote-docker", SandboxLimits({"remote-docker": 5})
    p = _prompts(tmp_path, store, gate, groups=4)
    assert len(p.take(4)) == 2  # 5 sandboxes // 2 per group


def test_affinity_strict_and_prefer(tmp_path):
    store = sd.DirObjects(tmp_path / "store")
    _manifest(store, {"t0": {"h2": True}, "t1": {"h1": True}, "t2": {"h2": True}, "t3": {"h1": True}})
    p = _prompts(tmp_path, store, _gate(store), groups=4)
    assert sorted(p.slots.row(s) for s in p.take(4)) == [1, 3]
    store2 = sd.DirObjects(tmp_path / "store2")
    _manifest(store2, {"t0": {"h2": True}, "t1": {"h1": True}, "t2": {"h2": True}, "t3": {"h1": True}})
    p2 = _prompts(tmp_path, store2, _gate(store2, affinity="prefer"), groups=2)
    assert sorted(p2.slots.row(s) for s in p2.take(2)) == [1, 3]  # own host first
    assert sorted(p2.slots.row(s) for s in p2.take(2)) == [0, 2]  # then portable tasks ready elsewhere


def test_gate_does_not_change_the_plan(tmp_path):
    store = sd.DirObjects(tmp_path / "a")
    _prompts(tmp_path, store, None)
    plain = store.read("ns/plan.json")
    store2 = sd.DirObjects(tmp_path / "b")
    _manifest(store2, {"t0": {"h1": True}})
    _prompts(tmp_path, store2, _gate(store2))
    assert store2.read("ns/plan.json") == plain


def test_gate_from_spec_finds_the_host(tmp_path):
    spec = {"sampling": {"n": 8}, "agentic": {"task_defaults": {"provider": "remote-docker"}, "envs": {
        "run": "r1", "hosts": [{"name": "h1", "docker_host": "ssh://a@h1", "max_sandboxes": 64},
                               {"name": "h2", "docker_host": "ssh://a@h2"}]}}}
    store = sd.DirObjects(tmp_path)
    g = gate_from_spec(spec, store, environ={"HARBOR_ENV_KWARGS": '{"docker_host": "ssh://a@h2"}'})
    assert g.host_name == "h2" and g.group_size == 8 and g.provider == "remote-docker" and g.docker is None
    from yeto.rl.harness.harbor.capacity import SandboxLimits

    g = gate_from_spec(spec, store, environ={"YETO_SANDBOX_HOST": "h1"},
                       docker_factory=lambda h: FixedRoom(60), limits=SandboxLimits({"remote-docker": 32}))
    assert g.host_name == "h1" and g.free_groups() == 0  # (64 - 60) // 8
    assert gate_from_spec({"agentic": {}}, store) is None


def test_prepare_lists_the_task_images_and_checks_the_disk_first(world):
    report = _prepare(world)
    images = report["host_facts"]["h1"]["images"]
    # the task images and the prefetch image are all missing before the first prepare
    assert images["missing"] == images["images"] >= 5 and "runner:cg-1-vul" in images["missing_refs"]
    assert "warning" not in images  # 600 GB free on the fake disk
    again = _prepare(world)["host_facts"]["h1"]["images"]
    assert again["missing"] == 0  # every image pulled or built once; a second prepare pulls nothing
    st = ops.status(world["cfg"], world["store"], factory=world["factory"])["hosts"]["h1"]
    assert st["images_missing"] == []


def test_a_task_image_pull_is_retried(world):
    d = world["daemon"]
    fails = {"left": 2}

    def flaky(argv, env, stdin):
        if list(argv)[1:3] == ["pull", "hub/tb:b"] and fails["left"]:
            fails["left"] -= 1
            return 1, "", "toomanyrequests: rate limit"
        return d(argv, env, stdin)

    world["factory"] = lambda h: DockerHost(h.docker_host, ssh=h.ssh, runner=flaky)
    _prepare(world)
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["tasks"]["tb_b"]["hosts"]["h1"]["prepared"] and fails["left"] == 0


def _ssh_255(argv, env, stdin):
    return 255, "", "ssh: connect to host h1 port 22: Connection refused"


def test_gate_docker_check_fails_with_the_fix_when_ssh_is_refused(tmp_path):
    from yeto.sandbox.docker import DockerError

    spec = {"sampling": {"n": 8}, "agentic": {"envs": {
        "run": "r1", "hosts": [{"name": "h1", "docker_host": "ssh://a@h1", "max_sandboxes": 64}]}}}
    g = gate_from_spec(spec, sd.DirObjects(tmp_path), environ={},
                       docker_factory=lambda h: DockerHost(h.docker_host, runner=_ssh_255))
    with pytest.raises(DockerError, match=r"h1.*unreachable.*rc=255.*DOCKER_HOST=ssh://a@h1 docker info"):
        g.check_docker()
    ok = gate_from_spec(spec, sd.DirObjects(tmp_path), environ={},
                        docker_factory=lambda h: DockerHost(h.docker_host, runner=FakeDaemon()))
    ok.check_docker()  # a reachable daemon passes


def test_sampler_start_exits_when_the_sandbox_docker_is_unreachable(tmp_path, monkeypatch):
    import yeto.sandbox.gate as gate_mod
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.task_dispense import SAMPLER_INCARNATION_ENV

    monkeypatch.setattr(gate_mod, "DockerHost", lambda dh: DockerHost(dh, runner=_ssh_255))
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps({"text": f"q{i}", "label": "a"}) + "\n" for i in range(2)))
    spec = {"sampler_id": "nc-0", "store": str(tmp_path / "store"),
            "prompts": {"path": str(path), "input_key": "text", "label_key": "label", "groups_per_round": 2,
                        "dispense": {"backend": "store", "namespace": "task-dispense"}},
            "agentic": {"envs": {"run": "r1",
                                 "hosts": [{"name": "h1", "docker_host": "ssh://a@h1", "max_sandboxes": 64}]}}}
    with pytest.raises(entry.SamplerSpecError, match="unreachable.*docker info"):
        entry.store_dispensed_prompts(spec, environ={SAMPLER_INCARNATION_ENV: "inc-1"})


def test_teardown_unreachable_host_is_reported_and_the_other_hosts_are_torn_down(world):
    """One sandbox host down must not abort ``yeto sandbox teardown``: the reachable
    hosts are still cleaned, their ready state is still written, and the
    unreachable host is named (exit code 1 from the CLI)."""
    section = dict(world["section"], hosts=[{"name": "h0", "docker_host": "ssh://u@h0"}] + world["section"]["hosts"])
    _prepare(world)
    cfg = parse_envs(section)
    good = world["daemon"]

    def factory(h):
        if h.name == "h0":
            return DockerHost(h.docker_host, runner=lambda a, e, s: (255, "", "ssh: connect refused"))
        return DockerHost(h.docker_host, ssh=h.ssh, runner=good)

    out = ops.teardown(cfg, world["store"], factory=factory)
    assert "connect refused" in out["h0"]["error"]
    assert len(out["h1"]["containers"]) == 2
    assert not any(c["Config"]["Labels"].get(LABEL_RUN) == "r1" for c in good.containers.values())
    doc = ready_mod.read_manifest(world["store"], "sandbox/ready.json")
    assert doc["tasks"]["cg_1"]["hosts"]["h1"]["prepared"] is False
