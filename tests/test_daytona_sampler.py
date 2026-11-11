"""The no-docker sampler route of a Daytona GPU sandbox, driven by a fake exec handle."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from yeto.launch import daytona_sampler as ds
from yeto.launch.nc import elastic

GOOD_FACTS = {
    "kfd": "rw", "dri": "8", "driver": "6.16.0 6.8.0", "rocm_image": "10.0.0", "shm_kib": str(800 << 20),
    "mem_kib": str(1500 << 20), "memlock": "unlimited", f"free_kib:{ds.WORK}": str(3000 << 20),
    f"free_kib:{ds.MODELS}": str(3000 << 20), f"write:{ds.WORK}": "ok", f"write:{ds.MODELS}": "ok",
    "python": "ok", "aws": "ok", "docker_cli": "yes", "proxy_ca": "no", "vllm": ds.VLLM_SAMPLER_ROCM_VERSION,
    "store": "403",
}
GOOD_PY = {"missing": [], "torch": "2.13.0+rocm10.0.0", "hip": "10.0", "available": True, "count": 8,
           "arch": ["gfx950"], "names": ["AMD Instinct MI355X"]}


def report(facts=None, py=None) -> str:
    lines = [f"YETO_DAYTONA_FACT {k} {v}" for k, v in (GOOD_FACTS if facts is None else facts).items()]
    return "\n".join([*lines, "YETO_DAYTONA_PY " + json.dumps(GOOD_PY if py is None else py)]) + "\n"


class FakeBox:
    """Records every command and upload; answers the preflight with ``facts`` / ``py``."""

    def __init__(self, *, facts=None, py=None, started=True):
        self.name = "sbx-1"
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.facts, self.py, self.started = facts, py, started

    def exec(self, command: str, *, timeout: float = 600) -> ds.ExecResult:
        self.commands.append(command)
        if "YETO_DAYTONA_FACT" in command:
            return ds.ExecResult(0, report(self.facts, self.py))
        if "setsid bash -c" in command and ds.MARKER in command:
            return ds.ExecResult(0, "YETO_DAYTONA_STARTED 4242\n" if self.started else "YETO_DAYTONA_START_FAILED\n")
        if command.endswith("/yeto"):  # the code-tree test: absent
            return ds.ExecResult(1)
        return ds.ExecResult(0, "")

    def upload(self, data: bytes, remote: str) -> None:
        self.uploads[remote] = data


SPEC = {"sampler_id": "k31-0", "engine": {"adapter": "vllm-server", "server": {"in_place_reload": True}},
        "event_tape": "/work/sampler/events-k31-0.jsonl"}


def test_preflight_passes_on_a_good_sandbox():
    checks = ds.preflight(FakeBox(), ds.Requirements(data_gib=1000))
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert {"kfd", "dri", "rocm", "gpus", "arch", "shm", "memory", "disk", "store"} <= {c.name for c in checks}


@pytest.mark.parametrize("facts, py, failed", [
    ({"kfd": "absent"}, {}, "kfd"),
    ({"dri": "4"}, {}, "dri"),
    ({"shm_kib": str(8 << 20)}, {}, "shm"),
    ({"mem_kib": str(100 << 20)}, {}, "memory"),
    ({f"free_kib:{ds.WORK}": str(400 << 20), f"free_kib:{ds.MODELS}": str(400 << 20)}, {}, "disk"),
    ({f"write:{ds.MODELS}": "no"}, {}, f"write:{ds.MODELS}"),
    ({}, {"arch": ["gfx942"]}, "arch"),
    ({}, {"count": 4}, "gpus"),
    ({}, {"available": False}, "rocm"),
    ({}, {"missing": ["boto3"]}, "modules"),
    ({"aws": "absent"}, {}, "aws"),
    ({"store": "000"}, {}, "store"),
])
def test_preflight_names_the_failed_check_and_its_remedy(facts, py, failed):
    box = FakeBox(facts={**GOOD_FACTS, **facts}, py={**GOOD_PY, **py})
    with pytest.raises(ds.PreflightError) as err:
        ds.preflight(box, ds.Requirements(data_gib=1000))
    names = [c.name for c in err.value.failed]
    assert names == [failed]
    assert ds.REMEDY[failed.split(":")[0]] in str(err.value)


def test_preflight_warns_only_on_vllm_memlock_and_arch_without_hot_reload():
    box = FakeBox(facts={**GOOD_FACTS, "vllm": "0.33.0", "memlock": "65536"}, py={**GOOD_PY, "arch": ["gfx942"]})
    checks = ds.preflight(box, ds.Requirements(hot_reload=False))
    assert {c.name for c in checks if not c.ok} == {"vllm", "memlock", "arch"}
    assert all(c.warn for c in checks if not c.ok)


def test_preflight_without_a_report_fails_clearly():
    class Dead(FakeBox):
        def exec(self, command, *, timeout=600):
            return ds.ExecResult(255, "", "Permission denied")
    with pytest.raises(ds.PreflightError, match="Permission denied"):
        ds.preflight(Dead(), ds.Requirements())


def test_preflight_script_is_read_only_and_prints_no_secret():
    script = ds.preflight_script(ds.Requirements(ssh_hosts=("203.0.113.5:22",)))
    assert "docker" not in script.replace("docker_cli", "").replace("command -v docker", "")
    assert "AWS_SECRET_ACCESS_KEY" not in script and "echo $AWS" not in script
    assert "/dev/tcp/203.0.113.5/22" in script and ds.STORE_ENV in script


def test_requirements_of_reads_hot_reload_and_ssh_hosts(tmp_path):
    key = tmp_path / "k"
    key.write_text("x")
    spec = {**SPEC, "agentic": {"proxy": {"host": "127.0.0.1", "port": 18080}, "setup": {
        "sandbox": {"docker_host": "ssh://u@198.51.100.7:2222", "key": {"path": str(key)}},
        "tunnel": {"listen": "10.0.0.2:19000", "user": "t", "host": "198.51.100.7", "port": 22,
                   "key": {"path": str(key)}}, "records_secret": "generate"}}}
    req = ds.requirements_of(spec, plan=ds.k31_disk_plan(), shm_gib=32.0)
    assert req.hot_reload and req.shm_gib == 32.0
    assert req.ssh_hosts == ("198.51.100.7:2222", "198.51.100.7:22")
    assert req.data_gib == ds.k31_disk_plan().data_gib


def test_disk_plan_for_k31():
    plan = ds.k31_disk_plan()
    assert plan.required_gib == 1029  # (500 + 2 x 103 + 60 + 64 + 64) x 1.15
    assert plan.data_gib == 834
    assert ds.k31_disk_plan(base_gib=1560).required_gib == 2248
    assert plan.required_gib <= 8 * 512  # Daytona's per-GPU disk ceiling


def test_start_without_docker(tmp_path):
    env_file = tmp_path / "r2.env"
    env_file.write_text("AWS_SECRET_ACCESS_KEY=s3cr3t\nR2_BUCKET=b\n")
    box = FakeBox()
    out = ds.start_sampler(box, spec=SPEC, shard="k31-0", run_prefix="/runs-x/", r2_env_file=env_file,
                           code=("abc123", b"TAR"))
    joined = "\n".join(box.commands)
    assert "docker run" not in joined and "docker inspect" not in joined and "docker stop" not in joined
    assert box.uploads[ds.STORE_ENV + ".up"] == env_file.read_bytes()
    assert "s3cr3t" not in joined  # the secret travels as an upload, never in a command
    script = box.uploads[out.paths["script"] + ".up"].decode()
    assert "s3cr3t" not in script and f". {ds.STORE_ENV}" in script
    assert "export PYTHONPATH=/work/yeto/abc123" in script
    assert "export YETO_NC_RUN_PREFIX=runs-x" in script
    assert "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in script
    assert script.rstrip().endswith("-m yeto.rl.engine.engine_sampler --spec /work/sampler/k31-0.json")
    spec = json.loads(box.uploads["/work/sampler/k31-0.json.up"])
    assert spec["start_on_latest"] is True and spec["sampler_id"] == "k31-0"
    assert box.uploads["/work/yeto/abc123.tar"] == b"TAR"
    # order: preflight before the code upload, an old sampler stopped before the launch
    pre = next(i for i, c in enumerate(box.commands) if "YETO_DAYTONA_FACT" in c)
    untar = next(i for i, c in enumerate(box.commands) if "tar xf /work/yeto/abc123.tar" in c)
    stop = next(i for i, c in enumerate(box.commands) if "kill -TERM -- -$p" in c)
    launch = next(i for i, c in enumerate(box.commands) if "setsid bash -c" in c)
    assert pre < untar < stop < launch


def test_start_refusals(tmp_path):
    env_file = tmp_path / "r2.env"
    env_file.write_text("R2_BUCKET=b\n")
    kw = dict(r2_env_file=env_file, code=("abc", b""))
    with pytest.raises(ValueError, match="protected"):
        ds.start_sampler(FakeBox(), spec=SPEC, shard="k31-0", run_prefix="p", protected=frozenset({"p"}), **kw)
    with pytest.raises(ValueError, match="not the shard"):
        ds.start_sampler(FakeBox(), spec=SPEC, shard="k31-1", run_prefix="p", **kw)
    with pytest.raises(ds.PreflightError):
        ds.start_sampler(FakeBox(facts={**GOOD_FACTS, "kfd": "absent"}), spec=SPEC, shard="k31-0",
                         run_prefix="p", **kw)
    with pytest.raises(RuntimeError, match="did not start"):
        ds.start_sampler(FakeBox(started=False), spec=SPEC, shard="k31-0", run_prefix="p", **kw)


def test_process_lifecycle_commands_never_match_by_name():
    for cmd in (ds.launch_command("s"), ds.stop_command("s"), ds.state_command("s")):
        assert "pkill" not in cmd and "pgrep" not in cmd and "killall" not in cmd
    assert "kill -TERM -- -$p" in ds.stop_command("s") and "kill -KILL -- -$p" in ds.stop_command("s")
    assert ds.MARKER in ds.launch_command("s") and "/proc/$p/cmdline" in ds.state_command("s")


def test_process_lifecycle_runs_in_a_real_shell(tmp_path, monkeypatch):
    """launch / state / stop against /bin/bash with the paths moved under tmp_path."""
    if not (__import__("pathlib").Path("/proc/self/cmdline").exists()):
        pytest.skip("needs /proc")
    work = tmp_path / "work"
    monkeypatch.setattr(ds, "SAMPLER_DIR", str(work / "sampler"))
    monkeypatch.setattr(ds, "LOG_DIR", str(work / "logs"))
    (work / "sampler").mkdir(parents=True)
    script = work / "sampler" / "s.start.sh"
    script.write_text("#!/bin/bash\necho hello\nsleep 30\n")
    run = lambda c: subprocess.run(["bash", "-c", c], capture_output=True, text=True, timeout=60)  # noqa: E731
    assert "YETO_DAYTONA_STARTED" in run(ds.launch_command("s")).stdout
    assert run(ds.state_command("s")).stdout.strip() == "running -"
    run(ds.stop_command("s", grace_s=5))
    assert run(ds.state_command("s")).stdout.split()[0] == "exited"


def test_probe_script_parses_like_the_container_probe(tmp_path, monkeypatch):
    script = ds.probe_script("k31-0", "/work/sampler/events-k31-0*.jsonl", metrics_port=8000)
    assert "docker" not in script and "YETO_NC_PROBE" in script and "127.0.0.1:8000/metrics" in script
    # a sampler that exited 75 (reclaim) and wrote its JOIN to the log, probed in a real shell
    monkeypatch.setattr(ds, "SAMPLER_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "LOG_DIR", str(tmp_path))
    (tmp_path / "k31-0.exit").write_text("75\n")
    (tmp_path / "k31-0.log").write_text("rl_sampler_join\n")
    script = ds.probe_script("k31-0", f"{tmp_path}/events-k31-0*.jsonl")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    probe = elastic.Probe.parse(out.stdout, out.returncode)
    assert (probe.node, probe.sampler, probe.exit_code, probe.joined) == ("ok", "exited", 75, True)
    # the container route is unchanged
    assert "sudo docker inspect" in elastic.probe_script("yeto-sampler-k31-0", "/x*.jsonl")


def test_pull_command_runs_the_nc_body_without_a_container():
    cmd = ds.pull_command("models/k31", max_mb_s=2000)
    assert "docker" not in cmd and "aws s3 cp" in cmd and ds.MODEL_DIR in cmd and f". {ds.STORE_ENV}" in cmd
    with pytest.raises(ValueError, match="under /models"):
        ds.pull_command("models/k31", "/work/x")
    with pytest.raises(ValueError, match="3000"):
        ds.pull_command("models/k31", max_mb_s=4000)


def test_process_provisioner_has_no_docker(tmp_path):
    from yeto.launch import agentic_setup

    key = tmp_path / "k"
    key.write_text("-----KEY-----")
    spec = {**SPEC, "agentic": {"proxy": {"host": "127.0.0.1", "port": 18080}, "setup": {
        "sandbox": {"docker_host": "ssh://u@198.51.100.7", "key": {"path": str(key)}},
        "tunnel": {"listen": "10.0.0.2:19000", "user": "t", "host": "198.51.100.7", "key": {"path": str(key)}},
        "tasks": {"source": "store:tasks/tb", "ids": ["a"]}, "records_secret": "generate"}}}
    setup = agentic_setup.parse(spec)
    box = FakeBox()
    prov = ds.ProcessProvisioner(setup, ds.ProcessNode(box, yeto_dir="/work/yeto/abc"),
                                 store_env_file=ds.STORE_ENV)
    assert prov.tunnel_dir == "/tunnel"
    sync = prov._store_sync_command("/work/harbor/tasks", "aws s3 sync x /work/harbor/tasks/a")
    assert "docker" not in sync and f". {ds.STORE_ENV}" in sync
    prov._tunnel_start()
    start = box.commands[-1]
    assert "docker" not in start and "setsid" in start and "yeto.launch.agentic_node_agent tunnel" in start
    assert "-R 10.0.0.2:19000:127.0.0.1:18080" in start
    assert ds.ProcessNode(box, yeto_dir="/y").host("/work/harbor") == "/work/harbor"


def test_ssh_exec_wraps_commands_and_uploads_by_stdin():
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    box = ds.SshExec("sbx", ["alias-1"], run=run)
    assert box.exec("echo hi").stdout == "ok"
    assert calls[0][0][-2:] == ["alias-1", "bash -c 'echo hi'"]
    box.upload(b"DATA", "/work/x")
    assert calls[1][1]["input"] == b"DATA" and "cat > /work/x.part" in calls[1][0][-1]


def test_pool_uses_the_process_route(monkeypatch, tmp_path):
    from yeto.launch.nc import cli

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    ops = cli.NcSshOps.__new__(cli.NcSshOps)
    ops.cfg, ops.provider, ops._run = {"sampler_route": "process", "spec": str(spec), "run_prefix": "r"}, None, None
    assert isinstance(ops.sampler_route, ds.DaytonaSamplerRoute)
    assert ops._tape_glob("k31-0") == "/work/sampler/events-k31-0*.jsonl"
    ops2 = cli.NcSshOps.__new__(cli.NcSshOps)
    ops2.cfg, ops2.provider = {"sampler_route": "bogus"}, None
    with pytest.raises(cli.NcError, match="sampler_route"):
        _ = ops2.sampler_route
    route = object()
    ops3 = cli.NcSshOps.__new__(cli.NcSshOps)
    ops3.cfg, ops3.provider = {}, SimpleNamespace(sampler_route=route)
    assert ops3.sampler_route is route
    ops4 = cli.NcSshOps.__new__(cli.NcSshOps)
    ops4.cfg, ops4.provider = {}, None
    assert ops4.sampler_route is None
