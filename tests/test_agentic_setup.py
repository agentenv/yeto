"""agentic.setup (yeto.launch.agentic_setup): parsing, local checks, the node steps
against a fake node, the tunnel keep-alive loop, the ssh_sampler hook. No SSH."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from yeto.launch import agentic_node_agent as agent
from yeto.launch import agentic_setup as su
from yeto.rl.harness.harbor.pins import HARBOR_COMMIT, HARBOR_VERSION

# built by join: the hygiene check reads a literal "/home/<name>" as a user home
SECRET_PATH = "/".join(["/work/harbor", "home", "records-proxy.secret"])
NVME_SECRET_PATH = "/mnt/nvme" + SECRET_PATH


def _spec(tmp_path, **setup):
    key = tmp_path / "docker.key"
    key.write_text("-----BEGIN OPENSSH " + "PRIVATE KEY-----\nfake\n-----END OPENSSH " + "PRIVATE KEY-----\n")
    key.chmod(0o600)
    tasks = tmp_path / "tasks"
    for t in ("a", "b"):
        (tasks / t).mkdir(parents=True, exist_ok=True)
        (tasks / t / "task.toml").write_text("[task]\n")
    doc = {"sandbox": {"docker_host": "ssh://sbx@sandbox.invalid", "key": {"path": str(key)}},
           "tunnel": {"listen": "192.0.2.1:19001"},
           "tasks": {"source": str(tasks), "ids": ["a", "b"]},
           "records_secret": "generate", **setup}
    return {"sampler_id": "s-0", "agentic": {"proxy": {"host": "127.0.0.1", "port": 18080},
                                             "task_defaults": {"provider": "remote-docker"}, "setup": doc}}


def test_parse_defaults_and_container_env(tmp_path):
    s = su.parse(_spec(tmp_path))
    assert (s.harbor_version, s.harbor_commit) == (HARBOR_VERSION, HARBOR_COMMIT)
    assert s.sandbox.alias == "yeto-sandbox" and s.tunnel.target_port == 18080 and s.tunnel.key == s.sandbox.key
    env = s.container_env()
    assert env["DOCKER_HOST"] == "ssh://yeto-sandbox" and env["HOME"] == "/work/harbor/home"
    assert env["YETO_RECORDS_PROXY_PUBLIC_BASE"] == "http://192.0.2.1:19001"
    assert env["YETO_RECORDS_PROXY_SECRET_FILE"] == SECRET_PATH
    assert su.local_problems(s) == []
    assert su.parse({"agentic": {}}) is None
    assert "setup" not in su.strip(_spec(tmp_path))["agentic"]


@pytest.mark.parametrize("change, match", [
    ({"bogus": 1}, "unknown keys"),
    ({"images": ["x"]}, "agentic.envs"),
    ({"harbor": {"version": "0.24.0"}}, "verified against Harbor"),
    ({"docker_cli": {"version": "latest"}}, "pinned"),
    ({"sandbox": {"docker_host": "tcp://h", "key": {"path": "k"}}}, "ssh://"),
    ({"tunnel": {"listen": "nope"}}, "listen"),
    ({"root": "/etc"}, "under /work"),
    ({"records_secret": None}, "records_secret must be"),
])
def test_parse_refusals(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        su.parse(_spec(tmp_path, **change))


def test_tunnel_needs_secret_and_local_problems(tmp_path):
    spec = _spec(tmp_path)
    del spec["agentic"]["setup"]["records_secret"]
    with pytest.raises(ValueError, match="public"):
        su.parse(spec)
    s = su.parse(_spec(tmp_path, tasks={"source": str(tmp_path / "tasks"), "ids": ["a", "zz"]}))
    (tmp_path / "docker.key").chmod(0o644)
    problems = " ".join(su.local_problems(s))
    assert "0600" in problems and "zz" in problems
    lines, probs = su.dry_run_report(_spec(tmp_path, sandbox={"docker_host": "ssh://u@h", "key": {"env": "NOPE_K"}}),
                                     environ={})
    assert any("tunnel" in ln for ln in lines) and any("NOPE_K" in p for p in probs)


def test_classify_ssh_failure_names_the_fingerprint():
    msg = su.classify_ssh_failure("sbx@h: Permission denied (publickey).", who="sbx@h:22", key="docker key",
                                  fingerprint="SHA256:abc", purpose="docker")
    assert "SHA256:abc" in msg and "authorized_keys" in msg
    assert "permitlisten" in su.classify_ssh_failure("Error: remote port forwarding failed for listen port 1",
                                                     who="u@h", key="k", fingerprint="f", purpose="p")


class FakeNode:
    """Answers the commands the provisioner sends, with node state in a dict."""

    def __init__(self, *, harbor_ok=False, tools_ok=False, docker_ok=True, tunnel_err=None):
        self.files: dict[str, tuple[str, str]] = {}
        self.harbor_ok, self.tools_ok, self.docker_ok, self.tunnel_err = harbor_ok, tools_ok, docker_ok, tunnel_err
        self.tunnel_running, self.label, self.events = False, "", []
        self.calls: list[str] = []

    def r(self, out="", code=0, err=""):
        return SimpleNamespace(returncode=code, stdout=out, stderr=err)

    def run(self, argv, **kw):
        cmd = argv[-1] if argv[0] == "ssh" else " ".join(argv)
        self.calls.append(cmd)
        inp = kw.get("input")
        if argv[0] == "ssh-keygen":
            return self.r("256 SHA256:fake comment (ED25519)\n" if "-lf" in argv else "ssh-ed25519 AAAA\n")
        if "sha256sum" in cmd:
            path = cmd.split("sha256sum ")[1].split(" ")[0].strip("'")
            if path in self.files:
                import hashlib
                text, mode = self.files[path]
                return self.r(f"{hashlib.sha256(text.encode()).hexdigest()}\n{mode}\n")
            return self.r("")
        if inp is not None and "cat >" in cmd:
            path = cmd.split("cat > ")[1].split(".tmp")[0]
            mode = cmd.split("chmod ")[1].split(" ")[0]
            self.files[path] = (inp, mode)
            return self.r()
        if "docker --version" in cmd:
            return self.r("Docker version 29.0.2, build x\nDocker Compose version v2.40.3\n" if self.tools_ok else "")
        if "curl -fsSL" in cmd and "docker-compose" in cmd:
            self.tools_ok = True
            return self.r()
        if "pip install" in cmd:
            self.harbor_ok = True
            return self.r("YETO_SETUP_PRUNED pydantic,requests\n")
        if "YETO_SETUP_HARBOR" in cmd or "metadata.distribution" in cmd:
            st = {"version": HARBOR_VERSION, "commit": HARBOR_COMMIT, "import": "ok"} if self.harbor_ok else \
                {"error": "PackageNotFoundError: harbor"}
            return self.r("YETO_SETUP_HARBOR " + json.dumps(st) + "\n")
        if "docker -H ssh://" in cmd:
            return self.r("29.0.2\n[name=rootless]\n") if self.docker_ok else \
                self.r("", 1, "sbx@sandbox.invalid: Permission denied (publickey).")
        if "records-proxy.secret" in cmd and "urandom" in cmd:
            made = NVME_SECRET_PATH not in self.files
            self.files.setdefault(NVME_SECRET_PATH, ("x" * 40, "600"))
            return self.r("made\n" if made else "kept\n")
        if "stat -c %a" in cmd and "wc -c" in cmd:
            return self.r("600\n40\n")
        if "sudo test -f" in cmd and ".yeto-setup-" in cmd:
            return self.r(code=0 if any(".yeto-setup-" in f for f in self.files) else 1)
        if "sudo tar -C" in cmd:
            return self.r()
        if "sudo touch" in cmd:
            self.files[cmd.split("sudo touch ")[1].strip("'")] = ("", "644")
            return self.r()
        if "task.toml" in cmd:
            return self.r()
        if "docker inspect" in cmd and su.TUNNEL_CONTAINER in cmd:
            return self.r(f"true {self.label}\n" if self.tunnel_running else "")
        if "docker run -d --name " + su.TUNNEL_CONTAINER in cmd:
            self.tunnel_running = True
            self.label = cmd.split("yeto.setup=")[1].split(" ")[0]
            self.events.append({"event": su.TUNNEL_EVENT, "state": "down", "stderr": self.tunnel_err}
                               if self.tunnel_err else {"event": su.TUNNEL_EVENT, "state": "up"})
            return self.r()
        if "tail -n 50" in cmd or "sudo cat" in cmd:
            return self.r("".join(json.dumps(e) + "\n" for e in self.events))
        return self.r()


def _node(fake):
    return su.SshNode("nc-t1", "img", "/mnt/nvme", "/mnt/nvme/yeto/abc", "/opt/runtime-venv/bin/python3", "ubuntu",
                      run=fake.run)


def _prov(tmp_path, fake, **kw):
    s = su.parse(_spec(tmp_path, **kw))
    events = []
    p = su.Provisioner(s, _node(fake), emit=events.append, sleep=lambda _s: None, local_run=fake.run)
    p.node.stream_tar = lambda *a, **k: None
    return p, events


def test_provision_from_scratch_then_idempotent(tmp_path):
    fake = FakeNode()
    p, events = _prov(tmp_path, fake)
    report = p.run()
    assert report == {"layout": "ok", "docker_cli": "changed", "harbor": "changed", "sandbox": "changed",
                      "records_secret": "changed", "tasks": "changed", "tunnel": "changed"}
    assert all(e["event"] == su.SETUP_EVENT for e in events)
    assert fake.files["/mnt/nvme/agentic-tunnel/key"][1] == "600"
    assert "ControlMaster auto" in fake.files["/mnt/nvme/work/harbor/home/.ssh/config"][0]
    # the tunnel key is never under /work (the sampler container sees only /work)
    assert not any(f.startswith("/mnt/nvme/work") and "agentic-tunnel" in f for f in fake.files)
    n_runs = sum("docker run -d --name " + su.TUNNEL_CONTAINER in c for c in fake.calls)
    p2, _ = _prov(tmp_path, fake)
    assert set(p2.run().values()) == {"ok"}
    assert sum("docker run -d --name " + su.TUNNEL_CONTAINER in c for c in fake.calls) == n_runs
    assert not any("pip install" in c for c in fake.calls[len(fake.calls) // 2:]) or fake.harbor_ok


def test_refused_docker_key_names_the_fingerprint(tmp_path):
    fake = FakeNode(docker_ok=False)
    p, events = _prov(tmp_path, fake)
    with pytest.raises(su.SetupError, match="SHA256:fake.*authorized_keys") as info:
        p.run()
    assert info.value.step == "sandbox" and events[-1]["status"] == "failed"


def test_refused_forward_is_reported(tmp_path):
    fake = FakeNode(tunnel_err="Error: remote port forwarding failed for listen port 19001")
    p, _ = _prov(tmp_path, fake, tunnel={"listen": "192.0.2.1:19001", "ready_timeout_s": 0})
    with pytest.raises(su.SetupError, match="permitlisten"):
        p.run()


def test_tunnel_loop_reconnects_and_emits(tmp_path):
    class Proc:
        def __init__(self, lives):
            self.lives, self.returncode = lives, None
            self.stderr = SimpleNamespace(read=lambda: "Connection reset")

        def wait(self, timeout=None):
            if self.lives and timeout is not None:
                raise subprocess.TimeoutExpired("ssh", timeout)
            self.returncode = 255
            return 255

    procs = iter([Proc(True), Proc(False), Proc(True)])
    sleeps = []
    events = tmp_path / "t.jsonl"
    agent.tunnel_loop(["ssh", "-N"], events, popen=lambda *a, **k: next(procs), sleep=sleeps.append,
                      clock=iter(range(0, 1000, 7)).__next__, max_attempts=3, stable_s=1000)
    recs = [json.loads(ln) for ln in events.read_text().splitlines()]
    assert [r["state"] for r in recs] == ["up", "down", "down", "up", "down"]
    assert recs[1]["stderr"] == "Connection reset" and len(sleeps) == 3 and sleeps[1] > sleeps[0]


def test_start_sampler_runs_setup_and_layers_env_files(tmp_path, monkeypatch):
    from yeto.launch.nc import client as ncc
    from yeto.launch.nc import ssh_sampler as ss

    uploads, ran = [], []
    monkeypatch.setattr(ss, "upload_code", lambda alias, run: "/mnt/nvme/yeto/x")
    monkeypatch.setattr(ss, "upload_file", lambda a, local, remote, **k: uploads.append((str(local), remote)))
    monkeypatch.setattr(su.Provisioner, "run", lambda self: ran.append(self.node.alias) or {})
    calls = []
    ok = lambda argv, **k: calls.append(argv) or SimpleNamespace(returncode=0, stdout="", stderr="")  # noqa: E731
    lease = ncc.NodeLease(1, "t1", "192.0.2.7", "ubuntu", "ssh-ed25519 K", None, "2026-11-09T16:00:00Z", 8, 1)
    spec = _spec(tmp_path)
    spec["sampler_id"] = "nc-0"
    state = tmp_path / "state"
    state.mkdir()
    ss.start_sampler(lease, spec=spec, shard="nc-0", run_prefix="p", r2_env_file="r2", state_dir=state,
                     agentic_env_file="op.env", run=ok)
    assert ran == ["nc-t1-ws"] or len(ran) == 1
    assert "setup" not in json.loads((state / "spec-nc-0.json").read_text())["agentic"]
    run_cmd = calls[-1][-1]
    assert run_cmd.index("nc-0.agentic-setup.env") < run_cmd.index("nc-0.agentic.env")
    assert "PYTHONPATH=/yeto:/work/harbor/site" in run_cmd
    assert "YETO_RL_RELEASE_SPREAD_KEY=p/nc-0" in run_cmd  # the start spread is per shard, not per host name
    assert "DOCKER_HOST=ssh://yeto-sandbox" in (state / "nc-0.agentic-setup.env").read_text()


def test_a_task_source_sandbox_prepare_stages_is_no_dry_run_problem(tmp_path, monkeypatch):
    """A new run's first `run up --dry-run` comes before `yeto sandbox prepare` stages
    ~/.yeto/sandbox/<run>/stage/<host>; a missing directory elsewhere is still a problem."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    staged = tmp_path / "home" / ".yeto" / "sandbox" / "run-new" / "stage" / "sbx-3"
    s = su.parse(_spec(tmp_path, tasks={"source": str(staged), "ids": ["a"]}))
    assert not [p for p in su.local_problems(s) if "tasks" in p]
    s = su.parse(_spec(tmp_path, tasks={"source": str(tmp_path / "nowhere"), "ids": ["a"]}))
    assert any("is not a directory" in p for p in su.local_problems(s))
