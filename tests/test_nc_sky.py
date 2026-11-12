"""National Compute as sky SSH node pools (yeto.launch.nc): CPU tests with fake tools."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from yeto.launch.nc import client as ncc
from yeto.launch.nc import node_pool, tasks, tunnel
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool.island_pool import GONE, READY, IslandPool


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("NC_MANAGE", "nc-manage")
    monkeypatch.setenv("NC_QUERY", "nc-query")


class FakeNc:
    """nc-manage / nc-query over one account's current ticket."""

    def __init__(self, ticket=None, others=None, balance=100_000_000):
        self.ticket = ticket
        self.others = others or []
        self.balance = balance
        self.calls = []
        self.next_id = "abc123def456"

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw.get("env")))
        if argv[0] == "nc-query":
            t = dict(self.ticket or {"state": "none"}, others=self.others)
            doc = {"ticket": t, "balance": {"balance_ucredits": self.balance}}
        elif argv[1] == "create":
            self.ticket = {"id": self.next_id, "state": "queued"}
            doc = {"ok": True, "created": True, "ticket": dict(self.ticket)}
        elif argv[1] == "delete":
            assert "--all" not in argv
            self.ticket = dict(self.ticket, state="ended", ended_at="2026-11-09T12:00:00Z")
            doc = {"ok": True, "released": True, "ticket": dict(self.ticket)}
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(doc), stderr="")


def _client(fake, account=1):
    return ncc.NcClient(account, ncc.Ledger(tunnel.state_dir() / "tickets.json"), run=fake,
                        sleep=lambda s: None)


def test_create_refuses_open_ticket_and_ignores_others():
    fake = FakeNc(ticket={"id": "old", "state": "active"})
    with pytest.raises(ncc.NcRefused):
        _client(fake).create()
    assert not any(c[0][1:2] == ["create"] for c in fake.calls)
    fake = FakeNc(ticket={"id": "old", "state": "ended"}, others=[{"id": "prod", "state": "active"}])
    c = _client(fake)
    assert c.create(pubkey="ssh-ed25519 AAAA test") == "abc123def456"
    create = next(c for c in fake.calls if c[0][1:2] == ["create"])
    assert create[1]["NC_PUBKEY"] == "ssh-ed25519 AAAA test"
    assert c.ledger.owns(1, "abc123def456")


def test_account_allow_list():
    with pytest.raises(ncc.NcRefused):
        ncc.NcClient(2, ncc.Ledger(tunnel.state_dir() / "t.json"), allowed_accounts={1})


def test_delete_only_own_current_ticket():
    fake = FakeNc(ticket={"id": "prod-ticket", "state": "active"})
    c = _client(fake)
    with pytest.raises(ncc.NcRefused, match="not created by this tool"):
        c.delete("prod-ticket")
    c.ledger.add(1, "mine")
    with pytest.raises(ncc.NcRefused, match="current ticket is prod-ticket"):
        c.delete("mine")
    assert not any(c_[0][1:2] == ["delete"] for c_ in fake.calls)


def test_delete_confirms_ended():
    fake = FakeNc(ticket={"id": "mine", "state": "active"})
    c = _client(fake)
    c.ledger.add(1, "mine")
    out = c.delete("mine")
    assert out["state"] == "ended"
    deletes = [a for a, _ in fake.calls if a[1:2] == ["delete"]]
    assert deletes == [["nc-manage", "delete", "--account", "1", "--yes"]]
    assert not c.ledger.open_rows()


def test_never_all():
    c = _client(FakeNc())
    with pytest.raises(ncc.NcRefused):
        c._call(["nc-manage", "delete", "--all", "--yes"])


def test_wait_active_returns_lease():
    fake = FakeNc()
    c = _client(fake)
    t = c.create()
    fake.ticket = {"id": t, "state": "active", "lease_until": "2026-11-09T16:00:00Z", "gpus": 8,
                   "node": {"public_ip": "192.0.2.7", "ssh_user": "ubuntu", "host_key": "ssh-ed25519 KEY"}}
    lease = c.wait_active(t, log=lambda m: None)
    assert (lease.ip, lease.gpus) == ("192.0.2.7", 8)
    assert lease.lease_until_ts == ncc.parse_time("2026-11-09T16:00:00Z")


# -- tunnel ------------------------------------------------------------------

def test_frames_roundtrip():
    payload = bytes(range(256)) * 300
    frame = tunnel.encode_frame(payload, mask=b"\x01\x02\x03\x04")
    reader = tunnel.FrameReader()
    assert reader.feed(frame[:5]) == []
    assert reader.feed(frame[5:]) == [(2, payload)]


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_relay_end_to_end(tmp_path):
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen()
    eport = echo.getsockname()[1]

    def serve():
        conn, _ = echo.accept()
        while data := conn.recv(65536):
            conn.sendall(data)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    rport = s.getsockname()[1]
    s.close()
    allow = tmp_path / "allow"
    allow.write_text(f"127.0.0.1:{eport}\n")
    relay = subprocess.Popen([shutil.which("node"), str(tunnel.RELAY_JS)], env={"RELAY_PORT": str(rport),
                             "RELAY_ALLOW": str(allow), "PATH": "/usr/bin:/bin"}, stdout=subprocess.PIPE)
    try:
        relay.stdout.readline()
        ep = tunnel.Endpoint(f"http://127.0.0.1:{rport}/user/x/port/{rport}", "gardengate_port=c")
        with pytest.raises(tunnel.HandshakeRefused) as exc:
            tunnel.open_ws(ep, "127.0.0.1:1")
        assert exc.value.status == 403
        sock, rest = tunnel.open_ws(ep, f"127.0.0.1:{eport}")
        blob = b"SSH-2.0-test\r\n" + b"x" * 200_000
        sock.sendall(tunnel.encode_frame(blob))
        reader, got = tunnel.FrameReader(), b""
        for op, data in reader.feed(rest):
            got += data
        while len(got) < len(blob):
            for op, data in reader.feed(sock.recv(65536)):
                got += data
        assert got == blob
        sock.close()
    finally:
        relay.terminate()


def test_relay_start_command_has_allow_list():
    cmd = tunnel.relay_start_command(["192.0.2.7:22"])
    assert "__RELAY_UP__" in cmd and "relay-allow" in cmd and "nohup node" in cmd


# -- node pool ---------------------------------------------------------------

def _lease():
    return ncc.NodeLease(1, "t1", "192.0.2.7", "ubuntu", "ssh-ed25519 HOSTKEY", None,
                         "2026-11-09T16:00:00Z", 8, 3_000_000)


def test_ssh_alias_and_pool(tmp_path):
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Host other\n  HostName 1.2.3.4\n")
    alias = node_pool.write_ssh_alias(_lease(), ssh_config=cfg)
    text = (tunnel.state_dir() / "ssh_config").read_text()
    assert alias == "yeto-nc-t1" and "HostName 192.0.2.7" in text
    assert "ProxyCommand env " in text and "-m yeto.launch.nc proxy --account 1" in text
    assert "StrictHostKeyChecking yes" in text
    assert cfg.read_text().startswith("Include ")
    node_pool.write_ssh_alias(_lease(), ssh_config=cfg)
    assert cfg.read_text().count("Include ") == 1
    assert (tunnel.state_dir() / "known_hosts").read_text() == "192.0.2.7 ssh-ed25519 HOSTKEY\n"
    pools = tmp_path / "pools.yaml"
    pools.write_text(yaml.safe_dump({"onprem": {"hosts": ["a"]}}))
    assert node_pool.write_pool("t1", path=pools) == "nc-t1"
    doc = yaml.safe_load(pools.read_text())
    assert doc == {"onprem": {"hosts": ["a"]}, "nc-t1": {"hosts": ["yeto-nc-t1"]}}
    node_pool.remove_pool("t1", path=pools)
    assert yaml.safe_load(pools.read_text()) == {"onprem": {"hosts": ["a"]}}
    node_pool.remove_ssh_alias("t1")
    assert "yeto-nc-t1" not in (tunnel.state_dir() / "ssh_config").read_text()


def test_host_bootstrap_and_gpu_scripts():
    s = node_pool.host_bootstrap_script()
    for d in ("/var/lib/rancher", "/var/lib/kubelet", "/var/lib/containerd"):
        assert d in s
    assert "a k3s not installed by sky" in s
    assert s.index("a k3s not installed by sky") < s.index("systemctl stop")
    assert "mountpoint -q" in s and "mount --bind" in s and "YETO_NC_BOOTSTRAP_OK" in s
    subprocess.run(["bash", "-n"], input=s, text=True, check=True)
    g = node_pool.amd_gpu_script(8)
    assert "skypilot.co/accelerator=mi355x" in g and "amd.com/gpu" in g
    subprocess.run(["bash", "-n"], input=g, text=True, check=True)


def test_sky_ssh_up_argv():
    seen = []
    node_pool.sky_ssh_up("nc-t1", run=lambda argv, **kw: seen.append(argv))
    assert seen[0][-4:] == ["ssh", "up", "--infra", "nc-t1"]


# -- tasks -------------------------------------------------------------------

def test_hello_and_pull_tasks():
    h = tasks.hello_task("nc-t1")
    assert h["resources"] == {"infra": "ssh/nc-t1", "accelerators": "MI355X:8"}
    vol = h["config"]["ssh"]["pod_config"]["spec"]["volumes"][0]
    assert vol["hostPath"]["path"] == "/mnt/nvme"
    p = tasks.model_pull_task("nc-t1", src="s3://b/models/m", dest="/mnt/nvme/models/m", include="config.json")
    assert "max_bandwidth = 2500MB/s" in p["run"] and "--include config.json" in p["run"]
    with pytest.raises(ValueError, match="exceeds"):
        tasks.model_pull_task("nc-t1", src="s3://b/m", dest="/mnt/nvme/m", max_mb_s=7000)
    with pytest.raises(ValueError, match="dest"):
        tasks.model_pull_task("nc-t1", src="s3://b/m", dest="/root/m")


def test_sampler_task_and_spec(tmp_path):
    spec = tasks.sampler_spec({"sampler_id": "old", "engine": {}}, sampler_id="nc-1", shards=["nc-0"])
    assert spec["sampler_id"] == "nc-1" and spec["buffer_shards"] == ["nc-0", "nc-1"]
    t = tasks.sampler_task("nc-t1", spec_path="/x/spec.json", run_prefix="nc-sky-test", workdir="/repo")
    assert t["resources"]["image_id"] == f"docker:{tasks.SERVING_IMAGE}" and "@sha256:" in tasks.SERVING_IMAGE
    assert "yeto.rl.engine.engine_sampler --spec" in t["run"]
    assert "secrets" not in t and "AWS_SECRET_ACCESS_KEY" not in json.dumps(t)
    with pytest.raises(ValueError, match="protected"):
        tasks.sampler_task("nc-t1", spec_path="/x", run_prefix="live-run", workdir="/r",
                           protected=frozenset({"live-run"}))
    env = tmp_path / "r2.env"
    env.write_text("AWS_ACCESS_KEY_ID=a\nAWS_SECRET_ACCESS_KEY=b\nAWS_ENDPOINT_URL=https://e\nR2_BUCKET=k\nOTHER=z\n")
    s = tasks.with_secrets(t, env)
    assert s["secrets"] == {"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
                            "AWS_ENDPOINT_URL": "https://e", "R2_BUCKET": "k"}
    env.write_text("AWS_ACCESS_KEY_ID=a\n")
    with pytest.raises(ValueError, match="missing"):
        tasks.with_secrets(t, env)


def test_tasks_parse_as_sky_tasks():
    sky = pytest.importorskip("sky")
    for doc in (tasks.hello_task("nc-t1"),
                tasks.model_pull_task("nc-t1", src="s3://b/m", dest="/mnt/nvme/m")):
        task = sky.Task.from_yaml_config(json.loads(json.dumps(doc)))
        (res,) = task.resources
        assert res.accelerators == {"MI355X": 8}


# -- lease pool --------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeOps:
    def __init__(self, clock, lease_s=4 * 3600, ready_after=1800, balance=200_000_000):
        self.clock, self.lease_s, self.ready_after = clock, lease_s, ready_after
        self.bal = balance
        self.tickets = {}  # ticket -> dict(account, granted, drained)
        self.n = 0
        self.events = []

    def order(self, account):
        self.n += 1
        t = f"t{self.n}"
        self.tickets[t] = {"account": account, "granted": self.clock() + 600, "drained": False}
        self.events.append(("order", t, account, self.clock()))
        return t

    def poll(self, m):
        rec = self.tickets[m.ticket]
        if rec["drained"]:
            return "ended", None
        if self.clock() >= rec["granted"]:
            return "active", rec["granted"] + self.lease_s
        return "waiting", None

    def start(self, m, shards):
        self.tickets[m.ticket]["started"] = self.clock()
        self.tickets[m.ticket]["shards"] = list(shards)

    def ready(self, m):
        return self.clock() >= self.tickets[m.ticket]["started"] + self.ready_after

    def drain(self, m):
        self.tickets[m.ticket]["drained"] = True
        self.events.append(("drain", m.ticket, self.clock()))

    def balance(self):
        return self.bal

    def account_free(self, account):
        return not any(r["account"] == account and not r["drained"] for r in self.tickets.values())


def _run(pool, clock, until, step=60):
    while clock.t < until:
        pool.tick()
        clock.t += step


def test_pool_replaces_before_lease_end_on_second_account():
    clock = Clock()
    ops = FakeOps(clock)
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 4], shard_base="nc"), ops, clock=clock, log=lambda m: None)
    _run(pool, clock, 5 * 3600)
    orders = [e for e in ops.events if e[0] == "order"]
    drains = [e for e in ops.events if e[0] == "drain"]
    assert [o[2] for o in orders[:2]] == [1, 4]
    # t1 lease ends at 600 + 4h; replacement ordered 1h before, t1 drained once t2 is ready
    assert orders[1][3] == pytest.approx(600 + 3 * 3600, abs=60)
    assert drains[0][1] == "t1" and drains[0][2] < 600 + 4 * 3600 - 600
    assert ops.tickets["t2"]["shards"] == ["nc-0", "nc-1"]
    assert {m.shard for m in pool.members} >= {"nc-0", "nc-1"}


def test_pool_single_account_waits_for_the_old_order():
    clock = Clock()
    ops = FakeOps(clock)
    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="nc"), ops, clock=clock, log=lambda m: None)
    _run(pool, clock, 4 * 3600 + 1200)
    orders = [e for e in ops.events if e[0] == "order"]
    drains = [e for e in ops.events if e[0] == "drain"]
    assert len(orders) == 2 and drains[0][1] == "t1"
    assert orders[1][3] >= drains[0][2]  # never two orders on one account
    assert drains[0][2] == pytest.approx(600 + 4 * 3600 - 600, abs=60)  # end margin


def test_pool_stops_below_balance():
    clock = Clock()
    ops = FakeOps(clock)
    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="nc", stop_below_ucr=50_000_000),
                      ops, clock=clock, log=lambda m: None)
    _run(pool, clock, 3600)
    assert any(m.state == READY for m in pool.members)
    ops.bal = 40_000_000
    pool.tick()
    assert pool.stopped and all(m.state == GONE for m in pool.members)
    clock.t += 3600
    pool.tick()
    assert len([e for e in ops.events if e[0] == "order"]) == 1


def test_pool_does_not_order_below_floor():
    clock = Clock()
    ops = FakeOps(clock, balance=30_000_000)  # above stop (20M) but below stop + 1 h (44M)
    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="nc"), ops, clock=clock, log=lambda m: None)
    pool.tick()
    assert not ops.events and not pool.stopped


def test_pool_state_roundtrip(tmp_path):
    clock = Clock()
    ops = FakeOps(clock)
    cfg = PoolConfig(target=1, accounts=[1], shard_base="nc")
    pool = IslandPool(cfg, ops, clock=clock, log=lambda m: None)
    _run(pool, clock, 3600)
    path = tmp_path / "pool.json"
    pool.save(path)
    again = IslandPool.restore(path, cfg, ops, clock=clock, log=lambda m: None)
    assert again.snapshot() == pool.snapshot()
    time.sleep(0)


# -- SSH + docker sampler route ----------------------------------------------

def test_ssh_sampler_inventory_and_docker_argv():
    from yeto.launch.nc import ssh_sampler as ss

    node = ss.node_of(_lease())
    assert (node.host, node.card, node.device_family, node.cards_per_node) == ("yeto-nc-t1", "MI355X", "rocm", 8)
    argv = ss.docker_run_argv(node, name=ss.container_name("nc-0"), command=ss.sampler_command("/work/sampler/nc-0.json"),
                              env=ss.sampler_env("nc-sky-run"), yeto_dir="/mnt/nvme/yeto/abc",
                              env_file="/home/ubuntu/.yeto-nc-r2.env", detach=True)
    text = " ".join(argv)
    assert "--device /dev/kfd" in text and "--env-file /home/ubuntu/.yeto-nc-r2.env" in text
    assert "/mnt/nvme/models/Kimi-K3.1-Pro:/models/kimi-k3.1-pro:ro" in text and "/mnt/nvme/yeto/abc:/yeto:ro" in text
    assert argv[-3:-1] == [ss.IMAGE, "-c"] and "@sha256:" in ss.IMAGE
    assert "yeto.rl.engine.engine_sampler --spec /work/sampler/nc-0.json" in argv[-1]
    assert "SECRET" not in text


def test_docker_host_script_leaves_provider_k3s_alone():
    from yeto.launch.nc import ssh_sampler as ss

    s = ss.docker_host_script()
    assert "rancher" not in s and "kubelet" not in s and "systemctl stop docker.socket docker containerd" in s
    subprocess.run(["bash", "-n"], input=s, text=True, check=True)


def test_start_sampler_guards(tmp_path):
    from yeto.launch.nc import ssh_sampler as ss

    calls = []
    fake = lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(ValueError, match="protected"):
        ss.start_sampler(_lease(), spec={"sampler_id": "nc-0"}, shard="nc-0", run_prefix="live",
                         r2_env_file="x", protected=frozenset({"live"}), state_dir=tmp_path, run=fake)
    with pytest.raises(ValueError, match="not the shard"):
        ss.start_sampler(_lease(), spec={"sampler_id": "other"}, shard="nc-0", run_prefix="nc-sky",
                         r2_env_file="x", state_dir=tmp_path, run=fake)
    assert not calls


def test_probe_command_has_no_store_writes():
    from yeto.launch.nc import ssh_sampler as ss

    cmd = ss.probe_command("/work/sampler/probe.json")[-1]
    assert "vllm --version" in cmd and "rocm-smi" in cmd and "/work/dry-preflight" in cmd
    assert "s3://" not in cmd and "engine_sampler --spec" not in cmd


def test_ssh_pull_command_caps():
    from yeto.launch.nc import ssh_sampler as ss

    cmd = ss.pull_model_command("models/Kimi-K3.1-Pro", "/mnt/nvme/models/Kimi-K3.1-Pro")
    assert "max_bandwidth 156MB/s" in cmd and "-P 16" in cmd  # 2500 MB/s over 16 jobs
    assert "s3://$R2_BUCKET/models/Kimi-K3.1-Pro" in cmd
    assert "/models/Kimi-K3.1-Pro" in cmd and "--env-file" in cmd
    with pytest.raises(ValueError, match="exceeds"):
        ss.pull_model_command("m", "/mnt/nvme/models/m", max_mb_s=7000)
    with pytest.raises(ValueError, match="dest"):
        ss.pull_model_command("m", "/root/m")


def _run_pull_body(tmp_path, cmd, listing, have=None):
    """Run the container body of ``pull_model_command`` with a fake ``aws`` on PATH;
    ``/models`` is mapped to ``tmp_path/models``. Returns (proc, copied keys)."""
    import os
    import shlex
    import subprocess

    body = shlex.split(cmd)[-1].replace("/models", str(tmp_path / "models"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "listing").write_text("".join(f"2026-11-10 00:00:00 {size} {key}\n" for key, size in listing))
    log = tmp_path / "cp.log"
    fake = bin_dir / "aws"
    fake.write_text("#!/bin/sh\n"
                    f'case "$2" in ls) cat {tmp_path / "listing"};; '
                    f'cp) k=${{4#s3://bkt/}}; echo "$k" >> {log}; '
                    f'grep " $k$" {tmp_path / "listing"} | awk \'{{printf "%*s", $3, ""}}\' > "$5";; esac\n')
    fake.chmod(0o755)
    for key, size in (have or {}).items():
        path = tmp_path / "models" / "M" / key.split("/", 2)[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(" " * size)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", R2_BUCKET="bkt", AWS_ENDPOINT_URL="http://x")
    proc = subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True, timeout=60)
    return proc, sorted(log.read_text().split()) if log.exists() else []


def test_ssh_pull_runs_parallel_jobs_with_a_shared_cap(tmp_path):
    """One aws-cli process reached ~275 MB/s on R2: the pull runs ``jobs`` processes,
    each capped at max_mb_s / jobs, so the node total stays under max_mb_s."""
    from yeto.launch.nc import ssh_sampler as ss

    cmd = ss.pull_model_command("models/K", "/mnt/nvme/models/M", max_mb_s=2400, jobs=16)
    assert "max_bandwidth 150MB/s" in cmd and "-P 16" in cmd
    listing = [("models/K/config.json", 7), ("models/K/a/model-1.safetensors", 30),
               ("models/K/model-2.safetensors", 20)]
    proc, copied = _run_pull_body(tmp_path, cmd, listing, have={"models/K/model-2.safetensors": 20})
    assert proc.returncode == 0, proc.stderr
    assert "YETO_NC_PULL done" in proc.stdout
    assert copied == ["models/K/a/model-1.safetensors", "models/K/config.json"]  # resume skips model-2
    assert (tmp_path / "models/M/a/model-1.safetensors").stat().st_size == 30


def test_ssh_pull_include_glob_filters_keys(tmp_path):
    from yeto.launch.nc import ssh_sampler as ss

    cmd = ss.pull_model_command("models/K", "/mnt/nvme/models/M", include="*.json", jobs=2)
    proc, copied = _run_pull_body(tmp_path, cmd, [("models/K/config.json", 3), ("models/K/m.safetensors", 4)])
    assert proc.returncode == 0, proc.stderr
    assert copied == ["models/K/config.json"]


def test_create_follows_live_status_not_the_ledger():
    """An old ticket in the ledger never blocks an order: nc-query's live state decides."""
    fake = FakeNc(ticket={"id": "old", "state": "ended"})
    c = _client(fake)
    c.ledger.add(1, "old", ip="192.0.2.1")  # still "open" in the local ledger
    assert c.create() == "abc123def456"


def test_preflight_builds_the_reward():
    from yeto.launch.nc import ssh_sampler as ss

    assert "build_reward(spec)" in ss.PREFLIGHT_PY


def test_backlog_groups_matches_sample_roles():
    from types import SimpleNamespace as NS

    from yeto.rl.engine import sample_roles as sr

    m = sr.DEFAULT_MAX_BACKLOG_ROUNDS
    # no-sync: the launcher's plain-sampler cap
    plain = sr._max_backlog_rounds(NS(rl_sampler_max_backlog_rounds=None), sr.SAMPLER)
    assert tasks.backlog_groups(groups_per_round=10, synced=False) == plain * 10 == m * 10
    assert tasks.backlog_groups(groups_per_round=10, synced=False, explicit_rounds=6) == 60
    # synced: the extracted pure function equals min_full_parameter_backlog_rounds
    args = NS(fragments=8, local_rl_rounds_per_sync=2, pipeline=2)
    sr_need = sr.stall_free_backlog_rounds(trainers_fed=2, sharing=3, fragments=8, local_rounds=2, pipeline=2)
    assert sr_need == -(-2 * (8 * 2 + 2) // 3) == 12
    assert tasks.backlog_groups(groups_per_round=10, synced=True, trainers=2, samplers=3, fragments=8,
                                local_rounds=2, pipeline=2) == (sr_need + m) * 10
    with pytest.raises(ValueError, match="stalls"):
        tasks.backlog_groups(groups_per_round=10, synced=True, trainers=2, samplers=1, fragments=8,
                             explicit_rounds=4)
    del args


def test_gsm8k_reward_scores_three_rows():
    from yeto.rl.engine.engine_sampler_entry import PromptItem, miles_reward
    from yeto.rl.gsm8k_reward import score

    rows = [json.loads(line) for line in Path("data/gsm8k/gsm8k-tasks256.jsonl").read_text().splitlines()[:3]]
    reward = miles_reward(score)
    for row in rows:
        item = PromptItem(prompt_id="x", prompt="p", label=row["label"], metadata={})
        good = SimpleNamespace(text="so \\boxed{{{}}}".format(row["label"]), token_ids=[1], prompt_token_ids=[])
        bad = SimpleNamespace(text="\\boxed{999999}", token_ids=[1], prompt_token_ids=[])
        assert reward(item, good) == 1.0 and reward(item, bad) == 0.0


def _base_spec():
    return json.loads((Path(__file__).resolve().parents[1] / "examples/k31/sampler-spec.nc-mi355x.json").read_text())


def test_sampler_overrides_match_the_modal_launcher_knobs():
    spec = tasks.apply_sampler_overrides(tasks.sampler_spec(_base_spec(), sampler_id="nc-1", shards=["nc-0"]),
                                         data_slot="1/3", drop_zero_variance=True, thinking_effort="high",
                                         reload="restart", n=4, max_tokens=2048)
    assert spec["sampler_id"] == "nc-1" and spec["buffer_shards"] == ["nc-0", "nc-1"]
    assert spec["prompts"]["data_slot"] == "1/3"
    assert spec["prompts"]["chat_template_kwargs"] == {"thinking_effort": "high"}
    assert spec["drop_zero_variance_groups"] is True
    assert spec["sampling"]["n"] == 4 and spec["sampling"]["max_tokens"] == 2048
    assert spec["engine"]["server"]["in_place_reload"] is False
    assert spec["engine"]["options"]["expert_file_update"] == "off"
    assert spec["engine"]["options"]["expert_update_probe"] is False
    hot = tasks.apply_sampler_overrides(spec, reload="hot")
    assert hot["engine"]["server"]["in_place_reload"] is True and hot["engine"]["options"]["expert_file_update"] == "auto"
    assert tasks.apply_sampler_overrides(_base_spec()) == _base_spec()  # nothing set: the file as is


def test_thinking_effort_reaches_the_agentic_template():
    # an agentic round renders with agentic.template_kwargs; prompts.chat_template_kwargs is unused there
    base = dict(_base_spec(), agentic={"template": "kimi_k31", "template_kwargs": {"x": 1}})
    spec = tasks.apply_sampler_overrides(base, thinking_effort="max")
    assert spec["agentic"]["template_kwargs"] == {"x": 1, "thinking_effort": "max"}
    assert "agentic" not in tasks.apply_sampler_overrides(_base_spec(), thinking_effort="max")


def test_reload_on_a_vllm_spec_sets_only_the_server_flag():
    from yeto.rl.engine import engine_sampler_entry as ese

    base = {"platform": "rocm", "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                           "adapter": "vllm-server", "server": {"base_dir": "/m", "port": 1, "card_model": "MI355X", "device_family": "rocm", "engine_version": "v", "precision": "mxfp4", "quant_scheme": "q", "topology": "tp8", "max_model_len": 4096},
                                           "options": {"tp": 8}}}
    spec = tasks.apply_sampler_overrides(base, reload="restart")
    assert spec["engine"]["server"]["in_place_reload"] is False
    assert spec["engine"]["options"] == {"tp": 8}
    ese.build_engine(spec)  # VllmOptions takes the options as they are


def test_sampler_overrides_refuse_bad_values():
    with pytest.raises(ValueError):
        tasks.apply_sampler_overrides(_base_spec(), data_slot="3/3")
    with pytest.raises(ValueError):
        tasks.apply_sampler_overrides(_base_spec(), reload="cold")
    with pytest.raises(ValueError, match="max_model_len"):
        tasks.apply_sampler_overrides(_base_spec(), max_tokens=32768)  # + 4096 prompt tokens


def test_pool_members_split_the_prompts_and_a_replacement_keeps_the_slot():
    clock = Clock()
    ops = FakeOps(clock)
    pool = IslandPool(PoolConfig(target=3, accounts=[1, 2, 3, 4, 5, 6], shard_base="nc", data_slots=3), ops,
                      clock=clock, log=lambda m: None)
    _run(pool, clock, 5 * 3600)
    first = {m.ticket: m.data_slot for m in pool.members[:3]}
    assert sorted(first.values()) == ["0/3", "1/3", "2/3"]
    by_ticket = {m.ticket: m for m in pool.members}
    replaced = [m for m in pool.members if m.replaced_by]
    assert len(replaced) == 3
    for m in replaced:
        assert by_ticket[m.replaced_by].data_slot == m.data_slot
    live = [m.data_slot for m in pool.members if m.state != GONE and m.replaced_by is None]
    assert len(live) == len(set(live))


def test_cli_sampler_override_flags():
    from yeto.launch.nc import cli as nc_cli

    p = __import__("argparse").ArgumentParser()
    nc_cli._add_sampler_override_args(p)
    args = p.parse_args(["--data-slot", "0/2", "--drop-zv", "--reload", "hot", "--n", "8", "--max-tokens", "512",
                         "--thinking-effort", "low"])
    assert nc_cli._arg_overrides(args) == {"data_slot": "0/2", "drop_zero_variance": True, "thinking_effort": "low",
                                           "reload": "hot", "n": 8, "max_tokens": 512,
                                           "continues_shard": None}
    assert nc_cli._arg_overrides(p.parse_args([]))["drop_zero_variance"] is None
    assert nc_cli._arg_overrides(p.parse_args(["--keep-zv"]))["drop_zero_variance"] is False
    with pytest.raises(ValueError, match="unknown"):
        nc_cli._overrides({"data_slot": "0/2"})


def test_ssh_sampler_agentic_env_file(tmp_path):
    from yeto.launch.nc import ssh_sampler as ss

    node = ss.node_of(_lease())
    argv = ss.docker_run_argv(node, name="x", command=["true"], env={}, yeto_dir=None, env_file="/a.env",
                              detach=True, extra_env_files=["/b.env"])
    text = " ".join(argv)
    assert text.index("--env-file /a.env") < text.index("--env-file /b.env")
    calls = []
    fake = lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(ValueError, match="agentic section"):
        ss.start_sampler(_lease(), spec={"sampler_id": "nc-0"}, shard="nc-0", run_prefix="p", r2_env_file="x",
                         state_dir=tmp_path, agentic_env_file="h.env", run=fake)
    assert not calls


def test_sampler_command_links_the_agentic_home_ssh_into_the_passwd_home(tmp_path):
    """OpenSSH ignores $HOME: with HOME=/work/harbor/home (agentic env file) the docker
    CLI's ssh found no host alias on the MI355X node until ~/.ssh was linked."""
    import os
    import subprocess

    from yeto.launch.nc import ssh_sampler as ss

    assert ss.sampler_command("/work/s.json")[-1].startswith(ss.SSH_HOME_LINK)
    home, pw = tmp_path / "harbor-home", tmp_path / "root"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "config").write_text("Host yeto-sbx-3\n")
    pw.mkdir()
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "getent").write_text(f"#!/bin/sh\necho \"root:x:0:0::{pw}:/bin/sh\"\n")
    (fake / "getent").chmod(0o755)
    env = dict(os.environ, HOME=str(home), PATH=f"{fake}:{os.environ['PATH']}")
    for _ in range(2):  # idempotent: an existing link is left alone
        subprocess.run(["sh", "-c", ss.SSH_HOME_LINK + "true"], env=env, check=True)
    assert (pw / ".ssh").is_symlink() and (pw / ".ssh" / "config").read_text() == "Host yeto-sbx-3\n"
    # HOME equal to the passwd home, or a real ~/.ssh already there: nothing changes
    (pw / ".ssh").unlink()
    (pw / ".ssh").mkdir()
    subprocess.run(["sh", "-c", ss.SSH_HOME_LINK + "true"], env=env, check=True)
    assert not (pw / ".ssh").is_symlink() and not (pw / ".ssh" / "config").exists()


def test_docker_host_script_waits_for_apt_lock_and_reports_failure():
    from yeto.launch.nc import ssh_sampler as ss

    s = ss.docker_host_script()
    assert s.count("DPkg::Lock::Timeout=900") == 1
    assert "YETO_NC_DOCKER_FAIL apt-get update" in s and "YETO_NC_DOCKER_FAIL apt-get install docker.io" in s
    subprocess.run(["bash", "-n"], input=s, text=True, check=True)


def test_ssh_sampler_starts_on_the_newest_release_unless_the_spec_says_no(tmp_path, monkeypatch):
    from yeto.launch.nc import ssh_sampler as ss

    monkeypatch.setattr(ss, "upload_code", lambda alias, run: "/mnt/nvme/yeto/x")
    monkeypatch.setattr(ss, "upload_file", lambda *a, **k: None)
    ok = lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="")  # noqa: E731
    for given, written in (({}, True), ({"start_on_latest": False}, False)):
        ss.start_sampler(_lease(), spec={"sampler_id": "nc-0", **given}, shard="nc-0", run_prefix="p",
                         r2_env_file="x", state_dir=tmp_path, run=ok)
        assert json.loads((tmp_path / "spec-nc-0.json").read_text())["start_on_latest"] is written


def test_relay_directory_is_configurable_and_checked(monkeypatch):
    monkeypatch.setenv("YETO_NC_RELAY_DIR", "/work/relay-a")
    cmd = tunnel.relay_start_command(["192.0.2.7:22"])
    assert "/work/relay-a/relay.pid" in cmd and "/data/home" not in cmd
    monkeypatch.setenv("YETO_NC_RELAY_DIR", "relative; rm -rf x")
    with pytest.raises(ValueError, match="YETO_NC_RELAY_DIR"):
        tunnel.relay_start_command([])
    monkeypatch.delenv("YETO_NC_RELAY_DIR")
    assert tunnel.DEFAULT_RELAY_DIR + "/relay.mjs" in tunnel.relay_start_command([])


def _run_pull_body_with_aws(tmp_path, cmd, aws_ls):
    """The pull body with a fake ``aws`` whose ``s3 ls`` runs ``aws_ls`` (shell)."""
    import os
    import shlex
    import subprocess

    body = shlex.split(cmd)[-1].replace("/models", str(tmp_path / "models"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "aws"
    fake.write_text(f'#!/bin/sh\ncase "$2" in ls) {aws_ls};; cp) echo "cp $4" >> {tmp_path / "cp.log"};; esac\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", R2_BUCKET="bkt", AWS_ENDPOINT_URL="http://x")
    return subprocess.run(["sh", "-c", body], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("aws_ls", ["echo 'An error occurred (AccessDenied)' >&2; exit 255", "true"])
def test_ssh_pull_refused_or_empty_listing_is_a_failed_pull(tmp_path, aws_ls):
    """A refused listing (credentials, endpoint) or an empty one (wrong src_key) copied
    nothing: the pull fails instead of printing ``done`` (a pipeline exits with xargs'
    status, and ``xargs -r`` on no input exits 0), so the sampler is not started on an
    empty model directory."""
    from yeto.launch.nc import cli
    from yeto.launch.nc import ssh_sampler as ss

    proc = _run_pull_body_with_aws(tmp_path, ss.pull_model_command("models/K", "/mnt/nvme/models/M"), aws_ls)
    assert proc.returncode != 0
    assert "YETO_NC_PULL done" not in proc.stdout and "YETO_NC_PULL fail" in proc.stdout
    assert not (tmp_path / "cp.log").exists()
    assert cli.pull_ended_without_done(f"exited {proc.returncode}\n{proc.stdout}")


def test_pull_status_over_without_done():
    """Any exit of the pull container without ``done`` (not only codes 1xx), a vanished
    container, or a nonzero exit of the current no-docker attempt ends the wait at once;
    a running pull and the lines of an earlier attempt in an appended log do not."""
    from yeto.launch.nc import cli

    over = ["exited 255\nYETO_NC_PULL start x jobs=16", "exited 2\n", "exited 3\n", "absent",
            "YETO_NC_PULL start a\nxargs: sh: exited with status 255; aborting\nYETO_NC_PULL exit 124",
            "running 0\nYETO_NC_PULL start a\nYETO_NC_PULL fail models/K/x"]
    for status in over:
        assert cli.pull_ended_without_done(status), status
    going = ["running 0\nYETO_NC_PULL start a jobs=16\n1.2T /mnt/nvme/models/M", "",
             "YETO_NC_PULL fail models/K/x\nYETO_NC_PULL exit 124\nYETO_NC_PULL start b"]
    for status in going:
        assert not cli.pull_ended_without_done(status), status


def test_pull_waits_end_on_a_container_exit_without_done(monkeypatch):
    """``_pull`` raises on the first status of an exited pull container instead of
    polling the billed node until ``timeout_s``."""
    from yeto.launch.nc import cli
    from yeto.launch.nc import ssh_sampler as ss

    ops = cli.NcSshOps.__new__(cli.NcSshOps)
    ops.cfg, ops._run, ops._route = {"r2_env_file": "/dev/null"}, None, None
    monkeypatch.setattr(ss, "pull_model", lambda *a, **k: ss.PULL_CONTAINER)
    statuses = iter(["running 0\nYETO_NC_PULL start", "exited 255\nYETO_NC_PULL start"])
    monkeypatch.setattr(ss, "pull_status", lambda *a, **k: next(statuses))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    lease = type("L", (), {"ticket": "t1"})()
    with pytest.raises(cli.NcError, match="model pull failed"):
        ops._pull(lease, {"src_key": "models/K", "timeout_s": 3600, "poll_s": 1})
