"""Reach of an own node (unified-sampler-pool 3.2): ``direct``, ``jump`` (``ProxyJump``)
and ``relay`` (the ``ProxyCommand`` of :mod:`yeto.launch.nc.tunnel`'s WebSocket relay),
and every ssh of the ssh sampler pool going through it."""

from __future__ import annotations

import io
import json
import os
import shlex
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_ssh_sampler_pool import FakeRun, pool_cfg
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.nc import cli, node_pool, tunnel
from yeto.launch.nc.client import NodeLease
from yeto.launch.node_inventory import NodeProfile, Reach, load_profiles, parse_node
from yeto.node import relay_proxy, transport

NODE = {"host": "ops@10.0.0.5", "device_family": "npu", "cpu_arch": "aarch64", "card": "910B4",
        "cards_per_node": 8, "hbm_gib": 64, "host_memory_gib": 1024, "runtime": "rootfs"}


def _option(argv, key):
    """The value of ``-o <key>=...`` in an ssh argv (None: absent)."""
    for a, b in zip(argv, argv[1:]):
        if a == "-o" and b.startswith(key + "="):
            return b[len(key) + 1:]
    return None


# -- the options ------------------------------------------------------------------

def test_direct_and_jump_options():
    assert transport.reach_options(None) == [] and transport.reach_options(Reach()) == []
    assert transport.reach_options(Reach("jump", "ops@bastion")) == ["-o", "ProxyJump=ops@bastion"]


def test_jump_is_the_inventory_proxy_jump():
    """``reach: jump`` and ``proxy_jump:`` give one and the same ssh route."""
    old = parse_node({**NODE, "name": "a", "proxy_jump": "ops@bastion"}, index=0)
    assert NodeProfile.from_node(old).ssh_options() == old.ssh_options() == ["-o", "ProxyJump=ops@bastion"]
    via = NodeProfile.from_node(parse_node({**NODE, "name": "b"}, index=0), reach=Reach("jump", "ops@bastion"))
    assert via.ssh_options() == old.ssh_options() and via.proxy_jump == "ops@bastion"
    plain = parse_node({**NODE, "name": "c"}, index=0)
    assert NodeProfile.from_node(plain).ssh_options() == plain.ssh_options() == []


def test_a_national_compute_relay_is_the_lease_alias_proxy_command(tmp_path, monkeypatch):
    """``via: nc:<account>`` dials exactly as a leased node's alias does: the very
    ``ProxyCommand`` line of ``node_pool.ssh_host_block`` (that block is unchanged)."""
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path))
    opts = transport.reach_options(Reach("relay", "nc:4"))
    assert opts[0] == "-o" and opts[1] == f"ProxyCommand={node_pool.proxy_command(4)}"
    lease = NodeLease(account=4, ticket="t1", ip="10.1.2.3", user="ubuntu", host_key="", granted_at=None,
                      lease_until=None, gpus=None, rate_ucr_per_gpu_hour=None)
    block = node_pool.ssh_host_block(lease)
    assert f"  ProxyCommand {node_pool.proxy_command(4)}" in block.splitlines()
    assert _option(opts, "ProxyCommand").endswith(f"-m yeto.launch.nc proxy --account 4 --relay-port "
                                                  f"{tunnel.RELAY_PORT} %h %p")
    port = transport.reach_options(Reach("relay", "nc:4:8799"))[1]
    assert port == f"ProxyCommand={node_pool.proxy_command(4, 8799)}"


def test_a_site_relay_url_dials_through_relay_proxy():
    cmd = transport.relay_proxy_command("wss://relay.example/yeto")
    argv = shlex.split(cmd)
    assert argv[0] == "env" and argv[1].startswith("PYTHONPATH=") and argv[2] == sys.executable
    assert argv[3:] == ["-P", "-m", "yeto.node.relay_proxy", "--url", "wss://relay.example/yeto", "%h", "%p"]
    assert Path(argv[1].split("=", 1)[1], "yeto", "node", "relay_proxy.py").is_file()
    url = "https://gw.example:8443/r"
    assert transport.reach_options(Reach("relay", url)) == ["-o", "ProxyCommand=" + transport.relay_proxy_command(url)]


@pytest.mark.parametrize("via", ["nc:", "nc:x", "nc:1:2:3", "relay.example", "wss://", "wss://a b",
                                 "wss://x;rm -rf /", "https://x/$(id)", "ftp://x"])
def test_a_relay_via_ssh_cannot_dial_is_refused(via):
    with pytest.raises(transport.ReachError):
        transport.reach_options(Reach("relay", via))


@pytest.mark.parametrize("mode", ["outbound", "function"])
def test_nodes_without_inbound_ssh_are_refused(mode):
    with pytest.raises(transport.ReachError, match=mode):
        transport.reach_options(Reach(mode))


def test_ssh_exec_dials_through_the_reach():
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    node = NodeProfile.from_node(parse_node({**NODE, "name": "a"}, index=0), reach=Reach("relay", "wss://r.example"))
    box = transport.ssh_exec(node, run=run)
    assert box.exec("true").stdout == "ok"
    argv = calls[0]
    assert argv[argv.index("ops@10.0.0.5") - 2:argv.index("ops@10.0.0.5")] == \
        ["-o", f"ProxyCommand={transport.relay_proxy_command('wss://r.example')}"]


# -- the ssh sampler pool ---------------------------------------------------------

def _inventory(tmp_path, **extra):
    nodes = [{**NODE, "name": "edge", "host": "ops@192.0.2.1"},
             {**NODE, "name": "behind-jump", "host": "ops@10.0.0.6", "reach": {"mode": "jump", "via": "ops@bastion"}},
             {**NODE, "name": "behind-relay", "host": "10.0.0.7", "reach": {"mode": "relay", "via": "wss://relay.example/y"}},
             {**NODE, "name": "old-jump", "host": "ops@10.0.0.8", "proxy_jump": "ops@bastion"}]
    for n in nodes:
        n.update(extra.get(n["name"], {}))
    path = tmp_path / "inv.yaml"
    path.write_text(json.dumps({"nodes": nodes}))
    return path


def _pool(tmp_path, monkeypatch, *, node, **section):
    class Pop:
        stdout = None

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(sp.subprocess, "Popen", Pop)
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    monkeypatch.setenv(sp.ssh_transport.MUX_ENV, "off")  # the mux options are their own test's
    cfg = pool_cfg(tmp_path, inventory=str(_inventory(tmp_path)), nodes=[node], **section)
    cfg["provider"].pop("devices", None)
    run = FakeRun()
    return sp.SshSamplerPoolOps(cfg, run=run, state_dir=tmp_path, code_ref="HEAD"), run


@pytest.mark.parametrize("node,want", [
    ("behind-relay", ("ProxyCommand", transport.relay_proxy_command("wss://relay.example/y"))),
    ("behind-jump", ("ProxyJump", "ops@bastion")),
    ("old-jump", ("ProxyJump", "ops@bastion")),
    ("edge", None),
])
def test_every_ssh_of_a_start_probe_and_stop_goes_through_the_reach(tmp_path, monkeypatch, node, want):
    """Preflight, code upload, spec upload, start, probe, log copy and stop: each ssh dials
    the node's reach right before its host; a direct node's argv has no proxy option."""
    ops, run = _pool(tmp_path, monkeypatch, node=node)
    m = SimpleNamespace(account=0, ticket=ops.order(0), shard="f09-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["f09-0"])
    assert ops.probe(m).joined
    ops.drain(m)
    ssh = [c[0] for c in run.calls if c[0][0] == "ssh"]
    host = ops.conf.nodes[0].host
    assert len(ssh) >= 6 and all(host in argv for argv in ssh)
    for argv in ssh:
        before = argv[:argv.index(host)]
        if want is None:
            assert _option(before, "ProxyCommand") is None and _option(before, "ProxyJump") is None
        else:
            assert before[-2:] == ["-o", f"{want[0]}={want[1]}"]
            other = "ProxyJump" if want[0] == "ProxyCommand" else "ProxyCommand"
            assert _option(before, other) is None


def test_a_direct_nodes_argv_is_unchanged(tmp_path, monkeypatch):
    """The argv of a node without reach is what the pool sent before reach existed."""
    ops, run = _pool(tmp_path, monkeypatch, node="edge")
    ops.ssh(ops.conf.nodes[0], "true")
    assert run.calls[-1][0] == ["ssh", "-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                                "ops@192.0.2.1", "true"]


def test_the_mux_probe_reads_the_route_with_its_reach(tmp_path, monkeypatch):
    seen = []
    monkeypatch.delenv(sp.ssh_transport.MUX_ENV, raising=False)
    monkeypatch.setattr(sp.ssh_transport, "user_controls", lambda target: seen.append(list(target)) or False)
    ops, _run = _pool(tmp_path, monkeypatch, node="behind-jump")
    monkeypatch.delenv(sp.ssh_transport.MUX_ENV, raising=False)
    ops.ssh(ops.conf.nodes[0], "true")
    assert seen[-1] == ["-o", "ProxyJump=ops@bastion", "ops@10.0.0.6"]


def test_inventory_reach_is_read_and_a_node_ssh_cannot_reach_is_refused(tmp_path):
    profiles = {p.name: p for p in load_profiles(_inventory(tmp_path))}
    assert profiles["behind-relay"].reach == Reach("relay", "wss://relay.example/y")
    assert profiles["old-jump"].reach == Reach("jump", "ops@bastion")
    for bad in ({"reach": {"mode": "outbound"}}, {"reach": {"mode": "relay", "via": "nowhere"}}):
        path = _inventory(tmp_path, edge=bad)
        with pytest.raises(ValueError, match="node edge"):
            sp.parse_config({"provider": {"kind": "ssh", "inventory": str(path), "nodes": ["edge"],
                                          "workdir": "/srv/yeto/r", "runtime": "venv",
                                          "venv": {"python": "/v/bin/python"}}})
    cfg = {"target": 1, "slots": [0], "spec": "s", "run_prefix": "p", "data_path": "syncer",
           "provider": {"kind": "ssh", "inventory": str(path), "nodes": ["edge"], "workdir": "/srv/yeto/r",
                        "runtime": "venv", "venv": {"python": "/v/bin/python"}}}
    assert any("node edge" in p and "nowhere" in p for p in sp.offline_problems(cfg))


# -- the relay proxy --------------------------------------------------------------

def test_relay_proxy_endpoint():
    assert relay_proxy.endpoint("wss://r.example/y").base_url == "https://r.example/y"
    assert relay_proxy.endpoint("ws://10.0.0.1:8791").base_url == "http://10.0.0.1:8791"
    assert relay_proxy.endpoint("https://r.example").cookie == ""
    with pytest.raises(tunnel.TunnelError):
        relay_proxy.endpoint("ftp://r.example")


def test_relay_proxy_pipes_ssh_bytes_through_a_relay(monkeypatch):
    """A loopback relay (101 upgrade, then echo of each binary frame, uppercased): the
    proxy asks for the node's ``host:port`` and moves the bytes both ways."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    asked = []

    def serve():
        conn, _ = srv.accept()
        with conn:
            head = b""
            while b"\r\n\r\n" not in head:
                head += conn.recv(4096)
            asked.append(head.split(b"\r\n", 1)[0].decode())
            conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
            reader, data = tunnel.FrameReader(), b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                for op, payload in reader.feed(chunk):
                    if op == 8:
                        conn.sendall(bytes([0x88, 0]))
                        return
                    reply = payload.upper()
                    conn.sendall(bytes([0x82, len(reply)]) + reply)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    out = io.BytesIO()
    r, w = os.pipe()  # the pipe reads stdin by its file descriptor, as ssh hands it over
    os.write(w, b"ssh-2.0-hello")
    os.close(w)
    stdin = os.fdopen(r, "rb")
    real_pipe = tunnel.pipe
    monkeypatch.setattr(tunnel, "pipe", lambda sock, rest: real_pipe(sock, rest, stdin=stdin, stdout=out))
    try:
        assert relay_proxy.main(["--url", f"ws://127.0.0.1:{port}/relay", "10.0.0.7", "22"]) == 0
    finally:
        srv.close()
        t.join(5)
    assert asked == ["GET /relay/?target=10.0.0.7:22 HTTP/1.1"]
    assert out.getvalue() == b"SSH-2.0-HELLO"
