"""agentic.setup.records_relay: the AgentENV records proxy of a sampler node that passes
port 22 only, served through a reverse forward to a relay host. Parsing, the sampler
env, the dry-run warnings, the node step (systemd unit) against a fake node, the
relay-side command (its script run locally against fake tools) and the in-microVM
self-check hint. No SSH, no machine."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_agentenv_run_up import _spec as _agentenv_spec
from tests.test_agentic_setup import FakeNode, _node
from yeto.launch import agentic_setup as su
from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc import records_relay as rr
from yeto.launch.nc import ssh_sampler as ss
from yeto.rl.harness.harbor import agentenv_network as net

RELAY = "relay.example.org"
FAKE_PRIV = "-----BEGIN OPENSSH " + "PRIVATE KEY-----\nfake-{n}\n-----END OPENSSH " + "PRIVATE KEY-----\n"


def _keys(tmp_path, n=3):
    for k in range(n):
        p = tmp_path / f"relay-{k}.key"
        p.write_text(FAKE_PRIV.format(n=k))
        p.chmod(0o600)
    return str(tmp_path / "relay-{shard}.key")


def _spec(tmp_path, *, sampler_id="k31-1", **relay):
    spec = _agentenv_spec(tmp_path)
    setup = spec["agentic"]["setup"]
    del setup["records_proxy"]
    setup["records_relay"] = {"host": RELAY, "key": {"path": _keys(tmp_path)}, "listen_port": "18281-18283",
                              "allow_cidrs": ["198.51.100.7", "203.0.113.0/28"], **relay}
    spec["sampler_id"] = sampler_id
    return spec


def fake_keygen(argv, **kw):
    """ssh-keygen on the launcher: a public key derived from the private key file."""
    if argv[0] != "ssh-keygen":
        raise AssertionError(argv)
    body = Path(argv[-1]).read_text().split("\n")[1].replace("-", "")
    if "-lf" in argv:
        return SimpleNamespace(returncode=0, stdout=f"256 SHA256:{body} c (ED25519)\n", stderr="")
    return SimpleNamespace(returncode=0, stdout=f"ssh-ed25519 AAAAC3{body} c\n", stderr="")


# -- parsing and the sampler env -------------------------------------------------

def test_relay_picks_the_shard_port_and_key_and_sets_the_public_base(tmp_path):
    s = su.parse(_spec(tmp_path))
    r = s.relay
    assert (r.listen_port, r.target_port, r.user, r.port) == (18282, 18080, "root", 22)
    assert r.key.path.endswith("relay-1.key") and r.forward == "0.0.0.0:18282:127.0.0.1:18080"
    env = s.container_env()
    assert env["YETO_RECORDS_PROXY_PUBLIC_BASE"] == f"http://{RELAY}:18282"
    assert su.local_problems(s) == []
    lines, problems = su.dry_run_report(_spec(tmp_path))
    text = "\n".join(lines)
    assert problems == [] and su.RELAY_UNIT in text and "records-relay --spec" in text
    assert not any("WARNING" in ln for ln in lines)
    pub = su.parse(_spec(tmp_path, public_host="198.51.100.200", listen_port=19000, key={"env": "RK"},
                         sampler_id="__SHARD__"))
    assert pub.container_env()["YETO_RECORDS_PROXY_PUBLIC_BASE"] == "http://198.51.100.200:19000"


@pytest.mark.parametrize("change, match", [
    ({"listen_port": "18281-18282", "sampler_id": "k31-2"}, "no port for shard"),
    ({"sampler_id": "__SHARD__"}, "-<shard index>"),
    ({"host": "<relay>"}, "records_relay.host"),
    ({"host": "root@relay"}, "records_relay.host"),
    ({"listen_port": None}, "listen_port is required"),
    ({"listen_port": "9-1"}, "empty"),
    ({"allow_cidrs": ["nope"]}, "not a CIDR"),
    ({"allow_cidrs": "198.51.100.7"}, "list of CIDRs"),
    ({"bogus": 1}, "unknown keys"),
])
def test_relay_refusals(tmp_path, change, match):
    sampler_id = change.pop("sampler_id", "k31-1")
    with pytest.raises(ValueError, match=match):
        su.parse(_spec(tmp_path, sampler_id=sampler_id, **change))


def test_relay_and_a_public_base_together_or_without_a_proxy_port_are_refused(tmp_path):
    spec = _spec(tmp_path)
    spec["agentic"]["setup"]["records_proxy"] = {"public_base": "https://records.example.org"}
    with pytest.raises(ValueError, match="not both"):
        su.parse(spec)
    spec = _spec(tmp_path)
    del spec["agentic"]["proxy"]["port"]
    with pytest.raises(ValueError, match="agentic.proxy.port"):
        su.parse(spec)
    remote = _spec(tmp_path)
    remote["agentic"]["setup"]["provider"] = "remote-docker"
    remote["agentic"]["task_defaults"]["provider"] = "remote-docker"
    del remote["agentic"]["setup"]["agentenv"]
    with pytest.raises(ValueError, match="records_relay is for provider agentenv"):
        su.parse(remote)


def test_a_missing_shard_key_is_a_local_problem(tmp_path):
    s = su.parse(_spec(tmp_path, sampler_id="k31-2"))
    os.unlink(s.relay.key.path)
    assert any("records_relay.key" in p for p in su.local_problems(s))


def test_a_records_base_on_a_node_ip_warns_in_the_dry_run_and_at_start(tmp_path):
    spec = _agentenv_spec(tmp_path, public="http://198.51.100.9:18080")
    lines, problems = su.dry_run_report(spec)
    warn = [ln for ln in lines if "WARNING" in ln]
    assert problems == [] and len(warn) == 1 and "port 22 only" in warn[0] and "records_relay" in warn[0]
    for ok in ("https://records.example.org", "http://10.0.0.5:18080", "http://198.51.100.9:8443"):
        assert not any("WARNING" in ln for ln in su.dry_run_report(_agentenv_spec(tmp_path, public=ok))[0])
    # at start the node's address is known: any port on it is the same trap
    setup = su.parse(_agentenv_spec(tmp_path, public="http://198.51.100.9:28080"))
    lease = SimpleNamespace(ip="198.51.100.9", ticket="t1")
    assert "sampler node's own address" in ss.direct_records_warning(setup, spec, lease)
    assert ss.direct_records_warning(su.parse(_spec(tmp_path)), spec, lease) is None


# -- the node step ----------------------------------------------------------------

class RelayNode(FakeNode):
    """FakeNode plus systemd: the unit is active once restarted; ``flaps`` counts restarts."""

    def __init__(self, *, flaps=False, journal="", **kw):
        super().__init__(**kw)
        self.active, self.restarts, self.flaps, self.journal = False, 0, flaps, journal

    def run(self, argv, **kw):
        cmd = argv[-1] if argv[0] == "ssh" else " ".join(argv)
        if argv[0] == "ssh" and "YETO_INIT" in cmd:
            self.calls.append(cmd)
            return self.r("YETO_INIT systemd\n")
        if argv[0] == "ssh" and "systemctl is-active" in cmd:
            self.calls.append(cmd)
            if self.flaps:
                self.restarts += 1
            return self.r(f"{'active' if self.active else 'inactive'}\n{self.restarts}\n")
        if argv[0] == "ssh" and "systemctl restart" in cmd:
            self.calls.append(cmd)
            self.active = True
            return self.r()
        if argv[0] == "ssh" and "journalctl" in cmd:
            self.calls.append(cmd)
            return self.r(self.journal)
        return super().run(argv, **kw)


def _prov(tmp_path, fake, spec=None):
    now = [0.0]

    def sleep(s):
        now[0] += s

    p = su.Provisioner(su.parse(spec or _spec(tmp_path)), _node(fake), sleep=sleep, clock=lambda: now[0],
                       local_run=fake_keygen)
    p.node.stream_tar = lambda *a, **k: None
    return p


def test_the_node_step_writes_the_key_and_unit_and_is_idempotent(tmp_path):
    fake = RelayNode()
    report = _prov(tmp_path, fake).run()
    assert report["records_relay"] == "changed"
    key, unit = fake.files["/mnt/nvme/records-relay/key"], fake.files[su.RELAY_UNIT_PATH]
    assert key[1] == "600" and "fake-1" in key[0]
    assert "Restart=always" in unit[0] and "StartLimitIntervalSec=0" in unit[0]
    assert "-R 0.0.0.0:18282:127.0.0.1:18080 yeto-records-relay" in unit[0]
    config = fake.files["/mnt/nvme/records-relay/config"][0]
    assert f"HostName {RELAY}" in config and "ExitOnForwardFailure yes" in config
    assert "ServerAliveInterval 15" in config and "IdentityFile /mnt/nvme/records-relay/key" in config
    assert not any("fake-1" in c for c in fake.calls)  # the key travels on stdin only
    assert not any(k.startswith("/mnt/nvme/work/") and "relay" in k for k in fake.files)  # never in /work
    restarts = sum("systemctl restart" in c for c in fake.calls)
    assert _prov(tmp_path, fake).run()["records_relay"] == "ok"
    assert sum("systemctl restart" in c for c in fake.calls) == restarts


def test_a_flapping_unit_fails_the_setup_with_the_relay_fix(tmp_path):
    fake = RelayNode(flaps=True, journal="Error: remote port forwarding failed for listen port 18282\n")
    with pytest.raises(su.SetupError, match="records_relay") as info:
        _prov(tmp_path, fake).run()
    msg = str(info.value)
    assert "permitlisten" in msg and "records-relay --spec" in msg and "restarts" in msg
    denied = RelayNode(flaps=True, journal="root@relay: Permission denied (publickey).\n")
    with pytest.raises(su.SetupError, match="SHA256:fake1 needs its restricted authorized_keys line"):
        _prov(tmp_path, denied).run()


class NoSystemdNode(FakeNode):
    """A docker node whose PID 1 is not systemd: the supervisor container's events file
    gets an ``up`` line once it starts (``down`` lines first with ``flaps``)."""

    def __init__(self, *, flaps=0, stderr="", **kw):
        super().__init__(**kw)
        self.running, self.flaps, self.stderr, self.ev = False, flaps, stderr, []

    def run(self, argv, **kw):
        cmd = argv[-1] if argv[0] == "ssh" else " ".join(argv)
        if argv[0] != "ssh":
            return super().run(argv, **kw)
        if "YETO_INIT" in cmd:
            self.calls.append(cmd)
            return self.r("YETO_INIT none\n")
        if "systemctl" in cmd or "journalctl" in cmd:
            raise AssertionError(f"systemd used on a node without it: {cmd}")
        if "docker inspect" in cmd and su.RELAY_CONTAINER in cmd:
            self.calls.append(cmd)
            return self.r("true\n" if self.running else "")
        if "docker run -d" in cmd and su.RELAY_CONTAINER in cmd:
            self.calls.append(cmd)
            self.running = True
            self.ev = []
            return self.r("cid\n")
        if "grep -c" in cmd and "events.jsonl" in cmd:
            self.calls.append(cmd)
            if self.flaps:
                self.ev.append({"event": "rl_agentic_tunnel", "state": "down", "stderr": self.stderr})
            elif not self.ev or self.ev[-1]["state"] != "up":
                self.ev.append({"event": "rl_agentic_tunnel", "state": "up"})
            downs = sum(e["state"] == "down" for e in self.ev)
            return self.r(f"{downs}\n{json.dumps(self.ev[-1])}\n")
        if "tail -n 30" in cmd and "events.jsonl" in cmd:
            self.calls.append(cmd)
            return self.r("".join(json.dumps(e) + "\n" for e in self.ev))
        return super().run(argv, **kw)


def test_without_systemd_the_forward_runs_under_the_supervisor_container(tmp_path):
    fake = NoSystemdNode()
    report = _prov(tmp_path, fake).run()
    assert report["records_relay"] == "changed" and su.RELAY_UNIT_PATH not in fake.files
    start = next(c for c in fake.calls if "docker run -d" in c)
    assert "--restart unless-stopped" in start and "--network host" in start
    assert "-m yeto.launch.agentic_node_agent tunnel --events /mnt/nvme/records-relay/events.jsonl" in start
    assert "-R 0.0.0.0:18282:127.0.0.1:18080 yeto-records-relay" in start
    assert "-v /mnt/nvme/records-relay:/mnt/nvme/records-relay" in start  # the ssh config names that path
    assert ": > /mnt/nvme/records-relay/events.jsonl" in start  # a stale up line of an old supervisor is gone
    argv = fake.files["/mnt/nvme/records-relay/supervisor.argv"][0]
    assert "ssh -N -F /mnt/nvme/records-relay/config" in argv
    assert not any("fake-1" in c for c in fake.calls)
    for c in fake.calls:
        assert "pkill" not in c and "pgrep" not in c and "killall" not in c
    # idempotent: the running supervisor with the same argv is kept
    starts = sum("docker run -d" in c for c in fake.calls)
    assert _prov(tmp_path, fake).run()["records_relay"] == "ok"
    assert sum("docker run -d" in c for c in fake.calls) == starts
    # a stopped one (a rebooted host: the sampler start runs the step again) is started again
    fake.running = False
    assert _prov(tmp_path, fake).run()["records_relay"] == "changed"


def test_a_flapping_supervisor_fails_with_the_relay_fix(tmp_path):
    fake = NoSystemdNode(flaps=1, stderr="Error: remote port forwarding failed for listen port 18282")
    with pytest.raises(su.SetupError, match="records_relay") as info:
        _prov(tmp_path, fake).run()
    msg = str(info.value)
    assert "permitlisten" in msg and "supervisor (no systemd)" in msg and "restarts" in msg


def test_without_a_relay_no_unit_is_written(tmp_path):
    fake = RelayNode()
    report = _prov(tmp_path, fake, spec=_agentenv_spec(tmp_path)).run()
    assert "records_relay" not in report and su.RELAY_UNIT_PATH not in fake.files


# -- the relay side ---------------------------------------------------------------

def test_plan_grants_one_restricted_line_per_shard_key(tmp_path):
    sec = rr.load_section(_spec(tmp_path), base_dir=tmp_path)
    p = rr.plan(sec, run=fake_keygen, label="k31")
    lines = [g.line("k31") for g in p.grants]
    assert len(lines) == 3 and p.ports == (18281, 18282, 18283)
    assert lines[1].startswith('restrict,port-forwarding,permitlisten="0.0.0.0:18282" ssh-ed25519 ')
    assert lines[1].endswith("yeto-records-relay:k31:1") and "fake1" in lines[1]
    assert "PRIVATE" not in rr.script(p)
    assert len(rr.plan(sec, shards=2, run=fake_keygen).grants) == 2
    with pytest.raises(rr.RelayError, match="one per shard"):
        rr.plan(sec, shards=4, run=fake_keygen)
    shared = rr.load_section(_spec(tmp_path, key={"path": str(tmp_path / "relay-0.key")}), base_dir=tmp_path)
    (one,) = rr.plan(shared, run=fake_keygen).grants
    assert one.listens == ("0.0.0.0:18281", "0.0.0.0:18282", "0.0.0.0:18283") and one.shards == (0, 1, 2)
    (tmp_path / "relay-2.key").write_text(FAKE_PRIV.format(n=0))
    with pytest.raises(rr.RelayError, match="share one key"):
        rr.plan(sec, run=fake_keygen)


FAKE_TOOLS = {
    "sudo": '#!/bin/sh\nexec "$@"\n',
    "id": '#!/bin/sh\necho 1000\n',
    "getent": '#!/bin/sh\necho "$2:x:0:0::$FAKE_HOME:/bin/sh"\n',
    "sshd": '#!/bin/sh\necho "sshd $*" >> "$FAKE_LOG"\n'
            '[ "$1" = -T ] && grep -i gatewayports "$FAKE_DROPIN" | tr A-Z a-z\nexit 0\n',
    "systemctl": '#!/bin/sh\necho "systemctl $*" >> "$FAKE_LOG"\n',
    "chown": '#!/bin/sh\nexit 0\n',
    "iptables": '#!/bin/sh\necho "iptables $*" >> "$FAKE_LOG"\nf="$FAKE_FW"\n'
                'case "$1" in\n'
                '  -N) [ -f "$f" ] && exit 1; echo "-N $2" > "$f";;\n'
                '  -S) cat "$f";;\n'
                '  -F) echo "-N $2" > "$f";;\n'
                '  -A) shift; echo "-A $*" >> "$f";;\n'
                '  -C) [ -f "$f.jump" ];;\n'
                '  -I) touch "$f.jump";;\n'
                'esac\n',
}


def _relay_host(tmp_path):
    bin_dir, home = tmp_path / "bin", tmp_path / "relayhome"
    bin_dir.mkdir()
    home.mkdir()
    for name, body in FAKE_TOOLS.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_HOME": str(home), "FAKE_LOG": str(tmp_path / "log"),
           "FAKE_FW": str(tmp_path / "fw"), "FAKE_DROPIN": str(tmp_path / "sshd.d" / "60.conf")}
    return env, home


def _run_script(text, env):
    return subprocess.run(["bash", "-c", text], env=env, capture_output=True, text=True, timeout=60, check=False)


def test_the_relay_script_is_idempotent_on_fake_tools(tmp_path):
    env, home = _relay_host(tmp_path)
    p = rr.plan(rr.load_section(_spec(tmp_path), base_dir=tmp_path), run=fake_keygen, label="k31")
    ak = home / ".ssh" / "authorized_keys"
    ak.parent.mkdir()
    ak.write_text("ssh-ed25519 AAAAoperator admin\n")
    text = rr.script(p, sshd_dropin=env["FAKE_DROPIN"])
    first = _run_script(text, env)
    assert first.returncode == 0, first.stderr
    assert "YETO_RELAY sshd changed" in first.stdout and "YETO_RELAY keys changed" in first.stdout
    assert "YETO_RELAY firewall changed" in first.stdout
    body = ak.read_text()
    assert body.startswith("ssh-ed25519 AAAAoperator admin\n") and body.count("permitlisten=") == 3
    assert "GatewayPorts clientspecified" in Path(env["FAKE_DROPIN"]).read_text()
    fw = Path(env["FAKE_FW"]).read_text()
    assert "-s 198.51.100.7/32 -p tcp -m tcp --dport 18281:18283 -j ACCEPT" in fw
    assert "-s 203.0.113.0/28" in fw and fw.rstrip().endswith("--dport 18281:18283 -j DROP")
    log = Path(env["FAKE_LOG"]).read_text()
    assert "systemctl reload ssh" in log and "restart" not in log
    second = _run_script(text, env)
    assert second.returncode == 0, second.stderr
    for part in ("sshd", "keys", "firewall"):
        assert f"YETO_RELAY {part} ok" in second.stdout
    assert ak.read_text() == body and Path(env["FAKE_LOG"]).read_text().count("reload") == 1
    # fewer shards replace the managed block, the operator's line stays
    two = rr.plan(rr.load_section(_spec(tmp_path), base_dir=tmp_path), shards=2, run=fake_keygen, label="k31")
    assert _run_script(rr.script(two, sshd_dropin=env["FAKE_DROPIN"]), env).returncode == 0
    assert ak.read_text().count("permitlisten=") == 2 and "AAAAoperator" in ak.read_text()


def test_the_relay_script_warns_on_an_unrestricted_line_of_a_node_key(tmp_path):
    env, home = _relay_host(tmp_path)
    p = rr.plan(rr.load_section(_spec(tmp_path, allow_cidrs=[]), base_dir=tmp_path), run=fake_keygen)
    (home / ".ssh").mkdir()
    (home / ".ssh" / "authorized_keys").write_text(p.grants[0].public_key + " by-hand\n")
    out = _run_script(rr.script(p, sshd_dropin=env["FAKE_DROPIN"]), env)
    assert out.returncode == 0, out.stderr
    assert "YETO_RELAY warn" in out.stdout and "YETO_RELAY firewall skipped" in out.stdout
    assert not Path(env["FAKE_FW"]).exists()


def test_apply_reads_the_parts_and_raises_on_a_failed_one(tmp_path):
    p = rr.plan(rr.load_section(_spec(tmp_path), base_dir=tmp_path), run=fake_keygen)
    seen = {}

    def ok(argv, **kw):
        seen["argv"], seen["input"] = argv, kw["input"]
        return SimpleNamespace(returncode=0, stdout="YETO_RELAY sshd ok\nYETO_RELAY keys changed\n"
                                                    "YETO_RELAY warn x\nYETO_RELAY firewall ok\n", stderr="")

    report = rr.apply(p, ssh_target="root@relay.example.org", run=ok)
    assert report == {"sshd": "ok", "keys": "changed", "firewall": "ok", "warnings": "x"}
    assert seen["argv"][-2:] == ["root@relay.example.org", "bash -s"] and "BatchMode=yes" in seen["argv"]
    assert "PRIVATE" not in seen["input"]

    def bad(argv, **kw):
        return SimpleNamespace(returncode=3, stdout="YETO_RELAY sshd failed sshd -T has no gatewayports\n",
                               stderr="")

    with pytest.raises(rr.RelayError, match=r"\[sshd\]: sshd -T"):
        rr.apply(p, ssh_target="root@relay", run=bad)


def test_the_cli_dry_run_needs_no_ssh(tmp_path, capsys, monkeypatch):
    import json

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec(tmp_path)))
    monkeypatch.setattr(su, "key_identity", lambda ref, **kw: ("SHA256:x", "ssh-ed25519 AAAA" + ref.path[-5]))
    assert nc_cli.main(["records-relay", "--spec", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "permitlisten 0.0.0.0:18283" in out and "--dry-run: ok" in out and "PRIVATE" not in out
    bare = _agentenv_spec(tmp_path)
    path.write_text(json.dumps(bare))
    assert nc_cli.main(["records-relay", "--spec", str(path), "--dry-run"]) == 2


# -- the in-microVM self-check --------------------------------------------------

@pytest.mark.asyncio
async def test_an_unreachable_records_proxy_names_the_relay():
    async def timeout(_cmd):
        return SimpleNamespace(return_code=28, stdout="", stderr="timed out")

    eps = net.endpoints_from_mapping({"records-proxy": "http://198.51.100.9:18080"})
    with pytest.raises(net.NetworkError, match="records_relay"):
        await net.self_check(timeout, eps, tries=1, sleep=_noop)
    other = net.endpoints_from_mapping({"task-server": "http://198.51.100.9:8300"})
    with pytest.raises(net.NetworkError) as info:
        await net.self_check(timeout, other, tries=1, sleep=_noop)
    assert "records_relay" not in str(info.value)


async def _noop(_s):
    return None
