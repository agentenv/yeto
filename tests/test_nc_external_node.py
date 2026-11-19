"""register-node: a node an operator ordered by hand is usable by the per-node
commands (lease record + direct ssh alias) and is never released by this tool.
No real ssh: ``subprocess.run`` is faked."""

from __future__ import annotations

import json
import subprocess

import pytest

from yeto.launch.nc import cli, external_node, node_pool

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleHostKeyOnlyForTests"


class FakeRun:
    def __init__(self, *, login_rc=0, keys=(KEY,)):
        self.calls, self.login_rc, self.keys = [], login_rc, keys

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[0] == "ssh-keyscan":
            out = "".join(f"{argv[-1]} {k}\n" for k in self.keys)
            return subprocess.CompletedProcess(argv, 0, f"# {argv[-1]}:22 SSH-2.0\n{out}", "")
        if argv[0] == "ssh":
            return subprocess.CompletedProcess(argv, self.login_rc, "224\n8\n", "")
        raise AssertionError(f"unexpected command {argv}")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    return tmp_path, fake


def _register(tmp_path, fake, **kw):
    return external_node.register("mi355-190", "ubuntu@192.0.2.190", save_lease=cli.save_lease,
                                  ssh_config=tmp_path / "ssh_config", run=fake, **kw)


def test_register_writes_a_lease_record_the_per_node_commands_load(env):
    tmp_path, fake = env
    report = _register(tmp_path, fake, lease_until="2026-11-19T13:24:00Z")
    assert report["ticket"] == "ext-mi355-190" and report["login_ok"]
    lease = cli.load_lease("ext-mi355-190")
    assert (lease.account, lease.ip, lease.user, lease.gpus) == (0, "192.0.2.190", "ubuntu", 8)
    assert lease.host_key == KEY and lease.lease_until == "2026-11-19T13:24:00Z"
    assert node_pool.host_alias(lease.ticket) == report["alias"] == "yeto-nc-ext-mi355-190"


def test_the_alias_dials_directly_with_the_pinned_host_key(env):
    tmp_path, fake = env
    _register(tmp_path, fake)
    ours = (tmp_path / "state" / "ssh_config").read_text()
    assert "Host yeto-nc-ext-mi355-190\n  HostName 192.0.2.190\n  User ubuntu\n" in ours
    assert "ProxyCommand" not in ours and "IdentityFile" not in ours
    assert "StrictHostKeyChecking yes" in ours
    assert (tmp_path / "state" / "known_hosts").read_text() == f"192.0.2.190 {KEY}\n"
    assert (tmp_path / "ssh_config").read_text().startswith(f"Include {tmp_path / 'state' / 'ssh_config'}\n")
    login = [c for c in fake.calls if c[0] == "ssh"]
    assert len(login) == 1 and "yeto-nc-ext-mi355-190" in login[0]


def test_identity_file_and_port_go_into_the_alias(env):
    tmp_path, fake = env
    _register(tmp_path, fake, identity_file="/keys/op", port=2222)
    ours = (tmp_path / "state" / "ssh_config").read_text()
    assert "  Port 2222\n" in ours and "  IdentityFile /keys/op\n  IdentitiesOnly yes\n" in ours
    assert (tmp_path / "state" / "known_hosts").read_text() == f"[192.0.2.190]:2222 {KEY}\n"
    assert ["ssh-keyscan", "-T", "15", "-p", "2222", "192.0.2.190"] in fake.calls


def test_re_register_replaces_the_pinned_key_and_keeps_other_hosts(env):
    tmp_path, fake = env
    known = tmp_path / "state" / "known_hosts"
    known.parent.mkdir(parents=True, exist_ok=True)
    known.write_text("198.51.100.7 ssh-ed25519 OTHER\n192.0.2.190 ssh-ed25519 OLD\n")
    _register(tmp_path, fake)
    assert known.read_text() == f"198.51.100.7 ssh-ed25519 OTHER\n192.0.2.190 {KEY}\n"


def test_nothing_goes_in_the_order_ledger(env):
    tmp_path, fake = env
    _register(tmp_path, fake)
    assert not cli.ledger().owns(0, "ext-mi355-190")
    assert all("ext-mi355-190" != r.get("ticket") for r in cli.ledger().load())


@pytest.mark.parametrize("cmd", [
    ["down", "--account", "1", "--ticket", "ext-mi355-190"],
    ["delete-ticket", "--config", "/nonexistent.json", "--ticket", "ext-mi355-190", "--yes"],
    ["adopt", "--account", "1", "--ticket", "ext-mi355-190", "--yes"],
])
def test_release_commands_refuse_a_registered_node(env, cmd, capsys):
    tmp_path, fake = env
    _register(tmp_path, fake)
    n = len(fake.calls)
    assert cli.main(cmd) == 2
    assert "not an order of this tool" in capsys.readouterr().err
    assert len(fake.calls) == n                       # nothing was called
    assert cli.load_lease("ext-mi355-190").ip == "192.0.2.190"


def test_unregister_drops_record_and_alias_only(env):
    tmp_path, fake = env
    _register(tmp_path, fake)
    n = len(fake.calls)
    assert cli.main(["unregister-node", "--name", "mi355-190"]) == 0
    assert len(fake.calls) == n                       # the node is not contacted
    assert "ext-mi355-190" not in json.loads(cli.leases_path().read_text())
    assert "yeto-nc-ext-mi355-190" not in (tmp_path / "state" / "ssh_config").read_text()


def test_cli_register_reports_a_failed_login(env, capsys, monkeypatch):
    tmp_path, _ = env
    monkeypatch.setattr(subprocess, "run", FakeRun(login_rc=255))
    monkeypatch.setattr(node_pool, "SSH_CONFIG", tmp_path / "ssh_config")
    assert cli.main(["register-node", "--name", "n1", "--host", "ubuntu@192.0.2.5"]) == 1
    assert "did not log in" in capsys.readouterr().err
    assert cli.load_lease("ext-n1").ip == "192.0.2.5"


@pytest.mark.parametrize("name,host", [
    ("Bad_Name", "ubuntu@192.0.2.1"), ("x.y", "ubuntu@192.0.2.1"), ("ok", "192.0.2.1"),
    ("ok", "ubuntu@"), ("ok", "ubuntu@bad host"), ("ok", "Root;x@192.0.2.1"),
])
def test_bad_name_or_host_is_refused_before_any_call(env, name, host, capsys):
    _, fake = env
    assert cli.main(["register-node", "--name", name, "--host", host]) == 2
    assert fake.calls == []


def test_a_bad_lease_until_is_refused(env):
    _, fake = env
    with pytest.raises(ValueError, match="ISO"):
        external_node.lease_of("n1", "ubuntu@192.0.2.5", lease_until="tomorrow")
    assert fake.calls == []
