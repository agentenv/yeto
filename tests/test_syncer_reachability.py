"""s13-g3-modal-20261007b: two Modal islands redialed a Nebius syncer (actor
29400 + critic 29401) silently for the whole 900 s stall timeout. The launcher
now probes every syncer port before starting any island, and the island-side
client reports its dial progress."""
import json
import socket
from types import SimpleNamespace

import pytest

import yeto.launcher as launcher
from yeto.protocol import SyncerClient


def _args(critic: bool):
    spec = {"execution": {"needs_critic": critic}}
    return SimpleNamespace(training_mode="rl", rl_algorithm_spec_json=json.dumps(spec),
                           syncer_region="nebius/eu-north1")


def test_probe_covers_actor_and_critic_ports():
    seen = []
    details = launcher.probe_syncer_ports(_args(True), "198.51.100.7",
                                          probe=lambda h, p: (seen.append((h, p)) or (True, f"{h}:{p} ok")))
    assert seen == [("198.51.100.7", 29400), ("198.51.100.7", 29401)]
    assert len(details) == 2


def test_probe_actor_only_without_critic():
    seen = []
    launcher.probe_syncer_ports(_args(False), "h", probe=lambda h, p: (seen.append(p) or (True, "ok")))
    assert seen == [29400]


def test_closed_critic_port_fails_before_any_island():
    def probe(host, port):
        return (port == 29400, f"{host}:{port} {'ok' if port == 29400 else 'unreachable'}")

    with pytest.raises(RuntimeError, match="syncer port 29401 not reachable"):
        launcher.probe_syncer_ports(_args(True), "h", probe=probe)


def test_client_reports_dial_progress_and_gives_up(capsys, monkeypatch):
    # A closed local port: the client must say what it dials and why it fails,
    # not redial silently.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    client = SyncerClient.__new__(SyncerClient)
    client.addr = ("127.0.0.1", port)
    client.learner_id = 1
    client.connect_timeout = 0.5
    monkeypatch.setattr("yeto.protocol.time.sleep", lambda s: None)
    with pytest.raises(ConnectionError, match="attempt"):
        client._connect_one()
    err = capsys.readouterr().err
    assert f"dialing syncer 127.0.0.1:{port}" in err
    assert "not reachable yet (attempt 1" in err


def test_client_reports_connection(capsys):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = SyncerClient.__new__(SyncerClient)
    client.addr = server.getsockname()
    client.learner_id = 0
    client.connect_timeout = 5.0
    sock = client._connect_one()
    sock.close()
    server.close()
    assert "connected to syncer" in capsys.readouterr().err


def test_island_syncer_addrs_line():
    assert launcher.island_syncer_addrs({"SYNCER_ADDR": "1.2.3.4:29400", "CRITIC_SYNCER_ADDR": "1.2.3.4:29401",
                                         "HF_TOKEN": "secret"}) == \
        "SYNCER_ADDR=1.2.3.4:29400 CRITIC_SYNCER_ADDR=1.2.3.4:29401"
    assert launcher.island_syncer_addrs(None) == "no syncer"


def test_island_probe_names_the_unreachable_critic_syncer(capsys):
    from yeto.rl.learner import probe_syncers

    class _C:
        def close(self):
            pass

    def connect(addr, timeout=None):
        if addr[1] == 29401:
            raise ConnectionRefusedError("refused")
        return _C()

    args = SimpleNamespace(syncer="203.0.113.4:29400", critic_syncer="203.0.113.4:29401")
    with pytest.raises(ConnectionError, match="203.0.113.4:29401 unreachable"):
        probe_syncers(args, connect=connect, attempts=3, sleep=lambda s: None)
    assert "203.0.113.4:29400 reachable" in capsys.readouterr().out
    assert probe_syncers(SimpleNamespace(syncer=None, critic_syncer=None)) == []
