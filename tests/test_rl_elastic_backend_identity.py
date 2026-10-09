"""Elastic JOIN carries the backend identity hash (yeto-framework-decoupling 6.2a). No Ray.

The JOIN golden frame is the syncer's own encode output
(syncer/src/elastic.rs ``join_frame_with_backend_identity_golden_and_old_format_refused``);
the real-syncer test needs YETO_TEST_ELASTIC_SYNCER (a built ``yeto-syncer``).
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import time
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.identity import backend_identity
from yeto.rl.elastic_client import (
    MSG_JOIN, ElasticClientConfig, ElasticIslandClient, ElasticProtocolError, Join, decode, encode,
    seal,
)
from yeto.rl.engine.backend_identity import BackendIdentity, param_map_sha256

JOIN_GOLDEN = ("03000000000000000100000007000000000000000000000000000440"
               + "ab" * 32
               + "3d2faaa5418d42914e93471c8f79f008699aee2190bd1d505af97476e9ec21a7")

MILES = backend_identity("ports")
VERL = BackendIdentity("verl", "0" * 40, "nvidia", param_map_sha256({"verl": "map"}))


def test_join_golden_frame_matches_rust():
    msg = Join(3, 1, 7, 2.5, bytes([0xab]) * 32)
    t, payload = encode(msg, b"k1")
    assert t == MSG_JOIN and payload.hex() == JOIN_GOLDEN
    assert decode(b"k1", t, payload) == msg
    tampered = bytearray(payload)
    tampered[28] ^= 1  # first identity byte is under the HMAC
    with pytest.raises(ElasticProtocolError):
        decode(b"k1", t, bytes(tampered))


def test_old_join_frame_without_identity_is_refused():
    old = seal(b"k1", MSG_JOIN, struct.pack("<QIQd", 3, 1, 7, 2.5))
    with pytest.raises(ElasticProtocolError, match="without backend identity"):
        decode(b"k1", MSG_JOIN, old)


def test_config_identity_bytes():
    assert ElasticClientConfig(("x", 0), 1).backend_identity_bytes() == bytes(32)
    cfg = ElasticClientConfig(("x", 0), 1, backend_identity_sha256=MILES.sha256())
    assert cfg.backend_identity_bytes() == bytes.fromhex(MILES.sha256())
    with pytest.raises(ElasticProtocolError):
        ElasticClientConfig(("x", 0), 1, backend_identity_sha256="ab").backend_identity_bytes()


def test_bridges_pass_identity_to_the_elastic_client(monkeypatch):
    """Both elastic bridge paths build the client with BridgeConfig.backend_identity_sha256."""
    import yeto.rl.elastic_client as ec
    from yeto.rl.bridge import make_island_bridge
    from yeto.rl.engine.bridges import ElasticAvgSync

    seen = []

    class Spy:
        def __init__(self, config, key, **kw):
            seen.append(config.backend_identity_sha256)

    monkeypatch.setattr(ec, "ElasticIslandClient", Spy)
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k")
    cfg = SimpleNamespace(syncer_addr=("x", 0), learner_id=1, backend_identity_sha256=MILES.sha256(),
                          expected_specs=(), global_rounds=1, local_optimizer_steps=1)
    try:
        make_island_bridge(SimpleNamespace(), cfg, island_scheduling="elastic")
    except Exception:  # noqa: BLE001 -- only the client construction matters here
        pass
    ElasticAvgSync(cfg)._client()
    assert seen == [MILES.sha256(), MILES.sha256()]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not os.environ.get("YETO_TEST_ELASTIC_SYNCER"),
                    reason="needs YETO_TEST_ELASTIC_SYNCER (syncer binary with elastic mode)")
def test_real_syncer_refuses_verl_island_in_miles_session(tmp_path):
    """Real Rust syncer, three fake islands: Miles island 1 joins and seeds the base, a
    verl island 2 is refused with both hashes in the error, Miles island 3 joins as
    before and one outer step merges islands 1 and 3."""
    port, key = _free_port(), "ident-e2e"
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_ELASTIC_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", "1", "--event-tape", str(tmp_path / "tape.jsonl"),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "1.0", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "6", "--syncer-epoch", "0", "--outer-lr", "1.0", "--outer-momentum", "0.0"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(tmp_path / "syncer.log", "w"), stderr=subprocess.STDOUT)

    def client(island, ident):
        return ElasticIslandClient(ElasticClientConfig(
            ("127.0.0.1", port), island, lease_s=6.0, join_timeout_s=5.0,
            backend_identity_sha256=ident.sha256()), key.encode())

    a, v, b = client(1, MILES), client(2, VERL), client(3, MILES)
    try:
        for _ in range(50):
            try:
                a.join()
                break
            except OSError:
                time.sleep(0.1)
        a.elastic_init([0.0] * 4)
        base0 = a.wait_base(newer_than=None, timeout_s=10.0)
        with pytest.raises(ElasticProtocolError, match="backend identity mismatch") as exc:
            v.join()
        text = str(exc.value)
        assert VERL.sha256() in text and MILES.sha256() in text and "Miles vs verl" in text
        assert b.join() is not None  # same backend: unchanged behaviour
        b.wait_base(newer_than=None, timeout_s=10.0)
        a.delta_tensor(base_version=base0.outer_version, c_tokens=10, c_steps=1, update=[1.0] * 4)
        b.delta_tensor(base_version=base0.outer_version, c_tokens=10, c_steps=1, update=[3.0] * 4)
        base1 = a.wait_base(newer_than=base0.outer_version, timeout_s=10.0)
        assert base1 is not None and base1.outer_version == 1, a.errors
        assert list(base1.params) == pytest.approx([2.0] * 4)  # equal weights, outer lr 1
    finally:
        for c in (a, v, b):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        syncer.kill()
        syncer.wait(5)


@pytest.mark.skipif(not os.environ.get("YETO_TEST_ELASTIC_SYNCER"),
                    reason="needs YETO_TEST_ELASTIC_SYNCER (syncer binary with elastic mode)")
def test_real_syncer_refuses_same_backend_with_other_lr_schedule(tmp_path):
    """S17 N17: two Miles islands whose LR schedules differ declare different bound
    identities (island_contract_sha256); the elastic syncer refuses the second JOIN."""
    from yeto.rl.engine.backend_identity import island_contract_sha256, lr_schedule_sha256

    port, key = _free_port(), "lr-e2e"
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_ELASTIC_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", "1", "--event-tape", str(tmp_path / "tape.jsonl"),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "1.0", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "6", "--syncer-epoch", "0", "--outer-lr", "1.0", "--outer-momentum", "0.0"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(tmp_path / "syncer.log", "w"), stderr=subprocess.STDOUT)
    const = island_contract_sha256(MILES.sha256(), lr_schedule_sha256("constant", 6, 1e-5))
    other = island_contract_sha256(MILES.sha256(), lr_schedule_sha256("constant", 6, 2e-5))

    def client(island, ident):
        return ElasticIslandClient(ElasticClientConfig(
            ("127.0.0.1", port), island, lease_s=6.0, join_timeout_s=5.0,
            backend_identity_sha256=ident), key.encode())

    a, bad = client(1, const), client(2, other)
    try:
        for _ in range(50):
            try:
                a.join()
                break
            except OSError:
                time.sleep(0.1)
        a.elastic_init([0.0] * 4)
        a.wait_base(newer_than=None, timeout_s=10.0)
        with pytest.raises(ElasticProtocolError, match="backend identity mismatch") as exc:
            bad.join()
        assert const in str(exc.value) and other in str(exc.value)
    finally:
        for c in (a, bad):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        syncer.kill()
        syncer.wait()
