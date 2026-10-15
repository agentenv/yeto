"""rl-identity-tiering L9.7 (with L9.5's syncer part): island contract v3. No Ray.

The contract binds the backend identity, the LR schedule, the algorithm's
semantic hash and the run code pack digest. The real Rust syncer (built with
cargo) admits only byte-identical contracts, so it refuses an elastic JOIN whose
code pack differs and a JOIN from an island that still declares contract v2.
"""

from __future__ import annotations

import hashlib
import socket
import subprocess
import time
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.identity import backend_identity
from yeto.rl.elastic_client import ElasticClientConfig, ElasticIslandClient, ElasticProtocolError
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, PluginRef
from yeto.rl.engine.backend_identity import (
    config_island_contract_sha256,
    island_contract_sha256,
    lr_schedule_sha256,
)

from test_rl_integration import syncer_binary  # noqa: F401

IDENTITY = backend_identity(compat_group="nvidia-h100").sha256()
LR = lr_schedule_sha256("constant", 6, 1e-5)
SPEC = AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=0.0, upper=2.0))
PACK_A, PACK_B = "a" * 64, "b" * 64


def _v2_contract(identity: str, lr: str, algorithm_sha256: str) -> str:
    """The contract an island of the previous version declares (domain v2, three slots)."""
    return hashlib.sha256(b"yeto-rl-island-contract-v2\0" + bytes.fromhex(identity)
                          + bytes.fromhex(lr) + bytes.fromhex(algorithm_sha256)).hexdigest()


def test_every_slot_changes_the_contract():
    base = island_contract_sha256(IDENTITY, LR, SPEC.semantic_sha256(), PACK_A)
    other_spec = SPEC.replace(correction=CorrectionSpec(method="tis", lower=0.0, upper=5.0))
    assert len({
        base,
        island_contract_sha256(IDENTITY, LR, SPEC.semantic_sha256(), PACK_B),
        island_contract_sha256(IDENTITY, LR, other_spec.semantic_sha256(), PACK_A),
        island_contract_sha256(IDENTITY, lr_schedule_sha256("constant", 6, 2e-5),
                               SPEC.semantic_sha256(), PACK_A),
        island_contract_sha256(IDENTITY, LR, SPEC.semantic_sha256(), None),
        _v2_contract(IDENTITY, LR, SPEC.sha256()),
    }) == 6
    assert island_contract_sha256(IDENTITY, None, None, None) == IDENTITY  # nothing declared
    with pytest.raises(ValueError, match="code pack digest"):
        island_contract_sha256(IDENTITY, LR, None, "not-a-digest")


def test_plugin_source_hash_does_not_change_the_contract():
    """A plugin source change moves the pack digest, not the algorithm slot."""
    path = "yeto.rl.algos.seq_adv.maxrl_advantage"
    pinned = AlgorithmSpec(plugins=[PluginRef(path, "1" * 64)])
    stale = AlgorithmSpec(plugins=[PluginRef(path, "2" * 64)])
    assert pinned.sha256() != stale.sha256()
    assert (island_contract_sha256(IDENTITY, LR, pinned.semantic_sha256(), PACK_A)
            == island_contract_sha256(IDENTITY, LR, stale.semantic_sha256(), PACK_A))


def test_bridge_config_fields_feed_the_contract():
    config = SimpleNamespace(backend_identity_sha256=IDENTITY, lr_schedule_sha256=LR,
                             algorithm_semantic_sha256=SPEC.semantic_sha256(),
                             code_pack_digest=PACK_A)
    assert config_island_contract_sha256(config) == island_contract_sha256(
        IDENTITY, LR, SPEC.semantic_sha256(), PACK_A)


def test_island_code_pack_digest_prefers_the_run_pack():
    from yeto.rl.adapters.miles.island_entry import _island_code_pack_digest

    assert _island_code_pack_digest(SimpleNamespace(code_pack_digest=PACK_A,
                                                    source_sha256=PACK_B)) == PACK_A
    assert _island_code_pack_digest(SimpleNamespace(code_pack_digest=None,
                                                    source_sha256=PACK_B)) == PACK_B


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("bad", ["other_pack", "v2_island"])
def test_real_syncer_refuses_a_join_with_another_contract(tmp_path, syncer_binary, bad):  # noqa: F811
    """Island 1 joins with contract v3 (pack A) and seeds the base. Island 2 then
    declares pack B, or the v2 contract of the same run; the syncer refuses the
    JOIN and names both contracts."""
    port, key = _free_port(), "contract-v3"
    log = open(tmp_path / "syncer.log", "w")
    syncer = subprocess.Popen(
        [str(syncer_binary), "--port", str(port), "--learners", "2",
         "--total-steps", "1", "--event-tape", str(tmp_path / "tape.jsonl"),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "1.0", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "6", "--syncer-epoch", "0", "--outer-lr", "1.0",
         "--outer-momentum", "0.0"],
        env={"YETO_ISLAND_HMAC_KEY": key, "PATH": "/usr/bin:/bin"},
        stdout=log, stderr=subprocess.STDOUT)
    good = island_contract_sha256(IDENTITY, LR, SPEC.semantic_sha256(), PACK_A)
    other = (island_contract_sha256(IDENTITY, LR, SPEC.semantic_sha256(), PACK_B)
             if bad == "other_pack" else _v2_contract(IDENTITY, LR, SPEC.sha256()))

    def client(island: int, contract: str) -> ElasticIslandClient:
        return ElasticIslandClient(ElasticClientConfig(
            ("127.0.0.1", port), island, lease_s=6.0, join_timeout_s=5.0,
            backend_identity_sha256=contract), key.encode())

    first, second = client(1, good), client(2, other)
    try:
        for _ in range(100):
            try:
                first.join()
                break
            except OSError:
                time.sleep(0.1)
        first.elastic_init([0.0] * 4)
        first.wait_base(newer_than=None, timeout_s=10.0)
        with pytest.raises(ElasticProtocolError, match="backend identity mismatch") as exc:
            second.join()
        assert good in str(exc.value) and other in str(exc.value)
    finally:
        for c in (first, second):
            try:
                c.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        syncer.kill()
        syncer.wait()
        log.close()
