"""Island-side elastic client: frames byte-identical to syncer elastic.rs (tasks 0.14). No Ray."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from yeto.protocol import MSG_ERROR, read_frame, write_frame
from yeto.rl import bridge as bridge_mod
from yeto.rl.core import canonical_state
from yeto.rl.elastic_client import (
    MSG_JOIN, MSG_LEAVE, DeltaReady, DeltaTensor, ElasticBase, ElasticClientConfig, ElasticInit,
    ElasticIslandClient, ElasticProtocolError, Join, JoinAck, LeaseHeartbeat, Leave, SampleIndex,
    SampleVerdict, VERDICT_ACCEPT_IS,
    decode, encode, hmac_key_from_env, hmac_sha256,
)

# Produced by the syncer's own ElasticMsg::encode(b"k1") (elastic.rs at s15-interisland-rs
# 101a172c; 19 SAMPLE_INDEX and 24 SAMPLE_VERDICT re-dumped at 1be303e3), the messages of all_frames_roundtrip_and_reject_tampering_and_wrong_key),
# dumped from a scratch copy of that source; see progress.md.
RUST_GOLDEN = {
    15: "03000000000000000100000007000000000000000000000000000440abababababababababababababababababababababababababababababababab0b0000006e76696469612d68313030d76ccd494a9ee9bb15d43cdcafb727ab8d0c651b9b81c6481e80ff4084314b4c",  # s17-elastic-identity: + backend identity; s19-compat: + compat_group
    16: "0300000000000000010000000400000000000000090000000000000007070707070707070707070707070707070707070707070707070707070707070193e467c96823a92f6a48b1fdc63e870e2cb17b0de0968b74790c97c7ad575810",
    17: "03000000000000000100000000d7833586bcbea7773fc51f18c8ded76ac70cb34571860c23d8ab8409dad86a27",
    18: "03000000000000000100000004000000000000000b00000000000000000000000000f43fa2b755da8b28ec797183f4a991453ba9ad27864e96abac41d01a6fcf3abcbd03",
    19: "030000000000000001000000090000000000000003000000000000000101010101010101010101010101010101010101010101010101010101010101020000006731010000007008000000000000001500000073333a2f2f6275636b65742f672e70617271756574000400000000000040000000616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161610101000000000000f83f170000007965746f2e726c2e73616d706c652d696e6465782f7631295b5352ea4d70bf182ed6fa1a98b95658ae152abc6b0f56e218de674a595904",
    24: "030000000000000001120000007374616c655f77697468696e5f626f756e64010000000000000068215e6ef9d59311adef9dafaffce6c5424f9b88e9a6eb4ca254b270b5db2bfd",
    20: "03000000000000000100000009000000000000006400000000000000040000000000000022e1d4320b934424bedffe64ef9bf39a7176a2794b028031c953a164bd51904c",
    21: "03000000000000000100000002000000000000000000803f000000c0490f5332c1f0c2ed1b73ccb1d0e4e134f86b9a2b75ad0ca3df85691045b8c12c",
    22: "03000000000000000100000002000000000000000500000000000000010000000000000003000000000000000000003f0000803e000080bfbfa7704da9c65533c2208a2adf0ae3249de4ac6fef91d8d1b44da8ee328cff6e",
    23: "03000000000000000400000000000000010000000000000000004040ddcbb3df54cd41f060864a9f6390e53d4b9969546c5688e6019e6b49056c6a3e",
}
MSGS = {
    15: Join(3, 1, 7, 2.5, bytes([0xab]) * 32, "nvidia-h100"),
    16: JoinAck(3, 1, 4, 9, bytes([7]) * 32, True),
    17: Leave(3, 1, 0),
    18: LeaseHeartbeat(3, 1, 4, 11, 1.25),
    19: SampleIndex(3, 1, 9, 3, bytes([1]) * 32, "g1", "p", 8, "s3://bucket/g.parquet", 1024,
                    "a" * 64, True, True, 1.5, "yeto.rl.sample-index/v1"),
    24: SampleVerdict(3, VERDICT_ACCEPT_IS, "stale_within_bound", 1),
    20: DeltaReady(3, 1, 9, 100, 4),
    21: ElasticInit(3, 1, (1.0, -2.0)),
    22: DeltaTensor(3, 1, 2, 5, 1, (0.5, 0.25, -1.0)),
    23: ElasticBase(3, 4, (3.0,)),
}


def test_hmac_rfc4231_case_2():
    assert hmac_sha256(b"Jefe", b"what do ya want for nothing?").hex() == (
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")


@pytest.mark.parametrize("msg_type", sorted(MSGS))
def test_frames_match_rust_bytes(msg_type):
    t, payload = encode(MSGS[msg_type], b"k1")
    assert t == msg_type and payload.hex() == RUST_GOLDEN[msg_type]
    assert decode(b"k1", t, bytes.fromhex(RUST_GOLDEN[msg_type])) == MSGS[msg_type]
    with pytest.raises(ElasticProtocolError):
        decode(b"k2", t, payload)
    bad = bytearray(payload); bad[0] ^= 1
    with pytest.raises(ElasticProtocolError):
        decode(b"k1", t, bytes(bad))
    other = MSG_JOIN if t == MSG_LEAVE else MSG_LEAVE  # the MAC binds the type byte
    with pytest.raises(ElasticProtocolError):
        decode(b"k1", other, payload)


def test_key_from_env():
    assert hmac_key_from_env({"YETO_ISLAND_HMAC_KEY": "abc"}) == b"abc"
    with pytest.raises(ElasticProtocolError):
        hmac_key_from_env({})


class FakeSyncer:
    """Socketpair stand-in for elastic_server.rs: JOIN->JOIN_ACK(+base), INIT->base,
    each DELTA_TENSOR -> base' = base + update (version + 1)."""

    def __init__(self, key=b"k", catch_up=False, refuse=False, base=None):
        self.a, self.b = socket.socketpair()
        self.key, self.catch_up, self.refuse = key, catch_up, refuse
        self.base, self.version = base, 0
        self.frames = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _send_base(self):
        write_frame(self.b, *encode(ElasticBase(0, self.version, tuple(self.base)), self.key))

    def _serve(self):
        while True:
            try:
                t, p = read_frame(self.b, 1 << 20)
            except Exception:
                return
            msg = decode(self.key, t, p)
            self.frames.append(msg)
            if isinstance(msg, Join):
                if self.refuse:
                    write_frame(self.b, MSG_ERROR, b"fenced: syncer_epoch 0 < 5")
                    continue
                write_frame(self.b, *encode(JoinAck(0, msg.island_id, 1, self.version,
                                                    bytes(32), self.catch_up), self.key))
                if self.base is not None:
                    self._send_base()
            elif isinstance(msg, ElasticInit):
                if self.base is None:
                    self.base = list(msg.params)
                self._send_base()
            elif isinstance(msg, DeltaTensor):
                self.base = [b + u for b, u in zip(self.base, msg.update)]
                self.version += 1
                self._send_base()

    def connect(self, addr):
        return self.a


def _client(fake, **kw):
    cfg = ElasticClientConfig(("x", 0), island_id=3, capacity=2.0, lease_s=0.15,
                              join_timeout_s=2.0, **kw)
    return ElasticIslandClient(cfg, fake.key, connect=fake.connect)


def test_join_heartbeat_delta_leave():
    fake = FakeSyncer(catch_up=True)
    c = _client(fake)
    ack = c.join()
    assert ack.catch_up and c.heartbeat_period_s == pytest.approx(0.05)
    time.sleep(0.25)
    c.delta_ready(base_version=0, c_tokens=100, c_steps=4)
    c.leave()
    time.sleep(0.2)
    c.close()
    kinds = [type(m).__name__ for m in fake.frames]
    assert kinds[0] == "Join" and fake.frames[0].capacity == 2.0
    assert kinds.count("LeaseHeartbeat") >= 2
    assert DeltaReady(0, 3, 0, 100, 4) in fake.frames
    assert kinds[-1] == "Leave"  # no heartbeat after LEAVE


def test_join_refused_fails_fast():
    c = _client(FakeSyncer(refuse=True))
    t0 = time.monotonic()
    with pytest.raises(ElasticProtocolError, match="fenced"):
        c.join()
    assert time.monotonic() - t0 < 1.0
    c.close()


NAMES = ("base_model.model.a.lora_A.weight", "base_model.model.z.lora_B.weight")


def _state(version, a, z):
    return canonical_state(version, {NAMES[0]: torch.tensor([a], dtype=torch.float32),
                                     NAMES[1]: torch.tensor(z, dtype=torch.float32).reshape(2, 1)},
                           base_model_revision="a" * 40, lora_config_hash="b" * 64)


class Runtime:
    """Each local round adds +1 to every parameter of the applied policy."""

    def __init__(self):
        self.applied, self.rounds, self.shut, self.cur = [], [], False, None

    def initialize(self):
        return _state(0, [1.0, 2.0], [3.0, 4.0])

    def apply_global_policy(self, state):
        self.applied.append(state.policy_version)
        self.cur = state

    def run_local_round(self, *, expected_policy_version, groups, samples_per_group,
                        optimizer_steps):
        self.rounds.append(expected_policy_version)
        return SimpleNamespace(action_tokens=50 + len(self.rounds))

    def export_local_policy(self):
        t = {k: v + 1.0 for k, v in self.cur.tensors.items()}
        return canonical_state(self.cur.policy_version, t, base_model_revision="a" * 40,
                               lora_config_hash="b" * 64)

    def record_local_round(self, stats):
        pass

    def shutdown(self):
        self.shut = True


def _cfg(tmp_path):
    return SimpleNamespace(learner_id=3, global_rounds=2, groups_per_round=1, samples_per_group=1,
                           local_optimizer_steps=4, event_tape=str(tmp_path / "tape.jsonl"),
                           syncer_addr=("x", 0), expected_specs=())


def test_elastic_bridge_init_delta_base(tmp_path):
    fake = FakeSyncer()
    rt = Runtime()
    br = bridge_mod.make_island_bridge(rt, _cfg(tmp_path), island_scheduling="elastic",
                                       elastic_client=_client(fake), base_wait_s=5.0)
    assert br.run() == 2
    assert rt.rounds == [0, 1] and rt.applied == [0, 1, 2] and rt.shut
    time.sleep(0.05)
    inits = [m for m in fake.frames if isinstance(m, ElasticInit)]
    assert inits == [ElasticInit(0, 3, (1.0, 2.0, 3.0, 4.0))]  # canonical (sorted) order
    deltas = [m for m in fake.frames if isinstance(m, DeltaTensor)]
    assert [(d.base_version, d.c_tokens, d.c_steps, d.update) for d in deltas] == [
        (0, 51, 4, (1.0, 1.0, 1.0, 1.0)), (1, 52, 4, (1.0, 1.0, 1.0, 1.0))]
    assert fake.base == [3.0, 4.0, 5.0, 6.0]
    assert isinstance(fake.frames[-1], Leave)
    assert not any(isinstance(m, DeltaReady) for m in fake.frames)  # tensor run: no DELTA_READY


def test_catch_up_island_uses_existing_base(tmp_path):
    fake = FakeSyncer(catch_up=True, base=[9.0, 9.0, 9.0, 9.0])
    rt = Runtime()
    cfg = _cfg(tmp_path); cfg.global_rounds = 1
    bridge_mod.make_island_bridge(rt, cfg, island_scheduling="elastic",
                                  elastic_client=_client(fake), base_wait_s=5.0).run()
    assert not any(isinstance(m, ElasticInit) for m in fake.frames)
    assert rt.applied == [0, 1]


def test_legacy_factory_is_strict_bridge(monkeypatch):
    made = []
    monkeypatch.setattr(bridge_mod, "StrictRlBridge", lambda r, c: made.append((r, c)) or "strict")
    assert bridge_mod.make_island_bridge("rt", "cfg") == "strict" and made == [("rt", "cfg")]
    with pytest.raises(ValueError):
        bridge_mod.make_island_bridge("rt", "cfg", island_scheduling="auto")


def test_legacy_never_imports_elastic_client():
    code = ("import sys; import yeto.rl.bridge, yeto.protocol; "
            "assert 'yeto.rl.elastic_client' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


# ----------------------------------------------------------------- 0.19 SAMPLE_INDEX wire alignment
def _entry(island="1", version=9, policy_hash="01" * 32, group="g1", has_lp=True):
    from yeto.rl.engine.sample_pool import SampleIndexEntry

    return SampleIndexEntry(island, version, 3, policy_hash, group, "p", 8, "s3://bucket/g.parquet",
                            1024, "a" * 64, has_lp, True, 1.5)


def test_sample_index_entry_roundtrip_matches_rust_frame():
    from yeto.rl.elastic_client import sample_index_from_entry, sample_index_to_entry

    msg = sample_index_from_entry(_entry(), syncer_epoch=3)
    assert msg == MSGS[19]
    assert encode(msg, b"k1")[1].hex() == RUST_GOLDEN[19]
    assert sample_index_to_entry(msg) == _entry()
    named = sample_index_from_entry(_entry(island="island-a"), syncer_epoch=3, island_ids={"island-a": 1})
    assert named == MSGS[19]
    assert sample_index_to_entry(named, island_names={1: "island-a"}).island_id == "island-a"


def test_sample_index_conversion_rejects_bad_ids_and_hashes():
    from yeto.rl.elastic_client import sample_index_from_entry

    with pytest.raises(ElasticProtocolError, match="numeric id"):
        sample_index_from_entry(_entry(island="island-a"), syncer_epoch=0)
    with pytest.raises(ElasticProtocolError, match="u32"):
        sample_index_from_entry(_entry(island=str(1 << 32)), syncer_epoch=0)
    with pytest.raises(ElasticProtocolError, match="not hex"):
        sample_index_from_entry(_entry(policy_hash="zz" * 32), syncer_epoch=0)
    with pytest.raises(ElasticProtocolError, match="32 bytes"):
        sample_index_from_entry(_entry(policy_hash="01" * 16), syncer_epoch=0)


def test_sample_verdict_names():
    assert MSGS[24].verdict_name == "ACCEPT_IS"
    assert SampleVerdict(0, 9, "", 0).verdict_name == "UNKNOWN(9)"


def _f32_hash(values):
    import hashlib
    import struct as _struct

    return hashlib.sha256(_struct.pack(f"<{len(values)}f", *values)).hexdigest()


@pytest.mark.skipif(not __import__("os").environ.get("YETO_TEST_ELASTIC_SYNCER"),
                    reason="needs YETO_TEST_ELASTIC_SYNCER (syncer binary with elastic mode)")
def test_sample_index_three_verdicts_against_real_rust_syncer(tmp_path):
    """One island, two outer versions: ACCEPT / ACCEPT_IS (needs IS correction) / REJECT,
    then the status.json the syncer exports is parsed by the 0.18 reader."""
    import json
    import os

    from yeto.rl.elastic_client import VERDICT_ACCEPT, VERDICT_REJECT
    from yeto.rl.engine.island_status import read_syncer_status, scheduling_probe_from_status

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = "sample-e2e"
    tape = tmp_path / "syncer-tape.jsonl"
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_ELASTIC_SYNCER"], "--port", str(port), "--learners", "1",
         "--total-steps", "3", "--event-tape", str(tape),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "0.75", "--carry-gamma", "0.5",
         "--soft-deadline-s", "30", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "6", "--syncer-epoch", "2"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(tmp_path / "syncer.log", "w"), stderr=subprocess.STDOUT)
    c = ElasticIslandClient(ElasticClientConfig(("127.0.0.1", port), 1, lease_s=6.0, syncer_epoch=2),
                            key.encode())
    try:
        for _ in range(50):
            try:
                c.join()
                break
            except OSError:
                time.sleep(0.1)
        c.elastic_init([0.0] * 4)
        base0 = c.wait_base(newer_than=None, timeout_s=10.0)
        c.delta_tensor(base_version=base0.outer_version, c_tokens=10, c_steps=1, update=[1.0] * 4)
        base1 = c.wait_base(newer_than=base0.outer_version, timeout_s=10.0)
        assert base1 is not None, c.errors
        h0, h1 = _f32_hash(base0.params), _f32_hash(base1.params)
        verdicts = [c.sample_index(_entry(version=base1.outer_version, policy_hash=h1, group="g1")),
                    c.sample_index(_entry(version=base0.outer_version, policy_hash=h0, group="g0")),
                    c.sample_index(_entry(version=base0.outer_version, policy_hash=h1, group="bad"))]
        assert [(v.verdict, v.reason, v.outer_lag) for v in verdicts] == [
            (VERDICT_ACCEPT, "same_version", 0), (VERDICT_ACCEPT_IS, "stale_within_bound", 1),
            (VERDICT_REJECT, "policy_hash_mismatch", 1)]
        status = None
        for _ in range(50):
            status = read_syncer_status(tmp_path)
            if status is not None and status["sample_index_count"] == 2:
                break
            time.sleep(0.1)
        assert status is not None and status["syncer_epoch"] == 2, status
        assert status["policy_hash"] == h1 and status["outer_version"] == base1.outer_version
        probe = scheduling_probe_from_status(tmp_path, island_id=1)()
        assert probe["capacity"] == 1.0 and probe["syncer_epoch"] == 2
        idx = (tmp_path / "sample_index.jsonl").read_text().splitlines()
        from yeto.rl.engine.sample_pool import SampleIndexEntry

        rows = [json.loads(l) for l in idx]
        assert [(r["verdict"], r["entry"]["group_id"]) for r in rows] == [("ACCEPT", "g1"), ("ACCEPT_IS", "g0")]
        assert SampleIndexEntry.from_json(rows[1]["entry"]) == _entry(version=base0.outer_version,
                                                                      policy_hash=h0, group="g0")
    finally:
        c.close()
        syncer.kill()
        syncer.wait(5)
