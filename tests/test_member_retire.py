"""MEMBER_RETIRE (``syncer/src/member_retire.rs``): the controller retires one
syncer member now; the Python codec matches the Rust golden bytes, and on the
real syncer the retired island's positions go straight to another island."""

from __future__ import annotations

import pytest

from tests.test_dispense_syncer import FP, KEY, Syncer, syncer_binary  # noqa: F401 - the fixture
from yeto.rl.engine import task_dispense as td


def test_wire_bytes_match_the_rust_golden_vectors():
    """Same vectors as ``member_retire::tests::wire_golden_vectors``."""
    body = td.encode_member_retire(0x8000_0007, 0x0102030405060708, "scale_down")
    assert body.hex() == "0100070000800108070605040302010a007363616c655f646f776e"
    sealed = td.seal_member_retire(body, b"k")
    assert sealed[len(body):].hex() == "e9c1742d535f0684c680c579c1091feee6d588c661aff106e512642b0acfddcf"
    assert td.seal_member_retire(body, None) == body
    assert td.decode_member_retired(bytes.fromhex("0009000000000000000600000000000000")) == {
        "status": "retired", "incarnation": 9, "reclaimed": 6}
    current = td.encode_member_retire(5, None)
    assert current[6] == 0 and current[7:15] == bytes(8)
    with pytest.raises(ValueError):
        td.encode_member_retire(1, None, "x" * 257)


def _conn(port, learner_id, incarnation, key=KEY):
    return td.DispenseConnection(("127.0.0.1", port), learner_id=learner_id, incarnation=incarnation, rows=8,
                                 fingerprint=FP, key=key, timeout_s=5.0)


def test_retire_on_the_real_syncer_releases_at_once_and_fences(syncer_binary, tmp_path):  # noqa: F811
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", lease_s=600.0)
    try:
        a, b = _conn(syncer.port, 1, 11), _conn(syncer.port, 2, 22)
        grant = a.claim(1, 3)
        assert grant.ranges == ((0, 3),)
        address = ("127.0.0.1", syncer.port)
        with pytest.raises(td.DispenseError, match="HMAC"):
            td.retire_member(address, 1, key=b"wrong")
        assert td.retire_member(address, 1, reason="scale_down", key=KEY) == {
            "status": "retired", "incarnation": 11, "reclaimed": 3}
        assert td.retire_member(address, 1, 11, key=KEY)["status"] == "already_retired"
        assert td.retire_member(address, 1, key=KEY)["status"] == "no_member"
        # no wait for the 600 s lease: the next claim of another island takes them first
        taken = sorted(p for start, end in b.claim(1, 4).ranges for p in range(start, end))
        assert taken == [0, 1, 2, 3]
        with pytest.raises(td.DispenseFenced, match="retired by the controller"):
            a.claim(2, 1)
        a.close()
        b.close()
        reclaims = [r for r in syncer.records() if r.get("kind") == "dispense_reclaim"]
        assert [(r["reason"], r["positions"]) for r in reclaims] == [("retired", 3)]
    finally:
        syncer.stop()


def test_without_a_key_only_the_head_itself_may_retire(syncer_binary, tmp_path):  # noqa: F811
    syncer = Syncer(syncer_binary, tmp_path, key=None)
    try:
        a = _conn(syncer.port, 3, 33, key=None)
        a.claim(1, 2)
        assert td.retire_member(("127.0.0.1", syncer.port), 3)["reclaimed"] == 2
        a.close()
    finally:
        syncer.stop()


def test_a_retired_task_source_stops_instead_of_rejoining(syncer_binary, tmp_path):  # noqa: F811
    """A task source retired by the controller does not reopen under a new
    incarnation (it would take work again after a scale-down)."""
    import time

    syncer = Syncer(syncer_binary, tmp_path, lease_s=600.0)
    island = td.sampler_holder_id("nc-2")
    try:
        events: list = []
        connect = td.tcp_connector(("127.0.0.1", syncer.port), learner_id=island, rows=100,
                                   fingerprint=FP, key=KEY, timeout_s=5.0)
        source = td.TaskSource(connect=connect, block=4, prefetch_blocks=1, retry_s=0.05,
                               incarnation=lambda: 0x7777, emit=lambda kind, **f: events.append(kind)).start()
        assert len(source.take(4, timeout_s=5)) == 4
        answer = td.retire_member(("127.0.0.1", syncer.port), island, reason="scale_down", key=KEY)
        assert answer["status"] == "retired" and answer["incarnation"] == 0x7777 and answer["reclaimed"] >= 4
        with pytest.raises(td.DispenseFenced, match=r"\(retired by the controller\)$"):
            while True:
                source.take(4, timeout_s=5)
        assert "rl_dispense_retired" in events
        time.sleep(0.5)
        opens = [r for r in syncer.records() if r.get("kind") == "dispense_open"]
        assert [r["incarnation"] for r in opens] == [f"{0x7777:016x}"]
        source.close(release=False)
    finally:
        syncer.stop()
