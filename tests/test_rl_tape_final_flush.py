"""The trainer's last tape upload at shutdown (yeto issue #14, k31-mfu16-n1): the
final drain uploads every complete line, retries a failed chunk under the same
key, is bounded in wall time even when a store write hangs, and holds a teardown
SIGINT until it ends instead of dying inside the last R2 put."""

from __future__ import annotations

import json
import os
import signal
import threading
import time

import pytest

from yeto.rl.engine import tape_upload as tu
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.tape_upload import TapeUploader, chunk_key


def _append(path, *records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _stored(objects) -> bytes:
    return b"".join(objects.read(k) for k in objects.list("tapes/"))


def test_final_drain_uploads_a_backlog_larger_than_one_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "MAX_CHUNK_BYTES", 64)
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "t.jsonl"
    _append(tape, *({"event": "x", "i": i} for i in range(20)))  # ~20 chunks of 64 bytes
    up = TapeUploader(objects, tape, island="trainer", incarnation="a", interval_s=999)
    assert up.stop(timeout_s=10) is True
    assert _stored(objects) == tape.read_bytes()  # the old stop() uploaded one chunk only


def test_final_drain_retries_a_failed_chunk_under_the_same_key(tmp_path):
    class Flaky:
        def __init__(self):
            self.keys, self.data = [], {}

        def write(self, key, data):
            self.keys.append(key)
            if len(self.keys) <= 2:
                raise OSError("read timeout")  # the put may even have landed: same key again
            self.data[key] = data

    tape = tmp_path / "t.jsonl"
    _append(tape, {"event": "rl_learner_finalized"})
    store, logs = Flaky(), []
    up = TapeUploader(store, tape, island="trainer", incarnation="a", interval_s=999, log=logs.append)
    assert up.stop(timeout_s=10, sleep=lambda s: None) is True
    assert store.keys == [chunk_key("trainer", "a", 0)] * 3
    assert store.data[chunk_key("trainer", "a", 0)] == tape.read_bytes()
    assert up.failures == 2


def test_final_drain_is_bounded_when_a_store_write_hangs(tmp_path):
    release = threading.Event()

    class Hung:
        def write(self, key, data):
            release.wait(30)  # an R2 put stuck reading the response

    tape = tmp_path / "t.jsonl"
    _append(tape, {"event": "rl_learner_finalized"})
    logs = []
    up = TapeUploader(Hung(), tape, island="trainer", incarnation="a", interval_s=999, log=logs.append)
    t0 = time.monotonic()
    try:
        assert up.stop(timeout_s=0.3) is False
        assert time.monotonic() - t0 < 5
        assert any("final upload incomplete" in line for line in logs)
    finally:
        release.set()


def test_a_torn_last_line_does_not_make_the_drain_incomplete(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "t.jsonl"
    _append(tape, {"event": "a"})
    with open(tape, "a") as f:
        f.write('{"event": "tor')
    up = TapeUploader(objects, tape, island="trainer", incarnation="a", interval_s=999)
    assert up.stop(timeout_s=5) is True
    assert _stored(objects) == b'{"event": "a"}\n'


def test_zero_budget_turns_the_drain_off(tmp_path, monkeypatch):
    monkeypatch.setenv(tu.FINAL_FLUSH_ENV, "0")
    assert tu.final_flush_s() == 0.0
    monkeypatch.setenv(tu.FINAL_FLUSH_ENV, "12.5")
    assert tu.final_flush_s() == 12.5
    monkeypatch.delenv(tu.FINAL_FLUSH_ENV)
    assert tu.final_flush_s() == tu.DEFAULT_FINAL_FLUSH_S


def test_sigint_during_the_final_put_is_held_until_the_chunk_is_stored(tmp_path):
    """n1: the teardown SIGINT reached the head inside put_object; now the put
    finishes, the chunk is stored, then the SIGINT is delivered as before."""
    objects = DirectoryObjectStore(tmp_path / "store")

    class SlowPut:
        def write(self, key, data):
            os.kill(os.getpid(), signal.SIGINT)  # the teardown, mid-request
            time.sleep(0.2)
            objects.write(key, data)

    tape = tmp_path / "t.jsonl"
    _append(tape, {"event": "rl_learner_finalized"})
    logs = []
    up = TapeUploader(SlowPut(), tape, island="trainer", incarnation="a", interval_s=999, log=logs.append)
    # An earlier test in the same process may leave another SIGINT handler installed.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            up.stop(timeout_s=10)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)
    assert _stored(objects) == tape.read_bytes()
    assert any("held until the final tape upload ends" in line for line in logs)


def test_held_signals_restore_handlers_and_redeliver_once(monkeypatch):
    seen = []
    previous = signal.signal(signal.SIGTERM, lambda num, frame: seen.append(num))
    try:
        with tu.held_signals(log=lambda line: None) as arrived:
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            assert seen == []  # held
        assert arrived == [signal.SIGTERM, signal.SIGTERM]
        assert seen == [signal.SIGTERM]  # delivered again, once
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_held_signals_off_the_main_thread_is_a_no_op():
    out = []

    def body():
        with tu.held_signals() as arrived:
            out.append(arrived)

    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert out == [[]]
