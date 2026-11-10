"""Sampler side of the dashboard (k31/dashboard-rl 6fc76e99 + fb729db8, sender parts only):
the engine sampler's round summary and its event-tape upload to the run store."""

from __future__ import annotations

import json

import pytest

from yeto.rl.engine.engine_sampler import round_summary, zero_variance
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.tape_upload import TapeUploader, chunk_key, store_objects, upload_interval_s


T0 = 1_790_000_000.0


def _append(path, *records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _round(sid, rid, t):
    return {"event": "rl_sampler_round", "sampler_id": sid, "rollout_id": rid, "groups": 4,
            "generate_s": 30.0, "policy_version": 0, "version_lag": 0, "time_unix": t}


def test_uploader_writes_complete_lines_in_numbered_chunks(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "events.a1.jsonl"
    up = TapeUploader(objects, tape, island="nc/s1", incarnation="a1", interval_s=999)
    assert up.flush() == 0  # no file yet
    _append(tape, _round("s1", 0, T0))
    with open(tape, "a") as f:
        f.write('{"event": "rl_sampler_ro')  # torn last line: waits
    assert up.flush() > 0
    _append(tape, {"x": 1})  # completes the torn line... as garbage + a record
    up.flush()
    keys = objects.list("tapes/")
    assert keys == [chunk_key("nc/s1", "a1", 0), chunk_key("nc/s1", "a1", 1)]
    assert keys[0] == "tapes/nc_s1/a1/00000000.jsonl"
    assert b"".join(objects.read(k) for k in keys) == tape.read_bytes()


def test_uploader_failure_never_raises_and_retries(tmp_path):
    class Flaky:
        def __init__(self):
            self.calls, self.data = 0, {}

        def write(self, key, data):
            self.calls += 1
            if self.calls == 1:
                raise OSError("store down")
            self.data[key] = data

    tape = tmp_path / "t.jsonl"
    _append(tape, _round("s1", 0, T0))
    obj, logs = Flaky(), []
    up = TapeUploader(obj, tape, island="s1", incarnation="a", log=logs.append)
    assert up.flush() == 0 and up.failures == 1 and logs
    assert up.flush() > 0 and list(obj.data) == [chunk_key("s1", "a", 0)]


def test_uploader_thread_and_final_flush(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "t.jsonl"
    up = TapeUploader(objects, tape, island="s1", incarnation="a", interval_s=0.01).start()
    _append(tape, _round("s1", 0, T0), _round("s1", 1, T0 + 30))
    up.stop()
    assert b"".join(objects.read(k) for k in objects.list("tapes/")) == tape.read_bytes()


def test_upload_interval_env_and_off():
    assert upload_interval_s(None, {}) == 30.0
    assert upload_interval_s(5, {}) == 5.0
    assert upload_interval_s(5, {"YETO_RL_TAPE_UPLOAD_S": "0"}) is None


def test_store_objects_of_a_directory_store(tmp_path):
    class S:
        root = tmp_path

    assert isinstance(store_objects(S()), DirectoryObjectStore)
    sentinel = object()

    class O:
        objects = sentinel

    assert store_objects(O()) is sentinel



def _group(rewards, statuses=None, length=10):
    statuses = statuses or ["completed"] * len(rewards)
    return [{"reward": r, "status": s, "response_length": length + i} for i, (r, s) in enumerate(zip(rewards, statuses))]


def test_round_summary_reward_length_truncation_zero_variance():
    scored = [(None, _group([1.0, 0.0, 1.0, 0.0], ["completed", "truncated", "completed", "completed"])),
              (None, _group([1.0, 1.0, 1.0, 1.0]))]
    s = round_summary(scored)
    assert s["reward_mean"] == pytest.approx(0.75)
    assert s["reward_std"] == pytest.approx(0.4330127)
    assert (s["reward_p10"], s["reward_p50"], s["reward_p90"]) == (0.0, 1.0, 1.0)
    assert s["resp_len_mean"] == pytest.approx(11.5) and s["resp_len_p95"] == pytest.approx(13.0)
    assert s["truncated"] == 1
    assert s["zero_variance_seen"] == 1 == sum(zero_variance(g) for _, g in scored)


def test_round_summary_empty_round_adds_nothing():
    assert round_summary([]) == {}


def test_engine_sampler_uploads_its_tape(tmp_path, ports_v1_pin):
    """A real (fake-engine) sampler run: its whole tape lands under tapes/ in the store,
    and its round events carry the reward summary."""
    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry

    assert entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None) == 2
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    objects = DirectoryObjectStore(tmp_path / "store")
    keys = objects.list("tapes/ppu4/")
    assert keys and b"".join(objects.read(k) for k in keys) == tape.read_bytes()
    rounds = [json.loads(l) for l in tape.read_text().splitlines() if '"rl_sampler_engine_round"' in l]
    assert rounds and all("reward_mean" in r for r in rounds)


def test_engine_sampler_upload_off(tmp_path, ports_v1_pin, monkeypatch):
    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry

    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "0")
    assert entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None) == 2
    assert DirectoryObjectStore(tmp_path / "store").list("tapes/") == []
