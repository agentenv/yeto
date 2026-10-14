"""rl-trainer-start-stop L4 (CPU, fakes): the durable sample buffer, its watermark
in the cut, arrival polling, and samplers that follow the trainer."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from test_rl_engine_driver import NAME, PROFILE, _driver

from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine import recovery
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.fake_lifecycle import (
    BufferFedFakeRollout,
    FakeMembership,
    FakeRoundCuts,
    FakeSampler,
    cuttable_fake_engine,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError
from yeto.rl.engine.trainer_lifecycle import (
    DriverLifecycle,
    DurableSampleBuffer,
    LifecycleError,
    LifecycleState,
    SampleIndexEntry,
    StopCondition,
)

REPO = Path(__file__).resolve().parents[1]
BATCH = 4  # fake engine: 2 groups x 2 samples


def _buffer(root, **kw):
    return FileSampleBuffer(CheckpointStore(root), batch_size=BATCH, **kw)


def _add(buf, sample_id, version):
    ref = buf.write_payload(sample_id, sample_id.encode())
    buf.append(SampleIndexEntry(sample_id, version, ref, tokens=3))


# ---------------------------------------------------------------- L4.1 index
def test_append_select_commit_and_backlog(tmp_path):
    buf = _buffer(tmp_path)
    assert isinstance(buf, DurableSampleBuffer)
    for i, v in enumerate([0, 0, 1, 1, 1, 2]):
        _add(buf, f"s{i}", v)
    stats = buf.backlog(current_version=3)
    assert (stats.samples, stats.batches, stats.oldest_version, stats.max_version_gap) == (6, 1.5, 0, 3)
    assert stats.version_histogram == {0: 2, 1: 3, 2: 1} and stats.arrival_rate > 0
    got = buf.select("trainer", 4)
    assert [s.position for s in got] == [0, 1, 2, 3] and got[2].entry.sample_id == "s2"
    assert buf.read_payload(got[0].entry.payload_ref) == b"s0"
    assert buf.backlog(3).samples == 2 and buf.watermark == 0
    buf.commit_consumed(4)
    assert buf.watermark == 4
    with pytest.raises(SampleBufferError):
        buf.commit_consumed(5)  # beyond what was selected
    with pytest.raises(SampleBufferError):
        buf.select("other", 1)
    with pytest.raises(SampleBufferError, match="payload"):
        buf.append(SampleIndexEntry("x", 0, "payloads/missing.bin"))


def test_reopen_after_process_exit(tmp_path):
    """Samples a sampler process appended survive that process; a torn last line
    is ignored and cut off by the next append."""
    script = (
        "import sys\n"
        "from yeto.rl.engine.resume import CheckpointStore\n"
        "from yeto.rl.engine.sample_buffer import FileSampleBuffer\n"
        "from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry\n"
        "buf = FileSampleBuffer(CheckpointStore(sys.argv[1]), batch_size=4)\n"
        "for i in range(5):\n"
        "    ref = buf.write_payload(f'p{i}', b'x' * i)\n"
        "    buf.append(SampleIndexEntry(f'p{i}', i // 2, ref))\n"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, cwd=REPO)
    index = tmp_path / "sample-buffer" / "index.jsonl"
    with open(index, "ab") as fh:
        fh.write(b'{"schema": "yeto.rl.sample-buf')  # crash in the middle of a line
    buf = _buffer(tmp_path)
    assert len(buf) == 5 and buf.read_payload(buf.select("t", 5)[4].entry.payload_ref) == b"xxxx"
    _add(buf, "p5", 3)
    again = _buffer(tmp_path)
    assert len(again) == 6 and [s.entry.sample_id for s in again.select("t", 9)][-1] == "p5"
    again.mark_data_end()
    assert _buffer(tmp_path).backlog(0).data_ended
    with pytest.raises(SampleBufferError, match="data end"):
        _add(again, "late", 3)


# ---------------------------------------------------------------- L4.3 arrival
def test_arrival_callback_fires_within_one_interval(tmp_path):
    reader = _buffer(tmp_path)
    writer = _buffer(tmp_path)  # another process in production
    fired = threading.Event()
    reader.on_arrival(fired.set)
    interval = 0.5
    reader.start_polling(interval)
    try:
        assert not fired.wait(interval * 1.5)  # nothing appended yet
        appended = time.monotonic()
        _add(writer, "a0", 0)
        assert fired.wait(interval * 3)
        # one interval plus scheduling slack
        assert time.monotonic() - appended <= interval + 0.25
    finally:
        reader.stop_polling()
    assert reader.poll() == 0  # the polled sample is not reported twice


# ---------------------------------------------------------------- L4.2 / L4.4 lifecycle
class _Island:
    def __init__(self, tmp_path, *, fail_rounds=None):
        self.tmp_path = tmp_path
        self.buffer = _buffer(tmp_path / "store")
        self.cuts = FakeRoundCuts(tmp_path / "cuts", watermark=lambda: self.buffer.watermark)
        self.membership = FakeMembership()
        self.fail_rounds = dict(fail_rounds or {})  # incarnation -> rounds whose step fails
        self.rollouts = []
        self.engine = None

    def build(self, handle):
        engine = cuttable_fake_engine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                                      capability_profile=PROFILE,
                                      failed_step_rounds=set(self.fail_rounds.get(handle.incarnation, ())))
        resume = handle.resume_from
        self.buffer.reset_to(0 if resume is None else resume.watermark)
        engine.rollout = BufferFedFakeRollout(engine, self.buffer)
        self.rollouts.append(engine.rollout)
        self.engine = engine
        sync = LocalOnlySync(1000)
        sync.round_cuts = self.cuts
        return _driver(engine, sync, self.tmp_path, name=f"events-{handle.incarnation}.jsonl")

    def lifecycle(self):
        return DriverLifecycle(self.build, membership=self.membership, cuts=self.cuts)

    def published(self):
        return None if self.engine is None or self.engine.published is None else self.engine.published[0]


def test_crash_after_consuming_trains_the_samples_again(tmp_path):
    """Spec "Trainer crash after consuming": the trainer consumes samples and fails
    before the next cut; the next start resumes at the watermark of the last cut."""
    island = _Island(tmp_path, fail_rounds={1: {2}})
    sampler = FakeSampler(island.buffer, island.published)
    lc = island.lifecycle()
    sampler.produce(3 * BATCH)

    h = lc.start(lc.plan_start({}), resume_from=None)
    lc.run(h, island.buffer, StopCondition(2))
    cut = lc.drain_and_cut(h, LeaveReason.STOPPED)
    lc.stop(h, LeaveReason.STOPPED)
    assert (cut.trigger, cut.policy_version, cut.watermark) == (recovery.SCHEDULED, 2, 2 * BATCH)
    record = island.cuts.pointer()[recovery.CUT_KEY]
    assert record["buffer_watermark"] == 2 * BATCH

    h = lc.start(lc.plan_start({}), resume_from=cut)
    with pytest.raises(LifecycleError):
        lc.run(h, island.buffer, StopCondition(1))  # round 2 consumes, then its step fails
    assert h.state is LifecycleState.STOPPED and island.cuts.latest() == cut
    crashed = [sid for sid, _ in island.rollouts[1].trained[2]]

    h = lc.start(lc.plan_start({}), resume_from=island.cuts.latest())
    lc.run(h, island.buffer, StopCondition(1))
    lc.stop(h, LeaveReason.STOPPED)
    again = [sid for sid, _ in island.rollouts[2].trained[2]]
    assert again == crashed == [f"s0-{i:05d}" for i in range(2 * BATCH, 3 * BATCH)]
    first = [sid for r in (0, 1) for sid, _ in island.rollouts[0].trained[r]]
    assert not set(first) & set(again)  # nothing before the watermark trained again
    assert island.cuts.latest().watermark == 3 * BATCH and island.cuts.latest().policy_version == 3
    assert [t[0] for t in island.membership.tape] == ["JOIN", "LEAVE"] * 3
    assert island.membership.tape[3] == ("LEAVE", 1, LeaveReason.TRAINING_FAILED)


def test_samplers_follow_the_trainer_and_keep_the_last_version(tmp_path):
    """D5: while training, each published version reaches the sampler; samples
    record the version the sampler loaded; after a stop the sampler keeps it."""
    island = _Island(tmp_path)
    sampler = FakeSampler(island.buffer, island.published)
    lc = island.lifecycle()
    h = lc.start(lc.plan_start({}), resume_from=None)
    assert lc.ready(h, time.monotonic() + 30)
    for _ in range(3):
        sampler.load()
        sampler.produce(BATCH)
        receipt = lc.run(h, island.buffer, StopCondition(1))
        assert h.published_version == receipt.trained_version  # published before the wait
    assert sampler.load() == 3
    lc.stop(h, LeaveReason.STOPPED)
    island.engine = None  # no trainer: nothing new to load
    assert sampler.load() == 3
    sampler.produce(BATCH)
    versions = [e.behavior_version for e in (s.entry for s in _buffer(tmp_path / "store").select("x", 99))]
    assert versions == [0] * BATCH + [1] * BATCH + [2] * BATCH + [3] * BATCH
    trained = island.rollouts[0].trained
    assert [sorted({v for _, v in trained[r]}) for r in range(3)] == [[0], [1], [2]]
    # the next start trains the backlog the sampler wrote while the trainer was off
    h = lc.start(lc.plan_start({}), resume_from=island.cuts.latest())
    lc.run(h, island.buffer, StopCondition(1))
    lc.stop(h, LeaveReason.STOPPED)
    assert {v for _, v in island.rollouts[1].trained[3]} == {3}
    assert island.buffer.backlog(4).samples == 0 and island.cuts.latest().watermark == 4 * BATCH
