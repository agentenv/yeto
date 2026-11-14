"""The backlog's arrival rate is measured on the reader's clock: a sampler whose
wall clock is skewed, or writes a non-finite ``appended_at``, must not change it."""

from __future__ import annotations

import math

from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

WINDOW = 60.0


class _Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _pair(tmp_path, skew: float):
    reader_clock = _Clock(1_000_000.0)
    writer_clock = lambda: reader_clock.t + skew  # noqa: E731 - the sampler machine
    reader = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2, clock=reader_clock,
                              arrival_window_s=WINDOW)
    writer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2, clock=writer_clock)
    return reader, writer, reader_clock


def _add(buf, sample_id, appended_at=0.0):
    ref = buf.write_payload(sample_id, b"x")
    buf.append(SampleIndexEntry(sample_id, 0, ref, tokens=1, appended_at=appended_at))


def test_sampler_clock_behind_still_counts_arrivals(tmp_path):
    reader, writer, _ = _pair(tmp_path, skew=-3600.0)
    for i in range(6):
        _add(writer, f"s{i}")
    assert reader.backlog(0).arrival_rate == 6 / WINDOW


def test_sampler_clock_ahead_does_not_count_old_arrivals_forever(tmp_path):
    reader, writer, clock = _pair(tmp_path, skew=+3600.0)
    for i in range(6):
        _add(writer, f"s{i}")
    assert reader.backlog(0).arrival_rate == 6 / WINDOW
    clock.t += 2 * WINDOW  # nothing new arrived since
    assert reader.backlog(0).arrival_rate == 0.0


def test_non_finite_appended_at_is_counted(tmp_path):
    reader, writer, _ = _pair(tmp_path, skew=0.0)
    _add(writer, "nan", appended_at=math.nan)
    _add(writer, "inf", appended_at=math.inf)
    _add(writer, "ok")
    assert reader.backlog(0).arrival_rate == 3 / WINDOW


def test_history_read_at_start_is_no_arrival_burst(tmp_path):
    _, writer, _ = _pair(tmp_path, skew=0.0)
    for i in range(6):
        _add(writer, f"s{i}")
    late = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2, clock=_Clock(1_000_000.0),
                            arrival_window_s=WINDOW)
    stats = late.backlog(0)
    assert stats.samples == 6 and stats.arrival_rate == 0.0
    _add(writer, "new")
    assert late.backlog(0).arrival_rate == 1 / WINDOW


def test_head_counts_unmerged_shard_records_on_its_own_clock(tmp_path):
    """The head's view of shard records the trainer has not merged: a sampler an
    hour behind still shows arrivals; the shard history at the first read does not."""
    from yeto.rl.engine.object_store import DirectoryObjectStore
    from yeto.rl.engine.sample_buffer import UnmergedShards, shard_index

    objects = DirectoryObjectStore(tmp_path)
    head_clock = _Clock(1_000_000.0)
    sampler = FileSampleBuffer(objects, batch_size=1, index=shard_index("a"),
                               clock=lambda: head_clock.t - 3600.0)
    _add(sampler, "old")
    head = FileSampleBuffer(objects, batch_size=1, create_index=False, clock=head_clock,
                            arrival_window_s=WINDOW)
    unmerged = UnmergedShards(objects)
    stats = head.backlog(0, start=0, unmerged=unmerged)
    assert stats.samples == 1 and stats.arrival_rate == 0.0  # history
    for i in range(3):
        _add(sampler, f"s{i}")
    stats = head.backlog(0, start=0, unmerged=unmerged)
    assert stats.samples == 4 and stats.arrival_rate == 3 / WINDOW
    head_clock.t += 2 * WINDOW
    assert head.backlog(0, start=0, unmerged=unmerged).arrival_rate == 0.0
