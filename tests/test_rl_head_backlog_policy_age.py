"""CPU only: the trainer scheduler's backlog (the head's view of the sample buffer)
counts only the groups the trainer can train. A group older than
``--rl-max-policy-age`` behind the trained version is passed over unread by the
trainer's reader (``BatchReader`` age gate), so it must not start a trainer: the
convertible pool already filtered it (``MergedBacklog``), the ``--rl-trainer-scheduler``
binding did not."""

from __future__ import annotations

from types import SimpleNamespace

from yeto.launch import trainer_island
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.sample_buffer import CommittedBacklog, FileSampleBuffer
from yeto.rl.engine.trainer_lifecycle import (
    BacklogStats,
    LifecycleState,
    SampleIndexEntry,
    Start,
)
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler


class _Islands:
    def release_incarnation(self, *_a, **_k):
        return None

    def start_events(self):
        return []


def _fill(objects, versions):
    buffer = FileSampleBuffer(objects, batch_size=1)
    for i, version in enumerate(versions):
        buffer.append(SampleIndexEntry(f"g{i}", version, "", 1, float(i + 1)))


def _binding_backlog(tmp_path, versions, current_version, **args):
    objects = DirectoryObjectStore(tmp_path)
    _fill(objects, versions)
    ns = SimpleNamespace(rollout_batch_size=2, rl_checkpoint_store="s3://bkt/run",
                         rl_trainer_command_poll_seconds=None, **args)
    binding = trainer_island.object_store_binding(ns, objects, _Islands(), journal_dir=tmp_path / "j")
    return binding.buffer.backlog(current_version)


def test_over_age_groups_do_not_start_the_trainer(tmp_path):
    # trained version 10, window 2: v7 and v5 are passed over by the reader, v8 is trainable
    stats = _binding_backlog(tmp_path, [5, 7, 7, 7, 7, 8], 10, rl_max_policy_age=2)
    assert (stats.samples, stats.batches) == (1, 0.5)
    assert stats.oldest_version == 5 and stats.version_histogram == {5: 1, 7: 4, 8: 1}
    s1 = S1Scheduler(S1Config(a_hard=2, start_seconds=5.0, fixed_start_seconds=True))
    assert not isinstance(s1.decide(stats, LifecycleState.COLD), Start)
    # the same backlog counted whole (before) starts it (Q_on = a_hard + 1 = 3 batches)
    whole = BacklogStats(batches=3.0, samples=6, oldest_version=5, version_histogram=stats.version_histogram)
    assert isinstance(s1.decide(whole, LifecycleState.COLD), Start)


def test_without_an_age_window_every_group_counts(tmp_path):
    stats = _binding_backlog(tmp_path, [5, 7, 8], 10)
    assert (stats.samples, stats.batches) == (3, 1.5)


def test_the_head_and_the_reader_share_one_floor():
    """``trainable_backlog`` keeps exactly the versions ``policy_age.age_floor`` keeps."""
    from yeto.rl.engine.policy_age import age_floor
    from yeto.rl.engine.sample_buffer import trainable_backlog

    hist = {v: 1 for v in range(12)}
    stats = BacklogStats(batches=12.0, samples=12, version_histogram=hist)
    for age in (1, 2, 5):
        kept = trainable_backlog(stats, 10, age, batch_size=1).samples
        assert kept == sum(1 for v in hist if v >= age_floor(10, age))
    assert trainable_backlog(stats, 10, None, batch_size=1) is stats


def test_committed_backlog_filters_only_when_given_a_window(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    _fill(objects, [1, 9])
    head = FileSampleBuffer(objects, batch_size=1)
    assert CommittedBacklog(head, lambda: 0).backlog(10).samples == 2
    assert CommittedBacklog(head, lambda: 0, max_policy_age=1).backlog(10).samples == 1
