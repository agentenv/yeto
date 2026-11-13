"""CPU only: the trainer scheduler's backlog uses the reader's window rule at
every ``--rl-max-policy-age``, 0 included. ``policy_age.age_floor(cur, 0)`` is
``cur`` (0 is the deterministic mode: only the current policy's samples, never
"no limit"); the head must not read 0 as "count everything"."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.launch import trainer_island
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.policy_age import PolicyAgeError, age_floor
from yeto.rl.engine.sample_buffer import CommittedBacklog, FileSampleBuffer, trainable_backlog
from yeto.rl.engine.trainer_lifecycle import BacklogStats, SampleIndexEntry


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


@pytest.mark.parametrize("args", [{"rl_max_policy_age": 0}, {"rl_max_policy_age": None}, {}])
def test_age_zero_counts_only_current_version_groups(tmp_path, args):
    stats = _binding_backlog(tmp_path, [7, 9, 10, 10], 10, **args)
    assert stats.samples == sum(1 for v in (7, 9, 10, 10) if v >= age_floor(10, 0)) == 2
    assert stats.batches == 1.0


def test_the_head_floor_is_age_floor_for_every_limit_including_zero():
    hist = {v: 1 for v in range(12)}
    stats = BacklogStats(batches=12.0, samples=12, version_histogram=hist)
    for age in (0, 1, 2, 5):
        kept = trainable_backlog(stats, 10, age, batch_size=1).samples
        assert kept == sum(1 for v in hist if v >= age_floor(10, age))


def test_committed_backlog_with_age_zero_filters(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    _fill(objects, [9, 10])
    head = FileSampleBuffer(objects, batch_size=1)
    assert CommittedBacklog(head, lambda: 0, max_policy_age=0).backlog(10).samples == 1


def test_a_negative_limit_is_refused_like_the_reader():
    stats = BacklogStats(batches=1.0, samples=1, version_histogram={10: 1})
    with pytest.raises(PolicyAgeError):
        trainable_backlog(stats, 10, -1, batch_size=1)
