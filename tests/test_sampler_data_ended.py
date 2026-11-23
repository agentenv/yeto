"""A sampler whose task dispenser settled every task ends normally (exit 0,
"data ended"); a fenced incarnation's end is not the data's end."""

from __future__ import annotations

import pytest

from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_loop import SamplerLoop
from yeto.rl.engine.task_dispense import DispenseEnded, DispenseFencedEnd


def _loop(tmp_path, generate, events):
    follower = PolicyFollower(PolicyReleaseStore(CheckpointStore(tmp_path / "releases")),
                              lambda release, directory: None)
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "buffer"), batch_size=1)
    return SamplerLoop(follower=follower, buffer=buffer, generate=generate, num_rollouts=None,
                       max_backlog_groups=8, emit=lambda name, **kw: events.append(name)), buffer


def test_dispense_end_finishes_the_loop_and_marks_the_data_end(tmp_path, capsys):
    calls, events = [], []

    def generate(rollout_id):
        calls.append(rollout_id)
        if rollout_id == 2:
            raise DispenseEnded("every task of the store dispenser is settled")

    loop, buffer = _loop(tmp_path, generate, events)
    assert loop.run() == 2
    assert calls == [0, 1, 2]
    assert buffer.data_ended
    assert "rl_sampler_data_ended" in events and events[-1] == "rl_sampler_data_end"
    assert "data ended" in capsys.readouterr().out


def test_a_fenced_dispense_end_is_not_the_data_end(tmp_path):
    def generate(rollout_id):
        raise DispenseFencedEnd("fenced: this incarnation takes no more tasks")

    loop, buffer = _loop(tmp_path, generate, [])
    with pytest.raises(DispenseFencedEnd):
        loop.run()
    assert not buffer.data_ended


def test_the_store_dispenser_end_names_its_plan_in_the_shard_end_record(tmp_path):
    """The shard's end record carries the dispenser plan: the trainer's merger then ends
    the shards no group of that plan can reach (sample_buffer module doc)."""
    detail = {"backend": "store", "namespace": "run/task-dispense", "plan": "run/task-dispense:00ff", "slots": 3}
    events = []

    def generate(rollout_id):
        error = DispenseEnded("every task of the store dispenser is settled")
        error.dispense = detail
        raise error

    follower = PolicyFollower(PolicyReleaseStore(CheckpointStore(tmp_path / "releases")),
                              lambda release, directory: None)
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "buffer"), batch_size=1)
    loop = SamplerLoop(follower=follower, buffer=buffer, generate=generate, num_rollouts=None,
                       max_backlog_groups=8, emit=lambda name, **kw: events.append((name, kw)))
    assert loop.run() == 0
    assert buffer.data_ended
    ended = [kw for name, kw in events if name == "rl_sampler_data_ended"]
    assert ended[0]["dispense"] == detail
    records = [line for line in (tmp_path / "buffer" / "sample-buffer" / "index.jsonl").read_text().splitlines()]
    assert '"dispense"' in records[-1] and '"run/task-dispense:00ff"' in records[-1]
