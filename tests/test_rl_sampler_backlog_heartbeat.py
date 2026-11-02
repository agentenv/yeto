"""S30 D29: the sampler's backlog follows the trainer's read heartbeat
(``sample-buffer/read-position.json``) when it is ahead of the last release."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.engine.sample_buffer import BUFFER_DIR, READ_POSITION, read_read_position, write_read_position
from yeto.rl.engine.sampler_loop import SamplerLoop


class _Buffer:
    def __init__(self, root, groups, shard=None):
        self.store = SimpleNamespace(root=root)
        self.groups, self.shard = groups, shard

    def refresh(self):
        pass

    def __len__(self):
        return self.groups


def _loop(buffer, shards=None, selected=None):
    seen = None if shards is None and selected is None else SimpleNamespace(
        buffer_shards=shards, buffer_selected=selected)
    follower = SimpleNamespace(last_seen=seen)
    return SamplerLoop(follower=follower, buffer=buffer, generate=lambda r: None, num_rollouts=1,
                       max_backlog_groups=8, emit=lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _buffer_dir(tmp_path):
    (tmp_path / BUFFER_DIR).mkdir()


def test_heartbeat_ahead_of_the_release_lowers_the_backlog(tmp_path):
    loop = _loop(_Buffer(tmp_path, 24, shard="spot-0"), shards={"spot-0": 8})
    assert loop.backlog() == 16  # the release's position only
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=16, buffer_shards={"spot-0": 16}, train_step=3)
    assert loop.backlog() == 8
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=20, buffer_shards={"spot-0": 20}, train_step=4)
    assert loop.backlog() == 4


def test_the_larger_position_wins(tmp_path):
    loop = _loop(_Buffer(tmp_path, 24, shard="spot-0"), shards={"spot-0": 16})
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=4, buffer_shards={"spot-0": 4}, train_step=1)
    assert loop.backlog() == 8  # an older heartbeat never raises the backlog
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=10, buffer_shards={"other": 30}, train_step=1)
    assert loop.backlog() == 8  # another shard's position does not count


def test_heartbeat_without_any_release_counts(tmp_path):
    loop = _loop(_Buffer(tmp_path, 10))
    assert loop.backlog() == 10
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=7, buffer_shards=None, train_step=None)
    assert loop.backlog() == 3


def test_broken_or_missing_heartbeat_keeps_the_release_position(tmp_path):
    loop = _loop(_Buffer(tmp_path, 24, shard="spot-0"), shards={"spot-0": 8})
    assert read_read_position(tmp_path / BUFFER_DIR) is None
    for text in ("{not json", "[]", '{"buffer_shards": "x"}', ""):
        (tmp_path / BUFFER_DIR / READ_POSITION).write_text(text)
        assert loop.backlog() == 16


def test_write_read_position_is_atomic_and_complete(tmp_path):
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=5, buffer_shards={"spot-0": 5}, train_step=2,
                        clock=lambda: 100.0)
    assert read_read_position(tmp_path / BUFFER_DIR) == {"buffer_selected": 5, "buffer_shards": {"spot-0": 5},
                                            "train_step": 2, "time_unix": 100.0}
    assert [p.name for p in (tmp_path / BUFFER_DIR).iterdir()] == [READ_POSITION]
