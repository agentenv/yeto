"""The sampler's backlog follows the trainer's read heartbeat
(``sample-buffer/read-position.json``) when it belongs to the release lineage."""

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


def _loop(buffer, shards=None, selected=None, seq=3, policy_hash="h3", released_at=50.0):
    seen = None if shards is None and selected is None else SimpleNamespace(
        buffer_shards=shards, buffer_selected=selected, seq=seq, policy_hash=policy_hash,
        released_at=released_at)
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


def test_an_old_format_heartbeat_only_moves_the_position_forward(tmp_path):
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


def _beat(tmp_path, read, *, release, clock=100.0, step=3):
    write_read_position(tmp_path / BUFFER_DIR, buffer_selected=read, buffer_shards={"spot-0": read},
                        train_step=step, release=release, clock=lambda: clock)


def test_a_heartbeat_of_the_same_release_is_the_trainers_position_even_when_lower(tmp_path):
    """A trainer resumed from an earlier checkpoint reads again from there: its
    heartbeat (same release lineage) lowers the read position below the release's."""
    loop = _loop(_Buffer(tmp_path, 24, shard="spot-0"), shards={"spot-0": 16})
    _beat(tmp_path, 4, release={"seq": 3, "policy_hash": "h3"})
    assert loop.backlog() == 20
    _beat(tmp_path, 20, release={"seq": 3, "policy_hash": "h3"})
    assert loop.backlog() == 4


def test_a_heartbeat_of_a_newer_release_counts_once_written_after_the_seen_one(tmp_path):
    loop = _loop(_Buffer(tmp_path, 24, shard="spot-0"), shards={"spot-0": 16}, released_at=50.0)
    _beat(tmp_path, 6, release={"seq": 4, "policy_hash": "h4"}, clock=60.0)
    assert loop.backlog() == 18
    _beat(tmp_path, 22, release={"seq": 4, "policy_hash": "h4"}, clock=40.0)  # older than the seen release
    assert loop.backlog() == 8


@pytest.mark.parametrize("release", [{"seq": 3, "policy_hash": "other-run"}, {"seq": 2, "policy_hash": "h2"},
                                     None])
def test_a_heartbeat_from_another_run_or_before_the_release_is_ignored(tmp_path, release):
    loop = _loop(_Buffer(tmp_path, 24, shard="spot-0"), shards={"spot-0": 16})
    _beat(tmp_path, 2, release=release)
    assert loop.backlog() == 8
    _beat(tmp_path, 22, release=release)
    assert loop.backlog() == 8  # neither lower nor higher


def test_the_heartbeat_names_its_release(tmp_path):
    _beat(tmp_path, 5, release={"seq": 7, "policy_hash": "abc"}, clock=100.0)
    assert read_read_position(tmp_path / BUFFER_DIR) == {
        "buffer_selected": 5, "buffer_shards": {"spot-0": 5}, "train_step": 3, "time_unix": 100.0,
        "release": {"seq": 7, "policy_hash": "abc"}}
    _beat(tmp_path, 5, release=None)
    assert read_read_position(tmp_path / BUFFER_DIR)["release"] is None
