"""When a buffer trainer's lane cursor record is written (S27 G-EQ4).

The island must read a round's groups again when it stopped after collecting them
but before training them -- what :mod:`yeto.rl.engine.data_position` promises
("again, the rows of the round that was running when the process stopped").

Measured on npu1 before this change: a kill during round 4 left the record
``cursor 32, positions [28, 29, 30, 31]`` for a round that never trained, and the
relaunch started at 32, so four groups were generated and never trained.
"""

from __future__ import annotations

import pytest

from yeto.rl.adapters.verl.buffer_feed import BufferCall, BufferFeed


class _Log:
    """A lane cursor log, in memory. ``last_cursor`` is what a relaunch starts from."""

    def __init__(self) -> None:
        self.records: list[tuple[int, list[int], int]] = []

    def append(self, rollout_id: int, positions: list[int], cursor: int) -> None:
        self.records.append((int(rollout_id), list(positions), int(cursor)))

    def last_cursor(self) -> int:
        return self.records[-1][2] if self.records else 0


class _Buffer:
    """Just the parts the feed touches for the cursor: a read position it can reset."""

    def __init__(self) -> None:
        self.read_cursor = 0
        self.reset_calls: list[int] = []

    def reset_to(self, cursor: int) -> None:
        self.reset_calls.append(int(cursor))
        self.read_cursor = int(cursor)


def _feed(log: _Log, buffer: _Buffer) -> BufferFeed:
    return BufferFeed(buffer, vmap=lambda: None, shape=None,
                      add_group=lambda row: None, record=lambda row: None, cursor_log=log)


def _collect(feed: BufferFeed, buffer: _Buffer, positions: list[int]) -> None:
    """Stand in for a round's collection: the feed keeps these positions and the
    buffer's read position ends after them."""
    feed._round_kept = list(positions)
    buffer.read_cursor = positions[-1] + 1


def test_a_round_that_trains_is_not_read_again():
    log, buffer = _Log(), _Buffer()
    feed = _feed(log, buffer)

    _collect(feed, buffer, [28, 29, 30, 31])
    feed.close_round(3)
    assert log.records == []  # collection alone writes nothing

    assert feed.commit_round() == (3, [28, 29, 30, 31], 32)
    assert log.records == [(3, [28, 29, 30, 31], 32)]
    assert log.last_cursor() == 32  # a relaunch starts after them

    assert feed.commit_round() is None  # nothing pending: not written twice
    assert len(log.records) == 1


def test_a_round_collected_but_never_trained_is_read_again():
    """The npu1 case: the island died between collection and the optimizer step."""
    log, buffer = _Log(), _Buffer()
    feed = _feed(log, buffer)

    _collect(feed, buffer, [24, 25, 26, 27])
    feed.close_round(2)
    feed.commit_round()  # this round trained

    _collect(feed, buffer, [28, 29, 30, 31])
    feed.close_round(3)
    # ... and here the process is killed: commit_round never runs

    assert log.last_cursor() == 28, "the cursor must still point at the untrained round"
    relaunched = _Buffer()
    _feed(log, relaunched)
    assert relaunched.reset_calls == [28]
    assert relaunched.read_cursor == 28  # positions 28..31 are read again, not skipped


def test_a_failing_train_call_leaves_the_round_to_be_read_again():
    """``BufferCall`` commits only after the train call returns."""
    log, buffer = _Log(), _Buffer()
    feed = _feed(log, buffer)
    _collect(feed, buffer, [8, 9])

    calls: list[str] = []

    def actor_call(name, *args, **kwargs):
        calls.append(name)
        if name == "yeto_train_step":
            raise RuntimeError("the step ran out of memory")
        return "ok"

    call = BufferCall(feed, actor_call, round_id=lambda: 1)
    assert call("yeto_assemble") == "ok"
    assert log.records == []  # the assemble call does not write the record

    with pytest.raises(RuntimeError):
        call("yeto_train_step")
    assert log.records == []  # the step failed, so the groups stay unread
    assert log.last_cursor() == 0

    # the next attempt trains, and only then is the record written
    def ok_call(name, *args, **kwargs):
        return "ok"

    BufferCall(feed, ok_call, round_id=lambda: 1)("yeto_train_step")
    assert log.records == [(1, [8, 9], 10)]
    assert calls == ["yeto_assemble", "yeto_train_step"]


def test_the_train_call_result_is_passed_through():
    log, buffer = _Log(), _Buffer()
    feed = _feed(log, buffer)
    _collect(feed, buffer, [0])
    feed.close_round(0)
    out = BufferCall(feed, lambda name, *a, **k: {"loss": 0.5}, round_id=lambda: 0)("yeto_train_step")
    assert out == {"loss": 0.5}
    assert log.records == [(0, [0], 1)]
