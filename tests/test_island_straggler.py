"""Island stragglers (own-cluster-fleet-stability 9.1): an island none of whose pushes the
syncer merged in N merge cycles is written as ``island_straggler`` by the fleet controller,
once per stretch, and cleared once a push of it is merged again. The tracker reads the
syncer's tape (a local file or its copy in the run store); the controller asks its policy
whether to wait (default) or replace, and the other islands are not touched."""

from __future__ import annotations

import json
import types

import pytest
from test_controller import RUNNING, SYNCER, FakeOps, ImmediateThread

from yeto.launch import island_lifecycle as lc
from yeto.launch import island_straggler as st
from yeto.launch.fleet import FleetController
from yeto.rl.engine.object_store import DirectoryObjectStore


def _step(step, merged, *, fragments=2, absent=()):
    return {"kind": "outer_step", "step": step, "fragment": (step - 1) % fragments,
            "weights": [[k, step - 1, 1.0 / len(merged)] for k in merged],
            "raw_weights": [[k, step - 1, 1.0] for k in merged], "absent": list(absent)}


def _join(k):
    return {"kind": "pool_join", "island_id": k, "generation": 0, "catch_up": False, "base_version": 0}


RUN2 = {0: (0, 1), 1: (0, 1)}


def test_an_island_merged_in_no_round_for_n_cycles_is_a_straggler_once_then_cleared():
    tracker = st.StragglerTracker(cycles=3)
    for k in (0, 1):
        tracker.feed(_join(k))
    step = 0
    for _ in range(4):  # both islands merged: nothing
        step += 1
        tracker.feed(_step(step, [0, 1]))
        assert tracker.check(RUN2) == []
    signals = []
    for _ in range(6):  # island 1 silent: 3 cycles of 2 fragments
        step += 1
        tracker.feed(_step(step, [0], absent=[1]))
        signals += tracker.check(RUN2)
    (signal,) = signals
    assert signal.island == 1 and not signal.cleared and signal.missed_cycles == 3 and signal.merges == 6
    assert signal.fragments == 2 and signal.cycles == 3 and signal.absent and signal.newest_step == step
    assert signal.last_merged_step == 4 and "3 merge cycle(s)" in signal.reason
    for _ in range(4):  # still silent: no second event for the same stretch
        step += 1
        tracker.feed(_step(step, [0], absent=[1]))
        assert tracker.check(RUN2) == []
    step += 1
    tracker.feed(_step(step, [0, 1]))  # a push of it is merged again (a late push carried in)
    (cleared,) = tracker.check(RUN2)
    assert cleared.island == 1 and cleared.cleared and cleared.last_merged_step == step


def test_one_cycle_short_is_not_a_straggler_and_a_merged_push_resets_the_count():
    tracker = st.StragglerTracker(cycles=3)
    for k in (0, 1):
        tracker.feed(_join(k))
    step = 0
    for _ in range(3):
        for _ in range(5):  # 5 of the 6 merges without it, then one with it
            step += 1
            tracker.feed(_step(step, [0]))
        step += 1
        tracker.feed(_step(step, [0, 1]))
        assert tracker.check(RUN2) == []


def test_counting_starts_at_the_join_of_the_launch_and_a_relaunch_starts_it_again():
    tracker = st.StragglerTracker(cycles=1)
    for step in range(1, 9):  # island 1 not joined yet (its launch still starts): nothing counts
        tracker.feed(_step(step, [0], fragments=1))
        assert tracker.check(RUN2) == []
    tracker.feed(_join(1))
    tracker.feed(_step(9, [0], fragments=1))
    (signal,) = tracker.check(RUN2)
    assert signal.island == 1 and signal.merges == 1
    # relaunched while flagged (a new incarnation token): cleared, and counted from its next join
    (cleared,) = tracker.check({0: (0, 1), 1: (1, 2)})
    assert cleared.cleared and cleared.reason == "relaunched"
    tracker.feed(_step(10, [0], fragments=1))
    assert tracker.check({0: (0, 1), 1: (1, 2)}) == []
    tracker.feed(_join(1))  # a re-JOIN of the same incarnation later does not restart it
    tracker.feed(_step(11, [0], fragments=1))
    tracker.feed(_join(1))
    assert [s.island for s in tracker.check({0: (0, 1), 1: (1, 2)})] == [1]


def test_only_the_running_islands_are_reported_and_zero_cycles_is_off():
    tracker = st.StragglerTracker(cycles=1)
    for k in (0, 1):
        tracker.feed(_join(k))
    tracker.feed(_step(1, [0], fragments=1))
    assert tracker.check({0: (0, 1)}) == []  # island 1 draining or recovering: not watched
    (flagged,) = tracker.check(RUN2)  # running again (same incarnation): reported
    (closed,) = tracker.check({0: (0, 1)})  # then it stops running: its stretch is closed
    assert not flagged.cleared and closed.cleared and closed.island == 1 and closed.reason == "not running"
    assert tracker.check({0: (0, 1)}) == []
    off = st.StragglerTracker(cycles=0)
    for k in (0, 1):
        off.feed(_join(k))
    for step in range(1, 20):
        off.feed(_step(step, [0], fragments=1))
    assert off.check(RUN2) == []


def test_the_flag_is_a_whole_number_of_cycles():
    assert st.straggler_cycles(types.SimpleNamespace()) == 3
    assert st.straggler_cycles(types.SimpleNamespace(rl_island_straggler_cycles=5)) == 5
    assert st.straggler_cycles(types.SimpleNamespace(rl_island_straggler_cycles=0)) == 0
    with pytest.raises(ValueError, match="whole number >= 0"):
        st.straggler_cycles(types.SimpleNamespace(rl_island_straggler_cycles=-1))


def test_the_flag_is_checked_with_the_island_pool_options(monkeypatch):
    from yeto.cli import build_parser
    from yeto.launch import rl_args

    base = ["launch", "--training-mode", "rl", "--model", "m", "--data", "d"]
    assert build_parser().parse_args([*base, "--rl-island-straggler-cycles", "4"]).rl_island_straggler_cycles == 4
    assert build_parser().parse_args(base).rl_island_straggler_cycles is None
    monkeypatch.setattr(rl_args, "rl_needs_critic", lambda args: False)
    args = types.SimpleNamespace(training_mode="rl", rl_sync_island_lease_s=30.0, rl_sync_preset="strict-avg",
                                 experimental_rl_sync=False, syncer_shards=None, rl_island_straggler_cycles=-2)
    with pytest.raises(ValueError, match="--rl-island-straggler-cycles must be a whole number"):
        rl_args._check_island_pool(args, num_islands=2)


def _at(record, t):
    return dict(record, time_unix=t)


def test_records_from_before_the_watch_started_are_not_counted():
    """A relaunched or moved island's new launch reads the syncer's history: the merges its
    earlier launch missed, and its earlier join, are stamped before the watch started."""
    tracker = st.StragglerTracker(cycles=1, since=10_000.0)
    tracker.feed(_at(_join(1), 9_000.0))
    for step in range(1, 9):
        tracker.feed(_at(_step(step, [0], fragments=1, absent=[1]), 9_000.0 + step))
    assert tracker.check(RUN2) == []  # nothing of it counts: island 1 has not joined since
    tracker.feed(_at(_join(1), 10_001.0))
    assert tracker.check(RUN2) == []
    tracker.feed(_at(_step(9, [0], fragments=1), 10_002.0))
    (signal,) = tracker.check(RUN2)
    assert signal.merges == 1 and signal.last_merged_step is None


def test_a_syncer_clock_300_s_behind_is_still_counted():
    """The syncer host's clock may run behind the launcher's: its records of now are stamped up
    to CLOCK_SKEW_S before the watch's start and still count."""
    tracker = st.StragglerTracker(cycles=1, since=10_000.0)
    tracker.feed(_at(_join(1), 9_700.0))
    tracker.feed(_at(_step(1, [0], fragments=1), 9_701.0))
    assert [s.island for s in tracker.check(RUN2)] == [1]
    assert st.CLOCK_SKEW_S >= 300


def test_a_link_that_reconnects_keeps_the_count_and_the_mark():
    """Each reconnect of the same process is a new connection generation, and the syncer
    writes a new pool_join for it: none restarts the count or clears the mark. An island that
    reconnects every 2 merges and never delivers is reported once, and stays reported."""
    tracker = st.StragglerTracker(cycles=3)
    signals = []
    for step in range(1, 41):
        if step % 2 == 1:
            tracker.feed(dict(_join(1), generation=step))
            signals += tracker.check(RUN2)
        tracker.feed(_step(step, [0], fragments=1, absent=[1]))
        signals += tracker.check(RUN2)
    assert [(s.island, s.cleared, s.merges) for s in signals] == [(1, False, 3)]


def test_a_new_syncer_term_keeps_the_count_and_the_mark():
    tracker = st.StragglerTracker(cycles=2)
    tracker.feed(_join(1))
    for step in (1, 2):
        tracker.feed(_step(step, [0], fragments=1))
    (signal,) = tracker.check(RUN2)
    assert not signal.cleared
    tracker.feed(dict(_join(1), syncer_epoch=4, generation=7))  # a syncer that took over
    assert tracker.check(RUN2) == []  # still reported: nothing written
    tracker.feed(_step(3, [1], fragments=1))
    (cleared,) = tracker.check(RUN2)
    assert cleared.cleared and cleared.reason == "a push of it is merged again"


def test_an_island_the_syncer_lists_absent_is_counted_without_a_join():
    """A restarted launcher adopts a stuck island: its join came before this watch. The syncer
    expects it (absent in its merges), so it is counted from the first such merge."""
    tracker = st.StragglerTracker(cycles=2)
    for step in (1, 2):
        tracker.feed(_step(step, [0], fragments=1, absent=[] if step == 1 else [1]))
    assert tracker.check(RUN2) == []  # one merge listed it absent: 1 of 2
    tracker.feed(_step(3, [0], fragments=1, absent=[1]))
    (signal,) = tracker.check(RUN2)
    assert signal.island == 1 and signal.merges == 2 and signal.absent


def test_records_of_the_earlier_incarnation_read_late_do_not_count():
    """Relaunched (it left running, then runs as a new token): merges of its earlier
    incarnation read late, even ones that list it absent, start nothing; it is counted
    from its next join."""
    tracker = st.StragglerTracker(cycles=1)
    tracker.feed(_join(1))
    tracker.check(RUN2)
    assert tracker.check({0: (0, 1)}) == []  # recovering
    tracker.feed(_step(1, [0], fragments=1, absent=[1]))  # read late
    assert tracker.check({0: (0, 1), 1: (1, 2)}) == []  # running as its next incarnation
    tracker.feed(_step(2, [0], fragments=1))
    assert tracker.check({0: (0, 1), 1: (1, 2)}) == []  # not joined yet: starting
    tracker.feed(_join(1))
    tracker.feed(_step(3, [0], fragments=1))
    assert [s.merges for s in tracker.check({0: (0, 1), 1: (1, 2)})] == [1]


def test_a_join_recorded_before_the_controller_holds_the_new_incarnation_running_counts():
    """Recovering (the island left running), its new incarnation joins, and only then does
    the controller hold it running (a new token): the join is the new incarnation's, so the
    island is counted from its return, not left uncounted until a push of it merges."""
    tracker = st.StragglerTracker(cycles=1)
    tracker.feed(_join(1))
    tracker.check(RUN2)
    assert tracker.check({0: (0, 1)}) == []  # island 1 recovering
    tracker.feed(dict(_join(1), catch_up=True))
    assert tracker.check({0: (0, 1)}) == []  # not held running: never reported
    assert tracker.check({0: (0, 1), 1: (1, 2)}) == []  # running again
    tracker.feed(_step(1, [0], fragments=1))
    (signal,) = tracker.check({0: (0, 1), 1: (1, 2)})
    assert not signal.cleared and signal.island == 1 and signal.merges == 1


def test_an_island_back_running_as_the_same_incarnation_keeps_its_place_in_the_pool():
    tracker = st.StragglerTracker(cycles=1)
    tracker.feed(_join(1))
    tracker.check(RUN2)
    assert tracker.check({0: (0, 1)}) == []
    tracker.check(RUN2)  # the same token back
    tracker.feed(_step(1, [0], fragments=1))
    assert [s.island for s in tracker.check(RUN2)] == [1]
    again = st.StragglerTracker(cycles=2)  # counted from its return: what it missed before is not
    again.feed(_join(1))
    again.feed(_step(1, [0], fragments=1))
    again.check(RUN2)
    again.check({0: (0, 1)})
    again.check(RUN2)
    again.feed(_step(2, [0], fragments=1))
    assert again.check(RUN2) == []
    fresh = st.StragglerTracker(cycles=1)  # never joined before it left: still not counted
    fresh.check(RUN2)
    fresh.check({0: (0, 1)})
    fresh.check(RUN2)
    fresh.feed(_step(1, [0], fragments=1))
    assert fresh.check(RUN2) == []


# -- the syncer's tape ---------------------------------------------------------------------------

def test_the_local_tape_is_read_line_by_line_from_its_offset(tmp_path):
    tape = tmp_path / "tape.jsonl"
    tape.write_text(json.dumps(_join(9)) + "\n")
    source = st.FileTape(tape, offset=tape.stat().st_size)  # an earlier syncer job's records
    assert source.read() == []
    with tape.open("a") as fh:
        fh.write(json.dumps(_join(0)) + "\n" + '{"kind": "outer_st')  # the syncer mid-write
    assert source.read() == [_join(0)]
    with tape.open("a") as fh:
        fh.write('ep", "step": 1}\nnot json\n')
    assert source.read() == [{"kind": "outer_step", "step": 1}]
    assert st.FileTape(tmp_path / "missing.jsonl").read() == []


def _chunk(objects, inc, seq, *records):
    objects.write(f"tapes/syncer/{inc}/{seq:08d}.jsonl", "".join(json.dumps(r) + "\n" for r in records).encode())


def test_the_store_tape_starts_at_the_tail_of_the_newest_syncer_and_reads_each_chunk_once(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    old, current = "hostb-20261126T120000Z", "hosta-20261126T130000Z"  # by name the newer sorts first
    _chunk(objects, old, 0, _join(1), _step(1, [0]))
    _chunk(objects, current, 0, _join(1), _step(2, [0]))  # the history before this watch
    _chunk(objects, current, 1, _step(3, [0]))
    objects.write("tapes/syncer-shard1/x/00000000.jsonl", b'{"kind": "outer_step"}\n')  # not this tape
    objects.write("tapes/l0-host-20261126T120000Z/00000000.jsonl", b'{"event": "rl_local_round"}\n')
    clock = [0.0]
    source = st.StoreTape(objects, interval_s=30, clock=lambda: clock[0])
    assert source.read() == [] and source.polled  # the tail: no history
    _chunk(objects, current, 2, _step(4, [0]))
    assert source.read() == [] and not source.polled  # within the poll interval: no request
    clock[0] = 31.0
    assert source.read() == [_step(4, [0])]
    clock[0] = 62.0
    assert source.read() == []
    restarted = "hosta-20261126T140000Z"  # the syncer restarted: its new tape is read from its start
    _chunk(objects, current, 3, _step(5, [0]))
    _chunk(objects, restarted, 0, _join(0), _step(6, [0]))
    got = []
    for _ in range(st.STORE_DISCOVER_POLLS):  # found by the next full listing
        clock[0] += 31.0
        got += source.read()
    assert got == [_step(5, [0]), _join(0), _step(6, [0])]  # the old one drained, then the new one


def test_a_watch_started_on_a_history_does_not_count_it(tmp_path):
    """The relaunch case through the store: an earlier launch's history (island 1 missing
    every merge) in the newest syncer incarnation; a new watch reports nothing until island 1
    joins again, then counts from 0."""
    objects = DirectoryObjectStore(tmp_path / "store")
    inc = "hosta-20261126T130000Z"
    _chunk(objects, inc, 0, _at(_join(0), 10.0), _at(_join(1), 10.0),
           *(_at(_step(n, [0], fragments=1), 10.0 + n) for n in range(1, 9)))
    clock = [0.0]
    watch = st.StragglerWatch(st.StoreTape(objects, interval_s=30, clock=lambda: clock[0]), cycles=2,
                              since=100.0, log=lambda m: None)
    assert watch.poll(RUN2) == []
    _chunk(objects, inc, 1, *(_at(_step(n, [0], fragments=1), 101.0 + n) for n in range(9, 12)))
    clock[0] = 31.0
    assert watch.poll(RUN2) == []  # island 1 has not joined since the watch started
    _chunk(objects, inc, 2, _at(_join(1), 120.0), _at(_step(12, [0], fragments=1), 121.0))
    clock[0] = 62.0
    assert watch.poll(RUN2) == []  # 1 merge since its join
    _chunk(objects, inc, 3, _at(_step(13, [0], fragments=1), 122.0))
    clock[0] = 93.0
    (signal,) = watch.poll(RUN2)
    assert signal.island == 1 and signal.merges == 2


def test_the_store_tape_reads_at_most_its_chunks_per_poll_and_keeps_what_it_read(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    inc = "hosta-20261126T130000Z"
    _chunk(objects, inc, 0, _join(0))
    clock = [0.0]
    source = st.StoreTape(objects, interval_s=30, clock=lambda: clock[0], gets_per_poll=4)
    assert source.read() == []
    for seq in range(1, 11):
        _chunk(objects, inc, seq, _step(seq, [0]))
    got = []
    for _ in range(3):
        clock[0] += 31.0
        batch = source.read()
        assert len(batch) <= 4
        got += batch
    assert [r["step"] for r in got] == list(range(1, 11))

    class Flaky:  # the 3rd chunk read of a poll fails once
        def __init__(self, inner):
            self.inner, self.reads, self.failed = inner, 0, False

        def list(self, prefix, *, start_after=""):
            return self.inner.list(prefix, start_after=start_after)

        def read(self, key):
            self.reads += 1
            if self.reads == 4 and not self.failed:  # the tail's read is the 1st
                self.failed = True
                raise OSError("read timed out")
            return self.inner.read(key)

    flaky = st.StoreTape(Flaky(objects), interval_s=30, clock=lambda: clock[0])
    assert flaky.read() == []
    for seq in range(11, 16):
        _chunk(objects, inc, seq, _step(seq, [0]))
    clock[0] += 31.0
    first = flaky.read()
    assert [r["step"] for r in first] == [11, 12] and "read timed out" in flaky.error  # kept, not lost
    clock[0] += 31.0
    second = flaky.read()
    assert [r["step"] for r in second] == [13, 14, 15] and flaky.error is None  # none twice, none skipped


def test_a_failed_first_read_tries_the_tail_again_and_never_reads_the_history(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    inc = "hosta-20261126T130000Z"
    for seq in range(5):
        _chunk(objects, inc, seq, _step(seq + 1, [0]))  # the history

    class Down:
        def __init__(self, inner, fail):
            self.inner, self.fail = inner, set(fail)

        def list(self, prefix, *, start_after=""):
            if ("root" if prefix == "tapes/syncer/" else "inc") in self.fail:
                raise OSError("list failed")
            return self.inner.list(prefix, start_after=start_after)

        def read(self, key):
            return self.inner.read(key)

    clock = [0.0]
    for fail in ({"root"}, {"inc"}):  # the incarnations' listing, then the newest one's chunks
        clock[0] = 0.0
        down = Down(objects, fail)
        source = st.StoreTape(down, interval_s=30, clock=lambda: clock[0])
        assert source.read() == [] and "list failed" in source.error and not source.tailed
        down.fail = set()
        clock[0] = 31.0
        assert source.read() == [] and source.tailed  # the tail now: still no history
        _chunk(objects, inc, 5 + len(fail) + (fail == {"inc"}), _step(99, [0]))
        clock[0] = 62.0
        assert [r["step"] for r in source.read()] == [99]


def test_the_newest_incarnation_is_the_furthest_syncer_not_the_newest_name(tmp_path):
    """No clock: the syncer that took over (a newer term) is read even when its folder's name
    sorts first and its steps are behind."""
    objects = DirectoryObjectStore(tmp_path / "store")
    _chunk(objects, "z-20991231T000000Z", 0, dict(_step(50, [0]), syncer_epoch=1))
    _chunk(objects, "a-20000101T000000Z", 0, dict(_step(40, [0]), syncer_epoch=2))
    clock = [0.0]
    source = st.StoreTape(objects, interval_s=30, clock=lambda: clock[0])
    assert source.read() == [] and source.queue == ["a-20000101T000000Z"]
    _chunk(objects, "a-20000101T000000Z", 1, dict(_step(41, [0]), syncer_epoch=2))
    _chunk(objects, "z-20991231T000000Z", 1, dict(_step(51, [0]), syncer_epoch=1))  # the old one, late
    clock[0] = 31.0
    assert [r["step"] for r in source.read()] == [41]


def test_a_store_poll_starts_no_request_past_its_budget(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    inc = "hosta-20261126T130000Z"
    _chunk(objects, inc, 0, _join(0))
    ticks = [0.0]

    def clock():  # every call of the poll's clock: 1 s
        ticks[0] += 1.0
        return ticks[0]

    source = st.StoreTape(objects, interval_s=0, clock=clock, budget_s=4.0)
    assert source.read() == []
    for seq in range(1, 20):
        _chunk(objects, inc, seq, _step(seq, [0]))
    before = source.requests
    batch = source.read()
    assert 0 < len(batch) < 16 and source.requests - before <= 4


def test_a_watch_whose_tape_does_not_read_logs_once_and_goes_on():
    lines = []

    class Broken:
        label = "s3://b/runs/r/tapes/syncer/"

        def read(self):
            raise OSError("store down")

    watch = st.StragglerWatch(Broken(), cycles=1, log=lines.append)
    assert watch.poll(RUN2) == [] and watch.poll(RUN2) == []
    assert len(lines) == 1 and "store down" in lines[0]


def test_the_launcher_watches_an_island_pool_only(tmp_path):
    lines = []
    pool = types.SimpleNamespace(training_mode="rl", rl_sync_island_lease_s=30.0)
    assert st.watch_for(types.SimpleNamespace(training_mode="rl"), objects=object(), log=lines.append) is None
    off = st.watch_for(types.SimpleNamespace(**vars(pool), rl_island_straggler_cycles=0), objects=object(),
                       log=lines.append)
    assert isinstance(off, st.UnavailableWatch)
    assert off.notices() == [(st.UNAVAILABLE_EVENT, "turned off (--rl-island-straggler-cycles 0)")]
    assert off.notices() == [] and off.poll(RUN2) == []
    none = st.watch_for(pool, log=lines.append, no_source="the syncer runs on its own VM")
    assert isinstance(none, st.UnavailableWatch)
    assert none.notices() == [(st.UNAVAILABLE_EVENT, "the syncer runs on its own VM")]
    assert "watch off: the syncer runs on its own VM" in lines[-1]
    watch = st.watch_for(pool, local_tape=tmp_path / "tape.jsonl", objects=object(), log=lines.append)
    assert isinstance(watch.source, st.FileTape) and watch.tracker.cycles == 3 and watch.tracker.since > 0
    watch = st.watch_for(pool, objects=DirectoryObjectStore(tmp_path / "s"), log=lines.append)
    assert isinstance(watch.source, st.StoreTape)


# -- the fleet controller ------------------------------------------------------------------------

class ScriptedTape:
    label = "test tape"

    def __init__(self):
        self.pending: list[dict] = []

    def read(self):
        out, self.pending = self.pending, []
        return out


class Recorder:
    def __init__(self):
        self.events = []

    def event(self, kind, name, **fields):
        self.events.append((kind, name, fields))

    def kinds(self, kind):
        return [e for e in self.events if e[0] == kind]

    def maybe_tick(self):
        pass

    def cost_tick(self):
        pass


def _controller(tmp_path, *, policy=None, cycles=1):
    ops = FakeOps()
    ops.status_seq.update({SYNCER: [RUNNING], "r-l0": [RUNNING], "r-l1": [RUNNING]})
    store = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    refs = {f"r-l{k}": lc.IslandRef(k, "ssh") for k in (0, 1)}
    tape, log = ScriptedTape(), Recorder()
    ctl = FleetController(learners={n: (f"task-{n}", 1) for n in refs}, syncer=(SYNCER, "task-syncer", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                          recorder=lc.FleetRecorder(store, refs), elastic=True, fleet_log=log,
                          straggler=st.StragglerWatch(tape, cycles=cycles, log=lambda m: None),
                          straggler_policy=policy)
    return ctl, ops, store, tape, log


def test_the_controller_writes_the_straggler_once_waits_and_touches_no_other_island(tmp_path):
    ctl, ops, store, tape, log = _controller(tmp_path)
    tape.pending += [_join(0), _join(1), _step(1, [0, 1], fragments=1), _step(2, [0], fragments=1, absent=[1])]
    ctl._poll_stragglers()
    (event,) = [e for e in log.events if e[0] == st.EVENT]
    assert event[1] == "r-l1" and event[2]["action"] == st.WAIT and event[2]["missed_cycles"] == 1
    record = store.read(1)
    assert record["straggler"]["action"] == "wait" and record["straggler"]["last_merged_step"] == 1
    assert "straggler" not in (store.read(0) or {})
    ctl._poll_stragglers()  # nothing new: no second event
    assert len([e for e in log.events if e[0] == st.EVENT]) == 1
    tape.pending.append(_step(3, [0, 1], fragments=1))
    ctl._poll_stragglers()
    assert [e[:2] for e in log.events if e[0] == st.CLEARED_EVENT] == [(st.CLEARED_EVENT, "r-l1")]
    assert store.read(1)["straggler"] is None
    assert ops.down_calls == [] and ops.relaunch_calls == []  # waiting stops and relaunches nothing
    assert all(rec["state"] == "running" for rec in ctl.learners.values())


def test_a_draining_island_is_not_reported(tmp_path):
    ctl, _ops, _store, tape, log = _controller(tmp_path)
    ctl.learners["r-l1"]["draining"] = True
    tape.pending += [_join(0), _join(1), _step(1, [0], fragments=1), _step(2, [0], fragments=1)]
    ctl._poll_stragglers()
    assert not [e for e in log.events if e[0] == st.EVENT]


def test_the_policy_decides_and_a_replace_is_handed_to_it(tmp_path):
    asked, replaced = [], []

    class Policy:
        def __call__(self, name, signal):
            asked.append((name, signal.island))
            return st.REPLACE

        def replace(self, name, signal):
            replaced.append(name)

    ctl, _ops, store, tape, _log = _controller(tmp_path, policy=Policy())
    tape.pending += [_join(0), _join(1), _step(1, [0], fragments=1)]
    ctl._poll_stragglers()
    assert asked == [("r-l1", 1)] and replaced == ["r-l1"]
    assert store.read(1)["straggler"]["action"] == "replace"


def test_a_policy_that_fails_or_answers_nonsense_waits(tmp_path):
    def broken(name, signal):
        raise RuntimeError("no spare")

    ctl, _ops, store, tape, _log = _controller(tmp_path, policy=broken)
    tape.pending += [_join(0), _join(1), _step(1, [0], fragments=1)]
    ctl._poll_stragglers()
    assert store.read(1)["straggler"]["action"] == "wait"
    ctl, _ops, store, tape, _log = _controller(tmp_path / "b", policy=lambda name, signal: "reboot")
    tape.pending += [_join(0), _join(1), _step(1, [0], fragments=1)]
    ctl._poll_stragglers()
    assert store.read(1)["straggler"]["action"] == "wait"


def test_without_a_watch_the_controller_reads_nothing(tmp_path):
    ctl, _ops, _store, _tape, log = _controller(tmp_path)
    ctl.straggler = None
    ctl._poll_stragglers()
    assert log.events == []


def test_a_slow_start_is_never_said_unavailable():
    """Nothing about this launch's islands read yet (its trainers still load): no notice."""
    lines, tape = [], ScriptedTape()
    watch = st.StragglerWatch(tape, cycles=1, log=lines.append)
    tape.pending.append(_join(5))  # another launch's island: not this one's first record
    for _ in range(st.QUIET_POLLS * 3):
        watch.poll(RUN2)
    assert lines == [] and watch.notices() == []


def test_a_tape_silent_after_its_first_record_is_said_once_then_taken_back():
    lines, tape = [], ScriptedTape()
    watch = st.StragglerWatch(tape, cycles=1, log=lines.append)
    tape.pending.append(_join(0))
    watch.poll(RUN2)
    for _ in range(st.QUIET_POLLS * 2):
        watch.poll(RUN2)
    assert len(lines) == 1 and "no syncer record read from test tape in 10 reads" in lines[0]
    ((kind, why),) = watch.notices()
    assert kind == st.UNAVAILABLE_EVENT and "no syncer record" in why and watch.notices() == []
    tape.pending.append(_step(1, [0, 1]))
    watch.poll(RUN2)
    assert watch.notices() == [(st.AVAILABLE_EVENT, "syncer records read again from test tape")]
    for _ in range(st.QUIET_POLLS - 1):
        watch.poll(RUN2)
    assert watch.notices() == []  # one read short of a new silent stretch


def test_the_launcher_reads_the_store_copy_for_an_island_launch_only(tmp_path, monkeypatch):
    from yeto import launcher

    monkeypatch.delenv("YETO_RL_TAPE_UPLOAD_S", raising=False)
    states = types.SimpleNamespace(objects=DirectoryObjectStore(tmp_path / "s"))
    base = {"training_mode": "rl", "rl_sync_island_lease_s": 30.0, "rl_checkpoint_store": "s3://b/runs/r"}
    island = types.SimpleNamespace(**base, rl_fleet_role="island")
    assert isinstance(launcher._island_straggler_watch(island, states, None).source, st.StoreTape)
    group = types.SimpleNamespace(**base, rl_fleet_role=None)  # one launch: its syncer's tape is not uploaded
    off = launcher._island_straggler_watch(group, states, None)
    assert isinstance(off, st.UnavailableWatch) and "a VM of its own" in off.reason
    local = types.SimpleNamespace(event_tape=str(tmp_path / "tape.jsonl"), _event_offset=5)
    watch = launcher._island_straggler_watch(group, states, local)
    assert isinstance(watch.source, st.FileTape) and watch.source.offset == 5
    assert launcher._island_straggler_watch(island, None, None) is None  # no island records: nothing written
    local_store = types.SimpleNamespace(**dict(base, rl_checkpoint_store=str(tmp_path / "d")), rl_fleet_role="island")
    assert "not an object store" in launcher._island_straggler_watch(local_store, states, None).reason
    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "0")
    assert "upload is off" in launcher._island_straggler_watch(island, states, None).reason
    with pytest.raises(SystemExit, match="whole number"):
        launcher._island_straggler_watch(types.SimpleNamespace(**base, rl_island_straggler_cycles=-1), states, local)


def test_a_flagged_island_that_stops_running_is_cleared_naming_why(tmp_path):
    from yeto.launch import fleet as lf

    for setup, why in ((lambda rec: rec.update(draining=True), "stopped or removed (island STOP)"),
                       (lambda rec: rec.update(state=lf.RECOVERING), "relaunched"),
                       (lambda rec: rec.update(state=lf.DONE), "not running (done)")):
        ctl, _ops, store, tape, log = _controller(tmp_path / why.split()[0])
        tape.pending += [_join(0), _join(1), _step(1, [0], fragments=1)]
        ctl._poll_stragglers()
        assert store.read(1)["straggler"]["action"] == "wait"
        setup(ctl.learners["r-l1"])
        ctl._poll_stragglers()
        (cleared,) = log.kinds(st.CLEARED_EVENT)
        assert cleared[1] == "r-l1" and cleared[2]["reason"] == why
        assert store.read(1)["straggler"] is None


def test_the_watch_s_notices_are_written_once_as_unavailable(tmp_path):
    ctl, _ops, _store, _tape, log = _controller(tmp_path)
    ctl.straggler = st.UnavailableWatch("the run store is not an object store")
    ctl._poll_stragglers()
    ctl._poll_stragglers()
    why = {"reason": "the run store is not an object store"}
    assert [e[1:] for e in log.kinds(st.UNAVAILABLE_EVENT)] == [("r-l0", why), ("r-l1", why)]  # once per island


def test_a_watch_read_again_writes_available(tmp_path):
    ctl, _ops, _store, _tape, log = _controller(tmp_path)
    ctl.straggler._notices.append((st.AVAILABLE_EVENT, "syncer records read again"))
    ctl._poll_stragglers()
    assert [e[1] for e in log.kinds(st.AVAILABLE_EVENT)] == ["r-l0", "r-l1"]


def test_a_mark_an_earlier_launcher_left_is_cleared_once_and_judged_again(tmp_path):
    ctl, _ops, store, tape, log = _controller(tmp_path)
    store.write(1, "running", straggler={"action": "wait", "missed_cycles": 3, "merges": 3, "fragments": 1,
                                         "newest_step": 9, "last_merged_step": 6, "at": 1.0})
    ctl._poll_stragglers()
    (cleared,) = log.kinds(st.CLEARED_EVENT)
    assert cleared[1] == "r-l1" and cleared[2]["reason"] == "controller restarted"
    assert cleared[2]["missed_cycles"] == 3 and store.read(1)["straggler"] is None
    assert (store.read(0) or {}).get("straggler") is None and not log.kinds(st.EVENT)
    tape.pending += [_join(1), _step(1, [0], fragments=1)]
    ctl._poll_stragglers()
    assert [e[1] for e in log.kinds(st.EVENT)] == ["r-l1"]  # judged again by this watch
    assert len(log.kinds(st.CLEARED_EVENT)) == 1
