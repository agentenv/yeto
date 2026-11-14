"""``BatchReader._stop_prefetching`` called from two threads at once
(``executor_set_mode`` and ``reset``/``take``): the later caller must not see the
read-ahead thread cleared between its check and its join."""

from __future__ import annotations

import threading

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


def test_concurrent_stops_do_not_join_a_cleared_prefetcher(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=2)
    reader = br.BatchReader(buffer, max_policy_age=2, wait_s=1.0, poll_s=0.01)
    second_in_set = threading.Event()
    first_done = threading.Event()
    calls = [0]

    class Stop(threading.Event):
        def set(self):
            calls[0] += 1
            super().set()
            if calls[0] == 2:  # the second stopper: let the first one finish first
                second_in_set.set()
                first_done.wait(2)

    class Prefetcher:
        def join(self, timeout=None):
            second_in_set.wait(0.5)  # the second stopper passed its check by now

    reader._stop_prefetch = Stop()
    reader._prefetcher = Prefetcher()
    errors: list[BaseException] = []

    def stop(done=None):
        try:
            reader._stop_prefetching()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=stop, args=(first_done,))
    second = threading.Thread(target=stop)
    first.start()
    while calls[0] < 1:
        pass
    second.start()
    first.join(5)
    second.join(5)
    assert errors == []
    assert reader._prefetcher is None


def test_settle_warns_on_a_long_read_and_keeps_waiting(monkeypatch, capsys):
    from concurrent.futures import Future

    monkeypatch.setattr(br, "SETTLE_WARN_S", 0.05)
    read: Future = Future()
    threading.Timer(0.2, read.set_result, args=(None,)).start()
    reader = br.BatchReader.__new__(br.BatchReader)
    reader.clock = __import__("time").monotonic
    waited = reader._settle([read])
    assert read.done() and waited >= 0.15
    assert "still open" in capsys.readouterr().out


def test_take_ending_after_switch_to_sampler_does_not_restart_the_prefetch(tmp_path):
    """``executor_set_mode(SAMPLER)`` stops the read-ahead while a take on another
    thread is still finishing: that take must not start the read-ahead again (it
    would poll the index -- a Volume reload every poll -- through the sampler stint)."""
    from tests.buffer_rollout_procs import sample
    from yeto.rl.adapters.miles.buffer_rollout import GroupWriter

    store = CheckpointStore(tmp_path)
    writer = GroupWriter(FileSampleBuffer(store, batch_size=2), "s0")
    for g in range(3):
        writer.write([sample(2 * g, g, 0), sample(2 * g + 1, g, 0)], group_key=f"g{g}")
    switched = threading.Event()
    at_end = threading.Event()

    class CursorLog:  # appended just before the take starts its read-ahead
        def append(self, *args, **kwargs):
            at_end.set()
            assert switched.wait(5)

    reader = br.BatchReader(FileSampleBuffer(store, batch_size=2), max_policy_age=2, wait_s=30.0,
                            poll_s=0.01)
    reader.cursor_log = CursorLog()
    br.register_reader(reader)
    taken: list = []
    try:
        br.executor_set_mode(None, br.TRAINER)
        take = threading.Thread(target=lambda: taken.append(reader.take(0, 2, current_version=0)))
        take.start()
        assert at_end.wait(5)
        br.executor_set_mode(None, br.SAMPLER)  # the take is still finishing
        switched.set()
        take.join(5)
        assert len(taken) == 1 and len(taken[0][0]) == 2
        assert reader._prefetcher is None  # no read-ahead in the sampler stint
        # the next trainer stint reads ahead again
        br.executor_set_mode(None, br.TRAINER)
        reader.cursor_log = None
        reader.take(1, 1, current_version=0)
        assert reader._prefetcher is not None
    finally:
        reader._stop_prefetching()
        br._ACTIVE.pop(br.TRAINER, None)
        br.executor_set_mode(None, br.SAMPLER)
