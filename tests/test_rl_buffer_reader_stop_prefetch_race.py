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
