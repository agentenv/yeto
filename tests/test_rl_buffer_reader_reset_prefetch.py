"""``BatchReader.reset`` (a restore to a cut) while the read-ahead of the last take
runs: the restore's refresh (a Volume reload) must not run with a payload read
open, and the read-ahead of the abandoned position must stop."""

from __future__ import annotations

import threading
import time

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer


def _group(g, version):
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


def test_reset_stops_the_read_ahead_before_it_refreshes(tmp_path):
    open_reads = [0]
    overlaps: list[int] = []
    hold = threading.Event()
    held = threading.Event()
    armed = [False]

    class Watched(FileSampleBuffer):
        def read_payload(self, ref):
            open_reads[0] += 1
            try:
                if armed[0]:  # a read ahead: stays open until the test lets it go
                    held.set()
                    hold.wait(10)
                return super().read_payload(ref)
            finally:
                open_reads[0] -= 1

        def refresh(self):
            if open_reads[0]:
                overlaps.append(open_reads[0])
            return super().refresh()

    buffer = Watched(CheckpointStore(tmp_path), batch_size=2)
    writer = br.GroupWriter(buffer, "s0")
    for g in range(2):
        writer.write(_group(g, 1), group_key=f"r0-g{g}")
    reader = br.BatchReader(buffer, max_policy_age=2, wait_s=30.0, poll_s=0.01, read_workers=4)
    reader.take(0, 2, current_version=1)
    armed[0] = True
    for g in range(2, 4):
        writer.write(_group(g, 2), group_key=f"r1-g{g}")
    assert held.wait(10), "the read-ahead never started"
    overlaps.clear()
    restore = threading.Thread(target=reader.reset, args=(0,))
    restore.start()
    time.sleep(0.2)  # the restore runs while a read ahead is open
    hold.set()
    restore.join(10)
    assert not restore.is_alive()
    assert overlaps == []
    assert reader._prefetcher is None
    assert buffer.read_cursor == 0
