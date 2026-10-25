"""S27 G-FD: one island's slow downlink must not stall the other islands (real syncer, CPU, no Ray).

Run s27-fa-r1 (Qwen3-0.6B, 2 verl islands, quorum 1): the syncer's send to
island 1 fell to about 0.14 MB/s after 637 MB. The scheduler sent each merged
fragment to every island itself and waited for each island's send queue
(``publish_round``), so the second merge waited about 6 min on island 1's link:
no PULL went out, no push was merged, and both islands only trained locally.

Now a broadcast that does not fit in an island's send queues at once goes to a
forwarder task of that island (``BulkResends::spill``, the G-EL/G-EY resend
forwarder); the newest version of a fragment replaces an older pending one.
The other islands keep merging and every island keeps getting PULLs. Once
nothing is pending, the island's broadcasts are sent inline again.
``--inline-broadcast`` keeps the old behavior; that case shows the stall.

The syncer lease (60 s) is longer than the test's slow phase: under the old
behavior a short lease ends the stall by removing the slow island (G-EL's
test_bulk_resend covers leases; the s27-fa-r1 syncer log shows no lease
expiry), and the 64 MiB INIT
at startup must not lapse one either. Here the stall itself is measured.

The slow downlink is a test-only read limit on island 1's data sockets
(``SlowDownlinkClient``); control frames (PULLs) are read at full speed, as
in the GPU run, where the small frames still arrived. With bf16 downlink
deltas (``--bcast-delta-full-every``) the forwarder sends each version as a
FULL with the shared anchor, because the island may have skipped deltas; the
island must never need a resync (a redial).
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time

import pytest

from tests.test_bulk_resend import FRAGMENTS, LEASE_S, LightIsland, merges_with, tape_records
from tests.test_peer_recovery import build_syncer, free_port, wait_until
from yeto.protocol import (
    DTYPE_F32,
    DTYPE_FLAG_BCAST_DELTA,
    SyncerClient,
    _RateLimitedSocket,
    _ReadRate,
)
from yeto.rl.pool_client import LeaseHeartbeat

# Island 1 reads its data socket at this rate while slow: one 256 KiB
# fragment per 2 s, about 1/80 of the merge rate here. The syncer's writer
# queue (128 frames) and the socket buffers fill within a few seconds.
SLOW_BPS = 128 * 1024
SLOW_S = 14.0
# The writer queue and socket buffers fill within about 5 s (bf16 deltas are
# half the size); merges are counted from then on.
FILLED_S = 8.0
FULL_EVERY = 4
SYNCER_LEASE_S = 60


class SlowDownlinkClient(SyncerClient):
    """Reads its data streams at ``bps`` while ``slow`` is set; control
    frames always at full speed."""

    def __init__(self, *args, slow: threading.Event, bps: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slow_rate = _ReadRate(bps, slow.is_set)

    def _recv_loop(self, gen, sock):
        if self._socks and sock is not self._socks[0]:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)
            sock = _RateLimitedSocket(sock, self._slow_rate)
        super()._recv_loop(gen, sock)


class CountingIsland(LightIsland):
    """A :class:`LightIsland` that records the PULLs and broadcasts it gets;
    with ``slow`` its downlink is slow while the event is set."""

    def __init__(self, learner_id: int, port: int, store, *, dtype: int, slow: threading.Event | None = None):
        super().__init__(learner_id, port, store)
        kwargs = dict(syncer_epoch=0, recover=False, num_streams=1)
        address = ("127.0.0.1", port)
        if slow is None:
            self.client = SyncerClient(address, learner_id, self.layout, dtype, **kwargs)
        else:
            self.client = SlowDownlinkClient(address, learner_id, self.layout, dtype, slow=slow,
                                             bps=SLOW_BPS, **kwargs)
        self.heartbeat = LeaseHeartbeat(self.client, LEASE_S, progress=lambda: self.steps)
        self.pull_times: list[float] = []
        self.bcasts: list[tuple[float, int, int]] = []  # (time, fragment, version)
        self.out_of_order: list[tuple[int, int, int]] = []
        drain_pulls = self.client.drain_pulls

        def counted_pulls():
            pulls = drain_pulls()
            now = time.time()
            self.pull_times.extend(now for _ in pulls)
            return pulls

        self.client.drain_pulls = counted_pulls

    def apply_broadcast(self, update) -> None:
        p = update.fragment_id
        if update.version <= self.versions[p] and self.anchored[p]:
            self.out_of_order.append((p, self.versions[p], update.version))
        self.bcasts.append((time.time(), p, update.version))
        super().apply_broadcast(update)


def steady(tape, syncer_log) -> bool:
    text = syncer_log.read_text()
    if text.count("background resend started") != text.count("background resend drained"):
        return False
    records = tape_records(tape)
    joined = max((r["time_unix"] for r in records if r.get("kind") == "pool_join"), default=0.0)
    return all(merges_with(records, island, joined, time.time() + 1) >= 10 for island in (0, 1))


def start_syncer(tmp_path, port, *extra):
    binary = build_syncer()
    tape = tmp_path / "tape.jsonl"
    log = (tmp_path / "syncer.log").open("w")
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "2", "--quorum", "1", "--grace-ms", "50",
         "--total-steps", "1200", "--outer-lr", "0.7", "--outer-momentum", "0.0",
         "--min-round-interval-ms", "20", "--quorum-timeout-s", "30", "--event-tape", str(tape),
         "--island-lease-s", str(SYNCER_LEASE_S), "--syncer-epoch", "0", "--max-base-lag", "2", *extra],
        stdout=log, stderr=subprocess.STDOUT, text=True, env={**os.environ, "TOKIO_WORKER_THREADS": "4"})
    return proc, tape, log


@pytest.mark.timeout(300)
@pytest.mark.parametrize("delta", [False, True], ids=["f32-bcast", "bf16-delta-bcast"])
@pytest.mark.parametrize("inline", [False, True], ids=["forwarder", "inline-old-behavior"])
def test_one_slow_downlink_does_not_stall_merges_or_pulls(tmp_path, inline, delta):
    port = free_port()
    extra = [*(["--inline-broadcast"] if inline else []), *(["--bcast-delta-full-every", str(FULL_EVERY)] if delta else [])]
    proc, tape, log = start_syncer(tmp_path, port, *extra)
    dtype = DTYPE_F32 | (DTYPE_FLAG_BCAST_DELTA if delta else 0)
    slow = threading.Event()
    store = tmp_path / "store"
    a = CountingIsland(0, port, store, dtype=dtype)
    b = CountingIsland(1, port, store, dtype=dtype, slow=slow)
    try:
        a.start()
        b.start()
        wait_until(lambda: sum(a.versions) >= 40 and sum(b.versions) >= 40, 60, "no early rounds")
        # Both islands are ordinary members: a late joiner's catch-up is done
        # and both pushes are being merged.
        wait_until(lambda: steady(tape, tmp_path / "syncer.log"), 60, "islands not merging steadily")
        generations = (a.client.connection_generation, b.client.connection_generation)
        slow.set()  # island 1's downlink is slow from now on
        slow_from = time.time()
        time.sleep(SLOW_S)
        slow_until = time.time()
        filled = slow_from + FILLED_S
        during = merges_with(tape_records(tape), 0, filled, slow_until)
        pulls_to_slow = sum(1 for t in b.pull_times if filled <= t <= slow_until)
        slow.clear()
        print(f"inline={inline} delta={delta}: merges with island 0 in the last {SLOW_S - FILLED_S:.0f} s of "
              f"the slow phase: {during}; PULLs read by the slow island meanwhile: {pulls_to_slow}")
        if inline:
            # Old behavior: every merge waited for island 1's send queue (one
            # f32 fragment per 2 s, a bf16 delta per 1 s).
            assert during <= 10, f"{during} merges while the scheduler waited on the slow downlink"
            return
        for island in (a, b):
            island.join(timeout=180)
            assert not island.is_alive(), f"island {island.learner_id} did not finish"
            if island.exc:
                raise island.exc
        assert proc.wait(timeout=60) == 0, "syncer exited nonzero"
    except BaseException:
        log.flush()
        print((tmp_path / "syncer.log").read_text()[-6000:])
        raise
    finally:
        for island in (a, b):
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text()
    # Island 0 kept merging at full rate while island 1's downlink was slow,
    # and island 1 kept getting PULLs.
    assert during >= 60, during
    assert pulls_to_slow >= 10, pulls_to_slow
    # Nobody lost its lease or redialed (a skipped bf16 delta would make the
    # slow island resync, that is redial).
    assert not [r for r in records if r.get("kind") == "pool_leave"]
    assert (a.client.connection_generation, b.client.connection_generation) == generations
    assert a.heartbeat.errors == [] and b.heartbeat.errors == []
    # The slow island went to a forwarder and back to inline broadcasts.
    assert "slow downlink: the learner's send queues are full" in syncer_log
    assert "slow downlink drained; broadcasts go inline again" in syncer_log
    # It never applied a version older than one it held, skipped versions
    # while it was slow, and caught up: its pushes were merged again and it
    # ends at the same final cut as island 0.
    assert b.out_of_order == []
    assert any(w[0] == 1 for r in records if r.get("kind") == "outer_step" and r["time_unix"] > slow_until + 2
               for w in r.get("weights", [])), "no merge with island 1 after its downlink recovered"
    assert a.final_manifest is not None and a.final_manifest == b.final_manifest
    # Every fragment's last broadcast reached island 1 (the terminal round's
    # fragment comes with the final cut instead).
    final_versions = a.final_manifest.versions
    behind = [p for p in range(FRAGMENTS) if b.versions[p] < final_versions[p]]
    assert len(behind) <= 1, behind
    print("\n".join(line for line in syncer_log.splitlines() if "slow downlink" in line))
