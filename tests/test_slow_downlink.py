"""One island's slow downlink must not stall the other islands (real syncer, CPU, no Ray).

A GPU run (Qwen3-0.6B, 2 verl islands, quorum 1): the syncer's send to
island 1 fell to about 0.14 MB/s after 637 MB. The scheduler sent each merged
fragment to every island itself and waited for each island's send queue
(``publish_round``), so the second merge waited about 6 min on island 1's link:
no PULL went out, no push was merged, and both islands only trained locally.

Now a broadcast that does not fit in an island's send queues at once goes to a
forwarder task of that island (``BulkResends::spill``, the background resend
forwarder); the newest version of a fragment replaces an older pending one.
The other islands keep merging and every island keeps getting PULLs. Once
nothing is pending, the island's broadcasts are sent inline again.
``--inline-broadcast`` keeps the old behavior; that case shows the stall.

The syncer lease (60 s) is longer than the test's slow phase: under the old
behavior a short lease ends the stall by removing the slow island (the
test_bulk_resend covers leases; that run's syncer log shows no lease
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

import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime

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
# After the slow phase island 1 drains its backlog and is merged again.
RECOVER_S = 8.0
SYNCER_LEASE_S = 60
# --min-round-interval-ms 20: the syncer merges at most this many rounds a
# second, so a phase of S seconds uses at most S * MAX_MERGES_PER_S steps.
MAX_MERGES_PER_S = 50
TOTAL_STEPS = 1200


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


def stall_report(tape, syncer_log, pull_times, since: float, until: float, other_pull_times=None) -> str:
    """What the syncer did in [since, until]: merges with each island and
    PULLs read by the slow island, per second; the rounds that ended at the
    quorum timeout (``timed_out`` on the tape) or took long, and the longest
    gaps between merges; the scheduler's quorum-timeout restarts in its log (a
    restarted round merges later with ``timed_out`` false and a fresh
    ``step_seconds``, so the log lines are what marks it). Printed when a slow-phase check fails.
    ``other_pull_times``: PULLs read by the other island; with them, the tape records and syncer
    log lines of the longest gap tell a scheduler that sent no PULL from an island that got its
    PULL but whose push was refused or never arrived."""
    records = tape_records(tape)
    steps = [r for r in records if r.get("kind") == "outer_step" and since - 30 <= r["time_unix"] <= until + 1]
    seconds = range(int(until - since) + 1)

    def per_second(times) -> list[int]:
        counts = [0 for _ in seconds]
        for t in times:
            if since <= t <= until:
                counts[min(int(t - since), len(counts) - 1)] += 1
        return counts

    merges = {island: per_second(r["time_unix"] for r in steps if any(w[0] == island for w in r.get("weights", [])))
              for island in (0, 1)}
    timed_out = [(round(r["time_unix"] - since, 2), r["step"], r["fragment"], r["arrived"], r["absent"])
                 for r in steps if r.get("timed_out")]
    long_rounds = [(round(r["time_unix"] - since, 2), r["step"], r["fragment"], round(r["step_seconds"], 2))
                   for r in steps if r.get("step_seconds", 0.0) >= 2.0]
    times = sorted(r["time_unix"] for r in steps if since <= r["time_unix"] <= until)
    gaps = sorted(((b - a, a - since) for a, b in zip([since, *times], [*times, until])), reverse=True)[:3]
    restarts = [line for line in syncer_log.read_text().splitlines()
                if "quorum timeout" in line or "fixed quorum unavailable" in line]
    tail = "\n".join(tape.read_text().splitlines()[-20:]) if tape.exists() else "(no tape)"
    gap_from = since + gaps[0][1] if gaps else since
    gap_until = gap_from + (gaps[0][0] if gaps else 0.0) + 0.5
    in_gap = [json.dumps(r) for r in records
              if gap_from - 0.2 <= r.get("time_unix", 0.0) <= gap_until and r.get("kind") != "outer_step"
              or r.get("kind") == "outer_step" and r.get("step_seconds", 0.0) >= 2.0][:40]
    gap_log = []
    for line in syncer_log.read_text().splitlines():
        try:
            at = datetime.fromisoformat(line.split(" ", 1)[0].replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if gap_from - 0.2 <= at <= gap_until and "outer step" not in line and "grace window" not in line:
            gap_log.append(line)
    return "\n".join([
        f"slow window {since:.2f}..{until:.2f} (unix), per second from its start:",
        f"  merges with island 0: {merges[0]}",
        f"  merges with island 1: {merges[1]}",
        f"  PULLs read by the slow island: {per_second(pull_times)}",
        *([f"  PULLs read by the other island: {per_second(other_pull_times)}"] if other_pull_times is not None else []),
        f"  rounds ended by the quorum timeout (t, step, fragment, arrived, absent): {timed_out or 'none'}",
        f"  longest gaps without a merge (seconds, from t): {[(round(g, 2), round(t, 2)) for g, t in gaps]}",
        f"  rounds of >= 2 s (t, step, fragment, step_seconds): {long_rounds or 'none'}",
        f"  quorum-timeout lines in the syncer log: {len(restarts)}",
        *(f"    {line}" for line in restarts[-10:]),
        "tape records of the longest gap (and of rounds of >= 2 s):", *in_gap,
        f"syncer log lines of the longest gap ({len(gap_log)}, at most 40):", *gap_log[:40],
        "tape tail:", tail,
    ])


def start_as_first_members(tmp_path, first, initializer) -> None:
    """Start ``first``, and only once the syncer has registered it start
    ``initializer`` (island 0, which uploads the INIT), so both islands are
    members when the syncer sends the initial cut.

    The syncer starts its rounds as soon as the INIT is in and the launch
    quorum (here one island) is connected. An island that connects after
    that joins through a catch-up and gets no initial cut; started together,
    the two islands race (the syncer hears island 1's HELLO, or island 0's
    whole INIT, first)."""
    log = tmp_path / "syncer.log"
    first.start()
    registered = f"learner_id={first.learner_id} "
    wait_until(lambda: first.first_generation is not None  # its data stream's hello is sent too
               and any("learner connected" in line and registered in line for line in log.read_text().splitlines()),
               60, f"island {first.learner_id} did not connect")
    initializer.start()


def start_syncer(tmp_path, port, *extra, total_steps=TOTAL_STEPS):
    binary = build_syncer()
    tape = tmp_path / "tape.jsonl"
    log = (tmp_path / "syncer.log").open("w")
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "2", "--quorum", "1", "--grace-ms", "50",
         "--total-steps", str(total_steps), "--outer-lr", "0.7", "--outer-momentum", "0.0",
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
        # Island 1 is a member before island 0's INIT: started together, it
        # could join after the rounds began, as a late joiner whose catch-up
        # used up rounds of the fixed run (on CI the run then ended before the
        # slow phase, which saw no merge and no PULL at all).
        start_as_first_members(tmp_path, b, a)
        wait_until(lambda: sum(a.versions) >= 40 and sum(b.versions) >= 40, 60, "no early rounds")
        # Both islands are ordinary members and both pushes are being merged.
        wait_until(lambda: steady(tape, tmp_path / "syncer.log"), 60, "islands not merging steadily")
        # The slow phase and island 1's recovery after it fit in the run's
        # remaining steps whatever the merge rate (it is at most
        # MAX_MERGES_PER_S), so the phase never outlasts the run.
        step = max((r["step"] for r in tape_records(tape) if r.get("kind") == "outer_step"), default=0)
        assert step + (SLOW_S + RECOVER_S) * MAX_MERGES_PER_S <= TOTAL_STEPS, (
            f"the islands merged steadily only at step {step} of {TOTAL_STEPS}")
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
        # Island 0 kept merging at full rate while island 1's downlink was
        # slow, and island 1 kept getting PULLs.
        if during < 60 or pulls_to_slow < 10:
            time.sleep(1.0)  # the tape's last records
            report = stall_report(tape, tmp_path / "syncer.log", b.pull_times, slow_from, slow_until,
                                  other_pull_times=a.pull_times)
            assert during >= 60, f"{during} merges with island 0\n{report}"
            assert pulls_to_slow >= 10, f"{pulls_to_slow} PULLs to the slow island\n{report}"
        for island in (a, b):
            island.join(timeout=180)
            assert not island.is_alive(), f"island {island.learner_id} did not finish"
            if island.exc:
                raise island.exc
        assert proc.wait(timeout=60) == 0, "syncer exited nonzero"
    except BaseException:
        log.flush()
        print((tmp_path / "syncer.log").read_text()[-6000:])
        print("tape tail:", *(tape.read_text().splitlines()[-20:] if tape.exists() else []), sep="\n")
        raise
    finally:
        for island in (a, b):
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text()
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


def test_env_downlink_limit_names_one_learner_and_opens_at_its_pull_step(monkeypatch):
    """The test-only downlink limit for GPU runs (``--modal-env``)
    applies to the named learner only, starts at its first PULL for a step
    >= FROM_STEP and ends SECONDS later. Unset: no limit."""
    import struct

    from tests.test_bulk_resend import make_layout
    from yeto.protocol import (
        DOWNLINK_READ_LIMIT_ENV,
        MSG_PULL_REQ,
        downlink_read_limit_from_env,
    )

    assert downlink_read_limit_from_env(1, {}) is None
    assert downlink_read_limit_from_env(0, {DOWNLINK_READ_LIMIT_ENV: "1:1e6:5:2"}) is None
    window = downlink_read_limit_from_env(1, {DOWNLINK_READ_LIMIT_ENV: "1:1.5e6:5:2"})
    assert (window.bps, window.from_step, window.seconds) == (1.5e6, 5, 2.0)
    for bad in ("1:1e6:5", "1:0:5:2", "1:1e6:5:0", "x:1e6:5:2"):
        with pytest.raises(ValueError):
            downlink_read_limit_from_env(1, {DOWNLINK_READ_LIMIT_ENV: bad})

    monkeypatch.setenv(DOWNLINK_READ_LIMIT_ENV, "1:1000000:5:0.3")
    layout = make_layout()
    other = SyncerClient(("127.0.0.1", 9), 0, layout, DTYPE_F32, syncer_epoch=0)
    assert other._downlink_window is None and other._downlink_read_rate is None
    client = SyncerClient(("127.0.0.1", 9), 1, layout, DTYPE_F32, syncer_epoch=0)
    rate = client._downlink_read_rate
    assert rate is not None and not rate.active()
    client._dispatch_live(client._gen, MSG_PULL_REQ, struct.pack("<IQI", 0, 4, 0))
    assert not rate.active(), "a PULL before FROM_STEP opened the window"
    client._dispatch_live(client._gen, MSG_PULL_REQ, struct.pack("<IQI", 1, 5, 0))
    assert rate.active()
    assert client._downlink_window.pulls_in_window == [5]
    time.sleep(0.35)
    assert not rate.active()
    client._dispatch_live(client._gen, MSG_PULL_REQ, struct.pack("<IQI", 2, 9, 0))
    assert not rate.active(), "the window opens only once"
    assert len(client.drain_pulls()) == 3


ENV_FROM_STEP = 400
# The window opens at step ENV_FROM_STEP and the run must go on long enough
# after it for island 1 to drain its forwarder backlog (about 100 MB) and be
# merged again.  Rounds are at least 20 ms apart, so at most 50 merges/s: by
# the end of the window the syncer is at step <= 400 + 14 * 50 = 1100, and
# 1700 steps leave at least 12 s after it.  1200 steps left 2 s at CI's merge
# rate, and the run could end before island 1 was merged again.
ENV_TOTAL_STEPS = 1700


@pytest.mark.timeout(300)
@pytest.mark.parametrize("inline", [False, True], ids=["forwarder", "inline-old-behavior"])
def test_env_downlink_limit_makes_a_running_island_slow(tmp_path, monkeypatch, inline):
    """The env limit (the GPU run's mechanism) on a plain client
    reproduces the slow downlink of the test above: the syncer sends island
    1's broadcasts through its forwarder, island 0 keeps merging, and island
    1 is merged again after the window. With ``--inline-broadcast`` (the old
    behavior) the same window stalls the merges."""
    from yeto.protocol import DOWNLINK_READ_LIMIT_ENV

    monkeypatch.setenv(DOWNLINK_READ_LIMIT_ENV, f"1:{SLOW_BPS}:{ENV_FROM_STEP}:{SLOW_S}")
    port = free_port()
    proc, tape, log = start_syncer(tmp_path, port, *(["--inline-broadcast"] if inline else []),
                                   total_steps=ENV_TOTAL_STEPS)
    store = tmp_path / "store"
    a = CountingIsland(0, port, store, dtype=DTYPE_F32)
    b = CountingIsland(1, port, store, dtype=DTYPE_F32)
    assert a.client._downlink_window is None and b.client._downlink_window is not None
    try:
        start_as_first_members(tmp_path, b, a)
        wait_until(lambda: steady(tape, tmp_path / "syncer.log"), 60, "islands not merging steadily")
        window = b.client._downlink_window
        assert window._started is None, "the limit started before both islands merged steadily"
        wait_until(lambda: window._started is not None, 60, "no PULL reached the window's step")
        slow_from = time.time() - (time.monotonic() - window._started)
        slow_until = slow_from + SLOW_S
        time.sleep(max(0.0, slow_until - time.time()) + 0.5)
        filled = slow_from + FILLED_S
        during = merges_with(tape_records(tape), 0, filled, slow_until)
        pulls_to_slow = sum(1 for t in b.pull_times if filled <= t <= slow_until)
        if not inline:
            assert len(window.pulls_in_window) >= pulls_to_slow, "the window's own PULL count is short"
        print(f"env limit inline={inline}: merges with island 0 in the last {SLOW_S - FILLED_S:.0f} s of the "
              f"window: {during}; PULLs read by the slow island meanwhile: {pulls_to_slow}")
        if inline:
            assert during <= 10, f"{during} merges while the scheduler waited on the slow downlink"
            return
        if during < 60 or pulls_to_slow < 10:
            time.sleep(1.0)
            report = stall_report(tape, tmp_path / "syncer.log", b.pull_times, slow_from, slow_until,
                                  other_pull_times=a.pull_times)
            assert during >= 60, f"{during} merges with island 0\n{report}"
            assert pulls_to_slow >= 10, f"{pulls_to_slow} PULLs to the slow island\n{report}"
        for island in (a, b):
            island.join(timeout=180)
            assert not island.is_alive(), f"island {island.learner_id} did not finish"
            if island.exc:
                raise island.exc
        assert proc.wait(timeout=60) == 0, "syncer exited nonzero"
    except BaseException:
        log.flush()
        print((tmp_path / "syncer.log").read_text()[-6000:])
        print("tape tail:", *(tape.read_text().splitlines()[-20:] if tape.exists() else []), sep="\n")
        raise
    finally:
        for island in (a, b):
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text()
    assert "slow downlink: the learner's send queues are full" in syncer_log
    assert "slow downlink drained; broadcasts go inline again" in syncer_log
    assert b.out_of_order == []
    assert any(w[0] == 1 for r in records if r.get("kind") == "outer_step" and r["time_unix"] > slow_until + 2
               for w in r.get("weights", [])), "no merge with island 1 after the window"
    assert a.final_manifest is not None and a.final_manifest == b.final_manifest
