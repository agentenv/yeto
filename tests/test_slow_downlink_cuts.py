"""S27 G-FH: the initial cut, the final cut and a bf16-downlink catch-up do not wait on one slow downlink (real syncer, CPU, no Ray).

G-FD moved the per-round broadcast off the scheduler (``test_slow_downlink``).
Three sends still waited on each learner's link in turn (SLOW-DOWNLINK.md
section 9.3):

- the initial cut (``broadcast_all_fragments``): one slow island held the other
  islands' initial cut and the first round;
- the final cut (``finalize_learners``): a slow island sorted first held the
  other islands' final cut, and the scheduler read no event meanwhile;
- a catch-up on a bf16-downlink session (``CatchUps``): each fragment send
  waited on the rejoining island's link, so merges stalled.

Now the initial cut spills to the island's forwarder (exactly and in order:
version 0 first, then newer versions), each island's final cut goes out on its
own task, and a bf16 catch-up goes out on the forwarder as FULLs with the
shared anchor. ``--inline-broadcast`` keeps the old initial and final cuts,
``--inline-resend`` the old catch-up; those cases show the stall.

The slow downlink is the test-only read limit of ``test_slow_downlink``
(``SlowDownlinkClient``, 128 KiB/s on the data socket while an event is set;
control frames at full speed).
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time

import pytest

from tests.test_bulk_resend import FRAGMENTS, _log_seconds, merges_with, tape_records
from tests.test_peer_recovery import build_syncer, free_port, wait_until
from tests.test_slow_downlink import FILLED_S, FULL_EVERY, SYNCER_LEASE_S, CountingIsland, steady
from yeto.protocol import DTYPE_F32, DTYPE_FLAG_BCAST_DELTA

# Island 1 reads its initial cut slowly for this long after the INIT (the
# 64 MiB INIT upload itself takes about 5 s here). Its writer queue and socket
# buffers take about half of the cut at once; the rest needs about 4 min at
# 128 KiB/s, so the whole window is slow.
INITIAL_SLOW_S = 12.0
# The slow island at the final cut: it turns slow this many steps before the
# last one (a few seconds of merges) and stays slow for FINAL_SLOW_S.
FINAL_LEAD_STEPS = 150
FINAL_SLOW_S = 20.0
CATCH_UP_SLOW_S = 14.0


class CuttingIsland(CountingIsland):
    """A :class:`CountingIsland` that also records when it held every
    fragment for the first time and when it acknowledged the final cut."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.full_cut_at: float | None = None
        self.acked_at: float | None = None
        acknowledge = self.client.acknowledge_finalization

        def timed_acknowledge(*a, **k):
            result = acknowledge(*a, **k)
            self.acked_at = time.time()
            return result

        self.client.acknowledge_finalization = timed_acknowledge

    def apply_broadcast(self, update) -> None:
        super().apply_broadcast(update)
        if self.full_cut_at is None and all(self.anchored):
            self.full_cut_at = time.time()


def start_syncer(tmp_path, port, total_steps: int, *extra):
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


def finish(proc, islands) -> None:
    for island in islands:
        island.join(timeout=180)
        assert not island.is_alive(), f"island {island.learner_id} did not finish"
        if island.exc:
            raise island.exc
    assert proc.wait(timeout=60) == 0, "syncer exited nonzero"


def dump_on_failure(tmp_path, log) -> None:
    log.flush()
    print((tmp_path / "syncer.log").read_text()[-6000:])


@pytest.mark.timeout(300)
@pytest.mark.parametrize("inline", [False, True], ids=["forwarder", "inline-old-behavior"])
def test_a_slow_initial_cut_does_not_hold_the_other_island(tmp_path, inline):
    """Island 1's downlink is slow from its first byte. Island 0 must get its
    initial cut at once and merge; island 1 gets version 0 of every fragment
    before any newer version (an island outside a pool refuses anything
    else), then catches up and is merged."""
    port = free_port()
    proc, tape, log = start_syncer(tmp_path, port, 1500, *(["--inline-broadcast"] if inline else []))
    slow = threading.Event()
    slow.set()
    store = tmp_path / "store"
    a = CuttingIsland(0, port, store, dtype=DTYPE_F32)
    b = CuttingIsland(1, port, store, dtype=DTYPE_F32, slow=slow)
    try:
        started = time.time()
        a.start()
        b.start()
        wait_until(lambda: "global parameters initialized" in (tmp_path / "syncer.log").read_text(), 60, "no INIT")
        initialized = next(_log_seconds(line) for line in (tmp_path / "syncer.log").read_text().splitlines()
                           if "global parameters initialized" in line)
        time.sleep(max(0.0, initialized + INITIAL_SLOW_S - time.time()))
        slow_until = time.time()
        a_cut_s = None if a.full_cut_at is None else round(a.full_cut_at - initialized, 2)
        during = merges_with(tape_records(tape), 0, started, slow_until)
        b_held = sum(b.anchored)
        slow.clear()
        print(f"inline={inline}: island 0 held its initial cut {a_cut_s} s after INIT; merges with island 0 in the "
              f"{INITIAL_SLOW_S:.0f} s after INIT: {during}; fragments island 1 held then: {b_held}/{FRAGMENTS}")
        if inline:
            # Old behavior: the scheduler sent each fragment to island 1 in
            # turn, so island 0's cut and the first round waited on its link.
            assert a_cut_s is None, f"island 0 held its cut after {a_cut_s} s"
            assert during == 0, during
            return
        finish(proc, (a, b))
    except BaseException:
        dump_on_failure(tmp_path, log)
        raise
    finally:
        for island in (a, b):
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text()
    assert a_cut_s is not None and a_cut_s < 4.0, a_cut_s
    assert during >= 100, during
    assert b_held < FRAGMENTS, "island 1's initial cut was not slow"
    assert "initial cut: the learner's send queues are full; a forwarder sends the rest of it" in syncer_log
    # Island 1 got exactly version 0 of every fragment first, then newer ones.
    first_newer = next(i for i, (_, _, version) in enumerate(b.bcasts) if version > 0)
    initial = b.bcasts[:first_newer]
    assert sorted(p for _, p, _ in initial) == list(range(FRAGMENTS)), "initial cut incomplete or repeated"
    assert b.out_of_order == []
    assert any(w[0] == 1 for r in records if r.get("kind") == "outer_step" and r["time_unix"] > slow_until + 2
               for w in r.get("weights", [])), "no merge with island 1 after its downlink recovered"
    assert not [r for r in records if r.get("kind") == "pool_leave"]
    assert a.final_manifest is not None and a.final_manifest == b.final_manifest


@pytest.mark.timeout(300)
@pytest.mark.parametrize("inline", [False, True], ids=["per-learner-task", "inline-old-behavior"])
def test_a_slow_island_at_the_final_cut_holds_only_itself(tmp_path, inline):
    """Island 0 (sorted first) turns slow just before the last step and stays
    slow past it. Island 1 must get the final cut and acknowledge it at once;
    island 0 acknowledges once its link is back; both hold the same cut."""
    total = 1200
    port = free_port()
    proc, tape, log = start_syncer(tmp_path, port, total, *(["--inline-broadcast"] if inline else []))
    slow = threading.Event()
    store = tmp_path / "store"
    a = CuttingIsland(0, port, store, dtype=DTYPE_F32, slow=slow)
    b = CuttingIsland(1, port, store, dtype=DTYPE_F32)
    try:
        a.start()
        b.start()
        wait_until(lambda: steady(tape, tmp_path / "syncer.log"), 60, "islands not merging steadily")
        wait_until(lambda: max(b.versions) >= total - FINAL_LEAD_STEPS, 120, "no late rounds")
        slow.set()
        slow_from = time.time()
        time.sleep(FINAL_SLOW_S)
        slow_until = time.time()
        b_acked = b.acked_at
        slow.clear()
        finish(proc, (a, b))
    except BaseException:
        dump_on_failure(tmp_path, log)
        raise
    finally:
        for island in (a, b):
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    syncer_log = (tmp_path / "syncer.log").read_text()
    last_merge = max(r["time_unix"] for r in tape_records(tape) if r.get("kind") == "outer_step")
    print(f"inline={inline}: last merge {last_merge - slow_from:.1f} s after island 0 turned slow; island 1 "
          f"acknowledged after {None if b_acked is None else round(b_acked - slow_from, 1)} s, island 0 after "
          f"{a.acked_at - slow_from:.1f} s (slow window {FINAL_SLOW_S:.0f} s)")
    assert "all learners acknowledged final cut learners=2 absent=0" in syncer_log
    assert a.final_manifest is not None and a.final_manifest == b.final_manifest
    assert a.acked_at > slow_until, "island 0's final cut was not slow"
    if inline:
        # Old behavior: every broadcast, the final cut included, waited on
        # island 0's link (the rounds stall too, as in test_slow_downlink);
        # island 1 got its final cut only after island 0's window.
        assert b_acked is None, f"island 1 acknowledged {b_acked - slow_from:.1f} s into the window"
        return
    assert last_merge < slow_until - 5, "the rounds did not end inside the slow window"
    assert b_acked is not None and b_acked < last_merge + 5, (b_acked, last_merge)
    assert "final cut: each learner's send task started" in syncer_log


@pytest.mark.timeout(300)
@pytest.mark.parametrize("inline", [False, True], ids=["forwarder", "inline-old-behavior"])
def test_a_slow_catch_up_on_a_delta_session_does_not_stall_merges(tmp_path, inline):
    """bf16-downlink session: island 1 redials and its catch-up (every
    fragment as a FULL) goes over a slow link. Island 0 must keep merging;
    island 1 decodes every FULL (no resync redial), and is merged again."""
    port = free_port()
    extra = ["--bcast-delta-full-every", str(FULL_EVERY), *(["--inline-resend"] if inline else [])]
    proc, tape, log = start_syncer(tmp_path, port, 1500, *extra)
    dtype = DTYPE_F32 | DTYPE_FLAG_BCAST_DELTA
    slow = threading.Event()
    store = tmp_path / "store"
    a = CuttingIsland(0, port, store, dtype=dtype)
    b = CuttingIsland(1, port, store, dtype=dtype, slow=slow)
    try:
        a.start()
        b.start()
        wait_until(lambda: steady(tape, tmp_path / "syncer.log"), 60, "islands not merging steadily")
        a_generation, first = a.client.connection_generation, b.client.connection_generation
        slow.set()
        b.client._socks[0].shutdown(socket.SHUT_RDWR)  # the connection drops; the client redials
        wait_until(lambda: b.client.connection_generation != first, 30, "island 1 did not redial")
        catching_up = b.client.connection_generation
        slow_from = time.time()
        time.sleep(CATCH_UP_SLOW_S)
        slow_until = time.time()
        during = merges_with(tape_records(tape), 0, slow_from + FILLED_S, slow_until)
        slow.clear()
        print(f"inline={inline}: merges with island 0 in the last {CATCH_UP_SLOW_S - FILLED_S:.0f} s of the slow "
              f"catch-up: {during}")
        if inline:
            # Old behavior: each catch-up fragment send waited on the link.
            assert during <= 10, f"{during} merges while the scheduler waited on the slow catch-up"
            return
        finish(proc, (a, b))
    except BaseException:
        dump_on_failure(tmp_path, log)
        raise
    finally:
        for island in (a, b):
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text()
    assert during >= 60, during
    assert "background resend started" in syncer_log and "kind=CatchUp" in syncer_log
    # No resync redial: every FULL decoded on the island's anchors.
    assert b.client.connection_generation == catching_up and a.client.connection_generation == a_generation
    assert b.out_of_order == []
    joined = [r for r in records if r.get("kind") == "pool_join" and r.get("island_id") == 1
              and r.get("generation") == catching_up]
    assert len(joined) == 1 and joined[0]["catch_up"], joined
    assert any(w[0] == 1 for r in records if r.get("kind") == "outer_step" and r["time_unix"] > slow_until + 2
               for w in r.get("weights", [])), "no merge with the caught-up island"
    assert a.final_manifest is not None and a.final_manifest == b.final_manifest


def test_env_downlink_limit_can_open_at_connect_and_name_two_learners():
    """S27 G-FH: the GPU check slows island 1's initial cut (FROM_STEP
    "connect") and island 0's final cut (a PULL step near the end) in one
    run: one spec per learner, comma-separated."""
    from tests.test_bulk_resend import make_layout
    from yeto.protocol import DOWNLINK_READ_LIMIT_ENV, SyncerClient, downlink_read_limit_from_env

    value = "1:3e6:connect:0.3,0:1.5e6:60:600"
    first = downlink_read_limit_from_env(1, {DOWNLINK_READ_LIMIT_ENV: value})
    assert (first.bps, first.from_step, first.seconds) == (3e6, None, 0.3)
    last = downlink_read_limit_from_env(0, {DOWNLINK_READ_LIMIT_ENV: value})
    assert (last.bps, last.from_step, last.seconds) == (1.5e6, 60, 600.0)
    assert downlink_read_limit_from_env(2, {DOWNLINK_READ_LIMIT_ENV: value}) is None
    for bad in ("1:3e6:connect:1,1:1e6:5:2", "1:3e6:connect:1,0:1e6:5"):
        with pytest.raises(ValueError):
            downlink_read_limit_from_env(0, {DOWNLINK_READ_LIMIT_ENV: bad})
    # A PULL never opens a "connect" window; the client's start does.
    first.see_pull(10**6)
    assert not first.active()
    first.see_connect()
    assert first.active()
    last.see_connect()
    assert not last.active(), "a PULL-step window ignores the connect"
    time.sleep(0.35)
    assert not first.active()
    first.see_connect()
    assert not first.active(), "the window opens only once"
    # The client opens it in start(): constructing one does not.
    os.environ[DOWNLINK_READ_LIMIT_ENV] = value
    try:
        client = SyncerClient(("127.0.0.1", 9), 1, make_layout(), DTYPE_F32, syncer_epoch=0)
    finally:
        del os.environ[DOWNLINK_READ_LIMIT_ENV]
    assert client._downlink_window.from_step is None and not client._downlink_read_rate.active()
