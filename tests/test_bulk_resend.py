"""S27 G-EL: a slow receiver of a whole-model resend against the real syncer (CPU, no Ray).

Run s27-elastic-a (Nemotron Nano): after a degraded recovery the syncer sent
all 126 GB of fragments from its scheduler, waiting on the island's link for
8.8 min. No heartbeat was read meanwhile, so both islands' 150 s leases
expired and both were removed from the pool. Now a forwarder task per learner
sends a catch-up or recovery replay (``syncer/src/resend.rs``) while the
scheduler keeps merging and renewing leases. ``--inline-resend`` keeps the
old behavior; the inline case below shows the failure the default fixes.

The slow receiver is a recovering island whose data socket is not read while
a gate is closed, so the syncer's writer queue and the socket buffers fill.
The model (256 fragments of 256 KiB) is larger than the 128-frame writer
queue plus the socket buffers.

S27 G-EY: in run s27-elastic-c the replay then never ended. With one island
merging, versions were published about as fast as the link to the recovering
island took them, so the forwarder's pending map almost never emptied, and
"empty" was the tail condition. The last test reads the recovering island's
data sockets at a fixed rate below the publish rate: the tail must run once the
versions owed at the start are handed over. That island is slowed by the
client's own test-only read limit (``YETO_SYNC_TEST_RECOVERY_READ_BPS``), the
one a GPU run sets with ``--modal-env``.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tests.test_peer_recovery import build_syncer, free_port, wait_until
from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, BcastFragment, SyncerClient
from yeto.rl.peer_recovery import DirectorySnapshotStore, recover
from yeto.rl.pool_client import LeaseHeartbeat

FRAGMENTS = 256
NUMEL = 1 << 16  # 256 KiB of f32: one chunk frame per fragment
# The INIT of the 64 MiB model and its initial broadcast take several seconds
# here, and the scheduler still sends that broadcast itself before training,
# so a lease may lapse once at startup (both islands re-JOIN). The checks
# below cover the slow-link phase only, which starts after the early rounds.
LEASE_S = 4.0
GATE_S = 2.5 * LEASE_S


def make_layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_AVG, [(f"model.layers.{i}.weight", NUMEL)]) for i in range(FRAGMENTS)])


class _GatedSocket:
    """A data socket whose reads wait while ``gate`` is clear."""

    def __init__(self, sock: socket.socket, gate: threading.Event) -> None:
        self._sock = sock
        self._gate = gate

    def recv_into(self, *args, **kwargs):
        self._gate.wait()
        return self._sock.recv_into(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sock, name)


class GatedClient(SyncerClient):
    """Reads its data streams only while ``gate`` is set (control frames
    always): a learner on a slow link."""

    def __init__(self, *args, gate: threading.Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate = gate

    def _recv_loop(self, gen, sock):
        if self._socks and sock is not self._socks[0]:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)
            sock = _GatedSocket(sock, self.gate)
        super()._recv_loop(gen, sock)


class LightIsland(threading.Thread):
    """Answers every PULL with a zero delta (weights stay put; versions are
    what the test follows), heartbeats its lease, finalizes. A
    :class:`~yeto.rl.peer_recovery.RecoverableIsland` for the degraded path."""

    def __init__(self, learner_id: int, port: int, store: Path, *, recover_on_start: bool = False,
                 gate: threading.Event | None = None) -> None:
        super().__init__(daemon=True)
        self.learner_id = learner_id
        self.store = DirectorySnapshotStore(store)
        self.layout = make_layout()
        self.recover_on_start = recover_on_start
        self.recover_timeout = 120.0
        self.versions = [0] * FRAGMENTS
        self.anchored = [False] * FRAGMENTS
        self.steps = 0
        kwargs = dict(syncer_epoch=0, recover=recover_on_start, num_streams=1)
        if gate is None:
            self.client = SyncerClient(("127.0.0.1", port), learner_id, self.layout, DTYPE_F32, **kwargs)
        else:
            self.client = GatedClient(("127.0.0.1", port), learner_id, self.layout, DTYPE_F32, gate=gate,
                                      **kwargs)
        self.heartbeat = LeaseHeartbeat(self.client, LEASE_S, progress=lambda: self.steps)
        self.zero = np.zeros(NUMEL, np.float32).tobytes()
        self.killed = threading.Event()
        self.first_generation: int | None = None
        self.restored_at: float | None = None
        self.outcome = None
        self.final_manifest = None
        self.pushes = 0
        self.exc: BaseException | None = None

    # -- RecoverableIsland ------------------------------------------------------
    def fragment_versions(self):
        return tuple(self.versions)

    def layout_tag(self) -> str:
        return f"light-{FRAGMENTS}x{NUMEL}"

    def export_recovery_state(self):
        return {"versions": np.array(self.versions, np.int64)}

    def import_recovery_state(self, state, versions) -> None:
        self.versions = list(versions)

    def apply_broadcast(self, update: BcastFragment) -> None:
        p = update.fragment_id
        self.versions[p] = max(self.versions[p], update.version)
        self.anchored[p] = True

    def restore_degraded(self):
        self.restored_at = time.monotonic()
        return None  # nothing held: the syncer resends every fragment

    # -- loop -------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # surfaced by the test
            self.exc = exc
        finally:
            self.heartbeat.stop()
            self.client.close()

    def _run(self) -> None:
        self.client.start()
        self.first_generation = self.client.connection_generation
        self.heartbeat.start()
        if self.recover_on_start:
            self.outcome = recover(self.client, self, self.store, timeout=self.recover_timeout)
            self.client.mark_initial_cut_received()  # as the stream does after a recovery
        elif self.learner_id == 0:
            for p in range(FRAGMENTS):
                self.client.send_init(p, self.zero)
        pending = []
        deadline = time.monotonic() + 180
        while not self.killed.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("no final manifest")
            self.client.check_health()
            if self.client.finalizing.is_set():
                manifest, _ = self.client.wait_for_final_fragments(timeout=60)
                self.client.acknowledge_finalization(manifest, timeout=60)
                self.final_manifest = manifest
                return
            for update in self.client.drain_updates():
                self.apply_broadcast(update)
            self.steps += 1
            pending.extend(self.client.drain_pulls())
            still = []
            for pull in pending:
                p = pull.fragment_id
                if not self.anchored[p]:
                    still.append(pull)
                    continue
                self.client.push_fragment(p, pull.global_step, pull.round_attempt, self.versions[p],
                                          self.steps, 1, 16, self.zero)
                self.pushes += 1
            pending = still
            time.sleep(0.005)


def tape_records(tape: Path) -> list[dict]:
    if not tape.exists():
        return []
    return [json.loads(line) for line in tape.read_text().splitlines() if line.strip()]


def merges_with(records: list[dict], learner_id: int, since_unix: float, until_unix: float) -> int:
    return sum(1 for r in records if r.get("kind") == "outer_step" and since_unix <= r["time_unix"] <= until_unix
               and any(w[0] == learner_id for w in r.get("weights", [])))


@pytest.mark.timeout(300)
@pytest.mark.parametrize("inline", [False, True], ids=["background", "inline-old-behavior"])
def test_slow_receiver_of_a_degraded_replay_does_not_stall_rounds_or_leases(tmp_path, inline):
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    log = (tmp_path / "syncer.log").open("w")
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "2", "--quorum", "1", "--grace-ms", "50",
         "--total-steps", "1000", "--outer-lr", "0.7", "--outer-momentum", "0.0",
         "--min-round-interval-ms", "20", "--quorum-timeout-s", "30", "--event-tape", str(tape),
         "--island-lease-s", str(LEASE_S), "--syncer-epoch", "0", "--max-base-lag", "2",
         "--recovery-max-donor-tries", "1", "--recovery-donor-timeout-s", "1",
         *(["--inline-resend"] if inline else [])],
        stdout=log, stderr=subprocess.STDOUT, text=True)
    gate = threading.Event()
    islands: list[LightIsland] = []
    try:
        store = tmp_path / "store"
        a = LightIsland(0, port, store)
        b = LightIsland(1, port, store)
        islands += [a, b]
        a.start()
        b.start()
        wait_until(lambda: sum(a.versions) >= 40, 60, "no early rounds")
        phase_from, a_generation = time.time(), a.client.connection_generation
        b.killed.set()  # trainer state lost
        b.join(timeout=10)
        # A recovering island 1 on a slow link; island 0 never answers the
        # snapshot request, so the donor times out and the replay is everything.
        back = LightIsland(1, port, store, recover_on_start=True, gate=gate)
        islands.append(back)
        back.start()
        wait_until(lambda: back.restored_at is not None or back.exc is not None, 60, "no degraded source")
        if back.exc:
            raise back.exc
        gate_from = time.time()
        time.sleep(GATE_S)
        gate_until = time.time()
        records = tape_records(tape)
        during = merges_with(records, 0, gate_from + 0.5, gate_until)
        gate.set()
        if inline:
            # Old behavior: the scheduler waited on the slow link, so no round
            # merged, and island 0's lease expired once it was free again.
            def leave():
                return [r for r in tape_records(tape) if r.get("kind") == "pool_leave"
                        and r.get("island_id") == 0 and r["time_unix"] > phase_from]

            wait_until(leave, 30, "island 0 kept its lease")
            assert during == 0, f"{during} merges while the scheduler sent the replay"
            assert leave()[0]["reason"] == "lease_expired"
            return
        for island in (a, back):
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
        for island in islands:
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text()
    print(f"merges with island 0 while the replay waited on the slow link: {during}")
    # Rounds kept merging while the replay waited on the slow link.
    assert during >= 10, during
    # Island 0 kept its lease and its connection from then on.
    assert not [r for r in records if r.get("kind") == "pool_leave" and r.get("island_id") == 0
                and r["time_unix"] > phase_from]
    assert a.client.connection_generation == a_generation
    assert a.heartbeat.errors == []
    # The recovering island kept its lease too, took the degraded path, got
    # every fragment, rejoined the pool and its pushes were merged.
    gen = back.client.connection_generation
    assert not [r for r in records if r.get("kind") == "pool_leave" and r.get("generation") == gen]
    assert back.outcome.path == "degraded" and back.outcome.replayed == FRAGMENTS, back.outcome
    joined = [r for r in records if r.get("kind") == "pool_join" and r.get("recovered")]
    assert [(r["island_id"], r["generation"]) for r in joined] == [(1, gen)]
    after = [r for r in records if r.get("kind") == "outer_step" and r["time_unix"] > joined[0]["time_unix"]]
    assert any(w[0] == 1 for r in after for w in r.get("weights", [])), "no merge with the recovered island"
    assert a.final_manifest is not None and a.final_manifest == back.final_manifest
    assert "background resend started" in syncer_log and "background resend drained" in syncer_log


@pytest.mark.timeout(300)
@pytest.mark.parametrize("inline", [False, True], ids=["background", "inline-old-behavior"])
def test_slow_receiver_of_a_rejoin_catch_up_does_not_stall_rounds_or_leases(tmp_path, inline):
    """The other whole-model resend: an island reconnects (a new generation:
    the pool's re-JOIN after a lease expiry takes the same path) and gets the
    current value of every fragment over a slow link."""
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    log = (tmp_path / "syncer.log").open("w")
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "2", "--quorum", "1", "--grace-ms", "50",
         "--total-steps", "1000", "--outer-lr", "0.7", "--outer-momentum", "0.0",
         "--min-round-interval-ms", "20", "--quorum-timeout-s", "30", "--event-tape", str(tape),
         "--island-lease-s", str(LEASE_S), "--syncer-epoch", "0", "--max-base-lag", "2",
         *(["--inline-resend"] if inline else [])],
        stdout=log, stderr=subprocess.STDOUT, text=True)
    gate = threading.Event()
    gate.set()
    store = tmp_path / "store"
    a = LightIsland(0, port, store)
    b = LightIsland(1, port, store, gate=gate)
    try:
        a.start()
        b.start()
        wait_until(lambda: sum(a.versions) >= 40 and sum(b.versions) >= 40, 60, "no early rounds")
        phase_from, a_generation = time.time(), a.client.connection_generation
        first = b.client.connection_generation
        gate.clear()  # a slow link from now on
        b.client._socks[0].shutdown(socket.SHUT_RDWR)  # the connection drops; the client redials
        wait_until(lambda: b.client.connection_generation != first, 30, "island 1 did not redial")
        catching_up = b.client.connection_generation
        gate_from = time.time()
        time.sleep(GATE_S)
        gate_until = time.time()
        during = merges_with(tape_records(tape), 0, gate_from + 0.5, gate_until)
        # The inline catch-up sends one fragment per scheduler pass: rounds go on
        # until the writer queue is full, so look at the second half of the window.
        late = merges_with(tape_records(tape), 0, gate_from + GATE_S / 2, gate_until)
        gate.set()
        if inline:
            # Old behavior: the scheduler waited on the slow link, so no round
            # merged and heartbeats were not read: a lease expired (which island's
            # depends on the order of the queued events; that island re-JOINs for
            # another whole-model catch-up).
            def leave():
                return [r for r in tape_records(tape) if r.get("kind") == "pool_leave"
                        and r.get("reason") == "lease_expired" and r["time_unix"] > gate_from]

            wait_until(leave, 30, "every lease held")
            assert late == 0, f"{late} merges while the scheduler waited on the slow link"
            assert {r["generation"] for r in leave()} & {catching_up, a_generation}
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
        a.killed.set()
        b.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    print(f"merges with island 0 while the catch-up waited on the slow link: {during} ({late} late)")
    assert during >= 10 and late >= 5, (during, late)
    assert not [r for r in records if r.get("kind") == "pool_leave" and r.get("island_id") == 0
                and r["time_unix"] > phase_from]
    assert a.client.connection_generation == a_generation and a.heartbeat.errors == []
    gen = b.client.connection_generation
    assert not [r for r in records if r.get("kind") == "pool_leave" and r.get("generation") == gen]
    joined = [r for r in records if r.get("kind") == "pool_join" and r.get("island_id") == 1
              and r.get("generation") == gen]
    assert len(joined) == 1 and joined[0]["catch_up"], joined
    after = [r for r in records if r.get("kind") == "outer_step" and r["time_unix"] > joined[0]["time_unix"]]
    assert any(w[0] == 1 for r in after for w in r.get("weights", [])), "no merge with the caught-up island"
    assert a.final_manifest is not None and a.final_manifest == b.final_manifest


# S27 G-EY: island 0 alone merges about 40 fragments of 256 KiB per second
# here (about 10 MiB/s); the recovering island reads 8 MiB/s.
SLOW_LINK = 8 * 1024 * 1024
# The old tail condition ("nothing pending") was not met in 63 s at this rate
# (the 3000 steps ended first); the new one is met after about 4 s, and the
# island holds the versions RECOVERY_DONE names after about 16 s (behind a
# 32 MiB writer queue).
RECOVERY_WINDOW_S = 90.0


def _log_seconds(line: str) -> float:
    """Unix time of a syncer log line (RFC 3339 UTC prefix)."""
    from datetime import datetime

    return datetime.fromisoformat(line.split()[0].replace("Z", "+00:00")).timestamp()


def test_recovery_read_limit_holds_only_while_a_recovering_client_waits_for_its_cut(monkeypatch):
    from yeto.protocol import RECOVERY_READ_BPS_ENV, _ReadRate, recovery_read_bps_from_env

    assert recovery_read_bps_from_env({}) is None
    assert recovery_read_bps_from_env({RECOVERY_READ_BPS_ENV: "0"}) is None
    assert recovery_read_bps_from_env({RECOVERY_READ_BPS_ENV: "2.5e6"}) == 2.5e6
    monkeypatch.setenv(RECOVERY_READ_BPS_ENV, "1000000")
    layout = make_layout()
    plain = SyncerClient(("127.0.0.1", 9), 0, layout, DTYPE_F32, syncer_epoch=0)
    assert plain._recovery_read_rate is None
    recovering = SyncerClient(("127.0.0.1", 9), 1, layout, DTYPE_F32, syncer_epoch=0, recover=True)
    rate = recovering._recovery_read_rate
    assert rate is not None and rate.active()
    recovering.mark_initial_cut_received()
    assert not rate.active()
    # The budget is shared: 4 x 50 kB at 1 MB/s take at least 0.2 s.
    budget = _ReadRate(1e6, lambda: True)
    started = time.monotonic()
    for _ in range(4):
        budget.spend(50_000)
    assert time.monotonic() - started >= 0.19


@pytest.mark.timeout(300)
def test_replay_tail_runs_when_the_link_is_slower_than_the_merges(tmp_path, monkeypatch):
    """A recovering island on a link slower than island 0's publications gets
    RECOVERY_DONE while the link is still slow: the tail waits only for the
    versions owed when the replay started, not for an empty pending map.
    Rounds keep merging every second meanwhile, and the island's pushes are
    merged afterwards."""
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    log = (tmp_path / "syncer.log").open("w")
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "2", "--quorum", "1", "--grace-ms", "50",
         "--total-steps", "3000", "--outer-lr", "0.7", "--outer-momentum", "0.0",
         "--min-round-interval-ms", "20", "--quorum-timeout-s", "30", "--event-tape", str(tape),
         "--island-lease-s", str(LEASE_S), "--syncer-epoch", "0", "--max-base-lag", "2",
         "--recovery-max-donor-tries", "1", "--recovery-donor-timeout-s", "1"],
        stdout=log, stderr=subprocess.STDOUT, text=True)
    from yeto.protocol import RECOVERY_READ_BPS_ENV

    monkeypatch.setenv(RECOVERY_READ_BPS_ENV, str(SLOW_LINK))  # applies to the recovering client only
    islands: list[LightIsland] = []
    try:
        store = tmp_path / "store"
        a = LightIsland(0, port, store)
        b = LightIsland(1, port, store)
        islands += [a, b]
        a.start()
        b.start()
        wait_until(lambda: sum(a.versions) >= 40, 60, "no early rounds")
        phase_from, a_generation = time.time(), a.client.connection_generation
        b.killed.set()  # trainer state lost
        b.join(timeout=10)
        # Island 0 never answers the snapshot request: the donor times out,
        # the source is degraded and the replay is every fragment.
        back = LightIsland(1, port, store, recover_on_start=True)
        back.recover_timeout = RECOVERY_WINDOW_S + 60
        islands.append(back)
        back.start()
        deadline = time.monotonic() + RECOVERY_WINDOW_S
        while back.outcome is None and back.exc is None:
            if time.monotonic() > deadline:
                pytest.fail(f"no RECOVERY_DONE within {RECOVERY_WINDOW_S:.0f}s on the slow link")
            time.sleep(0.05)
        if back.exc:
            raise back.exc
        done_at = time.time()
        # Not the degraded DONE an island gets when the run ends first (the
        # old tail condition never let the replay end before that). The read
        # limit ends with the recovery (mark_initial_cut_received).
        assert back.outcome.replayed == FRAGMENTS, f"RECOVERY_DONE without the replay: {back.outcome}"
        for island in (a, back):
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
        for island in islands:
            island.killed.set()
        if proc.poll() is None:
            proc.kill()
        log.close()
    records = tape_records(tape)
    syncer_log = (tmp_path / "syncer.log").read_text().splitlines()
    # (The 64 MiB start may also lapse a lease once: those re-JOIN catch-ups
    # are other resends.)
    started = [line for line in syncer_log if "background resend started" in line and "Recovery" in line]
    assert len(started) == 1, started
    replay = re.search(r"learner_id=1 generation=\d+ ", started[0]).group(0)
    caught_up = [line for line in syncer_log if "background resend caught up" in line and replay in line]
    assert len(caught_up) == 1, caught_up
    resend_from, tail_at = _log_seconds(started[0]), _log_seconds(caught_up[0])
    backlog = int(re.search(r"pending=(\d+)", caught_up[0]).group(1))
    merges = [r["time_unix"] for r in records if r.get("kind") == "outer_step" and resend_from <= r["time_unix"] <= tail_at]
    rate = len(merges) / max(tail_at - resend_from, 1e-9)
    print(f"replay of {FRAGMENTS} x {NUMEL * 4 >> 10} KiB: tail after {tail_at - resend_from:.1f}s with "
          f"{backlog} newer version(s) still pending; island 0 merged {rate:.1f}/s "
          f"({rate * NUMEL * 4 / SLOW_LINK:.2f}x the link); RECOVERY_DONE applied after "
          f"{done_at - resend_from:.1f}s")
    # The tail did not wait for an empty pending map.
    assert backlog > 0
    # Rounds merged in every whole second of the replay, and island 0 kept its lease.
    seconds = range(int(resend_from) + 1, int(tail_at))
    assert all(any(int(t) == second for t in merges) for second in seconds), "a second without a merge"
    assert not [r for r in records if r.get("kind") == "pool_leave" and r.get("island_id") == 0
                and r["time_unix"] > phase_from]
    assert a.client.connection_generation == a_generation and a.heartbeat.errors == []
    # The recovering island kept its lease, got every fragment, rejoined the
    # pool and its pushes were merged.
    gen = back.client.connection_generation
    assert not [r for r in records if r.get("kind") == "pool_leave" and r.get("generation") == gen]
    assert back.outcome.path == "degraded" and back.outcome.replayed == FRAGMENTS, back.outcome
    joined = [r for r in records if r.get("kind") == "pool_join" and r.get("recovered")]
    assert [(r["island_id"], r["generation"]) for r in joined] == [(1, gen)]
    after = [r for r in records if r.get("kind") == "outer_step" and r["time_unix"] > joined[0]["time_unix"]]
    assert any(w[0] == 1 for r in after for w in r.get("weights", [])), "no merge with the recovered island"
    assert a.final_manifest is not None and a.final_manifest == back.final_manifest
