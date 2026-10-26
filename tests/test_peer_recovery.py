"""Peer recovery of a killed island against the real syncer binary (CPU, no Ray).

Fake islands train ``||w - target||^2`` with a numpy Adam, so parameters and
inner optimizer state are plain arrays that can be compared bitwise. One island
is killed (its process state is gone), a fresh one with the same id connects
with ``recover=True`` and must come back with a donor's parameters and
optimizer state, receive only the fragments newer than the donor's snapshot,
and be back on the roster within the bound H (in syncer steps).
"""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from yeto import recovery_protocol as rp
from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, BcastFragment, SyncerClient, encode_hello, layout_fingerprint
from yeto.rl.pool_client import LeaseHeartbeat
from yeto.rl.peer_recovery import DirectorySnapshotStore, DonorService, Snapshot, read_array_state, recover

ROOT = Path(__file__).resolve().parent.parent
NUMEL = 64
FRAGMENTS = 4
BOUND_H = 8


def build_syncer() -> Path:
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_AVG, [(f"model.layers.{i}.weight", NUMEL)]) for i in range(FRAGMENTS)])


class FakeIsland(threading.Thread):
    """One island: inner Adam steps, applies broadcasts, answers pulls,
    serves donor snapshots, and finalizes. Implements RecoverableIsland."""

    def __init__(self, learner_id: int, port: int, target: np.ndarray, store: DirectorySnapshotStore,
                 *, recover_on_start: bool = False, syncer_epoch: int | None = None,
                 lease_s: float | None = None) -> None:
        super().__init__(daemon=True)
        self.learner_id = learner_id
        self.layout = make_layout()
        self.target = target.reshape(FRAGMENTS, NUMEL)
        self.store = store
        self.recover_on_start = recover_on_start
        self.params = np.zeros((FRAGMENTS, NUMEL), np.float32)
        self.anchors = np.zeros((FRAGMENTS, NUMEL), np.float32)
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.adam_step = 0
        self.steps_total = 0
        self.steps_at_anchor = [0] * FRAGMENTS
        self.versions = [0] * FRAGMENTS
        self.anchored = [False] * FRAGMENTS
        self.client = SyncerClient(("127.0.0.1", port), learner_id, self.layout, DTYPE_F32, num_streams=1,
                                   syncer_epoch=syncer_epoch, recover=recover_on_start)
        self.donor = DonorService(self.client, self, store)
        # Island pool member: heartbeats keep its lease (also during a restore).
        self.heartbeat = (None if lease_s is None
                          else LeaseHeartbeat(self.client, lease_s, progress=lambda: self.steps_total))
        self.killed = threading.Event()
        self.outcome = None
        self.imported: dict[str, np.ndarray] | None = None
        self.after_recovery: dict[str, np.ndarray] | None = None
        self.final_manifest = None
        self.pushes_after_recovery = 0
        self.exc: BaseException | None = None

    # -- RecoverableIsland ------------------------------------------------------
    def fragment_versions(self):
        return tuple(self.versions)

    def layout_tag(self) -> str:
        return f"fake-dp1-{FRAGMENTS}x{NUMEL}"

    def export_recovery_state(self):
        return {
            "params": self.params.copy(), "anchors": self.anchors.copy(), "adam_m": self.m.copy(),
            "adam_v": self.v.copy(), "adam_step": np.array([self.adam_step], np.int64),
            "steps_at_anchor": np.array(self.steps_at_anchor, np.int64) - self.steps_total,
        }

    def import_recovery_state(self, state, versions) -> None:
        self.imported = {k: np.array(v) for k, v in state.items()}
        self.params[:] = state["params"]
        self.anchors[:] = state["anchors"]
        self.m[:] = state["adam_m"]
        self.v[:] = state["adam_v"]
        self.adam_step = int(state["adam_step"][0])
        # Counters are relative to the donor's step count.
        self.steps_at_anchor = [self.steps_total + int(s) for s in state["steps_at_anchor"]]
        self.versions = list(versions)
        self.anchored = [True] * FRAGMENTS

    def apply_broadcast(self, update: BcastFragment) -> None:
        values = np.frombuffer(bytes(update.data), np.float32)
        p = update.fragment_id
        self.params[p] = values
        self.anchors[p] = values
        self.versions[p] = update.version
        self.steps_at_anchor[p] = self.steps_total
        self.anchored[p] = True

    def restore_degraded(self):
        self.m[:] = 0
        self.v[:] = 0
        self.adam_step = 0
        return None  # nothing reliable held: resend every fragment

    # -- loop -------------------------------------------------------------------
    def kill(self) -> None:
        self.killed.set()

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # surfaced by the test
            self.exc = exc
        finally:
            if self.heartbeat is not None:
                self.heartbeat.stop()
            self.client.close()

    def _adam(self) -> None:
        grad = 2.0 * (self.params - self.target)
        self.adam_step += 1
        self.m = (0.9 * self.m + 0.1 * grad).astype(np.float32)
        self.v = (0.999 * self.v + 0.001 * grad * grad).astype(np.float32)
        mhat = self.m / (1 - 0.9 ** self.adam_step)
        vhat = self.v / (1 - 0.999 ** self.adam_step)
        self.params = (self.params - 0.02 * mhat / (np.sqrt(vhat) + 1e-8)).astype(np.float32)
        self.steps_total += 1

    def _run(self) -> None:
        self.client.start()
        if self.heartbeat is not None:
            self.heartbeat.start()
        if self.recover_on_start:
            self.outcome = recover(self.client, self, self.store, timeout=60)
            self.after_recovery = self.export_recovery_state()
        elif self.learner_id == 0:
            for p in range(FRAGMENTS):
                self.client.send_init(p, self.params[p].tobytes())
        pending = []
        deadline = time.monotonic() + 120
        while not self.killed.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("no final manifest")
            self.client.check_health()
            if self.client.finalizing.is_set():
                manifest, _ = self.client.wait_for_final_fragments(timeout=20)
                self.client.acknowledge_finalization(manifest, timeout=20)
                self.final_manifest = manifest
                return
            self._adam()
            for update in self.client.drain_updates():
                self.apply_broadcast(update)
            self.donor.poll()
            pending.extend(self.client.drain_pulls())
            still = []
            for pull in pending:
                p = pull.fragment_id
                c_steps = self.steps_total - self.steps_at_anchor[p]
                if not self.anchored[p] or c_steps < 1:
                    still.append(pull)
                    continue
                delta = (self.params[p] - self.anchors[p]).astype(np.float32)
                self.client.push_fragment(p, pull.global_step, pull.round_attempt, self.versions[p],
                                          self.steps_total, c_steps, c_steps * 16, delta.tobytes())
                if self.outcome is not None:
                    self.pushes_after_recovery += 1
            pending = still
            time.sleep(0.004)


def start_syncer(binary: Path, port: int, learners: int, quorum: int, tape: Path, *extra: str,
                 total_steps: int = 80) -> subprocess.Popen:
    return subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", str(learners), "--quorum", str(quorum),
         "--grace-ms", "50", "--total-steps", str(total_steps), "--outer-lr", "0.7", "--outer-momentum", "0.0",
         "--min-round-interval-ms", "40", "--quorum-timeout-s", "30", "--event-tape", str(tape),
         "--recovery-bound-steps", str(BOUND_H), *extra],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def wait_until(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        time.sleep(0.01)


def finish(proc: subprocess.Popen, islands) -> str:
    try:
        for island in islands:
            island.join(timeout=120)
            assert not island.is_alive(), f"island {island.learner_id} did not finish"
            if island.exc:
                raise island.exc
        assert proc.wait(timeout=30) == 0, "syncer exited nonzero"
    finally:
        if proc.poll() is None:
            proc.kill()
    return proc.stdout.read() if proc.stdout else ""


def recovery_records(tape: Path) -> list[dict]:
    return [r for r in map(json.loads, tape.read_text().splitlines()) if r.get("kind") == "recovery"]


class RecordingStore(DirectorySnapshotStore):
    """Keeps every committed snapshot's array state, so the test can compare
    after the recovering island removed it."""

    def __init__(self, root, puts=None) -> None:
        super().__init__(root)
        self.puts = {} if puts is None else puts

    def manifest(self, locator, sha256):
        manifest = super().manifest(locator, sha256)
        if locator not in self.puts:
            files = self.root / self.snapshot_prefix(locator) / "files"
            self.puts[locator] = Snapshot(manifest.versions, manifest.layout_tag, read_array_state(files))
        return manifest


class SlowStore(RecordingStore):
    """Snapshot reads take ``delay`` seconds (a transfer over a slow link)."""

    def __init__(self, root, delay: float, puts=None) -> None:
        super().__init__(root, puts)
        self.delay = delay

    def fetch(self, locator, sha256, into):
        time.sleep(self.delay)
        return super().fetch(locator, sha256, into)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("transfer_s, expect_missed", [(0.3, False), (1.0, True)])
def test_killed_island_recovers_from_a_peer_within_bound(tmp_path, transfer_s, expect_missed):
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    store = RecordingStore(tmp_path / "store")
    rng = np.random.default_rng(0)
    targets = [rng.standard_normal(FRAGMENTS * NUMEL).astype(np.float32) for _ in range(3)]
    proc = start_syncer(binary, port, 3, 2, tape)
    out = ""
    try:
        islands = [FakeIsland(i, port, targets[i], store) for i in range(3)]
        for island in islands:
            island.start()
        wait_until(lambda: min(islands[0].versions) >= 8, 60, "no early rounds")
        islands[2].kill()  # trainer state lost
        islands[2].join(timeout=10)
        kill_versions = list(islands[0].versions)
        wait_until(lambda: sum(islands[0].versions) > sum(kill_versions) + 8, 60, "rounds stopped without island 2")
        back = FakeIsland(2, port, targets[2], SlowStore(store.root, transfer_s, store.puts), recover_on_start=True)
        back.start()
        wait_until(lambda: back.outcome is not None or back.exc is not None, 60, "recovery did not finish")
        if back.exc:
            raise back.exc
        live = [islands[0], islands[1], back]
        out = finish(proc, live)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-4000:])
        raise
    outcome = back.outcome
    print(f"recovery outcome: {outcome}")
    assert outcome.path == "peer" and outcome.donor in (0, 1), outcome
    assert outcome.steps > 0, "rounds kept committing while the snapshot was in transfer"
    assert (outcome.steps > BOUND_H) == expect_missed, outcome
    donor = islands[outcome.donor]
    (request, locator), = donor.donor.served
    assert (request.recovering_id, request.recovering_generation) == (2, back.client.connection_generation)
    snapshot = store.puts[locator]
    assert store.locators() == [], "a restored snapshot is removed"
    # Inner optimizer state equals the donor's at its snapshot, bitwise.
    for key in ("adam_m", "adam_v", "adam_step", "anchors", "params", "steps_at_anchor"):
        np.testing.assert_array_equal(back.imported[key], snapshot.state[key], err_msg=key)
    # The snapshot was taken at or after the syncer clock.
    assert all(d >= s for d, s in zip(snapshot.versions, request.versions))
    # Only fragments newer than the donor snapshot were resent; the island
    # now holds the syncer versions named in RECOVERY_DONE.
    assert outcome.restored_versions == snapshot.versions
    newer = [p for p in range(FRAGMENTS) if outcome.versions[p] > snapshot.versions[p]]
    assert outcome.replayed == len(newer) > 0
    for p in range(FRAGMENTS):
        if p not in newer:
            np.testing.assert_array_equal(back.after_recovery["params"][p], snapshot.state["params"][p])
    record, = recovery_records(tape)
    assert record["path"] == "peer" and record["missed"] is expect_missed and record["steps"] == outcome.steps
    assert record["bound_steps"] == BOUND_H
    # Back on the roster: it trains, pushes, and acknowledges the final cut.
    assert back.pushes_after_recovery > 0
    assert back.final_manifest is not None and back.final_manifest == islands[0].final_manifest
    assert ("peer recovery missed its bound" in out) == expect_missed


@pytest.mark.timeout(240)
def test_island_pool_slot_filled_by_peer_recovery(tmp_path):
    """Island pool and peer recovery together: the pool admits a new
    generation for a lost island; its state comes from a peer. The restore
    takes longer than the lease, so only heartbeats keep the island in the
    pool; it has weight 0 (no pushes) until RECOVERY_DONE."""
    lease_s = 1.5
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    store = RecordingStore(tmp_path / "store")
    rng = np.random.default_rng(1)
    targets = [rng.standard_normal(FRAGMENTS * NUMEL).astype(np.float32) for _ in range(3)]
    proc = start_syncer(binary, port, 3, 2, tape, "--island-lease-s", str(lease_s), "--syncer-epoch", "5",
                        "--max-base-lag", "2", total_steps=120)
    pool = {"syncer_epoch": 5, "lease_s": lease_s}
    try:
        islands = [FakeIsland(i, port, targets[i], store, **pool) for i in range(3)]
        for island in islands:
            island.start()
        wait_until(lambda: min(islands[0].versions) >= 8, 60, "no early rounds")
        islands[2].kill()
        islands[2].join(timeout=10)
        kill_versions = list(islands[0].versions)
        wait_until(lambda: sum(islands[0].versions) > sum(kill_versions) + 8, 60, "rounds stopped without island 2")
        back = FakeIsland(2, port, targets[2], SlowStore(store.root, 2 * lease_s, store.puts),
                          recover_on_start=True, **pool)
        back.start()
        wait_until(lambda: back.outcome is not None or back.exc is not None, 60, "recovery did not finish")
        if back.exc:
            raise back.exc
        finish(proc, [islands[0], islands[1], back])
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-4000:])
        raise
    outcome = back.outcome
    assert outcome.path == "peer" and outcome.donor in (0, 1), outcome
    assert outcome.seconds > lease_s, "the restore outlasted the lease"
    snapshot = store.puts[islands[outcome.donor].donor.served[0][1]]
    for key in ("adam_m", "adam_v", "adam_step"):
        np.testing.assert_array_equal(back.imported[key], snapshot.state[key], err_msg=key)
    records = [json.loads(line) for line in tape.read_text().splitlines()]
    generation = back.client.connection_generation
    left = [r for r in records if r.get("kind") == "pool_leave" and r.get("island_id") == 2
            and r.get("generation") == generation]
    assert not left, "heartbeats kept the recovering island's lease"
    joined = [r for r in records if r.get("kind") == "pool_join" and r.get("recovered")]
    assert [(r["island_id"], r["generation"]) for r in joined] == [(2, generation)]
    assert back.heartbeat.errors == []
    assert back.pushes_after_recovery > 0
    assert back.final_manifest is not None and back.final_manifest == islands[0].final_manifest


@pytest.mark.timeout(240)
def test_recovery_still_in_transfer_at_the_end_of_training_takes_the_final_cut(tmp_path):
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    store = DirectorySnapshotStore(tmp_path / "store")
    target = np.ones(FRAGMENTS * NUMEL, np.float32)
    proc = start_syncer(binary, port, 2, 1, tape)
    try:
        islands = [FakeIsland(i, port, target, store) for i in range(2)]
        for island in islands:
            island.start()
        wait_until(lambda: min(islands[0].versions) >= 60, 60, "training did not get near its end")
        islands[1].kill()
        islands[1].join(timeout=10)
        back = FakeIsland(1, port, target, SlowStore(store.root, 6.0), recover_on_start=True)
        back.start()
        finish(proc, [islands[0], back])
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-4000:])
        raise
    assert back.outcome.path == "degraded", back.outcome
    assert back.final_manifest is not None and back.final_manifest == islands[0].final_manifest


@pytest.mark.timeout(240)
def test_killed_island_without_peer_takes_the_degraded_path(tmp_path):
    binary = build_syncer()
    port = free_port()
    tape = tmp_path / "tape.jsonl"
    store = DirectorySnapshotStore(tmp_path / "store")
    target = np.ones(FRAGMENTS * NUMEL, np.float32)
    # The only island is killed: no peer is left to donate a snapshot.
    proc = start_syncer(binary, port, 1, 1, tape)
    try:
        first = FakeIsland(0, port, target, store)
        first.start()
        wait_until(lambda: min(first.versions) >= 4, 60, "no early rounds")
        first.kill()
        first.join(timeout=10)
        back = FakeIsland(0, port, target, store, recover_on_start=True)
        back.start()
        wait_until(lambda: back.outcome is not None or back.exc is not None, 60, "recovery did not finish")
        if back.exc:
            raise back.exc
        assert back.outcome.path == "degraded" and back.outcome.donor is None
        assert back.outcome.replayed == FRAGMENTS, "degraded recovery resends every fragment"
        assert back.outcome.restored_versions is None
        assert back.imported is None
        assert back.after_recovery["adam_step"][0] == 0, "fresh optimizer state"
        # The island alone finishes the run.
        finish(proc, [back])
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        print((proc.stdout.read() if proc.stdout else "")[-4000:])
        raise
    record, = recovery_records(tape)
    assert record["path"] == "degraded" and record["donor"] is None


@pytest.mark.timeout(120)
def test_sharded_hello_syncer_epoch_fence(tmp_path):
    binary = build_syncer()
    port = free_port()
    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "1", "--quorum", "1", "--total-steps", "1000000",
         "--syncer-epoch", "5", "--checkpoint-path", str(tmp_path / "syncer.ckpt")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        layout = make_layout()
        stale = SyncerClient(("127.0.0.1", port), 0, layout, DTYPE_F32, num_streams=0, syncer_epoch=4)
        stale.start()
        wait_until(lambda: stale._err is not None, 30, "stale epoch was admitted")
        from yeto.rl.core import StrictRlInvariantError

        with pytest.raises(StrictRlInvariantError, match="syncer_epoch"):
            stale.check_health()
        stale.close()
        for epoch in (5, None):  # current epoch and a client that sends none
            ok = SyncerClient(("127.0.0.1", port), 0, layout, DTYPE_F32, num_streams=0, syncer_epoch=epoch)
            ok.start()
            for p in range(FRAGMENTS):
                ok.send_init(p, np.zeros(NUMEL, np.float32).tobytes())
            wait_until(lambda: len(ok._bcast_seen) == FRAGMENTS and all(ok._bcast_seen), 30, "no broadcast")
            ok.check_health()
            ok.close()
        record = Path(str(tmp_path / "syncer.ckpt") + ".membership")
        wait_until(record.exists, 10, "membership record not written")
        assert record.read_bytes()[:4] == b"YMB1"
        assert struct.unpack_from("<Q", record.read_bytes(), 4)[0] == 5
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_hello_without_extension_keeps_legacy_bytes():
    layout = make_layout()
    legacy = encode_hello(3, DTYPE_F32, layout, 2, 99)
    assert encode_hello(3, DTYPE_F32, layout, 2, 99, syncer_epoch=None, recover=False) == legacy
    extended = encode_hello(3, DTYPE_F32, layout, 2, 99, syncer_epoch=7, recover=True)
    assert extended[: len(legacy)] == legacy
    assert extended[len(legacy):] == b"YEX1" + bytes([3]) + struct.pack("<Q", 7)
    assert layout_fingerprint(layout) in legacy


# -- island-side logic without a syncer ----------------------------------------


class _ScriptedClient:
    """The SyncerClient surface recover()/DonorService use, scripted."""

    def __init__(self, events=(), requests=()):
        self.events = list(events)
        self.requests = list(requests)
        self.sent: list[tuple] = []
        self.updates: list[BcastFragment] = []

    def check_health(self):
        pass

    def heartbeat(self, local_step):
        pass

    def next_recovery_event(self, timeout):
        return self.events.pop(0) if self.events else None

    def drain_snapshot_requests(self):
        out, self.requests = self.requests, []
        return out

    def drain_updates(self):
        out, self.updates = self.updates, []
        return out

    def send_snapshot_ready(self, request, versions, sha256, locator):
        self.sent.append(("ready", request, tuple(versions), sha256, locator))
        return True

    def send_recovery_restored(self, versions):
        self.sent.append(("restored", versions))
        return True

    def send_recovery_failed(self, reason):
        self.sent.append(("failed", reason))
        return True


class _ArrayIsland:
    def __init__(self, versions, tag="t"):
        self.versions = list(versions)
        self.tag = tag
        self.state = {"w": np.arange(4, dtype=np.float32), "adam_m": np.full(4, 0.5, np.float32)}
        self.imported = None

    def fragment_versions(self):
        return tuple(self.versions)

    def layout_tag(self):
        return self.tag

    def export_recovery_state(self):
        return self.state

    def import_recovery_state(self, state, versions):
        self.imported = dict(state)
        self.versions = list(versions)

    def apply_broadcast(self, update):
        self.versions[update.fragment_id] = update.version

    def restore_degraded(self):
        return None


def test_donor_waits_for_the_clock_and_refused_snapshot_is_reported(tmp_path):
    store = DirectorySnapshotStore(tmp_path)
    request = rp.SnapshotRequest(2, 77, 9, (5, 6))
    donor_island = _ArrayIsland([5, 5])
    donor_client = _ScriptedClient(requests=[request])
    donor = DonorService(donor_client, donor_island, store)
    assert donor.poll() == 0, "fragment 1 is behind the clock"
    donor_island.versions = [5, 7]
    assert donor.poll() == 1
    (_, sent_request, versions, sha256, locator), = donor_client.sent
    assert sent_request == request and versions == (5, 7)

    good = rp.RecoverySource(rp.SOURCE_PEER, 0, (5, 7), sha256, locator)
    corrupt = rp.RecoverySource(rp.SOURCE_PEER, 1, (5, 7), b"\0" * 32, locator)
    other_layout = _ArrayIsland([0, 0], tag="other")
    client = _ScriptedClient(events=[rp.RecoveryClock(9, (5, 6), 1), corrupt, good,
                                     rp.RecoveryDone(9, 10, 1, (5, 8))])
    client.updates = [BcastFragment(1, 8, b"")]
    me = _ArrayIsland([0, 0])
    outcome = recover(client, me, store, timeout=5)
    assert [s[0] for s in client.sent] == ["failed", "restored"]
    assert "sha256 mismatch" in client.sent[0][1]
    assert client.sent[1] == ("restored", (5, 7))
    np.testing.assert_array_equal(me.imported["adam_m"], donor_island.state["adam_m"])
    assert me.versions == [5, 8], "replayed fragment applied before returning"
    assert (outcome.path, outcome.donor, outcome.steps, outcome.refused_sources) == ("peer", 0, 1, 1)

    locator, sha256 = store.put(Snapshot((5, 7), "t", donor_island.state))
    good = rp.RecoverySource(rp.SOURCE_PEER, 0, (5, 7), sha256, locator)
    client = _ScriptedClient(events=[good])
    with pytest.raises(TimeoutError):
        recover(client, other_layout, store, timeout=0.3)
    assert client.sent[0][0] == "failed" and "layout" in client.sent[0][1]


def test_snapshot_store_rejects_path_locators(tmp_path):
    from yeto.rl.peer_recovery import SnapshotIntegrityError

    store = DirectorySnapshotStore(tmp_path)
    for bad in ("../x.npz", "a/b.npz", ".hidden", "", "s123-abc"):
        with pytest.raises(SnapshotIntegrityError):
            store.get(bad, b"\0" * 32)
