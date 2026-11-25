"""Peer snapshot transport without a shared directory, and snapshot cleanup.

The object store is an in-memory fake with the ``ObjectStore`` surface the S3
and OSS backends have (no shared path between donor and recovering island);
the Modal Volume case is a directory with commit/reload hooks that hide
uncommitted writes from the other side.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from tests.test_peer_recovery import _ArrayIsland, _ScriptedClient
from yeto import recovery_protocol as rp
from yeto.protocol import BcastFragment
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.peer_recovery import (
    DirectorySnapshotStore,
    DonorService,
    Snapshot,
    SnapshotIntegrityError,
    SnapshotStore,
    locator_created_unix,
    recover,
)


class FakeObjectStore:
    """Whole objects in a dict; ``list`` in byte order like S3."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.uploads = 0

    def read(self, key):
        return self.data.get(key)

    def write(self, key, data):
        self.data[key] = bytes(data)

    def create(self, key, data):
        if key in self.data:
            return False
        self.data[key] = bytes(data)
        return True

    def exists(self, key):
        return key in self.data

    def write_file(self, key, source):
        self.uploads += 1
        self.data[key] = Path(source).read_bytes()

    def list(self, prefix, *, start_after=""):
        return sorted(k for k in self.data if k.startswith(prefix) and k > start_after)

    def delete(self, key):
        self.data.pop(key, None)


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class FileIsland(_ArrayIsland):
    """A sharded trainer: one file per rank, restore asks for every fragment."""

    def __init__(self, versions, ranks=2, tag="tp2"):
        super().__init__(versions, tag)
        self.ranks = ranks
        self.restored: dict[str, bytes] | None = None

    def write_recovery_snapshot(self, directory):
        for rank in range(self.ranks):
            (directory / f"rank{rank}").mkdir()
            (directory / f"rank{rank}" / "shard.pt").write_bytes(bytes([rank]) * (1000 + rank))

    def read_recovery_snapshot(self, directory, versions):
        self.restored = {p.relative_to(directory).as_posix(): p.read_bytes()
                         for p in sorted(directory.rglob("*")) if p.is_file()}
        return None


def _serve(store, island, request):
    client = _ScriptedClient(requests=[request])
    donor = DonorService(client, island, store)
    served = donor.poll()
    if donor.background:  # a file island: hash + commit on the worker, reported at the next poll
        assert donor.wait(5)
        served += donor.poll()
    assert served == 1
    (_, _, versions, sha256, locator), = client.sent
    return rp.RecoverySource(rp.SOURCE_PEER, 0, versions, sha256, locator)


def test_object_store_carries_a_snapshot_between_islands_without_a_shared_path(tmp_path):
    objects = FakeObjectStore()
    donor_store = SnapshotStore(objects, staging=tmp_path / "donor-stage")
    recovering_store = SnapshotStore(objects, staging=tmp_path / "recovering-stage")
    request = rp.SnapshotRequest(2, 77, 9, (5, 6))
    source = _serve(donor_store, _ArrayIsland([5, 6]), request)
    assert objects.uploads == 1 and not any((tmp_path / "donor-stage").iterdir()), "staging is cleaned"

    client = _ScriptedClient(events=[rp.RecoveryClock(9, (5, 6), 1), source, rp.RecoveryDone(9, 10, 0, (5, 6))])
    me = _ArrayIsland([0, 0])
    outcome = recover(client, me, recovering_store, timeout=5)
    assert outcome.path == "peer" and client.sent == [("restored", (5, 6))]
    np.testing.assert_array_equal(me.imported["adam_m"], np.full(4, 0.5, np.float32))
    assert objects.data == {}, "a restored snapshot is removed from the object store"


def test_sharded_snapshot_files_and_corruption_of_one_file(tmp_path):
    objects = FakeObjectStore()
    store = SnapshotStore(objects, staging=tmp_path / "stage")
    source = _serve(store, FileIsland([3, 3]), rp.SnapshotRequest(1, 4, 2, (3, 3)))
    manifest_key = store.manifest_key(source.text)
    assert hashlib.sha256(objects.data[manifest_key]).digest() == source.sha256
    shard_key = store.file_key(source.text, "rank1/shard.pt")
    good = objects.data[shard_key]
    objects.data[shard_key] = b"x" + good[1:]

    client = _ScriptedClient(events=[source])
    me = FileIsland([0, 0])
    with pytest.raises(TimeoutError):
        recover(client, me, store, timeout=0.3)
    assert client.sent[0][0] == "failed" and "rank1/shard.pt" in client.sent[0][1]
    assert me.restored is None, "nothing restored from a corrupt snapshot"

    objects.data[shard_key] = good
    client = _ScriptedClient(events=[source, rp.RecoveryDone(2, 3, 2, (4, 4))])
    client.updates = [BcastFragment(0, 4, b""), BcastFragment(1, 4, b"")]
    outcome = recover(client, me, store, timeout=5)
    assert client.sent == [("restored", None)], "a sharded island asks for every fragment"
    assert sorted(me.restored) == ["rank0/shard.pt", "rank1/shard.pt"]
    assert me.restored["rank1/shard.pt"] == bytes([1]) * 1001
    assert outcome.restored_versions is None and outcome.versions == (4, 4)


class VolumeStore(CheckpointStore):
    """A Modal-Volume-like store: another mount sees a write only after this
    side committed and the reader reloaded."""

    kind = "modal-volume"

    def __init__(self, root, remote):
        super().__init__(root)
        self.remote = Path(remote)
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1
        shutil.copytree(self.root, self.remote, dirs_exist_ok=True)
        for path in sorted(self.remote.rglob("*"), reverse=True):  # deletions
            if not (self.root / path.relative_to(self.remote)).exists():
                path.unlink() if path.is_file() else path.rmdir()
        return 0.0

    def reload(self):
        self.reloads += 1
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.copytree(self.remote, self.root, dirs_exist_ok=True)


def test_volume_store_commits_after_write_and_reloads_before_read(tmp_path):
    remote = tmp_path / "volume"
    remote.mkdir()
    donor_volume = VolumeStore(tmp_path / "mount-a", remote)
    recovering_volume = VolumeStore(tmp_path / "mount-b", remote)
    donor_store = SnapshotStore.for_checkpoint_store(donor_volume)
    recovering_store = SnapshotStore.for_checkpoint_store(recovering_volume)
    source = _serve(donor_store, _ArrayIsland([1, 1]), rp.SnapshotRequest(1, 4, 2, (1, 1)))
    assert donor_volume.commits == 1

    client = _ScriptedClient(events=[source, rp.RecoveryDone(2, 3, 0, (1, 1))])
    me = _ArrayIsland([0, 0])
    recover(client, me, recovering_store, timeout=5)
    assert recovering_volume.reloads >= 1 and client.sent == [("restored", (1, 1))]
    assert recovering_volume.commits == 1, "the removal is committed"
    donor_volume.reload()
    assert donor_store.locators() == []


def test_checkpoint_store_with_object_api_uses_it(tmp_path):
    objects = FakeObjectStore()
    store = CheckpointStore(tmp_path / "mount")
    store.objects = objects
    snapshots = SnapshotStore.for_checkpoint_store(store)
    locator, _ = snapshots.put(Snapshot((0,), "t", {"w": np.zeros(2, np.float32)}))
    assert any(k.startswith(f"peer-snapshots/{locator}/") for k in objects.data)
    assert not (tmp_path / "mount").exists()


@pytest.mark.parametrize("backend", ["directory", "objects"])
def test_sweep_removes_expired_and_partial_snapshots(tmp_path, backend):
    clock = Clock()
    if backend == "directory":
        store = DirectorySnapshotStore(tmp_path / "store", clock=clock)
    else:
        store = SnapshotStore(FakeObjectStore(), staging=tmp_path / "stage", clock=clock)
    old, _ = store.put(Snapshot((0,), "t", {"w": np.zeros(2, np.float32)}))
    abandoned = store.begin()  # a donor that died while writing its snapshot
    (abandoned.directory / "rank0.pt").write_bytes(b"partial")
    if backend == "objects":
        store.objects.write(store.file_key(abandoned.locator, "rank0.pt"), b"partial")
    clock.now += 1000
    fresh, _ = store.put(Snapshot((0,), "t", {"w": np.ones(2, np.float32)}))
    assert locator_created_unix(fresh) - locator_created_unix(old) == pytest.approx(1000)

    assert store.sweep(1200) == []
    clock.now += 500
    removed = store.sweep(1200)
    assert sorted(removed) == sorted([old, abandoned.locator])
    assert store.locators() == [fresh]
    assert store.sweep(100, keep=[fresh]) == []


def test_donor_sweeps_on_its_interval(tmp_path):
    clock = Clock()
    store = DirectorySnapshotStore(tmp_path, clock=clock)
    stale, _ = store.put(Snapshot((0,), "t", {"w": np.zeros(2, np.float32)}))
    clock.now += 7200
    donor = DonorService(_ScriptedClient(), _ArrayIsland([0]), store, ttl_s=3600, sweep_every_s=60)
    assert donor.poll() == 0
    assert donor.swept == [stale] and store.locators() == []


def test_unreported_snapshot_is_removed(tmp_path):
    class Offline(_ScriptedClient):
        def send_snapshot_ready(self, *args):
            return False

    store = DirectorySnapshotStore(tmp_path)
    donor = DonorService(Offline(requests=[rp.SnapshotRequest(1, 1, 1, (0,))]), _ArrayIsland([0]), store)
    assert donor.poll() == 0
    assert store.locators() == [], "a snapshot nobody was told about is not left behind"


def test_manifest_with_escaping_file_path_is_refused(tmp_path):
    objects = FakeObjectStore()
    store = SnapshotStore(objects, staging=tmp_path)
    locator = store.new_locator()
    data = (b'{"created_unix":1,"files":{"../x":{"bytes":1,"sha256":"00"}},'
            b'"layout_tag":"t","schema":"yeto.peer_snapshot/v1","versions":[0]}')
    objects.write(store.manifest_key(locator), data)
    with pytest.raises(SnapshotIntegrityError, match="invalid snapshot file path"):
        store.fetch(locator, hashlib.sha256(data).digest(), tmp_path / "into")


def test_refused_snapshot_drops_the_request_and_leaves_nothing(tmp_path):
    from yeto.rl.peer_recovery import SnapshotRefused

    class Refusing(FileIsland):
        def write_recovery_snapshot(self, directory):
            (directory / "rank0.pt").write_bytes(b"half")
            raise SnapshotRefused("rank 1 refused")

    store = DirectorySnapshotStore(tmp_path)
    client = _ScriptedClient(requests=[rp.SnapshotRequest(1, 1, 1, (0,))])
    donor = DonorService(client, Refusing([0]), store)
    assert donor.poll() == 0 and client.sent == []
    assert [reason for _, reason in donor.refused] == ["rank 1 refused"]
    assert store.locators() == [] and donor.poll() == 0, "the request is not retried"


class _ConcurrentS3Objects:
    """An ``S3ObjectStore`` over the in-memory fake S3 whose single-file transfers
    must overlap: every ``write_file`` / ``read_file`` waits at a barrier of
    ``parties`` callers, so moving the files one after the other times out."""

    def __init__(self, parties):
        from tests.test_rl_object_store import URI, FakeS3
        from yeto.cloud import s3_store

        self.objects = s3_store.S3ObjectStore(URI, client=FakeS3())
        self.barrier = threading.Barrier(parties, timeout=5)
        self.single_writes = 0
        self.single_reads = 0
        write_file, read_file = self.objects.write_file, self.objects.read_file

        def concurrent_write(key, source):
            self.single_writes += 1
            self.barrier.wait()
            return write_file(key, source)

        def concurrent_read(key, target):
            self.single_reads += 1
            self.barrier.wait()
            return read_file(key, target)

        self.objects.write_file = concurrent_write
        self.objects.read_file = concurrent_read


def test_rank_files_of_a_snapshot_move_in_parallel_through_the_store(tmp_path):
    ranks = 4
    bulk = _ConcurrentS3Objects(ranks)
    donor_store = SnapshotStore(bulk.objects, staging=tmp_path / "donor-stage")
    recovering_store = SnapshotStore(bulk.objects, staging=tmp_path / "recovering-stage")
    source = _serve(donor_store, FileIsland([3, 3], ranks=ranks), rp.SnapshotRequest(1, 4, 2, (3, 3)))
    assert bulk.single_writes == ranks, "every rank file went up, all at once"
    assert not bulk.barrier.broken

    client = _ScriptedClient(events=[source, rp.RecoveryDone(2, 3, 2, (4, 4))])
    client.updates = [BcastFragment(0, 4, b""), BcastFragment(1, 4, b"")]
    me = FileIsland([0, 0], ranks=ranks)
    recover(client, me, recovering_store, timeout=10)
    assert bulk.single_reads == ranks and not bulk.barrier.broken
    assert sorted(me.restored) == [f"rank{r}/shard.pt" for r in range(ranks)]
    assert all(me.restored[f"rank{r}/shard.pt"] == bytes([r]) * (1000 + r) for r in range(ranks))
    assert client.sent == [("restored", None)]


class _BulkObjects(FakeObjectStore):
    """The many-file surface of ``S3ObjectStore`` (``write_files`` / ``read_files``)."""

    def __init__(self, *, failed_uploads=(), missing_downloads=()):
        super().__init__()
        self.failed_uploads = set(failed_uploads)
        self.missing_downloads = set(missing_downloads)
        self.batches: list[tuple[str, int]] = []

    def write_files(self, items):
        self.batches.append(("write", len(items)))
        written, errors = [], {}
        for key, source in items:
            if any(key.endswith(rel) for rel in self.failed_uploads):
                errors[key] = "ConnectionError: reset"
                continue
            self.data[key] = Path(source).read_bytes()
            written.append(key)
        return written, errors

    def read_file(self, key, target):
        data = self.data.get(key)
        if data is None:
            return False
        Path(target).write_bytes(data)
        return True

    def read_files(self, items):
        self.batches.append(("read", len(items)))
        missing = []
        for key, target in items:
            if any(key.endswith(rel) for rel in self.missing_downloads) or not self.read_file(key, target):
                missing.append(key)
        return missing


def test_a_failed_rank_upload_refuses_the_snapshot_and_leaves_nothing(tmp_path):
    objects = _BulkObjects(failed_uploads={"rank1/shard.pt"})
    store = SnapshotStore(objects, staging=tmp_path / "stage")
    client = _ScriptedClient(requests=[rp.SnapshotRequest(1, 4, 2, (3, 3))])
    donor = DonorService(client, FileIsland([3, 3]), store)
    donor.poll()
    assert donor.wait(5)
    assert donor.poll() == 0 and client.sent == []
    (_, reason), = donor.refused
    assert "1 of 2 snapshot file(s) failed to upload" in reason and "rank1/shard.pt" in reason
    assert objects.batches == [("write", 2)] and objects.uploads == 0, "one many-file upload"
    assert store.locators() == [] and not any((tmp_path / "stage").iterdir())


def test_a_rank_file_missing_from_the_download_fails_the_restore(tmp_path):
    objects = _BulkObjects()
    store = SnapshotStore(objects, staging=tmp_path / "stage")
    source = _serve(store, FileIsland([3, 3]), rp.SnapshotRequest(1, 4, 2, (3, 3)))
    objects.missing_downloads = {"rank0/shard.pt"}
    with pytest.raises(SnapshotIntegrityError, match="misses 'rank0/shard.pt'"):
        store.fetch(source.text, source.sha256, tmp_path / "into")
    assert ("read", 2) in objects.batches


def test_a_one_file_snapshot_keeps_the_single_file_transfer(tmp_path):
    objects = _BulkObjects()
    store = SnapshotStore(objects, staging=tmp_path / "stage")
    locator, sha256 = store.put(Snapshot((0,), "t", {"w": np.zeros(2, np.float32)}))
    assert objects.uploads == 1 and objects.batches == []
    assert store.get(locator, sha256).state["w"].tolist() == [0.0, 0.0]
    assert objects.batches == []


class _FailingDownloadObjects(_BulkObjects):
    """``read_files`` whose download process died (``S3ObjectStore`` raises ``ObjectStoreError``)."""

    def read_files(self, items):
        from yeto.rl.engine.object_store import ObjectStoreError

        self.batches.append(("read", len(items)))
        raise ObjectStoreError("s3://bkt/runs/a: 1 of 2 download processes failed (exit [-9]); their stderr is above")


def test_a_failed_download_process_is_a_refused_snapshot_not_a_failed_island(tmp_path):
    objects = _FailingDownloadObjects()
    store = SnapshotStore(objects, staging=tmp_path / "stage")
    source = _serve(store, FileIsland([3, 3]), rp.SnapshotRequest(1, 4, 2, (3, 3)))
    with pytest.raises(SnapshotIntegrityError, match="download processes failed"):
        store.fetch(source.text, source.sha256, tmp_path / "into")

    client = _ScriptedClient(events=[source])
    me = FileIsland([0, 0])
    with pytest.raises(TimeoutError):
        recover(client, me, store, timeout=0.3)
    assert client.sent[0][0] == "failed" and "download processes failed" in client.sent[0][1], \
        "the syncer is told (next donor or degraded) instead of the island failing"
    assert me.restored is None
