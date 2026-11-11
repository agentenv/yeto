"""TreeMirror: a policy release commits its own paths and never waits for a round
cut of the same staged store (GPU-A it10: RELEASE.json and LATEST.json of a 6 GB
release waited 40 minutes behind a 178 GB cut push), concurrent pushes share
.yeto-pushed.json without losing each other's entries, and the files of one tier
move concurrently."""
from __future__ import annotations

import json
import threading
import time

from test_rl_object_store import URI, FakeS3

from yeto.cloud import s3_store
from yeto.rl.engine.object_store import PUSHED_STATE, DirectoryObjectStore, TreeMirror
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import STORE_IMPL_ENV, store_for

DEADLINE_S = 10.0


class _GatedObjects(DirectoryObjectStore):
    """Uploads of keys under ``gated`` block until ``gate`` is set (a slow cut
    upload); every upload is recorded with the thread that made it."""

    def __init__(self, root, gated: str = "round-cuts/") -> None:
        super().__init__(root)
        self.gated = gated
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.lock = threading.Lock()
        self.writes: list[str] = []

    def write_file(self, key, source):
        if key.startswith(self.gated):
            self.entered.set()
            assert self.gate.wait(30), "the test never opened the gate"
        with self.lock:
            self.writes.append(key)
        super().write_file(key, source)


def _mirror(root, objects, **kw):
    return TreeMirror(root, objects, subtrees=s3_store.STAGED_SUBTREES, pointers=s3_store.POINTER_FILES,
                      records=s3_store.RECORD_FILES, independent=s3_store.INDEPENDENT_DIRS, **kw)


def _cut(root, n_shards=2):
    cut = root / "round-cuts" / "r000001-x"
    cut.mkdir(parents=True)
    for i in range(n_shards):
        (cut / f"trainer_tp0_pp0_dp{i}.pt").write_bytes(b"s" * 100)
    (root / "LATEST").write_text("1")


def _release(root, name="r1-v1"):
    rel = root / "policy-releases" / name
    rel.mkdir(parents=True)
    (rel / "model-00001.safetensors").write_bytes(b"w" * 50)
    return rel


def test_release_record_and_latest_are_visible_while_a_cut_push_is_stuck(tmp_path):
    objects = _GatedObjects(tmp_path / "bucket")
    root = tmp_path / "stage"
    _cut(root)
    cut_push = threading.Thread(target=_mirror(root, objects).push, daemon=True)
    cut_push.start()  # the cut's commit: the whole tree minus the independent release dirs
    assert objects.entered.wait(DEADLINE_S)

    rel = _release(root)
    started = time.monotonic()
    release_mirror = _mirror(root, objects)  # another thread (the release uploader)
    release_mirror.push(["policy-releases/r1-v1"])
    (rel / "RELEASE.json").write_text("{}")
    (root / "policy-releases" / "LATEST.json").write_text("{}")
    release_mirror.push(["policy-releases/r1-v1/RELEASE.json", "policy-releases/LATEST.json"])
    assert time.monotonic() - started < DEADLINE_S
    assert objects.read("policy-releases/LATEST.json") == b"{}"
    assert objects.read("policy-releases/r1-v1/RELEASE.json") == b"{}"
    assert objects.read("LATEST") is None  # the cut's pointer still waits for its shards

    objects.gate.set()
    cut_push.join(DEADLINE_S)
    assert not cut_push.is_alive()
    assert objects.read("LATEST") == b"1"
    # the cut's push never uploaded a release file, the release push never a cut file
    assert objects.writes.count("policy-releases/LATEST.json") == 1
    assert objects.writes.count("policy-releases/r1-v1/model-00001.safetensors") == 1
    assert objects.writes.index("LATEST") > objects.writes.index("round-cuts/r000001-x/trainer_tp0_pp0_dp1.pt")


def test_release_through_a_staged_s3_store_does_not_wait_for_a_cut_commit(tmp_path):
    """The same through the store API the trainer uses: S3Store.commit (the cut) on
    one thread, PolicyReleaseStore.release_staged (put_directory, put_files) on another."""
    entered, gate = threading.Event(), threading.Event()

    class SlowCutS3(FakeS3):
        def put_object(self, Bucket, Key, Body, **kw):
            if "/round-cuts/" in Key:
                entered.set()
                assert gate.wait(30)
            return super().put_object(Bucket, Key, Body, **kw)

    s3 = SlowCutS3()
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI, s3_store.STAGE_ENV: str(tmp_path / "stage")}
    store = store_for(tmp_path / "mount", environ=env, client=s3)
    _cut(store.root)
    cut = threading.Thread(target=store.commit, daemon=True)
    cut.start()
    assert entered.wait(DEADLINE_S)

    staged = tmp_path / "export" / "v1"
    staged.mkdir(parents=True)
    (staged / "model-00001.safetensors").write_bytes(b"w" * 64)
    releases = PolicyReleaseStore(store)
    started = time.monotonic()
    record = releases.release_staged(1, "hash-1", staged, position=lambda: (8, None))
    assert time.monotonic() - started < DEADLINE_S
    latest = json.loads(s3.objects[("bkt", "runs/a/policy-releases/LATEST.json")])
    assert latest["path"] == record.path == "r1-v1" and latest["files"] == {"model-00001.safetensors": 64}
    assert ("bkt", "runs/a/policy-releases/r1-v1/RELEASE.json") in s3.objects
    assert ("bkt", "runs/a/LATEST") not in s3.objects
    gate.set()
    cut.join(DEADLINE_S)
    assert ("bkt", "runs/a/LATEST") in s3.objects


def test_concurrent_pushes_keep_both_entries_of_the_pushed_state(tmp_path):
    objects = _GatedObjects(tmp_path / "bucket")
    root = tmp_path / "stage"
    _cut(root, n_shards=4)
    _release(root)
    first = threading.Thread(target=_mirror(root, objects).push, daemon=True)
    first.start()
    assert objects.entered.wait(DEADLINE_S)
    _mirror(root, objects).push(["policy-releases"])  # writes the state while the cut push is mid-way
    objects.gate.set()
    first.join(DEADLINE_S)
    state = json.loads((root / PUSHED_STATE).read_text())
    assert "policy-releases/r1-v1/model-00001.safetensors" in state
    assert {f"round-cuts/r000001-x/trainer_tp0_pp0_dp{i}.pt" for i in range(4)} <= set(state)
    assert "LATEST" in state
    objects.writes.clear()
    _mirror(root, objects).push()
    _mirror(root, objects).push(["policy-releases"])
    assert objects.writes == []  # nothing lost from the state: nothing is uploaded again


def test_files_of_one_tier_move_concurrently_and_tiers_stay_ordered(tmp_path):
    active, peak, lock = [0], [0], threading.Lock()

    class Timed(DirectoryObjectStore):
        order: list[str] = []

        def write_file(self, key, source):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
                self.order.append(key)
            super().write_file(key, source)

    objects = Timed(tmp_path / "bucket")
    root = tmp_path / "stage"
    _cut(root, n_shards=8)
    _mirror(root, objects, file_concurrency=4).push()
    assert peak[0] == 4
    assert objects.order[-1] == "LATEST" and len(objects.order) == 9


def test_scoped_push_deletes_only_inside_its_paths_and_whole_push_skips_releases(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "bucket")
    root = tmp_path / "stage"
    _cut(root)
    _release(root, "r1-v1")
    _release(root, "r2-v2")
    mirror = _mirror(root, objects)
    mirror.push()
    assert objects.read("policy-releases/r1-v1/model-00001.safetensors") is None  # independent: not pushed
    mirror.push(["policy-releases"])
    assert objects.read("policy-releases/r1-v1/model-00001.safetensors") is not None
    import shutil

    shutil.rmtree(root / "policy-releases" / "r1-v1")
    shutil.rmtree(root / "round-cuts" / "r000001-x")
    mirror.push(["policy-releases/r1-v1"])  # the release prune
    assert objects.read("policy-releases/r1-v1/model-00001.safetensors") is None
    assert objects.read("round-cuts/r000001-x/trainer_tp0_pp0_dp0.pt") is not None  # not in that scope
    mirror.push()  # the cut prune: never touches the release that is kept
    assert objects.read("round-cuts/r000001-x/trainer_tp0_pp0_dp0.pt") is None
    assert objects.read("policy-releases/r2-v2/model-00001.safetensors") is not None


def test_record_pushed_marks_a_file_its_writer_uploaded(tmp_path):
    objects = _GatedObjects(tmp_path / "bucket", gated="nothing/")
    root = tmp_path / "stage"
    _cut(root, n_shards=1)
    shard = "round-cuts/r000001-x/trainer_tp0_pp0_dp0.pt"
    objects.write_file(shard, root / shard)  # the rank uploaded it itself
    mirror = _mirror(root, objects)
    mirror.record_pushed(shard)
    objects.writes.clear()
    mirror.push()
    assert objects.writes == ["LATEST"]
