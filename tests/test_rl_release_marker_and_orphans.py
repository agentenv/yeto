"""Policy release directories: the exporter's ``.complete`` marker is never put, and
directories a stopped writer left without ``RELEASE.json`` are pruned (orphans)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from yeto.cloud import s3_store
from yeto.rl.engine.hf_export import COMPLETE_MARKER
from yeto.rl.engine.policy_release import (
    LATEST,
    LORA,
    RECORD,
    PolicyFollower,
    PolicyRelease,
    PolicyReleaseStore,
)
from yeto.rl.engine.resume import CheckpointStore

from test_rl_object_store import FakeS3

PREFIX = "policy-releases/"


def _store(tmp: Path, name: str, client: FakeS3) -> s3_store.S3Store:
    return s3_store.S3Store(tmp / name, "s3://bucket/run", environ={}, client=client, stage=tmp / name)


def _keys(client: FakeS3) -> list[str]:
    return list(s3_store.S3ObjectStore("s3://bucket/run", client=client).list(PREFIX))


def _dirs(client: FakeS3) -> set[str]:
    return {k[len(PREFIX):].split("/", 1)[0] for k in _keys(client) if k[len(PREFIX):].count("/") >= 1}


def _staged(tmp: Path, name: str, files: dict[str, bytes], *, marker: bool = True) -> Path:
    d = tmp / "export" / name
    d.mkdir(parents=True)
    for rel, data in files.items():
        (d / rel).write_bytes(data)
    if marker:  # what Miles' write_checkpoint_dir leaves once every rank wrote
        (d / COMPLETE_MARKER).touch()
    return d


def _pos():
    return None, None


def _orphan(client: FakeS3, name: str, files: dict[str, bytes]) -> None:
    """A release directory a writer uploaded and never recorded."""
    objects = s3_store.S3ObjectStore("s3://bucket/run", client=client)
    for rel, data in files.items():
        objects.write(f"{PREFIX}{name}/{rel}", data)


# -- the completion marker -------------------------------------------------------------
def test_a_staged_release_never_puts_the_exporters_completion_marker(tmp_path):
    client = FakeS3()
    releases = PolicyReleaseStore(_store(tmp_path, "t", client), keep=2, cache_dir=tmp_path / "c")
    staged = _staged(tmp_path, "v1", {"model.safetensors": b"w" * 64, "config.json": b"{}"})
    record = releases.release_staged(1, "h1", staged, position=_pos)
    assert COMPLETE_MARKER not in record.files
    assert sorted(record.files) == ["config.json", "model.safetensors"]
    assert not any(k.endswith(COMPLETE_MARKER) for k in _keys(client))
    assert not (staged / COMPLETE_MARKER).exists()  # the staged copy (a delta base) has none either
    # a reader loads it as before: LATEST and the record's files are the commit point
    reader = PolicyReleaseStore(_store(tmp_path, "s", client), cache_dir=tmp_path / "cs")
    loaded = reader.resolve(reader.latest())
    assert sorted(p.name for p in loaded.iterdir()) == ["RELEASE.json", "config.json", "model.safetensors"]


def test_the_marker_is_not_uploaded_before_the_weights(tmp_path, monkeypatch):
    """Before the fix the empty marker (a data file, first in name order) was the first
    object of the release in the store, minutes before the weights."""
    client = FakeS3()
    order: list[str] = []
    real = s3_store.S3ObjectStore.write_file

    def write_file(self, key, source, *args, **kwargs):
        order.append(key)
        return real(self, key, source, *args, **kwargs)

    monkeypatch.setattr(s3_store.S3ObjectStore, "write_file", write_file)
    releases = PolicyReleaseStore(_store(tmp_path, "t", client), keep=2, cache_dir=tmp_path / "c")
    releases.release_staged(1, "h1", _staged(tmp_path, "v1", {"a.safetensors": b"a" * 32,
                                                             "b.safetensors": b"b" * 32}), position=_pos)
    uploads = [k for k in order if k.startswith(PREFIX)]
    assert uploads and not any(k.endswith(COMPLETE_MARKER) for k in uploads)
    assert uploads.index(f"{PREFIX}r1-v1/{RECORD}") > max(uploads.index(f"{PREFIX}r1-v1/{n}")
                                                         for n in ("a.safetensors", "b.safetensors"))


def test_an_inline_release_drops_the_marker_and_its_digest_still_matches(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))

    def write(directory: Path) -> None:
        (directory / "adapter_model.safetensors").write_bytes(b"lora")
        (directory / "adapter_config.json").write_text("{}")
        (directory / COMPLETE_MARKER).touch()

    record = releases.release(1, "h1", write, weights=LORA)
    assert COMPLETE_MARKER not in record.files
    assert not (releases.root / record.path / COMPLETE_MARKER).exists()
    loaded = []
    follower = PolicyFollower(releases, lambda r, d: loaded.append(r.seq))
    assert follower.refresh().seq == 1 and loaded == [1]  # the sha256 check passes without it


# -- orphans: staged releases ----------------------------------------------------------
def test_an_orphan_below_latest_never_takes_a_kept_place(tmp_path):
    """r2-v7 (no record) sorts after r2-v2: before the fix it took one of the kept
    places, and the prune removed r2-v2, a release samplers may still serve."""
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=3, cache_dir=tmp_path / "ca")
    for v in (1, 2, 3):
        first.release_staged(v, f"h{v}", _staged(tmp_path, f"a{v}", {"w.bin": b"x" * v}), position=_pos)
    _orphan(client, "r2-v7", {"w.bin": b"orphan"})
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=3, cache_dir=tmp_path / "cb")
    second.release_staged(4, "h4", _staged(tmp_path, "b4", {"w.bin": b"y" * 4}), position=_pos)
    assert _dirs(client) == {"r2-v2", "r3-v3", "r4-v4"}
    assert second.last_orphans_removed == ["r2-v7"]


def test_the_orphan_of_a_stopped_upload_goes_with_the_next_release(tmp_path):
    """The measured case: weights of r21-v25 uploaded, the trainer stopped before the
    record; the next trainer's release at that seq removes it and says so."""
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    for v in (1, 2):
        first.release_staged(v, f"h{v}", _staged(tmp_path, f"a{v}", {"w.bin": b"x" * v}), position=_pos)
    _orphan(client, "r3-v9", {"w.bin.delta": b"d"})
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    record = second.release_staged(3, "h3", _staged(tmp_path, "b3", {"w.bin": b"3"}), position=_pos)
    assert record.path == "r3-v3"
    assert _dirs(client) == {"r2-v2", "r3-v3"}
    assert second.last_orphans_removed == ["r3-v9"]
    # the next release of the same process finds nothing more to remove
    second.release_staged(4, "h4", _staged(tmp_path, "b4", {"w.bin": b"4"}), position=_pos)
    assert second.last_orphans_removed == []


def test_a_directory_above_the_committed_seq_is_left_alone(tmp_path):
    """No record above LATEST's seq may be another writer's upload in flight."""
    client = FakeS3()
    releases = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    releases.release_staged(1, "h1", _staged(tmp_path, "a1", {"w.bin": b"1"}), position=_pos)
    _orphan(client, "r5-v9", {"w.bin": b"in flight"})
    restarted = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    restarted.release_staged(2, "h2", _staged(tmp_path, "b2", {"w.bin": b"2"}), position=_pos)
    restarted.release_staged(3, "h3", _staged(tmp_path, "b3", {"w.bin": b"3"}), position=_pos)
    assert _dirs(client) == {"r2-v2", "r3-v3", "r5-v9"}
    assert restarted.last_orphans_removed == []
    # once the store's seq passes it, it is an orphan of a stopped writer
    restarted.release_staged(4, "h4", _staged(tmp_path, "b4", {"w.bin": b"4"}), position=_pos)
    restarted.release_staged(5, "h5", _staged(tmp_path, "b5", {"w.bin": b"5"}), position=_pos)
    assert _dirs(client) == {"r4-v4", "r5-v5"}
    assert "r5-v9" in restarted.last_orphans_removed


def test_a_kept_delta_releases_base_is_never_an_orphan(tmp_path):
    """A base whose record is gone stays while a kept delta release names it."""
    store = CheckpointStore(tmp_path / "store")
    root = store.root / PREFIX
    (root / "r1-v1").mkdir(parents=True)
    (root / "r1-v1" / "w.bin").write_bytes(b"base")  # its record lost
    (root / "r1-v8").mkdir()
    (root / "r1-v8" / "w.bin").write_bytes(b"orphan")
    delta = PolicyRelease(seq=2, version=2, policy_hash="h2", path="r2-v2", files={"w.bin.delta": 1},
                          delta={"base_seq": 1, "base_path": "r1-v1", "base_version": 1, "ratio": 0.1,
                                 "files": {}})
    (root / "r2-v2").mkdir()
    (root / "r2-v2" / "w.bin.delta").write_bytes(b"d")
    (root / "r2-v2" / RECORD).write_text(json.dumps(asdict(delta)))
    (root / LATEST).write_text(json.dumps(asdict(delta)))
    releases = PolicyReleaseStore(store, keep=2, cache_dir=tmp_path / "c")
    releases.release_staged(3, "h3", _staged(tmp_path, "s3", {"w.bin": b"3"}), position=_pos)
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ["r1-v1", "r2-v2", "r3-v3"]
    assert releases.last_orphans_removed == ["r1-v8"]


def test_a_directory_this_writer_is_uploading_is_not_pruned_by_a_failed_attempt(tmp_path):
    """The upload of the next seq has no record until it commits: a release that
    failed before LATEST leaves it, and the retry at that seq replaces it."""
    client = FakeS3()
    releases = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    releases.release_staged(1, "h1", _staged(tmp_path, "a1", {"w.bin": b"1"}), position=_pos)
    put_files = releases.store.put_files

    def fail_once(files):
        releases.store.put_files = put_files
        raise OSError("record put failed")

    releases.store.put_files = fail_once
    try:
        releases.release_staged(2, "h2", _staged(tmp_path, "a2", {"w.bin": b"2"}), position=_pos)
    except OSError:
        pass
    assert "r2-v2" in _dirs(client)  # weights up, no record: not removed by anyone yet
    record = releases.release_staged(2, "h2", _staged(tmp_path, "a2b", {"w.bin": b"2"}), position=_pos)
    assert record.path == "r2-v2" and _dirs(client) == {"r1-v1", "r2-v2"}
    assert releases.last_orphans_removed == []


# -- orphans: inline releases (PolicyReleaseStore.release) -----------------------------
def _write(data: bytes):
    def write(directory: Path) -> None:
        (directory / "adapter_model.safetensors").write_bytes(data)
    return write


def test_an_inline_release_prunes_an_orphan_instead_of_a_kept_release(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=2)
    for v in (1, 2):
        releases.release(v, f"h{v}", _write(b"x" * v), weights=LORA)
    orphan = releases.root / "r2-v8"
    orphan.mkdir()
    (orphan / "adapter_model.safetensors").write_bytes(b"orphan")
    above = releases.root / "r9-v9"
    above.mkdir()
    (above / "adapter_model.safetensors").write_bytes(b"in flight")
    releases.release(3, "h3", _write(b"x" * 3), weights=LORA)
    assert sorted(p.name for p in releases.root.iterdir() if p.is_dir()) == ["r2-v2", "r3-v3", "r9-v9"]
    assert releases.last_orphans_removed == ["r2-v8"]


def test_an_inline_release_on_an_object_store_prunes_an_earlier_processs_releases(tmp_path):
    """The releases of a trainer on another node are not under this stage: listed from
    the stage alone they were never pruned."""
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    for v in (1, 2):
        first.release(v, f"h{v}", _write(b"x" * v), weights=LORA)
    _orphan(client, "r2-v8", {"adapter_model.safetensors": b"orphan"})
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    second.release(3, "h3", _write(b"x" * 3), weights=LORA)
    assert _dirs(client) == {"r2-v2", "r3-v3"}
    assert second.last_orphans_removed == ["r2-v8"]


def test_a_restarted_writer_keeps_the_base_its_kept_delta_names(tmp_path):
    """The base read from a stored record (a restarted writer has no ``_bases``): the
    record was looked up under the release directory twice, never found, and the
    base of a kept delta was pruned."""
    store = CheckpointStore(tmp_path / "store")
    root = store.root / PREFIX
    full = PolicyRelease(seq=1, version=1, policy_hash="h1", path="r1-v1", files={"w.bin": 4})
    delta = PolicyRelease(seq=2, version=2, policy_hash="h2", path="r2-v2", files={"w.bin.delta": 1},
                          delta={"base_seq": 1, "base_path": "r1-v1", "base_version": 1, "ratio": 0.1,
                                 "files": {}})
    for record, name in ((full, "w.bin"), (delta, "w.bin.delta")):
        (root / record.path).mkdir(parents=True)
        (root / record.path / name).write_bytes(b"data")
        (root / record.path / RECORD).write_text(json.dumps(asdict(record)))
    (root / LATEST).write_text(json.dumps(asdict(delta)))
    restarted = PolicyReleaseStore(store, keep=2, cache_dir=tmp_path / "c")
    restarted.release_staged(3, "h3", _staged(tmp_path, "s3", {"w.bin": b"3"}), position=_pos)
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ["r1-v1", "r2-v2", "r3-v3"]
