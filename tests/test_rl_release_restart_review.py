"""Adversarial cases of a restarted trainer's staged releases (review of #377)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yeto.cloud import s3_store
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore

from test_rl_object_store import FakeS3


def _store(tmp: Path, name: str, client: FakeS3) -> s3_store.S3Store:
    return s3_store.S3Store(tmp / name, "s3://bucket/run", environ={}, client=client, stage=tmp / name)


def _staged(tmp: Path, name: str, data: bytes) -> Path:
    d = tmp / "export" / name
    d.mkdir(parents=True)
    (d / "w.bin").write_bytes(data)
    return d


def _dirs(client: FakeS3) -> set[str]:
    keys = s3_store.S3ObjectStore("s3://bucket/run", client=client).list("policy-releases/")
    return {k.split("/")[1] for k in keys if k.count("/") >= 2}


def _pos():
    return None, None


@pytest.mark.parametrize("version", [3, 4])
def test_a_release_published_while_the_restarted_trainer_lists_is_not_removed(tmp_path, version):
    """Two trainers for a moment (the old one not yet gone): the old one publishes
    r3-v3 after the new one read LATEST but before it listed the store. r3-v3 is
    LATEST then; the new trainer must not take it for a partial upload."""
    client = FakeS3()
    old = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    for v in (1, 2):
        old.release_staged(v, f"h{v}", _staged(tmp_path, f"a{v}", b"x" * v), position=_pos)
    new = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    listed = new._listed_names

    def list_after_old_publishes():
        old.release_staged(3, "h3", _staged(tmp_path, "a3", b"x" * 3), position=_pos)
        return listed()

    new._listed_names = list_after_old_publishes
    removed = []
    remove = new.store.remove_tree
    new.store.remove_tree = lambda rel: (removed.append(rel), remove(rel))[1]
    new.release_staged(version, "hb", _staged(tmp_path, "b", b"b"), position=_pos)
    assert "policy-releases/r3-v3" not in removed
    latest = new.latest()
    assert latest.seq == 4  # the new release follows the old trainer's r3, not over it
    assert "r3-v3" in _dirs(client)


def test_a_failed_listing_fails_the_release_before_it_writes(tmp_path):
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    first.release_staged(1, "h1", _staged(tmp_path, "a1", b"1"), position=_pos)
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    objects = second.store.objects
    real = objects.list
    objects.list = lambda prefix, **kw: (_ for _ in ()).throw(RuntimeError("LIST refused"))
    with pytest.raises(RuntimeError):
        second.release_staged(2, "h2", _staged(tmp_path, "b2", b"2"), position=_pos)
    assert second.latest().seq == 1 and _dirs(client) == {"r1-v1"}
    objects.list = real
    assert second.release_staged(2, "h2", _staged(tmp_path, "b2b", b"2"), position=_pos).seq == 2


def test_a_failed_residue_removal_leaves_latest_and_the_retry_cleans_it(tmp_path):
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    first.release_staged(1, "h1", _staged(tmp_path, "a1", b"1"), position=_pos)
    first.store.put_files = lambda files: (_ for _ in ()).throw(RuntimeError("killed"))
    with pytest.raises(RuntimeError):
        first.release_staged(2, "h2", _staged(tmp_path, "a2", b"22"), position=_pos)
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    remove = second.store.remove_tree
    second.store.remove_tree = lambda rel: (_ for _ in ()).throw(RuntimeError("DELETE refused"))
    with pytest.raises(RuntimeError):
        second.release_staged(2, "h2b", _staged(tmp_path, "b2", b"2b"), position=_pos)
    assert second.latest().path == "r1-v1"
    second.store.remove_tree = remove
    assert second.release_staged(2, "h2b", _staged(tmp_path, "b2c", b"2b"), position=_pos).path == "r2-v2"
    reader = PolicyReleaseStore(_store(tmp_path, "s", client), cache_dir=tmp_path / "cs")
    assert (reader.resolve(reader.latest()) / "w.bin").read_bytes() == b"2b"


def test_a_mount_store_restart_prunes_and_replaces_residue(tmp_path):
    root = tmp_path / "mount"
    first = PolicyReleaseStore(CheckpointStore(root), keep=2)
    for v in (1, 2):
        first.release_staged(v, f"h{v}", _staged(tmp_path, f"a{v}", b"x" * v), position=_pos)
    (root / "policy-releases" / "r3-v3").mkdir()
    (root / "policy-releases" / "r3-v3" / "stale.bin").write_bytes(b"s")
    second = PolicyReleaseStore(CheckpointStore(root), keep=2)
    rec = second.release_staged(3, "h3", _staged(tmp_path, "b3", b"3"), position=_pos)
    assert sorted(p.name for p in (root / "policy-releases" / rec.path).iterdir()) == ["RELEASE.json", "w.bin"]
    assert {p.name for p in (root / "policy-releases").iterdir() if p.is_dir()} == {"r2-v2", "r3-v3"}
