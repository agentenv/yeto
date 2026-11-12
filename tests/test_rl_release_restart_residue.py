"""Staged releases across a trainer restart on an object store (S3 stage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yeto.cloud import s3_store
from yeto.rl.engine.policy_release import PolicyReleaseStore

from test_rl_object_store import FakeS3


def _store(tmp: Path, name: str, client: FakeS3) -> s3_store.S3Store:
    return s3_store.S3Store(tmp / name, "s3://bucket/run", environ={}, client=client, stage=tmp / name)


def _staged(tmp: Path, name: str, files: dict[str, bytes]) -> Path:
    d = tmp / "export" / name
    d.mkdir(parents=True)
    for rel, data in files.items():
        (d / rel).write_bytes(data)
    return d


def _keys(client: FakeS3, prefix: str) -> set[str]:
    return {k for k in _store_keys(client) if k.startswith(prefix)}


def _store_keys(client: FakeS3) -> list[str]:
    return list(s3_store.S3ObjectStore("s3://bucket/run", client=client).list("policy-releases/"))


def _pos():
    return None, None


def test_restarted_trainer_prunes_releases_of_the_earlier_process(tmp_path):
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    for v in (1, 2):
        first.release_staged(v, f"h{v}", _staged(tmp_path, f"a{v}", {"w.bin": b"x" * v}), position=_pos)
    # the trainer restarts on another node (a fresh stage) and releases on
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    for v in (3, 4, 5):
        second.release_staged(v, f"h{v}", _staged(tmp_path, f"b{v}", {"w.bin": b"y" * v}), position=_pos)
    dirs = {k.split("/")[1] for k in _store_keys(client) if k.count("/") >= 2}
    assert dirs == {"r4-v4", "r5-v5"}


def test_upload_into_a_partial_release_of_a_killed_trainer_drops_its_files(tmp_path):
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    first.release_staged(1, "h1", _staged(tmp_path, "a1", {"w.bin": b"1"}), position=_pos)

    def killed(_files):
        raise RuntimeError("trainer killed before LATEST")

    first.store.put_files = killed
    with pytest.raises(RuntimeError):
        first.release_staged(2, "h2", _staged(tmp_path, "a2", {"w.bin": b"22", "stale.safetensors": b"s"}),
                             position=_pos)
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    record = second.release_staged(2, "h2b", _staged(tmp_path, "b2", {"w.bin": b"2b"}), position=_pos)
    assert record.path == "r2-v2"
    assert _keys(client, "policy-releases/r2-v2/") == {"policy-releases/r2-v2/w.bin",
                                                       "policy-releases/r2-v2/RELEASE.json"}
    reader = PolicyReleaseStore(_store(tmp_path, "s", client), cache_dir=tmp_path / "cs")
    loaded = reader.resolve(reader.latest())
    assert sorted(p.name for p in loaded.iterdir()) == ["RELEASE.json", "w.bin"]


def test_a_partial_upload_at_the_next_seq_does_not_take_a_kept_place(tmp_path):
    client = FakeS3()
    first = PolicyReleaseStore(_store(tmp_path, "a", client), keep=2, cache_dir=tmp_path / "ca")
    for v in (1, 2):
        first.release_staged(v, f"h{v}", _staged(tmp_path, f"a{v}", {"w.bin": b"x" * v}), position=_pos)

    def killed(_files):
        raise RuntimeError("trainer killed before LATEST")

    first.store.put_files = killed
    with pytest.raises(RuntimeError):
        first.release_staged(9, "h9", _staged(tmp_path, "a9", {"w.bin": b"9"}), position=_pos)
    # restored from an older cut: the next release has the orphan's seq, another version
    second = PolicyReleaseStore(_store(tmp_path, "b", client), keep=2, cache_dir=tmp_path / "cb")
    record = second.release_staged(3, "h3", _staged(tmp_path, "b3", {"w.bin": b"3"}), position=_pos)
    assert record.path == "r3-v3"
    dirs = {k.split("/")[1] for k in _store_keys(client) if k.count("/") >= 2}
    assert dirs == {"r2-v2", "r3-v3"}  # the release samplers serve stays; the orphan goes
