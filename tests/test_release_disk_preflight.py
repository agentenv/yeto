"""A release download (sampler hot load, trainer --rl-init-from-release) and a delta
rebuild check the local disk's free space against the release record's bytes first:
too little space is a clear error before any byte is written, not an ENOSPC part way
through a 200 GB download that leaves a half-written directory behind."""
from __future__ import annotations

import numpy as np
import pytest

from yeto.rl.engine import disk_space
from yeto.rl.engine import release_delta as rd
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.policy_release import DeltaChain, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore


def _object_store(root):
    store = CheckpointStore(root)
    store.objects = DirectoryObjectStore(root)
    return store


def _write(directory, payload: bytes):
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(payload)


def test_download_refuses_before_writing_when_the_disk_is_too_small(tmp_path, monkeypatch):
    writer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    writer.release(1, "h1", lambda d: _write(d, b"x" * 10_000))
    reader = PolicyReleaseStore(_object_store(tmp_path / "store"), cache_dir=tmp_path / "cache")
    monkeypatch.setattr(disk_space, "free_bytes", lambda path: 5_000)
    with pytest.raises(disk_space.InsufficientDiskSpace, match="free"):
        reader.resolve(reader.latest())
    cache = tmp_path / "cache"
    assert not cache.exists() or not any(cache.rglob("*.safetensors*"))  # nothing half-written


def test_download_proceeds_with_enough_space(tmp_path, monkeypatch):
    writer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    writer.release(1, "h1", lambda d: _write(d, b"x" * 10_000))
    reader = PolicyReleaseStore(_object_store(tmp_path / "store"), cache_dir=tmp_path / "cache")
    monkeypatch.setattr(disk_space, "free_bytes", lambda path: 1 << 40)
    directory = reader.resolve(reader.latest())
    assert (directory / "model.safetensors").stat().st_size == 10_000


def test_a_resumed_download_only_needs_space_for_the_missing_files(tmp_path, monkeypatch):
    writer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    release = writer.release(1, "h1", lambda d: _write(d, b"x" * 10_000))
    reader = PolicyReleaseStore(_object_store(tmp_path / "store"), cache_dir=tmp_path / "cache")
    staging = tmp_path / "cache" / f".{release.path}.partial"
    staging.mkdir(parents=True)
    (staging / "model.safetensors").write_bytes(b"x" * 10_000)  # fetched whole before a failure
    monkeypatch.setattr(disk_space, "free_bytes", lambda path: 100)
    assert (reader.resolve(reader.latest()) / "model.safetensors").stat().st_size == 10_000


def test_delta_rebuild_refuses_before_writing_when_the_disk_is_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=2)
    chain = DeltaChain(full_every=4)
    rng = np.random.default_rng(0)
    words = rng.integers(0, 1 << 16, 50_000, dtype=np.uint16)
    for version in range(2):
        staged = tmp_path / "staging" / f"v{version}"
        staged.mkdir(parents=True)
        (staged / "config.json").write_text("{}")
        if version:
            words = words.copy()
            words[:100] ^= 1
        (staged / "model.safetensors").write_bytes(words.astype("<u2").tobytes())
        release = trainer.release_staged(version, f"h{version}", staged, position=lambda: (version, None),
                                         delta_base=chain.base())
        assert chain.advance(release, staged)
    mirror = rd.DeltaMirror(tmp_path / "mirror", workers=1)
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), mirror=mirror)
    latest = sampler.latest()
    assert latest.delta is not None
    monkeypatch.setattr(disk_space, "free_bytes", lambda path: 1_000)
    with pytest.raises(disk_space.InsufficientDiskSpace, match="rebuild"):
        sampler.resolve(latest)
    assert not any((tmp_path / "mirror").glob(".*.partial")) if (tmp_path / "mirror").exists() else True
    monkeypatch.setattr(disk_space, "free_bytes", lambda path: 1 << 40)
    rebuilt = sampler.resolve(latest)
    assert (rebuilt / "model.safetensors").read_bytes() == words.astype("<u2").tobytes()


def test_free_bytes_of_a_path_not_created_yet_uses_its_nearest_parent(tmp_path):
    assert disk_space.free_bytes(tmp_path / "a" / "b" / "c") > 0


@pytest.mark.parametrize("raw", ["inf", "nan", "-1", "0.5", "abc", ""])
def test_invalid_headroom_falls_back_to_the_default(raw):
    assert disk_space.headroom({disk_space.HEADROOM_ENV: raw}) == disk_space.HEADROOM


def test_infinite_headroom_does_not_crash_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv(disk_space.HEADROOM_ENV, "inf")
    monkeypatch.setattr(disk_space, "free_bytes", lambda path: 10**12)
    disk_space.check_free(tmp_path, 1000, "x")
