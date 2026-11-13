"""Trainer release GC (``keep``) racing a sampler that is downloading or rebuilding
the release it prunes. CPU only, object-store reads over a directory, no network.

Seam: ``PolicyReleaseStore.release_staged`` (trainer side, prunes to ``keep`` plus
the delta bases kept releases need) against ``PolicyFollower.refresh`` /
``PolicyReleaseStore.resolve`` / ``DeltaMirror.materialize`` (sampler side). The
sampler must fail closed (never load a half-downloaded set), move on to the newest
release, and fall back correctly when a delta's base is gone from the store."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from yeto.rl.engine import policy_release as pr
from yeto.rl.engine import release_delta as rd
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.policy_release import (
    DeltaChain,
    PolicyFollower,
    PolicyReleaseStore,
    ReleaseIncomplete,
    ReleaseNotLoaded,
)
from yeto.rl.engine.resume import CheckpointStore

FILES = ("model-00001.safetensors", "model-00002.safetensors")


def _words(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, size=n, dtype=np.uint16)


def _bump(words: np.ndarray, seed: int) -> np.ndarray:
    out = words.copy()
    index = np.random.default_rng(seed).choice(words.size, size=words.size // 100, replace=False)
    out[index] ^= 1
    return out


def _stage(root: Path, version: int, weights: dict[str, np.ndarray]) -> Path:
    directory = root / f"v{version}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, words in weights.items():
        (directory / name).write_bytes(words.astype("<u2").tobytes())
    (directory / "config.json").write_text("{}")
    return directory


class Trainer:
    """A trainer that publishes two weight files per version."""

    def __init__(self, tmp: Path, *, keep: int, full_every: int | None = None) -> None:
        self.store = PolicyReleaseStore(CheckpointStore(tmp / "store"), keep=keep)
        self.chain = None if full_every is None else DeltaChain(full_every=full_every)
        self.staging = tmp / "staging"
        self.weights = {name: _words(20_000, seed=i) for i, name in enumerate(FILES)}
        self.version = -1

    def release(self):
        self.version += 1
        if self.version:
            self.weights = {n: _bump(w, self.version * 10 + i) for i, (n, w) in enumerate(self.weights.items())}
        staged = _stage(self.staging, self.version, self.weights)
        base = None if self.chain is None else self.chain.base()
        release = self.store.release_staged(self.version, f"h{self.version}", staged,
                                            position=lambda: (self.version, None), delta_base=base)
        if self.chain is not None:
            self.chain.advance(release, staged)
        return release

    def bytes_of(self, version_weights: dict[str, np.ndarray]) -> dict[str, bytes]:
        return {n: w.astype("<u2").tobytes() for n, w in version_weights.items()}

    def released(self) -> list[str]:
        root = self.store.root
        return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("r"))


class HookedObjects(DirectoryObjectStore):
    """Reads over the store directory; ``on_read(key)`` runs before a file is read
    (the trainer publishing and pruning while the sampler downloads)."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.on_read = None
        self.read_keys: list[str] = []

    def read_file(self, key, target):
        self.read_keys.append(key)
        if self.on_read is not None:
            self.on_read(key)
        return super().read_file(key, target)


def _reader(tmp: Path, *, mirror: bool = False) -> tuple[PolicyReleaseStore, HookedObjects]:
    store = CheckpointStore(tmp / "store")
    objects = HookedObjects(tmp / "store")
    store.objects = objects
    reader = PolicyReleaseStore(store, cache_dir=tmp / "cache",
                                mirror=rd.DeltaMirror(tmp / "mirror", keep=2, workers=1) if mirror else None)
    return reader, objects


def _loader(loads: list):
    def load(release, directory):
        loads.append((release.version, {n: (directory / n).read_bytes() for n in FILES}))
    return load


def _prune_during(objects: HookedObjects, trainer: Trainer, path: str, *, releases: int) -> list:
    """After the first weight file of ``path`` was read, the trainer publishes
    ``releases`` more (each prunes): the second file of ``path`` is gone."""
    published = []

    def hook(key: str) -> None:
        if f"/{path}/" in key and key.endswith(FILES[1]) and not published:
            for _ in range(releases):
                published.append(trainer.release())
    objects.on_read = hook
    return published


def test_a_full_release_pruned_mid_download_is_not_loaded_and_the_newest_is(tmp_path):
    trainer = Trainer(tmp_path, keep=1)
    first = trainer.release()
    reader, objects = _reader(tmp_path)
    loads: list = []
    follower = PolicyFollower(reader, _loader(loads))
    assert follower.refresh().version == 0
    second = trainer.release()
    published = _prune_during(objects, trainer, second.path, releases=2)
    assert follower.refresh() is None  # fail closed: the half-downloaded r2 is not loaded
    assert len(published) == 2 and second.path not in trainer.released()
    assert follower.version == 0 and [v for v, _ in loads] == [0]
    assert follower.incomplete and second.path.split("-")[0] in follower.incomplete
    assert not (tmp_path / "cache" / second.path).exists()  # never renamed into place
    objects.on_read = None
    newest = follower.refresh()
    assert newest.seq == published[-1].seq and newest.version == trainer.version
    assert loads[-1] == (trainer.version, trainer.bytes_of(trainer.weights))
    assert follower.counters.superseded == 2  # r2 and r3 were never served
    # the cache holds the served release only (no leftover partial of r2)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [newest.path]
    assert first.path not in trainer.released()


def test_a_background_download_of_a_pruned_release_moves_on_to_the_newest(tmp_path, monkeypatch):
    monkeypatch.setenv(pr.DOWNLOAD_SPREAD_ENV, "0")
    trainer = Trainer(tmp_path, keep=1)
    trainer.release()
    reader, objects = _reader(tmp_path)
    loads: list = []
    follower = PolicyFollower(reader, _loader(loads), background_download=True)
    assert follower.refresh().version == 0
    second = trainer.release()
    published = _prune_during(objects, trainer, second.path, releases=2)
    assert follower.refresh() is None  # the fetch of r2 failed incomplete, nothing loaded
    assert [v for v, _ in loads] == [0] and follower.version == 0
    objects.on_read = None
    loaded = None
    for _ in range(3):  # starts the newest release's fetch, then loads it
        loaded = follower.refresh() or loaded
        if loaded is not None:
            break
    assert loaded is not None and loaded.seq == published[-1].seq
    assert [v for v, _ in loads] == [0, trainer.version]
    assert loads[-1][1] == trainer.bytes_of(trainer.weights)


def test_a_delta_release_pruned_mid_rebuild_falls_through_to_the_next_full_chain(tmp_path, monkeypatch):
    """full_every=2: r1 full, r2 delta(r1), r3 full, r4 delta(r3). keep=1 prunes
    r1 and r2 once r4 lands (r4 needs only r3)."""
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = Trainer(tmp_path, keep=1, full_every=2)
    trainer.release()  # r1 full
    reader, objects = _reader(tmp_path, mirror=True)
    loads: list = []
    follower = PolicyFollower(reader, _loader(loads))
    assert follower.refresh().version == 0
    second = trainer.release()
    assert second.delta is not None
    # the base r1 is cached (served); the hook fires on r2's delta of the second file
    objects.on_read = None
    published = []

    def hook(key: str) -> None:
        if f"/{second.path}/" in key and FILES[1] in key and not published:
            published.extend(trainer.release() for _ in range(2))
    objects.on_read = hook
    assert follower.refresh() is None  # r2's rebuild has no second delta: nothing loaded
    assert published[0].delta is None and published[1].delta["base_path"] == published[0].path
    assert trainer.released() == sorted([published[0].path, published[1].path])
    assert follower.version == 0 and [v for v, _ in loads] == [0]
    assert not any(p.name.endswith(".partial") for p in (tmp_path / "mirror").iterdir())
    assert not (tmp_path / "mirror" / second.path).exists()
    objects.on_read = None
    newest = follower.refresh()
    assert newest.seq == published[1].seq
    assert loads[-1] == (trainer.version, trainer.bytes_of(trainer.weights))


def test_a_base_gone_from_the_store_fails_closed_until_a_full_release(tmp_path, monkeypatch):
    """The trainer's prune keeps every base a kept delta needs; a base deleted
    behind its back (an external lifecycle rule) breaks the chain. The sampler then
    never loads a partial rebuild: the delta is incomplete, the engines keep the
    served version, and the next full release ends the outage. A base whose record
    is gone is not taken from the mirror either (the mirror's stamp is checked
    against the base's record), so the mirror copy of the served release does not
    stand in for it."""
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = Trainer(tmp_path, keep=8, full_every=16)
    trainer.release()
    trainer.release()  # r2 delta(r1)
    reader, _objects = _reader(tmp_path, mirror=True)
    loads: list = []
    follower = PolicyFollower(reader, _loader(loads))
    served = follower.refresh()
    assert served.version == 1 and (tmp_path / "mirror" / served.path).is_dir()
    third = trainer.release()  # r3 delta(r2)
    assert third.delta["base_path"] == served.path
    for name in trainer.released():
        if name != third.path:
            shutil.rmtree(trainer.store.root / name)
    assert follower.refresh() is None and "no record" in follower.incomplete
    with pytest.raises(ReleaseIncomplete, match="no record"):
        reader.resolve(reader.latest(), from_store=True)
    assert follower.version == 1 and [v for v, _ in loads] == [1]
    assert not (tmp_path / "mirror" / third.path).exists()
    assert not any(p.name.endswith(".partial") for p in (tmp_path / "mirror").iterdir())
    trainer.chain = DeltaChain(full_every=16)  # a fresh chain: the next release is full
    fourth = trainer.release()
    assert fourth.delta is None
    assert follower.refresh().seq == fourth.seq
    assert loads[-1] == (trainer.version, trainer.bytes_of(trainer.weights))


def test_a_digest_retry_from_the_store_survives_a_prune_of_the_mirror_base(tmp_path, monkeypatch):
    """The mirror copy a delta was rebuilt on goes bad; the retry rebuilds from the
    store's full release. The trainer keeps that chain (bases of kept releases),
    so the retry succeeds even after keep=1 pruned every non-base release."""
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = Trainer(tmp_path, keep=1, full_every=16)
    trainer.release()
    trainer.release()
    reader, _objects = _reader(tmp_path, mirror=True)
    loads: list = []
    follower = PolicyFollower(reader, _loader(loads))
    served = follower.refresh()
    copy = tmp_path / "mirror" / served.path / FILES[0]
    data = bytearray(copy.read_bytes())
    data[0] ^= 0xFF
    copy.write_bytes(bytes(data))
    trainer.release()
    newest = trainer.release()  # keep=1: newest plus its whole base chain stay
    assert len(trainer.released()) == 4
    with pytest.raises(ReleaseNotLoaded) as caught:
        follower.refresh()
    assert caught.value.reason == "delta_rebuild_failed" and follower.version == 1
    assert follower.refresh().seq == newest.seq
    assert loads[-1] == (trainer.version, trainer.bytes_of(trainer.weights))
