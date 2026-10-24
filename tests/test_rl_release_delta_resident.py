"""S27 G-CJ: the resident delta engine (base in memory, chunk-parallel). CPU only, no Ray."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from yeto.rl.engine import release_delta as rd
from yeto.rl.engine.policy_release import (
    DeltaChain,
    PolicyFollower,
    PolicyReleaseStore,
    ReleaseNotLoaded,
)
from yeto.rl.engine.resume import CheckpointStore

CHUNK = 4096  # words per chunk in these tests (the real chunk is 16 Mi words)


@pytest.fixture(autouse=True)
def _small_weight_files(monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)


def _words(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, size=n, dtype=np.uint16)


def _changed(words: np.ndarray, fraction: float, seed: int = 1) -> np.ndarray:
    out = words.copy()
    index = np.random.default_rng(seed).choice(words.size, size=int(words.size * fraction), replace=False)
    out[index] ^= 1
    return out


def _release_dir(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def _versions(count: int, *, dense_at: int | None = None) -> list[dict[str, bytes]]:
    """``count`` versions of a release: an odd-size weight file, an even one, a
    sub-directory weight file, a small one (below the resident size) and a config."""
    a, b, c = _words(30_001, 1), _words(20_000, 2), _words(9_000, 3)
    out = []
    for version in range(count):
        if version:
            a = _changed(a, 0.01, seed=10 + version)
            b = _changed(b, 0.4 if version == dense_at else 0.01, seed=20 + version)
            c = _changed(c, 0.02, seed=30 + version)
        out.append({"a.safetensors": a.tobytes() + b"\x05", "b.safetensors": b.tobytes(),
                    "sub/c.safetensors": c.tobytes(), "tiny.safetensors": b"\x01\x02" * 100,
                    "config.json": f'{{"v": {version}}}'.encode()})
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_engine_selection_and_workers_from_the_environment():
    assert rd.delta_engine_kind({}) == rd.RESIDENT
    assert rd.delta_engine_kind({rd.ENGINE_ENV: "FILES"}) == rd.FILES
    assert rd.delta_engine({rd.ENGINE_ENV: "files"}) is None
    engine = rd.delta_engine({rd.WORKERS_ENV: "3"})
    assert isinstance(engine, rd.DeltaEngine) and engine.workers == 3
    assert 1 <= rd.delta_workers({}) <= rd.MAX_WORKERS
    for bad in ({rd.ENGINE_ENV: "gpu"}, {rd.WORKERS_ENV: "0"}, {rd.WORKERS_ENV: "x"}):
        with pytest.raises(ValueError):
            rd.delta_engine(bad)
    mirror = rd.default_mirror({rd.ENGINE_ENV: "files", rd.MIRROR_ENV: "/tmp/m", rd.WORKERS_ENV: "5"})
    assert mirror.engine is None and mirror.workers == 5 and mirror.root == Path("/tmp/m")
    assert isinstance(rd.default_mirror({}).engine, rd.DeltaEngine)


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_resident_encode_writes_the_file_engines_bytes(tmp_path, workers):
    versions = _versions(3, dense_at=2)
    dirs = [_release_dir(tmp_path / f"v{k}", files) for k, files in enumerate(versions)]
    engine = rd.DeltaEngine(workers=workers, chunk_words=CHUNK)
    full = engine.encode(None, None, dirs[0], "r1-v0", None)
    assert {n: e.sha256 for n, e in full.entries.items()} == {n: _sha(d) for n, d in versions[0].items()}
    assert full.delta_bytes == 0 and engine.resident.holds("r1-v0")
    assert engine.resident.file_bytes("a.safetensors") == versions[0]["a.safetensors"]
    assert engine.resident.size("tiny.safetensors") is None  # below the resident size: copied whole
    for k in (1, 2):
        got = engine.encode(dirs[k - 1], f"r{k}-v{k - 1}", dirs[k], f"r{k + 1}-v{k}", tmp_path / f"d{k}")
        want = rd.encode_directory(dirs[k - 1], dirs[k], tmp_path / f"ref{k}", workers=2)
        assert got.entries == want.entries
        for name, entry in got.entries.items():
            if entry.mode == rd.DELTA:  # same chunking -> the same delta file
                rd.encode_file(dirs[k - 1] / name, dirs[k] / name, tmp_path / "x.ydelta", chunk_words=CHUNK)
                expected = (tmp_path / "x.ydelta").read_bytes()
                assert (tmp_path / f"d{k}" / (name + rd.DELTA_SUFFIX)).read_bytes() == expected
            else:
                assert (tmp_path / f"d{k}" / name).read_bytes() == versions[k][name]
        assert got.delta_bytes == sum(p.stat().st_size for p in (tmp_path / f"d{k}").rglob("*") if p.is_file())
        assert engine.resident.holds(f"r{k + 1}-v{k}")
        assert engine.resident.file_bytes("sub/c.safetensors") == versions[k]["sub/c.safetensors"]


def test_resident_encode_needs_no_base_files_on_disk(tmp_path):
    versions = _versions(2)
    v0, v1 = (_release_dir(tmp_path / f"v{k}", f) for k, f in enumerate(versions))
    engine = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    engine.encode(None, None, v0, "r1-v0", None)
    shutil.rmtree(v0)
    got = engine.encode(tmp_path / "gone", "r1-v0", v1, "r2-v1", tmp_path / "d")
    assert got.entries["a.safetensors"].mode == rd.DELTA and got.ratio < 0.1
    out = tmp_path / "out"
    sampler = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    sampler.encode(None, None, _release_dir(tmp_path / "copy0", versions[0]), "r1-v0", None)
    sampler.rebuild(None, "r1-v0", [(tmp_path / "d", got.entries)], out, "r2-v1")
    assert {n: (out / n).read_bytes() for n in versions[1]} == versions[1]


def test_a_weight_file_whose_size_changed_goes_whole_and_stays_resident(tmp_path):
    old = {"a.safetensors": _words(5000, 1).tobytes()}
    new = {"a.safetensors": _words(6000, 2).tobytes() + b"\x09"}
    engine = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    engine.encode(None, None, _release_dir(tmp_path / "v0", old), "r1-v0", None)
    got = engine.encode(tmp_path / "v0", "r1-v0", _release_dir(tmp_path / "v1", new), "r2-v1", tmp_path / "d")
    assert got.entries["a.safetensors"].mode == rd.FULL_COPY
    assert (tmp_path / "d" / "a.safetensors").read_bytes() == new["a.safetensors"]
    assert engine.resident.file_bytes("a.safetensors") == new["a.safetensors"]


@pytest.mark.parametrize("workers", [1, 4])
def test_resident_rebuild_matches_the_file_engine_along_a_chain(tmp_path, workers):
    versions = _versions(4, dense_at=2)
    dirs = [_release_dir(tmp_path / f"v{k}", files) for k, files in enumerate(versions)]
    trainer = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    trainer.encode(None, None, dirs[0], "r1-v0", None)
    links = []
    for k in (1, 2, 3):
        result = trainer.encode(dirs[k - 1], f"r{k}-v{k - 1}", dirs[k], f"r{k + 1}-v{k}", tmp_path / f"d{k}")
        links.append((tmp_path / f"d{k}", result.entries))
    # a sampler that starts late: one pass over the whole chain from the full v0 on disk;
    # its chunk size is not the deltas' (the footers decide)
    late = rd.DeltaEngine(workers=workers, chunk_words=777)
    late.rebuild(dirs[0], "r1-v0", links, tmp_path / "late", "r4-v3")
    reference = tmp_path / "reference"
    rd.apply_directory(dirs[0], links, reference, workers=2)
    for name, data in versions[3].items():
        assert (tmp_path / "late" / name).read_bytes() == data == (reference / name).read_bytes()
    assert late.resident.holds("r4-v3") and late.resident.file_bytes("a.safetensors") == versions[3]["a.safetensors"]
    # a sampler that follows: one delta at a time, from its resident copy only
    follow = rd.DeltaEngine(workers=workers, chunk_words=CHUNK)
    follow.encode(None, None, dirs[0], "r1-v0", None)
    for k, link in enumerate(links, start=1):
        follow.rebuild(None, f"r{k}-v{k - 1}", [link], tmp_path / f"f{k}", f"r{k + 1}-v{k}")
        assert {n: (tmp_path / f"f{k}" / n).read_bytes() for n in versions[k]} == versions[k]


def test_a_failed_rebuild_names_the_file_and_drops_the_resident_copy(tmp_path):
    versions = _versions(2)
    v0, v1 = (_release_dir(tmp_path / f"v{k}", f) for k, f in enumerate(versions))
    trainer = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    trainer.encode(None, None, v0, "r1-v0", None)
    entries = trainer.encode(v0, "r1-v0", v1, "r2-v1", tmp_path / "d").entries
    sampler = rd.DeltaEngine(workers=3, chunk_words=CHUNK)
    sampler.encode(None, None, v0, "r1-v0", None)
    wrong = {**entries, "b.safetensors": rd.FileEntry(rd.DELTA, entries["b.safetensors"].size, "0" * 64)}
    with pytest.raises(rd.ReleaseDeltaError, match="rebuilt sha256") as error:
        sampler.rebuild(None, "r1-v0", [(tmp_path / "d", wrong)], tmp_path / "bad", "r2-v1")
    assert error.value.file == "b.safetensors" and sampler.resident.path is None
    # no resident copy and no base on disk: refused, not guessed
    with pytest.raises(rd.ReleaseDeltaError, match="no base copy"):
        sampler.rebuild(None, "r1-v0", [(tmp_path / "d", entries)], tmp_path / "bad2", "r2-v1")
    sampler.rebuild(v0, "r1-v0", [(tmp_path / "d", entries)], tmp_path / "good", "r2-v1")
    assert {n: (tmp_path / "good" / n).read_bytes() for n in versions[1]} == versions[1]


def test_a_delta_of_another_base_size_is_refused(tmp_path):
    base = _release_dir(tmp_path / "v0", {"a.safetensors": _words(5000).tobytes()})
    target = _release_dir(tmp_path / "v1", {"a.safetensors": _changed(_words(5000), 0.01).tobytes()})
    entries = rd.encode_directory(base, target, tmp_path / "d").entries
    other = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    other.encode(None, None, _release_dir(tmp_path / "w", {"a.safetensors": _words(4000).tobytes()}), "r1-v0", None)
    with pytest.raises(rd.ReleaseDeltaError, match="another size") as error:
        other.rebuild(None, "r1-v0", [(tmp_path / "d", entries)], tmp_path / "o", "r2-v1")
    assert error.value.file == "a.safetensors" and other.resident.path is None


def test_a_foreign_delta_file_is_refused_with_its_name(tmp_path):
    versions = _versions(2)
    v0, v1 = (_release_dir(tmp_path / f"v{k}", f) for k, f in enumerate(versions))
    entries = rd.encode_directory(v0, v1, tmp_path / "d").entries
    (tmp_path / "d" / ("a.safetensors" + rd.DELTA_SUFFIX)).write_bytes(b"not a delta file at all")
    engine = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    with pytest.raises(rd.ReleaseDeltaError, match="not a delta file") as error:
        engine.rebuild(v0, "r1-v0", [(tmp_path / "d", entries)], tmp_path / "o", "r2-v1")
    assert error.value.file == "a.safetensors"


def test_an_encode_error_leaves_no_resident_release(tmp_path, monkeypatch):
    versions = _versions(2)
    v0, v1 = (_release_dir(tmp_path / f"v{k}", f) for k, f in enumerate(versions))
    engine = rd.DeltaEngine(workers=2, chunk_words=CHUNK)
    engine.encode(None, None, v0, "r1-v0", None)

    def broken(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(rd, "_diff_chunk", broken)
    with pytest.raises(OSError, match="disk gone"):
        engine.encode(v0, "r1-v0", v1, "r2-v1", tmp_path / "d")
    assert engine.resident.path is None


# ---------------------------------------------------------------- store, mirror, follower
def _stage(root: Path, version: int, files: dict[str, bytes]) -> Path:
    return _release_dir(root / f"v{version}", files)


def _release(store: PolicyReleaseStore, chain: DeltaChain, staged: Path, version: int):
    release = store.release_staged(version, f"h{version}", staged, position=lambda: (version, None),
                                   delta_base=chain.base(), digest_full=True, engine=chain.engine)
    assert chain.advance(release, staged)
    return release


@pytest.mark.parametrize("trainer_engine,sampler_engine", [("resident", "resident"), ("resident", "files"),
                                                           ("files", "resident")])
def test_engines_interoperate_through_the_store(tmp_path, trainer_engine, sampler_engine):
    def engine(kind):
        return rd.DeltaEngine(workers=3, chunk_words=CHUNK) if kind == "resident" else None

    versions = _versions(6)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=2)
    chain = DeltaChain(full_every=4, engine=engine(trainer_engine))
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), mirror=rd.DeltaMirror(
        tmp_path / "mirror", workers=2, engine=engine(sampler_engine)))
    for version, files in enumerate(versions):
        release = _release(trainer, chain, _stage(tmp_path / "staging", version, files), version)
        assert (release.delta is None) == (version % 4 == 0)
        if release.delta is None:  # the chain's full releases carry each file's sha256
            assert release.file_sha256 == {n: _sha(d) for n, d in files.items()}
        directory = sampler.resolve(sampler.latest())
        assert {n: (directory / n).read_bytes() for n in files} == files
    if trainer_engine == "resident":
        assert chain.engine.resident.holds(release.path)


def test_the_resident_trainer_releases_deltas_without_its_base_on_disk(tmp_path):
    versions = _versions(3)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=4)
    chain = DeltaChain(full_every=8, engine=rd.DeltaEngine(workers=2, chunk_words=CHUNK))
    first = _release(trainer, chain, _stage(tmp_path / "s", 0, versions[0]), 0)
    assert chain.engine.resident.holds(first.path)  # the full release's digests filled it
    shutil.rmtree(chain.base().directory)
    second = _release(trainer, chain, _stage(tmp_path / "s", 1, versions[1]), 1)
    assert second.delta is not None and second.delta["base_path"] == first.path
    files_engine = PolicyReleaseStore(CheckpointStore(tmp_path / "store2"), keep=4)
    plain = DeltaChain(full_every=8)
    _release(files_engine, plain, _stage(tmp_path / "t", 0, versions[0]), 0)
    shutil.rmtree(plain.base().directory)
    assert _release(files_engine, plain, _stage(tmp_path / "t", 1, versions[1]), 1).delta is None


def test_the_resident_mirror_rebuilds_from_memory_and_retries_from_the_store(tmp_path, monkeypatch):
    versions = _versions(4)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=4)
    chain = DeltaChain(full_every=8, engine=rd.DeltaEngine(workers=2, chunk_words=CHUNK))
    mirror = rd.DeltaMirror(tmp_path / "mirror", keep=1, engine=rd.DeltaEngine(workers=2, chunk_words=CHUNK))
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), mirror=mirror)
    loaded = []
    follower = PolicyFollower(sampler, lambda release, directory: loaded.append(
        (release.version, {n: (directory / n).read_bytes() for n in versions[0]})))
    starts = []
    real = rd.DeltaEngine.rebuild
    monkeypatch.setattr(rd.DeltaEngine, "rebuild", lambda self, base, base_path, *a: (
        starts.append((None if base is None else base.parent.name, base_path, self.resident.holds(base_path))),
        real(self, base, base_path, *a))[1])
    _release(trainer, chain, _stage(tmp_path / "s", 0, versions[0]), 0)
    assert follower.refresh().version == 0  # a full release: loaded from the store
    for version in (1, 2):
        _release(trainer, chain, _stage(tmp_path / "s", version, versions[version]), version)
        assert follower.refresh().version == version
    # v1 from the full v0 in the store (not resident yet); v2 from the resident v1
    assert starts[0] == ("policy-releases", sampler.records()[0].path, False)
    assert starts[1][2] is True
    assert [v for v, _ in loaded] == [0, 1, 2] and all(files == versions[v] for v, files in loaded)
    # a corrupted resident copy fails the digest check; the retry rebuilds from the store
    mirror.engine.resident._words["a.safetensors"][0] ^= 1
    _release(trainer, chain, _stage(tmp_path / "s", 3, versions[3]), 3)
    with pytest.raises(ReleaseNotLoaded) as error:
        follower.refresh()
    assert error.value.reason == "delta_rebuild_failed" and follower.version == 2
    assert follower.refresh().version == 3 and loaded[-1][1] == versions[3]
    assert starts[-1] == ("policy-releases", None, False)  # from_store: the full v0 and the whole chain

