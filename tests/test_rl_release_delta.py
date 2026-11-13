"""Bf16 delta releases (trainer encodes, sampler rebuilds). CPU only, no Ray."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from yeto.rl.engine import release_delta as rd
from yeto.rl.engine.policy_release import (
    DeltaChain,
    PolicyReleaseStore,
    ReleaseIncomplete,
)
from yeto.rl.engine.resume import CheckpointStore


def _words(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, size=n, dtype=np.uint16)


def _changed(words: np.ndarray, fraction: float, seed: int = 1) -> np.ndarray:
    out = words.copy()
    rng = np.random.default_rng(seed)
    index = rng.choice(words.size, size=int(words.size * fraction), replace=False)
    out[index] ^= 1  # one bf16 step
    return out


def _write(path: Path, words: np.ndarray, tail: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(words.astype("<u2").tobytes() + tail)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("fraction", [0.0, 0.001, 0.05, 0.5])
@pytest.mark.parametrize("tail", [b"", b"\x07"])
def test_file_round_trip_is_byte_exact(tmp_path, fraction, tail):
    base_words = _words(70_001)
    base = _write(tmp_path / "base.safetensors", base_words, tail)
    target = _write(tmp_path / "new.safetensors", _changed(base_words, fraction), tail)
    changed, digest = rd.encode_file(base, target, tmp_path / "d.ydelta", chunk_words=4096)
    assert changed == int(70_001 * fraction) and digest == _sha(target)
    rebuilt = tmp_path / "rebuilt"
    assert rd.apply_file(base, tmp_path / "d.ydelta", rebuilt) == digest
    assert rebuilt.read_bytes() == target.read_bytes()


def test_sparse_chunks_use_indexes_and_dense_ones_a_bitmap(tmp_path):
    base_words = _words(1 << 16)
    target_words = base_words.copy()
    target_words[:3] ^= 1  # chunk 0: 3 changes -> indexes
    target_words[1 << 14:(1 << 14) + 4000] ^= 1  # chunk 1: dense -> bitmap
    base = _write(tmp_path / "a", base_words)
    target = _write(tmp_path / "b", target_words)
    rd.encode_file(base, target, tmp_path / "d", chunk_words=1 << 14)
    with (tmp_path / "d").open("rb") as handle:
        footer = rd._footer(handle, tmp_path / "d")
    assert footer["chunks"] == [["idx", 3], ["bits", 4000], ["same", 0], ["same", 0]]
    payload = 3 * 6 + ((1 << 14) // 8 + 2 * 4000)
    assert (tmp_path / "d").stat().st_size == payload + len(json.dumps(
        footer, separators=(",", ":"))) + 8 + len(rd.MAGIC)


def test_apply_refuses_another_base_and_a_foreign_file(tmp_path):
    base = _write(tmp_path / "a", _words(1000))
    target = _write(tmp_path / "b", _changed(_words(1000), 0.1))
    rd.encode_file(base, target, tmp_path / "d")
    with pytest.raises(rd.ReleaseDeltaError, match="another size"):
        rd.apply_file(_write(tmp_path / "short", _words(999)), tmp_path / "d", tmp_path / "x")
    with pytest.raises(rd.ReleaseDeltaError, match="not a delta file"):
        rd.apply_file(base, target, tmp_path / "x")
    with pytest.raises(rd.ReleaseDeltaError, match="sizes differ"):
        rd.encode_file(tmp_path / "short", target, tmp_path / "d2")


def _release_dir(root: Path, words: dict[str, np.ndarray]) -> Path:
    for name, value in words.items():
        _write(root / name, value)
    (root / "config.json").write_text('{"model_type": "x"}')
    return root


def test_directory_delta_rebuilds_every_file_and_copies_the_small_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    old = {"model-1.safetensors": _words(50_000, 1), "model-2.safetensors": _words(60_000, 2)}
    new = {"model-1.safetensors": _changed(old["model-1.safetensors"], 0.01),
           "model-2.safetensors": _words(70_000, 3)}  # another size: copied whole
    base, target = _release_dir(tmp_path / "v1", old), _release_dir(tmp_path / "v2", new)
    delta = rd.encode_directory(base, target, tmp_path / "d", workers=2)
    assert {n: e.mode for n, e in delta.entries.items()} == {
        "config.json": "full", "model-1.safetensors": "delta", "model-2.safetensors": "full"}
    assert sorted(p.name for p in (tmp_path / "d").iterdir()) == [
        "config.json", "model-1.safetensors.ydelta", "model-2.safetensors"]
    rd.apply_directory(base, [(tmp_path / "d", delta.entries)], tmp_path / "out", workers=2)
    for name in delta.entries:
        assert (tmp_path / "out" / name).read_bytes() == (target / name).read_bytes()
    corrupt = dict(delta.entries)
    corrupt["model-1.safetensors"] = rd.FileEntry("delta", corrupt["model-1.safetensors"].size, "0" * 64)
    with pytest.raises(rd.ReleaseDeltaError, match="rebuilt sha256"):
        rd.apply_directory(base, [(tmp_path / "d", corrupt)], tmp_path / "out2")


# ---------------------------------------------------------------- store, chain, mirror
def _stage(root: Path, version: int, words: np.ndarray) -> Path:
    directory = root / f"v{version}"
    _write(directory / "model.safetensors", words)
    (directory / "config.json").write_text("{}")
    return directory


def _release(store: PolicyReleaseStore, chain: DeltaChain, staged: Path, version: int):
    release = store.release_staged(version, f"h{version}", staged, position=lambda: (version, None),
                                   delta_base=chain.base())
    assert chain.advance(release, staged)
    return release


def test_trainer_chain_uploads_deltas_and_the_sampler_rebuilds_them(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    store = CheckpointStore(tmp_path / "store")
    trainer = PolicyReleaseStore(store, keep=1)
    chain = DeltaChain(full_every=3)
    words = [_words(100_000)]
    for _ in range(4):
        words.append(_changed(words[-1], 0.01, seed=len(words)))
    staging = tmp_path / "staging"
    releases = [_release(trainer, chain, _stage(staging, v, w), v) for v, w in enumerate(words)]
    # full, delta, delta, full (every 3rd), delta
    assert [r.delta is None for r in releases] == [True, False, False, True, False]
    assert releases[2].delta["base_path"] == releases[1].path and releases[2].delta["ratio"] < 0.1
    # the store holds the delta file (pruned since; its size is recorded)
    assert sorted(releases[2].files) == ["config.json", "model.safetensors.ydelta"]
    assert releases[2].files["model.safetensors.ydelta"] < 0.1 * 2 * words[0].size
    # keep=1: the newest release (a delta) and its base (the full v3) stay in the store
    names = sorted(p.name for p in (store.root / "policy-releases").iterdir() if p.is_dir())
    assert names == sorted([releases[3].path, releases[4].path])
    # only the newest base stays on the trainer's disk
    assert sorted(p.name for p in staging.iterdir()) == ["v4"]

    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror", workers=2))
    latest = sampler.latest()
    rebuilt = sampler.resolve(latest)
    assert rebuilt.parent == tmp_path / "mirror"
    assert (rebuilt / "model.safetensors").read_bytes() == words[4].astype("<u2").tobytes()


def test_mirror_applies_the_next_delta_to_its_own_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=4)
    chain = DeltaChain(full_every=8)
    words = [_words(50_000)]
    staging = tmp_path / "staging"
    _release(trainer, chain, _stage(staging, 0, words[0]), 0)
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror", keep=2, workers=1))
    applied = []
    real_apply = rd.apply_directory
    monkeypatch.setattr(rd, "apply_directory",
                        lambda base, *a, **k: (applied.append(Path(base)), real_apply(base, *a, **k)))
    for version in (1, 2, 3):
        words.append(_changed(words[-1], 0.02, seed=version))
        _release(trainer, chain, _stage(staging, version, words[-1]), version)
        rebuilt = sampler.resolve(sampler.latest())
        assert (rebuilt / "model.safetensors").read_bytes() == words[-1].astype("<u2").tobytes()
    # v1 from the full v0 in the store; v2 and v3 from the mirror's own copies
    assert applied[0].parent.name == "policy-releases"
    assert [p.parent for p in applied[1:]] == [tmp_path / "mirror"] * 2
    assert len([p for p in (tmp_path / "mirror").iterdir() if p.is_dir()]) == 2  # keep=2


def test_a_tracking_mirror_records_the_changes_against_the_copy_it_rebuilt_from(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=4)
    chain = DeltaChain(full_every=8)
    words = [_words(50_000)]
    staging = tmp_path / "staging"
    _release(trainer, chain, _stage(staging, 0, words[0]), 0)
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror", keep=2, workers=1, track_changes=True))
    previous = sampler.resolve(sampler.latest())  # the full v0 in the store: no manifest
    assert rd.read_changes(previous) is None
    for version in (1, 2):
        words.append(_changed(words[-1], 0.02, seed=version))
        _release(trainer, chain, _stage(staging, version, words[-1]), version)
        rebuilt = sampler.resolve(sampler.latest())
        manifest = rd.read_changes(rebuilt)
        assert Path(manifest["base_dir"]) == previous
        assert (manifest["base_stamp"] is None) == (version == 1)  # v2 starts from the mirror's v1
        changed = rd.read_changed_words(manifest["files"]["model.safetensors"])
        assert changed.tolist() == np.flatnonzero(words[-1] != words[-2]).tolist()
        previous = rebuilt


def test_a_delta_whose_base_is_gone_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=4)
    chain = DeltaChain(full_every=8)
    words = _words(20_000)
    _release(trainer, chain, _stage(tmp_path / "s", 0, words), 0)
    delta = _release(trainer, chain, _stage(tmp_path / "s", 1, _changed(words, 0.01)), 1)
    import shutil

    shutil.rmtree(tmp_path / "store" / "policy-releases" / delta.delta["base_path"])
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror"))
    with pytest.raises(ReleaseIncomplete, match="no record"):
        sampler.resolve(sampler.latest())
    with pytest.raises(Exception, match="no mirror"):
        PolicyReleaseStore(CheckpointStore(tmp_path / "store")).resolve(sampler.latest())


def test_a_large_delta_is_released_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    chain = DeltaChain(full_every=8)
    _release(trainer, chain, _stage(tmp_path / "s", 0, _words(20_000, 0)), 0)
    second = _release(trainer, chain, _stage(tmp_path / "s", 1, _words(20_000, 9)), 1)
    assert second.delta is None and chain.base().release == second


def test_chain_needs_full_every_two_or_more():
    with pytest.raises(ValueError):
        DeltaChain(1)


# ---------------------------------------------------------------- trainer publisher
class _State:
    def __init__(self, version: int) -> None:
        self.policy_version = version

    def policy_tensor_hash(self) -> str:
        return f"h{self.policy_version}"


def test_staged_publisher_releases_deltas_in_the_background(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import sampler_island as si
    from yeto.rl.engine.policy_release import BackgroundReleaser

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    words = [_words(40_000)]
    for v in range(1, 4):
        words.append(_changed(words[-1], 0.01, seed=v))
    events = []

    def export(version, directory):
        _write(directory / "model.safetensors", words[version])

    publisher = si.ReleasingPublisher(
        SimpleNamespace(publish=lambda state: None),
        PolicyReleaseStore(CheckpointStore(tmp_path / "store")), export, read_position=lambda: 0,
        emit=lambda e, **f: events.append((e, f)), staging=tmp_path / "staging",
        releaser=BackgroundReleaser(), delta=DeltaChain(full_every=4))
    for version in range(4):
        publisher.publish(_State(version))
    publisher.drain_releases()
    released = [f for e, f in events if e == "rl_policy_released"]
    assert [f["delta_base_seq"] for f in released] == [None, 1, 2, 3]
    assert all(f["delta_ratio"] < 0.1 for f in released[1:])
    assert released[3]["release_bytes"] < 0.1 * released[0]["release_bytes"]
    assert "delta_s" in released[1]["release_timings"]
    assert len(list((tmp_path / "staging").iterdir())) == 1  # the next base only
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror"))
    rebuilt = sampler.resolve(sampler.latest())
    assert (rebuilt / "model.safetensors").read_bytes() == words[3].astype("<u2").tobytes()
    # The chain's full release carries each file's sha256, the deltas do not
    records = sorted(sampler.records(), key=lambda r: r.seq)
    assert records[0].file_sha256 == {
        "model.safetensors": hashlib.sha256(words[0].astype("<u2").tobytes()).hexdigest()}
    assert all(r.file_sha256 is None for r in records[1:])


def test_publisher_refuses_deltas_without_staging(tmp_path):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import sampler_island as si

    with pytest.raises(ValueError, match="need staged releases"):
        si.ReleasingPublisher(SimpleNamespace(publish=lambda s: None),
                              PolicyReleaseStore(CheckpointStore(tmp_path)), lambda v, d: None,
                              read_position=lambda: 0, emit=lambda e, **f: None,
                              delta=DeltaChain(full_every=4))


# ---------------------------------------------------------------- the flag
def _trainer_args(**kw):
    from types import SimpleNamespace

    base = dict(rl_sample_role="trainer", rl_single_island_no_sync=True, tuning="full",
                rl_checkpoint_store="/store", rl_max_policy_age=2, rl_release_delta_full_every=8)
    return SimpleNamespace(**{**base, **kw})


def test_delta_flag_reaches_the_trainer_config_and_command():
    from yeto.rl.engine import sample_roles

    args = _trainer_args()
    sample_roles.check_launch(args)
    assert " --rl-release-delta-full-every 8" in sample_roles.island_flags(args)
    config = sample_roles.resolve(_trainer_args(rl_resume_store="/s"), groups_per_round=4)
    assert config.release_delta_full_every == 8
    sampler = sample_roles.resolve(_trainer_args(rl_sample_role="sampler", rl_resume_store="/s"),
                                   groups_per_round=4)
    assert sampler.release_delta_full_every == 0


@pytest.mark.parametrize("kw,message", [
    ({"rl_release_delta_full_every": 1}, "0 \\(off\\) or >= 2"),
    ({"rl_release_inline": True}, "staged releases"),
    ({"tuning": "lora"}, "full-weight releases"),
    ({"rl_rollout_quant_tier": 2}, "quant-tier"),
    # Verl is allowed, but only for a full-parameter island
    ({"rl_backend": "verl", "parameter_mode": "lora"}, "--parameter-mode full on a verl island"),
    ({"rl_backend": "torchtitan"}, "--rl-backend miles or verl"),
])
def test_delta_flag_refusals(kw, message):
    from yeto.rl.engine import sample_roles

    with pytest.raises(ValueError, match=message):
        sample_roles.check_launch(_trainer_args(**kw))


def test_a_verl_full_parameter_trainer_takes_the_delta_flag():
    """Its export writes the base checkpoint's own files, so the first
    delta against the base stays small (adapters/verl/hf_export)."""
    from yeto.rl.engine import sample_roles

    args = _trainer_args(rl_backend="verl", parameter_mode="full", rl_resume_store="/s")
    sample_roles.check_launch(args)
    assert sample_roles.resolve(args, groups_per_round=4).release_delta_full_every == 8


def test_delta_flag_on_a_both_role_island_is_refused():
    from types import SimpleNamespace

    from yeto.rl.engine import sample_roles

    with pytest.raises(ValueError, match="needs --rl-sample-role trainer"):
        sample_roles.check_launch(SimpleNamespace(rl_release_delta_full_every=4))


def test_pair_forwards_the_delta_flag_to_its_trainer_only():
    from types import SimpleNamespace

    from yeto.rl.engine import sample_roles

    pair = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:trainer"], rl_sampler_fused_stage=False,
                           rl_sampler_min_round_seconds=None, rl_sample_wait_s=None,
                           rl_sampler_max_backlog_rounds=None, rl_release_delta_full_every=8)
    assert sample_roles.island_args(pair, 0, 2).rl_release_delta_full_every == 0
    assert sample_roles.island_args(pair, 1, 2).rl_release_delta_full_every == 8


def test_follower_loads_the_rebuilt_copy_of_a_delta_release(tmp_path, monkeypatch):
    from yeto.rl.engine.policy_release import PolicyFollower

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    monkeypatch.setenv(rd.MIRROR_ENV, str(tmp_path / "mirror"))
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    chain = DeltaChain(full_every=4)
    words = _words(30_000)
    _release(trainer, chain, _stage(tmp_path / "s", 0, words), 0)
    _release(trainer, chain, _stage(tmp_path / "s", 1, _changed(words, 0.01)), 1)
    loads = []
    follower = PolicyFollower(PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                                 mirror=rd.default_mirror()),
                              lambda release, directory: loads.append((release.version, directory)))
    assert follower.refresh().version == 1
    assert loads == [(1, tmp_path / "mirror" / follower.release.path)]
    assert "resolve_s" in follower.last_timings


def test_a_base_whose_files_are_gone_gives_a_full_release(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    chain = DeltaChain(full_every=8)
    words = _words(20_000)
    _release(trainer, chain, _stage(tmp_path / "s", 0, words), 0)
    shutil.rmtree(chain.base().directory)
    assert _release(trainer, chain, _stage(tmp_path / "s", 1, _changed(words, 0.01)), 1).delta is None


def test_chain_apply_rebuilds_several_versions_in_one_pass(tmp_path):
    words = [_words(30_001)]
    for v in range(1, 4):
        words.append(_changed(words[-1], 0.02, seed=v))
    paths = [_write(tmp_path / f"v{v}", w, b"\x05") for v, w in enumerate(words)]
    deltas = []
    for v in range(1, 4):
        rd.encode_file(paths[v - 1], paths[v], tmp_path / f"d{v}", chunk_words=4096)
        deltas.append(tmp_path / f"d{v}")
    assert rd.apply_chain(paths[0], deltas, tmp_path / "out") == _sha(paths[3])
    assert (tmp_path / "out").read_bytes() == paths[3].read_bytes()
    with pytest.raises(rd.ReleaseDeltaError, match="another size or chunking"):
        rd.encode_file(paths[0], paths[1], tmp_path / "odd", chunk_words=2048)
        rd.apply_chain(paths[0], [tmp_path / "odd", deltas[1]], tmp_path / "x")


def test_directory_chain_restarts_at_a_whole_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    v0 = {"a.safetensors": _words(20_000, 1)}
    v1 = {"a.safetensors": _words(25_000, 2)}  # another size: whole in link 1
    v2 = {"a.safetensors": _changed(v1["a.safetensors"], 0.01)}
    dirs = [_release_dir(tmp_path / f"v{i}", v) for i, v in enumerate((v0, v1, v2))]
    links = []
    for i in (1, 2):
        result = rd.encode_directory(dirs[i - 1], dirs[i], tmp_path / f"d{i}")
        links.append((tmp_path / f"d{i}", result.entries))
    assert [e["a.safetensors"].mode for _, e in links] == ["full", "delta"]
    rd.apply_directory(dirs[0], links, tmp_path / "out")
    assert (tmp_path / "out" / "a.safetensors").read_bytes() == (dirs[2] / "a.safetensors").read_bytes()


def test_a_sampler_two_versions_behind_rebuilds_once(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=4)
    chain = DeltaChain(full_every=8)
    words = [_words(40_000)]
    _release(trainer, chain, _stage(tmp_path / "s", 0, words[0]), 0)
    for version in (1, 2, 3):
        words.append(_changed(words[-1], 0.01, seed=version))
        _release(trainer, chain, _stage(tmp_path / "s", version, words[-1]), version)
    calls = []
    real_chain = rd.apply_chain
    monkeypatch.setattr(rd, "apply_chain", lambda base, deltas, out: (
        calls.append(len(deltas)), real_chain(base, deltas, out))[1])
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror"))
    rebuilt = sampler.resolve(sampler.latest())
    assert calls == [3]  # v1..v3 on the full v0, one pass
    assert (rebuilt / "model.safetensors").read_bytes() == words[3].astype("<u2").tobytes()
    assert [p.name for p in (tmp_path / "mirror").iterdir()] == [sampler.latest().path]


# ---------------------------------------------------------------- rebuild failures
def _flip_first_word(path: Path) -> None:
    """Change the file's bytes, not its size (size checks pass, the digest does not)."""
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))


def _delta_releases(tmp_path: Path, versions: int):
    """A store with the full v0 and delta releases v1..``versions``; returns (trainer, words)."""
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=8)
    chain = DeltaChain(full_every=16)
    words = [_words(30_000)]
    _release(trainer, chain, _stage(tmp_path / "s", 0, words[0]), 0)
    for version in range(1, versions + 1):
        words.append(_changed(words[-1], 0.01, seed=version))
        _release(trainer, chain, _stage(tmp_path / "s", version, words[-1]), version)
    return trainer, chain, words


def _follower(tmp_path: Path, loads: list, **kw):
    from yeto.rl.engine.policy_release import PolicyFollower

    return PolicyFollower(PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                             mirror=rd.DeltaMirror(tmp_path / "mirror", keep=4)),
                          lambda release, directory: loads.append(
                              (release.version, (directory / "model.safetensors").read_bytes())), **kw)


def test_a_failed_rebuild_keeps_the_served_weights_and_the_retry_starts_from_the_store(
        tmp_path, monkeypatch):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer, chain, words = _delta_releases(tmp_path, 1)
    loads = []
    follower = _follower(tmp_path, loads)
    assert follower.refresh().version == 1  # rebuilt in the mirror from the full v0
    # the mirror's copy of v1 goes bad (same size): v2 rebuilt on it fails its digest
    _flip_first_word(tmp_path / "mirror" / follower.release.path / "model.safetensors")
    words.append(_changed(words[-1], 0.01, seed=9))
    _release(trainer, chain, _stage(tmp_path / "s", 2, words[-1]), 2)
    with pytest.raises(ReleaseNotLoaded) as caught:
        follower.refresh()
    assert caught.value.reason == "delta_rebuild_failed"
    assert caught.value.fields == {"release_version": 2, "file": "model.safetensors",
                                   "rebuild_failures": 1, "max_rebuild_failures": 3}
    assert follower.version == 1 and [v for v, _ in loads] == [1]  # the engines keep v1
    # the next try ignores the mirror: v0 from the store, deltas v1 and v2 in one pass
    applied = []
    real_apply = rd.apply_directory
    monkeypatch.setattr(rd, "apply_directory", lambda base, links, *a, **k: (
        applied.append((Path(base).parent.name, len(links))), real_apply(base, links, *a, **k))[1])
    assert follower.refresh().version == 2
    assert applied == [("policy-releases", 2)]
    assert loads[-1] == (2, words[2].astype("<u2").tobytes())
    assert follower.rebuild_failures == 0 and follower.counters.rebuild_failures == 1


def test_rebuild_failures_in_a_row_stop_the_sampler(tmp_path, monkeypatch):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    _delta_releases(tmp_path, 1)
    # the full v0 in the store goes bad: every rebuild of v1 fails
    v0 = sorted((tmp_path / "store" / "policy-releases").glob("r*-v0"))[0]
    _flip_first_word(v0 / "model.safetensors")
    loads = []
    follower = _follower(tmp_path, loads, max_rebuild_failures=2)
    with pytest.raises(ReleaseNotLoaded, match="rebuilds from the full release"):
        follower.refresh()
    with pytest.raises(rd.ReleaseDeltaError, match="2 delta rebuilds or digest checks in a row failed") as caught:
        follower.refresh()
    assert caught.value.file == "model.safetensors" and loads == [] and follower.version == 0
    with pytest.raises(ValueError, match="max_rebuild_failures"):
        _follower(tmp_path, loads, max_rebuild_failures=0)


def test_a_retry_downloads_the_chain_again(tmp_path, monkeypatch):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer, chain, words = _delta_releases(tmp_path, 1)
    store = CheckpointStore(tmp_path / "store")
    store.objects = DirectoryObjectStore(tmp_path / "store")
    reader = PolicyReleaseStore(store, cache_dir=tmp_path / "cache",
                                mirror=rd.DeltaMirror(tmp_path / "mirror"))
    latest = reader.latest()
    v0 = next((tmp_path / "store" / "policy-releases").glob("r*-v0")) / "model.safetensors"
    good = v0.read_bytes()
    _flip_first_word(v0)  # a bad copy is downloaded, the rebuild fails, the download stays
    with pytest.raises(rd.ReleaseDeltaError):
        reader.resolve(latest)
    v0.write_bytes(good)
    with pytest.raises(rd.ReleaseDeltaError):
        reader.resolve(latest)  # the cached bad download again
    rebuilt = reader.resolve(latest, from_store=True)
    assert (rebuilt / "model.safetensors").read_bytes() == words[1].astype("<u2").tobytes()


def test_the_sampler_loop_reports_a_failed_rebuild_and_generates_on(tmp_path, monkeypatch):
    from yeto.rl.engine.sampler_loop import SamplerLoop

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    trainer, chain, words = _delta_releases(tmp_path, 1)
    loads = []
    follower = _follower(tmp_path, loads)
    follower.refresh()
    _flip_first_word(tmp_path / "mirror" / follower.release.path / "model.safetensors")
    words.append(_changed(words[-1], 0.01, seed=9))
    _release(trainer, chain, _stage(tmp_path / "s", 2, words[-1]), 2)

    class _Buffer:
        def refresh(self):
            return 0

        def __len__(self):
            return 0

        def mark_data_end(self):
            pass

    # the retry fails too (the store's copy of v1's delta is changed in place)
    delta_file = next((tmp_path / "store" / "policy-releases").glob("r*-v1")) / "model.safetensors.ydelta"
    original = delta_file.read_bytes()
    _flip_first_word(delta_file)
    events, generated = [], []
    loop = SamplerLoop(follower=follower, buffer=_Buffer(), num_rollouts=3, max_backlog_groups=1,
                       generate=lambda rid: (generated.append((rid, follower.version)),
                                             rid == 1 and delta_file.write_bytes(original)),
                       emit=lambda event, **f: events.append((event, f)), poll_s=0.0)
    assert loop.run() == 3
    not_loaded = [f for e, f in events if e == "rl_sampler_release_not_loaded"]
    assert [(f["seq"], f["release_version"], f["rebuild_failures"], f["reason"]) for f in not_loaded] == [
        (3, 2, 1, "delta_rebuild_failed"), (3, 2, 2, "delta_rebuild_failed")]
    assert all(f["file"] == "model.safetensors" and f["policy_version"] == 1 for f in not_loaded)
    assert generated == [(0, 1), (1, 1), (2, 2)]  # v1 served until the store copy is good again
    rounds = [f for e, f in events if e == "rl_sampler_round"]
    assert rounds[-1]["releases"]["rebuild_failures"] == 2


def test_a_full_release_with_file_digests_is_checked_before_it_loads(tmp_path, monkeypatch):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    staged = _stage(tmp_path / "s", 0, _words(20_000))
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    plain = trainer.release_staged(0, "h0", staged, position=lambda: (0, None))
    assert plain.file_sha256 is None  # off by default
    record = trainer.release_staged(1, "h1", staged, position=lambda: (1, None), digest_full=True)
    assert record.file_sha256 == {"config.json": _sha(staged / "config.json"),
                                  "model.safetensors": _sha(staged / "model.safetensors")}
    weights = tmp_path / "store" / "policy-releases" / record.path / "model.safetensors"
    good = weights.read_bytes()
    _flip_first_word(weights)
    loads = []
    follower = _follower(tmp_path, loads)
    with pytest.raises(ReleaseNotLoaded) as caught:
        follower.refresh()
    assert caught.value.reason == "release_digest_mismatch"
    assert caught.value.fields["file"] == "model.safetensors" and loads == []
    weights.write_bytes(good)
    assert follower.refresh().version == 1 and [v for v, _ in loads] == [1]
    assert follower.counters.rebuild_failures == 1 and follower.rebuild_failures == 0


# ---------------------------------------------------------------- format v2
needs_zstd = pytest.mark.skipif(not rd.zstd_available(), reason="no zstd (Python < 3.14 without zstandard)")


def _bf16_step(words: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A trainer-like update: a fraction of the words move by +-1 bf16 step."""
    out = words.copy()
    rng = np.random.default_rng(seed)
    index = rng.choice(words.size, size=int(words.size * fraction), replace=False)
    out[index] = (out[index].astype(np.int32) + rng.choice([-1, 1], size=index.size)).astype(np.uint16)
    return out


@needs_zstd
@pytest.mark.parametrize("fraction", [0.0, 0.001, 0.05, 0.5, 1.0])
@pytest.mark.parametrize("tail", [b"", b"\x07"])
def test_v2_round_trip_is_byte_exact_with_the_same_digest(tmp_path, fraction, tail):
    base_words = _words(70_001)
    base = _write(tmp_path / "base", base_words, tail)
    target = _write(tmp_path / "new", _bf16_step(base_words, fraction, 3), tail)
    c1, d1 = rd.encode_file(base, target, tmp_path / "d1", chunk_words=4096)
    c2, d2 = rd.encode_file(base, target, tmp_path / "d2", chunk_words=4096, fmt=2)
    assert (c1, d1) == (c2, d2) and d2 == _sha(target)
    assert (tmp_path / "d2").read_bytes().endswith(rd.MAGIC_V2)
    assert rd.apply_file(base, tmp_path / "d2", tmp_path / "out") == d2
    assert (tmp_path / "out").read_bytes() == target.read_bytes()


@needs_zstd
def test_v2_is_smaller_than_v1_on_sparse_bf16_steps(tmp_path):
    base_words = _words(1 << 20)
    base = _write(tmp_path / "a", base_words)
    target = _write(tmp_path / "b", _bf16_step(base_words, 0.012, 4))
    rd.encode_file(base, target, tmp_path / "d1", chunk_words=1 << 18)
    rd.encode_file(base, target, tmp_path / "d2", chunk_words=1 << 18, fmt=2)
    assert (tmp_path / "d2").stat().st_size < 0.6 * (tmp_path / "d1").stat().st_size


@needs_zstd
def test_a_chain_mixes_v1_and_v2_deltas(tmp_path):
    words = [_words(30_001)]
    for v in range(1, 5):
        words.append(_bf16_step(words[-1], 0.02, seed=v))
    paths = [_write(tmp_path / f"v{v}", w, b"\x05") for v, w in enumerate(words)]
    deltas = []
    for v in range(1, 5):
        rd.encode_file(paths[v - 1], paths[v], tmp_path / f"d{v}", chunk_words=4096, fmt=1 + v % 2)
        deltas.append(tmp_path / f"d{v}")
    assert rd.apply_chain(paths[0], deltas, tmp_path / "out") == _sha(paths[4])
    assert (tmp_path / "out").read_bytes() == paths[4].read_bytes()


@needs_zstd
def test_v2_refuses_a_corrupt_frame_and_a_wrong_magic(tmp_path):
    base = _write(tmp_path / "a", _words(10_000))
    target = _write(tmp_path / "b", _bf16_step(_words(10_000), 0.1, 5))
    rd.encode_file(base, target, tmp_path / "d", fmt=2)
    raw = bytearray((tmp_path / "d").read_bytes())
    raw[10] ^= 0xFF
    (tmp_path / "bad").write_bytes(bytes(raw))
    with pytest.raises(rd.ReleaseDeltaError):
        rd.apply_file(base, tmp_path / "bad", tmp_path / "x")
    good = (tmp_path / "d").read_bytes()
    (tmp_path / "v1magic").write_bytes(good[:-len(rd.MAGIC)] + rd.MAGIC)
    with pytest.raises(rd.ReleaseDeltaError, match="format does not match"):
        rd.apply_file(base, tmp_path / "v1magic", tmp_path / "x")


@needs_zstd
def test_directory_delta_takes_the_format_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    v0 = {"a.safetensors": _words(20_000, 1)}
    v1 = {"a.safetensors": _bf16_step(v0["a.safetensors"], 0.01, 2)}
    d0, d1 = (_release_dir(tmp_path / f"v{i}", v) for i, v in enumerate((v0, v1)))
    monkeypatch.setenv(rd.FORMAT_ENV, "v2")
    result = rd.encode_directory(d0, d1, tmp_path / "d")
    assert (tmp_path / "d" / "a.safetensors.ydelta").read_bytes().endswith(rd.MAGIC_V2)
    rd.apply_directory(d0, [(tmp_path / "d", result.entries)], tmp_path / "out")
    assert (tmp_path / "out" / "a.safetensors").read_bytes() == (d1 / "a.safetensors").read_bytes()


def test_delta_format_parsing():
    assert rd.delta_format({}) == 1
    assert rd.delta_format({rd.FORMAT_ENV: "V2"}) == 2 and rd.delta_format({rd.FORMAT_ENV: "1"}) == 1
    with pytest.raises(ValueError):
        rd.delta_format({rd.FORMAT_ENV: "v3"})


# ---------------------------------------------------------------- mirror identity
def _v2_chain(tmp_path, monkeypatch, versions: int = 4, *, root: str = "store", seed: int = 0):
    """A trainer that releases a full v0, then v2 deltas, each against the release
    before it; returns (store, the releases, the staged words of each)."""
    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    monkeypatch.setenv(rd.FORMAT_ENV, "v2")
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / root), keep=2)
    chain = DeltaChain(full_every=64)
    words = [_words(60_000, seed)]
    staging = tmp_path / f"staging-{root}"
    releases = [_release(trainer, chain, _stage(staging, 0, words[0]), 0)]
    for version in range(1, versions):
        words.append(_changed(words[-1], 0.02, seed=seed * 100 + version))
        releases.append(_release(trainer, chain, _stage(staging, version, words[-1]), version))
    return trainer, releases, words


@needs_zstd
def test_v2_delta_chain_rebuilds_byte_exact_through_every_link(tmp_path, monkeypatch):
    _trainer, releases, words = _v2_chain(tmp_path, monkeypatch, versions=5)
    assert [r.delta["base_path"] for r in releases[1:]] == [r.path for r in releases[:-1]]
    for release in releases[1:]:
        delta = tmp_path / "store" / "policy-releases" / release.path / "model.safetensors.ydelta"
        with delta.open("rb") as handle:
            assert rd._footer(handle, delta).get("format") == 2
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                 mirror=rd.DeltaMirror(tmp_path / "mirror", workers=1))
    rebuilt = sampler.resolve(sampler.latest())
    assert (rebuilt / "model.safetensors").read_bytes() == words[-1].astype("<u2").tobytes()
    assert json.loads((rebuilt / rd.MIRROR_STAMP).read_text()) == rd.mirror_stamp(releases[-1])


@needs_zstd
def test_a_stale_mirror_copy_with_the_same_name_is_not_used_as_a_base(tmp_path, monkeypatch):
    """The host-wide mirror held another run's ``r<seq>-v<version>``; the
    rebuild started from it and failed its digest check. Unstamped or foreign
    copies are skipped; the chain is rebuilt from the full release in the store."""
    _trainer, releases, words = _v2_chain(tmp_path, monkeypatch, versions=4)
    _other, foreign, _ = _v2_chain(tmp_path, monkeypatch, versions=4, root="other", seed=7)
    assert foreign[-2].path == releases[-2].path  # same names, other weights
    mirror = rd.DeltaMirror(tmp_path / "mirror", workers=1)
    other = PolicyReleaseStore(CheckpointStore(tmp_path / "other"), mirror=mirror)
    stale = other.resolve(foreign[-2])  # a stamped copy of the other store's release
    assert stale == tmp_path / "mirror" / releases[-2].path
    assert rd.mirror_stamp(foreign[-2]) != rd.mirror_stamp(releases[-2])
    assert mirror.holds(releases[-2].path) and not mirror.holds(releases[-2].path, releases[-2])
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), mirror=mirror)
    rebuilt = sampler.resolve(sampler.latest())
    assert (rebuilt / "model.safetensors").read_bytes() == words[-1].astype("<u2").tobytes()
    # a copy from before the stamp (no stamp file) is not trusted either
    (stale / rd.MIRROR_STAMP).unlink(missing_ok=True)
    assert not mirror.holds(releases[-2].path, releases[-2])


@needs_zstd
def test_a_failed_rebuild_leaves_no_partial_and_a_retry_from_the_store_replaces_the_copy(tmp_path, monkeypatch):
    _trainer, releases, words = _v2_chain(tmp_path, monkeypatch, versions=3)
    mirror = rd.DeltaMirror(tmp_path / "mirror", workers=1)
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), mirror=mirror)
    first = sampler.resolve(releases[-1])
    second = sampler.resolve(releases[-1], from_store=True)  # the target directory exists already
    assert first == second
    assert (second / "model.safetensors").read_bytes() == words[-1].astype("<u2").tobytes()
    real = rd.apply_directory
    monkeypatch.setattr(rd, "apply_directory", lambda *a, **k: (real(*a, **k), (_ for _ in ()).throw(
        rd.ReleaseDeltaError("boom", file="model.safetensors")))[0])
    with pytest.raises(rd.ReleaseDeltaError):
        sampler.resolve(releases[-1], from_store=True)
    assert not [p for p in (tmp_path / "mirror").iterdir() if p.name.endswith(".partial")]


def test_chain_apply_refuses_deltas_with_another_chunk_count(tmp_path):
    words = [_words(30_001)]
    for v in range(1, 3):
        words.append(_changed(words[-1], 0.02, seed=v))
    paths = [_write(tmp_path / f"v{v}", w) for v, w in enumerate(words)]
    for v in range(1, 3):
        rd.encode_file(paths[v - 1], paths[v], tmp_path / f"d{v}", chunk_words=4096)
    # rewrite the second footer with one chunk record dropped (same size and chunking)
    delta = tmp_path / "d2"
    with delta.open("rb") as handle:
        footer = rd._footer(handle, delta)
        end = handle.seek(0, 2)
    raw = delta.read_bytes()
    trailer = rd._FOOTER.size + len(rd.MAGIC)
    (length,) = rd._FOOTER.unpack(raw[end - trailer:end - trailer + rd._FOOTER.size])
    body = raw[:end - trailer - length]
    footer["chunks"] = footer["chunks"][:-1]
    text = json.dumps(footer, separators=(",", ":")).encode()
    delta.write_bytes(body + text + rd._FOOTER.pack(len(text)) + raw[-len(rd.MAGIC):])
    with pytest.raises(rd.ReleaseDeltaError, match="chunks"):
        rd.apply_chain(paths[0], [tmp_path / "d1", delta], tmp_path / "x")


def test_default_mirror_puts_each_store_in_its_own_directory(tmp_path):
    env = {rd.MIRROR_ENV: str(tmp_path / "mirror")}
    a, b = rd.default_mirror(env, store="s3://bucket/run-a"), rd.default_mirror(env, store="s3://bucket/run-b")
    assert a.root != b.root and a.root.parent == b.root.parent == tmp_path / "mirror"
    assert rd.default_mirror(env, store="s3://bucket/run-a").root == a.root
    assert rd.default_mirror(env).root == tmp_path / "mirror"


@needs_zstd
def test_two_stores_sharing_a_mirror_root_do_not_touch_each_other(tmp_path, monkeypatch):
    """Two runs on one host with the same mirror root and the same release names:
    one run's rebuilds and pruning leave the copy the other one serves alone."""
    _t, mine, words = _v2_chain(tmp_path, monkeypatch, versions=4)
    _o, theirs, other_words = _v2_chain(tmp_path, monkeypatch, versions=4, root="other", seed=7)
    env = {rd.MIRROR_ENV: str(tmp_path / "mirror")}
    a = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                           mirror=rd.default_mirror(env, store=str(tmp_path / "store")))
    b = PolicyReleaseStore(CheckpointStore(tmp_path / "other"),
                           mirror=rd.default_mirror(env, store=str(tmp_path / "other")))
    a.mirror.keep = b.mirror.keep = 1
    served = b.resolve(theirs[1])
    for release in mine[1:]:
        rebuilt = a.resolve(release)
        assert (rebuilt / "model.safetensors").read_bytes() == words[mine.index(release)].astype("<u2").tobytes()
    a.resolve(mine[1], from_store=True)  # a stale same-named copy in a's own directory is replaced
    assert served.is_dir() and served.name == mine[1].path
    assert (served / "model.safetensors").read_bytes() == other_words[1].astype("<u2").tobytes()


@needs_zstd
def test_prune_skips_copies_stamped_for_another_scope(tmp_path, monkeypatch):
    _t, mine, _words_a = _v2_chain(tmp_path, monkeypatch, versions=4)
    root = tmp_path / "mirror"
    foreign = root / "r99-v99"
    foreign.mkdir(parents=True)
    (foreign / rd.MIRROR_STAMP).write_text(json.dumps({"path": "r99-v99", "seq": 99, "scope": "other"}))
    unstamped = root / "r98-v98"
    unstamped.mkdir()
    mirror = rd.DeltaMirror(root, keep=1, workers=1, scope="store")
    sampler = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), mirror=mirror)
    for release in mine[1:]:
        sampler.resolve(release)
    left = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    assert left == sorted([mine[-1].path, "r99-v99", "r98-v98"])


@needs_zstd
def test_concurrent_rebuilds_of_one_release_both_succeed(tmp_path, monkeypatch):
    """Two processes of one run rebuild the same delta at once: each works in its own
    partial directory, neither deletes the other's, both end on a correct copy."""
    import threading

    _t, releases, words = _v2_chain(tmp_path, monkeypatch, versions=3)
    root = tmp_path / "mirror"
    gate = threading.Barrier(2)
    real = rd.apply_directory

    def slow_apply(base, chain, out, **kw):
        result = real(base, chain, out, **kw)
        gate.wait(timeout=10)  # both partials exist at the same time
        return result

    monkeypatch.setattr(rd, "apply_directory", slow_apply)
    out, errors = [], []

    def rebuild():
        try:
            store = PolicyReleaseStore(CheckpointStore(tmp_path / "store"),
                                       mirror=rd.DeltaMirror(root, workers=1, scope="store"))
            out.append(store.resolve(releases[-1]))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=rebuild) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors and len(out) == 2 and out[0] == out[1]
    assert (out[0] / "model.safetensors").read_bytes() == words[-1].astype("<u2").tobytes()
    assert not [p for p in root.iterdir() if p.name.startswith(".")]


def test_a_restarted_trainer_keeps_the_delta_chain_bound(tmp_path):
    """--rl-release-delta-full-every N: a sampler rebuilds at most N-1 deltas. A
    restarted trainer adopts LATEST as its base; when LATEST is itself the
    (N-1)-th delta, its next release must be full, else every restart lengthens
    the chain a cold sampler rebuilds (and the store keeps every base of it)."""
    from yeto.rl.adapters.verl.buffer_ports import delta_chain_for
    from yeto.rl.engine.policy_release import DeltaChain, PolicyReleaseStore
    from yeto.rl.engine.release_delta import DeltaMirror
    from yeto.rl.engine.resume import CheckpointStore

    store = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=2,
                               mirror=DeltaMirror(tmp_path / "mirror"))
    chain = DeltaChain(3)
    for version in range(1, 4):
        staged = tmp_path / "staging" / f"v{version}"
        staged.mkdir(parents=True)
        data = bytearray(1 << 20)
        data[version * 8] = version
        (staged / "model.safetensors").write_bytes(bytes(data))
        release = store.release_staged(version, f"h{version}", staged, position=lambda: (None, None),
                                       delta_base=chain.base())
        chain.advance(release, staged)
    assert release.delta is not None and chain.base() is None  # the trainer's next release: full
    restarted = delta_chain_for(PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=2,
                                                   mirror=DeltaMirror(tmp_path / "mirror2")), 3)
    assert restarted.base() is None
