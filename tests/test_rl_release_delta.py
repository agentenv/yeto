"""S27 G-BG: bf16 delta releases (trainer encodes, sampler rebuilds). CPU only, no Ray."""

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
    rd.apply_directory(base, tmp_path / "d", delta.entries, tmp_path / "out", workers=2)
    for name in delta.entries:
        assert (tmp_path / "out" / name).read_bytes() == (target / name).read_bytes()
    corrupt = dict(delta.entries)
    corrupt["model-1.safetensors"] = rd.FileEntry("delta", corrupt["model-1.safetensors"].size, "0" * 64)
    with pytest.raises(rd.ReleaseDeltaError, match="rebuilt sha256"):
        rd.apply_directory(base, tmp_path / "d", corrupt, tmp_path / "out2")


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
    ({"rl_backend": "verl"}, "--rl-backend miles"),
])
def test_delta_flag_refusals(kw, message):
    from yeto.rl.engine import sample_roles

    with pytest.raises(ValueError, match=message):
        sample_roles.check_launch(_trainer_args(**kw))


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
