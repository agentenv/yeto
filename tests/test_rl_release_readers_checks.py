"""Readers of policy releases: a trainer started from a release checks the files
against the record (``--rl-init-from-release``), and a sampler's
``rl_sampler_policy_loaded.seen_at`` is when it first read *that* release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.cloud import s3_store
from yeto.rl.engine import init_from_release as ifr
from yeto.rl.engine.init_from_release import RECORD, InitFromReleaseError, prepare, prepare_if_released
from yeto.rl.engine.policy_release import LORA, PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_loop import SamplerLoop

from test_rl_object_store import FakeS3

WEIGHTS = {"model-00001.safetensors": b"w" * 64, "config.json": b"{}"}


def _staged(tmp: Path, name: str, files: dict[str, bytes] = WEIGHTS) -> Path:
    d = tmp / "export" / name
    d.mkdir(parents=True)
    for rel, data in files.items():
        (d / rel).write_bytes(data)
    return d


def _pos():
    return None, None


def _s3(tmp: Path, name: str, client: FakeS3) -> s3_store.S3Store:
    return s3_store.S3Store(tmp / name, "s3://bucket/run", environ={}, client=client, stage=tmp / name)


def _full_release_with_digests(releases: PolicyReleaseStore, tmp: Path, version: int = 1):
    """A full release whose record carries file_sha256 (a full release of a delta chain)."""
    record = releases.release_staged(version, f"h{version}", _staged(tmp, f"v{version}"), position=_pos,
                                     digest_full=True)
    assert record.file_sha256 and set(record.file_sha256) == set(WEIGHTS)
    return record


def _base(tmp: Path) -> Path:
    base = tmp / "base"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text("{}")
    return base


# -- --rl-init-from-release: file_sha256 ------------------------------------------------
def test_prepare_refuses_files_that_differ_from_the_records_sha256(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=3)
    record = _full_release_with_digests(releases, tmp_path)
    (releases.root / record.path / "model-00001.safetensors").write_bytes(b"x" * 64)  # same size
    out = tmp_path / "init"
    out.mkdir()
    (out / RECORD).write_text(json.dumps({"path": "r0-v0"}))  # an earlier preparation's record
    with pytest.raises(InitFromReleaseError, match=r"r1-v1: the weight files differ .*model-00001"):
        prepare(releases, "latest", _base(tmp_path), out, verify="full")
    assert not (out / RECORD).exists()  # no record that looks usable is left behind


def test_prepare_passes_matching_files_and_off_checks_sizes_only(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=3)
    record = _full_release_with_digests(releases, tmp_path)
    out = tmp_path / "init"
    assert prepare(releases, "latest", _base(tmp_path), out, verify="full")["path"] == record.path
    (releases.root / record.path / "model-00001.safetensors").write_bytes(b"x" * 64)
    assert prepare(releases, "latest", _base(tmp_path), out, verify="off")["path"] == record.path
    with pytest.raises(InitFromReleaseError):
        prepare(releases, "latest", _base(tmp_path), out, verify="fast")


def test_prepare_follows_yeto_verify_by_default(tmp_path, monkeypatch):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=3)
    record = _full_release_with_digests(releases, tmp_path)
    (releases.root / record.path / "model-00001.safetensors").write_bytes(b"x" * 64)
    monkeypatch.setenv("YETO_VERIFY", "off")
    assert prepare(releases, "latest", _base(tmp_path), tmp_path / "a")["path"] == record.path
    monkeypatch.setenv("YETO_VERIFY", "full")
    with pytest.raises(InitFromReleaseError):
        prepare_if_released(releases, _base(tmp_path), tmp_path / "b")
    assert not (tmp_path / "b" / RECORD).exists()


def test_a_damaged_download_is_fetched_again_from_the_store(tmp_path):
    client = FakeS3()
    writer = PolicyReleaseStore(_s3(tmp_path, "t", client), keep=3, cache_dir=tmp_path / "ct")
    record = _full_release_with_digests(writer, tmp_path)
    out = tmp_path / "init"
    reader = PolicyReleaseStore(_s3(tmp_path, "n", client), cache_dir=out / "files")
    cached = reader.resolve(record)
    (cached / "model-00001.safetensors").write_bytes(b"x" * 64)  # an earlier download, damaged on disk
    result = prepare(reader, "latest", _base(tmp_path), out, verify="full")
    assert result["path"] == record.path
    assert Path(result["model_dir"], "model-00001.safetensors").read_bytes() == WEIGHTS["model-00001.safetensors"]


def test_a_store_copy_that_differs_fails_after_one_fresh_download(tmp_path):
    client = FakeS3()
    writer = PolicyReleaseStore(_s3(tmp_path, "t", client), keep=3, cache_dir=tmp_path / "ct")
    record = _full_release_with_digests(writer, tmp_path)
    objects = s3_store.S3ObjectStore("s3://bucket/run", client=client)
    objects.write(f"policy-releases/{record.path}/model-00001.safetensors", b"x" * 64)
    out = tmp_path / "init"
    reader = PolicyReleaseStore(_s3(tmp_path, "n", client), cache_dir=out / "files")
    downloads = []
    resolve = reader.resolve
    reader.resolve = lambda r, **kw: (downloads.append(kw.get("from_store", False)), resolve(r, **kw))[1]
    with pytest.raises(InitFromReleaseError, match="also after a fresh download"):
        prepare(reader, "latest", _base(tmp_path), out, verify="full")
    assert downloads == [False, True] and not (out / RECORD).exists()


def test_a_delta_release_is_not_hashed_again(tmp_path, monkeypatch):
    """Its rebuild checked every file against the record already."""
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=3)
    record = _full_release_with_digests(releases, tmp_path)
    delta = type(record)(**{**record.__dict__, "seq": 2, "version": 2, "path": "r2-v2",
                            "delta": {"base_path": record.path}})
    calls = []
    monkeypatch.setattr(ifr, "check_release_files", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(releases, "resolve", lambda r, **kw: releases.root / record.path)
    assert ifr.resolve_checked(releases, delta, verify="full") == releases.root / record.path
    assert calls == []
    ifr.resolve_checked(releases, record, verify="full")
    assert len(calls) == 1


def test_main_passes_yeto_verify_of_the_environment(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ifr, "prepare", lambda *a, **kw: seen.update(kw) or {"path": "r1-v1"})
    monkeypatch.setattr(ifr, "_base_path", lambda model, revision: str(tmp_path))
    ifr.main(["--release", "latest", "--store", str(tmp_path / "store"), "--model", str(tmp_path),
              "--out", str(tmp_path / "out")], environ={"YETO_VERIFY": "fast"})
    assert seen["verify"] == "fast"


# -- seen_at: per release ---------------------------------------------------------------
def _adapter(value: str):
    def write(directory: Path) -> None:
        (directory / "adapter_config.json").write_text('{"r": 8}')
        (directory / "adapter_model.safetensors").write_text(value)
    return write


def test_a_release_loaded_after_a_newer_one_appeared_keeps_its_own_seen_at(tmp_path):
    """The measured case: v19 downloaded and drained for, v20 published during the
    drain; the follower loads v19, then v20. v19's event carried v20's time."""
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    now = {"t": 100.0}
    loaded = []
    follower = PolicyFollower(releases, lambda r, d: loaded.append(r.seq), background_download=True,
                              wall=lambda: now["t"])
    releases.release(1, "h1", _adapter("a"), weights=LORA)
    assert follower.prefetch() == 1  # on disk: the sampler drains for it
    now["t"] = 200.0
    releases.release(2, "h2", _adapter("b"), weights=LORA)  # arrives during the drain
    assert follower.pending_seq() == 1
    first = follower.refresh()
    assert first.seq == 1 and follower.seen_at == 200.0  # the newest seen: r2
    assert follower.loaded_seen_at == 100.0  # r1 was first read at 100
    now["t"] = 300.0
    second = None
    for _ in range(50):  # r2's background download
        second = follower.refresh() or second
        if second is not None:
            break
        follower._fetch.done.wait(1.0)
    assert second.seq == 2 and follower.loaded_seen_at == 200.0 and loaded == [1, 2]


def test_seen_at_of_a_release_is_its_first_read_across_refreshes(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    wall = iter([100.0, 150.0, 200.0, 250.0])
    follower = PolicyFollower(releases, lambda r, d: None, wall=lambda: next(wall))
    releases.release(1, "h1", _adapter("a"), weights=LORA)
    with follower.generating():
        assert follower.refresh() is None  # read, deferred by the in-flight rule
    assert follower.refresh().seq == 1 and follower.loaded_seen_at == 100.0


def test_the_loaded_event_carries_the_loaded_releases_seen_at(tmp_path):
    store = CheckpointStore(tmp_path)
    releases = PolicyReleaseStore(store, clock=lambda: 50.0)
    follower = PolicyFollower(releases, lambda r, d: None, wall=lambda: 60.0)
    follower.loaded_seen_at = None
    events = []
    loop = SamplerLoop(follower=follower, buffer=FileSampleBuffer(store, batch_size=2),
                       generate=lambda r: None, num_rollouts=1, max_backlog_groups=4,
                       emit=lambda e, **f: events.append((e, f)))
    releases.release(1, "h1", _adapter("a"), weights=LORA)
    follower._first_seen[1] = 55.0  # read once before (e.g. by a prefetch)
    loop.run()
    (event,) = [f for e, f in events if e == "rl_sampler_policy_loaded"]
    assert event["seen_at"] == 55.0
