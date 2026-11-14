"""YETO_VERIFY=full|fast|off for release/cut file checks (file_verify)."""
import json
import os

import pytest

from yeto.rl.engine import cut as cut_mod
from yeto.rl.engine import file_verify as fv
from yeto.rl.engine import policy_release as pr
from yeto.rl.spot_rollout import cut_manifest as cm


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv(fv.ENV_CACHE, str(tmp_path / "verify-cache"))
    monkeypatch.delenv(fv.ENV_MODE, raising=False)
    return tmp_path / "verify-cache"


def _count_hashes(monkeypatch):
    calls = []
    real = fv.sha256_file

    def counting(path, **kw):
        calls.append(os.fspath(path))
        return real(path, **kw)

    monkeypatch.setattr(fv, "sha256_file", counting)
    return calls


def _release(tmp_path, name="r2-v2"):
    root = tmp_path / "store" / "policy-releases"
    rel = root / name
    rel.mkdir(parents=True)
    (rel / "a.safetensors").write_bytes(b"A" * 4096)
    (rel / "b.safetensors").write_bytes(b"B" * 4096)
    shas = {n: fv.sha256_file(rel / n, remember=False) for n in ("a.safetensors", "b.safetensors")}
    record = {"schema": pr.SCHEMA, "policy_hash": "p" * 64, "path": name, "version": 2, "seq": 2,
              "weights": pr.FULL,
              "files": {"a.safetensors": 4096, "b.safetensors": 4096}, "file_sha256": shas}
    (root / "LATEST.json").write_text(json.dumps(record))
    return tmp_path / "store", rel, record


def test_default_is_full_and_writes_no_cache(tmp_path, monkeypatch, _cache):
    assert fv.verify_mode({}) == fv.FULL
    _, rel, _ = _release(tmp_path)
    calls = _count_hashes(monkeypatch)
    fv.digests([rel / "a.safetensors"])
    fv.digests([rel / "a.safetensors"])
    assert len(calls) == 2 and not _cache.exists()


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        fv.verify_mode({fv.ENV_MODE: "fastest"})


def test_fast_hashes_an_unchanged_file_once(tmp_path, monkeypatch):
    _, rel, _ = _release(tmp_path)
    calls = _count_hashes(monkeypatch)
    first = fv.digests([rel / "a.safetensors"], mode=fv.FAST)
    second = fv.digests([rel / "a.safetensors"], mode=fv.FAST)
    assert first == second and len(calls) == 1


def test_fast_rehashes_a_file_rewritten_with_its_size_and_mtime_kept(tmp_path):
    _, rel, record = _release(tmp_path)
    path = rel / "a.safetensors"
    assert fv.digests([path], mode=fv.FAST)[os.fspath(path)] == record["file_sha256"]["a.safetensors"]
    st = path.stat()
    path.write_bytes(b"X" * 4096)  # same size; mtime put back (rsync --inplace -t)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert fv.digests([path], mode=fv.FAST)[os.fspath(path)] != record["file_sha256"]["a.safetensors"]


@pytest.mark.parametrize("mode", [fv.FULL, fv.FAST])
def test_full_and_fast_refuse_truncation_and_swapped_files(tmp_path, mode):
    store, rel, record = _release(tmp_path)
    manifest = cm.make(store, "r2-v2", record=record, mode=mode)
    cm.verify(manifest, rel, mode=fv.FAST)  # warm the cache with the good files
    a, b = rel / "a.safetensors", rel / "b.safetensors"
    os.replace(a, rel / "tmp")
    os.replace(b, a)
    os.replace(rel / "tmp", b)  # same sizes, names swapped: misplacement
    with pytest.raises(cm.ManifestError, match="sha256 mismatch"):
        cm.verify(manifest, rel, mode=mode)
    with open(a, "r+b") as fh:
        fh.truncate(100)
    with pytest.raises(cm.ManifestError, match="size"):
        cm.verify(manifest, rel, mode=mode)


def test_off_checks_sizes_only(tmp_path):
    store, rel, record = _release(tmp_path)
    manifest = cm.make(store, "r2-v2", record=record)
    (rel / "a.safetensors").write_bytes(b"Z" * 4096)
    cm.verify(manifest, rel, mode=fv.OFF)  # same size: not caught by design
    (rel / "a.safetensors").write_bytes(b"Z" * 10)
    with pytest.raises(cm.ManifestError, match="size"):
        cm.verify(manifest, rel, mode=fv.OFF)


def test_fast_mirror_reuses_the_trainer_digest_full_rehashes(tmp_path, monkeypatch):
    store, rel, record = _release(tmp_path)
    calls = _count_hashes(monkeypatch)
    fast = cm.make(store, "r2-v2", record=record, mode=fv.FAST)
    assert calls == []  # every file carried its trainer digest
    full = cm.make(store, "r2-v2", record=record, mode=fv.FULL)
    assert len(calls) == 2 and fast == full


def test_fast_mirror_with_a_copy_damaged_before_it_is_refused_at_placement(tmp_path):
    store, rel, record = _release(tmp_path)
    (rel / "a.safetensors").write_bytes(b"Q" * 4096)  # store copy differs from what the trainer hashed
    manifest = cm.make(store, "r2-v2", record=record, mode=fv.FAST)
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in ("a.safetensors", "b.safetensors"):
        (staged / name).write_bytes((rel / name).read_bytes())
    with pytest.raises(cm.ManifestError, match="sha256 mismatch"):
        cm.place(manifest, staged, tmp_path / "far", expected_digest=cm.digest(manifest))


def test_sampler_check_after_fast_placement_does_not_rehash(tmp_path, monkeypatch):
    store, rel, record = _release(tmp_path)
    monkeypatch.setenv(fv.ENV_MODE, fv.FAST)
    manifest = cm.make(store, "r2-v2", record=record)
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in ("a.safetensors", "b.safetensors"):
        (staged / name).write_bytes((rel / name).read_bytes())
    calls = _count_hashes(monkeypatch)
    placed = cm.place(manifest, staged, tmp_path / "far", expected_digest=cm.digest(manifest))
    assert len(calls) == 2
    release = pr.PolicyRelease.from_record(record)
    pr.check_release_files(release, placed)
    assert len(calls) == 2  # the sampler reused the placement's digests
    pr.check_release_files(release, placed, mode=fv.FULL)
    assert len(calls) == 4


def test_sampler_check_refuses_a_bad_file_in_every_hashing_mode(tmp_path):
    _, rel, record = _release(tmp_path)
    release = pr.PolicyRelease.from_record(record)
    pr.check_release_files(release, rel, mode=fv.FAST)
    (rel / "b.safetensors").write_bytes(b"C" * 4096)
    for mode in (fv.FULL, fv.FAST):
        with pytest.raises(pr.ReleaseDigestMismatch):
            pr.check_release_files(release, rel, mode=mode)
    pr.check_release_files(release, rel, mode=fv.OFF)


def test_cut_file_problems_modes(tmp_path, monkeypatch):
    d = tmp_path / "cut"
    d.mkdir()
    files = []
    for i in range(4):
        p = d / f"shard{i}.pt"
        p.write_bytes(bytes([i]) * 2048)
        files.append(cut_mod.CutFile(path=p.name, sha256=fv.sha256_file(p, remember=False), bytes=2048))
    calls = _count_hashes(monkeypatch)
    assert cut_mod.file_problems(d, files) == [] and len(calls) == 4  # default full
    assert cut_mod.file_problems(d, files, mode=fv.FAST) == []
    assert cut_mod.file_problems(d, files, mode=fv.FAST) == [] and len(calls) == 8
    (d / "shard1.pt").write_bytes(b"\xff" * 2048)
    (d / "shard2.pt").write_bytes(b"\x02" * 10)
    for mode in (fv.FULL, fv.FAST):
        problems = cut_mod.file_problems(d, files, mode=mode)
        assert any("shard1.pt checksum mismatch" in p for p in problems)
        assert any("shard2.pt is 10 bytes" in p for p in problems)
    assert [p for p in cut_mod.file_problems(d, files, mode=fv.OFF)] == [
        "shard shard2.pt is 10 bytes, manifest says 2048 (truncated?)"]
