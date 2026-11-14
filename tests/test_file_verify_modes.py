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
    monkeypatch.delenv(fv.ENV_WORKERS, raising=False)
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


@pytest.mark.parametrize("raw", ["fastest", "1", "true"])
def test_unknown_mode_warns_and_falls_back_to_full(raw, caplog):
    with caplog.at_level("WARNING", logger=fv.__name__):
        assert fv.verify_mode({fv.ENV_MODE: raw}) == fv.FULL
    assert fv.ENV_MODE in caplog.text


def test_yeto_verify_1_does_not_break_the_default_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(fv.ENV_MODE, "1")
    p = tmp_path / "s.bin"
    p.write_bytes(b"x" * 100)
    sha = cut_mod.sha256_file(p)
    assert cut_mod.file_problems(tmp_path, [cut_mod.CutFile(path="s.bin", sha256=sha, bytes=100)]) == []
    cm.verify({"files": {"s.bin": {"size": 100, "sha256": sha}}}, tmp_path)


def test_workers_are_opt_in():
    assert fv.hash_workers({}) == 1
    assert fv.hash_workers({fv.ENV_WORKERS: "4"}) == 4
    assert fv.hash_workers({}, default=pr.DIGEST_WORKERS) == pr.DIGEST_WORKERS


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


def _mixed(tmp_path):
    """s1: wrong sha, zz: missing, s4: wrong size, ok: fine (main's own outputs below)."""
    d = tmp_path / "mixed"
    d.mkdir()
    for name in ("s1.bin", "s4.bin", "ok.bin"):
        (d / name).write_bytes(name.encode() * 64)
    good = {n: fv.sha256_file(d / n, remember=False) for n in ("s1.bin", "s4.bin", "ok.bin")}
    (d / "s1.bin").write_bytes(b"?" * len("s1.bin") * 64)
    (d / "s4.bin").write_bytes(b"short")
    return d, good


@pytest.mark.parametrize("workers", [None, "4"])
def test_file_problems_order_and_wording_match_main(tmp_path, monkeypatch, workers):
    if workers:
        monkeypatch.setenv(fv.ENV_WORKERS, workers)
    d, good = _mixed(tmp_path)
    files = [cut_mod.CutFile(path="s1.bin", sha256=good["s1.bin"], bytes=384),
             cut_mod.CutFile(path="zz.bin", sha256="0" * 64, bytes=1),
             cut_mod.CutFile(path="s4.bin", sha256=good["s4.bin"], bytes=384),
             cut_mod.CutFile(path="ok.bin", sha256=good["ok.bin"], bytes=384)]
    assert cut_mod.file_problems(d, files) == [
        "shard s1.bin checksum mismatch",
        "missing shard zz.bin",
        "shard s4.bin is 5 bytes, manifest says 384 (truncated?)",
    ]


def test_verify_default_reports_the_first_bad_file_like_main(tmp_path):
    d, good = _mixed(tmp_path)
    sha_first = {"files": {"s1.bin": {"size": 384, "sha256": good["s1.bin"]},
                           "s4.bin": {"size": 384, "sha256": good["s4.bin"]},
                           "ok.bin": {"size": 384, "sha256": good["ok.bin"]}}}
    with pytest.raises(cm.ManifestError) as err:
        cm.verify(sha_first, d)
    assert str(err.value) == "s1.bin: sha256 mismatch"
    size_first = {"files": dict(reversed(list(sha_first["files"].items())))}
    with pytest.raises(cm.ManifestError) as err:
        cm.verify(size_first, d)
    assert str(err.value) == "s4.bin: size 5 != 384"


def test_verify_parallel_opt_in_checks_sizes_first_then_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv(fv.ENV_WORKERS, "4")
    d, good = _mixed(tmp_path)
    files = {"s1.bin": {"size": 384, "sha256": good["s1.bin"]},
             "s4.bin": {"size": 384, "sha256": good["s4.bin"]},
             "ok.bin": {"size": 384, "sha256": good["ok.bin"]}}
    with pytest.raises(cm.ManifestError, match=r"^s4\.bin: size 5 != 384$"):
        cm.verify({"files": files}, d)
    (d / "s4.bin").unlink()
    del files["s4.bin"]
    with pytest.raises(cm.ManifestError, match=r"^s1\.bin: sha256 mismatch$"):
        cm.verify({"files": files}, d)


def test_parallel_digests_stop_early_on_error(tmp_path, monkeypatch):
    paths = []
    for i in range(6):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(bytes([i]) * (4 * fv.CHUNK))
        paths.append(p)
    read_blocks = []
    real = fv.sha256_file

    def slow(path, **kw):
        if os.fspath(path).endswith("f0.bin"):
            raise OSError("boom")
        read_blocks.append(path)
        return real(path, **kw)

    monkeypatch.setattr(fv, "sha256_file", slow)
    with pytest.raises(OSError, match="boom"):
        fv.digests(paths, mode=fv.FULL, workers=2)
    assert len(read_blocks) < len(paths) - 1  # queued files were cancelled


def test_parallel_digests_stop_on_first_mismatch(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(bytes([i]) * 1024)
        paths.append(p)
    expected = {os.fspath(p): fv.sha256_file(p, remember=False) for p in paths}
    expected[os.fspath(paths[0])] = "0" * 64
    got = fv.digests(paths, mode=fv.FULL, workers=2, expected=expected)
    assert got[os.fspath(paths[0])] != "0" * 64
