"""A release handed to a spot island through a near store (cut_manifest)."""
import errno
import json
import os
import stat
import time
import shutil

import pytest

from yeto.rl.spot_rollout import cut_manifest as cm


def _store(tmp_path):
    root = tmp_path / "store" / "policy-releases"
    (root / "r1-v0").mkdir(parents=True)
    (root / "r1-v0" / "model.safetensors").write_bytes(b"base")
    rel = root / "r2-v2"
    rel.mkdir()
    (rel / "model.safetensors.ydelta").write_bytes(b"delta" * 1000)
    (rel / "config.json").write_text("{}")
    record = {"path": "r2-v2", "version": 2, "seq": 2, "delta": {"base_path": "r1-v0"},
              "files": {"model.safetensors.ydelta": 5000, "config.json": 2}}
    (root / "LATEST.json").write_text(json.dumps(record))
    return tmp_path / "store", record


def _far(tmp_path):
    far = tmp_path / "far" / "policy-releases" / "r1-v0"
    far.mkdir(parents=True)
    (far / "model.safetensors").write_bytes(b"base")
    return tmp_path / "far"


def test_make_place_round_trip(tmp_path):
    store, record = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    assert manifest["version"] == 2 and manifest["base"] == "r1-v0" and manifest["bytes"] == 5002
    sha = cm.digest(json.loads(cm.encode(manifest)))
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    far = _far(tmp_path)
    target = cm.place(manifest, staged, far, expected_digest=sha)
    assert (target / "model.safetensors.ydelta").read_bytes() == b"delta" * 1000
    assert json.loads((far / "policy-releases" / "LATEST.json").read_text()) == record
    # idempotent: placing again keeps the release (LATEST already names it)
    assert cm.place(manifest, tmp_path / "gone", far, expected_digest=sha) == target


def test_a_corrupt_file_or_a_wrong_digest_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    (staged / "model.safetensors.ydelta").write_bytes(b"x" * 5000)  # same size, other bytes
    far = _far(tmp_path)
    with pytest.raises(cm.ManifestError, match="sha256"):
        cm.place(manifest, staged, far, expected_digest=cm.digest(manifest))
    with pytest.raises(cm.ManifestError, match="digest"):
        cm.place(manifest, staged, far, expected_digest="0" * 64)
    assert not (far / "policy-releases" / "LATEST.json").exists()


def test_missing_base_or_wrong_latest_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    with pytest.raises(cm.ManifestError, match="delta base"):
        cm.place(manifest, tmp_path, tmp_path / "empty", expected_digest=None)
    with pytest.raises(cm.ManifestError, match="names"):
        cm.make(store, "r1-v0")


def test_cli(tmp_path, capsys):
    store, _ = _store(tmp_path)
    out = tmp_path / "m.json"
    assert cm.main(["make", "--store", str(store), "--release", "r2-v2", "--out", str(out)]) == 0
    sha = capsys.readouterr().out.split()[0]
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    far = _far(tmp_path)
    assert cm.main(["place", "--manifest", str(out), "--digest", sha, "--staged", str(staged),
                    "--store", str(far)]) == 0
    assert cm.main(["place", "--manifest", str(out), "--digest", "1" * 64, "--staged", str(staged),
                    "--store", str(far)]) == 2


def test_placing_an_older_manifest_again_keeps_the_newer_latest(tmp_path):
    store, record = _store(tmp_path)
    old = cm.make(store, "r2-v2")
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    far = _far(tmp_path)
    cm.place(old, staged, far, expected_digest=None)
    root = store / "policy-releases"
    (root / "r3-v3").mkdir()
    (root / "r3-v3" / "config.json").write_text("{}")
    newer = {"path": "r3-v3", "version": 3, "seq": 3, "files": {"config.json": 2}}
    (root / "LATEST.json").write_text(json.dumps(newer))
    staged3 = tmp_path / "staged3"
    shutil.copytree(root / "r3-v3", staged3)
    cm.place(cm.make(store, "r3-v3"), staged3, far, expected_digest=None)
    latest = far / "policy-releases" / "LATEST.json"
    assert json.loads(latest.read_text()) == newer
    # the idempotent branch with the older manifest: the release is kept, LATEST is not rolled back
    assert cm.place(old, tmp_path / "gone", far, expected_digest=None) == far / "policy-releases" / "r2-v2"
    assert json.loads(latest.read_text()) == newer
    assert record["seq"] == 2


@pytest.mark.parametrize("name", ["../evil", "a/b", "..", ".", "", "x\\y", "r2..v2"])
def test_a_release_name_with_a_path_is_refused(tmp_path, name):
    store, _ = _store(tmp_path)
    manifest = dict(cm.make(store, "r2-v2"), release=name)
    with pytest.raises(cm.ManifestError, match="plain release directory name"):
        cm.place(manifest, tmp_path / "staged", _far(tmp_path), expected_digest=None)
    assert not (tmp_path / "far" / "policy-releases" / "LATEST.json").exists()


def _placed_inputs(tmp_path):
    store, _ = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    return manifest, staged, _far(tmp_path)


def test_every_file_and_directory_is_fsynced_before_the_renames(tmp_path, monkeypatch):
    """A power cut must never leave a renamed release (or LATEST) over unsynced bytes."""
    manifest, staged, far = _placed_inputs(tmp_path)
    log, real_fsync, real_replace = [], cm.os.fsync, cm.os.replace

    def fsync(fd):
        log.append(("fsync", fd, os.fstat(fd).st_ino))
        real_fsync(fd)

    def replace(src, dst):
        log.append(("replace", str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(cm.os, "fsync", fsync)
    monkeypatch.setattr(cm.os, "replace", replace)
    target = cm.place(manifest, staged, far, expected_digest=cm.digest(manifest))
    root = far / "policy-releases"
    synced = lambda upto: {entry[2] for entry in log[:upto] if entry[0] == "fsync"}  # noqa: E731
    first = next(i for i, e in enumerate(log) if e[0] == "replace" and e[1] == str(target))
    need = {p.stat().st_ino for p in [target, *target.iterdir()]}
    assert need <= synced(first), "release files/dir not fsynced before the rename"
    latest = next(i for i, e in enumerate(log) if e[0] == "replace" and e[1] == str(root / "LATEST.json"))
    assert (root / "LATEST.json").stat().st_ino in synced(latest)
    assert root.stat().st_ino in synced(latest)  # the release rename is durable before LATEST names it
    assert root.stat().st_ino in synced(len(log)) and log[-1][0] == "fsync"  # and LATEST's rename


def test_temporary_names_are_unique_per_writer(tmp_path, monkeypatch):
    """Two writers placing at once must not share (and so clobber) a temporary path."""
    manifest, staged, far = _placed_inputs(tmp_path)
    root = far / "policy-releases"
    other = root / ".r2-v2.placing"  # what another live writer's staging used to be named
    other.mkdir()
    (other / "half").write_bytes(b"x")
    names, real_replace = [], cm.os.replace
    monkeypatch.setattr(cm.os, "replace", lambda s, d: (names.append(os.path.basename(s)), real_replace(s, d)))
    cm.place(manifest, staged, far, expected_digest=cm.digest(manifest))
    assert (other / "half").read_bytes() == b"x"  # never removed under the other writer
    assert all(str(os.getpid()) in name for name in names) and ".LATEST.json.tmp" not in names
    assert sorted(p.name for p in root.iterdir() if not p.name.startswith(".")) == ["LATEST.json", "r1-v0", "r2-v2"]


def test_losing_a_race_to_place_the_same_release_keeps_the_winner(tmp_path, monkeypatch):
    manifest, staged, far = _placed_inputs(tmp_path)
    winner = tmp_path / "winner"
    shutil.copytree(staged, winner)
    target = far / "policy-releases" / "r2-v2"
    real_replace = cm.os.replace

    def replace(src, dst):
        if str(dst) == str(target) and not target.exists():  # the other writer gets there first
            shutil.copytree(winner, target)
        real_replace(src, dst)

    monkeypatch.setattr(cm.os, "replace", replace)
    assert cm.place(manifest, staged, far, expected_digest=cm.digest(manifest)) == target
    assert sorted(p.name for p in target.parent.iterdir()) == ["LATEST.json", "r1-v0", "r2-v2"]


def test_stale_staging_is_swept_by_age_not_pid(tmp_path):
    """A pid is meaningless across containers/hosts sharing the store: a recent staging
    dir is kept even if its pid is not alive here; an old one (or a pre-unique
    ``.<release>.placing``) is removed."""
    manifest, staged, far = _placed_inputs(tmp_path)
    root = far / "policy-releases"
    recent = root / ".r2-v2.placing.999999999.deadbeef"
    old_unique = root / ".r2-v2.placing.1.cafe"
    legacy = root / ".r2-v2.placing"
    other_release = root / ".r3-v3.placing.1.cafe"
    for d in (recent, old_unique, legacy, other_release):
        d.mkdir()
    stale = time.time() - cm.STALE_STAGING_SECONDS - 60
    for d in (old_unique, legacy, other_release):
        os.utime(d, (stale, stale))
    cm.place(manifest, staged, far, expected_digest=cm.digest(manifest))
    assert recent.exists() and other_release.exists()
    assert not old_unique.exists() and not legacy.exists()


def test_a_corrupt_release_that_won_the_rename_race_is_replaced(tmp_path, monkeypatch):
    manifest, staged, far = _placed_inputs(tmp_path)
    target = far / "policy-releases" / "r2-v2"
    real_replace, raced = cm.os.replace, []

    def replace(src, dst):
        if str(dst) == str(target) and not raced:  # a writer that placed bad bytes first
            raced.append(True)
            target.mkdir()
            (target / "model.safetensors.ydelta").write_bytes(b"x" * 5000)
            (target / "config.json").write_text("{}")
        real_replace(src, dst)

    monkeypatch.setattr(cm.os, "replace", replace)
    assert cm.place(manifest, staged, far, expected_digest=cm.digest(manifest)) == target
    assert (target / "model.safetensors.ydelta").read_bytes() == b"delta" * 1000
    assert sorted(p.name for p in target.parent.iterdir()) == ["LATEST.json", "r1-v0", "r2-v2"]


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP])
def test_a_filesystem_refusing_directory_fsync_is_tolerated(tmp_path, monkeypatch, code):
    manifest, staged, far = _placed_inputs(tmp_path)
    real_fsync = cm.os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        real_fsync(fd)

    monkeypatch.setattr(cm.os, "fsync", fsync)
    target = cm.place(manifest, staged, far, expected_digest=cm.digest(manifest))
    assert (target / "config.json").read_text() == "{}"


def test_other_directory_fsync_errors_still_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(cm.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "io")))
    with pytest.raises(OSError):
        cm._fsync_dir(tmp_path)


def test_directory_fsync_is_skipped_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "_WINDOWS", True)
    monkeypatch.setattr(cm.os, "fsync", lambda fd: pytest.fail("a directory cannot be opened on Windows"))
    cm._fsync_dir(tmp_path)
