"""ModalVolumeStore puts through the container's mount (S27 G-DR).

A Modal island used to put HF exports and staged policy releases with
``Volume.batch_upload``, which Modal bills as network egress (S27 G-DE: about
$2.79 per 63 GB export). When the Volume is mounted in the container the store
now writes through the mount (temporary names, then rename) and commits once,
under the writer guard; a reader in another container sees the files after its
reload, and never sees a partial file under its final name."""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.cloud import modal_ckpt_store as mcs
from yeto.cloud.modal_ckpt_store import PUT_MOUNT, PUT_UPLOAD, ModalVolumeStore, store_from_env, volume_mounted
from yeto.rl.engine import resume
from yeto.rl.engine.hf_export import COMPLETE_MARKER, FINAL_POINTER, ExportRounds, HfRoundExports
from yeto.rl.engine.policy_release import RELEASE_DIR, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore


@pytest.fixture(autouse=True)
def _guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(mcs.GUARD_DIR_ENV, str(tmp_path / "guards"))
    (tmp_path / "guards").mkdir()
    monkeypatch.delenv("YETO_MODAL_PYTHON", raising=False)
    monkeypatch.delenv(mcs.PUT_ENV, raising=False)


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _tree(root: Path) -> dict[str, bytes | None]:
    """{relative path: bytes, or None for a directory}; no shutil (tests patch copyfile)."""
    return {p.relative_to(root).as_posix(): None if p.is_dir() else p.read_bytes() for p in sorted(root.rglob("*"))}


def _write_tree(root: Path, tree: dict[str, bytes | None]) -> None:
    for rel, data in tree.items():
        target = root / rel
        if data is None:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


class _Container:
    """One container's view of a Modal Volume (v1 semantics, simplified): writes
    go to ``mount``; ``commit`` sends this container's changes since its last
    reload or commit to ``backend`` (the durable Volume); ``reload`` replaces
    the view with ``backend``. ``batch_upload`` / ``remove_file`` must not be
    used by a mount put."""

    def __init__(self, backend: Path, mount: Path) -> None:
        self.backend, self.mount = backend, mount
        backend.mkdir(parents=True, exist_ok=True)
        self.commits = 0
        self.reloads = 0
        self.uploads = 0
        self.reload()
        self.reloads = 0

    def commit(self) -> None:
        mine = _tree(self.mount)
        for rel in sorted(set(self._base) - set(mine), reverse=True):  # the deepest first
            target = self.backend / rel
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        _write_tree(self.backend, {rel: data for rel, data in mine.items()
                                   if rel not in self._base or self._base[rel] != data})
        self._base = mine
        self.commits += 1

    def reload(self) -> None:
        shutil.rmtree(self.mount, ignore_errors=True)
        self.mount.mkdir(parents=True)
        _write_tree(self.mount, _tree(self.backend))
        self._base = _tree(self.mount)
        self.reloads += 1

    def batch_upload(self, force=False):
        self.uploads += 1
        raise AssertionError("a mount put does not upload")

    def remove_file(self, path, recursive=False):
        raise AssertionError("a mount put does not remove by API")


def _store(container: _Container, sub: str, **kwargs) -> ModalVolumeStore:
    kwargs.setdefault("mounted", True)
    return ModalVolumeStore(container.mount / sub, "vol", volume_factory=lambda name: container,
                            mount=container.mount, **kwargs)


def _two_containers(tmp_path):
    backend = tmp_path / "volume"
    return _Container(backend, tmp_path / "writer-mount"), _Container(backend, tmp_path / "reader-mount")


def _write_model(rollout_id: int, directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"r": %d}' % rollout_id)
    (directory / "model-00001-of-00002.safetensors").write_bytes(b"a" * 4096)
    (directory / "model-00002-of-00002.safetensors").write_bytes(b"b" * 3000)
    (directory / "sub").mkdir()
    (directory / "sub" / "tokenizer.json").write_text("{}")
    (directory / "empty").mkdir()
    (directory / COMPLETE_MARKER).write_text("{}")  # Miles writes it; the export takes it out


# ---------------------------------------------------------------- which path a put takes
def test_volume_mounted_reads_the_mount_table(tmp_path):
    root = tmp_path.resolve()
    for name in ("yeto-checkpoint-store/run-a", "with space", "other"):
        (root / name).mkdir(parents=True)
    table = tmp_path / "mountinfo"
    escaped = str(root / "with space").replace(" ", "\\040")
    table.write_text(
        "22 1 0:21 / / rw - overlay overlay rw\n"
        f"640 22 0:88 / {root}/yeto-checkpoint-store rw,relatime - fuse modal rw\n"
        f"641 22 0:89 / {escaped} rw - fuse modal rw\n")
    assert volume_mounted(root / "yeto-checkpoint-store", mountinfo=str(table))
    assert volume_mounted(root / "with space", mountinfo=str(table))
    assert not volume_mounted(root / "yeto-checkpoint-store" / "run-a", mountinfo=str(table))
    assert not volume_mounted(root / "other", mountinfo=str(table))
    assert not volume_mounted(root / "missing", mountinfo=str(table))  # nothing there: no mount
    # no mount table: os.path.ismount decides
    assert volume_mounted("/", mountinfo=str(tmp_path / "missing"))
    assert not volume_mounted(str(tmp_path), mountinfo=str(tmp_path / "missing"))


def test_volume_mounted_knows_the_modal_container_layout(tmp_path, monkeypatch):
    """In a Modal container the mount path is a symlink to /__modal/volumes/<id>
    and is in no line of the mount table (S27 G-DR, yeto-gdr-3 diag.json)."""
    volumes = tmp_path.resolve() / "__modal" / "volumes"
    (volumes / "vo-kpBXryWNF9bENmb9AkP2yv").mkdir(parents=True)
    monkeypatch.setattr(mcs, "MODAL_VOLUMES_DIR", str(volumes))
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "yeto-checkpoint-store").symlink_to(volumes / "vo-kpBXryWNF9bENmb9AkP2yv")
    table = tmp_path / "mountinfo"
    table.write_text(f"36 18 0:31 / {volumes} rw,nosuid - 9p none rw\n")
    assert volume_mounted(tmp_path / "root" / "yeto-checkpoint-store", mountinfo=str(table))
    (tmp_path / "root" / "plain").mkdir()  # not mounted: a directory of the container's own disk
    assert not volume_mounted(tmp_path / "root" / "plain", mountinfo=str(table))
    assert not volume_mounted(tmp_path / "root" / "missing", mountinfo=str(table))
    (tmp_path / "root" / "dangling").symlink_to(volumes / "vo-gone")
    assert not volume_mounted(tmp_path / "root" / "dangling", mountinfo=str(table))


def test_the_put_path_is_the_mount_when_the_volume_is_mounted(tmp_path, monkeypatch):
    volume = SimpleNamespace()
    mount = tmp_path / "mount"

    def store(root=mount / "run", factory=lambda n: volume, **kwargs):
        return ModalVolumeStore(root, "vol", volume_factory=factory, mount=mount, **kwargs)

    assert store(mounted=True).put_path == PUT_MOUNT
    assert not store(mounted=True).puts_bypass_mount
    assert store(mounted=False).put_path == PUT_UPLOAD  # the API is the only way to the Volume
    assert store(mounted=False).puts_bypass_mount
    assert store(mounted=True, put_mode="upload").put_path == PUT_UPLOAD  # asked for
    assert store(mounted=True, put_mode=" Mount ").put_path == PUT_MOUNT
    assert store(root=tmp_path / "elsewhere", mounted=False).put_path == PUT_MOUNT  # not in the Volume
    assert store(factory=None, mounted=False).put_path == PUT_MOUNT  # no Modal client: as before
    monkeypatch.setenv("YETO_MODAL_PYTHON", "/bin/false")  # the island: the runner interpreter
    assert store(factory=None, mounted=False).put_path == PUT_UPLOAD
    assert store(factory=None, mounted=True).put_path == PUT_MOUNT
    assert store(factory=None).put_path == PUT_UPLOAD  # tmp_path/mount is no mount point
    with pytest.raises(ValueError, match=mcs.PUT_ENV):
        store(put_mode="batch")


def test_store_from_env_reads_the_put_mode(monkeypatch):
    env = {"YETO_RL_STORE_MODAL_VOLUME": "v"}
    assert store_from_env("/root/yeto-checkpoint-store/run", environ=env).put_mode == PUT_MOUNT
    built = store_from_env("/root/yeto-checkpoint-store/run", environ={**env, mcs.PUT_ENV: "upload"})
    assert built.put_mode == PUT_UPLOAD
    with pytest.raises(ValueError):
        store_from_env("/root/yeto-checkpoint-store/run", environ={**env, mcs.PUT_ENV: "nope"})


# ---------------------------------------------------------------- HF exports through the mount
def test_an_hf_export_goes_through_the_mount_and_a_reader_sees_it_after_its_reload(tmp_path):
    writer, reader = _two_containers(tmp_path)
    store = _store(writer, "run-a/hf/island0")
    exports = HfRoundExports(ExportRounds((2,), final=True), store, _write_model, staging=tmp_path / "staging",
                             sleep=lambda s: None)
    record = exports.export(1, policy_version=2, final=False)
    assert record["ok"] and record["store_put"] == PUT_MOUNT and record["upload_attempts"] == 1, record
    assert writer.uploads == 0 and writer.commits == 2  # the directory, then its marker (nothing to remove)
    exported = reader.mount / "run-a" / "hf" / "island0" / "round-2"
    assert not exported.exists()  # the reader's view is older than the commit
    reader.reload()
    assert sorted(_files(exported)) == [COMPLETE_MARKER, "config.json", "model-00001-of-00002.safetensors",
                                        "model-00002-of-00002.safetensors", "sub/tokenizer.json"]
    assert (exported / "empty").is_dir()
    assert _files(exported)["model-00001-of-00002.safetensors"] == b"a" * 4096
    assert json.loads((exported / COMPLETE_MARKER).read_text())["bytes"] == record["bytes"] == 4096 + 3000 + 8 + 2
    assert not [p for p in writer.mount.rglob("*.tmp")]  # every temporary name was renamed
    # a relaunch exports the round again: the older directory goes first (one more commit)
    final = exports.export(1, policy_version=2, final=True)
    assert final["ok"] and writer.commits == 5
    reader.reload()
    assert json.loads((reader.mount / "run-a" / "hf" / "island0" / FINAL_POINTER).read_text())["path"] == "round-2"


def test_a_background_commit_never_shows_a_partial_file_under_its_final_name(tmp_path, monkeypatch):
    writer, reader = _two_containers(tmp_path)
    store = _store(writer, "run-a/hf/island0")
    seen = []
    copy = shutil.copyfile
    monkeypatch.setattr(resume, "PUT_COPY_WORKERS", 1)  # one copy at a time: the fake commits are not thread safe

    def copy_then_background_commit(src, dst, *args, **kwargs):
        with open(dst, "wb") as handle:  # half of the file, then a Modal background commit
            handle.write(Path(src).read_bytes()[: max(1, os.path.getsize(src) // 2)])
        writer.commit()
        reader.reload()
        seen.append(_files(reader.mount))
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(resume.shutil, "copyfile", copy_then_background_commit)
    record = HfRoundExports(ExportRounds((1,)), store, _write_model, staging=tmp_path / "staging").export(
        0, policy_version=1, final=False)
    assert record["ok"], record
    monkeypatch.setattr(resume.shutil, "copyfile", copy)
    sources = {"config.json": b'{"r": 0}', "model-00001-of-00002.safetensors": b"a" * 4096,
               "model-00002-of-00002.safetensors": b"b" * 3000, "sub/tokenizer.json": b"{}"}
    assert len(seen) == 4
    for view in seen:
        for rel, data in view.items():
            name = rel.rsplit("/", 1)[-1]
            if name.endswith(".tmp"):
                continue  # a temporary name: no reader takes it
            assert data == sources[rel.split("round-1/", 1)[1]], rel  # complete, or not there at all
            assert not rel.endswith(COMPLETE_MARKER)  # the marker comes after the directory's commit


def test_a_reload_by_another_process_waits_out_the_put(tmp_path, monkeypatch):
    """The writer guard is held from the first file to the commit: a reload of
    the same Volume by another store object (another process of the container)
    is skipped meanwhile, so it never drops the files not committed yet."""
    writer, _ = _two_containers(tmp_path)
    store = _store(writer, "run-a/hf/island0")
    other = _store(writer, "run-a/sample-buffer")  # e.g. the trainer's buffer reader
    copy = shutil.copyfile

    def copy_and_reload(src, dst, *args, **kwargs):
        other.reload()
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(resume.shutil, "copyfile", copy_and_reload)
    staged = tmp_path / "stage"
    _write_model(0, staged)
    store.put_directory(staged, "round-1")
    assert writer.reloads == 0 and writer.commits == 1
    other.reload()
    assert writer.reloads == 1
    assert (writer.mount / "run-a" / "hf" / "island0" / "round-1" / "config.json").read_text() == '{"r": 0}'


def test_a_put_is_skipped_by_no_reader_guard_but_the_readers_reload_is(tmp_path):
    """G-CO (S27-FIXES §2): a reader holds the shared guard while its file is
    open. A put shares it (both are readers of the lock), so a put never waits
    for a reader; a reload waits for both."""
    writer, _ = _two_containers(tmp_path)
    store = _store(writer, "run-a/policy-releases")
    if not hasattr(store, "hold_reader"):  # s27-verl-megatron took ed981c36 without G-CO's c69b3307
        pytest.skip("needs the reader guard of s27-fixes c69b3307 (not on this branch)")
    reading = store.hold_reader()
    try:
        staged = tmp_path / "stage"
        _write_model(0, staged)
        store.put_directory(staged, "r1-v1")
        store.reload()
    finally:
        reading.release()
    assert writer.commits == 1 and writer.reloads == 0
    store.reload()
    assert writer.reloads == 1


def test_the_commit_of_a_put_may_take_as_long_as_its_bytes_need(tmp_path, monkeypatch):
    budgets = []
    writer, _ = _two_containers(tmp_path)
    store = _store(writer, "run-a/hf/island0")
    commit = writer.commit

    def commit_with_budget():
        budgets.append(store._commit_budgets().get(threading.get_ident()))
        commit()

    monkeypatch.setattr(writer, "commit", commit_with_budget)
    staged = tmp_path / "stage"
    _write_model(0, staged)
    store.put_directory(staged, "round-1")
    assert budgets == [4096 + 3000 + 8 + 2 + 2]  # every file of the directory (marker included)
    assert store._commit_budgets() == {}
    # the runner interpreter's commit gets the timeout of those bytes
    island = ModalVolumeStore(writer.mount / "run-a", "vol", mount=writer.mount, mounted=True)
    timeouts = []
    monkeypatch.setenv("YETO_MODAL_PYTHON", "/bin/false")
    monkeypatch.setattr(island, "_runner_script", lambda code, *a, verb, timeout_s: timeouts.append((verb, timeout_s)))
    monkeypatch.setattr(mcs, "COMMIT_MIN_BYTES_PER_S", 10.0)
    island.put_directory(staged, "round-2")
    assert timeouts == [("commit", max(mcs.COMMIT_TIMEOUT_S, (4096 + 3000 + 8 + 2 + 2) / 10.0))]


def test_the_upload_mode_still_uploads_and_leaves_the_mount_alone(tmp_path):
    from tests.test_rl_hf_export import _MissingPathVolume

    volume = _MissingPathVolume(tmp_path / "volume")
    mount = tmp_path / "mount"
    store = ModalVolumeStore(mount / "run-a" / "hf" / "island0", "vol", volume_factory=lambda n: volume,
                             mount=mount, mounted=True, put_mode="upload")
    record = HfRoundExports(ExportRounds((1,)), store, _write_model, staging=tmp_path / "staging").export(
        0, policy_version=1, final=False)
    assert record["ok"] and record["store_put"] == PUT_UPLOAD, record
    assert (tmp_path / "volume" / "run-a" / "hf" / "island0" / "round-1" / COMPLETE_MARKER).is_file()
    assert not mount.exists()


# ---------------------------------------------------------------- staged policy releases
def test_staged_releases_go_through_the_mount_and_prune_there(tmp_path):
    writer, reader = _two_containers(tmp_path)
    store = _store(writer, "run-a")
    releases = PolicyReleaseStore(store, keep=2)
    for version in (1, 2, 3):
        staged = tmp_path / f"stage-{version}"
        staged.mkdir()
        (staged / "model.safetensors").write_bytes(b"x" * (100 + version))
        commits = []
        record = releases.release_staged(version, f"h{version}", staged,
                                         position=lambda: commits.append(writer.commits) or (8 * version, None))
        assert record.seq == version and (staged / "model.safetensors").exists()
        assert releases.last_latest_source == ("reloaded" if version == 1 else "cached")
        assert commits == [writer.commits - 1 - (1 if version == 3 else 0)]  # weights committed first
    assert writer.uploads == 0
    root = writer.mount / "run-a" / RELEASE_DIR
    assert sorted(p.name for p in root.iterdir()) == ["LATEST.json", "r2-v2", "r3-v3"]
    seen = PolicyReleaseStore(_store(reader, "run-a"))
    latest = seen.latest()  # reloads the reader's container
    assert (latest.seq, latest.version, latest.buffer_selected) == (3, 3, 24)
    assert sorted(p.name for p in (reader.mount / "run-a" / RELEASE_DIR).iterdir()) == [
        "LATEST.json", "r2-v2", "r3-v3"]
    seen.resolve(latest)  # complete: every recorded file with its size
    assert not [p for p in reader.mount.rglob("*.tmp")]


# ---------------------------------------------------------------- the neutral store's put
def test_put_directory_replaces_the_target_and_keeps_the_source(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    staged = tmp_path / "stage"
    _write_model(3, staged)
    (store.root / "round-1").mkdir(parents=True)
    (store.root / "round-1" / "stale.bin").write_bytes(b"old")
    store.put_directory(staged, "round-1")
    assert _files(store.root / "round-1") == _files(staged)
    assert (store.root / "round-1" / "empty").is_dir()
    assert not [p for p in store.root.rglob("*.tmp")]


def test_a_failed_copy_raises_and_releases_the_writer_guard(tmp_path, monkeypatch):
    released = []

    class _Guard(resume.StoreWriterGuard):
        def release(self):
            released.append(1)

    store = CheckpointStore(tmp_path / "store")
    monkeypatch.setattr(store, "hold_writer", lambda: _Guard())
    staged = tmp_path / "stage"
    _write_model(0, staged)
    copy = shutil.copyfile

    def fail_on_one(src, dst, *args, **kwargs):
        if str(src).endswith("00002.safetensors"):
            raise OSError(28, "No space left on device")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(resume.shutil, "copyfile", fail_on_one)
    with pytest.raises(OSError, match="No space"):
        store.put_directory(staged, "round-1")
    assert released == [1]
    assert not (store.root / "round-1" / "model-00002-of-00002.safetensors").exists()


def test_a_bucket_mount_put_writes_the_final_names_straight(tmp_path, monkeypatch):
    """A rename on a bucket mount is a copy of the object: no temporary names
    there (an object shows only once its upload is whole)."""
    from yeto.cloud.s3_store import S3Store
    from yeto.rl.engine.resume import BucketMountStore

    written = []
    copy = shutil.copyfile
    monkeypatch.setattr(resume.shutil, "copyfile",
                        lambda src, dst, *a, **k: written.append(Path(dst).name) or copy(src, dst, *a, **k))
    staged = tmp_path / "stage"
    _write_model(0, staged)
    BucketMountStore(tmp_path / "bucket").put_directory(staged, "round-1")
    assert sorted(written) == sorted(p.name for p in staged.rglob("*") if p.is_file())
    assert _files(tmp_path / "bucket" / "round-1") == _files(staged)
    assert not S3Store(tmp_path / "s3", "s3://b/p", environ={}).put_renames
    assert S3Store(tmp_path / "s3", "s3://b/p", environ={}, stage=tmp_path / "stage-dir").put_renames


def test_a_missing_source_directory_puts_nothing(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    with pytest.raises(FileNotFoundError):
        store.put_directory(tmp_path / "no-such-stage", "round-1")
    assert not (store.root / "round-1").exists()
