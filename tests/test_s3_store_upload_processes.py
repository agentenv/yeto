"""Multi-process upload of a staged store's push (``S3ObjectStore.write_files``,
``TransferSettings.upload_processes``), the counterpart of ``read_files``."""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from tests.test_rl_ssh_trainer_island import b300, b300_spec
from yeto.cloud import s3_store
from yeto.launch import ssh_node_storage
from yeto.launch.ssh_trainer_island import remote_paths, start_script
from yeto.rl.engine.object_store import PUSHED_STATE, ObjectStoreError, TreeMirror


class DiskWriteS3:
    """A picklable client that writes ``root/<bucket>/<key>`` files (put and
    multipart), rebuilt in each upload process; ``pids`` records who wrote, and a
    key containing ``fail`` is refused."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, Bucket, Key):
        return self.root / Bucket / Key

    def _log(self):
        with open(self.root / "pids", "a") as fh:
            fh.write(f"{os.getpid()}\n")

    def put_object(self, Bucket, Key, Body, **kw):
        if "fail" in Key:
            raise ConnectionError(f"refused {Key}")
        self._log()
        p = self._path(Bucket, Key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(Body))
        return {"ETag": '"' + hashlib.md5(bytes(Body)).hexdigest() + '"'}

    def create_multipart_upload(self, Bucket, Key, **kw):
        if "fail" in Key:
            raise ConnectionError(f"refused {Key}")
        upload = hashlib.sha256(f"{Key}{os.getpid()}".encode()).hexdigest()[:16]
        (self.root / "uploads" / upload).mkdir(parents=True, exist_ok=True)
        return {"UploadId": upload}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, **kw):
        self._log()
        (self.root / "uploads" / UploadId / f"{PartNumber:05d}").write_bytes(bytes(Body))
        return {"ETag": '"' + hashlib.md5(bytes(Body)).hexdigest() + '"'}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload, **kw):
        parts = sorted((self.root / "uploads" / UploadId).iterdir())
        p = self._path(Bucket, Key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"".join(part.read_bytes() for part in parts))
        return {}

    def head_object(self, Bucket, Key, **kw):
        p = self._path(Bucket, Key)
        if not p.is_file():
            raise FileNotFoundError(Key)
        return {"ContentLength": p.stat().st_size}


def _objects(tmp_path, processes, **kw):
    factory = functools.partial(DiskWriteS3, str(tmp_path / "s3"))
    (tmp_path / "s3").mkdir(exist_ok=True)
    transfer = s3_store.TransferSettings(part_bytes=s3_store.MIN_PART_BYTES, concurrency=2, file_concurrency=4,
                                         upload_processes=processes, part_checksum=False, **kw)
    return s3_store.S3ObjectStore("s3://bkt/pfx", client=factory(), transfer=transfer, client_factory=factory)


def _release(root, sizes):
    files = {}
    for i, size in enumerate(sizes):
        p = Path(root) / "policy-releases" / "r1-v1" / f"model-{i:05d}.safetensors"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes([i % 251]) * size)
        files[p.relative_to(root).as_posix()] = p
    return files


def test_settings_read_the_upload_processes_and_ship_them_to_islands():
    assert s3_store.TransferSettings().upload_processes == 1  # unchanged one-process default
    assert s3_store.TransferSettings.from_env({s3_store.UPLOAD_PROCESSES_ENV: "6"}).upload_processes == 6
    assert s3_store.UPLOAD_PROCESSES_ENV in s3_store.SETTING_ENVS
    with pytest.raises(ValueError, match="upload processes"):
        s3_store.TransferSettings(upload_processes=0)


def test_split_by_size_balances_bytes_not_counts(tmp_path):
    items = []
    for name, size in (("big", 1000), ("a", 300), ("b", 300), ("c", 300), ("d", 100)):
        (tmp_path / name).write_bytes(b"x" * size)
        items.append((name, str(tmp_path / name)))
    groups = s3_store.split_by_size(items, 2)
    totals = sorted(sum(os.path.getsize(f) for _, f in g) for g in groups)
    assert totals == [1000, 1000] and sorted(k for g in groups for k, _ in g) == ["a", "b", "big", "c", "d"]
    assert len(s3_store.split_by_size(items[:1], 4)) == 1  # no empty group


def test_a_push_spreads_a_release_over_upload_processes(tmp_path):
    """A release tier goes up from upload_processes children, each with its own client;
    multipart files arrive whole and every file is marked pushed."""
    root = tmp_path / "stage"
    files = _release(root, [3 * s3_store.MIN_PART_BYTES + 11, 2 * s3_store.MIN_PART_BYTES, 1000, 2000, 3000, 7])
    objects = _objects(tmp_path, processes=3)
    mirror = TreeMirror(root, objects, subtrees=("policy-releases/",), pointers=frozenset({"LATEST.json"}),
                        independent=("policy-releases",))
    mirror.push(["policy-releases/r1-v1"])
    for rel, path in files.items():
        assert (tmp_path / "s3" / "bkt" / "pfx" / rel).read_bytes() == path.read_bytes()
    pids = set((tmp_path / "s3" / "pids").read_text().split())
    assert len(pids) == 3 and str(os.getpid()) not in pids
    import json

    assert set(json.loads((root / PUSHED_STATE).read_text())) == set(files)
    (tmp_path / "s3" / "pids").unlink()
    mirror.push(["policy-releases/r1-v1"])  # nothing changed: nothing sent again
    assert not (tmp_path / "s3" / "pids").exists()


def test_a_failed_file_is_reported_and_the_others_are_kept(tmp_path):
    root = tmp_path / "stage"
    _release(root, [1000, 2000])
    bad = root / "policy-releases" / "r1-v1" / "model-fail.safetensors"
    bad.write_bytes(b"z" * 10)
    objects = _objects(tmp_path, processes=2)
    written, errors = objects.write_files([(p.relative_to(root).as_posix(), p)
                                           for p in sorted((root / "policy-releases" / "r1-v1").iterdir())])
    assert sorted(errors) == ["policy-releases/r1-v1/model-fail.safetensors"]
    assert "ConnectionError" in next(iter(errors.values())) and len(written) == 2
    mirror = TreeMirror(root, objects, subtrees=(), pointers=frozenset(), independent=("policy-releases",))
    with pytest.raises(ObjectStoreError, match="1 of 3 files did not upload.*model-fail"):
        mirror.push(["policy-releases/r1-v1"])
    import json

    assert "policy-releases/r1-v1/model-fail.safetensors" not in json.loads((root / PUSHED_STATE).read_text())


def test_without_a_client_factory_the_upload_stays_in_this_process(tmp_path):
    root = tmp_path / "stage"
    _release(root, [100, 200, 300])
    objects = _objects(tmp_path, processes=4)
    objects.client_factory = None
    assert objects.upload_processes == 1
    TreeMirror(root, objects, subtrees=(), pointers=frozenset(), independent=("policy-releases",)).push(
        ["policy-releases/r1-v1"])
    assert set((tmp_path / "s3" / "pids").read_text().split()) == {str(os.getpid())}


def test_ssh_trainer_nodes_upload_a_staged_store_in_processes_unless_told_otherwise():
    staged = {s3_store.STAGE_ENV: s3_store.DEFAULT_STAGE}
    paths = remote_paths("k31", "t", 1)
    script = start_script(b300_spec(env=dict(staged)), 0, {}, paths, container="c0")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert f"export {s3_store.UPLOAD_PROCESSES_ENV}={ssh_node_storage.UPLOAD_PROCESSES}" in script
    own = start_script(b300_spec(env={**staged, s3_store.UPLOAD_PROCESSES_ENV: "8"}), 0, {}, paths, container="c0")
    assert f"export {s3_store.UPLOAD_PROCESSES_ENV}=8" in own
    pinned = b300(0, env={s3_store.UPLOAD_PROCESSES_ENV: "2"})
    assert ssh_node_storage.transfer_env(pinned, staged) == {}
    assert s3_store.UPLOAD_PROCESSES_ENV not in start_script(b300_spec(), 0, {}, paths, container="c0")
