"""The object API of the checkpoint store (CPU, fakes; no cloud resource):
the sample buffer, the trainer commands and the policy release pointer by object
key, for an ``s3://`` store whose bucket mount cannot append or rewrite."""

from __future__ import annotations

import json
import threading

import pytest
from botocore.exceptions import ClientError

from yeto.cloud import s3_store
from yeto.launch.credentials import SECRET_ENV_NAMES, split_secret_envs
from yeto.launch.storage import store_object_envs
from yeto.rl.engine.object_store import DirectoryObjectStore, ObjectStore, ObjectStoreError
from yeto.rl.engine.policy_release import LORA, PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore, store_for
from yeto.rl.engine.sample_buffer import FileSampleBuffer, ObjectIndex, SampleBufferError
from yeto.rl.pool_client import LeaveReason
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry
from yeto.rl.engine.trainer_remote import CommandReader, CommandWriter

URI = "s3://bkt/runs/a"
CREDS = {s3_store.KEY_ID_ENV: "AKIDSTORE", s3_store.SECRET_ENV: "store-secret"}


def _error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


class FakeS3:
    """The boto3 S3 client calls the store makes, over a dict. ``conditional``:
    False answers a conditional put with NotImplemented (an endpoint without it).
    Multipart uploads keep their parts until completed (``uploads``); ranged and
    ETag-pinned reads behave like S3's."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, *, conditional: bool = True) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.conditional = conditional
        self.lock = threading.Lock()
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.calls: list[str] = []

    @staticmethod
    def etag(data: bytes) -> str:
        import hashlib

        return '"' + hashlib.md5(data).hexdigest() + '"'

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        data = self.objects[(Bucket, Key)]
        if IfMatch is not None and IfMatch != self.etag(data):
            raise _error("PreconditionFailed", "GetObject")
        if Range is not None:
            first, last = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            data = data[first:last + 1]
        self.calls.append("get_object")
        return {"Body": type("Body", (), {"read": lambda self: data})()}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None):
        with self.lock:
            if IfNoneMatch is not None:
                if not self.conditional:
                    raise _error("NotImplemented", "PutObject")
                if (Bucket, Key) in self.objects:
                    raise _error("PreconditionFailed", "PutObject")
            self.objects[(Bucket, Key)] = bytes(Body)
            self.calls.append("put_object")

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _error("404", "HeadObject")
        data = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "ETag": self.etag(data)}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def create_multipart_upload(self, Bucket, Key):
        with self.lock:
            upload_id = f"u{len(self.uploads)}"
            self.uploads[upload_id] = {}
            self.calls.append("create_multipart_upload")
        return {"UploadId": upload_id}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        with self.lock:
            self.uploads[UploadId][PartNumber] = bytes(Body)
            self.calls.append("upload_part")
        return {"ETag": self.etag(bytes(Body))}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        parts = self.uploads.pop(UploadId)
        numbers = [p["PartNumber"] for p in MultipartUpload["Parts"]]
        assert numbers == sorted(parts) == list(range(1, len(parts) + 1))
        assert [p["ETag"] for p in MultipartUpload["Parts"]] == [self.etag(parts[n]) for n in numbers]
        self.objects[(Bucket, Key)] = b"".join(parts[n] for n in numbers)
        self.calls.append("complete_multipart_upload")

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.uploads.pop(UploadId, None)
        self.calls.append("abort_multipart_upload")

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix, StartAfter=""):
                keys = sorted(k for b, k in fake.objects if b == Bucket and k.startswith(Prefix) and k > StartAfter)
                for i in range(0, max(len(keys), 1), 2):  # two keys a page
                    yield {"Contents": [{"Key": k} for k in keys[i:i + 2]]} if keys else {}

        return Paginator()


def _stores(tmp_path):
    return [DirectoryObjectStore(tmp_path / "dir"), s3_store.S3ObjectStore(URI, client=FakeS3())]


# ---------------------------------------------------------------- object stores
@pytest.mark.parametrize("which", [0, 1], ids=["directory", "s3"])
def test_object_store_read_write_create_list(tmp_path, which):
    store = _stores(tmp_path)[which]
    assert isinstance(store, ObjectStore)
    assert store.read("a/b.json") is None and not store.exists("a/b.json")
    store.write("a/b.json", b"1")
    store.write("a/b.json", b"2")  # a whole-object rewrite
    assert store.read("a/b.json") == b"2" and store.exists("a/b.json")
    assert store.create("a/c.json", b"x") and not store.create("a/c.json", b"y")
    assert store.read("a/c.json") == b"x"
    store.write("a/d.json", b"")
    store.write("z.json", b"")
    assert store.list("a/") == ["a/b.json", "a/c.json", "a/d.json"]
    assert store.list("a/", start_after="a/b.json") == ["a/c.json", "a/d.json"]
    assert store.list("") == ["a/b.json", "a/c.json", "a/d.json", "z.json"]
    store.delete("a/c.json")
    assert store.list("a/") == ["a/b.json", "a/d.json"]
    for bad in ("", "/abs", "a//b", "a/../b"):
        with pytest.raises(ObjectStoreError):
            store.write(bad, b"")


def test_s3_object_store_keys_carry_the_store_prefix():
    s3 = FakeS3()
    s3_store.S3ObjectStore(URI + "/", client=s3).write("x/y", b"1")
    s3_store.S3ObjectStore("s3://bkt", client=s3).write("x/y", b"2")
    assert s3.objects == {("bkt", "runs/a/x/y"): b"1", ("bkt", "x/y"): b"2"}
    for bad in ("modal-volume://v/p", "s3://", "/local/path"):
        with pytest.raises(ValueError):
            s3_store.S3ObjectStore(bad, client=s3)


def test_s3_list_skips_directory_markers_of_a_bucket_mount():
    s3 = FakeS3()
    store = s3_store.S3ObjectStore(URI, client=s3)
    s3.objects[("bkt", "runs/a/policy-releases/r1-v0/")] = b""  # goofys mkdir
    store.write("policy-releases/r1-v0/RELEASE.json", b"{}")
    assert store.list("policy-releases/r1-v0/") == ["policy-releases/r1-v0/RELEASE.json"]


def test_s3_create_without_conditional_writes_falls_back_to_check_and_put():
    s3 = FakeS3(conditional=False)
    store = s3_store.S3ObjectStore(URI, client=s3)
    assert store.create("k", b"1") and not store.conditional_writes
    assert not store.create("k", b"2") and store.read("k") == b"1"


def test_s3_create_raises_other_errors():
    s3 = FakeS3()
    s3.put_object = lambda **kw: (_ for _ in ()).throw(_error("AccessDenied", "PutObject"))
    with pytest.raises(ClientError):
        s3_store.S3ObjectStore(URI, client=s3).create("k", b"1")


def test_directory_store_skips_staged_files(tmp_path):
    store = DirectoryObjectStore(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json.1234abcd.tmp").write_bytes(b"partial")
    store.write("a/y.json", b"1")
    assert store.list("a/") == ["a/y.json"]


# ---------------------------------------------------------------- sample buffer
def _add(buf, sample_id, version):
    ref = buf.write_payload(sample_id, sample_id.encode())
    buf.append(SampleIndexEntry(sample_id, version, ref, tokens=3))


@pytest.mark.parametrize("which", [0, 1], ids=["directory", "s3"])
def test_buffer_by_object_key(tmp_path, which):
    objects = _stores(tmp_path)[which]
    writer = FileSampleBuffer(objects, batch_size=2)
    assert isinstance(writer.index, ObjectIndex)
    for i, v in enumerate([0, 0, 1]):
        _add(writer, f"s{i}", v)
    assert objects.list("sample-buffer/index/") == [f"sample-buffer/index/{i:012d}.json" for i in range(3)]
    reader = FileSampleBuffer(objects, batch_size=2)  # another host
    assert len(reader) == 3 and reader.backlog(2).version_histogram == {0: 2, 1: 1}
    got = reader.select("trainer", 2)
    assert [s.entry.sample_id for s in got] == ["s0", "s1"]
    assert reader.read_payload(got[1].entry.payload_ref) == b"s1"
    reader.commit_consumed(2)
    _add(writer, "s3", 2)
    assert reader.poll() == 1 and [s.position for s in reader.select("trainer", 9)] == [2, 3]
    writer.mark_data_end()
    assert reader.backlog(2).data_ended and FileSampleBuffer(objects, batch_size=2).data_ended
    with pytest.raises(SampleBufferError, match="data end"):
        _add(writer, "late", 3)
    with pytest.raises(SampleBufferError, match="payload"):
        writer.append(SampleIndexEntry("x", 0, "payloads/missing.bin"))


def test_two_writers_never_share_a_position(tmp_path):
    """Writers on two hosts: each append takes the next free position with a
    create-if-absent write; the loser re-reads and takes the next one."""
    objects = s3_store.S3ObjectStore(URI, client=FakeS3())
    a, b = FileSampleBuffer(objects, batch_size=1), FileSampleBuffer(objects, batch_size=1)
    _add(a, "a0", 0)
    _add(b, "b0", 0)  # b had read nothing: loses position 0, takes 1
    _add(a, "a1", 0)
    threads = [threading.Thread(target=_add, args=(buf, f"{name}{i}", 0))
               for i in range(2, 12) for name, buf in (("a", a), ("b", b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [s.entry.sample_id for s in FileSampleBuffer(objects, batch_size=1).select("t", 99)]
    assert ids[:3] == ["a0", "b0", "a1"] and len(ids) == len(set(ids)) == 23


def test_reader_stops_before_a_missing_position(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    writer = FileSampleBuffer(objects, batch_size=1)
    _add(writer, "s0", 0)
    record = json.loads(objects.read("sample-buffer/index/000000000000.json"))
    objects.write("sample-buffer/index/000000000002.json", json.dumps({**record, "pos": 2}).encode())
    reader = FileSampleBuffer(objects, batch_size=1)
    assert len(reader) == 1  # position 1 is not there yet: 2 waits


def test_buffer_on_an_s3_store_uses_the_object_api_not_the_mount(tmp_path):
    s3 = FakeS3()
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI}
    store = store_for(tmp_path / "mount", environ=env, client=s3)
    assert isinstance(store, s3_store.S3Store) and store.describe()["uri"] == URI
    _add(FileSampleBuffer(store, batch_size=1), "s0", 0)
    assert not (tmp_path / "mount").exists()  # nothing through the mount
    assert ("bkt", "runs/a/sample-buffer/index/000000000000.json") in s3.objects
    assert ("bkt", "runs/a/sample-buffer/payloads/s0.bin") in s3.objects


def test_path_store_keeps_the_jsonl_index(tmp_path):
    buf = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    _add(buf, "s0", 0)
    assert (tmp_path / "sample-buffer" / "index.jsonl").read_text().count("\n") == 1


# ---------------------------------------------------------------- commands
def test_trainer_commands_through_the_object_api():
    objects = s3_store.S3ObjectStore(URI, client=FakeS3())
    writer, reader = CommandWriter(objects, 3), CommandReader(objects, 3)
    assert reader.poll() == []
    writer.send("run", rounds=2)
    writer.send("stop", reason=LeaveReason.STOPPED)
    assert [(c.seq, c.verb) for c in reader.poll()] == [(1, "run"), (2, "stop")]
    assert reader.poll() == []


# ---------------------------------------------------------------- policy releases
class _ObjectsMount(CheckpointStore):
    def __init__(self, root, objects):
        super().__init__(root)
        self.objects = objects


def test_policy_release_pointer_and_weights_by_object_key(tmp_path):
    """The trainer writes through its mount; the sampler reads LATEST by key and
    downloads the weights (its own mount may hold a stale copy)."""
    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(_ObjectsMount(shared, DirectoryObjectStore(shared)))
    sampler_mount = tmp_path / "sampler-mount"  # a stale mount: never updated
    sampler = PolicyReleaseStore(_ObjectsMount(sampler_mount, DirectoryObjectStore(shared)),
                                 cache_dir=tmp_path / "cache")
    assert sampler.latest() is None

    def weights(text):
        def write(directory):
            (directory / "adapter_model.safetensors").write_text(text)
            (directory / "sub").mkdir()
            (directory / "sub" / "cfg.json").write_text("{}")
        return write

    loaded = []
    follower = PolicyFollower(sampler, lambda release, directory: loaded.append(
        (release.version, (directory / "adapter_model.safetensors").read_text())))
    trainer.release(1, "h1", weights("w1"), weights=LORA)
    assert follower.refresh().version == 1
    trainer.release(2, "h2", weights("w2"), weights=LORA)
    assert follower.refresh().version == 2 and follower.refresh() is None
    assert loaded == [(1, "w1"), (2, "w2")]
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["r2-v2"]  # older download dropped
    assert not sampler_mount.exists()


# ---------------------------------------------------------------- credentials + launch
def test_island_store_uses_only_the_store_credentials(tmp_path, monkeypatch):
    import boto3

    sessions = []

    class Session:
        def __init__(self, **kw):
            sessions.append(kw)

        def client(self, name, **kw):
            sessions.append(kw)
            return FakeS3()

    monkeypatch.setattr(boto3.session, "Session", Session)
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI}
    with pytest.raises(ObjectStoreError, match="never the default AWS chain"):
        store_for(tmp_path, environ=env).objects
    assert sessions == []
    env.update(CREDS, **{s3_store.ENDPOINT_ENV: "http://minio:9000"})
    assert store_for(tmp_path, environ=env).objects.bucket == "bkt"
    assert sessions == [{"aws_access_key_id": "AKIDSTORE", "aws_secret_access_key": "store-secret",
                         "aws_session_token": None},
                        {"endpoint_url": "http://minio:9000", "region_name": None}]


def test_island_envs_copy_the_store_settings_and_need_credentials():
    with pytest.raises(ValueError, match="store credentials"):
        s3_store.island_envs(URI, environ={"AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "y"})
    env = s3_store.island_envs(URI, environ={**CREDS, s3_store.REGION_ENV: "eu-north1", "AWS_SECRET_ACCESS_KEY": "y"})
    assert env == {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI,
                   s3_store.REGION_ENV: "eu-north1", **CREDS}
    plain, secrets = split_secret_envs(env)
    assert set(secrets) == {s3_store.SECRET_ENV} and s3_store.SECRET_ENVS <= SECRET_ENV_NAMES


def test_store_object_envs_only_for_sky_islands_with_a_sample_role(monkeypatch):
    from types import SimpleNamespace

    for key, value in CREDS.items():
        monkeypatch.setenv(key, value)
    trainer = SimpleNamespace(rl_checkpoint_store=URI, rl_sample_role="trainer")
    assert store_object_envs(trainer, "nebius")[s3_store.URI_ENV] == URI
    assert store_object_envs(trainer, "nebius")[s3_store.STAGE_ENV] == s3_store.DEFAULT_STAGE
    assert s3_store.STAGE_ENV not in store_object_envs(trainer, "nebius", num_nodes=2)  # ranks on 2 disks
    assert store_object_envs(trainer, "modal") == {}
    assert store_object_envs(SimpleNamespace(rl_checkpoint_store=URI, rl_sample_role="both"), "nebius") == {}
    assert store_object_envs(SimpleNamespace(rl_checkpoint_store="modal-volume://v/p",
                                             rl_sample_role="trainer"), "nebius") == {}
    monkeypatch.delenv(s3_store.KEY_ID_ENV)
    with pytest.raises(ValueError, match="store credentials"):
        store_object_envs(trainer, "nebius")


# ---------------------------------------------------------------- staged cuts (TreeMirror)
class _RecordingObjects(DirectoryObjectStore):
    def __init__(self, root):
        super().__init__(root)
        self.writes: list[str] = []
        self.deletes: list[str] = []

    def write(self, key, data):
        self.writes.append(key)
        super().write(key, data)

    def write_file(self, key, source):
        self.writes.append(key)
        super().write_file(key, source)

    def delete(self, key):
        self.deletes.append(key)
        super().delete(key)


def _mirror(root, objects):
    from yeto.rl.engine.object_store import TreeMirror

    return TreeMirror(root, objects, subtrees=s3_store.STAGED_SUBTREES, pointers=s3_store.POINTER_FILES)


def test_tree_mirror_pushes_pointers_last_once_and_deletes_pruned(tmp_path):
    objects = _RecordingObjects(tmp_path / "bucket")
    a = tmp_path / "a"
    (a / "state" / "00000000").mkdir(parents=True)
    (a / "LATEST").write_text("0")
    (a / "state" / "00000000" / "x").write_text("x")
    (a / "round-cuts" / "r1").mkdir(parents=True)
    (a / "round-cuts" / "r1" / "shard").write_bytes(b"s" * 10)
    (a / "x.tmp").write_text("partial")
    _mirror(a, objects).push()
    assert objects.writes == ["round-cuts/r1/shard", "state/00000000/x", "LATEST"]
    objects.writes.clear()
    _mirror(a, objects).push()  # another process of the host: nothing new
    assert objects.writes == []
    (a / "LATEST").write_text("1")
    import shutil
    shutil.rmtree(a / "round-cuts" / "r1")
    _mirror(a, objects).push()
    assert objects.writes == ["LATEST"] and objects.deletes == ["round-cuts/r1/shard"]


def test_tree_mirror_pull_fetches_missing_keys_and_every_pointer(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "bucket")
    objects.write("LATEST", b"0")
    objects.write("state/00000000/x", b"x")
    objects.write("policy-releases/LATEST.json", b"{}")  # not a staged subtree
    b = tmp_path / "b"
    mirror = _mirror(b, objects)
    mirror.pull()
    assert (b / "LATEST").read_bytes() == b"0" and (b / "state" / "00000000" / "x").read_bytes() == b"x"
    assert not (b / "policy-releases").exists()
    objects.write("LATEST", b"1")
    mirror.pull()
    assert (b / "LATEST").read_bytes() == b"1"
    rec = _RecordingObjects(tmp_path / "bucket")
    _mirror(b, rec).push()  # pulled files are not uploaded again
    assert rec.writes == []


def test_staged_s3_store_carries_snapshots_and_cuts_between_hosts(tmp_path):
    """The trainer of one incarnation snapshots its state dir and a round cut on its
    own disk; commit uploads them; the next incarnation (another host) restores."""
    from yeto.rl.engine.resume import ROUND_CUTS, ResumeController

    s3 = FakeS3()
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI}

    def store(host):
        return store_for(tmp_path / "mount", environ={**env, s3_store.STAGE_ENV: str(tmp_path / host)}, client=s3)

    first = store("host-a")
    assert first.staged and first.root == tmp_path / "host-a" / "bkt" / "runs" / "a"
    ctl = ResumeController(state_dir=tmp_path / "state-a", store=first)
    cut = first.root / ROUND_CUTS / "r000001-x"
    cut.mkdir(parents=True)
    (cut / "shard-0.pt").write_bytes(b"w" * 100)
    (ctl.state_dir / "round-cut.json").write_text(json.dumps({"cut_id": "r000001-x", "next_rollout_id": 1}))
    assert ctl.sync_checkpoint_store("round_cut")
    assert ctl.last_store_sync["commit_s"] >= 0
    assert ("bkt", "runs/a/round-cuts/r000001-x/shard-0.pt") in s3.objects
    assert ("bkt", "runs/a/LATEST") in s3.objects
    assert not (tmp_path / "mount").exists()

    second = ResumeController(state_dir=tmp_path / "state-b", store=store("host-b"))
    assert second.restored["seq"] == 0
    assert json.loads((tmp_path / "state-b" / "round-cut.json").read_text())["cut_id"] == "r000001-x"
    assert (second.store.root / ROUND_CUTS / "r000001-x" / "shard-0.pt").read_bytes() == b"w" * 100
