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
        self.upload_checksum: dict[str, str | None] = {}
        self.checksums: dict[tuple[str, str], str | None] = {}
        self.created = 0
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

    @staticmethod
    def sha(data: bytes) -> str:
        import base64
        import hashlib

        return base64.b64encode(hashlib.sha256(data).digest()).decode()

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, ChecksumSHA256=None):
        with self.lock:
            if IfNoneMatch is not None:
                if not self.conditional:
                    raise _error("NotImplemented", "PutObject")
                if (Bucket, Key) in self.objects:
                    raise _error("PreconditionFailed", "PutObject")
            if ChecksumSHA256 is not None and ChecksumSHA256 != self.sha(bytes(Body)):
                raise _error("BadDigest", "PutObject")
            self.objects[(Bucket, Key)] = bytes(Body)
            self.checksums[(Bucket, Key)] = ChecksumSHA256
            self.calls.append("put_object")

    def head_object(self, Bucket, Key, ChecksumMode=None):
        if (Bucket, Key) not in self.objects:
            raise _error("404", "HeadObject")
        data = self.objects[(Bucket, Key)]
        out = {"ContentLength": len(data), "ETag": self.etag(data)}
        if ChecksumMode == "ENABLED" and self.checksums.get((Bucket, Key)):
            out["ChecksumSHA256"] = self.checksums[(Bucket, Key)]
        return out

    def delete_object(self, Bucket, Key):
        with self.lock:
            self.objects.pop((Bucket, Key), None)
            self.checksums.pop((Bucket, Key), None)

    def create_multipart_upload(self, Bucket, Key, ChecksumAlgorithm=None):
        with self.lock:
            upload_id = f"u{self.created}"
            self.created += 1
            self.uploads[upload_id] = {}
            self.upload_checksum[upload_id] = ChecksumAlgorithm
            self.calls.append("create_multipart_upload")
        return {"UploadId": upload_id}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, ChecksumSHA256=None):
        with self.lock:
            if UploadId not in self.uploads:
                raise _error("NoSuchUpload", "UploadPart")
            if ChecksumSHA256 is not None and ChecksumSHA256 != self.sha(bytes(Body)):
                raise _error("BadDigest", "UploadPart")
            self.uploads[UploadId][PartNumber] = bytes(Body)
            self.calls.append("upload_part")
        return {"ETag": self.etag(bytes(Body))}

    def list_parts(self, Bucket, Key, UploadId, PartNumberMarker=0, MaxParts=2):
        with self.lock:
            if UploadId not in self.uploads:
                raise _error("NoSuchUpload", "ListParts")
            numbers = sorted(n for n in self.uploads[UploadId] if n > int(PartNumberMarker))
            page, rest = numbers[:MaxParts], numbers[MaxParts:]
            self.calls.append("list_parts")
            sums = self.upload_checksum[UploadId] == "SHA256"
            return {"Parts": [{"PartNumber": n, "ETag": self.etag(self.uploads[UploadId][n]),
                               "Size": len(self.uploads[UploadId][n]),
                               **({"ChecksumSHA256": self.sha(self.uploads[UploadId][n])} if sums else {})}
                              for n in page],
                    "IsTruncated": bool(rest), **({"NextPartNumberMarker": page[-1]} if rest else {})}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        import base64
        import hashlib

        parts = self.uploads.pop(UploadId)
        numbers = [p["PartNumber"] for p in MultipartUpload["Parts"]]
        assert numbers == sorted(parts) == list(range(1, len(parts) + 1))
        assert [p["ETag"] for p in MultipartUpload["Parts"]] == [self.etag(parts[n]) for n in numbers]
        checksum = None
        if self.upload_checksum.get(UploadId) == "SHA256":
            assert [p["ChecksumSHA256"] for p in MultipartUpload["Parts"]] == [self.sha(parts[n]) for n in numbers]
            raw = b"".join(hashlib.sha256(parts[n]).digest() for n in numbers)
            checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode() + f"-{len(numbers)}"
        self.objects[(Bucket, Key)] = b"".join(parts[n] for n in numbers)
        self.checksums[(Bucket, Key)] = checksum
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
            pools.append(kw.pop("config").max_pool_connections)
            sessions.append(kw)
            return FakeS3()

    pools = []
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
    assert pools == [8 * 16]  # one connection per transfer thread (default 8 files x 16 parts)


def test_client_pool_covers_the_transfer_threads():
    """botocore's default pool of 10 serialized the 8 x 16 range reads of a release
    download (221.7 MB/s for a 207 GB K3.1 release on an MI355X node)."""
    assert s3_store.client_pool_size({}) == 128
    assert s3_store.client_pool_size({s3_store.FILE_CONCURRENCY_ENV: "32", s3_store.PART_CONCURRENCY_ENV: "8"}) == 256
    assert s3_store.client_pool_size({s3_store.FILE_CONCURRENCY_ENV: "1", s3_store.PART_CONCURRENCY_ENV: "2"}) == 10
    assert s3_store.client_pool_size({s3_store.MAX_CONNECTIONS_ENV: "40"}) == 40
    assert s3_store.MAX_CONNECTIONS_ENV in s3_store.SETTING_ENVS


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
    # a Modal island with an s3 store has no mount: the whole store goes by object API
    # from a local stage, the credential shipped like on sky...
    modal_envs = store_object_envs(trainer, "modal", num_nodes=2)
    assert modal_envs[s3_store.STAGE_ENV] == s3_store.DEFAULT_STAGE and modal_envs[s3_store.KEY_ID_ENV]
    assert s3_store.CREDENTIALS_ENV not in modal_envs
    # ...or, with --modal-store-secret, read from the Secret's AWS_* variables on the island
    secret = SimpleNamespace(rl_checkpoint_store=URI, modal_store_secret="k31-store",
                             rl_object_store_endpoint="https://acct.r2.cloudflarestorage.com")
    envs = store_object_envs(secret, "modal")
    assert envs[s3_store.CREDENTIALS_ENV] == s3_store.CREDENTIALS_AWS_ENV
    assert not any(name in envs for name in s3_store.CREDENTIAL_ENVS)
    assert envs[s3_store.ENDPOINT_ENV] == "https://acct.r2.cloudflarestorage.com"
    assert envs[s3_store.REGION_ENV] == "auto"
    assert store_object_envs(SimpleNamespace(rl_checkpoint_store="modal-volume://v/p"), "modal") == {}
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
    # the data files of a tier move concurrently (any order), the pointer only after them
    assert sorted(objects.writes[:2]) == ["round-cuts/r1/shard", "state/00000000/x"]
    assert objects.writes[2:] == ["LATEST"]
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


def test_a_stalled_range_read_is_retried_not_fatal(tmp_path, monkeypatch):
    from botocore.exceptions import ReadTimeoutError

    s3 = FakeS3()
    part = s3_store.MIN_PART_BYTES
    data = bytes(range(256)) * (3 * part // 256)  # three ranges
    store = s3_store.S3ObjectStore(URI, client=s3, transfer=s3_store.TransferSettings(part_bytes=part))
    s3.objects[("bkt", store._key("big"))] = data
    real, stalls = s3.get_object, {"left": 2}

    class Body:
        def __init__(self, inner):
            self.inner = inner

        def read(self):
            if stalls["left"]:
                stalls["left"] -= 1
                raise ReadTimeoutError(endpoint_url="r2")
            return self.inner.read()

    def get_object(**kw):
        out = real(**kw)
        return {**out, "Body": Body(out["Body"])} if kw.get("Range") else out

    monkeypatch.setattr(s3, "get_object", get_object)
    monkeypatch.setattr(s3_store.time, "sleep", lambda s: None)
    assert store.read_file("big", tmp_path / "out")
    assert (tmp_path / "out").read_bytes() == data and stalls["left"] == 0



def test_a_broken_range_stream_is_retried_not_fatal(tmp_path, monkeypatch):
    from botocore.exceptions import ResponseStreamingError

    s3 = FakeS3()
    part = s3_store.MIN_PART_BYTES
    data = bytes(range(256)) * (3 * part // 256)  # three ranges
    store = s3_store.S3ObjectStore(URI, client=s3, transfer=s3_store.TransferSettings(part_bytes=part))
    s3.objects[("bkt", store._key("big"))] = data
    real, stalls = s3.get_object, {"left": 2}

    class Body:
        def __init__(self, inner):
            self.inner = inner

        def read(self):
            if stalls["left"]:
                stalls["left"] -= 1
                raise ResponseStreamingError(error="IncompleteRead(8454090 bytes read)")
            return self.inner.read()

    def get_object(**kw):
        out = real(**kw)
        return {**out, "Body": Body(out["Body"])} if kw.get("Range") else out

    monkeypatch.setattr(s3, "get_object", get_object)
    monkeypatch.setattr(s3_store.time, "sleep", lambda s: None)
    assert store.read_file("big", tmp_path / "out")
    assert (tmp_path / "out").read_bytes() == data and stalls["left"] == 0

def test_staged_release_leaves_no_weights_on_the_trainer_disk(tmp_path):
    """A staged store pushes each release, then drops its stage copy (a kept
    207 GB Kimi-K3.1 release per version filled the trainer's disk); the bucket
    keeps the release until the prune deletes it there too."""
    s3 = FakeS3()
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI,
           s3_store.STAGE_ENV: str(tmp_path / "stage")}
    store = store_for(tmp_path / "mount", environ=env, client=s3)
    trainer = PolicyReleaseStore(store)

    for version in range(1, 6):
        staged = tmp_path / f"staging-v{version}"  # the trainer's export, then the upload
        staged.mkdir()
        (staged / "model.safetensors").write_text(f"w{version}")
        trainer.release_staged(version, f"h{version}", staged, position=lambda: (None, None))
    local = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*.safetensors"))
    assert local == []
    remote = sorted(k for _, k in s3.objects if k.endswith(".safetensors"))
    assert remote and "runs/a/policy-releases/r5-v5/model.safetensors" in remote
    assert "runs/a/policy-releases/r1-v1/model.safetensors" not in remote  # pruned in the bucket
    store.commit()  # a later full push takes the evicted files for neither upload nor prune
    assert sorted(k for _, k in s3.objects if k.endswith(".safetensors")) == remote


def test_a_sampler_keeps_the_served_release_until_the_next_one_is_loaded(tmp_path, monkeypatch):
    """A restart or a whole reload of the engines reads the served release's files
    (an engine that reloads by restart, vLLM on the A cards): the download of the
    next release never drops it, nor its overlay. Once the next one is loaded the
    older download and overlay go (two 207 GB Kimi-K3.1 releases stay at most)."""
    from yeto.rl.engine import policy_release

    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(_ObjectsMount(shared, DirectoryObjectStore(shared)))
    cache = tmp_path / "cache"
    sampler = PolicyReleaseStore(_ObjectsMount(tmp_path / "sampler-mount", DirectoryObjectStore(shared)),
                                 cache_dir=cache)
    on_disk = []
    real = policy_release.download_file

    def download(objects, key, path):
        on_disk.append(sorted(p.name for p in cache.iterdir() if not p.name.startswith(".")))
        return real(objects, key, path)

    monkeypatch.setattr(policy_release, "download_file", download)

    def weights(text):
        def write(directory):
            (directory / "model.safetensors").write_text(text)
        return write

    fail = []

    def load(release, directory):
        if sampler.serving is not None:  # the served release is still readable
            assert (cache / sampler.serving / "model.safetensors").is_file()
        if fail:
            raise RuntimeError("engine load failed")

    follower = PolicyFollower(sampler, load)
    trainer.release(1, "h1", weights("w1"))
    assert follower.refresh().version == 1 and sampler.serving == "r1-v1"
    (cache / "r1-v1.overlay").mkdir()  # the overlay the engine serves
    (cache / "r0-v9.overlay").mkdir()  # an overlay of a release no longer served
    on_disk.clear()
    trainer.release(2, "h2", weights("w2"))
    fail.append(True)
    with pytest.raises(RuntimeError):
        follower.refresh()  # the load fails: the served release and its overlay stay
    assert all("r1-v1" in names and "r1-v1.overlay" in names for names in on_disk)
    assert {"r1-v1", "r1-v1.overlay", "r2-v2"} <= {p.name for p in cache.iterdir()}
    fail.clear()
    assert follower.refresh().version == 2 and sampler.serving == "r2-v2"
    assert sorted(p.name for p in cache.iterdir()) == ["r2-v2"]
    trainer.release(3, "h3", weights("w3"))
    (cache / "r2-v2.overlay").mkdir()
    assert follower.refresh().version == 3
    assert sorted(p.name for p in cache.iterdir()) == ["r3-v3"]


def test_a_release_without_its_record_is_never_read(tmp_path):
    """A trainer stopped during an upload leaves weights without RELEASE.json and
    LATEST still names the previous release (k31-rl-3 r3-v3): readers skip it."""
    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(_ObjectsMount(shared, DirectoryObjectStore(shared)))
    trainer.release(1, "h1", lambda d: (d / "model.safetensors").write_text("w1"))
    torn = shared / "policy-releases" / "r2-v2"
    torn.mkdir()
    (torn / "model.safetensors").write_text("half")
    (torn / ".complete").write_text("")
    reader = PolicyReleaseStore(_ObjectsMount(tmp_path / "m", DirectoryObjectStore(shared)),
                                cache_dir=tmp_path / "cache")
    assert reader.latest().path == "r1-v1"
    assert [r.path for r in reader.records()] == ["r1-v1"]
    follower = PolicyFollower(reader, lambda release, directory: None)
    assert follower.refresh().path == "r1-v1" and follower.refresh() is None


def test_a_staged_store_links_a_release_export_instead_of_copying_it(tmp_path, monkeypatch):
    s3 = FakeS3()
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI,
           s3_store.STAGE_ENV: str(tmp_path / "stage")}
    store = store_for(tmp_path / "mount", environ=env, client=s3)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(b"w" * 10)
    seen = []
    real_push = store._tree().push

    def push(paths=None):
        seen.append((store.root / "rel" / "model.safetensors").stat().st_ino)
        return real_push(paths)

    monkeypatch.setattr(store._tree(), "push", push)
    store.put_directory(export, "rel")
    assert seen == [(export / "model.safetensors").stat().st_ino]
    assert s3.objects[("bkt", "runs/a/rel/model.safetensors")] == b"w" * 10


def test_a_release_downloads_several_files_at_once(tmp_path, monkeypatch):
    import threading
    import time as _time

    from yeto.rl.engine import policy_release

    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(_ObjectsMount(shared, DirectoryObjectStore(shared)))
    objects = DirectoryObjectStore(shared)
    objects.transfer = s3_store.TransferSettings(file_concurrency=4)
    sampler = PolicyReleaseStore(_ObjectsMount(tmp_path / "m", objects), cache_dir=tmp_path / "cache")
    active, peak, lock = [0], [0], threading.Lock()
    real = policy_release.download_file

    def download(objs, key, path):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.05)
        try:
            return real(objs, key, path)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(policy_release, "download_file", download)

    def weights(directory):
        for i in range(8):
            (directory / f"model-{i}.safetensors").write_text(str(i))

    trainer.release(1, "h1", weights)
    directory = sampler.resolve(sampler.latest())
    assert sorted(p.name for p in directory.glob("*.safetensors")) == [f"model-{i}.safetensors" for i in range(8)]
    assert peak[0] > 1


def _slow_release_world(tmp_path, monkeypatch):
    import threading

    from yeto.rl.engine import policy_release

    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(_ObjectsMount(shared, DirectoryObjectStore(shared)))
    sampler = PolicyReleaseStore(_ObjectsMount(tmp_path / "m", DirectoryObjectStore(shared)),
                                 cache_dir=tmp_path / "cache")
    gate, fail = threading.Event(), {"left": 0}
    real = policy_release.download_file

    def download(objects, key, path):
        gate.wait(5)
        if fail["left"]:
            fail["left"] -= 1
            raise ConnectionResetError("reset by peer")
        return real(objects, key, path)

    monkeypatch.setattr(policy_release, "download_file", download)
    monkeypatch.setattr(policy_release, "DOWNLOAD_WAIT_S", 0.0)

    def weights(directory):
        (directory / "model.safetensors").write_text("w")

    return trainer, sampler, gate, fail, weights


def test_a_release_downloads_in_the_background_while_the_engines_generate(tmp_path, monkeypatch):
    trainer, sampler, gate, _, weights = _slow_release_world(tmp_path, monkeypatch)
    loaded = []
    follower = PolicyFollower(sampler, lambda r, d: loaded.append(r.version), background_download=True)
    trainer.release(1, "h1", weights)
    assert follower.refresh() is None and follower.incomplete == "downloading r1-v1"
    with follower.generating():  # a round runs while the files arrive
        assert follower.refresh() is None
    gate.set()
    follower._fetch.done.wait(5)
    assert follower.refresh().version == 1 and loaded == [1]
    assert follower.last_timings["resolve_s"] >= 0


def test_a_failed_download_keeps_the_served_weights_and_is_tried_again(tmp_path, monkeypatch):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    trainer, sampler, gate, fail, weights = _slow_release_world(tmp_path, monkeypatch)
    now = [0.0]
    follower = PolicyFollower(sampler, lambda r, d: None, background_download=True,
                              download_retry_s=60.0, timer=lambda: now[0])
    trainer.release(1, "h1", weights)
    fail["left"] = 1
    gate.set()
    follower.refresh()
    follower._fetch.done.wait(5)
    with pytest.raises(ReleaseNotLoaded) as raised:
        follower.refresh()
    assert raised.value.reason == "download_failed" and follower.version == 0
    assert follower.refresh() is None  # reported once, then waits to retry
    now[0] = 61.0
    assert follower.refresh() is None  # retry window passed: a new download starts next
    follower.refresh()
    follower._fetch.done.wait(5)
    assert follower.refresh().version == 1


class DiskS3:
    """A picklable read-only client over ``root/<bucket>/<key>`` files, built again
    in each process of ``S3ObjectStore.read_files``; ``pids`` records who read."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, root):
        self.root = root

    def _path(self, Bucket, Key):
        import pathlib

        return pathlib.Path(self.root) / Bucket / Key

    def head_object(self, Bucket, Key):
        import hashlib

        p = self._path(Bucket, Key)
        if not p.is_file():
            raise _error("404", "HeadObject")
        return {"ContentLength": p.stat().st_size, "ETag": '"' + hashlib.md5(p.read_bytes()).hexdigest() + '"'}

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        import os as _os
        import pathlib

        p = self._path(Bucket, Key)
        if not p.is_file():
            raise self.exceptions.NoSuchKey(Key)
        data = p.read_bytes()
        if Range is not None:
            first, last = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            data = data[first:last + 1]
        log = pathlib.Path(self.root) / "pids"
        with open(log, "a") as fh:
            fh.write(f"{_os.getpid()}\n")
        return {"Body": _Bytes(data)}


class _Bytes:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


def _disk_store(tmp_path, files, *, processes):
    import functools

    for key, data in files.items():
        p = tmp_path / "s3" / "bkt" / "pfx" / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    factory = functools.partial(DiskS3, str(tmp_path / "s3"))
    transfer = s3_store.TransferSettings(part_bytes=s3_store.MIN_PART_BYTES, concurrency=2, file_concurrency=4,
                                         download_processes=processes)
    return s3_store.S3ObjectStore("s3://bkt/pfx", client=factory(), transfer=transfer, client_factory=factory)


def test_read_files_spreads_the_files_over_processes(tmp_path):
    """One Python process reads R2 at ~700 MB/s at most (MI355X): a release download
    splits its files over download_processes processes, each with its own client."""
    import os as _os

    big = bytes(range(256)) * (s3_store.MIN_PART_BYTES // 256 * 2 + 7)  # three ranged parts
    files = {f"r1-v1/model-{i:05d}.safetensors": big[i:] for i in range(6)}
    objects = _disk_store(tmp_path, files, processes=3)
    out = tmp_path / "out"
    out.mkdir()
    items = [(k, out / k.split("/")[-1]) for k in files] + [("r1-v1/gone.safetensors", out / "gone")]
    assert objects.read_files(items) == ["r1-v1/gone.safetensors"]
    for key, data in files.items():
        assert (out / key.split("/")[-1]).read_bytes() == data
    assert not (out / "gone").exists()
    pids = set((tmp_path / "s3" / "pids").read_text().split())
    assert len(pids) == 3 and str(_os.getpid()) not in pids


def test_read_files_without_a_factory_stays_in_process(tmp_path):
    import os as _os

    objects = _disk_store(tmp_path, {"a": b"x" * 10, "b": b"y" * 20}, processes=4)
    objects.client_factory = None
    out = tmp_path / "out"
    out.mkdir()
    assert objects.read_files([("a", out / "a"), ("b", out / "b")]) == []
    assert (out / "b").read_bytes() == b"y" * 20
    assert set((tmp_path / "s3" / "pids").read_text().split()) == {str(_os.getpid())}


def test_release_download_goes_through_read_files(tmp_path):
    """PolicyReleaseStore downloads a release with the object API's read_files."""
    from pathlib import Path

    calls = []

    class Objects(DirectoryObjectStore):
        def read_files(self, items):
            calls.append(sorted(Path(t).name for _, t in items))
            missing = []
            for key, target in items:
                data = self.read(key)
                if data is None:
                    missing.append(key)
                else:
                    Path(target).write_bytes(data)
            return missing

    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(_ObjectsMount(shared, DirectoryObjectStore(shared)))
    sampler = PolicyReleaseStore(_ObjectsMount(tmp_path / "m", Objects(shared)), cache_dir=tmp_path / "cache")

    def write(directory):
        (directory / "adapter_model.safetensors").write_text("w1")
        (directory / "cfg.json").write_text("{}")

    trainer.release(1, "h1", write, weights=LORA)
    got = sampler.resolve(sampler.latest())
    assert (got / "adapter_model.safetensors").read_text() == "w1"
    assert len(calls) == 1 and "adapter_model.safetensors" in calls[0]


def test_read_files_from_an_unguarded_script_runs_the_script_once(tmp_path):
    """multiprocessing's spawn re-imported an unguarded __main__ on the MI355X node
    (the download never started); the subprocess children do not touch it."""
    import subprocess
    import sys
    import textwrap

    _disk_store(tmp_path, {"a": b"1" * 9, "b": b"2" * 9}, processes=2)  # writes the objects
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(f"""
        import functools, sys
        sys.path[:0] = {[p for p in sys.path if p]!r}
        open({str(tmp_path / 'runs')!r}, "a").write("run\\n")
        from yeto.cloud import s3_store
        from test_rl_object_store import DiskS3
        f = functools.partial(DiskS3, {str(tmp_path / 's3')!r})
        t = s3_store.TransferSettings(file_concurrency=2, download_processes=2)
        o = s3_store.S3ObjectStore("s3://bkt/pfx", client=f(), transfer=t, client_factory=f)
        print(o.read_files([("a", {str(tmp_path / 'a')!r}), ("b", {str(tmp_path / 'b')!r})]))
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    assert (tmp_path / "runs").read_text() == "run\n"
    assert (tmp_path / "b").read_bytes() == b"2" * 9
    assert len(set((tmp_path / "s3" / "pids").read_text().split())) == 2
