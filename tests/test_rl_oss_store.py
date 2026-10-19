"""Cross-site store on Alibaba Cloud OSS (rl-cross-site-sample-store N3.1-N3.4; CPU,
fake S3 client; no cloud resource): the OSS store and its credentials, one buffer
shard per sampler island merged by the trainer, complete releases only, multipart
transfer of large files, and the trainer scheduler's store check."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tests.test_rl_object_store import FakeS3, _error
from yeto.cloud import oss_store, s3_store
from yeto.cloud.s3_store import MIN_PART_BYTES, TransferSettings
from yeto.launch.credentials import SECRET_ENV_NAMES, split_secret_envs
from yeto.launch.storage import rl_checkpoint_store_plan, store_object_envs, store_unreachable_reason
from yeto.rl.engine.object_store import ObjectStoreError, TreeMirror
from yeto.rl.engine.policy_release import (LATEST, RECORD, RELEASE_DIR, W4A8_RELEASE_DIR, PolicyFollower,
                                          PolicyReleaseStore)
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore, store_for
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError, needs_shards, shard_index
from yeto.rl.engine.sampler_loop import SamplerLoop
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

URI = "oss://bkt/runs/a"
ENDPOINT = "https://oss-cn-hangzhou.aliyuncs.com"
CREDS = {oss_store.KEY_ID_ENV: "AKIDOSS", oss_store.SECRET_ENV: "oss-secret"}


class FakeOss(FakeS3):
    """FakeS3 plus the ``ForbidOverwrite`` parameter of :func:`oss_store.oss_client`."""

    accepts_forbid_overwrite = True

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, ForbidOverwrite=False):
        if ForbidOverwrite and (Bucket, Key) in self.objects:
            raise _error("FileAlreadyExists", "PutObject")
        super().put_object(Bucket, Key, Body, IfNoneMatch=IfNoneMatch)


def _env(tmp_path, **extra):
    return {STORE_IMPL_ENV: oss_store.STORE_IMPL, oss_store.URI_ENV: URI, oss_store.ENDPOINT_ENV: ENDPOINT,
            oss_store.STAGE_ENV: str(tmp_path / "stage"), **CREDS, **extra}


def _objects(fake=None, **transfer):
    return oss_store.OssObjectStore(URI, client=fake or FakeOss(), transfer=TransferSettings(**transfer))


# ---------------------------------------------------------------- N3.1 store + credentials
def test_store_for_builds_a_staged_oss_store_through_the_registry_hook(tmp_path):
    fake = FakeOss()
    store = store_for("/ignored", environ=_env(tmp_path), client=fake)
    assert isinstance(store, oss_store.OssStore) and store.kind == "oss"
    assert store.root == tmp_path / "stage" / "bkt" / "runs" / "a"
    assert store.objects.bucket == "bkt" and store.objects.prefix == "runs/a" and needs_shards(store)
    (store.root / "state").mkdir(parents=True)
    (store.root / "state" / "x").write_bytes(b"x")
    (store.root / "LATEST").write_text("0")
    store.commit()
    assert set(fake.objects) == {("bkt", "runs/a/state/x"), ("bkt", "runs/a/LATEST")}
    other = store_for("/ignored", environ=_env(tmp_path / "host2"), client=fake)
    other.reload()
    assert (other.root / "state" / "x").read_bytes() == b"x"
    with pytest.raises(ValueError, match="not a store URI oss://"):
        oss_store.parse_oss_uri("s3://bkt")


def test_create_uses_forbid_overwrite_and_reports_a_lost_race():
    objects = _objects()
    assert objects.create("k", b"1") is True
    assert objects.create("k", b"2") is False and objects.read("k") == b"1"


def test_oss_client_sends_the_forbid_overwrite_header_virtual_hosted_and_signed():
    client = oss_store.oss_client({oss_store.ENDPOINT_ENV: ENDPOINT, **CREDS})
    seen = {}

    class Sent(Exception):
        pass

    def capture(request, **_):
        seen.update(url=request.url, headers=dict(request.headers))
        raise Sent

    client.meta.events.register("before-send.s3.PutObject", capture)
    with pytest.raises(Sent):
        client.put_object(Bucket="bkt", Key="runs/a/k", Body=b"x", ForbidOverwrite=True)
    assert seen["url"] == "https://bkt.oss-cn-hangzhou.aliyuncs.com/runs/a/k"
    assert seen["headers"][oss_store.FORBID_OVERWRITE_HEADER] == b"true"
    assert seen["headers"]["Authorization"].startswith(b"AWS4-HMAC-SHA256")
    assert not any("checksum" in name.lower() for name in seen["headers"])


def test_credentials_come_from_the_environment_or_a_private_file(tmp_path):
    with pytest.raises(ObjectStoreError, match="set YETO_RL_STORE_OSS_ACCESS_KEY_ID"):
        oss_store.credentials({})
    assert oss_store.credentials(CREDS) == ("AKIDOSS", "oss-secret", None)
    file = tmp_path / "oss.json"
    file.write_text(json.dumps({"access_key_id": "AKIDFILE", "access_key_secret": "file-secret",
                                "security_token": "sts"}))
    env = {oss_store.CREDENTIALS_FILE_ENV: str(file)}
    file.chmod(0o644)
    with pytest.raises(ObjectStoreError, match="chmod 600") as refused:
        oss_store.credentials(env)
    assert "file-secret" not in str(refused.value)
    file.chmod(0o600)
    assert oss_store.credentials(env) == ("AKIDFILE", "file-secret", "sts")
    file.write_text("{not json")
    with pytest.raises(ObjectStoreError, match="cannot read") as bad:
        oss_store.credentials(env)
    assert "not json" not in str(bad.value)
    with pytest.raises(ObjectStoreError, match="ENDPOINT_URL is not set"):
        oss_store.oss_client(CREDS)


def test_island_envs_carry_the_settings_and_ship_the_secret_as_a_task_secret(tmp_path):
    with pytest.raises(ValueError, match="ENDPOINT_URL"):
        oss_store.island_envs(URI, environ=CREDS)
    with pytest.raises(ValueError, match="store credentials"):
        oss_store.island_envs(URI, environ={oss_store.ENDPOINT_ENV: ENDPOINT, "AWS_SECRET_ACCESS_KEY": "y"})
    env = oss_store.island_envs(URI, environ={oss_store.ENDPOINT_ENV: ENDPOINT, s3_store.PART_MB_ENV: "128",
                                              "AWS_SECRET_ACCESS_KEY": "y", **CREDS})
    assert env == {STORE_IMPL_ENV: oss_store.STORE_IMPL, oss_store.URI_ENV: URI,
                   oss_store.ENDPOINT_ENV: ENDPOINT, s3_store.PART_MB_ENV: "128", **CREDS}
    _plain, secrets = split_secret_envs(env)
    assert set(secrets) == {oss_store.SECRET_ENV} and oss_store.SECRET_ENVS <= SECRET_ENV_NAMES


# ---------------------------------------------------------------- N3.1 sharded buffer
def _entry(sample_id, version=0, appended_at=0.0):
    return SampleIndexEntry(sample_id, version, "", 1, appended_at)


def test_each_sampler_writes_its_own_shard_and_the_trainer_merges_them():
    objects = _objects()
    npu = FileSampleBuffer(objects, batch_size=1, index=shard_index("npu-sampler"))
    ppu = FileSampleBuffer(objects, batch_size=1, index=shard_index("ppu-sampler"))
    assert (npu.shard, ppu.shard) == ("npu-sampler", "ppu-sampler")
    npu.append(_entry("n0", appended_at=2.0))
    ppu.append(_entry("p0", appended_at=1.0))
    npu.append(_entry("n1", appended_at=3.0))
    assert not any(k.startswith("sample-buffer/index/") for k in objects.list("sample-buffer/"))
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    assert [s.entry.sample_id for s in trainer.select("t", 10)] == ["p0", "n0", "n1"]
    trainer.commit_consumed(2)
    assert trainer.shard_positions(trainer.watermark) == {"npu-sampler": 1, "ppu-sampler": 1}
    assert trainer.shard_positions(trainer.read_cursor) == {"npu-sampler": 2, "ppu-sampler": 1}
    # a restarted trainer resumes at the cut's watermark; merged records are not merged again
    ppu.append(_entry("p1", appended_at=4.0))
    again = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    again.reset_to(2)
    assert [s.entry.sample_id for s in again.select("t", 10)] == ["n1", "p1"]
    assert len(again) == 4
    # a reader that does not merge (the head) sees the merged index only
    assert len(FileSampleBuffer(objects, batch_size=1)) == 4
    with pytest.raises(SampleBufferError, match="only the index of an object store merges"):
        FileSampleBuffer(objects, batch_size=1, index=shard_index("x"), merge_shards=True)


def test_the_merged_index_ends_when_every_shard_ended():
    objects = _objects()
    a = FileSampleBuffer(objects, batch_size=1, index=shard_index("a"))
    b = FileSampleBuffer(objects, batch_size=1, index=shard_index("b"))
    a.append(_entry("a0"))
    a.mark_data_end()
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    assert len(trainer) == 1 and not trainer.data_ended
    b.append(_entry("b0", appended_at=1.0))
    b.mark_data_end()
    assert trainer.refresh() == 2 and trainer.data_ended


def test_a_clock_step_back_in_one_shard_keeps_that_shard_in_order():
    """The merged records of a shard stay a prefix of it, so a restarted merger
    (which resumes each shard after its merged count) neither skips nor repeats."""
    objects = _objects()
    a = FileSampleBuffer(objects, batch_size=1, index=shard_index("a"))
    a.append(_entry("a0", appended_at=5.0))
    a.append(_entry("a1", appended_at=1.0))  # the sampler host's clock stepped back
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    assert [s.entry.sample_id for s in trainer.select("t", 10)] == ["a0", "a1"]
    a.append(_entry("a2", appended_at=6.0))
    again = FileSampleBuffer(objects, batch_size=1, merge_shards=True)
    assert [s.entry.sample_id for s in again.select("t", 10)] == ["a0", "a1", "a2"]


def test_a_sampler_caps_its_backlog_on_its_own_shard(tmp_path):
    objects = _objects()
    store = CheckpointStore(tmp_path / "releases")
    releases = PolicyReleaseStore(store)
    follower = PolicyFollower(releases, lambda release, directory: None)
    shard = FileSampleBuffer(objects, batch_size=1, index=shard_index("npu"))
    loop = SamplerLoop(follower=follower, buffer=shard, generate=lambda r: None, num_rollouts=1,
                       max_backlog_groups=1, emit=lambda *a, **k: None)
    for i in range(3):
        shard.append(_entry(f"n{i}"))
    assert loop.backlog() == 3
    releases.release(1, "h1", lambda d: (d / "w").write_bytes(b"w"), buffer_selected=9,
                     buffer_shards={"npu": 2, "ppu": 7})
    follower.refresh()
    assert loop.backlog() == 1  # 3 written, 2 read in this shard; the merged 9 does not apply


# ---------------------------------------------------------------- N3.2 complete releases only
def test_a_reader_never_loads_a_partial_release(tmp_path):
    fake = FakeOss()
    trainer = store_for("/t", environ=_env(tmp_path / "trainer"), client=fake)
    sampler = store_for("/s", environ=_env(tmp_path / "sampler"), client=fake)
    releases = PolicyReleaseStore(trainer)
    loaded = []
    follower = PolicyFollower(PolicyReleaseStore(sampler, cache_dir=tmp_path / "cache"),
                              lambda release, directory: loaded.append(
                                  (release.version, (directory / "model.safetensors").read_bytes())))

    def weights(value):
        def write(directory):
            (directory / "model.safetensors").write_bytes(value)
            (directory / "config.json").write_text("{}")
        return write

    v1 = releases.release(1, "h1", weights(b"one"))
    assert v1.files == {"config.json": 2, "model.safetensors": 3}
    assert follower.refresh().version == 1
    # the next release's record and pointer are visible, one weight file is not yet (upload running)
    releases.release(2, "h2", weights(b"two-two"))
    key = f"runs/a/{RELEASE_DIR}/r2-v2/model.safetensors"
    held = fake.objects.pop(("bkt", key))
    assert follower.refresh() is None and follower.release.version == 1
    assert "1 of 2 files" in follower.incomplete
    fake.objects[("bkt", key)] = held[:3]  # a part of it: another size
    assert follower.refresh() is None and follower.release.version == 1
    fake.objects[("bkt", key)] = held
    assert follower.refresh().version == 2 and follower.incomplete is None
    assert loaded == [(1, b"one"), (2, b"two-two")]


def test_a_bf16_follower_keeps_the_w4a8_download_of_the_same_cache(tmp_path):
    fake = FakeOss()
    trainer = store_for("/t", environ=_env(tmp_path / "trainer"), client=fake)
    sampler = store_for("/s", environ=_env(tmp_path / "sampler"), client=fake)
    cache = tmp_path / "cache"
    for release_dir in (RELEASE_DIR, W4A8_RELEASE_DIR):
        PolicyReleaseStore(trainer, release_dir=release_dir).release(
            1, "h1", lambda d: (d / "w").write_bytes(b"w"))
    w4a8 = PolicyReleaseStore(sampler, cache_dir=cache, release_dir=W4A8_RELEASE_DIR)
    kept = w4a8.resolve(w4a8.latest())
    bf16 = PolicyReleaseStore(sampler, cache_dir=cache)
    bf16.resolve(bf16.latest())
    assert (kept / "w").is_file()


def test_a_mirror_uploads_data_then_the_release_record_then_the_pointer(tmp_path):
    uploads = []

    class Recording(FakeOss):
        def put_object(self, Bucket, Key, Body, **kw):
            uploads.append(Key.rsplit("/", 1)[-1])
            super().put_object(Bucket, Key, Body, **kw)

    root = tmp_path / "stage"
    release = root / RELEASE_DIR / "r1-v1"
    release.mkdir(parents=True)
    for name in ("a.bin", "z.bin", RECORD):
        (release / name).write_bytes(b"x")
    (root / RELEASE_DIR / LATEST).write_text("{}")
    TreeMirror(root, _objects(Recording()), subtrees=(), pointers=s3_store.POINTER_FILES,
               records=s3_store.RECORD_FILES).push()
    assert uploads == ["a.bin", "z.bin", RECORD, LATEST]


# ---------------------------------------------------------------- N3.4 large files
def test_a_large_sparse_file_goes_up_in_parts_and_comes_back_by_ranges(tmp_path):
    fake = FakeOss()
    objects = _objects(fake, part_bytes=MIN_PART_BYTES, concurrency=4)
    source = tmp_path / "shard.safetensors"
    size = 4 * MIN_PART_BYTES + 12345
    with open(source, "wb") as fh:  # sparse: holes between a few written blocks
        for offset in (0, MIN_PART_BYTES + 7, size - 5):
            fh.seek(offset)
            fh.write(os.urandom(5))
        fh.truncate(size)
    objects.write_file("policy-releases/r1-v1/shard.safetensors", source)
    assert fake.calls.count("upload_part") == 5 and "complete_multipart_upload" in fake.calls
    assert fake.uploads == {} and objects.size("policy-releases/r1-v1/shard.safetensors") == size
    target = tmp_path / "back"
    fake.calls.clear()
    assert objects.read_file("policy-releases/r1-v1/shard.safetensors", target)
    assert fake.calls.count("get_object") == 5 and target.read_bytes() == source.read_bytes()
    assert objects.read_file("missing", tmp_path / "none") is False


def test_a_failed_multipart_upload_is_aborted(tmp_path):
    class Failing(FakeOss):
        def upload_part(self, **kw):
            if kw["PartNumber"] == 2:
                raise _error("InternalError", "UploadPart")
            return super().upload_part(**kw)

    fake = Failing()
    source = tmp_path / "f"
    with open(source, "wb") as fh:
        fh.truncate(3 * MIN_PART_BYTES)
    with pytest.raises(Exception, match="InternalError"):
        _objects(fake, part_bytes=MIN_PART_BYTES).write_file("k", source)
    assert "abort_multipart_upload" in fake.calls and fake.uploads == {} and not fake.objects


def test_a_download_fails_when_the_object_is_replaced_midway(tmp_path):
    class Replacing(FakeOss):
        def get_object(self, Bucket, Key, Range=None, IfMatch=None):
            if Range and not Range.startswith("bytes=0-"):
                self.objects[(Bucket, Key)] = b"y" * len(self.objects[(Bucket, Key)])
            return super().get_object(Bucket, Key, Range=Range, IfMatch=IfMatch)

    fake = Replacing()
    objects = _objects(fake, part_bytes=MIN_PART_BYTES, concurrency=1)
    objects.write("k", b"x" * (2 * MIN_PART_BYTES))
    with pytest.raises(Exception, match="PreconditionFailed"):
        objects.read_file("k", tmp_path / "out")


def test_a_small_object_is_read_as_it_is_now_and_a_vanished_one_is_missing(tmp_path):
    class Rewriting(FakeOss):
        def get_object(self, Bucket, Key, Range=None, IfMatch=None):
            if Key.endswith("LATEST.json"):  # a pointer rewritten after the HEAD
                self.objects[(Bucket, Key)] = b"new"
            if Key.endswith("gone"):  # pruned after the HEAD
                self.objects.pop((Bucket, Key), None)
            return super().get_object(Bucket, Key, Range=Range, IfMatch=IfMatch)

    objects = _objects(Rewriting())
    objects.write("LATEST.json", b"old")
    assert objects.read_file("LATEST.json", tmp_path / "latest") and (tmp_path / "latest").read_bytes() == b"new"
    objects.write("gone", b"x")
    assert objects.read_file("gone", tmp_path / "gone") is False


def test_transfer_settings_keep_a_300_gb_file_within_the_part_limit():
    settings = TransferSettings.from_env({s3_store.PART_MB_ENV: "16", s3_store.PART_CONCURRENCY_ENV: "32"})
    assert (settings.part_bytes, settings.concurrency) == (16 << 20, 32)
    size = 301 * 10**9
    assert -(-size // settings.part_for(size)) <= s3_store.MAX_PARTS
    with pytest.raises(ValueError, match="below"):
        TransferSettings(part_bytes=1 << 20)


# ---------------------------------------------------------------- N3.3 launch + scheduler check
def test_store_reachability_by_island_cloud():
    assert store_unreachable_reason("oss://b/p", "nebius") is None
    assert store_unreachable_reason("s3://b/p", "aws") is None
    assert store_unreachable_reason("modal-volume://v", "modal") is None
    assert "only into Modal" in store_unreachable_reason("modal-volume://v", "nebius")
    assert "modal-volume://" in store_unreachable_reason("oss://b/p", "modal")
    assert "host path" in store_unreachable_reason("/shared/store", "nebius")
    assert "no object API" in store_unreachable_reason("gs://b", "gcp")


def test_an_oss_store_is_staged_on_the_island_and_gets_its_settings(monkeypatch):
    args = SimpleNamespace(rl_checkpoint_store=URI, rl_sample_role="sampler", rl_island_sample_role=None)
    assert rl_checkpoint_store_plan(args) == (f"{oss_store.DEFAULT_STAGE}/bkt/runs/a", None)
    for name, value in {oss_store.ENDPOINT_ENV: ENDPOINT, **CREDS}.items():
        monkeypatch.setenv(name, value)
    assert store_object_envs(args, "nebius")[oss_store.URI_ENV] == URI
    assert store_object_envs(args, "modal") == {}


def test_the_scheduler_takes_an_oss_store_that_the_trainer_and_the_sampler_reach(tmp_path, monkeypatch):
    from tests.test_rl_trainer_remote import _args
    from yeto.launch import trainer_loop as launch_loop

    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "1"])
    args.gpu, args.rl_checkpoint_store = "nebius:1xh100,nebius:1xh100", URI
    args.rl_island_sample_role = ["0:sampler", "1:trainer"]
    launch_loop.check(args)
    args.gpu = "modal:1xh100,nebius:1xh100"
    with pytest.raises(SystemExit, match=r"sampler island 0 \(modal\) does not reach"):
        launch_loop.check(args)
    args.rl_checkpoint_store = "/shared/store"
    with pytest.raises(SystemExit, match="a --rl-checkpoint-store the head reads"):
        launch_loop.check(args)
