"""An object API write above the single-PUT limit (CPU, fakes; no cloud resource).

A sample buffer group is one object (``ObjectIndex.write_payload`` ->
``S3ObjectStore.write``); a group of eight long agentic trials passes the 5 GiB
that S3, OSS and R2 take in one PUT. Above ``TransferSettings.multipart_bytes``
the write goes as a multipart upload, ``create`` keeps create-if-absent on its
CompleteMultipartUpload, and a reader sees the whole object or none. At or below
the threshold every call is the one PutObject of before. The fakes scale the
limits down (5 MiB parts)."""

from __future__ import annotations

import os

import pytest
from botocore.exceptions import ClientError
from test_rl_object_store import FakeS3, _error
from test_rl_oss_store import FakeOss

from yeto.cloud import oss_store, s3_store
from yeto.cloud.s3_store import MIN_PART_BYTES, S3ObjectStore, TransferSettings
from yeto.rl.engine.buffer_payload import GroupWriter, decode_group
from yeto.rl.engine.sample_buffer import FileSampleBuffer

URI = "s3://bkt/runs/a"
MIB = 1 << 20
THRESHOLD = 8 * MIB


class LimitedS3(FakeS3):
    """S3 with its single-request limit (EntityTooLarge above ``put_limit``) and
    the conditional CompleteMultipartUpload of S3 and R2 (``If-None-Match: *``,
    412 PreconditionFailed when the key exists, applied with the complete)."""

    def __init__(self, *, put_limit: int = THRESHOLD, conditional_complete: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.put_limit = put_limit
        self.conditional_complete = conditional_complete
        self.completes: list[dict] = []
        self.on_part = None  # callable(number): runs before a part is stored

    def put_object(self, Bucket, Key, Body, **kw):
        if len(Body) > self.put_limit:
            raise _error("EntityTooLarge", "PutObject")
        return super().put_object(Bucket=Bucket, Key=Key, Body=Body, **kw)

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        """As S3 answers a GET: the ETag, and a ranged one's ContentRange."""
        response = dict(super().get_object(Bucket=Bucket, Key=Key, Range=Range, IfMatch=IfMatch))
        response["ETag"] = self.object_etag(Bucket, Key)
        if Range is not None:
            first, last = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            size = len(self.objects[(Bucket, Key)])
            response["ContentRange"] = f"bytes {first}-{min(last, size - 1)}/{size}"
        return response

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, ChecksumSHA256=None):
        if self.on_part is not None:
            self.on_part(PartNumber)
        return super().upload_part(Bucket=Bucket, Key=Key, UploadId=UploadId, PartNumber=PartNumber,
                                   Body=Body, ChecksumSHA256=ChecksumSHA256)

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload, **condition):
        self.completes.append(dict(condition))
        if condition:
            if not self.conditional_complete:
                raise _error("NotImplemented", "CompleteMultipartUpload")
            if (Bucket, Key) in self.objects:
                raise _error("PreconditionFailed", "CompleteMultipartUpload")
        return super().complete_multipart_upload(Bucket=Bucket, Key=Key, UploadId=UploadId,
                                                 MultipartUpload=MultipartUpload)

    def multipart_calls(self) -> list[str]:
        return [c for c in self.calls if "multipart" in c or c == "upload_part"]


def _store(fake=None, **transfer) -> S3ObjectStore:
    settings = {"part_bytes": MIN_PART_BYTES, "multipart_bytes": THRESHOLD, **transfer}
    return S3ObjectStore(URI, client=fake or LimitedS3(), transfer=TransferSettings(**settings))


def _data(size: int) -> bytes:
    return os.urandom(size)


# ---------------------------------------------------------------- below the threshold: as before
@pytest.mark.parametrize("size", [0, 1, THRESHOLD])
def test_a_write_at_or_below_the_threshold_is_the_one_put_of_before(size):
    fake = LimitedS3()
    store = _store(fake)
    data = _data(size)
    store.write("sample-buffer/payloads/g", data)
    assert fake.calls == ["put_object"] and fake.objects[("bkt", "runs/a/sample-buffer/payloads/g")] == data
    assert fake.checksums[("bkt", "runs/a/sample-buffer/payloads/g")] is None  # no checksum argument either
    fake.calls.clear()
    assert store.create("k", data) and not store.create("k", b"x")
    assert fake.calls == ["put_object"] and fake.completes == []  # the lost create raised before storing


def test_the_default_threshold_keeps_every_group_below_1_gib_on_one_put():
    t = TransferSettings()
    assert t.multipart_bytes == s3_store.DEFAULT_MULTIPART_BYTES == 1 << 30
    assert TransferSettings.from_env({}).multipart_bytes == 1 << 30
    assert TransferSettings.from_env({s3_store.MULTIPART_MB_ENV: "256"}).multipart_bytes == 256 * MIB
    for bad in ("1", str(6 * 1024)):  # below one part; above what one PUT carries
        with pytest.raises(ValueError, match="multipart threshold"):
            TransferSettings.from_env({s3_store.MULTIPART_MB_ENV: bad})
    assert s3_store.MULTIPART_MB_ENV in s3_store.SETTING_ENVS
    assert s3_store.MULTIPART_MB_ENV in oss_store.SETTING_ENVS


# ---------------------------------------------------------------- above it: multipart
def test_a_write_above_the_threshold_goes_in_parts_and_reads_back_whole():
    fake = LimitedS3()
    store = _store(fake)
    data = _data(THRESHOLD + 3 * MIB + 17)  # 11 MiB + 17 B: parts of 5, 5, 1 MiB + 17 B
    with pytest.raises(ClientError, match="EntityTooLarge"):  # what main's one PUT does with it
        fake.put_object(Bucket="bkt", Key="runs/a/x", Body=data)
    fake.calls.clear()
    store.write("sample-buffer/payloads/g", data)
    assert "put_object" not in fake.calls
    assert fake.multipart_calls() == ["create_multipart_upload", *["upload_part"] * 3, "complete_multipart_upload"]
    assert fake.completes == [{}]  # a plain write: no condition
    assert store.read("sample-buffer/payloads/g") == data
    assert fake.uploads == {}
    # S3 with part checksums: the store's composite checksum is the parts' (verified by write)
    assert fake.checksums[("bkt", "runs/a/sample-buffer/payloads/g")].endswith("-3")


def test_parts_are_equal_but_the_last_and_stay_under_the_part_limit():
    fake = LimitedS3()
    store = _store(fake, part_checksum=False)
    sizes = []
    real = fake.upload_part
    fake.upload_part = lambda **kw: (sizes.append(len(kw["Body"])), real(**kw))[1]
    store.write("g", _data(3 * MIN_PART_BYTES + 1))
    assert sizes[:-1] == [MIN_PART_BYTES] * 3 and sizes[-1] == 1  # R2: all parts but the last the same size


def test_a_reader_never_sees_a_large_object_before_its_upload_completes():
    fake = LimitedS3()
    store = _store(fake)
    seen = []
    fake.on_part = lambda n: seen.append((store.exists("g"), store.read("g")))
    data = _data(2 * THRESHOLD)
    store.write("g", data)
    assert seen and all(s == (False, None) for s in seen)
    assert store.read("g") == data


def test_a_large_write_that_fails_leaves_no_object_and_aborts_its_upload():
    fake = LimitedS3()
    store = _store(fake, upload_attempts=1)

    def broken(number):
        if number == 2:
            raise _error("InternalError", "UploadPart")

    fake.on_part = broken
    with pytest.raises(ClientError, match="InternalError"):
        store.write("g", _data(2 * THRESHOLD))
    assert store.read("g") is None and fake.uploads == {} and "abort_multipart_upload" in fake.calls


def test_a_large_write_retries_a_failed_part_within_the_call():
    fake = LimitedS3()
    store = _store(fake)
    failed = []

    def once(number):
        if number == 2 and not failed:
            failed.append(number)
            raise _error("InternalError", "UploadPart")

    fake.on_part = once
    data = _data(2 * THRESHOLD)
    store.write("g", data)
    assert failed == [2] and store.read("g") == data and fake.uploads == {}


def test_an_endpoint_without_part_checksums_gets_the_large_write_without_them():
    class NoChecksums(LimitedS3):
        def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, ChecksumSHA256=None):
            if ChecksumSHA256 is not None:
                raise _error("NotImplemented", "UploadPart")
            return super().upload_part(Bucket=Bucket, Key=Key, UploadId=UploadId, PartNumber=PartNumber,
                                       Body=Body)

    fake = NoChecksums()
    store = _store(fake)
    data = _data(THRESHOLD + 1)
    store.write("g", data)
    assert store.read("g") == data and store.part_checksums_refused and fake.uploads == {}


# ---------------------------------------------------------------- create-if-absent above the threshold
def test_a_large_create_puts_the_condition_on_the_complete_and_loses_to_an_existing_key():
    fake = LimitedS3()
    store = _store(fake)
    first, second = _data(THRESHOLD + 1), _data(THRESHOLD + 2)
    assert store.create("g", first) is True
    assert fake.completes == [{"IfNoneMatch": "*"}]
    assert store.create("g", second) is False
    assert store.read("g") == first  # never replaced
    assert fake.uploads == {} and "abort_multipart_upload" in fake.calls
    assert store.conditional_writes


def test_a_large_create_loses_to_a_writer_that_lands_while_its_parts_go_up():
    fake = LimitedS3()
    store = _store(fake)
    other = _data(10)

    def race(number):
        if number == 1 and ("bkt", "runs/a/g") not in fake.objects:
            fake.objects[("bkt", "runs/a/g")] = other  # another host's create wins meanwhile

    fake.on_part = race
    assert store.create("g", _data(THRESHOLD + 1)) is False
    assert store.read("g") == other and fake.uploads == {}


def test_a_large_create_whose_complete_answer_was_lost_is_this_writers():
    """The complete applied, its answer was lost; the SDK's retry is refused (the
    key exists now): the object is these parts, so the create is this one's."""

    class LostAnswer(LimitedS3):
        def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload, **condition):
            super().complete_multipart_upload(Bucket, Key, UploadId, MultipartUpload)
            raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "exists"},
                               "ResponseMetadata": {"HTTPStatusCode": 412, "RetryAttempts": 1}},
                              "CompleteMultipartUpload")

    fake = LostAnswer()
    store = _store(fake, part_checksum=False)  # R2: the multipart ETag proves it
    data = _data(THRESHOLD + 5)
    assert store.create("g", data) is True and store.read("g") == data


def test_a_large_create_whose_retried_complete_finds_no_upload_is_this_writers():
    class Gone(LimitedS3):
        def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload, **condition):
            super().complete_multipart_upload(Bucket, Key, UploadId, MultipartUpload)
            raise _error("NoSuchUpload", "CompleteMultipartUpload")  # R2's answer to a completed upload

    fake = Gone()
    store = _store(fake)
    data = _data(THRESHOLD + 5)
    assert store.create("g", data) is True and store.read("g") == data


def test_a_large_create_on_an_endpoint_without_the_condition_falls_back_to_check_then_write():
    fake = LimitedS3(conditional_complete=False)
    store = _store(fake)
    data = _data(THRESHOLD + 1)
    assert store.create("g", data) is True
    assert not store.conditional_writes and fake.uploads == {}
    assert store.read("g") == data
    assert store.create("g", _data(THRESHOLD + 2)) is False and store.read("g") == data


def test_oss_completes_a_large_create_with_forbid_overwrite():
    class Oss(FakeOss):
        completes: list

        def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload, **condition):
            type(self).completes.append(dict(condition))
            if condition.get("ForbidOverwrite") and (Bucket, Key) in self.objects:
                raise _error("FileAlreadyExists", "CompleteMultipartUpload")
            return FakeS3.complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload)

    Oss.completes = []
    objects = oss_store.OssObjectStore("oss://bkt/runs/a", client=Oss(), transfer=TransferSettings(
        part_bytes=MIN_PART_BYTES, multipart_bytes=THRESHOLD))
    first = _data(THRESHOLD + 1)
    assert objects.create("g", first) is True and objects.create("g", _data(THRESHOLD + 1)) is False
    assert Oss.completes == [{"ForbidOverwrite": True}] * 2 and objects.read("g") == first
    assert objects.create("small", b"1") and not objects.create("small", b"2")  # small: one PUT, as before


# ---------------------------------------------------------------- the buffer group path end to end
def test_a_group_above_the_threshold_reaches_the_trainer_through_the_s3_buffer():
    from tests.buffer_rollout_procs import sample

    fake = LimitedS3()
    objects = _store(fake)
    writer_buffer = FileSampleBuffer(objects, batch_size=1)
    group = [sample(i, 0, 3) for i in range(2)]
    big = "x" * (THRESHOLD + MIB)  # a long prompt text: the payload passes the threshold
    group = [{**s, "prompt": big} for s in group]
    entry = GroupWriter(writer_buffer, "s0").write(group, group_key="r0-g0")
    assert "put_object" in fake.calls  # the index record (small): a conditional PUT
    assert fake.completes == [{}]  # the payload: in parts
    reader = FileSampleBuffer(objects, batch_size=1)
    [selected] = reader.select("trainer", 1)
    assert selected.entry.payload_ref == entry.payload_ref
    assert decode_group(reader.read_payload(entry.payload_ref))[1]["prompt"] == big
    assert reader.payload_size(entry.payload_ref) == objects.size(f"sample-buffer/{entry.payload_ref}")


# ---------------------------------------------------------------- v1 and v2 payloads, ranged reads
def _r3_group(rows: int = 3000) -> list[dict]:
    """Two K3.1-like samples whose R3 (uint16 [rows, 92, 16], 2944 B a row) makes the
    payload pass the threshold in every format."""
    import numpy

    from yeto.rl.engine.buffer_payload import ROLLOUT_ROUTED_EXPERTS

    rng = numpy.random.default_rng(3)
    return [{"index": i, "group_index": 0, "tokens": list(range(rows + 1)), "response_length": 3,
             "loss_mask": [1, 1, 1], "rollout_log_probs": [-0.1, -0.2, -0.3], "reward": float(i),
             "status": "completed", "weight_versions": [[{"version": "2", "abs_start": rows - 2, "abs_end": rows + 1}]],
             ROLLOUT_ROUTED_EXPERTS: rng.integers(0, 896, (rows, 92, 16)).astype(numpy.uint16)} for i in range(2)]


@pytest.mark.parametrize("codec", ["json", "raw", "zstd"])
def test_v1_and_v2_groups_above_the_threshold_go_up_in_parts_and_come_back_by_ranges(monkeypatch, codec):
    import numpy

    from yeto.rl.engine.buffer_payload import PAYLOAD_CODEC_ENV, ROLLOUT_ROUTED_EXPERTS

    monkeypatch.setenv(PAYLOAD_CODEC_ENV, codec)
    fake = LimitedS3()
    objects = _store(fake)
    group = _r3_group()
    entry = GroupWriter(FileSampleBuffer(objects, batch_size=1), "s0").write(group, group_key="r0-g0")
    key = ("bkt", f"runs/a/sample-buffer/{entry.payload_ref}")
    size = len(fake.objects[key])
    assert size > THRESHOLD and fake.completes == [{}]
    assert fake.multipart_etags[key].endswith(f'-{-(-size // MIN_PART_BYTES)}"')
    reader = FileSampleBuffer(objects, batch_size=1)
    ranges, real = [], fake.get_object
    fake.get_object = lambda **kw: (ranges.append(kw.get("Range")), real(**kw))[1]
    fake.calls.clear()
    data = reader.read_payload(entry.payload_ref, size=reader.payload_size(entry.payload_ref))
    assert len(ranges) == -(-size // MIN_PART_BYTES) and None not in ranges  # one ranged GET a part
    assert "head_object" not in fake.calls  # the size came from the caller
    assert bytes(data) == fake.objects[key]
    back = decode_group(data)
    for want, got in zip(group, back):
        assert numpy.array_equal(got[ROLLOUT_ROUTED_EXPERTS], want[ROLLOUT_ROUTED_EXPERTS])
        assert got["reward"] == want["reward"]


@pytest.mark.parametrize("sized", [False, True], ids=["head", "size-given"])
def test_a_ranged_read_is_pinned_to_one_version_of_the_object(sized):
    fake = LimitedS3()
    store = _store(fake)
    store.write("g", _data(3 * MIN_PART_BYTES))
    real = fake.get_object

    def replaced_midway(**kw):
        if kw.get("Range", "").startswith(f"bytes={MIN_PART_BYTES}-"):
            fake.objects[("bkt", "runs/a/g")] = _data(3 * MIN_PART_BYTES)  # rewritten meanwhile
            fake.multipart_etags.pop(("bkt", "runs/a/g"), None)
        return real(**kw)

    fake.get_object = replaced_midway
    with pytest.raises(ClientError, match="PreconditionFailed"):
        S3ObjectStore(URI, client=fake, transfer=TransferSettings(part_bytes=MIN_PART_BYTES, concurrency=1)
                      ).read_ranged("g", 3 * MIN_PART_BYTES if sized else None)


def test_a_ranged_read_with_a_stale_size_reads_the_object_whole():
    fake = LimitedS3()
    store = _store(fake)
    data = _data(3 * MIN_PART_BYTES + 9)
    store.write("g", data)
    assert store.read_ranged("g", 2 * MIN_PART_BYTES + 1) == data
    assert store.read_ranged("missing", 2 * MIN_PART_BYTES) is None


def test_a_ranged_read_retries_a_cut_short_range_only(monkeypatch):
    fake = LimitedS3()
    store = _store(fake)
    data = _data(3 * MIN_PART_BYTES)
    store.write("g", data)
    real, cut = fake.get_object, []

    def reset_once(**kw):
        response = real(**kw)
        if kw.get("Range", "").startswith(f"bytes={MIN_PART_BYTES}-") and not cut:
            cut.append(kw["Range"])

            class Reset:
                def read(self):
                    raise ConnectionResetError("reset by peer")

            return {"Body": Reset()}
        return response

    fake.get_object = reset_once
    monkeypatch.setattr(s3_store, "_read_retry_pause", lambda attempt: None)
    fake.calls.clear()
    assert store.read_ranged("g", len(data)) == data and len(cut) == 1
    assert fake.calls.count("get_object") == 4  # three parts, the reset one twice: not the object again


def test_a_small_payload_is_read_with_one_get_after_the_head():
    fake = LimitedS3()
    store = _store(fake)
    store.write("g", b"small")
    fake.calls.clear()
    assert store.read_ranged("g") == b"small" and fake.calls == ["get_object"]
    assert store.read_ranged("missing") is None
