"""Resumable, checksummed multipart uploads of the s3 object store
(``S3ObjectStore.write_file``) and the cut shard checksums the driver checks before
its commit (``cut.uploaded_shard_checksums``). CPU only, over the fake S3 client."""
from __future__ import annotations

import json

import pytest
from test_rl_multinode_staged_cut import _env, _store, _summary, _write_shard
from test_rl_object_store import URI, FakeS3, _error

from yeto.cloud import s3_store
from yeto.cloud.s3_store import MIN_PART_BYTES, UPLOAD_JOURNAL_SUFFIX, S3ObjectStore, TransferSettings, composite_sha256
from yeto.rl.engine.cut import (
    CutError,
    checksum_problems,
    push_shard_from_rank,
    uploaded_shard_checksums,
)
from yeto.rl.engine.object_store import ObjectStoreError, TreeMirror

KEY = "round-cuts/r1/trainer_tp0_pp0_dp0.pt"


def _objects(fake, **transfer):
    return S3ObjectStore(URI, client=fake, transfer=TransferSettings(part_bytes=MIN_PART_BYTES, **transfer))


def _source(tmp_path, parts=4, tail=123):
    path = tmp_path / "shard.pt"
    block = bytes(range(256)) * (MIN_PART_BYTES // 256)
    path.write_bytes(b"".join(bytes([i]) + block[1:] for i in range(parts - 1)) + b"z" * tail)
    return path


def _full(key):
    bucket, prefix = s3_store.parse_s3_uri(URI)
    return bucket, f"{prefix}/{key}" if prefix else key


def _journal(source):
    return source.with_name(source.name + UPLOAD_JOURNAL_SUFFIX)


class Flaky(FakeS3):
    """Fails the upload of chosen part numbers (once each, or always)."""

    def __init__(self, fail=(), always=False):
        super().__init__()
        self.fail = set(fail)
        self.always = always

    def upload_part(self, **kw):
        if kw["PartNumber"] in self.fail:
            if not self.always:
                self.fail.discard(kw["PartNumber"])
            raise _error("InternalError", "UploadPart")
        return super().upload_part(**kw)


# ---------------------------------------------------------------- checksums
def test_every_part_carries_its_sha256_and_the_composite_checksum_is_checked(tmp_path):
    fake, source = FakeS3(), _source(tmp_path)
    returned = _objects(fake).write_file(KEY, source)
    assert fake.objects[_full(KEY)] == source.read_bytes()
    assert returned == fake.checksums[_full(KEY)] and returned.endswith("-4")
    data = source.read_bytes()
    parts = [FakeS3.sha(data[i:i + MIN_PART_BYTES]) for i in range(0, len(data), MIN_PART_BYTES)]
    assert returned == composite_sha256(parts) == _objects(fake).checksum(KEY)
    assert not _journal(source).exists()


def test_a_small_file_is_one_put_with_its_sha256(tmp_path):
    fake = FakeS3()
    source = tmp_path / "small"
    source.write_bytes(b"abc")
    assert _objects(fake).write_file("k", source) == FakeS3.sha(b"abc") == _objects(fake).checksum("k")


def test_a_part_corrupted_on_the_way_is_refused_and_sent_again(tmp_path):
    class Corrupting(FakeS3):
        hit = False

        def upload_part(self, **kw):
            if kw["PartNumber"] == 2 and not self.hit:
                self.hit = True
                kw["Body"] = b"x" + bytes(kw["Body"])[1:]  # the store checks it against ChecksumSHA256
            return super().upload_part(**kw)

    fake, source = Corrupting(), _source(tmp_path)
    _objects(fake).write_file(KEY, source)
    assert fake.hit and fake.objects[_full(KEY)] == source.read_bytes()
    assert fake.calls.count("create_multipart_upload") == 1


def test_a_composite_checksum_the_store_reports_differently_fails_the_upload(tmp_path):
    class Wrong(FakeS3):
        def head_object(self, Bucket, Key, ChecksumMode=None):
            out = super().head_object(Bucket, Key, ChecksumMode=ChecksumMode)
            if "ChecksumSHA256" in out:
                out["ChecksumSHA256"] = "AAAA-4"
            return out

    with pytest.raises(ObjectStoreError, match="reports checksum AAAA-4"):
        _objects(Wrong()).write_file(KEY, _source(tmp_path))


def test_checksums_off_sends_none(tmp_path):
    fake, source = FakeS3(), _source(tmp_path)
    objects = S3ObjectStore(URI, client=fake, transfer=TransferSettings.from_env(
        {s3_store.PART_MB_ENV: "5", s3_store.PART_CHECKSUM_ENV: "0"}))
    assert objects.write_file(KEY, source) is None
    assert fake.upload_checksum == {"u0": None} and objects.checksum(KEY) is None
    assert fake.objects[_full(KEY)] == source.read_bytes()


# ---------------------------------------------------------------- resume
def test_a_failed_part_is_retried_in_the_same_call_without_resending_the_others(tmp_path):
    fake, source = Flaky(fail={3}), _source(tmp_path)
    _objects(fake).write_file(KEY, source)
    assert fake.calls.count("create_multipart_upload") == 1
    assert fake.calls.count("upload_part") == 4  # 1, 2, 4, then 3 again
    assert "list_parts" in fake.calls and fake.objects[_full(KEY)] == source.read_bytes()


def test_an_upload_cut_short_continues_in_the_next_call_from_the_journal(tmp_path):
    fake, source = Flaky(fail={2, 4}, always=True), _source(tmp_path)
    with pytest.raises(Exception, match="InternalError"):
        _objects(fake, upload_attempts=2).write_file(KEY, source)
    journal = json.loads(_journal(source).read_text())
    assert journal["upload_id"] == "u0" and sorted(journal["parts"]) == ["1", "3"]
    assert fake.uploads["u0"].keys() == {1, 3} and not fake.objects
    fake.fail.clear()
    fake.calls.clear()
    returned = _objects(fake).write_file(KEY, source)  # a new store object: a restarted process
    assert fake.calls.count("upload_part") == 2 and "create_multipart_upload" not in fake.calls
    assert fake.objects[_full(KEY)] == source.read_bytes() and returned == fake.checksums[_full(KEY)]
    assert not _journal(source).exists() and fake.uploads == {}


def test_a_part_the_store_holds_differently_from_the_journal_is_sent_again(tmp_path):
    fake, source = Flaky(fail={4}, always=True), _source(tmp_path)
    with pytest.raises(Exception, match="InternalError"):
        _objects(fake, upload_attempts=1).write_file(KEY, source)
    fake.uploads["u0"][2] = b"y" * MIN_PART_BYTES  # not what the journal recorded
    fake.fail.clear()
    fake.calls.clear()
    _objects(fake).write_file(KEY, source)
    assert fake.calls.count("upload_part") == 2  # part 2 (disagrees) and part 4 (missing)
    assert fake.objects[_full(KEY)] == source.read_bytes()


def test_a_journal_of_another_version_of_the_file_aborts_its_upload(tmp_path):
    fake, source = Flaky(fail={2}, always=True), _source(tmp_path)
    with pytest.raises(Exception, match="InternalError"):
        _objects(fake, upload_attempts=1).write_file(KEY, source)
    source.write_bytes(b"new" + source.read_bytes()[3:])  # rewritten: new mtime
    fake.fail.clear()
    fake.calls.clear()
    _objects(fake).write_file(KEY, source)
    assert fake.calls[:2] == ["abort_multipart_upload", "create_multipart_upload"]
    assert fake.calls.count("upload_part") == 4 and fake.objects[_full(KEY)] == source.read_bytes()


def test_an_upload_gone_from_the_store_starts_again(tmp_path):
    fake, source = Flaky(fail={2}, always=True), _source(tmp_path)
    with pytest.raises(Exception, match="InternalError"):
        _objects(fake, upload_attempts=1).write_file(KEY, source)
    fake.uploads.clear()  # expired by a lifecycle rule
    fake.fail.clear()
    _objects(fake).write_file(KEY, source)
    assert fake.calls.count("create_multipart_upload") == 2 and fake.objects[_full(KEY)] == source.read_bytes()


def test_the_journal_is_never_pushed_as_a_store_file(tmp_path):
    fake = Flaky(fail={2}, always=True)
    root = tmp_path / "stage"
    shard = root / "round-cuts" / "r1" / "shard.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(_source(tmp_path).read_bytes())
    mirror = TreeMirror(root, _objects(fake, upload_attempts=1), subtrees=("round-cuts/",), pointers=frozenset())
    with pytest.raises(Exception, match="InternalError"):
        mirror.push()
    assert _journal(shard).is_file()
    fake.fail.clear()
    mirror.push()
    assert sorted(k for _, k in fake.objects) == [_full("round-cuts/r1/shard.pt")[1]]


# ---------------------------------------------------------------- cut shards
def test_a_rank_reports_the_checksum_and_the_driver_checks_it_again(tmp_path):
    s3 = FakeS3()
    head = _store(tmp_path, "node-a", s3)
    final = _write_shard(head, 0)
    pushed = push_shard_from_rank(final, environ=_env(tmp_path, "node-a"), store_kwargs={"client": s3})
    assert pushed["object_checksum"] == FakeS3.sha(final.read_bytes())
    summaries = [_summary(final, pushed)]
    checks = uploaded_shard_checksums(head, summaries)
    assert checks == {final.name: (pushed["object_checksum"],) * 2} and checksum_problems(checks) == []
    s3.checksums[(head.objects.bucket, head.objects._key(pushed["object_key"]))] = "BBBB"  # replaced since
    problems = checksum_problems(uploaded_shard_checksums(head, summaries))
    assert problems and "reports checksum BBBB" in problems[0]


def test_commit_manifest_refuses_a_shard_whose_checksum_changed(tmp_path):
    from test_rl_multinode_staged_cut import CUT, _cut_manifest
    from yeto.rl.engine.cut import MANIFEST_NAME, commit_manifest, cut_dir
    from yeto.rl.engine.resume import ROUND_CUTS

    s3 = FakeS3()
    head = _store(tmp_path, "node-a", s3)
    summaries = []
    for dp in (0, 1):
        final = _write_shard(head, dp)
        summaries.append(_summary(final, push_shard_from_rank(
            final, environ=_env(tmp_path, "node-a"), store_kwargs={"client": s3})))
    manifest = _cut_manifest(head, summaries)
    good = uploaded_shard_checksums(head, summaries)
    bad = {**good, summaries[1]["path"]: (good[summaries[1]["path"]][0], "CCCC")}
    with pytest.raises(CutError, match="reports checksum CCCC"):
        commit_manifest(head.root / ROUND_CUTS, manifest, remote_checksums=bad)
    assert not (cut_dir(head.root / ROUND_CUTS, CUT) / MANIFEST_NAME).exists()
    commit_manifest(head.root / ROUND_CUTS, manifest, remote_checksums=good)
    assert (cut_dir(head.root / ROUND_CUTS, CUT) / MANIFEST_NAME).is_file()


# ---------------------------------------------------------------- R2: part checksums refused
class NoPartChecksums(FakeS3):
    """Cloudflare R2 (measured): UploadPart with x-amz-checksum-sha256 answers
    NotImplemented; a single put with one is accepted."""

    def upload_part(self, **kw):
        if kw.get("ChecksumSHA256") is not None:
            raise _error("NotImplemented", "UploadPart")
        return super().upload_part(**kw)


@pytest.mark.parametrize("value, endpoint, expected", [
    (None, None, True),
    ("auto", "https://s3.eu-north-1.amazonaws.com", True),
    (None, "https://0123abcd.r2.cloudflarestorage.com", False),
    ("auto", "https://0123abcd.eu.r2.cloudflarestorage.com", False),
    ("1", "https://0123abcd.r2.cloudflarestorage.com", True),
    ("0", None, False),
    ("off", None, False),
])
def test_the_part_checksum_default_is_off_on_r2_and_on_elsewhere(value, endpoint, expected):
    env = {**({s3_store.PART_CHECKSUM_ENV: value} if value else {}),
           **({s3_store.ENDPOINT_ENV: endpoint} if endpoint else {})}
    assert TransferSettings.from_env(env).part_checksum is expected


def test_an_r2_endpoint_given_as_aws_endpoint_url_also_turns_checksums_off():
    env = {"AWS_ENDPOINT_URL": "https://0123abcd.r2.cloudflarestorage.com"}
    assert TransferSettings.from_env(env).part_checksum is False


def test_an_unknown_part_checksum_value_is_refused():
    with pytest.raises(ValueError, match="expected 1, 0 or auto"):
        TransferSettings.from_env({s3_store.PART_CHECKSUM_ENV: "maybe"})


def test_not_implemented_turns_checksums_off_once_with_one_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s3_store, "_PART_CHECKSUM_WARNED", False)
    fake, source = NoPartChecksums(), _source(tmp_path)
    objects = _objects(fake)
    assert objects.write_file(KEY, source) is None
    assert fake.objects[_full(KEY)] == source.read_bytes() and fake.uploads == {}
    assert objects.part_checksums_refused and objects.checksum(KEY) is None
    assert not _journal(source).exists()
    assert fake.calls.count("create_multipart_upload") == 2 and "abort_multipart_upload" in fake.calls
    other = tmp_path / "other"
    other.write_bytes(source.read_bytes()[::-1])
    assert objects.write_file("round-cuts/r1/other.pt", other) is None
    assert fake.calls.count("create_multipart_upload") == 3  # no second probe on this store
    _objects(NoPartChecksums()).write_file(KEY, source)  # another store of the process: no second warning
    err = capsys.readouterr().err
    assert err.count("does not take SHA-256 part checksums") == 1
    assert s3_store.PART_CHECKSUM_ENV in err


def test_a_not_implemented_failure_without_checksums_is_raised(tmp_path):
    class Refusing(FakeS3):
        def upload_part(self, **kw):
            raise _error("NotImplemented", "UploadPart")

    with pytest.raises(Exception, match="NotImplemented"):
        _objects(Refusing(), part_checksum=False).write_file(KEY, _source(tmp_path))


# ---------------------------------------------------------------- a complete whose answer was lost
class LostComplete(FakeS3):
    """The first CompleteMultipartUpload raises ``error``: after it took effect
    (``applied``, the answer lost on the way back) or instead of it. A complete of an
    upload the store does not hold any more is ``NoSuchUpload``, as on S3 and R2."""

    def __init__(self, error, *, applied=True):
        super().__init__()
        self.error = error
        self.applied = applied

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        if UploadId not in self.uploads:
            raise _error("NoSuchUpload", "CompleteMultipartUpload")
        if self.error is None:
            return super().complete_multipart_upload(Bucket, Key, UploadId, MultipartUpload)
        error, self.error = self.error, None
        if self.applied:
            super().complete_multipart_upload(Bucket, Key, UploadId, MultipartUpload)
        raise error


# what the caller sees: the SDK's retries of a 5xx used up, or (measured on R2) the
# SDK's own second send of a complete whose first answer timed out
LOST_ANSWERS = {"retries exhausted": lambda: _error("ServiceUnavailable", "CompleteMultipartUpload"),
                "the sdk's retry found the upload gone": lambda: _error("NoSuchUpload", "CompleteMultipartUpload")}


@pytest.mark.parametrize("answer", sorted(LOST_ANSWERS))
@pytest.mark.parametrize("part_checksum", [False, True])  # False: R2 (the multipart ETag is the proof)
def test_a_completed_upload_whose_answer_was_lost_is_not_uploaded_again(tmp_path, answer, part_checksum):
    fake, source = LostComplete(LOST_ANSWERS[answer]()), _source(tmp_path)
    returned = _objects(fake, part_checksum=part_checksum).write_file(KEY, source)
    assert fake.calls.count("upload_part") == 4 and fake.calls.count("create_multipart_upload") == 1
    assert fake.calls.count("complete_multipart_upload") == 1
    assert fake.objects[_full(KEY)] == source.read_bytes()
    assert returned == (fake.checksums[_full(KEY)] if part_checksum else None)
    assert not _journal(source).exists() and fake.uploads == {}


@pytest.mark.parametrize("part_checksum", [False, True])
def test_a_completed_upload_whose_answer_was_lost_still_ends_as_the_whole_object(tmp_path, part_checksum):
    fake, source = LostComplete(_error("ServiceUnavailable", "CompleteMultipartUpload")), _source(tmp_path)
    _objects(fake, part_checksum=part_checksum).write_file(KEY, source)
    assert fake.objects[_full(KEY)] == source.read_bytes()


@pytest.mark.parametrize("part_checksum", [False, True])
def test_the_last_try_of_an_upload_also_recognises_its_lost_complete(tmp_path, part_checksum):
    fake, source = LostComplete(_error("ServiceUnavailable", "CompleteMultipartUpload")), _source(tmp_path)
    _objects(fake, part_checksum=part_checksum, upload_attempts=1).write_file(KEY, source)
    assert fake.calls.count("upload_part") == 4 and not _journal(source).exists()


def test_a_process_killed_between_the_complete_and_its_answer_does_not_upload_again(tmp_path):
    class Killed(BaseException):
        pass

    fake, source = LostComplete(Killed()), _source(tmp_path)
    with pytest.raises(Killed):
        _objects(fake, part_checksum=False).write_file(KEY, source)
    assert _journal(source).exists() and fake.objects[_full(KEY)] == source.read_bytes()
    fake.calls.clear()
    _objects(fake, part_checksum=False).write_file(KEY, source)  # the restarted process
    assert "upload_part" not in fake.calls and "create_multipart_upload" not in fake.calls
    assert not _journal(source).exists() and fake.objects[_full(KEY)] == source.read_bytes()


@pytest.mark.parametrize("part_checksum", [False, True])
@pytest.mark.parametrize("left", ["nothing", "another object of the same size", "the same bytes as one put"])
def test_an_upload_that_is_gone_without_its_object_is_uploaded_again(tmp_path, left, part_checksum):
    """Aborted or expired after its last part, not completed: whatever the key holds
    is not this upload unless its ETag / composite checksum says so."""
    fake, source = LostComplete(_error("InternalError", "CompleteMultipartUpload"), applied=False), _source(tmp_path)
    real = fake.complete_multipart_upload

    def complete(**kw):
        try:
            return real(**kw)
        except Exception:
            fake.uploads.clear()  # aborted by someone else / expired by a lifecycle rule
            data = source.read_bytes()
            if left == "another object of the same size":
                fake.put_object(Bucket=kw["Bucket"], Key=kw["Key"], Body=b"\0" * len(data))
            elif left == "the same bytes as one put":  # cannot be told from this upload's parts: sent again
                fake.put_object(Bucket=kw["Bucket"], Key=kw["Key"], Body=data)
            raise

    fake.complete_multipart_upload = complete
    _objects(fake, part_checksum=part_checksum).write_file(KEY, source)
    assert fake.calls.count("upload_part") == 8 and fake.calls.count("create_multipart_upload") == 2
    assert fake.objects[_full(KEY)] == source.read_bytes() and not _journal(source).exists()


def test_a_still_failing_store_after_the_last_try_raises_the_upload_error_and_keeps_the_journal(tmp_path):
    fake, source = LostComplete(_error("InternalError", "CompleteMultipartUpload"), applied=False), _source(tmp_path)

    def list_parts(**kw):
        raise _error("ServiceUnavailable", "ListParts")

    fake.list_parts = list_parts
    with pytest.raises(Exception, match="InternalError"):
        _objects(fake, upload_attempts=1).write_file(KEY, source)
    assert _journal(source).exists() and "u0" in fake.uploads


def test_the_multipart_etag_is_the_md5_of_the_part_md5s_and_only_of_md5s():
    import hashlib

    bodies = [b"a" * 10, b"b" * 10, b"tail"]
    etags = [FakeS3.etag(b) for b in bodies]
    expected = hashlib.md5(b"".join(hashlib.md5(b).digest() for b in bodies)).hexdigest() + "-3"
    assert s3_store.multipart_etag(etags) == s3_store.multipart_etag([e.strip('"') for e in etags]) == expected
    assert s3_store.multipart_etag([FakeS3.etag(b"one")]).endswith("-1")
    # an SSE-KMS / OSS style ETag is no MD5: nothing can be derived, so nothing is confirmed
    for odd in ([], [None], ['"abc"'], [etags[0], '"' + "z" * 32 + '"'], [etags[0], '"0123-2"']):
        assert s3_store.multipart_etag(odd) is None


@pytest.mark.parametrize("part_checksum", [False, True])
def test_a_part_cut_off_by_a_closed_connection_is_sent_again_like_any_failed_part(tmp_path, part_checksum):
    from botocore.exceptions import ConnectionClosedError

    class Closing(FakeS3):
        closed = False

        def upload_part(self, **kw):
            if kw["PartNumber"] == 2 and not self.closed:
                self.closed = True
                raise ConnectionClosedError(endpoint_url="https://store")  # its ``response`` is None
            return super().upload_part(**kw)

    fake, source = Closing(), _source(tmp_path)
    _objects(fake, part_checksum=part_checksum).write_file(KEY, source)
    assert fake.calls.count("upload_part") == 4 and fake.calls.count("create_multipart_upload") == 1
    assert fake.objects[_full(KEY)] == source.read_bytes()
