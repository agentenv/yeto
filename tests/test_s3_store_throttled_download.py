"""A range read the store refuses for its request rate lowers the range reads the
process runs at once and is tried again (``yeto.cloud.s3_store.RangeGate``)."""

from __future__ import annotations

import threading

import pytest
from botocore.exceptions import ClientError

from tests.test_rl_object_store import URI, FakeS3
from yeto.cloud import s3_store


def _refusal(code="ServiceUnavailable", status=503):
    return ClientError({"Error": {"Code": code, "Message": "Reduce your concurrent request rate for the same object."},
                        "ResponseMetadata": {"HTTPStatusCode": status, "RetryAttempts": 8}}, "GetObject")


class Limited(FakeS3):
    """Refuses a range read whenever more than ``allowed`` are in flight."""

    def __init__(self, allowed):
        super().__init__()
        self.allowed, self.flying, self.most, self.refused = allowed, 0, 0, 0
        self.gate = threading.Lock()

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):  # noqa: N803
        if Range is None:
            return super().get_object(Bucket, Key, Range, IfMatch)
        with self.gate:
            self.flying += 1
            self.most = max(self.most, self.flying)
            over = self.flying > self.allowed
            if over:
                self.refused += 1
        try:
            if over:
                raise _refusal()
            threading.Event().wait(0.002)  # the read takes a moment: reads overlap
            return super().get_object(Bucket, Key, Range, IfMatch)
        finally:
            with self.gate:
                self.flying -= 1


def _store(s3, ranges, concurrency):
    part = s3_store.MIN_PART_BYTES
    store = s3_store.S3ObjectStore(URI, client=s3, transfer=s3_store.TransferSettings(part_bytes=part,
                                                                                    concurrency=concurrency))
    data = bytes(range(251)) * (ranges * part // 251 + 1)
    s3.objects[("bkt", store._key("big"))] = data[:ranges * part]
    return store, data[:ranges * part]


def test_throttling_is_told_from_other_refusals():
    assert s3_store.is_throttled(_refusal())
    assert s3_store.is_throttled(_refusal("SlowDown"))
    assert s3_store.is_throttled(_refusal("TooManyRequests", 429))
    assert s3_store.is_throttled(_refusal("Unknown", 503))
    assert not s3_store.is_throttled(_refusal("PreconditionFailed", 412))
    assert not s3_store.is_throttled(_refusal("AccessDenied", 403))
    assert not s3_store.is_throttled(OSError("connection reset"))


def test_the_gate_halves_on_a_refusal_and_gives_reads_back_one_by_one():
    gate = s3_store.RangeGate()
    for _ in range(8):
        gate.__enter__()
    assert gate.limit is None and gate.active == 8
    assert gate.refused() == 4 and gate.refused() == 2 and gate.cuts == 2
    for _ in range(8):
        gate.__exit__()
    assert gate.refused() == 1 and gate.refused() == 1  # never below one
    limits = []
    for _ in range(8 * s3_store.RANGE_RESTORE_READS):
        with gate:
            pass
        gate.succeeded()
        limits.append(gate.limit)
    assert limits[::s3_store.RANGE_RESTORE_READS] == [1, 2, 3, 4, 5, 6, 7, 8] and limits[-1] == 9


def test_a_download_refused_for_its_rate_slows_down_and_completes(tmp_path, monkeypatch, capsys):
    """main: the first range refused after botocore's retries failed the whole file
    (and with it the release download of that sampler)."""
    s3 = Limited(allowed=3)
    store, data = _store(s3, ranges=16, concurrency=16)
    pauses = []
    monkeypatch.setattr(s3_store.time, "sleep", pauses.append)
    assert store.read_file("big", tmp_path / "out")
    assert (tmp_path / "out").read_bytes() == data
    assert s3.most > 3 and s3.refused >= 1  # it started at 16 and was refused
    assert store.range_gate.cuts >= 1 and len(pauses) == s3.refused and all(0 < p <= 30.0 for p in pauses)
    assert "refused a range read for its request rate (ServiceUnavailable)" in capsys.readouterr().err
    # it probes upward again (one read per RANGE_RESTORE_READS), so a refusal now and then, not a storm
    assert s3.refused <= 16 and store.range_gate.limit is not None and store.range_gate.limit <= 4


def test_a_range_refused_every_time_fails_the_download_after_its_tries(tmp_path, monkeypatch):
    s3 = Limited(allowed=0)
    store, _data = _store(s3, ranges=2, concurrency=1)
    pauses = []
    monkeypatch.setattr(s3_store.time, "sleep", pauses.append)
    with pytest.raises(ClientError, match="ServiceUnavailable"):
        store.read_file("big", tmp_path / "out")
    # each range that ran was tried THROTTLED_READ_ATTEMPTS times (the second one may be cancelled)
    ran, rest = divmod(s3.refused, s3_store.THROTTLED_READ_ATTEMPTS)
    assert ran in (1, 2) and rest == 0 and len(pauses) == ran * (s3_store.THROTTLED_READ_ATTEMPTS - 1)
    assert not (tmp_path / "out").exists() and store.range_gate.limit == 1


def test_a_replaced_object_still_fails_the_download_at_once(tmp_path, monkeypatch):
    s3 = FakeS3()
    store, _data = _store(s3, ranges=3, concurrency=2)
    real = s3.get_object
    monkeypatch.setattr(s3, "get_object", lambda **kw: real(**{**kw, "IfMatch": '"other"'}) if kw.get("Range") else real(**kw))
    monkeypatch.setattr(s3_store.time, "sleep", lambda s: pytest.fail("no pause: not a rate refusal"))
    with pytest.raises(ClientError, match="PreconditionFailed"):
        store.read_file("big", tmp_path / "out")
    assert store.range_gate.cuts == 0



def test_the_retry_layers_of_a_refused_range_stay_bounded(tmp_path, monkeypatch):
    """botocore's retries times ours: 54 requests and at most 577 s of waits for a
    range refused every time (the bound in the module comment)."""
    retries = s3_store.client_config({})["retries"]
    assert retries == {"mode": "standard", "total_max_attempts": 9}
    botocore_worst = sum(min(2 ** (n - 1), 20) for n in range(1, retries["total_max_attempts"]))
    s3 = Limited(allowed=0)
    store, _data = _store(s3, ranges=2, concurrency=1)
    pauses = []
    monkeypatch.setattr(s3_store.random, "uniform", lambda lo, hi: hi)  # the longest pause each time
    monkeypatch.setattr(s3_store.time, "sleep", pauses.append)
    with pytest.raises(ClientError):
        store.read_file("big", tmp_path / "out")
    tries = s3_store.THROTTLED_READ_ATTEMPTS
    ranges, rest = divmod(s3.refused, tries)
    assert ranges in (1, 2) and rest == 0 and retries["total_max_attempts"] * tries == 54
    assert botocore_worst == 91 and tries * botocore_worst + sum(pauses) / ranges == 577
