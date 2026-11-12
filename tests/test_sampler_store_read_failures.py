"""A sampler's reads of the shared store are polls: one that fails is no verdict.

CPU, fakes (a real ``S3ObjectStore`` over a fake boto3 client; no cloud). The
unit cases place one failed read at one call site (the incarnation lease, the
handoff record, the dispenser ledger, ``LATEST``); the last section runs the
whole engine sampler (``engine_sampler_entry.run``) through an outage of every
read, and of every request, and checks the invariant of
``yeto.launch.checks.store_outage_check``.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from email.utils import formatdate
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from tests.test_rl_object_store import URI, FakeS3
from tests.test_stage1_gaps import _sampler_spec
from yeto.cloud import s3_store
from yeto.launch.checks.store_outage_check import read_outage_problems, released_in_flight
from yeto.launch.nc.elastic import ObjectRecords
from yeto.launch.pool.data_path import R2DataPath
from yeto.rl.engine import engine_sampler
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.incarnation_lease import (IncarnationGate, IncarnationLease, LeaseUnreadable, lease_key,
                                              read_lease)
from yeto.rl.engine.policy_release import LATEST_UNREAD, PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import STORE_IMPL_ENV
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.sampler_loop import READ_FAILED_EVENT, READ_OK_EVENT, SamplerLoop
from yeto.rl.engine.sampler_resume import ReclaimWatcher, handoff_requested, request_handoff
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

PREFIX = "runs/a/"  # of URI
READS = frozenset({"get_object", "head_object", "list"})
WRITES = frozenset({"put_object", "delete_object"})


def unavailable(op: str = "Op") -> ClientError:
    """What the caller sees once botocore's retries of a 503 are used up."""
    return ClientError({"Error": {"Code": "ServiceUnavailable", "Message": "retries exhausted"},
                        "ResponseMetadata": {"HTTPStatusCode": 503, "RetryAttempts": 7}}, op)


def expired_token(op: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": "ExpiredToken", "Message": "expired"},
                        "ResponseMetadata": {"HTTPStatusCode": 403, "RetryAttempts": 0}}, op)


class OutageS3(FakeS3):
    """``FakeS3`` with faults and server times.

    ``fail(op, error, key=..., skip=n)``: the ``n+1``-th call of ``op`` whose key
    contains ``key`` raises ``error`` (once). ``down``: the ops that fail on every
    call (an outage; :data:`READS`, :data:`WRITES` or both). Ops: the client
    method names and ``list`` (a listing; ``key`` matches the listed prefix).
    ``peer()``: another client of the same bucket with its own faults (a host on
    another network). ``now``: the server clock."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.now = 1_700_000_000.0
        self.mtime: dict[tuple[str, str], float] = {}
        self.faults: list[dict] = []
        self.down: frozenset[str] = frozenset()
        self.refused: list[str] = []

    def peer(self) -> "OutageS3":
        other = OutageS3(conditional=self.conditional)
        other.objects, other.mtime, other.lock, other.checksums = self.objects, self.mtime, self.lock, self.checksums
        return other

    def fail(self, op: str, error: BaseException, *, key: str = "", skip: int = 0) -> None:
        self.faults.append({"op": op, "error": error, "key": key, "skip": skip})

    def _check(self, op: str, key: str) -> None:
        if op in self.down:
            self.refused.append(f"{op} {key}")
            raise (unavailable(op) if self.down == READS else EndpointConnectionError(endpoint_url="https://store"))
        for fault in self.faults:
            if fault["op"] == op and fault["key"] in key:
                if fault["skip"]:
                    fault["skip"] -= 1
                    return
                self.faults.remove(fault)
                self.refused.append(f"{op} {key}")
                raise fault["error"]

    def get_object(self, Bucket, Key, **kwargs):
        self._check("get_object", Key)
        return FakeS3.get_object(self, Bucket, Key, **kwargs)

    def put_object(self, Bucket, Key, Body, **kwargs):
        self._check("put_object", Key)
        FakeS3.put_object(self, Bucket, Key, Body, **kwargs)
        self.mtime[(Bucket, Key)] = self.now

    def head_object(self, Bucket, Key, **kwargs):
        self._check("head_object", Key)
        return FakeS3.head_object(self, Bucket, Key, **kwargs)

    def delete_object(self, Bucket, Key):
        self._check("delete_object", Key)
        return FakeS3.delete_object(self, Bucket, Key)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix, StartAfter=""):
                fake._check("list", Prefix)
                with fake.lock:
                    keys = sorted(k for b, k in fake.objects if b == Bucket and k.startswith(Prefix)
                                  and k > StartAfter)
                yield {"ResponseMetadata": {"HTTPHeaders": {"date": formatdate(fake.now, usegmt=True)}},
                       "Contents": [{"Key": k, "LastModified": datetime.datetime.fromtimestamp(
                           fake.mtime.get((Bucket, k), fake.now), tz=datetime.timezone.utc)} for k in keys]}

        return Paginator()


def _world():
    s3 = OutageS3()
    return s3, s3_store.S3ObjectStore(URI, client=s3)


def objects_store(objects):
    """A store as the sampler sees it: ``.objects`` is the object API."""
    return SimpleNamespace(objects=objects, root="/nonexistent")


class Mono:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ===================================================================== incarnation lease
LEASE_ERRORS = {"503 after the SDK's retries": lambda: unavailable("GetObject"),
                "no route to the endpoint": lambda: EndpointConnectionError(endpoint_url="https://store"),
                "expired credential": lambda: expired_token("GetObject")}


@pytest.mark.parametrize("error", list(LEASE_ERRORS), ids=list(LEASE_ERRORS))
def test_one_failed_lease_read_does_not_end_the_lease_watcher(error):
    s3, objects = _world()
    lease = IncarnationLease(objects_store(objects), "s-0", "i1")
    lease.claim()
    consumed = threading.Event()

    def poll():
        try:
            return lease.poll()
        finally:
            if s3.refused:
                consumed.set()

    s3.fail("get_object", LEASE_ERRORS[error](), key="leases/s-0")
    watcher = ReclaimWatcher(poll, lambda: None, poll_s=0.01, name="lease-watcher-under-test")
    threading.excepthook, hook = (lambda args: None), threading.excepthook  # the thread's traceback
    try:
        watcher.start()
        assert consumed.wait(5.0)
        IncarnationLease(objects_store(objects), "s-0", "i2").claim()  # the pool fences / a successor claims
        assert watcher.fired.wait(2.0), "the watcher thread ended on the failed read"
    finally:
        watcher.stop()
        threading.excepthook = hook
    assert lease.read_failures == 1 and lease.fenced_by == "i2"


def test_a_watcher_poll_that_raises_is_polled_again():
    answers = iter([RuntimeError("store"), unavailable(), True])

    def poll():
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    watcher = ReclaimWatcher(poll, lambda: None, poll_s=0.01, name="watcher-under-test").start()
    try:
        assert watcher.fired.wait(5.0) and watcher.poll_failures == 2
    finally:
        watcher.stop()


def test_a_failed_lease_read_before_a_group_write_fails_open():
    s3, objects = _world()
    lease = IncarnationLease(objects_store(objects), "s-0", "i1", write_check_s=0.0)
    lease.claim()
    s3.fail("get_object", unavailable("GetObject"), key="leases/s-0")
    assert lease.check_before_write() is False and lease.read_failures == 1
    IncarnationLease(objects_store(objects), "s-0", "i2").claim()
    assert lease.check_before_write() is True and lease.fenced_by == "i2"


def test_a_lease_unread_for_longer_than_the_bound_refuses_group_writes_until_it_is_read_again():
    """Fail open has a bound: an incarnation that could not read its lease for
    ``max_unread_s`` does not know whether it was fenced, and does not write."""
    s3, objects = _world()
    mono = Mono()
    lease = IncarnationLease(objects_store(objects), "s-0", "i1", write_check_s=1.0, monotonic=mono,
                             max_unread_s=300.0)
    lease.claim()
    s3.down = READS
    for mono.now in (10.0, 150.0, 299.0):
        assert lease.check_before_write() is False and not lease.unknown()
    mono.now = 301.0
    assert lease.check_before_write() is True and lease.unknown() and not lease.fenced
    assert lease.unread_refusals == 1 and lease.health()["lease_unread_s"] == 301.0
    assert IncarnationLease(objects_store(objects), "s-0", "i1").max_unread_s == 60.0  # the default: two polls
    s3.down = frozenset()  # the store answers again: this incarnation still holds the lease
    mono.now = 303.0
    assert lease.check_before_write() is False and not lease.unknown()
    s3.down = READS  # a second outage, during which the pool fences it
    IncarnationLease(objects_store(s3_store.S3ObjectStore(URI, client=s3.peer())), "s-0", "pool-fence:t1").claim()
    mono.now = 700.0
    assert lease.check_before_write() is True and lease.unknown()
    s3.down = frozenset()
    mono.now = 702.0
    assert lease.check_before_write() is True and lease.fenced_by == "pool-fence:t1" and not lease.unknown()


def test_a_lease_without_a_bound_fails_open_for_as_long_as_it_is_unread():
    s3, objects = _world()
    mono = Mono()
    lease = IncarnationLease(objects_store(objects), "s-0", "i1", monotonic=mono, max_unread_s=None)
    lease.claim()
    s3.down = READS
    mono.now = 86400.0
    assert lease.check_before_write() is False and not lease.unknown()


def test_a_fence_whose_read_of_the_previous_lease_timed_out_still_supersedes_its_holder():
    s3, objects = _world()
    IncarnationLease(objects_store(objects), "s-0", "i1", clock=lambda: 100.0).claim()
    s3.fail("get_object", ReadTimeoutError(endpoint_url="https://store"), key="leases/s-0")
    slept: list[float] = []
    record = IncarnationLease(objects_store(objects), "s-0", "pool-fence:t1",
                              clock=lambda: 1000.0).claim(sleep=slept.append)
    assert record["superseded"] == ["i1"] and record["generation"] == 2 and slept == [0.5]
    late = SampleIndexEntry("s-0-i1-r7-row3", 0, "payloads/x.bin", tokens=1, appended_at=2000.0,
                            sampler_round={"sampler": "s-0", "incarnation": "i1", "rollout_id": 7})
    assert IncarnationGate(objects_store(objects), lambda: [late]).verdict(late, 0) is not None


def test_a_claim_that_cannot_read_the_previous_lease_writes_no_lease():
    """A claim written over an unread lease would name nothing superseded: the
    trainer's gate would then train what the fenced incarnation writes."""
    s3, objects = _world()
    IncarnationLease(objects_store(objects), "s-0", "i1", clock=lambda: 100.0).claim()
    before = objects.read(lease_key("s-0"))
    s3.down = READS
    slept: list[float] = []
    fence = IncarnationLease(objects_store(objects), "s-0", "pool-fence:t1", clock=lambda: 1000.0)
    with pytest.raises(LeaseUnreadable, match="could not be read in 4 tries"):
        fence.claim(sleep=slept.append)
    s3.down = frozenset()
    assert objects.read(lease_key("s-0")) == before and slept == [0.5, 1.0, 2.0]
    assert fence.claim()["superseded"] == ["i1"]


def test_a_first_claim_finds_no_lease_and_writes_generation_one():
    _s3, objects = _world()
    record = IncarnationLease(objects_store(objects), "s-0", "i1").claim(sleep=lambda s: pytest.fail("no retry"))
    assert record["generation"] == 1 and record["superseded"] == []


# ===================================================================== the trainer's gate
def _late_group(objects, mono=None, **kwargs):
    """``s-0``'s incarnation ``i1`` superseded by a pool fence, one group of ``i1``
    appended after it, and the trainer's gate."""
    IncarnationLease(objects_store(objects), "s-0", "i1", clock=lambda: 100.0).claim()
    late = SampleIndexEntry("s-0-i1-r7-row3", 0, "payloads/x.bin", tokens=1, appended_at=2000.0,
                            sampler_round={"sampler": "s-0", "incarnation": "i1", "rollout_id": 7})
    gate = IncarnationGate(objects_store(objects), lambda: [late], monotonic=mono or time.monotonic, **kwargs)
    return late, gate


def _fence(objects):
    IncarnationLease(objects_store(objects), "s-0", "pool-fence:t1", clock=lambda: 1000.0).claim()


@pytest.mark.parametrize("error", [lambda: ReadTimeoutError(endpoint_url="https://store"),
                                   lambda: unavailable("GetObject")], ids=["times out", "refused"])
def test_a_lease_the_gate_never_read_is_read_again_before_its_entries_get_a_verdict(error):
    """Neither "no lease" (the fenced incarnation's group trained) nor an error out
    of the trainer's take."""
    s3, objects = _world()
    slept: list[float] = []
    late, gate = _late_group(objects, sleep=slept.append, unread_retry_s=2.0)
    _fence(objects)
    s3.fail("get_object", error(), key="leases/s-0")
    s3.fail("get_object", error(), key="leases/s-0")
    assert gate.verdict(late, 0) is not None and slept == [2.0, 2.0] and gate.read_failures == 2


def test_a_gate_that_cannot_read_a_lease_for_too_long_raises_instead_of_letting_the_entry_through():
    s3, objects = _world()
    slept: list[float] = []
    late, gate = _late_group(objects, sleep=slept.append, unread_retry_s=2.0, unread_wait_s=6.0)
    s3.down = READS
    with pytest.raises(LeaseUnreadable):
        gate.verdict(late, 0)
    assert slept == [2.0, 2.0, 2.0]


def test_a_failed_refresh_of_a_lease_keeps_the_lease_the_gate_read_last():
    """Not cached as "no lease" for the refresh period."""
    s3, objects = _world()
    mono = Mono()
    late, gate = _late_group(objects, mono, sleep=lambda s: pytest.fail("a known lease is not waited for"))
    _fence(objects)
    assert gate.verdict(late, 0) is not None
    mono.now = 31.0  # past refresh_s
    s3.fail("get_object", ReadTimeoutError(endpoint_url="https://store"), key="leases/s-0")
    assert gate.verdict(late, 0) is not None and gate.read_failures == 1
    calls = s3.calls.count("get_object")
    mono.now = 32.0  # read again unread_retry_s later, not at every verdict
    assert gate.verdict(late, 0) is not None and s3.calls.count("get_object") == calls
    mono.now = 34.0
    assert gate.verdict(late, 0) is not None and s3.calls.count("get_object") == calls + 1


# ===================================================================== handoff records
def test_a_failed_read_of_the_handoff_record_is_not_a_reason_to_stop_or_crash():
    s3, objects = _world()
    store = objects_store(objects)
    request_handoff(store, "s-0", "s-1")
    heard: list[BaseException] = []
    s3.fail("get_object", unavailable("GetObject"), key="handoffs/s-0")
    assert handoff_requested(store, "s-0", on_error=heard.append) is False and len(heard) == 1
    assert handoff_requested(store, "s-0", on_error=heard.append) is True and len(heard) == 1  # the next poll


# ===================================================================== store dispenser
ROWS = [{"prompt": f"p{i}", "metadata": {"task_id": f"t{i}"}} for i in range(4)]
NS = "task-dispense"


def _sampler(objects, holder, incarnation, *, lease_s=900.0, lease_of_holder=None, emit=None):
    ledger = sd.StoreLedger(objects, NS, rows=len(ROWS), fingerprint=b"f", holder=holder,
                            incarnation=incarnation, lease_s=lease_s, lease_of_holder=lease_of_holder)
    ledger.open()
    return sd.StoreDispensedPrompts("unused", ledger, input_key="prompt", label_key=None, groups_per_round=1,
                                    rows=ROWS, renew=False, sleep=lambda s: None, scan_s=0.0, emit=emit)


def _ledger_keys(objects, kind):
    head = f"{NS}/{kind}/"
    return sorted(k[len(head):] for k in objects.list(head))


def test_a_failed_ledger_listing_costs_one_scan_not_the_sampler():
    s3, objects = _world()
    events: list[tuple[str, dict]] = []
    a = _sampler(objects, "s-0", "i1", emit=lambda event, **fields: events.append((event, fields)))
    assert a.take(1) == [0]
    s3.fail("list", unavailable("ListObjectsV2"), key=NS + "/")
    assert a.take(1) == []  # holding one: nothing new this time
    assert a.take(1) == [1]
    assert a.stats["scans_failed"] == 1 and a.holding() == 2 and _ledger_keys(objects, "fail") == []
    assert [f["state"] for e, f in events if e == sd.SCAN_FAILED_EVENT] == ["failed", "recovered"]


def test_a_sampler_holding_nothing_waits_for_the_next_scan_after_a_failed_listing():
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1")
    s3.fail("list", unavailable("ListObjectsV2"), key=NS + "/")
    assert a.take(1) == [0] and a.stats["scans_failed"] == 1
    s3.fail("list", unavailable("ListObjectsV2"), key=NS + "/")
    assert a.poll(1) == ("wait", []) and a.poll(1) == ("items", [1])


def test_a_drain_request_that_cannot_be_read_costs_one_scan():
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1")
    assert a.take(1) == [0]
    seq = sd.request_drain(objects, NS, "s-0", seq="t1:1.000")
    s3.fail("get_object", unavailable("GetObject"), key=f"{NS}/drain/s-0")
    assert a.poll(1) == ("wait", []) and a.stats["scans_failed"] == 1 and not a.draining()
    assert a.poll(1) == ("wait", []) and a.draining() and a.holding() == 1
    assert not sd.drain_answered(objects, NS, "s-0", seq)  # it still holds a claim


def test_a_claim_whose_holder_cannot_be_read_stays_a_claim_until_its_lease_runs_out():
    """The fence check of a live claim reads its body: a failed read (or a body
    that is no JSON) is no fence verdict and does not end the scan."""
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1", lease_s=90.0)
    assert a.take(1) == [0]
    b = _sampler(objects, "s-1", "j1", lease_s=90.0, lease_of_holder={"s-0": "i2"}.get)  # s-0 was fenced
    s3.fail("get_object", unavailable("GetObject"), key=f"{NS}/claims/000000000000/0000")
    assert b.take(1) == [1] and _ledger_keys(objects, "fail") == []
    s3.objects[("bkt", f"{PREFIX}{NS}/claims/000000000001/0000")] = b"{not json"
    c = _sampler(objects, "s-2", "k1", lease_s=90.0, lease_of_holder={"s-0": "i2"}.get)
    assert c.take(1) == [0]  # s-0's claim: read now, its holder is fenced; slot 1 is left alone
    assert _ledger_keys(objects, "fail") == ["000000000000/0000.release"]
    assert c.recover() == {"recovered": 0, "released": 0}  # a restart reads every open claim's body
    s3.now += 100.0
    assert c.renew() == 1  # its own claim of slot 0
    assert c.take(1) == [1] and "000000000001/0000.expired" in _ledger_keys(objects, "fail")


def test_a_hold_check_the_store_did_not_answer_is_unknown_and_the_claim_stays():
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1")
    assert a.take(1) == [0]
    s3.fail("head_object", unavailable("HeadObject"), key=f"{NS}/claims/")
    assert a.hold_verdict(0) is None and a.stats["hold_checks_failed"] == 1
    s3.fail("head_object", EndpointConnectionError(endpoint_url="https://store"), key=f"{NS}/fail/")
    assert a.still_holds(0) is False  # the contract's answer: do not write now
    assert a.holding() == 1 and a.stats["lost"] == 0 and _ledger_keys(objects, "fail") == []
    assert a.hold_verdict(0) is True and a.still_holds(0) is True


def test_a_renewal_that_cannot_read_whether_it_was_superseded_still_writes_its_beat():
    """Renewals are writes: an outage of the reads does not let a claim run out."""
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1", lease_s=90.0)
    assert a.take(2) == [0, 1]
    for _ in range(4):  # every read of the outage fails; 160 s is longer than the lease
        s3.now += 40.0
        s3.down = READS
        assert a.renew() == 2
        s3.down = frozenset()
    assert a.stats["renew_checks_failed"] == 8
    b = _sampler(objects, "s-1", "j1", lease_s=90.0)
    assert b.take(2) == [2, 3] and _ledger_keys(objects, "fail") == []
    assert a.still_holds(0) and a.still_holds(1)


def test_a_refused_request_for_one_slot_does_not_cost_the_other_slots_their_beats():
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1", lease_s=90.0)
    assert a.take(3) == [0, 1, 2]
    s3.fail("put_object", unavailable("PutObject"), key=f"{NS}/beats/000000000001/")
    s3.fail("head_object", unavailable("HeadObject"), key=f"{NS}/claims/000000000000/0001")
    with pytest.raises(ClientError):  # reported by the renew loop, which renews again a period later
        a.renew()
    assert _ledger_keys(objects, "beats") == ["000000000000/0000", "000000000002/0000"]


def test_a_sampler_back_from_an_outage_of_every_request_longer_than_its_lease_does_not_write_its_group():
    """Its beats failed for longer than the lease: the claim ran out, another
    sampler took the task, and the first one's write is refused."""
    s3, objects = _world()
    a = _sampler(objects, "s-0", "i1", lease_s=90.0)
    assert a.take(1) == [0]
    s3.down = READS | WRITES
    for _ in range(3):
        with pytest.raises(EndpointConnectionError):
            a.renew()
        assert a.hold_verdict(0) is None
        s3.now += 40.0
    s3.down = frozenset()
    b = _sampler(objects, "s-1", "j1", lease_s=90.0)
    assert b.take(1) == [0] and _ledger_keys(objects, "fail") == ["000000000000/0000.expired"]
    assert a.hold_verdict(0) is False and a.renew() == 0
    assert a.written(0, [], unwritten="release")["trained"] == 0 and _ledger_keys(objects, "done") == []
    assert b.still_holds(0) is True


# ===================================================================== the sampler loop
class _ReleaseStoreView:
    """A ``PolicyReleaseStore``'s store: the bucket by object API, no mount."""

    def __init__(self, objects, root) -> None:
        self.objects, self.root = objects, root

    def reload(self) -> None:
        pass


def _loop(objects, tmp_path, rounds, events, **kwargs):
    releases = PolicyReleaseStore(_ReleaseStoreView(objects, tmp_path / "mount"), cache_dir=tmp_path / "cache")
    follower = PolicyFollower(releases, lambda release, directory: None)
    buffer = FileSampleBuffer(objects, batch_size=1, index=shard_index("s-0"))
    return SamplerLoop(follower=follower, buffer=buffer, generate=rounds.append, max_backlog_groups=4,
                       emit=lambda event, **fields: events.append((event, fields)), sleep=lambda s: None, **kwargs)


LATEST_KEY = ("bkt", PREFIX + "policy-releases/LATEST.json")


def test_a_sampler_keeps_generating_on_its_served_weights_while_latest_is_unreadable(tmp_path):
    s3, objects = _world()
    s3.objects[LATEST_KEY] = b"{not json"
    rounds: list[int] = []
    events: list[tuple[str, dict]] = []
    loop = _loop(objects, tmp_path, rounds, events, num_rollouts=2)
    assert loop.run() == 2 and rounds == [0, 1]
    not_loaded = [f for e, f in events if e == "rl_sampler_release_not_loaded"]
    assert [(f["reason"], f["damaged"]) for f in not_loaded] == [(LATEST_UNREAD, True)]  # once, not per poll


def test_a_latest_that_stays_damaged_ends_the_sampler_after_the_bound(tmp_path):
    s3, objects = _world()
    s3.objects[LATEST_KEY] = b"[1, 2]"
    rounds: list[int] = []
    loop = _loop(objects, tmp_path, rounds, [], num_rollouts=None)
    loop.max_damaged_latest_reads = 3
    with pytest.raises(Exception, match="could not be read"):
        loop.run()
    assert rounds == [0, 1]


def test_a_latest_the_store_does_not_answer_for_never_ends_the_sampler(tmp_path):
    s3, objects = _world()
    rounds: list[int] = []
    events: list[tuple[str, dict]] = []
    loop = _loop(objects, tmp_path, rounds, events, num_rollouts=6)
    loop.max_damaged_latest_reads = 2
    for _ in range(6):
        s3.fail("get_object", unavailable("GetObject"), key="policy-releases/LATEST.json")
    assert loop.run() == 6 and loop.follower.latest_read_failures == 6
    assert loop.follower.pending_seq() is None


def test_a_round_is_not_admitted_while_the_buffer_cannot_be_read_and_the_loop_reads_again(tmp_path):
    s3, objects = _world()
    rounds: list[int] = []
    events: list[tuple[str, dict]] = []
    loop = _loop(objects, tmp_path, rounds, events, num_rollouts=1)
    for _ in range(3):
        s3.fail("list", unavailable("ListObjectsV2"), key="sample-buffer/shards/s-0/")
    assert loop.run() == 1 and rounds == [0] and loop.read_failures == 3
    polls = [(e, f.get("what"), f.get("failed_reads")) for e, f in events if e in (READ_FAILED_EVENT, READ_OK_EVENT)]
    assert polls == [(READ_FAILED_EVENT, "backlog", None), (READ_OK_EVENT, "backlog", 3)]


# ===================================================================== the whole sampler
GROUPS, ROUNDS, N = 2, 3, 2  # groups a round, rounds, samples a group


class Run:
    """One engine sampler (``engine_sampler_entry.run``, fake engine, store
    dispenser, incarnation lease) on a thread, over ``s3``. ``at_reward(k, fn)``:
    ``fn()`` runs on the sampler's thread right before the ``k``-th reward call
    (``k = 1 + round * GROUPS * N``: the round's tasks are claimed and generated,
    none of its groups is written); ``at_event(name, fn)``: right after the first
    tape event ``name`` for which ``fn(fields)`` is true."""

    def __init__(self, monkeypatch, tmp_path, s3, *, dispense=True, lease=None, **spec_keys) -> None:
        self.s3, self.events, self.rewards = s3, [], 0
        self.stop = threading.Event()  # the caller's stop (``entry.run(stop=...)``)
        self._at_reward: dict[int, object] = {}
        self._at_event: list[tuple[str, object, object]] = []
        self.result: object = None
        monkeypatch.setattr(s3_store, "_client", lambda environ, require_credentials=True: s3)
        monkeypatch.setattr(engine_sampler, "DEFAULT_WRITE_RETRY_S", 0.01)

        def reward(item, completion) -> float:
            self.rewards += 1
            hook = self._at_reward.pop(self.rewards, None)
            if hook is not None:
                hook()
            return 0.5

        monkeypatch.setattr(entry, "build_reward", lambda spec: reward)
        spec = _sampler_spec(tmp_path, sampler_id="s-0", num_rollouts=ROUNDS, poll_s=0.01, buffer_shards=["s-0"],
                             incarnation_lease={"poll_s": 0.01, "write_check_s": 0.0, **(lease or {})}, **spec_keys)
        rows = [{"q": f"question {i}", "a": str(i)} for i in range(GROUPS * ROUNDS)]
        (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        spec["prompts"] = {**spec["prompts"], "groups_per_round": GROUPS}
        if dispense:
            spec["prompts"]["dispense"] = {"backend": "store", "namespace": NS, "scan_s": 0.01, "max_scan_s": 0.02}
        self.env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI}
        self.thread = threading.Thread(target=self._run, args=(spec,), name="sampler-under-test", daemon=True)

    def _run(self, spec) -> None:
        try:
            self.result = entry.run(spec, environ=self.env, resource_sampler=lambda emit: None,
                                    on_event=self._on_event, stop=self.stop.is_set)
        except BaseException as error:  # noqa: BLE001 - the test asserts on it
            self.result = error

    def _on_event(self, event, **fields) -> None:
        self.events.append({"event": event, "time_unix": time.time(), "sampler_id": "s-0", **fields})
        for hook in list(self._at_event):
            if hook[0] == event and hook[1](fields):
                self._at_event.remove(hook)
                hook[2]()

    def at_reward(self, k, fn) -> None:
        self._at_reward[k] = fn

    def at_event(self, name, fn, when=lambda fields: True) -> None:
        self._at_event.append((name, when, fn))

    def wait(self, what, pred, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            assert self.thread.is_alive(), f"the sampler ended while the test waited for {what}: {self.result!r}"
            time.sleep(0.005)
        raise AssertionError(f"timed out waiting for {what}; events: {[e['event'] for e in self.events][-12:]}")

    def seen(self, event, **fields) -> int:
        return sum(1 for e in self.events if e["event"] == event and all(e.get(k) == v for k, v in fields.items()))

    def finish(self, timeout=30.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the sampler did not end"
        return self.result

    def ledger(self):
        return s3_store.S3ObjectStore(URI, client=self.s3.peer()).list_meta(NS + "/")[0]

    def groups(self) -> list[dict]:
        objects = s3_store.S3ObjectStore(URI, client=self.s3.peer())
        return [e.sampler_round for e in FileSampleBuffer(objects, batch_size=1, index=shard_index("s-0"),
                                                          create_index=False).entries()]


class Outage:
    """An outage of ``ops`` on the sampler's client, with the ledger as it began."""

    def __init__(self, run: Run, ops) -> None:
        self.run, self.ops = run, frozenset(ops)
        self.start = self.end = None
        self.before = None

    def begin(self) -> None:
        self.before = self.run.ledger()
        self.start = time.time()
        self.run.s3.down = self.ops

    def stop(self) -> None:
        self.run.s3.down = frozenset()
        self.end = time.time()

    def problems(self, **kwargs) -> list[str]:
        return read_outage_problems(self.run.events, start=self.start, end=self.end, ledger_before=self.before,
                                    ledger_after=self.run.ledger(), namespace=NS, **kwargs)


def _all_trained_once(run: Run) -> None:
    states = sd.parse_ledger(run.ledger(), NS)
    assert {slot: state.done for slot, state in states.items()} == {i: "trained" for i in range(GROUPS * ROUNDS)}
    assert all(not state.fails and list(state.claims) == [0] for state in states.values())
    positions = sorted(g["dispense_position"] for g in run.groups())
    assert positions == list(range(GROUPS * ROUNDS))


def test_a_sampler_lives_through_an_outage_of_every_read_with_tasks_in_flight_and_between_rounds(
        monkeypatch, tmp_path):
    """Writes still work. First outage: round 1's tasks are claimed and generated,
    its groups not yet written. Second: between two rounds (the handoff record,
    ``LATEST``, the backlog, the ledger scan)."""
    run = Run(monkeypatch, tmp_path, OutageS3())
    in_flight, between = Outage(run, READS), Outage(run, READS)
    run.at_reward(1 + GROUPS * N, in_flight.begin)
    run.at_event("rl_sampler_round", between.begin, when=lambda fields: fields["rollout_id"] == 1)
    run.thread.start()
    run.wait("the held-back group write", lambda: run.seen(engine_sampler.WRITE_DEFERRED_EVENT, reason="claim_unread"))
    run.wait("failed lease polls", lambda: sum("leases/" in r for r in run.s3.refused) >= 3
             and run.seen(READ_FAILED_EVENT, what="lease"))
    assert len(run.groups()) == GROUPS  # round 0 only: nothing is written on an unread claim
    in_flight.stop()
    run.wait("the outage between rounds", lambda: between.start is not None)
    run.wait("the failed polls of the loop", lambda: (
        run.seen(READ_FAILED_EVENT, what="handoff") and run.seen(READ_FAILED_EVENT, what="backlog")
        and run.seen("rl_sampler_release_not_loaded", reason=LATEST_UNREAD)))
    between.stop()
    assert run.finish() == ROUNDS
    _all_trained_once(run)
    assert in_flight.problems() == [] and between.problems() == []
    assert run.seen(READ_OK_EVENT, what="handoff") and run.seen(READ_OK_EVENT, what="backlog")
    assert [e["reason"] for e in run.events if e["event"] == "rl_sampler_leave"] == ["FINISHED"]


def test_a_sampler_lives_through_an_outage_of_every_request_and_writes_its_groups_in_flight_afterwards(
        monkeypatch, tmp_path):
    """Reads and writes fail (shorter than the dispense lease): the round's groups
    wait for the store, no claim is given back, and the sampler goes on."""
    run = Run(monkeypatch, tmp_path, OutageS3())
    outage = Outage(run, READS | WRITES)
    run.at_reward(1 + GROUPS * N, outage.begin)
    run.thread.start()
    run.wait("the held-back group write", lambda: run.seen(engine_sampler.WRITE_DEFERRED_EVENT))
    run.wait("more refused requests", lambda: len(run.s3.refused) >= 20)
    assert run.thread.is_alive() and len(run.groups()) == GROUPS
    outage.stop()
    assert run.finish() == ROUNDS
    _all_trained_once(run)
    assert outage.problems() == []


def test_a_sampler_fenced_during_an_outage_of_its_reads_writes_no_group_once_it_reads_again(monkeypatch, tmp_path):
    s3 = OutageS3()
    run = Run(monkeypatch, tmp_path, s3)
    outage = Outage(run, READS)
    run.at_reward(1 + GROUPS * N, outage.begin)
    run.thread.start()
    run.wait("the held-back group write", lambda: run.seen(engine_sampler.WRITE_DEFERRED_EVENT))
    pool = s3_store.S3ObjectStore(URI, client=s3.peer())  # the control host reaches the store
    R2DataPath(ObjectRecords(pool), (pool, NS)).fence_shard("s-0", by="t1")
    outage.stop()
    assert run.finish() == 2  # round 1 ran; its groups are dropped, not written
    assert len(run.groups()) == GROUPS and {g["rollout_id"] for g in run.groups()} == {0}
    assert [e["reason"] for e in run.events if e["event"] == "rl_sampler_leave"] == ["FENCED"]
    states = sd.parse_ledger(run.ledger(), NS)
    assert [states[i].fails for i in (2, 3)] == [{0: "release"}, {0: "release"}]  # given out again
    holders = {slot: "s-0" for slot in states}
    assert outage.problems(fenced=["s-0"], holders=holders) == []
    assert any("ended release" in p for p in outage.problems(expect_progress=False))


def test_group_writes_wait_while_the_lease_stays_unread_past_its_bound_and_go_on_when_it_is_read(
        monkeypatch, tmp_path):
    """No dispenser: only the lease stands between a fenced incarnation and the
    buffer. Past ``max_unread_s`` the round's groups wait instead of being written
    on an unknown lease (or dropped)."""
    run = Run(monkeypatch, tmp_path, OutageS3(), dispense=False, lease={"max_unread_s": 0.2})
    outage = Outage(run, {"get_object"})  # the write's own HEAD of its payload name still answers
    run.at_reward(1 + GROUPS * N, lambda: (outage.begin(), time.sleep(0.3)))  # unread for longer than the bound
    run.thread.start()
    run.wait("the held-back group write", lambda: run.seen(engine_sampler.WRITE_DEFERRED_EVENT, reason="lease_unread"))
    assert len(run.groups()) == GROUPS
    outage.stop()
    assert run.finish() == ROUNDS and len(run.groups()) == GROUPS * ROUNDS
    assert [e.get("dropped_groups") for e in run.events if e["event"] == "rl_sampler_leave"] == [0]


def _given_up(run: Run) -> list[dict]:
    return [e for e in run.events if e["event"] == engine_sampler.WRITE_DEFERRED_EVENT and e.get("given_up")]


def test_past_the_longest_wait_a_group_is_given_up_its_task_released_and_the_sampler_goes_on(monkeypatch, tmp_path):
    """The store does not say for longer than ``write_defer_max_s`` whether the
    sampler still holds its tasks: the waiting groups are not written, their tasks
    go back uncounted, and the sampler takes them again once it can read."""
    run = Run(monkeypatch, tmp_path, OutageS3(), write_defer_max_s=0.05)
    outage = Outage(run, READS)
    run.at_reward(1 + GROUPS * N, outage.begin)
    run.thread.start()
    run.wait("the groups given up", lambda: len(_given_up(run)) == GROUPS)
    assert _given_up(run)[0]["reason"] == "claim_unread" and _given_up(run)[0]["dispense_position"] == 2
    assert _given_up(run)[1]["waited_s"] == 0.0  # the second group asks once: the bound is the outage's, not its own
    run.wait("the tasks released", lambda: sum(e["event"] == "rl_dispense_release" for e in run.events) == GROUPS)
    assert run.thread.is_alive() and len(run.groups()) == GROUPS
    outage.stop()
    assert run.finish() == ROUNDS  # its last round takes the released tasks again (returned tasks go first)
    states = sd.parse_ledger(run.ledger(), NS)
    assert {slot: state.done for slot, state in states.items()} == {i: "trained" for i in range(4)}
    assert [(states[i].fails, sorted(states[i].claims)) for i in range(4)] == [
        ({}, [0]), ({}, [0]), ({0: "release"}, [0, 1]), ({0: "release"}, [0, 1])]
    assert sorted(g["dispense_position"] for g in run.groups()) == [0, 1, 2, 3]  # each written once
    assert [e["reason"] for e in run.events if e["event"] == "rl_sampler_leave"] == ["FINISHED"]
    assert any("ended release" in p for p in outage.problems())  # longer than the sampler was set to ride out


def test_a_sampler_asked_to_stop_gives_up_the_group_it_cannot_verify_and_stops(monkeypatch, tmp_path):
    run = Run(monkeypatch, tmp_path, OutageS3())
    outage = Outage(run, READS)
    run.at_reward(1 + GROUPS * N, outage.begin)
    run.thread.start()
    run.wait("the held-back group write", lambda: run.seen(engine_sampler.WRITE_DEFERRED_EVENT, reason="claim_unread"))
    assert not _given_up(run)
    run.stop.set()
    assert run.finish() == 2 and len(_given_up(run)) == GROUPS  # the round ends; no group written on an unknown claim
    assert len(run.groups()) == GROUPS
    outage.stop()
    states = sd.parse_ledger(run.ledger(), NS)
    assert [states[i].fails for i in (2, 3)] == [{0: "release"}, {0: "release"}]
    assert [e["reason"] for e in run.events if e["event"] == "rl_sampler_leave"] == ["FINISHED"]


# ===================================================================== the invariant check
def _claim(slot, attempt=0):
    return f"{NS}/claims/{slot:012d}/{attempt:04d}"


def test_the_outage_check_names_a_sampler_that_left_and_a_task_that_was_given_back():
    before = [_claim(0), _claim(1), _claim(2), f"{NS}/done/{2:012d}.trained"]
    after = before + [f"{NS}/fail/{0:012d}/0000.release", f"{NS}/done/{1:012d}.trained"]
    assert released_in_flight(before, after, NS) == {0: (0, "release")}
    tape = [{"sampler_id": "s-0", "time_unix": 90.0, "event": "rl_dispense_claim", "slot": 0},
            {"sampler_id": "s-0", "time_unix": 120.0, "event": "rl_dispense_release", "slot": 0, "attempt": 0,
             "state": "release"},
            {"sampler_id": "s-0", "time_unix": 121.0, "event": "rl_sampler_leave", "reason": "INFRA_FAILED",
             "error": "ClientError: 503"},
            {"sampler_id": "s-1", "time_unix": 95.0, "event": "rl_dispense_claim", "slot": 1},
            {"sampler_id": "s-1", "time_unix": 230.0, "event": "rl_dispense_settle", "slot": 1, "state": "trained"}]
    problems = read_outage_problems(tape, start=100.0, end=200.0, ledger_before=before, ledger_after=after,
                                    namespace=NS)
    assert len(problems) == 4 and all("s-1" not in p for p in problems)
    assert read_outage_problems(tape, start=100.0, end=200.0, max_outage_s=60.0) == []  # outside the invariant
    assert read_outage_problems(tape[3:], start=100.0, end=200.0) == []


def test_a_prefetch_whose_read_of_latest_fails_reports_nothing_on_disk_and_reads_again(tmp_path):
    s3, objects = _world()
    releases = PolicyReleaseStore(_ReleaseStoreView(objects, tmp_path / "mount"), cache_dir=tmp_path / "cache")
    follower = PolicyFollower(releases, lambda release, directory: None)
    follower.background_download = True
    s3.fail("get_object", unavailable("GetObject"), key="policy-releases/LATEST.json")
    assert follower.prefetch() is False and follower.latest_read_failures == 1
    assert follower.prefetch() is True  # read again: no release yet, nothing to fetch
