"""The trainer generation lease (CPU, fakes; no cloud resource): one trainer owns
a run's store at a time, by a lease record that is created once per generation."""

from __future__ import annotations

import json
import threading

import pytest
from botocore.exceptions import ClientError

from test_rl_object_store import URI, FakeS3
from yeto.cloud import s3_store
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import BucketMountStore, CheckpointStore
from yeto.rl.engine.trainer_lease import (
    LEASE_DIR,
    SCHEMA,
    TrainerFenced,
    TrainerLease,
    TrainerLeaseError,
    lease_key,
    lease_objects,
)


def _s3(client: FakeS3 | None = None) -> s3_store.S3ObjectStore:
    return s3_store.S3ObjectStore(URI, client=client or FakeS3())


@pytest.fixture(params=["directory", "s3"])
def objects(request, tmp_path):
    return DirectoryObjectStore(tmp_path / "store") if request.param == "directory" else _s3()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class LossyS3(FakeS3):
    """``arm(n)``: the n-th conditional put from now is applied and its answer lost.
    ``answer`` is what the caller then sees: ``refused`` (the SDK retried the request
    and the retry met the object: 412) or ``error`` (the SDK gave up: a timeout)."""

    def __init__(self, answer: str = "refused") -> None:
        super().__init__()
        self.answer = answer
        self.armed: int | None = None

    def arm(self, n: int = 1) -> None:
        self.armed = n

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **kwargs):
        if IfNoneMatch is not None and self.armed is not None:
            self.armed -= 1
            if self.armed == 0:
                self.armed = None
                super().put_object(Bucket, Key, Body, IfNoneMatch=IfNoneMatch, **kwargs)  # applied
                if self.answer == "error":
                    raise ClientError({"Error": {"Code": "RequestTimeout", "Message": "lost"}}, "PutObject")
                raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "lost"},
                                   "ResponseMetadata": {"RetryAttempts": 1, "HTTPStatusCode": 412}}, "PutObject")
        return super().put_object(Bucket, Key, Body, IfNoneMatch=IfNoneMatch, **kwargs)


class StaleFirstListing:
    """An object API whose first ``stale`` listings of the lease directory are empty:
    the claimer then races for a generation that is already taken."""

    def __init__(self, objects, stale: int) -> None:
        self.objects = objects
        self.stale = stale

    def list(self, prefix, **kwargs):
        if self.stale > 0:
            self.stale -= 1
            return []
        return self.objects.list(prefix, **kwargs)

    def __getattr__(self, name):
        return getattr(self.objects, name)


def test_a_store_without_lease_records_is_the_legacy_generation_zero(objects):
    assert TrainerLease.current(objects) == 0
    assert TrainerLease.chain(objects) == []


def test_the_first_claim_on_a_legacy_store_is_generation_one_and_records_its_holder(objects):
    lease = TrainerLease.claim(objects, incarnation="inc-a", host="node-1", clock=lambda: 12.5)

    assert lease.generation == 1 and lease.incarnation == "inc-a" and lease.fork is None
    assert TrainerLease.current(objects) == 1
    record = json.loads(objects.read(lease_key(1)))
    assert lease_key(1) == f"{LEASE_DIR}/00000001.json"
    assert record == {"schema": SCHEMA, "generation": 1, "incarnation": "inc-a", "claim": record["claim"],
                      "claimed_at": 12.5, "host": "node-1", "fork": None}
    lease.check(force=True)  # the holder of the highest generation is not fenced
    assert not lease.fenced


def test_concurrent_claims_get_distinct_consecutive_generations(objects):
    n = 16
    start = threading.Barrier(n)
    leases: list[TrainerLease] = []

    def claim(i: int) -> None:
        start.wait()
        leases.append(TrainerLease.claim(objects, incarnation=f"inc-{i}"))

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(lease.generation for lease in leases) == list(range(1, n + 1))
    chain = TrainerLease.chain(objects)
    assert [record["generation"] for record in chain] == list(range(1, n + 1))
    # every record names the claimer that holds that generation
    assert {record["generation"]: record["incarnation"] for record in chain} == {
        lease.generation: lease.incarnation for lease in leases}
    assert TrainerLease.current(objects) == n


def test_a_claimer_that_lost_the_race_for_a_generation_takes_the_next_one(objects):
    first = TrainerLease.claim(objects, incarnation="inc-a")

    late = TrainerLease.claim(StaleFirstListing(objects, stale=1), incarnation="inc-b")

    assert (first.generation, late.generation) == (1, 2)
    assert json.loads(objects.read(lease_key(1)))["incarnation"] == "inc-a"  # not overwritten
    assert [record["incarnation"] for record in TrainerLease.chain(objects)] == ["inc-a", "inc-b"]


def test_a_trainer_fenced_by_a_higher_generation_fails_every_later_check(objects):
    clock = Clock()
    zombie = TrainerLease.claim(objects, incarnation="inc-a", monotonic=clock, check_cache_s=5.0)
    zombie.check()
    successor = TrainerLease.claim(objects, incarnation="inc-b", monotonic=clock)

    clock.now = 10.0
    with pytest.raises(TrainerFenced) as fenced:
        zombie.check()
    assert (fenced.value.generation, fenced.value.fenced_by) == (1, 2)
    assert zombie.fenced and zombie.detail()["fenced_by"] == 2

    checks = zombie.checks
    for _ in range(3):  # sticky: no listing is needed to refuse again
        with pytest.raises(TrainerFenced):
            zombie.check(force=True)
    assert zombie.checks == checks
    successor.check(force=True)
    assert not successor.fenced


def test_a_check_reuses_its_listing_for_the_cache_window_unless_forced(objects):
    clock = Clock()
    lease = TrainerLease.claim(objects, incarnation="inc-a", monotonic=clock, check_cache_s=5.0)
    lease.check()
    TrainerLease.claim(objects, incarnation="inc-b")

    clock.now = 4.0
    lease.check()  # inside the window: the cached verdict
    assert lease.checks == 1 and not lease.fenced
    with pytest.raises(TrainerFenced):
        lease.check(force=True)
    assert lease.checks == 2


def test_a_claim_whose_create_answer_was_lost_to_a_retry_keeps_its_generation():
    client = LossyS3("refused")
    objects = _s3(client)
    client.arm()

    lease = TrainerLease.claim(objects, incarnation="inc-a")

    # the applied create is this claim's: no second record, no hole, one holder
    assert lease.generation == 1
    assert [record["incarnation"] for record in TrainerLease.chain(objects)] == ["inc-a"]
    lease.check(force=True)
    assert TrainerLease.claim(objects, incarnation="inc-b").generation == 2
    with pytest.raises(TrainerFenced):
        lease.check(force=True)


def test_a_claim_whose_create_was_applied_before_it_raised_keeps_its_generation():
    client = LossyS3("error")
    objects = _s3(client)
    client.arm()

    lease = TrainerLease.claim(objects, incarnation="inc-a")

    assert lease.generation == 1
    assert [record["generation"] for record in TrainerLease.chain(objects)] == [1]


def test_a_create_that_failed_without_being_applied_raises_and_claims_nothing():
    class Down(FakeS3):
        def put_object(self, *args, **kwargs):
            raise ClientError({"Error": {"Code": "RequestTimeout", "Message": "down"}}, "PutObject")

    objects = _s3(Down())

    with pytest.raises(ClientError):
        TrainerLease.claim(objects, incarnation="inc-a")
    assert TrainerLease.current(objects) == 0


def test_a_lost_answer_is_not_taken_for_a_win_when_another_claimer_holds_the_generation():
    """Two claimers with the same incarnation id (a platform retry that reuses it):
    the one refused by the other's record must not read it back as its own."""
    objects = _s3()
    first = TrainerLease.claim(objects, incarnation="inc-a")

    second = TrainerLease.claim(StaleFirstListing(objects, stale=1), incarnation="inc-a")

    assert (first.generation, second.generation) == (1, 2)


def test_the_fork_of_a_claim_is_kept_as_given_and_read_back_from_the_chain(objects):
    fork = {"parent_generation": 1, "parent_version": 7, "start_version": 12}
    TrainerLease.claim(objects, incarnation="inc-a")

    lease = TrainerLease.claim(objects, incarnation="inc-b", fork=fork)

    assert lease.fork == fork
    assert [record["fork"] for record in TrainerLease.chain(objects)] == [None, fork]


def test_a_lease_record_of_another_schema_refuses_the_claim(objects):
    TrainerLease.claim(objects, incarnation="inc-a")
    objects.write(lease_key(2), json.dumps({"schema": "yeto.rl.trainer-lease/v2", "generation": 2}).encode())

    with pytest.raises(TrainerLeaseError, match="another yeto version"):
        TrainerLease.claim(objects, incarnation="inc-b")
    assert TrainerLease.current(objects) == 2  # readers still see that a newer trainer owns the store


def test_an_unknown_object_under_the_lease_directory_is_refused_not_skipped(objects):
    objects.write(f"{LEASE_DIR}/g3.json", b"{}")

    with pytest.raises(TrainerLeaseError, match="unexpected object"):
        TrainerLease.current(objects)


def test_a_check_that_no_longer_finds_its_own_record_refuses_to_go_on(objects):
    lease = TrainerLease.claim(objects, incarnation="inc-a")
    objects.delete(lease_key(1))

    with pytest.raises(TrainerLeaseError, match="is gone"):
        lease.check(force=True)


def test_an_endpoint_without_conditional_writes_cannot_carry_the_lease():
    objects = _s3(FakeS3(conditional=False))

    with pytest.raises(TrainerLeaseError, match="atomic exclusive create"):
        TrainerLease.claim(objects, incarnation="inc-a")


def test_the_lease_lives_on_the_object_api_of_a_store_or_on_its_directory(tmp_path):
    staged = CheckpointStore(tmp_path / "stage")
    staged.objects = _s3()
    assert lease_objects(staged) is staged.objects
    assert lease_objects(staged.objects) is staged.objects

    plain = CheckpointStore(tmp_path / "plain")
    assert TrainerLease.claim(plain, incarnation="inc-a").generation == 1
    assert (tmp_path / "plain" / LEASE_DIR / "00000001.json").is_file()
    assert TrainerLease.current(plain) == 1 and TrainerLease.current(tmp_path / "plain") == 1

    with pytest.raises(TrainerLeaseError, match="bucket mount"):
        lease_objects(BucketMountStore(tmp_path / "mount"))
