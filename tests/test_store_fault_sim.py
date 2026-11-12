"""Shared-store consistency of an async GRPO run under faults (tests/store_sim).

Real sampler, trainer and pool store code runs on a fault-injecting fake S3
under a seeded scheduler; after every run the store and what the trainer
consumed are checked: nothing trained twice, nothing lost, no wrong version, no
fenced writer trained, and the run ends.

``YETO_STORE_SIM_SEEDS`` (default 20): seeds per scenario;
``YETO_STORE_SIM_FIRST_SEED`` (default 0). A failure prints the seed and its
trace; ``python -m tests.store_sim.run --seeds S:S+1 --faults <preset>`` replays it.
"""

from __future__ import annotations

import inspect
import logging
import os
import random

import pytest
from botocore.exceptions import ClientError

from store_sim.fault_s3 import Crash, FaultConfig, FaultyS3Client, Injector, S3Server, SimClock
from store_sim.sim import Result, SimConfig, run_seed
from yeto.cloud.s3_store import S3ObjectStore
from yeto.rl.engine import sample_buffer
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_resume import handoff_requested

SEEDS = int(os.environ.get("YETO_STORE_SIM_SEEDS", "20"))
FIRST = int(os.environ.get("YETO_STORE_SIM_FIRST_SEED", "0"))

# PR #354: a refused conditional put that the SDK had retried is read back
CREATE_ANSWER_FIX = hasattr(S3ObjectStore, "_created_by_retry")
needs_create_answer_fix = pytest.mark.xfail(
    not CREATE_ANSWER_FIX, strict=True,
    reason="S3ObjectStore.create takes its own retried create for a lost race (fixed by _created_by_retry)")

# PR #359: the shard merge is idempotent (a merged record is its shard position)
IDEMPOTENT_MERGE = hasattr(FileSampleBuffer, "_is_merged")
MERGE_VIOLATIONS = ("shard_record_merged_twice", "shard_record_never_merged", "shard_merged_out_of_order",
                    "merged_index_position_gap", "trained_twice_same_record")
# shard ownership records: a fenced writer's group is refused at the write or left out of the merge
SHARD_OWNERSHIP = hasattr(sample_buffer, "ShardFenced")
# a failed store read neither ends a sampler nor releases its claims
SAMPLER_SURVIVES_READS = "on_error" in inspect.signature(handoff_requested).parameters
try:  # the trainer generation lease
    import yeto.rl.engine.trainer_lease  # noqa: F401
    TRAINER_LEASE = True
except ImportError:
    TRAINER_LEASE = False
# more invariants of other suites, as package.module:name (store_sim.sim.load_check)
CHECKS = tuple(c for c in os.environ.get("YETO_STORE_SIM_CHECKS", "").split(",") if c)

# Violations the simulation finds that are reported and not fixed yet. A scenario
# that can hit one tolerates exactly these kinds; everything else stays strict.
OPEN_FINDINGS = {
    # a sampler that fails (or is overtaken by an expiry) between the buffer append of
    # a group and its ``trained`` settlement leaves the slot to be claimed again
    "trained_twice_same_slot",
    # a claim taken while ``store_dispense.retire`` lists the ledger is not released by it, and
    # the retired process learns of its marker only at its next scan: it still writes that group
    "retired_writer_group_trained",
}


@pytest.fixture(autouse=True)
def _quiet_dispenser():
    logger = logging.getLogger("yeto")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(level)


def _run(preset: str, **config) -> list[Result]:
    config.setdefault("checks", CHECKS)
    return [run_seed(SimConfig(seed=seed, faults=FaultConfig.preset(preset), **config))
            for seed in range(FIRST, FIRST + SEEDS)]


def _assert_holds(results: list[Result], *, tolerate: frozenset[str] | set[str] = frozenset()) -> None:
    bad = [r for r in results if set(r.kinds()) - set(tolerate)]
    if bad:
        kinds = sorted({k for r in bad for k in r.kinds()})
        pytest.fail(f"{len(bad)} of {len(results)} seeds violate an invariant ({', '.join(kinds)}); "
                    f"seeds {[r.config.seed for r in bad][:20]}\n" + bad[0].report(), pytrace=False)


def _seeds_with(results: list[Result], kind: str) -> list[int]:
    return [r.config.seed for r in results if kind in r.violations]


# ------------------------------------------------------------------ the fault layer


def _store(injector: Injector | None = None) -> tuple[S3Server, S3ObjectStore]:
    server = S3Server(SimClock())
    return server, S3ObjectStore("s3://store-sim", client=FaultyS3Client(server, "writer", injector))


def test_a_lost_create_answer_is_refused_on_the_sdk_retry_with_retry_attempts():
    clock = SimClock()
    injector = Injector(random.Random(0), clock, FaultConfig(lost_response=1.0, retry_fault=0.0))
    server = S3Server(clock)
    client = FaultyS3Client(server, "writer", injector)
    with pytest.raises(ClientError) as refused:
        client.put_object(Bucket="store-sim", Key="k", Body=b"mine", IfNoneMatch="*")
    assert refused.value.response["Error"]["Code"] == "PreconditionFailed"
    assert refused.value.response["ResponseMetadata"]["RetryAttempts"] == 1
    assert server.get("k") == b"mine" and server.own_retry_refusals == {"k": "writer"}


def test_a_create_lost_to_another_writer_is_refused_on_the_first_attempt():
    server, objects = _store()
    assert objects.create("k", b"theirs") is True
    with pytest.raises(ClientError) as refused:
        objects.client.put_object(Bucket="store-sim", Key="k", Body=b"mine", IfNoneMatch="*")
    assert refused.value.response["ResponseMetadata"]["RetryAttempts"] == 0
    assert objects.create("k", b"mine") is False and server.get("k") == b"theirs"


def test_a_process_killed_after_a_write_leaves_the_object_in_the_store():
    clock = SimClock()
    injector = Injector(random.Random(0), clock, FaultConfig())
    server = S3Server(clock)
    objects = S3ObjectStore("s3://store-sim", client=FaultyS3Client(server, "writer", injector))
    injector.crash_in["writer"] = (0, True)
    with pytest.raises(Crash):
        objects.write("k", b"data")
    assert server.get("k") == b"data"


def test_a_partitioned_node_reaches_nothing_and_the_listing_carries_server_time():
    clock = SimClock()
    injector = Injector(random.Random(0), clock, FaultConfig())
    server = S3Server(clock, page_size=2)
    objects = S3ObjectStore("s3://store-sim", client=FaultyS3Client(server, "writer", injector, node="n1"))
    for i in range(5):
        objects.write(f"p/{i}", b"x")
    keys, now = objects.list_meta("p/")
    assert [k for k, _ in keys] == [f"p/{i}" for i in range(5)] and int(now) == int(clock.now)
    injector.partitioned["n1"] = clock.now + 1000.0
    with pytest.raises(Exception, match="Could not connect"):
        objects.write("p/9", b"x")
    assert server.get("p/9") is None


# ---------------------------------------------------------------------- the runs


def test_a_run_without_faults_trains_every_group_exactly_once():
    results = _run("none")
    _assert_holds(results)
    for r in results:
        assert r.metrics["groups_trained"] == r.metrics["groups_written"] == r.config.slots, r.report()


def test_a_seed_replays_the_same_run():
    config = SimConfig(seed=FIRST + 3, faults=FaultConfig.preset("process"))
    first, again = run_seed(config), run_seed(config)
    assert first.trace == again.trace and first.metrics == again.metrics


@needs_create_answer_fix
def test_lost_answers_of_conditional_writes_neither_repeat_nor_lose_a_group():
    _assert_holds(_run("lost-response"))


@pytest.mark.skipif(CREATE_ANSWER_FIX, reason="the lost create answer is fixed in this tree")
def test_the_simulation_finds_what_a_lost_create_answer_breaks():
    """Without PR #354: a shard and the merged index hold a record twice, the
    trainer trains it twice, and a sampler holds a claim it does not know of."""
    results = _run("lost-response")
    kinds = ["shard_record_written_twice", "trained_twice_same_sample_id", "claim_held_without_the_holder_knowing"]
    if not IDEMPOTENT_MERGE:  # the trainer's merge loses the answer of a create too
        kinds.append("shard_record_merged_twice")
    for kind in kinds:
        assert _seeds_with(results, kind), f"no seed of {len(results)} shows {kind}"


@needs_create_answer_fix
def test_network_faults_break_nothing_but_the_open_findings():
    _assert_holds(_run("network"), tolerate=OPEN_FINDINGS)


def test_crashes_partitions_and_pool_actions_break_nothing_but_the_open_findings():
    _assert_holds(_run("process"), tolerate=OPEN_FINDINGS)


@needs_create_answer_fix
def test_every_fault_at_once_breaks_nothing_but_the_open_findings():
    _assert_holds(_run("all"), tolerate=OPEN_FINDINGS)


@pytest.mark.xfail(strict=False, reason="open finding: a slot can be trained twice when its sampler fails "
                                        "between the buffer append and the trained settlement")
def test_no_slot_is_trained_twice_under_every_fault_at_once():
    _assert_holds(_run("all"))


@pytest.mark.skipif(IDEMPOTENT_MERGE, reason="the shard merge is idempotent in this tree")
def test_the_simulation_finds_two_trainers_merging_the_same_shards():
    """Off in every other scenario (``two_trainers``): a trainer that is not dead
    yet and its restarted successor both merge the shards."""
    results = _run("two-trainers")
    assert _seeds_with(results, "shard_record_merged_twice"), "no seed shows a shard record merged twice"


@pytest.mark.xfail(not IDEMPOTENT_MERGE, strict=True,
                   reason="two trainers at once merge a shard record twice (fixed by the idempotent merge)")
def test_two_running_trainers_merge_every_shard_record_once():
    results = _run("two-trainers")
    bad = {kind: _seeds_with(results, kind) for kind in MERGE_VIOLATIONS if _seeds_with(results, kind)}
    assert not bad, f"seeds by violation: {bad}\n" + next(r for r in results if not r.ok).report()


@pytest.mark.xfail(not TRAINER_LEASE, strict=False,
                   reason="two trainers at once write the same release directory (fixed by the trainer generation)")
def test_two_running_trainers_never_hand_a_sampler_mixed_release_files():
    seeds = _seeds_with(_run("two-trainers", steps=600), "release_files_of_another_version")
    assert not seeds, f"seeds whose sampler loaded files of two releases: {seeds}"


def test_a_pool_fence_with_a_zombie_writer_and_skewed_clocks_trains_no_fenced_group():
    """The pool fences shards whose node is partitioned while the old process still
    writes, under clocks that are a minute apart."""
    results = _run("fence-zombie")
    assert sum(r.metrics.get("pool_fence_gone", 0) for r in results) > 0
    _assert_holds(results, tolerate=OPEN_FINDINGS)


def test_samplers_of_the_rules_before_shard_ownership_run_next_to_new_ones():
    """Half of the shards are written by samplers without the shard ownership
    records (a legacy bucket that is continued, or old and new code side by side)."""
    _assert_holds(_run("process", legacy_samplers=2, samplers=4), tolerate=OPEN_FINDINGS)


def test_the_nc_ownership_check_plugs_in():
    result = run_seed(SimConfig(seed=FIRST, checks=("store_sim.checks:nc_ownership",)))
    assert result.ok, result.report()
    assert "nc_changing_calls" in result.metrics or "nc_ownership_not_available" in result.metrics


@pytest.mark.xfail(not SAMPLER_SURVIVES_READS, strict=True,
                   reason="a sampler exits on a failed store read and gives its claims back")
def test_a_read_outage_of_ten_minutes_ends_no_sampler_and_releases_no_claim():
    results = _run("read-outage", lease_s=1800.0, checks=(*CHECKS, "store_sim.checks:read_outage"))
    assert sum(r.metrics.get("read_outages", 0) for r in results) > 0
    _assert_holds(results, tolerate=OPEN_FINDINGS)
