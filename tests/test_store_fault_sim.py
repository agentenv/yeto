"""Shared-store consistency of an async GRPO run under faults (tests/store_sim).

Real sampler, trainer and pool store code runs on a fault-injecting fake S3
under a seeded scheduler; after every run the store and what the trainer
consumed are checked: nothing trained twice, nothing lost, no wrong version, no
fenced writer trained, and the run ends.

What runs by default is the short subset for CI: ``YETO_STORE_SIM_SEEDS`` seeds per
scenario (default 4, about twenty seconds for the file). The full run, by hand or
nightly, is ``YETO_STORE_SIM_FULL=1`` (100 seeds per scenario, and the mutation
cases of ``tests/test_store_fault_sim_mutations.py``); ``YETO_STORE_SIM_FIRST_SEED``
moves the window. A failure prints the seed and its trace;
``python -m tests.store_sim.run --seeds S:S+1 --faults <preset>`` replays it.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import random

import pytest
from botocore.exceptions import ClientError

from store_sim.fault_s3 import Crash, FaultConfig, FaultyS3Client, Injector, S3Server, SimClock
from store_sim.sim import Result, SimConfig, run_seed
from yeto.cloud.s3_store import S3ObjectStore
from yeto.rl.adapters.miles import buffer_rollout
from yeto.rl.engine import incarnation_lease, sample_buffer, store_dispense
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_resume import handoff_requested

FULL = os.environ.get("YETO_STORE_SIM_FULL", "") not in ("", "0")
SEEDS = int(os.environ.get("YETO_STORE_SIM_SEEDS", "100" if FULL else "4"))
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

# Findings of the simulation that are reported and not fixed on main yet: the kind, whether
# this tree has the fix (then the kind is strict), and the largest share of a scenario's
# seeds that may show it meanwhile (above it something else broke the same invariant).
# Everything that is not listed here is strict everywhere.
ONE_GROUP_PER_TASK = hasattr(buffer_rollout, "TaskSlots")  # the reader trains one group per dispensed task
PLAN_IDENTITY = hasattr(store_dispense.StoreLedger, "plan_id")  # the dispenser names its plan in every record
# retire writes its marker before it lists the claims; a claim pass that won checks the marker
RETIRE_MARKER_FIRST = hasattr(store_dispense.StoreLedger, "retired_now")
GATE_RECHECK = hasattr(incarnation_lease.IncarnationGate, "recheck")  # the round is judged again before training
OPEN_FINDINGS: dict[str, tuple[bool, float]] = {
    # a sampler that fails between the buffer append of a group and its ``trained`` settlement
    # (or whose claim expires between its check and its write) leaves the slot to be run again
    "trained_twice_same_slot": (ONE_GROUP_PER_TASK, 0.5),
    # a claim taken while ``store_dispense.retire`` lists the ledger is not released by it, and
    # the retired process learns of its marker only at its next scan: it still writes that group
    "retired_writer_group_trained": (RETIRE_MARKER_FIRST, 0.25),
    # the gate judges a group when the take selects it, against a lease it caches for
    # ``refresh_s`` (30 s); the take may then wait for later groups and train it after a fence
    # the cached lease did not show yet (seen with a partitioned zombie, preset fence-zombie).
    # The fail-open read (a failed lease read taken for "no lease") is fixed (#373); the take
    # judges its round again, the leases read again, right before it hands it over (GATE_RECHECK)
    "stale_incarnation_group_trained": (GATE_RECHECK, 0.05),
    # until the old incarnation of a second start reads its lease both write the shard; a round
    # id written by both leaves the shard unresumable (resume_point: two data cursors / last groups)
    # Measured with 100 seeds (YETO_STORE_SIM_FULL=1, main at 1c1783af): preset second-start
    # shows it in 7 seeds (7, 14, 67, 75, 86, 93, 96), under the ceiling of 0.1 (11 seeds)
    "shard_left_unresumable": (SHARD_OWNERSHIP, 0.1),
}
# A violation that is only the consequence of an open finding in the same seed: kind -> (the
# finding, what the detail of the violation must name). An unresumable shard makes every restart
# of its sampler fail on the same SamplerResumeError, and the run ends on the crash-loop budget
# (seeds 75, 86 and 96 of second-start); a run that does not end for any other reason is strict.
CONSEQUENCES: dict[str, tuple[str, str]] = {
    "run_did_not_end": ("shard_left_unresumable", "SamplerResumeError"),
}
# ... and only where a fenced process writes without any check of its own (preset fence-bypass):
# the lease gate caches the lease for 30 s, and the next incarnation seals what it finds
UNCHECKED_WRITES = {"fenced_writer_group_trained": (SHARD_OWNERSHIP, 0.9)}


@pytest.fixture(autouse=True)
def _quiet_dispenser():
    logger = logging.getLogger("yeto")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(level)


def _run(preset: str, *, seeds: int | None = None, **config) -> list[Result]:
    config.setdefault("checks", CHECKS)
    return [run_seed(SimConfig(seed=seed, faults=FaultConfig.preset(preset), **config))
            for seed in range(FIRST, FIRST + (seeds or SEEDS))]


def _assert_holds(results: list[Result], *, open_findings: dict[str, tuple[bool, float]] | None = None) -> None:
    """No seed violates an invariant; ``open_findings``: the kinds this scenario may
    still show (:data:`OPEN_FINDINGS`), each up to its share of the seeds."""
    allowed = {kind: share for kind, (fixed, share) in (open_findings or {}).items() if not fixed}
    bad = [r for r in results if _unexplained(r, allowed)]
    if bad:
        kinds = sorted({k for r in bad for k in _unexplained(r, allowed)})
        pytest.fail(f"{len(bad)} of {len(results)} seeds violate an invariant ({', '.join(kinds)}); "
                    f"seeds {[r.config.seed for r in bad][:20]}\n" + bad[0].report(), pytrace=False)
    for kind, share in allowed.items():
        seeds = _seeds_with(results, kind)
        if len(seeds) > int(share * len(results)) + 1:
            pytest.fail(f"{kind} (a known, unfixed finding) shows in {len(seeds)} of {len(results)} seeds, "
                        f"above its ceiling of {share:.0%}: seeds {seeds[:20]}\n"
                        + next(r for r in results if kind in r.violations).report(), pytrace=False)


def _unexplained(result: Result, allowed: dict[str, float]) -> set[str]:
    """The kinds of a seed that neither are allowed open findings nor only their consequence
    (:data:`CONSEQUENCES`: the cause shows in the same seed and every item names it)."""
    kinds = set(result.kinds()) - set(allowed)
    for kind, (cause, names) in CONSEQUENCES.items():
        if (kind in kinds and cause in allowed and cause in result.violations
                and all(names in json.dumps(item, default=str) for item in result.violations[kind])):
            kinds.discard(kind)
    return kinds


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
    # strict but for the claim that expires between its check and its write (OPEN_FINDINGS)
    _assert_holds(_run("lost-response"), open_findings={"trained_twice_same_slot": OPEN_FINDINGS["trained_twice_same_slot"]})


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
    _assert_holds(_run("network"), open_findings=OPEN_FINDINGS)


def test_crashes_partitions_and_pool_actions_break_nothing_but_the_open_findings():
    _assert_holds(_run("process"), open_findings=OPEN_FINDINGS)


@needs_create_answer_fix
def test_every_fault_at_once_breaks_nothing_but_the_open_findings():
    _assert_holds(_run("all"), open_findings=OPEN_FINDINGS)


@pytest.mark.xfail(not ONE_GROUP_PER_TASK, strict=False,
                   reason="open finding: a slot can be trained twice when its sampler fails between the "
                          "buffer append and the trained settlement (fixed by one group per dispensed task)")
def test_no_slot_is_trained_twice_under_every_fault_at_once():
    # strict on the slot; the other open findings (of other layers) keep their shares
    _assert_holds(_run("all"), open_findings={kind: finding for kind, finding in OPEN_FINDINGS.items()
                                              if kind != "trained_twice_same_slot"})


def test_the_tasks_of_samplers_that_died_are_taken_over_by_the_others():
    """Samplers die holding claims and nobody restarts them: only the lease expiry
    frees their tasks, and every task still ends."""
    # a sampler dies only while it holds groups in flight: a few seeds may see none
    results = _run("abandon", seeds=max(SEEDS, 12))
    assert sum(r.metrics.get("samplers_abandoned", 0) for r in results) > 0
    _assert_holds(results, open_findings=OPEN_FINDINGS)


def test_a_pool_that_acts_all_the_time_breaks_nothing_but_the_open_findings():
    _assert_holds(_run("pool"), open_findings=OPEN_FINDINGS)


def test_a_second_start_of_a_running_sampler_leaves_its_shard_resumable():
    results = _run("second-start")
    assert sum(r.metrics.get("second_starts", 0) for r in results) > 0
    _assert_holds(results, open_findings=OPEN_FINDINGS)


def test_a_fenced_process_that_writes_without_any_check_is_kept_out_by_the_trainer():
    """The fenced process writes what it has in flight unchecked (it passed its checks
    before the fence): only the trainer's side keeps those groups out."""
    results = _run("fence-bypass")
    assert sum(r.metrics.get("writes_unchecked", 0) for r in results) > 0
    _assert_holds(results, open_findings={**OPEN_FINDINGS, **UNCHECKED_WRITES})


@pytest.mark.xfail(not PLAN_IDENTITY, strict=True,
                   reason="a restart settles the tasks of a new dispenser namespace from the groups its shard "
                          "holds of the old one (recover matches by sampler and position only)")
def test_a_new_dispenser_namespace_over_the_same_buffer_runs_every_one_of_its_tasks():
    """More data or one more pass: the operator restarts the samplers on a new
    namespace while the buffer and its shards stay."""
    _assert_holds(_run("none", namespace_switch_at=20), open_findings=OPEN_FINDINGS)


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
    assert sum(n for r in results for name, n in r.metrics.items() if name.startswith("pool_")) > 0
    _assert_holds(results, open_findings=OPEN_FINDINGS)


def test_samplers_of_the_rules_before_shard_ownership_run_next_to_new_ones():
    """Half of the shards are written by samplers without the shard ownership
    records (a legacy bucket that is continued, or old and new code side by side)."""
    _assert_holds(_run("process", legacy_samplers=2, samplers=4), open_findings=OPEN_FINDINGS)


def test_the_nc_ownership_check_plugs_in():
    result = run_seed(SimConfig(seed=FIRST, checks=("store_sim.checks:nc_ownership",)))
    assert result.ok, result.report()
    assert "nc_changing_calls" in result.metrics or "nc_ownership_not_available" in result.metrics


@pytest.mark.xfail(not SAMPLER_SURVIVES_READS, strict=True,
                   reason="a sampler exits on a failed store read and gives its claims back")
def test_a_read_outage_of_ten_minutes_ends_no_sampler_and_releases_no_claim():
    results = _run("read-outage", lease_s=1800.0, checks=(*CHECKS, "store_sim.checks:read_outage"))
    assert sum(r.metrics.get("read_outages", 0) for r in results) > 0
    _assert_holds(results, open_findings=OPEN_FINDINGS)


def test_releases_whose_multipart_uploads_fail_part_way_are_never_named_half_uploaded():
    """Parts and completes of the policy releases fail on every attempt, often: the
    release LATEST names is whole, and samplers load only whole releases."""
    results = _run("upload")
    assert sum(r.metrics.get("fault_upload_outage", 0) for r in results) > 0
    _assert_holds(results, open_findings=OPEN_FINDINGS)


def test_a_claim_create_applied_whose_call_raised_never_lets_a_drain_be_answered_over_it():
    # PR #380: before it, seeds 4 / 5 / 19 / 27 of this preset each answered a drain while
    # the store held a claim of the drained process (drain_answered_while_holding_a_claim)
    seeds = sorted({4, 5, 19, 27} | set(range(FIRST, FIRST + SEEDS)))
    results = [run_seed(SimConfig(seed=seed, faults=FaultConfig.preset("create-answer-lost"), checks=CHECKS))
               for seed in seeds]
    assert not _seeds_with(results, "drain_answered_while_holding_a_claim")
    assert sum(r.metrics.get("fault_create_answer_lost", 0) for r in results)
