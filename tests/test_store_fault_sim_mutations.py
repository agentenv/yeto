"""The store simulation must turn red when the product is wrong (tests/store_sim).

Each case breaks one rule of the product code in memory and runs seeds of one
scenario until an invariant that names the damage is violated. The same seeds
must not violate that invariant without the mutation, so the invariant is one
this tree really holds. A case that stays green is a hole in the simulation: add
the scenario or the invariant, never widen what is tolerated.

Part of the full run (``YETO_STORE_SIM_FULL=1``, by hand or nightly; about half a
minute): the default run of the repository skips this file to stay short.
``YETO_STORE_SIM_MUTATION_SEEDS`` (default 40): the seeds a mutation gets. With
``-rA`` each case prints the seed that turned red and the invariant that did.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from store_sim.fault_s3 import FaultConfig
from store_sim.sim import MERGE_DROPS_FENCED_WRITES, Result, SimConfig, run_seed
from yeto.cloud import s3_store
from yeto.rl.adapters.miles import buffer_rollout
from yeto.rl.engine import buffer_payload, incarnation_lease, policy_release, sample_buffer, store_dispense

SEEDS = int(os.environ.get("YETO_STORE_SIM_MUTATION_SEEDS", "40"))
pytestmark = pytest.mark.skipif(os.environ.get("YETO_STORE_SIM_FULL", "") in ("", "0"),
                                reason="part of the full store simulation run (YETO_STORE_SIM_FULL=1)")
IDEMPOTENT_MERGE = hasattr(sample_buffer.FileSampleBuffer, "_is_merged")


@pytest.fixture(autouse=True)
def _quiet_dispenser():
    logger = logging.getLogger("yeto")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(level)


# ------------------------------------------------------------------ the mutations


def _still_holds_never_sees_a_later_claim(mp: Any) -> None:
    mp.setattr(store_dispense.StoreLedger, "superseded", lambda self, slot, attempt: False)


def _a_lost_claim_counts_as_won(mp: Any) -> None:
    real = store_dispense.StoreLedger.claim

    def claim(self, slot, attempt, **detail):
        real(self, slot, attempt, **detail)
        self._bodies[(int(slot), int(attempt))] = {"holder": self.holder, "incarnation": self.incarnation}
        return True

    mp.setattr(store_dispense.StoreLedger, "claim", claim)


def _the_gate_fences_nothing(mp: Any) -> None:
    mp.setattr(incarnation_lease.IncarnationGate, "verdict", lambda self, entry, position: None)


def _a_restored_trainer_starts_one_past_its_watermark(mp: Any) -> None:
    real = buffer_rollout.BatchReader.reset

    def reset(self, watermark):
        self.buffer.refresh()
        return real(self, min(int(watermark) + 1, len(self.buffer)) if int(watermark) else 0)

    mp.setattr(buffer_rollout.BatchReader, "reset", reset)


def _the_merge_drops_every_seventh_shard_record(mp: Any) -> None:
    real = sample_buffer.ObjectIndex.shard_reader

    def shard_reader(self, shard, start):
        reader = real(self, shard, start)
        read_new = reader.read_new
        reader.read_new = lambda: [r for r in read_new() if "sample_id" not in r or int(r["pos"]) % 7 != 3]
        return reader

    mp.setattr(sample_buffer.ObjectIndex, "shard_reader", shard_reader)


def _a_restarted_merger_merges_the_last_record_again(mp: Any) -> None:
    real = sample_buffer.ObjectIndex.shard_reader
    mp.setattr(sample_buffer.ObjectIndex, "shard_reader",
               lambda self, shard, start: real(self, shard, max(0, int(start) - 1)))


def _the_trainer_merges_only_its_first_shard(mp: Any) -> None:
    real = sample_buffer.open_trainer_buffer

    def open_trainer_buffer(store, *, shards=None, **kwargs):
        return real(store, shards=list(shards or ())[:1], **kwargs)

    import store_sim.actors as actors

    mp.setattr(actors, "open_trainer_buffer", open_trainer_buffer)


def _a_restart_does_not_settle_what_its_shard_holds(mp: Any) -> None:
    real = store_dispense.StoreDispensedPrompts.recover
    mp.setattr(store_dispense.StoreDispensedPrompts, "recover", lambda self, entries=(): real(self, ()))


def _the_follower_reports_the_version_before(mp: Any) -> None:
    real = policy_release.PolicyFollower.refresh

    def refresh(self):
        out = real(self)
        if out is not None and self.version:
            self.version -= 1
        return out

    mp.setattr(policy_release.PolicyFollower, "refresh", refresh)


def _a_group_is_indexed_with_the_next_version(mp: Any) -> None:
    real = buffer_payload.group_entry_fields

    def group_entry_fields(samples):
        version, tokens = real(samples)
        return version + 1, tokens

    mp.setattr(buffer_payload, "group_entry_fields", group_entry_fields)


def _every_fifth_group_is_reported_written_without_a_write(mp: Any) -> None:
    real = buffer_payload.GroupWriter.write
    calls = [0]

    def write(self, samples, **kwargs):
        calls[0] += 1
        if calls[0] % 5 == 0:
            return None
        return real(self, samples, **kwargs)

    import store_sim.actors as actors

    mp.setattr(actors.GroupWriter, "write", write)


def _an_expired_attempt_gets_no_ending(mp: Any) -> None:
    real = store_dispense.StoreLedger.fail
    mp.setattr(store_dispense.StoreLedger, "fail",
               lambda self, slot, attempt, kind, **detail:
               False if kind == "expired" else real(self, slot, attempt, kind, **detail))


def _a_claim_never_expires(mp: Any) -> None:
    mp.setattr(store_dispense.TaskState, "last_seen", lambda self, attempt: float("inf"))


def _the_trainer_reads_the_payload_of_the_next_position(mp: Any) -> None:
    real = sample_buffer.FileSampleBuffer.read_payload

    def read_payload(self, ref):
        refs = [e.payload_ref for e in self._entries]
        i = refs.index(ref) if ref in refs else -1
        return real(self, refs[i + 1] if 0 <= i < len(refs) - 1 else ref)

    mp.setattr(sample_buffer.FileSampleBuffer, "read_payload", read_payload)


def _a_release_carries_a_file_of_the_release_before(mp: Any) -> None:
    import store_sim.actors as actors

    real = policy_release.PolicyReleaseStore.release_staged

    def release_staged(self, version, policy_hash, staged, **kwargs):
        name = "weights-00001.bin"
        (staged / name).write_bytes(actors.release_files(int(version) - 1, "another")[name])
        return real(self, version, policy_hash, staged, **kwargs)

    mp.setattr(policy_release.PolicyReleaseStore, "release_staged", release_staged)


def _the_prune_deletes_every_release(mp: Any) -> None:
    # ``_kept_names(names)`` (#377): the releases of the store in prune order, the new one last.
    # A mutation of the wrong signature raises in the trainer after LATEST is written and is
    # caught by nothing: keep the arguments of the product method
    mp.setattr(policy_release.PolicyReleaseStore, "_kept_names", lambda self, names: set())


def _a_new_lease_supersedes_nobody(mp: Any) -> None:
    real = incarnation_lease.IncarnationLease.claim

    def claim(self, entries=(), earlier=()):
        record = real(self, entries, earlier)
        record.update(superseded=[], sealed={})
        incarnation_lease._put(self.store, incarnation_lease.lease_key(self.sampler), record)
        return record

    mp.setattr(incarnation_lease.IncarnationLease, "claim", claim)


def _a_fenced_incarnation_never_learns_it(mp: Any) -> None:
    mp.setattr(incarnation_lease.IncarnationLease, "poll", lambda self: False)
    mp.setattr(incarnation_lease.IncarnationLease, "check_before_write", lambda self: False)


def _a_drain_is_answered_with_claims_still_held(mp: Any) -> None:
    def answer(self):
        seq = self.ledger.drain_seq
        if seq is not None and self._drain_answered != seq:
            self.ledger.answer_drain(seq)
            self._drain_answered = seq

    mp.setattr(store_dispense.StoreDispensedPrompts, "_answer_drain", answer)


# -- the object store under the real S3ObjectStore: lost answers, cut bodies, multipart, one key


def _a_create_refused_on_the_retry_is_taken_for_a_loss(mp: Any) -> None:
    mp.setattr(s3_store.S3ObjectStore, "_created_by_retry", lambda self, error, key, data: False)


def _every_refused_create_is_taken_for_a_win(mp: Any) -> None:
    mp.setattr(s3_store.S3ObjectStore, "_created_by_retry", lambda self, error, key, data: True)


def _a_create_checks_then_writes(mp: Any) -> None:
    def create(self, key, data):  # no IfNoneMatch: two writers of one key both win
        if self.exists(key):
            return False
        self.write(key, data)
        return True

    mp.setattr(s3_store.S3ObjectStore, "create", create)


def _a_cut_body_is_read_as_what_arrived(mp: Any) -> None:
    def read(self, key):  # the body that arrived is taken for the object instead of read again
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except self.client.exceptions.NoSuchKey:
            return None
        try:
            return response["Body"].read()
        except s3_store.RANGE_READ_RETRYABLE:
            data = response["Body"]._data  # store_sim.fault_s3._Body: what was sent
            return data[: len(data) // 2]

    mp.setattr(s3_store.S3ObjectStore, "read", read)


def _a_failed_multipart_upload_is_reported_done(mp: Any) -> None:
    real = s3_store.S3ObjectStore._write_file

    def write_file(self, key, source):
        try:
            return real(self, key, source)
        except Exception:  # noqa: BLE001 - the mutation: half an upload is taken for the file
            return None

    mp.setattr(s3_store.S3ObjectStore, "_write_file", write_file)


def _a_multipart_upload_completes_with_its_first_part_only(mp: Any) -> None:
    real = s3_store.S3ObjectStore._send_parts

    def send_parts(self, journal, full, source, part, size, count, checks):
        real(self, journal, full, source, part, size, count, checks)
        for n in range(2, count + 1):
            journal.parts.pop(n, None)

    mp.setattr(s3_store.S3ObjectStore, "_send_parts", send_parts)
    real_complete = s3_store.S3ObjectStore._write_file

    def write_file(self, key, source):
        import os as _os

        size = _os.path.getsize(source)
        part = self.transfer.part_for(size)
        if size <= part:
            return real_complete(self, key, source)
        with open(source, "rb") as fh:
            head = fh.read(part)
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=head)
        return None

    mp.setattr(s3_store.S3ObjectStore, "_write_file", write_file)


def _a_restored_trainer_starts_one_before_its_watermark(mp: Any) -> None:
    real = buffer_rollout.BatchReader.reset
    mp.setattr(buffer_rollout.BatchReader, "reset",
               lambda self, watermark: real(self, max(0, int(watermark) - 1)))


def _the_gate_trains_one_entry_past_the_seal(mp: Any) -> None:
    real = incarnation_lease.IncarnationGate._lease

    def _lease(self, sampler, *, force=False):  # ``ordinal <= sealed`` becomes ``ordinal <= sealed + 1``
        record = real(self, sampler, force=force)
        if record and record.get("superseded"):
            sealed = record.get("sealed") or {}
            record = dict(record, sealed={str(i): int(sealed.get(str(i), 0)) + 1 for i in record["superseded"]})
        return record

    mp.setattr(incarnation_lease.IncarnationGate, "_lease", _lease)


def _the_merge_takes_a_record_of_another_shard_for_merged(mp: Any) -> None:
    def _is_merged(self, record):  # keyed by the shard position alone, not by (shard, position)
        if record.get("end"):
            return self._ended
        if record.get("shard") is None:
            return False
        return any(int(record["shard_pos"]) in seen for seen in self._shard_seen.values())

    mp.setattr(sample_buffer.FileSampleBuffer, "_is_merged", _is_merged)


def _a_restart_resumes_from_another_shard(mp: Any) -> None:
    import store_sim.actors as actors

    real = actors.resume_point

    def resume_point(entries, sampler):
        sid = str(sampler)
        other = f"{sid[:-1]}{(int(sid[-1]) + 1) % 2}" if sid[-1:].isdigit() else sid
        return real(entries, other)

    mp.setattr(actors, "resume_point", resume_point)


@dataclass(frozen=True)
class Mutation:
    apply: Callable[[Any], None]
    preset: str  # the scenario that exposes it (store_sim.fault_s3.PRESETS)
    red: tuple[str, ...]  # the invariants that name the damage
    config: tuple[tuple[str, Any], ...] = ()
    skip: str | None = None  # why this tree cannot show it
    # seeds to try (None: SEEDS): a damage that needs a rare interleaving (a fence that lands
    # while a zombie still writes) shows in few seeds, and which ones moves with any timing change
    seeds: int | None = None

    @property
    def id(self) -> str:
        return self.apply.__name__.strip("_")


_GATE = ("stale_incarnation_group_trained",) if not MERGE_DROPS_FENCED_WRITES else ("fenced_writer_group_trained",)
MUTATIONS = [
    # -- the dispenser: one holder per task, the holder knows when it lost the task
    Mutation(_still_holds_never_sees_a_later_claim, "network", ("group_written_without_its_claim",)),
    Mutation(_a_lost_claim_counts_as_won, "network", ("group_written_without_its_claim",)),
    # red on main at 1c1783af in seeds 50, 71, 74, 86, 94, 99, 117 of the first 120 (none of the first 40)
    Mutation(_a_restart_does_not_settle_what_its_shard_holds, "process", ("restart_left_a_written_slot_unsettled",),
             seeds=max(SEEDS, 120)),
    Mutation(_an_expired_attempt_gets_no_ending, "network", ("attempt_superseded_without_an_ending",)),
    Mutation(_a_claim_never_expires, "abandon", ("task_of_a_dead_holder_never_taken_over", "run_did_not_end")),
    Mutation(_every_fifth_group_is_reported_written_without_a_write, "none",
             ("slot_settled_trained_without_a_group",)),
    Mutation(_a_drain_is_answered_with_claims_still_held, "pool", ("drain_answered_while_holding_a_claim",)),
    # -- fencing: a superseded incarnation's late groups are not trained
    Mutation(_the_gate_fences_nothing, "fence-zombie", _GATE, seeds=max(SEEDS, 120)),
    Mutation(_the_gate_trains_one_entry_past_the_seal, "fence-zombie", _GATE, seeds=max(SEEDS, 120)),
    Mutation(_a_new_lease_supersedes_nobody, "process", ("lease_forgot_a_superseded_incarnation",)),
    Mutation(_a_fenced_incarnation_never_learns_it, "second-start", ("fenced_incarnation_still_writes",)),
    # -- the buffer: every shard record merged once, read as itself, consumed once per cut lineage
    Mutation(_the_merge_drops_every_seventh_shard_record, "none", ("shard_record_never_merged", "run_did_not_end")),
    Mutation(_a_restarted_merger_merges_the_last_record_again, "process", ("shard_record_merged_twice",),
             skip="the idempotent merge leaves the repeated record out" if IDEMPOTENT_MERGE else None),
    Mutation(_the_merge_takes_a_record_of_another_shard_for_merged, "process", ("shard_record_never_merged",),
             skip=None if IDEMPOTENT_MERGE else "the merge of this tree keeps no merged positions"),
    Mutation(_a_restart_resumes_from_another_shard, "process", ("shard_left_unresumable",)),
    Mutation(_the_trainer_merges_only_its_first_shard, "none",
             ("slot_settled_trained_but_never_consumed", "shard_record_never_merged", "run_did_not_end")),
    Mutation(_a_restored_trainer_starts_one_past_its_watermark, "process", ("merged_record_never_consumed",)),
    Mutation(_the_trainer_reads_the_payload_of_the_next_position, "none", ("trained_payload_of_another_group",)),
    # -- versions and releases
    Mutation(_the_follower_reports_the_version_before, "none",
             ("follower_reports_another_version_than_was_loaded",)),
    Mutation(_a_group_is_indexed_with_the_next_version, "none", ("trained_with_another_behavior_version",)),
    Mutation(_a_release_carries_a_file_of_the_release_before, "none", ("release_files_of_another_version",)),
    Mutation(_the_prune_deletes_every_release, "none", ("latest_release_is_incomplete",)),
    # -- the object store itself
    Mutation(_a_create_refused_on_the_retry_is_taken_for_a_loss, "lost-response",
             ("shard_record_written_twice", "claim_held_without_the_holder_knowing", "trained_twice_same_slot",
              "shard_record_merged_twice")),
    Mutation(_every_refused_create_is_taken_for_a_win, "network",
             ("group_written_without_its_claim", "shard_record_nobody_wrote", "trained_group_nobody_wrote")),
    Mutation(_a_create_checks_then_writes, "network",
             ("group_written_without_its_claim", "shard_record_nobody_wrote", "trained_group_nobody_wrote")),
    Mutation(_a_cut_body_is_read_as_what_arrived, "network", ("read_returned_bytes_the_key_never_held",)),
    # red on main at 1c1783af in seeds 42, 81 and 93 of the first 120 (none of the first 40)
    Mutation(_a_failed_multipart_upload_is_reported_done, "upload-heavy",
             ("latest_release_is_incomplete", "release_files_of_another_version"), seeds=max(SEEDS, 120)),
    Mutation(_a_multipart_upload_completes_with_its_first_part_only, "none",
             ("latest_release_is_incomplete", "release_files_of_another_version", "store_left_unreadable")),
    Mutation(_a_restored_trainer_starts_one_before_its_watermark, "process",
             ("trained_twice_same_slot", "trained_twice_same_sample_id", "merged_record_consumed_twice")),
]

_baseline: dict[tuple[Any, ...], Result] = {}
FIRST_RED: dict[str, int] = {}  # mutation -> the seed that turned red (printed by -rA / the summary)


def _config(mutation: Mutation, seed: int) -> SimConfig:
    return SimConfig(seed=seed, faults=FaultConfig.preset(mutation.preset), **dict(mutation.config))


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.id)
def test_the_simulation_turns_red_when(mutation: Mutation, monkeypatch, record_property):
    if mutation.skip:
        pytest.skip(mutation.skip)
    seeds = mutation.seeds or SEEDS
    for seed in range(seeds):  # the invariant holds on this tree as it is ...
        key = (mutation.preset, mutation.config, seed)
        if key not in _baseline:
            _baseline[key] = run_seed(_config(mutation, seed))
        held = set(_baseline[key].kinds()) & set(mutation.red)
        assert not held, f"{held} is violated without the mutation (seed {seed}):\n" + _baseline[key].report()
        with monkeypatch.context() as patch:  # ... and breaks with the mutation
            mutation.apply(patch)
            result = run_seed(_config(mutation, seed))
        hit = set(result.kinds()) & set(mutation.red)
        if hit:
            FIRST_RED[mutation.id] = seed
            record_property("first_red_seed", seed)
            print(f"{mutation.id}: red at seed {seed} of preset {mutation.preset}: {sorted(hit)}")
            return
    pytest.fail(f"{mutation.id}: {seeds} seeds of preset {mutation.preset} never violated {mutation.red}",
                pytrace=False)
