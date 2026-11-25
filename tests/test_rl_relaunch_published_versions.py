"""CPU: a full-parameter island relaunched (or peer-recovered) over its own policy
releases continues its version account above them.

The relaunch of a fleet island: one trainer rank died, the island left and the
launcher started a new process. The peer recovery restored the model, the
optimizer, the RNG and the fragment versions from a donor, but the session counts
its rounds from 0 in every process (``FullParameterSync.start``), so the driver
published v0 again while the island's sample buffer held groups its own samplers
generated with the releases v6..v8 of the earlier process, and the store's release
index of the island named v7, v8. The first round stopped the island
("policy v6 of a trained trajectory was not published") and the launcher
relaunched it again.

Now the driver continues the account at the newest release + 1, restores the
published table from the island's release index (rollout 0 included), and a
buffered group of a version that no record names any more is consumed untrained
and counted instead of stopping the island. Here: a real ``IslandDriver``, the
real ``PolicyReleaseStore`` (it keeps the newest two releases), the real
``ReleasingPublisher`` with the own-lane release guard, the real ``BatchReader``
over a ``FileSampleBuffer``; the session is a stand-in that counts its rounds from
0 in every process as ``FullParameterSync`` does (its own verb is tested below on
the real session).
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch
from test_rl_full_parameter_sync import _candidate, _Driver, _PairedRole
from test_rl_full_parameter_sync import _session as _full_parameter_session

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.adapters.miles.buffer_rollout import BatchReader, GroupWriter
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
from yeto.rl.engine.bridges import LocalOnlySync, _local_state
from yeto.rl.engine.buffer_payload import group_entry_fields
from yeto.rl.engine.driver import EventTape, IslandDriver, SyncStart
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import FakeEngine, FakeRolloutPool, fake_capabilities
from yeto.rl.engine.merged_release import MergedReleaseGuard
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

NAME = "base_model.model.layer.lora_A.weight"
GROUPS = 2  # groups per round


def _hash(version: int) -> str:
    return f"{version:064x}"


class _RestartingSession(LocalOnlySync):
    """A session that counts its rounds from 0 in every process, as
    ``FullParameterSync`` does, with its verb to continue above the releases."""

    local_rounds_restart = True

    def __init__(self, rounds: int) -> None:
        super().__init__(rounds)
        self.continued_at: int | None = None

    def continue_version_account_at(self, driver, rollout_id):
        self.continued_at = self.resumed_at = int(rollout_id)  # the rounds of this process from there
        return SyncStart(_local_state(driver, rollout_id), rollout_id)


class _Samplers:
    """The island's samplers: before every trainer round they write one round of
    groups generated with the newest release of the island's store."""

    def __init__(self, store) -> None:
        self.releases = PolicyReleaseStore(store)
        self.buffer = FileSampleBuffer(store, batch_size=1)
        self.writer = GroupWriter(self.buffer, "s0")
        self.written = 0
        self.versions: dict[str, int] = {}

    def write(self, version: int) -> None:
        g = self.written
        self.writer.write([sample(2 * g, g, version), sample(2 * g + 1, g, version)], group_key=f"g{g}")
        self.versions[f"g{g}"] = version
        self.written += 1

    def write_round(self) -> None:
        latest = self.releases.latest()
        for _ in range(GROUPS):
            self.write(0 if latest is None else int(latest.version))


class _BufferPool(FakeRolloutPool):
    """The rollout port of a buffer trainer: a round is the reader's next groups,
    each named by its plain behavior version (``MilesRolloutPool`` for a buffer trainer)."""

    def __init__(self, engine, reader, samplers) -> None:
        super().__init__(engine)
        self.reader = reader
        self.samplers = samplers

    def generate(self, rollout_id, *, expected_policy_version=None):
        version, digest = self.engine.published
        self.samplers.write_round()
        taken, report = self.reader.take(rollout_id, GROUPS, current_version=version)
        groups = []
        for _position, samples in taken:
            behavior, tokens = group_entry_fields(samples)
            groups.append(GroupMetadata(
                f"g{samples[0]['group_index']}", tuple(f"s{s['index']}" for s in samples),
                str(behavior), 0.5, 0.5, int(tokens), policy_versions=(int(behavior),)))
        return RolloutBatchHandle(
            rollout_id, version, digest, tuple(groups), len(groups), 0, payload=object(),
            buffer_stale_groups=report.stale_groups,
            buffer_unidentified_groups=report.unidentified_policy_groups or None,
            buffer_unidentified_versions=[[v, n] for v, n in report.unidentified_policy_versions.items()] or None)

    def set_unidentified_policy_gate(self, start_version, known_versions):
        self.reader.set_unidentified_gate(start_version, known_versions)


def _earlier_process(store, samplers, *, released=(6, 7, 8), buffered=(6, 7, 7, 8, 8)):
    """What the island's earlier process left: its releases (the store keeps the
    newest two) and the groups its samplers generated with them, not trained yet."""
    releases = PolicyReleaseStore(store)
    for version in released:
        releases.release(version, _hash(version), lambda d, v=version: (d / "w.bin").write_text(str(v)))
    for version in buffered:
        samplers.write(version)
    return releases


def _relaunch(tmp_path, *, age, rounds=3, released=(6, 7, 8), buffered=(6, 7, 7, 8, 8)):
    store = CheckpointStore(tmp_path / "store")
    samplers = _Samplers(store)
    releases = _earlier_process(store, samplers, released=released, buffered=buffered)
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, groups=GROUPS)
    reader = BatchReader(FileSampleBuffer(store, batch_size=1, create_index=False), max_policy_age=age,
                         wait_s=0.0, poll_s=0.0, read_workers=1, prefetch=False, incarnation_gate=None)
    engine.rollout = _BufferPool(engine, reader, samplers)
    algorithm = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=age),
                              correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    caps = fake_capabilities("miles-like")
    caps = dataclasses.replace(caps, execution=dataclasses.replace(caps.execution, max_policy_staleness=age))
    profile = ExecutionProfile("p", "colocated-serial", "strict-avg", max_policy_age=age,
                               algorithm_contract="bounded-staleness", groups_per_batch=GROUPS,
                               samples_per_group=engine.samples_per_group,
                               algorithm_spec_sha256=algorithm.sha256())
    tape = tmp_path / "relaunch.jsonl"
    session = _RestartingSession(rounds)
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, algorithm=algorithm, capabilities=caps,
        sync=session, events=EventTape(tape, 0), profile=profile)
    # wire_policy_releases on a buffer lane: the lane cursor holds the data position,
    # the island's own release index is the history of its published versions
    driver.data_position_owner = "buffer_lane"
    driver.published_policy_history = lambda: [
        (r.version, r.policy_hash) for r in releases.records() if r.derived_from is None]
    driver.publisher = si.ReleasingPublisher(
        driver.publisher, releases, lambda version, directory: (directory / "w.bin").write_text(str(version)),
        read_position=lambda: reader.buffer.read_cursor, read_trained=reader.trained_before,
        emit=driver.emit, guard=MergedReleaseGuard(lambda previous, version: None))
    return driver, session, releases, samplers, tape


def _events(tape, kind):
    return [e for e in map(json.loads, tape.read_text().splitlines()) if e.get("event") == kind]


def _trained(tape):
    """rollout id -> the generating versions of the groups it trained."""
    return {e["rollout_id"]: sorted(int(g[1].split(":")[1]) for g in e["groups"])
            for e in _events(tape, "rl_group_policies")}


def test_a_relaunched_island_trains_its_buffered_groups_and_publishes_above_its_releases(tmp_path):
    """The p5h relaunch: releases v6..v8 (the store names v7, v8), buffered groups of
    v6..v8. The island continues at v9: v7 and v8 are restored from the release index,
    so their groups train; v6 is older than the window and passed over as stale."""
    driver, session, releases, _samplers, tape = _relaunch(tmp_path, age=2)
    driver.run()

    assert session.continued_at == 9
    [continued] = _events(tape, "rl_version_account_continued")
    assert (continued["rollout_id"], continued["session_start"], continued["latest_release_version"]) == (9, 0, 8)
    assert [e["versions"] for e in _events(tape, "rl_published_policies_restored")] == [[7, 8]]
    publishes = [e["policy_version"] for e in _events(tape, "rl_driver_phase") if e["phase"] == "publish"]
    assert publishes == [9, 10, 11, 12]  # never a version the store already names
    assert _trained(tape) == {9: [7, 7], 10: [8, 8], 11: [9, 9]}
    # the buffered groups are named with the hashes the island released them with
    first = _events(tape, "rl_group_policies")[0]
    assert {g[1] for g in first["groups"]} == {f"yeto:7:{_hash(7)}"}
    assert sum(e["stale_groups"] for e in _events(tape, "rl_buffer_stale_skipped")) == 1  # the v6 group
    assert [r.version for r in releases.records()] == [11, 12]
    assert not _events(tape, "rl_policy_release_skipped")  # v9 is newer than LATEST v8: released


def test_a_buffered_group_of_a_version_no_record_names_is_dropped_and_the_rest_train(tmp_path):
    """age 4: the v6 group is inside the window, but the store pruned the v6 release, so
    nothing names its policy. That one group is consumed untrained and counted; the
    island keeps training the others (it used to stop the island at the evidence)."""
    driver, _session, _releases, samplers, tape = _relaunch(tmp_path, age=4)
    driver.run()

    [gate] = _events(tape, "rl_buffer_unidentified_policy_gate")
    assert (gate["start_version"], gate["known_versions"]) == (9, [7, 8])
    skipped = _events(tape, "rl_buffer_unidentified_policy_skipped")
    assert [(e["unidentified_policy_groups"], e["versions"]) for e in skipped] == [(1, [[6, 1]])]
    trained = _trained(tape)
    assert trained == {9: [7, 7], 10: [8, 8], 11: [9, 9]}
    trained_groups = {g[0] for e in _events(tape, "rl_group_policies") for g in e["groups"]}
    assert "g0" not in trained_groups and samplers.versions["g0"] == 6


def test_a_session_start_at_rollout_zero_restores_the_published_table_from_the_history(tmp_path):
    """The session starts at rollout 0; the history names the island's releases
    v0..v2: the table is restored (v1, v2 in the window) and v3 is the first publication."""
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0)
    age = 2
    algorithm = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=age),
                              correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    caps = fake_capabilities("miles-like")
    caps = dataclasses.replace(caps, execution=dataclasses.replace(caps.execution, max_policy_staleness=age))
    profile = ExecutionProfile("p", "colocated-serial", "strict-avg", max_policy_age=age,
                               algorithm_contract="bounded-staleness", groups_per_batch=engine.groups,
                               samples_per_group=engine.samples_per_group,
                               algorithm_spec_sha256=algorithm.sha256())
    shared = []
    engine.rollout.set_published_policies = lambda published: shared.append(published.to_dict())
    session = _RestartingSession(1)
    tape = tmp_path / "events.jsonl"
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, algorithm=algorithm, capabilities=caps,
        sync=session, events=EventTape(tape, 0), profile=profile)
    driver.data_position_owner = "buffer_lane"
    driver.published_policy_history = lambda: [(0, _hash(0)), (1, _hash(1)), (2, _hash(2))]
    driver.run()

    assert [e["versions"] for e in _events(tape, "rl_published_policies_restored")] == [[1, 2]]
    assert shared[0] == {"1": _hash(1), "2": _hash(2)}  # handed to the rollout side before v3
    publishes = [e["policy_version"] for e in _events(tape, "rl_driver_phase") if e["phase"] == "publish"]
    assert publishes == [3, 4]


def test_sessions_without_the_verb_and_first_starts_keep_their_start(tmp_path):
    """A single trainer (``LocalOnlySync.resume_at``) counts its own account; a store
    without releases is a first start: neither is moved."""
    from types import SimpleNamespace

    events = []
    start = SyncStart(state=SimpleNamespace(policy_version=4), rollout_id=4)
    plain = SimpleNamespace(sync=LocalOnlySync(1), published_policy_history=lambda: [(7, _hash(7))],
                            emit=lambda *a, **k: events.append(a))
    assert IslandDriver._continue_above_own_releases(plain, start) is start
    first = SimpleNamespace(sync=_RestartingSession(1), published_policy_history=lambda: [],
                            emit=lambda *a, **k: events.append(a))
    assert IslandDriver._continue_above_own_releases(first, start) is start
    above = SimpleNamespace(sync=_RestartingSession(1), published_policy_history=lambda: [(3, _hash(3))],
                            emit=lambda *a, **k: events.append(a))
    assert IslandDriver._continue_above_own_releases(above, start) is start  # already above v3
    assert events == []


# ------------------------------------------------------------------ the real session
def test_full_parameter_sync_continues_its_account_after_a_peer_recovery():
    """``FullParameterSync``: after a peer recovery the session still starts at
    rollout 0; the driver's verb moves the start above the island's releases. The
    state keeps the recovered cut (only its label moves), the next boundary publishes
    the version after it, and the lead cap counts this process's rounds from 1."""
    from yeto.rl.peer_recovery import RecoveryOutcome

    actor = _PairedRole("actor", [_candidate("actor", 1)], [])
    actor.recovery_outcome = RecoveryOutcome(
        path="peer", donor=1, t_s=0, done_step=4, steps=4, replayed=2, refused_sources=0,
        seconds=1.0, restored_versions=(15, 14), versions=(15, 14))
    session, driver = _full_parameter_session(actor), _Driver()
    start = session.start(driver)
    assert (start.rollout_id, session.joined_running_pool) == (0, True)
    continued = session.continue_version_account_at(driver, 9)
    assert (continued.rollout_id, continued.state.policy_version) == (9, 9)
    assert continued.state.policy_tensor_hash() == start.state.policy_tensor_hash()
    boundary = session.boundary(driver, rollout_id=9, stats=None)
    assert boundary.state.policy_version == 10
    assert session.pairing.local_round == 1  # as the first round of a process that starts at 0
    with pytest.raises(ValueError):
        session.continue_version_account_at(driver, -1)


# ------------------------------------------------------------------ the buffer reader
def test_the_reader_leaves_out_a_group_of_a_version_the_island_never_published(monkeypatch):
    """Before the take is trained (as #767 does for unknown-version tokens): a group
    whose trajectories name a version missing from the driver's published table is
    left out, counted in ``unpublished_version_groups`` and reported as a
    ``trainer.sample_alignment`` warning; the other groups are kept."""
    trips = []
    monkeypatch.setattr(br.trainer_breakers, "trip", lambda name, emit, **fields: trips.append((name, fields)))
    groups = [(10, [sample(0, 0, 7), sample(1, 0, 7)]),
              (11, [sample(2, 1, 6), sample(3, 1, 6)]),
              (12, [sample(4, 2, 8), sample(5, 2, 8)])]
    stats: dict[str, int] = {}
    kept, problems = br.miles_groups(groups, samples_per_group=2, stats=stats, require_known_versions=True,
                                     published_versions=(7, 8, 9))
    assert [position for position, _ in kept] == [10, 12]
    assert len(problems) == 1 and "buffer position 11" in problems[0] and "[6]" in problems[0]
    assert stats == {"unpublished_version_groups": 1}
    assert [name for name, _ in trips] == [br.trainer_breakers.SAMPLE_ALIGNMENT]
    assert trips[0][1]["count"] == 1


def test_a_group_whose_oldest_version_is_published_is_kept():
    """The evidence names a trajectory by its oldest version: a newer version on
    some of its tokens need not be in the table (it is the round's own or newer)."""
    crossing = dict(sample(0, 0, 7))
    crossing["weight_versions"] = [[{"version": "7", "abs_start": 2, "abs_end": 4},
                                    {"version": "8", "abs_start": 4, "abs_end": 5}]]
    kept, problems = br.miles_groups([(3, [crossing, sample(1, 0, 7)])], samples_per_group=2,
                                     require_known_versions=True, published_versions={7})
    assert [position for position, _ in kept] == [3] and problems == []


def test_without_a_published_table_the_groups_are_kept_as_before():
    """Relay and merged-lane trainers name external merged cuts, and a non
    full-parameter trainer writes no evidence: no table is passed, nothing changes."""
    groups = [(11, [sample(2, 1, 6), sample(3, 1, 6)])]
    stats: dict[str, int] = {}
    kept, problems = br.miles_groups(groups, samples_per_group=2, stats=stats, require_known_versions=True)
    assert [position for position, _ in kept] == [11] and problems == [] and stats == {}
