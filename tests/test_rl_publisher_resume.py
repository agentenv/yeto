"""A relaunched synced trainer (sample buffer lane) resumes.

* its buffer lane cursor log, not the data source, holds its data position:
  the restart restores no data cursor;
* it learns the versions an earlier publisher process released (the store's
  policy releases), so groups generated with them pass the policy-age check.

The same holds for a buffer trainer that counts its versions
itself (no syncer): :class:`~yeto.rl.engine.merged_release.ResumedReleases`
continues the version account above the store's releases, and
:meth:`~yeto.rl.engine.bridges.LocalOnlySync.resume_at` starts its rounds there.
The verl island end to end is in test_rl_verl_buffer_trainer.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.engine import sample_roles
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import IslandDriver
from yeto.rl.engine.merged_release import BufferLane, ResumedReleases, resumed_releases
from yeto.rl.engine.policy_age import check_batch_ages
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.policy_versions import PublishedPolicies
from yeto.rl.engine.resume import CheckpointStore


def _release(store, version, policy_hash):
    store.release(version, policy_hash, lambda d: (d / "w.bin").write_text(policy_hash))


def _driver(*, limit, history=None, owner=None, rollout=None):
    events = []
    fake = SimpleNamespace(
        published_policies=PublishedPolicies(), published_policy_history=history,
        data_position_owner=owner, rollout=rollout or SimpleNamespace(),
        emit=lambda event, **fields: events.append((event, fields)),
        _max_policy_age=lambda: limit)
    return fake, events


def test_records_lists_the_kept_releases_in_seq_order(tmp_path):
    store = PolicyReleaseStore(CheckpointStore(tmp_path))
    assert store.records() == []
    for version in range(4):
        _release(store, version, f"h{version}")
    assert [(r.seq, r.version, r.policy_hash) for r in store.records()] == [(3, 2, "h2"), (4, 3, "h3")]


def test_restart_restores_the_published_versions_in_the_window():
    shared = []
    rollout = SimpleNamespace(set_published_policies=lambda p: shared.append(p.to_dict()))
    history = lambda: [(1, "h1"), (2, "h2"), (3, "h3"), (4, "old")]  # noqa: E731
    fake, events = _driver(limit=2, history=history, rollout=rollout)
    IslandDriver._restore_published_policies(fake, 4)
    # v1 is outside the window; v4 is the start version, published again
    assert fake.published_policies.hashes() == {2: "h2", 3: "h3"}
    assert shared == [{"2": "h2", "3": "h3"}]
    assert events == [("rl_published_policies_restored", {"rollout_id": 4, "versions": [2, 3]})]
    group = SimpleNamespace(group_id="g128", policy_token="yeto:2:h2", policy_versions=None)
    assert check_batch_ages([group], 4, 2, {**fake.published_policies.hashes(), 4: "h4"}) == []


def test_no_history_or_first_rollout_restores_nothing():
    fake, events = _driver(limit=2)
    IslandDriver._restore_published_policies(fake, 3)
    fake2, events2 = _driver(limit=2, history=lambda: [(0, "h0")])
    IslandDriver._restore_published_policies(fake2, 0)
    assert fake.published_policies.hashes() == fake2.published_policies.hashes() == {}
    assert events == events2 == []


def test_a_buffer_lane_restart_restores_no_data_cursor():
    seeks = []
    rollout = SimpleNamespace(seek_data_cursor=lambda c: seeks.append(c))
    fake, events = _driver(limit=2, owner="buffer_lane", rollout=rollout)
    IslandDriver._restore_data_cursor(fake, 2)
    assert seeks == []
    assert events == [("rl_data_cursor_restored", {"rollout_id": 2, "source": "buffer_lane"})]


def test_wiring_a_synced_trainer_sets_the_lane_owner_and_the_release_history(tmp_path):
    store = PolicyReleaseStore(CheckpointStore(tmp_path))
    _release(store, 1, "h1")
    _release(store, 2, "h2")
    events = []
    for index in (0, 1):  # the publisher and the other synced trainer
        driver = SimpleNamespace(publisher=SimpleNamespace(publish=lambda s: None),
                                 emit=lambda event, **fields: events.append(event),
                                 rollout=SimpleNamespace(_in_executor=lambda fn, *a: None))
        args = SimpleNamespace(lora_rank=0, yeto_rl_sample_lane=BufferLane(index, 2),
                               yeto_rl_sample_buffer=sample_roles.SampleBufferConfig(
                                   role="trainer", store=str(tmp_path)))
        si.wire_policy_releases(driver, miles_args=args, actor_model=SimpleNamespace(),
                                runner=SimpleNamespace(run=lambda value: value))
        br._ACTIVE.pop(br.TRAINER, None)
        assert driver.data_position_owner == "buffer_lane"
        assert driver.published_policy_history() == [(1, "h1"), (2, "h2")]


# -------------------------------------------- a trainer without a syncer
def test_the_account_continues_above_every_release_the_store_holds(tmp_path):
    store = PolicyReleaseStore(CheckpointStore(tmp_path), keep=4)
    assert resumed_releases(store, 2) is None  # a run that starts here
    for version in (1, 2, 3):
        _release(store, version, f"h{version}")
    _release(store, 1, "h1b")  # the bug this fixes: a relaunch released v1 again
    resumed = resumed_releases(store, 2)
    # LATEST names v1, but the account must continue above v3: K = 3, not 1
    assert store.latest().version == 1
    assert (resumed.base_version, resumed.start_version) == (3, 3)
    assert resumed.known_versions() == (1, 2, 3)
    assert resumed.window() == {1: "h1b", 2: "h2"}  # [K - limit, K); the newest record of v1
    assert resumed.history() == [(1, "h1b"), (2, "h2"), (3, "h3")]


def test_derived_releases_are_not_policy_versions_of_this_trainer():
    records = [SimpleNamespace(seq=1, version=1, policy_hash="h1", derived_from=None),
               SimpleNamespace(seq=2, version=2, policy_hash="w2",
                               derived_from={"scheme": "w4a8", "version": 2}),
               SimpleNamespace(seq=3, version=2, policy_hash="h2", derived_from=None)]
    resumed = resumed_releases(SimpleNamespace(records=lambda: records), 4)
    assert resumed.base_version == 2 and resumed.window() == {1: "h1"}
    assert resumed.history() == [(1, "h1"), (2, "h2")]


def test_a_window_wider_than_the_account_stops_at_zero():
    resumed = ResumedReleases(2, 5, ((0, "h0"), (1, "h1"), (2, "h2")))
    assert resumed.window() == {0: "h0", 1: "h1"} and resumed.known_versions() == (0, 1, 2)


def test_local_only_sync_runs_its_rounds_from_the_resumed_version():
    driver = SimpleNamespace(
        export_local_resident=lambda *, policy_version: SimpleNamespace(policy_version=policy_version))
    sync = LocalOnlySync(2)
    sync.resume_at(3)
    start = sync.start(driver)
    assert (start.rollout_id, start.state.policy_version, start.finished) == (3, 3, False)
    # the round budget counts the rounds of this process: 3 and 4, the last one final
    assert sync._end() == 5
    assert not sync.is_final_round(driver, rollout_id=3)
    assert sync.is_final_round(driver, rollout_id=4)
    fresh = LocalOnlySync(2)
    assert (fresh.start(driver).rollout_id, fresh._end()) == (0, 2)


def test_a_resumed_account_and_a_round_cut_resume_are_refused_together():
    sync = LocalOnlySync(2)
    sync.round_cuts = SimpleNamespace(resume=lambda driver: None)
    with pytest.raises(ValueError, match="round-cut resume"):
        sync.resume_at(3)
    with pytest.raises(ValueError, match="below 0"):
        LocalOnlySync(2).resume_at(-1)


def test_a_relaunch_without_a_cut_reads_the_buffer_from_where_the_newest_release_is_trained(tmp_path):
    """The record of vK holds the buffer position its weights are trained up to
    (``buffer_trained``); a relaunched single trainer hands it to its round cuts as
    the place to read from when the store holds no cut (RoundCutCheckpoint.resume),
    and its sync knows the account continued without one. ``buffer_selected`` (the
    read position when the record was written) is never that place."""
    store = PolicyReleaseStore(CheckpointStore(tmp_path), keep=4)
    for version, trained in ((1, 4), (2, 9), (3, None)):
        store.release(version, f"h{version}", lambda d: (d / "w.bin").write_text("w"),
                      buffer_selected=30, buffer_trained=trained)
    assert resumed_releases(store, 2).buffer_trained is None  # v3 recorded none
    store.release(4, "h4", lambda d: (d / "w.bin").write_text("w"), buffer_selected=30, buffer_trained=17)
    assert resumed_releases(store, 2).buffer_trained == 17

    calls = []
    cuts = SimpleNamespace(resume=lambda driver: None, buffer_start_without_cut=0)
    sync = LocalOnlySync(2)
    sync.round_cuts = cuts
    driver = SimpleNamespace(sync=sync, _max_policy_age=lambda: 2, emit=lambda event, **f: calls.append((event, f)),
                             export_local_resident=lambda *, policy_version: SimpleNamespace(
                                 policy_version=policy_version))
    assert si._resume_single_trainer(driver, store) == {"resumed_base": 4}
    assert cuts.buffer_start_without_cut == 17 and calls[-1][1]["buffer_trained"] == 17
    assert sync.start(driver).rollout_id == 4 and sync.continues_releases_without_cut is True
    fresh = LocalOnlySync(2)
    fresh.start(driver)
    assert fresh.continues_releases_without_cut is False  # a run that starts here


def test_a_reader_older_than_buffer_trained_reads_a_record_that_has_it(tmp_path):
    """The record gained a field; a reader built before it (the samplers of a
    running pool) keeps only the fields it knows, with the real ``from_record``."""
    import dataclasses
    import json

    from yeto.rl.engine.policy_release import LATEST, RELEASE_DIR, PolicyRelease

    store = PolicyReleaseStore(CheckpointStore(tmp_path))
    store.release(3, "h3", lambda d: (d / "w.bin").write_text("w"), buffer_selected=8, buffer_trained=6)
    record = json.loads((tmp_path / RELEASE_DIR / LATEST).read_text())
    assert record["buffer_trained"] == 6
    earlier = dataclasses.make_dataclass(
        "PolicyRelease", [(f.name, f.type, f) for f in dataclasses.fields(PolicyRelease)
                          if f.name != "buffer_trained"], frozen=True)
    earlier.from_record = classmethod(PolicyRelease.from_record.__func__)
    read = earlier.from_record(record)
    assert not hasattr(read, "buffer_trained")
    assert (read.seq, read.version, read.policy_hash, read.buffer_selected) == (1, 3, "h3", 8)


def test_the_reader_reports_where_a_version_is_trained_up_to(tmp_path):
    """``BatchReader.trained_before(v)``: the end of the last round below v, not the
    read position (a later round's selection, a carried-over group, do not move it);
    the start position while this reader took no round."""
    from tests.buffer_rollout_procs import sample
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader, GroupWriter
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    store = CheckpointStore(tmp_path)
    writer = GroupWriter(FileSampleBuffer(store, batch_size=1), "s0")
    for g in range(6):
        writer.write([sample(2 * g, g, 0), sample(2 * g + 1, g, 0)], group_key=f"g{g}")
    reader = BatchReader(FileSampleBuffer(store, batch_size=1, create_index=False), max_policy_age=2,
                         wait_s=0.0, poll_s=0.0, read_workers=1, prefetch=False, incarnation_gate=None)
    reader.reset(1)
    assert reader.trained_before(0) == 1 and reader.trained_before(5) == 1
    reader.take(0, 2, current_version=0)
    assert reader.trained_before(0) == 1 and reader.trained_before(1) == 3
    reader.take(1, 2, current_version=1)  # round 1 is selected: v1 is still trained up to 3
    assert (reader.buffer.read_cursor, reader.trained_before(1), reader.trained_before(2)) == (5, 3, 5)
    assert reader.watermark_before(1) == 3 and reader.trained_before(1) == 3  # a cut before round 1


def test_the_reader_passes_over_groups_without_an_identity_and_warns_when_it_keeps_none(tmp_path, caplog):
    """The gate of a relaunched trainer (start version 5, the store still names v4):
    groups of v2 and v3 inside the window are consumed untrained and counted, v4 and
    v5 are kept, and a version above the start version is not the gate's business.
    Selections in a row that keep nothing raise the starvation warning."""
    from tests.buffer_rollout_procs import sample
    from yeto.rl.adapters.miles import buffer_rollout as br2
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    store = CheckpointStore(tmp_path)
    writer = br2.GroupWriter(FileSampleBuffer(store, batch_size=1), "s0")
    for g, version in enumerate((2, 3, 3, 2, 3, 2, 4, 5, 9)):
        writer.write([sample(2 * g, g, version), sample(2 * g + 1, g, version)], group_key=f"g{g}")
    reader = br2.BatchReader(FileSampleBuffer(store, batch_size=1, create_index=False), max_policy_age=8,
                             wait_s=0.0, poll_s=0.0, read_workers=1, prefetch=False, incarnation_gate=None)
    events = []
    reader.emit = lambda event, **fields: events.append((event, fields))
    reader.set_unidentified_gate(5, [4])
    with caplog.at_level("WARNING"):
        taken, report = reader.take(5, 1, current_version=5)
    assert [group[0]["group_index"] for _, group in taken] == [6]  # the group of v4
    assert report.unidentified_policy_groups == 6 and report.unidentified_policy_versions == {2: 3, 3: 3}
    assert report.metrics()["sample_buffer/unidentified_policy_groups"] == 6.0
    # six selections of one group kept nothing: the warning at the third and the sixth
    assert [(e, f["selections"], f["unidentified_policy_groups_total"]) for e, f in events] == [
        (br2.UNIDENTIFIED_STARVED_EVENT, 3, 3), (br2.UNIDENTIFIED_STARVED_EVENT, 6, 6)]
    assert events[0][1]["start_version"] == 5 and events[0][1]["known_versions"] == [4]
    assert sum("kept no group" in r.message for r in caplog.records) == 2
    taken, report = reader.take(6, 2, current_version=5)
    assert [group[0]["group_index"] for _, group in taken] == [7, 8]  # v5, and v9: not below the start
    assert report.unidentified_policy_groups == 0 and len(events) == 2
    assert br2._starved_selects_limit({br2.UNIDENTIFIED_STARVED_SELECTS_ENV: "7"}) == 7
    assert br2._starved_selects_limit({br2.UNIDENTIFIED_STARVED_SELECTS_ENV: "0"}) == 0


def test_the_miles_pool_hands_the_gate_to_the_reader_in_the_executor():
    """The driver's gate reaches the reader through the rollout executor, also when
    the reader does not exist yet (the first rollout call builds it)."""
    from yeto.rl.adapters.miles import buffer_rollout as br2
    from yeto.rl.adapters.miles.rollout import MilesRolloutPool

    pool = SimpleNamespace(_in_executor=lambda fn, *args: fn(None, *args))
    br2._ACTIVE.pop(br2.TRAINER, None)
    MilesRolloutPool.set_unidentified_policy_gate(pool, 7, {6})
    assert br2._PENDING_UNIDENTIFIED_GATE[br2.TRAINER] == (7, [6])
    gates = []
    reader = SimpleNamespace(set_unidentified_gate=lambda start, known: gates.append((start, list(known))),
                             gate_version=None)
    br2.register_reader(reader)
    MilesRolloutPool.set_unidentified_policy_gate(pool, 9, {7, 8})
    br2._ACTIVE.pop(br2.TRAINER, None)
    assert gates == [(7, [6]), (9, [7, 8])] and br2.TRAINER not in br2._PENDING_UNIDENTIFIED_GATE
    report = SimpleNamespace(unidentified_policy_groups=3, unidentified_policy_versions={5: 2, 4: 1},
                             unidentified_policy_positions=[11, 12, 40])
    assert br2.unidentified_metadata(report) == {
        "buffer_unidentified_groups": 3, "buffer_unidentified_versions": [[4, 1], [5, 2]],
        "buffer_unidentified_positions": [[11, 12, 1], [40, 40, 1]]}
    assert br2.unidentified_metadata(SimpleNamespace(unidentified_policy_groups=0)) == {}
