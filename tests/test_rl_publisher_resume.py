"""S27 L11: a relaunched synced trainer (sample buffer lane) resumes.

* its buffer lane cursor log, not the data source, holds its data position:
  the restart restores no data cursor;
* it learns the versions an earlier publisher process released (the store's
  policy releases), so groups generated with them pass the policy-age check.

S27 G-EQ5 (gap 22) adds the same for a buffer trainer that counts its versions
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


# -------------------------------------------- S27 G-EQ5 (gap 22): a trainer without a syncer
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
