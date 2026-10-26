"""S27 L11: a relaunched synced trainer (sample buffer lane) resumes.

* its buffer lane cursor log, not the data source, holds its data position:
  the restart restores no data cursor;
* it learns the versions an earlier publisher process released (the store's
  policy releases), so groups generated with them pass the policy-age check.
"""

from __future__ import annotations

from types import SimpleNamespace

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.engine import sample_roles
from yeto.rl.engine.driver import IslandDriver
from yeto.rl.engine.merged_release import BufferLane
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
