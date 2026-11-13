"""CPU, no Ray, no Miles: fewer policy releases (``--rl-release-every N`` and
``--rl-release-skip-busy``) on the Miles trainer's ReleasingPublisher, and their
launch checks."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tests.test_rl_sampler_island import _launch_args, _State
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.engine import sample_roles
from yeto.rl.engine.policy_release import BackgroundReleaser, PolicyReleaseError, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore


def _export(version, directory):
    (directory / "model.safetensors").write_text(f"w{version}")


def _publisher(tmp_path, events, published=None, export=_export, **kw):
    inner = SimpleNamespace(publish=lambda state: (published if published is not None else []).append(
        state.policy_version))
    return si.ReleasingPublisher(
        inner, PolicyReleaseStore(CheckpointStore(tmp_path / "store")), export, read_position=lambda: 0,
        emit=lambda e, **f: events.append((e, f)), staging=tmp_path / "staging",
        releaser=BackgroundReleaser(), **kw)


def _released(events):
    return [f["policy_version"] for e, f in events if e == "rl_policy_released"]


def _skipped(events):
    return [(f["policy_version"], f["reason"]) for e, f in events if e == "rl_policy_release_skipped"]


def test_release_every_n_releases_the_n_th_versions_and_the_last_one(tmp_path):
    events, published = [], []
    publisher = _publisher(tmp_path, events, published, release_every=2)
    for version in range(1, 6):
        publisher.publish(_State(version, f"h{version}"))
    assert published == [1, 2, 3, 4, 5]  # the trainer's own engines take every version
    publisher.drain_releases()
    assert _released(events) == [2, 4, 5]  # 5: the last version, released at the drain
    assert _skipped(events) == [(1, "release_every"), (3, "release_every"), (5, "release_every")]
    store = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    assert store.latest().version == 5


def test_the_default_releases_every_version(tmp_path):
    events = []
    publisher = _publisher(tmp_path, events)
    for version in range(1, 4):
        publisher.publish(_State(version, f"h{version}"))
    publisher.drain_releases()
    assert _released(events) == [1, 2, 3] and _skipped(events) == []


def test_skip_busy_never_waits_for_an_upload_and_the_latest_wins(tmp_path, monkeypatch):
    events = []
    gate = threading.Event()
    store_cls = PolicyReleaseStore
    real = store_cls.release_staged

    def slow(self, *args, **kw):
        gate.wait(5)
        return real(self, *args, **kw)

    monkeypatch.setattr(store_cls, "release_staged", slow)
    publisher = _publisher(tmp_path, events, skip_busy=True)
    publisher.publish(_State(1, "h1"))  # its upload hangs on the gate
    publisher.publish(_State(2, "h2"))
    publisher.publish(_State(3, "h3"))
    assert _skipped(events) == [(2, "upload_in_flight"), (3, "upload_in_flight")]
    gate.set()
    publisher.drain_releases()  # v1's upload ends, then the newest skipped version goes out
    assert _released(events) == [1, 3]


def test_a_resumed_base_is_still_not_released_again(tmp_path):
    events = []
    publisher = _publisher(tmp_path, events, release_every=2, resumed_base=4)
    publisher.publish(_State(4, "h4"))
    publisher.publish(_State(5, "h5"))
    publisher.publish(_State(6, "h6"))
    publisher.drain_releases()
    assert _released(events) == [6]
    assert (4, "resumed_base_version") in _skipped(events)


def test_the_publisher_refuses_bad_cadence_settings(tmp_path):
    with pytest.raises(ValueError, match="release_every"):
        _publisher(tmp_path, [], release_every=0)
    with pytest.raises(ValueError, match="skip_busy needs staged"):
        si.ReleasingPublisher(SimpleNamespace(publish=lambda s: None),
                              PolicyReleaseStore(CheckpointStore(tmp_path / "s")), _export,
                              read_position=lambda: 0, emit=lambda *a, **k: None, skip_busy=True)


def _trainer(**kw):
    return _launch_args(rl_sample_role="trainer", rl_max_policy_age=2, **kw)


def test_launch_checks_policy_age_against_the_cadence():
    sample_roles.check_launch(_trainer(rl_release_every=2))
    with pytest.raises(ValueError, match="--rl-max-policy-age >= 3"):
        sample_roles.check_launch(_trainer(rl_release_every=3))
    with pytest.raises(ValueError, match="--rl-max-policy-age >= 3 for --rl-release-skip-busy"):
        sample_roles.check_launch(_trainer(rl_release_skip_busy=True))
    sample_roles.check_launch(_launch_args(rl_sample_role="trainer", rl_max_policy_age=3,
                                           rl_release_skip_busy=True))
    with pytest.raises(ValueError, match="no --rl-release-inline"):
        sample_roles.check_launch(_launch_args(rl_sample_role="trainer", rl_max_policy_age=3,
                                               rl_release_skip_busy=True, rl_release_inline=True))
    with pytest.raises(ValueError, match="--rl-backend miles"):
        sample_roles.check_launch(_trainer(rl_release_every=2, rl_backend="verl"))
    with pytest.raises(ValueError, match="need --rl-sample-role trainer"):
        sample_roles.check_launch(_launch_args(rl_release_every=2))


def test_cadence_flags_reach_the_trainer_island_only():
    args = _launch_args(rl_sample_role="trainer", rl_max_policy_age=3, rl_release_every=2,
                        rl_release_skip_busy=True)
    flags = sample_roles.island_flags(args)
    assert " --rl-release-every 2" in flags and " --rl-release-skip-busy" in flags
    config = sample_roles.resolve(SimpleNamespace(rl_sample_role="trainer", rl_resume_store="/s",
                                                  rl_release_every=2, rl_release_skip_busy=True),
                                  groups_per_round=8)
    assert (config.release_every, config.release_skip_busy) == (2, True)
    sampler = sample_roles.resolve(SimpleNamespace(rl_sample_role="sampler", rl_resume_store="/s",
                                                   rl_release_every=2, rl_release_skip_busy=True),
                                   groups_per_round=8)
    assert (sampler.release_every, sampler.release_skip_busy) == (1, False)


def test_round_trained_summarizes_the_policy_age_of_the_trained_groups():
    from yeto.rl.engine.driver import policy_age_fields

    groups = [SimpleNamespace(policy_versions=v) for v in ([10], [9, 10], [7], None, [10, 8])]
    assert policy_age_fields(groups, 10) == {
        "policy_age_mean": (0 + 1 + 3 + 2) / 4, "policy_age_max": 3,
        "policy_age_groups": {"0": 1, "1": 1, "2": 1, "3": 1}}
    assert policy_age_fields([SimpleNamespace(policy_versions=None)], 4) == {}


def test_the_dashboard_charts_policy_age_and_the_is_clip_share():
    from yeto.dashboard.reducer import METRICS, first_finite

    paths = {key: p for key, _label, p in METRICS}
    record = {"policy_age_mean": 2.5, "policy_age_max": 6,
              "train_metrics": {"sc_is_clipped": 0.01, "sc_importance_weight": 0.99, "sc_tail_ratio": 1.0}}
    assert first_finite(record, paths["policy_age"]) == 2.5
    assert first_finite(record, paths["policy_age_max"]) == 6
    assert first_finite(record, paths["is_clip"]) == 0.01
    assert first_finite(record, paths["is_weight"]) == 0.99


def test_a_failed_upload_stops_the_trainer_at_its_next_publish_whatever_the_cadence(tmp_path, monkeypatch):
    # v2's upload fails (R2 throttling past every retry). v3 is not a release
    # version (release_every=2): the trainer must still learn of the failure at
    # v3's publish instead of training on while the samplers stay at v0.
    events = []

    def fail(self, *args, **kw):
        raise OSError("SlowDown: retries exhausted")

    monkeypatch.setattr(PolicyReleaseStore, "release_staged", fail)
    publisher = _publisher(tmp_path, events, release_every=2)
    publisher.publish(_State(1, "h1"))
    publisher.publish(_State(2, "h2"))  # its upload fails on the worker
    publisher.releaser._thread.join()
    with pytest.raises(PolicyReleaseError, match="SlowDown"):
        publisher.publish(_State(3, "h3"))
    assert (3, "release_every") not in _skipped(events)
