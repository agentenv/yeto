"""A release that holds the weights the sampler already serves is adopted without a load
(a fleet's synced lane released its base-checkpoint v0 and a vLLM drain-mode pool
drained its agentic trials for 303 s to reload the same weights)."""

from __future__ import annotations

import threading

from yeto.rl.engine import policy_release
from yeto.rl.engine.policy_release import (
    BASE_CHECKPOINT_KEY,
    BASE_FINGERPRINT_KEY,
    FollowerCounters,
    PolicyFollower,
    PolicyRelease,
    checkpoint_fingerprint,
    is_base_checkpoint_release,
)
from yeto.rl.engine.sampler_loop import SamplerLoop


class Store:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.newest = None
        self.resolved: list[int] = []
        self.served: list[int] = []

    def publish(self, seq, version, policy_hash, **fields):
        self.newest = PolicyRelease(seq=seq, version=version, policy_hash=policy_hash,
                                    path=f"r{seq}-v{version}", **fields)
        return self.newest

    def latest(self):
        return self.newest

    def resolve(self, release, *, from_store=False):
        self.resolved.append(release.seq)
        directory = self.root / release.path
        directory.mkdir(exist_ok=True)
        return directory

    def mark_served(self, release):
        self.served.append(release.seq)


FP = "f" * 64
BASE = {"origin": {"kind": "local", "learner_id": 0, "num_learners": 2, "local_version": 0,
                   BASE_CHECKPOINT_KEY: True, BASE_FINGERPRINT_KEY: FP}}


def _follower(tmp_path, monkeypatch, *, background=True, **kwargs):
    monkeypatch.setattr(policy_release, "DOWNLOAD_WAIT_S", 5.0)
    store = Store(tmp_path)
    loaded: list[int] = []
    kwargs.setdefault("base_fingerprint", FP)
    follower = PolicyFollower(store, lambda release, directory: loaded.append(release.seq), version=0,
                              background_download=background, **kwargs)
    return store, follower, loaded


def test_a_base_checkpoint_v0_is_adopted_by_a_sampler_serving_the_base(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch)
    store.publish(1, 0, "h-base", **BASE)
    # the drain-mode round asks first: nothing pending, so no drain starts
    assert follower.pending_seq() is None
    assert follower.release.seq == 1 and follower.version == 0 and follower.counters.adopted == 1
    assert loaded == [] and store.resolved == [] and store.served == [1]
    assert follower.prefetch() is True and follower.refresh() is None  # nothing more to do
    # the next trained release is loaded as before
    store.publish(2, 2, "h2", origin={"kind": "local"})
    assert follower.pending_seq() == 2
    assert follower.refresh().seq == 2 and loaded == [2] and store.resolved == [2]


def test_the_refresh_path_adopts_too(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False)
    store.publish(1, 0, "h-base", **BASE)
    assert follower.refresh() is None and follower.last_adopted.seq == 1 and loaded == []
    assert store.resolved == []


def test_a_v0_release_without_the_base_marker_is_loaded(tmp_path, monkeypatch):
    """A version alone never makes two sets of weights the same."""
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False)
    store.publish(1, 0, "h-other", origin={"kind": "merged", "learner_id": 1})
    assert follower.pending_seq() == 1
    assert follower.refresh().seq == 1 and loaded == [1] and follower.counters.adopted == 0


def test_a_base_marked_release_is_not_adopted_once_a_release_is_served(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False)
    store.publish(1, 0, "h-trained-v0", origin={"kind": "local"})
    assert follower.refresh().seq == 1
    store.publish(2, 0, "h-base", **BASE)  # same version, other weights than the served release
    assert follower.refresh().seq == 2 and loaded == [1, 2] and follower.counters.adopted == 0


def test_the_same_version_and_hash_as_the_served_release_is_adopted(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False)
    store.publish(3, 4, "h4")
    assert follower.refresh().seq == 3
    store.publish(4, 4, "h4")  # e.g. a relaunched publisher wrote the served weights again
    assert follower.pending_seq() is None and follower.release.seq == 4 and loaded == [3]
    store.publish(5, 4, "h4-other")  # same number, other weights: loaded
    assert follower.refresh().seq == 5 and loaded == [3, 5]
    store.publish(6, 5, "h4-other")  # same hash, other number: loaded (the engines stamp the version)
    assert follower.refresh().seq == 6 and loaded == [3, 5, 6]


def test_a_lora_or_derived_or_delta_v0_is_never_a_base_checkpoint():
    def release(**fields):
        return PolicyRelease(seq=1, version=0, policy_hash="h", path="r1-v0", **fields)
    assert is_base_checkpoint_release(release(**BASE))
    assert not is_base_checkpoint_release(release(weights="lora", **BASE))
    assert not is_base_checkpoint_release(release(derived_from={"seq": 1}, **BASE))
    assert not is_base_checkpoint_release(release(delta={"base_seq": 0}, **BASE))
    assert not is_base_checkpoint_release(PolicyRelease(seq=2, version=2, policy_hash="h", path="r2-v2", **BASE))
    assert not is_base_checkpoint_release(release(origin={"kind": "local"}))
    assert not is_base_checkpoint_release(release(origin={BASE_CHECKPOINT_KEY: "yes", BASE_FINGERPRINT_KEY: FP}))
    assert not is_base_checkpoint_release(release(origin={BASE_CHECKPOINT_KEY: True}))  # no fingerprint


def test_the_counters_keep_their_fields_until_a_release_is_adopted():
    counters = FollowerCounters()
    assert "adopted" not in counters.to_dict()
    assert set(counters.to_dict()) == {"loaded", "superseded", "stale_reads", "lineage_resets",
                                       "deferred_in_flight", "rebuild_failures", "forced_loads"}
    counters.adopted = 1
    assert counters.to_dict()["adopted"] == 1


def test_the_sampler_loop_reports_an_adopted_release_once(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False)
    events: list[tuple[str, dict]] = []
    loop = SamplerLoop.__new__(SamplerLoop)
    loop.follower = follower
    loop.emit = lambda name, **fields: events.append((name, fields))
    loop._adopted_reported = None
    store.publish(1, 0, "h-base", **BASE)
    assert follower.pending_seq() is None  # adopted on the round's thread
    loop._report_adopted()
    loop._report_adopted()
    adopted = [fields for name, fields in events if name == "rl_sampler_policy_adopted"]
    assert adopted == [{"seq": 1, "policy_version": 0, "policy_hash": "h-base", "released_at": 0.0,
                        "reason": "base_checkpoint", "buffer_selected": None}]


def test_a_download_in_flight_for_the_adopted_release_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_release, "DOWNLOAD_WAIT_S", 0.0)
    store = Store(tmp_path)
    gate = threading.Event()
    resolve = store.resolve

    def slow(release, *, from_store=False):
        assert gate.wait(10)
        return resolve(release, from_store=from_store)
    store.resolve = slow
    follower = PolicyFollower(store, lambda release, directory: None, version=4, background_download=True)
    follower.release = store.publish(3, 4, "h4")
    store.publish(4, 5, "h5")
    assert follower.prefetch() is False  # downloading r4
    fetch = follower._fetch
    store.publish(5, 4, "h4")  # the served weights again (a lineage back to v4)
    follower.version = 4
    assert follower._adopt(store.newest) is True and follower._fetch is None
    gate.set()
    assert fetch.done.wait(10)


# ---------------------------------------------------------------------- the trainer marks its base v0
def _publisher(tmp_path, events, **kw):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import sampler_island as si
    from yeto.rl.engine.policy_release import BackgroundReleaser, PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore

    def export(version, directory):
        (directory / "model.safetensors").write_text(f"w{version}")
    return si.ReleasingPublisher(
        SimpleNamespace(publish=lambda state: None), PolicyReleaseStore(CheckpointStore(tmp_path / "store")),
        export, read_position=lambda: 0, emit=lambda e, **f: events.append((e, f)),
        staging=tmp_path / "staging", releaser=BackgroundReleaser(), **kw)


def _records(tmp_path):
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore
    return {r.version: r for r in PolicyReleaseStore(CheckpointStore(tmp_path / "store")).records()}


def test_a_synced_lane_marks_its_first_v0_release_as_the_base_checkpoint(tmp_path):
    from tests.test_rl_sampler_island import _State

    events = []
    lane = {"kind": "local", "learner_id": 0, "num_learners": 2}
    publisher = _publisher(tmp_path, events, origin=lambda state: {**lane, "local_version": int(state.policy_version)},
                           base_fingerprint=FP)
    for version in range(0, 2):  # the store keeps the newest two releases
        publisher.publish(_State(version, f"h{version}"))
    publisher.drain_releases()
    records = _records(tmp_path)
    assert sorted(records) == [0, 1]  # the lane still releases v0 (restarts restore it)
    assert is_base_checkpoint_release(records[0])
    assert records[0].origin == {**lane, "local_version": 0, BASE_CHECKPOINT_KEY: True, BASE_FINGERPRINT_KEY: FP}
    assert BASE_CHECKPOINT_KEY not in (records[1].origin or {})


def test_a_lane_that_loaded_a_release_or_a_v0_after_releases_is_not_marked(tmp_path):
    from tests.test_rl_sampler_island import _State

    events = []
    publisher = _publisher(tmp_path / "a", events, origin=lambda state: {"kind": "local"}, base_fingerprint=None)
    publisher.publish(_State(0, "h0"))
    publisher.drain_releases()
    assert not is_base_checkpoint_release(_records(tmp_path / "a")[0])


def test_a_single_trainer_still_releases_nothing_for_v0(tmp_path):
    """The customer's single-trainer release sequence is unchanged: v0 skipped, origin None."""
    from tests.test_rl_sampler_island import _State

    events = []
    publisher = _publisher(tmp_path, events, skip_base=True)
    for version in range(0, 3):
        publisher.publish(_State(version, f"h{version}"))
    publisher.drain_releases()
    records = _records(tmp_path)
    assert sorted(records) == [1, 2] and all(r.origin is None for r in records.values())
    assert ("rl_policy_release_skipped", "base_policy") in [(e, f.get("reason")) for e, f in events]


def _checkpoint(root, *, sizes=(10, 20), config='{"model_type": "qwen3"}'):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(config)
    for n, size in enumerate(sizes):
        (root / f"model-{n:05d}.safetensors").write_bytes(b"x" * size)
    return root


def test_checkpoint_fingerprints_agree_for_copies_and_differ_otherwise(tmp_path):
    a = checkpoint_fingerprint(_checkpoint(tmp_path / "a"))
    assert a and a == checkpoint_fingerprint(_checkpoint(tmp_path / "b"))
    assert a != checkpoint_fingerprint(_checkpoint(tmp_path / "c", sizes=(10, 21)))
    assert a != checkpoint_fingerprint(_checkpoint(tmp_path / "d", config='{"model_type": "other"}'))
    assert checkpoint_fingerprint(tmp_path / "missing") is None and checkpoint_fingerprint(None) is None
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "config.json").write_text("{}")
    assert checkpoint_fingerprint(tmp_path / "empty") is None  # no weight file


def test_a_base_release_of_another_checkpoint_or_without_a_sampler_fingerprint_is_loaded(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False, base_fingerprint="e" * 64)
    store.publish(1, 0, "h-base", **BASE)  # the trainer's checkpoint is not the sampler's (a misconfigured base)
    assert follower.refresh().seq == 1 and loaded == [1] and follower.counters.adopted == 0
    (tmp_path / "x").mkdir()
    store2, follower2, loaded2 = _follower(tmp_path / "x", monkeypatch, background=False, base_fingerprint=None)
    store2.publish(1, 0, "h-base", **BASE)
    assert follower2.refresh().seq == 1 and loaded2 == [1]


def test_an_empty_hash_or_another_manifest_is_never_the_same_weights(tmp_path, monkeypatch):
    store, follower, loaded = _follower(tmp_path, monkeypatch, background=False)
    store.publish(1, 4, "")
    follower.version = 4
    assert follower.refresh().seq == 1
    store.publish(2, 4, "")  # no hash: nothing proves the weights are the served ones
    assert follower.refresh().seq == 2
    store.publish(3, 4, "h4", manifest={"expert_selection_sha256": "a"})
    assert follower.refresh().seq == 3
    store.publish(4, 4, "h4", manifest={"expert_selection_sha256": "b"})  # same hash, other declared selection
    assert follower.refresh().seq == 4 and loaded == [1, 2, 3, 4] and follower.counters.adopted == 0


def test_a_restarted_sampler_adopts_the_marked_v0_already_in_the_store(tmp_path, monkeypatch):
    """A sampler that (re)starts on its base checkpoint while the store's newest release is the
    marked v0 takes it on its first poll, as one that saw it published."""
    store, follower, loaded = _follower(tmp_path, monkeypatch)
    store.publish(1, 0, "h-base", **BASE)
    assert follower.prefetch() is True and follower.release.seq == 1 and loaded == [] and store.resolved == []


def test_base_checkpoint_fingerprint_marks_only_a_cold_synced_full_weight_lane(tmp_path):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.sampler_island import base_checkpoint_fingerprint

    ckpt = _checkpoint(tmp_path / "hf")
    fp = checkpoint_fingerprint(ckpt)
    lane = object()
    args = SimpleNamespace(hf_checkpoint=str(ckpt), load=None, yeto_rl_init_release=None)
    assert base_checkpoint_fingerprint(args, lora=False, lane=lane) == fp
    assert base_checkpoint_fingerprint(args, lora=False, lane=None) is None  # a single trainer: no v0 release
    assert base_checkpoint_fingerprint(args, lora=True, lane=lane) is None
    warm = SimpleNamespace(hf_checkpoint=str(ckpt), load=str(tmp_path / "megatron"), yeto_rl_init_release=None)
    assert base_checkpoint_fingerprint(warm, lora=False, lane=lane) is None  # Megatron --load warm start
    released = SimpleNamespace(hf_checkpoint=str(ckpt), load=None, yeto_rl_init_release={"seq": 3})
    assert base_checkpoint_fingerprint(released, lora=False, lane=lane) is None
    remote = SimpleNamespace(hf_checkpoint="Qwen/Qwen3-4B", load=None, yeto_rl_init_release=None)
    assert base_checkpoint_fingerprint(remote, lora=False, lane=lane) is None  # no local directory


def test_wire_policy_releases_marks_only_a_synced_full_weight_lane():
    import inspect

    from yeto.rl.adapters.miles import sampler_island as si
    source = inspect.getsource(si.wire_policy_releases)
    assert "skip_base=not lora and lane is None" in source  # the single trainer's v0 skip is unchanged
    assert "base_fingerprint=base_checkpoint_fingerprint(miles_args, lora=lora, lane=lane)" in source


def test_the_dashboard_shows_an_adopted_release_on_its_sampler():
    from yeto.dashboard.async_rl import AsyncState

    asy = AsyncState()
    asy.fold({"event": "rl_policy_released", "policy_version": 0, "seq": 1, "released_at": 100.0},
             "trainer", 100.0, "", None)
    asy.fold({"event": "rl_sampler_policy_adopted", "seq": 1, "policy_version": 0, "policy_hash": "h",
              "released_at": 100.0, "reason": "base_checkpoint", "buffer_selected": 3}, "p5-i0-0", 105.0, "", None)
    loads = [ld for ld in asy.loads if ld["sampler"] == "p5-i0-0"]
    assert loads and loads[0]["adopted"] and loads[0]["pause_s"] == 0.0 and loads[0]["policy_version"] == 0
