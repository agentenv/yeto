"""A drain-mode weight switch is bounded by reload_max_defer_s (k31-s1m sweep: a drain
waited 2684 s on one agent-timeout trial with the engine idle)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from yeto.rl.engine import agentic_round as ar
from yeto.rl.engine import policy_release
from yeto.rl.engine.policy_release import PolicyFollower, PolicyRelease, ReleaseNotLoaded, SchemeMismatch
from yeto.rl.engine.engine_sampler import DEFAULT_RELOAD_MAX_DEFER_S


@pytest.mark.parametrize(("drain", "defer", "want"), [
    (3600.0, 300.0, 300.0), (120.0, 300.0, 120.0), (None, 300.0, 300.0), (3600.0, None, 3600.0),
    (None, None, None), (300.0, 300.0, 300.0)])
def test_the_effective_drain_limit_is_the_smaller_bound(drain, defer, want):
    assert ar.effective_drain_timeout(drain, defer) == want


def test_reload_max_defer_reads_the_default_and_an_explicit_null():
    assert ar.reload_max_defer_s({}) == DEFAULT_RELOAD_MAX_DEFER_S
    assert ar.reload_max_defer_s({"reload_max_defer_s": 300}) == 300.0
    assert ar.reload_max_defer_s({"reload_max_defer_s": None}) is None


def test_a_drain_timeout_above_the_reload_bound_is_warned():
    spec = {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 3600}}
    warning = ar.drain_timeout_warning(spec)
    assert warning is not None and "3600 s" in warning and "300 s" in warning
    unset = ar.drain_timeout_warning({"agentic": {"weight_switch": "drain"}})
    assert unset is not None and "unset" in unset and f"{DEFAULT_RELOAD_MAX_DEFER_S:g} s" in unset
    auto = ar.drain_timeout_warning({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600}})
    assert auto is not None  # auto resolves to drain on an engine without live updates


@pytest.mark.parametrize("spec", [
    {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 300}},
    {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 60}},
    {"reload_max_defer_s": None, "agentic": {"weight_switch": "drain", "drain_timeout_s": 3600}},
    {"reload_max_defer_s": 300, "agentic": {"weight_switch": "live", "drain_timeout_s": 3600}},
    {"reload_max_defer_s": 300}])
def test_no_warning_when_the_drain_is_within_the_bound_or_not_used(spec):
    assert ar.drain_timeout_warning(spec) is None


def test_auto_on_a_live_update_engine_is_not_warned():
    spec = {"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600},
            "engine": {"options": {"live_update": True}}}
    assert ar.spec_live_engine(spec) and ar.drain_timeout_warning(spec) is None
    assert ar.drain_timeout_warning(spec, live_engine=False) is not None  # the engine itself says drain
    spec["agentic"]["weight_switch"] = "drain"  # asked for: warned on any engine
    assert ar.drain_timeout_warning(spec) is not None
    assert not ar.spec_live_engine({"engine": {"options": {"live_update": "yes"}}})


# ---------------------------------------------------------------------- round_factory's drain settings
@pytest.mark.parametrize(("spec", "drain", "want"), [
    ({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600}}, True, (300.0, ar.ABANDON_GRACE_S)),
    ({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 60, "abandon_grace_s": 5}}, True, (60.0, 5.0)),
    ({"agentic": {}}, True, (DEFAULT_RELOAD_MAX_DEFER_S, ar.ABANDON_GRACE_S)),  # old spec: now cut at the default
    ({"reload_max_defer_s": None, "agentic": {}}, True, (None, ar.ABANDON_GRACE_S)),  # both unset: no cut
    ({"reload_max_defer_s": None, "agentic": {"drain_timeout_s": 90}}, True, (90.0, ar.ABANDON_GRACE_S)),
    ({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600}}, False, (3600.0, ar.ABANDON_GRACE_S)),
    ({"reload_max_defer_s": 300, "agentic": {"abandon_grace_s": 0}}, True, (300.0, 0.0))])
def test_the_drain_settings_of_a_spec(spec, drain, want):
    assert ar.drain_settings(spec, drain=drain) == want


@pytest.mark.parametrize("section", [
    {"abandon_grace_s": None}, {"abandon_grace_s": -1}, {"abandon_grace_s": "soon"}, {"abandon_grace_s": True},
    {"abandon_grace_s": float("nan")}, {"abandon_grace_s": float("inf")},
    {"drain_timeout_s": -5}, {"drain_timeout_s": "x"}, {"drain_timeout_s": float("nan")}])
def test_a_bad_drain_setting_is_a_config_error(section):
    from yeto.rl.agentic import HarnessConfigError

    with pytest.raises(HarnessConfigError, match="abandon_grace_s|drain_timeout_s"):
        ar.drain_settings({"agentic": section}, drain=True)


def _factory_world(monkeypatch):
    """round_factory with its heavy parts (tokenizer, records proxy, Harbor) replaced."""
    from yeto.rl.harness import chat_templates, harbor
    from yeto.rl.harness.gateway import proxy, recorder

    monkeypatch.setattr(chat_templates.ChatTemplate, "load", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(proxy, "RecordsProxy", lambda *a, **k: object())
    monkeypatch.setattr(recorder, "ProxyTokenRecorder", lambda *a, **k: object())
    monkeypatch.setattr(harbor, "HarborHarness", lambda *a, **k: object())
    monkeypatch.setattr(harbor.HarborSettings, "from_env", classmethod(lambda cls: harbor.HarborSettings(tasks_dir=Path("tasks"))))


def _built(spec, tmp_path, *, live=False):
    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.engine.buffer_payload import GroupWriter
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.engine.sampler_engine import Provenance, SamplingParams

    engine = FakeEngine()
    engine.start(None, 0)
    engine.updates_during_generation = live
    spec = {"sampling": {}, **spec, "agentic": {"turn_max_tokens": 8, "template": "t", "max_in_flight": 2,
                                                "max_response_len": 16, **spec.get("agentic", {})}}
    make = ar.round_factory(spec, engine, {})
    sampling = SamplingParams(max_tokens=8, n=2)
    rnd = make(engine, GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), "s"),
               prompts=lambda rollout_id: [], sampling=sampling, emit=lambda name, **fields: None,
               provenance=Provenance.of(engine.identity, sampler_id="s", sampling=sampling))
    try:
        return rnd.weight_switch, rnd.drain_timeout_s, rnd.abandon_grace_s
    finally:
        rnd.close()


@pytest.mark.parametrize(("spec", "live", "want"), [
    ({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600}}, False, ("drain", 300.0, ar.ABANDON_GRACE_S)),
    ({"agentic": {"abandon_grace_s": 7}}, False, ("drain", DEFAULT_RELOAD_MAX_DEFER_S, 7.0)),
    ({"reload_max_defer_s": None, "agentic": {}}, False, ("drain", None, ar.ABANDON_GRACE_S)),
    ({"reload_max_defer_s": 300, "agentic": {"drain_timeout_s": 3600}}, True, ("live", 3600.0, ar.ABANDON_GRACE_S))])
def test_round_factory_builds_the_round_with_the_spec_drain_settings(monkeypatch, tmp_path, spec, live, want):
    _factory_world(monkeypatch)
    assert _built(spec, tmp_path, live=live) == want


@pytest.mark.parametrize("spec", [
    {"agentic": {"abandon_grace_s": None}}, {"agentic": {"abandon_grace_s": -1}},
    {"agentic": {"drain_timeout_s": "x"}}, {"reload_max_defer_s": "soon", "agentic": {}},
    {"reload_max_defer_s": 0, "agentic": {}}])
def test_round_factory_refuses_bad_drain_settings(monkeypatch, tmp_path, spec):
    from yeto.rl.agentic import HarnessConfigError

    _factory_world(monkeypatch)
    with pytest.raises(HarnessConfigError, match="abandon_grace_s|drain_timeout_s|reload_max_defer_s"):
        _built(spec, tmp_path)


# ---------------------------------------------------------------------- PolicyFollower.prefetch (real follower)
class Store:
    """A release store in memory: ``publish`` sets LATEST; ``resolve`` is the download
    (``gates[seq]``: it waits for the event; ``fail[seq]``: errors still to raise)."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.newest = None
        self.read = None  # what latest() returns when not the newest (a stale read)
        self.resolved: list[int] = []
        self.gates: dict[int, threading.Event] = {}
        self.fail: dict[int, list[BaseException]] = {}

    def publish(self, seq, **fields):
        self.newest = PolicyRelease(seq=seq, version=seq, policy_hash=f"h{seq}", path=f"r{seq}-v{seq}", **fields)
        return self.newest

    def latest(self):
        return self.read or self.newest

    def resolve(self, release, *, from_store=False):
        self.resolved.append(release.seq)
        gate = self.gates.get(release.seq)
        if gate is not None:
            assert gate.wait(10)
        if self.fail.get(release.seq):
            raise self.fail[release.seq].pop(0)
        directory = self.root / release.path
        directory.mkdir(exist_ok=True)
        return directory


def _world(tmp_path, monkeypatch, *, served=3, wait_s=5.0, **kwargs):
    monkeypatch.setattr(policy_release, "DOWNLOAD_WAIT_S", wait_s)
    store = Store(tmp_path)
    loaded: list[int] = []
    follower = PolicyFollower(store, lambda release, directory: loaded.append(release.seq), version=served,
                              background_download=True, **kwargs)
    follower.release = store.publish(served)
    return store, follower, loaded


def _downloaded(follower):
    assert follower._fetch.done.wait(10)


def test_prefetch_downloads_the_newer_release_and_the_refresh_loads_it(tmp_path, monkeypatch):
    store, follower, loaded = _world(tmp_path, monkeypatch, wait_s=0.0)
    assert follower.prefetch() is True and store.resolved == []  # nothing newer: nothing to fetch
    store.gates[4] = threading.Event()
    store.publish(4)
    assert follower.prefetch() is False and follower.prefetch() is False  # downloading
    assert follower.incomplete == "downloading r4-v4" and follower.pending_seq() == 4
    store.gates[4].set()
    _downloaded(follower)
    assert follower.prefetch() == 4 and store.resolved == [4] and loaded == []  # on disk, nothing loaded
    assert follower.refresh().seq == 4 and loaded == [4] and store.resolved == [4]
    assert follower.prefetch() is True and follower.pending_seq() is None


def test_prefetch_without_background_download_has_nothing_to_wait_for(tmp_path):
    store = Store(tmp_path)
    follower = PolicyFollower(store, lambda release, directory: None)
    store.publish(4)
    assert follower.prefetch() is True and store.resolved == []


def test_a_release_published_during_the_drain_does_not_cancel_the_load(tmp_path, monkeypatch):
    store, follower, loaded = _world(tmp_path, monkeypatch)
    store.publish(4)
    assert follower.prefetch() == 4  # r4 is on disk: the round starts its drain for r4
    store.publish(5)  # published while the trials drain
    assert follower.pending_seq() == 4 and follower.prefetch() == 4  # still the one to load
    release = follower.refresh()  # the drain ended: it gets its load
    assert release is not None and release.seq == 4 and loaded == [4] and store.resolved == [4]
    assert follower.version == 4 and follower.last_seen.seq == 5 and follower.version_lag() == 1
    assert follower.pending_seq() == 5  # r5 follows: fetched, drained for and loaded next
    assert follower.prefetch() == 5 and follower.refresh().seq == 5
    assert loaded == [4, 5] and store.resolved == [4, 5]


def test_releases_faster_than_a_drain_still_load_and_the_lag_stays_bounded(tmp_path, monkeypatch):
    store, follower, loaded = _world(tmp_path, monkeypatch)
    per_drain = 3  # releases published during one drain
    newest = 3
    for _cycle in range(12):
        newest += 1
        store.publish(newest)
        ready = follower.prefetch()
        assert ready is not True and ready is not False  # a release is on disk: the drain starts
        for _ in range(per_drain):  # the trainer keeps releasing while the trials drain
            newest += 1
            store.publish(newest)
            assert follower.prefetch() == ready
        release = follower.refresh()
        assert release is not None and release.seq == ready  # every drain ends in a load
        assert newest - follower.release.seq <= per_drain  # behind by one drain's releases at most
    assert len(loaded) == 12 and loaded == sorted(loaded)
    assert store.resolved == loaded  # no download was thrown away
    assert follower.counters.loaded == 12 and follower.counters.superseded == newest - 3 - 12 - per_drain


def test_a_failed_prefetch_is_still_reported_by_the_refresh(tmp_path, monkeypatch):
    now = [0.0]
    store, follower, loaded = _world(tmp_path, monkeypatch, timer=lambda: now[0], download_retry_s=60.0)
    store.fail[4] = [ConnectionResetError("reset by peer")]
    store.publish(4)
    assert follower.prefetch() is False and follower.prefetch() is False  # failed: not ready, not raised
    assert store.resolved == [4]  # and not downloaded again on every poll
    with pytest.raises(ReleaseNotLoaded) as raised:  # the loop's refresh reports it (the event)
        follower.refresh()
    assert raised.value.reason == "download_failed" and follower.version == 3
    assert follower.refresh() is None and follower.prefetch() is False  # reported once; waits to retry
    now[0] = 61.0
    assert follower.prefetch() is False  # retry window passed: the next poll downloads again
    assert follower.prefetch() == 4 and store.resolved == [4, 4]
    assert follower.refresh().seq == 4 and loaded == [4]


def test_the_sampler_loop_emits_release_not_loaded_for_a_failed_prefetch(tmp_path, monkeypatch):
    from yeto.rl.engine.sampler_loop import SamplerLoop

    store, follower, loaded = _world(tmp_path, monkeypatch)
    store.fail[4] = [ConnectionResetError("reset by peer")]
    store.publish(4)
    assert follower.prefetch() is False
    events = []
    loop = SamplerLoop.__new__(SamplerLoop)
    loop.follower, loop._not_loaded = follower, None
    loop.read_failures, loop._damaged_latest, loop.max_damaged_latest_reads = 0, 0, 3
    loop.emit = lambda name, **fields: events.append((name, fields))
    loop._load_newest()
    loop._load_newest()
    assert [(n, f["seq"], f["reason"]) for n, f in events] == [("rl_sampler_release_not_loaded", 4, "download_failed")]


def test_prefetch_does_not_download_a_release_of_another_scheme(tmp_path, monkeypatch):
    store, follower, loaded = _world(tmp_path, monkeypatch, scheme="MXFP4_EXPERTS_RTN")
    store.publish(4)  # trained weights (bf16): not what this sampler serves
    assert follower.prefetch() is False and store.resolved == []
    with pytest.raises(SchemeMismatch):
        follower.refresh()
    assert store.resolved == [] and loaded == []
    store.publish(5, derived_from={"scheme": "MXFP4_EXPERTS_RTN"})
    assert follower.prefetch() == 5 and store.resolved == [5]


def test_a_stale_latest_does_not_restart_the_download(tmp_path, monkeypatch):
    store, follower, loaded = _world(tmp_path, monkeypatch, wait_s=0.0)
    r4 = store.publish(4)
    store.gates[5] = threading.Event()
    store.publish(5)
    assert follower.prefetch() is False  # r5 downloads
    deadline = time.time() + 10
    while not store.resolved and time.time() < deadline:  # its thread has started
        time.sleep(0.001)
    assert store.resolved == [5]
    store.read = r4  # stale reads of LATEST (an eventually consistent store): r4, also unloaded
    for _ in range(3):
        assert follower.prefetch() is False and follower.pending_seq() == 5
    store.gates[5].set()
    _downloaded(follower)
    assert follower.prefetch() == 5 and store.resolved == [5]  # r4 was never fetched, r5 only once
    assert follower.refresh().seq == 5 and loaded == [5]  # the refresh counts the stale read
    assert follower.counters.stale_reads == 1
    store.read = None
    assert follower.prefetch() is True


def test_a_stale_latest_after_the_download_keeps_the_ready_release(tmp_path, monkeypatch):
    store, follower, loaded = _world(tmp_path, monkeypatch)
    r4 = store.publish(4)
    store.publish(5)
    assert follower.prefetch() == 5
    store.read = r4
    assert follower.prefetch() == 5 and follower.pending_seq() == 5
    assert follower.refresh().seq == 5 and store.resolved == [5] and loaded == [5]
