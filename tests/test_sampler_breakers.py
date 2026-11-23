"""The sampler's breaker trips (yeto.rl.engine.sampler_breakers): one rl_breaker
record next to each trip site's own event, the stop reason of the breakers that
stop the sampler, and the infrastructure streak (sampler.infra_streak, P5).
CPU, fakes only."""

from __future__ import annotations

import pytest

from yeto.breakers.events import clear_stop_reason, current_stop_reason, validate_event
from yeto.rl.engine import agentic_round as ar
from yeto.rl.engine import engine_sampler, sampler_checks
from yeto.rl.engine.infra_streak import InfraStreak
from yeto.rl.engine.policy_release import ReleaseNotLoaded
from yeto.rl.engine.sampler_breakers import spec_breakers, stop_reason_of, trip
from yeto.rl.engine.sampler_loop import SamplerLoop, StoreReadOutage

from test_store_dispense import FakeObjects, _open, _rows, _slots, _write


class Tape:
    """The sampler's ``emit(event, **fields)``."""

    def __init__(self):
        self.records: list[tuple[str, dict]] = []

    def __call__(self, event, **fields):
        self.records.append((event, fields))

    def breakers(self, name=None):
        return [f for e, f in self.records if e == "rl_breaker" and (name is None or f["name"] == name)]


@pytest.fixture(autouse=True)
def _no_stop_reason():
    clear_stop_reason()
    yield
    clear_stop_reason()


def test_a_trip_is_one_valid_rl_breaker_record_and_a_stop_names_the_breaker():
    tape = Tape()
    trip(tape, "sampler.store_read_outage", 1801.0, 1800.0, detail="latest", legacy_event="StoreReadOutage")
    (event,) = tape.breakers()
    assert validate_event({"event": "rl_breaker", **event}) == []
    assert (event["layer"], event["kind"], event["action"], event["unit"]) == ("sampler", "breaker", "stop", "s")
    assert event["stop_reason"] == "breaker:sampler.store_read_outage"
    assert current_stop_reason() == "breaker:sampler.store_read_outage"


def test_a_trip_never_raises_and_a_drop_sets_no_stop_reason():
    def broken(event, **fields):
        raise OSError("tape full")

    trip(broken, "sampler.write_defer", 601.0, 600.0)
    trip(None, "sampler.min_samples", None, 2, count=3)
    assert current_stop_reason() is None


def test_the_stop_reason_of_the_errors_that_stop_the_sampler():
    assert stop_reason_of(StoreReadOutage("x")) == "breaker:sampler.store_read_outage"
    assert stop_reason_of(sampler_checks.SamplerLoadFatal("x")) == "breaker:sampler.load_failures"
    damaged = ReleaseNotLoaded("x", reason="latest_unread", fields={"damaged": True})
    assert stop_reason_of(damaged) == "breaker:sampler.store_read_outage"
    assert stop_reason_of(ReleaseNotLoaded("x", reason="latest_unread")) is None
    assert stop_reason_of(RuntimeError("x")) is None


# ------------------------------------------------------------------ trip sites
def test_a_store_outage_past_the_bound_trips_before_the_island_ends():
    tape, now = Tape(), [250.0]
    loop = SamplerLoop(follower=None, buffer=None, generate=lambda rid: None, num_rollouts=None,
                       max_backlog_groups=1, emit=tape, clock=lambda: now[0])
    loop.max_read_outage_s = 100.0
    loop._outage_bound("latest", 200.0, 3, OSError("reset"))  # 50 s: inside the bound
    assert tape.breakers() == []
    with pytest.raises(StoreReadOutage):
        loop._outage_bound("latest", 100.0, 7, OSError("reset"))
    (event,) = tape.breakers("sampler.store_read_outage")
    assert event["value"] == 150.0 and event["limit"] == 100.0 and "latest" in event["detail"]


def test_the_last_load_failure_trips_before_the_sampler_load_fatal():
    tape = Tape()
    loader = object.__new__(sampler_checks.PortReleaseLoader)
    loader.emit, loader.failures, loader.max_consecutive_failures = tape, 1, 3
    loader._count_failure(4, "probe", "nan")
    assert tape.breakers() == []
    with pytest.raises(sampler_checks.SamplerLoadFatal):
        loader._count_failure(4, "probe", "nan")
    (event,) = tape.breakers("sampler.load_failures")
    assert (event["value"], event["limit"], event["stop_reason"]) == (3, 3, "breaker:sampler.load_failures")


def _waiting_round(tape, *, stop=False):
    now = [0.0]
    round_ = object.__new__(engine_sampler.EngineRound)
    round_.reclaimed, round_.write_fenced, round_.write_unknown = (lambda: False), (lambda: True), (lambda: True)
    round_.should_stop, round_.emit = (lambda: stop), tape
    round_.write_retry_s, round_.write_defer_max_s, round_.deferred_writes, round_._unread_since = 5.0, 30.0, 0, None
    round_.monotonic = lambda: now[0]
    round_.sleep = lambda s: now.__setitem__(0, now[0] + s)
    return round_


def test_a_group_given_up_on_the_write_defer_bound_trips_and_the_lease_pause_is_reported_once():
    tape = Tape()
    assert _waiting_round(tape)._write_refusal(None) == engine_sampler.STORE_UNREAD
    assert [f["name"] for f in tape.breakers()] == ["sampler.incarnation_lease", "sampler.write_defer"]
    pause, given_up = tape.breakers()
    assert pause["action"] == "pause" and given_up["value"] == 30.0 and given_up["limit"] == 30.0


def test_a_group_given_up_because_the_loop_stops_is_no_write_defer_trip():
    tape = Tape()
    assert _waiting_round(tape, stop=True)._write_refusal(None) == engine_sampler.STORE_UNREAD
    assert [f["name"] for f in tape.breakers()] == ["sampler.incarnation_lease"]


def _draining_round(tape, *, cut_at, abandoned, detached):
    round_ = object.__new__(ar.AgenticRound)
    round_.emit, round_.clock, round_.recorder = tape, (lambda: 700.0), None
    round_._on_loop = lambda fn, *a: None
    round_._drain_started, round_._drain_quiet_at, round_._drain_since, round_._drain_total = 100.0, 690.0, 100.0, 0.0
    round_._drain_seq, round_._drain_in_flight, round_._detached = 7, 3, set()
    round_._drain_cut_at, round_._drain_abandoned, round_._drain_detached = cut_at, abandoned, detached
    round_.drain_timeout_s = 600.0
    round_.engine = type("E", (), {"served_version": lambda self: 6})()
    return round_


def test_a_drain_that_cut_trials_trips_once_with_the_trials_lost():
    tape = Tape()
    _draining_round(tape, cut_at=700.0, abandoned=2, detached=1)._end_drain()
    (event,) = tape.breakers("sampler.drain")
    assert (event["count"], event["value"], event["limit"], event["action"]) == (3, 600.0, 600.0, "drop")
    assert [e for e, _ in tape.records] == [ar.DRAIN_EVENT, "rl_breaker"]  # next to the drain's own event
    tape = Tape()
    _draining_round(tape, cut_at=None, abandoned=0, detached=0)._end_drain()  # every trial ended in time
    assert tape.breakers() == []


def test_the_round_reports_its_drops_once_per_entry():
    tape = Tape()
    round_ = object.__new__(ar.AgenticRound)
    round_.emit, round_.min_samples, round_.infra_retries = tape, 2, 2
    round_.retries_exhausted, round_._exhausted_reported = 5, 1
    tally = ar._Tally()
    tally.small = 3
    round_._round_breakers(12, tally)
    small, retries = tape.breakers()
    assert (small["name"], small["count"], small["limit"]) == ("sampler.min_samples", 3, 2)
    assert (retries["name"], retries["count"]) == ("sampler.infra_retries", 4)
    round_._round_breakers(13, ar._Tally())  # nothing new: nothing written
    assert len(tape.breakers()) == 2


def test_an_expired_claim_and_an_exhausted_slot_trip_the_dispense_entries(tmp_path):
    store, tape = FakeObjects(), Tape()
    path = _write(tmp_path, _rows(1))
    dead = _open(path, store, holder="dead", groups=1, config={"max_redo": 0})
    assert _slots(dead(0)) == [0]  # dies holding it
    live = _open(path, store, holder="live", groups=1, config={"max_redo": 0}, emit=tape)
    store.now += 120  # past the 90 s lease: expired, a counted redo past max_redo 0
    with pytest.raises(Exception):
        live(0)
    lease, redo = tape.breakers("sampler.dispense_lease"), tape.breakers("sampler.dispense_redo")
    assert len(lease) == 1 and lease[0]["limit"] == 90 and lease[0]["value"] >= 90
    assert len(redo) == 1 and redo[0]["scope"]["slot"] == 0


def test_the_store_dispenser_takes_its_lease_and_redo_from_the_registry(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(2))
    prompts = _open(path, store, limits=(450.0, 4))
    assert (prompts.ledger.lease_s, prompts.ledger.max_redo) == (450.0, 4)
    assert (_open(path, FakeObjects()).ledger.lease_s, _open(path, FakeObjects()).ledger.max_redo) == (90.0, 1)


# ------------------------------------------------------------------ the infrastructure streak (P5)
def test_the_streak_warns_once_per_streak_by_default():
    tape = Tape()
    streak = InfraStreak(max_n=3)
    streak.emit = tape
    for reason in ["below_min_samples", "below_min_samples", "fenced", "below_min_samples", "below_min_samples"]:
        streak.group(reason)
    (event,) = tape.breakers("sampler.infra_streak")
    assert (event["action"], event["value"], event["limit"], event["state"]) == ("warn", 3, 3, "trip")
    assert not streak.paused()  # warn holds nothing
    streak.group(None)  # a written group ends the streak and re-arms it
    for _ in range(3):
        streak.group("below_min_samples")
    assert len(tape.breakers()) == 2 and streak.trips == 2


def test_a_pause_holds_claims_until_a_group_is_written_or_the_cool_down():
    tape, now = Tape(), [0.0]
    streak = InfraStreak(max_n=2, cooldown_s=300, action="pause", clock=lambda: now[0])
    streak.emit = tape
    streak.group("below_min_samples")
    assert not streak.paused()
    streak.group("below_min_samples")
    assert streak.paused()
    now[0] = 299.0
    assert streak.paused()
    now[0] = 300.0
    assert not streak.paused()  # the cool-down: claims again
    trip_, clear = tape.breakers()
    assert (trip_["state"], trip_["action"], clear["state"]) == ("trip", "pause", "clear")
    assert "cool-down" in clear["detail"]
    streak.group("below_min_samples"), streak.group("below_min_samples")  # a new streak: paused again
    assert streak.paused()
    streak.group(None)  # an infra-ok group lands
    assert not streak.paused() and tape.breakers()[-1]["state"] == "clear"
    assert "written" in tape.breakers()[-1]["detail"]


def test_the_streak_follows_its_resolved_entry():
    assert InfraStreak.of(spec_breakers({})).action == "warn"
    paused = InfraStreak.of(spec_breakers({"agentic": {"infra_streak": {"action": "pause", "max_n": 8}}}))
    assert (paused.action, paused.max_n, paused.cooldown_s) == ("pause", 8, 300.0)
    assert InfraStreak.of(spec_breakers({"breakers": {"sampler": {"infra_streak": {"enabled": False}}}})) is None
    with pytest.raises(ValueError):
        spec_breakers({"agentic": {"infra_streak": {"action": "stop"}}})


def test_a_paused_store_dispenser_claims_nothing_and_keeps_its_claims(tmp_path):
    store = FakeObjects()
    path = _write(tmp_path, _rows(4))
    prompts = _open(path, store, groups=2)
    held = _slots(prompts(0))
    paused = [True]
    prompts.paused = lambda: paused[0]
    assert prompts._claim(2) == ([], False)
    assert prompts.still_holds(held[0]) and prompts.holding()
    paused[0] = False
    assert len(prompts._claim(2)[0]) == 2


def _agentic_round(monkeypatch, tmp_path, spec, prompts):
    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.engine.buffer_payload import GroupWriter
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.engine.sampler_engine import Provenance, SamplingParams

    from test_agentic_drain_defer import _factory_world

    _factory_world(monkeypatch)
    engine = FakeEngine()
    engine.start(None, 0)
    spec = {"sampling": {}, "agentic": {"turn_max_tokens": 8, "template": "t", "max_in_flight": 2,
                                        "max_response_len": 16, **spec}}
    sampling = SamplingParams(max_tokens=8, n=2)
    return ar.round_factory(spec, engine, {})(
        engine, GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), "s"), prompts=prompts,
        sampling=sampling, emit=Tape(), provenance=Provenance.of(engine.identity, sampler_id="s", sampling=sampling))


class _Dispenser:
    paused = None

    def __call__(self, rollout_id):
        return []


def test_round_factory_hooks_a_pausing_streak_to_the_store_dispenser(monkeypatch, tmp_path):
    prompts = _Dispenser()
    rnd = _agentic_round(monkeypatch, tmp_path, {"infra_streak": {"action": "pause", "max_n": 3}}, prompts)
    try:
        assert rnd.infra_streak.max_n == 3 and prompts.paused == rnd.infra_streak.paused
        assert rnd.infra_streak.emit is rnd.emit
    finally:
        rnd.close()
    plain = _Dispenser()
    rnd = _agentic_round(monkeypatch, tmp_path, {}, plain)  # default: warn only, the dispenser untouched
    try:
        assert rnd.infra_streak.action == "warn" and plain.paused is None
    finally:
        rnd.close()


def test_round_factory_reads_min_samples_and_infra_retries_from_the_registry(monkeypatch, tmp_path):
    rnd = _agentic_round(monkeypatch, tmp_path, {"min_samples": 1, "infra_retries": 0}, _Dispenser())
    try:
        assert (rnd.min_samples, rnd.infra_retries) == (1, 0)
    finally:
        rnd.close()
    from yeto.rl.agentic import HarnessConfigError

    with pytest.raises(HarnessConfigError, match="infra_retries"):
        _agentic_round(monkeypatch, tmp_path, {"infra_retries": "x"}, _Dispenser())
