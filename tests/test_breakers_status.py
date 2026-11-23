"""rl_breaker_status snapshots (yeto.breakers.status): items, cadence, trips, failure isolation."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from yeto.breakers import status as st
from yeto.breakers.status import StatusEmitter, StatusThread, band, primary_limit


@dataclass
class FakeEntry:
    name: str
    params: dict
    enabled: bool = True
    action: Any = "stop"
    kind: str = "breaker"
    implicit: bool = False
    source: Any = field(default_factory=lambda: SimpleNamespace(kind="default", location="default"))

    @property
    def spec(self):
        return SimpleNamespace(name=self.name, layer=self.name.split(".")[0], kind=self.kind, implicit=self.implicit)


class FakeResolved:
    def __init__(self, *entries):
        self.rows = list(entries)

    def entries(self, layer=None):
        return [e for e in self.rows if layer is None or e.spec.layer == layer]


def make(gauges=None, resolved=None, **kw):
    sent = []
    resolved = resolved or FakeResolved(
        FakeEntry("sampler.drain", {"timeout_s": 600.0}),
        FakeEntry("sampler.token_alignment", {}, kind="invariant"),
        FakeEntry("harness.setup_retries", {"max_n": 2}, action=None))
    em = StatusEmitter(sent.append, ("sampler", "harness"), resolved=resolved, scope={"sampler_id": "s-0"},
                       gauges=gauges, clock=lambda: 0.0, **kw)
    return em, sent


def test_band_and_primary_limit():
    assert [band(x) for x in (None, 0.2, 0.5, 0.75, 0.95)] == [0, 0, 1, 2, 3]
    assert primary_limit({"warn_ratio": 0.8, "usd": 450}) == 450.0
    assert primary_limit({"enabled_flag": True, "max_n": 3}) == 3.0
    assert primary_limit({}) is None


def test_first_snapshot_carries_settings_sources_and_gauges():
    em, sent = make({"sampler.drain": lambda: (120.0, None, "longest drain")})
    ev = em.poll(0.0)
    assert sent == [ev] and ev["event"] == "rl_breaker_status" and ev["layer"] == "sampler"
    assert ev["scope"] == {"sampler_id": "s-0"}
    items = {i["name"]: i for i in ev["items"]}
    assert set(items) == {"sampler.drain", "harness.setup_retries"}  # invariants have no status
    d = items["sampler.drain"]
    assert (d["value"], d["limit"], d["ratio"], d["detail"]) == (120.0, 600.0, 0.2, "longest drain")
    assert d["params"] == {"timeout_s": 600.0} and d["source"] == {"kind": "default", "location": "default"}
    assert d["enabled"] is True and d["action"] == "stop" and d["layer"] == "sampler"
    assert items["harness.setup_retries"]["layer"] == "harness" and items["harness.setup_retries"]["value"] is None


def test_unchanged_snapshot_waits_for_the_refresh():
    em, sent = make({"sampler.drain": lambda: 10.0}, check_s=15, refresh_s=600)
    em.poll(0.0)
    assert em.poll(5.0) is None  # between two checks: no gauge read
    assert em.poll(30.0) is None and em.poll(599.0) is None
    assert em.poll(601.0) is not None and len(sent) == 2


def test_rising_band_is_sent_at_once_a_falling_one_after_the_gap():
    value = [100.0]
    em, sent = make({"sampler.drain": lambda: value[0]}, check_s=0, min_gap_s=60)
    em.poll(0.0)
    value[0] = 450.0  # 0.75: band 2
    assert em.poll(1.0) is not None
    value[0] = 100.0  # back down: not urgent
    assert em.poll(2.0) is None
    assert em.poll(62.0) is not None and len(sent) == 3


def test_value_change_inside_a_band_is_not_a_change():
    value = [100.0]
    em, sent = make({"sampler.drain": lambda: value[0]}, check_s=0, min_gap_s=60)
    em.poll(0.0)
    value[0] = 200.0  # 0.33, still band 0
    assert em.poll(100.0) is None


def test_setting_change_is_sent_at_once():
    entry = FakeEntry("sampler.drain", {"timeout_s": 600.0})
    em, sent = make(resolved=FakeResolved(entry), check_s=0)
    em.poll(0.0)
    entry.params = {"timeout_s": 900.0}
    assert em.poll(1.0)["items"][0]["params"] == {"timeout_s": 900.0}


def test_noted_trip_goes_out_at_the_next_poll():
    em, sent = make(check_s=15, min_gap_s=60)
    em.poll(0.0)
    em.note({"event": "rl_breaker", "name": "sampler.drain", "state": "trip", "action": "drop", "value": 601.0,
             "limit": 600.0, "time_unix": 3.0})
    ev = em.poll(3.0)  # inside the check interval, before the gap: a trip is a band rise
    assert ev is not None
    last = {i["name"]: i for i in ev["items"]}["sampler.drain"]["last"]
    assert last == {"state": "trip", "action": "drop", "value": 601.0, "limit": 600.0, "time_unix": 3.0}
    em.note({"event": "rl_sampler_round"})  # not a breaker event: ignored
    assert em.poll(4.0) is None


def test_gauge_without_settings_is_sent_bare():
    em, sent = make({"pool.engine_stall": lambda: (30.0, 600.0)}, resolved=FakeResolved(), names=None)
    ev = em.poll(0.0)
    assert [i["name"] for i in ev["items"]] == ["pool.engine_stall"] and ev["items"][0]["ratio"] == 0.05


def test_disabled_entry_has_no_limit():
    em, _ = make(resolved=FakeResolved(FakeEntry("sampler.drain", {"timeout_s": 600.0}, enabled=False)))
    item = em.poll(0.0)["items"][0]
    assert item["enabled"] is False and item["limit"] is None


def test_failing_gauge_and_writer_never_raise(capsys):
    def boom():
        raise RuntimeError("probe down")

    em, _ = make({"sampler.drain": boom})
    assert em.poll(0.0)["items"][0]["value"] is None

    def bad_writer(event):
        raise OSError("disk full")

    em2 = StatusEmitter(bad_writer, "sampler", resolved=FakeResolved(FakeEntry("sampler.drain", {"timeout_s": 1.0})),
                        clock=lambda: 0.0)
    assert em2.poll(0.0) is None and em2.poll(1000.0, force=True) is None
    assert capsys.readouterr().err.count("status snapshot not sent") == 1


def test_thread_sends_the_first_snapshot_on_start():
    sent = []
    em = StatusEmitter(sent.append, "sampler", resolved=FakeResolved(FakeEntry("sampler.drain", {"timeout_s": 1.0})))
    th = StatusThread(em, interval_s=3600).start()
    th.stop()
    assert len(sent) == 1


def test_status_thread_resolves_this_process(monkeypatch):
    seen = {}

    def fake_resolved_for(**kw):
        seen.update(kw)
        return FakeResolved(FakeEntry("trainer.sample_wait", {"timeout_s": 3600.0}))

    monkeypatch.setattr(st, "resolved_for", fake_resolved_for)
    sent = []
    th = st.status_thread(sent.append, "trainer", scope={"island": "0"}, argv=["--rl-sample-wait-s", "3600"], env={})
    assert seen["argv"] == ["--rl-sample-wait-s", "3600"] and not sent  # not started yet
    th.start()
    th.stop()
    assert sent[0]["items"][0]["name"] == "trainer.sample_wait" and sent[0]["scope"] == {"island": "0"}


def test_nothing_resolved_means_no_thread(monkeypatch):
    monkeypatch.setattr(st, "resolved_for", lambda **kw: None)
    assert st.start_status(lambda e: None, "sampler") is None


def test_resolved_for_real_registry_and_resolved_file(tmp_path):
    rb = st.resolved_for(env={})
    assert rb is not None
    path = tmp_path / "r.json"
    path.write_text(__import__("json").dumps(rb.to_json()))
    again = st.resolved_for(env={st.RESOLVED_ENV: str(path)})
    assert again is not None
