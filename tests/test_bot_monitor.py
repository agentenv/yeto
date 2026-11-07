"""yeto.bot rules and alert policy: severity routing, transitions, cooldown, re-notify,
ack, mute, quiet hours, merging one poll into one card."""

import math
import time

import pytest

from yeto.bot.config import Policy, Thresholds, parse_clock, parse_hours
from yeto.bot.monitor import Monitor
from yeto.bot.rules import P0, P1, P2, EventTracker, conditions, missed_streaks
from yeto.dashboard.alerts import DEFAULT_THRESHOLDS

from bot_helpers import MERGE_COLS, STEP_COLS, T0, FakeClock, clone, eval_row, relayout, round_row, row, view


def mon(clock=None, policy=None, **th):
    return Monitor(Thresholds(**th), policy or Policy(), clock=clock or FakeClock())


def keys(notices):
    return sorted(n.key for n in notices)


def add_row(v, isl=0, **kw):
    rows = v["sft"]["islands"][isl]["rows"]
    rows.append(row(rows[-1][0] + 1, **kw))
    return v


def noisy(v, isl=0):
    """Give an island's loss some spread (0.69 / 0.71) so the sigma rule has a baseline."""
    for i, r in enumerate(v["sft"]["islands"][isl]["rows"]):
        r[2] = 0.69 if i % 2 else 0.71
    return v


# -- rules: severity routing ----------------------------------------------------------

def sev_of(v, key, **th):
    c = conditions(v, Thresholds(**th)).get(key)
    return None if c is None else c.sev


def test_island_condition_severities():
    v = view(1)
    assert conditions(v, Thresholds()) == {}
    st = clone(v)
    st["sft"]["islands"][0]["status"] = "stale"
    assert sev_of(st, ("0", "stale")) == P0
    assert sev_of(add_row(clone(v), loss=None, gn=None), ("0", "loss_nan")) == P0  # dashboard null = NaN
    assert sev_of(add_row(clone(v), loss=math.nan), ("0", "loss_nan")) == P0
    assert sev_of(add_row(clone(v), loss=None, gn=None, lr=None), ("0", "loss_nan")) is None  # not logged yet
    assert sev_of(add_row(clone(v), hbm=53.9), ("0", "hbm")) == P0  # refs sft_hbm_limit_gib
    assert sev_of(add_row(clone(v), hbm=53.0), ("0", "hbm_warn")) == P1
    assert sev_of(add_row(clone(v), hbm=53.0), ("0", "hbm"), hbm_limit_gib=52.5) == P0  # env override
    assert sev_of(add_row(clone(v), hbm=52.1), ("0", "hbm_warn")) == P1  # > 0.965 x 53.9 = 52.01
    assert sev_of(add_row(clone(v), hbm=52.1), ("0", "hbm_warn"), hbm_warn_frac=0.98) is None
    assert sev_of(add_row(clone(v), hbm=60.0), ("0", "hbm_warn"), hbm_limit_gib=80.0) is None  # follows the limit
    assert sev_of(add_row(clone(v), hbm=78.0), ("0", "hbm_warn"), hbm_limit_gib=80.0) == P1
    assert sev_of(add_row(clone(v), hbm=50.5), ("0", "hbm_warn"), hbm_warn_gib=50.0) == P1  # explicit line
    no_limit = add_row(clone(v), hbm=200.0)
    del no_limit["sft"]["refs"]["sft_hbm_limit_gib"]
    assert conditions(no_limit, Thresholds()) == {}  # no limit known: neither P0 nor the derived warning
    assert sev_of(no_limit, ("0", "hbm_warn"), hbm_warn_gib=100.0) == P1
    assert sev_of(add_row(clone(v), gn=1.5), ("0", "grad_norm")) == P1
    assert sev_of(add_row(clone(v), gn=1.5), ("0", "grad_norm"), grad_norm_max=2.0) is None
    done = clone(v)
    done["sft"]["islands"][0]["status"] = "done"
    assert sev_of(done, ("0", "done")) == P2


def test_rules_resolve_columns_by_name():
    v = stale(add_row(add_row(view(2), gn=2.0, hbm=53.95), isl=1, loss=None, gn=None), isl=0)
    v["sft"]["now"] = v["sft"]["syncer"]["outer"][-1][0] + 4000  # syncer stuck (merges carry the round times)
    v["sft"]["syncer"]["outer"] = []
    want = conditions(v, Thresholds())
    assert {("0", "stale"), ("0", "grad_norm"), ("0", "hbm"), ("1", "loss_nan"), ("syncer", "stuck")} <= set(want)
    shuffled = relayout(v, cols=list(reversed(STEP_COLS)) + ["new_metric"], merge_cols=list(reversed(MERGE_COLS)))
    assert shuffled["sft"]["islands"][0]["rows"][0][-2:] == [1, None]  # step moved to the end, then new_metric
    got = conditions(shuffled, Thresholds())
    assert {k: (c.sev, c.lines) for k, c in got.items()} == {k: (c.sev, c.lines) for k, c in want.items()}
    no_hbm = relayout(v, cols=[c for c in STEP_COLS if c != "hbm_gib"])
    got = conditions(no_hbm, Thresholds())
    assert ("0", "hbm") not in got and ("0", "grad_norm") in got
    bare = relayout(v, cols=["step", "ts"], merge_cols=["ts"])  # almost nothing named: rules go quiet, no crash
    assert set(conditions(bare, Thresholds())) == {("0", "stale")}


def test_loss_spike_mean_plus_three_sigma():
    v = noisy(view(1))
    assert sev_of(add_row(clone(v), loss=0.72), ("0", "loss_spike")) is None  # inside 3 sd (0.01)
    c = conditions(add_row(clone(v), loss=0.75), Thresholds())[("0", "loss_spike")]
    assert c.sev == P1 and "> mean 0.7 + 3 sd (0.01) of the previous 10 steps" in c.lines[0]
    assert sev_of(add_row(view(1), loss=0.9), ("0", "loss_spike")) is None  # flat history: no sd


def test_syncer_stuck_is_median_round_plus_two_grace():
    v = view(outer_steps=8, round_s=600.0)
    last = v["sft"]["syncer"]["outer"][-1][0]
    v["sft"]["now"] = last + 600 + 2 * 600 - 1
    assert sev_of(v, ("syncer", "stuck")) is None
    v["sft"]["now"] = last + 1801
    c = conditions(v, Thresholds())[("syncer", "stuck")]
    assert c.sev == P0 and "limit 30min" in c.lines[0]
    assert sev_of(v, ("syncer", "stuck"), grace_s=900) is None
    assert sev_of(v, ("syncer", "stuck"), stuck_round_s=1000) == P0
    v["sft"]["syncer"]["outer_step"] = 960  # finished
    assert sev_of(v, ("syncer", "stuck")) is None


def test_rss_host_disk_and_missed_rounds():
    assert sev_of(view(rss=1.5e12), ("syncer", "rss")) == P1  # refs sft_syncer_rss_limit_bytes 1.4e12
    assert sev_of(view(rss=1.5e12), ("syncer", "rss"), rss_warn_bytes=1.6e12) is None
    v = view(1)
    v["sft"]["islands"][0]["host"] = {"node-a": [[T0, 80e9, 1.5e12]]}
    assert sev_of(v, ("0", "host", "node-a")) == P1
    v["sft"]["syncer"]["disk"] = [[T0, 4e9, 100e9]]  # hook: fires once the view carries disk samples
    assert sev_of(v, ("syncer", "disk")) == P0
    rounds = [round_row(1), round_row(2, missed=[1]), round_row(3, missed=[1]), round_row(4, missed=[0, 1])]
    assert missed_streaks(rounds) == {"1": 3, "0": 1}
    assert Thresholds().missed_rounds == DEFAULT_THRESHOLDS["missed_consecutive_warn"]
    c = conditions(view(rounds=rounds), Thresholds())
    assert c[("1", "missed")].sev == P1 and ("0", "missed") not in c
    assert "isl1 not merged for 3 rounds" == c[("1", "missed")].title


def test_forwarded_dashboard_alerts_and_oom():
    crit = {"sev": 0, "rule": "recovery_required", "title": "island 1 RECOVERY_REQUIRED", "detail": "x", "island": "1"}
    oom = {"sev": 0, "rule": "recovery_required", "title": "island 2 RECOVERY_REQUIRED",
           "detail": "CUDA out of memory", "island": "2"}
    warn = {"sev": 1, "rule": "kl", "title": "island 1 KL", "detail": "y", "island": "1"}
    c = conditions(view(alerts=[crit, oom, warn]), Thresholds())
    assert c[("dash", "recovery_required", "1")].sev == P0
    assert c[("oom", "2")].sev == P0 and c[("oom", "2")].title.startswith("OOM")
    assert len(c) == 2
    assert conditions(view(alerts=[crit]), Thresholds(forward_dashboard_alerts=False)) == {}


def test_event_tracker_syncer_events_and_eval():
    tr = EventTracker()
    base = [[T0, "rejected", 1, "log", None]]
    assert tr.events(view(events=base, eval_rows=[eval_row(32)]), seed=True) == []
    new = base + [[T0 + 5, "rejected", 1, "log", None], [T0 + 6, "rejected", 2, "log", None],
                  [T0 + 7, "checkpoint", None, "log", 12.5], [T0 + 8, "joined", 0, "log", None],
                  [T0 + 9, "late", 0, "log", None], [T0 + 10, "checkpoint_failed", None, "log", None]]
    evs = {e.key: e for e in tr.events(view(events=new, eval_rows=[eval_row(32)]))}
    assert evs[("syncer", "rejected")].sev == P1 and evs[("syncer", "rejected")].count == 2
    assert evs[("syncer", "rejected")].lines == ["learner 1 x1, learner 2 x1"]
    assert evs[("syncer", "checkpoint")].sev == P2 and "write took 12s" in evs[("syncer", "checkpoint")].lines
    assert evs[("syncer", "joined")].sev == P2 and evs[("syncer", "checkpoint_failed")].sev == P0
    assert ("syncer", "late") not in evs
    assert tr.events(view(events=new, eval_rows=[eval_row(32)])) == []  # nothing new
    better = tr.events(view(events=new, eval_rows=[eval_row(32), eval_row(64, loss=0.69)]))
    assert [(e.key, e.sev) for e in better] == [(("eval", "new"), P2)]
    assert "vs previous point -0.0100" in better[0].lines[0]
    worse = tr.events(view(events=new, eval_rows=[eval_row(32), eval_row(64, loss=0.69), eval_row(96, loss=0.70)]))
    assert [(e.key, e.sev) for e in worse] == [(("eval", "worse"), P1)]
    assert worse[0].title == "eval step 96 worse than step 64"
    assert worse[0].lines[1:] == ["vs base -0.0500", "vs ckpt100 -0.0100"]
    renamed = tr.events(view(events=new, eval_rows=[eval_row(96, loss=0.70), eval_row(128, loss=0.71, ref="w0")]))
    assert renamed[0].key == ("eval", "worse") and renamed[0].lines[1:] == ["vs base -0.0500", "vs w0 -0.0100"]


# -- policy: transitions, re-notify, cooldown ---------------------------------------------

def stale(v, isl=1):
    v = clone(v)
    v["sft"]["islands"][isl]["status"] = "stale"
    return v


def test_first_poll_seeds_then_p0_renotifies_until_acked():
    clock = FakeClock()
    m = mon(clock)
    st = stale(view())
    assert m.evaluate(st) == []  # already stale at startup: seeded, not paged yet
    clock.t += 1799
    assert m.evaluate(st) == []
    clock.t += 2
    n = m.evaluate(st)
    assert [(x.key, x.kind) for x in n] == [(("1", "stale"), "renotify")]
    clock.t += 60
    assert m.evaluate(st) == []
    assert m.ack(None, "alice") == [("1", "stale")]
    clock.t += 1800
    assert m.evaluate(st) == []  # acked: muted for 2 h
    clock.t += 7200
    assert [x.kind for x in m.evaluate(st)] == ["renotify"]  # ack expired, still firing
    clock.t += 60
    assert [(x.key, x.kind) for x in m.evaluate(view())] == [(("1", "stale"), "resolve")]


def test_p0_fire_then_resolve_and_flap_suppression():
    clock = FakeClock()
    m = mon(clock)
    ok = view()
    m.evaluate(ok)
    clock.t += 60
    assert [(x.sev, x.kind) for x in m.evaluate(stale(ok))] == [(P0, "fire")]
    clock.t += 60
    assert [x.kind for x in m.evaluate(ok)] == ["resolve"]
    clock.t += 60
    assert m.evaluate(stale(ok)) == []  # flapped back inside the re-notify window
    clock.t += 1800
    assert [x.kind for x in m.evaluate(stale(ok))] == ["fire"]  # still there after 30 min: paged


def test_p1_cooldown_per_subject():
    clock = FakeClock()
    m = mon(clock)
    v = view(1)
    hot = add_row(clone(v), gn=2.0)
    m.evaluate(v)
    clock.t += 60
    n = m.evaluate(hot)
    assert [(x.key, x.sev) for x in n] == [(("0", "grad_norm"), P1)]
    clock.t += 60
    assert m.evaluate(hot) == []  # P1 never re-notifies while it holds
    clock.t += 60
    assert [x.kind for x in m.evaluate(v)] == ["resolve"]
    clock.t += 60
    assert m.evaluate(hot) == []  # within the 1 h cooldown
    clock.t += 60
    assert m.evaluate(v) == []  # suppressed one is not resolved either
    clock.t += 3600
    assert [x.key for x in m.evaluate(hot)] == [("0", "grad_norm")]


def test_p1_events_accumulate_through_cooldown():
    clock = FakeClock()
    m = mon(clock)
    ev = [[T0 + 1, "rejected", 1, "log", None]]
    m.evaluate(view(events=ev))
    ev += [[T0 + 100, "rejected", 1, "log", None]]
    clock.t += 60
    assert [(x.key, x.count) for x in m.evaluate(view(events=ev))] == [(("syncer", "rejected"), 1)]
    ev += [[T0 + 200, "rejected", 2, "log", None]]
    clock.t += 60
    assert m.evaluate(view(events=ev)) == []
    ev += [[T0 + 300, "rejected", 2, "log", None]]
    clock.t += 60
    assert m.evaluate(view(events=ev)) == []
    clock.t += 3600
    n = m.evaluate(view(events=ev))  # flushed with no new event once cooled down
    assert [(x.key, x.count) for x in n] == [(("syncer", "rejected"), 2)]


def test_p2_events_have_no_cooldown():
    clock = FakeClock()
    m = mon(clock)
    ev = []
    m.evaluate(view(events=ev))
    for i in range(3):
        ev = ev + [[T0 + i, "checkpoint", None, "log", 10.0]]
        clock.t += 60
        assert [(x.sev, x.kind) for x in m.evaluate(view(events=ev))] == [(P2, "event")]


def test_mute_and_quiet_hours():
    clock = FakeClock()
    m = mon(clock)
    v = view(1)
    m.evaluate(v)
    m.mute(3600)
    clock.t += 60
    both = stale(add_row(clone(v), gn=2.0), isl=0)
    assert [x.sev for x in m.evaluate(both)] == [P0]  # P0 pages through a plain mute
    m.unmute()
    m2 = mon(clock)
    m2.evaluate(v)
    m2.mute(3600, include_p0=True)
    clock.t += 60
    assert m2.evaluate(both) == []
    lt = time.localtime(clock.t)
    now_min = lt.tm_hour * 60 + lt.tm_min
    quiet = Policy(quiet_hours=(now_min, (now_min + 120) % 1440 or 1440))
    m3 = mon(clock, quiet)
    m3.evaluate(v)
    clock.t += 60
    assert [x.sev for x in m3.evaluate(both)] == [P0]  # only P0 inside quiet hours
    assert any(not h[5] and h[1] == P1 for h in m3.history)  # the P1 is recorded as suppressed
    assert m3.quiet(clock.t) and not m3.quiet(clock.t + 3 * 3600)


def test_p1_held_back_by_mute_goes_out_after_the_mute():
    clock = FakeClock()
    m = mon(clock)
    v = view(1)
    hot = add_row(clone(v), gn=2.0)
    m.evaluate(v)
    m.mute(3600)
    clock.t += 60
    assert m.evaluate(hot) == []  # appeared while muted
    for _ in range(5):
        clock.t += 300
        assert m.evaluate(hot) == []
    suppressed = [h for h in m.history if h[2] == ("0", "grad_norm")]
    assert [(h[4], h[5]) for h in suppressed] == [("fire", False)]  # recorded once, not every poll
    clock.t += 3600  # mute over, still firing
    [n] = m.evaluate(hot)
    assert (n.key, n.sev, n.kind) == (("0", "grad_norm"), P1, "fire")
    assert "held back until now" in n.lines[-1] and n.lines[-1].startswith("active for 85min")
    clock.t += 60
    assert m.evaluate(hot) == []  # delivered once
    clock.t += 60
    assert [x.kind for x in m.evaluate(v)] == ["resolve"]


def test_p1_held_back_by_quiet_hours_or_cooldown_and_dropped_if_resolved():
    clock = FakeClock(time.mktime((2026, 11, 7, 7, 0, 0, 0, 0, -1)))
    m = mon(clock, Policy(quiet_hours=(23 * 60, 8 * 60)))
    v = view(1)
    hot = add_row(clone(v), gn=2.0)
    m.evaluate(v)
    clock.t += 60
    assert m.evaluate(hot) == []  # 07:01, quiet
    clock.t += 3600  # 08:01
    assert [x.key for x in m.evaluate(hot)] == [("0", "grad_norm")]
    # resolved, then back inside the cooldown and still active once the cooldown ends
    clock.t += 60
    m.evaluate(v)
    clock.t += 60
    assert m.evaluate(hot) == []
    clock.t += 3600
    assert [x.key for x in m.evaluate(hot)] == [("0", "grad_norm")]
    # held back, then cleared before the hold ended: never announced, no resolve either
    m2 = mon(FakeClock(), Policy())
    m2.evaluate(v)
    m2.mute(3600)
    m2.clock.t += 60
    assert m2.evaluate(hot) == []
    m2.clock.t += 60
    assert m2.evaluate(v) == []
    m2.clock.t += 7200
    assert m2.evaluate(v) == [] and m2.unsent == {}


def test_p1_active_at_startup_stays_unannounced():
    clock = FakeClock()
    m = mon(clock)
    hot = add_row(view(1), gn=2.0)
    assert m.evaluate(hot) == []
    clock.t += 7200
    assert m.evaluate(hot) == []


def test_parse_quiet_hours_and_clock():
    assert parse_hours("23-8") == (23 * 60, 8 * 60)
    assert parse_hours("22:30-06:15") == (22 * 60 + 30, 6 * 60 + 15)
    assert parse_hours("") is None
    with pytest.raises(ValueError):
        parse_hours("night")
    assert parse_clock("09:00") == 540 and parse_clock("7") == 420
    m = mon(policy=Policy(quiet_hours=(23 * 60, 8 * 60)))
    day = time.mktime((2026, 11, 7, 12, 0, 0, 0, 0, -1))
    night = time.mktime((2026, 11, 7, 2, 0, 0, 0, 0, -1))
    late = time.mktime((2026, 11, 7, 23, 30, 0, 0, 0, -1))
    assert not m.quiet(day) and m.quiet(night) and m.quiet(late)


def test_dashboard_down_after_n_failures_and_recovery():
    m = mon(dashboard_down_after=3)
    m.evaluate(view())
    assert m.fetch_failed(OSError("refused")) == []
    assert m.fetch_failed(OSError("refused")) == []
    n = m.fetch_failed(OSError("refused"))
    assert [(x.sev, x.title) for x in n] == [(P1, "dashboard unreachable")]
    assert m.fetch_failed(OSError("refused")) == []
    assert [(x.kind, x.title) for x in m.evaluate(view())] == [("resolve", "dashboard reachable again")]


# -- one card per poll ----------------------------------------------------------------------

def test_several_alerts_merge_into_one_card():
    clock = FakeClock()
    m = mon(clock)
    v = view(2, events=[])
    m.evaluate(v)
    clock.t += 60
    bad = stale(add_row(add_row(clone(v), isl=1, gn=2.0), isl=1, gn=2.5), isl=0)
    bad["sft"]["syncer"]["events"] = [[T0 + 50, "checkpoint", None, "log", 9.0]]
    n = m.evaluate(bad)
    assert keys(n) == [("0", "stale"), ("1", "grad_norm"), ("syncer", "checkpoint")]
    msg = m.card(n, link="https://dash.example/")
    assert msg.level == "p0" and msg.mention
    assert msg.title == "P0 isl0 stale (process dead or hung?) (+2 more)"
    text = msg.plain()
    assert text.index("P0 isl0 stale") < text.index("P1 isl1 grad_norm high") < text.index("P2 syncer checkpoint")
    assert msg.buttons == [("Ack 2h", "/ack c1"), ("Details", "/details c1"),
                           ("Open dashboard", "https://dash.example/")]
    assert [x.key for x in m.card_notices("c1")] == [x.key for x in n]
    p2 = m.card([n[2]])
    assert p2.level == "p2" and not p2.mention and p2.buttons == [("Details", "/details c2")]
    clock.t += 60
    res = m.card(m.evaluate(v))
    assert res.level == "ok" and res.title.startswith("resolved: ") and "(+1 more)" in res.title
    assert m.card([]) is None


def test_alerts_message_lists_active_ack_and_history():
    clock = FakeClock()
    m = mon(clock)
    m.evaluate(view())
    clock.t += 60
    m.evaluate(stale(view()))
    m.ack(None, "bob")
    text = m.alerts_message().plain()
    assert "Active (1)" in text and "isl1 stale" in text and "acked by bob" in text
    assert "P0 1 | P1 0 | P2 0" in text
