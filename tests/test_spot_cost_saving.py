"""rl-spot-cost-saving groups 1-2 on CPU: capability table, reclaim handling, replacement planning."""

from __future__ import annotations

import json
import signal
import time

import pytest

from yeto.cloud import capabilities as caps
from yeto.cloud import preemption as pre
from yeto.cloud import replace as rp

# --- 1.1 / 1.8 capability table -----------------------------------------------------


def test_checked_fields_have_source_and_date():
    cap = caps.capability("modal", "notice_s")
    assert cap.value == 30 and cap.checked
    assert cap.source == "https://modal.com/docs/guide/lifecycle-functions" and cap.checked_on == "2026-10-09"
    aws = caps.capability("aws", "notice_s")
    assert aws.value == 120 and aws.checked and "spot-instance-termination-notices" in aws.source


@pytest.mark.parametrize("cloud", ["nebius", "verda"])
def test_unchecked_clouds_are_empty_and_mean_no_notice(cloud):
    for field in caps.FIELDS:
        cap = caps.capability(cloud, field)
        assert cap.value is None and cap.status == "unchecked" and cap.source is None
    assert caps.notice_seconds(cloud) is None
    assert caps.notice_seconds("someothercloud") is None


def test_table_validation_rejects_estimates_and_missing_sources(tmp_path):
    table = json.loads(caps.TABLE_PATH.read_text())
    table["clouds"]["nebius"]["notice_s"]["value"] = 30
    with pytest.raises(caps.CapabilityTableError, match="unchecked"):
        caps.validate(table)
    table = json.loads(caps.TABLE_PATH.read_text())
    table["clouds"]["modal"]["notice_s"]["source"] = None
    with pytest.raises(caps.CapabilityTableError, match="without source"):
        caps.validate(table)


def test_single_reader_used_by_handlers():
    # 1.8: the handlers read the same table (no second copy of 30 / 120)
    assert pre.MODAL_HANDLER_CAP_S == caps.notice_seconds("modal") - 5
    assert pre.parse_aws_action(200, json.dumps({"action": "terminate"}), now=0) == caps.notice_seconds("aws")


# --- 1.3 handling order -------------------------------------------------------------


def _notice(deadline=30.0, at=100.0):
    return pre.ReclaimNotice("isl-1", "aws", "us-east-1", "spot", "eval", "test", at, deadline)


def test_plan_response_three_spec_scenarios():
    assert pre.plan_response(30, 8, 10).save is True
    p = pre.plan_response(30, 90, 10)
    assert not p.save and p.reason == pre.SKIP_NOT_ENOUGH
    p = pre.plan_response(30, None, 10)
    assert not p.save and p.reason == pre.SKIP_NO_MEASUREMENT


def test_handle_notice_enough_time_saves_then_leaves_then_emits():
    order, events = [], []
    ev = pre.handle_notice(_notice(), save=lambda b: order.append(("save", b)) or {"dropped_trajectories": 2},
                           leave=lambda: order.append("leave") or True,
                           emit=lambda e, **f: order.append(e) or events.append(f),
                           last_save_s=8, margin_s=10, clock=lambda: 100.0)
    assert order == [("save", 20.0), "leave", "spot_reclaim"]
    assert ev["saved"] and ev["outcome"] == "saved" and ev["leave_confirmed"] is True
    assert ev["remaining_s"] == 30.0 and ev["dropped_trajectories"] == 2
    for key in ("island", "cloud", "region", "billing", "notified_at", "source", "remaining_s", "saved",
                "save_s", "dropped_trajectories", "dropped_tokens", "leave_confirmed"):
        assert key in events[0]


def test_handle_notice_not_enough_time_writes_marker_and_skips_save():
    marks, saves = [], []
    ev = pre.handle_notice(_notice(), save=saves.append, leave=lambda: True, emit=lambda e, **f: None,
                           last_save_s=90, write_marker=marks.append, clock=lambda: 100.0)
    assert saves == [] and marks == [pre.SKIP_NOT_ENOUGH] and ev["saved"] is False


def test_handle_notice_elapsed_time_counts_against_deadline():
    ev = pre.handle_notice(_notice(deadline=30, at=100), save=lambda b: None, leave=lambda: True,
                           emit=lambda e, **f: None, last_save_s=8, clock=lambda: 115.0)
    assert ev["remaining_s"] == 15.0 and ev["saved"] is False  # 15 <= 8 + 10


# --- 1.4 Modal exit handler -----------------------------------------------------------


def test_modal_handler_timeout_branch_and_25s_cap():
    events = []
    h = pre.ModalExitHandler("isl", save=lambda b: time.sleep(5), leave=lambda: True,
                             emit=lambda e, **f: events.append(f), last_save_s=0.0, margin_s=24.8)
    t0 = time.time()
    ev = h.handle()
    assert time.time() - t0 < 2  # budget 25 - 24.8 = 0.2 s; the save is abandoned
    assert ev["outcome"] == "save timed out" and ev["saved"] is False
    assert ev["remaining_s"] == 25.0 and h.stop.is_set()
    assert h.handle() is ev and len(events) == 1  # runs once (signal + @modal.exit)


def test_modal_handler_slow_measured_save_skips_save():
    saves = []
    h = pre.ModalExitHandler("isl", save=saves.append, leave=lambda: True, emit=lambda e, **f: None,
                             last_save_s=lambda: 90.0)
    assert h.handle()["outcome"] == pre.SKIP_NOT_ENOUGH and saves == []


def test_modal_handler_real_signal():
    events = []
    h = pre.ModalExitHandler("isl", save=None, leave=lambda: True, emit=lambda e, **f: events.append(f),
                             last_save_s=None)
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        assert h.install() == ["SIGINT", "SIGTERM"]
        signal.raise_signal(signal.SIGTERM)
    finally:
        for s, f in old.items():
            signal.signal(s, f)
    assert h.stop.is_set() and events[0]["source"] == "modal_signal" and events[0]["cloud"] == "modal"


# --- 1.5 AWS metadata poll -----------------------------------------------------------------


def test_aws_poller_404_then_terminate_fires_once():
    replies = [(404, ""), (404, ""),
               (200, json.dumps({"action": "terminate", "time": "2026-10-09T12:02:00Z"}))]
    got = []
    from datetime import datetime, timezone

    now = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    p = pre.AwsMetadataPoller("isl", got.append, region="us-east-1", fetch=lambda: replies.pop(0),
                              clock=lambda: now)
    sleeps = []
    p.run(sleep=sleeps.append)
    assert len(got) == 1 and got[0].deadline_s == 120.0 and got[0].source == "aws_metadata"
    assert sleeps == [5.0, 5.0, 5.0] and p.fired
    assert p.poll_once() is None


def test_aws_poller_survives_fetch_errors():
    def boom():
        raise OSError("no route")

    p = pre.AwsMetadataPoller("isl", lambda n: None, fetch=boom)
    assert p.poll_once() is None and p.errors == 1


# --- 1.6 summary ----------------------------------------------------------------------------


def test_summary_per_cloud_region():
    evs = [{"event": "spot_reclaim", "cloud": "aws", "region": "r1", "notified_at": t, "saved": s,
            "dropped_trajectories": d, "dropped_tokens": d * 100}
           for t, s, d in ((100, True, 1), (400, False, 2), (1000, True, 0))]
    evs.append({"event": "spot_reclaim", "cloud": "modal", "region": None, "notified_at": 5})
    evs.append({"event": "other", "cloud": "aws", "region": "r1"})
    s = pre.summarize(evs)
    assert s[("aws", "r1")] == {"count": 3, "intervals_s": [300, 600], "saved": 2,
                                "dropped_trajectories": 3, "dropped_tokens": 300}
    assert s[("modal", None)]["count"] == 1


# --- 1.7 training-island admission ----------------------------------------------------------


def test_training_island_spot_admission():
    ok, why = pre.training_spot_admission("modal", measured_save_s=60)
    assert not ok and "not enough to save a checkpoint" in why
    assert not pre.training_spot_admission("nebius", measured_save_s=1)[0]
    assert not pre.training_spot_admission("aws", measured_save_s=None)[0]
    assert pre.training_spot_admission("aws", measured_save_s=30)[0]


# --- 2.1 / 2.2 candidates and scoring ----------------------------------------------------------


def _offer(cloud="aws", region="r2", gpu="H200", price=10.0, **kw):
    return rp.Offer(cloud, region, gpu, 8, price, **{"has_weights": True, **kw})


def test_candidates_origin_out_of_stock_and_compat_group():
    offers = [_offer(region="r1", in_stock=False), _offer(region="r2", in_stock=True),
              _offer(cloud="nebius", region="eu", gpu="H100", price=1.0),
              _offer(cloud="modal", region="any", has_weights=False)]
    kept, rejected = rp.candidates(offers, compat_gpu="H200", origin=("aws", "r1"))
    assert [o.region for o in kept] == ["r2"]
    reasons = {(o.cloud, o.region): why for o, why in rejected}
    assert reasons[("aws", "r1")] == "origin region out of stock"
    assert reasons[("nebius", "eu")] == "compatibility group differs"
    assert "copy cost unknown" in reasons[("modal", "any")]


def test_score_terms_weight_copy_and_unknown_rate():
    a = rp.score(_offer(), prep_s=360, prep_source="pool_grow", expected_stay_s=3600)
    assert a.terms["price"]["value"] == 10.0
    assert a.terms["prep"]["value"] == pytest.approx(1.0)  # 10 $/h * 0.1 h / 1 h
    assert a.terms["reclaim"]["value"] is None and "reclaim rate unknown" in a.risks
    assert a.score == pytest.approx(11.0)
    b = rp.score(_offer(has_weights=False, weight_copy_s=360), prep_s=360, prep_source="x",
                 expected_stay_s=3600)
    assert b.terms["prep"]["prep_s"] == 720 and b.score > a.score
    summ = {("aws", "r2"): {"count": 3, "intervals_s": [3600, 3600]}}
    c = rp.score(_offer(), prep_s=360, prep_source="x", expected_stay_s=3600, reclaim_summary=summ,
                 loss_per_reclaim_usd=2.0)
    assert c.terms["reclaim"]["value"] == pytest.approx(2.0) and c.score == pytest.approx(13.0)


# --- 2.3 / 2.4 planner switch and budget ----------------------------------------------------------


def test_planner_switch_off_never_launches():
    events = []
    launched = []
    p = rp.ReplacementPlanner(offers=lambda n: [_offer()], compat_gpu="H200",
                              emit=lambda e, **f: events.append(e), prep_s=300,
                              launch=lambda n, o: launched.append(n))
    out = p.plan("isl")
    assert out["action"] == "advise" and launched == [] and events == [rp.ADVICE_EVENT]


def test_planner_auto_requires_budget_and_caps():
    with pytest.raises(ValueError):
        rp.ReplacementPlanner(offers=lambda n: [], compat_gpu="H200", emit=lambda e, **f: None,
                              prep_s=1, auto_launch=True)
    events, launched = [], []
    p = rp.ReplacementPlanner(offers=lambda n: [_offer()], compat_gpu="H200",
                              emit=lambda e, **f: events.append(e), prep_s=1, auto_launch=True,
                              budget_usd=15.0, spent_usd=lambda: 10.0,
                              launch=lambda n, o: launched.append(n) or 42)
    assert p.plan("isl")["action"] == "budget_cap" and launched == []
    assert events == [rp.ADVICE_EVENT, rp.BUDGET_EVENT]
    p.spent_usd = lambda: 0.0
    out = p.plan("isl")
    assert out["action"] == "launch" and out["result"] == 42 and launched == ["isl"]


def test_fleet_controller_consults_planner_in_advise_mode():
    from yeto.launcher import FleetController

    calls = []

    class Planner:
        def plan(self, name):
            calls.append(name)
            return {"action": "advise", "candidates": []}

    fc = FleetController.__new__(FleetController)
    fc.replacement_planner = Planner()
    fc.fleet_log = None
    assert fc._plan_replacement({"name": "isl-1"}) is None and calls == ["isl-1"]

    class Launch(Planner):
        def plan(self, name):
            return {"action": "launch", "result": 7}

    fc.replacement_planner = Launch()
    att = fc._plan_replacement({"name": "isl-1"})
    assert att.finished and att.result == 7


# --- 2.5 dashboard ---------------------------------------------------------------------------------


def test_dashboard_offline_page_shows_reclaim_rows(tmp_path):
    from yeto.dashboard.export import export_html
    from yeto.dashboard.reducer import Reducer

    r = Reducer(prices={})
    r.feed({"event": "spot_reclaim", "time_unix": 10.0, "island_id": 1, "island": "isl-1", "cloud": "aws",
            "region": "r1", "source": "aws_metadata", "remaining_s": 110.0, "saved": True, "outcome": "saved"})
    r.feed({"event": "spot_replace_advice", "time_unix": 11.0, "island_id": 1, "island": "isl-1",
            "auto_launch": False, "candidates": [{"cloud": "aws", "region": "r2"}], "rejected": []})
    view = r.page_view()
    assert [x["event"] for x in view["spot"]] == ["spot_reclaim", "spot_replace_advice"]
    assert view["spot"][1]["region"] == "r2" and view["spot"][1]["candidates"] == 1
    html = export_html(r, tmp_path / "p.html").read_text()
    assert "spotBox" in html and "spot_replace_advice" in html
