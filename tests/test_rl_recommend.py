from types import SimpleNamespace as NS

import pytest

from yeto.rl.engine.controller import Rejected
from yeto.rl.engine.recommend import (
    BALANCED, GPU_SAT, HOLD, OVERLAP, PUBLISH, SERIAL, TAIL, TOOL_WAIT, CandidateEdge,
    EdgeCost, LoadWindow, Recommender, RecommendMode, attribute, predict_gain, to_load_window)

PH = "a" * 64
EDGE = CandidateEdge("P2", "P4", 2, 4)
COSTS = {(PH, "P2", "P4"): EdgeCost(PH, "P2", "P4", 30.0, 60.0, 30.0)}


class FakeController:
    def __init__(self, epoch=3, config="P2", phash=PH, pause_ok=True):
        self.journal = NS(epochs=NS(config_epoch=epoch, config_id=config))
        self.profile = NS(contract_hash=phash)
        self.pause_ok = pause_ok
        self.plans, self.requests = [], []

    def plan(self, target, expected_epoch, *, deadline_s=0.0):
        self.plans.append(target)
        if not self.pause_ok:
            raise Rejected("pause not allowed: budget")
        return NS(target=target)

    def request(self, request_id, target, expected_epoch, deadline_s):
        self.requests.append((request_id, target, expected_epoch, deadline_s))
        return {"request_id": request_id}


def win(i=0, busy=0.6, tool=0.05, tail=0.05, pub=0.05, queued=5, epoch=3, ph=PH, train=None):
    return LoadWindow(i * 60.0, (i + 1) * 60.0, ph, epoch, busy, tool, tail, pub,
                      queued=queued, train_fraction=train)


def stable(n=4, **kw):
    return [win(i, **kw) for i in range(n)]


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def rec_mode(**kw):
    return Recommender(mode=RecommendMode.RECOMMEND, clock=kw.pop("clock", Clock()), **kw)


def test_attribution_distinguishes_four_loads():
    assert attribute(win(busy=0.9))["dominant"] == GPU_SAT
    assert attribute(win(busy=0.3, tool=0.5, queued=0))["dominant"] == TOOL_WAIT
    assert attribute(win(busy=0.3, tail=0.4, queued=0))["dominant"] == TAIL
    assert attribute(win(busy=0.3, pub=0.4, queued=0))["dominant"] == PUBLISH
    assert attribute(win(busy=0.5, queued=0))["dominant"] == BALANCED


def test_tool_wait_and_tail_do_not_scale():
    a = predict_gain(stable(busy=0.3, tool=0.6), EDGE, SERIAL)
    b = predict_gain(stable(busy=0.3, tool=0.0), EDGE, SERIAL)
    assert a.gain_upper == b.gain_upper == pytest.approx(0.15)


def test_serial_vs_overlap_not_max():
    ws = stable(busy=0.6, train=0.5)
    s = predict_gain(ws, EDGE, SERIAL)
    o = predict_gain(ws, EDGE, OVERLAP)
    assert s.gain_upper == pytest.approx(0.3)
    # overlap: old critical path 0.7, new max(0.4, 0.5)=0.5 -> 0.2, not 0.3
    assert o.gain_upper == pytest.approx(0.2)
    # trainer-bound overlap: no gain at all
    assert predict_gain(stable(busy=0.2, tool=0, tail=0, train=0.9), EDGE, OVERLAP).gain_upper == 0
    assert predict_gain(stable(), EDGE, OVERLAP).gain_lower is None


def test_unknown_cost_gives_no_recommendation():
    r = rec_mode().recommend(FakeController(), stable(), [EDGE], {})
    assert r.target == r.source and r.reason == HOLD and "unknown transition cost" in r.rejection_reason


def test_no_net_gain_holds():
    big = {(PH, "P2", "P4"): EdgeCost(PH, "P2", "P4", 1e5, 1e6)}
    r = rec_mode().recommend(FakeController(), stable(), [EDGE], big)
    assert not r.actionable and "no net gain" in r.rejection_reason
    r = rec_mode().recommend(FakeController(), stable(busy=0.05, tool=0.8, queued=0), [EDGE], COSTS)
    assert not r.actionable


def test_recommendation_fields_and_no_execution():
    c = FakeController()
    r = rec_mode().recommend(c, stable(), [EDGE], COSTS)
    assert r.actionable and (r.source, r.target, r.expected_epoch, r.profile_hash) == ("P2", "P4", 3, PH)
    assert r.gain_lower <= r.gain_upper and r.cost_upper == 60.0 and r.expires_at > r.issued_at
    assert r.evidence["n"] == 4
    assert c.requests == [] and c.plans == []


def test_windows_of_other_epoch_or_profile_ignored():
    r = rec_mode().recommend(FakeController(), stable(epoch=2) + stable(ph="b" * 64), [EDGE], COSTS)
    assert not r.actionable


def test_approve_uses_same_request_entry():
    c = FakeController()
    rm = rec_mode()
    r = rm.recommend(c, stable(), [EDGE], COSTS)
    out = rm.approve(r, c, stable(), [EDGE], COSTS, deadline_s=600)
    assert c.requests == [(r.request_id, "P4", 3, 600)] and out == {"request_id": r.request_id}


@pytest.mark.parametrize("change,msg", [
    ("expire", "expired"), ("epoch", "epoch changed"), ("profile", "profile hash"),
    ("pause", "pause not allowed"), ("load", "no longer justifies"), ("edge", "no longer a candidate"),
])
def test_revalidation_rejects(change, msg):
    clock = Clock()
    rm = rec_mode(clock=clock)
    c = FakeController()
    r = rm.recommend(c, stable(), [EDGE], COSTS)
    ws, edges = stable(), [EDGE]
    if change == "expire":
        clock.t += 1e4
    elif change == "epoch":
        c.journal.epochs.config_epoch = 4
    elif change == "profile":
        c.profile.contract_hash = "b" * 64
    elif change == "pause":
        c.pause_ok = False
    elif change == "load":
        ws = stable(busy=0.05, tool=0.8, queued=0)
    elif change == "edge":
        edges = [CandidateEdge("P2", "P6", 2, 6)]
    with pytest.raises(Rejected, match=msg):
        rm.approve(r, c, ws, edges, COSTS, deadline_s=600)
    assert c.requests == []


def test_hold_cannot_be_approved():
    c = FakeController()
    rm = rec_mode()
    r = rm.recommend(c, stable(), [EDGE], {})
    with pytest.raises(Rejected, match="not actionable"):
        rm.approve(r, c, stable(), [EDGE], COSTS, deadline_s=600)


@pytest.mark.parametrize("mode", [RecommendMode.DISABLED, RecommendMode.MANUAL])
def test_disabled_and_manual_do_nothing(mode):
    c = FakeController()
    rm = Recommender(mode=mode)
    assert rm.recommend(c, stable(), [EDGE], COSTS) is None
    r = rec_mode().recommend(c, stable(), [EDGE], COSTS)
    with pytest.raises(Rejected, match="recommend mode"):
        rm.approve(r, c, stable(), [EDGE], COSTS, deadline_s=600)
    assert c.requests == [] and Recommender().mode is RecommendMode.DISABLED


def test_adapter_duck_typed():
    obj = NS(window_start=0, window_end=60, profile_hash=PH, epoch=3, gpu_busy_fraction=0.5,
             tool_wait_fraction=0.1, tail_wait_fraction=0.1, publish_block_fraction=0.0,
             queued=1, active=2, ready_groups=0, consume_rate=1.0, policy_age=0, extra=1)
    w = to_load_window(obj)
    assert w.duration_s == 60 and w.active == 2 and w.train_fraction is None
    with pytest.raises(ValueError):
        to_load_window(NS(window_start=0, window_end=1))
