"""D2 6.4/6.5 + D1<->1.7 wiring (CPU only)."""
import json
from types import SimpleNamespace as NS

import pytest

from yeto.rl.engine.auto import AutoController, AutoPolicy
from yeto.rl.engine.controller import SUCCEEDED, Rejected, read_journal
from yeto.rl.engine.recommend import (CandidateEdge, EdgeCost, LoadWindow, Recommender,
                                      RecommendMode, candidate_edges_from_attestation,
                                      edge_costs_from_table, to_load_window)
from yeto.rl.engine.timeline import load_windows

PH = "a" * 64
EDGE = CandidateEdge("P2", "P4", 2, 4)
COSTS = {(PH, "P2", "P4"): EdgeCost(PH, "P2", "P4", 30.0, 60.0, 30.0)}


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeCtl:
    def __init__(self, mode="auto", fail=False):
        self.journal = NS(epochs=NS(config_epoch=3, config_id="P2"))
        self.profile = NS(contract_hash=PH)
        self.recommend_mode = mode
        self.finalizing = None
        self.fail = fail
        self.plans, self.requests, self.cancels = [], [], []
        self._status = {}
        self.pending_tx = False
        self.certified = {("P2", "P4")}

    def plan(self, target, expected_epoch, *, deadline_s=0.0):
        self.plans.append(target)
        if (self.journal.epochs.config_id, target) not in self.certified:
            raise Rejected("edge is not certified")
        return NS(target=target)

    def request(self, rid, target, epoch, deadline_s):
        self.plan(target, epoch, deadline_s=deadline_s)
        self.requests.append((rid, target))
        self._status[rid] = {"terminal": False, "phase": "VALIDATING"}
        return {"request_id": rid}

    def finish(self, ok=True):
        rid = self.requests[-1][0]
        self._status[rid] = {"terminal": True, "phase": SUCCEEDED if ok else "REBUILT_OLD"}
        if ok:
            e = self.journal.epochs
            self.journal.epochs = NS(config_epoch=e.config_epoch + 1, config_id=self.requests[-1][1])

    def status(self, rid):
        return self._status.get(rid, {"known": False})

    def has_pending(self):
        return self.pending_tx or any(not s["terminal"] for s in self._status.values())

    def set_recommend_mode(self, mode, *, reason=""):
        self.recommend_mode = mode
        return {"mode": mode}


def win(i, busy=0.8, tool=0.02, tail=0.02, queued=5, epoch=3):
    return LoadWindow(i * 60.0, (i + 1) * 60.0, PH, epoch, busy, tool, tail, 0.02, queued=queued)


def auto(clock=None, **kw):
    clock = clock or Clock()
    pol = AutoPolicy(**{"k_windows": 4, "min_dwell_s": 600, "cooldown_s": 600,
                        "max_switches": 1, "switch_window_s": 3600, **kw})
    return AutoController(Recommender(clock=clock), pol, clock=clock), clock


def step(a, c, ws, **kw):
    kw.setdefault("remaining_budget_s", 10_000.0)
    return a.step(c, ws, [EDGE], COSTS, **kw)


# ---------------------------------------------------------------- 6.5 modes
def test_modes_and_default_disabled():
    assert [m.value for m in RecommendMode] == ["disabled", "manual", "recommend", "auto"]
    assert Recommender().mode is RecommendMode.DISABLED
    a, _ = auto()
    c = FakeCtl(mode="disabled")
    for m in ("disabled", "manual", "recommend"):
        c.recommend_mode = m
        assert step(a, c, [win(i) for i in range(6)])["action"] == "hold"
    assert c.requests == [] and c.plans == []


# ---------------------------------------------------------------- 6.4 triggers
def test_steady_imbalance_triggers_via_revalidate_and_plan():
    a, clock = auto()
    c = FakeCtl()
    d = step(a, c, [win(i) for i in range(6)])
    assert d["action"] == "requested" and d["target"] == "P4"
    assert c.plans.count("P4") >= 2  # revalidate's plan + request's plan
    assert c.requests[0][0].startswith("auto-")


def test_k_windows_required():
    a, _ = auto()
    c = FakeCtl()
    assert "consecutive" in step(a, c, [win(i) for i in range(3)])["reason"]
    assert c.requests == []


def test_oscillating_load_does_not_switch():
    a, clock = auto()
    c = FakeCtl()
    hist = []
    for i in range(40):
        hist.append(win(i, busy=0.8 if i % 2 == 0 else 0.05, queued=5 if i % 2 == 0 else 0))
        clock.t += 60
        step(a, c, hist)
    assert c.requests == []


def test_oscillating_regimes_are_rate_limited():
    """Long alternating regimes (each > K windows): switches bounded by dwell/cooldown/rate."""
    a, clock = auto(max_switches=1, switch_window_s=1e9)
    c = FakeCtl()
    c.certified |= {("P4", "P2")}
    edges = [EDGE, CandidateEdge("P4", "P2", 4, 2)]
    costs = dict(COSTS)
    costs[(PH, "P4", "P2")] = EdgeCost(PH, "P4", "P2", 30.0, 60.0, 30.0)
    hist = []
    for i in range(200):
        high = (i // 5) % 2 == 0
        hist.append(win(i, busy=0.8 if high else 0.05, queued=5 if high else 0,
                        epoch=c.journal.epochs.config_epoch))
        clock.t += 60
        a.step(c, hist, edges, costs, remaining_budget_s=1e6)
        if c.has_pending():
            c.finish()
    assert len(c.requests) <= 1


def test_safety_margin_and_horizon_bounded_by_budget():
    a, _ = auto(safety_margin_s=1e6)
    c = FakeCtl()
    assert "no net gain" in step(a, c, [win(i) for i in range(6)])["reason"]
    a, _ = auto()
    d = step(a, c, [win(i) for i in range(6)], remaining_budget_s=100.0)
    assert d["action"] == "hold" and "no net gain" in d["reason"]
    d = step(a, c, [win(i) for i in range(6)], remaining_budget_s=None)
    assert "budget" in d["reason"] and c.requests == []


def test_tool_heavy_and_finalization_hold():
    a, _ = auto()
    c = FakeCtl()
    assert "tool" in step(a, c, [win(i) for i in range(6)], tool_heavy=True)["reason"]
    assert "tool" in step(a, c, [win(i, busy=0.3, tool=0.6, queued=0) for i in range(6)])["reason"]
    c.finalizing = 7
    assert "final" in step(a, c, [win(i) for i in range(6)])["reason"]
    assert c.requests == []


def test_no_cost_table_always_holds(tmp_path):
    a, _ = auto()
    c = FakeCtl()
    costs = edge_costs_from_table(tmp_path / "missing.json")
    assert costs == {}
    d = a.step(c, [win(i) for i in range(6)], [EDGE], costs, remaining_budget_s=1e6)
    assert d["action"] == "hold" and c.requests == []


def test_uncertified_trainer_edge_not_selectable():
    att = NS(certified_edges=frozenset({("P2", "P4", "rollout-only")}))
    cfg = {n: NS(rollout=r, rollout_engine_gpus=1) for n, r in
           (("P2", 2), ("P4", 4), ("T6R2", 2))}
    edges = candidate_edges_from_attestation(att, cfg, source="P2")
    assert edges == [EDGE]
    # a cost row for an uncertified trainer edge does not make it a candidate
    a, _ = auto()
    c = FakeCtl()
    costs = dict(COSTS)
    costs[(PH, "P2", "T6R2")] = EdgeCost(PH, "P2", "T6R2", 0.0, 0.0, 0.0)
    d = a.step(c, [win(i) for i in range(6)], edges, costs, remaining_budget_s=1e6)
    assert d["target"] == "P4"
    # even if handed an uncertified edge, controller.plan refuses (via revalidate)
    a, _ = auto()
    c = FakeCtl()
    c.certified = set()
    d = step(a, c, [win(i) for i in range(6)])
    assert "revalidation refused" in d["reason"] and c.requests == []


def test_failed_switch_disables_auto_keeps_manual():
    a, clock = auto()
    c = FakeCtl()
    assert step(a, c, [win(i) for i in range(6)])["action"] == "requested"
    c.finish(ok=False)
    clock.t += 10_000
    d = step(a, c, [win(i) for i in range(6)])
    assert d["action"] == "disabled" and c.recommend_mode == "manual"
    clock.t += 10_000
    assert step(a, c, [win(i) for i in range(6)])["reason"] == "auto mode is off"
    assert len(c.requests) == 1


def test_dwell_and_cooldown_after_success():
    a, clock = auto(max_switches=5)
    c = FakeCtl()
    c.certified |= {("P4", "P8")}
    e2 = CandidateEdge("P4", "P8", 4, 8)
    costs = dict(COSTS)
    costs[(PH, "P4", "P8")] = EdgeCost(PH, "P4", "P8", 1.0, 1.0, 1.0)
    step(a, c, [win(i) for i in range(6)])
    c.finish()
    ws = [win(i, epoch=4) for i in range(6)]
    clock.t += 60
    assert a.step(c, ws, [e2], costs, remaining_budget_s=1e6)["reason"] == "minimum dwell not reached"
    clock.t += 700
    assert a.step(c, ws, [e2], costs, remaining_budget_s=1e6)["action"] == "requested"


# ---------------------------------------------------------------- 1.7 wiring
def test_load_summary_train_fraction_and_adapter():
    span = lambda s, e, role: {"event": "rl_timeline_span", "start": s, "end": e,  # noqa: E731
                               "kind": "compute", "role": role, "profile_hash": PH, "epoch": 3}
    evs = [span(0, 30, "rollout"), span(20, 50, "trainer"), span(40, 45, "trainer+rollout")]
    (s,) = load_windows(evs, 60.0)
    assert s.train_fraction == pytest.approx(30 / 60)
    assert s.gpu_busy_fraction == pytest.approx(50 / 60)
    w = to_load_window(s)
    assert (w.start_s, w.end_s, w.train_fraction, w.tool_wait_fraction) == (0, 60, 0.5, 0.0)


def test_edge_costs_from_table(tmp_path):
    row = {"profile_hash": PH, "source": "P2", "target": "P4", "cost_lower_s": 30,
           "cost_upper_s": 60, "recovery_upper_s": 30, "n": 3, "provenance": "5.7 run x"}
    p = tmp_path / "c.json"
    p.write_text(json.dumps([row]))
    assert edge_costs_from_table(p) == COSTS
    for bad in ({**row, "n": 0}, {**row, "cost_lower_s": 99}, {k: v for k, v in row.items()
                                                                 if k != "provenance"}):
        with pytest.raises(ValueError):
            edge_costs_from_table([bad])
    with pytest.raises(ValueError, match="duplicate"):
        edge_costs_from_table([row, row])


# ---------------------------------------------------------------- 6.5 real controller
def test_mode_switch_does_not_interrupt_transaction(tmp_path):
    from test_rl_reconfig_e1 import _setup
    from yeto.rl.engine.controller import CommandInbox

    driver, ctl, *_ = _setup(tmp_path, inbox=True)
    assert ctl.recommend_mode == "disabled"  # legacy default
    inbox = CommandInbox(tmp_path / "state" / "inbox")
    with pytest.raises(Rejected, match="auto_controller"):  # d2-wire: capability gate
        ctl.set_recommend_mode("auto")
    import dataclasses
    ctl.attestation = dataclasses.replace(ctl.attestation, auto_controller=True)
    ctl.set_recommend_mode("auto")
    ctl.request("m1", "T4R4S0", 0, 60)
    inbox.submit("sw", "mode", {"mode": "disabled"})
    driver.run()
    assert ctl.recommend_mode == "disabled"
    assert ctl.status("m1")["phase"] == SUCCEEDED
    st = json.loads((tmp_path / "state/inbox/sw.status.json").read_text())
    assert st["mode"] == "disabled" and st["previous"] == "auto"
    kinds = [r["kind"] for r in read_journal(tmp_path / "state/reconfig")]
    assert "recommend_mode" in kinds and "cancel" not in kinds
    with pytest.raises(Rejected):
        ctl.set_recommend_mode("bogus")
    # manual path still validated by plan regardless of mode
    with pytest.raises(Rejected):
        ctl.request("m2", "T4R4S0", 0, 60)  # stale epoch
