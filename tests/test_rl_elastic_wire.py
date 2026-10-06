"""d2-wire: D1/D2 wired into the driver safe point with the Flash-Next profile (CPU only).

Real IslandController + IslandDriver + elastic fakes (test_rl_reconfig_e1), the
Flash-Next ExecutionProfile and elastic declaration, synthetic load windows for
four load types and a fake 5.7 cost table.
"""
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from test_rl_reconfig_e1 import (FP, NAME, ElasticFakePool, ElasticFakePublisher,
                                 ForkMembership, _events)
from yeto.rl.elastic_benchmark.capabilities import Attestation
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.auto import AutoController, AutoPolicy
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.controller import SUCCEEDED, IslandController, Rejected, read_journal
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.elastic import AUTO_STATE_KIND, ElasticHook, restore_auto_state
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.miles_adapter.elastic_placement import ElasticPlacement
from yeto.rl.engine.ports import PlacementDescription
from yeto.rl.engine.recommend import Recommender
from yeto.rl.profiles import qwen3_8_next as q

DECL = q.flash_next_elastic_declaration()
SMALL, BIG = "FN-T16R8S8", "FN-T16R16S0"
W = 100.0  # window seconds


def _profile():
    return q.flash_next_execution_profile(AlgorithmSpec())


PH = _profile().contract_hash


class _Static:
    def describe(self):
        return PlacementDescription("fixed-partition", tuple(f"g{i}" for i in range(16)),
                                    tuple(f"g{i}" for i in range(16, 24)))


class _Placement:
    def __init__(self):
        self.inner = ElasticPlacement(_Static(), pool_gpus=tuple(f"g{i}" for i in range(32)))

    def describe(self):
        return self.inner.describe()

    def reconfigure(self, plan, *, epoch):
        return self.inner.reconfigure(plan, epoch=epoch)


def _attestation(certified=True, auto=True):
    edges = frozenset((a, b, q.ELASTIC_EDGE_KIND) for a, b in DECL["declared_edges"]) \
        if certified else frozenset()
    return Attestation(FP, frozenset({"colocated-serial", "partitioned-serial"}), edges,
                       frozenset(), auto, True)


def synth(kind, n, epoch=0, t0=0.0):
    """n windows of one load type, as the driver's observe events would show it."""
    busy, pub, sample = {
        "saturated": (0.9, 0.0, dict(queued_requests=6, running_requests=8, tool_wait_trajectories=0)),
        "tool_wait": (0.2, 0.0, dict(queued_requests=0, running_requests=0, tool_wait_trajectories=5)),
        "long_tail": (0.4, 0.0, dict(queued_requests=0, running_requests=2, tool_wait_trajectories=0)),
        "publish": (0.3, 0.5, dict(queued_requests=0, running_requests=8, tool_wait_trajectories=0)),
    }[kind]
    evs = []
    for i in range(n):
        lo = t0 + i * W
        lab = dict(profile_hash=PH, epoch=epoch)
        evs.append({"event": "rl_timeline_span", "task": "generate", "role": "rollout",
                    "kind": "compute", "start": lo, "end": lo + busy * W, **lab})
        if pub:
            evs.append({"event": "rl_timeline_span", "task": "publish", "role": "trainer",
                        "kind": "transfer", "start": lo + busy * W, "end": lo + (busy + pub) * W,
                        **lab})
        for j in range(4):
            evs.append({"event": "rl_load_sample", "t": lo + j * W / 4, "engine_capacity": 8,
                        "ready_groups": 0, **sample, **lab})
    return evs


def cost_table(tmp_path, ph=PH):
    rows = [{"profile_hash": ph, "source": SMALL, "target": BIG, "cost_lower_s": 40,
             "cost_upper_s": 90, "recovery_upper_s": 60, "n": 3, "provenance": "fake 5.7 table"}]
    p = tmp_path / "edge_costs.json"
    p.write_text(json.dumps(rows))
    return str(p)


class Clock:
    def __init__(self, t=0.0, step=1.0):
        self.t, self.step = t, step

    def __call__(self):
        self.t += self.step
        return self.t


def setup(tmp_path, *, mode="disabled", hook=True, events=None, costs=None, certified=True,
          auto_cap=True, rounds=4, wall=None, observe=True):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                        placement_kind="fixed-partition")
    fork = ForkMembership(engine, declared=["engine:c0", "engine:c1"], running=["engine:c0"])
    pool = ElasticFakePool(engine, fork)
    publisher = ElasticFakePublisher(engine, fork)
    configs = dict(DECL["configs"])
    configs[SMALL] = dataclasses.replace(configs[SMALL], placement={"rollout": [f"g{i}" for i in range(16, 24)]})
    configs[BIG] = dataclasses.replace(configs[BIG], placement={"rollout": [f"g{i}" for i in range(16, 32)]})
    wall = wall or Clock(10_000.0, 0.0)
    ctl = IslandController(state_dir=tmp_path / "state", configs=configs,
                           attestation=_attestation(certified, auto_cap), profile=_profile(),
                           initial_config=SMALL, runtime_fingerprint=FP, wall_clock=wall,
                           sleep=lambda s: None)
    ctl.open(pool)
    if mode != "disabled":
        ctl.set_recommend_mode(mode, reason="test")
    h = None
    if hook:
        h = ElasticHook(configs, window_s=W, total_rounds=1000, edge_costs_path=costs,
                        declared_edges=DECL["declared_edges"],
                        recommender=Recommender(min_windows=3, clock=wall),
                        auto=AutoController(Recommender(clock=wall),
                                            AutoPolicy(k_windows=3, safety_margin_s=60.0,
                                                       min_dwell_s=0.0, cooldown_s=600.0),
                                            clock=wall),
                        events_source=(lambda: events(ctl)) if events else None)
    driver = IslandDriver(
        learner_id=0, rollout=pool, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=publisher, placement=_Placement(), algorithm=AlgorithmSpec(),
        sync=LocalOnlySync(rounds), events=EventTape(tmp_path / "events.jsonl", 0),
        capabilities=fake_capabilities(execution_modes={"colocated-serial", "partitioned-serial"},
                                       auto_controller=auto_cap),
        profile=_profile(), controller=ctl, observe=observe, clock=Clock(0.0, 10.0),
        elastic_hook=h)
    return driver, ctl, h, fork


def _ev(kind):
    return lambda ctl: synth(kind, 4, epoch=int(ctl.journal.epochs.config_epoch))


# ---------------------------------------------------------------- declaration
def test_flash_next_declaration_is_data_only_and_uncertified_edges_unselectable(tmp_path):
    assert DECL["model"] == "Qwen3.8-Flash-Next" and DECL["gpu"] == "H200"
    assert DECL["nodes"] * DECL["gpus_per_node"] == 32 and DECL["trainable"] == "lora"
    assert set(DECL["configs"]) == {SMALL, BIG}
    for c in DECL["configs"].values():
        assert c.total == 32 and c.dims["ep"] == 2 and c.rollout_engine_gpus == 8
    assert DECL["declared_edges"] == {(SMALL, BIG), (BIG, SMALL)}
    driver, ctl, h, _ = setup(tmp_path, certified=False, mode="recommend",
                              events=_ev("saturated"), costs=cost_table(tmp_path))
    assert h.candidates(ctl) == []  # declared but not attested -> not a candidate
    with pytest.raises(Rejected):
        ctl.plan(BIG, 0)


def test_profile_hash_threads_windows_recommendation_and_cost_lookup(tmp_path):
    driver, ctl, h, _ = setup(tmp_path, mode="recommend", events=_ev("saturated"),
                              costs=cost_table(tmp_path))
    driver.run()
    recs = [e for e in _events(tmp_path) if e["event"] == "rl_elastic_recommendation"]
    assert recs and all(e["profile_hash"] == PH for e in recs)
    actionable = [e for e in recs if e["recommendation"]["rejection_reason"] is None]
    assert actionable and all(e["recommendation"]["profile_hash"] == PH
                              and e["recommendation"]["target"] == BIG for e in actionable)
    # recommend never executes
    assert ctl.journal.epochs.config_epoch == 0
    kinds = [r["kind"] for r in read_journal(tmp_path / "state/reconfig")]
    assert "request" not in kinds and "recommendation" in kinds
    # a cost row under another profile hash is not found -> hold
    driver2, ctl2, _, _ = setup(tmp_path / "b", mode="recommend", events=_ev("saturated"),
                                costs=cost_table(tmp_path / "b", ph="sha256:" + "f" * 64)
                                if (tmp_path / "b").mkdir() is None else None)
    driver2.run()
    recs2 = [e for e in _events(tmp_path / "b") if e["event"] == "rl_elastic_recommendation"]
    assert all("unknown transition cost" in (e["recommendation"]["rejection_reason"] or "")
               or "valid windows" in (e["recommendation"]["rejection_reason"] or "")
               for e in recs2)


@pytest.mark.parametrize("kind,dominant", [("saturated", "gpu_saturation"),
                                           ("tool_wait", "tool_wait"),
                                           ("long_tail", "long_tail"),
                                           ("publish", "publish_block")])
def test_recommend_mode_attributes_four_load_types(tmp_path, kind, dominant):
    driver, ctl, h, _ = setup(tmp_path, mode="recommend", events=_ev(kind),
                              costs=cost_table(tmp_path))
    driver.run()
    recs = [e for e in _events(tmp_path) if e["event"] == "rl_elastic_recommendation"]
    assert recs
    ev = recs[-1]["recommendation"]["evidence"]["dominant"]
    assert max(ev, key=ev.get) == dominant
    assert recs[-1]["tool_heavy"] == (kind == "tool_wait")
    assert ctl.journal.epochs.config_epoch == 0


def test_auto_switches_once_through_the_same_request_entry(tmp_path):
    driver, ctl, h, fork = setup(tmp_path, mode="auto", events=_ev("saturated"),
                                 costs=cost_table(tmp_path), rounds=5)
    calls = []
    real = ctl.request
    ctl.request = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    driver.run()
    assert len(calls) == 1 and calls[0][0].startswith("auto-") and calls[0][1] == BIG
    assert ctl.status(calls[0][0])["phase"] == SUCCEEDED
    assert ctl.journal.epochs.config_id == BIG and ctl.journal.epochs.config_epoch == 1
    autos = [e for e in _events(tmp_path) if e["event"] == "rl_elastic_auto"]
    assert [e["action"] for e in autos].count("requested") == 1
    assert any(e["event"] == "rl_reconfiguration" and e["result"] == SUCCEEDED
               for e in _events(tmp_path))
    states = [r for r in read_journal(tmp_path / "state/reconfig") if r["kind"] == AUTO_STATE_KIND]
    assert states and states[-1]["state"]["pending"] is None
    assert states[-1]["state"]["last_change_at"] is not None


@pytest.mark.parametrize("kind", ["tool_wait", "saturated"])
def test_auto_holds_without_costs_or_on_tool_heavy(tmp_path, kind):
    costs = cost_table(tmp_path) if kind == "tool_wait" else None
    driver, ctl, h, _ = setup(tmp_path, mode="auto", events=_ev(kind), costs=costs)
    driver.run()
    autos = [e for e in _events(tmp_path) if e["event"] == "rl_elastic_auto"]
    assert autos and all(e["action"] == "hold" for e in autos)
    assert ctl.journal.epochs.config_epoch == 0


def test_auto_refused_without_capability(tmp_path):
    driver, ctl, h, _ = setup(tmp_path, auto_cap=False)
    with pytest.raises(Rejected, match="auto_controller"):
        ctl.set_recommend_mode("auto")
    assert ctl.recommend_mode == "disabled"
    ctl.set_recommend_mode("recommend")  # other modes unaffected


def test_auto_state_survives_restart(tmp_path):
    wall = Clock(10_000.0, 0.0)
    driver, ctl, h, _ = setup(tmp_path, mode="auto", events=_ev("saturated"),
                              costs=cost_table(tmp_path), rounds=3, wall=wall)
    driver.run()
    before = {k: getattr(h.auto, k) for k in ("last_change_at", "last_attempt_at", "switches")}
    assert before["switches"]
    ctl.close()
    fresh = AutoController(Recommender(), AutoPolicy(k_windows=3), clock=wall)
    assert restore_auto_state(fresh, read_journal(tmp_path / "state/reconfig"))
    assert {k: getattr(fresh, k) for k in before} == before
    # restarted controller: mode replayed, a new hook restores and stays in cooldown
    ctl2 = IslandController(state_dir=tmp_path / "state", configs=ctl.configs,
                            attestation=ctl.attestation, profile=_profile(),
                            initial_config=SMALL, runtime_fingerprint=FP, wall_clock=wall,
                            sleep=lambda s: None)
    assert ctl2.recommend_mode == "auto"
    hook2 = ElasticHook(ctl.configs, window_s=W, auto=AutoController(
        Recommender(), AutoPolicy(k_windows=3, cooldown_s=600.0), clock=wall))
    hook2._restore(ctl2)
    assert hook2.auto.switches == before["switches"]
    ctl2.close()


def _strip(evs):
    return [{k: v for k, v in e.items() if k not in ("time_unix", "train_seconds", "rollout_seconds")} for e in evs]


@pytest.mark.parametrize("observe", [True, False])
def test_disabled_hook_is_event_identical_to_legacy(tmp_path, observe):
    a, ctl_a, _, _ = setup(tmp_path / "a", hook=False, observe=observe)
    a.run()
    b, ctl_b, h, _ = setup(tmp_path / "b", hook=True, events=_ev("saturated"),
                           costs=None, observe=observe)
    b.run()
    assert _strip(_events(tmp_path / "a")) == _strip(_events(tmp_path / "b"))
    assert h.decisions == []


def test_train_heavy_load_does_not_inflate_resize_gain():
    """gpu_busy_fraction includes trainer compute; only rollout compute scales."""
    from yeto.rl.engine.recommend import SERIAL, CandidateEdge, predict_gain, to_load_window
    from yeto.rl.engine.timeline import load_windows

    lab = dict(profile_hash=PH, epoch=0)
    evs = []
    for i in range(3):
        lo = i * W
        evs.append({"event": "rl_timeline_span", "task": "generate", "role": "rollout",
                    "kind": "compute", "start": lo, "end": lo + 0.2 * W, **lab})
        evs.append({"event": "rl_timeline_span", "task": "train", "role": "trainer",
                    "kind": "compute", "start": lo + 0.2 * W, "end": lo + 0.9 * W, **lab})
    ws = load_windows(evs, W)
    assert all(abs(w.gpu_busy_fraction - 0.9) < 1e-9 and abs(w.train_fraction - 0.7) < 1e-9
               and abs(w.rollout_busy_fraction - 0.2) < 1e-9 for w in ws)
    edge = CandidateEdge(SMALL, BIG, 1, 2)
    g = predict_gain([to_load_window(w) for w in ws], edge, SERIAL, efficiency_lower=1.0)
    assert abs(g.gain_upper - 0.1) < 1e-9  # 0.2 * (1 - 1/2), not 0.9 * (1 - 1/2)
    # legacy windows without the field keep gpu_busy_fraction as the scalable share
    legacy = dataclasses.replace(to_load_window(ws[0]), rollout_busy_fraction=None)
    assert abs(predict_gain([legacy], edge, SERIAL, efficiency_lower=1.0).gain_upper - 0.45) < 1e-9
