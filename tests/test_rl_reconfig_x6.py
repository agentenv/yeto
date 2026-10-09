"""rl-infra-spec 3.8 (X6) CPU protocol tests: manual bidirectional rollout
switching on one island of a two-island strict-avg fleet.

The syncer and fork are fakes (FakeStrictSyncer re-sends the same PULL the
way ``server.rs`` does on a quorum timeout; ForkMembership models the fork
contract). These prove the yeto-side protocol only; the X6 acceptance is the
GPU experiment in ``evidence/infra-e1/plan-3.8-4.4.md`` (A5), not these tests.
"""

from __future__ import annotations

import json
import threading

import pytest
import torch

from yeto.protocol import PullRequest
from yeto.rl.engine.bridges import StrictAvgSync
from yeto.rl.engine.controller import CANCELLED, SUCCEEDED, IslandController, Rejected
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.journal import read_journal
from yeto.rl.engine.pause_audit import PAUSABLE_PHASE

from test_rl_engine_driver import _strict_config, _strict_syncer
from test_rl_reconfig_e1 import CONFIGS, FP, MODES, NAME, _attestation, _profile, _setup


def _counting(syncer):
    pushes = []
    original = syncer.push

    def push(client, fragment_id, step, attempt, base, data):
        pushes.append((client.learner_id, step, attempt, base))
        return original(client, fragment_id, step, attempt, base, data)

    syncer.push = push
    return pushes


def _plain_island(tmp_path, syncer, *, learner_id, rounds):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=3.0,
                        placement_kind="fixed-partition")
    sync = StrictAvgSync(
        _strict_config(tmp_path, engine, learner_id=learner_id, rounds=rounds,
                       tape=f"bridge-{learner_id}.jsonl"),
        client_factory=lambda _b: syncer.client(learner_id),
    )
    trained = []
    original = engine.trainer.train_step
    engine.trainer.train_step = lambda b: trained.append(
        tuple(s for g in b.groups for s in g.sample_ids)) or original(b)
    driver = IslandDriver(
        learner_id=learner_id, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, algorithm=AlgorithmSpec(), sync=sync,
        events=EventTape(tmp_path / f"island-{learner_id}.jsonl", learner_id),
        capabilities=fake_capabilities(execution_modes=MODES),
        profile=_profile(outer="strict-avg"),
    )
    return driver, trained


def _fleet(tmp_path, *, rounds, controller_kw=None):
    """Island 0: elastic (controller + fork fakes); island 1: plain strict."""
    engine0 = FakeEngine(tensors={NAME: torch.zeros(1, 2)})
    syncer = _strict_syncer(engine0, learners=2, rounds=rounds)
    pushes = _counting(syncer)
    clients = {}

    def sync0(engine):
        def factory(_b):
            clients[0] = syncer.client(0)
            return clients[0]
        return StrictAvgSync(_strict_config(tmp_path, engine, learner_id=0, rounds=rounds,
                                            tape="bridge-0.jsonl"), client_factory=factory)

    island0 = _setup(tmp_path / "i0", rounds=rounds, sync_factory=sync0, outer="strict-avg",
                     controller_kw=controller_kw)
    island1 = _plain_island(tmp_path, syncer, learner_id=1, rounds=rounds)
    return syncer, pushes, clients, island0, island1


def _run(drivers):
    results, errors = {}, {}

    def run(i, d):
        try:
            results[i] = d.run()
        except BaseException as error:  # noqa: BLE001
            errors[i] = error

    threads = [threading.Thread(target=run, args=(i, d), daemon=True)
               for i, d in enumerate(drivers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()
    return results, errors


def _journal(tmp_path):
    return read_journal(tmp_path / "i0" / "state" / "reconfig")


def test_bidirectional_switch_on_one_strict_island_keeps_fleet_invariants(tmp_path):
    rounds = 4
    # baseline: the same fleet, no request
    b_syncer, b_pushes, _, (b0, *_, b_trained0, _), (b1, b_trained1) = _fleet(
        tmp_path / "base", rounds=rounds)
    b_results, b_errors = _run([b0, b1])
    assert b_errors == {}

    syncer, pushes, clients, island0, (d1, trained1) = _fleet(tmp_path / "x", rounds=rounds)
    d0, ctl, fork, pool, publisher, trained0, _ = island0
    requests = {1: ("up", "T4R4S0", 0), 2: ("down", "T4R2S2", 1)}
    resent = []
    orig_add = pool.add_engines

    def add_engines(count, *, epoch, members=None):
        # the fleet is paused here with island 0 holding PULL(2): the syncer's
        # quorum timeout re-sends the SAME PULL (server.rs); it must be ignored
        permit = d0.sync.permit
        dup = PullRequest(permit.fragment_id, permit.global_step, permit.round_attempt)
        with syncer.lock:
            clients[0].pulls.append(dup)
        resent.append(dup)
        return orig_add(count, epoch=epoch, members=members)

    pool.add_engines = add_engines
    orig_safe = d0.safe_point

    def safe_point(rollout_id):
        if rollout_id in requests:
            rid, target, epoch = requests[rollout_id]
            ctl.request(rid, target, epoch, 60)
        return orig_safe(rollout_id)

    d0.safe_point = safe_point
    results, errors = _run([d0, d1])
    assert errors == {}
    assert resent and resent[0].global_step == 2
    # the resend is on island 0's JSONL tape (A5 quorum criterion)
    tape = [json.loads(x) for x in (tmp_path / "x" / "bridge-0.jsonl").read_text().splitlines()]
    resends = [e for e in tape if e["event"] == "rl_pull_resend"]
    assert [(e["global_step"], e["round_attempt"], e["pulls_received"]) for e in resends] == [
        (2, 1, 2)]
    base_tape = (tmp_path / "base" / "bridge-0.jsonl").read_text()
    assert "rl_pull_resend" not in base_tape  # default path: no extra event without a resend
    assert ctl.status("up")["phase"] == SUCCEEDED and ctl.status("down")["phase"] == SUCCEEDED
    # samples / steps / policy / roster unchanged vs the no-switch fleet
    assert trained0 == b_trained0 and trained1 == b_trained1
    assert syncer.history == b_syncer.history == list(range(rounds + 1))
    assert sorted(pushes) == sorted(b_pushes)  # same learners, steps, attempts, bases
    assert len(pushes) == 2 * rounds and set(syncer.clients) == {0, 1}
    assert results[0].policy_tensor_hash() == b_results[0].policy_tensor_hash()
    assert results[1].policy_tensor_hash() == results[0].policy_tensor_hash()
    # the pause was decided at the pausable phase with the strict budget
    decisions = [r for r in _journal(tmp_path / "x") if r["kind"] == "pause_decision"]
    assert [d["outer_phase"] for d in decisions] == [PAUSABLE_PHASE] * 2
    assert all(d["allowed"] and d["stalls_peers"] and d["budget_s"] == 450.0 for d in decisions)


def test_request_during_last_round_is_cancelled_by_finalization(tmp_path):
    rounds = 2
    syncer, _, _, island0, (d1, _) = _fleet(tmp_path, rounds=rounds)
    d0, ctl, *_ = island0
    engine_train = d0.trainer.train_step
    seen = []

    def train_step(batch):
        if not seen and d0.rounds_completed == rounds - 1:  # inside the final round
            seen.append(ctl.request("late", "T4R4S0", 0, 60))
        return engine_train(batch)

    d0.trainer.train_step = train_step
    _, errors = _run([d0, d1])
    assert errors == {} and seen
    status = ctl.status("late")
    assert status["phase"] == CANCELLED
    assert "finalization" in (status.get("error") or json.dumps(status))
    with pytest.raises(Rejected, match="finalization"):
        ctl.request("after", "T4R4S0", 0, 60)
    kinds = [r["kind"] for r in _journal(tmp_path)]
    assert "finalization" in kinds
    events = [json.loads(x) for x in (tmp_path / "i0" / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "rl_reconfiguration" and e["result"] == "CANCELLED"
               and e.get("request_id") == "late" for e in events)
    assert d0.config_epoch == 0  # no switch happened


class _Client:
    def __init__(self):
        self.finalizing = threading.Event()


def test_strict_outer_phase_states():
    sync = StrictAvgSync.__new__(StrictAvgSync)
    sync.bridge = type("B", (), {"client": _Client()})()
    sync.current, sync.permit = object(), object()
    assert sync.outer_phase(None, rollout_id=0) == PAUSABLE_PHASE
    sync.permit = None
    assert sync.outer_phase(None, rollout_id=0) == "stop-round"
    sync.current = None
    assert sync.outer_phase(None, rollout_id=0) == "in-boundary"
    sync.bridge.client.finalizing.set()
    assert sync.outer_phase(None, rollout_id=0) == "finalizing"


def test_safe_point_in_finalizing_phase_cancels_instead_of_pausing(tmp_path):
    driver, ctl, fork, *_ = _setup(tmp_path, outer="strict-avg")
    ctl.request("r", "T4R4S0", 0, 60)
    driver.sync.outer_phase = lambda _d, *, rollout_id: "finalizing"
    orig = driver.safe_point
    driver.safe_point = lambda rid: orig(rid)
    driver.run()
    assert ctl.status("r")["phase"] == CANCELLED
    assert not [c for c in fork.calls if c[0] in ("start", "stop", "drain")]
    decision = [r for r in read_journal(tmp_path / "state" / "reconfig")
                if r["kind"] == "pause_decision"][0]
    assert decision["outer_phase"] == "finalizing" and not decision["allowed"]


@pytest.mark.parametrize("kw, deadline, ok, budget", [
    ({}, 450, True, 450.0),
    ({}, 451, False, 450.0),
    ({"quorum_timeout_s": 120.0}, 61, False, 60.0),
    ({"quorum_timeout_s": 120.0, "idle_flow_timeout_s": 30.0}, 31, False, 30.0),
    ({"quorum_timeout_s": 120.0, "pause_margin": 2.0}, 200, True, 240.0),
])
def test_strict_pause_budget_inputs(tmp_path, kw, deadline, ok, budget):
    ctl = IslandController(state_dir=tmp_path, configs=CONFIGS, initial_config="T4R2S2",
                           attestation=_attestation(), profile=_profile(outer="strict-avg"),
                           runtime_fingerprint=FP, **kw)
    if ok:
        assert ctl.plan("T4R4S0", 0, deadline_s=deadline).pause_budget_s == budget
    else:
        with pytest.raises(Rejected, match="exceeds budget"):
            ctl.plan("T4R4S0", 0, deadline_s=deadline)


def test_pause_inputs_reach_the_controller_through_the_wiring(tmp_path):
    import json as _json

    from yeto.rl.adapters.miles.elastic_wiring import build_elastic

    res = tmp_path / "res.json"
    res.write_text(_json.dumps({"configs": {"T4R2S2": {"trainer": 4, "rollout": 2, "standby": 2}},
                                "edges": []}))
    w = build_elastic(state_dir=tmp_path / "s", resources=res, attestation=None,
                      profile=_profile(outer="strict-avg"), initial_config="T4R2S2",
                      runtime_fingerprint=FP, declared_cells=("c0",), quorum_timeout_s=120,
                      idle_flow_timeout_s=50, pause_margin=0.25)
    c = w.controller
    assert (c.quorum_timeout_s, c.idle_flow_timeout_s, c.pause_margin) == (120.0, 50.0, 0.25)
    c.close()
