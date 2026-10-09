"""rl-infra-spec 3.7 restart recovery (E1-D §7 ruling (b); design in
evidence/infra-e1/recovery-design.md) -- CPU fault tests F1-F13.

After a learner restart the fork comes back at its startup shape; the
controller rebuilds the committed membership (epochs.json) at ``open()``,
the driver's first full publication covers the rebuilt members and
``confirm_recovery`` verifies before admission reopens. Every failure is
RECOVERY_REQUIRED and nothing is consumed. The fakes below follow the real
fork more closely than test_rl_reconfig_e1's where it matters here: a
started cell is a member (``Running`` = PendingWeights | Serving) before it
is published, and a full publication marks every member serving. CPU
stand-ins only: the GPU acceptance of E1-D ⑤⑥⑦ is not claimed here.
"""
from __future__ import annotations

import json

import pytest
import torch
from test_rl_reconfig_e1 import (
    CONFIGS,
    FP,
    MODES,
    NAME,
    ElasticFakePlacement,
    ElasticFakePool,
    ElasticFakePublisher,
    ForkMembership,
    _attestation,
    _profile,
)

from yeto.rl.elastic_benchmark.capabilities import ResourceConfig
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.controller import (
    CANCELLED,
    REBUILT_OLD,
    RECOVERY_REQUIRED,
    SUCCEEDED,
    IslandController,
    RecoveryRequired,
    Rejected,
    Timeouts,
)
from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.journal import EpochState, read_epochs, read_journal
from yeto.rl.engine.ledger import BatchLedger

C = [f"engine:c{i}" for i in range(4)]
TWO, FOUR = frozenset(C[:2]), frozenset(C)


class Fork(ForkMembership):
    """Members = every running cell (fork: PendingWeights and Serving are ``Running``)."""

    def __init__(self, engine, running=("engine:c0", "engine:c1"), unbound=()):
        super().__init__(engine, declared=C, running=running)
        self.unbound = set(unbound)
        self.serving = set(running)

    def sync(self):
        self.engine.members_ids = tuple(sorted(self.running))

    def change(self, op, members, expected):
        if op == "start" and set(members) & self.unbound:
            raise RuntimeError("CellUnboundError")
        epoch = super().change(op, members, expected)
        if op == "stop":
            self.serving -= set(members)
        return epoch


class Pool(ElasticFakePool):
    def member_states(self):
        f = self.fork
        return {m: {"state": "unbound" if m in f.unbound else "running" if m in f.running else "stopped",
                    "tracked": m in f.running, "serving": m in f.serving,
                    "awaiting_admission": m in f.awaiting_admission} for m in f.declared}


class Publisher(ElasticFakePublisher):
    """A full publication reaches every member (upstream update_weights) and marks
    it serving; ``verify_serving_policy`` is the fork's start_commit_weight_version."""

    def publish(self, state):
        result = super().publish(state)
        f = self.fork
        f.awaiting_admission -= set(result.members)
        f.serving |= set(result.members)
        f.sync()
        return result

    def verify_serving_policy(self, *, epoch, token_rollout_id, state):
        f = self.fork
        if epoch != f.epoch:
            raise RuntimeError("MembershipEpochMismatchError")
        want = f"{token_rollout_id}:{state.policy_tensor_hash()}"
        stale = {m: f.versions.get(m) for m in f.serving if f.versions.get(m) != want}
        if stale:
            raise RuntimeError(f"serving engines {stale} do not report weight version {want}")


def _configs():
    cfg = dict(CONFIGS)
    cfg["T4R4S0"] = ResourceConfig("T4R4S0", 4, 4, 0, placement={"rollout": ["g4", "g5", "g6", "g7"]})
    cfg["T4R2S2"] = ResourceConfig("T4R2S2", 4, 2, 2, placement={"rollout": ["g4", "g5"]})
    return cfg


def _ctl(state, clock, **kw):
    kw.setdefault("attestation", _attestation())
    kw.setdefault("profile", _profile())
    kw.setdefault("configs", _configs())
    kw.setdefault("initial_config", "T4R2S2")
    return IslandController(state_dir=state, runtime_fingerprint=FP,
                            wall_clock=lambda: clock["t"],
                            sleep=lambda s: clock.__setitem__("t", clock["t"] + s), **kw)


def _island(tmp, *, running=("engine:c0", "engine:c1"), unbound=(), clock=None, ledger=None,
            rounds=3, controller_kw=None):
    """A fresh learner process: new fork at its startup shape, controller on the
    shared state dir, driver."""
    clock = clock or {"t": 1000.0}
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                        placement_kind="fixed-partition")
    fork = Fork(engine, running=running, unbound=unbound)
    pool, publisher = Pool(engine, fork), Publisher(engine, fork)
    ctl = _ctl(tmp / "state", clock, **(controller_kw or {}))
    ctl.open(pool)
    driver = IslandDriver(
        learner_id=0, rollout=pool, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=publisher, placement=ElasticFakePlacement(), algorithm=AlgorithmSpec(),
        sync=LocalOnlySync(rounds), events=EventTape(tmp / "events.jsonl", 0),
        capabilities=fake_capabilities(execution_modes=MODES), profile=_profile(),
        controller=ctl, ledger=ledger,
    )
    return driver, ctl, fork, pool, publisher, engine


def _commit_four(tmp, clock=None):
    """Startup T4R2S2 (c0,c1), then an up transaction committed (T4R4S0, 4 members)."""
    clock = clock or {"t": 1000.0}
    driver, ctl, fork, *_ = _island(tmp, clock=clock, ledger=BatchLedger(tmp / "state"))
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("up", "T4R4S0", 0, 60)
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    assert ctl.status("up")["phase"] == SUCCEEDED
    e = read_epochs(tmp / "state/reconfig")
    assert (e.config_epoch, e.config_id, set(e.members), e.fork_membership_epoch) == (1, "T4R4S0", FOUR, 1)
    ctl.close()
    return clock


def _recoveries(tmp):
    return [(r["tx_id"], r["status"]) for r in read_journal(tmp / "state/reconfig") if r["kind"] == "recovery"]


def _events(tmp):
    return [json.loads(x) for x in (tmp / "events.jsonl").read_text().splitlines()]


class _Pub:
    def __init__(self, fail=None):
        self.fail = fail

    def verify_serving_policy(self, **_):
        if self.fail:
            raise RuntimeError(self.fail)


class _State:
    def policy_tensor_hash(self):
        return "h"


class _Driver:
    """Enough of IslandDriver for confirm_recovery after a full publication (which
    marks every member serving: fork end_update_weights)."""

    def __init__(self, pool, publisher=None, ledger=None, trainer=None, version=0, published=True):
        self.rollout, self.publisher, self.ledger, self.trainer = pool, publisher or _Pub(), ledger, trainer
        self.published_state, self.published_version = _State(), version
        if published:
            f = pool.fork
            f.serving |= set(f.running)
            f.awaiting_admission.clear()


# ------------------------------------------------------------- ⑤ / general recovery
def test_restart_after_commit_recovers_committed_members(tmp_path):
    """E1-D ⑤ in general (the committed config differs from the startup shape):
    the restarted learner rebuilds the 4 committed members, publishes to all of
    them, verifies and trains on."""
    clock = _commit_four(tmp_path)
    driver, ctl, fork, pool, publisher, engine = _island(tmp_path, clock=clock)
    assert ctl.inspect().health == "RECOVERING" and not ctl.admission_open
    assert [c for c in fork.calls if c[0] in ("restore", "start", "stop")] == [
        ("restore", 1, None, ["start", ["engine:c2", "engine:c3"]], 0),
        ("start", ("engine:c2", "engine:c3"), 1)]
    assert pool.members() == FOUR
    e = read_epochs(tmp_path / "state/reconfig")
    assert (e.config_epoch, e.config_id, e.fork_membership_epoch, set(e.members)) == (1, "T4R4S0", 2, FOUR)
    with pytest.raises(Rejected, match="recovering"):
        ctl.request("late", "T4R2S2", 1, 60)  # F11: nothing runs during a recovery
    driver.run()
    assert ctl.inspect().health == "RUNNING" and ctl.admission_open
    rid = _recoveries(tmp_path)[0][0]
    assert _recoveries(tmp_path) == [(rid, "planned"), (rid, "membership_restored"), (rid, "verified")]
    verified = [r for r in read_journal(tmp_path / "state/reconfig") if r.get("status") == "verified"][0]
    assert verified["checks"]["policy_token"] == "verified"
    assert verified["checks"]["router"] == {"not_admitted": []}
    assert verified["members"] == sorted(FOUR)
    # the first publication reached all four and every generation used them
    pubs = [c for c in engine.calls if c[0] == "publish"]
    assert pubs and engine.calls.index(("generate", 0)) > engine.calls.index(pubs[0])
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert [e["result"] for e in ev] == [SUCCEEDED, "RECOVERED"]
    assert ev[1]["members"] == sorted(FOUR) and ev[1]["recovery_id"] == rid
    # S14/A19 (FINAL-REPORT-S7 §6.1): the recovered epoch's membership view is on
    # this incarnation's tape too, with the same fields a COMMITTED tx writes.
    mem = [e for e in _events(tmp_path) if e["event"] == "rl_membership"]
    assert [m["kind"] for m in mem] == ["up", "recovered"]  # "up" is the first incarnation's
    rec = mem[1]
    assert rec["config_epoch"] == e.config_epoch == 1
    assert rec["members"] == sorted(FOUR)
    assert rec["tx_id"] == e.last_tx_id == mem[0]["tx_id"]
    assert rec["kind"] == "recovered" and rec["round"] == 0 and rec["recovery_id"] == rid
    # the tape's rl_membership follows RECOVERED (a consumer reading in order sees
    # the reconfiguration result before the membership it re-serves)
    order = [x["event"] for x in _events(tmp_path) if x["event"] in ("rl_reconfiguration", "rl_membership")]
    assert order[-2:] == ["rl_reconfiguration", "rl_membership"]
    assert ctl.status("up")["phase"] == SUCCEEDED
    ctl.close()


def test_committed_equals_startup_shape_needs_no_recovery(tmp_path):
    """Ruling (c) kept as a regression: up then down committed -> restart finds the
    startup shape and journals no recovery (behaviour of 3be4933)."""
    clock = {"t": 1000.0}
    driver, ctl, *_ = _island(tmp_path, clock=clock, ledger=BatchLedger(tmp_path / "state"), rounds=4)
    orig = driver.safe_point
    reqs = {1: ("up", "T4R4S0", 0), 2: ("down", "T4R2S2", 1)}

    def safe_point(rollout_id):
        if rollout_id in reqs:
            ctl.request(*reqs[rollout_id], 60)
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    ctl.close()
    driver, ctl, fork, pool, *_ = _island(tmp_path, clock=clock)
    assert ctl.inspect().health == "RUNNING" and ctl.admission_open
    assert _recoveries(tmp_path) == []
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == []
    driver.run()
    ctl.close()


def test_restart_in_quiescing_is_cancelled_without_recovery(tmp_path):
    """E1-D ⑥: killed before release -> CANCELLED; members already match."""
    clock = {"t": 1000.0}
    _, ctl, *_ = _island(tmp_path, clock=clock)
    ctl.request("r", "T4R4S0", 0, 60)
    ctl._phase(ctl._tx, "QUIESCING")
    ctl.close()
    _, ctl, fork, *_ = _island(tmp_path, clock=clock)
    assert ctl.status("r")["phase"] == CANCELLED
    assert _recoveries(tmp_path) == [] and ctl.inspect().health == "RUNNING"
    ctl.close()


def test_epoch0_member_loss_is_recovered(tmp_path):
    """E1-D ⑦ shape: no transaction yet, the fork restarted at epoch 0 and one of
    the two committed members is missing -> it is started again."""
    clock = {"t": 1000.0}
    _, ctl, *_ = _island(tmp_path, clock=clock)
    assert set(read_epochs(tmp_path / "state/reconfig").members) == TWO
    ctl.close()
    _, ctl, fork, pool, *_ = _island(tmp_path, clock=clock, running=("engine:c0",))
    assert [c for c in fork.calls if c[0] in ("restore", "start")] == [("start", ("engine:c1",), 0)]
    assert pool.members() == TWO and ctl.inspect().health == "RECOVERING"
    ctl.confirm_recovery(_Driver(pool))
    assert ctl.admission_open and _recoveries(tmp_path)[-1][1] == "verified"
    ctl.close()


def test_restart_after_release_rebuilds_old_members(tmp_path):
    """D1: a down released (stop done) and killed before its CAS -> the restart
    rebuilds the committed (old, 4) members and ends the transaction REBUILT_OLD."""
    clock = _commit_four(tmp_path)
    _, ctl, fork, pool, *_ = _island(tmp_path, clock=clock, running=C)
    assert _recoveries(tmp_path) == []
    ctl.request("down", "T4R2S2", 1, 60)
    tx = ctl._tx
    ctl._pool = pool
    ctl._phase(tx, "TRANSFERRING", rollback_boundary="release", release=C[2:])
    ctl._fork_call(tx, "stop", frozenset(C[2:]), lambda e: pool.remove_engines(frozenset(C[2:]), epoch=e))
    assert pool.members() == TWO
    ctl.close()
    _, ctl, fork, pool, *_ = _island(tmp_path, clock=clock)
    assert pool.members() == FOUR
    assert ctl.status("down")["phase"] == REBUILT_OLD
    phase = [r for r in read_journal(tmp_path / "state/reconfig")
             if r["kind"] == "phase" and r["phase"] == REBUILT_OLD][-1]
    assert phase["recovered_after_restart"] is True
    kinds = [(r["kind"], r.get("status") or r.get("phase")) for r in read_journal(tmp_path / "state/reconfig")]
    assert kinds.index(("recovery", "membership_restored")) < kinds.index(("phase", REBUILT_OLD))
    ctl.confirm_recovery(_Driver(pool, version=0))
    assert ctl.inspect().health == "RUNNING"
    ctl.close()


def test_restart_after_release_of_trainer_edge_stays_recovery_required(tmp_path):
    """D1 limit: a trainer edge released before its CAS is not recovered (hint only)."""
    clock = {"t": 1000.0}
    _, ctl, *_ = _island(tmp_path, clock=clock)
    ctl.journal.append("request", request_id="rt", tx_id="tx-rt", body={"target": "T2R6S0", "kind": "role-transfer",
                       "expected_config_epoch": 0, "deadline_s": 60}, body_hash="x", plan={}, deadline_wall=0)
    ctl.journal.append("phase", tx_id="tx-rt", request_id="rt", phase="TRANSFERRING", config_epoch=0, fork_epoch=0)
    ctl.close()
    _, ctl, *_ = _island(tmp_path, clock=clock)
    assert ctl.status("rt")["phase"] == RECOVERY_REQUIRED
    assert _recoveries(tmp_path) == []
    ctl.close()


# ------------------------------------------------------------- F1 / F2: crash during recovery, budget
def test_recovery_crash_before_verification_is_idempotent(tmp_path):
    """F1: the learner dies after the membership was restored but before the
    verification -> the next restart supersedes it and recovers again."""
    clock = _commit_four(tmp_path)
    _, ctl, *_ = _island(tmp_path, clock=clock)  # recovery 1: restored, never confirmed
    ctl.close()
    _, ctl, fork, pool, *_ = _island(tmp_path, clock=clock)
    r = _recoveries(tmp_path)
    assert [s for _, s in r] == ["planned", "membership_restored", "superseded", "planned", "membership_restored"]
    assert r[0][0] == r[2][0] and r[3][0] != r[0][0]
    assert fork.calls[-1] == ("start", ("engine:c2", "engine:c3"), 2)  # journal fork epoch moved on
    assert read_epochs(tmp_path / "state/reconfig").fork_membership_epoch == 3
    ctl.confirm_recovery(_Driver(pool))
    assert _recoveries(tmp_path)[-1][1] == "verified" and ctl._recovery_unverified == 0
    ctl.close()


def test_recovery_budget_spent(tmp_path):
    """F2: three consecutive unverified recoveries -> the fourth restart is RECOVERY_REQUIRED."""
    clock = _commit_four(tmp_path)
    for _ in range(3):
        _, ctl, *_ = _island(tmp_path, clock=clock, controller_kw={"max_recovery_attempts": 3})
        assert ctl.inspect().health == "RECOVERING"
        ctl.close()
    _, ctl, fork, pool, *_ = _island(tmp_path, clock=clock, controller_kw={"max_recovery_attempts": 3})
    assert ctl.inspect().health == RECOVERY_REQUIRED and "recovery budget spent" in ctl.recovery_required
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == []
    assert pool.members() == TWO
    ctl.close()


# ------------------------------------------------------------- F3-F6: fork failures, timeout
def test_start_rolled_back_enters_recovery_required(tmp_path):
    """F3: no resources (the worker manager rolled the start back) -> RECOVERY_REQUIRED."""
    clock = _commit_four(tmp_path)
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, placement_kind="fixed-partition")
    fork = Fork(engine)
    fork.start_fail = "rolled_back"
    ctl = _ctl(tmp_path / "state", clock)
    ctl.open(Pool(engine, fork))
    assert ctl.inspect().health == RECOVERY_REQUIRED
    assert "membership recovery failed" in ctl.recovery_required
    ops = [r for r in read_journal(tmp_path / "state/reconfig") if r["kind"] == "fork_op"]
    assert ops[-1]["status"] == "failed" and ops[-1]["tx_id"].startswith("rec-")
    assert _recoveries(tmp_path)[-1][1] == "failed"
    ctl.close()


def test_half_failed_ops_are_retried_then_recovery_goes_on(tmp_path):
    """F4/F5: a start whose rollback failed (fork incomplete = stop them) and a
    half-failed stop are retried inside the recovery budget."""
    clock = _commit_four(tmp_path)
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, placement_kind="fixed-partition")
    fork = Fork(engine)
    fork.start_fail = "rollback_failed"
    ctl = _ctl(tmp_path / "state", clock)
    pool = Pool(engine, fork)
    ctl.open(pool)
    assert ctl.inspect().health == "RECOVERING" and pool.members() == FOUR
    ops = [(r["op"], r["status"]) for r in read_journal(tmp_path / "state/reconfig")
           if r["kind"] == "fork_op" and r["tx_id"].startswith("rec-")]
    assert ops == [("start", "issued"), ("start", "incomplete"), ("stop", "issued"), ("stop", "done"),
                   ("start", "issued"), ("start", "done")]
    ctl.close()
    # a down committed in between: the recovery stops the extra cells; the stop half fails once
    _, ctl, *_ = _island(tmp_path / "b", clock=clock, running=C)
    ctl.journal.compare_and_swap(expected_config_epoch=0, new=EpochState(1, "T4R2S2", 0, tuple(TWO), "tx-d"))
    ctl.close()
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, placement_kind="fixed-partition")
    fork = Fork(engine, running=C)
    fork.stop_failures = 1
    ctl = _ctl(tmp_path / "b/state", clock)
    pool = Pool(engine, fork)
    ctl.open(pool)
    assert pool.members() == TWO and ctl.inspect().health == "RECOVERING"
    ops = [(r["op"], r["status"]) for r in read_journal(tmp_path / "b/state/reconfig")
           if r["kind"] == "fork_op" and r["tx_id"].startswith("rec-")]
    assert ops == [("stop", "issued"), ("stop", "incomplete"), ("stop", "issued"), ("stop", "done")]
    ctl.close()


def test_recovery_timeout(tmp_path):
    """F6: the fork start outlives T_recovery -> RECOVERY_REQUIRED (nothing released)."""
    clock = _commit_four(tmp_path)
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, placement_kind="fixed-partition")
    fork = Fork(engine)
    pool = Pool(engine, fork)
    real = pool.add_engines

    def slow(*a, **k):
        clock["t"] += 50.0
        return real(*a, **k)

    pool.add_engines = slow
    ctl = _ctl(tmp_path / "state", clock, timeouts=Timeouts(recovery=30.0))
    ctl.open(pool)
    assert ctl.inspect().health == RECOVERY_REQUIRED and "deadline" in ctl.recovery_required
    assert _recoveries(tmp_path)[-1][1] == "failed"
    ctl.close()
    # and a verification that comes after the deadline is refused too
    clock2 = _commit_four(tmp_path / "v")
    _, ctl, _, pool, *_ = _island(tmp_path / "v", clock=clock2, controller_kw={"timeouts": Timeouts(recovery=30.0)})
    clock2["t"] += 31.0
    with pytest.raises(RecoveryRequired, match="T_recovery"):
        ctl.confirm_recovery(_Driver(pool))
    assert ctl.inspect().health == RECOVERY_REQUIRED
    ctl.close()


# ------------------------------------------------------------- F9: verification
@pytest.mark.parametrize("case", ["token", "router", "members", "ledger", "layout", "fork_epoch"])
def test_verification_failures_are_recovery_required(tmp_path, case):
    clock = _commit_four(tmp_path)
    _, ctl, fork, pool, *_ = _island(tmp_path, clock=clock)
    kw = {}
    if case == "token":
        kw["publisher"] = _Pub(fail="serving engines {'engine:c2': 'old'} do not report weight version")
    elif case == "router":
        kw["published"] = False  # c2/c3 never marked serving (no publication reached them)
    elif case == "members":
        fork.running.discard("engine:c3")
        fork.sync()
    elif case == "ledger":
        led = BatchLedger(tmp_path / "ledger")

        class B:  # a batch prepared and never consumed
            rollout_id, policy_version, policy_hash, group_ids = 7, 7, "h", ("g",)

            @property
            def groups(self):
                return ()
        led.prepare(B(), policy_token="7:h")
        kw["ledger"] = led
    elif case == "layout":
        class T:
            def actual_layout(self):
                return {"world": 4, "tp": 2, "pp": 1, "cp": 1, "ep": 1, "dp": 2}
        kw["trainer"] = T()
    elif case == "fork_epoch":
        fork.epoch += 1
    with pytest.raises(RecoveryRequired):
        ctl.confirm_recovery(_Driver(pool, **kw))
    assert ctl.inspect().health == RECOVERY_REQUIRED and not ctl.admission_open
    failed = [r for r in read_journal(tmp_path / "state/reconfig") if r.get("status") == "failed"][-1]
    assert failed["errors"] and {"token": "policy", "router": "router", "members": "members",
                                 "ledger": "unconsumed", "layout": "layout",
                                 "fork_epoch": "disagrees"}[case] in " ".join(failed["errors"])
    with pytest.raises(Rejected, match="RECOVERY_REQUIRED"):
        ctl.request("x", "T4R2S2", 1, 60)
    ctl.close()


def test_verification_passes_with_layout_and_ledger(tmp_path):
    clock = _commit_four(tmp_path)
    _, ctl, _, pool, *_ = _island(tmp_path, clock=clock)

    class T:
        def actual_layout(self):
            return {"world": 4, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 4}

    ctl.confirm_recovery(_Driver(pool, ledger=BatchLedger(tmp_path / "ledger"), trainer=T()))
    checks = [r for r in read_journal(tmp_path / "state/reconfig") if r.get("status") == "verified"][-1]["checks"]
    assert checks["trainer_layout"]["world"] == 4 and checks["unconsumed_batches"] == []
    ctl.close()


# ------------------------------------------------------------- F10 / F11 / F13
def test_unrecoverable_committed_config_falls_back(tmp_path):
    """F10: a committed trainer shape other than the startup one, or a committed
    member bound to no bundle, is not recovered by cell start/stop."""
    clock = {"t": 1000.0}
    _, ctl, *_ = _island(tmp_path / "t", clock=clock)
    ctl.journal.compare_and_swap(expected_config_epoch=0, new=EpochState(1, "T2R6S0", 0, tuple(C), "tx-rt"))
    ctl.close()
    _, ctl, fork, *_ = _island(tmp_path / "t", clock=clock)
    assert ctl.inspect().health == RECOVERY_REQUIRED and "trainer shape" in ctl.recovery_required
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == [] and _recoveries(tmp_path / "t") == []
    ctl.close()
    _commit_four(tmp_path / "u")
    _, ctl, fork, *_ = _island(tmp_path / "u", clock={"t": 2000.0}, unbound=("engine:c3",))
    assert ctl.inspect().health == RECOVERY_REQUIRED and "bound to no bundle" in ctl.recovery_required
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == []
    ctl.close()


def test_old_epoch_and_unconfirmed_safe_point_are_fenced(tmp_path):
    """F11: stale-epoch requests stay refused; a safe point reached while the
    recovery is unconfirmed closes the island instead of running anything."""
    clock = _commit_four(tmp_path)
    driver, ctl, _, pool, *_ = _island(tmp_path, clock=clock)
    with pytest.raises(Rejected, match="recovering"):
        ctl.request("old", "T4R2S2", 0, 60)
    assert ctl.has_pending()
    with pytest.raises(RecoveryRequired, match="before the restart recovery was confirmed"):
        ctl.run_at_safe_point(driver, driver.safe_point_snapshot(0))
    assert ctl.inspect().health == RECOVERY_REQUIRED
    ctl.close()


def test_driver_fails_fast_when_recovery_required_at_open(tmp_path):
    """F13: RECOVERY_REQUIRED at open -> no publication, no generation, no ledger entry."""
    clock = _commit_four(tmp_path)
    led = BatchLedger(tmp_path / "ledger")
    driver, ctl, fork, pool, publisher, engine = _island(
        tmp_path, clock=clock, controller_kw={"max_recovery_attempts": 1}, ledger=led)
    assert ctl.inspect().health == "RECOVERING"
    ctl.close()
    driver, ctl, fork, pool, publisher, engine = _island(
        tmp_path, clock=clock, controller_kw={"max_recovery_attempts": 1}, ledger=led)
    assert ctl.inspect().health == RECOVERY_REQUIRED
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        driver.run()
    assert not [c for c in engine.calls if c[0] in ("publish", "generate")]
    assert led.unconsumed() == [] and not any(r.get("kind") == "prepared" for r in led._journal.records)
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert ev[-1]["result"] == RECOVERY_REQUIRED
    ctl.close()


def test_driver_run_ends_in_driver_error_when_verification_fails(tmp_path):
    clock = _commit_four(tmp_path)
    driver, ctl, fork, pool, publisher, engine = _island(tmp_path, clock=clock)
    real = publisher.publish

    def publish(state):
        result = real(state)
        fork.versions["engine:c2"] = "stale"  # one rebuilt engine serves another policy
        return result

    publisher.publish = publish
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        driver.run()
    assert not [c for c in engine.calls if c[0] == "generate"]
    assert _recoveries(tmp_path)[-1][1] == "failed"
    ctl.close()


# ------------------------------------------------------------- adapter: member_states
def test_miles_pool_member_states_from_describe_cells():
    from yeto.rl.adapters.miles.rollout import MilesRolloutPool

    class Ctl:
        async def describe_cells(self):
            return {"c0": {"state": "running", "tracked": True, "serving": True, "awaiting_admission": False},
                    "c1": {"state": "unbound", "tracked": False, "serving": False, "awaiting_admission": False}}

    pool = MilesRolloutPool(inference_controller=Ctl(), rollout_executor=object(), metadata=None,
                            expected_policy=lambda: (1, "h"), declared_cells=["c0", "c1"])
    assert pool.member_states() == {
        "engine:c0": {"state": "running", "tracked": True, "serving": True, "awaiting_admission": False},
        "engine:c1": {"state": "unbound", "tracked": False, "serving": False, "awaiting_admission": False}}
    pool2 = MilesRolloutPool(inference_controller=object(), rollout_executor=object(), metadata=None,
                             expected_policy=lambda: (1, "h"), declared_cells=["c0"])
    assert pool2.member_states() is None


# ------------------------------------------------------------- no-sync island restart (blocker found in phase 3)
def test_no_sync_restart_hits_the_ledger_and_the_restart_loop_is_bounded(tmp_path):
    """A ``--rl-single-island-no-sync`` island restarts through LocalOnlySync, i.e. at rollout 0 with a freshly
    initialised policy: with the persisted ledger the restarted driver fails in ledger.rebase(0) (LedgerError:
    outer-recorded rollouts behind the restart point) before publishing or generating -- every restart, so E1-D
    ⑤⑥⑦ cannot run under no-sync (recovery-design.md; ruling: strict-avg + syncer). The learner's restart loop
    (launcher RESTART_LOOP_FN) only bounds the retries: it returns the last exit code after N attempts and journals
    no RECOVERY_REQUIRED (recorded here, not changed)."""
    import subprocess
    from yeto.launcher import RESTART_LOOP_FN
    from yeto.rl.engine.ledger import LedgerError

    clock = {"t": 1000.0}
    driver, ctl, *_ = _island(tmp_path, clock=clock, ledger=BatchLedger(tmp_path / "state"))
    driver.run(); ctl.close(); driver.ledger.close()
    driver, ctl, fork, pool, publisher, engine = _island(tmp_path, clock=clock, ledger=BatchLedger(tmp_path / "state"))
    with pytest.raises(LedgerError, match="behind outer-recorded rollouts"):
        driver.run()
    assert not [c for c in engine.calls if c[0] in ("publish", "generate")]
    assert ctl.inspect().health == "RUNNING" and _recoveries(tmp_path) == []  # not a recovery problem: no RECOVERY_REQUIRED journaled
    ctl.close()
    script = RESTART_LOOP_FN + "n=0; yeto_rl_restart_loop sh -c 'exit 86'; echo rc=$? attempts=$(grep -c . $LOG)"
    log = tmp_path / "loop.log"
    out = subprocess.run(["bash", "-c", script], env={"YETO_RL_RESTART_ATTEMPTS": "2", "LOG": str(log), "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True)
    assert "rc=86" in out.stdout and out.stderr.count("in-place restart") == 2  # bounded: 1 + 2 attempts, then the exit code
