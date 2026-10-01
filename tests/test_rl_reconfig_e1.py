"""rl-infra-spec 3.1-3.7 (E1) CPU tests: journal, ledger, controller, driver safe point.

The fakes below model the fork's membership contract (michaellchung/miles
yeto/ports 0af62f4d: epoch CAS, idempotent repeat right after commit,
``incomplete`` after a half-failed stop, cordon/in-flight drain, cordoned
admission). They are CPU stand-ins for protocol tests only; the GPU
acceptance of 3.3/3.4/3.5/3.7 is NOT claimed by these tests (see
evidence/infra-e1/plan.md).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading

import pytest
import torch

from yeto.rl.contracts import InferencePublicationManifest
from yeto.rl.elastic_benchmark.capabilities import Attestation, ResourceConfig
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.capabilities import RESERVED_PORT_VERBS
from yeto.rl.engine.controller import (
    CANCELLED,
    RECOVERY_REQUIRED,
    REBUILT_OLD,
    SUCCEEDED,
    CommandInbox,
    IslandController,
    Rejected,
    Timeouts,
    main as controller_main,
)
from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import FakeEngine, FakePublisher, FakeRolloutPool, fake_capabilities
from yeto.rl.engine.journal import (
    EpochConflict,
    EpochState,
    Journal,
    JournalCorrupt,
    JournalLocked,
    read_epochs,
    read_journal,
)
from yeto.rl.engine.ledger import BatchLedger, LedgerError
from yeto.rl.engine.miles_adapter.elastic_placement import ElasticPlacement, PlacementPlanError
from yeto.rl.engine.ports import (
    ElasticRolloutPool,
    MemberPublisher,
    PlacementDescription,
    PublicationResult,
    ReconfigurablePlacement,
)

NAME = "base_model.model.layer.lora_A.weight"
FP = "sha256:" + "0" * 64
MODES = {"colocated-serial", "partitioned-serial"}
CONFIGS = {
    "T4R2S2": ResourceConfig("T4R2S2", 4, 2, 2),
    "T4R4S0": ResourceConfig("T4R4S0", 4, 4, 0),
    "T2R6S0": ResourceConfig("T2R6S0", 2, 6, 0),
}


def _attestation(edges=(("T4R2S2", "T4R4S0", "rollout-only"), ("T4R4S0", "T4R2S2", "rollout-only"),
                        ("T4R2S2", "T2R6S0", "role-transfer")), fingerprint=FP):
    return Attestation(fingerprint, frozenset(MODES), frozenset(edges), frozenset(), False, True)


def _profile(mode="partitioned-serial", outer="none"):
    return ExecutionProfile(name=f"t-{mode}", execution_mode=mode,
                            outer_protocol=outer).bind_algorithm(AlgorithmSpec())


# --------------------------------------------------------------------------- fakes
class ForkMembership:
    """Fork M2/M3/M4 membership semantics (inference_controller.py @0af62f4d)."""

    def __init__(self, engine: FakeEngine, declared, running) -> None:
        self.engine = engine
        self.declared = tuple(declared)
        self.running = set(running)
        self.epoch = 0
        self.last_op = None
        self.incomplete = None
        self.cordoned: set[str] = set()
        self.awaiting_admission: set[str] = set()
        self.inflight: dict[str, int] = {}
        self.versions: dict[str, str] = {}
        self.stop_failures = 0  # stop fails half way this many times
        self.start_fail = None  # None | "rolled_back" | "rollback_failed"
        self.calls: list[tuple] = []
        self.sync()

    def sync(self) -> None:
        self.engine.members_ids = tuple(sorted(
            m for m in self.running if m not in self.awaiting_admission))

    def change(self, op, members, expected):
        ids = tuple(sorted(members))
        self.calls.append((op, ids, expected))
        if self.last_op == (op, ids) and expected == self.epoch - 1:
            return self.epoch
        if self.incomplete is not None and self.incomplete != (op, ids):
            raise RuntimeError("MembershipIncompleteError")
        if expected != self.epoch:
            raise RuntimeError("MembershipEpochMismatchError")
        if op == "stop":
            self.running -= set(ids)
            self.cordoned -= set(ids)
            self.sync()
            if self.stop_failures:
                self.stop_failures -= 1
                self.incomplete = (op, ids)
                raise RuntimeError("stop failed half way")
        else:
            if self.start_fail == "rolled_back":
                self.start_fail = None
                raise RuntimeError("start failed (rolled back)")
            if self.start_fail == "rollback_failed":
                self.start_fail = None
                self.incomplete = ("stop", ids)
                raise RuntimeError("StartRollbackFailedError")
            self.running |= set(ids)
            self.awaiting_admission |= set(ids)  # pending weights: not routed
            self.sync()
        self.incomplete = None
        self.epoch += 1
        self.last_op = (op, ids)
        return self.epoch


class ElasticFakePool(FakeRolloutPool):
    def __init__(self, engine, fork: ForkMembership, *, load=None) -> None:
        super().__init__(engine)
        self.fork = fork
        self.load = load  # callable -> {"active_requests", "tool_wait"} or None
        self.drain_results: list[bool] = []

    def plan_add(self, count):
        free = [c for c in self.fork.declared if c not in self.fork.running]
        return frozenset(free[:count])

    def add_engines(self, count, *, epoch, members=None):
        chosen = frozenset(members) if members is not None else self.plan_add(count)
        self.fork.change("start", chosen, epoch)
        return chosen

    def remove_engines(self, members, *, epoch):
        self.fork.change("stop", members, epoch)
        return self.members()

    def drain(self, members, deadline):
        self.fork.calls.append(("drain", tuple(sorted(members))))
        self.fork.cordoned |= set(members)
        if self.drain_results:
            return self.drain_results.pop(0)
        return all(self.fork.inflight.get(m, 0) == 0 for m in members)

    def undrain(self, members):
        self.fork.calls.append(("undrain", tuple(sorted(members))))
        self.fork.cordoned -= set(members)

    def membership_status(self):
        inc = self.fork.incomplete
        return {"epoch": self.fork.epoch,
                "incomplete": None if inc is None else [inc[0], list(inc[1])]}

    def restore_membership(self, *, epoch, incomplete, last_op, expected_current_epoch):
        self.fork.calls.append(("restore", epoch, incomplete, last_op, expected_current_epoch))
        if expected_current_epoch != self.fork.epoch:
            raise RuntimeError("MembershipEpochMismatchError")
        self.fork.epoch = epoch
        self.fork.last_op = None if last_op is None else (last_op[0], tuple(last_op[1]))
        return self.membership_status()

    def trajectory_load(self):
        return self.load() if self.load else None


class ElasticFakePublisher(FakePublisher):
    def __init__(self, engine, fork: ForkMembership) -> None:
        super().__init__(engine)
        self.fork = fork
        self.corrupt_payload = False
        self.published_members: list[tuple] = []

    def publish(self, state):
        result = super().publish(state)
        for m in result.members:
            self.fork.versions[m] = f"{state.policy_version}:{state.policy_tensor_hash()}"
        return result

    def publish_members(self, state, members, *, epoch, token_rollout_id=None):
        f = self.fork
        self.published_members.append((tuple(sorted(members)), epoch, token_rollout_id))
        if epoch != f.epoch:
            raise RuntimeError("MembershipEpochMismatchError")
        version = f"{token_rollout_id}:{state.policy_tensor_hash()}"
        for m in members:
            f.versions[m] = "garbage" if self.corrupt_payload else version
        if self.corrupt_payload:  # read-back differs: never admitted
            raise RuntimeError("payload read-back differs")
        f.awaiting_admission -= set(members)
        f.sync()
        digest = state.policy_tensor_hash()
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None,
            target_policy_version=state.policy_version, target_policy_hash=digest,
            target_manifest_hash="a" * 64, payload_hash="b" * 64, payload_bytes=1, complete=True)
        return PublicationResult(manifest, frozenset(members))


class ElasticFakePlacement:
    def __init__(self) -> None:
        self.inner = ElasticPlacement(
            _StaticPlacement(), pool_gpus=tuple(f"g{i}" for i in range(8)))

    def describe(self):
        return self.inner.describe()

    def reconfigure(self, plan, *, epoch):
        return self.inner.reconfigure(plan, epoch=epoch)


class _StaticPlacement:
    def describe(self):
        return PlacementDescription("fixed-partition", ("g0", "g1", "g2", "g3"), ("g4", "g5"))


def _setup(tmp_path, *, rounds=4, load=None, ledger=True, controller_kw=None, inbox=False,
           driver_kw=None, sync_factory=None, outer="none", learner_id=0):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                        placement_kind="fixed-partition")
    fork = ForkMembership(engine, declared=[f"engine:c{i}" for i in range(4)],
                          running=["engine:c0", "engine:c1"])
    pool = ElasticFakePool(engine, fork, load=load)
    publisher = ElasticFakePublisher(engine, fork)
    configs = dict(CONFIGS)
    configs["T4R4S0"] = ResourceConfig("T4R4S0", 4, 4, 0,
                                       placement={"rollout": ["g4", "g5", "g6", "g7"]})
    configs["T4R2S2"] = ResourceConfig("T4R2S2", 4, 2, 2,
                                       placement={"rollout": ["g4", "g5"]})
    clock = {"t": 1000.0}
    ctl = IslandController(
        state_dir=tmp_path / "state", configs=configs, attestation=_attestation(),
        profile=_profile(outer=outer), initial_config="T4R2S2", runtime_fingerprint=FP,
        wall_clock=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        inbox=CommandInbox(tmp_path / "state" / "inbox") if inbox else None,
        **(controller_kw or {}),
    )
    ctl.open(pool)
    led = BatchLedger(tmp_path / "state") if ledger else None
    trained = []
    original = engine.trainer.train_step

    def train_step(batch):
        trained.append(tuple(s for g in batch.groups for s in g.sample_ids))
        return original(batch)

    engine.trainer.train_step = train_step
    driver = IslandDriver(
        learner_id=learner_id, rollout=pool, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=publisher, placement=ElasticFakePlacement(),
        algorithm=AlgorithmSpec(),
        sync=sync_factory(engine) if sync_factory else LocalOnlySync(rounds),
        events=EventTape(tmp_path / "events.jsonl", learner_id),
        **{"capabilities": fake_capabilities(execution_modes=MODES),
           "profile": _profile(outer=outer), "controller": ctl, "ledger": led,
           **(driver_kw or {})},
    )
    return driver, ctl, fork, pool, publisher, trained, clock


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]


# --------------------------------------------------------------------------- 3.4a
def test_reserved_port_verbs_and_optional_protocols():
    assert {"RolloutPool.add_engines", "RolloutPool.remove_engines", "RolloutPool.drain",
            "Publisher.publish_members", "Placement.reconfigure"} <= RESERVED_PORT_VERBS
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)})
    # R0 fakes do not satisfy the E1 protocols; the elastic fakes do.
    assert not isinstance(engine.rollout, ElasticRolloutPool)
    assert not isinstance(engine.publisher, MemberPublisher)
    fork = ForkMembership(engine, ["engine:c0"], ["engine:c0"])
    assert isinstance(ElasticFakePool(engine, fork), ElasticRolloutPool)
    assert isinstance(ElasticFakePublisher(engine, fork), MemberPublisher)
    assert isinstance(ElasticFakePlacement(), ReconfigurablePlacement)


def test_miles_capabilities_do_not_declare_e1_verbs_yet():
    from yeto.rl.engine.miles_adapter.entry import miles_capabilities

    caps = miles_capabilities(FP)
    assert not (set(caps.port_verbs) & RESERVED_PORT_VERBS)


# --------------------------------------------------------------------------- journal
def test_journal_append_fsync_torn_tail_and_single_writer(tmp_path):
    with Journal(tmp_path) as j:
        j.append("a", x=1)
        j.append("b", x=2)
        with pytest.raises(JournalLocked):
            Journal(tmp_path)
    with open(tmp_path / "journal.jsonl", "ab") as fh:
        fh.write(b'{"seq": 3, "kind": "c"')  # crash mid-write
    assert [r["kind"] for r in read_journal(tmp_path)] == ["a", "b"]
    with Journal(tmp_path) as j:
        assert j.append("d")["seq"] == 3
    assert [r["kind"] for r in read_journal(tmp_path)] == ["a", "b", "d"]
    lines = (tmp_path / "journal.jsonl").read_bytes().splitlines()
    lines.insert(1, b"not json")
    (tmp_path / "journal.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(JournalCorrupt):
        read_journal(tmp_path)


def test_journal_compare_and_swap(tmp_path):
    with Journal(tmp_path) as j:
        j.compare_and_swap(expected_config_epoch=0, new=EpochState(0, "A"))
        with pytest.raises(EpochConflict):
            j.compare_and_swap(expected_config_epoch=1, new=EpochState(2, "B"))
        j.compare_and_swap(expected_config_epoch=0, expected_fork_epoch=0,
                           new=EpochState(1, "B", 2, ("engine:c0",), "tx"))
        with pytest.raises(EpochConflict):
            j.compare_and_swap(expected_config_epoch=1, new=EpochState(0, "A"))
    assert read_epochs(tmp_path) == EpochState(1, "B", 2, ("engine:c0",), "tx")
    assert not (tmp_path / "epochs.json.tmp").exists()


# --------------------------------------------------------------------------- ledger (3.6)
class _G:
    def __init__(self, gid, samples=2, filtered=None):
        self.group_id = gid
        self.sample_ids = tuple(f"s{i}" for i in range(samples))
        self.filtered_samples = filtered


class _B:
    def __init__(self, rid, gids, filtered=None, carried=None):
        self.rollout_id = rid
        self.groups = tuple(_G(g) for g in gids)
        self.filtered = filtered
        self.carried_over = carried


def test_ledger_states_no_double_training_and_restart(tmp_path):
    led = BatchLedger(tmp_path)
    led.prepare(_B(0, ["g0", "g1"], filtered=3), policy_token="t0")
    with pytest.raises(LedgerError):
        led.outer_recorded(0)  # may not skip optimizer_applied
    led.optimizer_applied(0)
    led.optimizer_applied(0)  # idempotent after a lost acknowledgement
    with pytest.raises(LedgerError):
        led.prepare(_B(1, ["g1", "g2"]), policy_token="t1")  # g1 already trained
    with pytest.raises(LedgerError):
        led.prepare(_B(0, ["g5"]), policy_token="t0")  # rollout 0 is past prepared
    led.outer_recorded(0)
    led.prepare(_B(1, ["g2"]), policy_token="t1")
    led.discard(1, error="train gate refused")
    led.prepare(_B(1, ["g2"]), policy_token="t1")  # retried after an explicit discard
    assert led.batch(1)["attempt"] == 1
    led.optimizer_applied(1)
    led.close()
    # restart: the durable ledger remembers; restart behind rollout 1's update supersedes it
    led = BatchLedger(tmp_path)
    assert led.state(0) == "outer_recorded" and led.state(1) == "optimizer_applied"
    assert led.batch(0)["filtered"]["groups"] == 3
    with pytest.raises(LedgerError):
        led.rebase(0)  # behind an outer-recorded batch: refused
    assert led.rebase(1) == [1]
    led.prepare(_B(1, ["g2"]), policy_token="t1b")  # may be trained again
    led.close()


def test_ledger_carried_over_must_be_consumed_or_filtered(tmp_path):
    led = BatchLedger(tmp_path)
    led.carried_over(0, ["g9", "g8"], policy_token="t0")
    assert set(led.open_carried_over()) == {"g8", "g9"}
    led.prepare(_B(1, ["g9"]), policy_token="t1")
    assert set(led.open_carried_over()) == {"g8"}
    led.filter_carried(["g8"], reason="stale", mechanism="over_sampling")
    assert led.open_carried_over() == {}
    with pytest.raises(LedgerError):
        led.filter_carried(["g7"], reason="x", mechanism="y")
    kinds = [r["kind"] for r in read_journal(tmp_path / "ledger")]
    assert "carried_over_report" in kinds
    led.close()


# --------------------------------------------------------------------------- plan / request (3.2)
def _controller(tmp_path, **kw):
    kw.setdefault("attestation", _attestation())
    kw.setdefault("profile", _profile())
    return IslandController(state_dir=tmp_path, configs=CONFIGS, initial_config="T4R2S2",
                            runtime_fingerprint=FP, **kw)


def test_plan_rejections_are_side_effect_free(tmp_path):
    ctl = _controller(tmp_path)
    plan = ctl.plan("T4R4S0", 0, deadline_s=60)
    assert (plan.source_engines, plan.target_engines, plan.add, plan.remove) == (2, 4, 2, 0)
    for target, epoch, why in [("T9", 0, "unknown"), ("T4R4S0", 1, "epoch"),
                               ("T2R6S0", 0, "not E1"), ("T4R2S2", 0, "current")]:
        with pytest.raises(Rejected, match=why):
            ctl.plan(target, epoch, deadline_s=60)
    ctl.close()
    ctl = _controller(tmp_path / "b", attestation=Attestation.none())
    with pytest.raises(Rejected, match="attestation"):
        ctl.plan("T4R4S0", 0, deadline_s=60)
    ctl.close()
    ctl = _controller(tmp_path / "c", attestation=_attestation(fingerprint="sha256:" + "1" * 64))
    with pytest.raises(Rejected, match="fingerprint"):
        ctl.plan("T4R4S0", 0, deadline_s=60)
    ctl.close()
    ctl = _controller(tmp_path / "d", profile=_profile("colocated-serial"))
    with pytest.raises(Rejected, match="partitioned"):
        ctl.plan("T4R4S0", 0, deadline_s=60)
    ctl.close()
    ctl = _controller(tmp_path / "e", profile=_profile(outer="strict-avg"))
    with pytest.raises(Rejected, match="pause not allowed"):  # 3000s > 900*0.5 budget
        ctl.plan("T4R4S0", 0, deadline_s=3000)
    assert ctl.plan("T4R4S0", 0, deadline_s=60).pause_budget_s == 450.0
    ctl.close()
    assert [r["kind"] for r in read_journal(tmp_path / "reconfig")] == []


def test_request_idempotent_conflicts_and_lost_ack(tmp_path):
    ctl = _controller(tmp_path)
    first = ctl.request("r1", "T4R4S0", 0, 60)
    assert ctl.request("r1", "T4R4S0", 0, 60) == first  # same body: same answer
    with pytest.raises(Rejected, match="another body"):
        ctl.request("r1", "T4R4S0", 0, 61)
    with pytest.raises(Rejected, match="in progress"):
        ctl.request("r2", "T4R4S0", 0, 60)
    assert ctl.cancel("r1") == "cancelled"
    assert ctl.cancel("r1") == "cancelled"
    assert ctl.cancel("nope") == "unknown"
    with pytest.raises(Rejected, match="epoch"):
        ctl.request("r3", "T4R4S0", 7, 60)
    ctl.close()
    # the requester lost the answer: status is readable from the journal
    ctl = _controller(tmp_path)
    assert ctl.status("r1")["phase"] == CANCELLED
    ctl.close()


# --------------------------------------------------------------------------- driver e2e
def test_rollout_reconfiguration_round_trip_keeps_samples_and_policy(tmp_path):
    base_driver, _, _, _, _, base_trained, _ = _setup(tmp_path / "base")
    base_final = base_driver.run()

    driver, ctl, fork, pool, publisher, trained, _ = _setup(tmp_path / "x")
    requests = {1: ("up", "T4R4S0", 0), 2: ("down", "T4R2S2", 1)}

    orig_safe = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id in requests:
            rid, target, epoch = requests[rollout_id]
            ctl.request(rid, target, epoch, 60)
        return orig_safe(rollout_id)

    driver.safe_point = safe_point
    final = driver.run()
    assert trained == base_trained  # same sample IDs and optimizer order
    assert final.policy_tensor_hash() == base_final.policy_tensor_hash()
    assert ctl.status("up")["phase"] == SUCCEEDED and ctl.status("down")["phase"] == SUCCEEDED
    assert driver.config_epoch == 2 and read_epochs(tmp_path / "x/state/reconfig").config_id == "T4R2S2"
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == ["start", "stop"]
    # new engines were published the served policy at the fork epoch, then admitted
    assert publisher.published_members[0] == (("engine:c2", "engine:c3"), 1, 1)
    # the removed engines were drained before the stop
    drain_i = fork.calls.index(("drain", ("engine:c2", "engine:c3")))
    stop_i = [i for i, c in enumerate(fork.calls) if c[0] == "stop"][0]
    assert drain_i < stop_i
    ev = _events(tmp_path / "x")
    recon = [e for e in ev if e["event"] == "rl_reconfiguration"]
    assert [(e["result"], e["config_epoch"]) for e in recon] == [(SUCCEEDED, 1), (SUCCEEDED, 2)]
    assert recon[0]["members"] == ["engine:c0", "engine:c1", "engine:c2", "engine:c3"]
    # every publication after the switch was acknowledged by the new member set
    pubs = [e for e in ev if e["event"] == "rl_publication"]
    assert pubs[2]["sync/publication_members"] == recon[0]["members"]
    assert driver.placement.describe().rollout_gpus == ("g4", "g5")
    assert driver.placement.describe().extra["config_epoch"] == 2


def test_no_controller_tape_is_unchanged(tmp_path):
    driver, *_ = _setup(tmp_path / "a")
    driver.run()
    kinds = {e["event"] for e in _events(tmp_path / "a")}
    assert "rl_reconfiguration" not in kinds
    assert not any(e.get("phase") == "reconfigure" for e in _events(tmp_path / "a"))


def _run_with_request(driver, ctl, at, target="T4R4S0", epoch=0, deadline=60, rid="r"):
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == at:
            ctl.request(rid, target, epoch, deadline)
        return orig(rollout_id)

    driver.safe_point = safe_point
    return driver.run()


def test_tool_wait_keeps_old_routing_and_times_out_to_cancel(tmp_path):
    """3.3 X5 (CPU protocol part): active=0 but tool-wait>0 -> nothing released, cancel."""
    driver, ctl, fork, pool, publisher, trained, clock = _setup(
        tmp_path, load=lambda: {"active_requests": 0, "tool_wait": 1},
        controller_kw={"timeouts": Timeouts(drain=5.0, retry_interval=1.0)})
    _run_with_request(driver, ctl, at=1)
    assert ctl.status("r")["phase"] == CANCELLED
    assert "drain" in ctl.status("r")["error"]
    assert not [c for c in fork.calls if c[0] in ("start", "stop")]
    assert driver.config_epoch == 0 and ctl.admission_open
    kinds = [r["kind"] for r in read_journal(tmp_path / "state/reconfig")]
    assert "drain_timeout" in kinds


def test_drain_timeout_on_removal_uncordons_and_cancels(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path, rounds=4)
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("up", "T4R4S0", 0, 60)
        if rollout_id == 2:
            pool.drain_results = [False]
            ctl.request("down", "T4R2S2", 1, 60)
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    assert ctl.status("down")["phase"] == CANCELLED
    assert ("undrain", ("engine:c2", "engine:c3")) in fork.calls
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == ["start"]
    assert driver.config_epoch == 1


def test_payload_readback_failure_rebuilds_old_set(tmp_path):
    driver, ctl, fork, pool, publisher, trained, _ = _setup(tmp_path)
    publisher.corrupt_payload = True
    _run_with_request(driver, ctl, at=1)
    assert ctl.status("r")["phase"] == REBUILT_OLD
    # started engines were stopped again, never admitted, old members kept serving
    assert [c[0] for c in fork.calls if c[0] in ("start", "stop")] == ["start", "stop"]
    assert fork.engine.members_ids == ("engine:c0", "engine:c1")
    assert driver.config_epoch == 0
    assert read_epochs(tmp_path / "state/reconfig").fork_membership_epoch == 2


def test_start_rolled_back_by_fork_keeps_membership(tmp_path):
    driver, ctl, fork, *_ = _setup(tmp_path)
    fork.start_fail = "rolled_back"
    _run_with_request(driver, ctl, at=1)
    assert ctl.status("r")["phase"] == REBUILT_OLD
    assert fork.epoch == 0 and driver.config_epoch == 0


def test_start_rollback_failed_is_stopped_by_retry(tmp_path):
    driver, ctl, fork, *_ = _setup(tmp_path)
    fork.start_fail = "rollback_failed"
    _run_with_request(driver, ctl, at=1)
    assert ctl.status("r")["phase"] == REBUILT_OLD
    assert fork.incomplete is None and fork.epoch == 1  # the retried stop advanced it


def _up_then_down(driver, ctl, pool, before_down=None):
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("up", "T4R4S0", 0, 60)
        if rollout_id == 2:
            if before_down:
                before_down()
            ctl.request("down", "T4R2S2", 1, 60)
        return orig(rollout_id)

    driver.safe_point = safe_point
    return driver.run()


def test_half_failed_stop_is_retried_then_succeeds(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    _up_then_down(driver, ctl, pool, before_down=lambda: setattr(fork, "stop_failures", 1))
    assert ctl.status("down")["phase"] == SUCCEEDED
    stops = [c for c in fork.calls if c[0] == "stop"]
    assert len(stops) == 2 and stops[0][1] == stops[1][1]  # the same stop, retried
    kinds = [(r["kind"], r.get("status")) for r in read_journal(tmp_path / "state/reconfig")]
    assert ("fork_op", "incomplete") in kinds


def test_half_failed_stop_past_deadline_is_recovery_required(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(
        tmp_path, controller_kw={"timeouts": Timeouts(recovery=3.0, retry_interval=10.0)})
    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        _up_then_down(driver, ctl, pool, before_down=lambda: setattr(fork, "stop_failures", 99))
    assert ctl.status("down")["phase"] == RECOVERY_REQUIRED
    assert ctl.inspect().health == "RECOVERY_REQUIRED"
    with pytest.raises(Rejected, match="RECOVERY_REQUIRED"):
        ctl.plan("T4R4S0", 1)


def test_restart_reconciles_fork_epoch(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path, rounds=3)
    _run_with_request(driver, ctl, at=1)
    ctl.close()
    driver.ledger.close()
    assert fork.epoch == 1
    # fork restarted (epoch 0): restore from the journal
    fork.epoch = 0
    fork.last_op = None
    ctl2 = IslandController(state_dir=tmp_path / "state", configs=ctl.configs,
                            attestation=_attestation(), profile=_profile(),
                            initial_config="T4R2S2", runtime_fingerprint=FP)
    status = ctl2.open(pool)
    assert status.health == "RUNNING" and status.config_epoch == 1
    assert fork.calls[-1] == ("restore", 1, None, ["start", ["engine:c2", "engine:c3"]], 0)
    ctl2.close()
    # fork ahead of the journal without an issued op: recovery required, journal kept
    fork.epoch = 5
    ctl3 = IslandController(state_dir=tmp_path / "state", configs=ctl.configs,
                            attestation=_attestation(), profile=_profile(),
                            initial_config="T4R2S2", runtime_fingerprint=FP)
    assert ctl3.open(pool).health == "RECOVERY_REQUIRED"
    assert read_epochs(tmp_path / "state/reconfig").fork_membership_epoch == 1
    ctl3.close()


def test_lost_fork_answer_is_completed_from_the_fork_on_restart(tmp_path):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, placement_kind="fixed-partition")
    fork = ForkMembership(engine, ["engine:c0", "engine:c1"], ["engine:c0"])
    pool = ElasticFakePool(engine, fork)
    ctl = _controller(tmp_path)
    ctl.open(pool)
    ctl.journal.append("fork_op", tx_id="t", op="start", cells=["engine:c1"],
                       expected_fork_epoch=0, status="issued")
    ctl.close()
    fork.change("start", ["engine:c1"], 0)  # applied; the answer never reached the journal
    ctl = _controller(tmp_path)
    assert ctl.open(pool).health == "RUNNING"
    done = [r for r in read_journal(tmp_path / "reconfig") if r.get("status") == "done"]
    assert done and done[-1]["recovered"] is True and done[-1]["result_fork_epoch"] == 1
    ctl.close()


def test_restart_mid_transaction(tmp_path):
    """3.7: before release -> CANCELLED; after release -> RECOVERY_REQUIRED; after CAS -> SUCCEEDED."""
    for case in ("before", "after_release", "after_commit"):
        d = tmp_path / case
        engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, placement_kind="fixed-partition")
        fork = ForkMembership(engine, ["engine:c0", "engine:c1"], ["engine:c0", "engine:c1"])
        pool = ElasticFakePool(engine, fork)
        ctl = _controller(d)
        ctl.open(pool)
        ctl.request("r", "T4R4S0", 0, 60)
        tx = ctl._tx
        if case != "before":
            ctl._phase(tx, "TRANSFERRING")
        if case == "after_commit":
            e = ctl.journal.epochs
            ctl.journal.compare_and_swap(expected_config_epoch=0, new=EpochState(
                1, "T4R4S0", 0, e.members, tx.tx_id))
        ctl.close()
        ctl = _controller(d)
        ctl.open(pool)
        expected = {"before": CANCELLED, "after_release": RECOVERY_REQUIRED,
                    "after_commit": SUCCEEDED}[case]
        assert ctl.status("r")["phase"] == expected, case
        ctl.close()


def test_watchdog_fires_at_the_absolute_deadline(tmp_path):
    fired = []
    driver, ctl, fork, pool, publisher, *_ = _setup(
        tmp_path, controller_kw={"on_watchdog": lambda tx, phase: fired.append(phase)})
    gate = threading.Event()
    slow = publisher.publish_members

    def publish_members(*a, **k):
        gate.wait(5)
        return slow(*a, **k)

    publisher.publish_members = publish_members
    # real-time deadline for the watchdog timer: 0.2 s after the request
    ctl._wall = __import__("time").time
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.2)
            threading.Timer(0.6, gate.set).start()
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    assert fired == ["VERIFYING"]
    assert ctl.status("r")["phase"] == REBUILT_OLD
    kinds = [r["kind"] for r in read_journal(tmp_path / "state/reconfig")]
    assert "watchdog" in kinds


def test_command_inbox_and_cli(tmp_path, capsys):
    driver, ctl, fork, pool, *_ = _setup(tmp_path, inbox=True)
    assert controller_main(["--state-dir", str(tmp_path / "state"), "request", "cli1",
                            "--target", "T4R4S0", "--expected-epoch", "0",
                            "--deadline-s", "60"]) == 0
    driver.run()
    status = json.loads((tmp_path / "state/inbox/cli1.status.json").read_text())
    assert status["tx_id"] and status["request_id"] == "cli1"
    ctl.close()
    driver.ledger.close()
    controller_main(["--state-dir", str(tmp_path / "state"), "status", "cli1"])
    out = json.loads(capsys.readouterr().out)
    assert out["request"]["phase"] == SUCCEEDED and out["epochs"]["config_epoch"] == 1


def test_admission_fence_blocks_generation(tmp_path):
    driver, ctl, *_ = _setup(tmp_path)
    ctl.admission_open = False
    with pytest.raises(DriverError, match="admission fenced"):
        driver.run()


def test_ledger_records_every_round_once(tmp_path):
    driver, ctl, *_ = _setup(tmp_path, rounds=3)
    driver.run()
    recs = read_journal(tmp_path / "state/ledger")
    per = [(r["kind"], r["rollout_id"]) for r in recs if r["kind"] in
           ("prepared", "optimizer_applied", "outer_recorded")]
    assert per == [(k, i) for i in range(3) for k in ("prepared", "optimizer_applied",
                                                      "outer_recorded")]


# --------------------------------------------------------------------------- placement
def test_elastic_placement_refuses_trainer_moves():
    p = ElasticPlacement(_StaticPlacement(), pool_gpus=tuple(f"g{i}" for i in range(8)))
    base = p.describe()
    with pytest.raises(PlacementPlanError, match="epoch"):
        p.reconfigure(base, epoch=2)
    with pytest.raises(PlacementPlanError, match="trainer"):
        p.reconfigure(PlacementDescription("fixed-partition", ("g0",), ("g4",)), epoch=1)
    with pytest.raises(PlacementPlanError, match="outside"):
        p.reconfigure(PlacementDescription("fixed-partition", base.trainer_gpus, ("g9",)), epoch=1)
    with pytest.raises(PlacementPlanError, match="overlap"):
        p.reconfigure(PlacementDescription("fixed-partition", base.trainer_gpus, ("g0",)), epoch=1)
    out = p.reconfigure(PlacementDescription("fixed-partition", base.trainer_gpus,
                                             ("g4", "g5", "g6", "g7")), epoch=1)
    assert out.extra["standby_gpus"] == () and p.epoch == 1


def test_journal_files_are_fsynced(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    with Journal(tmp_path) as j:
        j.append("x")
        n = len(calls)
        j.compare_and_swap(expected_config_epoch=0, new=EpochState(1, "A"))
    assert n >= 1 and len(calls) >= n + 2  # file + directory


# --------------------------------------------------------------------------- review fixes
def test_safe_point_with_a_deferred_eval_in_flight_does_not_drain(tmp_path):
    """M2: an overlapped eval still holding the rollout role blocks WAIT_SAFE."""
    from dataclasses import replace

    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    driver.handshake()
    start = driver.sync.start(driver)
    driver.publish(start.state, rollout_id=0)
    driver.at_safe_point = True
    ctl.request("r", "T4R4S0", 0, 60)
    snap = replace(driver.safe_point_snapshot(0), eval_in_flight=1)
    assert ctl.run_at_safe_point(driver, snap) == "WAIT_SAFE"
    assert not [c for c in fork.calls if c[0] in ("drain", "start", "stop")]
    assert ctl.has_pending() and ctl.admission_open
    blocked = [r for r in read_journal(tmp_path / "state/reconfig") if r["kind"] == "safe_point_blocked"]
    assert "overlapped evals in flight" in blocked[-1]["blockers"][0]


def test_tool_wait_board_feeds_the_drain_and_unknown_fails_closed(tmp_path):
    from yeto.rl.engine.miles_adapter.rollout import MilesRolloutPool
    from yeto.rl.engine.tool_wait import ToolWaitBoard

    board = ToolWaitBoard()
    pool = MilesRolloutPool(inference_controller=object(), rollout_executor=object(), metadata=None,
                            expected_policy=lambda: (0, "h"), tool_wait_board=board)
    assert "router in-flight count unknown" in pool.trajectory_load()["blockers"]
    pool.load_sample = lambda: {"active_requests": 0, "workers": 2, "cordoned": 0}
    board.enter("t1")
    load = pool.trajectory_load()
    assert load["tool_wait"] == 1 and load["blockers"] == ["1 trajectories waiting on tools"]
    board.exit("t1")
    assert pool.trajectory_load()["blockers"] == []
    no_board = MilesRolloutPool(inference_controller=object(), rollout_executor=object(),
                                metadata=None, expected_policy=lambda: (0, "h"))
    assert no_board.trajectory_load() is None


def test_committed_placement_is_restored_after_restart():
    p = ElasticPlacement(_StaticPlacement(), pool_gpus=tuple(f"g{i}" for i in range(8)), epoch=2)
    out = p.restore_committed(("g4", "g5", "g6", "g7"), epoch=2)
    assert out.rollout_gpus == ("g4", "g5", "g6", "g7") and p.epoch == 2
    with pytest.raises(PlacementPlanError):
        p.restore_committed(("g4",), epoch=1)


def test_rebase_promotes_applied_batches_below_the_restart(tmp_path):
    led = BatchLedger(tmp_path)
    led.prepare(_B(0, ["g0"]), policy_token="t0")
    led.optimizer_applied(0)  # crashed before outer_recorded, but the outer took it
    assert led.rebase(1) == []
    assert led.state(0) == "outer_recorded"
    rec = [r for r in read_journal(tmp_path / "ledger") if r["kind"] == "outer_recorded"]
    assert rec[-1]["recovered"] is True
    led.close()


def test_engine_discarded_survives_replay(tmp_path):
    led = BatchLedger(tmp_path)
    b = _B(0, ["g0"])
    b.aborted_in_flight_groups = 2
    led.prepare(b, policy_token="t")
    led.close()
    led = BatchLedger(tmp_path)
    assert led.batch(0)["engine_discarded"] == 2
    assert led.cut_summary()["engine_discarded_groups"] == 2
    led.close()


def test_missing_group_index_fails_closed():
    from types import SimpleNamespace

    from yeto.rl.engine.miles_adapter.rollout_meta_hook import group_record

    sample = SimpleNamespace(index=3, reward=1.0, weight_versions=[])
    for args in (SimpleNamespace(), SimpleNamespace(yeto_rl_elastic_metadata=True)):
        with pytest.raises(RuntimeError, match="group_index"):
            group_record(args, [sample])


def test_reconfiguration_event_records_the_scheduled_eval_under_overlap(tmp_path):
    """Integ-s2 finding 4: elastic + eval overlap -- the rl_reconfiguration event
    carries ``eval_due`` (an eval scheduled but not started yet)."""
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.overlap import IMPLEMENTED_OVERLAP

    profile = ExecutionProfile(name="t-overlap", execution_mode="partitioned-overlap",
                               outer_protocol="none",
                               allowed_overlap=IMPLEMENTED_OVERLAP).bind_algorithm(AlgorithmSpec())

    class Handle:
        begin = end = None

        def __init__(self, rid):
            self.rid = rid

        def cancel(self):
            pass

        def result(self):
            return {"score": float(self.rid)}

    driver, ctl, *_ = _setup(tmp_path, driver_kw={
        "profile": profile, "evaluate": lambda rid: {"score": 0.0}, "eval_interval": 1,
        "evaluate_start": Handle,
        "capabilities": fake_capabilities(execution_modes=set(MODES) | {"partitioned-overlap"}),
    })
    _run_with_request(driver, ctl, at=2)
    recon = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert [e["result"] for e in recon] == [SUCCEEDED]
    assert "eval_due" in recon[0]
    # the eval of v2 was scheduled at round 2 and starts after the next generation
    assert recon[0]["eval_due"] == 2
    starts = [e["policy_version"] for e in _events(tmp_path) if e["event"] == "rl_eval_overlap_start"]
    assert 2 in starts  # it did start, after the reconfiguration


def test_reconfiguration_event_has_no_eval_due_without_overlap(tmp_path):
    driver, ctl, *_ = _setup(tmp_path)
    _run_with_request(driver, ctl, at=2)
    recon = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"]
    assert recon and "eval_due" not in recon[0]


def test_build_elastic_and_journal_expand_user_paths(tmp_path, monkeypatch):
    """Integ-s2 finding 5: '~' in --rl-elastic-state-dir/resources/attestation
    resolves to $HOME, never a literal './~' directory."""
    from yeto.rl.engine.journal import Journal, read_epochs, read_journal
    from yeto.rl.engine.miles_adapter.elastic_wiring import build_elastic

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "res.json").write_text(json.dumps(
        {"configs": {"c0": {"trainer": 1, "rollout": 1}}, "edges": []}))
    (tmp_path / "att.json").write_text("{}")
    wiring = build_elastic(state_dir="~/st", resources="~/res.json", attestation="~/att.json",
                           profile=_profile(), initial_config="c0", runtime_fingerprint=FP,
                           declared_cells=("a",))
    assert wiring.controller.state_dir == tmp_path / "st"
    assert (tmp_path / "st" / "reconfig").is_dir() and not (tmp_path / "~").exists()
    wiring.controller.journal.close()
    with Journal("~/j") as journal:
        assert journal.dir == tmp_path / "j"
    assert read_journal("~/st/reconfig") == read_journal(tmp_path / "st" / "reconfig")
    assert read_epochs("~/st/reconfig") == read_epochs(tmp_path / "st" / "reconfig")
    assert not (tmp_path / "~").exists()


class _FakeRay:
    """ray.get/ray.kill over plain values (the manager's methods are sync fakes)."""

    def __init__(self, on_kill):
        self.on_kill = on_kill
        self.killed = []

    def get(self, value, timeout=None):
        return value

    def kill(self, handle, no_restart=False):
        self.killed.append((handle, no_restart))
        self.on_kill(handle)


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _FakeManager:
    def __init__(self):
        from types import SimpleNamespace

        self.infos = {c: [SimpleNamespace(name=f"{c}/w0", generation=3)]
                      for c in ("c2", "c3")}

        def get_worker_infos(cell):  # the fork's lookup: bare cell ids only, else AssertionError
            matches = self.infos.get(cell, [])
            assert matches, \
                f"cell_id={cell!r} matches=[]"
            return matches

        self.get_worker_infos = _Remote(get_worker_infos)
        self.get_actor_handle = _Remote(
            lambda name, expected_generation: f"handle:{name}@{expected_generation}")


def test_default_watchdog_kills_the_target_generation_and_unblocks(tmp_path):
    from yeto.rl.engine.miles_adapter.elastic_wiring import kill_target_generation

    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    gate, dead = threading.Event(), []
    fake_ray = _FakeRay(lambda handle: (dead.append(handle), gate.set()))
    ctl.set_on_watchdog(kill_target_generation(ctl, manager=_FakeManager(), ray_module=fake_ray))
    slow = publisher.publish_members

    def publish_members(*a, **k):
        gate.wait(5)  # blocked on the new engines until they are killed
        if dead:
            raise RuntimeError("engine actor died")
        return slow(*a, **k)

    publisher.publish_members = publish_members
    ctl._wall = __import__("time").time
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.2)
        return orig(rollout_id)

    driver.safe_point = safe_point
    started = __import__("time").monotonic()
    driver.run()
    assert __import__("time").monotonic() - started < 4  # not the 5 s gate: the kill unblocked it
    assert ctl.status("r")["phase"] == REBUILT_OLD
    assert sorted(h for h, _ in fake_ray.killed) == ["handle:c2/w0@3", "handle:c3/w0@3"]
    assert all(no_restart for _, no_restart in fake_ray.killed)
    records = read_journal(tmp_path / "state/reconfig")
    action = next(r for r in records if r["kind"] == "watchdog_action")
    assert [k["cell"] for k in action["killed"]] == ["engine:c2", "engine:c3"]
    assert next(r for r in records if r["kind"] == "watchdog")["target_cells"] == [
        "engine:c2", "engine:c3"]
    # old members were never touched
    assert set(driver.rollout.members()) == {"engine:c0", "engine:c1"}


def test_watchdog_unmappable_target_enters_recovery_required_not_silent_wait(tmp_path):
    """GPU a4s3/a4s4 regression: a manager that cannot resolve the targets (fork-style
    lookup, real ``engine:`` member ids) must not leave the island waiting silently."""
    from yeto.rl.engine.miles_adapter.elastic_wiring import kill_target_generation

    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    manager = _FakeManager()
    manager.infos.clear()  # nothing resolvable
    fake_ray = _FakeRay(lambda handle: None)
    ctl.set_on_watchdog(kill_target_generation(ctl, manager=manager, ray_module=fake_ray))
    gate = threading.Event()
    slow = publisher.publish_members

    def publish_members(*a, **k):
        gate.wait(1.5)
        return slow(*a, **k)

    publisher.publish_members = publish_members
    ctl._wall = __import__("time").time
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.2)
        return orig(rollout_id)

    driver.safe_point = safe_point
    with contextlib.suppress(Exception):
        driver.run()
    records = read_journal(tmp_path / "state/reconfig")
    action = next(r for r in records if r["kind"] == "watchdog_action")
    assert action["killed"] == [] and "matches=[]" in action["errors"][0]["error"]
    assert action["errors"][0]["kind"] == "unknown_target"
    assert ctl.recovery_required and "could not kill" in ctl.recovery_required
    assert not ctl.admission_open


def test_watchdog_target_without_live_workers_is_unresolved_not_silent(tmp_path):
    """A cell the fork knows but with no live worker actors (not started yet / already
    stopped) cannot release the blocked step by being killed: journaled as kind
    ``no_workers`` and the island enters RECOVERY_REQUIRED (never a bare killed=[])."""
    from yeto.rl.engine.miles_adapter.elastic_wiring import kill_target_generation

    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    manager = _FakeManager()
    infos = dict(manager.infos)
    manager.get_worker_infos = _Remote(lambda cell: list(infos.get(cell, [])) if cell in ("c2", "c3")
                                       else (_ for _ in ()).throw(AssertionError(f"cell_id={cell!r} matches=[]")))
    infos["c3"] = []  # exists, no actors
    fake_ray = _FakeRay(lambda handle: None)
    ctl.set_on_watchdog(kill_target_generation(ctl, manager=manager, ray_module=fake_ray))
    gate = threading.Event()
    slow = publisher.publish_members

    def publish_members(*a, **k):
        gate.wait(1.5)
        return slow(*a, **k)

    publisher.publish_members = publish_members
    ctl._wall = __import__("time").time
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.2)
        return orig(rollout_id)

    driver.safe_point = safe_point
    with contextlib.suppress(Exception):
        driver.run()
    records = read_journal(tmp_path / "state/reconfig")
    action = next(r for r in records if r["kind"] == "watchdog_action")
    assert [k["cell"] for k in action["killed"]] == ["engine:c2"]  # c2 had a worker: killed
    assert [(e["cell"], e["kind"]) for e in action["errors"]] == [("engine:c3", "no_workers")]
    assert ctl.recovery_required and "could not kill" in ctl.recovery_required


def test_watchdog_outside_start_verify_kills_nothing(tmp_path):
    from yeto.rl.engine.miles_adapter.elastic_wiring import kill_target_generation

    driver, ctl, *_ = _setup(tmp_path)
    fake_ray = _FakeRay(lambda h: None)
    handler = kill_target_generation(ctl, manager=_FakeManager(), ray_module=fake_ray)
    handler("tx-none", "QUIESCING")
    assert fake_ray.killed == []
    action = [r for r in read_journal(tmp_path / "state/reconfig") if r["kind"] == "watchdog_action"]
    assert action and action[0]["killed"] == []


def test_build_elastic_wires_the_default_watchdog_action(tmp_path):
    from yeto.rl.engine.miles_adapter.elastic_wiring import build_elastic

    res = tmp_path / "res.json"
    res.write_text(json.dumps({"configs": {"T4R2S2": {"trainer": 4, "rollout": 2, "standby": 2}},
                               "edges": []}))
    kw = dict(resources=res, attestation=None, profile=_profile(), initial_config="T4R2S2",
              runtime_fingerprint=FP, declared_cells=("c0",))
    w = build_elastic(state_dir=tmp_path / "a", **kw)
    assert w.controller._on_watchdog is not None
    w.controller.close()
    off = build_elastic(state_dir=tmp_path / "b", on_watchdog=None, **kw)
    assert off.controller._on_watchdog is None
    off.controller.close()
    with pytest.raises(ValueError, match="unknown watchdog action"):
        build_elastic(state_dir=tmp_path / "c", on_watchdog="nope", **kw)


def test_watchdog_firing_after_the_last_verify_prevents_the_commit(tmp_path):
    """Review F2: the deadline passes after the last _check_deadline but before
    the CAS (a slow verify_serving_policy); the new cells may have been killed,
    so the transaction must not commit."""
    from yeto.rl.engine.miles_adapter.elastic_wiring import kill_target_generation

    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    fake_ray = _FakeRay(lambda handle: None)
    ctl.set_on_watchdog(kill_target_generation(ctl, manager=_FakeManager(), ray_module=fake_ray))
    fired = ctl._watchdog_fired

    def verify_serving_policy(**_):
        assert fired.wait(5)  # returns normally, but only after the watchdog fired

    publisher.verify_serving_policy = verify_serving_policy
    ctl._wall = __import__("time").time
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.3)
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    assert ctl.status("r")["phase"] == REBUILT_OLD
    assert "watchdog fired before the commit point" in (ctl.status("r")["error"] or "")
    assert read_epochs(tmp_path / "state/reconfig").config_epoch == 0
    assert set(driver.rollout.members()) == {"engine:c0", "engine:c1"}


def test_watchdog_after_the_commit_point_kills_nothing(tmp_path):
    driver, ctl, *_ = _setup(tmp_path)
    killed = []
    ctl.set_on_watchdog(lambda tx, phase: killed.append(phase))
    ctl._wall = __import__("time").time
    orig_cas = ctl.journal.compare_and_swap

    def slow_cas(**kw):
        out = orig_cas(**kw)
        if kw["new"].config_epoch == 1:
            __import__("time").sleep(0.6)  # the deadline passes right after the commit
        return out

    ctl.journal.compare_and_swap = slow_cas
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.3)
        return orig(rollout_id)

    driver.safe_point = safe_point
    driver.run()
    assert ctl.status("r")["phase"] == SUCCEEDED and killed == []
    notes = [r.get("note", "") for r in read_journal(tmp_path / "state/reconfig")
             if r["kind"] == "watchdog"]
    assert notes and "after the commit point" in notes[0]


def test_member_publication_is_on_the_tape_before_the_next_generation(tmp_path):
    """A4 E1-A (c) audit gap: the new engines get the published policy through
    publish_members at the up transaction; the tape records it, so the members
    serving the next generation are auditable before the next rl_publication."""
    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    _run_with_request(driver, ctl, at=1, rid="up")
    ev = _events(tmp_path)
    kinds = [e["event"] for e in ev]
    i = kinds.index("rl_member_publication")
    member_pub = ev[i]
    assert member_pub["policy_version"] == 1
    assert member_pub["sync/publication_members"] == ["engine:c2", "engine:c3"]
    assert member_pub["sync/serving_members"] == [f"engine:c{k}" for k in range(4)]
    assert member_pub["rl/policy_token"] == next(
        e for e in ev if e["event"] == "rl_publication" and e["policy_version"] == 1)["rl/policy_token"]
    # it comes before rollout 1's generation
    gen1 = next(j for j, e in enumerate(ev) if e["event"] == "rl_driver_phase"
                and e.get("phase") == "generate" and e.get("rollout_id") == 1)
    assert i < gen1


def test_no_member_publication_event_without_a_reconfiguration(tmp_path):
    driver, *_ = _setup(tmp_path)
    driver.run()
    assert "rl_member_publication" not in {e["event"] for e in _events(tmp_path)}
