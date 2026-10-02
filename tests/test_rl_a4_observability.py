"""A4 observability / injection fixes (E1-B cause, membership event, hold, watchdog
classification, A4b tool-wait injection, undrain failure). CPU only, no Ray."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from test_rl_e1_injections import _perturb_publisher
from test_rl_reconfig_e1 import (
    ElasticFakePublisher,
    _events,
    _run_with_request,
    _setup,
    _up_then_down,
)
from yeto.rl.engine.controller import (
    CANCELLED,
    RECOVERY_REQUIRED,
    REBUILT_OLD,
    SUCCEEDED,
    Timeouts,
)
from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.journal import read_journal
from yeto.rl.engine.miles_adapter.publish import (
    HOLD_BEFORE_CHECK_ENV,
    INJECT_LORA_PERTURB_ENV,
    PublicationError,
)
from yeto.rl.engine.ports import PublicationCause


def _journal(tmp_path, kind):
    return [r for r in read_journal(tmp_path / "state/reconfig") if r["kind"] == kind]


# ---------------------------------------------------------------- 1. cause enum
def test_cause_enum_values():
    assert {c.value for c in PublicationCause} >= {
        "payload_mismatch", "token_mismatch", "update_failed", "other"}


def test_payload_readback_refusal_carries_cause_and_engine_ids(monkeypatch):
    pub, world = _perturb_publisher(monkeypatch, 0.01)
    with pytest.raises(PublicationError) as raised:
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert raised.value.cause is PublicationCause.PAYLOAD_MISMATCH
    assert raised.value.engine_ids == ("engine0",)  # index in the check_weights read-back


def test_update_weights_failure_cause(monkeypatch):
    pub, world = _perturb_publisher(monkeypatch, None)

    async def broken(*a, **k):
        raise RuntimeError("NCCL timeout")

    pub._update_weights = broken
    with pytest.raises(PublicationError) as raised:
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert raised.value.cause is PublicationCause.UPDATE_FAILED
    assert raised.value.engine_ids == ("engine:c2",)


def test_token_not_reported_cause(monkeypatch):
    pub, world = _perturb_publisher(monkeypatch, None)

    class Stale:  # accepts the call but keeps reporting an older policy token
        async def update_weight_version(self, token):
            pass

        async def get_weight_version(self):
            return "yeto:0:old"

    async def start(members, expected_epoch):
        return SimpleNamespace(rollout_engines=[Stale() for _ in members],
                               snapshot_cell_id_to_hashes={c: "h" for c in members})

    pub._controller.start_update_weights = start
    with pytest.raises(PublicationError) as raised:
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert raised.value.cause is PublicationCause.TOKEN_MISMATCH


def test_rebuild_old_journal_and_tape_carry_cause_and_engines(tmp_path):
    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    error = PublicationError("payload read-back differs", cause=PublicationCause.PAYLOAD_MISMATCH,
                             engine_ids=["engine1", "engine0"])  # what MilesPublisher raises

    def refuse(*a, **k):
        raise error

    publisher.publish_members = refuse
    _run_with_request(driver, ctl, at=1)
    assert ctl.status("r")["phase"] == REBUILT_OLD
    rows = {r["phase"]: r for r in _journal(tmp_path, "phase")
            if r["phase"] in ("REBUILD_OLD", "REBUILT_OLD")}
    for phase in ("REBUILD_OLD", "REBUILT_OLD"):
        assert rows[phase]["cause"] == "payload_mismatch", phase
        assert rows[phase]["inconsistent_engines"] == ["engine0", "engine1"], phase
    tape = [e for e in _events(tmp_path) if e["event"] == "rl_reconfiguration"
            and e["result"] == REBUILT_OLD]
    assert len(tape) == 1
    assert tape[0]["cause"] == "payload_mismatch" and "read-back differs" in tape[0]["error"]
    assert tape[0]["inconsistent_engines"] == ["engine0", "engine1"]


def test_unstructured_refusal_is_cause_other(tmp_path):
    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    publisher.corrupt_payload = True  # fake raises a plain RuntimeError
    _run_with_request(driver, ctl, at=1)
    row = next(r for r in _journal(tmp_path, "phase") if r["phase"] == "REBUILT_OLD")
    assert row["cause"] == "other" and row["inconsistent_engines"] == []
    tape = next(e for e in _events(tmp_path) if e.get("result") == REBUILT_OLD)
    assert tape["cause"] == "other"


# ------------------------------------------------- 1b. test_injection records
def _sinked(monkeypatch, eps):
    pub, world = _perturb_publisher(monkeypatch, eps)
    records = []
    pub.event_sink = lambda event, **f: records.append((event, f))
    return pub, world, records


def test_perturbation_writes_an_applied_test_injection_record(monkeypatch):
    pub, world, records = _sinked(monkeypatch, 0.01)
    with pytest.raises(PublicationError):
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    (event, f), = [r for r in records if r[0] == "test_injection"]  # member_engines is not an injection
    assert event == "test_injection" and f["kind"] == "lora_perturb" and f["applied"] is True
    assert f["target_members"] == ["engine:c2"] and f["scale"] == 0.01


def test_perturbation_hook_failure_is_applied_false(monkeypatch):
    pub, world, records = _sinked(monkeypatch, 0.01)

    def broken(scale):
        raise RuntimeError("This event loop is already running")

    pub.perturb_trainer = broken
    with pytest.raises(PublicationError, match="already running"):
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    (event, f), = records
    assert event == "test_injection" and f["applied"] is False
    assert "already running" in f["error"]


def test_missing_perturbation_hook_is_applied_false(monkeypatch):
    pub, world, records = _sinked(monkeypatch, 0.01)
    pub.perturb_trainer = None
    with pytest.raises(PublicationError):
        asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert records[0][1]["applied"] is False


def test_controller_sink_writes_journal_and_tape_with_tx_and_phase(tmp_path):
    driver, ctl, *_ = _setup(tmp_path)
    ctl.record_test_event(driver, "test_injection", kind="lora_perturb",
                          target_members=["engine:c2"], applied=False)
    (row,) = _journal(tmp_path, "test_injection")
    assert row["injection_kind"] == "lora_perturb" and row["applied"] is False
    assert {"ts", "tx_id", "phase", "target_members"} <= set(row)
    tape = next(e for e in _events_live(driver) if e["event"] == "test_injection")
    assert tape["kind"] == "lora_perturb" and tape["applied"] is False and "ts" in tape


def _events_live(driver):
    import json

    return [json.loads(x) for x in driver.events.path.read_text().splitlines()]


# ---------------------------------------------------------------- 2. membership
def test_rl_membership_on_up_and_down_without_fabricated_publication(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    _up_then_down(driver, ctl, pool)
    assert ctl.status("up")["phase"] == SUCCEEDED and ctl.status("down")["phase"] == SUCCEEDED
    ev = [e for e in _events(tmp_path) if e["event"] == "rl_membership"]
    assert [(e["kind"], e["config_epoch"], e["round"]) for e in ev] == [("up", 1, 1), ("down", 2, 2)]
    assert ev[0]["members"] == ["engine:c0", "engine:c1", "engine:c2", "engine:c3"]
    assert ev[1]["members"] == ["engine:c0", "engine:c1"]
    assert ev[0]["tx_id"] != ev[1]["tx_id"]
    assert not any(k.startswith("sync/") or k.startswith("rl/") for e in ev for k in e)  # no payload
    # the scale-down published nothing: its tx has no rl_member_publication
    publications = [e for e in _events(tmp_path) if e["event"] == "rl_member_publication"]
    assert {e["tx_id"] for e in publications} == {ev[0]["tx_id"]}


def test_rl_membership_not_written_when_the_transaction_rolls_back(tmp_path):
    driver, ctl, fork, pool, publisher, *_ = _setup(tmp_path)
    publisher.corrupt_payload = True
    _run_with_request(driver, ctl, at=1)
    assert ctl.status("r")["phase"] == REBUILT_OLD
    assert not [e for e in _events(tmp_path) if e["event"] == "rl_membership"]


# ----------------------------------------------------------------- 3. hold
def _hold_publisher(monkeypatch, seconds, *, injection_mode=True):
    monkeypatch.setenv(HOLD_BEFORE_CHECK_ENV, str(seconds))
    if injection_mode:
        monkeypatch.setenv("YETO_RL_TEST_KILL_LEARNER_AT", "NEVER")  # any injection variable
    else:
        monkeypatch.delenv("YETO_RL_TEST_KILL_LEARNER_AT", raising=False)
    pub, world, records = _sinked(monkeypatch, None)
    c = pub._controller
    for name in ("end_update_weights", "check_weights", "admit_cells"):
        original = getattr(c, name)

        def wrapped(*a, _o=original, _n=name, **k):
            world["order"].append(("call", _n, time.time()))
            return _o(*a, **k)

        setattr(c, name, wrapped)
    return pub, world, records


def test_hold_sits_between_end_update_and_check_without_blocking_the_loop(monkeypatch):
    pub, world, records = _hold_publisher(monkeypatch, 0.3)
    ticks = []

    async def main():
        async def ticker():  # the loop must stay live during the hold
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        try:
            await pub._publish_members("tok", ["c2"], 1)
        finally:
            task.cancel()

    asyncio.run(main())
    calls = [(n, t) for tag, n, t in (o for o in world["order"] if isinstance(o, tuple) and o[0] == "call")]
    names = [n for n, _ in calls]
    assert names == ["end_update_weights", "check_weights", "admit_cells"]
    start, end = [f for e, f in records if e == "test_hold"]
    assert start["stage"] == "start" and end["stage"] == "end"
    assert end["end_ts"] - end["start_ts"] >= 0.3
    assert calls[0][1] <= end["start_ts"] + 0.05 and calls[1][1] >= end["end_ts"] - 0.05
    assert len(ticks) >= 10  # ~30 expected: not blocked
    # once per process: the next member publication is not held again
    records.clear()
    world["engine"].pop("c2")
    asyncio.run(pub._publish_members("tok", ["c3"], 2))
    assert [e for e, _ in records if e in ("test_hold", "test_injection")] == []


def test_hold_is_ignored_outside_test_injection_mode(monkeypatch, capsys):
    pub, world, records = _hold_publisher(monkeypatch, 5, injection_mode=False)
    started = time.monotonic()
    asyncio.run(pub._publish_members("tok", ["c2"], 1))
    assert time.monotonic() - started < 2
    assert [e for e, _ in records if e in ("test_hold", "test_injection")] == []
    assert "ignored" in capsys.readouterr().err


# ------------------------------------------------------------ 4. watchdog
class _Probe:
    def __init__(self):
        self.dead = threading.Event()

    async def snapshot(self, cells):
        pass

    async def status(self):
        return "dead: killed" if self.dead.is_set() else "alive"


def _watchdog_setup(tmp_path):
    fired = []
    probe = _Probe()
    driver, ctl, fork, pool, publisher, *_ = _setup(
        tmp_path, controller_kw={"on_watchdog": lambda tx, phase: (fired.append(phase),
                                                                    probe.dead.set())})
    ctl._wall = time.time
    orig = driver.safe_point

    def safe_point(rollout_id):
        if rollout_id == 1:
            ctl.request("r", "T4R4S0", 0, 0.3)
        return orig(rollout_id)

    driver.safe_point = safe_point
    return driver, ctl, fork, pool, publisher, fired, probe


def _watchdog_row(tmp_path):
    (row,) = _journal(tmp_path, "watchdog")
    return row


def test_watchdog_on_the_blocked_update_is_classified(tmp_path, monkeypatch):
    """The REAL publisher's injected block (reached record, liveness probe) under the
    watchdog: it fires while the update is blocked."""
    monkeypatch.setenv("YETO_RL_TEST_INJECT_UPDATE_WEIGHTS_BLOCK_S", "5")
    from yeto.rl.engine.miles_adapter.publish import MilesPublisher

    driver, ctl, fork, pool, publisher, fired, probe = _watchdog_setup(tmp_path)
    real = MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                          inference_controller=None)
    real.liveness_probe, real.block_poll_s = probe, 0.02
    real.event_sink = lambda event, **f: ctl.record_test_event(driver, event, **f)
    slow = publisher.publish_members

    def publish_members(state, members, **k):
        asyncio.run(real._injected_block(sorted(members), 5.0))  # raises once the probe is dead
        return slow(state, members, **k)

    publisher.publish_members = publish_members
    started = time.monotonic()
    driver.run()
    assert time.monotonic() - started < 4
    row = _watchdog_row(tmp_path)
    assert fired == ["VERIFYING"] and row["phase"] == "VERIFYING"
    assert row["injection_reached"] is True and row["classification"] == "FIRED_ON_BLOCKED_UPDATE"
    reached = next(r for r in _journal(tmp_path, "test_injection") if "reached_ts" in r)
    assert reached["injection_kind"] == "block_update" and reached["applied"] is True
    assert ctl.status("r")["phase"] == REBUILT_OLD
    released = [r for r in _journal(tmp_path, "test_injection") if "released_ts" in r]
    assert released and "dead" in released[0]["outcome"]


def test_watchdog_while_the_block_was_never_reached(tmp_path):
    driver, ctl, fork, pool, publisher, fired, probe = _watchdog_setup(tmp_path)
    gate = threading.Event()
    slow = publisher.publish_members

    def publish_members(*a, **k):  # slow, but not at the injection point
        gate.wait(0.8)
        return slow(*a, **k)

    publisher.publish_members = publish_members
    driver.run()
    row = _watchdog_row(tmp_path)
    assert fired == ["VERIFYING"]
    assert row["injection_reached"] is False and row["classification"] == "INJECTION_NOT_REACHED"


def test_watchdog_still_initializing_is_not_reached(tmp_path):
    driver, ctl, fork, pool, publisher, fired, probe = _watchdog_setup(tmp_path)
    gate = threading.Event()
    add = pool.add_engines

    def slow_start(*a, **k):
        gate.wait(0.8)
        return add(*a, **k)

    pool.add_engines = slow_start
    driver.run()
    row = _watchdog_row(tmp_path)
    assert row["phase"] == "INITIALIZING"
    assert row["injection_reached"] is False and row["classification"] == "INJECTION_NOT_REACHED"


# -------------------------------------------------- 5. A4b: undrain failure
def test_undrain_failure_after_drain_timeout_is_recovery_required(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)

    def broken_undrain(members):
        raise RuntimeError("router unreachable")

    def before_down():
        pool.drain_results = [False]  # drain times out
        pool.undrain = broken_undrain

    with pytest.raises(DriverError, match="RECOVERY_REQUIRED"):
        _up_then_down(driver, ctl, pool, before_down=before_down)
    assert ctl.recovery_required and "undrain" in ctl.recovery_required
    assert not ctl.admission_open  # NOT reopened on a half-cordoned fleet
    assert ctl.status("down")["phase"] == RECOVERY_REQUIRED
    assert _journal(tmp_path, "undrain_failed")
    assert not any(r["phase"] == CANCELLED and r.get("request_id") == "down"
                   for r in _journal(tmp_path, "phase"))
    assert [e for e in _events(tmp_path)
            if e["event"] == "rl_reconfiguration" and e["result"] == "RECOVERY_REQUIRED"]


def test_drain_timeout_with_a_working_undrain_stays_cancelled(tmp_path):
    driver, ctl, fork, pool, *_ = _setup(tmp_path)
    _up_then_down(driver, ctl, pool, before_down=lambda: setattr(pool, "drain_results", [False]))
    assert ctl.status("down")["phase"] == CANCELLED and ctl.admission_open


# ------------------------------------------------ 5b. A4b: tool-wait injection
def _real_pool(monkeypatch, seconds, board):
    from yeto.rl.engine.miles_adapter.rollout import (
        HARNESS_NOT_AGENTIC, INJECT_TOOL_WAIT_ENV, MilesRolloutPool,
    )

    if seconds is None:
        monkeypatch.delenv(INJECT_TOOL_WAIT_ENV, raising=False)
    else:
        monkeypatch.setenv(INJECT_TOOL_WAIT_ENV, str(seconds))

    class Fork:
        async def drain_cells(self, cells, timeout_seconds):
            return True

    pool = MilesRolloutPool(inference_controller=Fork(), rollout_executor=None, metadata=None,
                            expected_policy=lambda: (0, "h"), tool_wait_board=board,
                            harness=HARNESS_NOT_AGENTIC,  # IR-2: explicit zeros, not unknown
                            runner=SimpleNamespace(run=asyncio.run), declared_cells=("c0", "c1"))
    pool.load_sample = lambda: {"active_requests": 0, "workers": 2, "cordoned": 0}
    records = []
    pool.event_sink = lambda event, **f: records.append((event, f))
    return pool, records


def test_undrain_fail_injection_fails_the_next_n_undrains_and_records(monkeypatch):
    from yeto.rl.engine.miles_adapter.rollout import INJECT_UNDRAIN_FAIL_ENV

    monkeypatch.setenv(INJECT_UNDRAIN_FAIL_ENV, "1")
    pool, records = _real_pool(monkeypatch, None, None)
    calls = []

    async def uncordon_cells(cells):
        calls.append(sorted(cells))

    pool._controller.uncordon_cells = uncordon_cells
    with pytest.raises(RuntimeError, match="injected undrain failure"):
        pool.undrain(frozenset({"engine:c1"}))
    assert calls == [] and pool.injected_undrain_failures == [frozenset({"engine:c1"})]
    (event, f), = records
    assert event == "test_injection" and f["kind"] == "undrain_fail" and f["applied"] is True
    assert f["target_members"] == ["engine:c1"]
    pool.undrain(frozenset({"engine:c1"}))  # N exhausted: the real uncordon runs
    assert calls == [["c1"]]


def test_without_the_env_undrain_is_the_plain_uncordon(monkeypatch):
    from yeto.rl.engine.miles_adapter.rollout import INJECT_UNDRAIN_FAIL_ENV

    monkeypatch.delenv(INJECT_UNDRAIN_FAIL_ENV, raising=False)
    pool, records = _real_pool(monkeypatch, None, None)
    calls = []

    async def uncordon_cells(cells):
        calls.append(sorted(cells))

    pool._controller.uncordon_cells = uncordon_cells
    pool.undrain(frozenset({"engine:c0"}))
    assert calls == [["c0"]] and records == []


def test_tool_wait_injection_counts_on_the_board_for_n_seconds(monkeypatch):
    from yeto.rl.engine.tool_wait import ToolWaitBoard

    board = ToolWaitBoard()
    pool, records = _real_pool(monkeypatch, 0.4, board)
    assert pool.drain(frozenset({"engine:c1"}), time.time() + 30) is True  # the router part
    load = pool.trajectory_load()
    assert load["tool_wait"] == 1 and load["blockers"] == ["1 trajectories waiting on tools"]
    (event, f), = records
    assert event == "test_injection" and f["kind"] == "tool_wait" and f["applied"] is True
    assert f["target_members"] == ["engine:c1"]
    deadline = time.time() + 5
    while board.snapshot().in_flight and time.time() < deadline:
        time.sleep(0.05)
    assert pool.trajectory_load()["blockers"] == []
    # the injected tool executes on every drain (so a replayed drain is observable)
    pool.drain(frozenset({"engine:c1"}), time.time() + 30)
    assert board.snapshot().in_flight == 1 and len(records) == 2
    assert [f["attempt"] for _, f in records] == [1, 2]
    assert records[0][1]["tool_call_id"] == records[1][1]["tool_call_id"]
    assert records[0][1]["side_effect_log"] is False


# ------------------------------------------- 5c. A4bc: side-effect journal, no replay (3.3 X5)
def _side_effects(path):
    from yeto.rl.engine.tool_wait import read_side_effects

    return read_side_effects(path)


def test_side_effect_log_records_each_injected_execution_once(monkeypatch, tmp_path):
    from yeto.rl.engine.miles_adapter.rollout import INJECTED_TOOL_WAIT_ID, injected_tool_call_id
    from yeto.rl.engine.tool_wait import ToolWaitBoard, side_effect_duplicates

    board = ToolWaitBoard()
    log = tmp_path / "elastic-state" / "side_effects.jsonl"
    pool, records = _real_pool(monkeypatch, 0.3, board)
    pool._side_effects = None
    from yeto.rl.engine.tool_wait import ToolSideEffectLog

    pool._side_effects = ToolSideEffectLog(log)
    members = frozenset({"engine:c3"})
    assert pool.drain(members, time.time() + 30) is True
    recs = _side_effects(log)
    assert [r["kind"] for r in recs] == ["tool_side_effect"]  # written BEFORE the wait
    assert recs[0]["trajectory_id"] == INJECTED_TOOL_WAIT_ID
    assert recs[0]["tool_call_id"] == injected_tool_call_id(members) == f"{INJECTED_TOOL_WAIT_ID}:drain:engine:c3"
    assert recs[0]["seq"] == 1 and recs[0]["attempt"] == 1 and recs[0]["target_members"] == ["engine:c3"]
    assert isinstance(recs[0]["wall_time"], float) and "monotonic" in recs[0]
    assert records[0][1]["side_effect_log"] is True and records[0][1]["tool_call_id"] == recs[0]["tool_call_id"]
    # the cancel path (drain timeout -> undrain) adds no execution
    calls = []

    async def uncordon_cells(cells):
        calls.append(sorted(cells))

    pool._controller.uncordon_cells = uncordon_cells
    pool.undrain(members)
    assert calls == [["c3"]]
    deadline = time.time() + 5
    while board.snapshot().in_flight and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)
    recs = _side_effects(log)
    assert [r["kind"] for r in recs] == ["tool_side_effect", "tool_complete"]
    assert recs[1]["seq"] == 2 and recs[1]["tool_call_id"] == recs[0]["tool_call_id"]
    assert side_effect_duplicates(recs) == []
    # a replayed drain of the same members IS visible: a second record of the same pair
    pool.drain(members, time.time() + 30)
    recs = _side_effects(log)
    assert side_effect_duplicates(recs) == [(INJECTED_TOOL_WAIT_ID, injected_tool_call_id(members))]
    assert recs[-1]["attempt"] == 2 and records[-1][1]["applied"] is True
    # a replay while the first execution is still waiting is journaled too (applied=False)
    pool.drain(members, time.time() + 30)
    assert _side_effects(log)[-1]["attempt"] == 3 and records[-1][1]["applied"] is False


def test_pool_accepts_a_path_and_entry_wires_it_from_the_env(monkeypatch, tmp_path):
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.rollout import SIDE_EFFECT_LOG_ENV
    from yeto.rl.engine.tool_wait import ToolSideEffectLog, ToolWaitBoard

    pool, _ = _real_pool(monkeypatch, 1.0, ToolWaitBoard())
    pool2, _ = _real_pool(monkeypatch, None, None)
    assert pool._side_effects is None and pool2._side_effects is None
    from yeto.rl.engine.miles_adapter.rollout import HARNESS_NOT_AGENTIC, MilesRolloutPool

    p = MilesRolloutPool(inference_controller=None, rollout_executor=None, metadata=None,
                         expected_policy=lambda: (0, "h"), harness=HARNESS_NOT_AGENTIC,
                         side_effect_log=tmp_path / "se.jsonl")
    assert isinstance(p._side_effects, ToolSideEffectLog) and p._side_effects.path == str(tmp_path / "se.jsonl")
    elastic = SimpleNamespace(controller=SimpleNamespace(state_dir=tmp_path / "st"))
    monkeypatch.delenv(SIDE_EFFECT_LOG_ENV, raising=False)
    assert entry.side_effect_log_kwargs(elastic) == {} and entry.side_effect_log_kwargs(None) == {}
    monkeypatch.setenv(SIDE_EFFECT_LOG_ENV, "0")
    assert entry.side_effect_log_kwargs(elastic) == {}
    monkeypatch.setenv(SIDE_EFFECT_LOG_ENV, "1")
    assert entry.side_effect_log_kwargs(elastic) == {"side_effect_log": tmp_path / "st" / "side_effects.jsonl"}
    assert entry.side_effect_log_kwargs(None) == {}


def test_drain_timeout_cancel_does_not_replay_the_tool_and_training_continues(tmp_path):
    """3.3 X5 at the controller: the tool executes once during the drain of ``down``;
    drain timeout -> CANCELLED -> undrain; no second execution of the same
    (trajectory, tool call) after CANCELLED; the run finishes its rounds."""
    from yeto.rl.engine.tool_wait import ToolSideEffectLog, side_effect_duplicates

    log = ToolSideEffectLog(tmp_path / "state" / "side_effects.jsonl")
    waiting = {"n": 0}

    driver, ctl, fork, pool, *_ = _setup(tmp_path, rounds=6)
    fake_undrain = pool.undrain

    def drain(members, deadline):
        # the "tool": its external call happens here, before the wait
        log.record("traj-1", "call-1", target_members=sorted(members))
        waiting["n"] += 1
        fork.calls.append(("drain", tuple(sorted(members))))
        fork.cordoned |= set(members)
        return True  # router part drained; the tool wait (trajectory_load) blocks until the deadline

    def undrain(members):
        fake_undrain(members)
        waiting["n"] -= 1
        log.complete("traj-1", "call-1")

    pool.drain, pool.undrain = drain, undrain
    pool.load = lambda: {"active_requests": 0, "tool_wait": waiting["n"]}
    _up_then_down(driver, ctl, pool)
    assert ctl.status("down")["phase"] == CANCELLED and ctl.admission_open
    recs = log.records()
    assert [r["kind"] for r in recs] == ["tool_side_effect", "tool_complete"]
    assert [r["seq"] for r in recs] == [1, 2] and side_effect_duplicates(recs) == []
    calls = [c[0] for c in fork.calls]
    assert calls.count("drain") == 1 and calls.count("undrain") == 1 and "stop" not in calls[calls.index("drain"):]
    journal = read_journal(tmp_path / "state/reconfig")
    assert len([r for r in journal if r.get("kind") == "phase" and r.get("phase") == CANCELLED]) == 1
    assert [r for r in journal if r.get("kind") == "drain_timeout" and r.get("tool_wait") == 1]
    rounds = [e for e in _events(tmp_path) if e["event"] == "rl_round_trained"]
    assert len(rounds) >= 5  # training continued after the cancel


def test_tool_wait_injection_without_a_board_is_applied_false(monkeypatch):
    pool, records = _real_pool(monkeypatch, 0.4, None)
    pool.drain(frozenset({"engine:c1"}), time.time() + 30)
    (event, f), = records
    assert f["applied"] is False and "no ToolWaitBoard" in f["error"]


def test_injected_tool_wait_drives_drain_timeout_cancel_undrain_without_replay(tmp_path):
    """A trajectory in a tool call (real ToolWaitBoard) outlasts the drain budget: the
    scale-down is CANCELLED, undrained, and the tool-waiting trajectory is neither
    aborted nor executed a second time."""
    from yeto.rl.engine.tool_wait import ToolWaitBoard

    board = ToolWaitBoard()
    executed, aborted = [], []

    def load():
        snap = board.snapshot()
        return {"active_requests": 0, "tool_wait": snap.in_flight,
                "blockers": ([f"{snap.in_flight} trajectories waiting on tools"]
                             if snap.in_flight else [])}

    driver, ctl, fork, pool, publisher, trained, clock = _setup(
        tmp_path, load=load,
        controller_kw={"timeouts": Timeouts(drain=5.0, retry_interval=1.0)})
    release = threading.Event()

    def tool_task():  # one trajectory inside a tool call; runs exactly once
        executed.append("t0")
        board.enter("t0")
        release.wait(10)
        board.exit("t0")

    pool.abort = lambda *a, **k: aborted.append(a)

    def before_down():
        threading.Thread(target=tool_task, daemon=True).start()
        while not board.snapshot().in_flight:
            time.sleep(0.01)

    _up_then_down(driver, ctl, pool, before_down=before_down)
    assert ctl.status("down")["phase"] == CANCELLED and "drain" in ctl.status("down")["error"]
    assert _journal(tmp_path, "drain_timeout")[0]["tool_wait"] == 1
    assert ("undrain", ("engine:c2", "engine:c3")) in fork.calls
    assert not [c for c in fork.calls if c[0] == "stop"]  # nothing of the removed was stopped
    assert board.snapshot().in_flight == 1 and board.snapshot().exited_total == 0  # still waiting
    release.set()
    deadline = time.time() + 5
    while board.snapshot().in_flight and time.time() < deadline:
        time.sleep(0.01)
    assert executed == ["t0"] and aborted == []
    assert board.snapshot().entered_total == 1 and board.snapshot().exited_total == 1
    assert driver.config_epoch == 1  # the up edge committed, the down edge did not
