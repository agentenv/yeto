"""An island a trainer breaker, a strict check or a refusal stopped is never relaunched.

A trainer that stopped on ``trainer.sample_wait`` (no new group in the buffer) exited 1 like any
crash: the decision table read it as an infrastructure failure and the fleet controller ordered
the island again, and again (an island pool had no relaunch limit); Modal's own function retries
ran the failed container up to 10 more times first. Each relaunch billed a GPU machine that
waited for the same samples and stopped the same way. Now:

* a breaker's stop marks the error it raises (``trainer_breakers.stop_error``); the island
  process exits ``ISLAND_STOP_EXIT`` with a ``[yeto-island-stop]`` line for it, for a strict
  failure and for a refusal (LEAVE ``TRAINING_FAILED`` / ``CONFIG_MISMATCH``); the Modal
  function returns that code instead of raising (no Modal retry);
* the fleet controller reads why from the island's own tape records (its last trainer breaker
  stop, strict failure and LEAVE of the incarnation: ``TapeCollector.island_stop``), from the
  marker line or from the source's exit code, not from the rank's exit status, and gives the
  island up instead of relaunching it (any roster);
* an infrastructure failure is still relaunched; an island pool island is relaunched at most
  ``launch.island_relaunches`` times within its window, then given up.

All of it is ``launch.deterministic_stop``: on for a fleet (an island pool, an island launch of a
fleet, two or more trainer islands), off for a single trainer (the K3.1 customer's), which keeps
main's behavior (its exit retried by Modal, relaunched by the controller) unless
``--rl-deterministic-stop``. With it on, ``trainer.nan`` and ``trainer.policy_hash`` (a faulty
card may cause either) get one relaunch on a new machine, counted in the island's record and tape.
"""

from __future__ import annotations

import io
import json
import pickle

import pytest

import yeto.launcher as launcher
import yeto.modal_runner as mr
from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
from test_modal_runner import _cfg, fake_modal
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_launcher_island_failure import Clock, _setup
from yeto.breakers.registry import default_registry
from yeto.config_errors import (CONFIG_ERROR_EXIT, DETERMINISTIC_STOP_ENV, ISLAND_STOP_EXIT, ISLAND_STOP_MARKER,
                                ConfigError)
from yeto.launch import fleet
from yeto.launch import island_lifecycle as lc
from yeto.launch.fleet import FixedRosterIslandAbandoned, FleetController, IslandsFailedForGood
from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine import batch_reader, trainer_breakers
from yeto.rl.engine.buffer_payload import DamagedPayloadLimit, SampleBufferExhausted, SampleBufferRolloutError
from yeto.rl.engine.driver import PolicyAgeExceeded
from yeto.rl.event_echo import TapeCollector


@pytest.fixture
def island_switch(monkeypatch):
    """The island process of a launch with launch.deterministic_stop on (the launch sets the env)."""
    monkeypatch.setenv(DETERMINISTIC_STOP_ENV, "1")


# -- the stops' errors --------------------------------------------------------------------------------------

def _sample_wait():
    error = SampleBufferRolloutError("rollout 3: 0/4 groups after 300 s without a new group")
    batch_reader.trip_sample_wait(None, waited_s=301.0, limit_s=300.0, error=error)
    return error


def _buffer_damaged():
    try:
        batch_reader.check_damaged_run(run=8, damaged=8, decoded=0, limit=8, position=11, sample_id="s",
                                       reason="schema", emit=None)
    except DamagedPayloadLimit as error:
        return error
    raise AssertionError("no raise")


def _sample_alignment():
    return batch_reader.unconvertible_round(3, 4, "R3 routing rows", emit=lambda *a, **k: None)


def _compaction_round():
    return batch_reader.refuse_compaction_round("segments are not rollout siblings", emit=lambda *a, **k: None)


def _release_drain():
    from yeto.rl.engine.trainer_remote import RemoteTrainerError

    return trainer_breakers.stop_error(RemoteTrainerError("upload still running"), trainer_breakers.RELEASE_DRAIN)


BREAKER_STOPS = {"trainer.sample_wait": _sample_wait, "trainer.buffer_damaged": _buffer_damaged,
                 "trainer.sample_alignment": _sample_alignment, "trainer.compaction_round": _compaction_round,
                 "trainer.release_drain": _release_drain}


@pytest.mark.parametrize("name", sorted(BREAKER_STOPS))
def test_a_trainer_breakers_stop_is_never_relaunched(name):
    error = BREAKER_STOPS[name]()
    assert trainer_breakers.breaker_of(error) == name
    found = lc.classify_error(error)
    assert (found.cause, found.action, found.state) == (lc.BREAKER, lc.FAIL, lc.FAILED)
    assert name in found.detail
    assert not found.relaunch


def test_the_mark_survives_pickling_a_chain_and_a_ray_task_error():
    error = pickle.loads(pickle.dumps(_sample_wait()))
    assert trainer_breakers.breaker_of(error) == "trainer.sample_wait"
    try:
        try:
            raise _sample_wait()
        except SampleBufferRolloutError as inner:
            raise RuntimeError("rollout manager failed") from inner
    except RuntimeError as outer:
        assert lc.classify_error(outer).cause == lc.BREAKER

    class RayTaskErrorLike(RuntimeError):  # ray.exceptions.RayTaskError keeps the actor's error as .cause
        def __init__(self, cause):
            super().__init__("ray task failed")
            self.cause = cause

    assert lc.classify_error(RayTaskErrorLike(_sample_wait())).cause == lc.BREAKER


def _training_failed_members():
    """Every LEAVE TRAINING_FAILED error class (``leave_reason._TABLE``) -> an instance, and
    whether it is a deterministic stop (exit 11, never relaunched) or keeps its retries."""
    from yeto.rl.applied_lr import AppliedLrError
    from yeto.rl.adapters.miles.publish import PublicationError as MilesPublicationError
    from yeto.rl.critic_state import CriticStateError
    from yeto.rl.engine.bridges import CrossChannelCommitError
    from yeto.rl.engine.driver import DriverError, PolicyIdentityError, PublicationError, RoundFailedError
    from yeto.rl.engine.overlap import OverlapGuardError
    from yeto.rl.engine.policy_age import PolicyAgeError
    from yeto.rl.engine.policy_digest import PolicyDigestError
    from yeto.rl.engine.sample_pool import SampleIndexError
    from yeto.rl.engine.version_segments import ProvenanceError
    from yeto.rl.grad_audit import GradAuditError
    from yeto.rl.tbench_outcome import UntrustedTBenchOutcome
    from yeto.rl.engine.ports import PublicationCause
    from yeto.rl.peer_recovery import RecoveryIncomplete

    return {
        # deterministic: the same check fails the same way on every attempt
        "StrictRlInvariantError": (StrictRlInvariantError("layout_hash", "layout hash differs"), True),
        "StrictRlInvariantError (HELLO refused)": (StrictRlInvariantError(
            "syncer_profile_mismatch", "syncer refused this island: HELLO syncer semantic profile does not match"), True),
        "StrictRlInvariantError (nan)": (StrictRlInvariantError("nonfinite_grad_norm", "grad_norm is nan"), True),
        "PolicyAgeExceeded": (PolicyAgeExceeded("groups older than max_policy_age"), True),
        "PolicyAgeError": (PolicyAgeError("max_policy_age must be an int"), True),
        "AppliedLrError": (AppliedLrError("LR 0 before the final round"), True),
        "RecoveryIncomplete": (RecoveryIncomplete("actor recovery did not receive every fragment"), True),
        # may be an NCCL / CUDA / OOM / storage / network fault underneath: retried as before
        "RoundFailedError": (RoundFailedError("rollout 7: optimizer step failed"), False),
        "PublicationError (update failed)": (PublicationError(
            "engine update_weights raised", cause=PublicationCause.UPDATE_FAILED), False),
        "PolicyIdentityError": (PolicyIdentityError("groups not sampled from the published policy"), False),
        "DriverError": (DriverError("rollout 7: the batch generated ahead is rollout 6"), False),
        "MilesPublicationError": (MilesPublicationError("publication refused"), False),
        "CrossChannelCommitError": (CrossChannelCommitError("critic channel did not reach v+1"), False),
        "PolicyDigestError": (PolicyDigestError("weights no longer match the digest"), False),
        "OverlapGuardError": (OverlapGuardError("an overlapped eval would observe the wrong policy"), False),
        "ProvenanceError": (ProvenanceError("version segment out of order"), False),
        "SampleIndexError": (SampleIndexError("sample index out of range"), False),
        "GradAuditError": (GradAuditError("grad audit failed"), False),
        "CriticStateError": (CriticStateError("critic state does not load"), False),
        "UntrustedTBenchOutcome": (UntrustedTBenchOutcome("terminal-bench metadata is absent"), False),
    }


@pytest.mark.parametrize("member", sorted(_training_failed_members()))
def test_each_training_failed_error_is_a_stop_only_on_the_whitelist(member):
    from yeto.rl.engine.leave_reason import leave_reason_for

    error, deterministic = _training_failed_members()[member]
    assert leave_reason_for(error).name == "TRAINING_FAILED"
    found = lc.island_stop_decision(error)
    assert (found is not None) == deterministic
    if deterministic:
        assert (found.cause, found.action) == (lc.STRICT, lc.FAIL)
    record = {"leave": {"reason_name": "TRAINING_FAILED", "error_type": type(error).__name__, "error": str(error)}}
    assert (lc.classify_records(record) is not None) == deterministic


def test_every_training_failed_class_is_judged():
    """A class added to the TRAINING_FAILED row must be judged here (whitelisted or not)."""
    from yeto.rl.engine import leave_reason
    from yeto.rl.leave_code import LeaveReason

    (row,) = [names for reason, names in leave_reason._TABLE if reason is LeaveReason.TRAINING_FAILED]
    judged = {type(e).__name__ for e, _ in _training_failed_members().values()} | {"PublicationError"}
    assert {name for _module, name in row} <= judged | {"DriverError"}


@pytest.mark.parametrize("error", [
    RuntimeError("[Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=88, OpType=ALLREDUCE) "
                 "ran for 1800000 milliseconds before timing out"),
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    OSError("[Errno 5] Input/output error: '/store/ckpt/round-7'"),
    ConnectionResetError("Connection reset by peer"),
    TimeoutError("distributed init timed out"),
])
def test_a_transient_fault_keeps_its_modal_retries_and_relaunches(error, monkeypatch, island_switch):
    """NCCL / distributed timeouts, CUDA and driver errors, OOM, storage I/O, the network: the
    island process exits 1 as before (Modal retries it), and the launcher relaunches it."""
    from yeto.rl.adapters.miles.island_entry import island_stop_exit

    for wrapped in (error, _wrapped(error)):
        assert island_stop_exit(wrapped, io.StringIO()) is None
        assert lc.island_stop_decision(wrapped) is None
        assert lc.classify_error(wrapped).relaunch
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 1)
    with pytest.raises(RuntimeError, match="exited with 1"):  # raised: Modal retries the container
        mr.island_main(_cfg(num_nodes=1, run_script="exit 1").to_json())


def _wrapped(error):
    from yeto.rl.engine.driver import RoundFailedError

    try:
        try:
            raise error
        except BaseException as inner:
            raise RoundFailedError("rollout 7: optimizer step failed") from inner
    except RoundFailedError as outer:
        return outer


@pytest.mark.parametrize("error", [
    RuntimeError("NCCL error: unhandled system error"),
    # an unmarked reader error (a sampler HTTP error, a store that did not answer) is not a stop
    SampleBufferRolloutError("http://sampler:30000/generate: HTTP 503: busy"),
])
def test_an_infrastructure_failure_is_still_relaunched(error):
    found = lc.classify_error(error)
    assert (found.cause, found.action) == (lc.INFRA, lc.RELAUNCH)


def test_the_samplers_normal_end_is_not_a_stop():
    assert trainer_breakers.breaker_of(SampleBufferExhausted("the samplers ended the buffer")) is None


# -- the island process: its exit code and marker line ----------------------------------------------------------

def test_the_island_entry_exits_island_stop_for_a_deterministic_stop(island_switch):
    from yeto.rl.adapters.miles.island_entry import config_error_exit, island_stop_exit

    stream = io.StringIO()
    try:
        raise _sample_wait()
    except SampleBufferRolloutError as error:
        assert island_stop_exit(error, stream) == ISLAND_STOP_EXIT
    text = stream.getvalue()
    assert "Traceback" in text and f"{ISLAND_STOP_MARKER} {lc.BREAKER} trainer.sample_wait" in text
    assert island_stop_exit(StrictRlInvariantError("layout_hash", "x"), io.StringIO()) == ISLAND_STOP_EXIT
    assert island_stop_exit(RuntimeError("CUDA out of memory"), io.StringIO()) is None
    # a configuration error keeps its own exit (10), not this one
    assert island_stop_exit(ConfigError("bad option"), io.StringIO()) is None
    assert config_error_exit(ConfigError("bad option"), io.StringIO()) == CONFIG_ERROR_EXIT


def test_the_exit_code_is_distinct_from_every_registered_exit():
    registered = {fleet.ISLAND_FAILED_EXIT, fleet.TEARDOWN_UNVERIFIED_EXIT, fleet.RUN_STALLED_EXIT,
                  fleet.CONTAINER_CHANGED_EXIT, fleet.TRAINER_GAVE_UP_EXIT, fleet.RUN_STOPPED_EXIT,
                  launcher.NO_SYNC_INCOMPLETE_EXIT, launcher.ISLAND_RECORDS_LOCKED_EXIT,
                  launcher.ISLAND_RECORDS_UNKNOWN_EXIT, CONFIG_ERROR_EXIT, 0, 1, 2}
    from yeto.wan_probe import EXIT_SLOW_LINK

    assert ISLAND_STOP_EXIT == 11 and ISLAND_STOP_EXIT not in registered | {EXIT_SLOW_LINK}


def test_the_exit_code_and_the_marker_line_are_never_relaunched():
    assert lc.classify(exit_code=ISLAND_STOP_EXIT).action == lc.FAIL
    line = f"{ISLAND_STOP_MARKER} strict StrictRlInvariantError: x (exit 11, not retried)"
    assert (lc.classify(log_text=f"noise\n{line}").cause, lc.classify(log_text=line).action) == (lc.STRICT, lc.FAIL)
    line = f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: no group (exit 11, not retried)"
    assert lc.classify(log_text=line).cause == lc.BREAKER


def test_the_modal_function_returns_an_island_stop_instead_of_raising(monkeypatch):
    """A raise is retried by Modal (up to 10 times, a new GPU container each); a return ends the call."""
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: ISLAND_STOP_EXIT)
    assert mr.island_main(_cfg(num_nodes=1, run_script="exit 11").to_json()) == ISLAND_STOP_EXIT
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 1)
    with pytest.raises(RuntimeError, match="exited with 1"):
        mr.island_main(_cfg(num_nodes=1, run_script="exit 1").to_json())


def test_modal_status_reads_a_returned_island_stop_as_failed(monkeypatch, capsys):
    state = fake_modal(monkeypatch)
    ops = mr.ModalOps("yeto-run")
    cfg = _cfg()
    ops.define(cfg)
    cid = ops.spawn(cfg)
    state["calls"][cid].get = lambda timeout=None, index=0: ISLAND_STOP_EXIT
    assert ops.status(cid) == "FAILED"
    assert "deterministic stop" in capsys.readouterr().err


def test_the_log_probes_report_the_marker():
    from yeto.launch.ssh_resident import STRICT_MARKERS

    assert ISLAND_STOP_MARKER in STRICT_MARKERS
    ops = mr.ModalIslandOps(type("T", (), {"tail_logs": lambda self, job_id, entries=400: [
        "noise", f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: x"]})())
    assert ops.rl_strict_failure("island", "fc-1").startswith(ISLAND_STOP_MARKER)


# -- the island's own records (the launcher's tape echo) ----------------------------------------------------

def _echo(event, **fields):
    return "YETO_RL_EVENT " + json.dumps({"event": event, "island_id": 1, "time_unix": 1.0, **fields},
                                         sort_keys=True)


BREAKER_RECORD = {"layer": "trainer", "action": "stop", "state": "trip", "name": "trainer.sample_wait",
                  "stop_reason": "breaker:trainer.sample_wait", "detail": "no new group in 300 s"}


def test_the_tape_echo_keeps_the_incarnations_stop_records(tmp_path):
    tape = TapeCollector(tmp_path / "l1.jsonl")
    tape.feed(_echo("rl_driver_start") + "\n")
    assert tape.island_stop is None
    # a drop (an action that does not stop), a sampler's trip: no stop
    tape.feed(_echo("rl_breaker", layer="trainer", action="drop", state="trip", name="trainer.policy_age") + "\n")
    tape.feed(_echo("rl_breaker", layer="sampler", action="stop", state="trip", name="sampler.loop_guard") + "\n")
    assert tape.island_stop is None
    tape.feed(_echo("rl_breaker", **BREAKER_RECORD) + "\n")
    tape.feed(_echo("island_leave", reason=4, reason_name="INFRA_FAILED", error_type="SampleBufferRolloutError",
                    error="no new group", breaker="trainer.sample_wait") + "\n")
    stop = tape.island_stop
    assert stop["breaker"]["name"] == "trainer.sample_wait" and stop["leave"]["breaker"] == "trainer.sample_wait"
    assert lc.classify_records(stop).cause == lc.BREAKER
    # the relaunched incarnation starts clean
    tape.feed(_echo("rl_driver_start", incarnation=2) + "\n")
    assert tape.island_stop is None


@pytest.mark.parametrize("records,cause", [
    ({"breaker": BREAKER_RECORD}, lc.BREAKER),
    ({"strict": {"error": "StrictRlInvariantError: syncer refused this island"}}, lc.STRICT),
    ({"leave": {"reason_name": "TRAINING_FAILED", "error_type": "StrictRlInvariantError", "error": "x"}}, lc.STRICT),
    ({"leave": {"reason_name": "CONFIG_MISMATCH", "error_type": "ProfileError", "error": "x"}}, lc.CONFIG_ERROR),
    ({"leave": {"reason_name": "INFRA_FAILED", "error_type": "SampleBufferRolloutError", "error": "x",
                "breaker": "trainer.sample_wait"}}, lc.BREAKER),
    ({"leave": {"reason_name": "INFRA_FAILED", "error_type": "RuntimeError", "error": "NCCL"}}, None),
    (None, None),
])
def test_the_records_decide(records, cause):
    found = lc.classify_records(records)
    assert (found.cause if found is not None else None) == cause
    if found is not None:
        assert found.action == lc.FAIL


# -- the fleet controller ----------------------------------------------------------------------------------

class FleetLog:
    def __init__(self):
        self.events = []

    def event(self, kind, name=None, **fields):
        self.events.append((kind, name, fields))

    def maybe_tick(self):
        pass

    def cost_tick(self):
        pass


def _controller(ops, learners, *, records=None, elastic=False, fixed_roster=False, log=None, deterministic=True,
                probe=None, incarnations=None):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(
        learners={name: (f"task-{name}", job) for name, job in learners.items()},
        syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30, recover_timeout=900,
        thread_cls=ImmediateThread, elastic=elastic, fixed_roster=fixed_roster, fleet_log=log,
        stop_probe=probe or (None if records is None else (lambda name: records.get(name))),
        deterministic_stop=deterministic, incarnations=incarnations)


STOPS = {
    "trainer.sample_wait": {"breaker": BREAKER_RECORD},
    "trainer.buffer_damaged": {"breaker": {**BREAKER_RECORD, "name": "trainer.buffer_damaged"}},
    "trainer.sample_alignment": {"breaker": {**BREAKER_RECORD, "name": "trainer.sample_alignment"}},
    "trainer.compaction_round": {"breaker": {**BREAKER_RECORD, "name": "trainer.compaction_round"}},
    "trainer.policy_age": {"leave": {"reason_name": "TRAINING_FAILED", "error_type": "PolicyAgeExceeded",
                                     "error": "groups older than max_policy_age"}},
    "trainer.strict_invariant": {"breaker": {**BREAKER_RECORD, "name": "trainer.strict_invariant", "kind": "invariant"},
                                 "strict": {"error": "StrictRlInvariantError: layout hash"}},
    "profile refused": {"leave": {"reason_name": "CONFIG_MISMATCH", "error_type": "ProfileError", "error": "x"}},
    "HELLO refused": {"strict": {"error": "StrictRlInvariantError: syncer refused this island: HELLO refused"},
                      "leave": {"reason_name": "TRAINING_FAILED", "error_type": "StrictRlInvariantError"}},
}


@pytest.mark.parametrize("stop", sorted(STOPS))
@pytest.mark.parametrize("mode", ["island pool", "fixed roster single island", "single island"])
def test_an_island_a_breaker_or_refusal_stopped_is_given_up_not_relaunched(stop, mode):
    """Island pool: the pool goes on. Fixed roster (an RL launch without island pool options, the
    K3.1 single trainer island): the launch ends as an island given up (exit 4). A roster that is
    neither (one island alone): the launch ends with the reasons (a fleet island's launch exits 11)."""
    ops = FakeOps()
    learners = {"l0": 1, "l1": 2} if mode == "island pool" else {"l1": 2}
    ops.status_seq["l0"] = [RUNNING] * 6 + [SUCCEEDED]
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2", "j3"]
    log = FleetLog()
    ctl = _controller(ops, learners, records={"l1": STOPS[stop]}, elastic=mode == "island pool",
                      fixed_roster=mode == "fixed roster single island", log=log)
    if mode == "island pool":
        ctl.run()
    elif mode == "fixed roster single island":
        with pytest.raises(FixedRosterIslandAbandoned):
            ctl.run()
    else:
        with pytest.raises(IslandsFailedForGood) as ended:
            ctl.run()
        assert str(ended.value) == "all learners abandoned; nothing left to train"
        assert set(ended.value.reasons) == {"l1"}
    assert ops.relaunch_calls == []  # never ordered again
    assert ctl.learners["l1"]["state"] == "abandoned"
    assert ctl.learners["l1"]["decision"].action == lc.FAIL
    assert ops.down_calls.count("l1") >= 1  # released
    assert any(kind == "island_lost" and name == "l1" for kind, name, _ in log.events)
    if mode == "island pool":
        assert ctl.learners["l0"]["state"] == "done"  # the pool goes on


def test_a_fixed_roster_ends_the_fleet_on_a_breaker_stop_without_a_relaunch():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ctl = _controller(ops, {"l0": 1, "l1": 2}, records={"l1": STOPS["trainer.sample_wait"]}, fixed_roster=True)
    with pytest.raises(FixedRosterIslandAbandoned):
        ctl.run()
    assert ops.relaunch_calls == []


def test_an_island_without_an_echo_is_given_up_on_its_marker_line():
    ops = FakeOps()
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ops.rl_strict_failure = lambda name, job: f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: no group"
    ctl = _controller(ops, {"l1": 2})
    with pytest.raises(IslandsFailedForGood):
        ctl.run()
    assert ops.relaunch_calls == [] and ctl.learners["l1"]["state"] == "abandoned"


@pytest.mark.parametrize("records", [
    None,
    {"l1": {"leave": {"reason_name": "INFRA_FAILED", "error_type": "RuntimeError", "error": "NCCL error"}}},
    # a failed optimizer step (LEAVE TRAINING_FAILED, not whitelisted): an NCCL timeout underneath
    {"l1": {"leave": {"reason_name": "TRAINING_FAILED", "error_type": "RoundFailedError",
                      "error": "rollout 7: optimizer step failed"}}},
    {"l1": {"leave": {"reason_name": "TRAINING_FAILED", "error_type": "RayTaskError(RoundFailedError)",
                      "error": "optimizer step failed"}}},
])
def test_an_infrastructure_failure_is_relaunched(records):
    ops = FakeOps()
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ops.after_relaunch["l1"] = [RUNNING, SUCCEEDED]
    ctl = _controller(ops, {"l1": 2}, records=records or {}, elastic=True)
    ctl.run()
    assert ops.relaunch_calls == ["l1"] and ctl.learners["l1"]["state"] == "done"


# -- the island pool's relaunch limit -----------------------------------------------------------------------

def _keeps_failing(run_polls, *, poll_offset=0):
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING] * 400 + [SUCCEEDED]
    ops.status_seq["l1"] = [RUNNING] * run_polls + [FAILED]
    ops.relaunch_results["l1"] = [f"j{i}" for i in range(100)]
    ops.after_relaunch["l1"] = [RUNNING] * run_polls + [FAILED]
    return ops


@pytest.mark.parametrize("run_polls", [4, 12])  # 120 s after each boot; 360 s (> RECOVERY_STABLE_S)
def test_an_island_pool_island_that_keeps_failing_is_given_up_after_the_limit(run_polls):
    ops = _keeps_failing(run_polls)
    log = FleetLog()
    ctl = _controller(ops, {"l0": 1, "l1": 2}, elastic=True, log=log)
    ctl.run()
    assert len(ops.relaunch_calls) == fleet.ISLAND_MAX_RELAUNCHES == 3
    assert ctl.learners["l1"]["state"] == "abandoned" and ctl.learners["l0"]["state"] == "done"
    limits = [f for kind, name, f in log.events if kind == "island_relaunch_limit" and name == "l1"]
    assert len(limits) == 1 and limits[0]["limit"] == 3 and limits[0]["window_s"] == 3600.0


def test_relaunches_spread_beyond_the_window_go_on():
    """Failures more than the window apart: each relaunch is the first of its window."""
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING] * 4000 + [SUCCEEDED]
    stay = int(fleet.ISLAND_RELAUNCH_WINDOW_S / 30) + 10  # polls of 30 s each incarnation runs
    ops.status_seq["l1"] = [RUNNING] * stay + [FAILED]
    ops.relaunch_results["l1"] = [f"j{i}" for i in range(10)]
    ops.after_relaunch["l1"] = [RUNNING] * stay + [FAILED]

    def relaunched(name, job):
        if len(ops.relaunch_calls) == 5:  # the fifth incarnation ends normally
            ops.status_seq["l1"] = [RUNNING, SUCCEEDED]
            ops.status_seq["l0"] = [SUCCEEDED]

    ops.sleep = lambda seconds: setattr(ops, "t", ops.t + seconds)  # many polls: no 500-poll guard
    ctl = _controller(ops, {"l0": 1, "l1": 2}, elastic=True)
    ctl.on_relaunch = relaunched
    ctl.run()
    assert len(ops.relaunch_calls) == 5 and ctl.learners["l1"]["state"] == "done"


def test_the_registry_holds_the_limit_the_controller_applies():
    (entry,) = [b for b in default_registry() if b.name == "launch.island_relaunches"]
    params = {p.name: p.default for p in entry.params}
    assert params == {"max_n": fleet.ISLAND_MAX_RELAUNCHES, "window_s": fleet.ISLAND_RELAUNCH_WINDOW_S}
    assert entry.kind == "retry_budget" and not entry.can_disable


# -- the customer's shape: one Modal island, no syncer -------------------------------------------------------

def test_a_single_modal_island_a_breaker_stopped_is_not_spawned_again(monkeypatch, tmp_path):
    """``yeto launch --rl-deterministic-stop`` of one Modal island without a syncer (the K3.1 async
    GRPO trainer's shape, switched on): an island that stopped on its sample-wait breaker is
    released and the launch ends; no second container."""
    record, clock = [], Clock()
    line = f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: no new group (exit 11, not retried)"
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record, log_lines=("noise", line))
    extra = ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
             "--rl-single-island-no-sync", "--rl-deterministic-stop")
    args = _launcher_args("ports", extra, gpu="modal:1xa100")
    assert args.rl_deterministic_stop
    args.keep = False
    args.recover_timeout = 1200.0  # the default: recovery on, still no relaunch
    args.controller_poll = 30.0
    assert launcher.run(args) == fleet.ISLAND_FAILED_EXIT  # an island given up (fixed roster), as before
    assert not [r for r in record if r[0] == "relaunch"]
    assert record.count(("spawn", 0)) == 1
    assert record.count(("stop_app",)) == 1


def test_a_lone_island_pool_island_given_up_on_the_limit_ends_the_launch_for_good():
    """The relaunch limit is a decision too: a fleet island's launch whose island hit it is not
    started again by the run (it ends with the reasons; a fleet island's launch exits 11)."""
    ops = _keeps_failing(4)
    ctl = _controller(ops, {"l1": 2}, elastic=True)
    with pytest.raises(IslandsFailedForGood) as ended:
        ctl.run()
    assert "relaunch limit" in ended.value.reasons["l1"]
    assert len(ops.relaunch_calls) == 3 and ctl.learners["l1"]["decision"].state == lc.ABANDONED


def test_a_fleet_islands_launch_ends_with_the_island_stop_exit(monkeypatch, tmp_path):
    """``yeto launch`` of one fleet island (``--rl-fleet-role island``) whose island a breaker
    stopped: exit 11, so the run's trainer worker does not start it again."""
    record, clock = [], Clock()
    line = f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: no new group (exit 11, not retried)"
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record, log_lines=("noise", line))
    extra = ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
             "--rl-single-island-no-sync")
    args = _launcher_args("ports", extra, gpu="modal:1xa100")
    args.keep = False
    args.recover_timeout = 1200.0
    args.controller_poll = 30.0
    args.rl_fleet_role = "island"
    args.rl_fleet_learner_id, args.rl_fleet_learners = 0, 1
    monkeypatch.setattr(launcher, "controller_fixed_roster", lambda args: False)  # an island pool island
    # the fleet member checks (a synced full-parameter trainer fed by external pools, its learner id,
    # the fleet's learners) are not what this test is about: the exit of the launch is
    from yeto.rl.engine import sample_roles

    monkeypatch.setattr(sample_roles, "check_fleet_member", lambda args: None)
    assert launcher.run(args) == ISLAND_STOP_EXIT
    assert not [r for r in record if r[0] == "relaunch"] and record.count(("spawn", 0)) == 1


# -- launch.deterministic_stop: on for a fleet, off (main's behavior) for a single trainer ----------------------

@pytest.mark.parametrize("extra,gpu,on", [
    ((), "modal:1xa100", False),  # a single trainer island (the K3.1 customer's shape)
    (("--rl-single-island-no-sync",), "modal:1xa100", False),
    (("--rl-deterministic-stop",), "modal:1xa100", True),  # switched on for a single trainer
    ((), "modal:1xa100,modal:1xa100", True),  # two trainer islands: a fleet
    (("--rl-sync-island-lease-s", "150"), "modal:1xa100", True),  # an island pool option
    (("--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer"), "modal:1xa100", True),  # a fleet plan
    (("--rl-sample-role", "trainer"), "modal:1xa100", False),  # the K3.1 buffer trainer: a single trainer
])
def test_the_switch_is_on_for_a_fleet_and_off_for_a_single_trainer(extra, gpu, on):
    args = _launcher_args("ports", extra, gpu=gpu)
    assert fleet.deterministic_stop(args)[0] is on
    assert fleet.deterministic_stop_env(args) == ({DETERMINISTIC_STOP_ENV: "1"} if on else {})


def test_an_island_launch_of_a_fleet_has_the_switch_on():
    args = _launcher_args("ports", (), gpu="modal:1xa100")
    args.rl_fleet_role = "island"
    assert fleet.deterministic_stop(args) == (True, "an island launch of a fleet")


def test_switched_off_the_island_raises_every_stop_as_before(monkeypatch):
    """No env (a single trainer): the island process prints no marker and raises (exit 1), so
    Modal retries it and its log carries nothing the launcher's probes read as a stop."""
    from yeto.rl.adapters.miles.island_entry import island_stop_exit

    monkeypatch.delenv(DETERMINISTIC_STOP_ENV, raising=False)
    for error in (_sample_wait(), StrictRlInvariantError("layout_hash", "x"),
                  StrictRlInvariantError("nonfinite_grad_norm", "grad_norm=nan")):
        stream = io.StringIO()
        assert island_stop_exit(error, stream) is None and stream.getvalue() == ""


@pytest.mark.parametrize("stop", sorted(STOPS))
@pytest.mark.parametrize("mode", ["island pool", "fixed roster single island", "single island"])
def test_switched_off_the_controller_relaunches_as_on_main(stop, mode):
    """launch.deterministic_stop off: the same records change nothing; the island is relaunched
    exactly as main relaunches it (an exit 1 and no strict line in its log)."""
    ops = FakeOps()
    learners = {"l0": 1, "l1": 2} if mode == "island pool" else {"l1": 2}
    ops.status_seq["l0"] = [RUNNING] * 6 + [SUCCEEDED]
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ops.after_relaunch["l1"] = [RUNNING, SUCCEEDED]
    ctl = _controller(ops, learners, records={"l1": STOPS[stop]}, elastic=mode == "island pool",
                      fixed_roster=mode == "fixed roster single island", deterministic=False)
    ctl.run()
    assert ops.relaunch_calls == ["l1"] and ctl.learners["l1"]["state"] == "done"


def test_switched_off_a_lone_island_given_up_raises_the_plain_error():
    ops = FakeOps()
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = [None] * 50  # never comes back: abandoned on the recover timeout
    ctl = _controller(ops, {"l1": 2}, records={"l1": STOPS["trainer.sample_wait"]}, deterministic=False)
    with pytest.raises(RuntimeError) as ended:
        ctl.run()
    assert type(ended.value) is RuntimeError  # not IslandsFailedForGood: the fleet island's launch exits as before


def test_switched_off_the_customer_launch_relaunches_as_on_main(monkeypatch, tmp_path):
    """``yeto launch`` of one Modal island without a syncer and without --rl-deterministic-stop
    (the K3.1 customer's launch): a trainer that stopped on trainer.sample_wait is relaunched
    as main relaunches it (the fixed roster's two relaunches, then exit 4)."""
    record, clock = [], Clock()
    lines = ("noise", "yeto.rl.engine.buffer_payload.SampleBufferRolloutError: rollout 3: 0/4 groups after "
             "10800 s without a new group")
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record, log_lines=lines)
    extra = ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
             "--rl-single-island-no-sync")
    args = _launcher_args("ports", extra, gpu="modal:1xa100")
    args.keep = False
    args.recover_timeout = 1200.0
    args.controller_poll = 30.0
    assert not fleet.deterministic_stop(args)[0]
    assert launcher.run(args) == fleet.ISLAND_FAILED_EXIT
    assert record.count(("spawn", 0)) == 1 + fleet.FIXED_ROSTER_MAX_RELAUNCHES


def test_switched_off_the_trainer_scheduler_reads_its_failures_as_on_main():
    error = RuntimeError("NCCL timeout")
    from yeto.rl.engine.driver import RoundFailedError

    try:
        raise RoundFailedError("optimizer step failed") from error
    except RoundFailedError as failed:
        assert lc.loop_classifier(None, failed) == lc.loop_classifier(None, failed, deterministic=False)
        assert lc.loop_classifier(None, _sample_wait(), deterministic=False) == (lc.INFRA, True)
        assert lc.loop_classifier(None, _sample_wait(), deterministic=True) == (lc.BREAKER, False)


def test_the_registry_holds_the_switch_off_by_default():
    reg = default_registry()
    entry = reg.get("launch.deterministic_stop")
    assert not entry.default_enabled and entry.can_disable
    assert {p.name: p.default for p in entry.params} == {"new_machine_retries_n": lc.NEW_MACHINE_RETRIES}
    assert [a.path for a in entry.aliases] == ["--rl-deterministic-stop"]
    assert not reg.get("launch.island_relaunches").default_enabled  # an island pool turns it on (native)


# -- trainer.nan / trainer.policy_hash: one relaunch on a new machine ---------------------------------------------

NAN_LINES = [("rl_breaker", {"layer": "trainer", "action": "stop", "state": "trip", "name": "trainer.nan",
                             "stop_reason": "breaker:trainer.nan", "detail": "nan_skip_breaker: 3 in a row"}),
             ("rl_strict_failure", {"metric": "nan_skip_breaker", "value": 1,
                                    "error": "StrictRlInvariantError: rollout 9: 3 skipped rounds in a row"}),
             ("island_leave", {"reason": 3, "reason_name": "TRAINING_FAILED", "error_type": "StrictRlInvariantError",
                               "error": "rollout 9: 3 skipped rounds in a row", "breaker": "trainer.nan"})]


def _nan_incarnation(tape, incarnation):
    tape.feed(_echo("rl_driver_start", incarnation=incarnation) + "\n")
    for event, fields in NAN_LINES:
        tape.feed(_echo(event, incarnation=incarnation, **fields) + "\n")


class _NanIsland:
    """An island whose every incarnation stops on trainer.nan (its tape fed per incarnation)."""

    def __init__(self, ops, tape, *, first=1):
        self.ops, self.tape, self.n = ops, tape, first
        _nan_incarnation(tape, self.n)
        ops.status_seq["l1"] = [RUNNING, FAILED]
        ops.relaunch_results["l1"] = [f"j{i}" for i in range(10)]
        ops.after_relaunch["l1"] = [RUNNING, FAILED]

    def relaunched(self, name, job):
        self.n += 1
        _nan_incarnation(self.tape, self.n)


@pytest.mark.parametrize("mode", ["island pool", "fixed roster single island", "single island"])
def test_a_nan_stop_gets_one_relaunch_on_a_new_machine_and_the_second_ends_the_island(mode, tmp_path):
    ops, log = FakeOps(), FleetLog()
    tape = TapeCollector(tmp_path / "l1.jsonl")
    island = _NanIsland(ops, tape)
    learners = {"l0": 1, "l1": 2} if mode == "island pool" else {"l1": 2}
    ops.status_seq["l0"] = [RUNNING] * 20 + [SUCCEEDED]
    ctl = _controller(ops, learners, probe=lambda name: tape.island_stop if name == "l1" else None,
                      elastic=mode == "island pool", fixed_roster=mode == "fixed roster single island", log=log)
    ctl.on_relaunch = island.relaunched
    if mode == "island pool":
        ctl.run()
    else:
        with pytest.raises((FixedRosterIslandAbandoned, IslandsFailedForGood)):
            ctl.run()
    assert ops.relaunch_calls == ["l1"]  # once, never twice
    first_down = ops.down_calls.index("l1")
    assert first_down >= 0  # the old machine went down before the relaunch ordered a new one
    assert ctl.learners["l1"]["state"] == "abandoned" and ctl.learners["l1"]["decision"].action == lc.FAIL
    assert ctl.learners["l1"]["new_machine_retries"] == {"trainer.nan": 1}
    retries = [f for kind, name, f in log.events if kind == "island_new_machine_retry"]
    assert len(retries) == 1 and retries[0]["breaker"] == "trainer.nan" and retries[0]["limit"] == 1


def test_the_old_machine_goes_down_before_the_new_order():
    order = []
    ops = FakeOps()
    real_down, real_relaunch = ops.down, ops.relaunch
    ops.down = lambda name: order.append(("down", name)) or real_down(name)
    ops.relaunch = lambda task, name: order.append(("relaunch", name)) or real_relaunch(task, name)
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ops.after_relaunch["l1"] = [RUNNING, SUCCEEDED]
    records = {"l1": {"breaker": NAN_LINES[0][1], "strict": NAN_LINES[1][1], "leave": NAN_LINES[2][1],
                      "stops": {"trainer.nan": 1}}}
    ctl = _controller(ops, {"l1": 2}, records=records, elastic=True)
    ctl.run()
    assert order[:2] == [("down", "l1"), ("relaunch", "l1")] and ctl.learners["l1"]["state"] == "done"


def test_a_nan_stop_after_a_launcher_restart_counts_the_earlier_relaunch_from_the_record(tmp_path):
    """A restarted launcher reads the island's record (its new_machine_retries): the NaN stop after
    the restart is the second one, and ends the island."""
    ops = FakeOps()
    tape = TapeCollector(tmp_path / "l1.jsonl")
    _NanIsland(ops, tape, first=3)  # the restarted launcher's tape holds only this incarnation
    ctl = _controller(ops, {"l1": 2}, probe=lambda name: tape.island_stop, elastic=True,
                      incarnations={"l1": {"incarnation": 2, "relaunches": 1, "adopted": False,
                                           "new_machine_retries": {"trainer.nan": 1}}})
    with pytest.raises(IslandsFailedForGood):
        ctl.run()
    assert ops.relaunch_calls == []


def test_a_nan_stop_after_a_launcher_restart_counts_the_earlier_stop_from_the_tape(tmp_path):
    """No record (a single trainer has no run store record): a restarted launcher reads the
    island's tape file again (TapeCollector fresh=False) and counts the earlier NaN stop."""
    path = tmp_path / "l1.jsonl"
    first = TapeCollector(path)
    _nan_incarnation(first, 1)  # the first launcher: incarnation 1 stopped on NaN, relaunched
    first.feed(_echo("rl_driver_start", incarnation=2) + "\n")
    tape = TapeCollector(path, fresh=False)  # the restarted launcher, the same tape
    assert tape.island_stop is None
    for event, fields in NAN_LINES:  # incarnation 2 stops on NaN too (its log replays the start)
        tape.feed(_echo(event, incarnation=2, **fields) + "\n")
    assert tape.island_stop["stops"] == {"trainer.nan": 2}
    ops = FakeOps()
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j9"]
    ctl = _controller(ops, {"l1": 2}, probe=lambda name: tape.island_stop, elastic=True)
    with pytest.raises(IslandsFailedForGood):
        ctl.run()
    assert ops.relaunch_calls == []


def test_a_replayed_log_does_not_count_a_stop_twice(tmp_path):
    path = tmp_path / "l1.jsonl"
    first = TapeCollector(path)
    _nan_incarnation(first, 1)
    again = TapeCollector(path, fresh=False)
    _nan_incarnation(again, 1)  # the same records replayed by the new launcher's log stream
    assert again.island_stop is None or again.island_stop.get("stops", {}) == {"trainer.nan": 1}
    assert again._stop_counts == {"trainer.nan": 1}


def test_the_new_machine_breakers_are_named_in_every_reading():
    nan = {"breaker": NAN_LINES[0][1], "strict": NAN_LINES[1][1], "leave": NAN_LINES[2][1]}
    assert lc.new_machine_breaker(lc.classify_records(nan)) == "trainer.nan"
    leave_only = {"leave": NAN_LINES[2][1]}
    assert lc.new_machine_breaker(lc.classify_records(leave_only)) == "trainer.nan"
    for metric, name in (("nonfinite_grad_norm", "trainer.nan"), ("policy_hash_mismatch_after_apply",
                                                                   "trainer.policy_hash")):
        line = lc.island_stop_line(StrictRlInvariantError(metric, "x"))
        assert lc.new_machine_breaker(lc.classify(log_text=line, detail=line)) == name
    for metric in ("nonfinite_reward", "xhw_guard_stop", "zero_grad_norm_with_nonzero_advantages", "layout_hash"):
        line = lc.island_stop_line(StrictRlInvariantError(metric, "x"))
        assert lc.new_machine_breaker(lc.classify(log_text=line, detail=line)) is None
    assert lc.new_machine_breaker(lc.classify_records({"breaker": BREAKER_RECORD})) is None  # sample_wait


@pytest.mark.parametrize("stop", sorted(set(STOPS)))
def test_no_other_stop_gets_a_new_machine(stop):
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING] * 6 + [SUCCEEDED]
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ctl = _controller(ops, {"l0": 1, "l1": 2}, records={"l1": STOPS[stop]}, elastic=True)
    ctl.run()
    assert ops.relaunch_calls == []


def test_a_resident_ssh_island_gets_no_new_machine_relaunch():
    from yeto.launch.ssh_resident import SshIslandOps

    assert SshIslandOps.replaces_machine(object.__new__(SshIslandOps), "l1") is False
    ops = FakeOps()
    ops.replaces_machine = lambda name: False
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2"]
    ops.status_seq["l0"] = [RUNNING] * 6 + [SUCCEEDED]
    records = {"l1": {"strict": NAN_LINES[1][1], "leave": NAN_LINES[2][1], "stops": {"trainer.nan": 1}}}
    ctl = _controller(ops, {"l0": 1, "l1": 2}, records=records, elastic=True)
    ctl.run()
    assert ops.relaunch_calls == []


# -- which trainer breakers' stop records alone are final ---------------------------------------------------------

def test_every_trainer_stop_breaker_is_judged():
    """A trainer-layer entry that can stop is final by its stop record only when it marks its error
    (BREAKER_STOPS); a strict check goes by its rl_strict_failure record; anything else must be
    added here on purpose (a transient breaker would otherwise be read as never worth a relaunch)."""
    strict = set(trainer_breakers.STRICT_METRICS.values()) | {trainer_breakers.STRICT_INVARIANT}
    stop = {b.name for b in default_registry() if b.layer == "trainer" and "stop" in (b.actions or ())}
    by_record_elsewhere = {"trainer.policy_age"}  # PolicyAgeExceeded: its LEAVE (whitelisted)
    unjudged = stop - set(lc.BREAKER_STOPS) - strict - by_record_elsewhere
    assert not unjudged, f"trainer stop breakers not judged by the deterministic stop: {sorted(unjudged)}"
    assert lc.classify_records({"breaker": {**BREAKER_RECORD, "name": "trainer.some_new_transient"}}) is None


# -- the customer's configs keep the switch off ----------------------------------------------------------------

def _customer_runs():
    from test_k31_customer_plan_golden import K31, KIT

    return sorted([*K31.glob("run*.json"), KIT / "run.json"])


@pytest.mark.parametrize("run", _customer_runs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_customer_config_keeps_the_switch_off(run, tmp_path, monkeypatch, capsys):
    """Every examples/k31/run*.json (and the customer kit's run.json): launch.deterministic_stop is
    off, no island gets YETO_RL_DETERMINISTIC_STOP and the dry-run plan reports no switch (it is
    printed only when on). A customer config that comes to turn it on (an island pool option,
    --rl-sampler-follow trainer, a second trainer island, --rl-deterministic-stop) fails here:
    the customer's retries and relaunches would change, which needs the owner's decision."""
    import yeto.cli as cli
    from test_k31_customer_plan_golden import _argv, _plan

    args = cli.parse_args(_argv(run, tmp_path))
    assert fleet.deterministic_stop(args) == (False, "a single trainer (--rl-deterministic-stop turns it on)")
    assert fleet.deterministic_stop_env(args) == {}
    plan = _plan(run, tmp_path, monkeypatch, capsys)
    assert "deterministic_stop" not in plan["plan"]
    assert DETERMINISTIC_STOP_ENV not in json.dumps(plan)


# -- REV's gaps: the stop already relaunched, the trainer scheduler's wiring ------------------------------------

def test_a_relaunched_island_that_fails_before_its_driver_starts_is_an_infrastructure_relaunch(tmp_path):
    """After the NaN stop's relaunch on a new machine, the new container fails while it boots
    (before its rl_driver_start): the tape still holds the old incarnation's NaN records. They
    were handled already; the failure is an infrastructure failure and is relaunched."""
    ops = FakeOps()
    tape = TapeCollector(tmp_path / "l1.jsonl")
    _nan_incarnation(tape, 1)
    ops.status_seq["l0"] = [RUNNING] * 20 + [SUCCEEDED]
    ops.status_seq["l1"] = [RUNNING, FAILED]
    ops.relaunch_results["l1"] = ["j2", "j3"]
    boots = iter([[RUNNING, FAILED], [RUNNING, SUCCEEDED]])  # the first new container fails at boot
    ops.after_relaunch["l1"] = next(boots)

    def relaunched(name, job):
        ops.after_relaunch["l1"] = next(boots, [RUNNING, SUCCEEDED])

    ctl = _controller(ops, {"l0": 1, "l1": 2}, probe=lambda name: tape.island_stop if name == "l1" else None,
                      elastic=True)
    ctl.on_relaunch = relaunched
    ctl.run()
    assert ops.relaunch_calls == ["l1", "l1"]
    assert ctl.learners["l1"]["state"] == "done" and ctl.learners["l1"]["new_machine_retries"] == {"trainer.nan": 1}


@pytest.mark.parametrize("gpu,expected", [("modal:1xa100", (lc.INFRA, True)),  # single trainer: as on main
                                          ("modal:1xa100,modal:1xa100", (lc.BREAKER, False))])
def test_the_trainer_scheduler_classifies_with_the_launchs_switch(gpu, expected):
    """``yeto.launch.trainer_loop.build`` hands the loop ``_loop_classifier(args)``: a single
    trainer's scheduler reads a breaker's stop as main does (an infrastructure failure, started
    again); a fleet's never starts it again."""
    from yeto.launch import trainer_loop

    classify = trainer_loop._loop_classifier(_launcher_args("ports", (), gpu=gpu))
    assert classify(None, _sample_wait()) == expected
