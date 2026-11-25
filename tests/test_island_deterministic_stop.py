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
from yeto.config_errors import CONFIG_ERROR_EXIT, ISLAND_STOP_EXIT, ISLAND_STOP_MARKER, ConfigError
from yeto.launch import fleet
from yeto.launch import island_lifecycle as lc
from yeto.launch.fleet import FixedRosterIslandAbandoned, FleetController, IslandsFailedForGood
from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine import batch_reader, trainer_breakers
from yeto.rl.engine.buffer_payload import DamagedPayloadLimit, SampleBufferExhausted, SampleBufferRolloutError
from yeto.rl.engine.driver import PolicyAgeExceeded
from yeto.rl.event_echo import TapeCollector


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

    return {
        # deterministic: the same check fails the same way on every attempt
        "StrictRlInvariantError": (StrictRlInvariantError("layout_hash", "layout hash differs"), True),
        "StrictRlInvariantError (HELLO refused)": (StrictRlInvariantError(
            "syncer_profile_mismatch", "syncer refused this island: HELLO syncer semantic profile does not match"), True),
        "StrictRlInvariantError (nan)": (StrictRlInvariantError("nonfinite_grad_norm", "grad_norm is nan"), True),
        "PolicyAgeExceeded": (PolicyAgeExceeded("groups older than max_policy_age"), True),
        "PolicyAgeError": (PolicyAgeError("max_policy_age must be an int"), True),
        "AppliedLrError": (AppliedLrError("LR 0 before the final round"), True),
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
def test_a_transient_fault_keeps_its_modal_retries_and_relaunches(error, monkeypatch):
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

def test_the_island_entry_exits_island_stop_for_a_deterministic_stop():
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


def _controller(ops, learners, *, records=None, elastic=False, fixed_roster=False, log=None):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(
        learners={name: (f"task-{name}", job) for name, job in learners.items()},
        syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30, recover_timeout=900,
        thread_cls=ImmediateThread, elastic=elastic, fixed_roster=fixed_roster, fleet_log=log,
        stop_probe=None if records is None else (lambda name: records.get(name)))


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
    """``yeto launch`` of one Modal island without a syncer (the K3.1 async GRPO trainer): an island
    that stopped on its sample-wait breaker is released and the launch ends; no second container."""
    record, clock = [], Clock()
    line = f"{ISLAND_STOP_MARKER} breaker trainer.sample_wait: no new group (exit 11, not retried)"
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record, log_lines=("noise", line))
    extra = ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
             "--rl-single-island-no-sync")
    args = _launcher_args("ports", extra, gpu="modal:1xa100")
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
    monkeypatch.setattr(launcher, "controller_fixed_roster", lambda args: False)  # an island pool island
    assert launcher.run(args) == ISLAND_STOP_EXIT
    assert not [r for r in record if r[0] == "relaunch"] and record.count(("spawn", 0)) == 1
