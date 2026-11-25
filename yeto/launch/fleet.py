"""The fleet controller: island states, relaunch policy, failure types and exit codes."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

from ..config_errors import CONFIG_ERROR_EXIT, CONFIG_ERROR_MARKER  # noqa: F401 - registered exit code
from . import island_lifecycle as lifecycle
from .syncer import island_pool
from .sky_ops import SYNCER_REFUSAL_MARKERS


def controller_fixed_roster(args) -> bool:
    """FleetController ``fixed_roster``: RL runs without island pool options.
    A pool's membership is dynamic: an island that fails for good is dropped
    and the pool goes on (10-09 ruling of the main agent for the user)."""
    return getattr(args, "training_mode", "sft") == "rl" and not island_pool(args)


# FleetController states.
RUNNING = "running"
RECOVERING = "recovering"
DONE = "done"
ABANDONED = "abandoned"


@dataclass(frozen=True)
class _RlStrictFailure:
    reason: str


class RelaunchBlocked(Exception):
    """Raised by ``sky_ops.relaunch`` when the island must not be relaunched at all
    (not now, not on a later poll): e.g. its Modal app was stopped by a teardown.
    The message is the reason; the controller abandons the island instead of
    retrying until the recovery timeout."""


class _RelaunchAttempt:
    """Result slot for one background relaunch attempt."""

    def __init__(self):
        self.result = None  # new job id, or None if provisioning failed
        self.finished = False
        self.thread = None
        self.blocked = None  # reason the relaunch was refused (instance alive)
        self.name = None  # cluster name the relaunch used (may be a new one)


# Fixed-roster RL islands: consecutive relaunches allowed within one recovery
# window. A window closes (budget and count reset) once a relaunched island
# has stayed healthy for RECOVERY_STABLE_S; the time budget counts only the
# time spent recovering in the window.
FIXED_ROSTER_MAX_RELAUNCHES = 2
RECOVERY_STABLE_S = 300.0
# How long one rejected-link relaunch may take to get a new container.
WAN_RECOVERY_S = 900.0
# How long a failed-looking island is given for its rl_learner_finalized record
# to arrive over the log stream before recovery starts.
FINALIZE_GRACE_S = 60.0
# How long an island still waiting for the syncer may keep running after the
# syncer finished its run. The syncer is gone, so such an island (a replacement
# that came up after the final cut) can only dial a closed port until its
# connect timeout; the grace covers log-stream delay of its join records.
LATE_ISLAND_GRACE_S = 120.0
# Bound on waiting for an in-flight relaunch before abandoning (so a late
# sky.launch cannot re-create a cluster after teardown).
RELAUNCH_JOIN_S = 120.0
# An island STOP: how long its trainer may go on (to end on its own) before it is released.
ISLAND_STOP_GRACE_S = 60.0


class RunStalled(RuntimeError):
    """No island produced an event for the stall timeout (and not all finalized).

    The ``launch.stall`` breaker (``yeto.breakers.layers.launch``): ``silent_s`` is how long
    the islands were silent, ``timeout_s`` the limit."""

    def __init__(self, message: str, *, silent_s: float | None = None,
                 timeout_s: float | None = None) -> None:
        super().__init__(message)
        self.silent_s = silent_s
        self.timeout_s = timeout_s


# Exit code when the run stalled (no island event for --rl-stall-timeout).
RUN_STALLED_EXIT = 6
DEFAULT_RL_STALL_TIMEOUT_S = 900.0


def effective_stall_timeout(args) -> float:
    """``launch.stall``: the fleet's stall clock in seconds (``--rl-stall-timeout``; unset:
    the default; 0: off)."""
    value = getattr(args, "rl_stall_timeout", None)
    if value == 0:
        return 0.0
    return float(value or DEFAULT_RL_STALL_TIMEOUT_S)


class FixedRosterIslandAbandoned(RuntimeError):
    """A fixed-roster RL island could not be recovered: the run cannot finish."""
# run() exit code when a fixed-roster island failed for good (distinct from
# 2 = artifact not fetchable and 3 = incomplete event tape).
ISLAND_FAILED_EXIT = 4
FAILED_RUN_DRAIN_S = 20.0
# Modal app not confirmed stopped (state stopped, 0 tasks) after teardown.
TEARDOWN_UNVERIFIED_EXIT = 5
# Modal moved an island to a different container mid-run (preempted/rescheduled and the
# run script re-ran): the run is failed and the Modal app stopped. Registered exit codes:
# 1 failed, 2 artifact not fetchable, 3 incomplete tape, 4 island failed, 5 teardown
# unverified, 6 stalled, 7 Modal container id changed, 8 trainer scheduler gave up,
# 9 stopped on the STOP flag, 10 an island refused its configuration
# (CONFIG_ERROR_EXIT, yeto.config_errors: also the island's own exit code; not retried).
CONTAINER_CHANGED_EXIT = 7
# The trainer scheduler stopped after too many failed trainer incarnations in a row.
TRAINER_GAVE_UP_EXIT = 8
# The exit of an island its own STOP ended (``fleet/trainers/<k>/STOP``): a stop on request,
# not a failure of the run.
ISLAND_STOPPED_PREFIX = "STOPPED"


class RunStopRequested(RuntimeError):
    """``yeto stop-run`` wrote the run's STOP flag: the run ends and is torn down."""


# Exit code when the run ended on its STOP flag (``yeto stop-run``).
RUN_STOPPED_EXIT = 9


class IslandConfigError(RuntimeError):
    """An island refused its launch options (``[yeto-config-error]``, island exit
    :data:`CONFIG_ERROR_EXIT`): deterministic, so it is not relaunched; the run ends
    with :data:`CONFIG_ERROR_EXIT`."""


class FleetController:
    """Supervises the syncer + learner fleet after the initial launch.

    Per-learner state machine (evaluated once per poll):

        running    --(job terminal-not-SUCCEEDED, or cluster not UP)--> recovering
        recovering --(relaunch returns a new job id)------------------> running
        recovering --(recover_timeout exceeded)-----------------------> abandoned
        running    --(job SUCCEEDED)----------------------------------> done

    An abandoned learner's cluster is torn down immediately (even with
    --keep) and the run continues with the remaining fleet: the syncer's
    quorum design tolerates missing learners, and a learner that comes back
    later is caught up by the syncer's full rebroadcast. The syncer follows
    the same running/recovering cycle but is never abandoned — past the
    timeout it keeps retrying and logs an error every poll (its relaunch
    resumes from the on-VM checkpoint via the --resume flag already in its
    run command). recover_timeout <= 0 disables recovery: a failed learner
    is torn down on the spot.

    The syncer can be supervised in one of two ways:

    * as a cluster (local controller mode): pass ``syncer=(name, task,
      job_id)`` and it goes through the running/recovering cycle above,
      relaunched via sky like a learner but never abandoned;
    * as a subprocess of this process (head controller mode): pass
      ``syncer_probe``/``syncer_restart`` callables instead — ``probe()``
      returns None while healthy or a reason string when dead, at which
      point ``restart()`` is invoked (the syncer resumes from its local
      checkpoint). A local syncer is likewise never abandoned, and there
      is no syncer cluster to tear down.

    At most one relaunch attempt is in flight per cluster, each in a
    background thread so one slow re-provision never blocks polling the
    others; `thread_cls` exists so tests can substitute a synchronous stub.
    """

    def __init__(
        self,
        learners: dict,
        syncer: tuple | None,
        sky_ops,
        poll_interval: float,
        recover_timeout: float,
        on_relaunch=None,
        thread_cls=threading.Thread,
        syncer_probe=None,
        syncer_restart=None,
        fixed_roster: bool = False,
        finalized_probe=None,
        progress_probe=None,
        stall_timeout: float = 0.0,
        stop_flag=None,
        no_recover=(),
        instance_guard=None,
        on_rename=None,
        fleet_log=None,
        keep_abandoned: bool = False,
        syncer_finished_probe=None,
        left_pool_probe=None,
        awaiting_syncer_probe=None,
        elastic: bool = False,
        no_island_relaunch: bool = False,
        replacement_planner=None,
        wan_reject_probe=None,
        wan_probe_relaunches: int = 0,
        island_relaunch_envs=None,
        followers=(),
        follower_grace_s: float = 120.0,
        rollout=None,
        recorder=None,
        incarnations=None,
    ):
        """`learners` maps cluster name -> (task, job_id); `syncer` is
        (name, task, job_id) for a cluster syncer, or None with
        `syncer_probe`/`syncer_restart` callables for a subprocess syncer.
        `on_relaunch(name, new_job_id)` is called after every successful
        cluster relaunch (production spawns a new log tail)."""
        self.ops = sky_ops
        # Sampler islands of a synced group that run until their trainers end
        # (--rl-sampler-until-trainers-end): they stop on the trainers' end
        # records. A trainer that ends without one (it failed after the syncer
        # finished, e.g. a relaunched island still booting) would leave them
        # generating into a full backlog until the platform timeout; once every
        # other island is done they are ended here after follower_grace_s.
        self.followers = frozenset(followers or ())
        self.follower_grace_s = float(follower_grace_s)
        self._leaders_done_at: float | None = None
        # 0.26: () -> bool, the syncer finished its run (no island is relaunched
        # after that; a late island's failure then counts as a normal end).
        self.syncer_finished_probe = syncer_finished_probe
        # 0.27: name -> bool, the island left the elastic pool for good (re-JOIN
        # exhausted, learner exit 7); elastic only.
        self.left_pool_probe = left_pool_probe
        # name -> bool: the island's driver started and has not reached the syncer
        # yet (TapeCollector.awaiting_syncer). After the syncer finished such an
        # island is ended instead of left dialing until its connect timeout.
        self.awaiting_syncer_probe = awaiting_syncer_probe
        self.elastic = elastic
        self.no_island_relaunch = no_island_relaunch
        # name -> number of rejected link probes the island logged
        # (rl_wan_probe accepted=false, exit 75). Such an island never joined
        # the syncer session, so it is relaunched in a fresh container up to
        # wan_probe_relaunches times, also under --no-island-relaunch.
        self.wan_reject_probe = wan_reject_probe
        self.wan_probe_relaunches = wan_probe_relaunches
        # rl-spot-cost-saving 2.3: yeto.cloud.replace.ReplacementPlanner or None.
        # Asked once per recovery window before the same-place relaunch. Auto
        # launch off (default): it only writes spot_replace_advice and the
        # existing relaunch runs unchanged. On: its launch() result (a new job
        # id) replaces the same-place relaunch.
        self.replacement_planner = replacement_planner
        # Environment added to an island task before its relaunch (peer
        # recovery: the relaunched trainer restores from a peer, not from scratch).
        self.island_relaunch_envs = dict(island_relaunch_envs or {})
        # B12: True (--keep-abandoned) leaves an abandoned learner's cluster up for
        # post-mortem instead of tearing it down; the final teardown still honors --keep.
        self.keep_abandoned = keep_abandoned
        # Path of the run's STOP flag file (``runs.stop_flag_path``; ``yeto stop-run``
        # writes it): once it exists no island is relaunched any more.
        self.stop_flag = stop_flag
        self.poll_interval = poll_interval
        self._abort_error: BaseException | None = None  # set by abort()
        self.recover_timeout = recover_timeout
        self.on_relaunch = on_relaunch
        self.thread_cls = thread_cls
        self.fixed_roster = fixed_roster
        # name -> bool: the island's tape already holds rl_learner_finalized
        # (training complete); a non-zero exit after that is a shutdown-phase
        # error, not an island failure.
        self.finalized_probe = finalized_probe
        # () -> int: events received from the islands so far (log echo). With
        # stall_timeout > 0 a run with no new event for that long, and not every
        # learner finalized, is stalled (e.g. a dead syncer connection).
        self.progress_probe = progress_probe
        self.stall_timeout = stall_timeout
        self._progress = None
        self._progress_at = None
        self.learners = {
            name: self._make_record(name, task, job_id)
            for name, (task, job_id) in learners.items()
        }
        if syncer_probe is not None:
            if syncer_restart is None:
                raise ValueError("syncer_probe requires syncer_restart")
            self.syncer = None
            self.syncer_probe = syncer_probe
            self.syncer_restart = syncer_restart
        elif syncer is None:
            # --rl-single-island-no-sync: no syncer of any kind to supervise.
            self.syncer = None
            self.syncer_probe = self.syncer_restart = None
        else:
            syncer_name, syncer_task, syncer_job = syncer
            self.syncer = self._make_record(syncer_name, syncer_task, syncer_job)
            self.syncer_probe = self.syncer_restart = None
        self.downed_clusters: set = set()
        # teardowns skipped while the records lock could not be confirmed past its lease: tried
        # again each poll once it is confirmed (the island would bill until the launcher ends)
        self.pending_downs: dict = {}  # name -> force
        # Learners whose recovery is disabled regardless of recover_timeout
        # (Verda without the verified sky patch: a same-name relaunch there
        # can delete the running node).
        self.no_recover = set(no_recover)
        # Cloud-side check before a relaunch (VerdaInstanceGuard): None, or
        # an object whose check(name) returns ("alive", why) | ("gone", new
        # name) | ("unknown", why) | None (not guarded).
        self.instance_guard = instance_guard
        self.on_rename = on_rename
        # dashboard.fleet.FleetLog (fleet-dashboard 2.3): island lifecycle and
        # periodic cost_tick into the run's fleet.jsonl; best-effort, optional.
        self.fleet_log = fleet_log
        # yeto.launch.pool.controller.RolloutController or None: the rollout island
        # pools of the run, driven on their own thread for the life of run(); they
        # drain (machines released) when run() returns or raises, the STOP flag included.
        self.rollout = rollout
        self._rollout_warned = False
        # yeto.launch.island_lifecycle.FleetRecorder or None: every island state change is
        # written to the run store (fleet/trainers/<k>/state.json), and each island's own
        # STOP there ends that island only. None (a single-trainer launch by default):
        # nothing is written or read.
        self.recorder = recorder
        # name -> {"incarnation", "relaunches", "adopted"} from the island's record of an
        # earlier launcher of this run (yeto.launch.island_adopt): an adopted island keeps
        # its incarnation; a relaunched one continues the count.
        for name, known in dict(incarnations or {}).items():
            if name in self.learners:
                self.learners[name]["incarnation"] = int(known.get("incarnation") or 0)
                self.learners[name]["relaunches"] = int(known.get("relaunches") or 0)
                self.learners[name]["adopted"] = bool(known.get("adopted"))

    def _fleet(self, kind: str, name: str | None = None, *, force: bool = False, **fields) -> None:
        if self.fleet_log is None:
            return
        try:
            if kind == "cost_tick":
                if force:
                    self.fleet_log.cost_tick()
                else:
                    self.fleet_log.maybe_tick()
            else:
                self.fleet_log.event(kind, name, **fields)
        except Exception as e:  # noqa: BLE001 - telemetry never breaks supervision
            print(f"[launcher] fleet.jsonl: {e}", file=sys.stderr)

    @staticmethod
    def _make_record(name, task, job_id):
        return {
            "name": name,
            "task": task,
            "job_id": job_id,
            "state": RUNNING,
            "failed_at": None,
            "attempt": None,
            "exit": None,
            "incarnation": 0,
            "relaunches": 0,
        }

    def abort(self, error: BaseException) -> None:
        """Ask :meth:`run` (any thread) to end with ``error`` at its next poll, e.g.
        the trainer scheduler gave up; the launcher then tears the run down."""
        self._abort_error = error

    def run(self) -> dict:
        """Poll until every learner is done or abandoned (:meth:`_run_fleet`); with a
        ``rollout`` controller its pools run meanwhile and are drained at the end."""
        if getattr(self, "rollout", None) is None:
            return self._run_fleet()
        self.rollout.start()
        why = "trainer fleet finished"
        try:
            return self._run_fleet()
        except RunStopRequested:
            why = "STOP flag"
            raise
        except BaseException as e:
            why = f"run ended: {type(e).__name__}"
            raise
        finally:
            left = self.rollout.stop(why)
            billed = {name: [m.ticket for m in members] for name, members in left.items() if members}
            if billed:
                print(f"[launcher] ERROR: rollout machines still billed after the drain: {billed}",
                      file=sys.stderr)

    def _poll_rollout(self) -> None:
        if getattr(self, "rollout", None) is None:
            return
        if not self._rollout_warned:
            problem = self.rollout.check()
            if problem is not None:
                self._rollout_warned = True
                print(f"[launcher] ERROR: {problem}; its machines are drained when the run ends",
                      file=sys.stderr)
        self._check_rollout_stall()

    def _check_rollout_stall(self) -> None:
        """The rollout side's stall view (``RolloutController.stalled``): an alert once per
        silent stretch, never a kill (a sampler waiting for the trainer is not stalled)."""
        stalled = getattr(self.rollout, "stalled", None)
        if stalled is None:
            return
        try:
            why = stalled()
        except Exception:  # noqa: BLE001 - a view, never the end of supervision
            return
        if why and not getattr(self, "_rollout_stall_warned", False):
            self._rollout_stall_warned = True
            print(f"[launcher] WARNING: rollout stalled: {why}", file=sys.stderr)
            self._fleet("rollout_stalled", None, detail=why)
        elif not why and getattr(self, "_rollout_stall_warned", False):
            self._rollout_stall_warned = False
            self._fleet("rollout_progress_resumed", None)

    def _run_fleet(self) -> dict:
        """Poll until every learner is done or abandoned.

        Returns {learner name: final status string}; raises RuntimeError
        (after downing the syncer) if every learner was abandoned, the
        :meth:`abort` error when one was set, and :class:`RunStopRequested`
        when the STOP flag appears while islands still run (the launcher then
        stops the trainer with a cut and tears the run down; before, the
        worker kept supervising the live islands until it was killed).
        """
        for name, rec in self.learners.items():
            self._fleet("island_ready", name, initial=True)
            if rec.get("adopted"):
                self._record(rec, reason="adopted from an earlier launcher", adopted=True)
            else:
                self._record(rec, reason="launched", adopted=False)
        while True:
            if self._abort_error is not None:
                raise self._abort_error
            self._raise_if_fenced()
            self._retry_pending_downs()
            self._fleet("cost_tick")
            self._poll_rollout()
            if self.syncer is not None:
                self._poll(self.syncer, is_syncer=True)
            elif self.syncer_probe is not None:
                self._poll_local_syncer()
            for rec in self.learners.values():
                self._poll_recorded(rec)
            self._end_followers()
            if all(r["state"] in (DONE, ABANDONED) for r in self.learners.values()):
                break
            if self.stop_flag is not None and os.path.exists(self.stop_flag):
                print(f"[launcher] STOP flag {self.stop_flag} present; ending the run",
                      file=sys.stderr)
                raise RunStopRequested(f"STOP flag {self.stop_flag}")
            self._check_stall()
            self.ops.sleep(self.poll_interval)
        exit_codes = {name: rec["exit"] for name, rec in self.learners.items()}
        for name, rec in self.learners.items():
            if rec["state"] == DONE:
                self._fleet("island_stop", name, reason=str(rec["exit"]))
        self._fleet("cost_tick", force=True)
        print(f"[launcher] learner jobs finished: {exit_codes}")
        if not any(rec["state"] == DONE for rec in self.learners.values()):
            if self.syncer is not None:
                print(
                    "[launcher] ERROR: all learners abandoned; tearing down the syncer",
                    file=sys.stderr,
                )
                self._down(self.syncer["name"])
            else:
                print(
                    "[launcher] ERROR: all learners abandoned",
                    file=sys.stderr,
                )
            raise RuntimeError("all learners abandoned; nothing left to train")
        return exit_codes

    # -- the island records (yeto.launch.island_lifecycle) -------------------------------
    def _records_fenced(self):
        """The island records' lock error once another launcher took them over, else None."""
        return None if self.recorder is None else getattr(self.recorder.store, "fenced", None)

    def _raise_if_fenced(self) -> None:
        """A controller that lost the island records' lock stops supervising at once: the
        launcher that holds it orders, relaunches and tears down (this one touches nothing)."""
        fenced = self._records_fenced()
        if fenced is not None:
            raise lifecycle.IslandRecordsFenced(f"the fleet controller lost the island records' lock ({fenced})")

    @staticmethod
    def _unified_state(rec) -> str:
        """The lifecycle state of a controller record (an island dropped for a
        configuration error or a strict failure is ``failed``: never relaunched)."""
        state = rec["state"]
        if state == ABANDONED:
            decision = rec.get("decision")
            return lifecycle.FAILED if decision is not None and decision.action == lifecycle.FAIL \
                else lifecycle.ABANDONED
        if state == DONE:
            return lifecycle.DONE
        if state == RECOVERING:
            return lifecycle.RECOVERING
        return lifecycle.DRAINING if rec.get("draining") else lifecycle.RUNNING

    def _record(self, rec, state: str | None = None, **fields) -> None:
        """Write the island's record (state, machine handle, incarnation, relaunches, last
        exit and decision) to the store; never raises."""
        if self.recorder is None:
            return
        try:
            ref = self.recorder.ref(rec["name"])
            if ref is None:
                return
            decision = rec.get("decision")
            self.recorder.record(
                rec["name"], state or self._unified_state(rec),
                handle=lifecycle.handle_of(ref.source, rec["name"], rec["job_id"]),
                incarnation=int(rec.get("incarnation") or 0), relaunches=int(rec.get("relaunches") or 0),
                exit=None if rec.get("exit") is None else str(rec["exit"])[:500],
                decision=None if decision is None else decision.to_json(),
                fixed_roster=bool(self.fixed_roster), **fields)
        except Exception as e:  # noqa: BLE001 - the record is a view; supervision goes on
            print(f"[launcher] island record of {rec.get('name')}: {e}", file=sys.stderr)

    def _poll_recorded(self, rec) -> None:
        """:meth:`_poll` one learner, with its island STOP first; a state, machine or
        incarnation change is written to its record (a raise too: the island ``failed``
        or, for a fixed roster, ``abandoned``)."""
        self._island_start(rec)
        before = (rec["state"], rec["job_id"], rec["name"], rec.get("draining"))
        try:
            if not self._island_stop(rec):
                self._poll(rec, is_syncer=False)
        except BaseException as error:
            if self.recorder is not None and rec["state"] != DONE:  # a stopped island stays done
                state = lifecycle.ABANDONED if isinstance(error, FixedRosterIslandAbandoned) else lifecycle.FAILED
                if rec.get("decision") is None and state == lifecycle.FAILED:
                    rec["decision"] = lifecycle.classify(log_text=str(error), detail=str(error))
                self._record(rec, state, reason=f"{type(error).__name__}: {error}"[:300])
            raise
        if (rec["state"], rec["job_id"], rec["name"], rec.get("draining")) != before:
            decision = rec.get("decision")
            if rec["state"] == DONE and (decision is None or decision.relaunch):
                rec["decision"] = lifecycle.classify(succeeded=True, detail=str(rec.get("exit") or ""))
            elif rec["state"] == RUNNING and decision is not None and decision.relaunch:
                rec["decision"] = None  # relaunched: the failure is history (kept in the record's history)
            decision = rec.get("decision")
            reason = rec.get("exit") if rec.get("exit") is not None else (
                decision.detail if decision is not None else None)
            self._record(rec, reason=None if reason is None else str(reason)[:300])

    def _island_start(self, rec) -> bool:
        """Island ``rec``'s START request (``yeto island start``). Its record decides, not this
        controller's memory (a restarted or adopting launcher acts on it too): an island the
        record shows ``done`` on its island STOP (decision ``stopped``) is ordered again (the
        next incarnation; recovery drives the order, checked against the records' lock). A
        START is consumed: the record keeps it (``start``, by its time: never acted on twice)
        and the request is deleted. A START that reaches an island still draining on its STOP
        is kept, not consumed, and acted on once the island is done. A START of an island that runs, or that ended otherwise
        (finished, given up), orders nothing and is consumed too. A controller that lost the
        island records' lock acts on no request (:class:`IslandRecordsFenced`). True when
        the island is ordered again."""
        if self.recorder is None:
            return False
        ref = self.recorder.ref(rec["name"])
        if ref is None or ref.role == "sampler":
            return False
        try:
            request = self.recorder.start_requested(rec["name"])
            if request is None or self.recorder.stop_requested(rec["name"]) is not None:
                return False  # no START, or a STOP still stands (yeto island start clears it first)
        except Exception:  # noqa: BLE001 - asked again next poll
            return False
        self._raise_if_fenced()
        record = self.recorder.store.cached(ref.island) or {}
        if request.get("at") is not None and request.get("at") == (record.get("start") or {}).get("at"):
            self.recorder.store.clear_start(ref.island)  # acted on already (its delete did not land)
            return False
        why = str(request.get("reason") or "START")
        by = request.get("by")
        if rec.get("draining") or (rec.get("stop_request") is not None and rec["state"] != DONE):
            # acted on its STOP already (``yeto island start`` saw no draining record yet): the
            # START is kept, not consumed, and acted on once the island is done
            if rec.get("start_kept_at", object()) != request.get("at"):
                rec["start_kept_at"] = request.get("at")
                print(f"[launcher] {rec['name']}: island START ({why}) kept: the island drains on its island "
                      "STOP; it is ordered again once it is done", file=sys.stderr)
            return False
        stopped = (rec["state"] == DONE and record.get("state") == lifecycle.DONE
                   and (record.get("decision") or {}).get("cause") == lifecycle.STOPPED)
        if not stopped:
            print(f"[launcher] {rec['name']}: island START ({why}) ignored: the island is "
                  f"{record.get('state') or rec['state']}, not stopped on its island STOP", file=sys.stderr)
            self._record(rec, start=dict(request, outcome="ignored"))
            self.recorder.store.clear_start(ref.island)
            return False
        try:  # the lock checked now, not only the cached view (lost raises IslandRecordsFenced)
            self.recorder.store.guard_order(f"island START of {rec['name']}")
        except lifecycle.IslandRecordsUnconfirmed as error:
            print(f"[launcher] {rec['name']}: island START deferred: {error}", file=sys.stderr)
            return False
        rec.pop("stopped", None)
        rec.pop("stop_request", None)
        rec.pop("start_kept_at", None)
        rec.update(state=RECOVERING, failed_at=self.ops.now(), attempt=None, failures=0, recovering_s=0.0,
                   exit=None, decision=None, draining=False)
        self.downed_clusters.discard(rec["name"])  # its next incarnation is torn down at the end again
        print(f"[launcher] {rec['name']}: island START ({why}{', by ' + str(by) if by else ''}); "
              "ordering it again as its next incarnation", file=sys.stderr)
        # island_ready comes once the order runs (the recovery's relaunch), not now
        self._record(rec, lifecycle.ORDERING, reason=f"island START: {why}", start=dict(request, outcome="ordered"))
        self.recorder.store.clear_start(ref.island)
        return True

    def _island_stop(self, rec) -> bool:
        """Island ``rec``'s own STOP (``fleet/trainers/<k>/STOP``): it ends this island only.
        The island drains (no relaunch from here on): its trainer gets up to
        :data:`ISLAND_STOP_GRACE_S` to end on its own (polled, the other islands go on), then
        it is released and ``done`` with the request as its exit; the fleet goes on. The
        trainer has no command channel (the Miles island entry reads no STOP), so a trainer
        that still runs at the end of the grace is torn down and the syncer drops it once its
        lease runs out. A controller that lost the island records' lock tears nothing down
        and writes nothing (:class:`IslandRecordsFenced`: the launcher holding them acts).
        True while the island is draining or stopped now (no :meth:`_poll` of it)."""
        if self.recorder is None or rec["state"] in (DONE, ABANDONED):
            return False
        request = rec.get("stop_request")
        if request is None:
            try:
                request = self.recorder.stop_requested(rec["name"])
            except Exception:  # noqa: BLE001 - asked again next poll
                return False
            if request is None:
                return False
            self._raise_if_fenced()
            why = str(request.get("reason") or "STOP")
            by = request.get("by")
            rec["stop_request"] = dict(request)
            rec["draining"] = True
            rec["drain_until"] = self.ops.now() + ISLAND_STOP_GRACE_S
            self._record(rec, lifecycle.DRAINING, reason=f"island STOP: {why}")
            print(f"[launcher] {rec['name']}: island STOP ({why}{', by ' + str(by) if by else ''}); stopping "
                  f"this island only, the fleet goes on (its trainer gets {ISLAND_STOP_GRACE_S:.0f} s to end "
                  "before it is released)", file=sys.stderr)
            rec["stopped"] = True  # an in-flight relaunch tears its cluster down when it ends
            attempt = rec.get("attempt")
            if attempt is not None and attempt.thread is not None and not attempt.finished:
                attempt.thread.join(RELAUNCH_JOIN_S)
            if attempt is not None and attempt.finished and attempt.result is not None \
                    and attempt.name and attempt.name != rec["name"]:
                self._raise_if_fenced()
                self._down(attempt.name)  # relaunched under a new name meanwhile
            rec["attempt"] = None
        self._raise_if_fenced()
        why = str(request.get("reason") or "STOP")
        if rec["state"] == RUNNING and self.ops.now() < float(rec.get("drain_until") or 0.0):
            verdict, _status = self._probe(rec)
            if verdict is None:
                return True  # still training: the grace goes on
            print(f"[launcher] {rec['name']}: ended within its island STOP grace ({verdict})", file=sys.stderr)
        rec["state"] = DONE
        rec["exit"] = f"{ISLAND_STOPPED_PREFIX} (island STOP: {why})"
        rec["decision"] = lifecycle.classify(stopped=True, detail=why)
        rec["draining"] = False
        self._fleet("island_stop", rec["name"], reason=f"island STOP: {why}"[:500])
        if self.keep_abandoned:
            print(f"[launcher] keeping stopped cluster {rec['name']} (--keep-abandoned)")
        else:
            self._down(rec["name"])
        self._record(rec, lifecycle.DONE, reason=f"island STOP: {why}", stop=dict(request))
        if self.fixed_roster:
            # the syncer of a fixed roster waits for every learner: one island stopped ends the run
            message = (f"fixed roster: island {rec['name']} stopped on its island STOP ({why}); a fixed-roster "
                       "fleet cannot go on without it")
            print(f"[launcher] ERROR: {message}", file=sys.stderr)
            raise FixedRosterIslandAbandoned(message)
        return True

    def _poll_local_syncer(self) -> None:
        """Subprocess syncer: never abandoned — a dead process is restarted
        on the spot (it resumes from its local checkpoint)."""
        reason = self.syncer_probe()
        if reason is None:
            return
        if self._all_learners_finalized():
            print(f"[launcher] syncer: {reason} after every learner finalized; "
                  "not restarting", file=sys.stderr)
            return
        if isinstance(reason, _RlStrictFailure):
            raise RuntimeError(f"strict RL syncer failed: {reason.reason}")
        print(
            f"[launcher] syncer: {reason}; restarting the local syncer "
            "(resumes from its checkpoint)",
            file=sys.stderr,
        )
        try:
            self.syncer_restart()
        except Exception as e:
            print(
                f"[launcher] syncer restart failed: {e}; retrying next poll",
                file=sys.stderr,
            )

    def _poll(self, rec, is_syncer: bool) -> None:
        if rec["state"] == RUNNING:
            verdict, status = self._probe(rec)
            if verdict is None:
                if not is_syncer and self._end_late_island(rec):
                    return
                recovered_at = rec.get("recovered_at")
                if recovered_at is not None and self.ops.now() - recovered_at >= RECOVERY_STABLE_S:
                    # stable again: close the recovery window
                    rec.pop("recovered_at", None)
                    rec["failures"] = 0
                    rec["recovering_s"] = 0.0
                return  # healthy
            if verdict == "succeeded":
                rec["state"] = DONE
                rec["exit"] = str(status)
                print(f"[launcher] {rec['name']} job finished: {status}")
            elif is_syncer and self._all_learners_finalized():
                rec["state"] = DONE  # training is over; nothing to recover
                rec["exit"] = f"stopped after every learner finalized ({status})"
                print(f"[launcher] syncer {verdict} after every learner finalized; "
                      "not recovering", file=sys.stderr)
            elif not is_syncer and self._finalized(rec):
                # Training completed (rl_learner_finalized seen): an error while
                # shutting down (e.g. KeyboardInterrupt in ray.shutdown after
                # the syncer stopped) is neither a failure nor a reason to
                # relaunch.
                rec["state"] = DONE
                rec["exit"] = f"SUCCEEDED (finalized; shutdown ended as {status})"
                print(f"[launcher] WARN: {rec['name']} finalized, then {verdict}; "
                      "counted as succeeded, no recovery", file=sys.stderr)
            elif not is_syncer and self._syncer_finished():
                rec["state"] = DONE  # 0.26: the run is over; a late island's error is not a failure
                rec["exit"] = f"SUCCEEDED (syncer finished; island ended as {status})"
                self._fleet("island_done", rec["name"], reason="syncer finished")
                print(f"[launcher] {rec['name']}: {verdict} after the syncer finished; "
                      "counted as a normal end, not relaunched", file=sys.stderr)
            elif not is_syncer and self._rejected_link(rec):
                self._relaunch_rejected_link(rec)
            elif not is_syncer and self.elastic and self._left_pool(rec):
                # 0.27: re-JOIN exhausted (exit 7): the island left the elastic pool.
                # Not the fixed-roster abandon path (exit 4): relaunch unless
                # --no-island-relaunch, and the run goes on without it.
                self._fleet("island_lost", rec["name"], reason="left elastic pool (re-JOIN exhausted)")
                if self.no_island_relaunch:
                    rec["state"] = DONE
                    rec["exit"] = f"LEFT_POOL (re-JOIN exhausted; {status})"
                    print(f"[launcher] {rec['name']}: left the elastic pool (re-JOIN exhausted); "
                          "--no-island-relaunch: not relaunched", file=sys.stderr)
                else:
                    rec["left_pool"] = True
                    self._enter_recovering(rec, "left the elastic pool (re-JOIN exhausted)", is_syncer)
            elif not is_syncer and self._await_finalized(rec):
                return  # grace: the finalized record may still be in the log stream
            else:
                config_error = None if is_syncer else self._config_error(rec)
                if config_error is not None:
                    rec["decision"] = lifecycle.decision(lifecycle.CONFIG_ERROR, config_error.strip())
                    # the island refused its launch options: a relaunch fails the same way
                    self._fleet("island_lost", rec["name"], reason=f"configuration error: {config_error}"[:500])
                    if self.elastic:
                        print(f"[launcher] {rec['name']}: configuration error, not relaunched; the "
                              f"elastic pool goes on without it: {config_error}", file=sys.stderr)
                        self._abandon(rec, 0.0, reason="configuration error (elastic: island dropped)")
                        return
                    raise IslandConfigError(f"island {rec['name']} refused its configuration "
                                            f"(not relaunched): {config_error}")
                strict_failure = self._strict_failure(rec)
                if strict_failure is not None and not is_syncer:
                    rec["decision"] = lifecycle.classify(log_text=strict_failure, detail=strict_failure)
                    if rec["decision"].action != lifecycle.FAIL:  # a strict line without its marker
                        rec["decision"] = lifecycle.decision(lifecycle.STRICT, strict_failure)
                if strict_failure is not None and self.elastic and not is_syncer:
                    # Elastic: a refused or strictly failed island is a deterministic
                    # error -- relaunching it cannot help. Tear down only this island,
                    # record island_lost, keep the rest of the pool running
                    # (10-09 ruling of the main agent for the user).
                    self._fleet("island_lost", rec["name"], reason=f"strict failure: {strict_failure}"[:500])
                    print(f"[launcher] {rec['name']}: strict failure, not relaunched; the elastic "
                          f"pool goes on without it: {strict_failure}", file=sys.stderr)
                    self._abandon(rec, 0.0, reason="strict failure (elastic: island dropped)")
                    return
                if strict_failure is not None:
                    raise RuntimeError(
                        f"strict RL job {rec['name']} failed: {strict_failure}"
                    )
                self._enter_recovering(rec, verdict, is_syncer)
        elif rec["state"] == RECOVERING:
            self._drive_recovery(rec, is_syncer)

    def _end_followers(self) -> None:
        """End the follower islands (``followers``) still running
        ``follower_grace_s`` after every other island is done."""
        if not self.followers:
            return
        leaders = [r for n, r in self.learners.items() if n not in self.followers]
        if not leaders or not all(r["state"] in (DONE, ABANDONED) for r in leaders):
            self._leaders_done_at = None
            return
        now = self.ops.now()
        if self._leaders_done_at is None:
            self._leaders_done_at = now
        if now - self._leaders_done_at < self.follower_grace_s:
            return
        for name in sorted(self.followers):
            rec = self.learners.get(name)
            if rec is None or rec["state"] in (DONE, ABANDONED):
                continue
            rec["state"] = DONE
            rec["exit"] = "SUCCEEDED (its trainers ended; stopped by the launcher)"
            self._fleet("island_done", name, reason="trainers ended")
            print(f"[launcher] {name}: every trainer it feeds has ended; stopping the sampler",
                  file=sys.stderr)

    def _end_late_island(self, rec) -> bool:
        """End a running island that is still waiting for the syncer
        LATE_ISLAND_GRACE_S after the syncer finished its run. The syncer only
        waits for the learners connected at its final cut, so a replacement that
        comes up after that cut has no syncer to join and would otherwise hold
        its machine until its connect timeout (900 s and more) before failing.
        An island that reached the syncer (it may be saving its final policy)
        is left alone."""
        if (rec["name"] in self.followers or not self._awaiting_syncer(rec)
                or not self._syncer_finished()):
            rec.pop("late_since", None)
            return False
        since = rec.setdefault("late_since", self.ops.now())
        if self.ops.now() - since < LATE_ISLAND_GRACE_S:
            return False
        rec["state"] = DONE
        rec["exit"] = "SUCCEEDED (syncer finished before the island joined; stopped by the launcher)"
        self._fleet("island_done", rec["name"], reason="syncer finished before it joined")
        print(f"[launcher] {rec['name']}: the syncer finished and the island has not reached it "
              f"after {LATE_ISLAND_GRACE_S:.0f}s; stopping it", file=sys.stderr)
        return True

    def _awaiting_syncer(self, rec) -> bool:
        if self.awaiting_syncer_probe is None:
            return False
        try:
            return bool(self.awaiting_syncer_probe(rec["name"]))
        except Exception:
            return False

    def _syncer_finished(self) -> bool:
        if self.syncer is not None and self.syncer["state"] == DONE \
                and "SUCCEEDED" in str(self.syncer.get("exit", "")):
            return True
        if self.syncer_finished_probe is None:
            return False
        try:
            return bool(self.syncer_finished_probe())
        except Exception:
            return False

    def _rejected_link(self, rec) -> bool:
        """A new rejected link probe since the island's last (re)launch."""
        if self.wan_reject_probe is None:
            return False
        try:
            seen = int(self.wan_reject_probe(rec["name"]))
        except Exception:
            return False
        return seen > rec.get("wan_rejects_handled", 0)

    def _relaunch_rejected_link(self, rec) -> None:
        rec["wan_rejects_handled"] = rec.get("wan_rejects_handled", 0) + 1
        used = rec.get("wan_relaunches", 0)
        if used >= self.wan_probe_relaunches:
            self._fleet("island_lost", rec["name"], reason="link probe rejected; relaunch budget used")
            print(f"[launcher] {rec['name']}: link probe rejected again after {used} new "
                  "container(s); not relaunched", file=sys.stderr)
            self._abandon(rec, 0.0, reason="link probe rejected")
            return
        rec["wan_relaunches"] = used + 1
        self._fleet("island_lost", rec["name"],
                    reason=f"link probe rejected; new container {used + 1}/{self.wan_probe_relaunches}")
        print(f"[launcher] {rec['name']}: link probe rejected; requesting a new container "
              f"({used + 1}/{self.wan_probe_relaunches})", file=sys.stderr)
        rec["state"] = RECOVERING
        rec["failed_at"] = self.ops.now()
        rec["wan_recovery"] = True
        rec.pop("recovered_at", None)
        self._drive_recovery(rec, False)

    def _left_pool(self, rec) -> bool:
        if self.left_pool_probe is None:
            return False
        try:
            return bool(self.left_pool_probe(rec["name"]))
        except Exception:
            return False

    def _check_stall(self) -> None:
        if self.progress_probe is None or self.stall_timeout <= 0:
            return
        try:
            progress = int(self.progress_probe())
        except Exception:
            return
        now = self.ops.now()
        if progress == 0:
            return  # islands still starting (image pull, model load): no clock yet
        if self._progress is None or progress != self._progress:
            self._progress, self._progress_at = progress, now
            return
        silent = now - self._progress_at
        if silent >= self.stall_timeout and not self._all_learners_finalized():
            message = (f"no island event for {silent:.0f}s "
                       f"(stall timeout {self.stall_timeout:.0f}s, {progress} events so far)")
            print(f"[launcher] ERROR: run stalled: {message}", file=sys.stderr)
            raise RunStalled(message, silent_s=silent, timeout_s=self.stall_timeout)

    def _all_learners_finalized(self) -> bool:
        return self.finalized_probe is not None and all(
            r["state"] == DONE or self._finalized(r) for r in self.learners.values()
        )

    def _finalized(self, rec) -> bool:
        if self.finalized_probe is None:
            return False
        try:
            return bool(self.finalized_probe(rec["name"]))
        except Exception:
            return False

    def _await_finalized(self, rec) -> bool:
        """Hold a failure verdict for FINALIZE_GRACE_S while the log stream may
        still deliver the island's rl_learner_finalized record."""

        if self.finalized_probe is None:
            return False
        first = rec.setdefault("failure_seen_at", self.ops.now())
        if self.ops.now() - first < FINALIZE_GRACE_S:
            return True
        rec.pop("failure_seen_at", None)
        return False

    def _probe(self, rec):
        """Classify a running cluster: (None, status) if healthy,
        ("succeeded", status), or (failure reason, status)."""
        name, job_id = rec["name"], rec["job_id"]
        try:
            status = self.ops.job_status(name, job_id)
        except Exception as e:
            # sky refuses job_status while the cluster's health reads INIT
            # (e.g. its runtime Ray is unreachable).  That is not evidence
            # the job is gone: ask the job table before treating it as lost.
            if self._job_alive(name, job_id):
                return None, None
            return f"job status unavailable ({e})", None
        if status is not None and status.is_terminal():
            if "SUCCEEDED" in str(status):
                return "succeeded", status
            return f"job ended as {status}", status
        try:
            up = self.ops.cluster_up(name)
        except Exception as e:
            return f"cluster status unavailable ({e})", status
        if not up and not self._job_alive(name, job_id):
            return "cluster is not UP (preempted or deleted)", status
        return None, status

    def _job_alive(self, name: str, job_id) -> bool:
        probe = getattr(self.ops, "job_alive", None)
        if probe is None:
            return False
        try:
            return bool(probe(name, job_id))
        except Exception:
            return False

    def _config_error(self, rec) -> str | None:
        """The island's ``[yeto-config-error]`` log line (its own refusal of the launch
        options), read with the strict-failure log probe; None without one. Any roster:
        a configuration error is never worth a relaunch."""
        probe = getattr(self.ops, "rl_strict_failure", None)
        text = None
        if probe is not None:
            try:
                text = probe(rec["name"], rec["job_id"])
            except Exception:
                text = None
        if text is not None and CONFIG_ERROR_MARKER in text:
            return text
        if self.recorder is not None:  # island lifecycle: the source's exit code counts too (exit 10)
            observed = self._source_decision(rec)
            if observed is not None and observed.cause == lifecycle.CONFIG_ERROR:
                return observed.detail or f"exit {CONFIG_ERROR_EXIT}"
        return None

    def _strict_failure(self, rec) -> str | None:
        if not (self.fixed_roster or self.elastic):
            return None
        probe = getattr(self.ops, "rl_strict_failure", None)
        if probe is None:
            return None
        try:
            text = probe(rec["name"], rec["job_id"])
        except Exception:
            return None
        if text is not None and not self.elastic and any(m in text for m in SYNCER_REFUSAL_MARKERS):
            return None  # legacy unchanged: a refused island takes the fixed-roster path
        return text

    def _source_decision(self, rec):
        """The island source's own decision for its ended incarnation (``ops.island_decision``:
        resident ``ssh:`` islands read their ranks' exit codes, markers and reachability);
        None when the source has none. Recorded only: the relaunch policy is unchanged."""
        probe = getattr(self.ops, "island_decision", None)
        if probe is None:
            return None
        try:
            return probe(rec["name"], rec.get("job_id"))
        except Exception:  # noqa: BLE001 - the record falls back to the generic reading
            return None

    def _enter_recovering(self, rec, reason: str, is_syncer: bool) -> None:
        rec["state"] = RECOVERING
        rec["failed_at"] = self.ops.now()
        if not is_syncer:
            self._fleet("island_lost", rec["name"], reason=str(reason))
            text = str(reason)
            observed = self._source_decision(rec)
            rec["decision"] = observed if observed is not None and observed.relaunch else lifecycle.classify(
                detail=text, preempted="preempt" in text.lower(),
                lost=any(w in text for w in ("not UP", "unavailable", "left the elastic pool")))
        # Fixed-roster RL: the recovery budget is cumulative over relaunches
        # (an island that fails again right after every relaunch must not be
        # relaunched forever while the rest of the fleet waits on the syncer).
        rec["failures"] = rec.get("failures", 0) + 1  # within this recovery window
        rec.pop("recovered_at", None)
        print(
            f"[launcher] {rec['name']}: {reason}; starting recovery "
            f"(timeout {self.recover_timeout}s)",
            file=sys.stderr,
        )
        if not is_syncer and (self.recover_timeout <= 0 or rec["name"] in self.no_recover):
            if rec["name"] in self.no_recover:
                print(
                    f"[launcher] {rec['name']}: auto-recovery disabled for this island "
                    "(sky Verda provisioner unpatched); not relaunching",
                    file=sys.stderr,
                )
            self._abandon(rec, 0.0)
            return
        self._drive_recovery(rec, is_syncer)

    def _drive_recovery(self, rec, is_syncer: bool) -> None:
        if not is_syncer and self._syncer_finished():
            rec["state"] = DONE  # 0.26: never relaunch an island into a finished run
            rec["exit"] = "SUCCEEDED (syncer finished during recovery; not relaunched)"
            print(f"[launcher] {rec['name']}: the syncer finished; recovery stopped", file=sys.stderr)
            return
        attempt = rec["attempt"]
        if attempt is not None and attempt.finished:
            rec["attempt"] = None
            if attempt.blocked is not None and not is_syncer:
                print(
                    f"[launcher] {rec['name']}: NOT relaunching — {attempt.blocked}",
                    file=sys.stderr,
                )
                self._abandon(rec, self.ops.now() - rec["failed_at"])
                return
            if attempt.result is not None and attempt.name and attempt.name != rec["name"]:
                old = rec["name"]
                rec["name"] = attempt.name
                if self.recorder is not None:
                    self.recorder.rename(old, attempt.name)
                print(f"[launcher] {old} relaunched under a new cluster name: {attempt.name}")
                if self.on_rename is not None:
                    try:
                        self.on_rename(old, attempt.name)
                    except Exception as e:  # noqa: BLE001 - bookkeeping only
                        print(f"[launcher] on_rename hook failed: {e}", file=sys.stderr)
            if attempt.result is not None:
                rec["job_id"] = attempt.result
                rec["state"] = RUNNING
                rec["incarnation"] = int(rec.get("incarnation") or 0) + 1
                rec["relaunches"] = int(rec.get("relaunches") or 0) + 1
                # time spent recovering counts toward the window's budget
                rec["recovering_s"] = rec.get("recovering_s", 0.0) + (
                    self.ops.now() - rec["failed_at"])
                rec["failed_at"] = None
                rec["replace_planned"] = False
                rec.pop("wan_recovery", None)
                rec["recovered_at"] = self.ops.now()
                print(
                    f"[launcher] {rec['name']} recovered: relaunched as job "
                    f"{attempt.result}"
                )
                if not is_syncer:
                    self._fleet("island_ready", rec["name"], relaunched=True)
                if self.on_relaunch is not None:
                    self.on_relaunch(rec["name"], attempt.result)
                return
            print(
                f"[launcher] relaunch attempt for {rec['name']} failed; will retry",
                file=sys.stderr,
            )
        elapsed = self.ops.now() - rec["failed_at"]
        if self.fixed_roster and not is_syncer and not rec.get("left_pool"):
            elapsed += rec.get("recovering_s", 0.0)  # this window only
            if rec.get("failures", 0) > FIXED_ROSTER_MAX_RELAUNCHES:
                self._abandon(rec, elapsed)
                return
        # A rejected-link relaunch has its own budget (the island never joined).
        timeout = WAN_RECOVERY_S if rec.get("wan_recovery") else self.recover_timeout
        if timeout <= 0 or elapsed > timeout:
            if is_syncer:
                # The syncer is never abandoned: without it no learner can
                # make outer progress, so keep trying and complain loudly.
                print(
                    f"[launcher] ERROR: syncer unrecovered for {elapsed:.0f}s "
                    f"(recover timeout {self.recover_timeout}s exceeded); "
                    "still retrying — learners cannot sync until it returns",
                    file=sys.stderr,
                )
            else:
                self._abandon(rec, elapsed)
                return
        if rec["attempt"] is None:
            if self._stop_requested(rec, "recovery"):
                if not is_syncer:  # the syncer is never abandoned; it just is not relaunched
                    self._abandon(rec, elapsed, reason="STOP flag")
                return
            if self.recorder is not None:  # an island's or the syncer's relaunch is an order
                try:  # the lock still ours, checked now; lost raises IslandRecordsFenced (supervision ends)
                    self.recorder.store.guard_order(rec["name"])
                except lifecycle.IslandRecordsUnconfirmed as error:
                    print(f"[launcher] {rec['name']}: relaunch deferred: {error}", file=sys.stderr)
                    return
            if not is_syncer and self.replacement_planner is not None and not rec.get("replace_planned"):
                rec["replace_planned"] = True
                planned = self._plan_replacement(rec)
                if planned is not None:
                    rec["attempt"] = planned
                    return
            if not is_syncer:
                self._add_relaunch_envs(rec["task"])
            rec["attempt"] = self._start_relaunch(rec)

    def _add_relaunch_envs(self, task) -> None:
        if not self.island_relaunch_envs:
            return
        if callable(getattr(task, "update_envs", None)):  # sky.Task
            task.update_envs(self.island_relaunch_envs)
        elif isinstance(getattr(task, "envs", None), dict):  # ModalIslandConfig
            task.envs.update(self.island_relaunch_envs)
        else:
            raise TypeError(f"cannot add relaunch envs to a {type(task).__name__}")

    def _plan_replacement(self, rec):
        try:
            out = self.replacement_planner.plan(rec["name"])
        except Exception as e:  # noqa: BLE001 - advice never breaks supervision
            print(f"[launcher] {rec['name']}: replacement planner failed: {e}", file=sys.stderr)
            return None
        self._fleet("spot_replace_advice", rec["name"], action=out.get("action"),
                    candidates=len(out.get("candidates") or []))
        if out.get("action") != "launch":
            return None
        attempt = _RelaunchAttempt()
        attempt.result = out.get("result")
        attempt.finished = True
        return attempt

    def _stop_requested(self, rec, where: str) -> bool:
        flag = self.stop_flag
        if flag is None or not os.path.exists(flag):
            return False
        print(f"[launcher] {rec['name']}: STOP flag {flag} present; not relaunching "
              f"({where} ends here)", file=sys.stderr)
        return True

    def _start_relaunch(self, rec) -> _RelaunchAttempt:
        attempt = _RelaunchAttempt()
        name, task = rec["name"], rec["task"]

        def _run():
            target = name
            if self.instance_guard is not None:
                try:
                    verdict = self.instance_guard.check(name)
                except Exception as e:  # noqa: BLE001
                    verdict = ("unknown", f"instance check raised: {e}")
                if verdict is not None:
                    kind, detail = verdict
                    if kind == "alive":
                        attempt.blocked = detail
                        attempt.finished = True
                        return
                    if kind == "unknown":
                        # Could not tell whether the old node is gone: a
                        # relaunch now could delete it. Try again next poll.
                        print(f"[launcher] {name}: {detail}; relaunch deferred", file=sys.stderr)
                        attempt.finished = True
                        return
                    target = detail  # "gone": relaunch under a fresh name
            attempt.name = target
            try:
                if self._stop_requested(rec, "relaunch"):  # set after the check in the poll loop
                    attempt.result = None
                    return
                try:
                    attempt.result = self.ops.relaunch(task, target)
                except RelaunchBlocked as blocked:
                    attempt.blocked, attempt.result = str(blocked), None
                    return
                if attempt.result is not None and self.instance_guard is not None:
                    after = getattr(self.instance_guard, "after_relaunch", None)
                    if after is not None:
                        try:
                            after(target)
                        except Exception as e:  # noqa: BLE001
                            print(f"[launcher] recording ids of {target} failed: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[launcher] relaunch of {name} raised: {e}", file=sys.stderr)
                attempt.result = None
            finally:
                attempt.finished = True
            if attempt.result is not None and (rec["state"] == ABANDONED or rec.get("stopped")):
                # Abandoned while this attempt was in flight, but the
                # relaunch re-provisioned the cluster anyway: tear it back
                # down so nothing is left running unattended.
                if not self.keep_abandoned:
                    self._down(target, force=True)

        thread = self.thread_cls(target=_run, daemon=True)
        attempt.thread = thread
        thread.start()
        return attempt

    def _abandon(self, rec, elapsed: float, reason: str | None = None) -> None:
        rec["state"] = ABANDONED
        self._fleet("island_stop", rec["name"], reason=f"abandoned: {reason or ''}".strip())
        rec["exit"] = (f"ABANDONED ({reason}) after {elapsed:.0f}s" if reason
                       else f"ABANDONED after {elapsed:.0f}s")
        attempt = rec.get("attempt")
        if attempt is not None and attempt.thread is not None and not attempt.finished:
            # let an in-flight relaunch finish (bounded) so its cluster is torn
            # down here (the thread downs it when it sees ABANDONED), not left
            # behind after the process exits
            attempt.thread.join(RELAUNCH_JOIN_S)
            if not attempt.finished:
                print(f"[launcher] WARN: relaunch of {rec['name']} still in flight after "
                      f"{RELAUNCH_JOIN_S:.0f}s; verify the provider for leftovers",
                      file=sys.stderr)
        if self.keep_abandoned:
            print(f"[launcher] keeping abandoned cluster {rec['name']} (--keep-abandoned)")
        else:
            self._down(rec["name"])
        if self.fixed_roster and not rec.get("left_pool"):  # 0.27: elastic pool leave is not 4
            message = (
                f"fixed-roster learner {rec['name']} could not recover"
                f"{' (' + reason + ')' if reason else ''} "
                f"({rec.get('failures', 0)} failure(s) in the window, "
                f"{elapsed:.0f}s recovering; timeout {self.recover_timeout}s)"
            )
            print(f"[launcher] ERROR: {message}", file=sys.stderr)
            raise FixedRosterIslandAbandoned(message)
        remaining = sum(1 for r in self.learners.values() if r["state"] != ABANDONED)
        print(
            f"[launcher] LEARNER {rec['name']} ABANDONED after {elapsed:.0f}s "
            f"(could not recover within {self.recover_timeout}s); "
            f"fleet continues with {remaining} learner(s)",
            file=sys.stderr,
        )

    def _retry_pending_downs(self) -> None:
        """One lock check per poll for every pending teardown: not confirmed, the whole round
        waits (a store that hangs is asked once, not once per island); lost, nothing is torn
        down (the launcher holding the records owns them)."""
        if not self.pending_downs:
            return
        if self.recorder is not None:
            try:
                self.recorder.store.guard_order("pending teardowns")
            except lifecycle.IslandRecordsUnconfirmed as error:
                if error.lease_over:
                    return
            except lifecycle.IslandRecordsFenced:
                pass  # _down below sees the fence and tears nothing down
        pending, self.pending_downs = self.pending_downs, {}
        for name, force in sorted(pending.items()):
            self._down(name, force=force, checked=True)

    def _down(self, name: str, force: bool = False, *, checked: bool = False) -> None:
        if self.recorder is not None and not checked:  # the lock checked now, not only the cached view
            try:
                self.recorder.store.guard_order(f"teardown of {name}")
            except lifecycle.IslandRecordsFenced:
                pass  # the cached view below says so too
            except lifecycle.IslandRecordsUnconfirmed as error:
                if error.lease_over:  # another launcher may hold the records: no teardown now
                    if name not in self.pending_downs:
                        print(f"[launcher] NOT tearing down {name} now: {error}; tried again once the "
                              "lock is confirmed", file=sys.stderr)
                    self.pending_downs[name] = self.pending_downs.get(name, False) or force
                    return
                # within this holder's lease: still ours (nobody claims the records without the store)
                print(f"[launcher] {name}: tearing down unconfirmed ({error})", file=sys.stderr)
        fenced = self._records_fenced()
        if fenced is not None:  # the launcher that holds the island records owns this machine now
            print(f"[launcher] NOT tearing down {name}: this launcher lost the island records' lock "
                  f"({fenced})", file=sys.stderr)
            return
        if name in self.downed_clusters and not force:
            return
        self.downed_clusters.add(name)
        print(f"[launcher] tearing down {name}")
        try:
            self.ops.down(name)
        except Exception as e:
            print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)
