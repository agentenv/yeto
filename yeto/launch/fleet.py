"""The fleet controller: island states, relaunch policy, failure types and exit codes."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

from .syncer import island_scheduling_mode
from .sky_ops import SYNCER_REFUSAL_MARKERS


def controller_fixed_roster(args) -> bool:
    """FleetController ``fixed_roster``: RL runs whose island scheduling is NOT
    elastic. Elastic membership is dynamic: an island that fails for good is
    dropped and the pool goes on (10-09 ruling of the main agent for the user;
    before, every RL run, elastic included, was a fixed roster)."""
    return getattr(args, "training_mode", "sft") == "rl" and island_scheduling_mode(args) != "elastic"


# FleetController states.
RUNNING = "running"
RECOVERING = "recovering"
DONE = "done"
ABANDONED = "abandoned"


@dataclass(frozen=True)
class _RlStrictFailure:
    reason: str


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
# S26 E3b: how long one rejected-link relaunch may take to get a new container.
WAN_RECOVERY_S = 900.0
# How long a failed-looking island is given for its rl_learner_finalized record
# to arrive over the log stream before recovery starts.
FINALIZE_GRACE_S = 60.0
# Bound on waiting for an in-flight relaunch before abandoning (so a late
# sky.launch cannot re-create a cluster after teardown).
RELAUNCH_JOIN_S = 120.0


class RunStalled(RuntimeError):
    """No island produced an event for the stall timeout (and not all finalized)."""


# Exit code when the run stalled (no island event for --rl-stall-timeout).
RUN_STALLED_EXIT = 6
DEFAULT_RL_STALL_TIMEOUT_S = 900.0


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
# 9 stopped on the STOP flag.
CONTAINER_CHANGED_EXIT = 7
# The trainer scheduler stopped after too many failed trainer incarnations in a row.
TRAINER_GAVE_UP_EXIT = 8


class RunStopRequested(RuntimeError):
    """``yeto stop-run`` wrote the run's STOP flag: the run ends and is torn down."""


# Exit code when the run ended on its STOP flag (``yeto stop-run``).
RUN_STOPPED_EXIT = 9


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
        elastic: bool = False,
        no_island_relaunch: bool = False,
        replacement_planner=None,
        wan_reject_probe=None,
        wan_probe_relaunches: int = 0,
    ):
        """`learners` maps cluster name -> (task, job_id); `syncer` is
        (name, task, job_id) for a cluster syncer, or None with
        `syncer_probe`/`syncer_restart` callables for a subprocess syncer.
        `on_relaunch(name, new_job_id)` is called after every successful
        cluster relaunch (production spawns a new log tail)."""
        self.ops = sky_ops
        # 0.26: () -> bool, the syncer finished its run (no island is relaunched
        # after that; a late island's failure then counts as a normal end).
        self.syncer_finished_probe = syncer_finished_probe
        # 0.27: name -> bool, the island left the elastic pool for good (re-JOIN
        # exhausted, learner exit 7); elastic only.
        self.left_pool_probe = left_pool_probe
        self.elastic = elastic
        self.no_island_relaunch = no_island_relaunch
        # S26 E3b: name -> number of rejected link probes the island logged
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
        }

    def abort(self, error: BaseException) -> None:
        """Ask :meth:`run` (any thread) to end with ``error`` at its next poll, e.g.
        the trainer scheduler gave up; the launcher then tears the run down."""
        self._abort_error = error

    def run(self) -> dict:
        """Poll until every learner is done or abandoned.

        Returns {learner name: final status string}; raises RuntimeError
        (after downing the syncer) if every learner was abandoned, the
        :meth:`abort` error when one was set, and :class:`RunStopRequested`
        when the STOP flag appears while islands still run (the launcher then
        stops the trainer with a cut and tears the run down; before, the
        worker kept supervising the live islands until it was killed).
        """
        for name in self.learners:
            self._fleet("island_ready", name, initial=True)
        while True:
            if self._abort_error is not None:
                raise self._abort_error
            self._fleet("cost_tick")
            if self.syncer is not None:
                self._poll(self.syncer, is_syncer=True)
            elif self.syncer_probe is not None:
                self._poll_local_syncer()
            for rec in self.learners.values():
                self._poll(rec, is_syncer=False)
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
                strict_failure = self._strict_failure(rec)
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
        if now - self._progress_at >= self.stall_timeout and not self._all_learners_finalized():
            message = (f"no island event for {now - self._progress_at:.0f}s "
                       f"(stall timeout {self.stall_timeout:.0f}s, {progress} events so far)")
            print(f"[launcher] ERROR: run stalled: {message}", file=sys.stderr)
            raise RunStalled(message)

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

    def _enter_recovering(self, rec, reason: str, is_syncer: bool) -> None:
        rec["state"] = RECOVERING
        rec["failed_at"] = self.ops.now()
        if not is_syncer:
            self._fleet("island_lost", rec["name"], reason=str(reason))
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
                print(f"[launcher] {old} relaunched under a new cluster name: {attempt.name}")
                if self.on_rename is not None:
                    try:
                        self.on_rename(old, attempt.name)
                    except Exception as e:  # noqa: BLE001 - bookkeeping only
                        print(f"[launcher] on_rename hook failed: {e}", file=sys.stderr)
            if attempt.result is not None:
                rec["job_id"] = attempt.result
                rec["state"] = RUNNING
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
            if not is_syncer and self.replacement_planner is not None and not rec.get("replace_planned"):
                rec["replace_planned"] = True
                planned = self._plan_replacement(rec)
                if planned is not None:
                    rec["attempt"] = planned
                    return
            rec["attempt"] = self._start_relaunch(rec)

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
                attempt.result = self.ops.relaunch(task, target)
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
            if attempt.result is not None and rec["state"] == ABANDONED:
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

    def _down(self, name: str, force: bool = False) -> None:
        if name in self.downed_clusters and not force:
            return
        self.downed_clusters.add(name)
        print(f"[launcher] tearing down {name}")
        try:
            self.ops.down(name)
        except Exception as e:
            print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)
