"""The syncer launch of a fleet launched island by island (``--rl-fleet-role syncer``).

``yeto.rl.island_launches`` splits a per-island fleet into one launch of the syncer and
one launch per trainer island. The syncer launch runs :func:`yeto.launcher.run` up to
its syncer (a VM of its own, or the head's subprocess under ``--controller head``) and
then calls :func:`serve`:

1. once the syncer process accepts connections on every port (:func:`wait_listening`:
   checked on the syncer's machine itself, up to ``LISTEN_WAIT_S``; a syncer built on its
   VM takes minutes after the VM is up), it records the syncer's address and term in the run store
   (``fleet/island_syncer.json``, :func:`yeto.rl.island_launches.write_syncer_record`): the
   trainer worker of ``yeto run up`` gives them to every island launch
   (``--rl-fleet-syncer``, ``--rl-syncer-epoch``);
2. it supervises the syncer alone until the syncer ends its run (exit 0 / its status
   ``finished``), the run's STOP flag appears (exit 9, the launcher's
   ``RUN_STOPPED_EXIT``) or the syncer fails for good (exit 1): a strict failure, a
   cluster job that ended in failure, or a cluster that stayed unreadable or down for
   ``FAILURE_GRACE_S`` (``--rl-fleet-syncer-grace-s``; it includes time the sky API itself
   does not answer: a long API outage ends the fleet). A status read that fails is asked of the job table first
   (``job_alive``, as the fleet controller's probe does): a transient sky error never
   ends the fleet. A head subprocess syncer that dies is restarted (it resumes from its
   checkpoint, and its record is written again), as the fleet controller does.

The launcher's ``finally`` tears the syncer cluster down afterwards, as for any launch.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

POLL_S = 15.0
FAILURE_GRACE_S = 300.0  # a syncer cluster unreadable / not UP this long (in a row) has failed


class Final(str):
    """A failure that is final at once (a job that ended in failure): no grace."""


def _store_objects(args: Any) -> Any:
    from .island_lifecycle import store_objects
    from .storage import store_client_environ

    return store_objects(getattr(args, "rl_checkpoint_store", None), store_client_environ(args))


# How long the syncer launch waits for the syncer process to listen before it records the
# syncer for the islands: the VM is up (its address known) minutes before a syncer built on
# it (a non-x86 submitter) listens, and an island started then fails its ~30 s probe.
LISTEN_WAIT_S = 1200.0
LISTEN_POLL_S = 5.0
EPOCH_READ_S = 120.0  # an adopted syncer: how long its status.json is read for its term
RECORD_READ_ATTEMPTS = 3
RECORD_READ_RETRY_S = 5.0


class SyncerRecordUnreadable(RuntimeError):
    """The fleet's last syncer record exists (or may) but does not read: its term is
    unknown, so no record is written that could carry a lower one."""


def _recorded_epoch(args: Any, objects: Any, log: Callable[[str], None],
                    sleep: Callable[[float], None] = time.sleep) -> int | None:
    """The syncer_epoch of the fleet's last syncer record (an earlier syncer launch of this
    run); None when there is none, or it carries none (0 then). A store that does not answer
    is retried; still unreadable, or a record that does not parse:
    :class:`SyncerRecordUnreadable` (the term must never be written lower than it was)."""
    import json

    from ..rl.island_launches import SYNCER_RECORD

    for attempt in range(1, RECORD_READ_ATTEMPTS + 1):
        try:
            store = objects if objects is not None else _store_objects(args)
            raw = store.read(SYNCER_RECORD) if store is not None else None
            break
        except FileNotFoundError:  # a store that says "not there" by raising: no record
            raw = None
            break
        except Exception as error:  # noqa: BLE001 - retried, then refused
            if attempt == RECORD_READ_ATTEMPTS:
                raise SyncerRecordUnreadable(
                    f"the fleet's syncer record ({SYNCER_RECORD} in the run store) did not read after "
                    f"{attempt} attempts ({type(error).__name__}: {error}): its syncer_epoch is unknown and is "
                    "never written lower; retry once the run store answers, or give --rl-syncer-epoch") from error
            log(f"[launcher] the fleet's syncer record did not read ({type(error).__name__}: {error}); "
                f"retrying ({attempt}/{RECORD_READ_ATTEMPTS})")
            sleep(RECORD_READ_RETRY_S)
    if raw is None:
        return None
    try:
        record = json.loads(raw)
        epoch = record.get("syncer_epoch")
        if epoch is not None and (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0):
            raise ValueError(f"syncer_epoch {epoch!r}")
    except (ValueError, AttributeError) as error:
        raise SyncerRecordUnreadable(
            f"the fleet's syncer record ({SYNCER_RECORD} in the run store) does not parse ({error}): its "
            "syncer_epoch is unknown and is never written lower; give --rl-syncer-epoch at least the term "
            "the fleet's syncer ran (its status.json), or delete the record if no island launch runs") from None
    return epoch


def resolve_epoch(args: Any, role: str, *, adopted: bool, reader: Callable[[], Callable[[], Any]],
                  objects: Any = None, timeout_s: float | None = None,
                  log: Callable[[str], None] = print) -> int:
    """The syncer_epoch of a fleet launched island by island, on ``args.rl_syncer_epoch``
    (never left unset: the syncer record carries it to every island launch).

    * an island launch: the term its syncer record gave (``--rl-syncer-epoch``); a record
      without one (an earlier syncer launch) is 0. Nothing is read.
    * the syncer launch: the term its syncer runs, never lower than the fleet's last
      record (:func:`_recorded_epoch`; a HELLO at a higher epoch is admitted, a lower one
      fenced, and a syncer resumed from its checkpoint keeps its term): the higher of the
      record and its own term. Its own term: an adopted running syncer's status.json
      (``reader()``, ``timeout_s``) when it reads, else the term it was started with
      (``--syncer-epoch``, :func:`yeto.launch.syncer.island_syncer_epoch`). A syncer this
      launch just started is not read: it writes status.json only once its first learners
      are in, and the island launches wait for this record."""
    from ..rl.engine.sample_roles import FLEET_ROLE_ISLAND
    from .syncer import island_syncer_epoch, resolve_island_syncer_epoch

    if role == FLEET_ROLE_ISLAND:
        if getattr(args, "rl_syncer_epoch", None) is None:
            log("[launcher] WARN: the fleet's syncer record carries no syncer_epoch (an earlier syncer launch); "
                "the island JOINs with syncer_epoch 0")
        args.rl_syncer_epoch = island_syncer_epoch(args)
        return args.rl_syncer_epoch
    recorded = _recorded_epoch(args, objects, log)
    own = island_syncer_epoch(args)
    if adopted and getattr(args, "rl_syncer_epoch", None) is None:
        own = resolve_island_syncer_epoch(args, reader(), timeout_s=EPOCH_READ_S if timeout_s is None else timeout_s)
    epoch = max(own, recorded or 0)
    log(f"[launcher] fleet syncer {'adopted' if adopted else 'started'} at syncer_epoch {own}"
        + (f" (the fleet's last record: {recorded})" if recorded is not None else "")
        + f": the islands JOIN with syncer_epoch {epoch}")
    args.rl_syncer_epoch = epoch
    return epoch


def record(args: Any, syncer_addr: str, num_learners: int, *, objects: Any = None) -> dict[str, Any]:
    """Write ``fleet/island_syncer.json``; raises when the run store has no object API (the
    island launches could not find the syncer)."""
    from ..rl.island_launches import write_syncer_record

    objects = objects if objects is not None else _store_objects(args)
    if objects is None:
        raise ValueError(f"--rl-fleet-role syncer: the run store {getattr(args, 'rl_checkpoint_store', None)!r} "
                         "has no object API here; the island launches read the syncer's address from it")
    out = write_syncer_record(objects, addr=syncer_addr, syncer_epoch=getattr(args, "rl_syncer_epoch", None),
                              learners=num_learners, max_learners=getattr(args, "rl_sync_max_learners", None),
                              run=args.cluster_prefix)
    print(f"[launcher] fleet syncer at {syncer_addr} (epoch {out['syncer_epoch']}, {num_learners} launched "
          f"learner(s), ids below {out['max_learners']}) recorded in the run store (fleet/island_syncer.json)", flush=True)
    return out


_LISTEN_SCRIPT = ("import socket, sys\n"
                  "for port in sys.argv[1:]:\n"
                  "    socket.create_connection(('127.0.0.1', int(port)), 3).close()\n")


def syncer_listening(args: Any, *, head_mode: bool, syncer_cluster: str | None,
                     run: Callable[..., Any] | None = None) -> bool:
    """Whether every syncer port accepts a connection on the syncer's own machine: this
    host under ``--controller head``, else ``ssh <syncer cluster>``. A probe from this
    machine to the public address cannot tell: a local proxy accepts any connection."""
    import shlex
    import socket
    import subprocess

    from .syncer import _head_ports

    ports = [int(port) for port in _head_ports(args)]
    if head_mode:
        try:
            for port in ports:
                socket.create_connection(("127.0.0.1", port), 3).close()
        except OSError:
            return False
        return True
    if not syncer_cluster:
        return True  # nothing to check against: recorded as before
    command = f"python3 -c {shlex.quote(_LISTEN_SCRIPT)} " + " ".join(str(port) for port in ports)
    try:
        out = (run or subprocess.run)(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", syncer_cluster,
                                       command], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def wait_listening(listening: Callable[[], bool], failed: Callable[[], str | None], *, where: str,
                   timeout_s: float = LISTEN_WAIT_S, poll_s: float = LISTEN_POLL_S,
                   sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] | None = None,
                   log: Callable[[str], None] = print) -> str | None:
    """Wait until the syncer process listens (``listening()``); None then. The reason it
    never will otherwise: its job ended in failure (``failed()`` gives a :class:`Final`),
    or nothing listened within ``timeout_s``. Nothing is recorded meanwhile, so no
    island dials a syncer that is not up yet (bug 25)."""
    clock = clock or time.monotonic
    started = clock()
    said = False
    while True:
        if listening():
            if said:
                log(f"[launcher] fleet syncer: listening at {where} after {clock() - started:.0f}s")
            return None
        why = failed()
        if isinstance(why, Final):
            return f"{why} before it listened at {where}"
        waited = clock() - started
        if waited >= timeout_s:
            return (f"nothing listens at {where} {waited:.0f}s after the syncer launch (its process did "
                    "not start: see the syncer job's log); no island was started")
        if not said:
            log(f"[launcher] fleet syncer: waiting for the syncer process to listen at {where} "
                f"(up to {timeout_s:.0f}s) before the islands are started")
            said = True
        sleep(poll_s)


def supervise(*, finished: Callable[[], bool], failed: Callable[[], str | None],
              restart: Callable[[str], bool] | None = None, stop_flag: str | os.PathLike[str] | None = None,
              sleep: Callable[[float], None] = time.sleep, poll_s: float = POLL_S,
              grace_s: float = FAILURE_GRACE_S, clock: Callable[[], float] | None = None,
              log: Callable[[str], None] = print) -> int:
    """Poll the syncer until it ends (0), the STOP flag appears (9) or it fails for good
    (1). ``failed()``: why it is down (None: up; a :class:`Final`: failed for good now);
    ``restart(why)``: True when it was restarted (a head subprocess), False: the failure
    stands. A failure that is not final and not restarted must last ``grace_s`` in a
    row before it ends the syncer."""
    from .fleet import RUN_STOPPED_EXIT

    clock = clock or time.monotonic
    down_since: float | None = None
    while True:
        if finished():
            log("[launcher] fleet syncer: its run is finished")
            return 0
        if stop_flag is not None and os.path.exists(stop_flag):
            log(f"[launcher] fleet syncer: STOP flag {stop_flag}; ending the syncer")
            return RUN_STOPPED_EXIT
        why = failed()
        if why is None:
            if down_since is not None:
                log("[launcher] fleet syncer: readable and up again")
            down_since = None
        elif restart is not None and restart(why):
            down_since = None
        elif isinstance(why, Final):
            log(f"[launcher] ERROR: fleet syncer: {why}")
            return 1
        else:
            now = clock()
            if down_since is None:
                down_since = now
                log(f"[launcher] WARNING: fleet syncer: {why}; failing it after {grace_s:.0f}s of this")
            elif now - down_since >= grace_s:
                log(f"[launcher] ERROR: fleet syncer: {why} for {now - down_since:.0f}s")
                return 1
        sleep(poll_s)


def _sky_ops() -> Any:
    from .sky_ops import SkySDKOps

    return SkySDKOps()


def probe(ops: Any, cluster: str | None, job: Any) -> tuple[str | None, Any]:
    """The syncer job's state, as the fleet controller's ``_probe`` reads an island:
    ``(None, status)`` up, ``("succeeded", status)``, a :class:`Final` reason for a job
    that ended in failure, else a transient reason (unreadable, cluster not UP). A
    status read that raises asks the job table (``job_alive``) before it counts."""
    def alive() -> bool:
        check = getattr(ops, "job_alive", None)
        try:
            return bool(check(cluster, job)) if check is not None else False
        except Exception:  # noqa: BLE001 - "not known alive"
            return False

    try:
        status = ops.job_status(cluster, job)
    except Exception as error:  # noqa: BLE001 - sky refuses job_status while the cluster reads INIT
        if alive():
            return None, None
        return f"job status unavailable ({type(error).__name__}: {error})", None
    if status is not None and getattr(status, "is_terminal", None) and status.is_terminal():
        if "SUCCEEDED" in str(status):
            return "succeeded", status
        return Final(f"syncer job {job} on {cluster} ended {status}"), status
    try:
        up = ops.cluster_up(cluster)
    except Exception as error:  # noqa: BLE001
        return (None, status) if alive() else (f"cluster status unavailable ({error})", status)
    if not up and not alive():
        return "cluster is not UP (preempted or deleted)", status
    return None, status


def serve(args: Any, syncer_addr: str, num_learners: int, *, clusters: list[str], head_mode: bool,
          local_syncer: Any = None, syncer_cluster: str | None = None, syncer_job: Any = None,
          ops: Any = None, sleep: Callable[[float], None] = time.sleep) -> int:
    """Record the syncer once it listens, then supervise it alone (module doc)."""
    from .. import runs

    if head_mode:
        def dead() -> str | None:
            reason = local_syncer.probe()
            return None if reason is None else Final(f"the syncer process ended ({getattr(reason, 'reason', reason)})")
    else:
        ops = ops or _sky_ops()

        def dead() -> str | None:
            return probe(ops, syncer_cluster, syncer_job)[0]
    never = wait_listening(lambda: syncer_listening(args, head_mode=head_mode, syncer_cluster=syncer_cluster),
                           dead, where=syncer_addr, sleep=sleep)
    if never is not None:
        print(f"[launcher] ERROR: fleet syncer: {never}", file=sys.stderr, flush=True)
        return 1
    record(args, syncer_addr, num_learners)
    stop_flag = runs.stop_flag_path(args.cluster_prefix)
    if head_mode:
        from .fleet import _RlStrictFailure

        def failed() -> str | None:
            reason = local_syncer.probe()
            if reason is None:
                return None
            why = str(getattr(reason, "reason", reason))
            return Final(why) if isinstance(reason, _RlStrictFailure) else why

        def restart(why: str) -> bool:
            if isinstance(local_syncer.probe(), _RlStrictFailure):
                return False
            print(f"[launcher] fleet syncer: {why}; restarting it (resumes from its checkpoint)", file=sys.stderr)
            try:
                local_syncer.restart()
            except Exception as error:  # noqa: BLE001 - tried again next poll
                print(f"[launcher] fleet syncer restart failed: {error}", file=sys.stderr)
                return True
            try:  # the islands started from now on read it (same address and term)
                record(args, syncer_addr, num_learners)
            except Exception as error:  # noqa: BLE001 - the earlier record still names it
                print(f"[launcher] fleet syncer: record not rewritten ({error})", file=sys.stderr)
            return True

        return supervise(finished=local_syncer.finished, failed=failed, restart=restart, stop_flag=stop_flag,
                         sleep=sleep)
    last: dict[str, Any] = {}

    def finished() -> bool:
        reason, _status = probe(ops, syncer_cluster, syncer_job)
        last["reason"] = reason
        return reason == "succeeded"

    def failed() -> str | None:
        return last.pop("reason", None)

    grace = getattr(args, "rl_fleet_syncer_grace_s", None)
    return supervise(finished=finished, failed=failed, stop_flag=stop_flag, sleep=sleep,
                     grace_s=FAILURE_GRACE_S if grace is None else float(grace))
