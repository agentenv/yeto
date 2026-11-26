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
import re
import sys
import time
from collections.abc import Callable
from typing import Any

POLL_S = 15.0
FAILURE_GRACE_S = 300.0  # a syncer cluster unreadable / not UP this long (in a row) has failed


class Final(str):
    """A failure that is final at once (a job that ended in failure): no grace."""


_FINISHED_EARLY = "the syncer finished its run"


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


def record(args: Any, syncer_addr: str, num_learners: int, *, objects: Any = None,
           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``fleet/island_syncer.json``; raises when the run store has no object API (the
    island launches could not find the syncer). ``extra``: an on-site service's fields."""
    from ..rl.island_launches import write_syncer_record

    objects = objects if objects is not None else _store_objects(args)
    if objects is None:
        raise ValueError(f"--rl-fleet-role syncer: the run store {getattr(args, 'rl_checkpoint_store', None)!r} "
                         "has no object API here; the island launches read the syncer's address from it")
    out = write_syncer_record(objects, addr=syncer_addr, syncer_epoch=getattr(args, "rl_syncer_epoch", None),
                              learners=num_learners, max_learners=getattr(args, "rl_sync_max_learners", None),
                              run=args.cluster_prefix, extra=extra)
    print(f"[launcher] fleet syncer at {syncer_addr} (epoch {out['syncer_epoch']}, {num_learners} launched "
          f"learner(s), ids below {out['max_learners']}) recorded in the run store (fleet/island_syncer.json)", flush=True)
    return out


_LISTEN_SCRIPT = ("import socket, sys\n"
                  "for port in sys.argv[1:]:\n"
                  "    socket.create_connection(('127.0.0.1', int(port)), 3).close()\n")


# ssh exit codes that say nothing about the syncer: ssh itself failed (255: no host entry,
# host key, auth), or the shell found no python3 (127).
_CANNOT_CHECK_RC = {126, 127, 255}
UNCHECKED_LIMIT = 3  # checks in a row that cannot be made from here: the launch fails, naming why
# The syncer's line once its port is bound (syncer/src/server.rs, no ANSI: main.rs with_ansi(false)).
LISTENING_LINE = re.compile(r"syncer listening port=(\d+)")


def _ssh_config(cluster: str) -> list[str]:
    """``-F <sky's generated ssh config of the cluster>`` when this HOME has it (ssh reads
    the passwd home's config, which a launch under another HOME does not include)."""
    path = os.path.expanduser(os.path.join("~", ".sky", "generated", "ssh", cluster))
    return ["-F", path] if os.path.isfile(path) else []


def _ssh_listening(cluster: str, ports: list[int], run: Callable[..., Any] | None) -> bool | str:
    import shlex
    import subprocess

    command = f"python3 -c {shlex.quote(_LISTEN_SCRIPT)} " + " ".join(str(port) for port in ports)
    try:
        out = (run or subprocess.run)(["ssh", *_ssh_config(cluster), "-o", "BatchMode=yes",
                                       "-o", "ConnectTimeout=10", cluster, command],
                                      capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"ssh {cluster}: {type(error).__name__}: {error}"
    if out.returncode in _CANNOT_CHECK_RC:
        lines = [line for line in str(out.stderr or "").splitlines() if line.strip()]
        return f"ssh {cluster} exited {out.returncode}" + (f": {lines[-1].strip()}" if lines else "")
    return out.returncode == 0


def syncer_listening(args: Any, *, head_mode: bool, syncer_cluster: str | None, syncer_job: Any = None,
                     ops: Any = None, run: Callable[..., Any] | None = None) -> bool | str:
    """Whether every syncer port is bound on the syncer's own machine: this host under
    ``--controller head`` (a connect to 127.0.0.1), else the syncer job's log (its
    ``syncer listening port=N`` lines, read through sky: no ssh), else ``ssh <syncer
    cluster>`` and a connect to 127.0.0.1 there. A probe from this machine to the public
    address cannot tell: a local proxy accepts any connection. A string: why it cannot be
    checked from here (the log unreadable and ssh failed, or no python3 there)."""
    import socket

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
    log_read: str | None = None
    lines = getattr(ops, "job_log_lines", None)
    if callable(lines) and syncer_job is not None:
        try:
            bound = {int(m.group(1)) for line in lines(syncer_cluster, syncer_job)
                     if (m := LISTENING_LINE.search(str(line)))}
        except Exception as error:  # noqa: BLE001 - ssh is asked instead
            log_read = f"its job log: {type(error).__name__}: {error}"
        else:
            if set(ports) <= bound:
                return True
    state = _ssh_listening(syncer_cluster, ports, run)
    if isinstance(state, str):
        # a readable log that has no line yet: not listening (ssh is only the second witness)
        return f"{log_read}; {state}" if log_read else (False if callable(lines) and syncer_job is not None
                                                        else state)
    return state


def wait_listening(listening: Callable[[], bool | str], failed: Callable[[], str | None], *, where: str,
                   timeout_s: float = LISTEN_WAIT_S, poll_s: float = LISTEN_POLL_S,
                   sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] | None = None,
                   log: Callable[[str], None] = print) -> str | None:
    """Wait until the syncer process listens (``listening()``); None then. The reason it
    never will otherwise: its job ended in failure (``failed()`` gives a :class:`Final`),
    nothing listened within ``timeout_s``, or the check could not be made from here
    (``listening()`` gave a reason :data:`UNCHECKED_LIMIT` times in a row: named, not
    waited out). Nothing is recorded meanwhile, so no island dials a syncer that is not
    up yet (bug 25)."""
    clock = clock or time.monotonic
    started = clock()
    said = False
    unchecked = 0
    while True:
        state = listening()
        if state is True:
            if said:
                log(f"[launcher] fleet syncer: listening at {where} after {clock() - started:.0f}s")
            return None
        if isinstance(state, str):
            unchecked += 1
            if unchecked >= UNCHECKED_LIMIT:
                return (f"cannot check whether the syncer listens at {where}: {state} ({unchecked} checks in a "
                        "row); no island was started")
        else:
            unchecked = 0
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
              log: Callable[[str], None] = print, stopped: Callable[[], str | None] | None = None) -> int:
    """Poll the syncer until it ends (0), the STOP flag appears (9) or it fails for good
    (1). ``failed()``: why it is down (None: up; a :class:`Final`: failed for good now);
    ``restart(why)``: True when it was restarted (a head subprocess), False: the failure
    stands. A failure that is not final and not restarted must last ``grace_s`` in a
    row before it ends the syncer. ``stopped()``: a stop that is not a local flag (an
    on-site syncer service reads the fleet's STOP in the run store): why, or None."""
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
        why_stopped = stopped() if stopped is not None else None
        if why_stopped:
            log(f"[launcher] fleet syncer: {why_stopped}; ending the syncer")
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
          ops: Any = None, sleep: Callable[[float], None] = time.sleep,
          record_extra: dict[str, Any] | None = None, on_record: Callable[[dict[str, Any]], None] | None = None,
          stopped: Callable[[], str | None] | None = None, early_finish: bool = False) -> int:
    """Record the syncer once it listens, then supervise it alone (module doc).
    An on-site syncer service: ``record_extra``, the fields it adds to each record it
    writes; ``on_record(record)``, called after each record; ``stopped``, the fleet's STOP
    in the run store (:func:`supervise`); ``early_finish``, a head syncer that finished its
    run before it was seen listening (resumed at its last step) ends the wait: 0."""
    from .. import runs

    if head_mode:
        def dead() -> str | None:
            reason = local_syncer.probe()
            if reason is None and early_finish and local_syncer.finished():
                return Final(_FINISHED_EARLY)
            return None if reason is None else Final(f"the syncer process ended ({getattr(reason, 'reason', reason)})")
    else:
        ops = ops or _sky_ops()

        def dead() -> str | None:
            return probe(ops, syncer_cluster, syncer_job)[0]
    never = wait_listening(lambda: syncer_listening(args, head_mode=head_mode, syncer_cluster=syncer_cluster,
                                                    syncer_job=syncer_job, ops=ops),
                           dead, where=syncer_addr, sleep=sleep)
    if never is not None and early_finish and head_mode and never.startswith(_FINISHED_EARLY):
        print("[launcher] fleet syncer: its run is finished (before it was seen listening)", flush=True)
        return 0
    if never is not None:
        print(f"[launcher] ERROR: fleet syncer: {never}", file=sys.stderr, flush=True)
        return 1
    recorded = record(args, syncer_addr, num_learners, extra=record_extra)
    if on_record is not None:
        on_record(recorded)
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
                recorded = record(args, syncer_addr, num_learners, extra=record_extra)
                if on_record is not None:
                    on_record(recorded)
            except Exception as error:  # noqa: BLE001 - the earlier record still names it
                print(f"[launcher] fleet syncer: record not rewritten ({error})", file=sys.stderr)
            return True

        return supervise(finished=local_syncer.finished, failed=failed, restart=restart, stop_flag=stop_flag,
                         sleep=sleep, stopped=stopped)
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




# -- an on-site syncer service (fleet.syncer_service; ``yeto fleet service --role syncer``) ---------
#
# The syncer of a fleet launched island by island can run as a service of its own on the
# operator's machine (``yeto.rl.fleet_service``): a systemd unit restarts it, and it
# resumes from its checkpoint. One service drives the syncer at a time: it holds the
# service lock (:data:`SERVICE_LOCK`, a store lease its heartbeat renews), and the lock's
# token is the service's generation. It writes the syncer record (``fleet/island_syncer.json``,
# with ``service``, ``service_node``, ``service_generation`` and ``key_id``) and a heartbeat
# (:data:`SERVICE_RECORD`): ``starting``, then ``running`` once the record is written,
# then a final state (``finished``, ``failed``, ``stopped``). The fleet's syncer launch
# (``--rl-fleet-syncer-service HOST``) orders no syncer: it waits for that service, checks that
# both read the same island HMAC key, records the syncer again for this run up (a fresh ``at``)
# and watches the heartbeat (:func:`watch_service`). On the run's STOP flag it writes the
# fleet's STOP to the run store (:data:`ONSITE_STOP`), which ends the service too. The service
# keeps running when the control plane is lost, so the islands keep merging meanwhile.

SERVICE_RECORD = "fleet/island_syncer_service.json"
SERVICE_SCHEMA = "yeto.fleet.syncer-service/v1"
SERVICE_BEAT_S = 30.0  # the service rewrites its heartbeat (and renews its lock) this often
SERVICE_STALE_S = 3 * SERVICE_BEAT_S  # no new heartbeat seen this long (this host's clock): not up
SERVICE_LOCK = "fleet/syncer_service/lock"  # one syncer service at a time (StoreLock)
SERVICE_LOCK_TTL_S = 4 * SERVICE_BEAT_S
ONSITE_STOP = "fleet/onsite/STOP"  # the fleet's STOP in the run store: the on-site services end on it
ONSITE_STOP_SCHEMA = "yeto.fleet.onsite-stop/v1"
SERVICE_STARTING, SERVICE_RUNNING = "starting", "running"
SERVICE_FINISHED, SERVICE_FAILED, SERVICE_STOPPED = "finished", "failed", "stopped"
SERVICE_FINAL = frozenset({SERVICE_FINISHED, SERVICE_FAILED, SERVICE_STOPPED})
_KEY_ID_DOMAIN = b"yeto.island-hmac-key-id/v1\0"


def key_id(key: str | None) -> str | None:
    """A name of the island HMAC key that tells nothing of the key (a domain-separated
    SHA-256 prefix); None without a key. Kept in the run store so that two hosts can tell
    whether they read the same key, never the key itself."""
    if not key:
        return None
    import hashlib

    return hashlib.sha256(_KEY_ID_DOMAIN + str(key).encode()).hexdigest()[:16]


def environ_key_id(environ: Any = None) -> str | None:
    from .syncer import ISLAND_HMAC_KEY_ENV

    return key_id((os.environ if environ is None else environ).get(ISLAND_HMAC_KEY_ENV))


def write_service_status(objects: Any, *, host: str, state: str, key: str | None, reason: str | None = None,
                         generation: int | None = None, node: str | None = None,
                         clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """The on-site syncer service's heartbeat (:data:`SERVICE_RECORD`). ``at`` is the
    service host's clock: a reader on another host only looks whether it moves."""
    import json

    doc = {"schema": SERVICE_SCHEMA, "host": str(host), "state": str(state), "key_id": key,
           "at": round(clock(), 3), **({"reason": str(reason)[:500]} if reason else {}),
           **({"generation": int(generation)} if generation is not None else {}),
           **({"node": str(node)} if node else {})}
    objects.write(SERVICE_RECORD, json.dumps(doc, sort_keys=True).encode())
    return doc


def _read_doc(objects: Any, key: str, schema: str) -> dict[str, Any] | None:
    import json

    raw = objects.read(key)
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == schema else None


def read_service_status(objects: Any) -> dict[str, Any] | None:
    return _read_doc(objects, SERVICE_RECORD, SERVICE_SCHEMA)


def write_onsite_stop(objects: Any, *, why: str, by: str, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """The fleet's STOP in the run store (:data:`ONSITE_STOP`): the on-site syncer service
    ends on it (``stopped``, exit 9), whichever host it runs on."""
    import json

    doc = {"schema": ONSITE_STOP_SCHEMA, "why": str(why)[:500], "by": str(by), "at": round(clock(), 3)}
    objects.write(ONSITE_STOP, json.dumps(doc, sort_keys=True).encode())
    return doc


def read_onsite_stop(objects: Any) -> dict[str, Any] | None:
    return _read_doc(objects, ONSITE_STOP, ONSITE_STOP_SCHEMA)


def key_mismatch(mine: str | None, theirs: str | None) -> str | None:
    """Why this host may not use the syncer service's key (None: the same key)."""
    from .syncer import ISLAND_HMAC_KEY_ENV

    if mine is None:
        return (f"{ISLAND_HMAC_KEY_ENV} is not set on this host: with an on-site syncer service the island HMAC "
                "key comes from the env file both units read (none is generated: the syncer would refuse every "
                "island)")
    if theirs is not None and theirs != mine:
        return (f"the island HMAC key on this host (key id {mine}) is not the syncer service's (key id {theirs}): "
                "every island would be refused. Give both units the same env file (or the same "
                f"{ISLAND_HMAC_KEY_ENV}); no key is generated")
    return None


class _Beats:
    """Whether a heartbeat moves, by this host's monotonic clock: its ``at`` (the service
    host's clock) is only compared with the ``at`` seen before, never with this host's time.
    ``moved``: a heartbeat written after the first look (a new ``at``)."""

    _UNSEEN = object()

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.at: Any = self._UNSEEN
        self.since = clock()
        self.moved = False

    def see(self, status: dict[str, Any] | None) -> None:
        at = None if status is None else status.get("at")
        if self.at is self._UNSEEN:
            self.at, self.since = at, self.clock()
        elif at != self.at:
            self.at, self.since = at, self.clock()
            self.moved = self.moved or at is not None

    def silent_s(self) -> float:
        return self.clock() - self.since


def watch_service(args: Any, *, objects: Any = None, sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic, wait_s: float = LISTEN_WAIT_S,
                  poll_s: float = LISTEN_POLL_S, log: Callable[[str], None] = print) -> int:
    """The syncer launch of a fleet with an on-site syncer service (module section above):
    0 once the service says its run finished, 9 on the run's STOP flag (written to the run
    store as the fleet's STOP, which ends the service) or when the service stopped, 1 when
    it failed or stayed down past the grace, :data:`CONFIG_ERROR_EXIT` when the island HMAC
    key is missing here or is not the service's. ``clock``: this host's monotonic clock."""
    from .. import runs
    from ..config_errors import CONFIG_ERROR_EXIT
    from ..rl.island_launches import read_syncer_record, write_syncer_record
    from .fleet import RUN_STOPPED_EXIT

    host = str(args.rl_fleet_syncer_service)
    objects = objects if objects is not None else _store_objects(args)
    if objects is None:
        raise ValueError(f"--rl-fleet-syncer-service: the run store {getattr(args, 'rl_checkpoint_store', None)!r} "
                         "has no object API here; the syncer service records itself in it")
    mine = environ_key_id()
    refused = key_mismatch(mine, None)
    if refused:
        log(f"[launcher] ERROR: fleet syncer service on {host}: {refused}")
        return CONFIG_ERROR_EXIT
    stop_flag = runs.stop_flag_path(args.cluster_prefix)

    def stop_fleet() -> int:
        try:
            write_onsite_stop(objects, why=f"STOP flag of run {args.cluster_prefix}", by=_node())
            log(f"[launcher] fleet syncer: the fleet's STOP is in the run store ({ONSITE_STOP}); the syncer "
                f"service on {host} ends on it")
        except Exception as error:  # noqa: BLE001 - the service then runs on; said
            log(f"[launcher] ERROR: fleet syncer: the fleet's STOP was not written to the run store "
                f"({type(error).__name__}: {error}); stop the syncer unit on {host} (`systemctl stop`)")
        return RUN_STOPPED_EXIT

    started, said, beats = clock(), False, _Beats(clock)
    log(f"[launcher] fleet syncer: an on-site service on {host} (fleet.syncer_service); this launch starts no "
        "syncer, it waits for the service's record in the run store")
    while True:
        status, rec = read_service_status(objects), read_syncer_record(objects)
        ours = status is not None and status.get("host") == host
        beats.see(status if ours else None)
        if ours:
            refused = key_mismatch(mine, status.get("key_id"))
            if refused:
                log(f"[launcher] ERROR: fleet syncer service on {host}: {refused}")
                return CONFIG_ERROR_EXIT
            state = status.get("state")
            if state == SERVICE_FAILED:
                log(f"[launcher] ERROR: fleet syncer service on {host} failed: {status.get('reason')}")
                return 1
            if state == SERVICE_FINISHED:
                log(f"[launcher] fleet syncer service on {host}: its run is finished")
                return 0
            if state == SERVICE_STOPPED:
                log(f"[launcher] fleet syncer service on {host} stopped: {status.get('reason')}")
                return RUN_STOPPED_EXIT
            if (state == SERVICE_RUNNING and beats.moved and rec is not None and rec.get("service") == host
                    and status.get("generation") is not None
                    and rec.get("service_generation") == status.get("generation")):
                break
        if os.path.exists(stop_flag):
            log(f"[launcher] fleet syncer: STOP flag {stop_flag} before the syncer service on {host} was up")
            return stop_fleet()
        if clock() - started >= wait_s:
            log(f"[launcher] ERROR: fleet syncer: no running syncer service on {host} recorded in the run store "
                f"within {wait_s:.0f}s (start its unit on that host: `yeto fleet service`, docs/OWN_CLUSTERS.md); "
                "no island was started")
            return 1
        if not said:
            log(f"[launcher] fleet syncer: waiting up to {wait_s:.0f}s for the syncer service on {host} (its "
                "heartbeat must move once)")
            said = True
        sleep(poll_s)
    epoch = rec.get("syncer_epoch")
    args.rl_syncer_epoch = epoch
    # recorded again for this run up (wall clock: the trainer worker dials a record newer than its syncer launch)
    write_syncer_record(objects, addr=str(rec["addr"]), syncer_epoch=epoch, learners=int(rec.get("learners") or 0),
                        max_learners=rec.get("max_learners"), run=str(rec.get("run") or args.cluster_prefix),
                        extra={k: rec[k] for k in ("service", "service_node", "service_generation", "key_id")
                               if k in rec})
    log(f"[launcher] fleet syncer service on {host} at {rec['addr']} (epoch {epoch}, generation "
        f"{status.get('generation')}) recorded for the islands; watching its heartbeat")

    def current() -> dict[str, Any] | None:
        return read_service_status(objects)

    def finished() -> bool:
        status = current()
        return status is not None and status.get("host") == host and status.get("state") == SERVICE_FINISHED

    def service_stopped() -> str | None:
        status = current()
        if status is not None and status.get("host") == host and status.get("state") == SERVICE_STOPPED:
            return f"the syncer service on {host} stopped ({status.get('reason')})"
        return None

    def failed() -> str | None:
        try:
            status = current()
            rec = read_syncer_record(objects)
        except Exception as error:  # noqa: BLE001 - not known: down for now
            return f"its heartbeat did not read ({type(error).__name__}: {error})"
        if status is None:
            return "no heartbeat record"
        if status.get("host") != host:
            return f"the heartbeat names host {status.get('host')!r}, not {host}: no syncer service of this fleet"
        beats.see(status)
        refused = key_mismatch(mine, status.get("key_id"))
        if refused:
            return Final(refused)
        if status.get("state") == SERVICE_FAILED:
            return Final(f"the syncer service failed: {status.get('reason')}")
        if beats.silent_s() > SERVICE_STALE_S:
            return f"no new heartbeat from the syncer service on {host} for {beats.silent_s():.0f}s"
        if status.get("state") != SERVICE_RUNNING:
            return f"the syncer service on {host} is {status.get('state')}"
        if rec is None or rec.get("service_generation") != status.get("generation"):
            return (f"the syncer record (generation {None if rec is None else rec.get('service_generation')}) is not "
                    f"the running service's (generation {status.get('generation')}) yet")
        return None

    grace = getattr(args, "rl_fleet_syncer_grace_s", None)
    code = supervise(finished=finished, failed=failed, stop_flag=stop_flag, sleep=sleep,
                     grace_s=FAILURE_GRACE_S if grace is None else float(grace), clock=clock, log=log,
                     stopped=service_stopped)
    if code == RUN_STOPPED_EXIT and os.path.exists(stop_flag):
        return stop_fleet()
    return code


def _node() -> str:
    import socket

    return socket.gethostname()
