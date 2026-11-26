"""Node leases across runs: one run at a time on a node of an SSH inventory.

An inventory may name a ``lease_store`` at its top (:mod:`.node_inventory`): a store
every launcher of the site reaches (``s3://`` / ``oss://``, or a shared directory).
A run that starts something on a node of that inventory (an ``ssh:`` trainer island,
a ``kind: ssh`` sampler pool) first takes the node's lease there; a node another run
holds is skipped, and a run that finds no free node reports who holds them and waits.
Without ``lease_store`` nothing is leased and nothing here runs (as before).

Layout: a lease is a fleet store lock (:class:`~.pool.store_lock.StoreLock`) under
``<lease_store>/nodes/<node>/lease``: a token created once (S3 / R2 ``If-None-Match: *``,
a hard link on a directory) names the holder, and the token's lease file says whether it
still runs. One acquire lists the tokens once, judges the highest one and creates only
the next one (:meth:`StoreLock.claim`), so two runs claiming one node race for the same
create and one wins. A store whose create turns out not to be atomic is refused
(:class:`NodeLeaseError`).

Expiry: a lease runs for ``ttl_s`` (10 minutes) past its lease file's last write. On an
``s3://`` store both times are the store's (``LastModified`` and the listing's ``Date``,
``S3ObjectStore.list_meta``): the clocks of the hosts do not matter. A store without
those times (a directory, OSS) compares the holder's written end with the reader's clock:
the hosts must then run NTP (skew well below ``ttl_s``). The holder renews on its
supervision tick (the sampler pool's ticket reads, the trainer island's watch) and from a
keeper thread; a store that does not answer past the lease marks the node
``unconfirmed`` (an alert event), never free.

Takeover: a lease that ran out (its holder vanished) may be claimed by the next run, which
fences the old token for good and writes ``node_lease_taken_over`` under
``nodes/<node>/events/``. Before that the claimer asks the node (``probe``, over SSH:
:func:`still_runs_script`) whether anything of the previous run still runs there (its
trainer containers or process groups, its samplers by their start marker); if so, or if
the node does not answer, the takeover is refused (``node_lease_blocked``): a sampler whose
stop is still owed keeps its node out of other runs this way.

Holders: ``<run>/<part>@<identity>``: the run (its ``--cluster-prefix`` or pool run tag),
the part (``trainer`` / ``pool``) and the run's launch identity, a nonce made at the first
launch and kept in the run directory (the pool's: in its state, or in the run store on the
store control plane). Only the same identity takes a running lease back (a restarted
launcher); a second run of the same name elsewhere does not, and a node held by the same
run and part under another identity (a run dir restored without ``node-leases.json``) is
an error at once, not a wait. ``keep_all`` (a launcher that ends with ``--keep`` or with a
release not confirmed) leaves the leases held for good (``kept``: ``until`` null) until
``yeto cleanup --run <run>`` releases them; a node that stays unreachable keeps its lease
kept (docs/OWN_CLUSTERS.md, Node leases: the manual step).

Known follow-up: the nodes do not renew their own leases; a run whose launcher is gone
keeps its nodes only while ``kept`` or while the probe finds its processes.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pool.store_lock import DEFAULT_TTL_S as _LOCK_TTL_S, LockLost, StoreLock, StoreLockError, _TOKEN

DEFAULT_TTL_S = 600.0
WAIT_POLL_S = 30.0  # a run that waits for a free node reads the leases this often
WAIT_REPORT_S = 300.0  # ... and says so this often (and whenever the holders change)
EVENT_SCHEMA = "yeto.node-lease.event/v1"
TAKEOVER_EVENT = "node_lease_taken_over"
BLOCKED_EVENT = "node_lease_blocked"
LOST_EVENT = "node_lease_lost"
WAITING_EVENT = "node_lease_waiting"
UNCONFIRMED_EVENT = "node_lease_unconfirmed"
STORE_ERROR_EVENT = "node_lease_store_error"
PROBE_MARK = "yeto-lease-probe"
NO_LEASE_HINT = ("the node inventory declares no lease_store: its nodes are not leased across runs, so two runs "
                 "on one inventory may start on the same node (set lease_store: <store URI> at the top of the "
                 "inventory)")

_LOG = logging.getLogger(__name__)
_hinted = False


class NodeLeaseError(RuntimeError):
    """The leases cannot be used (a store without an atomic create, an unreadable store)."""


class NodeLeased(RuntimeError):
    """A node (or every free node) is held by another run, or its lease could not be
    confirmed; nothing was started on it.

    The sampler pool counts it as a refused order (``island_pool.REFUSED_ERRORS``: the
    slot is left out for a while, the next one is tried)."""

    def __init__(self, message: str, holders: Mapping[str, str | None] | None = None) -> None:
        super().__init__(message)
        self.holders = dict(holders or {})


@dataclass(frozen=True)
class LeaseResult:
    """One :meth:`NodeLeases.acquire`: ``ok`` (this run holds the node now), ``holder``
    (who holds it: this run when ``ok``), ``took_over`` (the claim ended another run's
    lease that had run out; ``previous`` names it), ``blocked`` (why such a takeover was
    refused: the previous run still runs there, or the node did not answer)."""

    ok: bool
    holder: str | None
    took_over: bool = False
    previous: str | None = None
    blocked: str | None = None


@dataclass(frozen=True)
class _State:
    top: int  # the highest token (0: none)
    holder: str | None
    running: bool
    released: bool = False
    kept: bool = False


def node_name(node: Any) -> str:
    return node if isinstance(node, str) else str(node.name)


def holder_id(run_id: str, identity: str | None) -> str:
    return f"{run_id}@{identity}" if identity else str(run_id)


def run_of(holder: str | None) -> str | None:
    """The run of a holder id (``<run>/<part>@<identity>`` -> ``<run>``)."""
    if not holder:
        return None
    base = str(holder).split("@", 1)[0]
    head, _, part = base.rpartition("/")
    return head if head and part in ("trainer", "pool") else base


def new_identity() -> str:
    return uuid.uuid4().hex[:16]


def local_identity(path: str | Path) -> str:
    """The launch identity kept in ``path`` (a JSON file; made at the first call)."""
    file = Path(path)
    try:
        doc = json.loads(file.read_text())
        if isinstance(doc, dict) and isinstance(doc.get("identity"), str) and doc["identity"]:
            return doc["identity"]
    except (OSError, ValueError):
        doc = {}
    identity = new_identity()
    doc = dict(doc) if isinstance(doc, dict) else {}
    doc["identity"] = identity
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    tmp.replace(file)
    return identity


def store_identity(objects: Any, key: str) -> str:
    """The launch identity kept in the store at ``key`` (created once: a controller that
    takes over on another host reads the same one)."""
    data = objects.read(key)
    if data is None:
        objects.create(key, json.dumps({"identity": new_identity()}).encode())
        data = objects.read(key)
    try:
        identity = json.loads(data or b"{}").get("identity")
    except ValueError:
        identity = None
    if not identity:
        raise NodeLeaseError(f"the lease identity record {key} is not readable")
    return str(identity)


def open_objects(uri: str, environ: Mapping[str, str] | None = None) -> Any:
    """The object API of a lease store: a directory (an absolute path or ``file://``),
    else the head's ``s3://`` / ``oss://`` client (the launcher's store credential)."""
    if uri.startswith("file://") or uri.startswith("/"):
        from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

        return DirectoryObjectStore(uri[len("file://"):] if uri.startswith("file://") else uri)
    from .trainer_island import head_store_objects  # noqa: PLC0415

    return head_store_objects(uri, environ)


def no_lease_hint(*, emit: Callable[[str], None] | None = None) -> str | None:
    """The one hint of an inventory without ``lease_store``, once per process: through
    ``emit`` (a dry-run's note, ``yeto cluster probe`` / ``check``), else a DEBUG log
    record only (the customer's trainer dry run, whose output stays as before). Returns
    the hint the first time, then None."""
    global _hinted
    if _hinted:
        return None
    _hinted = True
    if emit is None:
        _LOG.debug(NO_LEASE_HINT)
    else:
        emit(NO_LEASE_HINT)
    return NO_LEASE_HINT


def declares_new_fields(nodes: Any) -> bool:
    """The inventory uses a field of own-cluster-fleet-stability (``lease_store``, a node's
    ``fabric_group``): its trainer dry run then says the shared lease hint."""
    from .node_inventory import lease_store_of  # noqa: PLC0415

    return lease_store_of(nodes) is not None or any(getattr(n, "fabric_group", None) is not None for n in (nodes or ()))


# -- the node probe before a takeover ----------------------------------------------------

# what the takeover probe looks for, owned here (ssh_sampler_pool / ssh_trainer_island import them:
# node_lease imports neither)
SAMPLER_MARKER = "yeto-ssh-sampler"  # prefix of an ssh pool start's marker (run, shard, ticket)
TRAINER_STATE_DIR = ".yeto/trainer-islands"  # an ssh trainer island's state dirs, under the remote $HOME
RUN_LABEL = "yeto.run"  # a sampler container of a leased pool carries ``--label yeto.run=<run>``
RUN_LABEL_ENV = "YETO_RUN_LABEL"  # set on a leased pool's sampler: the node runtime labels its containers


def run_labels(run: str) -> dict[str, str]:
    """The labels of a sampler container whose node is leased (inventory ``lease_store``)."""
    return {RUN_LABEL: str(run)}


def still_runs_script(run: str, *, sudo: bool = False) -> str:
    """A shell script that prints ``yeto-lease-probe found=<what>`` for everything of
    run ``run`` that may still run on the node, then ``yeto-lease-probe done``:

    * ``container:<name>``: a container labelled ``yeto.run=<run>`` (a leased pool's
      sampler) or named ``yeto-<run>-...`` (a trainer incarnation);
    * ``unconfirmed:container:<name>``: a sampler container (``yeto-sampler-...``) with no
      ``yeto.run`` label (started before labels, or without lease_store): perhaps this
      run's, not confirmed;
    * ``unconfirmed:<view>``: an engine's containers could not be listed. Each engine is
      listed as the user and, when ``sudo -n`` works, through it. docker has one daemon:
      either view listing is enough (``unconfirmed:docker`` when neither does). podman's
      rootless and rootful containers are two sets: the user's view must list
      (``unconfirmed:podman``), and with ``sudo`` (a pool with ``container.sudo``) the
      rootful one too (``unconfirmed:sudo-podman``). A daemon that is not running ("Cannot
      connect to the Docker daemon", not "permission denied") lists no containers;
    * ``process:<island>``: a trainer process group (``~/.yeto/trainer-islands/<run>/*/pgid``
      without an ``rc``);
    * ``sampler:<pid>``: a venv sampler (the start marker ``yeto-ssh-sampler:<run>:`` in its
      arguments; assembled at run time, so the probe never finds itself).

    Every finding refuses a takeover; the caller treats a missing ``done`` line the same."""
    if not re.fullmatch(r"[A-Za-z0-9_.:+-]+", str(run)):
        raise NodeLeaseError(f"run {run!r}: not a plain word")
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "-", f"yeto-{run}-")
    head, tail = SAMPLER_MARKER[:5], SAMPLER_MARKER[5:]
    found = f"echo \"{PROBE_MARK} found="
    names = "--format '{{.Names}}'"
    return (
        f"m={head}; m=\"${{m}}{tail}:{run}:\"\n"
        "err=$(mktemp 2>/dev/null || echo \"/tmp/yeto-lease-probe.$$\")\n"
        "listed() {  # after a failed ps: a daemon that is not running lists nothing; any other failure: unknown\n"
        "  grep -qi 'permission denied' \"$err\" 2>/dev/null && return 1\n"
        "  grep -q 'Cannot connect to the Docker daemon' \"$err\" 2>/dev/null || return 1\n"
        "  mine=; labelled=; all=; ok=1\n"
        "}\n"
        "view() {  # the engine command: ok=1 and its listings, or ok=0\n"
        "  ok=0; mine=; labelled=; all=\n"
        f"  mine=$(\"$@\" ps --filter label={RUN_LABEL}={run} {names} 2>\"$err\") || {{ listed; return; }}\n"
        f"  labelled=$(\"$@\" ps --filter label={RUN_LABEL} {names} 2>\"$err\") || {{ listed; return; }}\n"
        f"  all=$(\"$@\" ps {names} 2>\"$err\") || {{ listed; return; }}\n"
        "  ok=1\n"
        "}\n"
        "report() {\n"
        f"  for n in $mine; do {found}container:$n\"; done\n"
        "  for n in $all; do\n"
        f"    case \"$n\" in {prefix}*) {found}container:$n\";;\n"
        "      yeto-sampler-*) case \" $(echo $labelled) \" in *\" $n \"*) ;; "
        f"*) {found}unconfirmed:container:$n\";; esac;;\n"
        "    esac\n  done\n}\n"
        "sudo_ok=0; command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && sudo_ok=1\n"
        "for e in docker podman; do command -v $e >/dev/null 2>&1 || continue\n"
        "  view $e; plain=$ok; [ $ok = 1 ] && report\n"
        "  root=0; if [ $sudo_ok = 1 ]; then view sudo -n $e; root=$ok; [ $ok = 1 ] && report; fi\n"
        "  case $e in\n"
        f"    docker) [ $plain = 1 ] || [ $root = 1 ] || {found}unconfirmed:docker\";;\n"
        f"    *) [ $plain = 1 ] || {found}unconfirmed:$e\"\n"
        + (f"       [ $root = 1 ] || {found}unconfirmed:sudo-$e\";;\n" if sudo else "       ;;\n")
        + "  esac\n"
        "done\n"
        "rm -f \"$err\"\n"
        f"for f in \"$HOME/{TRAINER_STATE_DIR}/{run}\"/*/pgid; do [ -f \"$f\" ] || continue; d=${{f%/pgid}}\n"
        "  [ -f \"$d/rc\" ] && continue; pid=$(cat \"$f\" 2>/dev/null)\n"
        f"  [ \"$pid\" -gt 1 ] 2>/dev/null && kill -0 -- -$pid 2>/dev/null && {found}process:${{d##*/}}\"\n"
        "done\n"
        f"ps -eo pid=,args= 2>/dev/null | M=\"$m\" awk -v me=$$ '$1 != me && index($0, ENVIRON[\"M\"]) "
        f"{{print \"{PROBE_MARK} found=sampler:\" $1}}'\n"
        f"echo \"{PROBE_MARK} done\"\n"
    )


def parse_probe(text: str) -> tuple[bool, list[str]]:
    """``(confirmed, found)`` of a :func:`still_runs_script` output (confirmed: it ran to
    its end)."""
    found, done = [], False
    for line in str(text or "").splitlines():
        if line.startswith(PROBE_MARK + " found="):
            found.append(line.split("=", 1)[1].strip())
        elif line.strip() == f"{PROBE_MARK} done":
            done = True
    return done, sorted(set(found))


def ssh_probe(run: Callable[[Any, str], Any], *, sudo: bool = False) -> Callable[[Any, str], str | None]:
    """A takeover probe over ``run(node, script) -> CompletedProcess`` (the caller's SSH):
    None when nothing of the previous holder's run runs on the node, else why the node is
    not free (it still runs that run, or it did not answer)."""

    def probe(node: Any, previous: str) -> str | None:
        name, previous_run = node_name(node), run_of(previous) or str(previous)
        try:
            done = run(node, still_runs_script(previous_run, sudo=sudo))
        except Exception as error:  # noqa: BLE001 - not confirmed: not free
            return f"node {name}: not confirmed free of run {previous_run} ({type(error).__name__})"
        confirmed, found = parse_probe(getattr(done, "stdout", "") or "")
        sure = [f for f in found if not f.startswith("unconfirmed:")]
        if sure:
            return f"node {name} still runs run {previous_run} ({', '.join(found)})"
        if found:
            return f"node {name}: not confirmed free of run {previous_run} ({', '.join(found)})"
        if not confirmed or getattr(done, "returncode", 0) not in (0, None):
            return (f"node {name}: not confirmed free of run {previous_run} "
                    f"(probe rc {getattr(done, 'returncode', None)})")
        return None

    return probe


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class NodeLeases:
    """The node leases of run ``run_id`` in the lease store ``store_uri`` (module doc).

    ``identity``: the run's launch identity (the holder is ``<run_id>@<identity>``).
    ``objects``: the store's object API (tests pass a fake; default :func:`open_objects`,
    opened on first use). ``on_event(name, **detail)``: where takeovers, refusals, losses
    and waits are reported besides the store (default: a line on stderr). ``probe(node,
    previous_holder) -> str | None``: asked before an expired lease of another run is taken
    over (:func:`ssh_probe`); None: no probe (the takeover goes ahead)."""

    def __init__(self, store_uri: str, run_id: str, ttl_s: float = DEFAULT_TTL_S, *, identity: str | None = None,
                 objects: Any = None, clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 on_event: Callable[..., None] | None = None,
                 probe: Callable[[Any, str], str | None] | None = None,
                 identity_path: str | None = None) -> None:
        if not run_id or not str(run_id).strip():
            raise NodeLeaseError("node leases need a run id (the holder)")
        if float(ttl_s) <= 0:
            raise NodeLeaseError(f"node lease ttl_s must be positive, got {ttl_s!r}")
        self.store_uri, self.run_id, self.ttl_s = str(store_uri), str(run_id), float(ttl_s)
        self.identity = identity
        self.holder_id = holder_id(self.run_id, identity)
        self.clock, self.monotonic = clock, monotonic
        self.on_event = on_event
        self.probe = probe
        self.identity_path = identity_path  # where the identity is kept (named in a same-run conflict)
        self._objects = objects
        self._locks: dict[str, StoreLock] = {}
        self._renewed: dict[str, float] = {}  # node -> monotonic time of its last renewal (the cadence)
        self._lost: dict[str, str | None] = {}  # node -> the holder that has it now
        self._unconfirmed: dict[str, float] = {}  # node -> since when its lease is not confirmed
        self.assigned: set[str] = set()  # nodes given to an island of this launch
        self.picks: dict[str, list[str]] = {}  # island -> its nodes (one pick per island and process)
        self._mutex = threading.RLock()
        self._keeper: threading.Thread | None = None
        self._stop = threading.Event()

    # -- store
    @property
    def objects(self) -> Any:
        if self._objects is None:
            self._objects = open_objects(self.store_uri)
        return self._objects

    @staticmethod
    def prefix(node: Any) -> str:
        return f"nodes/{node_name(node)}/lease"

    @staticmethod
    def event_key(node: Any, token: int, kind: str = "takeover") -> str:
        return f"nodes/{node_name(node)}/events/{int(token):08d}-{kind}.json"

    @property
    def renew_every_s(self) -> float:
        """A held lease is renewed once this much of it is gone (a quarter: three renewals
        may fail before it runs out)."""
        return self.ttl_s / 4

    def _doc(self, key: str) -> dict[str, Any] | None:
        data = self.objects.read(key)
        if data is None:
            return None
        try:
            doc = json.loads(data)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    def _listing(self, prefix: str) -> tuple[list[int], dict[str, float], float | None]:
        """One LIST of a node's lease: the tokens, the objects' store times and the store's
        own time (``list_meta``: S3; the last two empty / None without it)."""
        base = prefix.rstrip("/") + "/"
        meta = getattr(self.objects, "list_meta", None)
        rows = None
        if callable(meta):
            try:
                rows, now = meta(base, require_date=True)
            except (AttributeError, KeyError, NotImplementedError, TypeError):  # no store times: the clocks
                rows = None
        if rows is None:
            keys, times, now = self.objects.list(base), {}, None
        else:
            keys, times = [k for k, _ in rows], {k[len(base):]: float(t) for k, t in rows}
        tokens = sorted(int(m.group(1)) for k in keys if (m := _TOKEN.fullmatch(k[len(base):])) is not None)
        return tokens, times, (float(now) if rows is not None and now else None)

    def _state(self, name: str) -> _State:
        """The lease of ``name`` from one LIST and two reads (module doc: expiry)."""
        prefix = self.prefix(name)
        tokens, times, store_now = self._listing(prefix)
        if not tokens:
            return _State(0, None, False)
        top = tokens[-1]
        lease = self._doc(StoreLock.lease_key(prefix, top))
        record = self._doc(StoreLock.record_key(prefix, top)) or {}
        holder = str((lease or {}).get("holder") or record.get("holder") or "") or None
        if lease is not None and lease.get("released"):
            return _State(top, holder, False, released=True)
        if lease is not None and lease.get("kept"):
            return _State(top, holder, True, kept=True)
        ttl = float(record.get("ttl_s") or _LOCK_TTL_S)
        written = times.get(f"{int(top):08d}.lease" if lease is not None else f"{int(top):08d}.json")
        if store_now is not None and written is not None:  # the store's clock on both sides
            return _State(top, holder, store_now < written + ttl)
        until = (float(lease.get("until") or 0.0) if lease is not None
                 else float(record.get("claimed_at") or 0.0) + ttl)
        return _State(top, holder, until > self.clock())

    def _event(self, event: str, **detail: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, **detail)
                return
            except Exception:  # noqa: BLE001 - a report never fails the lease
                pass
        _stderr(f"[node-lease] {event}: " + " ".join(f"{k}={v}" for k, v in detail.items()))

    def _record(self, name: str, token: int, kind: str, doc: dict[str, Any]) -> bool:
        try:
            self.objects.write(self.event_key(name, token, kind), json.dumps(
                {"schema": EVENT_SCHEMA, "node": name, "at": self.clock(), **doc}, sort_keys=True).encode())
            return True
        except Exception:  # noqa: BLE001 - the store record is best effort
            return False

    # -- reading
    def holder(self, node: Any) -> str | None:
        """Who holds ``node`` now (None: free: no lease, released, or run out)."""
        state = self._state(node_name(node))
        return state.holder if state.running else None

    def held(self) -> list[str]:
        """The nodes this object holds the lease of."""
        with self._mutex:
            return sorted(self._locks)

    def lost(self) -> dict[str, str | None]:
        """The nodes whose lease this run lost (another run took them after the lease ran
        out), with the holder of each now."""
        with self._mutex:
            return dict(self._lost)

    def unconfirmed(self) -> dict[str, float]:
        """The held nodes whose lease the store did not confirm past its end (since when)."""
        with self._mutex:
            return dict(self._unconfirmed)

    # -- claim / renew / release
    def acquire(self, node: Any) -> LeaseResult:
        """Take ``node``'s lease for this run (held already: renewed). Refused (``ok``
        False) while another holder's lease runs (or is kept); an expired lease of another
        run is taken over (``took_over``, a takeover event) unless the probe finds that run
        still on the node (``blocked``, a blocked event)."""
        name = node_name(node)
        with self._mutex:
            lock = self._locks.get(name)
            if lock is not None and self._renew_lock(name, lock):
                return LeaseResult(True, self.holder_id)
            prefix = self.prefix(name)
            state = self._state(name)  # the one LIST of this claim
            if state.running and state.holder != self.holder_id:
                return LeaseResult(False, state.holder)
            previous = (state.holder if state.top and not state.running
                        and state.holder not in (None, self.holder_id) else None)
            if previous is not None and self.probe is not None:  # a released lease too: its run may linger
                why = self.probe(node, previous)
                if why:
                    self._record(name, state.top + 1, "blocked", {"event": BLOCKED_EVENT, "holder": self.holder_id,
                                                                  "previous": previous, "why": why})
                    self._event(BLOCKED_EVENT, node=name, holder=self.holder_id, previous=previous, why=why)
                    return LeaseResult(False, previous, blocked=why)
            try:
                lock = StoreLock.claim(self.objects, prefix, state.top + 1, holder=self.holder_id, ttl_s=self.ttl_s,
                                       clock=self.clock)
            except StoreLockError as error:
                raise NodeLeaseError(f"lease store {self.store_uri}: {error}") from None
            except LockLost:  # a higher token appeared right after our create: another claim won
                lock = None
            if lock is None:
                return LeaseResult(False, self.holder(name))
            self._locks[name] = lock
            self._renewed[name] = self.monotonic()
            self._lost.pop(name, None)
            self._unconfirmed.pop(name, None)
            took = previous is not None and not state.released
            if took:
                recorded = self._record(name, lock.token, "takeover", {
                    "event": TAKEOVER_EVENT, "holder": self.holder_id, "previous": previous, "token": lock.token})
                self._event(TAKEOVER_EVENT, node=name, holder=self.holder_id, previous=previous,
                            token=int(lock.token), recorded=recorded)
            return LeaseResult(True, self.holder_id, took, previous if took else None)

    def same_run_conflict(self, holder: str | None) -> str | None:
        """Why ``holder`` (another holder of a node) is a problem of this very run: the same
        run and part under another launch identity (its run dir moved, or was restored
        without the identity file). None: another run."""
        if not holder or holder == self.holder_id:
            return None
        if str(holder).split("@", 1)[0] != self.run_id:
            return None
        where = self.identity_path or "the run's node lease identity file"
        run = run_of(self.holder_id)
        return (f"held by {holder}: this run ({run}) under another launch identity (this launch is "
                f"{self.holder_id}; the identity is kept in {where}). Restore that file from the earlier "
                f"launch, or end the earlier launch's leases with `yeto cleanup --run {run}`")

    def _renew_lock(self, name: str, lock: StoreLock) -> bool:
        """Renew one held lease; False once another holder has the node (recorded in
        :meth:`lost`) or the store did not confirm it past its end (:meth:`unconfirmed`,
        an alert event once)."""
        try:
            lock.renew()
        except LockLost:
            self._locks.pop(name, None)
            try:
                now = self.holder(name)
            except Exception:  # noqa: BLE001 - who took it is information only
                now = None
            if now == self.holder_id:  # a restarted launcher of this run took it back: still this run's
                return True
            self._lost[name] = now
            self._event(LOST_EVENT, node=name, holder=self.holder_id, now_held_by=now)
            return False
        except Exception as error:  # noqa: BLE001 - not renewed this time; the lease may still run
            if lock.lease_left() > 0:
                return True
            if name not in self._unconfirmed:
                self._unconfirmed[name] = self.clock()
                self._event(UNCONFIRMED_EVENT, level="alert", node=name, holder=self.holder_id,
                            error=repr(error)[:200])
            return False
        self._renewed[name] = self.monotonic()
        if self._unconfirmed.pop(name, None) is not None:
            self._event("node_lease_confirmed", node=name, holder=self.holder_id)
        return True

    def renew(self, node: Any) -> bool:
        """Move ``node``'s lease end forward (one not held: taken, e.g. by a restarted
        launcher of this run). False when another run holds the node."""
        name = node_name(node)
        with self._mutex:
            lock = self._locks.get(name)
            if lock is None:
                return self.acquire(node).ok
            return self._renew_lock(name, lock)

    def renew_due(self, *, force: bool = False) -> list[str]:
        """Renew every held lease past :attr:`renew_every_s` (``force``: all), and every
        unconfirmed one; returns the nodes not renewed (lost or unconfirmed)."""
        out = []
        with self._mutex:
            now = self.monotonic()  # the cadence: never moved by a wall clock step
            for name, lock in sorted(self._locks.items()):
                last = self._renewed.get(name)
                if force or name in self._unconfirmed or last is None or now - last >= self.renew_every_s:
                    if not self._renew_lock(name, lock):
                        out.append(name)
        return out

    def release(self, node: Any) -> None:
        """Give ``node``'s lease up now (the next run does not wait for it to run out).
        Never raises."""
        name = node_name(node)
        with self._mutex:
            lock = self._locks.pop(name, None)
            self._unconfirmed.pop(name, None)
        if lock is not None:
            lock.release()

    def release_all(self, nodes: Iterable[Any] | None = None) -> None:
        for name in (self.held() if nodes is None else [node_name(n) for n in nodes]):
            self.release(name)

    def keep_all(self, nodes: Iterable[Any] | None = None) -> list[str]:
        """Hold every lease of this object for good (``kept``, ``until`` null: it never runs
        out): the launcher ends while the nodes may still run (``--keep``, a release not
        confirmed). ``yeto cleanup --run <run>`` (:meth:`force_release`) or a restarted
        launcher of this run ends it; a node that stays unreachable: docs/OWN_CLUSTERS.md,
        Node leases. ``nodes``: only those (one island of several; the keeper goes on
        renewing the others); None: all, and the keeper stops. A kept lease is no longer
        renewed (a renewal would end ``kept``). Returns the nodes kept; never raises."""
        only = None if nodes is None else {node_name(n) for n in nodes}
        if only is None:
            self.stop_keeper()
        kept = []
        with self._mutex:
            for name, lock in sorted(self._locks.items()):
                if only is not None and name not in only:
                    continue
                self._locks.pop(name, None)
                self._unconfirmed.pop(name, None)
                try:
                    lock.check(force=True)
                    self.objects.write(StoreLock.lease_key(lock.prefix, lock.token), json.dumps(
                        {"token": lock.token, "holder": lock.holder, "until": None, "released": False,
                         "kept": True, "kept_at": self.clock()}).encode())
                    kept.append(name)
                except BaseException as error:  # noqa: BLE001 - lost already, or the store: the lease runs out
                    _LOG.warning("node lease of %s not kept: %r", name, error)
        if kept:
            self._event("node_lease_kept", nodes=",".join(kept), holder=self.holder_id,
                        release=f"yeto cleanup --run {run_of(self.holder_id)} (after the nodes stopped); "
                                "an unreachable node: docs/OWN_CLUSTERS.md, Node leases")
        return kept

    def force_release(self, node: Any, *, holder: str | None = None, force: bool = False) -> bool:
        """End ``node``'s current lease when it is kept or ran out (``force``: a live one
        too; ``holder``: only when that one holds it): ``yeto cleanup --run`` of a run whose
        launcher kept its leases, ``yeto cluster release-node`` (with its own probe first).
        The next claim still probes the node. True when a lease was ended."""
        name = node_name(node)
        state = self._state(name)
        if not state.top or state.released or (holder is not None and state.holder != holder):
            return False
        if state.running and not state.kept and not force:
            return False
        self.objects.write(StoreLock.lease_key(self.prefix(name), state.top), json.dumps(
            {"token": state.top, "holder": state.holder, "until": 0.0, "released": True,
             "released_by": self.holder_id}).encode())
        with self._mutex:
            self._locks.pop(name, None)
        self._event("node_lease_released", node=name, holder=state.holder, by=self.holder_id)
        return True

    # -- the keeper (a holder with no tick of its own while it starts: preflight, weights)
    def start_keeper(self, interval_s: float | None = None) -> None:
        """Renew the held leases from a daemon thread every ``interval_s`` (default
        :attr:`renew_every_s`) until :meth:`stop_keeper`; idempotent."""
        if self._keeper is not None and self._keeper.is_alive():
            return
        every = float(interval_s or self.renew_every_s)
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(every):
                try:
                    self.renew_due()
                except Exception as error:  # noqa: BLE001 - the next round tries again
                    _LOG.warning("node lease renewal failed: %r", error)

        self._keeper = threading.Thread(target=loop, name="yeto-node-lease-keeper", daemon=True)
        self._keeper.start()

    def stop_keeper(self) -> None:
        self._stop.set()


class _NoPlacement(RuntimeError):
    """The default placement found fewer than ``need`` usable candidates."""


def in_order_placement(candidates: Sequence[Any], need: int, *, excluded: Iterable[str] = ()) -> list[Any]:
    """The placement of an inventory without groups: the first ``need`` candidates not in
    ``excluded``, in the order given (as on main). Raises when there are fewer."""
    out = [n for n in candidates if node_name(n) not in set(excluded)]
    if len(out) < int(need):
        raise _NoPlacement(f"{len(out)} usable node(s), {need} needed")
    return out[:int(need)]


def _placement_failed(error: BaseException) -> bool:
    """A placement that found no fit: the default's, or a ``PlacementError`` of a group
    placement (``fabric_groups.pick_placement``; told by class name, not imported)."""
    return isinstance(error, _NoPlacement) or any(c.__name__ == "PlacementError" for c in type(error).__mro__)


def pick_nodes(candidates: Sequence[Any], need: int, leases: NodeLeases, *, wait_s: float | None = None,
               poll_s: float = WAIT_POLL_S, report_s: float = WAIT_REPORT_S, prefer: Sequence[str] = (),
               excluded: Iterable[str] = (), place: Callable[..., Any] | None = None,
               stop: Callable[[], str | None] | None = None,
               sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
               log: Callable[[str], None] = _stderr) -> list[Any]:
    """``need`` of ``candidates`` whose leases this run takes, in ``candidates`` order.

    Each round: leave out ``excluded`` (the caller's: unhealthy, quarantined, given to
    another island of this launch) and the nodes another run's lease holds; ``place(usable,
    need, excluded=...)`` then chooses the nodes (a group placement such as
    ``fabric_groups.pick_placement``, whose result may carry ``.nodes``; default
    :func:`in_order_placement`: one group, as on main); the chosen nodes' leases are
    taken one by one, and a node whose claim fails (taken since the read, or a takeover
    the probe refused) is left out and the placement computed again. The ``prefer`` nodes
    (an island's recorded ones) and the nodes this run holds come first in the order
    given to ``place``, then free ones. No placement fits: the holders are reported
    (whenever they change, and every ``report_s`` with the time waited; a
    ``node_lease_waiting`` event each time) and the leases read every ``poll_s`` until
    one fits; :class:`NodeLeased` after ``wait_s`` (None: no limit) or once ``stop()``
    names a reason."""
    need = int(need)
    if need <= 0:
        return []
    if len(candidates) < need:
        raise NodeLeased(f"{need} node(s) needed, the inventory offers {len(candidates)}")
    place = place or in_order_placement
    left_out = {str(n) for n in excluded}
    order = {node_name(n): i for i, n in enumerate(candidates)}
    first = [n for n in candidates if node_name(n) in set(prefer)]
    ranked = first + [n for n in candidates if n not in first]
    began, reported, reported_at = monotonic(), None, None
    while True:
        holders = {node_name(n): leases.holder(n) for n in ranked}
        for name, h in sorted(holders.items()):  # this run under another identity: waiting never ends
            conflict = None if name in left_out else leases.same_run_conflict(h)
            if conflict:
                raise NodeLeased(f"node {name} is {conflict}", {name: h})
        ours = [n for n in ranked if holders[node_name(n)] == leases.holder_id]
        free = [n for n in ranked if holders[node_name(n)] is None]
        usable = ([n for n in first if n in ours or n in free] + [n for n in ours if n not in first]
                  + [n for n in free if n not in first])
        refusals: dict[str, str | None] = {}
        refused: set[str] = set()  # this round's failed claims: left out, the placement computed again
        for _attempt in range(len(usable) + 1):  # each attempt refuses one node more: then wait
            out = left_out | refused
            offered = [n for n in usable if node_name(n) not in out]
            try:
                chosen = place(offered, need, excluded=sorted(out))
            except Exception as error:  # noqa: BLE001 - only "no fit" waits; anything else is the caller's
                if not _placement_failed(error):
                    raise
                break
            chosen = list(getattr(chosen, "nodes", chosen))
            names = [node_name(n) for n in chosen]
            bad = sorted(set(names) - {node_name(n) for n in offered})
            if len(chosen) != need or len(set(names)) != need or bad:
                raise ValueError(f"placement {getattr(place, '__name__', place)!r} returned {names} for {need} "
                                 f"node(s) of {[node_name(n) for n in offered]}"
                                 + (f": {bad} not offered (excluded, held or unknown)" if bad else ""))
            taken: list[Any] = []
            for node in chosen:
                result = leases.acquire(node)
                if not result.ok:
                    holders[node_name(node)] = result.holder or "(another run)"
                    if result.blocked:
                        refusals[node_name(node)] = result.blocked
                    refused.add(node_name(node))
                    break
                taken.append(node)
            if len(taken) == need:
                return sorted(taken, key=lambda n: order[node_name(n)])
            for node in taken:  # give back what this placement took; place again without the refused node
                if node not in ours:
                    leases.release(node)
        free_now = sum(1 for n, h in holders.items() if h in (None, leases.holder_id) and n not in left_out)
        busy = {name: h for name, h in holders.items() if h not in (None, leases.holder_id)}
        text = ", ".join(f"{name} by {refusals.get(name) or 'run ' + str(h)}" for name, h in sorted(busy.items()))
        waited = monotonic() - began
        why = stop() if stop is not None else None
        if why:
            raise NodeLeased(f"{need} node(s) needed, {free_now} free; held: {text}; stopped waiting: {why}", busy)
        if wait_s is not None and waited >= wait_s:
            raise NodeLeased(f"{need} node(s) needed, {free_now} free after {waited:.0f}s; held: {text}", busy)
        if busy != reported or reported_at is None or waited - reported_at >= report_s:
            log(f"[node-lease] {need} node(s) needed, {free_now} free; held: {text}; waiting "
                f"{waited:.0f}s so far (the leases are read every {poll_s:.0f}s)")
            leases._event(WAITING_EVENT, need=need, free=free_now, waited_s=round(waited),  # noqa: SLF001
                          held=",".join(f"{n}={h}" for n, h in sorted(busy.items())))
            reported, reported_at = busy, waited
        sleep(poll_s)


def for_inventory(nodes: Any, run_id: str, **kwargs: Any) -> NodeLeases | None:
    """The :class:`NodeLeases` of an inventory's node list (None: no ``lease_store``)."""
    from .node_inventory import lease_store_of  # noqa: PLC0415

    store = lease_store_of(nodes)
    return None if store is None else NodeLeases(store, run_id, **kwargs)
