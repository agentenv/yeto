"""``yeto run up``'s state in the run store, for an on-site control plane
(``fleet.syncer_service``, ``docs/OWN_CLUSTERS.md``): a run up on another host continues
the run from it (:meth:`yeto.rl.up.Run.start_state_mirror`).

Only the fields of :data:`KEYS` are kept there (the spend's clocks and periods, the steps
done, the breaker and stop records); :data:`EXCLUDED` names the run up state fields that
never are (this host's processes). A removed pool's row keeps only its billing fields
(:data:`RETIRED_POOL_FIELDS`): never its frozen pool config, which may carry a provider's
environment. No key is ever in it: the island HMAC key stays in the environment file.

One writer at a time, by compare-and-swap alone. Every write carries its writer (host and a
nonce of this run up) and a sequence number one past the mirror's, and replaces the mirror
only while it still holds the bytes this writer read (``replace_if``: S3 ``If-Match``, a
directory store under its lock; the first write an exclusive create). A write that loses
reads the mirror again: another writer's number past the one this writer last wrote (or,
at its start, read) fences it (:attr:`StateMirror.fenced`): another control host drives the
run now (run up ends with :data:`CONTROL_FENCED_EXIT`); otherwise it tries again with the
next number, at most :data:`WRITE_TRIES` times in one save. A run up that starts takes the
number after the mirror's, so the newest control host wins. A store without conditional
writes (an endpoint that ignores ``If-Match``, the OSS relay) cannot hold one writer at a
time: an on-site control plane refuses it (:func:`conditional_write_problem`).

A writer that has not had a write confirmed for ``stale_after_s`` (the island records
lock's lease plus the takeover margin: by then another host may have taken the islands
over) while it tried is fenced too (:meth:`StateMirror.stale`): a control host cut off from
the store stops driving the run rather than driving it next to its successor.

A run up's first write (:meth:`StateMirror.start`) is made in its own thread and must be
confirmed: a first write that does not land, or that loses to another writer, refuses the
start (run up exits 2 and its unit tries again, reading the mirror again) and fences no one.

A control host is told by its machine id (:func:`host_identity`, a digest of
``/etc/machine-id``) when it has one, else by its host name: two hosts with the same host
name and no machine id would be taken for one.

The writes run on a background thread (:class:`StateMirror`): a store that does not answer
never holds the monitor. A forced write (a stop, a breaker, the end of the run) waits for
it at most :data:`WRITE_TIMEOUT_S`. A save throttled away is kept and written by
:meth:`StateMirror.flush` (run up's exit)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

STATE_MIRROR = "fleet/control/up-state.json"
SCHEMA = "yeto.fleet.up-state/v1"
EVERY_S = 30.0  # a save mirrors at most this often (a forced one always)
WRITE_TIMEOUT_S = 20.0  # a forced write waits this long for the store, then the monitor goes on
LOCAL_FILE = "state-mirror.json"  # in the run's state dir: the last sequence number this host wrote
WRITE_TRIES = 3  # conditional writes of the mirror lost to another write in one save before it gives up
# a writer with no confirmed write this long (while it tried) is fenced: the island records lock's
# lease plus the takeover margin (run up passes fleet.takeover_margin_s; default 60 s)
RECORDS_LOCK_TTL_S = 120.0  # yeto.launcher.ISLAND_RECORDS_LOCK_TTL_S
STALE_AFTER_S = RECORDS_LOCK_TTL_S + 60.0  # yeto.rl.fleet.DEFAULT_TAKEOVER_MARGIN_S
# run up's exit when another control host drives the run (yeto.rl.fleet_service CONTROL_PREVENT)
CONTROL_FENCED_EXIT = 76
MACHINE_ID_PATH = "/etc/machine-id"  # systemd's id of this machine (host_identity)
_MACHINE_ID = re.compile(r"[0-9a-f]{32}")  # systemd's format; "uninitialized" and the like are no id
CONDITIONAL_WRITES_NEEDED = "on-site control needs a store with conditional writes (If-Match)"

KEYS = frozenset({
    "run_t0", "t0", "trainer_t0", "ups", "steps", "scales", "breaker", "starvation", "release_stall",
    "fleet_stop", "fleet_stops", "fleet_island_stops", "fleet_starvation", "fleet_release_stall",
    "fleet_first_group", "fleet_data_ended", "fleet_pools_ended", "fleet_data_end_incomplete", "data_ended",
    "data_end_incomplete", "trainer_finished", "trainer_exit_seen", "retired_pools", "degraded_pools",
    "spend_usd_estimate", "finished_banner", "sandbox_over"})
# never mirrored: this host's processes (PIDs mean nothing on another host), the trainer launcher's
# refusal on this host, and the restore's own note
EXCLUDED = frozenset({"procs", "trainer_hands_off", "restored_from"})
RETIRED_POOL_FIELDS = ("name", "config", "usd_per_h", "shard_base", "removed_at")


def mirrored_state(state: dict[str, Any]) -> dict[str, Any]:
    """What the run store keeps of ``state`` (module doc), a copy."""
    out = {k: copy.deepcopy(state[k]) for k in KEYS if k in state}
    rows = out.get("retired_pools")
    if isinstance(rows, list):
        out["retired_pools"] = [{f: row.get(f) for f in RETIRED_POOL_FIELDS if f in row} if isinstance(row, dict)
                                else row for row in rows]
    return out


def _read_json(objects: Any, key: str) -> dict[str, Any] | None:
    raw = objects.read(key)
    if raw is None:
        return None
    doc = json.loads(raw)
    return doc if isinstance(doc, dict) else None


def read_mirror(objects: Any) -> dict[str, Any] | None:
    """The mirror (None: none, or not one of this schema). A store error or a file that is
    not JSON raises."""
    doc = _read_json(objects, STATE_MIRROR)
    return doc if doc is not None and doc.get("schema") == SCHEMA else None


def seq_of(doc: dict[str, Any] | None) -> int:
    try:
        return max(0, int((doc or {}).get("seq") or 0))
    except (TypeError, ValueError):
        return 0


def read_local(path: Path) -> dict[str, Any]:
    """What this host last wrote (``seq``, ``nonce``, ``host``); {} when it never did."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def host_identity(path: str | Path | None = None) -> str | None:
    """This control host's identity: a digest of its machine id (``/etc/machine-id``; never
    the id itself), or None when it has none that reads (then its host name tells it)."""
    try:
        raw = Path(path or MACHINE_ID_PATH).read_text().strip().lower()
    except (OSError, UnicodeDecodeError):
        return None
    if not _MACHINE_ID.fullmatch(raw):  # empty, "uninitialized", not systemd's 32 hex digits: no id
        return None
    return "machine-id:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def same_host(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """``a`` and ``b`` (each ``{"host", "host_id"}``) name the same control host: the same host
    name, and the same machine id too when both have one."""
    a, b = a or {}, b or {}
    same_name = bool(a.get("host")) and a.get("host") == b.get("host")
    if a.get("host_id") and b.get("host_id"):
        return same_name and a["host_id"] == b["host_id"]
    return same_name


def conditional_write_problem(objects: Any) -> str | None:
    """Why ``objects`` cannot hold one mirror writer at a time (None: it can): its create is
    not atomic (``atomic_create``), it has no ``replace_if``, or its endpoint does not apply
    ``If-Match`` (``replace_if_usable``, probed once: S3). The message names the need."""
    from yeto.rl.engine.object_store import atomic_create  # noqa: PLC0415

    if not atomic_create(objects):
        return f"{CONDITIONAL_WRITES_NEEDED}: this store's create is not atomic"
    if not callable(getattr(objects, "replace_if", None)):
        return f"{CONDITIONAL_WRITES_NEEDED}: this store has no conditional write"
    usable = getattr(objects, "replace_if_usable", None)
    if callable(usable):
        try:
            ok = usable()
        except Exception as error:  # noqa: BLE001 - not known: refused, tried again by the unit
            return f"{CONDITIONAL_WRITES_NEEDED}: the endpoint's conditional write was not probed ({error})"
        if not ok:
            return f"{CONDITIONAL_WRITES_NEEDED}: this endpoint does not apply If-Match on PutObject"
    return None


class StateMirror:
    """The run store's copy of one run up's state (module doc). ``submit`` is called at each
    save; ``fenced`` says why this run up may write no more (None: it may)."""

    def __init__(self, objects: Any, *, run: str, host: str, seq: int, local_path: Path,
                 log: Callable[[str], None] = print, on_fenced: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic,
                 every_s: float = EVERY_S, timeout_s: float = WRITE_TIMEOUT_S, background: bool = True,
                 stale_after_s: float | None = STALE_AFTER_S, host_id: str | None = None) -> None:
        self.objects, self.run, self.host, self.host_id = objects, run, host, host_id
        self.nonce = uuid.uuid4().hex
        self.seq = int(seq)  # the last number written (by anyone, when this run up starts)
        self.local_path = Path(local_path)
        self.log, self.on_fenced = log, on_fenced
        self.clock, self.monotonic = clock, monotonic
        self.every_s, self.timeout_s, self.background = float(every_s), float(timeout_s), background
        self.fenced: str | None = None
        # fenced because another writer wrote past this one (its writer; None: not, or cut off from the store)
        self.fenced_by: dict[str, Any] | None = None
        self.refused: str | None = None  # the first write (start) lost or did not land: why
        self._starting = False
        self._error: str | None = None
        self.writes = 0  # mirror writes done (tests)
        self._at: float | None = None
        self._warned = False
        self._cond = threading.Condition()
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._submitted = 0
        self._done = 0
        self._thread: threading.Thread | None = None
        # the fence of a writer cut off from the store (stale): no write confirmed since
        # ``_ok_at`` while ``_tried_at`` says it tried (monotonic)
        self.stale_after_s = None if stale_after_s is None else float(stale_after_s)
        if self.stale_after_s is not None and self.every_s + self.timeout_s >= self.stale_after_s:
            raise ValueError(f"the state mirror's write interval plus its write timeout ({self.every_s:.0f}s + "
                             f"{self.timeout_s:.0f}s) must stay under its stale limit ({self.stale_after_s:.0f}s): "
                             "a writer that only waits would fence itself")
        self._ok_at = self.monotonic()
        self._tried_at: float | None = None
        self._throttled: dict[str, Any] | None = None  # the newest save not written (flush)

    @property
    def writer(self) -> dict[str, str]:
        return {"host": self.host, "nonce": self.nonce, **({"host_id": self.host_id} if self.host_id else {})}

    def start(self, state: dict[str, Any]) -> str | None:
        """This run up's first write, made now in the caller's thread (no background thread,
        no timeout race). None: written and confirmed, this run up drives the run. Else why
        not: a write that did not land, or one that lost to another writer (a number past the
        one this run up started from, or every number it tried claimed). That is a refusal
        (run up exits 2), never a fence: no event, and nothing more is written."""
        self._starting = True
        try:
            self._at = self.monotonic()
            done = self.write_now(mirrored_state(state))
        finally:
            self._starting = False
        if done and self.writes >= 1 and self.fenced is None:
            return None
        why = self.refused or self.fenced or (f"the first write of run up's state to the run store did not land "
                                              f"({self._error or 'not confirmed'})")
        self.fenced = self.fenced or why  # nothing more is written
        return why

    # -- the caller's side
    def submit(self, state: dict[str, Any], *, force: bool = False) -> bool:
        """Mirror ``state`` (at most every ``every_s`` unless ``force``). A forced write waits
        for the store at most ``timeout_s``. False: nothing submitted (throttled, fenced)."""
        if self.fenced is not None:
            return False
        now = self.monotonic()
        if not force and self._at is not None and now - self._at < self.every_s:
            self._throttled = state
            return False
        self._at = now
        self._throttled = None
        snapshot = mirrored_state(state)
        if not self.background:
            self.write_now(snapshot)
            return True
        with self._cond:
            self._submitted += 1
            ticket = self._submitted
            self._pending = (ticket, snapshot)
            self._cond.notify_all()
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="up-state-mirror", daemon=True)
                self._thread.start()
            if force:
                deadline = self.monotonic() + self.timeout_s
                while self._done < ticket and self.fenced is None:
                    left = deadline - self.monotonic()
                    if left <= 0:
                        self._warn(f"the store did not answer within {self.timeout_s:.0f}s (the write goes on "
                                   "in the background)")
                        break
                    self._cond.wait(min(left, 1.0))
        return True

    def flush(self) -> bool:
        """The newest save that the throttle kept back, written now (forced: waited for at
        most the write timeout). False: none, or fenced."""
        state, self._throttled = self._throttled, None
        return False if state is None or self.fenced is not None else self.submit(state, force=True)

    def stale(self) -> str | None:
        """Why this writer may drive the run no more because it is cut off from the store
        (module doc): no write was confirmed for ``stale_after_s`` since its first try after
        the last confirmed one.
        Fenced then, as by another writer (:attr:`fenced`, the event). None: it may."""
        if self.fenced is not None:
            return self.fenced
        if self.stale_after_s is None or self._tried_at is None or self._tried_at <= self._ok_at:
            return None
        silent = self.monotonic() - self._tried_at  # since its first try after the last confirmed write
        if silent < self.stale_after_s:
            return None
        try:  # a successor there (a read still answers): fenced by it, so its islands are handed over
            _raw, current = self._read_raw()
            theirs = self._theirs_newer(current, self.seq)
        except Exception:  # noqa: BLE001 - cut off from the store: the stale fence below
            theirs = None
        if theirs is not None:
            self._fence(theirs, seq_of(current))
            return self.fenced
        self._fence({}, None, why=(f"no write of run up's state to the run store was confirmed for {silent:.0f}s "
                                   f"(limit {self.stale_after_s:.0f}s: the island records lock's lease plus the "
                                   "takeover margin): another control host may drive this run now"))
        return self.fenced

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                ticket, snapshot = self._pending
                self._pending = None
            try:
                self.write_now(snapshot)
            finally:
                with self._cond:
                    self._done = max(self._done, ticket)
                    self._cond.notify_all()
            if self.fenced is not None:
                return

    # -- the write
    def write_now(self, snapshot: dict[str, Any]) -> bool:
        """One conditional write (module doc); never raises. False: not written."""
        if self._tried_at is None or self._tried_at <= self._ok_at:
            self._tried_at = self.monotonic()
        try:
            done = self._write(snapshot)
        except Exception as error:  # noqa: BLE001 - kept local; tried again at the next save
            self._error = f"{type(error).__name__}: {error}"
            self._warn(self._error)
            return False
        if done:
            self._ok_at = self.monotonic()
        return done

    def _read_raw(self) -> tuple[bytes | None, dict[str, Any] | None]:
        raw = self.objects.read(STATE_MIRROR)
        if raw is None:
            return None, None
        doc = json.loads(raw)
        return raw, (doc if isinstance(doc, dict) and doc.get("schema") == SCHEMA else None)

    def _theirs_newer(self, current: dict[str, Any] | None, than: int) -> dict[str, Any] | None:
        """The writer of ``current`` when it is another writer's with a number past ``than``."""
        if current is None:
            return None
        theirs = current.get("writer") if isinstance(current.get("writer"), dict) else {}
        return theirs if theirs.get("nonce") != self.nonce and seq_of(current) > than else None

    def _replace(self, raw: bytes | None, data: bytes) -> bool:
        """The mirror replaced by ``data`` only while it still holds ``raw`` (None: absent, an
        exclusive create). Never a plain write: a store that takes no conditional write raises."""
        if raw is None:
            return bool(self.objects.create(STATE_MIRROR, data))
        replace = getattr(self.objects, "replace_if", None)
        done = replace(STATE_MIRROR, data, raw) if callable(replace) else None
        if done is None:
            raise RuntimeError(f"{CONDITIONAL_WRITES_NEEDED}: the store did not take the conditional write of "
                               f"{STATE_MIRROR}")
        return bool(done)

    def _mine(self, current: dict[str, Any] | None, seq: int) -> bool:
        """``current`` is this writer's own write of number ``seq``."""
        writer = (current or {}).get("writer")
        return isinstance(writer, dict) and writer.get("nonce") == self.nonce and seq_of(current) == seq

    def _write(self, snapshot: dict[str, Any]) -> bool:
        for _attempt in range(WRITE_TRIES):
            raw, current = self._read_raw()
            theirs = self._theirs_newer(current, self.seq)  # past what this writer wrote (at its start: read)
            if theirs is not None:
                return self._fence(theirs, seq_of(current))
            nxt = max(seq_of(current), self.seq) + 1
            doc = {"schema": SCHEMA, "run": self.run, "host": self.host, "writer": self.writer, "seq": nxt,
                   "at": round(self.clock(), 3), "state": snapshot}
            data = json.dumps(doc, sort_keys=True).encode()
            if self.fenced is not None:  # fenced meanwhile (stale, or by the caller): nothing more lands
                return False
            try:
                done = self._replace(raw, data)
            except Exception:
                _raw, after = self._read_raw()  # the answer lost: the write may have landed
                if not self._mine(after, nxt):
                    raise
                done = True
            if not done:
                _raw, after = self._read_raw()
                done = self._mine(after, nxt)  # this write itself, its answer lost
            if done:
                return self._written(nxt)
            if self._starting:  # the first write lost its compare-and-swap: a refusal, nobody fenced
                theirs = self._theirs_newer(after, self.seq) or (after or {}).get("writer") or {}
                return self._fence(theirs, seq_of(after))
        raise RuntimeError(f"{STATE_MIRROR} changed under {WRITE_TRIES} conditional writes in a row")

    def _written(self, nxt: int) -> bool:
        self.seq = nxt
        self.writes += 1
        self._save_local()
        if self._warned:
            self._warned = False
            self.log("on-site control plane: run up's state is written to the run store again")
        return True

    def _save_local(self) -> None:
        try:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.local_path.with_name(self.local_path.name + ".tmp")
            tmp.write_text(json.dumps({"run": self.run, "seq": self.seq, "host": self.host, "nonce": self.nonce},
                                      sort_keys=True))
            tmp.replace(self.local_path)
        except OSError:
            pass  # the next write tries again; a missing file only makes a restore more likely

    def _fence(self, theirs: dict[str, Any], seq: int | None, *, why: str | None = None) -> bool:
        why = why or (f"run up's state in the run store is written by host {theirs.get('host') or '?'} now (sequence "
                      f"{seq}, this run up wrote {self.seq}): another control host drives this run")
        if self._starting:  # the first write lost: a refusal of the start, no fence (module doc)
            self.refused = f"another control host writes run up's state in the run store ({why})"
            self.log(f"on-site control plane: {self.refused}; this run up starts nothing (exit 2, its unit tries "
                     "again and reads the mirror again)")
            return False
        self.fenced = why
        self.fenced_by = dict(theirs) if theirs else None
        self.log(f"CRITICAL: on-site control plane: {why}; this run up writes, stops and orders nothing more and "
                 f"exits {CONTROL_FENCED_EXIT} (its unit is not restarted)")
        if self.on_fenced is not None:
            try:
                self.on_fenced(why)
            except Exception:  # noqa: BLE001 - the fence holds whether the event is written or not
                pass
        return False

    def _warn(self, detail: str) -> None:
        if not self._warned:
            self._warned = True
            self.log(f"WARNING: on-site control plane: run up's state was not written to the run store ({detail}); "
                     "another host would start from older state")
