"""``yeto run up``'s state in the run store, for an on-site control plane
(``fleet.syncer_service``, ``docs/OWN_CLUSTERS.md``): a run up on another host continues
the run from it (:meth:`yeto.rl.up.Run.start_state_mirror`).

Only the fields of :data:`KEYS` are kept there (the spend's clocks and periods, the steps
done, the breaker and stop records); :data:`EXCLUDED` names the run up state fields that
never are (this host's processes). A removed pool's row keeps only its billing fields
(:data:`RETIRED_POOL_FIELDS`): never its frozen pool config, which may carry a provider's
environment. No key is ever in it: the island HMAC key stays in the environment file.

One writer at a time. Every write carries its writer (host and a nonce of this run up) and
a sequence number one past the last, claimed by an exclusive create of
``fleet/control/up-state-seq/<seq>.json``. A writer that finds the mirror written by
another writer with a higher number, or its next number claimed by another writer, writes
nothing more and is fenced (:attr:`StateMirror.fenced`): another control host drives the
run now (run up ends with :data:`CONTROL_FENCED_EXIT`). A run up that starts takes the next
number after the mirror's, so the newest control host wins.

The writes run on a background thread (:class:`StateMirror`): a store that does not answer
never holds the monitor. A forced write (a stop, a breaker, the end of the run) waits for
it at most :data:`WRITE_TIMEOUT_S`."""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

STATE_MIRROR = "fleet/control/up-state.json"
STATE_MIRROR_SEQ = "fleet/control/up-state-seq"
SCHEMA = "yeto.fleet.up-state/v1"
EVERY_S = 30.0  # a save mirrors at most this often (a forced one always)
WRITE_TIMEOUT_S = 20.0  # a forced write waits this long for the store, then the monitor goes on
KEEP_MARKERS = 16  # the sequence claims kept behind the newest (older ones are deleted)
LOCAL_FILE = "state-mirror.json"  # in the run's state dir: the last sequence number this host wrote
START_CLAIM_TRIES = 8  # a run up that starts skips this many numbers another writer claimed
# run up's exit when another control host drives the run (yeto.rl.fleet_service CONTROL_PREVENT)
CONTROL_FENCED_EXIT = 76

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


def marker_key(seq: int) -> str:
    return f"{STATE_MIRROR_SEQ}/{int(seq):012d}.json"


class StateMirror:
    """The run store's copy of one run up's state (module doc). ``submit`` is called at each
    save; ``fenced`` says why this run up may write no more (None: it may)."""

    def __init__(self, objects: Any, *, run: str, host: str, seq: int, local_path: Path,
                 log: Callable[[str], None] = print, on_fenced: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic,
                 every_s: float = EVERY_S, timeout_s: float = WRITE_TIMEOUT_S, background: bool = True) -> None:
        self.objects, self.run, self.host = objects, run, host
        self.nonce = uuid.uuid4().hex
        self.seq = int(seq)  # the last number written (by anyone, when this run up starts)
        self.local_path = Path(local_path)
        self.log, self.on_fenced = log, on_fenced
        self.clock, self.monotonic = clock, monotonic
        self.every_s, self.timeout_s, self.background = float(every_s), float(timeout_s), background
        self.fenced: str | None = None
        self.writes = 0  # mirror writes done (tests)
        self._started = False  # the first write of this run up claims past the numbers others hold
        self._at: float | None = None
        self._warned = False
        self._cond = threading.Condition()
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._submitted = 0
        self._done = 0
        self._thread: threading.Thread | None = None

    @property
    def writer(self) -> dict[str, str]:
        return {"host": self.host, "nonce": self.nonce}

    # -- the caller's side
    def submit(self, state: dict[str, Any], *, force: bool = False) -> bool:
        """Mirror ``state`` (at most every ``every_s`` unless ``force``). A forced write waits
        for the store at most ``timeout_s``. False: nothing submitted (throttled, fenced)."""
        if self.fenced is not None:
            return False
        now = self.monotonic()
        if not force and self._at is not None and now - self._at < self.every_s:
            return False
        self._at = now
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
        try:
            return self._write(snapshot)
        except Exception as error:  # noqa: BLE001 - kept local; tried again at the next save
            self._warn(f"{type(error).__name__}: {error}")
            return False

    def _write(self, snapshot: dict[str, Any]) -> bool:
        current = read_mirror(self.objects)
        if current is not None:
            theirs = current.get("writer") if isinstance(current.get("writer"), dict) else {}
            if theirs.get("nonce") != self.nonce and seq_of(current) > self.seq:
                return self._fence(theirs, seq_of(current))
        tries = START_CLAIM_TRIES if not self._started else 1
        nxt = self.seq + 1
        while True:
            if self.objects.create(marker_key(nxt), json.dumps(self.writer, sort_keys=True).encode()):
                break
            theirs = _read_json(self.objects, marker_key(nxt)) or {}
            if theirs.get("nonce") == self.nonce:  # our own create, its answer lost
                break
            tries -= 1
            if tries <= 0:
                return self._fence(theirs, nxt)
            nxt += 1  # a run up that starts: a number claimed by a writer that never wrote it
        doc = {"schema": SCHEMA, "run": self.run, "host": self.host, "writer": self.writer, "seq": nxt,
               "at": round(self.clock(), 3), "state": snapshot}
        self.objects.write(STATE_MIRROR, json.dumps(doc, sort_keys=True).encode())
        self.seq, self._started = nxt, True
        self.writes += 1
        self._save_local()
        if nxt > KEEP_MARKERS:
            try:
                self.objects.delete(marker_key(nxt - KEEP_MARKERS))
            except Exception:  # noqa: BLE001 - an old claim left behind is harmless
                pass
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

    def _fence(self, theirs: dict[str, Any], seq: int) -> bool:
        why = (f"run up's state in the run store is written by host {theirs.get('host') or '?'} now (sequence {seq}, "
               f"this run up wrote {self.seq}): another control host drives this run")
        self.fenced = why
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
