"""Crash-safe resource ledger of one run: ``<run dir>/resources.jsonl``.

Every billed resource a run creates is appended here the moment it exists
(Modal app, Modal FunctionCall, NC lease ticket, sky cluster), and a second
record when it is released. Lines are appended with ``O_APPEND`` and fsynced,
so a launcher that is SIGKILLed leaves a ledger that lists everything it had
created up to its last completed call. :mod:`.cleanup` releases exactly the
entries listed here - it never looks resources up by name pattern.

Record shape (one JSON object per line)::

    {"op": "intent"|"create"|"release"|"release_failed", "kind": <kind>,
     "id": <provider id or None>, "name": <display name>, "ts": <unix>, "attrs": {...}}

``intent`` is written right before a create whose id is only known
afterwards (``modal app deploy``); a crash between the two leaves an intent
that cleanup resolves by exact name AND creation time after the intent, or
reports as unresolved.

Kinds and the attrs each writer fills in:

* ``modal_app``  id = app id (``ap-...``), name = app name; attrs: gpu, gpus,
  usd_per_hour (optional estimate rate), shared (bool: app is reused by other runs).
* ``modal_call`` id = FunctionCall id (``fc-...``), name = island; attrs: app_id, app_name.
* ``nc_lease``   id = ticket; attrs: account, granted_at, lease_until (ISO or unix),
  gpus, rate_ucr_per_gpu_hour, usd_per_gpu_hour (optional).
* ``sky_cluster`` id = cluster name (sky names are exact ids).

Other processes of the same run (sampler launch scripts, the NC pool) find the
ledger through ``$YETO_RESOURCE_LEDGER`` (see :func:`current`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEDGER_ENV = "YETO_RESOURCE_LEDGER"
LEDGER_FILE = "resources.jsonl"

KINDS = ("modal_app", "modal_call", "nc_lease", "sky_cluster")


@dataclass
class Resource:
    kind: str
    id: str | None
    name: str | None
    created_at: float
    attrs: dict[str, Any] = field(default_factory=dict)
    released_at: float | None = None
    release_error: str | None = None
    intent_only: bool = False

    @property
    def key(self) -> str:
        return _key(self.kind, self.id, self.name)

    @property
    def open(self) -> bool:
        return self.released_at is None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _key(kind: str, rid: str | None, name: str | None) -> str:
    return f"{kind}:{rid if rid is not None else 'intent:' + str(name)}"


class ResourceLedger:
    def __init__(self, path: str | os.PathLike, *, clock=time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self._lock = threading.Lock()

    @classmethod
    def for_run(cls, run: str, **kw) -> "ResourceLedger":
        from .. import runs

        return cls(runs.run_dir(run) / LEDGER_FILE, **kw)

    # -- writing ---------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)

    def _rec(self, op: str, kind: str, rid: str | None, name: str | None, attrs: dict) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown resource kind {kind!r}; one of {KINDS}")
        self._append({"op": op, "kind": kind, "id": rid, "name": name, "ts": self.clock(),
                      "attrs": attrs})

    def intent(self, kind: str, name: str, **attrs: Any) -> None:
        self._rec("intent", kind, None, name, attrs)

    def created(self, kind: str, rid: str, name: str | None = None, **attrs: Any) -> None:
        if not rid:
            raise ValueError("a created resource needs its provider id")
        self._rec("create", kind, str(rid), name, attrs)

    def released(self, kind: str, rid: str | None, name: str | None = None, **attrs: Any) -> None:
        self._rec("release", kind, None if rid is None else str(rid), name, attrs)

    def release_failed(self, kind: str, rid: str | None, name: str | None, error: str) -> None:
        self._rec("release_failed", kind, rid, name, {"error": str(error)[:500]})

    # -- reading ---------------------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        out = []
        for line in text.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a torn last line after a crash
            if isinstance(rec, dict) and rec.get("kind") in KINDS:
                out.append(rec)
        return out

    def resources(self) -> list[Resource]:
        """Folded state, in creation order. A create resolves the newest open
        intent of the same kind and name."""
        res: dict[str, Resource] = {}
        for r in self.records():
            kind, rid, name, ts = r["kind"], r.get("id"), r.get("name"), float(r.get("ts") or 0)
            attrs = dict(r.get("attrs") or {})
            op = r.get("op")
            if op == "intent":
                k = _key(kind, None, name)
                res[k] = Resource(kind, None, name, ts, attrs, intent_only=True)
            elif op == "create":
                ik = _key(kind, None, name)
                prev = res.pop(ik, None) if name is not None else None
                k = _key(kind, rid, name)
                if k in res:  # a re-recorded create: keep the first creation time
                    res[k].attrs.update(attrs)
                    res[k].released_at = None
                    continue
                created = prev.created_at if prev is not None else ts
                res[k] = Resource(kind, rid, name, created, {**(prev.attrs if prev else {}), **attrs})
            elif op in ("release", "release_failed"):
                k = _key(kind, rid, name)
                if k not in res:
                    continue
                if op == "release":
                    res[k].released_at = ts
                    res[k].release_error = None
                else:
                    res[k].release_error = attrs.get("error")
        # a stopped app ends every FunctionCall in it
        stopped = [a for a in res.values() if a.kind == "modal_app" and a.released_at is not None]
        for c in res.values():
            if c.kind != "modal_call" or c.released_at is not None:
                continue
            for a in stopped:
                same = (c.attrs.get("app_id") and c.attrs.get("app_id") == a.id) or \
                    (not c.attrs.get("app_id") and c.attrs.get("app_name") == a.name)
                if same and a.released_at >= c.created_at:
                    c.released_at = a.released_at
                    break
        return list(res.values())

    def open_resources(self) -> list[Resource]:
        return [r for r in self.resources() if r.open]


class NullLedger:
    """Records nothing (a programmatic launch that opted out of the ledger)."""

    path = None

    def __getattr__(self, name):
        if name in ("intent", "created", "released", "release_failed"):
            return lambda *a, **k: None
        if name in ("records", "resources", "open_resources"):
            return lambda: []
        raise AttributeError(name)


def current(environ=None) -> ResourceLedger | None:
    """The ledger of the run this process belongs to (``$YETO_RESOURCE_LEDGER``), or None."""
    env = os.environ if environ is None else environ
    path = env.get(LEDGER_ENV)
    return ResourceLedger(path) if path else None


def record_safely(fn_name: str, *args: Any, ledger: ResourceLedger | None = None, **kw: Any) -> None:
    """Call ``ledger.<fn_name>(...)`` on the given or current ledger; a ledger
    write error is printed loudly but never breaks the create it records."""
    led = ledger if ledger is not None else current()
    if led is None:
        return
    try:
        getattr(led, fn_name)(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[cost] WARNING: resource ledger {led.path} write failed: {exc}", flush=True)
