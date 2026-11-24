"""``rl_breaker_status`` snapshots of one component: what to send and when (design 8.2).

A component (a trainer island, a sampler, the pool tick, the ``run up`` monitor) builds
one :class:`StatusEmitter` with its layer, its scope, its resolved settings and the
gauges it can read cheaply, and calls :meth:`StatusEmitter.poll` from a loop it already
runs. Each item of a snapshot carries the entry's effective settings (enabled, action,
params, the source that set them) and, where a gauge is registered, the current value
against its limit, so the dashboard and ``yeto run status`` see the limits a process
actually runs with, not only those of the config on the control host.

When a snapshot is sent:

* at the first poll (the component's start);
* when a setting or an item's band changes (a ratio crossing 0.5 / 0.7 / 0.9, a count
  gauge without a limit changing value, a new trip): at once for a rising band, a trip or a
  setting change, otherwise at most every ``min_gap_s``;
* every ``refresh_s`` while nothing changes, so a dashboard started late still sees it.

A poll between two checks costs one clock read; a check reads the gauges and compares
one tuple. Emitting never raises into the component: a failing gauge reads as None and
a failing writer is reported once on stderr.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .events import status_event, status_item

BANDS = (0.5, 0.7, 0.9)
CHECK_S = 15.0  # gauges are read at most this often
MIN_GAP_S = 60.0  # a change that is not urgent waits this long after the previous snapshot
REFRESH_S = 600.0  # an unchanged snapshot is sent again after this long
STATUS_KINDS = ("breaker", "guard", "retry_budget", "alert")  # invariants / semantic entries have no status
RESOLVED_ENV = "YETO_BREAKERS_RESOLVED"

# a gauge returns (value, limit) or a bare value (the limit then comes from the settings); None: unknown
GaugeFn = Callable[[], Any]


def band(ratio: float | None) -> int:
    """How many of ``BANDS`` ``ratio`` reached (0 below the first, 3 at or above 0.9)."""
    if ratio is None:
        return 0
    return sum(1 for b in BANDS if ratio >= b)


def primary_limit(params: Mapping[str, Any], spec: Any = None) -> float | None:
    """The limit a gauge without one of its own is compared with: the entry's first
    numeric parameter that is not a ratio or a switch (``timeout_s``, ``max_n``, ``usd``).
    With the entry's ``spec``, the first numeric parameter in the gauge's unit (its first
    param's, ``events._unit``): a ratio gauge (``sampler.vram_guard``) is compared with
    its ratio, never with a duration that happens to come first among the numbers."""
    units = {p.name: p.unit for p in getattr(spec, "params", None) or ()}
    want = spec.params[0].unit if units else None
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if want is not None and name in units:
            if units[name] == want:
                return float(value)
            continue
        if name.endswith("_ratio") or name == "warn_ratio":
            continue
        return float(value)
    return None


def resolved_for(*, argv: Sequence[str] | None = None, spec: Mapping[str, Any] | None = None,
                 env: Mapping[str, str] | None = None, run_config: Mapping[str, Any] | None = None,
                 pool: str | None = None, layers: Iterable[str] | None = None) -> Any:
    """The settings this process runs with: the resolved file handed to it
    (``YETO_BREAKERS_RESOLVED``, phase 4) when there is one, else the registry resolved
    over the surfaces the process can see (a sampler spec's own ``breakers`` section
    included; non-strict: a conflict never stops a component here, the component's own
    resolve or ``run up`` refuses it). None when nothing resolves; the snapshot then
    carries gauges only."""
    from .resolve import Sources, load_resolved, resolve  # noqa: PLC0415

    try:
        path = (env or {}).get(RESOLVED_ENV)
        if path and os.path.isfile(path):
            return load_resolved(path)
        section = spec.get("breakers") if isinstance(spec, Mapping) and run_config is None else None
        sources = Sources(run_config=run_config, argv=argv, spec=spec, env=env,
                          section=section if isinstance(section, Mapping) else None)
        return resolve(sources, pool=pool, strict=False, layers=None if layers is None else tuple(layers))
    except Exception as error:  # noqa: BLE001 - status is observation only
        print(f"[breakers] status: settings not resolved ({type(error).__name__}: {error})", file=sys.stderr)
        return None


def _source(entry: Any) -> dict[str, str] | None:
    src = getattr(entry, "source", None)
    if src is None:
        return None
    return {"kind": str(getattr(src, "kind", "")), "location": str(getattr(src, "location", ""))}


class StatusEmitter:
    """Snapshots of one component's entries (module doc). ``layers``: the layers whose
    entries this component reports (a sampler reports ``sampler`` and ``harness``);
    ``names`` narrows them further. ``emit`` gets the event dict (the component's own
    tape writer). ``gauges``: entry name -> :data:`GaugeFn`, more can be added with
    :meth:`gauge`."""

    def __init__(self, emit: Callable[[dict], Any], layers: str | Iterable[str], *,
                 resolved: Any = None, scope: Mapping[str, Any] | None = None,
                 gauges: Mapping[str, GaugeFn] | None = None, names: Iterable[str] | None = None,
                 clock: Callable[[], float] = time.time, check_s: float = CHECK_S,
                 min_gap_s: float = MIN_GAP_S, refresh_s: float = REFRESH_S) -> None:
        self.emit = emit
        self.layers = (layers,) if isinstance(layers, str) else tuple(layers)
        self.resolved = resolved
        self.scope = dict(scope or {})
        self.gauges: dict[str, GaugeFn] = dict(gauges or {})
        self.names = None if names is None else set(names)
        self.clock = clock
        self.check_s, self.min_gap_s, self.refresh_s = float(check_s), float(min_gap_s), float(refresh_s)
        self.last_trip: dict[str, dict] = {}  # entry -> its last rl_breaker (warn / trip / clear ...)
        self.sent = 0
        self._checked_at: float | None = None
        self._sent_at: float | None = None
        self._sent_sig: tuple | None = None
        self._sent_bands: dict[str, int] = {}
        self._dirty = False  # a trip was noted: the next poll checks at once
        self._failed = False

    def gauge(self, name: str, fn: GaugeFn) -> None:
        """Register (or replace) the gauge of entry ``name``."""
        self.gauges[name] = fn

    def note(self, event: Mapping[str, Any]) -> None:
        """An ``rl_breaker`` event this component emitted: it becomes the entry's last
        trip in the next snapshot, which is sent at the next poll."""
        if event.get("event") == "rl_breaker" and event.get("name"):
            self.last_trip[str(event["name"])] = {k: event.get(k) for k in (
                "state", "action", "value", "limit", "count", "detail", "stop_reason", "time_unix") if
                event.get(k) is not None}
            self._dirty = True

    # -- items ------------------------------------------------------------------
    def _entries(self) -> list[Any]:
        rb = self.resolved
        if rb is None:
            return []
        out = []
        for layer in self.layers:
            try:
                rows = rb.entries(layer)
            except Exception:  # noqa: BLE001
                rows = []
            out += [e for e in rows if e.spec.kind in STATUS_KINDS
                    and (self.names is None or e.spec.name in self.names)]
        return out

    def _read(self, name: str) -> tuple[float | None, float | None, str]:
        fn = self.gauges.get(name)
        if fn is None:
            return None, None, ""
        try:
            got = fn()
        except Exception:  # noqa: BLE001 - a gauge must never break the component
            return None, None, ""
        detail = ""
        if isinstance(got, tuple):
            got = tuple(got) + (None, None)
            value, limit, detail = got[0], got[1], got[2] or ""
        else:
            value, limit = got, None
        return _num(value), _num(limit), str(detail)

    def items(self) -> list[dict]:
        """The snapshot's items: every entry of the layers (settings, source, last trip)
        with its gauge where one is registered; gauges of names the settings do not
        know (no resolved settings in this process) are sent bare."""
        out, seen = [], set()
        for e in self._entries():
            name = e.spec.name
            seen.add(name)
            value, limit, detail = self._read(name)
            if limit is None and e.enabled:
                limit = primary_limit(e.params, e.spec)
            extra: dict[str, Any] = {"layer": e.spec.layer, "kind": e.spec.kind, "enabled": bool(e.enabled),
                                     "action": e.action, "params": dict(e.params), "source": _source(e)}
            if e.spec.implicit and (_source(e) or {}).get("kind") == "default":
                extra["implicit"] = True
            if name in self.last_trip:
                extra["last"] = self.last_trip[name]
            out.append(status_item(name, value, limit, detail=detail, **extra))
        for name in sorted(set(self.gauges) | set(self.last_trip)):
            if name in seen or (self.names is not None and name not in self.names):
                continue
            value, limit, detail = self._read(name)
            extra = {"last": self.last_trip[name]} if name in self.last_trip else {}
            out.append(status_item(name, value, limit, detail=detail, **extra))
        return out

    @staticmethod
    def _signature(items: list[dict]) -> tuple:
        sig = []
        for it in items:
            gauge_part = band(it.get("ratio")) if it.get("limit") is not None else it.get("value")
            sig.append((it["name"], it.get("enabled"), it.get("action"),
                        tuple(sorted((it.get("params") or {}).items(), key=lambda kv: kv[0])),
                        tuple(sorted((it.get("source") or {}).items())), gauge_part,
                        (it.get("last") or {}).get("time_unix")))
        return tuple(sig)

    # -- sending ----------------------------------------------------------------
    def poll(self, now: float | None = None, *, force: bool = False) -> dict | None:
        """Send a snapshot when one is due (module doc); returns it, or None."""
        now = self.clock() if now is None else now
        if not force and not self._dirty and self._checked_at is not None and now - self._checked_at < self.check_s:
            return None
        self._checked_at, self._dirty = now, False
        try:
            items = self.items()
        except Exception as error:  # noqa: BLE001
            self._warn(error)
            return None
        if not items:
            return None
        sig = self._signature(items)
        bands = {it["name"]: band(it.get("ratio")) for it in items}
        changed = sig != self._sent_sig
        settings_changed = self._sent_sig is not None and [s[:5] for s in sig] != [s[:5] for s in self._sent_sig]
        tripped = self._sent_sig is not None and [s[6] for s in sig] != [s[6] for s in self._sent_sig]
        rising = any(b > self._sent_bands.get(n, 0) for n, b in bands.items())
        since = None if self._sent_at is None else now - self._sent_at
        due = (force or since is None or since >= self.refresh_s
               or (changed and (rising or tripped or settings_changed or since >= self.min_gap_s)))
        if not due:
            return None
        event = status_event(self.layers[0], items, scope=self.scope or None, now=now)
        try:
            self.emit(event)
        except Exception as error:  # noqa: BLE001
            self._warn(error)
            return None
        self.sent += 1
        self._sent_at, self._sent_sig, self._sent_bands = now, sig, bands
        return event

    def _warn(self, error: BaseException) -> None:
        if not self._failed:
            self._failed = True
            print(f"[breakers] status snapshot not sent ({type(error).__name__}: {error})", file=sys.stderr)


class StatusThread:
    """Polls a :class:`StatusEmitter` every ``CHECK_S`` on a daemon thread, for a
    component without a loop of its own to call it from. The first snapshot is sent by
    :meth:`start` itself."""

    def __init__(self, emitter: StatusEmitter, *, interval_s: float | None = None) -> None:
        self.emitter = emitter
        self.interval_s = float(interval_s if interval_s is not None else emitter.check_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def note(self, event: Mapping[str, Any]) -> None:
        self.emitter.note(event)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.emitter.poll()

    def start(self) -> "StatusThread":
        if self._thread is None:
            self.emitter.poll(force=True)
            self._thread = threading.Thread(target=self._loop, name="yeto-breaker-status", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def status_thread(emit: Callable[[dict], Any], layers: str | Iterable[str], *,
                  scope: Mapping[str, Any] | None = None, argv: Sequence[str] | None = None,
                  spec: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None,
                  gauges: Mapping[str, GaugeFn] | None = None, **kw: Any) -> StatusThread | None:
    """A :class:`StatusThread` (not started) over the settings this process resolves
    (:func:`resolved_for`); None when nothing resolves or anything fails: a component
    never fails to start for its status snapshots."""
    try:
        layer_names = (layers,) if isinstance(layers, str) else tuple(layers)
        resolved = resolved_for(argv=argv, spec=spec, env=env, layers=layer_names)
        if resolved is None and not gauges:
            return None
        return StatusThread(StatusEmitter(emit, layers, resolved=resolved, scope=scope, gauges=gauges, **kw))
    except Exception as error:  # noqa: BLE001
        print(f"[breakers] status snapshots off ({type(error).__name__}: {error})", file=sys.stderr)
        return None


def start_status(emit: Callable[[dict], Any], layers: str | Iterable[str], **kw: Any) -> StatusThread | None:
    """:func:`status_thread`, started."""
    thread = status_thread(emit, layers, **kw)
    return None if thread is None else thread.start()


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if v == v and v not in (float("inf"), float("-inf")) else None
