"""``rl_breaker`` / ``rl_breaker_status`` events and the run's stop reason.

A component reports a trip with :func:`trip` through its existing tape writer, next to
(never instead of) its legacy event, and then raises or exits exactly as before. A trip
whose action stops the component records ``stop_reason = "breaker:<name>"``; exit codes
do not change.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yeto.breakers.registry import Registry, default_registry
from yeto.breakers.spec import STOP_ACTIONS, Breaker

SCHEMA = "yeto-breaker-v1"
BREAKER_EVENT = "rl_breaker"
STATUS_EVENT = "rl_breaker_status"
STATES = ("warn", "trip", "clear", "shadow", "config_mismatch")
STOP_REASON_ENV = "YETO_STOP_REASON"
STOP_REASON_PREFIX = "breaker:"
REQUIRED_KEYS = ("event", "schema", "name", "layer", "kind", "state", "action", "value", "limit", "unit",
                 "direction", "scope", "count", "time_unix")


def _spec(name: str, registry: Registry | None) -> Breaker | None:
    try:
        reg = registry if registry is not None else default_registry()
    except Exception as error:  # noqa: BLE001 - a broken layer file never stops a trip report
        print(f"[breakers] registry not loaded ({error}); event for {name} without registry fields",
              file=sys.stderr)
        return None
    return reg.get(name) if name in reg else None


def _number(value: Any) -> Any:
    return float(value) if isinstance(value, int) and not isinstance(value, bool) else value


def breaker_event(name: str, *, state: str, value: float | None, limit: float | None, action: str | None = None,
                  scope: Mapping[str, Any] | None = None, count: int = 1, detail: str = "",
                  legacy_event: str | None = None, config_source: str | None = None,
                  stop_reason: str | None = None, run: str | None = None, unit: str | None = None,
                  now: float | None = None, registry: Registry | None = None) -> dict[str, Any]:
    """One ``rl_breaker`` record (design 8.1); layer, kind, unit, direction from the registry."""
    if state not in STATES:
        raise ValueError(f"breaker event state {state!r} is not one of {STATES}")
    spec = _spec(name, registry)
    event: dict[str, Any] = {
        "event": BREAKER_EVENT, "schema": SCHEMA, "name": name,
        "layer": spec.layer if spec else name.split(".", 1)[0],
        "kind": spec.kind if spec else None, "state": state,
        "action": action if action is not None else (spec.default_action if spec else None),
        "value": value, "limit": limit,
        "unit": unit if unit is not None else (_unit(spec) if spec else ""),
        "direction": spec.direction if spec else "max",
        "scope": dict(scope or {}), "count": int(count),
        "time_unix": round(time.time() if now is None else now, 3)}
    for key, field_value in (("detail", detail), ("legacy_event", legacy_event),
                             ("config_source", config_source), ("stop_reason", stop_reason), ("run", run)):
        if field_value:
            event[key] = field_value
    return event


def _unit(spec: Breaker) -> str:
    """The unit of the entry's limit: its first param's (entries with one limit are the rule)."""
    return spec.params[0].unit if spec.params else ""


def trip(name: str, value: float | None, threshold: float | None, action: str | None = None, *,
         state: str = "trip", stop_reason: str | bool | None = None,
         emit: Callable[[dict[str, Any]], Any] | None = None, registry: Registry | None = None,
         **fields: Any) -> dict[str, Any]:
    """Report one trip (or warn / clear / shadow / config_mismatch): build the event, emit it, return it.

    Control flow is the caller's: this never raises from ``emit``. With a stopping action
    (``halt`` / ``stop``) and ``state="trip"``, the event carries ``stop_reason`` and the process
    records it (:func:`record_stop_reason`); ``stop_reason=False`` suppresses that."""
    spec = _spec(name, registry)
    act = action if action is not None else (spec.default_action if spec else None)
    reason: str | None = None
    if stop_reason is not False and state == "trip":
        if isinstance(stop_reason, str) and stop_reason:
            reason = stop_reason
        elif act in STOP_ACTIONS:
            reason = stop_reason_for(name)
    event = breaker_event(name, state=state, value=value, limit=threshold, action=act, stop_reason=reason,
                          registry=registry, **fields)
    if reason:
        record_stop_reason(reason)
    if emit is not None:
        try:
            emit(event)
        except Exception as error:  # noqa: BLE001 - reporting never changes what the breaker does
            print(f"[breakers] could not emit {BREAKER_EVENT} {name}: {error}", file=sys.stderr)
    return event


def print_event(event: Mapping[str, Any], prefix: str = "YETO_RL_EVENT ") -> None:
    """The trainer's tape: one ``YETO_RL_EVENT {json}`` line on stdout."""
    print(prefix + json.dumps(event, sort_keys=True, default=str), flush=True)


def status_item(name: str, value: float | None, limit: float | None, *, detail: str = "",
                registry: Registry | None = None, **extra: Any) -> dict[str, Any]:
    """One item of a ``rl_breaker_status`` snapshot; unit, gauge and ratio from the registry."""
    spec = _spec(name, registry)
    direction = spec.direction if spec else "max"
    item: dict[str, Any] = {"name": name, "value": value, "limit": limit,
                            "unit": _unit(spec) if spec else "", "gauge": spec.gauge if spec else "none",
                            "ratio": ratio(value, limit, direction)}
    if detail:
        item["detail"] = detail
    item.update(extra)
    return item


def ratio(value: float | None, limit: float | None, direction: str = "max") -> float | None:
    """How close ``value`` is to ``limit`` (1.0: at the limit); None when either is unknown or 0."""
    if value is None or limit is None:
        return None
    try:
        v, lim = float(value), float(limit)
    except (TypeError, ValueError):
        return None
    if direction == "min":
        return round(lim / v, 4) if v else None
    return round(v / lim, 4) if lim else None


def status_event(layer: str, items: Sequence[Mapping[str, Any]], *, scope: Mapping[str, Any] | None = None,
                 now: float | None = None) -> dict[str, Any]:
    """One ``rl_breaker_status`` snapshot (design 8.2)."""
    return {"event": STATUS_EVENT, "schema": SCHEMA, "layer": layer, "scope": dict(scope or {}),
            "items": [dict(i) for i in items], "time_unix": round(time.time() if now is None else now, 3)}


def validate_event(event: Mapping[str, Any]) -> list[str]:
    """Problems of an ``rl_breaker`` / ``rl_breaker_status`` record ([]: valid)."""
    problems = []
    if event.get("schema") != SCHEMA:
        problems.append(f"schema is {event.get('schema')!r}, want {SCHEMA}")
    if event.get("event") == BREAKER_EVENT:
        problems += [f"missing {k}" for k in REQUIRED_KEYS if k not in event]
        if event.get("state") not in STATES:
            problems.append(f"state {event.get('state')!r} is not one of {STATES}")
        if not isinstance(event.get("scope", {}), Mapping):
            problems.append("scope must be an object")
    elif event.get("event") == STATUS_EVENT:
        for k in ("layer", "scope", "items", "time_unix"):
            if k not in event:
                problems.append(f"missing {k}")
        for i, item in enumerate(event.get("items") or ()):
            problems += [f"items[{i}]: missing {k}" for k in ("name", "value", "limit", "unit", "ratio")
                         if k not in item]
    else:
        problems.append(f"event {event.get('event')!r} is not {BREAKER_EVENT} or {STATUS_EVENT}")
    return problems


# --- stop reason -----------------------------------------------------------------------------------

_STOP_LOCK = threading.Lock()
_STOP_REASON: str | None = None


def stop_reason_for(name: str) -> str:
    return f"{STOP_REASON_PREFIX}{name}"


def record_stop_reason(reason: str) -> None:
    """The first stop reason of this process wins (a later trip is a consequence, not the cause)."""
    global _STOP_REASON
    with _STOP_LOCK:
        if _STOP_REASON is None and reason:
            _STOP_REASON = reason


def current_stop_reason() -> str | None:
    return _STOP_REASON


def clear_stop_reason() -> None:
    """Tests, and a long-lived process starting a new run."""
    global _STOP_REASON
    with _STOP_LOCK:
        _STOP_REASON = None


def stop_reason_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    """The reason a parent handed down (``YETO_STOP_REASON``), if any."""
    value = (os.environ if environ is None else environ).get(STOP_REASON_ENV, "").strip()
    return value or None


def stop_reason_suffix(reason: str | None) -> str:
    """`` stop_reason=breaker:<name>`` for a final status line ('' when no reason is known)."""
    return f" stop_reason={reason}" if reason else ""
