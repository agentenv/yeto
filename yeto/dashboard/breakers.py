"""Breakers panel: every registered breaker's settings, headroom and trips (design 9).

Three inputs, each optional:

* ``breakers.resolved.json`` (``yeto breaker show --json`` format, written by ``run up``
  next to its ``state.json`` and under the run store's ``meta/``): on / off, action,
  effective parameters and their source, and the dependency-rule findings;
* ``rl_breaker_status`` snapshots (:mod:`yeto.breakers.status`): the settings each
  process runs with and the current value against the limit, per scope (island,
  sampler, pool);
* ``rl_breaker`` events: warns, trips, clears, shadow trips and config mismatches.

Tapes written before the registry existed carry none of these. Their legacy events
(``LEGACY_TRIPS``) are listed under the entry they belong to, and only when the record
says the breaker acted: most of them are routine telemetry (one ``rl_agentic_group``
per group, one ``rl_dispense_settle`` per task, ...). On a tape whose layer writes
``rl_breaker`` the new events are the record and that layer's legacy events are not used.

Pure observation, like the rest of the dashboard: missing stays None.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Callable

RESOLVED_FILE = "breakers.resolved.json"
STORE_RESOLVED_KEY = "meta/" + RESOLVED_FILE
STORE_RESOLVED_S = 300.0  # the store's resolved file is read again at most this often
TRIPS_MAX = 500  # rl_breaker / legacy trip records kept
TRIPS_VIEW = 60  # sent to the page, newest first
STATES = ("armed", "implicit", "off", "invariant")
SOURCE_RANK = {"breakers": 0, "switch": 1, "legacy": 2, "inherit": 3, "default": 4}
STOP_ACTIONS = ("halt", "stop")  # yeto.breakers.spec.STOP_ACTIONS (the page does not need the registry)
# an alert for a trip of an entry that drops / quarantines / pauses (it happens every round in a
# bad stretch) lasts this long after the trip; a stop / halt trip stays until the run restarts
RECENT_TRIP_S = 3600.0


def _f(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def scope_label(scope: Any) -> str:
    """``{"island": "0", "sampler_id": "s-1"}`` -> ``"s-1"``: the most specific name."""
    if not isinstance(scope, dict):
        return ""
    for key in ("slot", "sampler_id", "pool", "island"):
        if scope.get(key) not in (None, ""):
            return f"{'island ' if key == 'island' else ''}{scope[key]}"
    return ""


def _scope_key(scope: Any) -> str:
    if not isinstance(scope, dict):
        return ""
    return json.dumps({k: v for k, v in scope.items() if v is not None}, sort_keys=True, default=str)


def _n(r: Any, key: str) -> float:
    v = _f(r.get(key)) if isinstance(r, dict) else None
    return 0.0 if v is None else v


def _when(entry: str, state: str = "trip", test: Callable[[dict], bool] | None = None) -> Callable[[dict], list]:
    return lambda r: [(entry, state)] if test is None or test(r) else []


def _group(r: dict) -> list[tuple[str, str]]:
    """``rl_agentic_group`` is written for every group: a trip only for a group dropped
    below its kept-trial minimum, or with trials cut by the trial wall clock."""
    out = []
    if r.get("written") is False and r.get("dropped_reason") == "below_min_samples":
        out.append(("sampler.min_samples", "trip"))
    if _n(r.get("lost"), "wall_clock_cap") > 0:
        out.append(("sampler.trial_wall_clock", "trip"))
    return out


# Legacy event -> (record -> [(entry, state)]): when a record of a pre-registry event is
# a breaker acting. An event not listed here is never shown as a trip (routine telemetry,
# or one that does not say which entry acted). tests check each entry declares the event.
LEGACY_TRIPS: dict[str, Callable[[dict], list]] = {
    # routine records: a trip only under their condition
    "rl_agentic_group": _group,
    "rl_agentic_drain": _when("sampler.drain", test=lambda r: (
        _n(r, "abandoned_trials") > 0 or _n(r, "detached_trials") > 0 or _n(r, "detached_in_flight") > 0)),
    "rl_dispense_settle": _when("sampler.dispense_redo",
                                test=lambda r: r.get("state") in ("exhausted", "expired", "redo")),
    "rl_buffer_stale_skipped": _when("trainer.policy_age", test=lambda r: _n(r, "stale_groups") > 0),
    # one record per warn / trip / recovery
    "trainer_starvation_alert": _when("run.starvation", "warn"),
    "trainer_starvation_recovered": _when("run.starvation", "clear"),
    "trainer_starvation_stop": _when("run.starvation"),
    "cost_budget_warn": _when("launch.budget", "warn"),
    "cost_budget_exceeded": _when("launch.budget"),
    "idle_warn": _when("launch.idle", "warn"),
    "idle_cleared": _when("launch.idle", "clear"),
    "idle_shutdown": _when("launch.idle"),
    "RunStalled": _when("launch.stall"),
    "TrainerLoopGaveUp": _when("launch.trainer_failures"),
    "SampleBufferRolloutError": _when("trainer.sample_wait"),
    "DamagedPayloadLimit": _when("trainer.buffer_damaged"),
    "rl_nan_skip": _when("trainer.nan"),
    "SamplerLoadFatal": _when("sampler.load_failures"),
    "StoreReadOutage": _when("sampler.store_read_outage"),
    "nc_elastic_breaker_tripped": _when("pool.provision_breaker"),
    "nc_elastic_sampler_crash_loop": _when("pool.crash_loop"),
    "nc_elastic_engine_stalled": _when("pool.engine_stall"),
}


def best_source(sources: Any) -> dict | None:
    """The most specific source over an entry's per-param sources (section > switch >
    legacy alias > inherited > default), as ``{"kind", "location"}``."""
    if not isinstance(sources, dict):
        return None
    rows = [s for s in sources.values() if isinstance(s, dict) and s.get("kind")]
    if not rows:
        return None
    return dict(min(rows, key=lambda s: SOURCE_RANK.get(str(s.get("kind")), 9)))


def entry_state(kind: Any, enabled: Any, implicit: Any) -> str:
    if kind == "invariant":
        return "invariant"
    if enabled is False:
        return "off"
    if implicit:
        return "implicit"
    return "armed"


def _trip_row(r: dict, ts: float | None, *, legacy: str | None = None, name: str | None = None,
              state: str | None = None) -> dict:
    state = r.get("state") if legacy is None else state
    action = r.get("action")
    stop = r.get("stop_reason")
    if stop is None and legacy is None and state == "trip" and action in STOP_ACTIONS:
        stop = f"breaker:{r.get('name')}"
    scope = r.get("scope") if isinstance(r.get("scope"), dict) else (
        {"island": str(r["island_id"])} if r.get("island_id") is not None else {})
    return {"ts": ts, "name": name or r.get("name"), "layer": r.get("layer"), "state": state, "action": action,
            "value": _f(r.get("value")), "limit": _f(r.get("limit")), "unit": r.get("unit"),
            "count": r.get("count"), "scope": scope_label(scope), "scope_key": _scope_key(scope),
            "detail": str(r.get("detail") or "")[:300] or None, "stop_reason": stop,
            "legacy_event": legacy or r.get("legacy_event"), "legacy": legacy is not None,
            "config_source": r.get("config_source")}


class BreakerState:
    """What the reducer folded about breakers (module doc)."""

    def __init__(self) -> None:
        self.resolved: dict | None = None
        self.resolved_from: str | None = None
        self.status: dict[tuple[str, str], dict] = {}  # (entry, scope key) -> its newest status item
        self.settings: dict[str, dict] = {}  # entry -> settings a process reported (newest)
        self.trips: deque = deque(maxlen=TRIPS_MAX)
        self.legacy: deque = deque(maxlen=TRIPS_MAX)
        self.last: dict[str, dict] = {}  # entry -> its newest rl_breaker row
        self.latest: dict[tuple[str, str], dict] = {}  # (entry, scope key) -> newest rl_breaker row
        self.seen = False
        self.breaker_layers: set[str] = set()  # layers with an rl_breaker on the tapes

    # -- folding --------------------------------------------------------------------

    def fold(self, r: dict, iid: str | None, ts: float | None, *, cost_guard: bool = False) -> bool:
        """Fold one record; False for a record that is none of the breaker inputs.
        ``cost_guard``: the record is from the cost guard's tape, whose ``cost_tick`` is
        the ``launch.budget`` gauge (spend against the guard's budget)."""
        event = r.get("event")
        if cost_guard and event == "cost_tick":
            if _f(r.get("total_usd")) is None:
                return False
            self.seen = True
            limit = _f(r.get("budget_usd"))
            value = _f(r.get("total_usd"))
            self.status[("launch.budget", "")] = {
                "name": "launch.budget", "value": value, "limit": limit, "unit": "usd", "gauge": "level",
                "ratio": None if not limit else round(value / limit, 4), "detail": "cost guard spend",
                "ts": ts, "scope": "", "layer": "launch"}
            return True
        if event == "rl_breaker":
            if not r.get("name"):
                return False
            self.seen = True
            row = _trip_row(r, ts)
            self.trips.append(row)
            self.breaker_layers.add(str(row["layer"] or str(row["name"]).split(".")[0]))
            key = (str(row["name"]), row["scope_key"])
            if ts is None or (self.latest.get(key) or {}).get("ts") is None or ts >= self.latest[key]["ts"]:
                self.latest[key] = row
            if ts is None or (self.last.get(row["name"]) or {}).get("ts") is None or ts >= self.last[row["name"]]["ts"]:
                self.last[str(row["name"])] = row
            return True
        if event == "rl_breaker_status":
            self.seen = True
            scope = r.get("scope") if isinstance(r.get("scope"), dict) else (
                {"island": iid} if iid is not None else {})
            for it in r.get("items") or ():
                if not isinstance(it, dict) or not it.get("name"):
                    continue
                name = str(it["name"])
                key = (name, _scope_key(scope))
                prev = self.status.get(key)
                if prev is not None and ts is not None and prev.get("ts") is not None and ts < prev["ts"]:
                    continue  # an older snapshot read after a newer one (tapes read in any order)
                self.status[key] = {**it, "ts": ts, "scope": scope_label(scope), "layer": it.get("layer") or r.get("layer")}
                if "enabled" in it or "params" in it:
                    self.settings[name] = {k: it.get(k) for k in ("layer", "kind", "enabled", "action", "params",
                                                                  "source", "implicit")} | {"ts": ts}
            return True
        rule = LEGACY_TRIPS.get(event) if isinstance(event, str) else None
        if rule is None:
            return False
        hits = rule(r)
        for name, state in hits:
            self.seen = True
            self.legacy.append(_trip_row(r, ts, legacy=event, name=name, state=state))
        return bool(hits)

    def load_resolved(self, doc: Any, source: str) -> bool:
        """Take a ``breakers.resolved.json`` document; False when it is not one."""
        if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
            return False
        self.resolved, self.resolved_from = doc, source
        self.seen = True
        return True

    # -- views ----------------------------------------------------------------------
    def _rows(self) -> dict[str, dict]:
        """One row per entry: resolved settings first, the processes' own report
        filling what the resolved file does not have."""
        rows: dict[str, dict] = {}
        for e in (self.resolved or {}).get("entries") or ():
            if not isinstance(e, dict) or not e.get("name") or e.get("kind") == "semantic":
                continue
            rows[e["name"]] = {"name": e["name"], "layer": e.get("layer"), "kind": e.get("kind"),
                               "enabled": e.get("enabled"), "action": e.get("action"),
                               "params": e.get("params") or {}, "units": e.get("units") or {},
                               "implicit": bool(e.get("implicit")), "source": best_source(e.get("sources")),
                               "redundant": e.get("redundant") or [], "depends_on": e.get("depends_on") or [],
                               "switch_groups": e.get("switch_groups") or [], "from": "resolved"}
        for name, s in self.settings.items():
            row = rows.get(name)
            if row is None:
                rows[name] = {"name": name, "layer": s.get("layer") or name.split(".")[0], "kind": s.get("kind"),
                              "enabled": s.get("enabled"), "action": s.get("action"), "params": s.get("params") or {},
                              "units": {}, "implicit": bool(s.get("implicit")),
                              "source": s.get("source") if isinstance(s.get("source"), dict) else None,
                              "redundant": [], "depends_on": [], "switch_groups": [], "from": "status"}
            elif s.get("params") and row["params"] and s.get("params") != row["params"]:
                row["runtime_params"] = s["params"]  # the process runs with other values (config_mismatch)
        legacy = {t["name"] for t in self.legacy if str(t["name"]).split(".")[0] not in self.breaker_layers}
        for name in set(self.last) | legacy | {k[0] for k in self.status}:
            if name not in rows:
                trip = self.last.get(name) or {}
                rows[name] = {"name": name, "layer": trip.get("layer") or name.split(".")[0], "kind": None,
                              "enabled": None, "action": trip.get("action"), "params": {}, "units": {},
                              "implicit": False, "source": None, "redundant": [], "depends_on": [],
                              "switch_groups": [], "from": "events"}
        return rows

    def _gauge(self, name: str) -> dict | None:
        """The entry's worst current value over the scopes that report it."""
        rows = [v for (n, _), v in self.status.items() if n == name and (v.get("value") is not None
                                                                          or v.get("limit") is not None)]
        if not rows:
            return None
        worst = max(rows, key=lambda v: (_f(v.get("ratio")) if _f(v.get("ratio")) is not None else -1.0,
                                         v.get("ts") or 0))
        return {"value": _f(worst.get("value")), "limit": _f(worst.get("limit")), "unit": worst.get("unit"),
                "ratio": _f(worst.get("ratio")), "gauge": worst.get("gauge"), "detail": worst.get("detail"),
                "scope": worst.get("scope"), "scopes": len(rows), "ts": worst.get("ts")}

    def view(self, gauges: dict[str, dict] | None = None) -> dict | None:
        """The panel's data; None when no input has anything (older tapes, no file).
        ``gauges``: entry -> gauge the page derives itself (the run budget from the cost
        estimate), used for an entry no process reports a value for."""
        if not self.seen:
            return None
        # a layer that writes rl_breaker is told by it alone (its legacy events are telemetry)
        legacy = [t for t in self.legacy if str(t["name"]).split(".")[0] not in self.breaker_layers]
        layers: dict[str, list[dict]] = {}
        for name, row in self._rows().items():
            row["state"] = entry_state(row["kind"], row["enabled"], row["implicit"])
            row["gauge"] = self._gauge(name) or ((gauges or {}).get(name) if row["enabled"] is not False else None)
            last = self.last.get(name)
            if last is None:
                old = [t for t in legacy if t["name"] == name]
                last = old[-1] if old else None
            row["last"] = None if last is None else {k: last.get(k) for k in (
                "ts", "state", "action", "value", "limit", "unit", "scope", "detail", "stop_reason",
                "legacy_event", "legacy")}
            row["tripped"] = bool(last and last.get("state") == "trip")
            layers.setdefault(str(row["layer"] or "?"), []).append(row)
        try:
            from yeto.breakers.spec import LAYERS  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            LAYERS = ("run", "launch", "trainer", "sampler", "harness", "pool", "monitor")
        order = [x for x in LAYERS if x in layers] + sorted(x for x in layers if x not in LAYERS)
        out_layers = []
        for layer in order:
            rows = sorted(layers[layer], key=lambda r: (
                STATES.index(r["state"]) >= 2, -(((r["gauge"] or {}).get("ratio")) or -1.0) - (1.0 if r["tripped"] else 0),
                r["name"]))
            counts = {s: sum(1 for r in rows if r["state"] == s) for s in STATES}
            out_layers.append({"layer": layer, "rows": rows, "counts": counts})
        trips = sorted(list(self.trips) + legacy, key=lambda t: t["ts"] or 0, reverse=True)
        doc = self.resolved or {}
        counts = {s: sum(x["counts"][s] for x in out_layers) for s in STATES}
        return {"layers": out_layers, "counts": counts,
                "trips": [{k: v for k, v in t.items() if k != "scope_key"} for t in trips[:TRIPS_VIEW]],
                "trip_total": len(self.trips) + len(legacy),
                "findings": [f for f in doc.get("findings") or () if isinstance(f, dict)],
                "conflicts": [c for c in doc.get("conflicts") or () if isinstance(c, dict)],
                "switches": doc.get("switches") if isinstance(doc.get("switches"), dict) else {},
                "resolved_from": self.resolved_from, "has_section": doc.get("has_section"),
                "status_scopes": len({k[1] for k in self.status})}

    def alert_rows(self, now: float) -> list[dict]:
        """The newest ``rl_breaker`` of each (entry, scope) that still calls for an
        alert: a stop / halt trip, any other trip within ``RECENT_TRIP_S``, a warn, a
        config mismatch or a shadow trip; a clear ends it."""
        out = []
        for row in self.latest.values():
            state = row.get("state")
            if state in (None, "clear"):
                continue
            if state != "trip" or row.get("action") not in STOP_ACTIONS:
                if row.get("ts") is not None and now - row["ts"] > RECENT_TRIP_S:
                    continue
            out.append(row)
        return sorted(out, key=lambda r: r.get("ts") or 0)

    def section_param(self, name: str, param: str) -> Any:
        """``param`` of entry ``name`` when the run config's ``breakers:`` section set it,
        else None: without a section the page keeps its own defaults (no behavior change)."""
        for e in (self.resolved or {}).get("entries") or ():
            if isinstance(e, dict) and e.get("name") == name:
                src = (e.get("sources") or {}).get(param) or {}
                if src.get("kind") == "breakers":
                    return (e.get("params") or {}).get(param)
        return None

    def disabled(self, name: str) -> bool:
        for e in (self.resolved or {}).get("entries") or ():
            if isinstance(e, dict) and e.get("name") == name:
                return e.get("enabled") is False
        return False


class ResolvedFile:
    """Feeds ``breakers.resolved.json`` to a reducer: the first local candidate that
    exists (read again when its mtime changes), else the run store's copy (read at most
    every ``STORE_RESOLVED_S``)."""

    def __init__(self, candidates: list[str | os.PathLike], objects: Any = None, *,
                 clock: Callable[[], float] | None = None, interval_s: float = STORE_RESOLVED_S) -> None:
        import time  # noqa: PLC0415

        self.candidates = [Path(os.path.expanduser(str(c))) for c in candidates]
        self.objects = objects
        self.clock = clock or time.monotonic
        self.interval_s = float(interval_s)
        self._mtime: tuple[str, float] | None = None
        self._next_store: float | None = None
        self._store_bytes: bytes | None = None

    def pump(self, reducer) -> bool:
        for p in self.candidates:
            try:
                st = p.stat()
            except OSError:
                continue
            key = (str(p), st.st_mtime)
            if key == self._mtime:
                return False
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                return False
            self._mtime = key
            return reducer.load_resolved(doc, str(p))
        if self.objects is None:
            return False
        now = self.clock()
        if self._next_store is not None and now < self._next_store:
            return False
        self._next_store = now + self.interval_s
        try:
            data = self.objects.read(STORE_RESOLVED_KEY)
        except Exception:  # noqa: BLE001 - the store's own status reports read failures
            return False
        if not data or data == self._store_bytes:
            return False
        try:
            doc = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return False
        self._store_bytes = data
        return reducer.load_resolved(doc, "store " + STORE_RESOLVED_KEY)


def resolved_candidates(paths: list[str], *, run: str | None = None, up_state: str | None = None,
                        max_up: int = 3) -> list[str]:
    """Where a run's ``breakers.resolved.json`` may be: next to ``--up-state``, in the
    ``run up`` state directory of ``--run``, and in a tape's directory or its parents."""
    out: list[str] = []
    if up_state:
        out.append(str(Path(os.path.expanduser(up_state)).parent / RESOLVED_FILE))
    if run:
        root = Path(os.environ.get("YETO_STATE_DIR") or "~/.yeto/rl-up").expanduser()
        out.append(str(root / run / RESOLVED_FILE))
    for p in paths:
        base = Path(os.path.expanduser(p))
        for d in [base, *list(base.parents)[:max_up]]:
            c = str(d / RESOLVED_FILE)
            if c not in out:
                out.append(c)
    return out


def summary(doc: dict | None, records: list[dict]) -> dict:
    """``yeto run status``: the resolved settings in short and the newest trips, from
    the resolved file and the run's breaker events (``run up`` tape, trainer log)."""
    st = BreakerState()
    if doc is not None:
        st.load_resolved(doc, "")
    for r in records:
        if isinstance(r, dict):
            st.fold(r, None, _f(r.get("time_unix")))
    view = st.view()
    if view is None:
        return {"resolved": doc is not None, "entries": None, "trips": []}
    implicit = [r["name"] for x in view["layers"] for r in x["rows"] if r["state"] == "implicit"]
    off = [r["name"] for x in view["layers"] for r in x["rows"] if r["state"] == "off"]
    near = [{"name": r["name"], "ratio": r["gauge"]["ratio"], "scope": r["gauge"]["scope"]}
            for x in view["layers"] for r in x["rows"]
            if r["gauge"] and (r["gauge"]["ratio"] or 0) >= 0.7]
    return {"resolved": doc is not None, "has_section": view["has_section"], "entries": view["counts"],
            "implicit": implicit, "off": off, "near_limit": near,
            "findings": [{k: f.get(k) for k in ("rule", "level", "message")} for f in view["findings"]
                         if f.get("level") in ("error", "warn")],
            "conflicts": [c.get("message") for c in view["conflicts"]],
            "trips": [{k: t.get(k) for k in ("ts", "name", "state", "action", "scope", "stop_reason", "detail")}
                      for t in view["trips"][:5]]}
