"""D1/D2 wiring into the driver safe point (rl-infra-spec 6.5, branch d2-wire).

``ElasticHook.at_safe_point(driver, rollout_id)`` is called by
``IslandDriver.safe_point`` after the command inbox is polled and before the
controller runs a pending transaction.  It does nothing unless the driver
observes (``observe=True``) and the controller's ``recommend_mode`` is
``recommend`` or ``auto``:

* windows: ``timeline.load_windows(events, window_s)`` over the driver's own
  observe events (read incrementally from the event tape, or an injected source);
* remaining budget: ``(total_rounds - rollout_id) * mean(recent round wall time)``,
  unknown (None -> auto holds) until two safe points were seen;
* tool_heavy: any of the last ``k`` windows of the current (profile, epoch) is
  tool-wait dominated;
* candidates: ONLY ``attestation.certified_edges`` (optionally intersected with
  the run's declared edges) -- a declared but uncertified edge is never chosen;
* costs: ``edge_costs_from_table(edge_costs_path)``; None/missing -> {} -> hold.

``recommend`` emits ``rl_elastic_recommendation`` (and journals actionable ones);
nothing is executed.  ``auto`` calls ``AutoController.step`` whose only effect is
``controller.request`` -- the same transaction entry as a manual request -- which
the driver then executes in this very safe point.  AutoController dwell /
cooldown / switch history / pending request are journaled (kind ``auto_state``)
and restored after a restart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .auto import AutoController
from .recommend import (TOOL_WAIT, Recommender, RecommendMode, attribute,
                        candidate_edges_from_attestation, edge_costs_from_table,
                        recommendation_dict, to_load_window)
from .timeline import load_windows

_OBSERVED = ("rl_timeline_span", "rl_load_sample", "rl_readiness", "rl_round_labels")
AUTO_STATE_KIND = "auto_state"
_STATE_FIELDS = ("last_change_at", "last_attempt_at", "switches", "pending", "_seen_epoch", "_seq")


def auto_state(auto: AutoController) -> dict[str, Any]:
    return {k: getattr(auto, k) for k in _STATE_FIELDS}


def restore_auto_state(auto: AutoController, records: Iterable[Mapping[str, Any]]) -> bool:
    """Apply the last journaled ``auto_state`` record; False if there is none."""
    last = None
    for r in records:
        if r.get("kind") == AUTO_STATE_KIND:
            last = r
    if last is None:
        return False
    for k in _STATE_FIELDS:
        if k in last["state"]:
            v = last["state"][k]
            setattr(auto, k, list(v) if k == "switches" else v)
    return True


@dataclass
class ElasticHook:
    configs: Mapping[str, Any]
    window_s: float
    total_rounds: int | None = None
    edge_costs_path: str | None = None
    declared_edges: frozenset[tuple[str, str]] | None = None
    recommender: Recommender = field(default_factory=Recommender)
    auto: AutoController | None = None
    events_source: Callable[[], Iterable[Mapping[str, Any]]] | None = None
    recent_rounds: int = 5
    decisions: list[dict[str, Any]] = field(default_factory=list)
    _marks: list[tuple[int, float]] = field(default_factory=list)
    _buf: list[Mapping[str, Any]] = field(default_factory=list)
    _offset: int = 0
    _restored: bool = False
    _fed: bool = False

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")

    # ---------------------------------------------------------------- inputs
    def feed(self, event: Mapping[str, Any]) -> None:
        """In-memory mirror of the driver's own emits (``IslandDriver.emit``): the
        hook then never depends on the tape path (Miles ``_append_rl_event``)."""
        self._fed = True
        if event.get("event") in _OBSERVED:
            self._buf.append(dict(event))

    def _events(self, driver: Any) -> list[Mapping[str, Any]]:
        if self.events_source is not None:
            return list(self.events_source())
        if self._fed:
            return list(self._buf)
        path = getattr(driver.events, "path", None)
        if path is None or not Path(path).exists():
            return list(self._buf)
        with open(path, "rb") as fh:  # incremental: only new complete lines
            fh.seek(self._offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n") + 1
        self._offset += end
        for line in chunk[:end].splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("event") in _OBSERVED:
                self._buf.append(ev)
        return list(self._buf)

    def remaining_budget_s(self, rollout_id: int) -> float | None:
        marks = self._marks[-(self.recent_rounds + 1):]
        if self.total_rounds is None or len(marks) < 2:
            return None
        (r0, t0), (r1, t1) = marks[0], marks[-1]
        if r1 <= r0 or t1 <= t0:
            return None
        return max(0, int(self.total_rounds) - int(rollout_id)) * (t1 - t0) / (r1 - r0)

    def candidates(self, controller: Any) -> list:
        source = controller.journal.epochs.config_id
        edges = candidate_edges_from_attestation(controller.attestation, self.configs,
                                                 source=source)
        if self.declared_edges is not None:
            edges = [e for e in edges if (e.source, e.target) in self.declared_edges]
        return edges

    # ---------------------------------------------------------------- persistence
    def _restore(self, controller: Any) -> None:
        if self._restored or self.auto is None:
            return
        self._restored = True
        journal = getattr(controller, "journal", None)
        records = getattr(journal, "records", None)
        if records is not None and restore_auto_state(self.auto, records):
            self.decisions.append({"action": "restored", "state": auto_state(self.auto)})

    def _persist(self, controller: Any, before: dict[str, Any]) -> None:
        now = auto_state(self.auto)
        rec = getattr(controller, "_record", None)
        if now != before and callable(rec):
            rec(AUTO_STATE_KIND, tx_id=None, state=now)

    # ---------------------------------------------------------------- entry
    def at_safe_point(self, driver: Any, rollout_id: int) -> dict[str, Any] | None:
        controller = driver.controller
        self._restore(controller)
        mode = getattr(controller, "recommend_mode", RecommendMode.DISABLED.value)
        if not driver.observe or mode not in (RecommendMode.RECOMMEND.value,
                                              RecommendMode.AUTO.value):
            if self.auto is not None and self.auto.pending is not None:
                before = auto_state(self.auto)  # still settle our in-flight request
                self.auto.step(controller, [], [], {}, remaining_budget_s=None)
                self._persist(controller, before)
            return None
        # round wall time is sampled only while evaluating, so a disabled/manual
        # run never reads the driver clock here (event-identical to no hook)
        self._marks.append((int(rollout_id), float(driver.clock())))
        windows = load_windows(self._events(driver), self.window_s)
        epoch = int(controller.journal.epochs.config_epoch)
        phash = getattr(controller.profile, "contract_hash", None)
        current = [w for w in windows if w.epoch == epoch and w.profile_hash == phash]
        k = self.auto.policy.k_windows if self.auto is not None else self.recommender.min_windows
        tool_heavy = any(attribute(to_load_window(w))["dominant"] == TOOL_WAIT
                         for w in current[-k:])
        candidates = self.candidates(controller)
        costs = edge_costs_from_table(self.edge_costs_path)
        budget = self.remaining_budget_s(rollout_id)
        common = {"rollout_id": rollout_id, "mode": mode, "profile_hash": phash,
                  "config_epoch": epoch, "windows": len(current), "tool_heavy": tool_heavy,
                  "remaining_budget_s": budget,
                  "candidates": [[e.source, e.target] for e in candidates]}
        if mode == RecommendMode.RECOMMEND.value:
            rec = replace(self.recommender, mode=RecommendMode.RECOMMEND).recommend(
                controller, current, candidates, costs)
            body = recommendation_dict(rec) if rec is not None else None
            d = {"action": "recommend", **common, "recommendation": body}
            driver.emit("rl_elastic_recommendation", **d)
            if rec is not None and rec.actionable and callable(getattr(controller, "_record", None)):
                controller._record("recommendation", tx_id=None, recommendation=body)
            self.decisions.append(d)
            return d
        if self.auto is None:
            d = {"action": "hold", "reason": "no AutoController configured", **common}
        elif not getattr(driver.capabilities, "auto_controller", False):
            d = {"action": "hold", "reason": "engine capabilities do not declare auto_controller",
                 **common}
        else:
            before = auto_state(self.auto)
            d = {**self.auto.step(controller, current, candidates, costs,
                                  remaining_budget_s=budget, tool_heavy=tool_heavy), **common}
            self._persist(controller, before)
        driver.emit("rl_elastic_auto", **d)
        self.decisions.append(d)
        return d
