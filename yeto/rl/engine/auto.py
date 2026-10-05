"""D2 task 6.4: default-off automatic policy on top of the D1 recommender.

Every trigger goes through the very same path as a human approval:
``Recommender.approve`` -> ``Recommender.revalidate`` (expiry, epoch, profile,
load, ``controller.plan`` pause guard / certification / capacity) ->
``controller.request``.  There is no side channel.

Guards (design D2), all configurable, no paper numbers hard-coded:
* the controller's ``recommend_mode`` must be ``auto`` (default ``disabled``);
* the last ``k_windows`` valid windows must ALL show net gain for the edge
  (``Recommender._net`` takes the minimum gain over the windows), so an
  oscillating load holds;
* ``gain_lower(H) > cost_upper + recovery_upper + safety_margin`` with
  ``H = min(horizon_s, remaining training budget)``; unknown budget holds;
* min dwell since the last config change, cooldown after any auto attempt,
  at most ``max_switches`` per ``switch_window_s``;
* hold while a transaction/recovery is pending, while finalizing, or when any
  of the K windows is tool-wait dominated (tool-heavy);
* candidate edges are only the attestation's certified edges;
* any failed auto switch disables auto and falls back to ``fallback_mode``
  (manual by default); manual requests stay available.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping

from .controller import SUCCEEDED, Rejected
from .recommend import (TOOL_WAIT, CandidateEdge, EdgeCost, Recommender, RecommendMode,
                        attribute, to_load_window)


@dataclass(frozen=True)
class AutoPolicy:
    k_windows: int = 6
    safety_margin_s: float = 120.0
    horizon_s: float = 3600.0
    min_dwell_s: float = 1800.0
    cooldown_s: float = 1800.0
    max_switches: int = 2
    switch_window_s: float = 3600.0
    deadline_s: float = 600.0
    fallback_mode: str = RecommendMode.MANUAL.value

    def __post_init__(self) -> None:
        if self.k_windows < 1 or self.max_switches < 0 or self.horizon_s <= 0:
            raise ValueError("invalid auto policy")
        if self.fallback_mode not in (RecommendMode.MANUAL.value, RecommendMode.RECOMMEND.value,
                                      RecommendMode.DISABLED.value):
            raise ValueError("fallback_mode must be manual/recommend/disabled")


@dataclass
class AutoController:
    recommender: Recommender
    policy: AutoPolicy = field(default_factory=AutoPolicy)
    clock: Callable[[], float] = time.time
    last_change_at: float | None = None  # wall time of the last config change seen
    last_attempt_at: float | None = None
    switches: list[float] = field(default_factory=list)
    pending: str | None = None  # request_id of our in-flight auto request
    decisions: list[dict[str, Any]] = field(default_factory=list)
    _seen_epoch: int | None = None
    _seq: int = 0

    def _hold(self, why: str) -> dict[str, Any]:
        d = {"action": "hold", "reason": why, "t": self.clock()}
        self.decisions.append(d)
        return d

    def _disable(self, controller: Any, why: str) -> dict[str, Any]:
        controller.set_recommend_mode(self.policy.fallback_mode, reason=f"auto disabled: {why}")
        d = {"action": "disabled", "reason": why, "fallback": self.policy.fallback_mode,
             "t": self.clock()}
        self.decisions.append(d)
        return d

    def step(self, controller: Any, windows: Iterable[Any], candidates: Iterable[CandidateEdge],
             costs: Mapping[tuple[str, str, str], EdgeCost], *,
             remaining_budget_s: float | None, tool_heavy: bool = False) -> dict[str, Any]:
        """Call at safe points.  Returns the decision (hold / disabled / requested)."""
        now = self.clock()
        epoch = int(controller.journal.epochs.config_epoch)
        if self._seen_epoch is None or epoch != self._seen_epoch:
            if self._seen_epoch is not None:
                self.last_change_at = now
            self._seen_epoch = epoch
        # outcome of our own previous switch (checked even if auto was turned off meanwhile)
        if self.pending is not None:
            st = controller.status(self.pending)
            if not st.get("terminal"):
                return self._hold(f"auto request {self.pending} in flight")
            rid, self.pending = self.pending, None
            if st.get("phase") != SUCCEEDED:
                if getattr(controller, "recommend_mode", None) == RecommendMode.AUTO.value:
                    return self._disable(controller, f"switch {rid} ended {st.get('phase')}")
                return self._hold(f"switch {rid} ended {st.get('phase')}")
            self.last_change_at = now
        if getattr(controller, "recommend_mode", None) != RecommendMode.AUTO.value:
            return self._hold("auto mode is off")
        if controller.has_pending():
            return self._hold("a transaction or recovery is pending")
        if getattr(controller, "finalizing", None) is not None:
            return self._hold("finalization is near/in progress")
        p = self.policy
        if remaining_budget_s is None or remaining_budget_s <= 0:
            return self._hold("remaining training budget unknown or exhausted")
        if self.last_change_at is not None and now - self.last_change_at < p.min_dwell_s:
            return self._hold("minimum dwell not reached")
        if self.last_attempt_at is not None and now - self.last_attempt_at < p.cooldown_s:
            return self._hold("cooldown")
        self.switches = [t for t in self.switches if now - t < p.switch_window_s]
        if len(self.switches) >= p.max_switches:
            return self._hold("switch rate limit")
        phash = getattr(controller.profile, "contract_hash", None)
        ws = [w for w in map(to_load_window, windows)
              if w.epoch == epoch and w.profile_hash == phash]
        ws = ws[-p.k_windows:]
        if len(ws) < p.k_windows:
            return self._hold(f"only {len(ws)} of {p.k_windows} consecutive valid windows")
        if tool_heavy or any(attribute(w)["dominant"] == TOOL_WAIT for w in ws):
            return self._hold("tool-heavy load")
        horizon = min(p.horizon_s, float(remaining_budget_s))
        rec_eng = replace(self.recommender, mode=RecommendMode.AUTO, horizon_s=horizon,
                          safety_margin_s=p.safety_margin_s, min_windows=p.k_windows,
                          clock=self.clock)
        candidates = list(candidates)
        rec = rec_eng.recommend(controller, ws, candidates, costs)
        if rec is None or not rec.actionable:
            return self._hold(rec.rejection_reason if rec else "no recommendation")
        self._seq += 1
        rec = replace(rec, request_id=f"auto-{rec.request_id}-{self._seq}")
        self.last_attempt_at = now
        try:
            rec_eng.revalidate(rec, controller, ws, candidates, costs, deadline_s=p.deadline_s)
        except Rejected as exc:
            return self._hold(f"revalidation refused: {exc}")
        try:
            answer = controller.request(rec.request_id, rec.target, rec.expected_epoch,
                                        p.deadline_s)
        except Exception as exc:  # noqa: BLE001 -- any failure to start disables auto
            return self._disable(controller, f"request failed: {exc}")
        self.pending = rec.request_id
        self.switches.append(now)
        d = {"action": "requested", "request_id": rec.request_id, "source": rec.source,
             "target": rec.target, "gain_lower": rec.gain_lower, "cost_upper": rec.cost_upper,
             "horizon_s": horizon, "answer": answer, "t": now}
        self.decisions.append(d)
        return d
