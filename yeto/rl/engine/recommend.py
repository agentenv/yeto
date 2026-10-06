"""D1 (tasks 6.1-6.3): shadow load attribution, gain prediction, semi-automatic
recommendations and re-validation before a human-approved request.

Nothing here executes a transition on its own.  ``approve`` re-validates a
recommendation and then calls the controller's ordinary ``request`` entry --
the same transaction path as a manual request (no side channel).

Gain model (design D1/D2):
* serial timeline: a window's wall time is ``rollout(gpu busy + tool wait +
  tail wait) + publish block + train/other``; only the GPU-busy rollout part
  scales with the engine count, tool wait and tail are not double counted and
  do not shrink with more engines.  gain = busy * (1 - src/tgt * 1/eff).
* certified overlap timeline: rollout and train run concurrently; the cycle is
  the critical path ``max(R, T) + publish``; gain = old critical path - new
  critical path (0 when train is the critical path).  Needs ``train_fraction``;
  without it the gain is unknown and nothing is recommended.
* cost: per (profile_hash, source, target) upper bounds from 5.7.  Unknown
  cost -> no recommendation.  Recommend only if
  ``gain_lower * horizon > cost_upper + safety_margin``.
"""
from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .controller import Rejected


class RecommendMode(str, enum.Enum):
    DISABLED = "disabled"
    MANUAL = "manual"
    RECOMMEND = "recommend"
    AUTO = "auto"  # D2 6.4: driven only by ``auto.AutoController``; default off


SERIAL = "serial"
OVERLAP = "certified-overlap"

TOOL_WAIT, TAIL, PUBLISH, GPU_SAT, BALANCED = (
    "tool_wait", "long_tail", "publish_block", "gpu_saturation", "balanced")
HOLD = "hold-current-config"


@dataclass(frozen=True)
class LoadWindow:
    start_s: float
    end_s: float
    profile_hash: str | None
    epoch: int
    gpu_busy_fraction: float
    tool_wait_fraction: float
    tail_wait_fraction: float
    publish_block_fraction: float
    queued: int = 0
    active: int = 0
    ready_groups: int = 0
    consume_rate: float = 0.0
    policy_age: float = 0.0
    train_fraction: float | None = None  # needed by the overlap model only

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


_ALIASES = {"start_s": ("start_s", "window_start", "start"),
            "end_s": ("end_s", "window_end", "end")}


def to_load_window(obj: Any) -> LoadWindow:
    """Adapter from any duck-typed summary (e.g. timeline ``LoadSummary``)."""
    if isinstance(obj, LoadWindow):
        return obj
    get = (lambda k, d=None: obj.get(k, d)) if isinstance(obj, Mapping) else \
        (lambda k, d=None: getattr(obj, k, d))
    kw: dict[str, Any] = {}
    for name, keys in _ALIASES.items():
        val = next((get(k) for k in keys if get(k) is not None), None)
        if val is None:
            raise ValueError(f"load window lacks {name}")
        kw[name] = float(val)
    for name in LoadWindow.__dataclass_fields__:
        if name in kw:
            continue
        val = get(name)
        if val is not None:
            kw[name] = val
    # timeline.LoadSummary uses None for "no classifiable sample": no evidence of waiting
    for name in ("tool_wait_fraction", "tail_wait_fraction"):
        if name not in kw and _has(obj, name):
            kw[name] = 0.0
    for need in ("epoch", "gpu_busy_fraction", "tool_wait_fraction",
                 "tail_wait_fraction", "publish_block_fraction"):
        if need not in kw:
            raise ValueError(f"load window lacks {need}")
    kw.setdefault("profile_hash", None)
    return LoadWindow(**kw)


def _has(obj: Any, name: str) -> bool:
    return name in obj if isinstance(obj, Mapping) else hasattr(obj, name)


def attribute(w: LoadWindow, *, saturation: float = 0.85) -> dict[str, Any]:
    """Separate attribution of one window; ``dominant`` names the bottleneck."""
    parts = {TOOL_WAIT: w.tool_wait_fraction, TAIL: w.tail_wait_fraction,
             PUBLISH: w.publish_block_fraction, GPU_SAT: w.gpu_busy_fraction}
    if w.gpu_busy_fraction >= saturation and w.queued > 0:
        dominant = GPU_SAT
    else:
        waits = {k: parts[k] for k in (TOOL_WAIT, TAIL, PUBLISH)}
        k = max(waits, key=waits.get)
        dominant = k if waits[k] >= 0.2 else BALANCED
    return {"parts": parts, "dominant": dominant}


@dataclass(frozen=True)
class CandidateEdge:
    source: str
    target: str
    source_engines: int
    target_engines: int


@dataclass(frozen=True)
class EdgeCost:
    """5.7 output: per profile/fingerprint/direction cost and recovery bounds (s)."""
    profile_hash: str
    source: str
    target: str
    cost_lower_s: float
    cost_upper_s: float
    recovery_upper_s: float = 0.0


@dataclass(frozen=True)
class EdgeGain:
    source: str
    target: str
    timeline: str
    gain_lower: float | None  # fraction of wall time saved
    gain_upper: float | None
    reason: str | None = None


def _window_gain(w: LoadWindow, ratio: float, timeline: str) -> float | None:
    busy = w.gpu_busy_fraction
    new_busy = busy * ratio
    if timeline == SERIAL:
        return busy - new_busy
    if w.train_fraction is None:
        return None
    r_old = busy + w.tool_wait_fraction + w.tail_wait_fraction
    r_new = new_busy + w.tool_wait_fraction + w.tail_wait_fraction
    t = w.train_fraction
    return max(r_old, t) - max(r_new, t)


def predict_gain(windows: Sequence[LoadWindow], edge: CandidateEdge, timeline: str, *,
                 efficiency_lower: float = 0.7) -> EdgeGain:
    if timeline not in (SERIAL, OVERLAP):
        raise ValueError(f"unknown timeline {timeline!r}")
    if not windows:
        return EdgeGain(edge.source, edge.target, timeline, None, None, "no load windows")
    if edge.target_engines < 1 or edge.source_engines < 1:
        return EdgeGain(edge.source, edge.target, timeline, None, None, "no engines")
    ideal = edge.source_engines / edge.target_engines
    # lower bound: imperfect scaling when growing, perfect slowdown when shrinking
    pess = ideal / efficiency_lower
    lows, highs = [], []
    for w in windows:
        lo, hi = _window_gain(w, pess, timeline), _window_gain(w, ideal, timeline)
        if lo is None or hi is None:
            return EdgeGain(edge.source, edge.target, timeline, None, None,
                            "overlap model needs train_fraction")
        lows.append(min(lo, hi))
        highs.append(max(lo, hi))
    return EdgeGain(edge.source, edge.target, timeline, min(lows), max(highs))


@dataclass(frozen=True)
class Recommendation:
    request_id: str
    source: str
    target: str
    expected_epoch: int
    profile_hash: str | None
    timeline: str
    gain_lower: float | None
    gain_upper: float | None
    cost_lower: float | None
    cost_upper: float | None
    issued_at: float
    expires_at: float
    reason: str
    rejection_reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return self.target != self.source and self.rejection_reason is None


def summarize(windows: Sequence[LoadWindow]) -> dict[str, Any]:
    if not windows:
        return {"n": 0}
    n = len(windows)
    mean = lambda k: sum(getattr(w, k) for w in windows) / n  # noqa: E731
    dom: dict[str, int] = {}
    for w in windows:
        d = attribute(w)["dominant"]
        dom[d] = dom.get(d, 0) + 1
    return {"n": n, "start_s": windows[0].start_s, "end_s": windows[-1].end_s,
            "gpu_busy": mean("gpu_busy_fraction"), "tool_wait": mean("tool_wait_fraction"),
            "tail_wait": mean("tail_wait_fraction"),
            "publish_block": mean("publish_block_fraction"), "dominant": dom}


_EVALUATING = (RecommendMode.RECOMMEND, RecommendMode.AUTO)


def candidate_edges_from_attestation(attestation: Any, configs: Mapping[str, Any], *,
                                     source: str | None = None) -> list[CandidateEdge]:
    """Candidate edges come ONLY from the attestation's certified set (a trainer
    edge that is not certified can never be chosen).  ``configs`` maps a config
    name to a ResourceConfig-like object (``rollout``, ``rollout_engine_gpus``)."""
    out: dict[tuple[str, str], CandidateEdge] = {}
    for edge in sorted(getattr(attestation, "certified_edges", ()) or ()):
        src, dst = edge[0], edge[1]
        if (source is not None and src != source) or src == dst:
            continue
        a, b = configs.get(src), configs.get(dst)
        if a is None or b is None:
            continue
        n = lambda c: int(c.rollout) // max(1, int(getattr(c, "rollout_engine_gpus", 1) or 1))  # noqa: E731
        out[(src, dst)] = CandidateEdge(src, dst, n(a), n(b))
    return list(out.values())


EDGE_COST_FIELDS = ("profile_hash", "source", "target", "cost_lower_s", "cost_upper_s",
                    "recovery_upper_s", "n", "provenance")


def edge_costs_from_table(table: Any) -> dict[tuple[str, str, str], EdgeCost]:
    """5.7 cost table -> ``{(profile_hash, source, target): EdgeCost}``.

    ``table`` is a path to a JSON file holding a list of
    ``{profile_hash, source, target, cost_lower_s, cost_upper_s, recovery_upper_s, n,
    provenance}`` rows, or that list itself.  A missing file / None gives ``{}``,
    which makes every evaluation hold.  Rows with ``n < 1``, no provenance or
    ``cost_lower_s > cost_upper_s`` are refused (ValueError)."""
    import json
    from pathlib import Path

    if table is None:
        return {}
    if isinstance(table, (str, Path)):
        path = Path(table).expanduser()
        if not path.exists():
            return {}
        table = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], EdgeCost] = {}
    for row in table:
        missing = [k for k in EDGE_COST_FIELDS if k not in row]
        if missing:
            raise ValueError(f"cost row lacks {missing}")
        if int(row["n"]) < 1 or not row["provenance"]:
            raise ValueError("cost row needs n >= 1 measurements and a provenance")
        lo, hi = float(row["cost_lower_s"]), float(row["cost_upper_s"])
        if lo > hi or lo < 0:
            raise ValueError("cost row bounds are inconsistent")
        key = (str(row["profile_hash"]), str(row["source"]), str(row["target"]))
        if key in out:
            raise ValueError(f"duplicate cost row {key}")
        out[key] = EdgeCost(*key, lo, hi, float(row["recovery_upper_s"]))
    return out


@dataclass
class Recommender:
    mode: RecommendMode = RecommendMode.DISABLED
    timeline: str = SERIAL
    horizon_s: float = 3600.0
    safety_margin_s: float = 60.0
    ttl_s: float = 300.0
    min_windows: int = 3
    efficiency_lower: float = 0.7
    clock: Callable[[], float] = time.time

    def _net(self, windows: Sequence[LoadWindow], edge: CandidateEdge, profile_hash: str | None,
             costs: Mapping[tuple[str, str, str], EdgeCost]) -> tuple[EdgeGain, EdgeCost | None, str | None]:
        gain = predict_gain(windows, edge, self.timeline, efficiency_lower=self.efficiency_lower)
        if gain.gain_lower is None:
            return gain, None, gain.reason
        cost = costs.get((profile_hash, edge.source, edge.target)) if profile_hash else None
        if cost is None:
            return gain, None, "unknown transition cost for this profile/edge"
        span = sum(w.duration_s for w in windows)
        if span <= 0:
            return gain, cost, "empty window span"
        benefit = gain.gain_lower * self.horizon_s
        need = cost.cost_upper_s + cost.recovery_upper_s + self.safety_margin_s
        if benefit <= need:
            return gain, cost, f"no net gain: {benefit:.1f}s <= cost bound {need:.1f}s"
        return gain, cost, None

    def recommend(self, controller: Any, windows: Iterable[Any],
                  candidates: Iterable[CandidateEdge],
                  costs: Mapping[tuple[str, str, str], EdgeCost]) -> Recommendation | None:
        """Shadow evaluation.  Returns None unless mode is RECOMMEND.  Never executes."""
        if self.mode not in _EVALUATING:
            return None
        epochs = controller.journal.epochs
        source, epoch = epochs.config_id, int(epochs.config_epoch)
        phash = getattr(controller.profile, "contract_hash", None)
        ws = [w for w in map(to_load_window, windows)
              if w.epoch == epoch and w.profile_hash == phash]
        now = self.clock()
        evidence = summarize(ws)
        hold = lambda why: Recommendation(  # noqa: E731
            f"rec-hold-{epoch}-{int(now)}", source, source, epoch, phash, self.timeline,
            None, None, None, None, now, now + self.ttl_s, HOLD, why, evidence)
        if len(ws) < self.min_windows:
            return hold(f"only {len(ws)} valid windows (< {self.min_windows})")
        best, reasons = None, []
        for edge in candidates:
            if edge.source != source:
                continue
            gain, cost, why = self._net(ws, edge, phash, costs)
            if why is not None:
                reasons.append(f"{edge.source}->{edge.target}: {why}")
                continue
            score = gain.gain_lower * self.horizon_s - cost.cost_upper_s
            if best is None or score > best[0]:
                best = (score, edge, gain, cost)
        if best is None:
            return hold("; ".join(reasons) or "no candidate edge")
        _, edge, gain, cost = best
        digest = hashlib.sha256(f"{epoch}|{phash}|{edge.source}|{edge.target}|{now}".encode()
                                ).hexdigest()[:12]
        return Recommendation(
            f"rec-{epoch}-{digest}", edge.source, edge.target, epoch, phash, self.timeline,
            gain.gain_lower, gain.gain_upper, cost.cost_lower_s, cost.cost_upper_s,
            now, now + self.ttl_s, f"dominant load {max(evidence['dominant'], key=evidence['dominant'].get)}",
            None, evidence)

    def revalidate(self, rec: Recommendation, controller: Any, windows: Iterable[Any],
                   candidates: Iterable[CandidateEdge],
                   costs: Mapping[tuple[str, str, str], EdgeCost], *, deadline_s: float) -> None:
        """6.3: raise ``Rejected`` with the reason if anything changed.  The
        target is never replaced and an expired recommendation is never renewed."""
        if self.mode not in _EVALUATING:
            raise Rejected(f"recommend mode is {self.mode.value}")
        if not rec.actionable:
            raise Rejected(f"recommendation is not actionable: {rec.rejection_reason}")
        if self.clock() >= rec.expires_at:
            raise Rejected(f"recommendation {rec.request_id} expired at {rec.expires_at}")
        epochs = controller.journal.epochs
        if int(epochs.config_epoch) != rec.expected_epoch:
            raise Rejected(f"epoch changed: expected {rec.expected_epoch}, "
                           f"current {epochs.config_epoch}")
        if epochs.config_id != rec.source:
            raise Rejected(f"source changed: expected {rec.source}, current {epochs.config_id}")
        phash = getattr(controller.profile, "contract_hash", None)
        if phash != rec.profile_hash:
            raise Rejected("execution profile hash changed")
        edge = next((e for e in candidates
                     if e.source == rec.source and e.target == rec.target), None)
        if edge is None:
            raise Rejected(f"edge {rec.source}->{rec.target} is no longer a candidate")
        ws = [w for w in map(to_load_window, windows)
              if w.epoch == rec.expected_epoch and w.profile_hash == phash]
        if len(ws) < self.min_windows:
            raise Rejected(f"only {len(ws)} valid windows for re-validation")
        _, _, why = self._net(ws, edge, phash, costs)
        if why is not None:
            raise Rejected(f"load no longer justifies the edge: {why}")
        # pause guard, certification and capacity: the controller's own side-effect-free plan
        controller.plan(rec.target, rec.expected_epoch, deadline_s=deadline_s)

    def approve(self, rec: Recommendation, controller: Any, windows: Iterable[Any],
                candidates: Iterable[CandidateEdge],
                costs: Mapping[tuple[str, str, str], EdgeCost], *,
                deadline_s: float) -> dict[str, Any]:
        """Human approval: re-validate, then the SAME ``controller.request`` entry."""
        windows, candidates = list(windows), list(candidates)
        self.revalidate(rec, controller, windows, candidates, costs, deadline_s=deadline_s)
        return controller.request(rec.request_id, rec.target, rec.expected_epoch, deadline_s)


def recommendation_dict(rec: Recommendation) -> dict[str, Any]:
    return asdict(rec)
