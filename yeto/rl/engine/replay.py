"""rl-infra-spec 6.6 (CPU side): four-scenario trace replay.

Compares, under the SAME update budget (number of rounds) and the SAME pool:

* ``compat_default`` -- the compatible serial baseline, default config, fixed;
* ``fixed_default``  -- default config fixed, in the target ExecutionProfile mode
  (isolates the *mode* change from resize);
* ``best_fixed``     -- best fixed config in the target mode, chosen in hindsight;
* ``dynamic``        -- target mode, starting at the default config, driven by the
  real ``Recommender``/``AutoController`` against a fake controller.

``mode_gain_s = compat_default - fixed_default`` (only the execution mode changed);
``resize_gain_best_fixed_s = fixed_default - best_fixed`` and
``resize_gain_dynamic_s = fixed_default - dynamic`` (only resize, same mode).
No improvement ratio is assumed anywhere; all numbers come from the inputs.

Load model (per round, seconds, measured at the *reference* config):
``gen_s`` GPU-bound generation that scales with rollout engines, ``tool_s`` and
``tail_s`` that do not scale, ``train_s`` that scales with trainer GPUs,
``publish_s``.  Serial round = gen + tool + tail + train + publish; certified
overlap round = max(gen + tool + tail, train) + publish (no fill/drain model).
The pool is fixed: allocated GPU-hours = pool GPUs x wall time for every group.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace as NS
from typing import Any, Iterable, Mapping, Sequence

from .auto import AutoController, AutoPolicy
from .controller import SUCCEEDED, Rejected
from .recommend import (OVERLAP, SERIAL, CandidateEdge, EdgeCost, LoadWindow, Recommender,
                        RecommendMode, to_load_window)

SCENARIOS = ("stable", "changing", "long_tail", "tool_wait")
GROUPS = ("compat_default", "fixed_default", "best_fixed", "dynamic")
RESULT_FIELDS = ("config_path", "wall_s", "gpu_hours", "rounds", "samples_per_s", "wait_s",
                 "tool_wait_s", "tail_wait_s", "publish_s", "switch_block_s", "recovery_s",
                 "switches", "first_step_s")


@dataclass(frozen=True)
class SimConfig:
    name: str
    rollout_engines: int
    trainer_gpus: int


@dataclass(frozen=True)
class RoundLoad:
    gen_s: float
    tool_s: float = 0.0
    tail_s: float = 0.0
    train_s: float = 0.0
    publish_s: float = 0.0


@dataclass(frozen=True)
class Trace:
    scenario: str
    seed: int | None
    reference: str  # config name the per-round loads were measured at
    rounds: tuple[RoundLoad, ...]


# -- trace inputs ----------------------------------------------------------------

def synth_trace(scenario: str, *, rounds: int, seed: int, reference: str, gen_s: float,
                train_s: float, publish_s: float = 0.0, jitter: float = 0.02,
                step_factor: float = 3.0, period: int = 0, tail_prob: float = 0.1,
                tail_s: float = 0.0, tool_s: float = 0.0) -> Trace:
    """Parametric trace for one of the four scenarios (deterministic per seed).

    * stable: ``gen_s``/``train_s`` with relative ``jitter``;
    * changing: generation x ``step_factor`` in alternating phases of ``period``
      rounds (period 0 = a single step at half time; period 1 = oscillation);
    * long_tail: with prob ``tail_prob`` a round carries ``tail_s`` of non-scalable tail;
    * tool_wait: every round carries ``tool_s`` of tool waiting.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = random.Random(seed)
    j = lambda v: max(0.0, v * (1.0 + rng.uniform(-jitter, jitter)))  # noqa: E731
    out = []
    for i in range(rounds):
        g, tl, tw = gen_s, 0.0, 0.0
        if scenario == "changing":
            heavy = (i >= rounds // 2) if period <= 0 else ((i // period) % 2 == 1)
            g = gen_s * (step_factor if heavy else 1.0)
        elif scenario == "long_tail" and rng.random() < tail_prob:
            tl = tail_s
        elif scenario == "tool_wait":
            tw = tool_s
        out.append(RoundLoad(j(g), j(tw), j(tl), j(train_s), publish_s))
    return Trace(scenario, seed, reference, tuple(out))


def trace_from_windows(windows: Iterable[Any], *, reference: str,
                       scenario: str = "replayed") -> Trace:
    """Replay observed windows (``LoadSummary``/``LoadWindow``): one window = one round.
    ``gpu_busy_fraction`` covers trainer+rollout compute, so the rollout part is
    ``gpu_busy - train_fraction`` (needs ``train_fraction``)."""
    out = []
    for w in map(to_load_window, windows):
        d = w.duration_s
        tr = w.train_fraction or 0.0
        out.append(RoundLoad(max(0.0, w.gpu_busy_fraction - tr) * d, w.tool_wait_fraction * d,
                             w.tail_wait_fraction * d, tr * d, w.publish_block_fraction * d))
    return Trace(scenario, None, reference, tuple(out))


def trace_from_events(events: Iterable[Mapping[str, object]], window_s: float, *,
                      reference: str) -> Trace:
    """Replay driver ``observe=True`` events (rl_timeline_span / rl_load_sample /
    rl_readiness) via ``timeline.load_windows``."""
    from .timeline import load_windows
    return trace_from_windows(load_windows(events, window_s), reference=reference)


# -- round-time model ------------------------------------------------------------

@dataclass(frozen=True)
class TimeModel:
    configs: Mapping[str, SimConfig]
    reference: str
    mode: str = SERIAL  # target ExecutionProfile mode for fixed_default/best_fixed/dynamic
    gen_efficiency: float = 1.0  # scaling efficiency when engines grow
    train_efficiency: float = 1.0

    def parts(self, load: RoundLoad, cfg: str) -> tuple[float, float]:
        ref, c = self.configs[self.reference], self.configs[cfg]
        rg = ref.rollout_engines / c.rollout_engines
        rt = ref.trainer_gpus / c.trainer_gpus
        if rg < 1:
            rg /= self.gen_efficiency
        if rt < 1:
            rt /= self.train_efficiency
        return load.gen_s * rg, load.train_s * rt

    def round_s(self, load: RoundLoad, cfg: str, mode: str | None = None) -> float:
        g, t = self.parts(load, cfg)
        r = g + load.tool_s + load.tail_s
        if (mode or self.mode) == OVERLAP:
            return max(r, t) + load.publish_s
        return r + t + load.publish_s

    def window(self, load: RoundLoad, cfg: str, start: float, epoch: int,
               profile_hash: str) -> LoadWindow:
        g, t = self.parts(load, cfg)
        d = self.round_s(load, cfg)
        # gpu_busy_fraction here = rollout (engine-scalable) busy share, which is
        # what recommend.predict_gain scales; see REPLAY-RESULT.md caveat.
        return LoadWindow(start, start + d, profile_hash, epoch, g / d, load.tool_s / d,
                          load.tail_s / d, load.publish_s / d, queued=1 if g / d > 0.85 else 0,
                          train_fraction=t / d)


# -- fake controller for the real AutoController ---------------------------------

class _SimController:
    def __init__(self, source: str, profile_hash: str, certified: set[tuple[str, str]]):
        self.journal = NS(epochs=NS(config_epoch=1, config_id=source))
        self.profile = NS(contract_hash=profile_hash)
        self.recommend_mode = RecommendMode.AUTO.value
        self.finalizing = None
        self.certified = certified
        self._status: dict[str, dict[str, Any]] = {}
        self.inflight: tuple[str, str] | None = None

    def plan(self, target, expected_epoch, *, deadline_s=0.0):
        e = self.journal.epochs
        if expected_epoch != e.config_epoch:
            raise Rejected("epoch changed")
        if (e.config_id, target) not in self.certified:
            raise Rejected("edge is not certified")
        return NS(target=target)

    def request(self, rid, target, epoch, deadline_s):
        self.plan(target, epoch, deadline_s=deadline_s)
        self._status[rid] = {"terminal": False, "phase": "VALIDATING"}
        self.inflight = (rid, target)
        return {"request_id": rid}

    def complete(self) -> None:
        rid, target = self.inflight
        self._status[rid] = {"terminal": True, "phase": SUCCEEDED}
        e = self.journal.epochs
        self.journal.epochs = NS(config_epoch=e.config_epoch + 1, config_id=target)
        self.inflight = None

    def status(self, rid):
        return self._status.get(rid, {"known": False})

    def has_pending(self):
        return any(not s["terminal"] for s in self._status.values())

    def set_recommend_mode(self, mode, *, reason=""):
        self.recommend_mode = mode
        return {"mode": mode}


# -- simulator -------------------------------------------------------------------

@dataclass
class _Acc:
    path: list[str]
    wall: float = 0.0
    tool: float = 0.0
    tail: float = 0.0
    publish: float = 0.0
    block: float = 0.0
    recovery: float = 0.0
    switches: int = 0
    first_step: float = 0.0


def _account(acc: _Acc, model: TimeModel, load: RoundLoad, cfg: str, mode: str) -> float:
    d = model.round_s(load, cfg, mode)
    acc.wall += d
    acc.tool += load.tool_s
    acc.tail += load.tail_s
    acc.publish += load.publish_s
    return d


def _result(acc: _Acc, rounds: int, pool_gpus: int, samples_per_round: float) -> dict[str, Any]:
    wait = acc.tool + acc.tail + acc.publish + acc.block + acc.recovery
    return {"config_path": acc.path, "wall_s": acc.wall, "gpu_hours": pool_gpus * acc.wall / 3600,
            "rounds": rounds, "samples_per_s": rounds * samples_per_round / acc.wall if acc.wall else 0.0,
            "wait_s": wait, "tool_wait_s": acc.tool, "tail_wait_s": acc.tail,
            "publish_s": acc.publish, "switch_block_s": acc.block, "recovery_s": acc.recovery,
            "switches": acc.switches, "first_step_s": acc.first_step}


def simulate_fixed(trace: Trace, model: TimeModel, cfg: str, *, mode: str, pool_gpus: int,
                   samples_per_round: float = 1.0, first_step_s: float = 0.0) -> dict[str, Any]:
    acc = _Acc([cfg], first_step=first_step_s, wall=first_step_s)
    for load in trace.rounds:
        _account(acc, model, load, cfg, mode)
    return _result(acc, len(trace.rounds), pool_gpus, samples_per_round)


def all_pairs(configs: Iterable[str]) -> set[tuple[str, str]]:
    cs = list(configs)
    return {(a, b) for a in cs for b in cs if a != b}


def simulate_dynamic(trace: Trace, model: TimeModel, default: str, *, pool_gpus: int,
                     costs: Mapping[tuple[str, str, str], EdgeCost],
                     true_costs: Mapping[tuple[str, str], tuple[float, float]] | None = None,
                     certified: set[tuple[str, str]] | None = None,
                     policy: AutoPolicy | None = None, profile_hash: str = "replay-profile",
                     efficiency_lower: float = 0.7, samples_per_round: float = 1.0,
                     first_step_s: float = 0.0) -> dict[str, Any]:
    """Target mode, start at ``default``; at each round end (safe point) call the
    real ``AutoController.step``.  A requested switch blocks for the edge's true
    cost (``true_costs[(src, dst)] = (block_s, recovery_s)``, default: the cost
    table's upper bounds) and then completes.  ``costs`` is what the policy sees."""
    certified = all_pairs(model.configs) if certified is None else certified
    ctl = _SimController(default, profile_hash, certified)
    clock = [0.0]
    auto = AutoController(Recommender(timeline=model.mode, efficiency_lower=efficiency_lower),
                          policy or AutoPolicy(), clock=lambda: clock[0])
    by_cfg = {c: model.configs[c] for c in model.configs}
    acc = _Acc([default], first_step=first_step_s, wall=first_step_s)
    clock[0] = acc.wall
    windows: list[LoadWindow] = []
    n = len(trace.rounds)
    for i, load in enumerate(trace.rounds):
        cfg = ctl.journal.epochs.config_id
        start = acc.wall
        _account(acc, model, load, cfg, model.mode)
        windows.append(model.window(load, cfg, start, ctl.journal.epochs.config_epoch, profile_hash))
        clock[0] = acc.wall
        left = n - i - 1
        if left <= 0:
            break
        avg = acc.wall / (i + 1)
        cands = [CandidateEdge(cfg, b, by_cfg[cfg].rollout_engines, by_cfg[b].rollout_engines)
                 for (a, b) in sorted(certified) if a == cfg]
        d = auto.step(ctl, windows, cands, costs, remaining_budget_s=left * avg)
        if d["action"] == "requested":
            src, dst = d["source"], d["target"]
            ec = costs[(profile_hash, src, dst)]
            blk, rec = (true_costs or {}).get((src, dst), (ec.cost_upper_s, ec.recovery_upper_s))
            acc.wall += blk + rec
            acc.block += blk
            acc.recovery += rec
            acc.switches += 1
            acc.path.append(dst)
            ctl.complete()
            clock[0] = acc.wall
            auto.step(ctl, windows, [], costs, remaining_budget_s=left * avg)  # observe outcome
    out = _result(acc, n, pool_gpus, samples_per_round)
    out["decisions"] = sorted({x["reason"] for x in auto.decisions if x["action"] == "hold"})[:20]
    return out


def compare(trace: Trace, model: TimeModel, default: str, *, pool_gpus: int,
            costs: Mapping[tuple[str, str, str], EdgeCost], compat_mode: str = SERIAL,
            **dyn: Any) -> dict[str, Any]:
    """Run all groups on one trace; JSON-serialisable result."""
    kw = {"pool_gpus": pool_gpus, "samples_per_round": dyn.get("samples_per_round", 1.0),
          "first_step_s": dyn.get("first_step_s", 0.0)}
    groups: dict[str, Any] = {
        "compat_default": simulate_fixed(trace, model, default, mode=compat_mode, **kw),
        "fixed_default": simulate_fixed(trace, model, default, mode=model.mode, **kw)}
    fixed = {c: simulate_fixed(trace, model, c, mode=model.mode, **kw) for c in model.configs}
    best = min(fixed, key=lambda c: fixed[c]["wall_s"])
    groups["best_fixed"] = fixed[best]
    groups["dynamic"] = simulate_dynamic(trace, model, default, costs=costs, **kw,
                                         **{k: v for k, v in dyn.items() if k not in kw})
    w = {g: groups[g]["wall_s"] for g in groups}
    return {"scenario": trace.scenario, "seed": trace.seed, "rounds": len(trace.rounds),
            "mode": model.mode, "compat_mode": compat_mode, "default": default,
            "best_fixed_config": best, "groups": groups,
            "mode_gain_s": w["compat_default"] - w["fixed_default"],
            "resize_gain_best_fixed_s": w["fixed_default"] - w["best_fixed"],
            "resize_gain_dynamic_s": w["fixed_default"] - w["dynamic"]}


def payback_rounds(trace: Trace, model: TimeModel, src: str, dst: str,
                   cost: EdgeCost) -> float | None:
    """Rounds needed after switching src->dst for the mean per-round saving to
    repay ``cost_upper + recovery_upper``; None if there is no saving."""
    if not trace.rounds:
        return None
    save = sum(model.round_s(l, src) - model.round_s(l, dst) for l in trace.rounds) / len(trace.rounds)
    if save <= 0:
        return None
    return (cost.cost_upper_s + cost.recovery_upper_s) / save


def to_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
