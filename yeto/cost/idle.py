"""Idle detection for a run, with a loading-phase exemption.

A run is idle when, outside a loading phase, either

* no progress event (a trained round, a sampled round, a cut, a release ...)
  arrived for ``idle_after_s``, or
* every GPU source that reports utilization has read 0 % for ``idle_after_s``.

Heartbeats are not progress: a hung process still heartbeats.

Loading phase: the run start, and every loading event (driver start-up steps,
weight loading, sampler start, trainer incarnation start ...), opens a window
of ``load_grace_s`` in which neither rule fires - loading K3.1 weights takes
~20 min with 0 % GPU and no progress. A progress event closes the window early.
The window is bounded: loading events alone cannot keep a run alive past
``max_load_s`` since the last progress (a loader stuck in a restart loop).

Once idle for ``idle_after_s`` the detector says ``warn``; after a further
``stop_after_s`` it says ``stop``. Any progress resets it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

PROGRESS_EVENTS = frozenset({
    "rl_round_trained", "rl_local_round", "rl_sampler_round", "rl_sampler_admitted",
    "rl_cut_saved", "rl_round_cut", "rl_publication", "rl_member_publication",
    "rl_policy_release_delta", "rl_policy_release_role", "rl_sampler_policy_loaded",
    "rl_fragment_push", "rl_policy_apply", "rl_dispense_commit", "rl_learner_finalized",
    "rl_trajectory_reward", "sft_step", "step",
})
LOADING_EVENTS = frozenset({
    "rl_driver_start", "rl_startup_step", "rl_driver_phase", "rl_sampler_start",
    "rl_trainer_lifecycle", "rl_trainer_init_from_release", "rl_resume", "rl_data_cursor_restored",
    "rl_sampler_release_not_loaded", "island_ready", "rl_verl_island_start", "loading",
})
GPU_SAMPLE_EVENTS = frozenset({"modal_host_sample", "rl_resource_sample", "gpu_sample"})
# rl_driver_phase / rl_trainer_lifecycle count as loading only in these phases.
LOADING_PHASES = frozenset({"startup", "loading", "load", "init", "starting", "start", "weights",
                            "load_weights", "submitted", "provisioning", "ready_wait"})


def _gpu_util(rec: Mapping[str, Any]) -> float | None:
    vals = rec.get("gpu_util_pct")
    if vals is None:
        vals = rec.get("util_pct")
    if isinstance(vals, (int, float)):
        return float(vals)
    if isinstance(vals, list):
        nums = [float(v) for v in vals if isinstance(v, (int, float))]
        return max(nums) if nums else None
    return None


def classify(rec: Mapping[str, Any]) -> str | None:
    ev = rec.get("event")
    if ev in GPU_SAMPLE_EVENTS:
        return "gpu"
    if ev in PROGRESS_EVENTS:
        return "progress"
    if ev in ("rl_driver_phase", "rl_trainer_lifecycle"):
        phase = str(rec.get("phase") or rec.get("state") or "").lower()
        return "loading" if phase in LOADING_PHASES or not phase else None
    if ev in LOADING_EVENTS:
        return "loading"
    return None


@dataclass
class IdleConfig:
    idle_after_s: float = 15 * 60
    stop_after_s: float = 15 * 60
    load_grace_s: float = 45 * 60
    max_load_s: float = 120 * 60

    @property
    def enabled(self) -> bool:
        return self.idle_after_s > 0


@dataclass
class IdleVerdict:
    state: str  # "ok" | "loading" | "warn" | "stop"
    reason: str | None = None
    idle_s: float = 0.0


@dataclass
class IdleDetector:
    config: IdleConfig
    start: float
    last_progress: float | None = None
    loading_until: float | None = None
    gpu: dict[str, tuple[float, float | None]] = field(default_factory=dict)  # source -> (util, zero_since)

    def __post_init__(self) -> None:
        self.loading_until = self.start + self.config.load_grace_s

    def observe(self, rec: Mapping[str, Any], ts: float, source: str = "") -> str | None:
        kind = classify(rec)
        if kind == "progress":
            self.last_progress = max(self.last_progress or ts, ts)
            self.loading_until = None
        elif kind == "loading":
            self.loading_until = max(self.loading_until or 0.0, ts + self.config.load_grace_s)
        elif kind == "gpu":
            util = _gpu_util(rec)
            if util is not None:
                key = str(rec.get("island") or rec.get("learner_id") or source) + ":" + \
                    str(rec.get("node_rank") or rec.get("node") or 0)
                _prev, zero_since = self.gpu.get(key, (None, None))
                if util > 0:
                    zero_since = None
                elif zero_since is None:
                    zero_since = ts
                self.gpu[key] = (util, zero_since)
        return kind

    def _loading_end(self) -> float | None:
        """End of the exempt loading window (bounded by ``max_load_s`` after the last progress)."""
        if self.loading_until is None:
            return None
        anchor = self.last_progress if self.last_progress is not None else self.start
        return min(self.loading_until, anchor + self.config.max_load_s)

    def _loading(self, now: float) -> bool:
        end = self._loading_end()
        return end is not None and now < end

    def verdict(self, now: float) -> IdleVerdict:
        if not self.config.enabled:
            return IdleVerdict("ok")
        if self._loading(now):
            return IdleVerdict("loading")
        # the idle clocks start no earlier than the end of the last (exempt) loading window
        end = self._loading_end()
        quiet_from = self.start if end is None else min(end, now)
        no_progress = now - max(self.last_progress or self.start, quiet_from)
        reasons = []
        idle = 0.0
        if no_progress >= self.config.idle_after_s:
            reasons.append(f"no progress event for {no_progress / 60:.0f} min")
            idle = no_progress
        if self.gpu and all(z is not None for _u, z in self.gpu.values()):
            zero = now - max(max(z for _u, z in self.gpu.values()), quiet_from)
            if zero >= self.config.idle_after_s:
                reasons.append(f"GPU utilization 0 % on every reporting GPU for {zero / 60:.0f} min")
                idle = max(idle, zero)
        if not reasons:
            return IdleVerdict("ok")
        state = "stop" if idle >= self.config.idle_after_s + self.config.stop_after_s else "warn"
        return IdleVerdict(state, "; ".join(reasons), idle)
