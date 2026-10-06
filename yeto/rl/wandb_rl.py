"""Privacy-bounded RL event tape -> W&B scalar projection.

Only recognized round/eval events and an explicit scalar allowlist cross this
boundary.  The local tape remains authoritative; W&B is a failure-safe second
reader owned by the island head.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from ..wandb_logger import NullRun, WandbRun, init

log = logging.getLogger("wandb-rl")

# Training scalars use the exact optimizer step emitted by Miles, matching the
# source-attested local loss monitor. Rollout/eval scalars keep their existing
# island-local round axis.
_STEP_KEYS = ("policy_version", "base_policy_version", "local_round_id")

STEP_KEY = "rl/step"
TRAIN_STEP_KEY = "train/step"

STEP_METRICS = {
    "train/*": TRAIN_STEP_KEY,
    "rl/*": STEP_KEY,
    "eval/*": STEP_KEY,
}

# Closed output surface. Input aliases are limited to namespaced scalar fields
# produced by Yeto/Miles; bare keys are deliberately excluded because they can
# occur inside rollout payloads. Unknown keys and their values are never copied.
_METRIC_ALIASES = {
    "train/loss": ("train/loss",),
    "train/pg_loss": ("train/pg_loss",),
    "train/grad_norm": ("train/grad_norm",),
    "train/train_rollout_kl": (
        "train/train_rollout_kl",
        "train/ppo_kl",
        "train/approx_kl",
        "train/mean_kl",
        "train/kl",
        "rl/current_vs_rollout_kl",
    ),
    "train/ess_ratio": ("train/ess_ratio", "rl/ess_ratio"),
    "train/pg_clipfrac": (
        "train/pg_clipfrac",
        "train/clipfrac",
        "train/clip_fraction",
        "rl/clip_fraction",
    ),
    "train/lr": (
        "train/lr",
        "train/lr-pg_0",
        "train/learning_rate",
        "train/policy_lr",
    ),
    "rl/reward_mean": ("rl/reward_mean", "train/reward_mean"),
    "rl/pass_rate": ("rl/pass_rate", "train/pass_rate", "train/pass@1"),
    "eval/reward_mean": ("rl/eval/result",),
    "eval/pass_rate": ("rl/eval/pass_at_1",),
}

# Ports-engine rl_local_round records carry bare top-level names (driver
# ``_stats``); they are read only from that event's top level, never from
# payloads, and only when the namespaced alias is absent (fleet-dashboard 1.4).
_BARE_LOCAL_ROUND_ALIASES = {
    "train/loss": ("loss",),
    "train/pg_loss": ("pg_loss",),
    "train/grad_norm": ("grad_norm",),
    "train/train_rollout_kl": ("mean_kl",),
    "train/ess_ratio": ("ess_ratio",),
    "train/pg_clipfrac": ("clip_fraction",),
    "train/lr": ("lr", "applied_lr"),
    "rl/reward_mean": ("reward_mean",),
    "rl/pass_rate": ("pass_rate",),
}
# ``train_metrics{}`` (Miles loss dict round means, fleet-dashboard D4) keys
# become ``train/<key>``: plain metric-name keys with finite scalar values only.
_TRAIN_METRIC_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,63}$")
_TRAIN_STEP_ALIASES = (TRAIN_STEP_KEY, "train_step")
_derived_step_logged = False

_TRAIN_OUTPUTS = frozenset(key for key in _METRIC_ALIASES if key.startswith("train/"))
_RL_OUTPUTS = frozenset(key for key in _METRIC_ALIASES if key.startswith("rl/"))
_EVAL_OUTPUTS = frozenset(key for key in _METRIC_ALIASES if key.startswith("eval/"))


def _finite_scalar(event: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        value = event.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        try:
            scalar = float(value)
        except (OverflowError, TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            return scalar
    return None


def _non_negative_step(event: dict[str, Any], aliases: tuple[str, ...]) -> int | None:
    for key in aliases:
        value = event.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 2**63 - 1
        ):
            return value
    return None


def event_metrics(event: dict[str, Any]) -> dict[str, Any] | None:
    """Project one event onto the explicit scalar-only W&B surface."""

    global _derived_step_logged
    event_name = event.get("event")
    if event_name not in {"rl_local_round", "rl_eval_result", "rl_round_trained"}:
        return None

    metrics = {
        output: value
        for output, aliases in _METRIC_ALIASES.items()
        if (value := _finite_scalar(event, aliases)) is not None
    }
    train_extra: dict[str, float] = {}
    if event_name in ("rl_local_round", "rl_round_trained"):
        if event_name == "rl_local_round":
            for output, aliases in _BARE_LOCAL_ROUND_ALIASES.items():
                if output not in metrics:
                    value = _finite_scalar(event, aliases)
                    if value is not None:
                        metrics[output] = value
        tm = event.get("train_metrics")
        if isinstance(tm, dict):
            for key, raw in tm.items():
                if not isinstance(key, str) or not _TRAIN_METRIC_KEY.match(key):
                    continue
                value = _finite_scalar({key: raw}, (key,))
                if value is not None:
                    train_extra[f"train/{key}"] = value
    if event_name == "rl_round_trained":
        metrics = {}
    for key in ("rl/pass_rate", "eval/pass_rate"):
        value = metrics.get(key)
        if value is not None and not 0.0 <= value <= 1.0:
            metrics.pop(key)
    if event_name == "rl_local_round":
        for key in _EVAL_OUTPUTS:
            metrics.pop(key, None)
    else:
        for key in _TRAIN_OUTPUTS | _RL_OUTPUTS:
            metrics.pop(key, None)
    train_step = _non_negative_step(event, _TRAIN_STEP_ALIASES)
    if train_step is None and event_name == "rl_local_round":
        # ports path: Miles' optimizer step is not reported yet; the learner's
        # local round is the closest monotone axis (fleet-dashboard 1.4).
        train_step = _non_negative_step(event, ("local_round_id",))
        if train_step is not None and not _derived_step_logged:
            _derived_step_logged = True
            log.info("train/step derived from local_round_id (event has no train_step)")
    for key, value in train_extra.items():
        metrics.setdefault(key, value)
    train_outputs = _TRAIN_OUTPUTS | set(train_extra)
    if train_step is None:
        for key in train_outputs:
            metrics.pop(key, None)
    elif any(key in metrics for key in train_outputs):
        metrics[TRAIN_STEP_KEY] = train_step

    round_step = _non_negative_step(event, _STEP_KEYS)
    round_outputs = _RL_OUTPUTS | _EVAL_OUTPUTS
    if round_step is None:
        for key in round_outputs:
            metrics.pop(key, None)
    elif any(key in metrics for key in round_outputs):
        metrics[STEP_KEY] = round_step
    return metrics or None


class RlTelemetry:
    """The island's W&B run, started on the first event that reaches it."""

    def __init__(self) -> None:
        self._run: NullRun | WandbRun | None = None
        self._seen: set[tuple[tuple[str, int], tuple[str, ...]]] = set()

    def run(self, args) -> NullRun | WandbRun:
        if self._run is None:
            learner_id = getattr(args, "yeto_rl_learner_id", 0)
            self._run = init(
                args,
                job_type="rl-learner",
                name=f"learner-{learner_id}",
                config_override={
                    "island_backend": "rl-miles",
                    "learner_id": learner_id,
                    "rl_sync_preset": getattr(args, "yeto_rl_sync_preset", None),
                },
                step_metrics=STEP_METRICS,
            )
        return self._run

    def log(self, args, event: dict[str, Any]) -> None:
        metrics = event_metrics(event)
        if not metrics:
            return
        run = self.run(args)
        if not run.enabled:
            return
        axes = tuple(
            (name, int(metrics[name]))
            for name in (TRAIN_STEP_KEY, STEP_KEY)
            if name in metrics
        )
        series = tuple(
            sorted(name for name in metrics if name not in {TRAIN_STEP_KEY, STEP_KEY})
        )
        point = (axes, series)
        if point in self._seen:
            return
        run.log(metrics)
        self._seen.add(point)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
        self._seen.clear()


# One island per process, so one run per process.
_TELEMETRY = RlTelemetry()


def tee(args, event: dict[str, Any]) -> None:
    """Mirror one RL tape event into W&B. Never raises into the RL loop."""
    if not getattr(args, "wandb", False):
        return
    try:
        _TELEMETRY.log(args, event)
    except Exception as error:  # noqa: BLE001 - telemetry never breaks training
        log.warning(
            "RL W&B telemetry disabled after a %s",
            type(error).__name__,
        )


def finish() -> None:
    try:
        _TELEMETRY.finish()
    except Exception as error:  # noqa: BLE001 - telemetry never breaks training
        log.warning(
            "RL W&B telemetry cleanup failed with a %s",
            type(error).__name__,
        )
