"""RL event tape -> W&B.

The RL island already keeps a structured event tape: every rollout round,
fragment broadcast, fragment push, and strict-invariant failure goes
through ``_append_rl_event`` as a JSON record carrying reward, KL, ESS,
clip fraction, action tokens, delta norms, and payload sizes.

So RL needs no new instrumentation, only a tee — the same shape as
``yeto.wandb_tape`` for the syncer's merge record. The difference is that
the RL tape's writer is this process, so the tee is a direct call rather
than a follower thread.

The island's W&B run joins the fleet's group beside the SFT learner runs
and the syncer run, with ``job_type="rl-learner"``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..wandb_logger import NullRun, init

log = logging.getLogger("wandb-rl")

# Event fields that identify rather than measure: they belong in the run's
# config or its name, not in a time series.
_NOT_A_SERIES = frozenset(
    {
        "event",
        "island_id",
        "time_unix",
        "rl/policy_hash",
        "rl/canonical_layout_hash",
        "rl/sync_layout_fingerprint",
        "error",
        "metric",
    }
)

# No single key is present on every RL event kind: a rollout round carries
# base_policy_version and local_round_id, a policy apply carries
# policy_version, and an apply-progress tick carries only policy_version.
# event_metrics therefore synthesizes rl/step from the first of these that
# is present, in island-local order, so every event lands on one x-axis.
# (Checked against a real DeepSeek-V4 island tape: an earlier guess of
# "rl/rollout_id" appears in no event at all, which silently demoted every
# curve to W&B's internal step counter.)
_STEP_KEYS = ("policy_version", "base_policy_version", "local_round_id")

STEP_KEY = "rl/step"

STEP_METRICS = {
    "rl/*": STEP_KEY,
    "sync/*": STEP_KEY,
    "event/*": STEP_KEY,
}


def event_metrics(event: dict[str, Any]) -> dict[str, Any] | None:
    """One RL event -> the metrics logged for it, or None if it carries none.

    Scalars pass through. Lists become their length under a ``*_count``
    name — ``sync/applied_fragments`` is which fragments moved, and how
    many of them moved is the part a curve can show. Strings and hashes
    are dropped: they identify a state, they do not measure one.
    """
    name = str(event.get("event", "rl_event"))
    metrics: dict[str, Any] = {f"event/{name}": 1}
    for key, value in event.items():
        if key in _NOT_A_SERIES:
            continue
        if isinstance(value, bool):
            metrics[key] = int(value)
        elif isinstance(value, (int, float)):
            metrics[key] = value
        elif isinstance(value, (list, tuple)):
            metrics[f"{key}_count"] = len(value)
    for key in _STEP_KEYS:
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[STEP_KEY] = value
            break
    # An event with nothing but its own name is noise on every curve.
    if len(metrics) == 1:
        return None
    return metrics


class RlTelemetry:
    """The island's W&B run, started on the first event that reaches it."""

    def __init__(self) -> None:
        self._run: NullRun | None = None

    def run(self, args) -> NullRun:
        if self._run is None:
            learner_id = getattr(args, "yeto_rl_learner_id", 0)
            self._run = init(
                args,
                job_type="rl-learner",
                name=f"learner-{learner_id}",
                config_extra={
                    "island_backend": "rl-miles",
                    "learner_id": learner_id,
                    "rl_sync_preset": getattr(args, "yeto_rl_sync_preset", None),
                    "model": getattr(args, "yeto_rl_model", None),
                    "data": getattr(args, "yeto_rl_data", None),
                    "base_model_revision": getattr(
                        args, "yeto_rl_base_model_revision", None
                    ),
                    "reward_sha256": getattr(args, "yeto_rl_reward_sha256", None),
                },
                step_metrics=STEP_METRICS,
            )
        return self._run

    def log(self, args, event: dict[str, Any]) -> None:
        run = self.run(args)
        if not run.enabled:
            return
        metrics = event_metrics(event)
        if metrics:
            run.log(metrics)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


# One island per process, so one run per process.
_TELEMETRY = RlTelemetry()


def tee(args, event: dict[str, Any]) -> None:
    """Mirror one RL tape event into W&B. Never raises into the RL loop."""
    if not getattr(args, "wandb", False):
        return
    try:
        _TELEMETRY.log(args, event)
    except Exception as e:  # noqa: BLE001 - telemetry never breaks training
        log.warning("RL W&B telemetry disabled after an error: %s", e)


def finish() -> None:
    try:
        _TELEMETRY.finish()
    except Exception:  # noqa: BLE001
        pass
