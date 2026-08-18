"""Optional Weights & Biases telemetry.

Yeto runs are fleets, not single processes: one syncer plus one learner
island per ``--gpu`` entry, each island a separate torchrun world in a
different cloud/region. The mapping onto W&B is therefore

    group   = the run name (``--cluster-prefix``)
    run     = one island (job_type="learner") or the syncer (job_type="syncer")

with one W&B run per island rather than one for the fleet. Island-local
steps advance independently under async DiLoCo, so a shared run would log
a non-monotonic step series and silently drop points.

Everything here is import-safe and failure-safe: without the ``wandb``
extra installed, without ``--wandb``, or on any rank but 0, ``init``
returns a no-op sink, and a telemetry error downgrades a live run to that
same sink instead of interrupting training.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("wandb")

# Inner steps between training log lines / W&B points, shared by every
# backend. A run shorter than one window still emits a final point when its
# loop ends, so a late-joining island is never left without a curve.
TELEMETRY_EVERY = 10

# Launch flags that carry secrets or are pure noise in a config table.
_CONFIG_SKIP = frozenset(
    {
        "command",
        "func",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_mode",
        "wandb_group",
        # yeto.wandb_tape's own argv: the sidecar's config is the fleet's,
        # forwarded through --config-json, not how the reader was invoked.
        "tape",
        "follow",
        "from_start",
        "config_json",
    }
)

# Env vars that must never reach wandb.config.
_SECRET_SUBSTRINGS = ("token", "key", "secret", "password", "credential")


def _is_secret(name: str) -> bool:
    lowered = name.lower()
    return any(s in lowered for s in _SECRET_SUBSTRINGS)


def add_arguments(p) -> None:
    """The ``--wandb*`` flags, identical on every learner backend and on
    ``yeto launch`` (which forwards them verbatim)."""
    obs = p.add_argument_group("observability")
    obs.add_argument(
        "--wandb",
        action="store_true",
        help="stream training and sync metrics to Weights & Biases",
    )
    obs.add_argument("--wandb-project", default="yeto", help="W&B project name")
    obs.add_argument("--wandb-entity", default=None, help="W&B team/user")
    obs.add_argument(
        "--wandb-mode",
        choices=["online", "offline"],
        default="online",
        help="offline buffers to disk for a later `wandb sync`, which is the "
        "safer choice on a WAN-distant island",
    )


def build_config(args, extra: dict | None = None) -> dict:
    """JSON-safe ``wandb.config`` from a launch/learner argparse namespace.

    Private attributes (the ``_training_recipe`` / ``_adapter_lineage``
    scratch space the learner hangs off ``args``) and anything whose name
    smells like a credential are dropped.
    """
    import json

    config: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key.startswith("_") or key in _CONFIG_SKIP or _is_secret(key):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        config[key] = value
    for key, value in (extra or {}).items():
        config[key] = value
    return config


class NullRun:
    """The sink used whenever telemetry is off, unavailable, or broken."""

    enabled = False

    def log(self, metrics: dict) -> None:  # noqa: ARG002
        pass

    def summary(self, metrics: dict) -> None:  # noqa: ARG002
        pass

    def finish(self, exit_code: int = 0) -> None:  # noqa: ARG002
        pass


class WandbRun:
    """A live W&B run that degrades to a no-op on the first failure."""

    enabled = True

    def __init__(self, run, module):
        self._run = run
        self._wandb = module
        self._broken = False

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None)

    def log(self, metrics: dict) -> None:
        if self._broken:
            return
        try:
            self._wandb.log(metrics)
        except Exception as e:  # noqa: BLE001 - telemetry never breaks training
            self._fail(e)

    def summary(self, metrics: dict) -> None:
        if self._broken:
            return
        try:
            for key, value in metrics.items():
                self._run.summary[key] = value
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def finish(self, exit_code: int = 0) -> None:
        if self._broken:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def _fail(self, e: Exception) -> None:
        self._broken = True
        log.warning("W&B telemetry disabled after an error: %s", e)


def _resolve_mode(requested: str) -> str:
    """Force offline when the node has no way to authenticate.

    Learners are spot VMs with no TTY: an unauthenticated online
    ``wandb.init`` either errors out or blocks, so a missing key becomes
    an offline run (still recoverable with ``wandb sync``) rather than a
    dead island.
    """
    if requested != "online":
        return requested
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = os.path.expanduser("~/.netrc")
    if os.path.isfile(netrc):
        try:
            with open(netrc, encoding="utf-8", errors="replace") as f:
                if "api.wandb.ai" in f.read():
                    return "online"
        except OSError:
            pass
    log.warning(
        "no WANDB_API_KEY and no wandb entry in ~/.netrc; logging offline "
        "(recover the run later with: wandb sync)"
    )
    return "offline"


def init(
    args,
    *,
    job_type: str,
    name: str,
    rank: int = 0,
    group: str | None = None,
    config_extra: dict | None = None,
    step_metrics: dict[str, str] | None = None,
) -> NullRun | WandbRun:
    """Start this process's W&B run, or return a no-op sink.

    Only rank 0 of an island logs: every other rank holds the same
    all-reduced loss and the same merged fragments, so they would add
    duplicate runs and nothing else.

    ``step_metrics`` maps a metric glob (``"train/*"``) to the x-axis it
    should be plotted against (``"local_step"``). The default pairs
    training metrics with the island's local step and sync metrics with
    the fleet-global step, which is what makes an island's curve readable
    next to a fleet-wide one.
    """
    if rank != 0 or not getattr(args, "wandb", False):
        return NullRun()
    try:
        import wandb
    except ImportError:
        log.warning(
            "--wandb was requested but the wandb package is missing; "
            'install it with: pip install "yeto[wandb]"'
        )
        return NullRun()

    group = group or os.environ.get("YETO_RUN_GROUP") or getattr(args, "cluster_prefix", None)
    mode = _resolve_mode(getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE") or "online")
    # A deterministic id plus resume="allow" is what keeps a preempted spot
    # island on ONE curve: the fleet controller relaunches the learner and
    # the new process reattaches to the run it left behind.
    run_id = f"{group}-{name}" if group else name
    try:
        run = wandb.init(
            project=getattr(args, "wandb_project", None) or "yeto",
            entity=getattr(args, "wandb_entity", None),
            group=group,
            job_type=job_type,
            name=name,
            id=run_id,
            resume="allow",
            mode=mode,
            config=build_config(args, config_extra),
        )
        metrics = step_metrics or {"train/*": "local_step", "sync/*": "global_step"}
        for axis in sorted(set(metrics.values())):
            wandb.define_metric(axis)
        for glob, axis in metrics.items():
            wandb.define_metric(glob, step_metric=axis)
    except Exception as e:  # noqa: BLE001 - telemetry never breaks training
        log.warning("could not start W&B telemetry (%s); continuing without it", e)
        return NullRun()
    log.info("W&B run %s/%s started (mode=%s)", group, name, mode)
    return WandbRun(run, wandb)
