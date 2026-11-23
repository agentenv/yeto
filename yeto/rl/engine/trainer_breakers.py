"""The trainer's breakers, read through the breaker registry (layer ``trainer``).

The buffer reader, the driver and the commanded island ask this module for their
effective limits instead of parsing their env knobs one by one, and report trips here:
one ``rl_breaker`` record (``yeto.breakers.trip``) next to the legacy event or error,
which still goes out exactly as before. Exit codes do not change; a stopping trip
records ``stop_reason = "breaker:<name>"``.

Values resolve from the process environment (the launcher exports the trainer flags as
env on the island: ``--rl-sample-wait-s`` as ``YETO_SAMPLE_BUFFER_WAIT_S``,
``--rl-max-policy-age`` as ``YETO_SAMPLE_BUFFER_MAX_POLICY_AGE``), so an island without a
``breakers:`` section gets exactly the values it read before. Only the trainer layer is
resolved here: a malformed knob of another layer never stops the trainer.

Import-light: the registry is standard library only; no torch, no Miles.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from yeto import breakers
from yeto.breakers.registry import Registry, load_layers

from .nan_skip import NanSkipPolicy
from .xhw_guard import Limits, XhwGuard

LAYER = "trainer"

SAMPLE_WAIT = "trainer.sample_wait"
POLICY_AGE = "trainer.policy_age"
NAN = "trainer.nan"
XHW_GUARD = "trainer.xhw_guard"
BUFFER_DAMAGED = "trainer.buffer_damaged"
RELEASE_DRAIN = "trainer.release_drain"
GROUP_INTEGRITY = "trainer.group_integrity"
SAMPLE_ALIGNMENT = "trainer.sample_alignment"
COMPACTION_ROUND = "trainer.compaction_round"
STRICT_INVARIANT = "trainer.strict_invariant"

# StrictRlInvariantError.metric -> the entry it trips (any other metric: STRICT_INVARIANT)
STRICT_METRICS = {
    "nonfinite_grad_norm": NAN,
    "nan_skip_breaker": NAN,
    "xhw_guard_stop": XHW_GUARD,
    "nonfinite_reward": "trainer.nonfinite_reward",
    "zero_grad_norm_with_nonzero_advantages": "trainer.gradient_flow",
    "zero_grad_norm_with_required_gradient": "trainer.gradient_flow",
    "policy_hash_mismatch_after_apply": "trainer.policy_hash",
    "policy_hash_mismatch_after_rebuild": "trainer.policy_hash",
}

# While the buffer reader waits for new groups (inside trainer.sample_wait) it emits one
# rl_sample_wait island event per heartbeat, so launch.stall (no island event for
# --rl-stall-timeout) does not end a legitimate wait as a hang: every max(60 s, stall / 4).
WAIT_HEARTBEAT_ENV = "YETO_SAMPLE_BUFFER_HEARTBEAT_S"
MIN_WAIT_HEARTBEAT_S = 60.0

_REGISTRY: Registry | None = None


def registry() -> Registry:
    """The trainer layer's entries alone (loaded once per process)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_layers((LAYER,))
    return _REGISTRY


def resolved(environ: Mapping[str, str] | None = None, *,
             names: Sequence[str] | None = None) -> breakers.ResolvedBreakers:
    """The trainer entries resolved from ``environ`` (default: this process's env).
    Raises ``BreakerConfigError`` on a malformed or contradictory knob of ``names``, the
    entries the caller reads (None: every trainer entry); a knob the caller does not read
    stops nothing here, as before."""
    env = dict(os.environ if environ is None else environ)
    rb = breakers.resolve(breakers.Sources(env=env), registry=registry(), layers=(LAYER,), strict=False)
    problems = [f"{c.name}: {c.message}" for c in rb.conflicts if names is None or c.name in names]
    if problems:
        raise breakers.BreakerConfigError(problems)
    return rb


# ------------------------------------------------------------------ effective values
@dataclass(frozen=True)
class ReaderLimits:
    """What the Miles buffer reader applies (``BatchReader`` arguments)."""

    wait_s: float
    max_policy_age: int  # 0: no age gate (refused by a buffer trainer)
    max_damaged_run: int  # 0: no limit


READER = (SAMPLE_WAIT, POLICY_AGE, BUFFER_DAMAGED)  # the entries reader_limits reads


def reader_limits(rb: breakers.ResolvedBreakers) -> ReaderLimits:
    return ReaderLimits(
        wait_s=float(rb.get(f"{SAMPLE_WAIT}.timeout_s")),
        max_policy_age=int(rb.get(f"{POLICY_AGE}.max_versions") or 0) if rb.enabled(POLICY_AGE) else 0,
        max_damaged_run=int(rb.get(f"{BUFFER_DAMAGED}.max_run_n")) if rb.enabled(BUFFER_DAMAGED) else 0)


def nan_skip_policy(rb: breakers.ResolvedBreakers) -> NanSkipPolicy | None:
    """The opt-in NaN skip (None: strict, a non-finite grad_norm stops the trainer)."""
    if not rb.get(f"{NAN}.skip"):
        return None
    return NanSkipPolicy(max_consecutive=int(rb.get(f"{NAN}.skip_max_consecutive_n")),
                         max_fraction=float(rb.get(f"{NAN}.skip_max_fraction")),
                         min_rounds=int(rb.get(f"{NAN}.skip_min_rounds_n")))


def xhw_guard(rb: breakers.ResolvedBreakers) -> XhwGuard | None:
    """The per-source mismatch guard (None: off)."""
    if not rb.enabled(XHW_GUARD):
        return None

    def get(name: str) -> Any:
        return rb.get(f"{XHW_GUARD}.{name}")

    return XhwGuard(warn=Limits(get("warn_abs_diff"), get("warn_kl"), get("warn_clip")),
                    act=Limits(get("act_abs_diff"), get("act_kl"), get("act_clip")),
                    window=int(get("window_n")), min_rounds=int(get("min_rounds_n")),
                    action=rb.entry(XHW_GUARD).action, quarantine_rounds=int(get("quarantine_rounds_n")),
                    weight=float(get("weight")), max_cond=float(get("max_cond")),
                    confidence_z=float(get("confidence_z")))


def xhw_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the xhw guard is on (the reader applies its actions to a round)."""
    return resolved(environ, names=(XHW_GUARD,)).enabled(XHW_GUARD)


def wait_heartbeat_s(stall_s: float) -> float:
    """Seconds between the reader's wait heartbeats under a stall clock of ``stall_s``
    (0: the stall clock is off)."""
    return max(MIN_WAIT_HEARTBEAT_S, float(stall_s) / 4)


def wait_heartbeat_env(stall_s: float) -> dict[str, str]:
    """The env the launcher gives a buffer-trainer island ({} with the stall clock off:
    the reader then beats at the minimum, which nothing reads)."""
    if float(stall_s) <= 0:
        return {}
    return {WAIT_HEARTBEAT_ENV: f"{wait_heartbeat_s(stall_s):g}"}


def release_drain_s(rb: breakers.ResolvedBreakers) -> float:
    return float(rb.get(f"{RELEASE_DRAIN}.timeout_s"))


# ------------------------------------------------------------------ trips
def trip(name: str, emit: Callable[..., Any] | None, *, value: float | None = None,
         limit: float | None = None, **fields: Any) -> dict[str, Any]:
    """Report one trip of a trainer entry through ``emit``, a component's event writer
    with the signature ``emit(event_name, **fields)`` (the driver's ``emit``, the
    reader's stdout echo). Never raises and never changes control flow."""

    def forward(event: dict[str, Any]) -> None:
        if emit is not None:
            record = dict(event)
            emit(record.pop("event"), **record)

    try:
        entries = registry()
    except Exception as error:  # noqa: BLE001 - a broken declaration never hides the trip itself
        print(f"[breakers] trainer layer not loaded ({error}); {name} reported without registry fields",
              file=sys.stderr, flush=True)
        entries = Registry()
    return breakers.trip(name, value, limit, emit=forward, registry=entries, **fields)


def strict_breaker(metric: str) -> str:
    """The entry a ``StrictRlInvariantError`` of ``metric`` trips."""
    return STRICT_METRICS.get(str(metric), STRICT_INVARIANT)
