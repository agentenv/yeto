"""The harness layer's breaker settings, resolved by the breaker registry.

Harbor and its sandbox providers read their setup retries, reach check, sandbox caps
and sandbox TTL here (entries of :mod:`yeto.breakers.layers.harness`) instead of
parsing their own environment variables. A value comes from the entry's env alias
(``HARBOR_SETUP_RETRIES``, ``YETO_AGENTENV_MAX_ACTIVE``, ...), which is also the channel
the run's write-back uses to bring a ``breakers.harness.*`` value into the sampler, or
else from the registry default (the constant the harness used before). Two aliases of
one value that disagree, or a value out of bounds, is a :class:`HarnessConfigError`.

An empty variable counts as unset, as it did for every harness reader. Resolution is
cached per distinct value of the variables it reads, so a per-trial read is a dict
lookup.

Trips happen per trial, and the harness has no event tape: the harness counts them
and :func:`trip_events` folds the counts since the last round into aggregated
``rl_breaker`` events for the round's own tape.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from typing import Any

from yeto import breakers
from yeto.rl.agentic import HarnessConfigError

LAYER = "harness"
REACH_FAILURE = "NetworkError"  # agentenv_network: the reach check's exception


@functools.cache
def env_names() -> tuple[str, ...]:
    """Every environment variable a harness entry reads (its env aliases)."""
    names = [a.path for b in breakers.default_registry().layer(LAYER) for a in b.aliases if a.surface == "env"]
    return tuple(dict.fromkeys(names))


def resolved(environ: Mapping[str, str] | None = None) -> breakers.ResolvedBreakers:
    """The harness entries as resolved for ``environ`` (default: this process's environment)."""
    environ = os.environ if environ is None else environ
    return _resolve(tuple((name, value) for name in env_names()
                          if (value := str(environ.get(name, "") or "")).strip()))


def setting(path: str, environ: Mapping[str, str] | None = None) -> Any:
    """One resolved value, ``"harness.setup_retries.max_n"``."""
    return resolved(environ).get(path)


def enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    return resolved(environ).enabled(name)


def source(path: str, environ: Mapping[str, str] | None = None) -> str:
    """Where a value came from, for messages: ``"env HARBOR_SETUP_RETRIES"`` / ``"default"``."""
    name, _, key = path.rpartition(".")
    return resolved(environ).entry(name).sources[key].location


@functools.lru_cache(maxsize=32)
def _resolve(env: tuple[tuple[str, str], ...]) -> breakers.ResolvedBreakers:
    try:
        return breakers.resolve(breakers.Sources(env=dict(env)), layers=(LAYER,))
    except breakers.BreakerConfigError as exc:
        raise HarnessConfigError(f"harness breakers: {exc}") from exc


def reset() -> None:
    """Tests: resolve again on next use (the registry changed)."""
    _resolve.cache_clear()
    env_names.cache_clear()


def trip_events(harness: Any, *, scope: Mapping[str, Any] | None = None,
                environ: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """One aggregated ``rl_breaker`` event per harness entry that tripped since the last
    call (``harness.take_trips()``, :class:`~.trial.HarborHarness`); empty when none did
    or the harness counts no trips.

    * ``harness.setup_retries``: trials whose setup retries were all spent on failures;
      ``value`` = ``limit`` = ``max_n``;
    * ``harness.reach_check``: microVM starts whose reach check failed (retried
      attempts included); ``value`` = failures, no limit.
    """
    take = getattr(harness, "take_trips", None)
    trips = take() if take is not None else {}
    events = []
    if spent := trips.get("harness.setup_retries", 0):
        max_n = harness.setup_retries
        events.append(breakers.trip("harness.setup_retries", max_n, max_n, scope=dict(scope or {}), count=spent,
                                    detail=f"{spent} trial(s) failed before the agent ran after {max_n} retries",
                                    config_source=source("harness.setup_retries.max_n", environ)))
    if failed := trips.get("harness.reach_check", 0):
        events.append(breakers.trip("harness.reach_check", failed, None, scope=dict(scope or {}), count=failed,
                                    detail=f"{failed} microVM start(s) could not reach the judge / records "
                                           "endpoints"))
    return events
