"""The sampler's breaker settings and trips, through the breaker registry
(:mod:`yeto.breakers`, entries in ``yeto.breakers.layers.sampler``).

The sampler process sees only its spec. :func:`spec_breakers` resolves the
registry from it: the spec's native keys (``write_defer_max_s``,
``agentic.drain_timeout_s``, ``prompts.dispense.lease_s``, ...) are the entries'
``spec`` aliases, and a ``breakers`` key in the spec is the breakers section
(design 3.3: a sampler started without ``yeto run up``). A spec without a
``breakers`` key resolves to exactly the values the sampler read before; a bad
value or a section that disagrees with a native key is refused
(:class:`~yeto.breakers.BreakerConfigError`, a ``ValueError``).

:func:`trip` writes one ``rl_breaker`` record on the sampler's tape at a trip
site, next to the site's own event. It never raises and never changes what the
site does; a stop action also records the process's ``stop_reason``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from yeto import breakers

logger = logging.getLogger(__name__)


LAYERS = ("sampler", "harness")  # the entries that run in a sampler process


def spec_breakers(spec: Mapping[str, Any]) -> "breakers.ResolvedBreakers":
    """The resolved sampler and harness entries of a sampler spec (module doc)."""
    section = spec.get("breakers")
    return breakers.resolve(breakers.Sources(spec=spec, section=section if isinstance(section, Mapping) else None),
                            layers=LAYERS)


def dispense_limits(resolved: "breakers.ResolvedBreakers") -> tuple[float, int]:
    """``(lease_s, max_redo)`` of the store dispenser (``prompts.dispense``)."""
    return float(resolved.get("sampler.dispense_lease.lease_s")), int(resolved.get("sampler.dispense_redo.max_n"))


def config_dispense_limits(config: Mapping[str, Any]) -> tuple[float, int]:
    """:func:`dispense_limits` of a bare ``prompts.dispense`` config (no spec around it)."""
    return dispense_limits(spec_breakers({"prompts": {"dispense": dict(config)}}))


def stop_reason_of(error: BaseException) -> str | None:
    """``breaker:<name>`` of an error a sampler breaker raised to stop the process
    (its leave record carries it; the exit code is unchanged), else None."""
    from .policy_release import ReleaseNotLoaded  # noqa: PLC0415 - these modules import this one
    from .sampler_checks import SamplerLoadFatal  # noqa: PLC0415
    from .sampler_loop import StoreReadOutage  # noqa: PLC0415

    if isinstance(error, StoreReadOutage) or (isinstance(error, ReleaseNotLoaded)
                                              and getattr(error, "fields", {}).get("damaged")):
        return breakers.stop_reason_for("sampler.store_read_outage")
    if isinstance(error, SamplerLoadFatal):
        return breakers.stop_reason_for("sampler.load_failures")
    return None


def trip(emit: Callable[..., None] | None, name: str, value: float | None, limit: float | None, *,
         state: str = "trip", **fields: Any) -> None:
    """One ``rl_breaker`` record of entry ``name`` through the sampler's tape writer
    ``emit(event, **fields)`` (None: only the stop reason). ``fields``: the event's
    optional fields (``detail``, ``legacy_event``, ``count``, ``unit``, ``scope``)."""
    def sink(event: Mapping[str, Any]) -> None:
        if emit is not None:
            record = dict(event)
            emit(record.pop("event"), **record)

    try:
        breakers.trip(name, value, limit, state=state, emit=sink, **fields)
    except Exception as error:  # noqa: BLE001 - observation only: the trip site goes on as before
        logger.warning("rl_breaker %s not written: %r", name, error)
