"""One registry for yeto's breakers: every automatic stop, drop, pause or refusal.

Each layer declares its entries in ``yeto/breakers/layers/<layer>.py``; the resolver merges
a run config's ``breakers:`` section with every legacy knob (flags, spec fields, env) and
names the source of each effective value; components report trips with :func:`trip`.
Standard library only: importing this package never loads a training backend.
"""

from yeto.breakers.events import (
    clear_stop_reason,
    current_stop_reason,
    record_stop_reason,
    status_event,
    status_item,
    stop_reason_for,
    stop_reason_from_env,
    trip,
)
from yeto.breakers.registry import Registry, default_registry
from yeto.breakers.resolve import (
    Conflict,
    Materialized,
    ResolvedBreakers,
    ResolvedEntry,
    Source,
    Sources,
    load_resolved,
    materialize,
    parse_section,
    preflight,
    resolve,
    resolved,
)
from yeto.breakers.spec import (
    Alias,
    Breaker,
    BreakerConfigError,
    Finding,
    Param,
    RegistryError,
    SwitchGroup,
)

__all__ = [
    "Alias", "Breaker", "BreakerConfigError", "Conflict", "Finding", "Materialized", "Param", "Registry",
    "RegistryError", "ResolvedBreakers", "ResolvedEntry", "Source", "Sources", "SwitchGroup",
    "clear_stop_reason", "current_stop_reason", "default_registry", "load_resolved", "materialize",
    "parse_section", "preflight", "record_stop_reason", "resolve", "resolved", "status_event", "status_item",
    "stop_reason_for", "stop_reason_from_env", "trip",
]
