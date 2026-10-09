"""Neutral rollout-side metadata port (yeto-framework-decoupling 4.11, design D11).

Runs in the *rollout* process (and codex subprocesses): algorithm extensions
and harness code read the policy token the driver published and report
per-round counters through these functions.  The implementation comes from the
active backend (``yeto.rl.engine.backends``, role ``rollout_meta``), chosen by
``$YETO_RL_BACKEND`` (unset = miles).  The Miles implementation is
``yeto.rl.adapters.miles.rollout_meta_hook``.

Key names and :func:`counter_value` are neutral and live here; their string
values are unchanged (tape fields unchanged).
"""

from __future__ import annotations

from typing import Any

from . import backends

# IR-3/IR-4 sample-metadata keys written by agentic generate code.
EXPECTED_POLICY_VERSION_KEY = "expected_policy_version"
POLICY_AGE_VIOLATION_KEY = "policy_age_violation"
TITO_SESSION_MISMATCH_KEY = "tito_session_mismatch"
TITO_CHAIN_BREAKS_KEY = "tito_chain_breaks"


def counter_value(value: Any) -> int:
    """A per-sample counter as an int.

    Upstream Miles' session server writes ``tito_session_mismatch`` into the
    same sample-metadata key as a *list* of mismatch records
    (``compute_session_mismatch`` -> ``list[dict]``, empty when the replayed
    tokens match), while the harness bridge writes an int; both count
    mismatches (A-T3-6, codex-smoke-20261003-10 failed the rollout on
    ``int(list)``).  Dicts count their non-zero entries, None/"" count 0.
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        return sum(1 for v in value.values() if v)
    return int(value)


def port(name: str | None = None):
    """The active backend's rollout-metadata module."""

    return backends.module("rollout_meta", name)


def sink_available() -> bool:
    """Whether the driver's metadata sink can be reached from this process."""

    return port().sink_available()


def current_policy_token() -> str | None:
    return port().current_policy_token()


def expected_policy_version(sample: Any = None) -> str | None:
    return port().expected_policy_version(sample)


def record_round_metadata(args: Any, round_id: Any, **counters: int) -> None:
    port().record_round_metadata(args, round_id, **counters)
