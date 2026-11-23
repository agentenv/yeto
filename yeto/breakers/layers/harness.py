"""Breakers of the harness layer: the agentic harness inside the sampler (``yeto.rl.harness``).

Harbor and its sandbox providers read these through
:mod:`yeto.rl.harness.harbor.breaker_settings`. Every knob here used to be an env var
or a constant; an env var that existed stays as a deprecated alias of the entry's
parameter. A parameter that was a constant gets a plain env alias: the channel the
run's write-back uses to reach the sampler process (write ``breakers.harness.*``
instead of setting it by hand). The defaults are literals (the harness modules are
not light enough to import here); a test compares them with the runtime constants.

The trial's own caps (``wall_clock_s``, ``agent_timeout_s``) and the per-slot infra
retries are sampler-layer entries (``sampler.trial_wall_clock``,
``sampler.infra_retries``); the harness only reads the trial wall clock to size the
sandbox TTL.
"""

from __future__ import annotations

import json
from typing import Any

from yeto.breakers.spec import Alias, Breaker, Param


def sandbox_caps(raw: Any) -> dict[str, int] | None:
    """``HARBOR_MAX_SANDBOXES`` -> ``{provider or "*": cap}``: ``N`` is every provider, a JSON
    object names providers (``*`` for the others); empty is no cap. ``ValueError`` otherwise."""
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            raw = json.loads(raw)
        except ValueError:
            raise ValueError(f"must be N or a JSON object, got {raw!r}") from None
    caps = {"*": raw} if isinstance(raw, int) and not isinstance(raw, bool) else raw
    if not isinstance(caps, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 1 for k, v in caps.items()):
        raise ValueError(f"caps must be integers >= 1, got {raw!r}")
    return dict(caps) or None


BREAKERS: tuple[Breaker, ...] = (
    Breaker(
        name="harness.setup_retries", layer="harness", kind="retry_budget",
        params=(Param("max_n", int, 0, unit="n", lo=0,
                      doc="runs again of a trial that failed before the agent ran; 0: no retry"),
                Param("backoff_initial_s", float, 5.0, unit="s", lo=0,
                      doc="wait before the first retry; doubles per retry, with jitter (x0.5-1)"),
                Param("backoff_max_s", float, 60.0, unit="s", lo=0, doc="cap of one wait")),
        aliases=(Alias("env", "HARBOR_SETUP_RETRIES", "max_n", deprecated=True),
                 Alias("env", "HARBOR_SETUP_BACKOFF_S", "backoff_initial_s"),
                 Alias("env", "HARBOR_SETUP_BACKOFF_MAX_S", "backoff_max_s")),
        depends_on=("D9",), gauge="count", owner="yeto.rl.harness.harbor.trial",
        doc="sandbox start, image pull and agent install failures are retried on a new sandbox; "
            "once spent the trial is ABORTED (infra)"),
    Breaker(
        name="harness.reach_check", layer="harness", kind="guard",
        params=(Param("attempts_n", int, 15, unit="n", lo=1, doc="curl probes per endpoint"),
                Param("timeout_s", float, 5.0, unit="s", lo=0, doc="--max-time of one probe"),
                Param("interval_s", float, 2.0, unit="s", lo=0, doc="sleep between two probes")),
        actions=("drop",), default_action="drop",
        aliases=(Alias("env", "YETO_AGENTENV_SKIP_REACH_CHECK", None, off_values=("1",), deprecated=True),
                 Alias("env", "YETO_AGENTENV_REACH_TRIES", "attempts_n", deprecated=True),
                 Alias("env", "YETO_AGENTENV_REACH_TIMEOUT_S", "timeout_s", deprecated=True),
                 Alias("env", "YETO_AGENTENV_REACH_INTERVAL_S", "interval_s")),
        depends_on=("D9",), owner="yeto.rl.harness.harbor.agentenv_env",
        doc="an AgentENV microVM that cannot reach its judge / records endpoints fails its start "
            "(NetworkError, counted as infra) instead of scoring a silent zero"),
    Breaker(
        name="harness.sandbox_caps", layer="harness", kind="breaker",
        params=(Param("max_sandboxes", dict, None, nullable=True,
                      doc='sandboxes one trial process holds, {"<provider>": N, "*": N} ("*": every other '
                          "provider); null: no cap (remote-docker requires one)"),
                Param("agentenv_max_active_n", int, 64, unit="n", lo=1),
                Param("agentenv_starts_per_minute_n", int, 60, unit="n", lo=1),
                Param("agentenv_starts_per_hour_n", int, None, unit="n", lo=1, nullable=True),
                Param("agentenv_create_timeout_s", float, 120.0, unit="s", lo=0,
                      doc="one AgentENV API call (a cold create takes tens of seconds)"),
                Param("boat_max_active_n", int, 100, unit="n", lo=1),
                Param("boat_starts_per_minute_n", int, 12, unit="n", lo=1),
                Param("boat_starts_per_hour_n", int, None, unit="n", lo=1, nullable=True)),
        actions=("pause",), default_action="pause", can_disable=False,
        aliases=(Alias("env", "HARBOR_MAX_SANDBOXES", "max_sandboxes", to_canonical=sandbox_caps,
                       from_canonical=lambda caps: json.dumps(caps, sort_keys=True), deprecated=True),
                 Alias("env", "YETO_AGENTENV_MAX_ACTIVE", "agentenv_max_active_n", deprecated=True),
                 Alias("env", "YETO_AGENTENV_STARTS_PER_MINUTE", "agentenv_starts_per_minute_n", deprecated=True),
                 Alias("env", "YETO_AGENTENV_STARTS_PER_HOUR", "agentenv_starts_per_hour_n", deprecated=True),
                 Alias("env", "YETO_AGENTENV_REQUEST_TIMEOUT_S", "agentenv_create_timeout_s", deprecated=True),
                 Alias("env", "YETO_BOAT_MAX_ACTIVE", "boat_max_active_n", deprecated=True),
                 Alias("env", "YETO_BOAT_STARTS_PER_MINUTE", "boat_starts_per_minute_n", deprecated=True),
                 Alias("env", "YETO_BOAT_STARTS_PER_HOUR", "boat_starts_per_hour_n", deprecated=True)),
        gauge="level", owner="yeto.rl.harness.harbor.capacity",
        doc="a sampler whose max_in_flight exceeds a cap refuses to start; at run time a create "
            "above a cap or start rate waits"),
    Breaker(
        name="harness.sandbox_ttl", layer="harness", kind="breaker", can_disable=False,
        params=(Param("limit_s", int, None, unit="s", lo=60, nullable=True,
                      doc="every trial sandbox's TTL; null: an AgentENV trial gets its wall clock + "
                          "margin_s, a boat trial default_s"),
                Param("margin_s", float, 900.0, unit="s", lo=0,
                      doc="added to the trial wall clock: create (cold start, setup retries), "
                          "verifier upload, delete"),
                Param("default_s", int, 10800, unit="s", lo=60,
                      doc="a boat trial's TTL without limit_s (and the providers' constructor default)")),
        aliases=(Alias("env", "HARBOR_SANDBOX_TTL_S", "limit_s"),
                 Alias("env", "HARBOR_SANDBOX_TTL_MARGIN_S", "margin_s"),
                 Alias("env", "HARBOR_SANDBOX_TTL_DEFAULT_S", "default_s")),
        depends_on=("D3",), owner="yeto.rl.harness.harbor.trial",
        doc="the provider reclaims a sandbox after its TTL: the backstop for a lost delete "
            "(a killed sampler); HARBOR_ENV_KWARGS ttl_s still wins over it"),
    Breaker(
        name="harness.turn_max_tokens", layer="harness", kind="semantic",
        native="spec agentic.turn_max_tokens",
        owner="yeto.rl.harness.gateway.proxy",
        doc="the records proxy's signed max_tokens of one model call; a longer reply is cut "
            "(finish length) and the agent goes on"),
    Breaker(
        name="harness.agent_call", layer="harness", kind="semantic",
        native="pi_config DEFAULT_HTTP_IDLE_TIMEOUT_S / COMPACTION_HTTP_IDLE_TIMEOUT_S / DEFAULT_MAX_RETRIES; "
               "task agent kwargs http_idle_timeout_s",
        owner="yeto.rl.harness.harbor.pi_config",
        doc="pi's own per-call idle timeout (1800 s, 3600 s with proxy compaction) and model-call "
            "retries (8); a retried call forks the token chain"),
)
