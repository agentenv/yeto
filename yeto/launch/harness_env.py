"""Launch-side island environment: the Modal sandbox secret, the Harbor agent's
island pieces and the CompactionRL switch."""

from __future__ import annotations

import json
import os
import re

# Miles' Harbor examples that the Harbor agent imports by module name
# (``harbor_agent_function``; ``generate`` for ``harbor_rollout``), as Miles'
# own launcher puts them on the workers' PYTHONPATH
# (``examples/experimental/harbor/launch_common.agentic_pythonpath_dirs``).
HARBOR_EXAMPLE_DIRS = ("examples/experimental/harbor", "examples/swe-agent-harbor-docker")
# The gate relay port a Modal island forwards through a Modal tunnel when the
# launch sets none (one fixed port: the tunnel URL names exactly one).
DEFAULT_HARBOR_RELAY_PORT = 31500


def modal_sandbox_secret_name(args, spec) -> str | None:
    """--modal-sandbox-secret NAME, or None when unset.  Only Modal islands can
    attach a named Modal Secret: any other cloud is refused, never ignored."""
    name = getattr(args, "modal_sandbox_secret", None)
    if not name:
        return None
    if getattr(spec, "cloud", None) != "modal":
        raise ValueError(
            f"--modal-sandbox-secret attaches a Modal Secret to a Modal island; "
            f"island cloud {getattr(spec, 'cloud', None)!r} cannot use it")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(name)):
        raise ValueError(f"--modal-sandbox-secret {name!r} is not a Modal Secret name")
    return str(name)


def is_harbor_agent(args) -> bool:
    """True when the run's custom agent is yeto's Harbor agent (``harbor_agent.run``)."""
    from ..rl.adapters.miles.harbor_agent import AGENT_FUNCTION

    return getattr(args, "custom_agent_function_path", None) == AGENT_FUNCTION


def harbor_example_pythonpath(args, miles_root: str = "$HOME/miles") -> str:
    """PYTHONPATH entries (each with a trailing ``:``) the Harbor agent needs on the island, or ""."""
    if not is_harbor_agent(args):
        return ""
    return "".join(f"{miles_root}/{d}:" for d in HARBOR_EXAMPLE_DIRS)


def harbor_island_envs(envs: dict[str, str], *, load_boat_token=None) -> dict[str, str]:
    """Environment a Modal Harbor island adds to ``envs`` (which it does not change).

    * ``BOAT_API_KEY`` when the sandbox provider is boat (``HARBOR_ENV_TYPE=boat``)
      and the launch did not give one: the token of the launching machine
      (``boat_client.load_token``: ``BOAT_API_KEY``, else the boat CLI config).
      It travels as a Modal Secret like ``HF_TOKEN`` and is never printed.
    * ``YETO_HARBOR_GATE_RELAY_PORT`` (default :data:`DEFAULT_HARBOR_RELAY_PORT`)
      unless the launch set a public relay base itself: the island's rank 0
      forwards that port through a Modal tunnel (``harbor_relay_tunnel_port``).
    """
    from ..rl.adapters.miles import harbor_agent

    out: dict[str, str] = {}
    if str(envs.get("HARBOR_ENV_TYPE", "")).strip().lower() == "boat" and not envs.get("BOAT_API_KEY"):
        if load_boat_token is None:
            from ..rl.harness.harbor.boat_client import load_token as load_boat_token
        out["BOAT_API_KEY"] = load_boat_token()
    if not envs.get(harbor_agent.RELAY_PUBLIC_BASE_ENV) and not envs.get(harbor_agent.RELAY_PORT_ENV):
        out[harbor_agent.RELAY_PORT_ENV] = str(DEFAULT_HARBOR_RELAY_PORT)
    return out


def harbor_relay_tunnel_port(envs: dict[str, str]) -> int | None:
    """The relay port rank 0 forwards through a Modal tunnel, or None (the launch
    gave its own ``YETO_HARBOR_GATE_RELAY_PUBLIC_BASE``, or no relay port).
    The tunnel URL names one port, so a port range is refused."""
    from ..rl.adapters.miles import harbor_agent

    raw = str(envs.get(harbor_agent.RELAY_PORT_ENV, "") or "").strip()
    if not raw or envs.get(harbor_agent.RELAY_PUBLIC_BASE_ENV):
        return None
    if not raw.isdigit() or not 0 < int(raw) <= 65535:
        raise ValueError(f"{harbor_agent.RELAY_PORT_ENV}={raw!r}: a Modal tunnel forwards one fixed port")
    return int(raw)


def rl_gae_variant(args) -> str | None:
    """``advantage.gae_variant`` of the prepared RL algorithm spec (None if absent)."""
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if getattr(args, "training_mode", "sft") != "rl" or not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    value = ((payload or {}).get("advantage") or {}).get("gae_variant")
    return None if value is None else str(value)


def compactionrl_launch_env(args, environ, harness_compacts: bool) -> dict[str, str]:
    """Rollout-worker env for CompactionRL (design D8; progress "S13 Codex 压缩接线").

    A ``cross_segment_per_sample`` spec switches ``YETO_CODEX_COMPACTIONRL=1`` on
    (and forwards ``YETO_CODEX_COMPACTIONRL_T_COMP`` from the launching
    environment when set).  Any other spec returns {} (env unchanged).
    Raises when spec, harness and the launching environment disagree
    (``compactionrl.check_rollout_compaction``)."""
    from ..rl.algos import compactionrl as crl

    variant = rl_gae_variant(args)
    raw_switch = environ.get(crl.COMPACTION_SWITCH_ENV)
    t_comp = environ.get(crl.COMPACTION_T_COMP_ENV)
    if variant is None and not raw_switch and not t_comp:
        return {}
    try:
        explicit = crl.compaction_switch(raw_switch)
        wanted = variant == crl.COMPACTED_GAE_VARIANT
        if raw_switch not in (None, "") and explicit != wanted:
            if wanted:
                raise ValueError(
                    f"{crl.COMPACTION_SWITCH_ENV}={raw_switch!r} in the launching environment "
                    f"contradicts the CompactionRL spec ({crl.COMPACTED_GAE_VARIANT})"
                )
        crl.check_rollout_compaction(
            variant, wanted or explicit, harness_compacts=harness_compacts, t_comp=t_comp
        )
    except ValueError as error:
        raise ValueError(f"CompactionRL launch: {error}") from error
    if not wanted:
        return {}
    env = {crl.COMPACTION_SWITCH_ENV: "1"}
    if t_comp:
        env[crl.COMPACTION_T_COMP_ENV] = str(t_comp).strip()
    return env


def check_compaction_launch(args, environ=None) -> None:
    """Refuse a CompactionRL launch that cannot work, before any cloud resource.

    No agent harness compacts rollouts (the Codex OpenEnv bridge was removed in
    S26 E0.5).  The Harbor path does not produce compaction segments either
    (S26 E9.5 decision): on Miles the session server records one stream per
    agent session, and an agent-side compaction (Codex auto-compact) rewrites
    the history, which the stream cannot extend; on verl the records proxy is
    refused when it compacts (``gateway.recorder``).  So any custom agent
    (``harbor_agent.run``) leaves ``harness_compacts`` False and a CompactionRL
    spec is refused here, before any cloud resource.  TEST ONLY (task 6.4 G1): the synthetic two-segment reward stands
    in for a compacting harness; it is accepted only with
    ``cross_segment_per_sample`` and without a custom agent.
    """
    from ..rl.synthetic_segments import SYNTHETIC_SEGMENTS_REWARD

    environ = os.environ if environ is None else environ
    synthetic = getattr(args, "reward_function", None) == SYNTHETIC_SEGMENTS_REWARD
    if synthetic and rl_gae_variant(args) != "cross_segment_per_sample":
        raise ValueError(f"{SYNTHETIC_SEGMENTS_REWARD} is only valid with gae_variant cross_segment_per_sample")
    harness_compacts = synthetic and getattr(args, "custom_agent_function_path", None) is None
    compactionrl_launch_env(args, environ, harness_compacts)
