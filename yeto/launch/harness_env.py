"""Launch-side island environment: the Modal sandbox secret and the CompactionRL switch."""

from __future__ import annotations

import json
import os
import re


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
    S26 E0.5).  TEST ONLY (task 6.4 G1): the synthetic two-segment reward stands
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
