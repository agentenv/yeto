"""Launch preflight: how many bytes each island must download per synced version.

The syncer broadcasts every merged fragment in f32 (``BCAST``); only the upload
(``PUSH``) takes ``--rl-sync-push-dtype``. ``--rl-sync-bcast-delta-full-every N``
sends bf16 deltas between f32 snapshots (fixed roster only; it turns peer
recovery off). An island behind a slow link (a cross-region tunnel at about
10 MB/s) that cannot download each fragment version within its local rounds
keeps training on stale fragments and contributes little: the syncer then waits
out its grace for that island on every round.

:func:`downlink_demand` estimates the per-version bytes from the model
structure (:mod:`yeto.memory_estimate`, no weights) and :func:`downlink_preflight`
prints a WARNING above :data:`WARN_FRAGMENT_BYTES`, plus one for Modal islands
placed apart from the syncer (:mod:`yeto.launch.syncer_region`, measured
15.9 MB/s across regions against 132 MB/s within one). It never refuses and never
changes an argument. Not estimated: launches without a syncer, the sampler
streaming preset, and partial-expert selections (their synced set is a subset
of the model only the island knows).
"""

from __future__ import annotations

import sys
from typing import Any

from yeto.rl.engine.selection import (
    DECOUPLED_SYNC_PRESET,
    FULL_PARAMETER_SYNC_PRESET,
    STRICT_AVG_SYNC_PRESET,
)

MIB = 1 << 20
WARN_FRAGMENT_BYTES = 256 * MIB  # ~27 s per fragment version at SLOW_LINK_BYTES_S
SLOW_LINK_BYTES_S = 10 * MIB  # a cross-region tunnel
FAST_LINK_BYTES_S = 100 * MIB  # same-region object store / VPC
F32_BYTES = 4
BF16_BYTES = 2
_SYNCED_PRESETS = (STRICT_AVG_SYNC_PRESET, DECOUPLED_SYNC_PRESET, FULL_PARAMETER_SYNC_PRESET)


def _bcast_bytes_per_param(full_every: int) -> float:
    """Mean downlink bytes per parameter: f32 snapshots, bf16 deltas in between."""
    if full_every <= 1:
        return float(F32_BYTES)
    return (F32_BYTES + BF16_BYTES * (full_every - 1)) / full_every


def _skip(reason: str) -> dict[str, Any]:
    return {"estimated": False, "reason": reason}


def downlink_demand(args, *, arch=None) -> dict[str, Any] | None:
    """The downlink record of a synced RL launch, or None when no syncer runs.

    ``arch``: a :class:`yeto.memory_estimate.ModelArch` (resolved from
    ``--model`` when None)."""
    if getattr(args, "training_mode", "sft") != "rl" or getattr(args, "rl_single_island_no_sync", False):
        return None
    preset = getattr(args, "rl_sync_preset", None)
    if preset not in _SYNCED_PRESETS:
        return None
    if getattr(args, "rl_expert_selection", None) or int(getattr(args, "expert_full_count", 0) or 0):
        return _skip("partial-expert selection: the synced subset is known to the island only")
    from yeto import memory_estimate as me

    if arch is None:
        arch, repo, _ = me.resolve_arch(str(getattr(args, "model", "")))
        if arch is None:
            return _skip(f"model structure of {repo!r} unknown (no local or cached config.json)")
    full = preset == FULL_PARAMETER_SYNC_PRESET or (getattr(args, "tuning", "lora") or "lora") == "full"
    if full:
        params = float(arch.params)
    else:
        params = me.lora_params(arch, int(getattr(args, "lora_r", 16) or 16),
                                str(getattr(args, "lora_targets", "attention") or "attention"))
    fragments = max(1, int(getattr(args, "fragments", 1) or 1))
    full_every = int(getattr(args, "rl_sync_bcast_delta_full_every", 0) or 0)
    per_param = _bcast_bytes_per_param(full_every)
    fragment_bytes = int(params * per_param / fragments)
    return {
        "estimated": True,
        "synced": "full-parameter" if full else "lora",
        "synced_params": int(params),
        "fragments": fragments,
        "bcast": f"bf16 delta, f32 every {full_every}" if full_every > 1 else "f32",
        "fragment_bytes": fragment_bytes,
        "version_bytes": int(params * per_param),
        "slow_link_s": round(fragment_bytes / SLOW_LINK_BYTES_S, 1),
        "fast_link_s": round(fragment_bytes / FAST_LINK_BYTES_S, 1),
        "warn": fragment_bytes >= WARN_FRAGMENT_BYTES,
    }


def format_warning(rec: dict[str, Any]) -> str:
    gib = rec["version_bytes"] / (1 << 30)
    lines = [
        f"[preflight] WARNING: downlink demand: each island downloads ~{rec['fragment_bytes'] / MIB:.0f} MiB "
        f"per fragment version ({rec['bcast']} BCAST, {rec['fragments']} fragments, ~{gib:.1f} GiB for "
        f"every fragment once): ~{rec['slow_link_s']:.0f} s at 10 MB/s, ~{rec['fast_link_s']:.0f} s at 100 MB/s.",
        "[preflight]   An island on a slow link that cannot fetch each version within its local rounds "
        "trains on stale fragments and the syncer waits out its grace for it every round.",
        "[preflight]   Options: run the syncer in the islands' region; "
        "--rl-sync-bcast-delta-full-every N (bf16 deltas, fixed roster only, turns peer recovery off); "
        "more --fragments spreads the bytes over more rounds (the total is unchanged).",
    ]
    return "\n".join(lines)


def downlink_preflight(args, *, out=None, arch=None) -> dict[str, Any] | None:
    """Print the WARNINGs (large per-fragment download; Modal islands placed apart
    from the syncer, :mod:`yeto.launch.syncer_region`); the manifest record."""
    rec = downlink_demand(args, arch=arch)
    if rec is None:
        return None
    stream = out or sys.stdout
    if rec.get("warn"):
        print(format_warning(rec), file=stream, flush=True)
    findings = _region_findings(args)
    for finding in findings:
        print(f"[preflight] WARNING: syncer region: {finding}", file=stream, flush=True)
    if findings:
        rec = {**rec, "region_findings": findings}
    return rec


def _region_findings(args) -> list[str]:
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.syncer_region import region_findings

    gpu = getattr(args, "gpu", None)
    if not gpu:
        return []
    try:
        specs = parse_gpu_spec(gpu)
    except ValueError:
        return []
    return region_findings(args, specs)
