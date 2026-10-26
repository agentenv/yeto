"""Per-rank time split of Miles' full-parameter export, materialize and apply.

Each Megatron rank records what its last export, materialize, prepare and
commit did in ``actor._external_full_parameter_apply_stats`` (fragments read,
copied or skipped, seconds; ``commit`` also has ``copy_seconds``, so
``seconds - copy_seconds`` is the commit's whole-model part: master
propagation, the master-to-model copy, the model/master check and the weights
backup).  :func:`report_rank_stats` reads them through ``run_plugin`` after an
island operation and prints one ``[yeto-full-parameter] rank stats`` line per
rank (S27 G-AP: the evidence for where a boundary's time goes on the trainer).

Best effort: a group without ``run_plugin``, a Miles without the stats or a
failed call prints why and never fails the island.
``YETO_FULL_PARAMETER_RANK_STATS=0`` turns the readout off.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

RANK_STATS = f"{__name__}.rank_apply_stats"
RANK_STATS_ENV = "YETO_FULL_PARAMETER_RANK_STATS"


def rank_apply_stats(actor: Any) -> dict[str, Any]:
    """Trainer-rank plugin: this rank's last full-parameter operation stats."""
    stats = getattr(actor, "_external_full_parameter_apply_stats", None)
    return {phase: dict(values) for phase, values in stats.items()} if isinstance(stats, dict) else {}


async def report_rank_stats(group: Any, operation: str, role: str) -> None:
    """Print each rank's stats of the phases ``operation`` ran."""
    if os.environ.get(RANK_STATS_ENV, "1") == "0":
        return
    run_plugin = getattr(group, "run_plugin", None)
    if run_plugin is None:
        return
    try:
        results = await run_plugin(RANK_STATS, {})
    except Exception as error:  # noqa: BLE001 - diagnostics must not stop the island
        print(f"[yeto-full-parameter] rank stats after {operation} failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return
    phases = {"apply": ("prepare", "commit"), "materialize": ("materialize",), "export": ("export", "replica_export")}[operation]
    for rank, stats in enumerate(results or ()):
        if not isinstance(stats, dict):
            continue
        shown = {phase: _rounded(stats[phase]) for phase in phases if phase in stats}
        if shown:
            print(f"[yeto-full-parameter] rank stats op={operation} role={role} rank={rank} "
                  f"{json.dumps(shown, sort_keys=True)}", file=sys.stderr, flush=True)


def _rounded(values: Any) -> Any:
    if not isinstance(values, dict):
        return values
    return {key: round(value, 3) if isinstance(value, float) else value for key, value in values.items()}
