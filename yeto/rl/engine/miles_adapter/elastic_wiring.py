"""E1 wiring for ``compose_island`` (rl-infra-spec 3.2-3.7).

:func:`build_elastic` turns a resources manifest (1.6 / PR #66 ``resources``
block: ``configs`` and ``edges``) plus a capability attestation into the
objects ``compose_island(elastic=...)`` passes to the driver: the
:class:`~..controller.IslandController`, the :class:`~..ledger.BatchLedger`,
the startup-declared rollout cells and the pool GPUs. Nothing here starts,
stops or leases anything; without ``elastic`` the island is unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ElasticWiring:
    controller: Any  # IslandController
    ledger: Any  # BatchLedger
    declared_cells: tuple[str, ...]
    pool_gpus: tuple[str, ...] | None = None
    track_timeout_s: float = 600.0


def build_elastic(
    *,
    state_dir: str | Path,
    resources: dict[str, Any] | str | Path,
    attestation: str | Path | None,
    profile: Any,
    initial_config: str,
    runtime_fingerprint: str,
    declared_cells: Sequence[str],
    pool_gpus: Sequence[str] | None = None,
    timeouts: Any = None,
    on_watchdog: Any = None,
) -> ElasticWiring:
    """``on_watchdog(tx_id, phase)`` runs on the watchdog thread when the absolute
    transaction deadline passes while a step is still blocked. It is the only
    hook that can act on a blocked step (e.g. kill the target generation's
    SGLang processes by PID group); none is wired by default, so a blocked
    engine call is NOT bounded by the deadline (rl-infra-spec 3.7 limitation).
    """
    from yeto.rl.elastic_benchmark.capabilities import load_attestation, parse_configs

    from ..controller import CommandInbox, IslandController, Timeouts
    from ..ledger import BatchLedger

    if not isinstance(resources, dict):
        resources = json.loads(Path(resources).read_text(encoding="utf-8"))
    configs = parse_configs(resources)
    state = Path(state_dir)
    controller = IslandController(
        state_dir=state,
        configs=configs,
        attestation=load_attestation(Path(attestation) if attestation else None),
        profile=profile,
        initial_config=initial_config,
        runtime_fingerprint=runtime_fingerprint,
        timeouts=timeouts or Timeouts(),
        inbox=CommandInbox(state / "inbox"),
        on_watchdog=on_watchdog,
    )
    return ElasticWiring(
        controller=controller,
        ledger=BatchLedger(state),
        declared_cells=tuple(str(c) for c in declared_cells),
        pool_gpus=None if pool_gpus is None else tuple(pool_gpus),
    )
