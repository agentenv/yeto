"""Test-only driver loop for :class:`yeto.rl.bridge.StrictRlBridge`.

Production islands run ``yeto.rl.engine.driver.IslandDriver``, which drives
the bridge through ``engine.bridges.StrictAvgSync`` at each round boundary.
The self-driven loop below is what the removed legacy engine used; the
protocol tests keep it to exercise the bridge against a real syncer with a
fake island (rl-code-quality-cleanup 2.3).  The fake island runtime adds
``run_local_round`` and ``export_local_policy`` to the bridge's
``IslandRuntime``; ``shutdown`` is called once the loop ends.
"""

from __future__ import annotations

import sys
import time

from yeto.protocol import PullRequest
from yeto.rl.core import CanonicalLoraState, StrictRlInvariantError


def run_round(bridge, permit: PullRequest) -> None:
    """Train one local round on the fake island and submit it for ``permit``."""

    base = bridge.current
    if base is None or permit.global_step != base.policy_version + 1:
        raise RuntimeError("RL attempted a local round without its exact base")
    stats = bridge.runtime.run_local_round(
        expected_policy_version=base.policy_version,
        groups=bridge.config.groups_per_round,
        samples_per_group=bridge.config.samples_per_group,
        optimizer_steps=bridge.config.local_optimizer_steps,
    )
    if (
        stats.island_id != bridge.config.learner_id
        or stats.base_policy_version != base.policy_version
        or stats.local_round_id != permit.global_step
    ):
        raise RuntimeError("Miles returned LocalRoundStats for a different round")
    bridge.submit_local_state(permit, base, bridge.runtime.export_local_policy(), stats)


def run_strict_bridge(bridge) -> CanonicalLoraState:
    """Run rounds until the syncer finalizes; return the final policy."""

    try:
        bridge.start()
        while True:
            bridge.client.check_health()
            if bridge.client.finalizing.is_set():
                return bridge._finalize()
            progressed = bridge._drain_messages()
            permit = bridge._ready_permit()
            if permit is not None:
                run_round(bridge, permit)
                progressed = True
            if not progressed:
                time.sleep(0.05)
    except StrictRlInvariantError as error:
        bridge._append_event(
            {
                "event": "rl_strict_failure",
                "metric": error.metric,
                "value": 1,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        print(f"[yeto-rl-strict-failure] {error.metric}: {error}", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            bridge.runtime.shutdown()
        finally:
            bridge.client.close()
