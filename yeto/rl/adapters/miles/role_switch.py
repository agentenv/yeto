"""Trainer phase of :mod:`yeto.rl.engine.role_switch` on the Miles backend.

The island that was an engine sampler becomes a Miles trainer island in the same
container: :func:`subprocess_trainer` runs the ``role_switch.trainer.command``
argv of the spec (the learner command the launcher builds for a trainer island,
``python -m yeto.rl.adapters.miles.island_entry ... --rl-sample-role trainer
--rl-single-island-no-sync --tuning full --rl-model-recipe kimi_k31 ...``) as a
child process and waits for it. The trainer reads the buffer the sampler phase
wrote (every island's shard, merged by its reader) from the newest cut's
watermark, trains with ``--rl-max-policy-age`` gating the staleness, and releases
whole checkpoints into the store (``ReleasingPublisher``, manifest with the
expert selection sha). When the child ends, the release store's newest record
is the phase's result.

The environment of the child carries the role record (:data:`NODE_ENV`,
:data:`INCARNATION_ENV`, :data:`WATERMARK_ENV`) for the island's own tape; the
island entry reads none of them to decide anything (the buffer watermark comes
from the cut it restores, as on every trainer island).

Not run on a GPU yet.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from typing import Any

from yeto.rl.engine.role_switch import (
    RoleState,
    RoleSwitchError,
    RoleSwitchPlan,
    TrainerContext,
    TrainerPhaseReport,
)

NODE_ENV = "YETO_ROLE_SWITCH_NODE"
INCARNATION_ENV = "YETO_ROLE_SWITCH_INCARNATION"
WATERMARK_ENV = "YETO_ROLE_SWITCH_WATERMARK"
TRAINER_ARGV_MARKERS = ("--rl-sample-role", "trainer")


def check_trainer_command(command: Any) -> list[str]:
    """The argv of a trainer island: a non-empty list of strings naming the trainer
    role and no sync (the K3.1 trainer island form, GPU-A report §4)."""
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(t, str) for t in command):
        raise RoleSwitchError("role_switch.trainer.command must be a non-empty argv list")
    argv = [str(t) for t in command]
    if "--rl-sample-role" in argv and argv[argv.index("--rl-sample-role") + 1] != "trainer":
        raise RoleSwitchError("role_switch.trainer.command must run the island as --rl-sample-role trainer")
    return argv


def trainer_environ(state: RoleState, base: Mapping[str, str], extra: Mapping[str, Any] | None = None
                    ) -> dict[str, str]:
    env = {**base, NODE_ENV: state.node, INCARNATION_ENV: str(state.incarnation),
           WATERMARK_ENV: str(state.consumed_watermark)}
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    return env


def report_from_releases(releases: Any, *, before: Any, watermark: int | None) -> TrainerPhaseReport:
    """The phase result from the release store: the newest record if it is newer
    than ``before`` (the record before the phase), else no release."""
    latest = releases.latest()
    if latest is None or (before is not None and int(latest.seq) <= int(before.seq)):
        return TrainerPhaseReport(policy_version=0 if before is None else int(before.version),
                                  released_seq=None if before is None else int(before.seq),
                                  release_manifest=None if before is None else getattr(before, "manifest", None),
                                  consumed_watermark=watermark)
    selected = getattr(latest, "buffer_selected", None)
    return TrainerPhaseReport(policy_version=int(latest.version), released_seq=int(latest.seq),
                              release_manifest=getattr(latest, "manifest", None),
                              consumed_watermark=watermark if selected is None else int(selected))


def subprocess_trainer(state: RoleState, plan: RoleSwitchPlan, context: TrainerContext, *,
                       run: Callable[..., Any] = subprocess.run) -> TrainerPhaseReport:
    """Run the trainer command to its end; the release store tells what it released."""
    argv = check_trainer_command(plan.trainer.get("command"))
    before = context.releases.latest()
    env = trainer_environ(state, os.environ if not context.environ else context.environ, plan.trainer.get("env"))
    print(f"[role-switch] node {state.node}: trainer incarnation {state.incarnation} from watermark "
          f"{state.consumed_watermark} (buffer end {state.buffer_end}, shards {dict(state.shard_ends)})", flush=True)
    completed = run(argv, env=env, check=False)
    code = int(getattr(completed, "returncode", 1))
    if code != 0:
        raise RoleSwitchError(f"trainer incarnation {state.incarnation} exited with code {code}")
    report = report_from_releases(context.releases, before=before, watermark=None)
    if report.released_seq is None or (before is not None and report.released_seq == before.seq):
        raise RoleSwitchError(f"trainer incarnation {state.incarnation} ended without a new release")
    return report
