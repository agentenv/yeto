"""Island sample roles (rl-trainer-start-stop L4.6).

An island has one sample role:

* ``both`` (default): the island generates and trains in one loop, as before;
* ``sampler``: the island only generates and writes every finished group to the
  durable sample buffer. It builds no trainer model and takes no trainer GPU. It
  loads each policy the trainer releases (:mod:`.policy_release`) and keeps
  generating with the last one while the trainer is stopped, up to a backlog cap;
* ``trainer``: the island trains on groups read from the buffer and releases
  every published policy for the samplers.

The sampler and the trainer are separate single-island launches that mount the
same checkpoint store (the buffer and the releases live there). This module is
engine-neutral; the Miles mapping is in ``adapters.miles.buffer_rollout``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BOTH, SAMPLER, TRAINER = "both", "sampler", "trainer"
SAMPLE_ROLES = (BOTH, SAMPLER, TRAINER)

# groups a sampler may have ahead of the trainer's last released read position,
# in units of one round (rollout batch), when --rl-sampler-max-backlog-rounds is unset
DEFAULT_MAX_BACKLOG_ROUNDS = 4


@dataclass(frozen=True)
class SampleBufferConfig:
    """``role`` is sampler or trainer (``both`` resolves to no config).

    ``store``: the checkpoint store path that holds the buffer and the releases;
    ``max_backlog_groups``: sampler admission cap (unconsumed groups);
    ``wait_s``: trainer, max seconds to wait for one full round."""

    role: str
    store: str
    max_backlog_groups: int | None = None
    wait_s: float | None = None


def sample_role(args: Any) -> str:
    role = getattr(args, "rl_sample_role", None) or BOTH
    if role not in SAMPLE_ROLES:
        raise ValueError(f"--rl-sample-role {role!r}: expected one of {SAMPLE_ROLES}")
    return role


def check_launch(args: Any) -> None:
    """Launch-side checks, before any cloud work. ``both`` checks nothing."""
    role = sample_role(args)
    if role == BOTH:
        for flag, dest in (("--rl-sampler-max-backlog-rounds", "rl_sampler_max_backlog_rounds"),
                           ("--rl-sample-wait-s", "rl_sample_wait_s")):
            if getattr(args, dest, None) is not None:
                raise ValueError(f"{flag} needs --rl-sample-role sampler or trainer")
        return
    if not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError(f"--rl-sample-role {role} needs --rl-single-island-no-sync: a sampler "
                         "and a trainer are separate single-island launches over one store")
    if not getattr(args, "rl_checkpoint_store", None):
        raise ValueError(f"--rl-sample-role {role} needs --rl-checkpoint-store (the sample "
                         "buffer and the policy releases live in it)")
    if getattr(args, "rl_elastic", False):
        raise ValueError(f"--rl-sample-role {role} does not support --rl-elastic")
    if role == TRAINER and not int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1:
        raise ValueError("--rl-sample-role trainer trains on samples of earlier versions: "
                         "it needs --rl-max-policy-age >= 1")
    rounds = getattr(args, "rl_sampler_max_backlog_rounds", None)
    if rounds is not None and (role != SAMPLER or int(rounds) < 1):
        raise ValueError("--rl-sampler-max-backlog-rounds needs --rl-sample-role sampler "
                         "and a value >= 1")
    wait_s = getattr(args, "rl_sample_wait_s", None)
    if wait_s is not None and (role != TRAINER or not float(wait_s) > 0):
        raise ValueError("--rl-sample-wait-s needs --rl-sample-role trainer and a value > 0")


def resolve(args: Any, *, groups_per_round: int) -> SampleBufferConfig | None:
    """The island's buffer config (island side: the store is ``--rl-resume-store``)."""
    role = sample_role(args)
    if role == BOTH:
        return None
    store = getattr(args, "rl_resume_store", None)
    if not store:
        raise ValueError(f"--rl-sample-role {role} needs the checkpoint store "
                         "(--rl-resume-store on the island)")
    rounds = getattr(args, "rl_sampler_max_backlog_rounds", None)
    wait_s = getattr(args, "rl_sample_wait_s", None)
    return SampleBufferConfig(
        role=role, store=str(store),
        max_backlog_groups=(int(groups_per_round) * int(rounds or DEFAULT_MAX_BACKLOG_ROUNDS)
                            if role == SAMPLER else None),
        wait_s=None if wait_s is None else float(wait_s))


def island_flags(args: Any) -> str:
    """The learner flags the launcher forwards (empty for ``both``)."""
    role = sample_role(args)
    if role == BOTH:
        return ""
    flags = f" --rl-sample-role {role}"
    if getattr(args, "rl_sampler_max_backlog_rounds", None) is not None:
        flags += f" --rl-sampler-max-backlog-rounds {int(args.rl_sampler_max_backlog_rounds)}"
    if getattr(args, "rl_sample_wait_s", None) is not None:
        flags += f" --rl-sample-wait-s {float(args.rl_sample_wait_s)!r}"
    return flags
