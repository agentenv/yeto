"""Island sample roles (rl-trainer-start-stop L4.6).

An island has one sample role:

* ``both`` (default): the island generates and trains in one loop, as before;
* ``sampler``: the island only generates and writes every finished group to the
  durable sample buffer. It builds no trainer model and takes no trainer GPU. It
  loads each policy the trainer releases (:mod:`.policy_release`) and keeps
  generating with the last one while the trainer is stopped, up to a backlog cap;
* ``trainer``: the island trains on groups read from the buffer and releases
  every published policy for the samplers.

The sampler and the trainer mount the same checkpoint store (the buffer and the
releases live there). They are either separate single-island launches
(``--rl-sample-role``), or one launch of two islands, a *store-coupled pair*
(``--rl-island-sample-role 0:sampler --rl-island-sample-role 1:trainer``, L5.4):
there is no syncer, and each island runs as its own one-learner, no-sync island
(learner id 0 of 1) whose role comes from its island args (:func:`island_args`).
This module is engine-neutral; the Miles mapping is in
``adapters.miles.buffer_rollout``.
"""

from __future__ import annotations

import copy
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


def island_roles(args: Any, num_islands: int) -> dict[int, str]:
    """``--rl-island-sample-role ISLAND:ROLE`` entries as {island: role}; {} without
    them. Raises ValueError on a malformed, out-of-range or repeated entry."""
    roles: dict[int, str] = {}
    for item in getattr(args, "rl_island_sample_role", None) or ():
        island, sep, role = str(item).partition(":")
        if not sep or not island.strip().isdigit():
            raise ValueError(f"--rl-island-sample-role takes ISLAND:ROLE, got {item!r}")
        m = int(island)
        if not 0 <= m < num_islands:
            raise ValueError(f"--rl-island-sample-role {item!r}: --gpu has {num_islands} island(s)")
        if role not in (SAMPLER, TRAINER):
            raise ValueError(f"--rl-island-sample-role {item!r}: ROLE must be {SAMPLER} or {TRAINER}")
        if m in roles:
            raise ValueError(f"--rl-island-sample-role names island {m} twice")
        roles[m] = role
    return roles


def is_pair(args: Any) -> bool:
    """Whether the launch is a store-coupled sampler/trainer pair."""
    return bool(getattr(args, "rl_island_sample_role", None))


def island_args(args: Any, island: int, num_islands: int) -> Any:
    """``args`` as island ``island`` of a pair sees it: a copy with its
    ``rl_sample_role``, keeping only its own role's option (the sampler's backlog
    cap, the trainer's wait); ``args`` itself outside a pair."""
    roles = island_roles(args, num_islands)
    if not roles:
        return args
    out = copy.copy(args)
    out.rl_sample_role = roles[island]
    if out.rl_sample_role == SAMPLER:
        out.rl_sample_wait_s = None
    else:
        out.rl_sampler_max_backlog_rounds = None
    return out


def island_learner(args: Any, island: int, num_learners: int) -> tuple[int, int]:
    """``(learner id, number of learners)`` the island command carries. A pair
    island is a one-learner no-sync island: (0, 1). The launcher keeps ``island``
    for the cluster / Modal function name, the tape and the volumes."""
    return (0, 1) if is_pair(args) else (island, num_learners)


def buffer_full_parameter(args: Any) -> bool:
    """``--tuning full`` on sampler/trainer islands (single or pair, no-sync): the
    island trains every parameter without the full-parameter sync, since a buffer
    island has no syncer; the policy reaches the sampler as a store release."""
    return (getattr(args, "tuning", "lora") == "full"
            and bool(getattr(args, "rl_single_island_no_sync", False))
            and (is_pair(args) or sample_role(args) != BOTH))


def check_pair(args: Any, num_islands: int) -> None:
    """Launch checks of a store-coupled pair (nothing outside a pair): exactly two
    islands, one sampler and one trainer, ``--rl-single-island-no-sync`` (no
    syncer), no ``--rl-sample-role`` and no external learner; then every island's
    own role checks on its island args."""
    roles = island_roles(args, num_islands)
    if not roles:
        return
    if num_islands != 2 or sorted(roles.values()) != [SAMPLER, TRAINER] or len(roles) != 2:
        raise ValueError("--rl-island-sample-role: a pair is two --gpu islands, one "
                         f"{SAMPLER} and one {TRAINER} (got {num_islands} island(s), roles "
                         f"{dict(sorted(roles.items()))})")
    if sample_role(args) != BOTH:
        raise ValueError("--rl-island-sample-role replaces --rl-sample-role; give only one")
    if int(getattr(args, "external_learners", 0) or 0):
        raise ValueError("--rl-island-sample-role does not take external learners")
    if not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError("--rl-island-sample-role needs --rl-single-island-no-sync: the pair "
                         "has no syncer; each island is a one-learner no-sync island over the "
                         "shared --rl-checkpoint-store")
    for m in range(num_islands):
        check_launch(island_args(args, m, num_islands))


def check_launch(args: Any) -> None:
    """Launch-side checks of one island's role, before any cloud work. ``both``
    checks nothing (a pair checks each island with :func:`check_pair`)."""
    role = sample_role(args)
    if role == BOTH:
        for flag, dest in (("--rl-sampler-max-backlog-rounds", "rl_sampler_max_backlog_rounds"),
                           ("--rl-sample-wait-s", "rl_sample_wait_s")):
            if getattr(args, dest, None) is not None:
                raise ValueError(f"{flag} needs --rl-sample-role sampler or trainer")
        return
    if not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError(f"--rl-sample-role {role} needs --rl-single-island-no-sync: a sampler "
                         "and a trainer are separate no-sync islands over one store")
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
    """The learner flags the launcher forwards (empty for ``both``); ``args`` are
    the island args (:func:`island_args`)."""
    role = sample_role(args)
    if role == BOTH:
        return ""
    flags = f" --rl-sample-role {role}"
    if getattr(args, "rl_sampler_max_backlog_rounds", None) is not None:
        flags += f" --rl-sampler-max-backlog-rounds {int(args.rl_sampler_max_backlog_rounds)}"
    if getattr(args, "rl_sample_wait_s", None) is not None:
        flags += f" --rl-sample-wait-s {float(args.rl_sample_wait_s)!r}"
    return flags
