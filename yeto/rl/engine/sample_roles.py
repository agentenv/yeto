"""Island sample roles (rl-trainer-start-stop L4.6).

An island has one sample role:

* ``both`` (default): the island generates and trains in one loop, as before;
* ``sampler``: the island only generates and writes every finished group to the
  durable sample buffer. It builds no trainer model and takes no trainer GPU. It
  loads each policy the trainer releases (:mod:`.policy_release`) and keeps
  generating with the last one while the trainer is stopped, up to a backlog cap;
* ``trainer``: the island trains on groups read from the buffer and releases
  every published policy for the samplers;
* ``convertible`` (L7.5): a colocated LoRA island that generates into its own
  shard of the buffer, and converts to trainer in place when the head's
  scheduler selects it (``yeto.rl.engine.convertible_remote``). An island is
  convertible when it is declared so (``--rl-node-class islandM=convertible``);
  a launch whose islands are all convertible is a *convertible pool*.

The sampler and the trainer mount the same checkpoint store (the buffer and the
releases live there). They are either separate single-island launches
(``--rl-sample-role``), or one launch of two islands, a *store-coupled pair*
(``--rl-island-sample-role 0:sampler --rl-island-sample-role 1:trainer``, L5.4):
there is no syncer, and each island runs as its own one-learner, no-sync island
(learner id 0 of 1) whose role comes from its island args (:func:`island_args`).
Synced trainers (S25 C3, :mod:`.merged_release`): a launch of N >= 2 islands
with ``--rl-sample-role trainer`` and a syncer (strict or elastic sync, no
``--rl-single-island-no-sync``): every island is a trainer, learner m of N, and
reads lane m of the buffer; the island with learner id 0 releases the merged
policy. The store reaches the island as ``--rl-sample-store`` (buffer and
releases only: multi-island round cuts do not exist), the trainer count as
``--rl-sample-lanes``. The samplers are separate
no-sync launches over the same store.

This module is engine-neutral; the Miles mapping is in
``adapters.miles.buffer_rollout``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

BOTH, SAMPLER, TRAINER, CONVERTIBLE = "both", "sampler", "trainer", "convertible"
SAMPLE_ROLES = (BOTH, SAMPLER, TRAINER, CONVERTIBLE)

# groups a sampler may have ahead of the trainer's last released read position,
# in units of one round (rollout batch), when --rl-sampler-max-backlog-rounds is unset
DEFAULT_MAX_BACKLOG_ROUNDS = 4


@dataclass(frozen=True)
class SampleBufferConfig:
    """``role`` is sampler or trainer (``both`` resolves to no config).

    ``store``: the checkpoint store path that holds the buffer and the releases;
    ``max_backlog_groups``: sampler or convertible node admission cap (unconsumed
    groups);
    ``wait_s``: trainer, max seconds to wait for one full round;
    ``min_round_s``: sampler, min seconds from one round start to the next (a
    throttle, so a test can make the sampler slower than the trainer)."""

    role: str
    store: str
    max_backlog_groups: int | None = None
    wait_s: float | None = None
    min_round_s: float | None = None


def sample_role(args: Any) -> str:
    role = getattr(args, "rl_sample_role", None) or BOTH
    if role not in SAMPLE_ROLES:
        raise ValueError(f"--rl-sample-role {role!r}: expected one of {SAMPLE_ROLES}")
    return role


def is_synced_trainer(args: Any) -> bool:
    """A trainer island of a synced multi-island launch (module doc)."""
    return sample_role(args) == TRAINER and not getattr(args, "rl_single_island_no_sync", False)


def buffer_lane(args: Any) -> Any:
    """The buffer lane of a synced trainer island (island args: ``learner_id`` and
    ``--rl-sample-lanes``, the trainer count the launcher forwards); None for every
    other island."""
    from .merged_release import BufferLane

    if not is_synced_trainer(args):
        return None
    count = int(getattr(args, "rl_sample_lanes", None) or 1)
    return None if count == 1 else BufferLane(int(getattr(args, "learner_id", 0) or 0), count)


def check_synced_trainers(args: Any, num_islands: int) -> None:
    """Launch checks of synced trainers (module doc), before the per-role checks."""
    if not is_synced_trainer(args):
        return
    if int(num_islands) < 2:
        raise ValueError("--rl-sample-role trainer without --rl-single-island-no-sync is a synced "
                         f"multi-island launch: it needs at least two islands (got {num_islands})")
    if is_pair(args) or is_convertible_pool(args):
        raise ValueError("synced trainers do not combine with --rl-island-sample-role or "
                         "convertible nodes")
    if getattr(args, "tuning", "lora") != "lora":
        raise ValueError("synced trainers merge LoRA deltas: they need --tuning lora")


def check_sample_store(args: Any) -> None:
    """Island-side check: ``--rl-sample-store`` / ``--rl-sample-lanes`` belong to a
    synced trainer island, which needs both (the launcher forwards them) and no
    ``--rl-resume-store``; its learner id must name one of the lanes."""
    store, lanes = getattr(args, "rl_sample_store", None), getattr(args, "rl_sample_lanes", None)
    if not is_synced_trainer(args):
        if store is not None or lanes is not None:
            raise ValueError("--rl-sample-store / --rl-sample-lanes need --rl-sample-role trainer "
                             "with a syncer (no --rl-single-island-no-sync)")
        return
    if not store or lanes is None or int(lanes) < 2:
        raise ValueError("a synced trainer island needs --rl-sample-store and --rl-sample-lanes "
                         ">= 2 (the launcher forwards both)")
    if getattr(args, "rl_resume_store", None):
        raise ValueError("a synced trainer island takes --rl-sample-store, not --rl-resume-store")
    buffer_lane(args)  # learner id inside 0..lanes-1


def node_name(island: int) -> str:
    """The node name of island ``island`` (``--rl-node-class NAME=...``, buffer shard)."""
    return f"island{int(island)}"


def convertible_islands(args: Any) -> list[int]:
    """Indexes of the islands declared ``--rl-node-class islandM=convertible``."""
    from .convertible import NodeClass, parse_node_spec

    out = []
    for text in getattr(args, "rl_node_class", None) or ():
        spec = parse_node_spec(text)
        if spec.node_class is NodeClass.CONVERTIBLE:
            if not spec.name.startswith("island") or not spec.name[6:].isdigit():
                raise ValueError(f"--rl-node-class {text!r}: a convertible node is an island "
                                 "named islandM (M: index into --gpu)")
            out.append(int(spec.name[6:]))
    return sorted(out)


def is_convertible_pool(args: Any) -> bool:
    """Whether the launch declares convertible islands (the declaration enables the
    convertible pool: no other switch)."""
    return bool(convertible_islands(args))


def island_roles(args: Any, num_islands: int) -> dict[int, str]:
    """{island: role} from ``--rl-island-sample-role ISLAND:ROLE`` entries, or the
    convertible islands of a convertible pool; {} without them. Raises ValueError on
    a malformed, out-of-range or repeated entry."""
    if is_convertible_pool(args):
        return {m: CONVERTIBLE for m in convertible_islands(args)}
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


def is_store_coupled(args: Any) -> bool:
    """A pair or a convertible pool: no syncer, each island is its own one-learner
    no-sync island over the shared checkpoint store."""
    return is_pair(args) or is_convertible_pool(args)


def island_args(args: Any, island: int, num_islands: int) -> Any:
    """``args`` as island ``island`` of a pair sees it: a copy with its
    ``rl_sample_role``, keeping only its own role's option (the sampler's backlog
    cap, the trainer's wait; a convertible node keeps the backlog cap and the
    wait); ``args`` itself outside a pair."""
    roles = island_roles(args, num_islands)
    if not roles:
        return args
    out = copy.copy(args)
    out.rl_sample_role = roles[island]
    if out.rl_sample_role == CONVERTIBLE:
        out.rl_convertible_node = node_name(island)
        out.rl_sampler_min_round_seconds = None  # a node keeps its backlog cap
    elif out.rl_sample_role == SAMPLER:
        out.rl_sample_wait_s = None
    else:
        out.rl_sampler_max_backlog_rounds = None
        out.rl_sampler_min_round_seconds = None
    return out


def island_learner(args: Any, island: int, num_learners: int) -> tuple[int, int]:
    """``(learner id, number of learners)`` the island command carries. A pair
    island is a one-learner no-sync island: (0, 1). The launcher keeps ``island``
    for the cluster / Modal function name, the tape and the volumes."""
    return (0, 1) if is_store_coupled(args) else (island, num_learners)


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


def check_pool(args: Any, num_islands: int) -> None:
    """Launch checks of a convertible pool: at least two islands, every island
    declared convertible, no other role switch, ``--rl-single-island-no-sync``,
    LoRA on colocated islands (the trainer and the engines share the GPUs); then
    every island's own role checks."""
    islands = convertible_islands(args)
    if num_islands < 2 or islands != list(range(num_islands)):
        raise ValueError("--rl-node-class islandM=convertible: a convertible pool declares every "
                         f"--gpu island convertible, at least two (got {num_islands} island(s), "
                         f"convertible {islands}); the last rollout node is never converted")
    if getattr(args, "rl_island_sample_role", None) or sample_role(args) != BOTH:
        raise ValueError("a convertible pool sets the island roles itself; drop "
                         "--rl-island-sample-role / --rl-sample-role")
    if int(getattr(args, "external_learners", 0) or 0):
        raise ValueError("a convertible pool does not take external learners")
    if not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError("a convertible pool needs --rl-single-island-no-sync: there is no "
                         "syncer; the islands share the --rl-checkpoint-store")
    if getattr(args, "tuning", "lora") != "lora" or \
            (getattr(args, "rl_placement", None) or "colocated") != "colocated":
        raise ValueError("a convertible node trains and generates on the same GPUs: it needs "
                         "--tuning lora and colocated placement")
    for m in range(num_islands):
        check_launch(island_args(args, m, num_islands))


def check_launch(args: Any) -> None:
    """Launch-side checks of one island's role, before any cloud work. ``both``
    checks nothing (a pair checks each island with :func:`check_pair`)."""
    role = sample_role(args)
    if role == BOTH:
        for flag, dest in (("--rl-sampler-max-backlog-rounds", "rl_sampler_max_backlog_rounds"),
                           ("--rl-sampler-min-round-seconds", "rl_sampler_min_round_seconds"),
                           ("--rl-sample-wait-s", "rl_sample_wait_s")):
            if getattr(args, dest, None) is not None:
                raise ValueError(f"{flag} needs --rl-sample-role sampler or trainer")
        return
    if not getattr(args, "rl_single_island_no_sync", False) and role != TRAINER:
        raise ValueError(f"--rl-sample-role {role} needs --rl-single-island-no-sync: a sampler "
                         "is a no-sync island over the store (only trainers may sync)")
    if not getattr(args, "rl_checkpoint_store", None):
        raise ValueError(f"--rl-sample-role {role} needs --rl-checkpoint-store (the sample "
                         "buffer and the policy releases live in it)")
    if getattr(args, "rl_elastic", False):
        # --rl-elastic is placement reconfiguration (T/R cells), not the island sync:
        # synced trainers use --rl-island-scheduling elastic (or strict sync)
        raise ValueError(f"--rl-sample-role {role} does not support --rl-elastic "
                         "(placement reconfiguration)")
    if role in (TRAINER, CONVERTIBLE) and not int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1:
        raise ValueError(f"--rl-sample-role {role} trains on samples of earlier versions: "
                         "it needs --rl-max-policy-age >= 1")
    rounds = getattr(args, "rl_sampler_max_backlog_rounds", None)
    if rounds is not None and (role not in (SAMPLER, CONVERTIBLE) or int(rounds) < 1):
        raise ValueError("--rl-sampler-max-backlog-rounds needs --rl-sample-role sampler "
                         "(or a convertible pool) and a value >= 1")
    if rounds is not None and role == CONVERTIBLE and \
            int(rounds) < convertible_min_backlog_rounds(args):
        raise ValueError(f"--rl-sampler-max-backlog-rounds {int(rounds)}: a convertible pool "
                         "starts a trainer at up to --rl-max-policy-age + 1 rounds of backlog, "
                         f"so the cap must be at least {convertible_min_backlog_rounds(args)}")
    min_round_s = getattr(args, "rl_sampler_min_round_seconds", None)
    if min_round_s is not None and (role != SAMPLER or not float(min_round_s) > 0):
        raise ValueError("--rl-sampler-min-round-seconds needs --rl-sample-role sampler "
                         "and a value > 0")
    wait_s = getattr(args, "rl_sample_wait_s", None)
    if wait_s is not None and (role not in (TRAINER, CONVERTIBLE) or not float(wait_s) > 0):
        raise ValueError("--rl-sample-wait-s needs --rl-sample-role trainer and a value > 0")
    if role == CONVERTIBLE and not getattr(args, "rl_convertible_node", None):
        raise ValueError("--rl-sample-role convertible needs --rl-convertible-node NAME")


def convertible_min_backlog_rounds(args: Any) -> int:
    """The smallest backlog cap of a convertible node, in rounds: S1 starts a trainer
    at a backlog of at most ``a_hard + 1`` batches (``a_hard`` = the max policy
    age), so a lower cap would hold every node back before any trainer starts."""
    return int(getattr(args, "rl_max_policy_age", 0) or 0) + 1


def _max_backlog_rounds(args: Any, role: str) -> int | None:
    rounds = getattr(args, "rl_sampler_max_backlog_rounds", None)
    if role == SAMPLER:
        return int(rounds or DEFAULT_MAX_BACKLOG_ROUNDS)
    if role == CONVERTIBLE:
        return int(rounds or max(DEFAULT_MAX_BACKLOG_ROUNDS, convertible_min_backlog_rounds(args) + 1))
    return None


def resolve(args: Any, *, groups_per_round: int) -> SampleBufferConfig | None:
    """The island's buffer config (island side: the store is ``--rl-resume-store``,
    or ``--rl-sample-store`` on a synced trainer)."""
    role = sample_role(args)
    if role == BOTH:
        return None
    store = getattr(args, "rl_sample_store", None) or getattr(args, "rl_resume_store", None)
    if not store:
        raise ValueError(f"--rl-sample-role {role} needs the checkpoint store "
                         "(--rl-resume-store, or --rl-sample-store on a synced trainer)")
    rounds = _max_backlog_rounds(args, role)
    wait_s = getattr(args, "rl_sample_wait_s", None)
    min_round_s = getattr(args, "rl_sampler_min_round_seconds", None)
    return SampleBufferConfig(
        role=role, store=str(store),
        max_backlog_groups=None if rounds is None else int(groups_per_round) * rounds,
        wait_s=None if wait_s is None else float(wait_s),
        min_round_s=(float(min_round_s) if role == SAMPLER and min_round_s is not None
                     else None))


def island_flags(args: Any) -> str:
    """The learner flags the launcher forwards (empty for ``both``); ``args`` are
    the island args (:func:`island_args`)."""
    role = sample_role(args)
    if role == BOTH:
        return ""
    flags = f" --rl-sample-role {role}"
    if role == CONVERTIBLE:
        flags += f" --rl-convertible-node {args.rl_convertible_node}"
    if getattr(args, "rl_sampler_max_backlog_rounds", None) is not None:
        flags += f" --rl-sampler-max-backlog-rounds {int(args.rl_sampler_max_backlog_rounds)}"
    if getattr(args, "rl_sampler_min_round_seconds", None) is not None:
        flags += f" --rl-sampler-min-round-seconds {float(args.rl_sampler_min_round_seconds)!r}"
    if getattr(args, "rl_sample_wait_s", None) is not None:
        flags += f" --rl-sample-wait-s {float(args.rl_sample_wait_s)!r}"
    return flags
