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
(``--rl-island-sample-role 0:sampler --rl-island-sample-role 1:trainer``, L5.4),
or one launch of one trainer and several sampler islands (S25 C1: each sampler
writes its own shard of the buffer, :func:`buffer_shards`): there is no syncer, and each island runs as its own one-learner, no-sync island
(learner id 0 of 1) whose role comes from its island args (:func:`island_args`).
Synced trainers (S25 C3, :mod:`.merged_release`): a launch of N >= 2 islands
with ``--rl-sample-role trainer`` and a syncer (strict or elastic sync, no
``--rl-single-island-no-sync``): every island is a trainer, learner m of N, and
reads lane m of the buffer; the island with learner id 0 releases the merged
policy. The store reaches the island as ``--rl-sample-store`` (buffer and
releases only: multi-island round cuts do not exist), the trainer count as
``--rl-sample-lanes``. The samplers are separate no-sync launches over the
same store, or (S26 E9.4) islands of the same launch, a *synced group*
(``--rl-sample-role trainer --rl-island-sample-role M:sampler``, a syncer): the
listed islands are no-sync samplers (learner 0 of 1, like a pair's samplers)
and every other island is a synced trainer, learner k of K in island order
(:func:`synced_group_learners`). With the full-parameter preset (its
asynchronous rounds only) the publisher releases full weights at the merged
version of the fragments (``full_parameter_sync.merged_version``), and the
samplers run ``--tuning full``.

This module is engine-neutral; the Miles mapping is in
``adapters.miles.buffer_rollout``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .data_split import DataSlot  # noqa: F401 - re-exported (S26 E9.3 callers)
from .selection import FULL_PARAMETER_SYNC_PRESET

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
    throttle, so a test can make the sampler slower than the trainer);
    ``fused_stage``: LoRA sampler, check the release digest while copying the
    adapter to local disk (``--rl-sampler-fused-stage``, S26 E5);
    ``release_legacy``: trainer or convertible node, reload the store before every
    policy release and commit LATEST separately (``--rl-release-reload-latest``;
    ``PolicyReleaseStore.cached_latest``)."""

    role: str
    store: str
    max_backlog_groups: int | None = None
    wait_s: float | None = None
    min_round_s: float | None = None
    fused_stage: bool = False
    release_legacy: bool = False


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
    if is_convertible_pool(args):
        raise ValueError("synced trainers do not combine with convertible nodes")
    if getattr(args, "tuning", "lora") != "lora":
        raise ValueError("synced trainers merge LoRA deltas or run the full-parameter preset: "
                         "they need --tuning lora")
    if getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET and \
            getattr(args, "rl_sync_full_parameter_rounds", "sweep") != "async":
        # The strict sweep publishes in lockstep (policy age 0); a synced trainer
        # trains on buffered samples of earlier versions.
        raise ValueError("full-parameter synced trainers need --rl-sync-full-parameter-rounds "
                         "async (the strict sweep takes no --rl-max-policy-age)")
    if is_synced_group(args):
        check_synced_group(args, num_islands)


def check_synced_group(args: Any, num_islands: int) -> None:
    """Launch checks of a synced group (module doc): ``--rl-island-sample-role``
    names only samplers, at least two trainers remain, and the samplers split the
    trainers' rounds evenly; then every island's own role checks."""
    listed = _listed_roles(args, num_islands)
    if any(role != SAMPLER for role in listed.values()):
        raise ValueError("--rl-island-sample-role with --rl-sample-role trainer names the sampler "
                         f"islands only (every other island is a synced trainer); got {listed}")
    trainers = len(synced_group_learners(args, num_islands))
    if trainers < 2:
        raise ValueError(f"a synced group needs at least two trainer islands (got {trainers}); "
                         "one trainer with samplers is a pair (--rl-single-island-no-sync)")
    samplers = len(listed)
    if (trainers * int(getattr(args, "total_steps", 0) or 0)) % samplers:
        raise ValueError(f"--total-steps {args.total_steps}: {samplers} sampler islands share the "
                         f"rounds of {trainers} trainers; trainers x total steps must be a multiple "
                         "of the sampler count")
    if int(getattr(args, "external_learners", 0) or 0):
        raise ValueError("a synced group does not take external learners")
    if getattr(args, "rl_trainer_scheduler", None) not in (None, "off"):
        raise ValueError("--rl-trainer-scheduler does not combine with a synced group")
    for m in range(num_islands):
        check_launch(island_args(args, m, num_islands))


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
    roles = _listed_roles(args, num_islands)
    if is_synced_group(args):
        # the listed islands are samplers; every other island is a synced trainer
        return {m: roles.get(m, TRAINER) for m in range(int(num_islands))}
    return roles


def _listed_roles(args: Any, num_islands: int) -> dict[int, str]:
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
    """Whether the launch is a store-coupled group: one trainer island and one or
    more sampler islands (``--rl-island-sample-role`` without a synced trainer
    role; "pair" for short)."""
    return bool(getattr(args, "rl_island_sample_role", None)) and not is_synced_group(args)


def is_synced_group(args: Any) -> bool:
    """S26 E9.4: one launch of synced trainers and their sampler islands
    (``--rl-sample-role trainer`` with a syncer and ``--rl-island-sample-role
    M:sampler``; module doc)."""
    return bool(getattr(args, "rl_island_sample_role", None)) and is_synced_trainer(args)


def synced_group_learners(args: Any, num_islands: int) -> dict[int, int]:
    """{trainer island: learner id} of a synced group: the trainers in island order
    are learners 0..K-1 (learner 0 releases the merged policy); {} otherwise."""
    if not is_synced_group(args):
        return {}
    trainers = [m for m, role in sorted(island_roles(args, num_islands).items()) if role == TRAINER]
    return {m: k for k, m in enumerate(trainers)}


def syncer_learner_count(args: Any, num_islands: int) -> int:
    """The learners the syncer waits for among the launch's islands: the trainers
    of a synced group, every island otherwise (a no-sync launch has no syncer)."""
    learners = synced_group_learners(args, num_islands)
    return len(learners) if learners else int(num_islands)


def sampler_islands(args: Any, num_islands: int) -> list[int]:
    """The sampler islands of a pair, sorted."""
    return sorted(m for m, role in island_roles(args, num_islands).items() if role == SAMPLER)


def buffer_shards(args: Any, num_islands: int) -> list[str] | None:
    """The buffer shards of a pair with several sampler islands (S25 C1): one per
    sampler island, named by its node name; None with one sampler (it writes the
    index itself, as before)."""
    samplers = sampler_islands(args, num_islands)
    return [node_name(m) for m in samplers] if len(samplers) > 1 else None


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
    if is_synced_group(args):
        return _synced_group_island_args(args, island, num_islands, roles)
    out = copy.copy(args)
    out.rl_sample_role = roles[island]
    if out.rl_sample_role in (SAMPLER, TRAINER):
        out.rl_sampler_node = node_name(island) if out.rl_sample_role == SAMPLER else None
        out.rl_buffer_shards = buffer_shards(args, num_islands)
        samplers = len(sampler_islands(args, num_islands))
        # each sampler island draws its own slot of the data (S26 E9.3)
        out.rl_sampler_data_slot = (DataSlot(sampler_islands(args, num_islands).index(island), samplers).env()
                                    if out.rl_sample_role == SAMPLER and samplers > 1 else None)
        if out.rl_sample_role == SAMPLER and getattr(args, "rl_sampler_spot", False):
            # spot billing without the legacy --spot volume: the buffer is the sampler's
            # durable state (S25 C1)
            out.spot = True
            out.spot_without_volume = True
        if out.rl_sample_role == SAMPLER and samplers > 1:
            # the trainer trains --total-steps rounds; the samplers share them, so each
            # one reaches its data end (check_pair needs an even split)
            out.total_steps = int(args.total_steps) // samplers
            if getattr(args, "rl_total_fragment_steps", None) is not None:
                out.rl_total_fragment_steps = out.total_steps * int(getattr(args, "fragments", 1) or 1)
    if out.rl_sample_role == CONVERTIBLE:
        out.rl_convertible_node = node_name(island)
        out.rl_sampler_min_round_seconds = None  # a node keeps its backlog cap
    elif out.rl_sample_role == SAMPLER:
        out.rl_sample_wait_s = None
    else:
        out.rl_sampler_max_backlog_rounds = None
        out.rl_sampler_min_round_seconds = None
    if out.rl_sample_role != SAMPLER:
        out.rl_sampler_fused_stage = False
    if out.rl_sample_role not in (TRAINER, CONVERTIBLE):
        out.rl_release_reload_latest = False
    return out


def _synced_group_island_args(args: Any, island: int, num_islands: int, roles: dict[int, str]) -> Any:
    """Island args of a synced group. A trainer keeps the synced args (its lane and
    the sampler shards); a sampler becomes a pair-style no-sync sampler over the
    same store: no syncer and no island pool options, its node name and data
    slot, and its share of the trainers' rounds."""
    from ...launch.syncer import _ISLAND_POOL_PARAMS

    samplers = sampler_islands(args, num_islands)
    out = copy.copy(args)
    out.rl_island_sample_role = None  # the island is no group: its role is below
    out.rl_sample_role = roles[island]
    out.rl_buffer_shards = buffer_shards(args, num_islands)
    out.rl_synced_trainers = len(synced_group_learners(args, num_islands))
    if out.rl_sample_role == TRAINER:
        out.rl_sampler_node = None
        out.rl_sampler_data_slot = None
        out.rl_sampler_max_backlog_rounds = None
        out.rl_sampler_min_round_seconds = None
        return out
    out.rl_single_island_no_sync = True
    for name, _flag in _ISLAND_POOL_PARAMS:
        setattr(out, name, None)
    out.rl_syncer_epoch = None
    out.rl_sampler_node = node_name(island)
    out.rl_sampler_data_slot = (DataSlot(samplers.index(island), len(samplers)).env()
                                if len(samplers) > 1 else None)
    out.rl_sample_wait_s = None
    out.total_steps = out.rl_synced_trainers * int(args.total_steps) // len(samplers)
    if getattr(args, "rl_total_fragment_steps", None) is not None:
        out.rl_total_fragment_steps = out.total_steps * int(getattr(args, "fragments", 1) or 1)
    return out


def island_learner(args: Any, island: int, num_learners: int,
                   num_islands: int | None = None) -> tuple[int, int]:
    """``(learner id, number of learners)`` the island command carries. A pair
    island is a one-learner no-sync island: (0, 1); so is a synced group's sampler,
    and its trainers are learners 0..K-1 (:func:`synced_group_learners`). The
    launcher keeps ``island`` for the cluster / Modal function name, the tape and
    the volumes."""
    if is_store_coupled(args):
        return 0, 1
    learners = synced_group_learners(args, num_learners if num_islands is None else num_islands)
    if learners:
        return (learners[island], len(learners)) if island in learners else (0, 1)
    return island, num_learners


def buffer_full_parameter(args: Any) -> bool:
    """``--tuning full`` on a no-sync island: the island trains every parameter
    without the full-parameter sync, since it has no syncer. A sampler/trainer
    island (single or pair) passes the policy to the sampler as a store release;
    a single island with both roles (S26 E11) updates its own engines in place."""
    return (getattr(args, "tuning", "lora") == "full"
            and bool(getattr(args, "rl_single_island_no_sync", False)))


def check_pair(args: Any, num_islands: int) -> None:
    """Launch checks of a store-coupled pair (nothing outside a pair): every
    island has a role, exactly one trainer and at least one sampler,
    ``--rl-single-island-no-sync`` (no syncer), no ``--rl-sample-role`` and no
    external learner; several samplers split ``--total-steps`` evenly (each
    generates its share of the trainer's rounds) and take no trainer scheduler
    (the head reads no sampler shard of a shared directory); then every island's own role checks
    on its island args."""
    roles = island_roles(args, num_islands)
    if not roles:
        return
    values = list(roles.values())
    if len(roles) != num_islands or values.count(TRAINER) != 1 or values.count(SAMPLER) < 1:
        raise ValueError("--rl-island-sample-role: a pair gives every --gpu island a role, "
                         f"at least one {SAMPLER} and exactly one {TRAINER} (got {num_islands} "
                         f"island(s), roles {dict(sorted(roles.items()))})")
    samplers = values.count(SAMPLER)
    if samplers > 1 and int(getattr(args, "total_steps", 0) or 0) % samplers:
        raise ValueError(f"--total-steps {args.total_steps}: {samplers} sampler islands share the "
                         "trainer's rounds; it must be a multiple of the sampler count")
    if getattr(args, "rl_sampler_spot", False) and getattr(args, "spot", False):
        raise ValueError("--rl-sampler-spot bills only the samplers as spot; it does not take --spot")
    if samplers > 1 and getattr(args, "rl_trainer_scheduler", None) not in (None, "off"):
        raise ValueError("--rl-trainer-scheduler with several sampler islands is not supported "
                         "(the head does not read the sampler shards); drop the scheduler")
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
        if getattr(args, "rl_sampler_fused_stage", False):
            raise ValueError("--rl-sampler-fused-stage needs --rl-sample-role sampler")
        if getattr(args, "rl_release_reload_latest", False):
            raise ValueError("--rl-release-reload-latest needs --rl-sample-role trainer")
        return
    if not getattr(args, "rl_single_island_no_sync", False) and role != TRAINER:
        raise ValueError(f"--rl-sample-role {role} needs --rl-single-island-no-sync: a sampler "
                         "is a no-sync island over the store (only trainers may sync)")
    if not getattr(args, "rl_checkpoint_store", None):
        raise ValueError(f"--rl-sample-role {role} needs --rl-checkpoint-store (the sample "
                         "buffer and the policy releases live in it)")
    if getattr(args, "rl_elastic", False):
        # --rl-elastic is placement reconfiguration (T/R cells), not the island sync:
        # synced trainers use an island pool (or strict sync)
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
    if getattr(args, "rl_sampler_fused_stage", False) and (
            role != SAMPLER or (getattr(args, "tuning", None) or "lora") != "lora"):
        raise ValueError("--rl-sampler-fused-stage needs --rl-sample-role sampler and "
                         "--tuning lora (it fuses the LoRA adapter copy with its digest)")
    if getattr(args, "rl_release_reload_latest", False) and role not in (TRAINER, CONVERTIBLE):
        raise ValueError("--rl-release-reload-latest needs --rl-sample-role trainer "
                         "(or a convertible pool): only a trainer writes policy releases")
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
                     else None),
        fused_stage=role == SAMPLER and bool(getattr(args, "rl_sampler_fused_stage", False)),
        release_legacy=role in (TRAINER, CONVERTIBLE)
        and bool(getattr(args, "rl_release_reload_latest", False)))


def island_env(args: Any) -> dict[str, str]:
    """The environment a pair's sampler island gets from the launcher ({} for
    every other island; ``args`` are the island args): the reclaim marker path,
    so the Modal runner turns the reclaim signal into that file and the sampler
    leaves without being signalled (S25 C1)."""
    if sample_role(args) != SAMPLER or not getattr(args, "rl_sampler_node", None):
        return {}
    from .sampler_resume import DEFAULT_RECLAIM_MARKER, RECLAIM_MARKER_ENV

    return {RECLAIM_MARKER_ENV: DEFAULT_RECLAIM_MARKER}


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
    if role == SAMPLER and getattr(args, "rl_sampler_fused_stage", False):
        flags += " --rl-sampler-fused-stage"
    if role in (TRAINER, CONVERTIBLE) and getattr(args, "rl_release_reload_latest", False):
        flags += " --rl-release-reload-latest"
    if getattr(args, "rl_sampler_node", None):
        flags += f" --rl-sampler-node {args.rl_sampler_node}"
    if getattr(args, "rl_buffer_shards", None):
        flags += f" --rl-buffer-shards {','.join(args.rl_buffer_shards)}"
    if getattr(args, "rl_sampler_data_slot", None):
        flags += f" --rl-sampler-data-slot {args.rl_sampler_data_slot}"
    return flags

