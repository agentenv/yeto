"""Island sample roles.

An island has one sample role:

* ``both`` (default): the island generates and trains in one loop, as before;
* ``sampler``: the island only generates and writes every finished group to the
  durable sample buffer. It builds no trainer model and takes no trainer GPU. It
  loads each policy the trainer releases (:mod:`.policy_release`) and keeps
  generating with the last one while the trainer is stopped, up to a backlog cap;
* ``trainer``: the island trains on groups read from the buffer and releases
  every published policy for the samplers;
* ``convertible``: a colocated LoRA island that generates into its own
  shard of the buffer, and converts to trainer in place when the head's
  scheduler selects it (``yeto.rl.engine.convertible_remote``). An island is
  convertible when it is declared so (``--rl-node-class islandM=convertible``);
  a launch whose islands are all convertible is a *convertible pool*.

The sampler and the trainer mount the same checkpoint store (the buffer and the
releases live there). They are either separate single-island launches
(``--rl-sample-role``), or one launch of two islands, a *store-coupled pair*
(``--rl-island-sample-role 0:sampler --rl-island-sample-role 1:trainer``),
or one launch of one trainer and several sampler islands (each sampler
writes its own shard of the buffer, :func:`buffer_shards`): there is no syncer, and each island runs as its own one-learner, no-sync island
(learner id 0 of 1) whose role comes from its island args (:func:`island_args`).
Synced trainers (:mod:`.merged_release`): a launch of N islands with
``--rl-sample-role trainer`` and a syncer (strict or elastic sync, no
``--rl-single-island-no-sync``): every island is a trainer, learner m of N, and
reads lane m of the buffer; the island with learner id 0 releases the merged
policy. LoRA synced trainers need N >= 2 (they merge LoRA deltas); full-parameter
synced trainers (``--tuning full`` with the full-parameter preset, the K3.1
recipe: :func:`synced_full_parameter_trainer`) merge full-parameter
pseudo-gradients and run with N >= 1 (N = 1: learners 1 / quorum 1, the merge
is the identity). The store reaches the island as ``--rl-sample-store`` (buffer and
releases only: multi-island round cuts do not exist), the trainer count as
``--rl-sample-lanes``. The samplers are separate no-sync launches over the
same store, or islands of the same launch, a *synced group*
(``--rl-sample-role trainer --rl-island-sample-role M:sampler``, a syncer): the
listed islands are no-sync samplers (learner 0 of 1, like a pair's samplers)
and every other island is a synced trainer, learner k of K in island order
(:func:`synced_group_learners`). With the full-parameter preset (its
asynchronous rounds only) the publisher releases full weights at the merged
version of the fragments (``full_parameter_sync.merged_version``), and the
samplers run ``--tuning full``.

Samplers that follow their own trainer (``--rl-sampler-follow trainer``, on a backend
that releases per learner: ``per_learner_release``, :func:`releases_per_learner`):
in a synced group, sampler j (in island order) feeds trainer j % K only. Each
trainer and its samplers form one pair inside the group: they share the store
sub-directory ``trainer<k>`` (:func:`trainer_substore`; the buffer and the
releases), the trainer reads that whole buffer (its own lane, 0 of 1) and
releases every local policy version for its samplers
(``merged_release`` module doc). The trainers still merge through the syncer.

External sampler pools that follow their own trainer (``--rl-sampler-follow
trainer`` with ``--rl-island-buffer-shards K:A,B``, no ``--rl-island-sample-role``):
every island of the launch is a synced trainer, learner k of K in island order,
and its samplers are pools outside the launch (engine samplers on any cloud,
``yeto run up``) that read and write its sub-directory ``trainer<k>``. The flag
names, per trainer island, the buffer shards its pools write
(:func:`island_buffer_shards`); the trainer reads that whole buffer (its own
lane) as in a synced group. ``--rl-island-release-every K:N`` /
``--rl-island-max-policy-age K:N`` give one trainer its own release cadence and
policy-age limit (:func:`external_pool_island_args`). The sub-directory number
is the learner id in every follow-trainer launch (:func:`trainer_substore`).

Sampler islands and external pools together (a *mixed group*: a synced group
with ``--rl-sampler-follow trainer`` and ``--rl-island-buffer-shards K:A,B``;
:func:`is_mixed_group`): a trainer reads the shards of the sampler islands bound
to it (each writes the shard named by its node name, ``island<M>``) and the shards
of the external pools of its ``K:A,B`` entry, all in its sub-directory
``trainer<k>``. In a mixed group the ``K`` of ``--rl-island-buffer-shards`` /
``--rl-island-release-every`` / ``--rl-island-max-policy-age`` is the trainer's
learner id (0..K-1 in island order, sampler islands skipped), not its island index;
without sampler islands the two are the same. ``--rl-island-sampler-follows M:K``
binds sampler island M to the trainer with learner id K (every sampler island
named once); without it sampler j feeds trainer j % K as in a synced group. The
binding flag alone (no pool) takes the same form: each trainer reads the shards of
its own sampler islands. A trainer may have only sampler islands or only pools,
not neither. The sampler
islands generate until their own trainer ended (full-parameter or decoupled
preset), like the pools (:func:`check_mixed_group`). A sampler island draws its
rows from its data slot among the launch's sampler islands, or, with
``--rl-island-dispense-store M:STORE`` and ``--rl-store-dispense JSON`` (the pools'
``prompts.dispense``), claims them from the store ledger under STORE that the pools
claim from (an island fleet's ``<store>/fleet/dispense``), under its shard name
``island<M>``, so no prompt goes to a pool and a sampler island in the same pass
(:func:`check_store_dispense`; ``adapters.miles.task_dispense``).

Decoupled groups (``--rl-sync-preset decoupled``, ``per_learner_release``): a decoupled trainer
has one version per fragment, merged at different times, so its group needs
``--rl-sampler-follow trainer`` (:func:`check_decoupled_group`); its samplers
run as LoRA no-sync samplers (:func:`_as_lora_sampler`) and, like those of a
full-parameter group, generate until their trainer ended.

How long a full-parameter synced group's samplers run (``per_learner_release``): its trainers
run asynchronous rounds until the syncer stops them, so their round count is
not known before the launch (``run_config.resolve_lr_schedule``). Its samplers
therefore have no fixed round count: each generates until every trainer that
reads its store has ended (island flag ``--rl-sampler-until-trainers-end N``:
1 when the samplers follow their own trainer, K otherwise;
:func:`sampler_stop_trainers`). A trainer writes its end record when its run
ends normally (the syncer's final, or the samplers' data end;
``sampler_resume.write_trainer_end``). Other synced groups keep their split of
``K x --total-steps`` rounds.

This module is engine-neutral; the Miles mapping is in
``adapters.miles.buffer_rollout``.
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .data_split import DataSlot  # noqa: F401 - re-exported (callers import it from here)
from .selection import (
    DECOUPLED_SYNC_PRESET,
    FULL_PARAMETER_SYNC_PRESET,
    SAO_STREAMING_SYNC_PRESET,
    STRICT_AVG_SYNC_PRESET,
)

BOTH, SAMPLER, TRAINER, CONVERTIBLE = "both", "sampler", "trainer", "convertible"
SAMPLE_ROLES = (BOTH, SAMPLER, TRAINER, CONVERTIBLE)
# --rl-sampler-follow: what a synced group's samplers load (module doc)
FOLLOW_MERGED, FOLLOW_TRAINER = "merged", "trainer"

# groups a sampler may have ahead of the trainer's last released read position,
# in units of one round (rollout batch), when --rl-sampler-max-backlog-rounds is unset
DEFAULT_MAX_BACKLOG_ROUNDS = 4


@dataclass(frozen=True)
class SampleBufferConfig:
    """``role`` is sampler or trainer (``both`` resolves to no config).

    ``store``: the checkpoint store path that holds the buffer and the releases;
    ``max_backlog_groups``: sampler or convertible node admission cap (unconsumed
    groups);
    ``wait_s``: trainer, max seconds without a new group while a round fills;
    ``min_round_s``: sampler, min seconds from one round start to the next (a
    throttle, so a test can make the sampler slower than the trainer);
    ``fused_stage``: LoRA sampler, check the release digest while copying the
    adapter to local disk (``--rl-sampler-fused-stage``);
    ``release_legacy``: trainer or convertible node, reload the store before every
    policy release and commit LATEST separately (``--rl-release-reload-latest``;
    ``PolicyReleaseStore.cached_latest``);
    ``release_inline``: trainer, write each full-weight release into the store on
    the training critical path instead of staging it on local disk and uploading
    it in the background (``--rl-release-inline``; ``sampler_island`` module doc);
    ``release_delta_full_every``: trainer, upload staged full releases as bf16
    deltas, every N-th one full (``--rl-release-delta-full-every``; 0: off);
    ``release_every`` / ``release_skip_busy``: trainer, release every N-th
    version / skip a release while the previous upload is in flight
    (``--rl-release-every``, ``--rl-release-skip-busy``; ``sampler_island``
    ``ReleasingPublisher``);
    ``until_trainers_end``: sampler, generate until this many trainers that read
    the store have ended, with no fixed round count (module doc; 0: the
    island's own rounds);
    ``feeds_learner``: sampler of a synced group whose samplers follow their
    trainer, the learner id of the trainer it feeds (``rl_sampler_start.island``,
    as an external pool's sampler names it); None otherwise."""

    role: str
    store: str
    max_backlog_groups: int | None = None
    wait_s: float | None = None
    min_round_s: float | None = None
    fused_stage: bool = False
    release_legacy: bool = False
    release_inline: bool = False
    release_delta_full_every: int = 0
    release_every: int = 1
    release_skip_busy: bool = False
    until_trainers_end: int = 0
    feeds_learner: int | None = None


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
    other island. ``--rl-sample-lanes 1``: a trainer whose samplers follow it reads
    its own sub-store, the whole buffer (lane 0 of 1)."""
    from .merged_release import OWN_LANE, BufferLane

    if not is_synced_trainer(args) or getattr(args, "rl_sample_lanes", None) is None:
        return None
    count = int(args.rl_sample_lanes)
    return OWN_LANE if count == 1 else BufferLane(int(getattr(args, "learner_id", 0) or 0), count)


def check_synced_trainers(args: Any, num_islands: int) -> None:
    """Launch checks of synced trainers (module doc), before the per-role checks."""
    if not is_synced_trainer(args):
        return
    full = synced_full_parameter_trainer(args)
    if int(num_islands) < 2 and not full:
        raise ValueError("--rl-sample-role trainer without --rl-single-island-no-sync is a synced "
                         f"multi-island launch: it needs at least two islands (got {num_islands}); "
                         "one learner behind the syncer runs only the full-parameter preset "
                         f"(--tuning full --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET})")
    if is_convertible_pool(args):
        raise ValueError("synced trainers do not combine with convertible nodes")
    tuning = getattr(args, "tuning", "lora")
    if tuning not in ("lora", "full"):
        raise ValueError("synced trainers merge LoRA deltas or run a full-parameter preset: "
                         "they need --tuning lora or --tuning full")
    if tuning == "full" and not synced_full_parameter(args) \
            and getattr(args, "rl_model_recipe", "generic") != "kimi_k31":
        # --tuning full on a synced trainer: a full-parameter preset (the syncer merges the
        # islands' pseudo-gradients) or the K3.1 form (whole-checkpoint releases; no LoRA
        # delta to merge); the generic path keeps the LoRA merge and its golden argv
        raise ValueError("synced trainers with --tuning full need --rl-model-recipe kimi_k31 "
                         "(whole-checkpoint releases) or a full-parameter preset (full-parameter "
                         f"pseudo-gradients): --tuning full needs --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET} "
                         f"or {SAO_STREAMING_SYNC_PRESET}; other recipes need --tuning lora")
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
    # a mixed group of one full-parameter trainer is one learner behind the syncer,
    # as synced trainers fed by external pools only may be
    if trainers < 2 and not (trainers == 1 and is_mixed_group(args) and synced_full_parameter_trainer(args)):
        raise ValueError(f"a synced group needs at least two trainer islands (got {trainers}); "
                         "one trainer with samplers is a pair (--rl-single-island-no-sync)")
    samplers = len(listed)
    if (trainers * int(getattr(args, "total_steps", 0) or 0)) % samplers:
        raise ValueError(f"--total-steps {args.total_steps}: {samplers} sampler islands share the "
                         f"rounds of {trainers} trainers; trainers x total steps must be a multiple "
                         "of the sampler count")
    if getattr(args, "rl_sync_preset", None) == DECOUPLED_SYNC_PRESET:
        check_decoupled_group(args)
    if int(getattr(args, "external_learners", 0) or 0):
        raise ValueError("a synced group does not take external learners")
    if getattr(args, "rl_trainer_scheduler", None) not in (None, "off"):
        raise ValueError("--rl-trainer-scheduler does not combine with a synced group")
    for m in range(num_islands):
        check_launch(island_args(args, m, num_islands))


def check_decoupled_group(args: Any) -> None:
    """Launch checks of a decoupled synced group (module doc, decoupled groups).

    Its trainers have no single merged version (one version per fragment, merged
    at different times), so every sampler follows its own trainer and serves that
    trainer's local versions; the trainers merge their fragments through the
    syncer as decoupled islands do. The samplers start from the base checkpoint:
    an initial adapter would make the trainers' version 0 differ from what the
    samplers serve before the first release, so it is refused."""
    if getattr(args, "rl_sampler_follow", FOLLOW_MERGED) != FOLLOW_TRAINER:
        raise ValueError("--rl-sync-preset decoupled in a synced group (--rl-island-sample-role "
                         "M:sampler) needs --rl-sampler-follow trainer: a decoupled trainer has "
                         "no single merged version to release (each fragment has its own), so "
                         "each sampler serves its own trainer's local versions")
    if not releases_per_learner(args):
        raise ValueError(f"a decoupled synced group needs {per_learner_requirement(args)}")
    if getattr(args, "rl_initial_adapter", None) is not None:
        raise ValueError("a decoupled synced group does not take --rl-initial-adapter: its "
                         "samplers serve the base checkpoint until the first release, so the "
                         "trainers' version 0 must be the base policy")


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
    if not store or lanes is None or int(lanes) < 1:
        raise ValueError("a synced trainer island needs --rl-sample-store and --rl-sample-lanes "
                         ">= 1 (the launcher forwards both; 1: its samplers follow it)")
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
    """One launch of synced trainers and their sampler islands
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


FLEET_ROLE_SYNCER, FLEET_ROLE_ISLAND = "syncer", "island"
FLEET_ROLES = (FLEET_ROLE_SYNCER, FLEET_ROLE_ISLAND)


def fleet_role(args: Any) -> str | None:
    """A launch of one member of a fleet launched island by island (``--rl-fleet-role``):
    ``syncer`` (the fleet's syncer only, no island) or ``island`` (one trainer island that
    dials the fleet's syncer); None for every other launch (nothing changes)."""
    role = getattr(args, "rl_fleet_role", None)
    return role if role in FLEET_ROLES else None


def fleet_learner(args: Any) -> tuple[int, int] | None:
    """``(learner id, launched learners)`` of a fleet island launch (``--rl-fleet-role
    island``: ``--rl-fleet-learner-id K``, ``--rl-fleet-learners N``); None otherwise."""
    if fleet_role(args) != FLEET_ROLE_ISLAND:
        return None
    return int(args.rl_fleet_learner_id), int(args.rl_fleet_learners)


def syncer_learner_count(args: Any, num_islands: int) -> int:
    """The learners the syncer waits for among the launch's islands: the trainers
    of a synced group, every island otherwise (a no-sync launch has no syncer). A fleet
    member launch (:func:`fleet_role`) counts the fleet's launched learners
    (``--rl-fleet-learners``): its syncer profile and session contract are the fleet's."""
    if fleet_role(args) is not None:
        return int(args.rl_fleet_learners)
    learners = synced_group_learners(args, num_islands)
    return len(learners) if learners else int(num_islands)


def sampler_islands(args: Any, num_islands: int) -> list[int]:
    """The sampler islands of a pair, sorted."""
    return sorted(m for m, role in island_roles(args, num_islands).items() if role == SAMPLER)


def buffer_shards(args: Any, num_islands: int) -> list[str] | None:
    """The buffer shards of a pair with several sampler islands: one per
    sampler island, named by its node name; None with one sampler (it writes the
    index itself, as before)."""
    return _shards_of(sampler_islands(args, num_islands))


def _shards_of(samplers: list[int]) -> list[str] | None:
    return [node_name(m) for m in samplers] if len(samplers) > 1 else None


def follows_trainer(args: Any) -> bool:
    """Whether a synced group's samplers, or the external pools of synced trainers,
    follow their own trainer (``--rl-sampler-follow trainer``, module doc)."""
    return ((is_synced_group(args) or is_external_pool_group(args))
            and getattr(args, "rl_sampler_follow", FOLLOW_MERGED) == FOLLOW_TRAINER)


# A buffer shard name of --rl-island-buffer-shards: it reaches the island command
# unquoted (island_flags, comma separated) and names a buffer file or prefix.
_SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# Per-island trainer options of synced trainers fed by external pools:
# (launch flag, its args name, the island option it sets)
ISLAND_TRAINER_OVERRIDES = (
    ("--rl-island-release-every", "rl_island_release_every", "rl_release_every"),
    ("--rl-island-max-policy-age", "rl_island_max_policy_age", "rl_max_policy_age"),
    ("--rl-island-sample-wait-s", "rl_island_sample_wait_s", "rl_sample_wait_s"),
)
# island options whose value is seconds (a number > 0) rather than an integer >= 1
_SECONDS_OPTIONS = frozenset({"rl_sample_wait_s"})
EXTERNAL_POOL_FLAGS = (("--rl-island-buffer-shards", "rl_island_buffer_shards"),
                       *((flag, dest) for flag, dest, _name in ISLAND_TRAINER_OVERRIDES))
# --rl-island-sampler-follows M:K: a mixed group's explicit sampler binding (module doc)
SAMPLER_FOLLOWS_FLAG = ("--rl-island-sampler-follows", "rl_island_sampler_follows")
# --rl-island-dispense-store M:STORE: a mixed group's sampler island M claims its rows from
# the store ledger under STORE (module doc), with the pools' settings of --rl-store-dispense
DISPENSE_STORE_FLAG = ("--rl-island-dispense-store", "rl_island_dispense_store")
STORE_DISPENSE_FLAG = ("--rl-store-dispense", "rl_store_dispense")
# launch-only flags a mixed group's island args drop (the island command never takes them)
_MIXED_LAUNCH_DESTS = ("rl_island_buffer_shards", SAMPLER_FOLLOWS_FLAG[1], DISPENSE_STORE_FLAG[1],
                       *(dest for _flag, dest, _name in ISLAND_TRAINER_OVERRIDES))


def uses_external_pools(args: Any) -> bool:
    """Whether the launch names its trainers' buffer shards
    (``--rl-island-buffer-shards``): their samplers are pools outside the launch."""
    return bool(getattr(args, "rl_island_buffer_shards", None))


def is_external_pool_group(args: Any) -> bool:
    """Synced trainers fed by external pools that follow their own trainer (module
    doc): ``--rl-sample-role trainer`` with a syncer, ``--rl-sampler-follow trainer``
    and ``--rl-island-buffer-shards``, no sampler island."""
    return (uses_external_pools(args) and is_synced_trainer(args)
            and not getattr(args, "rl_island_sample_role", None)
            and getattr(args, "rl_sampler_follow", FOLLOW_MERGED) == FOLLOW_TRAINER)


def is_mixed_group(args: Any) -> bool:
    """A synced group whose samplers follow their trainer and whose trainers also
    read external pools (module doc): ``--rl-island-sample-role M:sampler`` with
    ``--rl-island-buffer-shards K:SHARDS`` (K: learner id) and ``--rl-sampler-follow
    trainer``. A group that binds its sampler islands explicitly
    (``--rl-island-sampler-follows``) takes this form too, also when no trainer has
    a pool (an island fleet whose islands have only Miles samplers)."""
    return ((uses_external_pools(args) or bool(getattr(args, SAMPLER_FOLLOWS_FLAG[1], None)))
            and is_synced_group(args)
            and getattr(args, "rl_sampler_follow", FOLLOW_MERGED) == FOLLOW_TRAINER)


def _trainer_entries(args: Any, dest: str, flag: str, num_islands: int) -> dict[int, str]:
    """{K: value} of a repeatable ``K:VALUE`` trainer flag of external pools: K is
    the island index without sampler islands (every island is a trainer) and the
    trainer's learner id in a synced group with sampler islands (module doc)."""
    if not is_synced_group(args):
        return _island_entries(args, dest, flag, num_islands)
    trainers = len(synced_group_learners(args, num_islands))
    return _island_entries(args, dest, flag, trainers,
                           unit=f"K is a trainer's learner id; the launch has {trainers} trainer(s)")


def _island_entries(args: Any, dest: str, flag: str, num_islands: int,
                    unit: str | None = None) -> dict[int, str]:
    """{island: value} of a repeatable ``ISLAND:VALUE`` flag; raises ValueError on a
    malformed, out-of-range or repeated entry (``unit``: what the range counts,
    by default the --gpu islands)."""
    out: dict[int, str] = {}
    for item in getattr(args, dest, None) or ():
        island, sep, value = str(item).partition(":")
        if not sep or not island.strip().isdigit():
            raise ValueError(f"{flag} takes ISLAND:VALUE, got {item!r}")
        m = int(island)
        if not 0 <= m < int(num_islands):
            raise ValueError(f"{flag} {item!r}: {unit or f'--gpu has {num_islands} island(s)'}")
        if m in out:
            raise ValueError(f"{flag} names island {m} twice")
        out[m] = value.strip()
    return out


def island_buffer_shards(args: Any, num_islands: int) -> dict[int, list[str]]:
    """{K: the buffer shards the pools of trainer K write} of
    ``--rl-island-buffer-shards K:A,B`` (repeatable; K: the trainer island, which
    is its learner id, or the learner id in a mixed group); {} without it. Raises
    ValueError on a malformed, out-of-range or repeated entry, an island without a
    shard, a name that is not a plain shard name, or a shard named twice (every
    pool writes its own shard, so a name is unique in the launch)."""
    out: dict[int, list[str]] = {}
    owner: dict[str, int] = {}
    entries = _trainer_entries(args, "rl_island_buffer_shards", "--rl-island-buffer-shards", num_islands)
    for m, value in sorted(entries.items()):
        shards = [s.strip() for s in value.split(",") if s.strip()]
        if not shards:
            raise ValueError(f"--rl-island-buffer-shards {m}:{value}: island {m} names no shard")
        for shard in shards:
            if not _SHARD_NAME.fullmatch(shard):
                raise ValueError(f"--rl-island-buffer-shards: island {m} shard {shard!r} is not a shard "
                                 "name (letters, digits, '_', '.', '-')")
            if shard in owner:
                raise ValueError(f"--rl-island-buffer-shards: shard {shard!r} is named for island "
                                 f"{owner[shard]} and island {m}; every pool writes its own shard")
            owner[shard] = m
        out[m] = shards
    return out


def island_trainer_overrides(args: Any, num_islands: int) -> dict[int, dict[str, Any]]:
    """{trainer island: {island option: value}} of ``--rl-island-release-every K:N``,
    ``--rl-island-max-policy-age K:N`` and ``--rl-island-sample-wait-s K:S``
    (:data:`ISLAND_TRAINER_OVERRIDES`); {} without them. Raises ValueError on an entry
    that is not ``ISLAND:N``, N an integer >= 1 (seconds > 0 for the sample wait)."""
    out: dict[int, dict[str, Any]] = {}
    for flag, dest, name in ISLAND_TRAINER_OVERRIDES:
        for m, value in _trainer_entries(args, dest, flag, num_islands).items():
            if name in _SECONDS_OPTIONS:
                try:
                    seconds = float(value)
                except ValueError:
                    seconds = math.nan
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"{flag} {m}:{value}: the value must be a number of seconds > 0")
                out.setdefault(m, {})[name] = seconds
                continue
            if not value.isdigit() or int(value) < 1:
                raise ValueError(f"{flag} {m}:{value}: the value must be an integer >= 1")
            out.setdefault(m, {})[name] = int(value)
    return out


def _bind_launch_policy_age(out: Any, args: Any) -> None:
    """Per-island limits (``--rl-island-max-policy-age``): the trainer ``out`` binds
    the launch-wide limit into its syncer session contract instead of its own
    (``--rl-contract-max-policy-age``, ``policy_age.bind_policy_age``). Its own limit
    gates only its own lane, counted in its local versions, and the shared algorithm
    spec bounds every island's limit (launch preflight), so trainers with different
    limits merge. Without the flag nothing changes."""
    if getattr(args, "rl_island_max_policy_age", None):
        out.rl_contract_max_policy_age = int(getattr(args, "rl_max_policy_age", 0) or 0)


def external_pool_island_args(args: Any, island: int, num_islands: int) -> Any:
    """Island args of a synced trainer fed by external pools (module doc): a copy
    with its sub-store ``trainer<k>`` (k: its learner id, which is its island index:
    every island is a trainer), its own lane (the launcher forwards
    ``--rl-sample-lanes 1``), the buffer shards of its pools, and its own release
    cadence and policy-age limit when the launch gives them. With per-island
    limits every trainer also carries the launch-wide limit as the one its session
    contract binds (``--rl-contract-max-policy-age``)."""
    out = copy.copy(args)
    out.rl_synced_trainers = int(num_islands)
    member = fleet_learner(args)  # a fleet island launch: its one island is learner K
    out.rl_sample_substore = trainer_substore(island if member is None else member[0])
    out.rl_buffer_shards = island_buffer_shards(args, num_islands).get(island)
    _bind_launch_policy_age(out, args)
    for name, value in island_trainer_overrides(args, num_islands).get(island, {}).items():
        setattr(out, name, value)
    return out


def sampler_bindings(args: Any, num_islands: int) -> dict[int, int]:
    """{sampler island: trainer island} of a synced group whose samplers follow
    their trainer: sampler j (in island order) feeds trainer j % K, or the trainer
    ``--rl-island-sampler-follows`` names (a mixed group); {} otherwise."""
    if not follows_trainer(args):
        return {}
    trainers = sorted(synced_group_learners(args, num_islands))
    if getattr(args, SAMPLER_FOLLOWS_FLAG[1], None):
        return {m: trainers[k] for m, k in _sampler_follows(args, num_islands).items()}
    return {m: trainers[j % len(trainers)] for j, m in enumerate(sampler_islands(args, num_islands))}


def _sampler_follows(args: Any, num_islands: int) -> dict[int, int]:
    """{sampler island: trainer learner id} of ``--rl-island-sampler-follows M:K``
    (a mixed group, module doc): every sampler island named once, K a learner id."""
    flag, dest = SAMPLER_FOLLOWS_FLAG
    samplers = sampler_islands(args, num_islands)
    trainers = len(synced_group_learners(args, num_islands))
    out: dict[int, int] = {}
    for m, value in _island_entries(args, dest, flag, num_islands).items():
        if m not in samplers:
            raise ValueError(f"{flag} {m}:{value}: island {m} is not a sampler island "
                             f"(sampler islands {samplers})")
        if not value.isdigit() or int(value) >= trainers:
            raise ValueError(f"{flag} {m}:{value}: K must be a trainer's learner id, 0..{trainers - 1}")
        out[m] = int(value)
    if sorted(out) != samplers:
        raise ValueError(f"{flag}: name every sampler island once (sampler islands {samplers}, "
                         f"named {sorted(out)})")
    return out


def sampler_dispense_stores(args: Any, num_islands: int) -> dict[int, str]:
    """{sampler island: the store of the ledger it claims from} of
    ``--rl-island-dispense-store M:STORE`` (a mixed group, module doc); {} without it.
    Raises ValueError on an entry that names no sampler island or no store."""
    flag, dest = DISPENSE_STORE_FLAG
    samplers = sampler_islands(args, num_islands)
    out = _island_entries(args, dest, flag, num_islands)
    for m, store in out.items():
        if m not in samplers:
            raise ValueError(f"{flag} {m}:{store}: island {m} is not a sampler island (sampler islands {samplers})")
        if not store:
            raise ValueError(f"{flag} {m}: name the ledger's store (M:STORE)")
    return out


def check_store_dispense(args: Any, num_islands: int) -> None:
    """``--rl-island-dispense-store`` with ``--rl-store-dispense`` (a mixed group's
    sampler islands on a store ledger): both or neither, the settings a store
    dispenser config without ``store`` (the flag names it), and not with the syncer's
    dispenser (``--rl-data-split dispense``: a sampler claims from one ledger)."""
    import json  # noqa: PLC0415

    stores = sampler_dispense_stores(args, num_islands)
    raw = getattr(args, STORE_DISPENSE_FLAG[1], None)
    if bool(stores) != bool(raw):
        raise ValueError(f"{DISPENSE_STORE_FLAG[0]} and {STORE_DISPENSE_FLAG[0]} go together (the ledger's "
                         "store per sampler island, and the pools' prompts.dispense settings)")
    if not stores:
        return
    from .store_dispense import DispenseError, check_store_config  # noqa: PLC0415

    try:
        config = json.loads(raw)
        check_store_config(config)
    except (ValueError, DispenseError) as error:
        raise ValueError(f"{STORE_DISPENSE_FLAG[0]}: {error}") from None
    if str(config.get("backend") or "store") != "store" or config.get("store") or config.get("holdout"):
        raise ValueError(f"{STORE_DISPENSE_FLAG[0]}: a store ledger config (backend store) without 'store' (named "
                         f"by {DISPENSE_STORE_FLAG[0]}) and without 'holdout' (a path on the pools' nodes; give "
                         "holdout_ids)")
    if getattr(args, "rl_data_split", None) == "dispense":
        raise ValueError(f"{DISPENSE_STORE_FLAG[0]}: the sampler islands claim from the store ledger; drop "
                         "--rl-data-split dispense (the syncer's dispenser)")


def mixed_trainer_shards(args: Any, trainer: int, num_islands: int) -> list[str]:
    """The buffer shards trainer island ``trainer`` of a mixed group reads (and its
    sampler islands are told): the node names of the sampler islands bound to it,
    then the shards of its external pools (the ``--rl-island-buffer-shards`` entry
    of its learner id)."""
    learners = synced_group_learners(args, num_islands)
    own = [node_name(m) for m, t in sorted(sampler_bindings(args, num_islands).items()) if t == trainer]
    return own + island_buffer_shards(args, num_islands).get(learners[trainer], [])


def plan_buffer_shards(args: Any, island: int, num_islands: int) -> list[str] | None:
    """The ``buffer_shards`` the dry-run plan records for island ``island``: the
    shards a trainer fed by external pools reads (in a mixed group, its sampler
    islands' shards first); None for every other island."""
    if is_mixed_group(args):
        learners = synced_group_learners(args, num_islands)
        return mixed_trainer_shards(args, island, num_islands) if island in learners else None
    return island_buffer_shards(args, num_islands).get(island)


def trainer_substore(learner: int) -> str:
    """The store sub-directory of trainer ``learner`` and its samplers (buffer and
    releases) when the samplers follow their trainer. ``learner`` is the trainer's
    learner id (k of K in island order), not its island index: in a synced group
    the sampler islands may come first."""
    return f"trainer{int(learner)}"


def island_substore(args: Any, island: int, num_islands: int) -> str | None:
    """The store sub-directory island ``island`` reads and writes when samplers
    follow their trainer: a trainer's own ``trainer<k>``, a sampler island's that of
    the trainer it feeds; None in every other launch."""
    if is_external_pool_group(args):
        if not 0 <= int(island) < int(num_islands):
            return None
        member = fleet_learner(args)
        return trainer_substore(island if member is None else member[0])
    learners = synced_group_learners(args, num_islands) if follows_trainer(args) else {}
    trainer = sampler_bindings(args, num_islands).get(island, island)
    return trainer_substore(learners[trainer]) if trainer in learners else None


def _capabilities(args: Any) -> frozenset[str]:
    from . import backends  # noqa: PLC0415

    try:
        return backends.capabilities(getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND)
    except backends.UnknownBackend:  # the launch's backend check names it
        return frozenset()


def releases_per_learner(args: Any) -> bool:
    """Whether the trainers' backend releases each trainer's own local versions to its
    ``trainer<k>/`` (``per_learner_release``: ``yeto.rl.engine.backends.capabilities``),
    which the samplers that follow it load (``--rl-sampler-follow trainer``, decoupled
    groups, the sampler stop count, per-trainer ``--rl-release-every``)."""
    from .backend_identity import PER_LEARNER_RELEASE  # noqa: PLC0415

    return PER_LEARNER_RELEASE in _capabilities(args)


def _able(capability: str) -> str:
    from . import backends  # noqa: PLC0415

    able = []
    for name in backends.registered():
        try:
            if capability in backends.capabilities(name):
                able.append(name)
        except backends.UnknownBackend:
            continue
    return " / ".join(able) or "none"


def per_learner_requirement(args: Any, *, short: bool = False) -> str:
    """What :func:`releases_per_learner` asks, for a refusal."""
    from .backend_identity import PER_LEARNER_RELEASE  # noqa: PLC0415

    able = f"--rl-backend {_able(PER_LEARNER_RELEASE)}"
    if short:
        return able
    return (f"a backend whose trainers release their own local versions ({PER_LEARNER_RELEASE}: {able} only); "
            f"--rl-backend {getattr(args, 'rl_backend', None) or 'miles'} does not")


def release_cadence_ok(args: Any, *, busy: bool = False) -> bool:
    """Whether the backend honours ``--rl-release-every N`` (and ``--rl-release-skip-busy``)
    for this trainer: in every role (``release_cadence``), or, without skip-busy, where
    it releases per learner and its samplers follow it (``per_learner_release``)."""
    from .backend_identity import RELEASE_CADENCE  # noqa: PLC0415

    if RELEASE_CADENCE in _capabilities(args):
        return True
    return not busy and releases_per_learner(args) and follows_trainer(args)


def release_cadence_requirement(args: Any) -> str:
    from .backend_identity import PER_LEARNER_RELEASE, RELEASE_CADENCE  # noqa: PLC0415

    backend = getattr(args, "rl_backend", None) or "miles"
    return (f"--rl-backend {_able(RELEASE_CADENCE)} ({RELEASE_CADENCE}), or a backend that releases per learner "
            f"({PER_LEARNER_RELEASE}: {_able(PER_LEARNER_RELEASE)}) with samplers that follow their trainer "
            f"(--rl-sampler-follow trainer) and no --rl-release-skip-busy; --rl-backend {backend} releases "
            "every version here")


def substore_uri(store: str, name: str) -> str:
    """``--rl-checkpoint-store`` (a path or a ``scheme://`` URI) with one more path
    segment."""
    return f"{str(store).rstrip('/')}/{name}"


def sampler_stop_trainers(args: Any, num_islands: int) -> dict[int, int]:
    """Sampler island -> how many trainers' ends stop it, for the samplers of a
    full-parameter or decoupled synced group on a backend that releases per learner
    (module doc); empty for every
    other launch."""
    if (not is_synced_group(args)
            or getattr(args, "rl_sync_preset", None) not in (FULL_PARAMETER_SYNC_PRESET,
                                                             DECOUPLED_SYNC_PRESET)
            or not releases_per_learner(args)):
        return {}
    trainers = 1 if sampler_bindings(args, num_islands) else len(synced_group_learners(args, num_islands))
    return {island: trainers for island in sampler_islands(args, num_islands)}


def describe_sampler_stop(args: Any, num_islands: int) -> str | None:
    """The launcher's line about :func:`sampler_stop_trainers` (None: no such samplers)."""
    stops = sampler_stop_trainers(args, num_islands)
    if not stops:
        return None
    trainers = set(stops.values()).pop()
    whose = "its own trainer" if trainers == 1 else f"all {trainers} trainers"
    return (f"synced group: sampler islands {', '.join(map(str, sorted(stops)))} generate until "
            f"{whose} ended (no fixed round count; the trainers run until the syncer stops them)")


def check_sampler_follow(args: Any, num_islands: int) -> None:
    """Launch checks of ``--rl-sampler-follow trainer``: a synced group on a backend that
    releases per learner whose samplers split evenly over its trainers, synced trainers fed
    by external pools (:func:`check_external_pools`), or both together (a mixed
    group, :func:`check_mixed_group`); the external-pool flags belong to the
    latter two (:func:`check_external_pool_flags`)."""
    check_external_pool_flags(args)
    if getattr(args, "rl_sampler_follow", FOLLOW_MERGED) != FOLLOW_TRAINER:
        return
    if not is_synced_group(args) and not is_external_pool_group(args):
        raise ValueError("--rl-sampler-follow trainer needs a synced group (--rl-sample-role "
                         "trainer --rl-island-sample-role M:sampler, a syncer), or synced "
                         "trainers fed by external sampler pools (--rl-sample-role trainer, a "
                         "syncer, --rl-island-buffer-shards K:SHARDS for every trainer island)")
    if not releases_per_learner(args):
        raise ValueError(f"--rl-sampler-follow trainer needs {per_learner_requirement(args)}")
    if is_external_pool_group(args):
        check_external_pools(args, num_islands)
        return
    if is_mixed_group(args):
        check_mixed_group(args, num_islands)
        return
    trainers = len(synced_group_learners(args, num_islands))
    samplers = len(sampler_islands(args, num_islands))
    if samplers % trainers:
        raise ValueError(f"--rl-sampler-follow trainer: {samplers} sampler islands do not split "
                         f"evenly over {trainers} trainers (each trainer needs the same count)")


def check_external_pool_flags(args: Any) -> None:
    """``--rl-island-buffer-shards`` / ``--rl-island-release-every`` /
    ``--rl-island-max-policy-age`` / ``--rl-island-sample-wait-s`` belong to synced
    trainers fed by external pools,
    with or without sampler islands, and ``--rl-island-sampler-follows`` to a mixed
    group (module doc); any other launch refuses them."""
    if getattr(args, SAMPLER_FOLLOWS_FLAG[1], None) and not is_mixed_group(args):
        raise ValueError(f"{SAMPLER_FOLLOWS_FLAG[0]} binds the sampler islands of a synced group whose "
                         "samplers follow their trainer (--rl-sample-role trainer, a syncer, "
                         "--rl-sampler-follow trainer, --rl-island-sample-role M:sampler)")
    for flag, dest in (DISPENSE_STORE_FLAG, STORE_DISPENSE_FLAG):
        if getattr(args, dest, None) and not is_mixed_group(args):
            raise ValueError(f"{flag} belongs to the sampler islands of a synced group whose samplers follow "
                             "their trainer (--rl-island-sampler-follows, an island fleet)")
    given = [flag for flag, dest in EXTERNAL_POOL_FLAGS if getattr(args, dest, None)]
    if not given or is_external_pool_group(args) or is_mixed_group(args):
        return
    if not uses_external_pools(args):
        raise ValueError(f"{', '.join(given)} need --rl-island-buffer-shards K:SHARDS (synced trainers "
                         "fed by external sampler pools)")
    raise ValueError(f"{', '.join(given)}: synced trainers fed by external sampler pools need "
                     "--rl-sample-role trainer with a syncer (no --rl-single-island-no-sync), "
                     "and --rl-sampler-follow trainer (the pools follow their own trainer, as "
                     "sampler islands named by --rl-island-sample-role M:sampler then do)")


def check_mixed_group(args: Any, num_islands: int) -> None:
    """Launch checks of a mixed group (module doc), after the synced-group checks:
    no global ``--rl-buffer-shards``, the external shards named per learner id, the
    sampler bindings, no shard named twice (a sampler island writes ``island<M>``),
    at least one source per trainer, and sampler islands that generate until their
    own trainer ended (the full-parameter or decoupled preset, per-learner releases, as the
    pools do)."""
    if getattr(args, "rl_buffer_shards", None):
        raise ValueError("--rl-buffer-shards: a synced group with external sampler pools takes each "
                         "trainer's shards from its sampler islands and --rl-island-buffer-shards K:SHARDS")
    learners = synced_group_learners(args, num_islands)
    pools = island_buffer_shards(args, num_islands)
    island_trainer_overrides(args, num_islands)  # K: a learner id (each trainer's own checks: check_synced_group)
    bindings = sampler_bindings(args, num_islands)
    for m in sorted(bindings):
        owner = next((k for k, shards in sorted(pools.items()) if node_name(m) in shards), None)
        if owner is not None:
            raise ValueError(f"--rl-island-buffer-shards: shard {node_name(m)!r} of trainer {owner} is "
                             f"the shard sampler island {m} writes; every sampler writes its own shard")
    fed = {learners[t] for t in bindings.values()} | set(pools)
    missing = sorted(set(learners.values()) - fed)
    if missing:
        raise ValueError(f"trainer(s) with learner id {missing} have no sampler: bind a sampler island "
                         f"({SAMPLER_FOLLOWS_FLAG[0]} M:K) or name pool shards (--rl-island-buffer-shards "
                         "K:SHARDS)")
    check_store_dispense(args, num_islands)
    if not sampler_stop_trainers(args, num_islands):
        raise ValueError("a synced group with sampler islands and external sampler pools needs "
                         f"--rl-sync-preset {FULL_PARAMETER_SYNC_PRESET} or {DECOUPLED_SYNC_PRESET} on "
                         f"{per_learner_requirement(args, short=True)}: its sampler islands generate until "
                         "their own trainer ended, as the pools do")


def check_external_pools(args: Any, num_islands: int) -> None:
    """Launch checks of synced trainers fed by external pools (module doc): every
    island names its buffer shards (the pools that write them follow that trainer
    only), and every trainer passes its own role checks with its release cadence
    and policy-age limit (``--rl-release-every`` <= ``--rl-max-policy-age``)."""
    if getattr(args, "rl_buffer_shards", None):
        raise ValueError("--rl-buffer-shards: synced trainers fed by external sampler pools take each "
                         "trainer's shards from --rl-island-buffer-shards K:SHARDS")
    shards = island_buffer_shards(args, num_islands)
    missing = [m for m in range(int(num_islands)) if m not in shards]
    if missing:
        raise ValueError(f"--rl-island-buffer-shards: trainer island(s) {missing} name no buffer shard; "
                         "every trainer island needs K:SHARDS (the shards of the pools bound to it)")
    if getattr(args, "rl_sync_preset", None) == DECOUPLED_SYNC_PRESET:
        check_decoupled_group(args)
    if int(getattr(args, "external_learners", 0) or 0):
        raise ValueError("synced trainers fed by external sampler pools do not take external learners")
    if getattr(args, "rl_trainer_scheduler", None) not in (None, "off"):
        raise ValueError("--rl-trainer-scheduler does not combine with external sampler pools")
    for m in range(int(num_islands)):
        try:
            check_launch(external_pool_island_args(args, m, num_islands))
        except ValueError as error:
            raise ValueError(f"trainer island {m}: {error}") from None


def is_store_coupled(args: Any) -> bool:
    """A pair or a convertible pool: no syncer, each island is its own one-learner
    no-sync island over the shared checkpoint store."""
    return is_pair(args) or is_convertible_pool(args)


def island_args(args: Any, island: int, num_islands: int) -> Any:
    """``args`` as island ``island`` of a pair sees it: a copy with its
    ``rl_sample_role``, keeping only its own role's option (the sampler's backlog
    cap, the trainer's wait; a convertible node keeps the backlog cap and the
    wait); ``args`` itself outside a pair. A synced trainer fed by external pools
    gets its sub-store and shards (:func:`external_pool_island_args`)."""
    if is_external_pool_group(args):
        return external_pool_island_args(args, island, num_islands)
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
        # each sampler island draws its own slot of the data
        out.rl_sampler_data_slot = (DataSlot(sampler_islands(args, num_islands).index(island), samplers).env()
                                    if out.rl_sample_role == SAMPLER and samplers > 1 else None)
        if out.rl_sample_role == SAMPLER and getattr(args, "rl_sampler_spot", False):
            # spot billing without the legacy --spot volume: the buffer is the sampler's
            # durable state
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
        # each node draws its own slot of the data, as sampler islands do;
        # without it every node generates the same prompts in the same order
        nodes = convertible_islands(args)
        out.rl_sampler_data_slot = (DataSlot(nodes.index(island), len(nodes)).env()
                                    if len(nodes) > 1 else None)
    elif out.rl_sample_role == SAMPLER:
        out.rl_sample_wait_s = None
    else:
        out.rl_sampler_max_backlog_rounds = None
        out.rl_sampler_min_round_seconds = None
    if out.rl_sample_role != SAMPLER:
        out.rl_sampler_fused_stage = False
    if out.rl_sample_role not in (TRAINER, CONVERTIBLE):
        out.rl_release_reload_latest = False
    if out.rl_sample_role != TRAINER:
        out.rl_release_inline = False
        out.rl_release_delta_full_every = 0
        out.rl_release_every = 1
        out.rl_release_skip_busy = False
    return out


def _synced_group_island_args(args: Any, island: int, num_islands: int, roles: dict[int, str]) -> Any:
    """Island args of a synced group. A trainer keeps the synced args (its lane and
    the sampler shards); a sampler becomes a pair-style no-sync sampler over the
    same store: no syncer and no island pool options, its node name and data
    slot, and its share of the trainers' rounds."""
    from ...launch.syncer import _ISLAND_POOL_PARAMS

    samplers = sampler_islands(args, num_islands)
    learners = synced_group_learners(args, num_islands)
    bindings = sampler_bindings(args, num_islands)
    ledgers = sampler_dispense_stores(args, num_islands) if is_mixed_group(args) else {}
    out = copy.copy(args)
    out.rl_dispense_store = None
    out.rl_island_sample_role = None  # the island is no group: its role is below
    out.rl_sample_role = roles[island]
    out.rl_buffer_shards = buffer_shards(args, num_islands)
    out.rl_synced_trainers = len(learners)
    out.rl_sample_substore = None
    if bindings:  # each trainer and its own samplers: one sub-store (module doc)
        trainer = island if out.rl_sample_role == TRAINER else bindings[island]
        out.rl_buffer_shards = _shards_of([m for m, t in sorted(bindings.items()) if t == trainer])
        out.rl_sample_substore = trainer_substore(learners[trainer])
    if is_mixed_group(args):
        # its sampler islands' shards and its pools' shards (a sampler island writes the
        # one of its node name); the trainer's own release cadence and policy-age limit
        out.rl_buffer_shards = mixed_trainer_shards(args, trainer, num_islands)
        if out.rl_sample_role == TRAINER:
            _bind_launch_policy_age(out, args)
            for name, value in island_trainer_overrides(args, num_islands).get(learners[island], {}).items():
                setattr(out, name, value)
        for dest in _MIXED_LAUNCH_DESTS:
            setattr(out, dest, None)
    if out.rl_sample_role == TRAINER:
        out.rl_sampler_node = None
        out.rl_sampler_data_slot = None
        out.rl_sampler_max_backlog_rounds = None
        out.rl_sampler_min_round_seconds = None
        return out
    out.rl_release_delta_full_every = 0
    out.rl_release_every = 1
    out.rl_release_skip_busy = False
    out.rl_sampler_until_trainers_end = sampler_stop_trainers(args, num_islands).get(island, 0)
    if out.rl_sample_substore is not None:
        out.rl_checkpoint_store = substore_uri(args.rl_checkpoint_store, out.rl_sample_substore)
        out.rl_sample_substore = None  # a sampler's store is its trainer's sub-store
        out.rl_sampler_feeds_learner = learners[bindings[island]]  # its rl_sampler_start.island
    out.rl_single_island_no_sync = True
    for name, _flag in _ISLAND_POOL_PARAMS:
        setattr(out, name, None)
    out.rl_syncer_epoch = None
    out.rl_sampler_node = node_name(island)
    out.rl_tape_island_id = island  # its records name its launch island, not learner 0
    if island in ledgers:
        # The sampler claims its rows from a store ledger, with the pools (no slot)
        out.rl_dispense_store = ledgers[island]
        out.rl_sampler_data_slot = None
    elif getattr(args, "rl_data_split", None) == "dispense":
        # The sampler claims its rows from the group's actor syncer (no slot)
        out.rl_sampler_dispense_island = island
        out.rl_sampler_data_slot = None
    else:
        out.rl_sampler_data_slot = (DataSlot(samplers.index(island), len(samplers)).env()
                                    if len(samplers) > 1 else None)
    out.rl_sample_wait_s = None
    out.total_steps = out.rl_synced_trainers * int(args.total_steps) // len(samplers)
    if getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET:
        out.rl_sampler_max_backlog_rounds = full_parameter_sampler_backlog_rounds(
            args, island, num_islands)
        _as_full_parameter_sampler(out)
    elif getattr(args, "rl_sync_preset", None) == DECOUPLED_SYNC_PRESET:
        _as_lora_sampler(out)
    if getattr(args, "rl_total_fragment_steps", None) is not None:
        out.rl_total_fragment_steps = out.total_steps * int(getattr(out, "fragments", 1) or 1)
    return out


def min_full_parameter_backlog_rounds(args: Any, island: int, num_islands: int) -> int:
    """The smallest backlog cap (sampler rounds) of a full-parameter synced group's
    sampler that cannot stall the group.

    The sampler counts its backlog from the read position of the newest release,
    and the publisher releases only when the merged version advances: every fragment
    merged once more, i.e. ``--fragments`` x ``--local-rl-rounds-per-sync`` local
    rounds of every trainer, plus the
    ``--pipeline`` fragments in flight. Until then the trainers it feeds (all K, or
    its own trainer with ``--rl-sampler-follow trainer``) consume
    ``trainers x (fragments + pipeline)`` rounds of groups, shared by the samplers of
    the same store. A lower cap stops the sampler before the next release exists and
    the trainers then wait for groups forever."""
    bindings = sampler_bindings(args, num_islands)
    samplers = sampler_islands(args, num_islands)
    if bindings:
        trainers_fed = 1
        sharing = sum(1 for t in bindings.values() if t == bindings[island])
    else:
        trainers_fed = len(synced_group_learners(args, num_islands))
        sharing = len(samplers)
    return stall_free_backlog_rounds(
        trainers_fed=trainers_fed, sharing=sharing,
        fragments=int(getattr(args, "fragments", 1) or 1),
        local_rounds=int(getattr(args, "local_rl_rounds_per_sync", 1) or 1),
        pipeline=int(getattr(args, "pipeline", 0) or 0))


def stall_free_backlog_rounds(*, trainers_fed: int, sharing: int, fragments: int = 1,
                              local_rounds: int = 1, pipeline: int = 0) -> int:
    """The pure part of :func:`min_full_parameter_backlog_rounds`: ``trainers_fed`` x
    (``fragments`` x ``local_rounds`` + ``pipeline``) rounds until the next merged
    release, shared by ``sharing`` samplers of the store (rounded up)."""
    rounds = trainers_fed * (fragments * local_rounds + pipeline)
    return -(-rounds // max(1, sharing))


def full_parameter_sampler_backlog_rounds(args: Any, island: int, num_islands: int) -> int:
    """``--rl-sampler-max-backlog-rounds`` of a full-parameter synced group's sampler:
    the stall-free minimum (:func:`min_full_parameter_backlog_rounds`) plus the
    default margin, or the operator's value, refused when it is below the minimum."""
    need = min_full_parameter_backlog_rounds(args, island, num_islands)
    explicit = getattr(args, "rl_sampler_max_backlog_rounds", None)
    if explicit is None:
        return need + DEFAULT_MAX_BACKLOG_ROUNDS
    if int(explicit) < need:
        raise ValueError(
            f"--rl-sampler-max-backlog-rounds {explicit} stalls a full-parameter synced group: "
            f"sampler island {island} must hold at least {need} rounds of groups until the "
            "next merged release (trainers x (fragments x local rounds + pipeline) / samplers "
            "per store)")
    return int(explicit)


def _as_full_parameter_sampler(out: Any) -> None:
    """A sampler of a full-parameter synced group runs as a pair's full-parameter
    sampler: a no-sync island with ``--tuning full`` and the default sync preset
    (it syncs nothing, so no fragment or merge option reaches its command)."""
    out.rl_sync_preset = STRICT_AVG_SYNC_PRESET
    out.tuning = "full"
    out.rl_sync_full_parameter_rounds = "sweep"
    out.rl_sync_matrix_merge = "avg"
    out.rl_sync_bcast_delta_full_every = 0
    out.local_rl_rounds_per_sync = 1
    out.fragments = 1


def _as_lora_sampler(out: Any) -> None:
    """A sampler of a decoupled synced group runs as a pair's LoRA sampler: a
    no-sync island with the default sync preset and one fragment (it syncs
    nothing, so no fragment, pipeline or merge option reaches its command). It
    loads its trainer's LoRA releases; it keeps the trainer's LoRA rank and
    targets, so the released adapter fits its engines."""
    out.rl_sync_preset = STRICT_AVG_SYNC_PRESET
    out.tuning = "lora"
    out.rl_sync_matrix_merge = "avg"
    out.local_rl_rounds_per_sync = 1
    out.fragments = 1
    out.pipeline = 1
    out.rl_initial_adapter = None
    out.rl_initial_adapter_sha256 = None


def island_learner(args: Any, island: int, num_learners: int,
                   num_islands: int | None = None) -> tuple[int, int]:
    """``(learner id, number of learners)`` the island command carries. A pair
    island is a one-learner no-sync island: (0, 1); so is a synced group's sampler,
    and its trainers are learners 0..K-1 (:func:`synced_group_learners`). The
    launcher keeps ``island`` for the cluster / Modal function name, the tape and
    the volumes."""
    if is_store_coupled(args):
        return 0, 1
    member = fleet_learner(args)
    if member is not None:  # a fleet island launch: its one trainer is learner K of the fleet's N
        return member
    learners = synced_group_learners(args, num_learners if num_islands is None else num_islands)
    if learners:
        return (learners[island], len(learners)) if island in learners else (0, 1)
    return island, num_learners


def synced_full_parameter_trainer(args: Any) -> bool:
    """A synced trainer island (:func:`is_synced_trainer`) training every
    parameter: ``--tuning full`` with the full-parameter preset. Its pseudo-gradient
    is merged by the syncer across the K learner islands (K >= 1: one learner is
    learners=1/quorum=1 and the merge is the identity); the samplers run as
    full-parameter samplers (:func:`_as_full_parameter_sampler`). The K3.1
    recipe (partial experts, ``--tuning full``) takes this form."""
    return (is_synced_trainer(args)
            and getattr(args, "tuning", "lora") == "full"
            and getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET)


def buffer_full_parameter(args: Any) -> bool:
    """``--tuning full`` on a no-sync island: the island trains every parameter
    without the full-parameter sync, since it has no syncer. A sampler/trainer
    island (single or pair) passes the policy to the sampler as a store release;
    a single island with both roles updates its own engines in place."""
    return (getattr(args, "tuning", "lora") == "full"
            and bool(getattr(args, "rl_single_island_no_sync", False)))


def synced_full_parameter(args: Any) -> bool:
    """``--tuning full`` on synced islands: every parameter trains and the
    full-parameter sync preset (whole-model fragments) carries it. The same
    islands run with ``--tuning lora`` and the full-parameter preset (the
    preset alone selects full parameter mode); ``--tuning full`` states it."""
    from .selection import FULL_PARAMETER_SYNC_PRESETS, canonical_sync_preset

    return (getattr(args, "tuning", "lora") == "full"
            and not bool(getattr(args, "rl_single_island_no_sync", False))
            and canonical_sync_preset(str(getattr(args, "rl_sync_preset", "strict-avg")))
            in FULL_PARAMETER_SYNC_PRESETS)


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
    LoRA or full parameters on colocated islands (the trainer and the engines share
    the GPUs; a full-parameter node releases whole checkpoints, the K3.1 form); then
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
    if getattr(args, "tuning", "lora") not in ("lora", "full") or \
            (getattr(args, "rl_placement", None) or "colocated") != "colocated":
        raise ValueError("a convertible node trains and generates on the same GPUs: it needs "
                         "--tuning lora or --tuning full and colocated placement")
    for m in range(num_islands):
        check_launch(island_args(args, m, num_islands))


FLEET_MEMBER_DESTS = (("--rl-fleet-learner-id", "rl_fleet_learner_id"), ("--rl-fleet-learners", "rl_fleet_learners"),
                      ("--rl-fleet-syncer", "rl_fleet_syncer"))


def check_fleet_member(args: Any) -> None:
    """``--rl-fleet-role`` (a fleet launched island by island, ``yeto.rl.island_launches``):
    synced full-parameter trainers fed by external pools, an island pool
    (``--rl-sync-max-learners``), the fleet's launched learners; an island launch also
    its learner id below that limit, one cluster in ``--gpu``, ``--controller local`` and,
    unless a dry run, the syncer's address. Without the role none of its options."""
    role = fleet_role(args)
    grace = getattr(args, "rl_fleet_syncer_grace_s", None)
    if grace is not None:
        if role != FLEET_ROLE_SYNCER:
            raise ValueError("--rl-fleet-syncer-grace-s belongs to the syncer launch (--rl-fleet-role syncer)")
        if not float(grace) > 0:
            raise ValueError(f"--rl-fleet-syncer-grace-s must be > 0 seconds (got {grace})")
    if role is None:
        named = [flag for flag, dest in FLEET_MEMBER_DESTS if getattr(args, dest, None) is not None]
        if named:
            raise ValueError(f"{' / '.join(named)} need --rl-fleet-role")
        return
    learners = getattr(args, "rl_fleet_learners", None)
    limit = getattr(args, "rl_sync_max_learners", None)
    if learners is None or int(learners) < 1:
        raise ValueError("--rl-fleet-role needs --rl-fleet-learners N >= 1 (the fleet's launched trainers)")
    if limit is None or int(limit) < int(learners):
        raise ValueError("--rl-fleet-role needs --rl-sync-max-learners >= --rl-fleet-learners (the syncer admits "
                         "the islands added later under ids below it)")
    if not synced_full_parameter_trainer(args) or not uses_external_pools(args) or not follows_trainer(args):
        raise ValueError("--rl-fleet-role: the fleet's trainers are synced full-parameter trainers fed by external "
                         "pools (--rl-sample-role trainer, --rl-sampler-follow trainer, --rl-island-buffer-shards, "
                         "--tuning full, the full-parameter preset)")
    if getattr(args, "rl_island_sample_role", None):
        raise ValueError("--rl-fleet-role: no sampler islands (--rl-island-sample-role) in a fleet launched island "
                         "by island; its samplers are external pools")
    if role == FLEET_ROLE_SYNCER:
        for flag, dest in FLEET_MEMBER_DESTS:
            if dest != "rl_fleet_learners" and getattr(args, dest, None) is not None:
                raise ValueError(f"--rl-fleet-role syncer: {flag} belongs to an island launch")
        return
    k = getattr(args, "rl_fleet_learner_id", None)
    if k is None or not 0 <= int(k) < int(limit):
        raise ValueError(f"--rl-fleet-role island needs --rl-fleet-learner-id K in 0..{int(limit) - 1} "
                         "(--rl-sync-max-learners)")
    gpu = str(getattr(args, "gpu", None) or "")
    if not gpu or "," in gpu:
        raise ValueError("--rl-fleet-role island launches one trainer island: --gpu names one cluster")
    if getattr(args, "controller", "local") != "local":
        raise ValueError("--rl-fleet-role island needs --controller local (the fleet's syncer runs elsewhere)")
    if not getattr(args, "rl_fleet_syncer", None) and not getattr(args, "dry_run", False):
        raise ValueError("--rl-fleet-role island needs --rl-fleet-syncer HOST:PORT (the fleet syncer's address, "
                         "fleet/island_syncer.json in the run store)")


def check_launch(args: Any) -> None:
    """Launch-side checks of one island's role, before any cloud work. ``both``
    checks nothing (a pair checks each island with :func:`check_pair`)."""
    check_fleet_member(args)
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
        if getattr(args, "rl_release_inline", False):
            raise ValueError("--rl-release-inline needs --rl-sample-role trainer")
        if int(getattr(args, "rl_release_delta_full_every", 0) or 0):
            raise ValueError("--rl-release-delta-full-every needs --rl-sample-role trainer")
        if _release_every(args) != 1 or getattr(args, "rl_release_skip_busy", False):
            raise ValueError("--rl-release-every / --rl-release-skip-busy need --rl-sample-role trainer")
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
    _check_release_delta(args, role)
    _check_release_cadence(args, role)
    wait_s = getattr(args, "rl_sample_wait_s", None)
    if wait_s is not None and (role not in (TRAINER, CONVERTIBLE) or not float(wait_s) > 0):
        raise ValueError("--rl-sample-wait-s needs --rl-sample-role trainer and a value > 0")
    if role == CONVERTIBLE and not getattr(args, "rl_convertible_node", None):
        raise ValueError("--rl-sample-role convertible needs --rl-convertible-node NAME")


RELEASE_SKIP_BUSY_MIN_AGE = 3


def _release_every(args: Any) -> int:
    return int(getattr(args, "rl_release_every", 1) or 1)


def _check_release_cadence(args: Any, role: str) -> None:
    """``--rl-release-every N`` / ``--rl-release-skip-busy``: a trainer whose samplers
    may sample on an older release than every version.

    Policy age stays counted in trainer versions (the buffer gate, the algorithm
    spec and resumed_base are unchanged): a release every N versions lets a
    sampler's newest weights trail the trainer by up to N - 1 more versions, so
    the launch needs --rl-max-policy-age >= N, else every sample of the
    in-between steps would be dropped as stale. Skipping busy releases has no
    fixed bound (an upload may span several steps). A sample's lag is about
    ceil(upload_s / step_s) + ceil(download_s / step_s) + backlog_groups / groups_per_step
    versions (Kimi-K3.1, 207 GB releases: a 12-20 min upload over ~200 s steps alone
    is 4-6), so pick the age from that sum and watch rl_buffer_stale_skipped; the
    launch only refuses an age below the floor of 3 (one release in flight, one
    being downloaded, one served). Samples older than the age are dropped as
    stale and counted."""
    every, busy = _release_every(args), bool(getattr(args, "rl_release_skip_busy", False))
    if every == 1 and not busy:
        return
    age = int(getattr(args, "rl_max_policy_age", 0) or 0)
    problems = []
    if every < 1:
        problems.append("--rl-release-every >= 1")
    if role != TRAINER:
        problems.append("--rl-sample-role trainer")
    if not release_cadence_ok(args, busy=busy):
        problems.append(release_cadence_requirement(args))
    if every > 1 and age < every:
        problems.append(f"--rl-max-policy-age >= {every} (a sampler may trail by up to "
                        f"{every} versions; got {age})")
    if busy:
        if getattr(args, "rl_release_inline", False):
            problems.append("staged releases for --rl-release-skip-busy (no --rl-release-inline)")
        if not _full_weights(args):
            # a LoRA trainer writes every release inline (sampler_island stages full
            # weights only): the flag would be dropped without a word
            problems.append("full-weight releases for --rl-release-skip-busy "
                            "(--tuning full or the full-parameter preset)")
        if age < RELEASE_SKIP_BUSY_MIN_AGE:
            problems.append(f"--rl-max-policy-age >= {RELEASE_SKIP_BUSY_MIN_AGE} for "
                            f"--rl-release-skip-busy (got {age})")
    if problems:
        raise ValueError("--rl-release-every / --rl-release-skip-busy need " + "; ".join(problems))


def _check_release_delta(args: Any, role: str) -> None:
    """``--rl-release-delta-full-every N``: 0 (off) or N >= 2, on a trainer whose
    full-weight releases are staged (``policy_release`` module doc).

    Backends: Miles, and a verl full-parameter buffer
    trainer, whose export writes the base checkpoint's own files
    (``adapters/verl/hf_export``) so the delta against the base stays small."""
    every = int(getattr(args, "rl_release_delta_full_every", 0) or 0)
    if every == 0:
        return
    problems = []
    if every < 2:
        problems.append("a value of 0 (off) or >= 2")
    if role != TRAINER:
        problems.append("--rl-sample-role trainer")
    if getattr(args, "rl_release_inline", False):
        problems.append("staged releases (no --rl-release-inline)")
    backend = getattr(args, "rl_backend", None) or "miles"
    if backend not in ("miles", "verl"):
        problems.append("--rl-backend miles or verl")
    if backend == "verl" and getattr(args, "parameter_mode", "lora") != "full":
        problems.append("--parameter-mode full on a verl island (a LoRA release is one adapter)")
    if int(getattr(args, "rl_rollout_quant_tier", 0) or 0):
        problems.append("no --rl-rollout-quant-tier (its re-quantizer reads whole releases)")
    if not _full_weights(args):
        problems.append("full-weight releases (--tuning full or the full-parameter preset)")
    if problems:
        raise ValueError("--rl-release-delta-full-every needs " + "; ".join(problems))


def _full_weights(args: Any) -> bool:
    """Whether a trainer releases whole checkpoints (not a LoRA adapter)."""
    if int(getattr(args, "expert_full_count", 0) or 0):
        return False
    return (getattr(args, "tuning", "lora") == "full"
            or getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET
            or getattr(args, "parameter_mode", "lora") == "full")


def convertible_min_backlog_rounds(args: Any) -> int:
    """The smallest backlog cap of a convertible node, in rounds: the trainer scheduler starts a trainer
    at a backlog of at most ``a_hard + 1`` batches (``a_hard`` = the max policy
    age), so a lower cap would hold every node back before any trainer starts."""
    return int(getattr(args, "rl_max_policy_age", 0) or 0) + 1


def sampler_backlog_rounds(explicit: int | None = None) -> int:
    """The backlog cap (rounds) of a plain sampler (no full-parameter synced group):
    the operator's value, else :data:`DEFAULT_MAX_BACKLOG_ROUNDS`."""
    return _max_backlog_rounds(SimpleNamespace(rl_sampler_max_backlog_rounds=explicit), SAMPLER)


# Release cadence vs the samplers' backlog cap (one buffer trainer, no sync).
# A sampler counts its backlog from the read position of the newest release it
# sees (0 before the first one; sampler_loop.SamplerLoop.backlog), so its cap
# moves only when a release lands, not as the trainer reads. Groups trained and
# skipped (stale, damaged) take buffer positions alike: the margin covers those.
RELEASE_BACKLOG_MARGIN_GROUPS = 8
# rounds a staged upload is assumed to last (the preflight does not know the
# upload or the step time; re-derive with the measured upload / step ratio)
ASSUMED_UPLOAD_ROUNDS = 2


def release_record_rounds(*, release_every: int, staged: bool, skip_busy: bool,
                          upload_rounds: int = ASSUMED_UPLOAD_ROUNDS) -> int:
    """``m``: the training rounds the trainer starts between a policy's publication
    and its release record (the record carries the read position the samplers'
    caps then move to). 0 for a release written inline; a staged upload: the
    round started right after the publication plus the rounds the upload lasts,
    at most ``release_every`` when the next release waits for the upload (no
    ``--rl-release-skip-busy``)."""
    if not staged:
        return 0
    rounds = 1 + max(0, int(upload_rounds))
    return rounds if skip_busy else min(max(1, int(release_every)), rounds)


def release_backlog_need(*, groups_per_round: int, release_every: int, record_rounds: int,
                         margin: int = RELEASE_BACKLOG_MARGIN_GROUPS) -> int:
    """Groups the samplers of a single trainer must be allowed to hold, all
    together, past the newest release's read position so that none waits at its
    cap between two releases: ``(release_every + m + 1) x groups_per_round +
    margin`` (``m``: :func:`release_record_rounds`). A sampler's
    ``max_backlog_groups`` is this over the samplers that keep writing, rounded up.
    The first release, at version ``release_every``, needs ``release_every x
    groups_per_round`` groups below the cap before it can be trained at all."""
    return (max(1, int(release_every)) + max(0, int(record_rounds)) + 1) * int(groups_per_round) + int(margin)


# ``max_backlog_groups: "auto"`` in a sampler spec: ``yeto run up`` derives the cap from
# the trainer's release cadence (:func:`auto_backlog_groups`) and hands it to its pools
# in these variables; a pool or a sampler started without them takes the fallback.
AUTO_BACKLOG = "auto"
AUTO_BACKLOG_ENV = "YETO_RL_AUTO_BACKLOG_GROUPS"
AUTO_BACKLOG_WHY_ENV = "YETO_RL_AUTO_BACKLOG_WHY"
# the spec key a resolved "auto" cap leaves its note in (the sampler prints it at start)
AUTO_BACKLOG_NOTE_KEY = "max_backlog_groups_auto"
# the cap of an "auto" sampler that cannot see the run config (started on its own, or
# a trainer the formula does not cover): large enough for one sampler of a trainer
# releasing every 24 rounds of 8 groups, so it never waits at its cap for a release
AUTO_BACKLOG_FALLBACK_GROUPS = 256


def auto_backlog_groups(*, groups_per_round: int, release_every: int, record_rounds: int,
                        samplers: int) -> int:
    """The ``max_backlog_groups`` an ``"auto"`` sampler runs with:
    :func:`release_backlog_need` over the samplers, rounded up
    (``ceil(((release_every + m + 1) x groups_per_round + 8) / samplers)``)."""
    need = release_backlog_need(groups_per_round=groups_per_round, release_every=release_every,
                                record_rounds=record_rounds)
    return -(-need // max(1, int(samplers)))


def resolve_auto_backlog(value: Any, environ: Mapping[str, str] | None = None) -> tuple[Any, str | None]:
    """``(cap, note)`` of a spec's ``max_backlog_groups``: a number is returned as is
    (note None); ``"auto"`` takes the value ``yeto run up`` derived
    (:data:`AUTO_BACKLOG_ENV`), else :data:`AUTO_BACKLOG_FALLBACK_GROUPS`; the note
    says which, for the start log."""
    if not (isinstance(value, str) and value.strip().lower() == AUTO_BACKLOG):
        return value, None
    env = os.environ if environ is None else environ
    raw = str(env.get(AUTO_BACKLOG_ENV) or "").strip()
    if raw.isdigit() and int(raw) > 0:
        why = str(env.get(AUTO_BACKLOG_WHY_ENV) or "").strip()
        return int(raw), "auto" + (f": {why}" if why else " (derived by yeto run up)")
    return AUTO_BACKLOG_FALLBACK_GROUPS, (
        f"auto: fallback {AUTO_BACKLOG_FALLBACK_GROUPS} (no run config: the release cadence and the sampler "
        f"count are known to `yeto run up` only; set a number to size it yourself)")


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
        and bool(getattr(args, "rl_release_reload_latest", False)),
        release_inline=role == TRAINER and bool(getattr(args, "rl_release_inline", False)),
        release_delta_full_every=(int(getattr(args, "rl_release_delta_full_every", 0) or 0)
                                  if role == TRAINER else 0),
        release_every=_release_every(args) if role == TRAINER else 1,
        release_skip_busy=role == TRAINER and bool(getattr(args, "rl_release_skip_busy", False)),
        until_trainers_end=(int(getattr(args, "rl_sampler_until_trainers_end", 0) or 0)
                            if role == SAMPLER else 0),
        feeds_learner=(int(args.rl_sampler_feeds_learner)
                       if role == SAMPLER and getattr(args, "rl_sampler_feeds_learner", None) is not None
                       else None))


def island_env(args: Any) -> dict[str, str]:
    """The environment a pair's sampler island gets from the launcher ({} for
    every other island; ``args`` are the island args): the reclaim marker path,
    so the Modal runner turns the reclaim signal into that file and the sampler
    leaves without being signalled."""
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
    if role == TRAINER and getattr(args, "rl_release_inline", False):
        flags += " --rl-release-inline"
    if role == TRAINER and int(getattr(args, "rl_release_delta_full_every", 0) or 0):
        flags += f" --rl-release-delta-full-every {int(args.rl_release_delta_full_every)}"
    if role == TRAINER and _release_every(args) != 1:
        flags += f" --rl-release-every {_release_every(args)}"
    if role == TRAINER and getattr(args, "rl_release_skip_busy", False):
        flags += " --rl-release-skip-busy"
    if getattr(args, "rl_sampler_node", None):
        flags += f" --rl-sampler-node {args.rl_sampler_node}"
    if getattr(args, "rl_buffer_shards", None):
        flags += f" --rl-buffer-shards {','.join(args.rl_buffer_shards)}"
    if getattr(args, "rl_sampler_data_slot", None):
        flags += f" --rl-sampler-data-slot {args.rl_sampler_data_slot}"
    if role == SAMPLER and int(getattr(args, "rl_sampler_until_trainers_end", 0) or 0):
        flags += f" --rl-sampler-until-trainers-end {int(args.rl_sampler_until_trainers_end)}"
    if role == SAMPLER and getattr(args, "rl_sampler_feeds_learner", None) is not None:
        flags += f" --rl-sampler-feeds-learner {int(args.rl_sampler_feeds_learner)}"
    if role == SAMPLER and getattr(args, "rl_tape_island_id", None) is not None:
        flags += f" --rl-tape-island-id {int(args.rl_tape_island_id)}"
    return flags

