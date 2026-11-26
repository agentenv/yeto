"""What the launches of a fleet launched island by island must agree on.

Each island launch (:mod:`yeto.rl.island_launches`) derives its options from its own
argv (one ``--gpu`` entry), the syncer launch from every declared trainer's: a default
derived from the launch itself can differ between them, and the syncer then refuses
every island at HELLO. Here each launch's argv is parsed and prepared as ``yeto launch``
prepares it (nothing is ordered or read remotely) and compared with the syncer launch:

* the syncer semantic profile (its sha256; the differing fields are named),
* the HELLO-bound island flags (``--num-learners``, layout, profile, rounds; the
  island's own learner id masked),
* the learners the syncer waits for and the syncer_epoch the island JOINs with,
* each island's learner id: unique, and below the syncer's ``--rl-sync-max-learners``,
* the backend identity bound into each island's session contract: the compat group
  its ``--gpu`` card declares (``--rl-compat-group`` replaces it) and the identity
  mode (``--rl-identity``). Under strict identity two islands on cards of different
  compat groups offer different session contract hashes (same layout), and the
  syncer refuses the later one's HELLO ("session mismatch"); ``yeto fleet island
  move/add`` also compares the island with the ones that hold the fleet's running
  session (the other trainer islands, and a moved island's previous machines).

``yeto run up`` (dry run too) checks the trainer islands of the desired state (else the
run config's) at its preflight; ``yeto fleet island add/move`` checks the island it
writes.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

PLACEHOLDER_ADDR = "SYNCER:29400"


@dataclass(frozen=True)
class LaunchSession:
    name: str
    learner_id: int | None
    profile: dict[str, Any]
    profile_sha256: str
    learners: int
    hello: str
    syncer_epoch: int
    max_learners: int | None
    # The backend identity's hardware slot this launch binds (None: not one island's
    # launch, or a card the catalog cannot place; the launch reports that itself).
    compat_group: str | None = None
    strict_identity: bool = True
    # (TP, PP, EP, CP, DP) of the trainer under a shape-bound full-parameter layout (None:
    # not one island's launch, or a layout that does not depend on the shape)
    shape: tuple[int, int, int, int, int] | None = None


def _prepare(args: Any) -> None:
    from yeto.launcher import _prepare_rl_args  # noqa: PLC0415 - the launch's own option rules

    _prepare_rl_args(args)


def launch_session(argv: list[str], name: str, *, prepare: Callable[[Any], None] | None = None) -> LaunchSession:
    """The session a launch with ``argv`` would bind (ValueError: the launch refuses it)."""
    from yeto.cli import parse_args  # noqa: PLC0415
    from yeto.launch.rl_island_task import _rl_trainable_island_flags  # noqa: PLC0415
    from yeto.launch.syncer import island_syncer_epoch, rl_syncer_profile  # noqa: PLC0415
    from yeto.launch.topology import rl_island_count  # noqa: PLC0415
    from yeto.rl.engine.sample_roles import syncer_learner_count  # noqa: PLC0415

    args = parse_args(list(argv))
    args.dry_run = True
    with contextlib.redirect_stdout(io.StringIO()):  # the launch's own notes: not this check's
        (prepare or _prepare)(args)
    learners = syncer_learner_count(args, rl_island_count(args))
    profile = rl_syncer_profile(args, learners)
    hello = re.sub(r"--learner-id \d+", "", _rl_trainable_island_flags(args, 0, learners))
    learner = getattr(args, "rl_fleet_learner_id", None)
    max_learners = getattr(args, "rl_sync_max_learners", None)
    return LaunchSession(name=name, learner_id=None if learner is None else int(learner),
                         profile=dict(profile.__dict__), profile_sha256=profile.sha256, learners=learners,
                         hello=" ".join(hello.split()), syncer_epoch=island_syncer_epoch(args),
                         max_learners=None if max_learners is None else int(max_learners),
                         compat_group=launch_compat_group_of(args),
                         strict_identity=getattr(args, "rl_identity", "strict") != "relaxed",
                         shape=launch_shape_of(args))


def launch_shape_of(args: Any) -> tuple[int, int, int, int, int] | None:
    """``(TP, PP, EP, CP, DP)`` of a one-cluster full-parameter launch on the shape-bound
    layout (the default ``--rl-sync-layout``), whose tensor names carry each rank's
    parallel position (``yeto.rl.engine.shape_free_layout``); None otherwise (a launch of
    several clusters, LoRA, ``shape-free``, a backend whose layout names no shard)."""
    shapes = launch_shapes_of(args)
    return shapes[0] if shapes is not None and len(shapes) == 1 else None


def launch_shapes_of(args: Any) -> list[tuple[int, int, int, int, int]] | None:
    """:func:`launch_shape_of` of each ``--gpu`` cluster of a launch (a synced-group
    fleet's one launch: one per island, in ``--gpu`` order); None when the layout does not
    depend on the shape or the launch does not parse."""
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from yeto.launch.fragment_preflight import _card_free  # noqa: PLC0415
    from yeto.launch.topology import rl_island_args, rl_island_spec, rl_trainer_shape  # noqa: PLC0415

    if (getattr(args, "rl_sync_layout", "shape-bound") != "shape-bound"
            or getattr(args, "tuning", "lora") != "full"):
        return None
    out = []
    try:
        specs = parse_gpu_spec(str(getattr(args, "gpu", None) or ""))
        if not specs or _card_free(args):
            return None
        for m, spec in enumerate(specs):  # each island as it sees itself (placement, GPUs used, parallel)
            iargs = rl_island_args(args, m)
            nodes, gpus = rl_trainer_shape(iargs, rl_island_spec(iargs, spec))
            tp, pp, ep, cp = (max(1, int(getattr(iargs, name, 1) or 1)) for name in
                              ("tensor_parallel", "pipeline_parallel", "expert_parallel", "context_parallel"))
            out.append((tp, pp, ep, cp, max(1, nodes * gpus // (tp * pp * cp))))
    except (ValueError, KeyError, AttributeError, ImportError):
        return None  # a shape the launch refuses: its own dry run says why
    return out


def _shape(shape: tuple[int, int, int, int, int]) -> str:
    return " ".join(f"{k}{v}" for k, v in zip(("TP", "PP", "EP", "CP", "DP"), shape))


def shape_problems(islands: list[LaunchSession], running: Iterable[LaunchSession] = ()) -> list[str]:
    """The islands whose trainer shape differs from the fleet's under the shape-bound
    layout: every tensor is named by its rank's TP/PP/EP/CP/DP position, so islands of
    two shapes bind two layouts and the syncer refuses the later HELLO. ``running``: the
    launches holding the fleet's running session (compared with, not reported for)."""
    out: list[str] = []
    peers = [r for r in running if r.shape is not None]
    for island in islands:
        if island.shape is None:
            continue
        peer = next((p for p in peers if p.shape != island.shape), None)
        if peer is not None:
            out.append(f"{_where(island)}: trainer shape {_shape(island.shape)}, {_where(peer)} trains on "
                       f"{_shape(peer.shape)}: the shape-bound full-parameter layout names every tensor by its "
                       "rank's TP/PP/EP/CP/DP position, so the syncer refuses this island's HELLO (layout "
                       "mismatch). Give every island the same shape (the same nodes x GPUs and parallel flags), "
                       "or set --rl-sync-layout shape-free for the whole fleet")
        else:
            peers.append(island)
    return out


def launch_compat_group_of(args: Any) -> str | None:
    """The compat group a one-cluster launch exports to its island (the launcher's
    ``rl_island_task._island_compat_group``: the ``--gpu`` card's catalog group, or
    ``--rl-compat-group``). None for a launch of several clusters (the syncer launch)
    or a card the catalog does not know (the island task refuses it with its own message)."""
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from yeto.hw.catalog import UnknownCardType, launch_compat_group  # noqa: PLC0415

    try:
        specs = parse_gpu_spec(str(getattr(args, "gpu", None) or ""))
        if len(specs) != 1:
            return None
        return launch_compat_group(specs[0].gpu, getattr(args, "rl_compat_group", None))
    except (UnknownCardType, ValueError, KeyError):
        return None


def _where(island: LaunchSession) -> str:
    return f"island {island.name} (learner {island.learner_id})"


def identity_problems(syncer: LaunchSession, islands: list[LaunchSession],
                      running: Iterable[LaunchSession] = ()) -> list[str]:
    """The islands whose session contract would differ from the others' only by the
    backend identity: a compat group or an identity mode of their own. ``running``: the
    launches holding the fleet's running session, compared with but not reported for.
    The syncer admits a different compat group only when it and the island run
    ``--rl-identity relaxed`` (the island then binds the semantic identity)."""
    from yeto.hw.catalog import compat_refusal  # noqa: PLC0415

    out: list[str] = []
    held = [r for r in running if r.compat_group is not None]
    for index, island in enumerate(islands):
        if island.compat_group is None:
            continue
        peers = held + [i for i in islands[:index] if i.compat_group is not None]
        for peer in peers:
            if peer.strict_identity != island.strict_identity:
                mode = lambda s: "strict" if s.strict_identity else "relaxed"  # noqa: E731
                out.append(f"{_where(island)}: --rl-identity {mode(island)}, {_where(peer)} runs "
                           f"--rl-identity {mode(peer)}: their session contracts bind different backend "
                           "identities and the syncer refuses the later HELLO (session_contract_mismatch)")
                break
            strict = island.strict_identity or syncer.strict_identity
            if compat_refusal(peer.compat_group, island.compat_group, strict=strict) is not None:
                others = sorted({p.name for p in peers if p.compat_group == peer.compat_group})
                out.append(
                    f"{_where(island)}: compat_group {island.compat_group}, the fleet's session holds "
                    f"compat_group {peer.compat_group} (island(s) {', '.join(others)}): under strict identity "
                    "the compat group is part of the session contract, so the syncer refuses this island's "
                    "HELLO (session_contract_mismatch; same layout, different session_contract_hash). Put it on "
                    f"a card of compat_group {peer.compat_group}, or relax it explicitly for the whole fleet: "
                    "--rl-compat-group <vendor>-<card> (one declared group for every island) or --rl-identity "
                    "relaxed (compat_group is provenance only); numeric tolerance across cards is not calibrated")
                break
    return out


def synced_shape_problems(trainer_argv: list[str], names: list[str]) -> list[str]:
    """:func:`shape_problems` of a synced-group fleet (one trainer launch of every island,
    ``yeto run up``; :func:`check` compiles per-island fleets only): the shape of each
    ``--gpu`` cluster of ``trainer_argv`` (island ``names[i]``, learner ``i``) under the
    launch parser's reading (a launch that does not parse is left to its own dry run)."""
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in trainer_argv])
    shapes = None if args is None else launch_shapes_of(args)
    if not shapes or len(shapes) != len(names):
        return []
    return shape_problems([LaunchSession(name=name, learner_id=i, profile={}, profile_sha256="", learners=0, hello="",
                                         syncer_epoch=0, max_learners=None, shape=shape)
                           for i, (name, shape) in enumerate(zip(names, shapes))])


def problems(syncer: LaunchSession, islands: Iterable[LaunchSession]) -> list[str]:
    """What each island launch does not share with the syncer launch (empty: they agree)."""
    out: list[str] = []
    seen: dict[int, str] = {}
    for island in islands:
        where = _where(island)
        if island.profile_sha256 != syncer.profile_sha256:
            fields = {k: (syncer.profile.get(k), island.profile.get(k))
                      for k in sorted(set(syncer.profile) | set(island.profile))
                      if syncer.profile.get(k) != island.profile.get(k)}
            out.append(f"{where}: syncer semantic profile {island.profile_sha256[:12]} != the syncer launch's "
                       f"{syncer.profile_sha256[:12]} (syncer, island): {fields}")
        if island.learners != syncer.learners:
            out.append(f"{where}: {island.learners} learner(s), the syncer waits for {syncer.learners}")
        if island.hello != syncer.hello:
            out.append(f"{where}: HELLO flags differ: syncer launch [{syncer.hello}] island [{island.hello}]")
        if island.syncer_epoch < syncer.syncer_epoch:
            out.append(f"{where}: JOINs with syncer_epoch {island.syncer_epoch}, below the syncer's "
                       f"{syncer.syncer_epoch} (fenced)")
        if island.learner_id is None:
            out.append(f"{where}: no --rl-fleet-learner-id")
        else:
            if island.learner_id in seen:
                out.append(f"{where}: learner id {island.learner_id} is island {seen[island.learner_id]}'s too")
            seen.setdefault(island.learner_id, island.name)
            if syncer.max_learners is not None and not 0 <= island.learner_id < syncer.max_learners:
                out.append(f"{where}: learner id outside 0..{syncer.max_learners - 1} (the syncer's "
                           "--rl-sync-max-learners)")
    return out


def check(syncer_argv: list[str], members: Iterable[dict[str, Any]], *, syncer_epoch: int | None = None,
          prepare: Callable[[Any], None] | None = None, running: Iterable[dict[str, Any]] = ()) -> None:
    """Raise ValueError naming every disagreement between the syncer launch (``syncer_argv``)
    and the island launches of ``members`` (trainer members of the desired state), each
    compiled with the syncer record's term (``syncer_epoch``; None: the syncer launch's),
    and between the backend identities of ``members`` and ``running`` (the trainer
    members whose launches hold the fleet's running session; compared with only: one
    whose launch does not compile is left out with a note)."""
    from yeto.rl import island_launches as il  # noqa: PLC0415

    try:
        syncer = launch_session(syncer_argv, "syncer launch", prepare=prepare)
    except (OSError, ValueError) as error:  # a missing or broken file it reads (algorithm spec, inventory)
        raise ValueError(f"the syncer launch refuses its options: {type(error).__name__}: {error}") from None
    record = {"addr": PLACEHOLDER_ADDR, "syncer_epoch": syncer.syncer_epoch if syncer_epoch is None else syncer_epoch}
    islands = []
    for member in members:
        try:
            islands.append(launch_session(il.launch_argv(member, syncer=record), str(member.get("name")),
                                          prepare=prepare))
        except (OSError, ValueError) as error:  # json.JSONDecodeError is a ValueError
            raise ValueError(f"island {member.get('name')} (learner {member.get('learner_id')}): its launch "
                             f"refuses its options: {type(error).__name__}: {error}") from None
    held = []
    for member in running:
        try:
            held.append(launch_session(il.launch_argv(member, syncer=record), str(member.get("name")),
                                       prepare=prepare))
        except (OSError, ValueError) as error:
            print(f"fleet session: island {member.get('name')}'s launch was not compiled for the identity "
                  f"check ({type(error).__name__}: {error})", file=sys.stderr)
    found = problems(syncer, islands) + identity_problems(syncer, islands, held) + shape_problems(islands, held)
    if found:
        raise ValueError("the fleet's launches do not agree on the syncer session (the syncer would refuse "
                         "these islands at HELLO): " + "; ".join(found))
