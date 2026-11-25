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
* each island's learner id: unique, and below the syncer's ``--rl-sync-max-learners``.

``yeto run up`` (dry run too) checks the trainer islands of the desired state (else the
run config's) at its preflight; ``yeto fleet island add/move`` checks the island it
writes.
"""

from __future__ import annotations

import contextlib
import io
import re
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
                         max_learners=None if max_learners is None else int(max_learners))


def problems(syncer: LaunchSession, islands: Iterable[LaunchSession]) -> list[str]:
    """What each island launch does not share with the syncer launch (empty: they agree)."""
    out: list[str] = []
    seen: dict[int, str] = {}
    for island in islands:
        where = f"island {island.name} (learner {island.learner_id})"
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
          prepare: Callable[[Any], None] | None = None) -> None:
    """Raise ValueError naming every disagreement between the syncer launch (``syncer_argv``)
    and the island launches of ``members`` (trainer members of the desired state), each
    compiled with the syncer record's term (``syncer_epoch``; None: the syncer launch's)."""
    from yeto.rl import island_launches as il  # noqa: PLC0415

    syncer = launch_session(syncer_argv, "syncer launch", prepare=prepare)
    record = {"addr": PLACEHOLDER_ADDR, "syncer_epoch": syncer.syncer_epoch if syncer_epoch is None else syncer_epoch}
    islands = []
    for member in members:
        try:
            islands.append(launch_session(il.launch_argv(member, syncer=record), str(member.get("name")),
                                          prepare=prepare))
        except ValueError as error:
            raise ValueError(f"island {member.get('name')} (learner {member.get('learner_id')}): its launch "
                             f"refuses its options: {error}") from None
    found = problems(syncer, islands)
    if found:
        raise ValueError("the fleet's launches do not agree on the syncer session (the syncer would refuse "
                         "these islands at HELLO): " + "; ".join(found))
