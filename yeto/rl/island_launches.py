"""An island fleet launched island by island (``fleet.launch: per_island``).

By default (``fleet.launch: synced_group``) :func:`yeto.rl.fleet.compile_fleet` makes
one ``yeto launch`` of every trainer island and the syncer: one island lost for good
ends with the launch, and no island can join while it trains. With ``per_island``
the fleet is ``1 + K`` launches that share one syncer:

* the **syncer launch** (what ``yeto run up`` starts as its trainer): the synced-group
  argv with ``--rl-fleet-role syncer``. It starts the syncer only (no island; ``--gpu``
  still names every declared trainer, which the syncer's profile and the fragment
  preflight read) and records its address and term in the run store,
  :data:`SYNCER_RECORD`;
* one **island launch** per trainer island k: ``fleet.trainer_argv`` with that island's
  ``--gpu``, ``--rl-island-buffer-shards 0:<its shards>`` (its one island is island 0
  of its launch), its own release cadence / policy age / sample wait,
  ``--rl-fleet-role island --rl-fleet-learner-id k --rl-fleet-learners K``, a cluster
  prefix of its own (:func:`island_prefix`), ``--controller local`` and
  ``--island-lifecycle on`` (its record ``fleet/trainers/<k>/state.json``). The
  trainer worker (:mod:`yeto.rl.trainer_worker`) adds ``--rl-fleet-syncer`` and
  ``--rl-syncer-epoch`` from the syncer record when it starts the launch.

Every launch carries ``--rl-sync-max-learners M`` (``fleet.max_islands``, default K):
the syncer admits learner ids ``0..M-1`` and every island binds that roster into its
session contract, so an island added later (learner id from
:func:`yeto.launch.learner_ids.allocate_learner_id`) joins the running session
through re-JOIN and catch-up.

The islands are trainer members of the fleet's desired state (``fleet/desired.json``,
:mod:`yeto.launch.pool.desired`), one per island (:func:`trainer_member`)::

    {"kind": "trainer", "name": "<island name>", "learner_id": k,
     "gpu": "<its --gpu cluster>", "usd_per_h": <its trainer's price or null>,
     "argv": [<its launch argv without --gpu / --cluster-prefix / the syncer flags>],
     "prefix": "<its cluster prefix>", "pools": ["i<k>-<pool>", ...],
     "fleet_shape": "<--rl-fleet-trainer-shape of the fleet>"}   # added / moved islands

A change of ``gpu`` (or ``argv``) moves the island: the worker stops its launch and
starts it again on the new machines under the same learner id, from its own
``trainer<k>/`` releases (:mod:`yeto.rl.trainer_worker`).

Import-light: no torch, ray or miles.
"""

from __future__ import annotations

import json
import time
from typing import Any

SYNCER_RECORD = "fleet/island_syncer.json"
SYNCER_SCHEMA = "yeto.fleet.syncer/v1"
LAUNCH_DEFAULT_NAME = "yeto"  # yeto launch's run name without --cluster-prefix


def _argv_value(argv: list[str], flag: str) -> str | None:
    value = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    return value


SYNCER_ONLY_FLAGS = ("--rl-fleet-syncer-grace-s",)  # fleet.trainer_argv flags of the syncer launch only


def _without(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    """``argv`` without ``flags`` and their values (``--flag V`` or ``--flag=V``)."""
    out, skip = [], False
    for a in (str(x) for x in argv):
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in flags):
            continue
        out.append(a)
    return out


def fleet_prefix(trainer_argv: list[str]) -> str:
    """The fleet's run name: ``--cluster-prefix`` of ``fleet.trainer_argv`` (the syncer
    launch's run name; ``yeto run up`` stops and downs it as its trainer)."""
    return _argv_value([str(a) for a in trainer_argv], "--cluster-prefix") or LAUNCH_DEFAULT_NAME


def island_prefix(base: str, incarnation: int = 0) -> str:
    """The cluster prefix of an island launch: its member ``prefix`` for its first
    machines, ``<prefix>-m<n>`` after its n-th move (a new run name: the moved-from
    launch's record stays as it ended)."""
    return base if int(incarnation) <= 0 else f"{base}-m{int(incarnation)}"


def fleet_flags(plan: Any) -> list[str]:
    """What every launch of a per-island fleet adds: the syncer's learner id limit."""
    return ["--rl-sync-max-learners", str(int(plan.max_islands))]


SYNCER_SERVICE_FLAG = "--rl-fleet-syncer-service"  # the syncer launch of a fleet with an on-site syncer service


def syncer_argv(group_argv: list[str], plan: Any) -> list[str]:
    """The syncer launch (module doc): the synced-group argv, syncer role only. With an
    on-site syncer service (``fleet.syncer_service``) the launch watches that service
    instead of starting a syncer (:func:`yeto.launch.fleet_syncer.watch_service`), on
    this host (``--controller local``: no head VM)."""
    service = getattr(plan, "syncer_service", None)
    return [*group_argv, "--rl-fleet-role", "syncer", "--rl-fleet-learners", str(len(plan.islands)),
            *fleet_flags(plan), *((SYNCER_SERVICE_FLAG, str(service), "--controller", "local") if service else ())]


def island_argv(trainer_argv: list[str], island: Any, plan: Any, *, learners: int,
                section_argv: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Island ``island``'s launch argv without ``--gpu`` and ``--cluster-prefix`` (the
    worker adds both) and without the syncer flags (module doc). ``learners``: the
    fleet's launched trainers (the syncer's ``--learners``). ``section_argv``: the flags
    the compile adds for the run config's ``breakers.trainer.*`` (every island's)."""
    argv = [*_without(trainer_argv, SYNCER_ONLY_FLAGS), "--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer"]
    argv += [str(a) for a in section_argv]
    if island.pools:
        argv += ["--rl-island-buffer-shards", f"0:{','.join(island.pool_shards())}"]
    if island.release_every is not None:
        argv += ["--rl-island-release-every", f"0:{island.release_every}"]
    if island.max_policy_age is not None:
        argv += ["--rl-island-max-policy-age", f"0:{island.max_policy_age}"]
    if island.sample_wait_s is not None:
        argv += ["--rl-island-sample-wait-s", f"0:{island.sample_wait_s:g}"]
    if getattr(island, "buffer_max_prompt_len", None) is not None:
        argv += ["--rl-island-buffer-max-prompt-len", f"0:{island.buffer_max_prompt_len}"]
    argv += ["--rl-fleet-role", "island", "--rl-fleet-learner-id", str(int(island.learner)),
             "--rl-fleet-learners", str(int(learners)), *fleet_flags(plan),
             "--controller", "local", "--island-lifecycle", "on"]
    return argv


def member_table_flags(k: int) -> list[str]:
    """The ``--island-env`` flags of island ``k`` on the store control plane: it follows
    the member tables of its own pools only (lane ``i<k>``: the pools a fleet compiles to
    ``<shard_base>-i<k>``)."""
    from yeto.rl.engine.member_table import MEMBER_LANES_ENV, MEMBER_TABLE_ENV  # noqa: PLC0415

    return ["--island-env", f"{MEMBER_TABLE_ENV}=1", "--island-env", f"{MEMBER_LANES_ENV}=i{int(k)}"]


def trainer_member(island: Any, plan: Any, argv: list[str], *, prefix: str,
                   fleet_shape: str | None = None) -> dict[str, Any]:
    """The desired state's trainer member of ``island`` (module doc)."""
    member: dict[str, Any] = {
        "kind": "trainer", "name": island.name, "learner_id": int(island.learner), "gpu": island.gpu,
        "usd_per_h": island.trainer_usd_per_h, "argv": list(argv), "prefix": prefix,
        "pools": [p.name for p in island.pools]}
    if fleet_shape:
        member["fleet_shape"] = fleet_shape
    return member


def island_section_argv(cfg: dict[str, Any], trainer_argv: list[str]) -> list[str]:
    """What the compile adds to every island launch of a per-island fleet: the flags of the
    run config's ``breakers.trainer.*`` and its roster's (``fleet.island_lease_s`` when set,
    :func:`yeto.rl.fleet.per_island_roster`; never a synced group's roster options)."""
    from yeto.rl import fleet as fleet_mod  # noqa: PLC0415

    section = fleet_mod._trainer_section_argv(cfg, trainer_argv)  # noqa: SLF001 - the compile's own reading
    return [*section, *fleet_mod.per_island_roster(cfg).argv]


def trainer_members(cfg: dict[str, Any], plan: Any) -> list[dict[str, Any]]:
    """The trainer members of a compiled per-island fleet config (one per island)."""
    from yeto.rl import fleet as fleet_mod  # noqa: PLC0415

    trainer_argv = [str(a) for a in cfg["fleet"]["trainer_argv"]]
    section = island_section_argv(cfg, trainer_argv)
    base = fleet_prefix(trainer_argv)
    return [trainer_member(i, plan, island_argv(trainer_argv, i, plan, learners=len(plan.islands),
                                                section_argv=section), prefix=f"{base}-i{i.learner}")
            for i in plan.islands]


# Launch options that only a Modal island takes (a launch without one refuses them), by
# the number of values each takes: the fleet's trainer argv may carry them for its Modal
# islands; an island launch whose --gpu is elsewhere (added there, or moved there)
# launches without them (its store credential is the launcher's, yeto.launch.storage;
# its GPU is its --gpu, not a pinned Modal GPU).
MODAL_ONLY_FLAGS = {"--modal-store-secret": 1, "--modal-gpu-name": 1, "--modal-gpu-exact": 0}


def own_cloud_argv(argv: list[str], gpu: str) -> list[str]:
    """``argv`` for an island launch on ``gpu``: without :data:`MODAL_ONLY_FLAGS` (and
    their values) unless ``gpu`` is a Modal cluster."""
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415

    if all(getattr(spec, "cloud", None) == "modal" for spec in parse_gpu_spec(gpu)):
        return list(argv)
    out, skip = [], 0
    for arg in argv:
        if skip:
            skip -= 1
            continue
        if arg in MODAL_ONLY_FLAGS:
            skip = MODAL_ONLY_FLAGS[arg]
            continue
        if any(arg.startswith(flag + "=") for flag in MODAL_ONLY_FLAGS):
            continue
        out.append(arg)
    return out


def launch_argv(member: dict[str, Any], *, incarnation: int = 0, syncer: dict[str, Any] | None = None) -> list[str]:
    """The ``yeto launch`` argv of a trainer member: its argv, its ``--gpu``, its cluster
    prefix for ``incarnation`` and, with the syncer record, ``--rl-fleet-syncer`` /
    ``--rl-syncer-epoch``; a moved or added island also checks its shape against the
    fleet's (``--rl-fleet-trainer-shape``, refused at dry-run when it differs)."""
    argv = [*own_cloud_argv([str(a) for a in member.get("argv") or ()], str(member["gpu"])),
            "--gpu", str(member["gpu"]), "--cluster-prefix", island_prefix(str(member["prefix"]), incarnation)]
    if member.get("fleet_shape"):
        argv += ["--rl-fleet-trainer-shape", str(member["fleet_shape"])]
    if syncer is not None:
        argv += ["--rl-fleet-syncer", str(syncer["addr"])]
        if syncer.get("syncer_epoch") is not None:
            argv += ["--rl-syncer-epoch", str(int(syncer["syncer_epoch"]))]
    return argv


def fleet_shape(trainer_argv: list[str]) -> str | None:
    """``--rl-fleet-trainer-shape`` of a fleet whose islands launch with ``trainer_argv``:
    the launch parser's reading of its trainer shape (None: the argv does not parse)."""
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in trainer_argv])
    if args is None:
        return None
    from yeto.launch import fragment_preflight  # noqa: PLC0415

    spec = getattr(fragment_preflight, "trainer_shape_spec", None)
    if callable(spec):
        try:
            return str(spec(args))
        except Exception:  # noqa: BLE001 - the plain shape below
            pass
    return f"{int(getattr(args, 'tensor_parallel', 1) or 1)}x{int(getattr(args, 'pipeline_parallel', 1) or 1)}"


# -- the syncer record --------------------------------------------------------------------------

def write_syncer_record(objects: Any, *, addr: str, syncer_epoch: int | None, learners: int,
                        max_learners: int | None, run: str, clock: Any = time.time,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The fleet syncer's address and term in the run store (the island launches dial it).
    ``extra``: fields an on-site syncer service adds (``service``, ``key_id``); none otherwise."""
    record = syncer_record_doc(addr=addr, syncer_epoch=syncer_epoch, learners=learners, max_learners=max_learners,
                               run=run, clock=clock, extra=extra)
    objects.write(SYNCER_RECORD, json.dumps(record, sort_keys=True).encode())
    return record


def syncer_record_doc(*, addr: str, syncer_epoch: int | None, learners: int, max_learners: int | None, run: str,
                      clock: Any = time.time, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The record :func:`write_syncer_record` writes."""
    return {"schema": SYNCER_SCHEMA, "addr": str(addr), "syncer_epoch": syncer_epoch, "learners": int(learners),
            "max_learners": None if max_learners is None else int(max_learners), "run": str(run),
            "at": round(clock(), 3), **dict(extra or {})}


def read_syncer_record(objects: Any) -> dict[str, Any] | None:
    """The fleet syncer's record (None: none yet, or unreadable)."""
    raw = objects.read(SYNCER_RECORD)
    if raw is None:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) and record.get("addr") else None
