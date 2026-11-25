"""Add, remove and move trainer islands of a running fleet (``yeto fleet island``).

A fleet launched island by island (``fleet.launch: per_island``,
:mod:`yeto.rl.island_launches`) keeps its islands in the desired state
(``fleet/desired.json``). Each command here is one new generation of it
(``DesiredStore.update``: a conditional write, retried on a conflict); the run's
controller applies it within seconds: the trainer worker starts, stops or moves the
island launch (:mod:`yeto.rl.trainer_worker`), the pool reconcile adds or drains its
sampler pools (``yeto.launch.pool.fleet_reconcile``).

* ``add``: a new island with its own sampler pools. Its learner id is the lowest one
  below ``fleet.max_islands`` no island ever had (:func:`yeto.launch.learner_ids.
  allocate_learner_id`, a claim in the run store, never reused); its pools are compiled
  as the fleet compile does for island k (``i<k>-<name>``, store ``trainer<k>``, shards
  ``<shard_base>-i<k>-<n>``, the fleet's task ledger); its launch dials the running
  syncer, catches up to the newest outer weights and merges from then on. It checks its
  trainer shape against the fleet's (``--rl-fleet-trainer-shape``).
* ``remove``: the island and its pools leave the desired state: its launch stops on
  its island STOP, its pools drain (claims back to the ledger) and their shards are
  retired, so the trainer data end never waits for them. The syncer merges over the
  remaining islands.
* ``move``: the island's machines change (``--gpu``, e.g. from Modal to an AWS spot
  cluster); same learner id, same store sub-directory: the next launch continues from
  ``trainer<k>/``. The syncer sees the island leave and come back (no RETIRE).

``--dry-run`` writes nothing: it prints the change and runs the island launch's
``yeto launch --dry-run`` (an island whose layout differs from the fleet's is refused
there, before anything is ordered).
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from typing import Any

from yeto.launch.pool import desired as dz
from yeto.rl import fleet as fleet_mod
from yeto.rl import island_launches as il


class IslandChangeError(ValueError):
    """A change the fleet cannot take (named island unknown, a name taken, no id left, ...)."""


def _plan(cfg: dict[str, Any]) -> Any:
    plan = fleet_mod.plan_of(cfg)
    if plan is None or not plan.per_island:
        raise IslandChangeError("yeto fleet island: the run config is not an island fleet with fleet.launch per_island")
    return plan


def trainer_by_name(doc: dict[str, Any], name: str) -> dict[str, Any]:
    member = dz.trainers(doc).get(name)
    if member is None:
        raise IslandChangeError(f"no trainer island named {name!r} in the desired state "
                                f"(islands: {', '.join(sorted(dz.trainers(doc))) or 'none'})")
    return member


def unused_claim(objects: Any, doc: dict[str, Any], name: str) -> int | None:
    """A learner id claimed for island ``name`` by an earlier ``add`` whose desired state
    write then failed: no trainer member has it, no island record exists under it (it
    never launched) and nothing was written under its store sub-directory
    (``trainer<k>/``: an island added and removed before it started may have had pools
    that sampled into it). That add's retry takes it again, so a failed write burns no id."""
    from yeto.launch.island_lifecycle import STATE_DIR  # noqa: PLC0415
    from yeto.launch.learner_ids import CLAIM_NAME  # noqa: PLC0415
    from yeto.rl.trainer_worker import LAUNCH_RECORD  # noqa: PLC0415

    in_doc = {int(m["learner_id"]) for m in dz.trainers(doc).values()}
    files: dict[int, set[str]] = {}
    for key in objects.list(f"{STATE_DIR}/"):
        parts = str(key).split("/")
        if len(parts) >= 4 and parts[-2].isdigit():
            files.setdefault(int(parts[-2]), set()).add(parts[-1])
    for k in sorted(files):
        if k in in_doc or CLAIM_NAME not in files[k] or files[k] & {"state.json", LAUNCH_RECORD}:
            continue
        try:
            claim = json.loads(objects.read(f"{STATE_DIR}/{k}/{CLAIM_NAME}") or b"{}")
        except ValueError:
            continue
        if isinstance(claim, dict) and claim.get("island") == str(name):
            if next(iter(objects.list(f"{fleet_mod.substore(k)}/")), None) is not None:
                continue  # its sub-directory holds data already: a fresh id instead
            return k
    return None


def next_learner_id(objects: Any, plan: Any, doc: dict[str, Any], name: str | None = None) -> int:
    """The id :func:`allocate_learner_id` would claim now (a dry run claims nothing), or
    the id an earlier failed ``add`` of ``name`` claimed (:func:`unused_claim`)."""
    from yeto.launch.learner_ids import taken_ids  # noqa: PLC0415

    if name is not None and (k := unused_claim(objects, doc, name)) is not None:
        return k
    taken = taken_ids(objects) | {int(m["learner_id"]) for m in dz.trainers(doc).values()}
    for k in range(len(plan.islands), int(plan.max_islands)):
        if k not in taken:
            return k
    raise IslandChangeError(f"no learner id left: ids 0..{int(plan.max_islands) - 1} are all taken "
                            "(fleet.max_islands; an id is never reused)")


def compile_island(cfg: dict[str, Any], plan: Any, *, k: int, name: str, gpu: str, usd_per_h: float | None,
                   samplers: list[Any], island_argv: list[str] | None = None) -> tuple[dict[str, Any], list[dict]]:
    """The trainer member and the sampler pool members of a new island k, compiled as
    :func:`yeto.rl.fleet.compile_fleet` compiles island k of the run config."""
    if not fleet_mod.NAME.fullmatch(str(name)):
        raise IslandChangeError(f"island name must be a word of letters, digits, '.', '_' or '-' (got {name!r})")
    where = f"island {name}"
    gpu = fleet_mod._gpu(gpu, f"{where} --gpu")  # noqa: SLF001
    price = fleet_mod._price(usd_per_h, f"{where} --usd-per-h")  # noqa: SLF001
    flags = fleet_mod._flag_values(list(island_argv or ()), f"{where} trainer argv")  # noqa: SLF001
    if not isinstance(samplers, list) or not samplers:
        raise IslandChangeError(f"{where}: give its sampler pools (a non-empty list of pool entries)")
    rows, members = [], []
    for j, entry in enumerate(samplers):
        if fleet_mod.is_miles_entry(entry):
            raise IslandChangeError(f"{where}: a fleet launched island by island takes external pools only")
        out, row, ports, _doc = fleet_mod._pool(entry, k, j, where, plan.dispense_store, plan.store)  # noqa: SLF001
        if ports is not None:
            raise IslandChangeError(f"{where}: pool {row.name} shares a port range with other pools (records relay / "
                                    "tunnel); an island added at run time cannot take a block of it")
        rows.append(row)
        pool = {key: value for key, value in out.items() if key != "name"}
        members.append({"kind": "sampler_pool", "name": row.name, "island": k,
                        "required": bool(pool.pop("required", True)), "pool": pool})
    island = fleet_mod.FleetIsland(index=k, name=name, gpu=gpu, usd_per_h=price,
                                   store=fleet_mod.substore_uri(plan.store, fleet_mod.substore(k)),
                                   release_every=flags.get("release_every"),
                                   max_policy_age=flags.get("max_policy_age"), pools=tuple(rows),
                                   trainer_usd_per_h=price, sample_wait_s=flags.get("sample_wait_s"))
    trainer_argv = [str(a) for a in cfg["fleet"]["trainer_argv"]]
    argv = il.island_argv(trainer_argv, island, plan, learners=len(plan.islands),
                          section_argv=il.island_section_argv(cfg, trainer_argv))
    argv += il.member_table_flags(k)
    member = il.trainer_member(island, plan, argv, prefix=f"{il.fleet_prefix(trainer_argv)}-i{k}",
                               fleet_shape=il.fleet_shape(trainer_argv))
    return member, members


def run_budget_usd(cfg: dict[str, Any]) -> float | None:
    """The run's ``run.budget`` in dollars, wherever the run config sets it (``budget.usd``
    or the ``breakers`` section), resolved as ``yeto run up`` resolves it; None when off."""
    from yeto import breakers  # noqa: PLC0415

    resolved = breakers.resolve(breakers.Sources.from_run_config(cfg), strict=False, layers=("run", "launch"))
    usd = resolved.get("run.budget.usd") if resolved.enabled("run.budget") else None
    if usd is None:
        budget = cfg.get("budget") if isinstance(cfg.get("budget"), dict) else {}
        usd = budget.get("usd")
    return None if usd is None else float(usd)


def unpriced_under_budget(cfg: dict[str, Any], gpu: str, usd_per_h: float | None) -> None:
    """Refuse an island (added, or moved to ``gpu``) without a price while the run has a
    budget: no ``--usd-per-h`` and no billing row of the unified price table for ``gpu``;
    its spend would count $0. Without a budget it passes (the spend warns)."""
    from yeto.rl import trainer_worker  # noqa: PLC0415

    if usd_per_h is None and run_budget_usd(cfg) is not None and trainer_worker.table_usd_per_h(gpu) is None:
        raise IslandChangeError(f"give the island's trainer price (--usd-per-h): the run has a budget (run.budget) "
                                f"and the price table has no billing row for {gpu}, so the island would bill nothing")


def add(doc: dict[str, Any], member: dict[str, Any], pools: list[dict[str, Any]]) -> None:
    """Add an island's trainer member and its pools to ``doc`` (in place)."""
    if member["name"] in dz.trainers(doc):
        raise IslandChangeError(f"a trainer island named {member['name']!r} is in the desired state already")
    if any(int(m["learner_id"]) == int(member["learner_id"]) for m in dz.trainers(doc).values()):
        raise IslandChangeError(f"learner id {member['learner_id']} belongs to another island")
    taken = set(dz.sampler_pools(doc))
    clash = [p["name"] for p in pools if p["name"] in taken]
    if clash:
        raise IslandChangeError(f"pool(s) {clash} are in the desired state already")
    doc.setdefault("members", []).extend([member, *pools])


def remove(doc: dict[str, Any], name: str) -> list[str]:
    """Remove island ``name`` and its pools from ``doc`` (in place); returns the pools removed."""
    member = trainer_by_name(doc, name)
    k = int(member["learner_id"])
    pools = set(member.get("pools") or ())
    gone = [m["name"] for m in doc["members"] if m.get("kind") == "sampler_pool"
            and (m["name"] in pools or m.get("island") == k)]
    doc["members"] = [m for m in doc["members"] if m is not member and m.get("name") not in gone]
    return gone


def move(doc: dict[str, Any], cfg: dict[str, Any], name: str, *, gpu: str, usd_per_h: float | None = None) -> dict:
    """Move island ``name`` to other machines (in place); returns its member. Its price is
    ``usd_per_h`` when given, else none of its own: the new machines are billed at the
    unified price table's price for their ``--gpu`` (``trainer_worker.table_usd_per_h``),
    never at the old machines' price."""
    member = trainer_by_name(doc, name)
    member["gpu"] = fleet_mod._gpu(gpu, f"island {name} --gpu")  # noqa: SLF001
    member["usd_per_h"] = (None if usd_per_h is None
                           else fleet_mod._price(usd_per_h, f"island {name} --usd-per-h"))  # noqa: SLF001
    shape = il.fleet_shape([str(a) for a in cfg["fleet"]["trainer_argv"]])
    if shape:
        member["fleet_shape"] = shape  # the moved island's layout is checked against the fleet's
    return member


# -- the launch's own dry run -------------------------------------------------------------------

def _prepare(args: Any) -> None:
    from yeto.launch.args import prepare_launch_args  # noqa: PLC0415

    prepare_launch_args(args)


def launch_dry_run(member: dict[str, Any], *, incarnation: int = 0,
                   prepare: Callable[[Any], None] | None = None) -> dict[str, Any]:
    """``yeto launch --dry-run`` of the island launch (its ``--rl-fleet-syncer`` a
    placeholder). Raises ValueError with the launch's refusal (e.g. a trainer shape whose
    layout differs from the fleet's)."""
    from yeto.cli import parse_args  # noqa: PLC0415
    from yeto.launcher import dry_run_plan  # noqa: PLC0415

    argv = il.launch_argv(member, incarnation=incarnation, syncer={"addr": "SYNCER:29400"})
    args = parse_args(argv)
    args.dry_run = True
    (prepare or _prepare)(args)
    return dry_run_plan(args)


# -- the command --------------------------------------------------------------------------------

def add_parser(fsub: Any) -> None:
    p = fsub.add_parser("island", help="add, remove or move a trainer island of a fleet launched island by island")
    isub = p.add_subparsers(dest="island_command", metavar="{add,remove,move}")
    isub.required = True
    a = isub.add_parser("add", help="a new island with its own sampler pools")
    a.add_argument("--name", required=True)
    a.add_argument("--gpu", required=True, help="its trainer cluster (yeto launch --gpu, one cluster)")
    a.add_argument("--usd-per-h", type=float, default=None, help="its trainer's price (spend view)")
    a.add_argument("--samplers", required=True, help="a JSON file: the island's samplers list (pool entries)")
    a.add_argument("--trainer-arg", action="append", default=None, metavar="ARG",
                   help="its own trainer flag (--rl-release-every / --rl-max-policy-age / --rl-sample-wait-s), "
                   "repeatable: --trainer-arg=--rl-release-every --trainer-arg=2")
    r = isub.add_parser("remove", help="the island leaves: its launch stops, its pools drain")
    r.add_argument("--name", required=True)
    m = isub.add_parser("move", help="the island moves to other machines under its learner id")
    m.add_argument("--name", required=True)
    m.add_argument("--gpu", required=True)
    m.add_argument("--usd-per-h", type=float, default=None)
    for q in (a, r, m):
        q.add_argument("--config", required=True, help="the run config")
        q.add_argument("--dry-run", action="store_true", help="show the change and dry-run the launch; write nothing")


def main(args: Any, *, prepare: Callable[[Any], None] | None = None) -> int:
    from yeto.launch.pool.fleet_cli import _store, changes  # noqa: PLC0415

    run, store = _store(args.config)
    try:
        plan = _plan(run.cfg)
        current = store.read()
        if current is None:
            raise IslandChangeError("the fleet has no desired state yet (`yeto run up` seeds it)")
        doc = copy.deepcopy(current)
        for m in doc.get("members") or ():
            for key in dz.ONE_SHOT:
                m.pop(key, None)
        launch_member = None
        if args.island_command == "add":
            unpriced_under_budget(run.cfg, args.gpu, args.usd_per_h)
            with open(args.samplers) as fh:
                samplers = json.load(fh)
            objects = run._objects()  # noqa: SLF001
            k = next_learner_id(objects, plan, doc, args.name)
            member, pools = compile_island(run.cfg, plan, k=k, name=args.name, gpu=args.gpu,
                                           usd_per_h=args.usd_per_h, samplers=samplers,
                                           island_argv=args.trainer_arg)
            add(doc, member, pools)
            launch_member = member
        elif args.island_command == "remove":
            remove(doc, args.name)
        else:
            unpriced_under_budget(run.cfg, args.gpu, args.usd_per_h)
            launch_member = move(doc, run.cfg, args.name, gpu=args.gpu, usd_per_h=args.usd_per_h)
        doc = dz.validate(doc)
        for line in changes(current, doc):
            print(f"fleet island: {line}")
        if launch_member is not None:
            launch_dry_run(launch_member, prepare=prepare)
            print(f"fleet island: launch of {launch_member['name']} (learner {launch_member['learner_id']}, "
                  f"--gpu {launch_member['gpu']}) passes its dry run")
    except (IslandChangeError, ValueError, OSError) as error:
        print(f"fleet island: refused: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"fleet island: dry run: generation {int(current['generation']) + 1} would be written; nothing changed")
        return 0
    if args.island_command == "add":
        from yeto.launch.learner_ids import allocate_learner_id  # noqa: PLC0415

        claimed = unused_claim(objects, current, args.name)
        if claimed is None:
            claimed = allocate_learner_id(objects, launched=len(plan.islands), max_learners=int(plan.max_islands),
                                          island=args.name)
        if claimed != int(launch_member["learner_id"]):  # another add claimed it meanwhile: compile again
            launch_member, pools = compile_island(run.cfg, plan, k=claimed, name=args.name, gpu=args.gpu,
                                                  usd_per_h=args.usd_per_h, samplers=samplers,
                                                  island_argv=args.trainer_arg)
    member_now = launch_member

    def change(fresh: dict[str, Any]) -> None:
        if args.island_command == "add":
            add(fresh, member_now, pools)
        elif args.island_command == "remove":
            remove(fresh, args.name)
        else:
            move(fresh, run.cfg, args.name, gpu=args.gpu, usd_per_h=args.usd_per_h)

    try:
        written = store.update(change, note=f"fleet island {args.island_command} {args.name}", attempts=3)
    except (dz.DesiredConflict, IslandChangeError, ValueError) as error:
        print(f"fleet island: refused: {error}", file=sys.stderr)
        return 3
    print(f"fleet island: generation {written['generation']} written; the run's controller applies it within seconds")
    return 0
