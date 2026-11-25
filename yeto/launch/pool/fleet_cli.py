"""``yeto fleet show|apply``: read and change the fleet's desired state (``desired``).

``yeto fleet show --config <run.json>`` prints the newest generation of
``fleet/desired.json`` in the run store. ``yeto fleet apply --config <run.json> --file
<desired.json> [--generation N]`` writes the file as the next generation: with
``--generation`` only when the store is still at N (else refused: read again), without
it on top of the generation read just before. ``--dry-run`` checks the document and
prints what would change, pool by pool. A scale up (a target raised, a pool added) goes
through the run's own check first, as ``yeto run scale`` does (the run's breaker, the
budget): refused with exit code 4. Needs ``rollout.control_plane: "store"``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import desired as dz


def add_parser(sub: Any) -> None:
    p = sub.add_parser("fleet", help="the fleet's desired state (rollout.control_plane store): show, apply")
    fsub = p.add_subparsers(dest="fleet_command", metavar="{show,apply}")
    fsub.required = True
    show = fsub.add_parser("show", help="print the newest generation of fleet/desired.json")
    show.add_argument("--config", required=True, help="the run config")
    show.add_argument("--generation", type=int, default=None, help="an older generation")
    apply = fsub.add_parser("apply", help="write a desired state as the next generation (conditional)")
    apply.add_argument("--config", required=True, help="the run config")
    apply.add_argument("--file", required=True, help="the desired state document (JSON)")
    apply.add_argument("--generation", type=int, default=None,
                       help="the generation the file was edited from: refused when the store moved on")
    apply.add_argument("--note", default=None, help="why (kept in the generation)")
    apply.add_argument("--dry-run", action="store_true", help="check and show the change; nothing is written")


def _store(config: str) -> Any:
    from yeto.rl import up  # noqa: PLC0415

    run = up.open_run(config)
    if up.rollout_section(run.cfg).get("control_plane") != "store":
        raise SystemExit("fleet: needs rollout.control_plane \"store\" in the run config")
    return run, run.desired_store()


def changes(old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    """What applying ``new`` over ``old`` does, pool by pool (the reconcile's view)."""
    before, after = dz.sampler_pools(old), dz.sampler_pools(new)
    out = [f"add pool {n} (target {dz.pool_target(after[n])})" for n in sorted(set(after) - set(before))]
    out += [f"remove pool {n} (target 0, drain, shards retired, machines released)"
            for n in sorted(set(before) - set(after))]
    for n in sorted(set(before) & set(after)):
        if dz.pool_shape(before[n]) != dz.pool_shape(after[n]):
            out.append(f"reconfigure pool {n} (rolling replacement, one member at a time)")
        if dz.pool_target(before[n]) != dz.pool_target(after[n]) or after[n].get("retire"):
            out.append(f"pool {n} target {dz.pool_target(before[n])} -> {dz.pool_target(after[n])}"
                       + (f", retire {after[n]['retire']}" if after[n].get("retire") else ""))
    tb, ta = dz.trainers(old), dz.trainers(new)
    out += [f"trainer {n}: {'added' if n not in tb else 'removed' if n not in ta else 'changed'} "
            "(reconciled by the trainer worker)" for n in sorted(set(tb) ^ set(ta) | {
                k for k in set(tb) & set(ta) if tb[k] != ta[k]})]
    return out or ["no change"]


def scale_refusals(run: Any, old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    """The run's own check of every scale up in ``new`` (``Run.scale_refusal``: the run's
    breaker fired, the budget), as ``yeto run scale`` does: a pool added counts its whole target."""
    from yeto.launch.rollout.pools import resolve  # noqa: PLC0415

    before, after = dz.sampler_pools(old), dz.sampler_pools(new)
    ups = {n: dz.pool_target(m) for n, m in after.items() if dz.pool_target(m)
           and (n not in before or (dz.pool_target(before[n]) or 0) < (dz.pool_target(m) or 0))}
    if not ups:
        return []
    try:
        run.pools = resolve(run._desired_cfg(new), run.dir)  # noqa: SLF001 - priced as the new generation's pools
    except Exception as error:  # noqa: BLE001 - validate() passed; a pool the controller cannot build adds nothing
        print(f"fleet apply: WARNING: scale ups not checked against the run's budget ({error})", file=sys.stderr)
        return []
    out = []
    for i, ref in enumerate(run.pools):
        if ref.name not in ups:
            continue
        try:
            why = run.scale_refusal(int(ups[ref.name]), i)
        except Exception as error:  # noqa: BLE001 - its config cannot be read: the controller cannot build it either
            print(f"fleet apply: WARNING: pool {ref.name}: scale up not checked ({error})", file=sys.stderr)
            continue
        if why:
            out.append(f"pool {ref.name} target {ups[ref.name]}: {why}")
    return out


def main(args: Any) -> int:
    run, store = _store(args.config)
    if args.fleet_command == "show":
        doc = store.read(args.generation)
        if doc is None:
            print("fleet: no desired state yet (`yeto run up` seeds it from the run config)", file=sys.stderr)
            return 1
        print(json.dumps(doc, indent=1, sort_keys=True))
        return 0
    try:
        with open(args.file) as fh:
            new = dz.validate(json.load(fh))
    except (OSError, ValueError) as error:
        print(f"fleet apply: refused: {error}", file=sys.stderr)
        return 2
    current = store.read()
    expect = int(current["generation"]) if current is not None else 0
    if args.generation is not None and args.generation != expect:
        print(f"fleet apply: refused: {dz.DesiredConflict(args.generation, expect)}", file=sys.stderr)
        return 3
    for line in changes(current, new):
        print(f"fleet apply: {line}")
    refused = scale_refusals(run, current, new)
    for line in refused:
        print(f"fleet apply: refused: {line}", file=sys.stderr)
    if refused:
        return 4
    if args.dry_run:
        print(f"fleet apply: dry run: generation {expect + 1} would be written; nothing changed")
        return 0
    try:
        doc = store.apply(new, expect=expect if args.generation is None else args.generation, note=args.note)
    except dz.DesiredConflict as error:
        print(f"fleet apply: refused: {error}", file=sys.stderr)
        return 3
    print(f"fleet apply: generation {doc['generation']} written; the run's controller applies it within seconds")
    return 0
