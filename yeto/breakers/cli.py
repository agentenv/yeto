"""``yeto breaker {check,show,list}``.

``check`` is the read-only health check over a run's event tapes (:mod:`yeto.rl.breaker`);
``yeto breaker --trainer-log ...`` (no subcommand) still runs it. ``show`` prints every
breaker's effective setting for a run config and where it came from; ``list`` prints the
registry. Exit codes of ``show``: 0 ok, 1 a conflict or an error-level finding, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SUBCOMMANDS = ("check", "show", "list")


def add_parser(sub: Any) -> argparse.ArgumentParser:
    from yeto.rl.breaker import add_check_args  # noqa: PLC0415 - the check's flags live with the check

    p = sub.add_parser(
        "breaker", help="breakers: check a run's tapes against health limits, show effective settings",
        description="check: read-only health limits over a run's event tapes (exit 1 on a trip; "
                    "`yeto breaker --trainer-log ...` is the same). show: every breaker's effective "
                    "setting and its source. list: the registry. See docs/BREAKER.md.")
    group = p.add_subparsers(dest="breaker_command", metavar="{check,show,list}")
    check = group.add_parser("check", help="check a run's event tapes against health limits (exit 1 on a trip)",
                             description="Read-only health breakers over a run's event tapes; exit 1 when one "
                                         "trips. See docs/BREAKER.md.")
    add_check_args(check)
    for action in check._actions:  # the tapes are not needed when --config / --breakers asks for the rules only
        if "--trainer-log" in action.option_strings:
            action.required = False
    rules = check.add_mutually_exclusive_group()
    rules.add_argument("--config", help="run only the dependency rules over this run config (no tapes)")
    rules.add_argument("--breakers", help="run only the dependency rules over this breakers file (no tapes)")
    show = group.add_parser("show", help="effective breaker settings of a run config, with sources and checks")
    src = show.add_mutually_exclusive_group()
    src.add_argument("--config", help="run config (JSON); its breakers: section and legacy knobs are read")
    src.add_argument("--breakers", help="a bare breakers: object, or a whole run config (JSON)")
    show.add_argument("--spec", help="sampler spec (JSON), for the sampler / harness aliases")
    show.add_argument("--layer", choices=_layers(), help="only this layer")
    show.add_argument("--pool", help="the run's pool whose config, sampler spec and breakers.pools.<POOL> "
                      "overrides the pool / sampler / harness rows read (default: the run's only pool)")
    show.add_argument("--json", action="store_true", help="print the breakers.resolved.json object")
    show.add_argument("--all", action="store_true", help="also list semantic entries")
    listing = group.add_parser("list", help="the breaker registry (no config needed)")
    listing.add_argument("--layer", choices=_layers(), help="only this layer")
    fmt = listing.add_mutually_exclusive_group()
    fmt.add_argument("--markdown", action="store_true", help="markdown tables, one per layer (docs)")
    fmt.add_argument("--json", action="store_true", help="one JSON object")
    return p


def _layers() -> tuple[str, ...]:
    from yeto.breakers.spec import LAYERS  # noqa: PLC0415

    return LAYERS


def normalize_argv(argv: Sequence[str]) -> list[str]:
    """``breaker <check flags>`` (the spelling before the command group) -> ``breaker check ...``."""
    out = list(argv)
    if out and out[0] == "breaker" and (len(out) == 1 or out[1] not in (*SUBCOMMANDS, "-h", "--help")):
        out.insert(1, "check")
    return out


def main(args: argparse.Namespace) -> int:
    command = getattr(args, "breaker_command", None) or "check"
    if command == "check":
        if getattr(args, "config", None) or getattr(args, "breakers", None):
            return cmd_rules(args)
        if not getattr(args, "trainer_log", None):
            print("yeto breaker check: error: the following arguments are required: --trainer-log "
                  "(or --config / --breakers for the dependency rules)", file=sys.stderr)
            return 2
        from yeto.rl.breaker import cmd_breaker  # noqa: PLC0415

        return cmd_breaker(args)
    if command == "list":
        return cmd_list(args)
    return cmd_show(args)


# --- show ------------------------------------------------------------------------------------------


def _read_json(path: str, what: str) -> Any:
    file = Path(path).expanduser()
    try:
        return json.loads(file.read_text())
    except OSError as error:
        raise ValueError(f"{what} {file}: cannot be read ({error.strerror or error})") from None
    except ValueError as error:
        raise ValueError(f"{what} {file}: not valid JSON ({error})") from None


def _sources(args: argparse.Namespace, *, view: bool = False) -> tuple[Any, Mapping[str, Any] | None]:
    """The resolver sources of the command's input. ``view`` (``show``): a run config's only pool
    (or ``--pool``) fills the pool, sampler and harness rows (its config, env file and spec;
    ``--spec`` replaces only the spec)."""
    from yeto.breakers.resolve import Sources  # noqa: PLC0415

    spec = _read_json(args.spec, "sampler spec") if args.spec else None
    if args.config:
        cfg = _read_json(args.config, "run config")
        if not isinstance(cfg, dict):
            raise ValueError(f"run config {args.config}: must be a JSON object")
        from yeto.breakers import fleet  # noqa: PLC0415

        if fleet.is_fleet(cfg) and not view:  # fleet.trainer_argv's knobs; the rules read each island
            return Sources.from_run_config(fleet.fleet_run_config(cfg), spec=spec), cfg
        if view:  # the pool, sampler and harness rows: the run's pool, its samplers' spec (or --spec) and env
            from yeto.breakers.rules_facts import run_view_sources  # noqa: PLC0415

            seen = run_view_sources(cfg, pool=getattr(args, "pool", None), spec=spec)
            args.pool, args.view_pools = seen.pool, seen.pools
            args.view_notes = _env_notes(cfg, [seen.pool] if seen.pool is not None else [])
            return seen.sources, cfg
        return Sources.from_run_config(cfg, spec=spec), cfg
    if args.breakers:
        data = _read_json(args.breakers, "breakers file")
        if not isinstance(data, dict):
            raise ValueError(f"breakers file {args.breakers}: must be a JSON object")
        if "breakers" in data:  # a whole run config
            return Sources.from_run_config(data, spec=spec), data
        return Sources(section=data, spec=spec), None
    return Sources(spec=spec), None


def _env_notes(cfg: Mapping[str, Any], pools: list[str] | None = None) -> list[str]:
    """The env file problems of the run's pools (``pools``: these names; None: every pool),
    one line each: the values they would give are shown as defaults (or refused)."""
    from yeto.breakers.rules_facts import env_file_problems, run_pools  # noqa: PLC0415

    try:
        found = run_pools(cfg)
    except ValueError:
        return []
    return [f"WARNING: {p}" + (f" (pool {name})" if len(found) > 1 else "")
            for name, pool_cfg, _ in found if pools is None or name in pools for p in env_file_problems(pool_cfg)]


def cmd_rules(args: argparse.Namespace) -> int:
    """``check --config``: the dependency rules alone; exit 1 on an error finding or a conflict."""
    from yeto.breakers.resolve import load_rules, preflight, resolve  # noqa: PLC0415
    from yeto.breakers.spec import BreakerConfigError, RegistryError  # noqa: PLC0415

    args.spec = getattr(args, "spec", None)
    try:
        sources, cfg = _sources(args)
        rb = resolve(sources, strict=False)
        findings = preflight(rb, run_config=cfg, sources=sources)
    except (BreakerConfigError, RegistryError, ValueError) as error:
        print(f"[breaker] {error}", file=sys.stderr)
        return 2
    notes = _env_notes(cfg) if cfg is not None else []
    rules = load_rules()
    fmt = getattr(rules, "format_finding", None) or (lambda f: f"{f.rule:<4} {f.level:<6} {f.message}")
    if args.json:
        print(json.dumps({"findings": [f.to_json() for f in findings],
                          "conflicts": [c.to_json() for c in rb.conflicts]}, sort_keys=True))
    else:
        for f in findings:
            print(fmt(f))
        for c in rb.conflicts:
            print(f"conflict {c.name}: {c.message}")
    for note in notes:
        print(note, file=sys.stderr if args.json else sys.stdout)
    return 1 if rb.conflicts or any(f.level == "error" for f in findings) else 0


def cmd_show(args: argparse.Namespace) -> int:
    from yeto.breakers.resolve import preflight, resolve  # noqa: PLC0415
    from yeto.breakers.spec import BreakerConfigError, RegistryError  # noqa: PLC0415

    args.view_notes = []
    if args.config:
        try:
            raw = _read_json(args.config, "run config")
        except ValueError as error:
            print(f"[breaker] {error}", file=sys.stderr)
            return 2
        from yeto.breakers import fleet  # noqa: PLC0415

        if fleet.is_fleet(raw):
            return cmd_show_fleet(args, raw)
    try:
        sources, cfg = _sources(args, view=True)
        rb = resolve(sources, pool=args.pool, strict=False)
        findings = preflight(rb, run_config=cfg, sources=sources)
    except (BreakerConfigError, RegistryError, ValueError) as error:
        print(f"[breaker] {error}", file=sys.stderr)
        for note in args.view_notes:
            print(note, file=sys.stderr)
        return 2
    data = rb.to_json(findings)
    if args.config:  # run up's view: what it leaves to run.budget (yeto launch --breakers applies it all)
        from yeto.breakers.rules_facts import mark_run_enforced  # noqa: PLC0415

        mark_run_enforced(data, cfg)
    if args.layer:
        data["entries"] = [e for e in data["entries"] if e["layer"] == args.layer]
    for note in args.view_notes:  # ahead of the rows they are about (stderr: --json stays one object)
        print(note, file=sys.stderr if args.json else sys.stdout)
    if args.json:
        print(json.dumps(data, sort_keys=True, default=str))
    else:
        print(render_show(data, show_all=args.all))
        pools = getattr(args, "view_pools", ())
        if args.pool is None and len(pools) > 1:
            print(f"pool / sampler / harness rows show defaults: {len(pools)} pools ({', '.join(pools)}), "
                  "pass --pool NAME")
    errors = [f for f in findings if f.level == "error"]
    return 1 if rb.conflicts or errors else 0


def cmd_show_fleet(args: argparse.Namespace, cfg: Mapping[str, Any]) -> int:
    """``show --config`` of a fleet run config: the fleet's own rows (run.budget, run.wall_clock,
    the launch layer; pool / sampler / harness / monitor rows as for one trainer, ``--pool``
    naming a pool ``i<k>-<name>``), then each island's run and trainer rows and its findings."""
    from yeto.breakers import fleet, rules  # noqa: PLC0415
    from yeto.breakers.resolve import resolve  # noqa: PLC0415
    from yeto.breakers.rules_facts import collect, mark_run_enforced, run_view_sources  # noqa: PLC0415
    from yeto.breakers.spec import BreakerConfigError, RegistryError  # noqa: PLC0415

    view = fleet.fleet_run_config(cfg)
    try:
        spec = _read_json(args.spec, "sampler spec") if args.spec else None
        seen = run_view_sources(view, pool=args.pool, spec=spec)
        args.pool = seen.pool
        notes = _env_notes(view, [seen.pool] if seen.pool is not None else [])
        rb = resolve(seen.sources, pool=args.pool, strict=False)
        facts = collect(rb, run_config=cfg)
        findings = rules.evaluate(rb, facts)
    except (BreakerConfigError, RegistryError, ValueError) as error:
        print(f"[breaker] {error}", file=sys.stderr)
        return 2
    data = mark_run_enforced(fleet_show_data(rb, facts.islands, findings, layer=args.layer), cfg)
    for note in notes:  # ahead of the rows they are about (stderr: --json stays one object)
        print(note, file=sys.stderr if args.json else sys.stdout)
    if args.json:
        print(json.dumps(data, sort_keys=True, default=str))
    else:
        print(render_fleet_show(data, show_all=args.all))
        if args.pool is None and len(seen.pools) > 1:
            print(f"pool / sampler / harness rows show defaults: {len(seen.pools)} pools ({', '.join(seen.pools)}), "
                  "pass --pool NAME")
    groups = [data, *data["islands"]]
    return 1 if any(g["conflicts"] or any(f["level"] == "error" for f in g["findings"]) for g in groups) else 0


def _group(data: dict[str, Any], keep: Callable[[str], bool], layer: str | None) -> dict[str, Any]:
    """``data`` (a ``ResolvedBreakers.to_json``) cut to the entries ``keep`` takes (and
    ``layer``), with their conflicts and deprecations only."""
    data["entries"] = [e for e in data["entries"] if keep(e["name"]) and (layer is None or e["layer"] == layer)]
    names = {e["name"] for e in data["entries"]}
    data["conflicts"] = [c for c in data["conflicts"] if c["name"] in names]
    data["deprecations"] = [d for e in data["entries"] for d in e.get("deprecations", ())]
    return data


def fleet_show_data(rb: Any, islands: Iterable[Any], findings: Iterable[Any], *,
                    layer: str | None = None) -> dict[str, Any]:
    """The ``show --json`` object of a fleet: the fleet's ``breakers.resolved.json`` cut to its
    own rows (:func:`yeto.breakers.fleet.breaker_scope` other than ``island``) with the
    fleet-wide findings, plus ``islands``: per island (``rules_facts.IslandFacts``) its index,
    name, pools, run / trainer rows, findings, conflicts and deprecations."""
    from yeto.breakers import fleet  # noqa: PLC0415

    findings = list(findings)
    data = _group(rb.to_json([f for f in findings if not f.scope]),
                  lambda name: fleet.breaker_scope(name) != "island", layer)
    data["islands"] = []
    for isl in islands:
        mine = [f for f in findings if f.scope.get("island") == str(isl.island.index)]
        group = _group(isl.resolved.to_json(mine), lambda name: fleet.breaker_scope(name) == "island", layer)
        data["islands"].append({"index": isl.island.index, "name": isl.island.name,
                                "pools": [p.name for p in isl.facts.pools],
                                **{k: group[k] for k in ("entries", "findings", "conflicts", "deprecations")}})
    return data


def render_fleet_show(data: Mapping[str, Any], *, show_all: bool = False) -> str:
    """The ``show`` text of a fleet: the fleet's table, then one table per island."""
    from yeto.breakers.fleet import FLEET_ENTRIES  # noqa: PLC0415

    lines = [f"FLEET  {' and '.join(FLEET_ENTRIES)} span every island; launch, pool, sampler, harness and "
             "monitor rows as for one trainer", render_show(data, show_all=show_all)]
    for isl in data["islands"]:
        lines += ["", f"ISLAND {isl['index']}  {isl['name']}  pools: {', '.join(isl['pools']) or '-'}"]
        lines.append(render_show(isl, show_all=show_all) if isl["entries"] or isl["findings"] else "  (no rows)")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _effective(entry: Mapping[str, Any]) -> str:
    if entry["kind"] in ("invariant", "semantic") and not entry["params"]:
        return "-"
    parts = []
    for name, value in entry["params"].items():
        src = entry["sources"].get(name, {})
        suffix = f" (inherit {src['location'].split(' ', 1)[1]})" if src.get("kind") == "inherit" else ""
        parts.append(f"{name}={_fmt(value)}{suffix}")
    return " ".join(parts) or "-"


def _source(entry: Mapping[str, Any]) -> str:
    if entry["kind"] == "invariant":
        return "(cannot disable)"
    if entry["kind"] == "semantic":
        return f"(native: {entry.get('native', '')})"
    order = ("breakers", "switch", "legacy", "inherit", "default")
    best = min(entry["sources"].values(), key=lambda s: order.index(s["kind"]),
               default={"kind": "default", "location": "default"})
    return best["location"] + (" (legacy)" if best["kind"] == "legacy" else "")


def _marks(entry: Mapping[str, Any], findings: Iterable[Mapping[str, Any]]) -> str:
    marks = []
    if entry.get("implicit"):
        marks.append("!")
    if entry.get("redundant"):
        marks.append("!redundant")
    if entry.get("enforced_by"):  # applied by another layer's breaker, not by this entry's component
        marks.append(f"!enforced_by_{entry['enforced_by']}")
    marks += [f"!{f['rule']}" for f in findings if entry["name"] in f.get("items", ())]
    return " ".join(dict.fromkeys(marks))


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]


def render_show(data: Mapping[str, Any], *, show_all: bool = False) -> str:
    """The ``show`` table of a ``breakers.resolved.json`` object."""
    findings = data.get("findings", [])
    rows = [["NAME", "LAYER", "KIND", "ON", "ACTION", "EFFECTIVE", "SOURCE", ""]]
    for e in data["entries"]:
        if e["kind"] == "semantic" and not show_all:
            continue
        rows.append([e["name"], e["layer"], e["kind"], "on" if e["enabled"] else "off", e["action"] or "-",
                     _effective(e), _source(e), _marks(e, findings)])
    lines = _table(rows)
    if data.get("switches"):
        lines.append("SWITCHES")
        for name, s in sorted(data["switches"].items()):
            src = "" if s["source"] == "default" else f"  ({s['source']})"
            lines.append(f"  {name:<20} {'on' if s['on'] else 'off':<4} {', '.join(s['members'])}{src}")
    overrides = (data.get("deps") or {}).get("overrides") or {}
    if findings or overrides:
        lines.append("DEPENDENCIES")
        for f in findings:
            over = f"  [override {f['overridden']}]" if f.get("overridden") else ""
            lines.append(f"  {f['rule']:<4} {f['level']:<6} {f['message']}{over}")
        for rule, level in sorted(overrides.items()):
            lines.append(f"  override  {rule} -> {level}")
    if data.get("deprecations"):
        lines.append("DEPRECATED")
        lines += [f"  {line}" for line in data["deprecations"]]
    if data.get("conflicts"):
        lines.append("CONFLICTS")
        lines += [f"  {c['name']}: {c['message']}" for c in data["conflicts"]]
    return "\n".join(lines)


# --- list ------------------------------------------------------------------------------------------


def _default(b: Any) -> str:
    return " ".join(f"{p.name}={_fmt(p.default)}" for p in b.params) or "-"


def _alias_text(b: Any) -> str:
    return ", ".join(a.location + (" (deprecated)" if a.deprecated else "") for a in b.aliases) or "-"


def cmd_list(args: argparse.Namespace) -> int:
    from yeto.breakers.registry import default_registry  # noqa: PLC0415
    from yeto.breakers.spec import LAYERS, RegistryError  # noqa: PLC0415

    try:
        reg = default_registry()
    except RegistryError as error:
        print(f"[breaker] {error}", file=sys.stderr)
        return 2
    layers = [args.layer] if args.layer else list(LAYERS)
    entries = [b for layer in layers for b in reg.layer(layer)]
    if args.json:
        print(json.dumps({"breakers": [_list_row(b, reg) for b in entries],
                          "switch_groups": [{"name": g.name, "members": list(g.members),
                                             "aliases": [a.location for a in g.aliases]}
                                            for g in reg.switch_groups]}, sort_keys=True))
        return 0
    if args.markdown:
        print(render_markdown(reg, layers))
        return 0
    rows = [["NAME", "LAYER", "KIND", "ON", "ACTION", "DEFAULT", "ALIASES"]]
    for b in entries:
        rows.append([b.name, b.layer, b.kind, "on" if b.default_enabled else "off", b.default_action or "-",
                     _default(b), _alias_text(b)])
    print("\n".join(_table(rows)))
    return 0


def _list_row(b: Any, reg: Any) -> dict[str, Any]:
    return {"name": b.name, "layer": b.layer, "kind": b.kind, "default_enabled": b.default_enabled,
            "can_disable": b.can_disable, "actions": list(b.actions), "default_action": b.default_action,
            "params": [{"name": p.name, "default": p.default, "unit": p.unit, "nullable": p.nullable,
                        "inherit_from": p.inherit_from} for p in b.params],
            "aliases": [a.location for a in b.aliases], "switch_groups": list(reg.groups_of(b.name)),
            "depends_on": list(b.depends_on), "owner": b.owner, "native": b.native, "doc": b.doc}


def render_markdown(reg: Any, layers: Iterable[str]) -> str:
    """One markdown table per layer (the reference tables of docs/BREAKERS.md)."""
    out = []
    for layer in layers:
        entries = reg.layer(layer)
        if not entries:
            continue
        out += [f"### {layer}", "", "| Name | Kind | Default | Action | Can disable | Legacy knobs | Notes |",
                "|---|---|---|---|---|---|---|"]
        for b in entries:
            cells = [f"`{b.name}`", b.kind, _default(b), b.default_action or "-",
                     "yes" if b.can_disable else "no", _alias_text(b), b.doc or b.native]
            out.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        out.append("")
    if reg.switch_groups:
        out += ["### switch groups", "", "| Group | Members | Legacy switch |", "|---|---|---|"]
        for g in reg.switch_groups:
            out.append(f"| `{g.name}` | {', '.join(g.members)} | {', '.join(a.location for a in g.aliases) or '-'} |")
        out.append("")
    return "\n".join(out).rstrip()
