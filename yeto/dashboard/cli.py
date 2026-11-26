"""``yeto dashboard serve|export|mirror`` argument parsing and dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_parser(sub) -> None:
    d = sub.add_parser("dashboard", help="read-only RL fleet dashboard (serve over SSH tunnel / export HTML)")
    dsub = d.add_subparsers(dest="dashboard_command", metavar="{serve,export,mirror}")
    dsub.required = True

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", default=None, help="run name: follow ~/.yeto/runs/<run>/{events,fleet.jsonl} "
                       "and the head's syncer tape if present")
        p.add_argument("--tapes", nargs="*", default=[], metavar="PATH",
                       help="extra tape files or directories (*.jsonl up to 2 levels deep): learner tapes, "
                       "syncer tape, controller journal.jsonl, fleet.jsonl")
        p.add_argument("--store", default=None, metavar="URI",
                       help="run store (s3://bucket/runs/<run> or a directory): read every island's tape "
                       "uploaded under tapes/ (samplers on leased nodes, replacements included)")
        p.add_argument("--prices", default=None, help="price table JSON (default: built-in example, not a bill)")
        p.add_argument("--budget", type=float, default=None, help="budget cap in $ for the cost bar/alerts")
        p.add_argument("--up-state", default=None, metavar="PATH",
                       help="state.json of a `yeto run up` run: show its cumulative runtime (every up, "
                       "stopped gaps excluded) and cumulative spend estimate")
        p.add_argument("--thresholds", default=None, help="alert threshold overrides JSON")
        p.add_argument("--compare", nargs="*", default=[], metavar="PATH",
                       help="tapes of a second run to compare generation-stage utilization with "
                       "(side by side / overlay; loaded once, not followed)")
        p.add_argument("--label", default=None, help="display label of this run in the comparison (default A)")
        p.add_argument("--compare-label", default=None, help="display label of the --compare run (default B)")

    s = dsub.add_parser("serve", help="loopback-only GET-only HTTP server")
    common(s)
    s.add_argument("--host", default="127.0.0.1", help="loopback address only (0.0.0.0 is refused)")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--poll", type=float, default=2.0, help="tape follow interval seconds")
    s.add_argument("--store-poll", type=float, default=10.0, metavar="SECONDS",
                   help="seconds between two polls of --store (default 10; the islands upload "
                   "a tape chunk every 30 s, and every poll costs listing requests)")
    s.add_argument("--head", default=None, help="head host name, only used in the printed ssh -L hint")

    e = dsub.add_parser("export", help="single self-contained offline HTML")
    common(e)
    e.add_argument("-o", "--out", required=True, help="output .html")

    m = dsub.add_parser("mirror", help="mirror a remote learner tape over ssh (tail -F) into a local file")
    m.add_argument("--target", required=True, help="ssh target, e.g. user@host")
    m.add_argument("--remote", required=True, help="remote tape path")
    m.add_argument("--local", required=True, help="local mirror path (point serve --tapes at it)")
    m.add_argument("--island", default=None, help="island id recorded on dashboard_source_lost events")


def store_run_name(store: str | None) -> str | None:
    """The run of a ``--store .../runs/<run>`` URI or directory (the page's run label when
    no ``--run`` is given); None for any other layout."""
    parts = [p for p in str(store or "").split("://")[-1].strip("/").split("/") if p]
    return parts[-1] if len(parts) >= 2 and parts[-2] == "runs" else None


def page_commands(args) -> dict[str, str]:
    """The page's "Commands" box: this dashboard's own serve / export command lines,
    rebuilt from its arguments (not the defaults), so a copied command reproduces it."""
    import shlex

    src: list[str] = []
    if args.run:
        src += ["--run", args.run]
    if getattr(args, "store", None):
        src += ["--store", args.store]
    if args.tapes:
        src += ["--tapes", *args.tapes]
    for flag, key in (("--up-state", "up_state"), ("--prices", "prices"), ("--thresholds", "thresholds"),
                      ("--label", "label")):
        if getattr(args, key, None):
            src += [flag, str(getattr(args, key))]
    if getattr(args, "budget", None) is not None:
        src += ["--budget", f"{args.budget:g}"]
    serve = ["yeto", "dashboard", "serve", *src]
    if getattr(args, "host", "127.0.0.1") != "127.0.0.1":
        serve += ["--host", args.host]
    serve += ["--port", str(getattr(args, "port", 8787))]
    if getattr(args, "store", None) and float(getattr(args, "store_poll", 10.0) or 0) != 10.0:
        serve += ["--store-poll", f"{args.store_poll:g}"]
    name = args.run or getattr(args, "label", None) or "run"
    export = ["yeto", "dashboard", "export", *src, "-o", f"{name}.html"]
    return {"serve": shlex.join(serve), "export": shlex.join(export)}


def _reducer(args):
    from .cost import load_prices
    from .reducer import Reducer

    thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    reducer = Reducer(run=args.run or store_run_name(getattr(args, "store", None)),
                      prices=load_prices(args.prices), budget_usd=args.budget,
                      thresholds=thresholds)
    reducer.up_state = getattr(args, "up_state", None)
    reducer.commands = page_commands(args)
    return reducer


def _paths(args) -> list[str]:
    from .sources import run_sources, up_events_path

    paths = list(args.tapes)
    up_state = getattr(args, "up_state", None)
    if args.run:
        paths = run_sources(args.run, up_state=up_state) + paths
    elif up_state:  # the run up monitor's tape (run-layer breakers) sits next to its state file
        paths.append(up_events_path(up_state=up_state))
    if not paths and not getattr(args, "store", None):
        raise SystemExit("[dashboard] no tapes: pass --run <run>, --tapes PATH... and/or --store URI")
    missing = [p for p in paths if not Path(p).expanduser().exists()]
    if missing and len(missing) == len(paths) and not getattr(args, "store", None):
        if args.dashboard_command == "export":  # nothing to export (a mistyped run name, say)
            raise SystemExit(f"[dashboard] no tapes: none of {', '.join(missing)} exists")
    for p in missing:  # serve follows them once they appear (a dashboard started before the run)
        print(f"[dashboard] waiting for {p} (not there yet)", file=sys.stderr)
    return paths


def _store(args):
    if not getattr(args, "store", None):
        return None
    from .fleet_store import store_readers
    from .store_tapes import StoreTapes, open_objects

    interval = float(getattr(args, "store_poll", None) or 0.0)
    # the island tapes, and the launchers' fleet.jsonl copies (fleet/log/, fleet_store)
    return store_readers(StoreTapes(open_objects(args.store), label="store", interval_s=interval),
                         interval_s=interval)


def _resolved(args, paths: list[str], store=None):
    """The run's ``breakers.resolved.json`` (Breakers panel): next to ``--up-state``, in
    the ``run up`` state directory of ``--run``, beside a tape, or under the store's ``meta/``."""
    from .breakers import ResolvedFile, resolved_candidates

    return ResolvedFile(resolved_candidates(paths, run=args.run, up_state=getattr(args, "up_state", None)),
                        getattr(store, "objects", None))


def _plan(paths: list[str], store=None, interval_s: float = 0.0):
    """A fleet's ``fleet/plan.json``: under a local store directory given as a tape path, or
    under the run store (``fleet_islands.PlanFile``); also reads the islands' lane tapes."""
    from .fleet_islands import PlanFile, plan_candidates

    return PlanFile(plan_candidates(paths), getattr(store, "objects", None), tape_interval_s=interval_s)


def _attach_compare(reducer, args) -> None:
    """load the --compare run into its own reducer."""
    if not getattr(args, "compare", None):
        return
    from .sources import load_all

    other = _reducer(args)
    other.run = None
    load_all(other, list(args.compare))
    reducer.label = args.label or "A"
    other.label = args.compare_label or "B"
    reducer.compare = [other]


def main(args) -> int:
    cmd = args.dashboard_command
    if cmd == "mirror":
        from .mirror import SshTapeMirror

        SshTapeMirror(args.target, args.remote, args.local, island=args.island).run_forever()
        return 0
    try:
        reducer = _reducer(args)
        paths = _paths(args)
        _attach_compare(reducer, args)
    except (OSError, ValueError) as exc:
        print(f"[dashboard] {exc}", file=sys.stderr)
        return 2
    if cmd == "serve":
        from .serve import serve

        try:
            st = _store(args)
            return serve(paths, reducer=reducer, host=args.host, port=args.port,
                         interval=args.poll, head=args.head, store=st, resolved=_resolved(args, paths, st),
                         plan=_plan(paths, st, float(getattr(args, "store_poll", None) or 0.0)))
        except ValueError as exc:
            print(f"[dashboard] {exc}", file=sys.stderr)
            return 2
    if cmd == "export":
        from .export import export_html
        from .sources import load_all

        sources = load_all(reducer, paths)
        st = _store(args)
        _resolved(args, paths, st).pump(reducer)
        plan = _plan(paths, st)
        plan.pump(reducer)
        if st is not None:
            reducer.store_status = st.status
            st.pump(reducer)
            plan.pump_tapes(reducer)
            sources = list(sources) + sorted(st.sources) + sorted(x for t in plan.tapes() for x in t.sources)
        out = export_html(reducer, args.out)
        print(f"[dashboard] {len(sources)} tape(s), {reducer.event_seq} record(s) -> {out}")
        return 0
    return 2
