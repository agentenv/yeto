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
        p.add_argument("--prices", default=None, help="price table JSON (default: built-in example, not a bill)")
        p.add_argument("--budget", type=float, default=None, help="budget cap in $ for the cost bar/alerts")
        p.add_argument("--thresholds", default=None, help="alert threshold overrides JSON")

    s = dsub.add_parser("serve", help="loopback-only GET-only HTTP server")
    common(s)
    s.add_argument("--host", default="127.0.0.1", help="loopback address only (0.0.0.0 is refused)")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--poll", type=float, default=2.0, help="tape follow interval seconds")
    s.add_argument("--head", default=None, help="head host name, only used in the printed ssh -L hint")

    e = dsub.add_parser("export", help="single self-contained offline HTML")
    common(e)
    e.add_argument("-o", "--out", required=True, help="output .html")

    m = dsub.add_parser("mirror", help="mirror a remote learner tape over ssh (tail -F) into a local file")
    m.add_argument("--target", required=True, help="ssh target, e.g. user@host")
    m.add_argument("--remote", required=True, help="remote tape path")
    m.add_argument("--local", required=True, help="local mirror path (point serve --tapes at it)")
    m.add_argument("--island", default=None, help="island id recorded on dashboard_source_lost events")


def _reducer(args):
    from .cost import load_prices
    from .reducer import Reducer

    thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    return Reducer(run=args.run, prices=load_prices(args.prices), budget_usd=args.budget,
                   thresholds=thresholds)


def _paths(args) -> list[str]:
    from .sources import run_sources

    paths = list(args.tapes)
    if args.run:
        paths = run_sources(args.run) + paths
    if not paths:
        raise SystemExit("[dashboard] no tapes: pass --run <run> and/or --tapes PATH...")
    return paths


def main(args) -> int:
    cmd = args.dashboard_command
    if cmd == "mirror":
        from .mirror import SshTapeMirror

        SshTapeMirror(args.target, args.remote, args.local, island=args.island).run_forever()
        return 0
    try:
        reducer = _reducer(args)
        paths = _paths(args)
    except (OSError, ValueError) as exc:
        print(f"[dashboard] {exc}", file=sys.stderr)
        return 2
    if cmd == "serve":
        from .serve import serve

        try:
            return serve(paths, reducer=reducer, host=args.host, port=args.port,
                         interval=args.poll, head=args.head)
        except ValueError as exc:
            print(f"[dashboard] {exc}", file=sys.stderr)
            return 2
    if cmd == "export":
        from .export import export_html
        from .sources import load_all

        sources = load_all(reducer, paths)
        out = export_html(reducer, args.out)
        print(f"[dashboard] {len(sources)} tape(s), {reducer.event_seq} record(s) -> {out}")
        return 0
    return 2
