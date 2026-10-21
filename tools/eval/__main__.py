"""CLI of the evaluation suite (see the package doc).

    python -m tools.eval list
    python -m tools.eval run E3 --backend verl --run-id eval-e3-verl-a [--out DIR] [--dry-run]
    python -m tools.eval judge DIR/eval-e3-verl-a
    python -m tools.eval report DIR/* --out report.md [--html report.html]

Efficiency evaluation (S26 E0.1, package :mod:`.efficiency`):

    python -m tools.eval eff-plan qwen3-0.6b [--arm ARM]
    python -m tools.eval eff-run qwen3-0.6b --arm bare-verl-sync --run-id eff-... [--out DIR] [--dry-run]
    python -m tools.eval eff-parse DIR/eff-...
    python -m tools.eval eff-report DIR/eff-* --out report.md

``run`` refuses a run id that is not pre-registered in the GPU ledger (the id
must appear in ``gpu-spend.md``), checks the local thread count and Ray, writes
``cell.json``, runs the cell's harness (raw data into ``raw/``), then judges.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import cells, judges, report
from .verdict import Verdict

DEFAULT_OUT = Path("/home/michael/work/infra-drafts/SESSION25/logs/T-EVAL")
LEDGER = Path("/home/michael/work/infra-drafts/gpu-spend.md")
THREAD_LIMIT = 3000


def _preflight(run_id: str) -> None:
    if not LEDGER.is_file() or run_id not in LEDGER.read_text():
        raise SystemExit(f"run id {run_id} is not pre-registered in {LEDGER}")
    threads = subprocess.run("ps -u michael -L --no-headers | wc -l", shell=True, capture_output=True,
                             text=True).stdout.strip()
    if int(threads or 0) >= THREAD_LIMIT:
        raise SystemExit(f"{threads} local threads (>= {THREAD_LIMIT}): do not launch")
    if subprocess.run(["pgrep", "-x", "raylet"], capture_output=True).returncode == 0:
        raise SystemExit("a local Ray is running: do not launch")


def cmd_list(_args) -> int:
    for key, cell in sorted(cells.CELLS.items()):
        flag = "" if cell.supported else "  [unsupported: expected to fail]"
        print(f"{key:10s} {cell.harness:12s} {cell.cards} card(s) {cell.config.get('gpu')}  "
              f"{cells.CAPABILITIES[cell.capability]}{flag}")
    return 0


def cmd_run(args) -> int:
    cell = cells.get(args.capability, args.backend)
    run_dir = Path(args.out) / args.run_id
    if not args.dry_run:
        if (run_dir / "cell.json").exists():
            raise SystemExit(f"{run_dir} already used")
        _preflight(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (run_dir / "cell.json").write_text(json.dumps({
        "capability": cell.capability, "backend": cell.backend, "run_id": args.run_id, "harness": cell.harness,
        "cards": cell.cards, "supported": cell.supported, "config": cell.config, "code_sha": sha}, indent=1))
    began = time.time()
    if cell.harness == "modal_verl":
        if args.dry_run:
            print(json.dumps(cell.config, indent=1))
            return 0
        from . import modal_verl

        summary = modal_verl.run(cell, args.run_id, run_dir)
    else:
        from . import miles

        summary = miles.run(cell, args.run_id, run_dir, dry_run=args.dry_run)
        if args.dry_run:
            print((run_dir / "dry-run.log").read_text())
            return summary["rc"]
    summary["wall_s"] = round(time.time() - began, 1)
    (run_dir / "run.json").write_text(json.dumps(summary, indent=1, default=str))
    return cmd_judge(argparse.Namespace(run_dir=str(run_dir)))


def judge(run_dir: Path) -> Verdict:
    from . import evidence, miles, modal_verl

    cell_doc = json.loads((run_dir / "cell.json").read_text())
    cell = cells.get(cell_doc["capability"], cell_doc["backend"])
    cfg = cell_doc["config"]
    verdict = Verdict(capability=cell.capability, backend=cell.backend, run_id=cell_doc["run_id"],
                      supported=cell.supported)
    run_doc = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    raw = run_dir / "raw"
    seconds = (modal_verl.card_seconds(raw) if cell.backend == "verl" else miles.card_seconds(raw))
    rate = cells.A10G_USD_PER_HOUR if cfg.get("gpu") == "A10G" else 5.3
    verdict.run.update(code_sha=cell_doc.get("code_sha"), wall_s=run_doc.get("wall_s"),
                       card_seconds=round(seconds, 1), usd_estimate=round(seconds / 3600 * rate, 3),
                       usd_rate=f"{rate}/container-hour incl. CPU+memory (list-price estimate, not a bill)")
    try:
        ev = evidence.load(run_dir)
        judges.JUDGES[cell.capability](ev, verdict, cfg)
    except Exception as exc:  # noqa: BLE001 - a judge crash is an error verdict, with the reason
        verdict.error = f"{type(exc).__name__}: {exc}"
    return verdict


def cmd_judge(args) -> int:
    run_dir = Path(args.run_dir)
    verdict = judge(run_dir)
    verdict.write(run_dir)
    doc = verdict.to_dict()
    print(json.dumps({"run": doc["run_id"], "cell": f"{doc['capability']}-{doc['backend']}",
                      "status": doc["status"], "failed": [c["id"] for c in doc["criteria"] if not c["passed"]],
                      "error": doc["error"]}))
    return 0 if doc["status"] in ("pass", "expected_fail") else 1


def cmd_report(args) -> int:
    dirs = [Path(d) for d in args.run_dirs if (Path(d) / "verdict.json").is_file()]
    md = report.markdown(dirs, title=args.title)
    Path(args.out).write_text(md)
    if args.html:
        Path(args.html).write_text(report.html(dirs, title=args.title))
    print(f"{len(dirs)} verdict(s) -> {args.out}")
    return 0


EFF_OUT = Path("/home/michael/work/infra-drafts/SESSION26/logs/E0")


def cmd_eff_plan(args) -> int:
    from .efficiency import spec

    sp = spec.get_spec(args.preset)
    arms = [args.arm] if args.arm else list(spec.ARMS)
    doc = {"spec": sp.to_dict(), "fingerprint": spec.control_fingerprint(sp),
           "arms": {a: spec.plan(sp, a, "<run id>") for a in arms},
           "verl_knob_diff": {m: spec.verl_knob_diff(sp, m) for m in spec.MODES}}
    print(json.dumps(doc, indent=1, default=str))
    return 0


def cmd_eff_run(args) -> int:
    from .efficiency import modal_eff, spec

    sp = spec.get_spec(args.preset, checkpoint=args.checkpoint)
    plan = spec.plan(sp, args.arm, args.run_id)
    run_dir = Path(args.out) / args.run_id
    if not args.dry_run:
        if (run_dir / "arm.json").exists():
            raise SystemExit(f"{run_dir} already used")
        _preflight(args.run_id)
    plan["code_sha"] = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    plan["images"] = spec.images()
    spec.write_plan(run_dir / "arm.json", plan)
    if plan["harness"] == "miles_local":
        summary = modal_eff.run_miles_launcher(plan, run_dir, dry_run=args.dry_run)
        if args.dry_run:
            print((run_dir / "dry-run.log").read_text())
            return summary["rc"]
    elif args.dry_run:
        print(plan.get("script") or json.dumps(plan.get("argv")))
        return 0
    else:
        summary = modal_eff.run(plan, run_dir)
    (run_dir / "run.json").write_text(json.dumps(summary, indent=1, default=str))
    return cmd_eff_parse(argparse.Namespace(run_dir=str(run_dir)))


def cmd_eff_parse(args) -> int:
    from .efficiency import metrics, parse

    run_dir = Path(args.run_dir)
    parsed = parse.parse_run(run_dir)
    (run_dir / "parsed.json").write_text(json.dumps(parsed, indent=1, default=str))
    summary = metrics.summarize(parsed)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"run": parsed["run_id"], "arm": parsed["arm"], "rounds": summary["rounds"],
                      "missing": summary["missing"]}))
    return 0 if summary["rounds"] == summary["rounds_expected"] else 1


def cmd_eff_report(args) -> int:
    from .efficiency import report as eff_report

    runs = eff_report.load([Path(d) for d in args.run_dirs])
    Path(args.out).write_text(eff_report.markdown(runs, title=args.title))
    print(f"{len(runs)} parsed run(s) -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    r = sub.add_parser("run")
    r.add_argument("capability", choices=sorted(cells.CAPABILITIES))
    r.add_argument("--backend", choices=cells.BACKENDS, required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)
    j = sub.add_parser("judge")
    j.add_argument("run_dir")
    j.set_defaults(fn=cmd_judge)
    rep = sub.add_parser("report")
    rep.add_argument("run_dirs", nargs="+")
    rep.add_argument("--out", required=True)
    rep.add_argument("--html", default=None)
    rep.add_argument("--title", default="yeto RL capability evaluation")
    rep.set_defaults(fn=cmd_report)
    from .efficiency.spec import ARMS, PRESETS

    ep = sub.add_parser("eff-plan")
    ep.add_argument("preset", choices=sorted(PRESETS))
    ep.add_argument("--arm", choices=ARMS, default=None)
    ep.set_defaults(fn=cmd_eff_plan)
    er = sub.add_parser("eff-run")
    er.add_argument("preset", choices=sorted(PRESETS))
    er.add_argument("--arm", choices=ARMS, required=True)
    er.add_argument("--run-id", required=True)
    er.add_argument("--out", default=str(EFF_OUT))
    er.add_argument("--dry-run", action="store_true")
    er.add_argument("--checkpoint", choices=("final", "every-round"), default="final",
                    help="yeto+Miles round cuts: only at the end (fair control) or every round "
                         "(product default)")
    er.set_defaults(fn=cmd_eff_run)
    eparse = sub.add_parser("eff-parse")
    eparse.add_argument("run_dir")
    eparse.set_defaults(fn=cmd_eff_parse)
    erep = sub.add_parser("eff-report")
    erep.add_argument("run_dirs", nargs="+")
    erep.add_argument("--out", required=True)
    erep.add_argument("--title", default="yeto RL efficiency evaluation")
    erep.set_defaults(fn=cmd_eff_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
