#!/usr/bin/env python3
"""One read-only training-health entry for a `yeto launch` run directory, whatever the
backend (S27 G-EU, VERL-PARITY #36).

Usage:
  python -I tools/eval/health.py RUN_DIR [--backend auto|miles|verl] [--json OUT.json]
                                 [--ref REF_RUN_DIR] [--no-table]          (Miles reader)
                                 [--learners N] [--total-steps S]          (verl reader)

The two readers stay where they are, each reading its backend's own logs; this
entry picks one and writes one summary:

  miles  tools/eval/health_gr.py: Miles' metric lines in RUN_DIR/launch.stream.log
         (rollout/raw_reward, train/*), the yeto events and the syncer log
  verl   tools/probes/s27_verl_full/health_launch.py: the island mirrors under
         RUN_DIR/tape-direct/<app>/l<k>/rank0 (verl metrics, criteria, publications,
         the latest container's files after a relaunch), the head syncer log

``--backend auto`` (default) reads the backend from the run: ``--rl-backend verl`` in
RUN_DIR/args.txt, or verl island files (``verl-plan-*.json`` / ``verl-runtime-*.json``)
in the island mirrors, else Miles. For the verl reader, ``--learners`` defaults to the
number of island mirrors and ``--total-steps`` to the syncer's fragment steps of
args.txt (``--total-steps`` x ``--fragments``).

Output: the reader's own report, then one line ``backend=<b> overall=<PASS|WARN|FAIL>``;
``--json`` writes ``{"backend", "overall", "checks": [{"check", "status", "detail"}],
"reader"}`` (refused inside the run dir). Exit 1 when a check FAILs. Standard library
only; never writes into the run dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
READERS = {
    "miles": HERE / "health_gr.py",
    "verl": HERE.parent / "probes" / "s27_verl_full" / "health_launch.py",
}
VERL_FILES = ("verl-plan-*.json", "verl-runtime-*.json")


def _load(backend: str):
    path = READERS[backend]
    if backend == "verl":  # health_launch imports its sibling health_p4
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"_yeto_health_{backend}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args_tokens(run: Path) -> list[str]:
    path = run / "args.txt"
    if not path.is_file():
        return []
    try:
        return shlex.split(path.read_text())
    except ValueError:
        return path.read_text().split()


def _flag(tokens: list[str], name: str) -> str | None:
    return tokens[tokens.index(name) + 1] if name in tokens and tokens.index(name) + 1 < len(tokens) else None


def island_mirrors(run: Path) -> list[Path]:
    return sorted(p for p in run.glob("tape-direct/*/l*/rank0") if p.is_dir())


def detect_backend(run: Path) -> tuple[str, str]:
    """``(backend, why)`` of a run directory (module doc)."""
    backend = _flag(_args_tokens(run), "--rl-backend")
    if backend:
        return backend, "--rl-backend in args.txt"
    for mirror in island_mirrors(run):
        for pattern in VERL_FILES:
            if any(mirror.glob(pattern)):
                return "verl", f"{pattern} in {mirror.relative_to(run)}"
    return "miles", "no verl island files (the Miles reader)"


def verl_defaults(run: Path) -> tuple[int, int | None]:
    """``(learners, syncer fragment steps)`` of a verl run directory (module doc)."""
    learners = len({p.parent.name for p in island_mirrors(run)}) or 2
    tokens = _args_tokens(run)
    steps, fragments = _flag(tokens, "--total-steps"), _flag(tokens, "--fragments")
    total = int(steps) * int(fragments or 1) if steps and steps.isdigit() else None
    return learners, total


def _verl(run: Path, args) -> dict:
    reader = _load("verl")
    learners, total = verl_defaults(run)
    learners = args.learners or learners
    total = args.total_steps or total or 32
    result = reader.check(run, learners, total)
    for r in result["rows"]:
        print(f"{r['status']:5} {r['check']:10} {r['detail']}")
    print(" ".join(f"{k}={v}" for k, v in result["counts"].items()))
    counts = result["counts"]
    overall = "FAIL" if counts.get("FAIL") else "WARN" if counts.get("WARN") else "PASS"
    return {"overall": overall, "checks": [dict(r) for r in result["rows"]],
            "reader": str(READERS["verl"].relative_to(HERE.parent.parent)),
            "options": {"learners": learners, "total_steps": total}}


def _miles(run: Path, args) -> dict:
    reader = _load("miles")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "health.json"
        argv = [str(run), "--json", str(out)] + (["--no-table"] if args.no_table else []) \
            + (["--ref", args.ref] if args.ref else [])
        reader.main(argv)
        data = json.loads(out.read_text())
    checks = [{"check": c.get("name"), "status": c.get("status"),
               "detail": f"value {c.get('value')}; threshold {c.get('threshold')}"} for c in data["checks"]]
    return {"overall": data["overall"], "checks": checks,
            "reader": str(READERS["miles"].relative_to(HERE.parent.parent))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--backend", choices=("auto", "miles", "verl"), default="auto")
    ap.add_argument("--json", help="write the summary JSON here (refused inside the run dir)")
    ap.add_argument("--ref", help="Miles reader: reference run dir with the same prompt order")
    ap.add_argument("--no-table", action="store_true", help="Miles reader: skip the per-step table")
    ap.add_argument("--learners", type=int, default=None, help="verl reader: islands (default: mirrors)")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="verl reader: syncer fragment steps (default: args.txt steps x fragments)")
    a = ap.parse_args(argv)
    run = Path(os.path.abspath(a.run_dir))
    if a.json:
        target = os.path.abspath(a.json)
        if target == str(run) or target.startswith(str(run) + os.sep):
            sys.exit("refusing to write inside the run dir")
    backend, why = (a.backend, "--backend") if a.backend != "auto" else detect_backend(run)
    if backend not in READERS:
        sys.exit(f"no health reader for backend {backend!r} ({why})")
    print(f"health: backend {backend} ({why}) -> {READERS[backend].name}")
    summary = _verl(run, a) if backend == "verl" else _miles(run, a)
    summary = {"backend": backend, "detected_from": why, **summary}
    print(f"backend={backend} overall={summary['overall']}")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1, default=str))
    return 1 if summary["overall"] == "FAIL" else 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
