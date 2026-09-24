"""``benchmark_rl_elastic`` command line: example, validate, plan, report.

None of these subcommands import torch, ray, miles or a cloud SDK, load a
model, or create resources. Execution of matrix items is delegated to runners
registered against a capability attestation and lives outside this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yeto.rl.elastic_benchmark import capabilities as caps
from yeto.rl.elastic_benchmark import results as results_mod
from yeto.rl.elastic_benchmark.manifest import (
    MODES,
    ManifestError,
    dump_manifest,
    example_manifest,
    load_manifest,
    manifest_hash,
)
from yeto.rl.elastic_benchmark.plan import build_plan, plan_as_dict, render_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_rl_elastic",
        description="Plan and report RL elastic resource benchmark studies (no GPU work here).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    example = commands.add_parser("example", help="write the documented starting manifest")
    example.add_argument("--mode", choices=MODES, default="calibration")
    example.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate a study manifest and print its hash")
    validate.add_argument("--study", type=Path, required=True)

    plan = commands.add_parser("plan", help="dry-run: expand the matrix and budget without running")
    plan.add_argument("--study", type=Path, required=True)
    plan.add_argument("--capabilities", type=Path, help="runtime capability attestation JSON")
    plan.add_argument("--json", action="store_true", help="print the plan as JSON")

    report = commands.add_parser("report", help="summarize verified evidence under a study directory")
    report.add_argument("--study", type=Path, required=True)
    report.add_argument("--capabilities", type=Path)
    report.add_argument("--study-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (ManifestError, caps.ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "example":
        digest = dump_manifest(example_manifest(args.mode), args.output)
        print(f"wrote {args.output} (study_hash {digest[:12]})")
        return 0
    manifest = load_manifest(args.study)
    study_hash = manifest_hash(manifest)
    if args.command == "validate":
        print(f"valid {manifest['mode']} study {manifest['identity']['study_id']} hash={study_hash}")
        return 0
    attestation = caps.load_attestation(args.capabilities)
    plan = build_plan(manifest, attestation, study_hash=study_hash)
    if args.command == "plan":
        print(json.dumps(plan_as_dict(plan), indent=2, sort_keys=True) if args.json else render_plan(plan))
        return 0
    summary = results_mod.summarize(plan, args.study_dir)
    results_mod.write_summary(summary, args.study_dir)
    path = results_mod.write_report(summary, args.study_dir)
    print(f"{summary['status']} benefit={summary['benefit']} report={path}")
    return 0 if summary["status"] == "complete" else 1
