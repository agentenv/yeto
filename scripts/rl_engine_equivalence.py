#!/usr/bin/env python3
"""Legacy-vs-ports RL engine equivalence harness (openspec rl-engine-ports 6.1).

Runs the same configuration under ``--rl-engine legacy`` and ``--rl-engine
ports``, pulls per-(island, round) metrics out of the JSONL RL event tapes and
writes a comparison report.

Metrics per (island, round), from the event names both paths share:

* ``rl_local_round``: ``reward_mean``, ``loss``, ``grad_norm``,
  ``completed_groups``, ``action_tokens`` (keyed by ``local_round_id``)
* ``rl_policy_apply``: ``sync/global_policy_hash`` for ``policy_version == round``
  (the policy applied after that round's outer sync)

Tolerance protocol: the legacy configuration is run ``--legacy-repeats`` times
(default 3) FIRST. Per-metric tolerance = ``--tolerance-factor`` x the largest
absolute spread across those repeats for that metric (floored at
``--tolerance-floor``). Count and hash metrics are exact when the repeats agree
and are marked ``nondeterministic`` (reported, not gated) when they do not.
The tolerance header is written to the report file BEFORE the ports run starts.

Modes:
  --dry-run   print the plan (commands, tapes, tolerance protocol) and exit
  --fake      CPU-only: ports side is the real IslandDriver on
              ``yeto.rl.engine.fake`` (two islands, strict-avg); legacy side is
              SYNTHETIC event tapes derived from it with seeded noise. The
              report is labeled FAKE throughout and proves only the harness.
  (default)   real mode: runs ``--launch-cmd`` for each engine. This may use
              paid resources; it is never invoked by tests.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

FLOAT_METRICS = ("reward_mean", "loss", "grad_norm")
COUNT_METRICS = ("completed_groups", "action_tokens")
HASH_METRIC = "post_sync_hash"
ALL_METRICS = FLOAT_METRICS + COUNT_METRICS + (HASH_METRIC,)

DEFAULT_LAUNCH_CMD = "yeto launch {launch_args} --rl-engine {engine}"

Key = tuple[int, int]  # (island_id, round)


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------
def read_events(paths: list[Path]) -> list[dict[str, Any]]:
    events = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def extract_rounds(events: list[dict[str, Any]]) -> dict[Key, dict[str, Any]]:
    """Per-(island, round) metrics from rl_local_round + rl_policy_apply."""

    rounds: dict[Key, dict[str, Any]] = {}
    hashes: dict[Key, str] = {}
    for e in events:
        island = int(e.get("island_id", 0))
        name = e.get("event")
        if name == "rl_local_round":
            r = int(e["local_round_id"])
            row = rounds.setdefault((island, r), {})
            for m in FLOAT_METRICS + COUNT_METRICS:
                row[m] = e.get(m, e.get(f"rl/{m}"))
        elif name == "rl_policy_apply" and "sync/global_policy_hash" in e:
            hashes[(island, int(e["policy_version"]))] = e["sync/global_policy_hash"]
    for key, row in rounds.items():
        row[HASH_METRIC] = hashes.get(key)
    return rounds


# ---------------------------------------------------------------------------
# Tolerance + comparison
# ---------------------------------------------------------------------------
def derive_tolerance(
    repeats: list[dict[Key, dict[str, Any]]], *, factor: float, floor: float
) -> dict[str, Any]:
    if len(repeats) < 2:
        raise ValueError("tolerance needs at least 2 legacy repeats")
    keys = sorted(set().union(*repeats))
    tol: dict[str, Any] = {}
    for m in FLOAT_METRICS:
        spread = 0.0
        for k in keys:
            vals = [r.get(k, {}).get(m) for r in repeats]
            vals = [float(v) for v in vals if v is not None]
            if len(vals) >= 2:
                spread = max(spread, max(vals) - min(vals))
        tol[m] = {"kind": "abs", "noise_spread": spread, "tolerance": max(floor, factor * spread)}
    for m in COUNT_METRICS + (HASH_METRIC,):
        agree = all(len({repr(r.get(k, {}).get(m)) for r in repeats}) == 1 for k in keys)
        tol[m] = {"kind": "exact" if agree else "nondeterministic"}
    return tol


def compare(
    legacy: dict[Key, dict[str, Any]],
    ports: dict[Key, dict[str, Any]],
    tolerance: dict[str, Any],
) -> dict[str, Any]:
    rows, failures = [], []
    missing = sorted(set(legacy) ^ set(ports))
    for key in sorted(set(legacy) & set(ports)):
        for m in ALL_METRICS:
            a, b = legacy[key].get(m), ports[key].get(m)
            spec = tolerance[m]
            if spec["kind"] == "abs":
                diff = None if a is None or b is None else abs(float(a) - float(b))
                ok = diff is not None and diff <= spec["tolerance"]
            elif spec["kind"] == "exact":
                diff = None
                ok = a == b
            else:
                diff, ok = None, None  # reported only
            row = {"island": key[0], "round": key[1], "metric": m,
                   "legacy": a, "ports": b, "abs_diff": diff, "pass": ok}
            rows.append(row)
            if ok is False:
                failures.append(row)
    passed = not failures and not missing
    return {"passed": passed, "rows": rows, "failures": failures,
            "missing_rounds": [list(k) for k in missing]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def header_text(meta: dict[str, Any], tolerance: dict[str, Any]) -> str:
    lines = []
    if meta.get("fake"):
        lines += ["# FAKE REPORT -- CPU fake engine + SYNTHETIC legacy tapes",
                  "# This validates the harness only; it is NOT equivalence evidence.", ""]
    lines += ["# RL engine equivalence report (legacy vs ports)", ""]
    for k in ("mode", "config", "created_unix", "legacy_repeats", "tolerance_factor",
              "tolerance_floor"):
        lines.append(f"- {k}: {meta.get(k)}")
    lines += ["", "## Tolerance (fixed from legacy repeats before the ports run)", "",
              "| metric | kind | legacy noise spread | tolerance |", "|---|---|---|---|"]
    for m in ALL_METRICS:
        s = tolerance[m]
        lines.append(f"| {m} | {s['kind']} | {s.get('noise_spread', '-')} | "
                     f"{s.get('tolerance', 'exact' if s['kind'] == 'exact' else '-')} |")
    return "\n".join(lines) + "\n"


def results_text(result: dict[str, Any], *, fake: bool) -> str:
    verdict = "PASS" if result["passed"] else "FAIL"
    if fake:
        verdict += " (FAKE)"
    lines = ["", f"## Result: {verdict}", ""]
    if result["missing_rounds"]:
        lines.append(f"Rounds present on one side only (island, round): {result['missing_rounds']}\n")
    lines += ["| island | round | metric | legacy | ports | abs diff | pass |",
              "|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        mark = {True: "ok", False: "FAIL", None: "n/a"}[r["pass"]]
        lines.append(f"| {r['island']} | {r['round']} | {r['metric']} | {r['legacy']} | "
                     f"{r['ports']} | {r['abs_diff']} | {mark} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Real mode
# ---------------------------------------------------------------------------
def launch_command(template: str, *, engine: str, launch_args: str, run_dir: Path) -> list[str]:
    return shlex.split(template.format(engine=engine, launch_args=launch_args,
                                       run_dir=str(run_dir)))


def plan(args) -> dict[str, Any]:
    out = Path(args.out_dir)
    runs = [(f"legacy-{i}", "legacy") for i in range(args.legacy_repeats)] + [("ports", "ports")]
    steps = []
    for name, engine in runs:
        run_dir = out / name
        steps.append({
            "name": name, "engine": engine,
            "command": ([] if args.fake else launch_command(
                args.launch_cmd, engine=engine, launch_args=args.launch_args, run_dir=run_dir)),
            "events_glob": str(run_dir / args.events_glob) if not args.fake else str(run_dir / "*.jsonl"),
        })
    return {
        "mode": "fake" if args.fake else "real",
        "config": args.launch_args if not args.fake else f"fake strict-avg islands=2 rounds={args.rounds}",
        "steps": steps,
        "tolerance": (f"after the {args.legacy_repeats} legacy runs: float metrics abs tol = "
                      f"{args.tolerance_factor} x max repeat spread (floor {args.tolerance_floor}); "
                      "counts/hash exact if repeats agree, else nondeterministic; header "
                      "written to report before the ports run"),
        "report": str(out / "report.md"),
    }


def run_real(step: dict[str, Any]) -> list[Path]:
    Path(step["events_glob"]).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(step["command"], check=True)
    paths = [Path(p) for p in sorted(glob.glob(step["events_glob"]))]
    if not paths:
        raise SystemExit(f"no event tapes matched {step['events_glob']}")
    return paths


# ---------------------------------------------------------------------------
# Fake mode
# ---------------------------------------------------------------------------
FAKE_TENSOR = "base_model.model.layer.lora_A.weight"


def run_fake_ports(run_dir: Path, *, rounds: int, islands: int = 2) -> list[Path]:
    """Real IslandDriver + StrictAvgSync on the CPU fake engine."""

    import torch

    from yeto.rl.bridge import BridgeConfig
    from yeto.rl.core import build_avg_layout
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import StrictAvgSync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import (
        LORA_CONFIG_HASH, MODEL_REVISION, FakeEngine, FakeStrictSyncer, fake_capabilities,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    engines = [FakeEngine(tensors={FAKE_TENSOR: torch.zeros(1, 2)},
                          step_delta=torch.tensor([1.0 + 2 * i, 3.0 + 2 * i]))
               for i in range(islands)]
    initial = engines[0].canonical(0)
    syncer = FakeStrictSyncer(build_avg_layout(initial.specs), learners=islands,
                              total_steps=rounds)
    drivers, paths = [], []
    for i, engine in enumerate(engines):
        tape = run_dir / f"island-{i}.jsonl"
        paths.append(tape)
        config = BridgeConfig(
            syncer_addr=("127.0.0.1", 1), learner_id=i, global_rounds=rounds,
            groups_per_round=engine.groups, samples_per_group=engine.samples_per_group,
            local_optimizer_steps=1, wan_streams=0, expected_specs=initial.specs,
            base_model_revision=MODEL_REVISION, lora_config_hash=LORA_CONFIG_HASH,
            layout_hash=initial.layout_hash, event_tape=str(tape),
        )
        sync = StrictAvgSync(config, client_factory=lambda _b, i=i: syncer.client(i))
        drivers.append(IslandDriver(
            learner_id=i, rollout=engine.rollout, trainer=engine.trainer,
            policy_state=engine.policy_state, publisher=engine.publisher,
            placement=engine.placement, algorithm=AlgorithmSpec(), sync=sync,
            events=EventTape(tape, i), capabilities=fake_capabilities(),
        ))
    errors: dict[int, BaseException] = {}

    def run(i, d):
        try:
            d.run()
        except BaseException as error:  # noqa: BLE001
            errors[i] = error

    threads = [threading.Thread(target=run, args=(i, d), daemon=True) for i, d in enumerate(drivers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        if t.is_alive():
            raise RuntimeError("fake ports island did not finish")
    if errors:
        raise RuntimeError(f"fake ports run failed: {errors}")
    return paths


def write_fake_legacy(run_dir: Path, template: dict[Key, dict[str, Any]], *,
                      seed: int, noise: float) -> list[Path]:
    """SYNTHETIC legacy tape: template metrics + seeded float noise. Labeled fake."""

    rng = random.Random(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    by_island: dict[int, list[str]] = {}
    for (island, r), row in sorted(template.items()):
        ev = {"event": "rl_local_round", "island_id": island, "local_round_id": r,
              "fake": True, "synthetic_legacy": True}
        for m in FLOAT_METRICS:
            ev[m] = None if row.get(m) is None else float(row[m]) + rng.uniform(-noise, noise)
        for m in COUNT_METRICS:
            ev[m] = row.get(m)
        apply = {"event": "rl_policy_apply", "island_id": island, "policy_version": r,
                 "sync/global_policy_hash": row.get(HASH_METRIC), "fake": True,
                 "synthetic_legacy": True}
        by_island.setdefault(island, []).extend(json.dumps(x, sort_keys=True) for x in (ev, apply))
    paths = []
    for island, lines in by_island.items():
        p = run_dir / f"island-{island}.jsonl"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--fake", action="store_true", help="CPU fake engine + synthetic legacy tapes")
    p.add_argument("--out-dir", default="rl-engine-equivalence")
    p.add_argument("--launch-cmd", default=DEFAULT_LAUNCH_CMD,
                   help="command template; placeholders {engine} {launch_args} {run_dir}")
    p.add_argument("--launch-args", default="", help="shared launch arguments (same config for both engines)")
    p.add_argument("--events-glob", default="**/rl-events*.jsonl",
                   help="event tape glob, relative to each run dir")
    p.add_argument("--legacy-repeats", type=int, default=3)
    p.add_argument("--tolerance-factor", type=float, default=2.0)
    p.add_argument("--tolerance-floor", type=float, default=1e-6)
    p.add_argument("--rounds", type=int, default=3, help="fake mode rounds")
    p.add_argument("--fake-noise", type=float, default=1e-3, help="fake legacy float noise")
    args = p.parse_args(argv)
    if args.legacy_repeats < 2:
        p.error("--legacy-repeats must be >= 2")
    if not args.fake and not args.dry_run and not args.launch_args:
        p.error("real mode requires --launch-args (or use --fake / --dry-run)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    the_plan = plan(args)
    if args.dry_run:
        print(json.dumps(the_plan, indent=2))
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = out / "report.md"
    steps = {s["name"]: s for s in the_plan["steps"]}

    if args.fake:
        # Ports reference run first so the synthetic legacy tapes have a shape to mimic.
        ports_paths = run_fake_ports(out / "ports", rounds=args.rounds)
        template = extract_rounds(read_events(ports_paths))
        legacy_runs = [
            extract_rounds(read_events(write_fake_legacy(
                out / f"legacy-{i}", template, seed=i, noise=args.fake_noise)))
            for i in range(args.legacy_repeats)
        ]
    else:
        legacy_runs = [extract_rounds(read_events(run_real(steps[f"legacy-{i}"])))
                       for i in range(args.legacy_repeats)]

    tolerance = derive_tolerance(legacy_runs, factor=args.tolerance_factor,
                                 floor=args.tolerance_floor)
    meta = {"mode": the_plan["mode"], "config": the_plan["config"], "fake": args.fake,
            "created_unix": time.time(), "legacy_repeats": args.legacy_repeats,
            "tolerance_factor": args.tolerance_factor, "tolerance_floor": args.tolerance_floor}
    # Header (tolerance) is committed to disk before the ports experiment.
    report.write_text(header_text(meta, tolerance), encoding="utf-8")

    if not args.fake:
        ports_paths = run_real(steps["ports"])
    ports = extract_rounds(read_events(ports_paths))
    result = compare(legacy_runs[0], ports, tolerance)
    with report.open("a", encoding="utf-8") as fh:
        fh.write(results_text(result, fake=args.fake))
    (out / "report.json").write_text(json.dumps(
        {"meta": meta, "tolerance": tolerance, **result}, indent=2, default=str), encoding="utf-8")
    print(f"{'PASS' if result['passed'] else 'FAIL'}{' (FAKE)' if args.fake else ''}: {report}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
