"""Run store-simulation seeds from the command line.

    python -m tests.store_sim.run --seeds 0:200 --samplers 16 --faults all

``--faults``: a preset of :data:`~.fault_s3.PRESETS` or ``name=value,...`` pairs of
:class:`~.fault_s3.FaultConfig` fields (both may be mixed: ``all,crash=0.05``).
Prints one line per violating seed and a summary (``--json`` for machines);
exits 1 when a seed violated an invariant.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import sys
import time
from collections.abc import Sequence

from .fault_s3 import PRESETS, FaultConfig
from .sim import Result, SimConfig, run_seed


def parse_faults(text: str) -> FaultConfig:
    fields = {f.name: f.type for f in dataclasses.fields(FaultConfig)}
    values: dict[str, object] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            if part not in PRESETS:
                raise SystemExit(f"unknown fault preset {part!r}; presets: {', '.join(sorted(PRESETS))}")
            values.update(PRESETS[part])
            continue
        name, _, raw = part.partition("=")
        if name not in fields:
            raise SystemExit(f"unknown fault {name!r}; faults: {', '.join(sorted(fields))}")
        values[name] = raw.lower() in ("1", "true", "yes", "on") if fields[name] == "bool" else float(raw)
    return FaultConfig(**values)  # type: ignore[arg-type]


def parse_seeds(text: str) -> range:
    first, sep, last = text.partition(":")
    return range(int(first), int(last)) if sep else range(int(first), int(first) + 1)


def summarize(results: Sequence[Result]) -> dict[str, object]:
    violations: collections.Counter[str] = collections.Counter()
    seeds: dict[str, list[int]] = collections.defaultdict(list)
    metrics: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    for result in results:
        for kind, items in result.violations.items():
            violations[kind] += len(items)
            seeds[kind].append(result.config.seed)
        metrics.update(result.metrics)
        errors.update(result.errors)
    return {"seeds": len(results), "violating_seeds": sum(1 for r in results if not r.ok),
            "violations": dict(sorted(violations.items())),
            "seeds_by_violation": {k: {"count": len(v), "first": v[:10]} for k, v in sorted(seeds.items())},
            "metrics": dict(sorted(metrics.items())), "process_errors": dict(sorted(errors.items()))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.store_sim.run", description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", default="0:20", help="A:B (B exclusive) or one seed")
    parser.add_argument("--samplers", type=int, default=SimConfig.samplers)
    parser.add_argument("--slots", type=int, default=SimConfig.slots)
    parser.add_argument("--steps", type=int, default=SimConfig.steps, help="scheduler steps with faults on")
    parser.add_argument("--faults", default="all")
    parser.add_argument("--lease-s", type=float, default=SimConfig.lease_s, help="the dispenser lease")
    parser.add_argument("--check", action="append", default=[], metavar="MODULE:NAME",
                        help="one more invariant (sim.load_check); may repeat")
    parser.add_argument("--backend", choices=("fake", "r2"), default="fake")
    parser.add_argument("--r2-prefix", help="with --backend r2: the temporary prefix of the bucket")
    parser.add_argument("--json", action="store_true", help="print the summary as one JSON object")
    parser.add_argument("--trace", type=int, default=40, help="trace lines per violating seed (0: none)")
    parser.add_argument("--show", type=int, default=5, help="violating seeds to print in full")
    args = parser.parse_args(argv)
    if args.backend != "fake":
        raise SystemExit("--backend r2 is not built yet: the simulation runs on the fake store only")
    faults = parse_faults(args.faults)
    logging.getLogger("yeto").setLevel(logging.CRITICAL)  # the dispenser warns on every expiry
    results, shown, started = [], 0, time.time()
    for seed in parse_seeds(args.seeds):
        result = run_seed(SimConfig(seed=seed, samplers=args.samplers, slots=args.slots, steps=args.steps,
                                    lease_s=args.lease_s, checks=tuple(args.check), faults=faults))
        results.append(result)
        if not result.ok and not args.json:
            shown += 1
            print(result.report(trace=args.trace if shown <= args.show else 0), flush=True)
    summary = {**summarize(results), "faults": args.faults, "samplers": args.samplers, "slots": args.slots,
               "steps": args.steps, "wall_s": round(time.time() - started, 1)}
    print(json.dumps(summary, sort_keys=True) if args.json else json.dumps(summary, indent=1, sort_keys=True))
    return 1 if summary["violating_seeds"] else 0


if __name__ == "__main__":
    sys.exit(main())
