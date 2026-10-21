#!/usr/bin/env python3
"""TB2 task lists on the Harbor path (S26 E9.5; module ``yeto.rl.harness.harbor.task_lists``).

    # hold-out + training split (data/eval/tb2-holdout.json, tb2-train.json shapes)
    python tools/reward_env/tb2_lists.py split --tasks ~/work/tb21-data --out-dir out/

    # curriculum stages of a training split
    python tools/reward_env/tb2_lists.py curriculum --tasks ~/work/tb21-data \
        --train out/tb2-train.json --out out/tb2-curriculum.json \
        [--reference pos-all.jsonl] [--pass-rates baseline.json --require-signal]

``--tasks`` is the Terminal-Bench checkout (TB2 flat, or TB2.1 with ``tasks/``).
The output is deterministic: same checkout and inputs, same bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yeto.rl.harness.harbor import task_lists as tl  # noqa: E402
from yeto.rl.harness.reward_env import benchmark as bm  # noqa: E402


def _split(a: argparse.Namespace) -> int:
    tasks = tl.HarborTaskSet(a.tasks, version=a.version)
    unusable = {} if a.no_unusable else tl.load_unusable(a.unusable)
    holdout, train = tl.build_tb2_lists(tasks, unusable=unusable, exclude={} if a.no_smoke6 else None)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in (("tb2-holdout.json", holdout), ("tb2-train.json", train)):
        (out / name).write_text(tl.dump_list(doc))
        print(name, len(doc["items"]), bm.holdout_sha256(doc))
    return 0


def _curriculum(a: argparse.Namespace) -> int:
    tasks = tl.HarborTaskSet(a.tasks, version=a.version)
    train = json.loads(Path(a.train).read_text())
    source = {"train": Path(a.train).name}
    reference = pass_rates = None
    if a.reference:
        reference = tl.load_reference_results(a.reference)
        source["reference"] = Path(a.reference).name
    if a.pass_rates:
        pass_rates = tl.load_pass_rates(a.pass_rates)
        source["pass_rates"] = Path(a.pass_rates).name
    doc = tl.build_curriculum(tasks, train, reference=reference, pass_rates=pass_rates,
                              require_signal=a.require_signal, source=source)
    Path(a.out).write_text(tl.dump_list(doc))
    for stage in doc["stages"]:
        print(stage["name"], len(stage["items"]))
    print("dropped", len(doc["dropped"]))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="hold-out and training lists")
    s.add_argument("--tasks", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--version", help="benchmark_version override (default: from the checkout)")
    s.add_argument("--unusable", help="unusable-task table (default data/eval/tb2-unusable.json)")
    s.add_argument("--no-unusable", action="store_true")
    s.add_argument("--no-smoke6", action="store_true", help="keep the S15 smoke-6 in the hold-out pool")
    s.set_defaults(run=_split)
    c = sub.add_parser("curriculum", help="ordered stages of a training list")
    c.add_argument("--tasks", required=True)
    c.add_argument("--train", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--version")
    c.add_argument("--reference", help="reference-solution runs (jsonl: task, passed, solve_s)")
    c.add_argument("--pass-rates", help="baseline pass counts (json: {task_id: {n, passed}})")
    c.add_argument("--require-signal", action="store_true", help="drop tasks with pass rate 0 or 1")
    c.set_defaults(run=_curriculum)
    a = p.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    raise SystemExit(main())
