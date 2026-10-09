#!/usr/bin/env python3
"""Write WP3 (rl-eval-difficulty-buckets D6b) hold-out lists for the built-in benchmarks.

    python tools/reward_env/holdout.py --tb2-tasks-dir ~/work/tb2-data \
        --swev-data swev-78f471bf.parquet --out-dir data/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yeto.rl.harness.reward_env import benchmark as bm  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tb2-tasks-dir")
    p.add_argument("--swev-data")
    p.add_argument("--tb2-exclude-jsonl", action="append", default=[],
                   help="dataset jsonl whose metadata.task_id rows leave the TB2 eval pool "
                        "(default: the built-in S15 smoke-6 list)")
    p.add_argument("--tb2-exclude-reason", default=None,
                   help="reason recorded for --tb2-exclude-jsonl tasks (default: S15 smoke training)")
    p.add_argument("--tb2-no-exclude", action="store_true", help="do not exclude any TB2 task")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    if a.tb2_tasks_dir:
        from yeto.rl.harness.reward_env import tb2

        if a.tb2_no_exclude:
            exclude = {}
        elif a.tb2_exclude_jsonl:
            reason = a.tb2_exclude_reason or tb2.SMOKE6_REASON
            exclude = {}
            for path in a.tb2_exclude_jsonl:
                for line in Path(path).read_text().splitlines():
                    if line.strip():
                        tid = bm.task_id_of(json.loads(line))
                        if tid is None:
                            raise SystemExit(f"{path}: row without metadata.task_id")
                        exclude[tid] = reason
        else:
            exclude = None  # built-in smoke-6
        adapter = tb2.Tb2Benchmark(a.tb2_tasks_dir)
        holdout = tb2.build_holdout(adapter, exclude=exclude)
        unknown = sorted({e["task_id"] for e in holdout.get("excluded", [])} - set(adapter.task_ids()))
        if unknown:
            raise SystemExit(f"excluded tasks not in the TB2 checkout: {unknown}")
        written["tb2-holdout.json"] = holdout
    if a.swev_data:
        from yeto.rl.harness.reward_env import swebench_verified as sv

        written["swebench-verified-eval.json"] = sv.build_holdout(sv.SwebenchVerified(a.swev_data))
    for name, holdout in written.items():
        (out / name).write_text(json.dumps(holdout, indent=1, ensure_ascii=False) + "\n")
        print(name, len(holdout["items"]), bm.holdout_sha256(holdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
