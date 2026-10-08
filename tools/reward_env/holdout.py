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
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    if a.tb2_tasks_dir:
        from yeto.rl.harness.reward_env import tb2

        written["tb2-holdout.json"] = tb2.build_holdout(tb2.Tb2Benchmark(a.tb2_tasks_dir))
    if a.swev_data:
        from yeto.rl.harness.reward_env import swebench_verified as sv

        written["swebench-verified-eval.json"] = sv.build_holdout(sv.SwebenchVerified(a.swev_data))
    for name, holdout in written.items():
        (out / name).write_text(json.dumps(holdout, indent=1, ensure_ascii=False) + "\n")
        print(name, len(holdout["items"]), bm.holdout_sha256(holdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
