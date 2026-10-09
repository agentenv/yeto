#!/usr/bin/env python3
"""TB2 eval / train jsonl from the hold-out list (rl-eval-difficulty-buckets 1.1 second half, 1.3).

    python tools/build_eval_data.py --tb2-tasks-dir ~/work/tb2-data \
        --holdout data/eval/tb2-holdout.json --out-dir data

Writes
* ``<out>/eval/tb2-holdout-eval.jsonl``  -- the 30 hold-out tasks (eval only);
* ``<out>/tb2/tb2-train.jsonl``           -- every other TB2 task except the unusable table
  ``data/eval/tb2-unusable.json`` (S17 N13; 46 at tb2@2fd12b8 = 89 - 30 hold-out - 13
  unusable; the S15 smoke-6 included: they were excluded from the eval pool, not from training);
* ``<out>/eval/tb2-data.sha256.json``     -- sha256 of the hold-out list and both files
  (pin these in the run config).

Row format = design D6.a: the smoke-6 system prompt (the task statement itself
comes from the task's ``instruction.md`` at rollout time, ``task_prompt``) and
``metadata`` with ``task_id``, ``benchmark``, ``benchmark_version``,
``difficulty``, ``difficulty_source`` and, on eval rows, ``eval_bucket``.
Deterministic: rows sorted by ``task_id``; the same inputs give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SYSTEM_PROMPT = (
    "You are an autonomous terminal agent solving a Terminal-Bench task. You will be given the task "
    "instruction, then interact with a real Linux shell. On each turn respond with EXACTLY ONE shell "
    "command inside a single ```bash code block and nothing else. Inspect the environment, make the "
    "required changes, and verify your work. When you are confident the task is fully complete, reply "
    "with TASK_COMPLETE (with no code block)."
)
EVAL_JSONL = "eval/tb2-holdout-eval.jsonl"
TRAIN_JSONL = "tb2/tb2-train.jsonl"
SHA_JSON = "eval/tb2-data.sha256.json"


def _row(spec: Any, *, with_bucket: bool) -> dict[str, Any]:
    meta = {"task_id": spec.task_id, "benchmark": spec.benchmark, "benchmark_version": spec.benchmark_version,
            "difficulty": spec.difficulty, "difficulty_source": spec.difficulty_source}
    if with_bucket:
        if not spec.eval_bucket:
            raise ValueError(f"{spec.task_id}: hold-out task without an eval bucket")
        meta["eval_bucket"] = spec.eval_bucket
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}], "metadata": meta}


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()


def build(tasks_dir: str | Path, holdout: dict[str, Any], *,
          unusable: dict[str, str] | None = None) -> dict[str, bytes]:
    """{relative path: bytes} for the eval jsonl, the train jsonl and the sha256 record."""
    from yeto.rl.harness.reward_env import benchmark as bm
    from yeto.rl.harness.reward_env import tb2

    adapter = tb2.Tb2Benchmark(tasks_dir)
    if holdout.get("benchmark_version") != adapter.version:
        raise ValueError(f"hold-out list is for {holdout.get('benchmark_version')}, checkout is {adapter.version}")
    held = bm.holdout_ids(holdout)
    every = adapter.task_ids()
    missing = sorted(set(held) - set(every))
    if missing:
        raise ValueError(f"hold-out tasks not in the checkout: {missing}")
    by_item = {i["task_id"]: i for i in holdout["items"]}
    eval_rows = []
    for tid in sorted(held):
        spec = adapter.task_spec(tid)
        if spec.eval_bucket != by_item[tid]["eval_bucket"]:
            raise ValueError(f"{tid}: bucket {spec.eval_bucket} != hold-out list {by_item[tid]['eval_bucket']}")
        eval_rows.append(_row(spec, with_bucket=True))
    unusable = {} if unusable is None else unusable
    unknown = sorted(set(unusable) - set(every))
    if unknown:
        raise ValueError(f"unusable tasks not in the checkout: {unknown}")
    clash = sorted(set(unusable) & set(held))
    if clash:
        raise ValueError(f"unusable tasks in the hold-out list (rebuild it with tools/reward_env/holdout.py): {clash}")
    train_rows = [_row(adapter.task_spec(t), with_bucket=False) for t in every
                  if t not in set(held) and t not in unusable]
    bm.assert_disjoint([r["metadata"]["task_id"] for r in train_rows], held)
    eval_bytes, train_bytes = _jsonl(eval_rows), _jsonl(train_rows)
    record = {
        "benchmark_version": adapter.version,
        "holdout_sha256": bm.holdout_sha256(holdout),
        "unusable_excluded": sorted(unusable),
        EVAL_JSONL: {"rows": len(eval_rows), "sha256": hashlib.sha256(eval_bytes).hexdigest()},
        TRAIN_JSONL: {"rows": len(train_rows), "sha256": hashlib.sha256(train_bytes).hexdigest()},
    }
    return {EVAL_JSONL: eval_bytes, TRAIN_JSONL: train_bytes,
            SHA_JSON: (json.dumps(record, indent=1, sort_keys=True) + "\n").encode()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tb2-tasks-dir", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tb2-unusable", default=None,
                   help="unusable-task table (default: data/eval/tb2-unusable.json); its tasks leave training")
    p.add_argument("--tb2-no-unusable", action="store_true", help="do not apply the unusable-task table")
    a = p.parse_args(argv)
    from yeto.rl.harness.reward_env import tb2

    unusable = {} if a.tb2_no_unusable else tb2.load_unusable(a.tb2_unusable)
    out = Path(a.out_dir)
    for rel, data in build(a.tb2_tasks_dir, json.loads(Path(a.holdout).read_text()),
                           unusable=unusable).items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
        print(rel, len(data), hashlib.sha256(data).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
