#!/usr/bin/env python
"""Run Harbor trials of TB2 tasks on boat sandboxes through the yeto Harbor path (S25 N10.8).

No model endpoint: use ``--agent oracle`` (Harbor runs the task's reference
solution) or ``--agent nop`` (does nothing; the verifier must fail).  Each
trial goes AgentTask -> build_trial_config (provider "boat" -> import_path)
-> Harbor Trial.run() -> TrialVerdict, and appends one JSON line to --out.

    python scripts/harbor_boat_trial.py --tasks-dir ~/work/tb2-data \\
        --agent oracle --out results.jsonl fix-git regex-log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from yeto.rl.agentic import AgentTask
from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config


async def run_one(task_id: str, agent: str, settings: HarborSettings) -> dict:
    task = AgentTask.from_mapping({"task_id": task_id, "agent": agent, "provider": "boat",
                                   "model_name": "none", "wall_clock_s": 3600.0})
    harness = HarborHarness(lambda t, endpoint: build_trial_config(t, endpoint, settings))
    started = time.time()
    verdict = await harness.run(task, endpoint="http://127.0.0.1:9/v1")  # oracle/nop never call it
    return {"task_id": task_id, "agent": agent, "rewards": verdict.rewards,
            "exception_type": verdict.exception_type,
            "exception_message": verdict.exception_message[-2000:],
            "timings_s": verdict.timings_s, "wall_s": round(time.time() - started, 1),
            "trial_dir": verdict.trial_dir}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, default=Path("/tmp/harbor_boat_trials"))
    parser.add_argument("--agent", choices=("oracle", "nop"), default="oracle")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = HarborSettings(tasks_dir=args.tasks_dir, trials_dir=args.trials_dir)
    slots = asyncio.Semaphore(args.concurrency)

    async def guarded(task_id: str) -> None:
        async with slots:
            row = await run_one(task_id, args.agent, settings)
        logging.info("RESULT %s", json.dumps(row))
        with args.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    await asyncio.gather(*(guarded(t) for t in args.tasks))


if __name__ == "__main__":
    asyncio.run(main())
