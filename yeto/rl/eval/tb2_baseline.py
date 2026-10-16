"""TB2 base-model baseline: rollout only, no training, no stored policy (S24 T-TB2).

Measures how often a base model passes the TB2 training tasks, to decide
whether TB2 gives a usable RL signal (pass rate neither ~0 nor ~1).  One run:

1. serve the base model on the local GPU without LoRA
   (:meth:`codex_infer.SglangSessionLoader.start_base`: SGLang + Miles'
   standalone session server with the training TITO settings);
2. run every ``(task, trial)`` pair through exactly the eval-island attempt
   (``tb2_attempt`` over the Modal reward-env provider and
   :class:`codex_infer.CodexEvalAgent`), ``concurrency`` attempts at a time in
   one event loop;
3. append one trace per attempt to ``traces.jsonl`` as it finishes and write
   ``summary.json`` at the end.

Pass-rate rule = the eval island's (D6.d): ``infra_error`` attempts are
excluded from the denominator; an agent that ran out of turns / context /
time is still judged.  Entry for Modal: ``yeto.cloud.modal_tb2_baseline``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .codex_infer import CodexEvalAgent, InferConfig, SglangSessionLoader
from .island import EvalTask
from .tb2_attempt import tb2_attempt

SUMMARY_SCHEMA = "yeto.tb2.baseline.summary/v1"
BASE_POLICY_VERSION = 0
TRACES = "traces.jsonl"
SUMMARY = "summary.json"


def load_tasks(split: Mapping[str, Any], rows: Iterable[Mapping[str, Any]],
               task_ids: Iterable[str] | None = None) -> list[EvalTask]:
    """Tasks of a TB2 split file (``tb2-train.json`` items) with their jsonl rows (prompt metadata)."""
    by_id = {str(r["metadata"]["task_id"]): r for r in rows}
    keep = None if task_ids is None else set(task_ids)
    tasks = [EvalTask(task_id=i["task_id"], eval_bucket=i.get("eval_bucket", "unknown"),
                      difficulty=i.get("difficulty", "unknown"), benchmark=str(split.get("benchmark", "tb2")),
                      row=by_id.get(i["task_id"], {}))
             for i in split["items"] if keep is None or i["task_id"] in keep]
    missing = [t.task_id for t in tasks if not t.row]
    if missing:
        raise ValueError(f"tasks without a jsonl row: {missing}")
    if keep is not None and len(tasks) != len(keep):
        raise ValueError(f"tasks not in the split: {sorted(keep - {t.task_id for t in tasks})}")
    return tasks


async def run_attempts(tasks: list[EvalTask], trials: int, attempt: Callable[..., Any], *,
                       concurrency: int, on_trace: Callable[[dict[str, Any]], None],
                       clock: Callable[[], float] = time.time) -> list[dict[str, Any]]:
    """``attempt(task, trial, ...)`` must return an awaitable (``tb2_attempt(..., run=lambda c: c)``).

    Trials are the outer loop so a run cut short still covers every task."""
    gate = asyncio.Semaphore(max(1, int(concurrency)))
    traces: list[dict[str, Any]] = []

    async def one(task: EvalTask, trial: int) -> None:
        async with gate:
            base = {"task_id": task.task_id, "difficulty": task.difficulty, "trial": trial, "start_ts": clock()}
            try:
                out = dict(await attempt(task, trial, policy_version=BASE_POLICY_VERSION, policy_token="base"))
            except Exception as exc:  # noqa: BLE001 - sandbox/verifier failure = infrastructure
                out = {"end_reason": "infra_error", "reward": 0.0, "success": False,
                       "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            trace = {**base, **out, "end_ts": clock()}
            traces.append(trace)
            on_trace(trace)

    await asyncio.gather(*(one(task, trial) for trial in range(trials) for task in tasks))
    return traces


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "mean": round(sum(ordered) / len(ordered), 3), "min": ordered[0],
            "p50": ordered[len(ordered) // 2], "max": ordered[-1]}


def summarize(traces: list[Mapping[str, Any]], *, trials: int) -> dict[str, Any]:
    judged = [t for t in traces if t.get("end_reason") != "infra_error"]
    passed = sum(1 for t in judged if t.get("success"))
    per_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"passed": 0, "judged": 0, "infra": 0})
    by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t in traces:
        row = per_task[str(t["task_id"])]
        row["difficulty"] = t.get("difficulty")
        if t.get("end_reason") == "infra_error":
            row["infra"] += 1
            continue
        row["judged"] += 1
        row["passed"] += int(bool(t.get("success")))
        by_diff[str(t.get("difficulty"))][0] += int(bool(t.get("success")))
        by_diff[str(t.get("difficulty"))][1] += 1
    # GRPO-style signal: a task gives a non-zero advantage only if its group mixes pass and fail
    mixed = sorted(k for k, v in per_task.items() if 0 < v["passed"] < v["judged"])
    timing = [t.get("timing") or {} for t in judged]
    return {
        "schema": SUMMARY_SCHEMA, "trials": trials, "attempts": len(traces), "judged": len(judged),
        "infra_errors": len(traces) - len(judged), "passed": passed,
        "pass_rate": round(passed / len(judged), 4) if judged else None,
        "by_difficulty": {k: {"passed": p, "judged": n, "pass_rate": round(p / n, 4) if n else None}
                          for k, (p, n) in sorted(by_diff.items())},
        "tasks": len(per_task),
        "tasks_any_pass": sum(1 for v in per_task.values() if v["passed"] > 0),
        "tasks_all_pass": sum(1 for v in per_task.values() if v["judged"] and v["passed"] == v["judged"]),
        "tasks_mixed": len(mixed), "mixed_task_ids": mixed,
        "per_task": dict(sorted(per_task.items())),
        "end_reasons": dict(Counter(str(t.get("end_reason")) for t in traces)),
        # the driver's own stop reason (e.g. a CodexModelFailure message), finer than end_reason
        "driver_end_reasons": dict(Counter(str((t.get("agent_detail") or {}).get("end_reason")) for t in traces)),
        "turns": _stats([float(t["turns"]) for t in judged if t.get("turns") is not None]),
        "tokens": _stats([float(t["tokens"]) for t in judged if t.get("tokens") is not None]),
        "agent_s": _stats([float(x["agent_s"]) for x in timing if "agent_s" in x]),
        "sandbox_s": _stats([float(x["sandbox_s"]) for x in timing if "sandbox_s" in x]),
        "judge_s": _stats([float(x["judge_s"]) for x in timing if "judge_s" in x]),
    }


def _provider(spec: str) -> Any:
    import importlib

    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(None)


def run_from_config(config: Mapping[str, Any], *, provider: Any = None, loader: Any = None,
                    agent: Any = None) -> dict[str, Any]:
    """``config``: ``split`` / ``rows`` (paths), ``out_dir``, ``trials``, ``concurrency``,
    optional ``task_ids``, ``provider`` (``module:factory``) and ``infer`` (:class:`InferConfig`)."""
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(config["split"]).read_text())
    rows = [json.loads(line) for line in Path(config["rows"]).read_text().splitlines() if line.strip()]
    tasks = load_tasks(split, rows, config.get("task_ids"))
    trials = int(config.get("trials", 4))
    infer = InferConfig.from_mapping(config.get("infer") or {})
    loader = loader or SglangSessionLoader(infer)
    t0 = time.time()
    try:
        serving = loader.start_base()
        (out / "serving.json").write_text(json.dumps(serving, indent=1, sort_keys=True, default=str))
        provider = provider or _provider(str(config.get("provider") or "yeto.cloud.modal_reward_env:modal_provider"))
        attempt = tb2_attempt(provider, agent or CodexEvalAgent(infer), run=lambda coro: coro)

        def on_trace(trace: dict[str, Any]) -> None:
            with open(out / TRACES, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace, sort_keys=True, default=str) + "\n")

        t1 = time.time()
        traces = asyncio.run(run_attempts(tasks, trials, attempt, concurrency=int(config.get("concurrency", 8)),
                                          on_trace=on_trace))
    finally:
        loader.close()
    summary = {**summarize(traces, trials=trials), "serving": serving,
               "serving_start_s": round(t1 - t0, 3), "attempts_wall_s": round(time.time() - t1, 3),
               "concurrency": int(config.get("concurrency", 8)), "infer": config.get("infer")}
    (out / SUMMARY).write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return summary
