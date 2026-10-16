"""S24 T-TB2: TB2 base-model baseline runner (CPU only; fake serving, provider and agent)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.eval import tb2_baseline as tb
from yeto.rl.eval.island import EvalTask

REPO = Path(__file__).resolve().parents[1]
SPLIT = {"benchmark": "tb2", "items": [{"task_id": "a", "difficulty": "easy", "eval_bucket": "tb2-easy"},
                                      {"task_id": "b", "difficulty": "hard", "eval_bucket": "tb2-hard"}]}
ROWS = [{"metadata": {"task_id": "a"}}, {"metadata": {"task_id": "b"}}]


def test_load_tasks_requires_rows_and_known_ids():
    tasks = tb.load_tasks(SPLIT, ROWS)
    assert [(t.task_id, t.difficulty) for t in tasks] == [("a", "easy"), ("b", "hard")]
    assert tb.load_tasks(SPLIT, ROWS, ["b"])[0].row == ROWS[1]
    with pytest.raises(ValueError, match="without a jsonl row"):
        tb.load_tasks(SPLIT, ROWS[:1])
    with pytest.raises(ValueError, match="not in the split"):
        tb.load_tasks(SPLIT, ROWS, ["a", "zz"])


def test_shipped_train_split_has_46_tasks_with_rows():
    split = json.loads((REPO / "data/eval/tb2-train.json").read_text())
    rows = [json.loads(x) for x in (REPO / "data/tb2/tb2-train.jsonl").read_text().splitlines() if x.strip()]
    assert len(tb.load_tasks(split, rows)) == 46


def test_run_attempts_bounds_concurrency_and_keeps_failures():
    live, peak, seen = [0], [0], []

    async def attempt(task, trial, *, policy_version, policy_token):
        assert policy_version == tb.BASE_POLICY_VERSION and policy_token == "base"
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0.01)
        live[0] -= 1
        if task.task_id == "b" and trial == 1:
            raise RuntimeError("sandbox create failed")
        return {"success": task.task_id == "a", "reward": float(task.task_id == "a"), "end_reason": "completed"}

    tasks = tb.load_tasks(SPLIT, ROWS)
    traces = asyncio.run(tb.run_attempts(tasks, 3, attempt, concurrency=2, on_trace=seen.append))
    assert len(traces) == len(seen) == 6 and peak[0] == 2
    infra = [t for t in traces if t["end_reason"] == "infra_error"]
    assert len(infra) == 1 and "sandbox create failed" in infra[0]["error"]


def test_summarize_excludes_infra_and_counts_mixed_tasks():
    def t(task, ok, end="completed", diff="easy"):
        return {"task_id": task, "difficulty": diff, "success": ok, "end_reason": end, "turns": 3,
                "timing": {"agent_s": 10.0}}

    traces = [t("a", True), {**t("a", False), "agent_detail": {"end_reason": "CodexModelFailure: x"}}, t("b", False, diff="hard"), t("b", False, "infra_error", "hard"),
              t("c", True), t("c", True)]
    s = tb.summarize(traces, trials=2)
    assert (s["attempts"], s["judged"], s["infra_errors"], s["passed"]) == (6, 5, 1, 3)
    assert s["pass_rate"] == 0.6 and s["by_difficulty"]["hard"] == {"passed": 0, "judged": 1, "pass_rate": 0.0}
    assert s["mixed_task_ids"] == ["a"] and s["tasks_any_pass"] == 2 and s["tasks_all_pass"] == 1
    assert s["end_reasons"] == {"completed": 5, "infra_error": 1} and s["agent_s"]["n"] == 5
    assert s["driver_end_reasons"] == {"None": 5, "CodexModelFailure: x": 1}


def test_run_from_config_writes_traces_and_summary_and_closes(tmp_path):
    (tmp_path / "split.json").write_text(json.dumps(SPLIT))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    closed = []
    loader = SimpleNamespace(start_base=lambda: {"sglang_start_s": 1.0}, close=lambda: closed.append(1))

    class Lease:
        verifier = SimpleNamespace(evaluate=None)

        async def destroy(self):
            pass

    async def evaluate(_episode):
        return {"passed": True}

    Lease.verifier.evaluate = evaluate

    async def acquire(task_id, tid):
        return Lease()

    async def agent(lease, task, trial, *, policy_token):
        return {"episode_id": "e", "end_reason": "completed", "turns": 2, "tokens": 100}

    out = tmp_path / "out"
    summary = tb.run_from_config({"split": str(tmp_path / "split.json"), "rows": str(tmp_path / "rows.jsonl"),
                                  "out_dir": str(out), "trials": 2, "concurrency": 3,
                                  "infer": {"base_model": "B"}},
                                 provider=SimpleNamespace(acquire=acquire), loader=loader, agent=agent)
    assert closed == [1] and summary["pass_rate"] == 1.0 and summary["attempts"] == 4
    assert len((out / tb.TRACES).read_text().splitlines()) == 4
    assert json.loads((out / tb.SUMMARY).read_text())["serving"] == {"sglang_start_s": 1.0}


def test_run_from_config_closes_serving_when_start_fails(tmp_path):
    (tmp_path / "split.json").write_text(json.dumps(SPLIT))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    closed = []

    def boom():
        raise RuntimeError("sglang exited")

    loader = SimpleNamespace(start_base=boom, close=lambda: closed.append(1))
    with pytest.raises(RuntimeError, match="sglang exited"):
        tb.run_from_config({"split": str(tmp_path / "split.json"), "rows": str(tmp_path / "rows.jsonl"),
                            "out_dir": str(tmp_path / "o"), "infer": {"base_model": "B"}}, provider=object(), loader=loader, agent=object())
    assert closed == [1]


def test_function_spec_env_pins_tb2_dir():
    from yeto.cloud.modal_tb2_baseline import BaselineFunctionSpec

    spec = BaselineFunctionSpec(app_name="a", workdir="w", codex_dir="c", tb2_dir="t", image_ref="img",
                                envs={"SECRLENV_MAX_TURNS": "32"})
    env = spec.container_env()
    assert env["YETO_HARNESS_TB2_TASKS_DIR"] == "/root/tb2-data" and env["SECRLENV_MAX_TURNS"] == "32"
    assert "PYTHONPATH" not in env and spec.gpu == "H200"
