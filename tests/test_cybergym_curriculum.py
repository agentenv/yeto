from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "select_cybergym_curriculum", ROOT / "scripts" / "select_cybergym_curriculum.py"
)
curriculum = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(curriculum)


def test_selector_interleaves_hard_tasks_and_preserves_heldout(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    rows = []
    for task_id in (
        "easy",
        "boundary-a",
        "boundary-b",
        "boundary-c",
        "hard-a",
        "heldout",
    ):
        rows.append(
            {
                "messages": [],
                "metadata": {"task_id": task_id},
            }
        )
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "baseline_attempts_per_task": 16,
                "boundary_baseline_successes": {
                    "boundary-a": 14,
                    "boundary-b": 8,
                    "boundary-c": 1,
                },
                "boundary_task_ids": ["boundary-a", "boundary-b", "boundary-c"],
                "hard_task_ids": ["hard-a"],
                "easy_success_task_ids": ["easy"],
                "heldout_task_ids": ["heldout"],
            }
        )
    )

    train, evaluation, manifest = curriculum.select(
        prompts,
        selection,
        train_count=5,
        eval_count=1,
    )

    assert len(train) == 5
    assert len(evaluation) == 1
    assert train[0]["metadata"]["task_id"] == "boundary-a"
    assert train[3]["metadata"]["training_bucket"] == "hard"
    assert evaluation[0]["metadata"]["task_id"] == "heldout"
    assert set(manifest["train_tasks"]).isdisjoint(manifest["eval_tasks"])


def test_selector_rejects_missing_selected_task(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"metadata": {"task_id": "present"}}) + "\n")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"boundary_task_ids": ["missing"]}))

    with pytest.raises(ValueError, match="missing selected task IDs"):
        curriculum.select(prompts, selection, train_count=1, eval_count=0)
