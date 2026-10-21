"""S26 E9.5: TB2 hold-out / training split / curriculum over Harbor task directories (CPU).

Synthetic checkouts in ``tmp_path`` for the rules; the real TB2 checkout
(``YETO_TB2_DIR``, default ~/work/tb2-data) for byte identity with the lists
in ``data/eval`` (skipped when absent).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from yeto.rl.harness.harbor import task_lists as tl
from yeto.rl.harness.reward_env import benchmark as bm

REPO = Path(__file__).resolve().parents[1]


def _task(root: Path, tid: str, difficulty: str, expert: float | None = 30.0) -> None:
    d = root / tid
    (d / "environment").mkdir(parents=True)
    meta = f'difficulty = "{difficulty}"\ncategory = "x"\n'
    if expert is not None:
        meta += f"expert_time_estimate_min = {expert}\njunior_time_estimate_min = {expert * 2}\n"
    (d / "task.toml").write_text(
        f'[task]\nname = "terminal-bench/{tid}"\n[metadata]\n{meta}'
        '[verifier]\ntimeout_sec = 900.0\n[agent]\ntimeout_sec = 1800.0\n'
        '[environment]\ndocker_image = "img"\ncpus = 2\nmemory_mb = 4096\n')
    (d / "environment" / "Dockerfile").write_text("FROM x\nWORKDIR /app\n")


def _checkout(root: Path, *, nested: bool = False) -> Path:
    tasks = root / "tasks" if nested else root
    for i in range(3):
        _task(tasks, f"easy-{i}", "easy", 5.0 * (i + 1))
    for i in range(20):
        _task(tasks, f"med-{i:02d}", "medium", 10.0 * (i % 7) + 5)
    for i in range(12):
        _task(tasks, f"hard-{i:02d}", "hard", None if i == 0 else 60.0 * i)
    for tid in tl.SMOKE6_TASK_IDS:
        _task(tasks, tid, "medium", 20.0)
    return root


def _unusable(tmp_path: Path, items) -> Path:
    path = tmp_path / "unusable.json"
    path.write_text(json.dumps({"schema": tl.UNUSABLE_SCHEMA, "benchmark": "tb2", "items": [
        {"task_id": t, "category": "timeout", "reason": "r"} for t in items]}))
    return path


def test_task_set_reads_both_layouts_and_the_task_facts(tmp_path):
    flat = tl.HarborTaskSet(_checkout(tmp_path / "flat"), version=None)
    nested = tl.HarborTaskSet(_checkout(tmp_path / "nested", nested=True))
    assert flat.task_ids() == nested.task_ids() and len(flat.task_ids()) == 41
    assert flat.version == "tb2@unknown" and nested.version == "tb2.1@unknown"  # not git checkouts
    spec = nested.task_spec("hard-03")
    assert (spec.difficulty, spec.eval_bucket, spec.difficulty_source) == ("hard", "tb2-hard", "tb2-task.toml")
    assert (spec.cpus, spec.memory_mb, spec.workdir, spec.agent_timeout_s) == (2, 4096, "/app", 1800.0)
    assert nested.facts("hard-00").expert_time_min is None
    with pytest.raises(FileNotFoundError):
        tl.HarborTaskSet(tmp_path / "empty")


def test_split_is_deterministic_disjoint_and_follows_the_rule(tmp_path):
    tasks = tl.HarborTaskSet(_checkout(tmp_path), version="tb2@test")
    unusable = tl.load_unusable(_unusable(tmp_path, ["med-03", "hard-05"]))
    holdout, train = tl.build_tb2_lists(tasks, unusable=unusable)
    again = tl.build_tb2_lists(tl.HarborTaskSet(tmp_path, version="tb2@test"), unusable=unusable)
    assert tl.dump_list(holdout) == tl.dump_list(again[0]) and tl.dump_list(train) == tl.dump_list(again[1])
    held = bm.holdout_ids(holdout)
    buckets = [i["eval_bucket"] for i in holdout["items"]]
    assert (buckets.count("tb2-easy"), buckets.count("tb2-medium"), buckets.count("tb2-hard")) == (2, 18, 10)
    assert not set(held) & (set(tl.SMOKE6_TASK_IDS) | set(unusable))
    train_ids = [i["task_id"] for i in train["items"]]
    assert not set(train_ids) & set(held) and not set(train_ids) & set(unusable)
    assert set(tl.SMOKE6_TASK_IDS) <= set(train_ids)  # smoke-6 leave the eval pool only
    assert train["holdout_sha256"] == bm.holdout_sha256(holdout)
    assert holdout["rule"] == tl.HOLDOUT_RULE_UNUSABLE
    assert len(held) + len(train_ids) + len(unusable) == len(tasks.task_ids())


def test_split_fails_closed_on_unknown_or_clashing_tasks(tmp_path):
    tasks = tl.HarborTaskSet(_checkout(tmp_path), version="v")
    with pytest.raises(ValueError, match="not in the checkout"):
        tl.build_tb2_lists(tasks, unusable={"nope": "x"})
    with pytest.raises(ValueError, match="both in the exclusion"):
        tl.build_tb2_lists(tasks, unusable={"fix-git": "x"})
    with pytest.raises(ValueError, match="quota"):
        tl.build_tb2_lists(tasks, unusable={f"hard-{i:02d}": "x" for i in range(5)})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "benchmark": "tb2", "items": []}))
    with pytest.raises(ValueError, match="not a"):
        tl.load_unusable(bad)


@pytest.mark.skipif(not (Path(os.environ.get("YETO_TB2_DIR", Path.home() / "work" / "tb2-data")) / "fix-git").is_dir(),
                    reason="TB2 checkout tb2@2fd12b8 not present")
def test_tb2_checkout_reproduces_the_lists_in_data_eval():
    root = Path(os.environ.get("YETO_TB2_DIR", Path.home() / "work" / "tb2-data"))
    tasks = tl.HarborTaskSet(root)
    if tasks.version != "tb2@2fd12b8":
        pytest.skip(f"checkout is {tasks.version}")
    holdout, train = tl.build_tb2_lists(tasks, unusable=tl.load_unusable())
    assert tl.dump_list(holdout) == (REPO / "data/eval/tb2-holdout.json").read_text()
    assert tl.dump_list(train) == (REPO / "data/eval/tb2-train.json").read_text()


def test_curriculum_stages_follow_public_facts_and_measured_signal(tmp_path):
    tasks = tl.HarborTaskSet(_checkout(tmp_path), version="tb2@test")
    _, train = tl.build_tb2_lists(tasks, unusable={})
    cur = tl.build_curriculum(tasks, train)
    names = [s["name"] for s in cur["stages"]]
    assert names == ["s1-short", "s2-medium", "s3-long"]
    placed = {i["task_id"]: s["name"] for s in cur["stages"] for i in s["items"]}
    assert set(placed) == {i["task_id"] for i in train["items"]} and cur["dropped"] == []
    for stage in cur["stages"]:
        for item in stage["items"]:
            rule = next(r for r in tl.DEFAULT_STAGES if r.name == stage["name"])
            assert rule.admits(tasks.facts(item["task_id"]))
            assert all(not r.admits(tasks.facts(item["task_id"])) for r in tl.DEFAULT_STAGES[:names.index(stage["name"])])
    assert placed["hard-00"] == "s3-long"  # no expert estimate: only the unbounded stage
    first = cur["stages"][0]["items"]
    assert first == sorted(first, key=lambda e: (tl._DIFFICULTY_RANK[e["difficulty"]], e["expert_time_estimate_min"], e["task_id"]))
    order = tl.curriculum_task_ids(cur, through="s1-short")
    assert order == [i["task_id"] for i in first]
    assert tl.curriculum_task_ids(cur) == [i["task_id"] for s in cur["stages"] for i in s["items"]]
    with pytest.raises(ValueError, match="unknown stage"):
        tl.curriculum_task_ids(cur, through="s9")

    ids = [i["task_id"] for i in train["items"]]
    ref = tmp_path / "ref.jsonl"
    ref.write_text("\n".join([json.dumps({"task": ids[0], "passed": False}),
                              json.dumps({"task": ids[1], "error": "infra"}),
                              json.dumps({"task": ids[2], "passed": True, "solve_s": 3.5})]))
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({ids[2]: {"n": 8, "passed": 3}, ids[3]: {"n": 8, "passed": 0},
                                 ids[4]: {"n": 8, "passed": 8}}))
    cur = tl.build_curriculum(tasks, train, reference=tl.load_reference_results(ref),
                              pass_rates=tl.load_pass_rates(rates), require_signal=True)
    kept = [i for s in cur["stages"] for i in s["items"]]
    assert [i["task_id"] for i in kept] == [ids[2]]
    assert kept[0]["reference_solution"] == {"passed": True, "solve_s": 3.5}
    assert (kept[0]["baseline_passed"], kept[0]["baseline_samples"]) == (3, 8)
    reasons = {d["task_id"]: d["reason"] for d in cur["dropped"]}
    assert reasons[ids[0]].startswith("reference solution fails") and "0/8" in reasons[ids[3]]
    assert "8/8" in reasons[ids[4]] and reasons[ids[1]] == "no measured pass rate"
    with pytest.raises(ValueError, match="needs measured"):
        tl.build_curriculum(tasks, train, require_signal=True)


def test_reference_results_fail_closed_on_disagreement(tmp_path):
    ref = tmp_path / "ref.jsonl"
    ref.write_text(json.dumps({"task": "a", "passed": True}) + "\n" + json.dumps({"task": "a", "passed": False}))
    with pytest.raises(ValueError, match="disagree"):
        tl.load_reference_results(ref)
    ref.write_text(json.dumps({"task": "a"}))
    with pytest.raises(ValueError, match="boolean passed"):
        tl.load_reference_results(ref)


def test_cli_writes_identical_bytes_twice(tmp_path):
    from importlib import util

    spec = util.spec_from_file_location("tb2_lists", REPO / "tools/reward_env/tb2_lists.py")
    cli = util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    root = _checkout(tmp_path / "co", nested=True)
    for out in ("a", "b"):
        assert cli.main(["split", "--tasks", str(root), "--out-dir", str(tmp_path / out), "--no-unusable",
                         "--version", "tb2.1@test"]) == 0
        assert cli.main(["curriculum", "--tasks", str(root), "--train", str(tmp_path / out / "tb2-train.json"),
                         "--out", str(tmp_path / out / "cur.json"), "--version", "tb2.1@test"]) == 0
    for name in ("tb2-holdout.json", "tb2-train.json", "cur.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
