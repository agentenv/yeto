"""RLF real-GPU RL gate runner orchestration (task 5.8): fake learner / fake
provider only — no GPU, no Miles, no torch, no Modal.

These tests cover the orchestration (collection -> gate rows -> rl_gate ->
evidence).  The injected collection can never announce a passing verdict;
only the real ``miles_island`` collector on the GPU run may do that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.eval.rlf_rl_gate import (
    EVIDENCE_SCHEMA,
    GateConfig,
    GateEvidenceError,
    build_gate_rows,
    collect_policy_update,
    collect_rollout_samples,
    prepare_prompt_data,
    run_gate,
    write_evidence,
)

FLAGS = {"fixture-ctf-ezshell": "flag{ezshell-secret}", "fixture-ctf-ret2win": "flag{ret2win-secret}"}


def _sample(group, index, reward, success, task_id="fixture-ctf-ezshell"):
    return {"reward": reward,
            "metadata": {"success": success, "rlf_task_id": task_id,
                         "rlf_episode_id": f"ep-{group}-{index}",
                         "rlf_trajectory_id": f"traj-{group}-{index}",
                         "grader_evidence": {"trusted": True}}}


def _payload(groups):
    return {"schema_version": 1, "completed_groups": groups}


def _passing_groups():
    # ezshell group: one success (variance + success_count); ret2win group: all fail.
    return [[_sample(0, 0, 1.0, True), _sample(0, 1, 0.0, False),
             _sample(0, 2, 0.0, False), _sample(0, 3, 0.0, False)],
            [_sample(1, 0, 0.0, False, "fixture-ctf-ret2win"),
             _sample(1, 1, 0.0, False, "fixture-ctf-ret2win")]]


def _update_loader(before, after):
    tables = {"before": before, "after": after}
    return lambda path: tables[str(path)]


def _update(updated=True):
    before = {"lora_a": 0.5, "lora_b": -0.25}
    after = {"lora_a": 0.5 + (0.01 if updated else 0.0), "lora_b": -0.25}
    return collect_policy_update("before", "after", step=2,
                                 loader=_update_loader(before, after))


def _config(tmp_path):
    return GateConfig(tasks_dir=str(tmp_path), flags=FLAGS)


# -- collection from the completed-groups checkpoint -------------------------

def test_collect_rollout_samples_reads_real_checkpoint_shape():
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    assert len(samples) == 6
    assert samples[0] == {"group_index": 0, "index": 0, "task_id": "fixture-ctf-ezshell",
                          "reward": 1.0, "success": True}
    assert sum(s["success"] for s in samples) == 1


def test_collect_flattens_compaction_segments_sharing_one_outcome():
    group = [[_sample(0, 0, 1.0, True), _sample(0, 0, 1.0, True)], _sample(0, 1, 0.0, False)]
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload([group]))
    assert [s["reward"] for s in samples] == [1.0, 0.0]


def test_collect_fails_closed_on_missing_success_or_reward():
    bad = _sample(0, 0, 1.0, True)
    del bad["metadata"]["success"]
    with pytest.raises(GateEvidenceError, match="metadata.success"):
        collect_rollout_samples("ckpt", loader=lambda _p: _payload([[bad]]))
    bad = _sample(0, 0, 1.0, True)
    bad["reward"] = None
    with pytest.raises(GateEvidenceError, match="numeric reward"):
        collect_rollout_samples("ckpt", loader=lambda _p: _payload([[bad]]))
    with pytest.raises(GateEvidenceError, match="no completed groups"):
        collect_rollout_samples("ckpt", loader=lambda _p: _payload([]))


# -- advantages and policy update are computed, never hand-filled ------------

def test_gate_rows_compute_grpo_group_advantages():
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    rows = build_gate_rows(samples, _update())
    first = [r["advantage"] for r in rows if r["metadata"]["group_index"] == 0]
    assert first == [0.75, -0.25, -0.25, -0.25]  # reward minus group mean
    assert all(r["policy_update"] for r in rows)


def test_policy_update_unchanged_params_marks_rows_not_updated():
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    rows = build_gate_rows(samples, _update(updated=False))
    assert not any(r["policy_update"] for r in rows)


# -- the five gate criteria ---------------------------------------------------

def test_all_five_criteria_pass_but_injected_collection_never_announces(tmp_path):
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    update = _update()
    rows = build_gate_rows(samples, update)
    evidence = run_gate(rows, collector="injected", config=_config(tmp_path), update=update)
    gate = evidence["gate"]
    assert gate["passed"] and gate["result"]["reward_std"] > 0
    assert gate["result"]["success_count"] == 1
    assert gate["result"]["advantage_abs_sum"] > 0
    # CPU orchestration only: the 5.8 gate is never announced off the real run.
    assert evidence["verdict"] is False
    assert "never announced" in evidence["verdict_note"]


def test_all_zero_rewards_fail():
    groups = [[_sample(0, i, 0.0, False) for i in range(4)]]
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(groups))
    evidence = run_gate(build_gate_rows(samples, _update()), collector="miles_island")
    assert evidence["verdict"] is False and evidence["gate"]["passed"] is False
    assert "reward_std_positive" in evidence["gate"]["error"]
    assert "success_count_positive" in evidence["gate"]["error"]
    assert "advantage_abs_sum_positive" in evidence["gate"]["error"]


def test_constant_nonzero_rewards_fail():
    groups = [[_sample(0, i, 1.0, True) for i in range(4)]]
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(groups))
    evidence = run_gate(build_gate_rows(samples, _update()), collector="miles_island")
    assert evidence["verdict"] is False
    assert "reward_std_positive" in evidence["gate"]["error"]


def test_no_policy_update_fails_even_with_real_collector():
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    update = _update(updated=False)
    rows = build_gate_rows(samples, update)
    evidence = run_gate(rows, collector="miles_island", update=update)
    assert evidence["verdict"] is False
    assert "policy_update" in evidence["gate"]["error"]
    pu = evidence["policy_update_evidence"]
    assert pu["updated"] is False and pu["before_digest"] == pu["after_digest"]


def test_real_collector_passing_run_announces_verdict(tmp_path):
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    update = _update()
    rows = build_gate_rows(samples, update)
    evidence = run_gate(rows, collector="miles_island", config=_config(tmp_path), update=update)
    assert evidence["verdict"] is True and evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["policy_update_evidence"]["before_digest"] != \
        evidence["policy_update_evidence"]["after_digest"]


# -- evidence on disk: content and flag hygiene --------------------------------

def test_write_evidence_scrubs_flags_and_keeps_detail(tmp_path):
    config = _config(tmp_path)
    samples = collect_rollout_samples("ckpt", loader=lambda _p: _payload(_passing_groups()))
    rows = build_gate_rows(samples, _update())
    rows[0]["metadata"]["note"] = f"grader saw {FLAGS['fixture-ctf-ezshell']} then submitted"
    evidence = run_gate(rows, collector="injected", config=config, update=_update())
    out = write_evidence(evidence, tmp_path / "evidence.json", secrets=config.flags.values())
    text = out.read_text()
    for flag in FLAGS.values():
        assert flag not in text
    assert "***" in text
    loaded = json.loads(text)
    assert loaded["schema"] == EVIDENCE_SCHEMA
    assert {r["reward"] for r in loaded["rows"]} == {0.0, 1.0}
    assert "flag{" not in json.dumps(loaded["config_digest"])  # digest only


def test_config_requires_flags_for_gate_tasks(tmp_path):
    with pytest.raises(GateEvidenceError, match="no trusted flag"):
        GateConfig(tasks_dir=str(tmp_path), flags={"fixture-ctf-ezshell": "flag{x}"})


# -- prompt data preparation ----------------------------------------------------

def _package(root, task_id):
    (root / "public").mkdir(parents=True)
    (root / "public" / "README.md").write_text(f"# {task_id}\nObtain the flag.")
    (root / "task.yaml").write_text(f"task_id: {task_id}\ngrader_mode: flag\n")
    (root / "MANIFEST.json").write_text(json.dumps({"task_id": task_id, "grader_mode": "flag"}))


def test_prepare_prompt_data_admits_gate_tasks(tmp_path):
    for task_id in FLAGS:
        _package(tmp_path / task_id, task_id)
    out = prepare_prompt_data(_config(tmp_path), tmp_path / "prompt.jsonl")
    rows = [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]
    assert len(rows) == 2 * len(FLAGS)  # repeat=2 per task
    assert {r["metadata"]["rlf_task_id"] for r in rows} == set(FLAGS)
    assert all(r["metadata"]["benchmark"] == "rlf" for r in rows)


def test_prepare_prompt_data_rejects_unadmitted_package(tmp_path):
    _package(tmp_path / "fixture-ctf-ezshell", "fixture-ctf-ezshell")
    from yeto.rl.rlf.contracts import AdmissionError
    with pytest.raises(AdmissionError):
        prepare_prompt_data(_config(tmp_path), tmp_path / "prompt.jsonl")
