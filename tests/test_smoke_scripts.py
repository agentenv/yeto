"""Pure-logic tests for the smoke/comparison harnesses (nothing launches)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load("smoke_models")
compare = _load("compare_diloco")
calibrate = _load("calibrate_fragment_score")
replay_merge = _load("replay_merge_utility")
group_local = _load("replay_group_local_probecommit")


def test_every_alias_has_a_tier():
    from yeto.models import MODEL_ALIASES

    for alias in MODEL_ALIASES:
        assert smoke.tier_of(alias) in smoke.TIERS


def test_tier_selection_is_cumulative_and_ordered():
    args = SimpleNamespace(only=None, skip=None, tier="small")
    small = smoke.select_models(args)
    assert small, "small tier should not be empty"
    assert "lfm25-230m" in small
    from yeto.models import MODEL_WEIGHT_GB

    sizes = [MODEL_WEIGHT_GB.get(a) for a in small]
    assert all(s is not None and s <= smoke.TIERS["small"] for s in sizes)
    assert sizes == sorted(sizes), "cheapest models must run first"
    args_med = SimpleNamespace(only=None, skip=None, tier="medium")
    assert set(small) <= set(smoke.select_models(args_med))


def test_only_and_skip():
    args = SimpleNamespace(only="lfm25-230m,qwen35-4b", skip="qwen35-4b", tier="small")
    assert smoke.select_models(args) == ["lfm25-230m"]
    import pytest

    with pytest.raises(SystemExit):
        smoke.select_models(SimpleNamespace(only="not-a-model", skip=None, tier="small"))


def test_run_names_are_cluster_safe():
    from yeto.models import MODEL_ALIASES

    for alias in MODEL_ALIASES:
        name = smoke.run_name(alias)
        assert all(c.isalnum() or c == "-" for c in name), name
        assert name.startswith("smk-")


def test_launch_command_uses_auto_planner():
    args = SimpleNamespace(
        data="org/ds", budget=15.0, total_steps=8, fragments=4,
        seq_len=1024, max_rows=512,
    )
    cmd = smoke.launch_command("lfm25-230m", args)
    assert "--gpu" not in cmd, "auto smoke must let the shape planner pick the fleet"
    assert "--budget" in cmd and "--confirm" in cmd
    # auto knobs stay at their defaults: no explicit micro-batch/lora-targets
    assert "--micro-batch-size" not in cmd and "--lora-targets" not in cmd


def test_compare_presets_cover_core_axes():
    arms = compare.select_arms("all")
    names = {a.name for a in arms}
    assert {"m2", "m4", "alpha0", "q4", "serial", "noheloco", "strided"} <= names
    assert compare.PRESETS["alpha0"].merge_alpha == 0.0
    assert compare.PRESETS["serial"].pipeline == 1
    assert compare.PRESETS["q4"].wire_dtype == "q4"


def test_token_budget_split_is_fair_across_learners():
    # Same budget, M learners -> per-learner steps shrink by ~M.
    b1 = compare.steps_for(1_000_000, 2, 512, 1)
    b4 = compare.steps_for(1_000_000, 2, 512, 4)
    assert b1 == 977 and b4 == 245  # ceil semantics
    assert compare.steps_for(1, 2, 512, 8) == 1  # never zero steps
    # DDP/FSDP ranks multiply tokens per step, so steps shrink by world too.
    assert compare.steps_for(1_000_000, 2, 512, 1, world=2) == 489


def test_multi_gpu_learner_uses_torchrun_and_gpu_blocks():
    args = SimpleNamespace(
        model="gemma4", lora_r=16, lora_alpha=32, seq_len=512,
        micro_batch_size=1, inner_lr=3e-4, device="cuda",
        shard="fsdp", learner_gpus=2,
    )
    cmd = compare.learner_command(
        args, Path("/tmp/w/m2"), learner_id=1, num_learners=2,
        syncer="127.0.0.1:1", max_steps=10, arm=compare.PRESETS["m2"],
    )
    assert "torch.distributed.run" in cmd and "--nproc_per_node=2" in cmd
    assert "--device" not in cmd  # torchrun ranks pick cuda from LOCAL_RANK
    assert cmd[cmd.index("--shard") + 1] == "fsdp"
    env = compare.gpu_env(1, 2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"  # learner 1 owns the 2nd block
    assert compare.gpu_env(0, 0) is None


def test_learner_command_arm_overrides():
    args = SimpleNamespace(
        model="lfm25-230m", lora_r=16, lora_alpha=32, seq_len=512,
        micro_batch_size=2, inner_lr=3e-4, device="cpu",
        shard="ddp", learner_gpus=0, fixed_window_tokens=8192,
        fixed_window_microsteps=64, pad_to_fixed_window_tokens=True,
        freeze_delta_before_delay=True, learner_push_delay_ms="0,50",
        learner_step_sleep_ms="0", learner_delay_jitter_ms=20.0,
    )
    arm = compare.PRESETS["q4"]
    cmd = compare.learner_command(
        args, Path("/tmp/w/q4"), learner_id=1, num_learners=2,
        syncer="127.0.0.1:1", max_steps=10, arm=arm,
    )
    assert cmd[cmd.index("--wire-dtype") + 1] == "q4"
    assert cmd[cmd.index("--data") + 1].endswith("train.jsonl")
    base = compare.learner_command(
        args, Path("/tmp/w/baseline"), learner_id=0, num_learners=1,
        syncer="none", max_steps=10, arm=None,
    )
    assert "--wire-dtype" not in base  # baseline never touches sync knobs
    assert "--fixed-window-tokens" in cmd
    assert "--fixed-window-tokens" not in base
    assert "--freeze-delta-before-delay" in cmd
    assert "--debug-push-delay-ms" in cmd
    assert "--debug-push-delay-ms" not in base


def test_syncer_command_quorum_defaults_to_all_learners():
    arm = compare.PRESETS["m4"]
    cmd = compare.syncer_command(arm, 1234, Path("/tmp/w/m4"), total_steps=100)
    assert cmd[cmd.index("--quorum") + 1] == "4"
    assert cmd[cmd.index("--checkpoint-every") + 1] == "1"
    assert cmd[cmd.index("--pipeline") + 1] == "2"
    # Experiment arms control sync frequency explicitly: the syncer's
    # adaptive H default is off for probe arms, on (24) only for m2h24.
    assert cmd[cmd.index("--sync-interval-steps") + 1] == "0.0"
    h24 = compare.syncer_command(compare.PRESETS["m2h24"], 1, Path("/tmp/w/h"), total_steps=1)
    assert h24[h24.index("--sync-interval-steps") + 1] == "24.0"


def test_syncer_command_can_enable_probe_capture():
    cmd = compare.syncer_command(
        compare.PRESETS["m12"],
        1234,
        Path("/tmp/w/m12"),
        total_steps=100,
        probe_capture=True,
        probe_capture_every=3,
    )
    assert cmd[cmd.index("--probe-capture-dir") + 1] == "/tmp/w/m12/syncer_probe"
    assert cmd[cmd.index("--probe-capture-every") + 1] == "3"


def test_calibrate_fragment_score_writes_heldout_split(tmp_path):
    log = tmp_path / "probe.jsonl"
    rows = []
    for learner_id in range(4):
        for step in range(8):
            update_norm = 0.05 if step < 4 else 1.0
            utility = 0.01 if step < 4 else -0.01
            rows.append(
                {
                    "learner_id": learner_id,
                    "seed": 1 if learner_id < 2 else 2,
                    "pull_step": step,
                    "fragment": 3 if step % 2 == 0 else 7,
                    "c_tokens": 8192,
                    "c_steps": 64,
                    "age": 1.0 + 0.1 * step,
                    "freshness": 1.0 / (2.0 + step),
                    "alignment": 0.5 if step < 4 else -0.5,
                    "uncertainty": 0.1 + 0.01 * step,
                    "norm_anomaly": update_norm,
                    "update_norm": update_norm,
                    "combined_score": 0.9 if step < 4 else 0.1,
                    "utility": utility,
                    "utility_se": 0.001,
                    "bad_strict": utility + 0.001 < 0.0,
                }
            )
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out_dir = tmp_path / "calibrated"
    rc = calibrate.main(
        [
            str(log),
            "--split",
            "heldout-learners",
            "--test-learners",
            "2,3",
            "--out-dir",
            str(out_dir),
            "--epochs",
            "20",
        ]
    )
    assert rc == 0
    assert (out_dir / "calibrated_train.jsonl").exists()
    assert (out_dir / "calibrated_test.jsonl").exists()
    summary = json.loads((out_dir / "calibration_summary.json").read_text())
    assert summary["train_records"] == 16
    assert summary["test_records"] == 16
    assert summary["split_meta"]["test_learners"] == [2, 3]
    assert "calibrated_score_auroc" in summary["test_summary"]
    test_rows = [
        json.loads(line)
        for line in (out_dir / "calibrated_test.jsonl").read_text().splitlines()
    ]
    assert all("calibrated_score" in row for row in test_rows)

    seed_out = tmp_path / "calibrated_seed"
    rc = calibrate.main(
        [
            str(log),
            "--split",
            "heldout-seed",
            "--test-seed",
            "2",
            "--out-dir",
            str(seed_out),
            "--epochs",
            "20",
        ]
    )
    assert rc == 0
    seed_summary = json.loads((seed_out / "calibration_summary.json").read_text())
    assert seed_summary["split_meta"]["test_seed"] == 2
    assert seed_summary["split_meta"]["train_seeds"] == [1]


def test_replay_merge_summary_reports_headroom():
    records = [
        {
            "candidate_count": 4,
            "individual_bad_rate": 0.5,
            "individual_strict_bad_rate": 0.25,
            "bad_weight_mass": 0.5,
            "strict_bad_weight_mass": 0.25,
            "token_weighted_utility": -0.01,
            "token_weighted_strict_negative": True,
            "token_weighted_selected_count": 4,
            "uniform_utility": -0.011,
            "uniform_strict_negative": True,
            "uniform_selected_count": 4,
            "freshness_weighted_utility": -0.005,
            "freshness_weighted_strict_negative": True,
            "freshness_weighted_selected_count": 4,
            "hand_score_weighted_utility": 0.001,
            "hand_score_weighted_strict_negative": False,
            "hand_score_weighted_selected_count": 4,
            "oracle_positive_utility": 0.02,
            "oracle_positive_strict_negative": False,
            "oracle_positive_selected_count": 2,
            "oracle_topk_utility": 0.03,
            "oracle_topk_strict_negative": False,
            "oracle_topk_selected_count": 2,
            "oracle_drop_strict_bad_utility": 0.015,
            "oracle_drop_strict_bad_strict_negative": False,
            "oracle_drop_strict_bad_selected_count": 3,
            "random_positive_count_utility": 0.0,
            "random_positive_count_strict_negative": False,
            "random_positive_count_selected_count": 2,
            "random_drop_strict_count_utility": -0.002,
            "random_drop_strict_count_strict_negative": False,
            "random_drop_strict_count_selected_count": 3,
        },
        {
            "candidate_count": 2,
            "individual_bad_rate": 0.0,
            "individual_strict_bad_rate": 0.0,
            "bad_weight_mass": 0.0,
            "strict_bad_weight_mass": 0.0,
            "token_weighted_utility": 0.01,
            "token_weighted_strict_negative": False,
            "token_weighted_selected_count": 2,
            "uniform_utility": 0.009,
            "uniform_strict_negative": False,
            "uniform_selected_count": 2,
            "freshness_weighted_utility": 0.011,
            "freshness_weighted_strict_negative": False,
            "freshness_weighted_selected_count": 2,
            "hand_score_weighted_utility": 0.012,
            "hand_score_weighted_strict_negative": False,
            "hand_score_weighted_selected_count": 2,
            "oracle_positive_utility": 0.02,
            "oracle_positive_strict_negative": False,
            "oracle_positive_selected_count": 2,
            "oracle_topk_utility": 0.025,
            "oracle_topk_strict_negative": False,
            "oracle_topk_selected_count": 1,
            "oracle_drop_strict_bad_utility": 0.02,
            "oracle_drop_strict_bad_strict_negative": False,
            "oracle_drop_strict_bad_selected_count": 2,
            "random_positive_count_utility": 0.01,
            "random_positive_count_strict_negative": False,
            "random_positive_count_selected_count": 2,
            "random_drop_strict_count_utility": 0.01,
            "random_drop_strict_count_strict_negative": False,
            "random_drop_strict_count_selected_count": 2,
        },
    ]
    summary = replay_merge.summarize(records)
    assert summary["records"] == 2
    assert summary["policies"]["token_weighted"]["negative_merge_rate"] == 0.5
    assert summary["headroom"]["oracle_positive_minus_token"]["mean"] == 0.02
    assert replay_merge._topk_count(6, None, 0.5) == 3
    assert replay_merge._topk_count(6, 10, 0.5) == 6
    groups = replay_merge._group_rows(
        [
            {"state_checkpoint": "b", "step": 2, "fragment": 1, "learner_id": 1},
            {"state_checkpoint": "a", "step": 1, "fragment": 1, "learner_id": 1},
            {"state_checkpoint": "a", "step": 1, "fragment": 1, "learner_id": 0},
        ]
    )
    assert [[r["learner_id"] for r in group] for group in groups] == [[0, 1], [1]]


def test_group_local_tuner_excludes_oracle_actions():
    def candidate(seed, step, fragment, learner, score, utility):
        return {
            "seed": seed,
            "pull_step": step,
            "fragment": fragment,
            "learner_id": learner,
            "probe_grad_dot": score,
            "calibrated_score": score,
            "freshness": 1.0,
            "combined_score": score,
            "utility": utility,
            "bad": utility < 0,
            "bad_strict": utility < -0.001,
            "weight": 1.0,
        }

    feature_groups = []
    policy_groups = []
    for seed in [1, 2, 3]:
        for step in range(4):
            rows = [
                candidate(seed, step, 0, 0, 0.9, 0.02),
                candidate(seed, step, 0, 1, 0.1, -0.01),
            ]
            stats = {"probe_grad_dot": group_local.group_stats(rows, "probe_grad_dot")}
            replay = {
                "token_weighted_utility": 0.0,
                "token_weighted_negative": False,
                "token_weighted_strict_negative": False,
                "token_weighted_selected_mass": 1.0,
                "token_weighted_selected_count": 2,
                "freshness_weighted_utility": 0.0,
                "freshness_weighted_negative": False,
                "freshness_weighted_strict_negative": False,
                "freshness_weighted_selected_mass": 1.0,
                "freshness_weighted_selected_count": 2,
                "anchor_drop_bottom25_utility": 0.01,
                "anchor_drop_bottom25_negative": False,
                "anchor_drop_bottom25_strict_negative": False,
                "anchor_drop_bottom25_selected_mass": 0.5,
                "anchor_drop_bottom25_selected_count": 1,
                "anchor_positive_threshold_utility": 0.01,
                "anchor_positive_threshold_negative": False,
                "anchor_positive_threshold_strict_negative": False,
                "anchor_positive_threshold_selected_mass": 0.5,
                "anchor_positive_threshold_selected_count": 1,
                "anchor_shrink_utility": 0.0,
                "anchor_shrink_negative": False,
                "anchor_shrink_strict_negative": False,
                "anchor_shrink_selected_mass": 1.0,
                "anchor_shrink_selected_count": 2,
                "probecommit_v1_utility": 0.01,
                "probecommit_v1_negative": False,
                "probecommit_v1_strict_negative": False,
                "probecommit_v1_selected_mass": 0.5,
                "probecommit_v1_selected_count": 1,
                # Oracle is deliberately much better. The tuner must not select it.
                "oracle_positive_utility": 10.0,
                "oracle_positive_negative": False,
                "oracle_positive_strict_negative": False,
                "oracle_positive_selected_mass": 0.5,
                "oracle_positive_selected_count": 1,
                "oracle_topk_utility": 10.0,
                "oracle_topk_negative": False,
                "oracle_topk_strict_negative": False,
                "oracle_topk_selected_mass": 0.5,
                "oracle_topk_selected_count": 1,
                "random_probecommit_count_utility": 0.0,
                "random_probecommit_count_negative": False,
                "random_probecommit_count_strict_negative": False,
                "random_probecommit_count_selected_mass": 0.5,
                "random_probecommit_count_selected_count": 1,
            }
            feature_groups.append(
                {
                    "seed": seed,
                    "step": step,
                    "fragment": 0,
                    "candidate_count": 2,
                    "candidates": rows,
                    "stats": stats,
                    "replay": replay,
                }
            )
            policy_groups.append(replay)

    rules = group_local.candidate_rules(feature_groups, ["probe_grad_dot"])
    assert all("oracle" not in rule.name and "random" not in rule.name for rule in rules)
    result = group_local.heldout_seed_replay(
        feature_groups,
        ["probe_grad_dot"],
        negative_penalty=0.0,
        strict_penalty=0.0,
    )
    selected = [split["test_result"]["actions"] for split in result["splits"]]
    assert all("oracle_positive" not in actions for actions in selected)
    assert result["aggregate"]["mean_gain_vs_token"] == 0.01
