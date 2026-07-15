"""Pure-logic tests for the smoke/comparison harnesses (nothing launches)."""

import importlib.util
import json
import os
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
build_group_features = _load("build_group_local_features")
policy_grid = _load("replay_group_local_policy_grid")
hard_search = _load("search_group_local_policy")
action_probe = _load("replay_action_probe_policy")
action_probe_agg = _load("aggregate_action_probe_results")
action_stability = _load("analyze_action_probe_stability")
buffered_robust = _load("replay_buffered_robust_syncer")
buffered_nesterov = _load("replay_buffered_nesterov_syncer")
buffered_nesterov_agg = _load("aggregate_buffered_nesterov")
lr_action_probe = _load("analyze_lr_action_probe")
anchor_gradient_syncer = _load("replay_anchor_gradient_syncer")
anchor_gradient_agg = _load("aggregate_anchor_gradient_syncer")
fragment_lr_profiles = _load("analyze_fragment_lr_profiles")


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


def test_compare_outer_optimizer_override_is_explicit():
    from dataclasses import replace

    arm = compare.PRESETS["m4"]
    tuned = replace(
        arm,
        outer_lr=0.35,
        outer_momentum=0.8,
        outer_optimizer="restarted-ema",
        outer_restart_cos_threshold=-0.25,
    )
    cmd = compare.syncer_command(tuned, 1234, Path("/tmp/w/m4"), total_steps=100)
    assert cmd[cmd.index("--outer-lr") + 1] == "0.35"
    assert cmd[cmd.index("--outer-momentum") + 1] == "0.8"
    assert cmd[cmd.index("--outer-optimizer") + 1] == "restarted-ema"
    assert cmd[cmd.index("--outer-restart-cos-threshold") + 1] == "-0.25"


def test_compare_outer_optimizer_defaults_match_syncer_defaults():
    arm = compare.PRESETS["m4"]

    assert arm.outer_optimizer == "nesterov"
    assert arm.outer_restart_cos_threshold == 0.0
    cmd = compare.syncer_command(arm, 1234, Path("/tmp/w/m4"), total_steps=40)
    assert cmd[cmd.index("--outer-optimizer") + 1] == "nesterov"
    assert cmd[cmd.index("--outer-restart-cos-threshold") + 1] == "0.0"


def test_compare_outer_optimizer_override_applies_to_every_selected_arm():
    original = compare.select_arms("m4,q4")
    overridden = compare.apply_arm_overrides(
        original,
        outer_optimizer="normalized-ema",
        outer_restart_cos_threshold=-0.1,
    )

    assert all(arm.outer_optimizer == "normalized-ema" for arm in overridden)
    assert all(arm.outer_restart_cos_threshold == -0.1 for arm in overridden)
    assert all(after is not before for before, after in zip(original, overridden))
    assert compare.PRESETS["m4"].outer_optimizer == "nesterov"
    assert compare.PRESETS["m4"].outer_restart_cos_threshold == 0.0


def test_compare_delta_correction_override_applies_to_every_selected_arm():
    original = compare.select_arms("m4,q4")
    overridden = compare.apply_arm_overrides(original, delta_correction="none")

    assert [arm.name for arm in overridden] == ["m4", "q4"]
    assert all(arm.delta_correction == "none" for arm in overridden)
    assert all(after is not before for before, after in zip(original, overridden))
    assert compare.PRESETS["m4"].delta_correction == "heloco"
    assert compare.PRESETS["q4"].delta_correction == "heloco"


def test_compare_delta_correction_cli_overrides_m4(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--settings",
            "m4",
            "--delta-correction",
            "none",
            "--outer-optimizer",
            "restarted-ema",
            "--outer-restart-cos-threshold",
            "-0.25",
            "--dry-run",
        ],
    )

    assert compare.main() == 0
    out = capsys.readouterr().out
    assert "m4" in out and "M=4" in out
    assert "optimizer=restarted-ema" in out
    assert "restart_cos=-0.25" in out
    assert "correction=none" in out


def test_compare_outer_optimizer_cli_rejects_invalid_choice(monkeypatch, capsys):
    import pytest

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--outer-optimizer",
            "ema",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        compare.main()
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_compare_arm_replacement_propagates_all_outer_overrides_to_syncer():
    arm = compare.apply_arm_overrides(
        [compare.PRESETS["m4"]],
        outer_lr=0.175,
        outer_momentum=0.5,
        delta_correction="none",
    )[0]

    assert arm.m == 4
    assert arm.fragments == compare.PRESETS["m4"].fragments
    cmd = compare.syncer_command(arm, 1234, Path("/tmp/w/m4"), total_steps=40)
    assert cmd[cmd.index("--outer-lr") + 1] == "0.175"
    assert cmd[cmd.index("--outer-momentum") + 1] == "0.5"
    assert cmd[cmd.index("--delta-correction") + 1] == "none"


def test_syncer_command_supports_fragment_outer_learning_rates():
    from dataclasses import replace

    tuned = replace(
        compare.PRESETS["m4"],
        outer_lr=0.175,
        outer_lr_by_fragment="0.2625,0.175,0.0875,0.14",
    )
    cmd = compare.syncer_command(tuned, 1234, Path("/tmp/w/m4"), total_steps=100)
    index = cmd.index("--outer-lr-by-fragment")
    assert cmd[index + 1] == "0.2625,0.175,0.0875,0.14"


def test_syncer_command_supports_strict_quorum():
    from dataclasses import replace

    tuned = replace(compare.PRESETS["m4"], strict_quorum=True)
    cmd = compare.syncer_command(tuned, 1234, Path("/tmp/w/m4"), total_steps=80)
    assert "--strict-quorum" in cmd


def test_syncer_command_supports_delta_norm_ref():
    arm = compare.PRESETS["m4"]
    # Default off: the flag must not appear, keeping production command
    # lines byte-identical.
    cmd = compare.syncer_command(arm, 1234, Path("/tmp/w/m4"), total_steps=80)
    assert "--delta-norm-ref" not in cmd
    cmd = compare.syncer_command(
        arm, 1234, Path("/tmp/w/m4"), total_steps=80, delta_norm_ref=0.0
    )
    assert "--delta-norm-ref" not in cmd
    # Active: forwarded verbatim to the syncer.
    cmd = compare.syncer_command(
        arm, 1234, Path("/tmp/w/m4"), total_steps=80, delta_norm_ref=2.869
    )
    assert cmd[cmd.index("--delta-norm-ref") + 1] == "2.869"


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


def test_compare_gpu_assignment_respects_offset_and_slots():
    args = SimpleNamespace(
        device="cuda", learner_gpus=2, gpu_slots=0, gpu_offset=4, settings="m2,m4"
    )
    assert compare.assigned_gpu_ids(args) == [4, 5, 6, 7, 8, 9, 10, 11]
    assert compare.learner_env(args, 1)["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert compare.eval_env(args)["CUDA_VISIBLE_DEVICES"] == "4,5,6,7,8,9,10,11"

    args = SimpleNamespace(
        device="cuda", learner_gpus=0, gpu_slots=3, gpu_offset=5, settings="m2"
    )
    assert compare.assigned_gpu_ids(args) == [5, 6, 7]
    assert compare.learner_env(args, 4)["CUDA_VISIBLE_DEVICES"] == "6"
    assert compare.eval_env(args)["CUDA_VISIBLE_DEVICES"] == "5,6,7"


def test_compare_gpu_wait_filters_to_assigned_gpus(monkeypatch, capsys):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    compute_queries = {"n": 0}

    def fake_run(cmd, **kwargs):
        if "--query-gpu=index,uuid" in cmd:
            return Result("0, GPU-a\n1, GPU-b\n")
        if "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in cmd:
            compute_queries["n"] += 1
            if compute_queries["n"] == 1:
                return Result("GPU-a, 111, python, 9000\nGPU-b, 222, python, 9000\n")
            return Result("")
        raise AssertionError(cmd)

    monkeypatch.setattr(compare.subprocess, "run", fake_run)
    monkeypatch.setattr(compare.time, "sleep", lambda _: None)

    compare.wait_for_free_gpus("cuda", timeout_s=1, gpu_ids=[1])

    out = capsys.readouterr().out
    assert "222" in out
    assert "111" not in out


def test_learner_command_arm_overrides():
    args = SimpleNamespace(
        model="lfm25-230m", lora_r=16, lora_alpha=32, seq_len=512,
        micro_batch_size=2, inner_lr=3e-4, device="cpu",
        shard="ddp", learner_gpus=0, fixed_window_tokens=8192,
        fixed_window_microsteps=64, pad_to_fixed_window_tokens=True,
        freeze_delta_before_delay=True, learner_push_delay_ms="0,50",
        learner_step_sleep_ms="0", learner_delay_jitter_ms=20.0,
        training_seed=123, wan_streams=0,
    )
    arm = compare.PRESETS["q4"]
    cmd = compare.learner_command(
        args, Path("/tmp/w/q4"), learner_id=1, num_learners=2,
        syncer="127.0.0.1:1", max_steps=10, arm=arm,
    )
    assert cmd[cmd.index("--wire-dtype") + 1] == "q4"
    assert cmd[cmd.index("--seed") + 1] == "123"
    assert cmd[cmd.index("--data") + 1].endswith("train.jsonl")
    assert cmd[cmd.index("--wan-streams") + 1] == "0"
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


def test_learner_command_forwards_fixed_window_schedule():
    args = SimpleNamespace(
        model="lfm25-230m", lora_r=16, lora_alpha=32, seq_len=128,
        micro_batch_size=1, inner_lr=3e-4, device="cpu",
        shard="ddp", learner_gpus=0, tuning="lora",
        fixed_window_schedule="0:16,160:256,170:16,330:256",
        training_seed=223223,
    )
    arm = compare.PRESETS["m4"]
    cmd = compare.learner_command(
        args, Path("/tmp/w/m4"), learner_id=0, num_learners=4,
        syncer="127.0.0.1:1", max_steps=10, arm=arm,
    )
    assert (
        cmd[cmd.index("--fixed-window-schedule") + 1] == "0:16,160:256,170:16,330:256"
    )
    base = compare.learner_command(
        args, Path("/tmp/w/baseline"), learner_id=0, num_learners=1,
        syncer="none", max_steps=10, arm=None,
    )
    assert "--fixed-window-schedule" not in base


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


def test_ensure_syncer_rebuilds_when_rust_source_is_newer(
    tmp_path, monkeypatch, capsys
):
    syncer_dir = tmp_path / "syncer"
    source_dir = syncer_dir / "src"
    binary = syncer_dir / "target" / "release" / "yeto-syncer"
    source_dir.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    inputs = [
        syncer_dir / "Cargo.toml",
        syncer_dir / "Cargo.lock",
        source_dir / "main.rs",
    ]
    for path in [*inputs, binary]:
        path.write_text("test")

    old_ns = 1_000_000_000
    binary_ns = 2_000_000_000
    for path in inputs:
        os.utime(path, ns=(old_ns, old_ns))
    os.utime(binary, ns=(binary_ns, binary_ns))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

    monkeypatch.setattr(compare, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(compare, "SYNCER_BIN", binary)
    monkeypatch.setattr(compare.subprocess, "run", fake_run)

    compare.ensure_syncer()
    assert calls == []
    assert capsys.readouterr().out == ""

    stale_ns = 3_000_000_000
    os.utime(inputs[-1], ns=(stale_ns, stale_ns))
    compare.ensure_syncer()

    assert calls == [
        (
            ["cargo", "build", "--release", "-q"],
            {"cwd": syncer_dir, "check": True},
        )
    ]
    assert capsys.readouterr().out == (
        "[compare] building syncer (cargo build --release)\n"
    )


def test_compare_eval_only_does_not_build_syncer(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diloco.py",
            "--data",
            "unused.jsonl",
            "--eval-only",
        ],
    )
    monkeypatch.setattr(compare, "eval_loss_per_token", lambda *args, **kwargs: 1.25)

    def unexpected_syncer_build():
        raise AssertionError("unexpected syncer build")

    monkeypatch.setattr(compare, "ensure_syncer", unexpected_syncer_build)

    assert compare.main() == 0
    assert capsys.readouterr().out == "EVAL_LOSS 1.25\n"


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


def test_exp27_feature_and_policy_grid_helpers_exclude_oracles():
    candidates = [
        {
            "learner_id": 0,
            "probe_grad_dot": 1.0,
            "calibrated_score": 0.9,
            "utility": 0.02,
            "bad": False,
            "bad_strict": False,
            "weight": 1.0,
        },
        {
            "learner_id": 1,
            "probe_grad_dot": -1.0,
            "calibrated_score": 0.2,
            "utility": -0.01,
            "bad": True,
            "bad_strict": True,
            "weight": 1.0,
        },
    ]
    stats = build_group_features._field_stats(candidates, "probe_grad_dot")
    assert stats["top1_learner"] == 0
    assert stats["bottom1_learner"] == 1
    assert stats["score_entropy"] < 1.0

    feature_row = {
        "seed": 1,
        "scores": {"probe_grad_dot": stats},
        "agreement": {},
        "actions": {
            "token_weighted": {
                "utility": 0.0,
                "negative": False,
                "strict_negative": False,
                "selected_mass": 1.0,
                "selected_count": 2,
            },
            "anchor_drop_bottom25": {
                "utility": 0.01,
                "negative": False,
                "strict_negative": False,
                "selected_mass": 0.5,
                "selected_count": 1,
            },
            "oracle_positive": {
                "utility": 10.0,
                "negative": False,
                "strict_negative": False,
                "selected_mass": 0.5,
                "selected_count": 1,
            },
        },
    }
    # Fill deployable action aliases required by baseline grid evaluation.
    for action in policy_grid.DEPLOYABLE_ACTIONS:
        feature_row["actions"].setdefault(action, feature_row["actions"]["token_weighted"])
    rules = policy_grid.candidate_rules([feature_row], ["probe_grad_dot"])
    assert all("oracle" not in rule.name and "random" not in rule.name for rule in rules)
    drop = policy_grid.Rule(
        "drop",
        "drop25_if_spread",
        "probe_grad_dot",
        {"spread": 0.1},
    )
    assert policy_grid.decide(drop, feature_row) == "anchor_drop_bottom25"


def test_hard_search_deployable_oracle_excludes_oracles():
    row = {
        "seed": 1,
        "step": 1,
        "fragment": 0,
        "candidate_count": 2,
        "scores": {},
        "agreement": {},
        "actions": {},
    }
    for action in hard_search.DEPLOYABLE_ACTIONS:
        row["actions"][action] = {
            "utility": 0.0,
            "negative": False,
            "strict_negative": False,
            "selected_mass": 1.0,
            "selected_count": 2,
        }
    row["actions"]["anchor_drop_bottom25"]["utility"] = 0.1
    row["actions"]["oracle_positive"] = {
        "utility": 10.0,
        "negative": False,
        "strict_negative": False,
        "selected_mass": 0.5,
        "selected_count": 1,
    }
    row["actions"]["oracle_topk"] = dict(row["actions"]["oracle_positive"])
    choices = hard_search.deployable_oracle_choices([row])
    assert choices == ["anchor_drop_bottom25"]
    assert all("oracle" not in action for action in hard_search.DEPLOYABLE_ACTIONS)


def test_action_probe_deployable_selection_excludes_oracles():
    record = {
        "token_weighted_anchor_utility": 0.0,
        "anchor_drop_bottom25_anchor_utility": 0.1,
        "oracle_positive_anchor_utility": 100.0,
    }
    action, utility, margin = action_probe.selected_action_by_anchor(
        record, ("token_weighted", "anchor_drop_bottom25")
    )
    assert action == "anchor_drop_bottom25"
    assert utility == 0.1
    assert margin == 0.1
    assert all("oracle" not in action for action in action_probe.DEFAULT_DEPLOYABLE_ACTIONS)


def test_action_probe_manifest_overlap_is_rejected():
    import pytest

    with pytest.raises(SystemExit):
        action_probe.validate_manifest({"overlap_count": 1}, require_disjoint=True)
    action_probe.validate_manifest({"overlap_count": 0}, require_disjoint=True)


def test_action_probe_margin_gated_fallback():
    record = {
        "token_weighted_anchor_utility": 0.10,
        "anchor_drop_bottom25_anchor_utility": 0.1002,
    }
    assert (
        action_probe.margin_gated_choice(
            record, ("token_weighted", "anchor_drop_bottom25"), margin=0.001
        )
        == "token_weighted"
    )
    assert (
        action_probe.margin_gated_choice(
            record, ("token_weighted", "anchor_drop_bottom25"), margin=0.0001
        )
        == "anchor_drop_bottom25"
    )


def test_action_probe_aggregate_rejects_missing_seed():
    import pytest

    summary = {
        "records": 1,
        "seeds": [53],
        "policies": {
            "action_probe_top1": {
                "mean_gain_vs_token": 0.1,
                "negative_rate_relative_drop": 0.1,
                "strict_negative_rate_relative_drop": 0.1,
                "oracle_positive_headroom_captured": 0.1,
                "selected_mass_mean": 1.0,
                "chosen_action_distribution": {"anchor_drop_bottom25": 1},
            },
            "action_probe_margin_gated": {
                "mean_gain_vs_token": 0.1,
                "negative_rate_relative_drop": 0.1,
                "strict_negative_rate_relative_drop": 0.1,
                "oracle_positive_headroom_captured": 0.1,
                "selected_mass_mean": 1.0,
                "chosen_action_distribution": {"anchor_drop_bottom25": 1},
            },
            "action_probe_risk_aware": {
                "mean_gain_vs_token": 0.1,
                "negative_rate_relative_drop": 0.1,
                "strict_negative_rate_relative_drop": 0.1,
                "oracle_positive_headroom_captured": 0.1,
                "selected_mass_mean": 1.0,
                "chosen_action_distribution": {"anchor_drop_bottom25": 1},
            },
        },
    }
    with pytest.raises(SystemExit):
        action_probe_agg.aggregate([summary], expected_seeds=[53, 67])


def test_action_probe_stability_uses_anchor_batch_utilities():
    actions = ("token_weighted", "anchor_drop_bottom25")
    records = [
        {
            "token_weighted_anchor_batch_utilities": [0.0, 0.0, 0.0, 0.0],
            "anchor_drop_bottom25_anchor_batch_utilities": [0.01, 0.01, 0.01, 0.01],
            "token_weighted_oracle_utility": 0.0,
            "token_weighted_oracle_negative": False,
            "token_weighted_oracle_strict_negative": False,
            "token_weighted_selected_mass": 1.0,
            "anchor_drop_bottom25_oracle_utility": 0.02,
            "anchor_drop_bottom25_oracle_negative": False,
            "anchor_drop_bottom25_oracle_strict_negative": False,
            "anchor_drop_bottom25_selected_mass": 0.5,
            "oracle_positive_oracle_utility": 0.03,
        },
        {
            "token_weighted_anchor_batch_utilities": [0.02, 0.02, 0.02, 0.02],
            "anchor_drop_bottom25_anchor_batch_utilities": [0.0, 0.0, 0.0, 0.0],
            "token_weighted_oracle_utility": 0.01,
            "token_weighted_oracle_negative": False,
            "token_weighted_oracle_strict_negative": False,
            "token_weighted_selected_mass": 1.0,
            "anchor_drop_bottom25_oracle_utility": -0.01,
            "anchor_drop_bottom25_oracle_negative": True,
            "anchor_drop_bottom25_oracle_strict_negative": True,
            "anchor_drop_bottom25_selected_mass": 0.5,
            "oracle_positive_oracle_utility": 0.02,
        },
    ]
    result = action_stability.evaluate_subset(records, actions, (0, 1), margin=0.005)
    assert result["top1_action_agreement"] == 1.0
    assert result["pairwise_concordance"] == 1.0
    assert result["top1_policy"]["chosen_action_distribution"] == {
        "anchor_drop_bottom25": 1,
        "token_weighted": 1,
    }
    assert result["top1_policy"]["mean_gain_vs_token"] == 0.01


def test_buffered_replay_policy_subset_is_validated():
    base = [
        "--capture-dir",
        "/tmp/capture",
        "--model",
        "qwen35-9b",
        "--data",
        "/tmp/eval.jsonl",
        "--out-jsonl",
        "/tmp/out.jsonl",
        "--out-summary",
        "/tmp/out.json",
    ]
    args = buffered_robust.parse_args(
        base
        + [
            "--policies",
            "buffer_coord_median,buffer_coord_median_blend25,buffer_coord_median",
        ]
    )
    assert args.policies == ("buffer_coord_median", "buffer_coord_median_blend25")

    import pytest

    with pytest.raises(SystemExit):
        buffered_robust.parse_args(base + ["--policies", "oracle_positive"])

    tuned = buffered_nesterov.parse_args(
        base + ["--policies", "current_outer_lr100,current_outer_lr150"]
    )
    assert tuned.policies == ("current_outer_lr100", "current_outer_lr150")


def test_buffered_nesterov_matches_syncer_equation():
    import torch

    current = torch.tensor([1.0])
    momentum = torch.tensor([0.2])
    merged_update = torch.tensor([-0.5])
    trial = buffered_nesterov._nesterov_trial(
        current, momentum, merged_update, outer_lr=0.7, outer_momentum=0.9
    )
    delta = -merged_update
    next_momentum = 0.9 * momentum + delta
    expected = current - 0.7 * (delta + 0.9 * next_momentum)
    assert torch.allclose(trial, expected)


def test_buffered_transport_caps_history_without_dropping_candidates():
    import torch

    updates = [torch.tensor([1.0]), torch.tensor([1.2]), torch.tensor([-4.0]), torch.tensor([-3.0])]
    merged, stats = buffered_nesterov._transport_slice(
        updates,
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 4.0, 4.0],
        torch.tensor([-1.0]),
        tau=4.0,
        history_cap=0.30,
        mode="mean",
    )
    assert stats["history_effective_share"] <= 0.300001
    assert stats["normalized_effective_sample_size"] > 0.5
    assert merged.item() >= 1.1 - 1e-6  # guard cannot trail the fresh mean on -momentum


def test_coordinate_midpoint_median_is_sign_equivariant():
    import torch

    values = [torch.tensor([float(value)]) for value in range(0, 16, 2)]
    median = buffered_nesterov._coordinate_midpoint_median(values)
    negated = buffered_nesterov._coordinate_midpoint_median([-value for value in values])
    assert median.item() == 7.0
    assert torch.equal(negated, -median)


def test_buffered_group_ema_keeps_current_round_dominant():
    import torch
    from yeto.fragments import MERGE_AVG

    def candidate(value, age):
        update = torch.tensor([float(value)])
        return buffered_robust.soft.Candidate(
            row={},
            tensor=update,
            update=update,
            weight=1.0,
            age=float(age),
            norm=abs(float(value)),
            align=0.0,
        )

    frag = SimpleNamespace(tensors=[("x", 1)], merge_mode=MERGE_AVG)
    merged, info = buffered_nesterov._group_policy(
        "buffer_group_ema25",
        [[candidate(1.0, 4)], [candidate(3.0, 0)]],
        torch.zeros(1),
        frag,
    )
    assert torch.allclose(merged, torch.tensor([2.5]))
    assert info["fresh_effective_share"] == 0.75
    assert info["history_effective_share"] == 0.25


def test_buffered_group_transport_supports_smaller_history_share():
    import math
    import torch
    from yeto.fragments import MERGE_AVG

    def candidate(value, age):
        update = torch.tensor([float(value)])
        return buffered_robust.soft.Candidate(
            row={},
            tensor=update,
            update=update,
            weight=1.0,
            age=float(age),
            norm=abs(float(value)),
            align=0.0,
        )

    frag = SimpleNamespace(tensors=[("x", 1)], merge_mode=MERGE_AVG)
    merged, info = buffered_nesterov._group_policy(
        "buffer_group_transport10",
        [[candidate(1.0, 4)], [candidate(3.0, 0)]],
        torch.zeros(1),
        frag,
    )
    assert torch.allclose(merged, torch.tensor([2.8]))
    assert info["fresh_effective_share"] == 0.9
    assert math.isclose(info["history_effective_share"], 0.1)


def test_buffered_group_normmatch_preserves_current_tensor_norm():
    import torch
    from yeto.fragments import MERGE_AVG

    def candidate(update, age):
        return buffered_robust.soft.Candidate(
            row={},
            tensor=update,
            update=update,
            weight=1.0,
            age=float(age),
            norm=float(update.norm().item()),
            align=0.0,
        )

    current = torch.tensor([3.0, 0.0])
    frag = SimpleNamespace(tensors=[("x", 2)], merge_mode=MERGE_AVG)
    merged, _ = buffered_nesterov._group_policy(
        "buffer_group_transport25_normmatch",
        [[candidate(torch.tensor([0.0, 2.0]), 4)], [candidate(current, 0)]],
        torch.zeros(2),
        frag,
    )
    assert torch.allclose(merged.norm(), current.norm())
    assert not torch.allclose(merged, current)


def test_consensus_rda_retains_disagreement_magnitude():
    import math
    import torch
    from yeto.fragments import MERGE_RDA

    def candidate(update):
        return buffered_robust.soft.Candidate(
            row={},
            tensor=update,
            update=update,
            weight=1.0,
            age=0.0,
            norm=float(update.norm().item()),
            align=0.0,
        )

    candidates = [candidate(torch.tensor([1.0, 0.0])), candidate(torch.tensor([0.0, 1.0]))]
    frag = SimpleNamespace(tensors=[("x", 2)], merge_mode=MERGE_RDA)
    linear, scale = buffered_nesterov._consensus_rda_update(
        candidates, torch.zeros(2), frag, "linear"
    )
    baseline = buffered_nesterov._production_merge_update(candidates, torch.zeros(2), frag)
    assert math.isclose(scale, math.sqrt(0.5), rel_tol=1e-6)
    assert linear.norm() < baseline.norm()
    assert torch.allclose(linear, torch.tensor([0.5, 0.5]), atol=1e-6)


def test_buffered_nesterov_aggregate_merges_policy_partitions(tmp_path):
    shared = {"seed": 59, "step": 5, "fragment": 0, "token_weighted_utility": 0.1}
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text(json.dumps({**shared, "policy_a_utility": 0.2}) + "\n")
    right.write_text(json.dumps({**shared, "policy_b_utility": 0.3}) + "\n")
    rows = buffered_nesterov_agg.merge_records([left, right])
    assert rows == [{**shared, "policy_a_utility": 0.2, "policy_b_utility": 0.3}]


def test_lr_action_probe_margin_falls_back_to_fixed_lr():
    actions = ("lr40", "lr50", "lr60")
    utility = {"lr40": 0.11, "lr50": 0.10, "lr60": 0.09}
    se = {"lr40": 0.02, "lr50": 0.02, "lr60": 0.02}
    assert lr_action_probe._choice(utility, se, actions, "lr50", 0.0) == "lr40"
    assert lr_action_probe._choice(utility, se, actions, "lr50", 1.0) == "lr50"


def test_anchor_gradient_tensor_norm_matching_is_per_tensor():
    import torch

    frag = SimpleNamespace(tensors=[("left", 2), ("right", 2)])
    source = torch.tensor([3.0, 4.0, 0.0, 2.0])
    target = torch.tensor([6.0, 8.0, 5.0, 12.0])
    matched = anchor_gradient_syncer._tensor_normmatch(source, target, frag)
    assert torch.allclose(matched[:2].norm(), target[:2].norm())
    assert torch.allclose(matched[2:].norm(), target[2:].norm())
    assert torch.allclose(matched[:2] / matched[:2].norm(), source[:2] / source[:2].norm())


def test_anchor_gradient_pcgrad_removes_tensor_conflict():
    import torch

    frag = SimpleNamespace(tensors=[("only", 2)])
    baseline = torch.tensor([1.0, 0.0])
    anchor = torch.tensor([-1.0, 1.0])
    corrected, conflict_fraction = anchor_gradient_syncer._pcgrad(baseline, anchor, frag)
    assert conflict_fraction == 1.0
    assert torch.dot(corrected, anchor).item() >= -1e-6
    assert torch.allclose(corrected.norm(), baseline.norm())


def test_anchor_gradient_manifest_overlap_is_rejected():
    import pytest

    with pytest.raises(SystemExit, match="content overlap"):
        anchor_gradient_syncer._validate_manifest({"overlap_count": 1})
    anchor_gradient_syncer._validate_manifest({"overlap_count": 0})


def test_anchor_gradient_policy_subset_is_validated():
    base = [
        "--capture-dir",
        "/tmp/capture",
        "--model",
        "qwen35-9b",
        "--anchor-data",
        "/tmp/anchor.jsonl",
        "--oracle-data",
        "/tmp/oracle.jsonl",
        "--split-manifest-out",
        "/tmp/split.json",
        "--out-jsonl",
        "/tmp/out.jsonl",
        "--out-summary",
        "/tmp/out.json",
    ]
    args = anchor_gradient_syncer.parse_args(
        base + ["--policies", "anchor_blend05,anchor_pcgrad_normmatch,anchor_blend05"]
    )
    assert args.policies == ("anchor_blend05", "anchor_pcgrad_normmatch")

    import pytest

    with pytest.raises(SystemExit):
        anchor_gradient_syncer.parse_args(base + ["--policies", "token_weighted"])


def test_anchor_gradient_aggregate_merges_policy_partitions(tmp_path):
    shared = {
        "schema": "anchor_gradient_syncer_replay_v1",
        "seed": 59,
        "step": 5,
        "fragment": 0,
        "token_weighted_utility": 0.1,
        "token_weighted_gain_vs_token": 0.0,
    }
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text(json.dumps({**shared, "anchor_blend05_utility": 0.2}) + "\n")
    right.write_text(json.dumps({**shared, "anchor_pcgrad_normmatch_utility": 0.3}) + "\n")
    rows = anchor_gradient_agg.merge_records([left, right])
    assert rows == [
        {
            **shared,
            "anchor_blend05_utility": 0.2,
            "anchor_pcgrad_normmatch_utility": 0.3,
        }
    ]


def test_fragment_lr_profile_selects_action_by_fragment():
    import math

    rows = []
    for fragment in range(4):
        row = {
            "seed": 59,
            "step": fragment + 1,
            "fragment": fragment,
            "token_weighted_utility": 0.0,
        }
        for action in fragment_lr_profiles.ACTIONS:
            utility = float(action) / 1000.0
            row[f"current_outer_lr{action}_utility"] = utility
            row[f"current_outer_lr{action}_negative"] = False
            row[f"current_outer_lr{action}_strict_negative"] = False
        rows.append(row)
    result = fragment_lr_profiles.evaluate_profile(rows, (75, 50, 25, 40))
    expected = (0.075 + 0.050 + 0.025 + 0.040) / 4
    baseline = 0.050
    assert math.isclose(result["mean_utility"], expected)
    assert math.isclose(result["mean_gain_vs_fixed_lr"], expected - baseline)


def test_fragment_lr_profile_accepts_custom_action_grid(tmp_path):
    args = fragment_lr_profiles.parse_args(
        [
            "--replays",
            str(tmp_path / "replay.jsonl"),
            "--actions",
            "50,80,100,125,150",
            "--baseline-action",
            "100",
            "--no-default-profiles",
            "--profile",
            "soft=125,100,50,80",
            "--out-json",
            str(tmp_path / "out.json"),
            "--out-md",
            str(tmp_path / "out.md"),
        ]
    )
    assert args.actions == (50, 80, 100, 125, 150)
    assert args.profiles == {"soft": (125, 100, 50, 80)}
