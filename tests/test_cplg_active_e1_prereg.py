"""Frozen scientific-contract tests for the CPLG active E1 preregistration."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments" / "optimizer"
CONFIG = EXPERIMENTS / "cplg-sgd-active-e1-r1-config.json"
DOSSIER = EXPERIMENTS / "cplg-sgd-active-e1-r1-prereg.md"
R2_CONFIG = EXPERIMENTS / "cplg-sgd-shadow-direction-r2-config.json"
R2_SPEC = EXPERIMENTS / "exp2-cplg-shadow-direction-r2.json"
CONFIG_SHA256 = "5afe2d4900051fda1ac99cc682c489dfeae85f0eb34d1816646b5bff5f0c26df"
VERDICTS = [
    "PASS",
    "FAIL",
    "INCONCLUSIVE",
    "UNIDENTIFIABLE",
    "INFRA_FAILURE",
]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        assert key not in value, f"duplicate JSON key {key!r}"
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise AssertionError(f"non-finite JSON constant {value}")


def _load(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(
        path.read_text(),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    return value


def test_config_checksum_closed_schema_and_deferred_cloud_binding() -> None:
    raw = CONFIG.read_bytes()
    config = _load()

    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    assert Path(f"{CONFIG}.sha256").read_text() == (f"{CONFIG_SHA256}  {CONFIG.name}\n")
    assert set(config) == {
        "schema_version",
        "schema",
        "run_id",
        "status",
        "candidate",
        "prerequisite_evidence",
        "source",
        "resource_envelope",
        "mechanism",
        "workload",
        "measurement",
        "gates",
        "artifacts",
        "lifecycle",
        "claim_boundary",
    }
    assert config["schema_version"] == 1
    assert config["schema"] == "cplg_sgd_active_e1_scientific_config_v1"
    assert config["run_id"] == "exp2-cplg-active-e1-m1-r1"
    assert config["status"] == (
        "frozen_before_active_e1_outcome_nonlaunchable_until_immutable_cloud_spec"
    )
    assert config["candidate"]["dossier"] == DOSSIER.relative_to(ROOT).as_posix()
    assert config["source"]["cloud_acquisition_spec"] is None
    assert "unknown until" in config["source"]["active_e1_commit_binding"]
    assert config["source"]["required_future_pair_wrapper"] == (
        "scripts/run_cplg_online_acquisition.py"
    )
    assert config["source"]["required_future_cpu_validator"] == (
        "scripts/validate_cplg_online_pair.py"
    )
    assert CONFIG_SHA256 in DOSSIER.read_text()


def test_r2_pass_is_exact_prerequisite_but_never_an_e1_input() -> None:
    config = _load()
    prerequisite = config["prerequisite_evidence"]

    assert prerequisite["required_run_id"] == "exp2-cplg-shadow-direction-r2"
    assert prerequisite["required_verdict"] == "PASS"
    assert prerequisite["required_terminal_stage"] == "completed_analysis"
    assert prerequisite["artifact_prefix"].endswith("/exp2-cplg-shadow-direction-r2")
    assert prerequisite["implementation_commit"] == (
        "eb6d21146011112ffe8df5cb518c985e8c0297bd"
    )
    assert (
        prerequisite["cloud_spec_sha256"]
        == hashlib.sha256(R2_SPEC.read_bytes()).hexdigest()
    )
    assert (
        prerequisite["scientific_config_sha256"]
        == hashlib.sha256(R2_CONFIG.read_bytes()).hexdigest()
    )
    assert prerequisite["analysis_sha256"] == (
        "a1143e818580c3b6b463bb180a04fb20991b04498f5fcd4820da2c6a52ada4fc"
    )
    assert prerequisite["final_manifest_sha256"] == (
        "cced40ef7bb2890b0ae92755070a259d8099abb3a4200d658f93e9b4db5b085f"
    )
    assert prerequisite["stock_tape_sha256"] == (
        "1e939222005f78d9d9711c4558d59b922e566af513485ce231d11ce417838c31"
    )
    assert prerequisite["stock_tape_manifest_sha256"] == (
        "1a0d19a27b99c82a0110a3e186269ebd2d76396b5163a625086862aacabac3fd"
    )
    assert prerequisite["simulated_nonstock_actions"] == 12
    assert prerequisite["positive_fragment_means"] == 4
    assert prerequisite["mean_shadow_cosine_gain"] == 0.0028530240058898928
    assert prerequisite["one_sided_bootstrap_lower_endpoint"] == (0.0027203083038330076)
    assert prerequisite["matched_capture_overhead_fraction"] == (0.009156851278518476)
    assert "may not be inputs" in prerequisite["forbidden_use"]


def test_candidate_mechanics_and_exact_fallback_contract_are_closed() -> None:
    mechanism = _load()["mechanism"]

    assert mechanism["outer_optimizer_name"] == "cplg-sgd"
    assert mechanism["angle_cap_f32_bits"] == "0x3e7adbb0"
    assert mechanism["arithmetic"].endswith("no FMA")
    assert mechanism["authoritative_transcendentals"].startswith("Rust libm 0.2.15")
    assert mechanism["interlock"] == (
        "select the current nonstock candidate only when the newest three "
        "already-resolved same-fragment f32 scores are all strictly positive"
    )
    assert mechanism["reason_vocabulary"] == [
        "not_active",
        "stock_warmup",
        "phase_warmup",
        "interlock_closed",
        "candidate_selected",
        "degenerate_stock",
        "nonacute_turn",
        "invalid_geometry",
        "invalid_shadow_score",
        "zero_or_rounded_phase",
    ]
    action = mechanism["action_contract"]
    assert "action SHA-256 equals candidate SHA-256" in action["candidate_selected"]
    assert "action SHA-256 exactly equal to stock SHA-256" in action["all_fallbacks"]
    assert "original stock byte object" in action["zero_or_rounded_phase"]
    assert "clears the complete affected-fragment causal state" in action["clearing"]
    assert mechanism["preview_commit_contract"] == (
        "preview is pure and commit installs exactly one preview once"
    )


def test_workload_is_truthful_fresh_sequential_r2_sized_active_pair() -> None:
    config = _load()
    workload = config["workload"]
    r2 = _load(R2_CONFIG)["workload"]

    assert workload["arms_in_order"] == ["cplg_m1_stock", "cplg_m1_candidate"]
    assert workload["outer_optimizer_by_arm"] == {
        "cplg_m1_stock": "nesterov",
        "cplg_m1_candidate": "cplg-sgd",
    }
    assert workload["result_rows_in_order"] == [
        "base (untrained)",
        "cplg_m1_stock",
        "cplg_m1_candidate",
    ]
    assert workload["execution_order"] == "sequential stock then candidate"
    assert "same separately restored initial" in workload["freshness"]
    assert workload["sequence_length"] == r2["sequence_length"] == 128
    assert workload["micro_batch_size"] == r2["micro_batch_size"] == 1
    assert (
        workload["raw_local_training_tokens_per_arm"]
        == (r2["raw_local_training_tokens"])
        == 4_352
    )
    assert (
        workload["compare_token_budget_per_arm"]
        == (r2["compare_token_budget"])
        == 4_352
    )
    assert (
        workload["expected_terminal_local_steps_per_arm"]
        == (r2["expected_terminal_local_steps"])
        == 34
    )
    assert 34 * 128 == workload["raw_local_training_tokens_per_arm"]
    assert workload["outer_commits_per_arm"] == r2["outer_commits"] == 32
    assert workload["commits_per_fragment_per_arm"] == 8
    assert workload["fragment_order"] == r2["fragment_order"]
    assert Counter(workload["fragment_order"]) == Counter({0: 8, 1: 8, 2: 8, 3: 8})
    for key in (
        "learner_max_steps_liveness_cap",
        "learners",
        "fragments",
        "quorum",
        "fixed_window_microsteps",
        "wire_dtype",
        "merge_alpha",
        "matrix_merge",
        "delta_correction",
        "outer_learning_rate",
        "outer_momentum",
        "inner_optimizer",
        "inner_learning_rate",
        "adamw_betas",
        "adamw_epsilon",
        "weight_decay",
        "warmup_steps",
        "scheduler",
        "gradient_clip_norm",
        "gradient_checkpointing",
        "loss_function",
        "train_on",
        "tuning",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "shard",
        "grad_accumulation",
        "tokenization",
        "shuffle_rows_seed",
        "training_seed",
        "max_rows",
        "evaluation_rows",
        "device",
        "gpu_slots",
        "arm_timeout_minutes",
        "strict_quorum",
        "barrier_sync",
        "deterministic_commit_order",
    ):
        assert workload[key] == r2[key]
    assert workload["outer_learning_rate_f32_bits"] == "0x3e8f5c29"
    assert workload["outer_momentum_f32_bits"] == "0x00000000"
    assert workload["skip_baseline_training"] is True


def test_activity_loss_and_unrounded_overhead_gates_are_exact() -> None:
    config = _load()
    measurement = config["measurement"]
    gates = config["gates"]

    assert "all 32 predefined" in measurement["activity"]["denominator"]
    assert "no row is removed" in measurement["activity"]["denominator"]
    assert gates["minimum_nonstock_actions"] == 8
    assert gates["minimum_fragments_with_nonstock_action"] == 3
    assert gates["maximum_candidate_minus_stock_eval_loss"] == 0.05
    assert gates["maximum_matched_interval_overhead_fraction"] == 0.02
    assert gates["maximum_nonfinite_losses"] == 0
    assert gates["maximum_invalid_or_missing_boundaries"] == 0
    assert gates["maximum_action_hash_contract_violations"] == 0
    assert gates["maximum_writer_drops_or_pending_items"] == 0
    assert measurement["loss"]["primary_contrast"] == (
        "candidate_eval_loss - stock_eval_loss"
    )
    assert measurement["loss"]["source"].startswith("unrounded finite binary64")
    overhead = measurement["overhead"]
    assert overhead["clock"] == "unrounded producer monotonic integer nanoseconds"
    assert overhead["formula"] == (
        "(candidate_interval_ns - stock_interval_ns) / stock_interval_ns"
    )
    assert overhead["negative_values"] == "retained and not clamped"
    assert overhead["computed_by"].startswith("separate CPU analysis")


def test_gpu_acquisition_and_cpu_analysis_are_distinct_lifecycles() -> None:
    config = _load()
    resources = config["resource_envelope"]
    artifacts = config["artifacts"]
    lifecycle = config["lifecycle"]

    gpu = resources["gpu_acquisition"]
    assert gpu["machine_type"] == "a2-highgpu-1g"
    assert gpu["accelerator_count"] == 1
    assert gpu["maximum_total_accelerators"] == 1
    assert gpu["provisioning_model"] == "SPOT"
    assert gpu["termination_action"] == "DELETE"
    assert gpu["maximum_run_duration_seconds"] == 3_600
    assert gpu["boot_disk_size_gb"] == 250
    assert gpu["boot_disk_type"] == "pd-ssd"
    assert gpu["expected_source_image_id"] == "7290368630472593484"
    assert resources["cpu_analysis"]["accelerator_count"] == 0
    assert resources["cpu_analysis"]["may_use_gpu_acquisition_vm"] is False
    assert resources["cpu_analysis"]["may_use_protected_unrelated_vm"] is False

    assert (
        artifacts["planned_gpu_acquisition_prefix"]
        != (artifacts["planned_cpu_analysis_prefix"])
    )
    assert "never modify" in artifacts["prefix_rule"]
    assert artifacts["forbidden_on_gpu_acquisition_vm"] == [
        "final gate arithmetic",
        "full causal replay",
        "bootstrap or resampling",
        "final scientific verdict publication",
    ]
    assert any(
        "scientific_verdict null" in item
        for item in artifacts["gpu_acquisition_required"]
    )
    assert lifecycle["gpu_acquisition_completion_state"] == ("GPU_ACQUISITION_COMPLETE")
    assert lifecycle["gpu_acquisition_completion_is_scientific_verdict"] is False
    assert lifecycle["terminal_scientific_verdicts"] == VERDICTS
    assert "round-trip verify" in lifecycle["gpu_success"]
    assert "exact-ID delete" in lifecycle["gpu_success"]
    assert "exact-ID abandon" in lifecycle["gpu_incomplete"]
    assert (
        "never retain, resume, or relaunch the A100"
        in lifecycle["cpu_analysis_failure"]
    )


def test_retry_rules_and_claim_limits_cannot_promote_short_e1() -> None:
    config = _load()
    lifecycle = config["lifecycle"]
    claims = config["claim_boundary"]

    assert lifecycle["terminal_scientific_verdicts"] == VERDICTS
    assert lifecycle["pass"].startswith("no retry")
    assert lifecycle["fail"].startswith("no retry")
    assert lifecycle["inconclusive"].startswith("no automatic retry")
    assert lifecycle["unidentifiable"].startswith("no retry")
    assert lifecycle["infra_failure"].startswith("at most one fresh-identity retry")
    assert "4,352-token" in claims["pass_claim"]
    assert "short matched online workload" in claims["pass_claim"]
    assert claims["forbidden_claims"] == [
        "CPLG-SGD beats SGD-0.28",
        "positive expected loss improvement",
        "same-state CRN finite-loss evidence",
        "32,768-token standard E1 completion",
        "convergence or unconditional dominance",
        "population, seed, model, H, learner-count, or inner-optimizer generalization",
        "authorization for E2 or production replacement",
    ]
    dossier = DOSSIER.read_text()
    for verdict in VERDICTS:
        assert f"`{verdict}`" in dossier
    assert "No GPU is authorized by this dossier alone" in dossier
    assert "not the SOP's 32,768-token standard E1 profile" in dossier
