from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_phase_map", ROOT / "scripts" / "validate_phase_map.py"
)
phase_map = importlib.util.module_from_spec(SPEC)
sys.modules["validate_phase_map"] = phase_map
assert SPEC.loader is not None
SPEC.loader.exec_module(phase_map)
ManifestError = phase_map.ManifestError
validate_and_summarize = phase_map.validate_and_summarize


HEX_A = "a" * 64
HEX_B = "b" * 64
GIT = "c" * 40


def _cell_id(h: int, mu: float, eta: float, seed: int) -> str:
    return f"h{h}-mu{mu:g}-eta{eta:g}-s{seed}"


def _manifest(
    *,
    seeds: tuple[int, ...] = (347,),
    etas: tuple[float, ...] = (0.021875, 0.04375, 0.0875),
    mode: str = "development",
) -> dict:
    seed_pairs = {str(seed): seed * 1000 + seed for seed in seeds}
    manifest = {
        "schema_version": "0.1",
        "status": "sealed_results",
        "study_id": "unit-phase-map-r0",
        "mode": mode,
        "min_confirmatory_seeds": 8,
        "expected_grid": {
            "h": [16],
            "mu": [0.0, 0.9],
            "eta": list(etas),
            "seeds": list(seeds),
        },
        "seed_pairs": seed_pairs,
        "frozen": {
            "git_commit": GIT,
            "image_id": "7290368630472593484",
            "image_digest": HEX_A,
            "model_id": "test/model",
            "model_revision": GIT,
            "model_hash": HEX_A,
            "data_hash": HEX_A,
            "train_rows_hash": HEX_A,
            "eval_hash": HEX_A,
            "eval_example_ids_hash": HEX_A,
            "eval_token_ids_hash": HEX_A,
            "command_hash": HEX_B,
            "randomization_plan_hash": HEX_B,
            "retry_policy_hash": HEX_A,
        },
        "protocol": {
            "matrix_merge": "rda",
            "strict_quorum": True,
            "barrier": True,
            "version_matched": True,
            "delta_correction": "none",
            "spot_only": True,
            "on_demand_fallback": False,
            "injected_baseline": False,
            "per_example_loss_required": True,
            "eval_rows": 1024,
            "token_budget": 655360,
            "seq_len": 128,
        },
        "horizon_work": {
            "16": {
                "fixed_window_microsteps": 16,
                "fixed_window_tokens": 2048,
                "outer_steps": 320,
            }
        },
        "adaptive_bracket": {"new_immutable_manifest_per_round": True},
        "randomization": {
            "required_mu_per_block": [0.0, 0.9],
            "block_fields": ["h", "eta", "seed"],
            "loss_blind": True,
        },
        "retry_policy": {
            "hash_definition": "sha256 canonical exact top-level retry_policy",
            "loss_blind_only": True,
            "rerun_entire_incomplete_block": True,
            "retain_all_attempts": True,
            "retry_lineage_required": True,
            "allowed_reasons": [
                "provider_spot_preemption",
                "peer_block_invalidated_by_infra_failure",
            ],
            "direct_infrastructure_failure_reasons": [
                "provider_spot_preemption"
            ],
            "peer_retry_reason": "peer_block_invalidated_by_infra_failure",
            "peer_retry_reason_is_never_failure_reason": True,
            "infra_failure_reason_must_be_direct_infrastructure_failure_reason": True,
            "preserve_completed_peer_status_and_artifacts": True,
            "retry_authorization_required_fields": [
                "loss_blind",
                "policy_hash",
                "trigger_attempt_id",
                "trigger_reason",
                "trigger_block_id",
                "prior_manifest_sha256",
            ],
            "retry_block_rows_must_be_contiguous": True,
            "result_acquisition_is_append_only": True,
            "trigger_must_be_genuine_infra_failure_in_immediately_prior_same_block": True,
            "shared_block_retry_authorization_required": True,
        },
        "analysis_policy": {"bracketing_tolerance": 0.0},
        "required_result_fields": [
            "cell_id",
            "h",
            "mu",
            "eta",
            "seed",
            "training_seed",
            "status",
            "failure_reason",
            "loss",
            "work",
            "git_commit",
            "image_digest",
            "model_hash",
            "data_hash",
            "eval_hash",
            "command_hash",
            "capture_uri",
            "capture_sha256",
            "per_example_loss_uri",
            "per_example_loss_sha256",
            "paired_control_id",
            "barrier",
            "version_matched",
            "matrix_merge",
            "strict_quorum",
            "delta_correction",
            "spot",
            "block_id",
            "order_index",
            "attempt",
            "retry_of",
            "retry_reason",
            "hardware",
            "started_at",
            "ended_at",
        ],
        "results": [],
    }
    loss_curve = {
        0.0: {etas[0]: 1.10, etas[1]: 1.00, etas[2]: 1.20},
        0.9: {etas[0]: 1.20, etas[1]: 1.05, etas[2]: 1.30},
    }
    for seed in seeds:
        for eta in etas:
            block_id = f"h16-eta{eta:g}-s{seed}"
            for order_index, mu in enumerate((0.0, 0.9)):
                cell_id = _cell_id(16, mu, eta, seed)
                control_id = _cell_id(16, 0.0, eta, seed)
                manifest["results"].append(
                    {
                        "attempt_id": f"{cell_id}#1",
                        "cell_id": cell_id,
                        "h": 16,
                        "mu": mu,
                        "eta": eta,
                        "seed": seed,
                        "training_seed": seed_pairs[str(seed)],
                        "status": "COMPLETED",
                        "failure_reason": None,
                        "loss": loss_curve[mu][eta] + (seed % 7) * 0.001,
                        "work": {
                            "fixed_window_microsteps": 16,
                            "fixed_window_tokens": 2048,
                            "outer_steps": 320,
                            "token_budget": 655360,
                            "eval_rows": 1024,
                        },
                        "observed_work": {
                            "tokens": 655360,
                            "microsteps": 5120,
                            "outer_steps": 320,
                            "per_fragment_outer_steps": {
                                "0": 80,
                                "1": 80,
                                "2": 80,
                                "3": 80,
                            },
                            "full_quorum": True,
                            "fixed_window_exact": True,
                            "version_matched_anchor_resolved": True,
                        },
                        "git_commit": GIT,
                        "image_digest": HEX_A,
                        "model_hash": HEX_A,
                        "data_hash": HEX_A,
                        "eval_hash": HEX_A,
                        "command_hash": hashlib.sha256(cell_id.encode("utf-8")).hexdigest(),
                        "capture_uri": f"gs://bucket/{cell_id}/capture",
                        "capture_sha256": HEX_A,
                        "result_uri": f"gs://bucket/{cell_id}/result.json",
                        "result_sha256": HEX_A,
                        "per_example_loss_uri": f"gs://bucket/{cell_id}/eval.jsonl",
                        "per_example_loss_sha256": HEX_A,
                        "paired_control_id": None if mu == 0.0 else control_id,
                        "barrier": True,
                        "version_matched": True,
                        "matrix_merge": "rda",
                        "strict_quorum": True,
                        "delta_correction": "none",
                        "injected_baseline": False,
                        "spot": True,
                        "block_id": block_id,
                        "order_index": order_index,
                        "attempt": 1,
                        "retry_of": None,
                        "retry_reason": None,
                        "retry_authorization": None,
                        "hardware": {
                            "market": "spot",
                            "provider": "gcp",
                            "instance_type": "a2-highgpu-4g",
                            "region": "us-central1-a",
                            "instance_id": "123456789",
                            "image_id": "7290368630472593484",
                            "provisioning_evidence_uri": f"gs://bucket/{cell_id}/spot.json",
                            "provisioning_evidence_sha256": HEX_B,
                        },
                        "started_at": "2026-07-14T12:00:00Z",
                        "ended_at": "2026-07-14T13:00:00Z",
                    }
                )
    manifest["frozen"]["cell_command_hashes"] = {
        row["cell_id"]: row["command_hash"] for row in manifest["results"]
    }
    canonical_retry_policy = json.dumps(
        manifest["retry_policy"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    manifest["frozen"]["retry_policy_hash"] = hashlib.sha256(
        canonical_retry_policy
    ).hexdigest()
    return manifest


def _result(manifest: dict, *, mu: float, eta: float, seed: int = 347) -> dict:
    return next(
        row
        for row in manifest["results"]
        if row["mu"] == mu and row["eta"] == eta and row["seed"] == seed
    )


def test_valid_development_manifest_is_bracketed_but_not_confirmatory() -> None:
    report = validate_and_summarize(_manifest(), claim_level="development")
    assert report["expected_cell_count"] == 6
    assert report["overall_bracket_decision"] == "BRACKETED_DEVELOPMENT_ONLY"
    assert report["confirmatory_eligible"] is False
    assert report["claim_scope"] == "development_only"
    candidate = next(
        row
        for row in report["paired_development_summaries"]
        if row["mu"] == 0.9 and row["eta"] == 0.04375
    )
    assert candidate["n_paired_completed"] == 1
    assert candidate["mean_candidate_minus_control"] == pytest.approx(0.05)
    assert candidate["sample_sd"] is None


def test_low_boundary_optimum_returns_extend_downward_not_pass() -> None:
    manifest = _manifest()
    for row in manifest["results"]:
        row["loss"] = 1.0 + 10.0 * row["eta"] + 0.1 * row["mu"]
    report = validate_and_summarize(manifest)
    assert {row["bracket_decision"] for row in report["phase_map"]} == {
        "EXTEND_DOWNWARD"
    }
    assert report["overall_bracket_decision"] == "EXTENSION_REQUIRED"
    assert "PASS" not in str(report)


def test_exact_cell_coverage_rejects_missing_and_unexpected_cells() -> None:
    missing = _manifest()
    missing["results"].pop()
    with pytest.raises(ManifestError, match="missing expected cells"):
        validate_and_summarize(missing)

    unexpected = _manifest()
    extra = copy.deepcopy(unexpected["results"][-1])
    extra.update(
        cell_id="unexpected",
        attempt_id="unexpected#1",
        mu=0.8,
        order_index=2,
    )
    unexpected["results"].append(extra)
    with pytest.raises(ManifestError, match="unexpected cells"):
        validate_and_summarize(unexpected)


def test_divergence_is_retained_with_null_loss_and_can_bracket_high_side() -> None:
    manifest = _manifest()
    row = _result(manifest, mu=0.9, eta=0.0875)
    row.update(status="DIVERGED", failure_reason="scientific_divergence", loss=None)
    row["per_example_loss_uri"] = None
    row["per_example_loss_sha256"] = None
    report = validate_and_summarize(manifest)
    assert report["final_status_counts"]["DIVERGED"] == 1
    candidate = next(row for row in report["phase_map"] if row["mu"] == 0.9)
    assert candidate["bracket_decision"] == "BRACKETED"
    assert candidate["points"][-1]["has_divergence"] is True


def test_preemption_must_be_infra_failure() -> None:
    manifest = _manifest()
    row = _result(manifest, mu=0.9, eta=0.0875)
    row.update(
        status="FAILED",
        failure_reason="provider_spot_preemption",
        loss=None,
        per_example_loss_uri=None,
        per_example_loss_sha256=None,
    )
    with pytest.raises(ManifestError, match="preemption must be classified as INFRA_FAILURE"):
        validate_and_summarize(manifest)


def test_spot_provisioning_and_exact_work_are_required() -> None:
    manifest = _manifest()
    row = manifest["results"][0]
    row["spot"] = False
    row["work"]["outer_steps"] = 319
    row["observed_work"]["outer_steps"] = 319
    with pytest.raises(ManifestError) as caught:
        validate_and_summarize(manifest)
    assert ".spot must be true" in str(caught.value)
    assert "does not match frozen target 320" in str(caught.value)
    assert "does not match scientific target 320" in str(caught.value)


def _make_full_block_retry(manifest: dict, eta: float = 0.0875) -> None:
    original_rows = [
        row
        for row in manifest["results"]
        if row["eta"] == eta and row["seed"] == 347
    ]
    trigger = next(row for row in original_rows if row["mu"] == 0.9)
    trigger.update(
        status="INFRA_FAILURE",
        failure_reason="provider_spot_preemption",
        loss=None,
        per_example_loss_uri=None,
        per_example_loss_sha256=None,
    )
    trigger["observed_work"]["outer_steps"] = 100
    trigger["observed_work"]["tokens"] = 200000
    trigger["observed_work"]["microsteps"] = 1562
    trigger["observed_work"]["full_quorum"] = False
    trigger["observed_work"]["fixed_window_exact"] = False
    trigger["observed_work"]["version_matched_anchor_resolved"] = False

    prior_manifest = copy.deepcopy(manifest)
    prior_hash = hashlib.sha256(
        json.dumps(
            prior_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    authorization = {
        "loss_blind": True,
        "policy_hash": manifest["frozen"]["retry_policy_hash"],
        "trigger_attempt_id": trigger["attempt_id"],
        "trigger_reason": trigger["failure_reason"],
        "trigger_block_id": trigger["block_id"],
        "prior_manifest_sha256": prior_hash,
    }

    retries = []
    for original in original_rows:
        retry = copy.deepcopy(original)
        retry.update(
            attempt=2,
            attempt_id=f"{original['cell_id']}#2",
            retry_of=f"{original['cell_id']}#1",
            retry_reason=(
                "provider_spot_preemption"
                if original is trigger
                else "peer_block_invalidated_by_infra_failure"
            ),
            retry_authorization=copy.deepcopy(authorization),
            status="COMPLETED",
            failure_reason=None,
            loss=1.2 if original["mu"] == 0.0 else 1.3,
            per_example_loss_uri=f"gs://bucket/{original['cell_id']}/retry-eval.jsonl",
            per_example_loss_sha256=HEX_A,
            started_at="2026-07-14T14:00:00Z",
            ended_at="2026-07-14T15:00:00Z",
        )
        retry["work"] = {
            "fixed_window_microsteps": 16,
            "fixed_window_tokens": 2048,
            "outer_steps": 320,
            "token_budget": 655360,
            "eval_rows": 1024,
        }
        retry["observed_work"] = {
            "tokens": 655360,
            "microsteps": 5120,
            "outer_steps": 320,
            "per_fragment_outer_steps": {
                "0": 80,
                "1": 80,
                "2": 80,
                "3": 80,
            },
            "full_quorum": True,
            "fixed_window_exact": True,
            "version_matched_anchor_resolved": True,
        }
        retries.append(retry)
    manifest["results"].extend(retries)


def test_loss_blind_exact_seed_full_block_retry_is_retained() -> None:
    manifest = _manifest()
    _make_full_block_retry(manifest)
    report = validate_and_summarize(manifest)
    assert report["retry_count"] == 2
    assert report["all_attempt_status_counts"]["INFRA_FAILURE"] == 1
    assert report["retained_noncompleted_attempt_count"] == 1


def test_retry_requires_frozen_loss_blind_authorization() -> None:
    manifest = _manifest()
    _make_full_block_retry(manifest)
    retry = next(row for row in manifest["results"] if row["attempt"] == 2)
    retry["retry_authorization"]["loss_blind"] = False
    with pytest.raises(ManifestError, match="loss_blind=true"):
        validate_and_summarize(manifest)


def test_partial_block_retry_is_rejected() -> None:
    manifest = _manifest()
    _make_full_block_retry(manifest)
    manifest["results"] = [
        row
        for row in manifest["results"]
        if not (row["attempt"] == 2 and row["mu"] == 0.9)
    ]
    with pytest.raises(ManifestError, match="partial"):
        validate_and_summarize(manifest)


def test_completed_peer_retry_requires_peer_reason_and_genuine_trigger() -> None:
    manifest = _manifest()
    _make_full_block_retry(manifest)
    peer_retry = next(
        row for row in manifest["results"] if row["attempt"] == 2 and row["mu"] == 0.0
    )
    peer_retry["retry_reason"] = "provider_spot_preemption"
    with pytest.raises(ManifestError, match="completed peer"):
        validate_and_summarize(manifest)

    no_trigger = _manifest()
    _make_full_block_retry(no_trigger)
    trigger = next(
        row
        for row in no_trigger["results"]
        if row["attempt"] == 1 and row["mu"] == 0.9 and row["eta"] == 0.0875
    )
    trigger.update(
        status="COMPLETED",
        failure_reason=None,
        loss=1.3,
        per_example_loss_uri="gs://bucket/restored/eval.jsonl",
        per_example_loss_sha256=HEX_A,
    )
    trigger["observed_work"].update(
        tokens=655360,
        microsteps=5120,
        outer_steps=320,
        full_quorum=True,
        fixed_window_exact=True,
        version_matched_anchor_resolved=True,
    )
    with pytest.raises(ManifestError, match="genuine direct INFRA_FAILURE"):
        validate_and_summarize(no_trigger)


def test_retry_prior_manifest_hash_and_policy_hash_are_verified() -> None:
    manifest = _manifest()
    _make_full_block_retry(manifest)
    for row in manifest["results"]:
        if row["attempt"] == 2:
            row["retry_authorization"]["prior_manifest_sha256"] = HEX_B
    with pytest.raises(ManifestError, match="prior_manifest_sha256"):
        validate_and_summarize(manifest)

    policy_tamper = _manifest()
    policy_tamper["retry_policy"]["forbidden_reasons"] = ["new_unfrozen_rule"]
    with pytest.raises(ManifestError, match="retry_policy_hash"):
        validate_and_summarize(policy_tamper)


def test_peer_retry_reason_can_never_be_failure_reason() -> None:
    manifest = _manifest()
    row = _result(manifest, mu=0.9, eta=0.0875)
    row.update(
        status="INFRA_FAILURE",
        failure_reason="peer_block_invalidated_by_infra_failure",
        loss=None,
        per_example_loss_uri=None,
        per_example_loss_sha256=None,
    )
    with pytest.raises(ManifestError, match="peer-only retry reason"):
        validate_and_summarize(manifest)


def test_one_seed_development_manifest_refuses_confirmatory_claim() -> None:
    with pytest.raises(ManifestError, match="REFUSED_CONFIRMATORY_CLAIM"):
        validate_and_summarize(_manifest(), claim_level="confirmatory")


def test_eight_seed_confirmation_can_be_confirmatory() -> None:
    seeds = (347, 359, 373, 383, 397, 409, 421, 433)
    report = validate_and_summarize(
        _manifest(seeds=seeds, mode="confirmation"),
        claim_level="confirmatory",
        require_bracketed=True,
    )
    assert report["confirmatory_eligible"] is True
    assert report["independent_seed_count"] == 8
    paired = next(
        row
        for row in report["paired_development_summaries"]
        if row["mu"] == 0.9 and row["eta"] == 0.04375
    )
    assert paired["n_paired_completed"] == 8
    assert paired["sample_sd"] == pytest.approx(0.0, abs=1e-15)


def test_provenance_mismatch_is_rejected() -> None:
    manifest = _manifest()
    manifest["results"][0]["eval_hash"] = HEX_B
    manifest["results"][0]["hardware"]["image_id"] = "999"
    with pytest.raises(ManifestError, match="eval_hash does not match"):
        validate_and_summarize(manifest)


def test_cli_emits_json_and_refuses_confirmation(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert phase_map.main(
        [
            str(manifest_path),
            "--claim-level",
            "development",
            "--output",
            str(output_path),
        ]
    ) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["claim_scope"] == "development_only"

    assert phase_map.main(
        [str(manifest_path), "--claim-level", "confirmatory"]
    ) == 2
    assert "REFUSED_CONFIRMATORY_CLAIM" in capsys.readouterr().err
