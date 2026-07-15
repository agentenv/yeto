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
_validate_with_authority = phase_map.validate_and_summarize


def validate_and_summarize(*args, **kwargs):
    """Exercise payload/statistical rules independently of Git authority.

    Dedicated tests at the end of this file exercise the production authority
    path against the hard-pinned preregistration commit.
    """

    kwargs["_skip_authority_for_tests"] = True
    return _validate_with_authority(*args, **kwargs)


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
    confirmation_policy = copy.deepcopy(
        phase_map._authoritative_template()["confirmation_policy"]
    )
    manifest = {
        "schema_version": "0.2",
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
        "expected_cells": [],
        "seed_pairs": seed_pairs,
        "frozen": {
            "git_commit": GIT,
            "image_id": "7290368630472593484",
            "image_digest": HEX_A,
            "model_id": "test/model",
            "model_revision": GIT,
            "model_hash": HEX_A,
            "data_hash": HEX_A,
            "development_eval_source_indices_hash": HEX_A,
            "audit_eval_source_indices_hash": HEX_B,
            "train_pool_source_indices_hash": HEX_A,
            "train_source_indices_hashes": {
                str(seed): hashlib.sha256(f"train-index-{seed}".encode()).hexdigest()
                for seed in seeds
            },
            "train_rows_hashes": {
                str(seed): hashlib.sha256(f"train-rows-{seed}".encode()).hexdigest()
                for seed in seeds
            },
            "development_eval_rows_hash": HEX_A,
            "development_eval_packed_hash": HEX_A,
            "development_eval_example_ids_hash": HEX_A,
            "development_eval_token_ids_hash": HEX_A,
            "audit_eval_rows_hash": HEX_B,
            "audit_eval_packed_hash": HEX_B,
            "audit_eval_example_ids_hash": HEX_B,
            "audit_eval_token_ids_hash": HEX_B,
            "audit_access_policy_hash": phase_map._sha256_canonical(
                confirmation_policy
            ),
            "command_hash": HEX_B,
            "randomization_plan_hash": HEX_B,
            "retry_policy_hash": HEX_A,
            "audit_command_hash": None,
            "audit_cell_command_hashes": None,
            "audit_randomization_plan_hash": None,
        },
        "protocol": {
            "tuning": "full",
            "train_rows": 5000,
            "development_eval_rows": 1024,
            "audit_eval_rows": 1024,
            "split_population_rows": 7048,
            "eval_split_seed": 331,
            "split_population_rule": "canonical_source_indices_0_through_7047",
            "split_assignment_rule": (
                "python_random.Random(331).shuffle_once_then_positions_0_5000_train_"
                "5000_6024_development_6024_7048_audit"
            ),
            "train_pool_slice_rule": "shuffled_indices_half_open_0_5000",
            "development_eval_slice_rule": "shuffled_indices_half_open_5000_6024",
            "audit_eval_slice_rule": "shuffled_indices_half_open_6024_7048",
            "train_shuffle_rule": (
                "per_study_shuffle_seed_applies_only_to_disjoint_pre_shuffle_train_pool"
            ),
            "matrix_merge": "rda",
            "strict_quorum": True,
            "barrier": True,
            "version_matched": True,
            "delta_correction": "none",
            "spot_only": True,
            "on_demand_fallback": False,
            "injected_baseline": False,
            "per_example_loss_required": True,
            "token_budget": 655360,
            "seq_len": 128,
            "machine_type": "a2-highgpu-4g",
            "gpu_slots": 4,
        },
        "confirmation_policy": confirmation_policy,
        "audit_checkpoint_registry": None,
        "audit_unblind_authorization": None,
        "audit_randomization": None,
        "audit_access_log": [],
        "audit_results": [],
        "audit_results_seal": None,
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
            "evaluation_role",
            "failure_reason",
            "loss",
            "work",
            "git_commit",
            "image_digest",
            "model_hash",
            "data_hash",
            "eval_source_indices_hash",
            "train_pool_source_indices_hash",
            "train_source_indices_hash",
            "train_rows_hash",
            "eval_rows_hash",
            "eval_hash",
            "eval_example_ids_hash",
            "eval_token_ids_hash",
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
    if len(etas) == 3:
        loss_curve = {
            0.0: {etas[0]: 1.10, etas[1]: 1.00, etas[2]: 1.20},
            0.9: {etas[0]: 1.20, etas[1]: 1.05, etas[2]: 1.30},
        }
    else:
        loss_curve = {
            0.0: {eta: 1.00 for eta in etas},
            0.9: {eta: 1.05 for eta in etas},
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
                        "evaluation_role": "development",
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
                        "eval_source_indices_hash": HEX_A,
                        "train_pool_source_indices_hash": HEX_A,
                        "train_source_indices_hash": manifest["frozen"][
                            "train_source_indices_hashes"
                        ][str(seed)],
                        "train_rows_hash": manifest["frozen"]["train_rows_hashes"][
                            str(seed)
                        ],
                        "eval_rows_hash": HEX_A,
                        "eval_hash": HEX_A,
                        "eval_example_ids_hash": HEX_A,
                        "eval_token_ids_hash": HEX_A,
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
                manifest["expected_cells"].append(
                    {
                        "cell_id": cell_id,
                        "h": 16,
                        "mu": mu,
                        "eta": eta,
                        "seed": seed,
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


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _p3_manifest() -> dict:
    seeds = (383, 397, 409, 421, 433, 443, 457, 461)
    manifest = _manifest(seeds=seeds, etas=(0.04375,), mode="confirmation")
    manifest["lineage"] = {"descendant_kind": "fresh_confirmation_stage"}
    frozen = manifest["frozen"]
    frozen["audit_command_hash"] = hashlib.sha256(b"audit-command").hexdigest()
    frozen["audit_cell_command_hashes"] = {
        row["cell_id"]: hashlib.sha256(
            f"audit-{row['cell_id']}".encode()
        ).hexdigest()
        for row in manifest["results"]
    }
    frozen["audit_randomization_plan_hash"] = hashlib.sha256(
        b"audit-randomization"
    ).hexdigest()

    original_losses = {row["cell_id"]: row["loss"] for row in manifest["results"]}
    for row in manifest["results"]:
        row["evaluation_role"] = "none"
        row["loss"] = None
        for field in (
            "eval_source_indices_hash",
            "eval_rows_hash",
            "eval_hash",
            "eval_example_ids_hash",
            "eval_token_ids_hash",
            "per_example_loss_uri",
            "per_example_loss_sha256",
        ):
            row[field] = None
        row["work"]["eval_rows"] = 0
        row["checkpoint_uri"] = f"gs://bucket/{row['cell_id']}/final.ckpt"
        row["checkpoint_sha256"] = hashlib.sha256(
            f"checkpoint-{row['cell_id']}".encode()
        ).hexdigest()
        row["checkpoint_sealed_at"] = "2026-07-14T13:00:00Z"

    registry = {
        "schema": "yeto_p3_checkpoint_registry_v1",
        "cells": [
            {
                "cell_id": row["cell_id"],
                "final_attempt_id": row["attempt_id"],
                "status": row["status"],
                "checkpoint_uri": row["checkpoint_uri"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "command_hash": row["command_hash"],
                "training_completed_at": row["ended_at"],
            }
            for row in manifest["results"]
        ],
        "sealed_at_utc": "2026-07-14T13:10:00Z",
    }
    manifest["audit_checkpoint_registry"] = registry
    ordered_ids = [row["cell_id"] for row in reversed(manifest["results"])]
    manifest["audit_randomization"] = {
        "schema": "yeto_p3_audit_randomization_v1",
        "ordered_cell_ids": ordered_ids,
        "plan_hash": frozen["audit_randomization_plan_hash"],
        "created_at_utc": "2026-07-14T11:00:00Z",
    }
    authorization = {
        "schema": "yeto_p3_audit_authorization_v1",
        "loss_blind": True,
        "all_training_cells_resolved_and_checkpoint_registry_sealed": True,
        "p3_manifest_canonical_sha256": hashlib.sha256(b"p3-training-manifest").hexdigest(),
        "checkpoint_registry_sha256": phase_map._sha256_canonical(registry),
        "audit_command_registry_sha256": phase_map._sha256_canonical(
            frozen["audit_cell_command_hashes"]
        ),
        "audit_randomization_plan_sha256": frozen[
            "audit_randomization_plan_hash"
        ],
        "training_completed_max_utc": "2026-07-14T13:00:00Z",
        "authorized_at_utc": "2026-07-14T13:20:00Z",
        "partial_results_withheld": True,
    }
    manifest["audit_unblind_authorization"] = authorization
    authorization_sha = phase_map._sha256_canonical(authorization)
    rows_by_id = {row["cell_id"]: row for row in manifest["results"]}
    audit_results = []
    access_log = []
    for order_index, cell_id in enumerate(ordered_ids):
        training = rows_by_id[cell_id]
        audit_row = {
            "cell_id": cell_id,
            "evaluation_role": "confirmation_audit",
            "audit_status": "COMPLETED",
            "audit_loss": original_losses[cell_id],
            "audit_eval_source_indices_hash": frozen[
                "audit_eval_source_indices_hash"
            ],
            "audit_eval_rows_hash": frozen["audit_eval_rows_hash"],
            "audit_eval_packed_hash": frozen["audit_eval_packed_hash"],
            "audit_eval_example_ids_hash": frozen[
                "audit_eval_example_ids_hash"
            ],
            "audit_eval_token_ids_hash": frozen["audit_eval_token_ids_hash"],
            "audit_command_hash": frozen["audit_cell_command_hashes"][cell_id],
            "audit_order_index": order_index,
            "audit_per_example_loss_uri": f"gs://bucket/{cell_id}/audit.jsonl",
            "audit_per_example_loss_sha256": hashlib.sha256(
                f"audit-output-{cell_id}".encode()
            ).hexdigest(),
            "checkpoint_uri": training["checkpoint_uri"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "training_attempt_id": training["attempt_id"],
            "training_completed_at": training["ended_at"],
            "audit_started_at": "2026-07-14T14:00:00Z",
            "audit_ended_at": "2026-07-14T14:01:00Z",
            "audit_unblind_authorization_sha256": authorization_sha,
        }
        audit_results.append(audit_row)
        access_log.append(
            {
                "cell_id": cell_id,
                "checkpoint_sha256": training["checkpoint_sha256"],
                "audit_eval_packed_hash": frozen["audit_eval_packed_hash"],
                "audit_command_hash": frozen["audit_cell_command_hashes"][cell_id],
                "access_started_at": audit_row["audit_started_at"],
                "access_ended_at": audit_row["audit_ended_at"],
            }
        )
    manifest["audit_results"] = audit_results
    manifest["audit_access_log"] = access_log
    manifest["audit_results_seal"] = {
        "schema": "yeto_p3_audit_results_seal_v1",
        "status": "SEALED",
        "audit_result_registry_sha256": phase_map._sha256_canonical(audit_results),
        "audit_cell_count": len(audit_results),
        "expected_cell_ids_covered_exactly": True,
        "sealed_at_utc": "2026-07-14T15:00:00Z",
        "partial_results_exposed": False,
        "unblinded_at_utc": "2026-07-14T15:01:00Z",
    }
    return manifest


def _authoritative_p0a() -> dict:
    manifest = copy.deepcopy(phase_map._authoritative_template())
    manifest["status"] = "sealed_results"
    manifest["study_id"] = "bp-phase-map-p0a-unit"
    manifest["expected_grid"] = {
        "h": [16],
        "mu": [0.0, 0.5, 0.9],
        "eta": [0.0875],
        "seeds": [337],
    }
    manifest["seed_pairs"] = {"337": 337337}
    manifest["protocol"]["token_budget"] = 65536
    manifest["protocol"]["gpu_slots"] = 1
    manifest["protocol"]["machine_type"] = "a2-highgpu-1g"
    manifest["horizon_work"] = {
        "16": {
            "fixed_window_microsteps": 16,
            "fixed_window_tokens": 2048,
            "outer_steps": 32,
        }
    }
    cells = []
    for mu in (0.0, 0.5, 0.9):
        cell_id = f"bp-phase-map-p0a-unit-h16-mu{mu:g}-eta0.0875-s337"
        cells.append(
            {"cell_id": cell_id, "h": 16, "mu": mu, "eta": 0.0875, "seed": 337}
        )
    manifest["expected_cells"] = cells
    frozen = manifest["frozen"]
    frozen.update(
        {
            "git_commit": _git_head(),
            "image_digest": HEX_A,
            "model_hash": HEX_A,
            "data_hash": HEX_A,
            "development_eval_source_indices_hash": HEX_A,
            "audit_eval_source_indices_hash": HEX_B,
            "train_pool_source_indices_hash": HEX_B,
            "train_source_indices_hashes": {"337": HEX_A},
            "train_rows_hashes": {"337": HEX_B},
            "development_eval_rows_hash": HEX_A,
            "development_eval_packed_hash": HEX_A,
            "development_eval_example_ids_hash": HEX_A,
            "development_eval_token_ids_hash": HEX_A,
            "audit_eval_rows_hash": HEX_B,
            "audit_eval_packed_hash": HEX_B,
            "audit_eval_example_ids_hash": HEX_B,
            "audit_eval_token_ids_hash": HEX_B,
            "audit_access_policy_hash": phase_map._sha256_canonical(
                manifest["confirmation_policy"]
            ),
            "command_hash": HEX_B,
            "cell_command_hashes": {
                cell["cell_id"]: hashlib.sha256(cell["cell_id"].encode()).hexdigest()
                for cell in cells
            },
            "randomization_plan_hash": HEX_B,
        }
    )
    frozen["retry_policy_hash"] = hashlib.sha256(
        json.dumps(
            manifest["retry_policy"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    manifest["randomization"]["plan_hash"] = HEX_B
    manifest["lineage"] = {
        "authoritative_prereg_path": phase_map.AUTHORITATIVE_PREREG_PATH,
        "authoritative_prereg_source_commit": (
            phase_map.AUTHORITATIVE_PREREG_SOURCE_COMMIT
        ),
        "authoritative_prereg_template_sha256": (
            phase_map.AUTHORITATIVE_PREREG_TEMPLATE_SHA256
        ),
        "parent_manifest_sha256": None,
        "parent_replay_report_sha256": None,
        "descendant_kind": "p0a_single_gpu_bound",
    }
    results = []
    for order_index, cell in enumerate(cells):
        cell_id = cell["cell_id"]
        mu = cell["mu"]
        results.append(
            {
                "attempt_id": f"{cell_id}-attempt-1",
                **cell,
                "training_seed": 337337,
                "status": "COMPLETED",
                "evaluation_role": "development",
                "failure_reason": None,
                "loss": 2.0 + 0.01 * mu,
                "work": {
                    "fixed_window_microsteps": 16,
                    "fixed_window_tokens": 2048,
                    "outer_steps": 32,
                    "token_budget": 65536,
                    "eval_rows": 1024,
                },
                "observed_work": {
                    "tokens": 65536,
                    "microsteps": 512,
                    "outer_steps": 32,
                    "full_quorum": True,
                    "fixed_window_exact": True,
                    "version_matched_anchor_resolved": True,
                },
                "git_commit": frozen["git_commit"],
                "image_digest": frozen["image_digest"],
                "model_hash": frozen["model_hash"],
                "data_hash": frozen["data_hash"],
                "eval_source_indices_hash": frozen[
                    "development_eval_source_indices_hash"
                ],
                "train_pool_source_indices_hash": frozen[
                    "train_pool_source_indices_hash"
                ],
                "train_source_indices_hash": frozen["train_source_indices_hashes"][
                    "337"
                ],
                "train_rows_hash": frozen["train_rows_hashes"]["337"],
                "eval_rows_hash": frozen["development_eval_rows_hash"],
                "eval_hash": frozen["development_eval_packed_hash"],
                "eval_example_ids_hash": frozen[
                    "development_eval_example_ids_hash"
                ],
                "eval_token_ids_hash": frozen[
                    "development_eval_token_ids_hash"
                ],
                "command_hash": frozen["cell_command_hashes"][cell_id],
                "normalized_workload_command_hash": hashlib.sha256(
                    f"normalized-{mu:g}".encode()
                ).hexdigest(),
                "capture_uri": f"gs://bucket/{cell_id}/capture",
                "capture_sha256": HEX_A,
                "result_uri": f"gs://bucket/{cell_id}/result.json",
                "result_sha256": HEX_A,
                "per_example_loss_uri": f"gs://bucket/{cell_id}/eval.jsonl",
                "per_example_loss_sha256": HEX_A,
                "paired_control_id": None if mu == 0.0 else cells[0]["cell_id"],
                "barrier": True,
                "version_matched": True,
                "matrix_merge": "rda",
                "strict_quorum": True,
                "delta_correction": "none",
                "injected_baseline": False,
                "spot": True,
                "block_id": "bp-phase-map-p0a-unit-block-h16-eta0.0875-s337",
                "order_index": order_index,
                "attempt": 1,
                "retry_of": None,
                "retry_reason": None,
                "retry_authorization": None,
                "hardware": {
                    "market": "spot",
                    "provider": "gcp",
                    "instance_type": "a2-highgpu-1g",
                    "region": "us-central1-a",
                    "instance_id": "123456789",
                    "image_id": frozen["image_id"],
                    "provisioning_evidence_uri": f"gs://bucket/{cell_id}/spot.json",
                    "provisioning_evidence_sha256": HEX_B,
                    "artifact_sealed_at": "2026-07-14T13:05:00Z",
                    "deletion_requested_at": "2026-07-14T13:10:00Z",
                    "deletion_completed_at": "2026-07-14T13:20:00Z",
                    "acquisition_status": "sealed_acquisition_pending_teardown",
                    "acquisition_manifest_sha256": HEX_A,
                    "acquisition_manifest_canonical_sha256": HEX_A,
                    "acquisition_checksum_sha256": HEX_A,
                    "acquisition_seal_sha256": HEX_A,
                    "final_manifest_status": "sealed_results",
                    "deletion_evidence_sha256": HEX_A,
                    "finalized_at": "2026-07-14T13:30:00Z",
                },
                "started_at": "2026-07-14T12:00:00Z",
                "ended_at": "2026-07-14T13:00:00Z",
            }
        )
    manifest["results"] = results
    return manifest


def _replay_for(parent: dict) -> tuple[dict, str]:
    replay = {
        "schema": "yeto_p0_cpu_replay_v1",
        "status": "PASS",
        "gpu_deleted_before_replay": True,
        "all_steps_replayed": True,
        "phase_map_manifest_canonical_sha256": phase_map._sha256_canonical(parent),
    }
    replay_bytes = (json.dumps(replay, indent=2, sort_keys=True) + "\n").encode()
    return replay, hashlib.sha256(replay_bytes).hexdigest()


def _authoritative_p0b(parent: dict) -> tuple[dict, dict, str]:
    manifest = copy.deepcopy(parent)
    manifest["study_id"] = "bp-phase-map-p0b-unit"
    manifest["protocol"]["machine_type"] = "a2-highgpu-4g"
    manifest["protocol"]["gpu_slots"] = 4
    replay, replay_sha = _replay_for(parent)
    manifest["lineage"] = {
        "authoritative_prereg_path": phase_map.AUTHORITATIVE_PREREG_PATH,
        "authoritative_prereg_source_commit": phase_map.AUTHORITATIVE_PREREG_SOURCE_COMMIT,
        "authoritative_prereg_template_sha256": phase_map.AUTHORITATIVE_PREREG_TEMPLATE_SHA256,
        "parent_manifest_sha256": phase_map._sha256_canonical(parent),
        "parent_replay_report_sha256": replay_sha,
        "descendant_kind": "p0b_four_gpu_bound",
    }
    old_cells = list(manifest["expected_cells"])
    id_map = {
        cell["cell_id"]: cell["cell_id"].replace("p0a-unit", "p0b-unit")
        for cell in old_cells
    }
    for cell in manifest["expected_cells"]:
        cell["cell_id"] = id_map[cell["cell_id"]]
    for row in manifest["results"]:
        old_id = row["cell_id"]
        cell_id = id_map[old_id]
        row["cell_id"] = cell_id
        row["attempt_id"] = f"{cell_id}-attempt-1"
        row["paired_control_id"] = (
            None if row["mu"] == 0.0 else id_map[row["paired_control_id"]]
        )
        row["block_id"] = row["block_id"].replace("p0a-unit", "p0b-unit")
        row["result_uri"] = f"gs://bucket/{cell_id}/result.json"
        row["capture_uri"] = f"gs://bucket/{cell_id}/capture"
        row["per_example_loss_uri"] = f"gs://bucket/{cell_id}/eval.jsonl"
        row["hardware"] = {
            "provider": "gcp",
            "zone": "us-central1-a",
            "region": "us-central1-a",
            "market": "spot",
            "instance_type": "a2-highgpu-4g",
            "instance_name": f"p0b-{row['order_index']}",
            "instance_id": f"900{row['order_index']}",
            "instance_numeric_id": f"900{row['order_index']}",
            "boot_disk_name": f"p0b-disk-{row['order_index']}",
            "boot_disk_numeric_id": f"800{row['order_index']}",
            "image_id": manifest["frozen"]["image_id"],
            "source_image_numeric_id": manifest["frozen"]["image_id"],
            "provisioning_evidence_uri": f"gs://bucket/{cell_id}/spot.json",
            "provisioning_evidence_sha256": HEX_B,
            "provisioning_started_at": "2026-07-14T11:00:00Z",
            "provisioning_completed_at": "2026-07-14T11:05:00Z",
            "nvidia_smi_inventory_uri": f"gs://bucket/{cell_id}/nvidia-smi.json",
            "nvidia_smi_inventory_sha256": HEX_A,
            "learner_gpu_map_uri": f"gs://bucket/{cell_id}/gpu-map.json",
            "learner_gpu_map_sha256": HEX_A,
            "barrier_version_trace_uri": f"gs://bucket/{cell_id}/barrier.jsonl",
            "barrier_version_trace_sha256": HEX_A,
            "distinct_a100_gpu_uuid_count": 4,
            "learner_gpu_uuid_bijection": {
                "0": "GPU-a",
                "1": "GPU-b",
                "2": "GPU-c",
                "3": "GPU-d",
            },
            "artifact_sealed_at": "2026-07-14T13:05:00Z",
            "deletion_requested_at": "2026-07-14T13:10:00Z",
            "deletion_completed_at": "2026-07-14T13:20:00Z",
            "acquisition_status": "sealed_acquisition_pending_teardown",
            "acquisition_manifest_sha256": HEX_A,
            "acquisition_manifest_canonical_sha256": HEX_A,
            "acquisition_checksum_sha256": HEX_A,
            "acquisition_seal_sha256": HEX_A,
            "final_manifest_status": "sealed_results",
            "deletion_evidence_sha256": HEX_A,
            "finalized_at": "2026-07-14T13:30:00Z",
            "instance_not_found_evidence_uri": f"gs://bucket/{cell_id}/vm-gone.json",
            "instance_not_found_evidence_sha256": HEX_A,
            "disk_not_found_evidence_uri": f"gs://bucket/{cell_id}/disk-gone.json",
            "disk_not_found_evidence_sha256": HEX_A,
            "zero_accelerator_evidence_uri": f"gs://bucket/{cell_id}/gpu-gone.json",
            "zero_accelerator_evidence_sha256": HEX_A,
        }
    manifest["frozen"]["cell_command_hashes"] = {
        row["cell_id"]: hashlib.sha256(row["cell_id"].encode()).hexdigest()
        for row in manifest["results"]
    }
    for row in manifest["results"]:
        row["command_hash"] = manifest["frozen"]["cell_command_hashes"][row["cell_id"]]
    manifest["frozen"]["command_hash"] = hashlib.sha256(b"p0b-command").hexdigest()
    manifest["frozen"]["randomization_plan_hash"] = hashlib.sha256(b"p0b-order").hexdigest()
    manifest["randomization"]["plan_hash"] = manifest["frozen"]["randomization_plan_hash"]
    return manifest, replay, replay_sha


def _authoritative_p1(parent: dict) -> tuple[dict, dict, str]:
    manifest = copy.deepcopy(phase_map._authoritative_template())
    cells = []
    for h in (16, 64, 256):
        for mu in (0.0, 0.5, 0.9):
            for eta in (0.021875, 0.04375, 0.0875, 0.175):
                cell_id = f"bp-phase-map-p1-r0-h{h}-mu{mu:g}-eta{eta:g}-s347"
                cells.append(
                    {"cell_id": cell_id, "h": h, "mu": mu, "eta": eta, "seed": 347}
                )
    frozen = manifest["frozen"]
    parent_frozen = parent["frozen"]
    for field in (
        "git_commit",
        "image_id",
        "image_digest",
        "model_id",
        "model_revision",
        "model_hash",
        "data_hash",
        "development_eval_source_indices_hash",
        "audit_eval_source_indices_hash",
        "train_pool_source_indices_hash",
        "development_eval_rows_hash",
        "development_eval_packed_hash",
        "development_eval_example_ids_hash",
        "development_eval_token_ids_hash",
        "audit_eval_rows_hash",
        "audit_eval_packed_hash",
        "audit_eval_example_ids_hash",
        "audit_eval_token_ids_hash",
        "audit_access_policy_hash",
        "retry_policy_hash",
    ):
        frozen[field] = parent_frozen[field]
    frozen["train_source_indices_hashes"] = {"347": HEX_A}
    frozen["train_rows_hashes"] = {"347": HEX_B}
    frozen["command_hash"] = HEX_B
    frozen["cell_command_hashes"] = {
        cell["cell_id"]: hashlib.sha256(cell["cell_id"].encode()).hexdigest()
        for cell in cells
    }
    frozen["randomization_plan_hash"] = HEX_A
    manifest["expected_cells"] = cells
    manifest["randomization"]["plan_hash"] = HEX_A
    replay, replay_sha = _replay_for(parent)
    manifest["lineage"] = {
        "authoritative_prereg_path": phase_map.AUTHORITATIVE_PREREG_PATH,
        "authoritative_prereg_source_commit": (
            phase_map.AUTHORITATIVE_PREREG_SOURCE_COMMIT
        ),
        "authoritative_prereg_template_sha256": (
            phase_map.AUTHORITATIVE_PREREG_TEMPLATE_SHA256
        ),
        "parent_manifest_sha256": phase_map._sha256_canonical(parent),
        "parent_replay_report_sha256": replay_sha,
        "descendant_kind": "initial_bound_p1_r0",
    }
    manifest["results"] = []
    manifest["status"] = "bound_launch_authority"
    return manifest, replay, replay_sha


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


def test_cell_id_is_bound_to_its_expected_coordinate() -> None:
    manifest = _manifest()
    first, second = manifest["results"][:2]
    first["cell_id"], second["cell_id"] = second["cell_id"], first["cell_id"]
    with pytest.raises(ManifestError, match="cell_id is not bound"):
        validate_and_summarize(manifest)


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


def test_eight_seed_nonregistered_confirmation_is_still_refused() -> None:
    seeds = (347, 359, 373, 383, 397, 409, 421, 433)
    with pytest.raises(ManifestError, match="fresh_confirmation_stage"):
        validate_and_summarize(
            _manifest(seeds=seeds, mode="confirmation"),
            claim_level="confirmatory",
            require_bracketed=True,
        )


def test_valid_p3_uses_sealed_audit_loss_as_sole_primary_endpoint() -> None:
    manifest = _p3_manifest()
    report = validate_and_summarize(manifest, claim_level="confirmatory")
    assert report["confirmatory_eligible"] is True
    assert report["independent_seed_count"] == 8
    assert report["paired_development_summaries"] == []
    paired = next(
        row
        for row in report["paired_audit_summaries"]
        if row["mu"] == 0.9 and row["eta"] == 0.04375
    )
    assert paired["n_paired_completed"] == 8
    assert paired["mean_candidate_minus_control"] == pytest.approx(0.05)
    assert paired["claim_scope"] == "paired_confirmation_audit_primary_endpoint"


def test_pre_p3_audit_outcome_or_access_is_rejected() -> None:
    manifest = _manifest()
    manifest["audit_results"] = [{"cell_id": manifest["results"][0]["cell_id"]}]
    manifest["results"][0]["audit_loss"] = 1.0
    with pytest.raises(ManifestError, match="audit_results must remain empty"):
        validate_and_summarize(manifest)


def test_p3_training_cannot_open_dev_or_audit_outcomes() -> None:
    manifest = _p3_manifest()
    training = manifest["results"][0]
    training["evaluation_role"] = "development"
    training["loss"] = 1.0
    training["eval_hash"] = manifest["frozen"]["development_eval_packed_hash"]
    manifest["audit_results"][0]["evaluation_role"] = "development"
    with pytest.raises(ManifestError) as caught:
        validate_and_summarize(manifest, claim_level="confirmatory")
    assert "evaluation_role must be 'none'" in str(caught.value)
    assert "loss must be null" in str(caught.value)
    assert "evaluation_role must be 'confirmation_audit'" in str(caught.value)


def test_p3_audit_must_start_after_all_training_and_shared_authorization() -> None:
    manifest = _p3_manifest()
    manifest["audit_results"][0]["audit_started_at"] = "2026-07-14T13:00:00Z"
    manifest["audit_access_log"][0]["access_started_at"] = "2026-07-14T13:00:00Z"
    with pytest.raises(ManifestError) as caught:
        validate_and_summarize(manifest, claim_level="confirmatory")
    assert "started before shared authorization" in str(caught.value)
    assert "started before all P3 training" in str(caught.value)


def test_p3_partial_or_unsealed_audit_bundle_is_rejected() -> None:
    manifest = _p3_manifest()
    removed = manifest["audit_results"].pop()
    manifest["audit_access_log"] = [
        row
        for row in manifest["audit_access_log"]
        if row["cell_id"] != removed["cell_id"]
    ]
    manifest["audit_results_seal"]["partial_results_exposed"] = True
    with pytest.raises(ManifestError) as caught:
        validate_and_summarize(manifest, claim_level="confirmatory")
    assert "cover expected P3 cells exactly" in str(caught.value)
    assert "partial audit results exposure is forbidden" in str(caught.value)


def test_p3_checkpoint_registry_rejects_incomplete_and_duplicate_cells() -> None:
    manifest = _p3_manifest()
    cells = manifest["audit_checkpoint_registry"]["cells"]
    cells.pop()
    cells.append(copy.deepcopy(cells[0]))
    with pytest.raises(ManifestError) as caught:
        validate_and_summarize(manifest, claim_level="confirmatory")
    assert "duplicates cell" in str(caught.value)
    assert "must cover expected P3 cells exactly" in str(caught.value)


def test_p3_randomization_must_precede_authorization() -> None:
    manifest = _p3_manifest()
    manifest["audit_randomization"]["created_at_utc"] = "2026-07-14T13:30:00Z"
    with pytest.raises(ManifestError, match="created before authorization"):
        validate_and_summarize(manifest, claim_level="confirmatory")


def test_p3_rejects_nonfinite_scientific_audit_loss() -> None:
    manifest = _p3_manifest()
    manifest["audit_results"][0]["audit_loss"] = float("nan")
    with pytest.raises(ManifestError, match="positive finite COMPLETED audit_loss"):
        validate_and_summarize(manifest, claim_level="confirmatory")


def test_spot_only_rejects_on_demand_or_fallback_evidence() -> None:
    manifest = _manifest()
    manifest["protocol"]["spot_only"] = False
    manifest["protocol"]["on_demand_fallback"] = True
    manifest["results"][0]["spot"] = False
    manifest["results"][0]["hardware"]["market"] = "on-demand"
    with pytest.raises(ManifestError) as caught:
        validate_and_summarize(manifest)
    assert "protocol.spot_only must be True" in str(caught.value)
    assert "protocol.on_demand_fallback must be False" in str(caught.value)
    assert ".spot must be true" in str(caught.value)
    assert "hardware.market must be 'spot'" in str(caught.value)


def test_provenance_mismatch_is_rejected() -> None:
    manifest = _manifest()
    manifest["results"][0]["eval_hash"] = HEX_B
    manifest["results"][0]["hardware"]["image_id"] = "999"
    with pytest.raises(ManifestError, match="eval_hash does not match"):
        validate_and_summarize(manifest)


def test_cli_emits_json_and_refuses_confirmation(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(_authoritative_p0a()), encoding="utf-8")
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


def test_hard_pinned_authority_rejects_forged_template_identity() -> None:
    manifest = _authoritative_p0a()
    manifest["lineage"]["authoritative_prereg_template_sha256"] = HEX_A
    with pytest.raises(ManifestError, match="template_sha256 is not hard-pinned"):
        _validate_with_authority(manifest)


def test_p0b_and_initial_p1_require_exact_parent_replay_chain() -> None:
    p0a = _authoritative_p0a()
    p0b, p0a_replay, p0a_replay_sha = _authoritative_p0b(p0a)
    p0b_report = _validate_with_authority(
        p0b,
        parent_manifest=p0a,
        parent_replay_report=p0a_replay,
        parent_replay_report_sha256=p0a_replay_sha,
    )
    assert p0b_report["integrity_status"] == "VALIDATED"

    manifest, replay, replay_sha = _authoritative_p1(p0b)

    report = _validate_with_authority(
        manifest,
        parent_manifest=p0b,
        parent_replay_report=replay,
        parent_replay_report_sha256=replay_sha,
    )
    assert report["integrity_status"] == "BOUND_LAUNCH_AUTHORITY_VALIDATED"
    assert report["expected_cell_count"] == 36

    with pytest.raises(ManifestError, match="requires the exact sealed parent"):
        _validate_with_authority(
            manifest,
            parent_replay_report=replay,
            parent_replay_report_sha256=replay_sha,
        )

    with pytest.raises(ManifestError, match="exact sealed parent CPU replay"):
        _validate_with_authority(manifest, parent_manifest=p0b)

    with pytest.raises(ManifestError, match="parent kind must be 'p0b_four_gpu_bound'"):
        _validate_with_authority(
            manifest,
            parent_manifest=p0a,
            parent_replay_report=p0a_replay,
            parent_replay_report_sha256=p0a_replay_sha,
        )


def test_p0b_rejects_coordinate_work_or_gpu_bijection_drift() -> None:
    p0a = _authoritative_p0a()
    p0b, replay, replay_sha = _authoritative_p0b(p0a)
    p0b["results"][0]["work"]["token_budget"] = 32768
    p0b["results"][1]["hardware"]["learner_gpu_uuid_bijection"]["3"] = "GPU-a"
    p0b["results"][2]["normalized_workload_command_hash"] = HEX_B
    with pytest.raises(ManifestError) as caught:
        _validate_with_authority(
            p0b,
            parent_manifest=p0a,
            parent_replay_report=replay,
            parent_replay_report_sha256=replay_sha,
        )
    assert "differs from P0a workload field work" in str(caught.value)
    assert "four distinct GPU UUIDs" in str(caught.value)
    assert "normalized_workload_command_hash" in str(caught.value)


def test_p0b_rejects_wrong_raw_parent_replay_hash() -> None:
    p0a = _authoritative_p0a()
    p0b, replay, replay_sha = _authoritative_p0b(p0a)
    p0b["lineage"]["parent_replay_report_sha256"] = HEX_A
    with pytest.raises(ManifestError, match="does not match the raw report bytes"):
        _validate_with_authority(
            p0b,
            parent_manifest=p0a,
            parent_replay_report=replay,
            parent_replay_report_sha256=replay_sha,
        )


def test_p0b_rejects_missing_exact_teardown_and_barrier_evidence() -> None:
    p0a = _authoritative_p0a()
    p0b, replay, replay_sha = _authoritative_p0b(p0a)
    del p0b["results"][0]["hardware"]["barrier_version_trace_sha256"]
    p0b["results"][0]["hardware"]["deletion_completed_at"] = (
        "2026-07-14T13:00:00Z"
    )
    with pytest.raises(ManifestError) as caught:
        _validate_with_authority(
            p0b,
            parent_manifest=p0a,
            parent_replay_report=replay,
            parent_replay_report_sha256=replay_sha,
        )
    assert "missing P0b evidence fields" in str(caught.value)
    assert "timestamps must order" in str(caught.value)


def test_canary_rejects_intermediate_legacy_or_mixed_lifecycle_state() -> None:
    intermediate = _authoritative_p0a()
    intermediate["status"] = "sealed_acquisition_pending_teardown"
    with pytest.raises(ManifestError, match="status must be"):
        _validate_with_authority(intermediate)

    mixed = _authoritative_p0a()
    mixed["results"][0]["hardware"]["final_manifest_status"] = (
        "sealed_acquisition_pending_teardown"
    )
    mixed["results"][1]["hardware"]["acquisition_manifest_sha256"] = None
    with pytest.raises(ManifestError) as caught:
        _validate_with_authority(mixed)
    assert "final_manifest_status must be 'sealed_results'" in str(caught.value)
    assert "acquisition_manifest_sha256 must be" in str(caught.value)

    legacy = _authoritative_p0a()
    legacy["lineage"]["descendant_kind"] = "p0_canary_bound"
    with pytest.raises(ManifestError, match="is not registered"):
        _validate_with_authority(legacy)

def test_initial_p1_cannot_drift_from_passing_p0b_or_expected_cells() -> None:
    p0a = _authoritative_p0a()
    parent, _, _ = _authoritative_p0b(p0a)
    manifest, replay, replay_sha = _authoritative_p1(parent)
    manifest["frozen"]["development_eval_packed_hash"] = HEX_B
    with pytest.raises(ManifestError, match="differs from its passing P0b"):
        _validate_with_authority(
            manifest,
            parent_manifest=parent,
            parent_replay_report=replay,
            parent_replay_report_sha256=replay_sha,
        )

    manifest, replay, replay_sha = _authoritative_p1(parent)
    manifest["expected_cells"].pop()
    with pytest.raises(ManifestError, match="exact frozen 36-cell grid"):
        _validate_with_authority(
            manifest,
            parent_manifest=parent,
            parent_replay_report=replay,
            parent_replay_report_sha256=replay_sha,
        )
