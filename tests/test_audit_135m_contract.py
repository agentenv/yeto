from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_135m_contract as audit
from scripts import run_parallel_phase_map as parallel
from scripts import run_phase_map as phase
from yeto import learner


def _runtime(tmp_path: Path) -> dict[str, object]:
    return {
        "run_dir": tmp_path / "runs",
        "model_path": tmp_path / "model",
        "python_executable": "/opt/yeto/venv/bin/python3",
        "command_repo_root": tmp_path / "repo",
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _selection(
    tmp_path: Path,
    *,
    stage: str,
    selected_etas: dict[str, float],
    boundary_status: str = "NOT_REQUIRED",
) -> Path:
    return _write_json(
        tmp_path / f"{stage.lower()}-selection.json",
        {
            "schema": f"audit_135m_{stage.lower()}_development_selection_v1",
            "status": "SEALED",
            "audit_stage": stage,
            "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
            "development_evidence_canonical_sha256": "d" * 64,
            "selection_rule": "lowest_pooled_development_mean",
            "boundary_extension_status": boundary_status,
            "selected_etas": selected_etas,
            "sealed_at_utc": "2026-07-16T23:30:00Z",
        },
    )


def _a1_selected() -> dict[str, float]:
    return {
        "H16_mu0": 0.021875,
        "H16_mu0.9": 0.002734375,
        "H256_mu0": 0.04375,
        "H256_mu0.5": 0.021875,
    }


def _a4_selected() -> dict[str, float]:
    values: dict[str, float] = {}
    for m in (1, 4):
        values.update(
            {
                f"M{m}_H16_mu0": 0.021875,
                f"M{m}_H16_mu0.9": 0.002734375,
                f"M{m}_H256_mu0": 0.04375,
                f"M{m}_H256_mu0.5": 0.021875,
            }
        )
    return values


def _precision_trigger(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "a4-precision-trigger.json",
        {
            "schema": "audit_135m_a4_precision_trigger_v1",
            "status": "SEALED",
            "audit_stage": "A4",
            "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
            "initial_confirmation_evidence_canonical_sha256": "e" * 64,
            "initial_confirmation_complete": True,
            "precision_expansion_rule": (
                "adjusted_ci_half_width_exceeds_epsilon_after_complete_initial_seed_block"
            ),
            "epsilon": 0.01,
            "triggered_sign_blindly": True,
            "expansion_required": True,
            "run_all_registered_expansion_seeds": True,
            "sealed_at_utc": "2026-07-16T23:40:00Z",
        },
    )


def _plan(
    tmp_path: Path,
    stage_code: str,
    *,
    decision_path: Path | None = None,
    precision_trigger_path: Path | None = None,
) -> dict[str, object]:
    plan, _hashes = audit.build_plan(
        stage_code=stage_code,
        study_id=f"audit-{stage_code}",
        runtime_config=_runtime(tmp_path),
        order_seed=20260716,
        decision_path=decision_path,
        precision_trigger_path=precision_trigger_path,
    )
    return plan


def _parent() -> dict[str, object]:
    cell = {
        "cell_id": "sealed-parent-cell",
        "h": 64,
        "m": 4,
        "mu": 0.0,
        "eta": 0.021875,
        "seed": 347,
        "training_seed": 347347,
        "block_id": "sealed-parent-block",
        "analysis_role": "sealed_parent",
        "pairing_identity_hash": "3" * 64,
        "pairing_command_hash": "4" * 64,
        "command_hash": "1" * 64,
        "normalized_workload_command_hash": "2" * 64,
    }
    return {
        "status": "sealed_results",
        "study_id": "sealed-parent",
        "expected_cells": [cell],
        "results": [{"cell_id": cell["cell_id"], "status": "COMPLETED"}],
        "seed_pairs": {"347": 347347},
        "frozen": {
            "model_id": audit.MODEL_ID,
            "model_revision": audit.MODEL_REVISION,
            "model_hash": audit.MODEL_HASH,
            "data_hash": audit.DATA_HASH,
            "image_id": audit.IMAGE_NUMERIC_ID,
            "image_digest": audit.IMAGE_DIGEST,
            "cell_command_hashes": {cell["cell_id"]: cell["command_hash"]},
            "train_rows_hashes": {"347": "5" * 64},
            "train_source_indices_hashes": {"347": "6" * 64},
        },
        "protocol": {
            "train_rows": audit.TRAIN_ROWS,
            "development_eval_rows": audit.DEVELOPMENT_EVAL_ROWS,
            "audit_eval_rows": audit.AUDIT_EVAL_ROWS,
            "seq_len": audit.SEQ_LEN,
            "micro_batch_size": audit.MICRO_BATCH_SIZE,
            "inner_lr": audit.INNER_LR,
            "token_budget": audit.TOKEN_BUDGET,
            "eval_split_seed": audit.EVAL_SPLIT_SEED,
            "spot_only": True,
            "barrier": True,
            "strict_quorum": True,
            "version_matched": True,
        },
    }


def _seed_registry(plan: dict[str, object]) -> dict[str, object]:
    seeds = sorted({int(cell["seed"]) for cell in plan["cells"]})
    return {
        "schema": "audit_135m_seed_bundle_registry_v1",
        "seeds": {
            str(seed): {
                "train_rows_sha256": f"{seed % 10:x}" * 64,
                "train_source_indices_sha256": f"{(seed + 1) % 10:x}" * 64,
                "parallel_eval_freeze_sha256": f"{(seed + 2) % 10:x}" * 64,
            }
            for seed in seeds
        },
    }


def _bound_and_parallel(tmp_path: Path, stage_code: str = "a1d"):
    plan = _plan(tmp_path, stage_code)
    parent = _parent()
    parent_hash = audit.canonical_sha256(parent)
    bound = audit.build_bound_manifest(
        stage_code=stage_code,
        study_id=f"audit-{stage_code}",
        git_commit="a" * 40,
        parent=parent,
        expected_parent_hash=parent_hash,
        plan=plan,
        seed_registry=_seed_registry(plan),
        decision_hashes=plan["decision_manifest_hashes"],
    )
    roster = parallel.build_parallel_roster(
        stage_code=stage_code,
        bound_manifest=bound,
        parent_manifest=parent,
        scientific_plan=plan,
    )
    parallel_plan = parallel.build_parallel_plan(roster)
    return parent, bound, plan, roster, parallel_plan


@pytest.mark.parametrize(
    ("stage_code", "cell_count", "seeds", "learner_counts"),
    [
        ("a1d", 24, {359, 373}, {4}),
        ("a3k", 3, {347}, {4}),
        ("a3r0", 18, {359, 373}, {4}),
        ("a4d", 48, {2069, 2081}, {1, 4}),
    ],
)
def test_registered_initial_stage_counts_seeds_and_m(
    tmp_path, stage_code, cell_count, seeds, learner_counts
):
    plan = _plan(tmp_path, stage_code)
    assert len(plan["cells"]) == cell_count
    assert {cell["seed"] for cell in plan["cells"]} == seeds
    assert {cell["M"] for cell in plan["cells"]} == learner_counts
    assert plan["loss_blind"] is True
    assert plan["decision_manifest_hashes"] == {}


def test_a1_confirmation_is_exact_fixed_and_independently_tuned_registry(tmp_path):
    plan = _plan(
        tmp_path,
        "a1c",
        decision_path=_selection(tmp_path, stage="A1", selected_etas=_a1_selected()),
    )
    assert len(plan["cells"]) == 64
    assert {cell["seed"] for cell in plan["cells"]} == {
        383,
        397,
        409,
        421,
        433,
        443,
        457,
        461,
    }
    for cells in _cells_by_block(plan).values():
        assert {cell["analysis_role"] for cell in cells} == {
            "fixed_control",
            "fixed_method",
            "tuned_control",
            "tuned_method",
        }
        assert {
            cell["eta"] for cell in cells if cell["pair_key"] == "fixed"
        } == {audit.FIXED_ETA}
        assert len(
            {cell["eta"] for cell in cells if cell["pair_key"] == "tuned"}
        ) == 2


def _cells_by_block(plan: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for cell in plan["cells"]:
        result.setdefault(cell["randomization"]["block_id"], []).append(cell)
    return result


def test_a3_h512_keeps_exact_tokens_with_one_256_step_terminal_window(tmp_path):
    plan = _plan(tmp_path, "a3r0")
    for cell in plan["cells"]:
        target = cell["target_work"]
        if cell["H"] == 512:
            assert target == {
                "tokens": 655_360,
                "microsteps": 5_120,
                "outer_steps": 12,
                "per_fragment_outer_steps": 3,
                "learner_count": 4,
                "quorum": 4,
                "learner_steps_per_learner": 1_280,
                "terminal_partial_window_registered": True,
                "terminal_partial_window_microsteps": 256,
            }
            assert "--allow-terminal-partial-fixed-window" in cell["command"]
        else:
            assert target["terminal_partial_window_registered"] is False
            assert "--allow-terminal-partial-fixed-window" not in cell["command"]

    assert learner.barrier_round_closure_target(
        barrier_sync=True,
        shutdown=False,
        steps_total=1280,
        max_local_steps=1280,
        fixed_window_steps=512,
        fragment_count=4,
        global_step=8,
        fixed_window_snapshots=[{}] * 4,
        allow_terminal_partial_fixed_window=True,
    ) == 12
    assert learner.barrier_round_closure_target(
        barrier_sync=True,
        shutdown=False,
        steps_total=1280,
        max_local_steps=1280,
        fixed_window_steps=512,
        fragment_count=4,
        global_step=8,
        fixed_window_snapshots=[{}] * 4,
        allow_terminal_partial_fixed_window=False,
    ) == 8


def test_audit_evaluation_modes_and_finite_kernel_registry_are_exact(tmp_path):
    expected_modes = {
        "a1d": {"development_endpoint"},
        "a1c": {"confirmation_audit_pending"},
        "a3k": {"capture_only_no_endpoint"},
        "a3r0": {"development_prediction_pending"},
        "a4d": {"development_endpoint"},
        "a4c": {"confirmation_audit_pending"},
    }
    decisions = {
        "a1c": _selection(tmp_path, stage="A1", selected_etas=_a1_selected()),
        "a4c": _selection(tmp_path, stage="A4", selected_etas=_a4_selected()),
    }
    captures = []
    for stage_code, modes in expected_modes.items():
        plan = _plan(
            tmp_path,
            stage_code,
            decision_path=decisions.get(stage_code),
        )
        assert {cell["evaluation_mode"] for cell in plan["cells"]} == modes
        captures.extend(
            (stage_code, cell["H"], cell["seed"], cell["mu"], cell["eta"])
            for cell in plan["cells"]
            if cell["finite_kernel_capture_required"]
        )
    assert len(captures) == 7
    assert {row[1] for row in captures} == {8, 16, 64, 256, 512}
    assert all(row[3] == 0.0 and row[4] == pytest.approx(0.021875) for row in captures)
    assert {row for row in captures if row[0] == "a3k"} == {
        ("a3k", 16, 347, 0.0, 0.021875),
        ("a3k", 64, 347, 0.0, 0.021875),
        ("a3k", 256, 347, 0.0, 0.021875),
    }
def test_unregistered_partial_fixed_window_fails_closed(tmp_path):
    args = audit._runtime_namespace(_runtime(tmp_path))
    with pytest.raises(phase.PhaseMapError, match="not divisible by H=512"):
        phase.compare_command(
            args,
            h=512,
            mu=0.0,
            eta=0.021875,
            seed=359,
            training_seed=359359,
            learner_count=4,
        )


def test_a4_precision_expansion_requires_separate_exact_sign_blind_trigger(tmp_path):
    selection = _selection(tmp_path, stage="A4", selected_etas=_a4_selected())
    with pytest.raises(audit.AuditContractError, match="trigger manifest"):
        _plan(tmp_path, "a4x", decision_path=selection)

    plan = _plan(
        tmp_path,
        "a4x",
        decision_path=selection,
        precision_trigger_path=_precision_trigger(tmp_path),
    )
    assert len(plan["cells"]) == 48
    assert {cell["seed"] for cell in plan["cells"]} == {2099, 2111, 2113}
    assert set(plan["decision_manifest_hashes"]) == {"selection", "precision_trigger"}

    malformed = json.loads(selection.read_text())
    malformed["precision_expansion"] = {"triggered_sign_blindly": True}
    bad_path = _write_json(tmp_path / "bad-a4-selection.json", malformed)
    with pytest.raises(audit.AuditContractError, match="field set differs"):
        _plan(
            tmp_path,
            "a4x",
            decision_path=bad_path,
            precision_trigger_path=_precision_trigger(tmp_path),
        )


def test_cumulative_manifest_preserves_parent_prefix_and_results_exactly(tmp_path):
    parent, bound, plan, _roster, _parallel_plan = _bound_and_parallel(tmp_path)
    assert bound["expected_cells"][: len(parent["expected_cells"])] == parent[
        "expected_cells"
    ]
    assert bound["results"] == parent["results"]
    assert len(bound["expected_cells"]) == len(parent["expected_cells"]) + 24
    assert bound["lineage"]["cumulative_parent_cells"] == 1
    assert bound["lineage"]["append_only_suffix_cells"] == 24
    assert set(bound["frozen"]["cell_command_hashes"]) == {
        cell["cell_id"] for cell in bound["expected_cells"]
    }
    assert plan["randomization_plan_hash"] == bound["frozen"][
        "randomization_plan_hash"
    ]


def test_six_arm_seed_blocks_split_into_deterministic_loss_blind_batches(tmp_path):
    _parent_value, _bound_value, _scientific, roster, plan = _bound_and_parallel(
        tmp_path
    )
    assert len(plan["waves"]) == 4
    for wave in plan["waves"]:
        assert len(wave["assigned_cells_in_dispatch_order"]) == 6
        assert wave["dispatch_batch_count"] == 2
        assert wave["whole_group_loss_blind_until_terminal"] is True
        by_batch: dict[int, list[dict[str, object]]] = {}
        for row in wave["assigned_cells_in_dispatch_order"]:
            by_batch.setdefault(row["dispatch_batch_index"], []).append(row)
        assert sorted(map(len, by_batch.values())) == [3, 3]
        assert all(
            len({row["logical_slot"] for row in rows}) == len(rows)
            for rows in by_batch.values()
        )
    assert parallel.canonical_json(parallel.build_parallel_plan(roster)) == (
        parallel.canonical_json(plan)
    )
    assert plan["capacity"]["zone_fallback_order"] == list(
        parallel.AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER
    )
    assert plan["capacity"]["allowed_zones"] == list(
        parallel.AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER
    )


def test_audit_concurrent_block_binding_is_prospective_deterministic_and_disjoint(
    tmp_path,
):
    _parent_value, _bound_value, _scientific, roster, plan = _bound_and_parallel(
        tmp_path
    )
    assert plan["schema"] == "yeto_parallel_plan_v4"
    assert plan["logical_slots"] == list(parallel.AUDIT_135M_LOGICAL_SLOTS)
    binding = plan["audit_concurrent_block_binding"]
    assert binding["amendment_raw_sha256"] == parallel.sha256_file(
        Path("docs/AMENDMENT-audit-135m-concurrent-blocks.md")
    )
    assert binding["block_width"] == 3
    assert binding["maximum_concurrent_blocks"] == 5
    assert plan["capacity"]["maximum_concurrent_scientific_cells"] == 15
    assert plan["capacity"]["maximum_campaign_owned_attached_a100s"] == 16

    contract_hash = roster["audit_135m_design_contract_hash"]
    batch_slots = parallel.AUDIT_135M_LOGICAL_SLOTS[:12]
    lanes = [
        parallel.audit_concurrent_block_slots(
            contract_hash=contract_hash,
            block_index=index,
            available_slots=batch_slots,
        )
        for index in range(4)
    ]
    assert all(len(lane) == 3 for lane in lanes)
    assert len({slot for lane in lanes for slot in lane}) == 12
    assert lanes == [
        parallel.audit_concurrent_block_slots(
            contract_hash=contract_hash,
            block_index=index,
            available_slots=reversed(batch_slots),
        )
        for index in range(4)
    ]
    assert [tuple(wave["available_slot_set"]) for wave in plan["waves"]] == lanes


@pytest.mark.parametrize("stage_code", ["a1d", "a3k", "a3r0", "a4d"])
def test_all_initial_135m_stages_inherit_concurrent_block_binding(
    tmp_path, stage_code
):
    _parent_value, _bound_value, _scientific, roster, plan = _bound_and_parallel(
        tmp_path, stage_code
    )
    assert plan["schema"] == "yeto_parallel_plan_v4"
    contract_hash = roster["audit_135m_design_contract_hash"]
    for batch_start in range(0, len(plan["waves"]), 5):
        waves = plan["waves"][batch_start : batch_start + 5]
        batch_slots = tuple(waves[0]["audit_registered_batch_slot_set"])
        lanes = [
            parallel.audit_concurrent_block_slots(
                contract_hash=contract_hash,
                block_index=batch_start + offset,
                available_slots=batch_slots,
            )
            for offset in range(len(waves))
        ]
        assert len(batch_slots) == 3 * len(waves)
        assert len({slot for lane in lanes for slot in lane}) == len(batch_slots)
        assert [tuple(wave["available_slot_set"]) for wave in waves] == lanes


def test_audit_capacity_allows_fifteen_1g_vms_but_not_sixteen():
    rows = [
        {
            "creation_timestamp": "2026-07-17T00:00:00Z",
            "deletion_completed_at_utc": "2026-07-17T01:00:00Z",
            "machine_type": "a2-highgpu-1g",
            "a100_count": 1,
        }
        for _ in range(15)
    ]
    parallel._validate_generation_capacity(rows, stage_code="a1d")
    with pytest.raises(parallel.LifecycleError, match="stage VM limit|16 A100s"):
        parallel._validate_generation_capacity(
            [*rows, dict(rows[-1])], stage_code="a1d"
        )


def test_concurrent_attempt_schedule_allows_disjoint_blocks_to_overlap(tmp_path):
    _parent_value, _bound_value, scientific, roster, plan = _bound_and_parallel(
        tmp_path
    )
    scientific_by_id = {cell["cell_id"]: cell for cell in scientific["cells"]}
    batch_slots = parallel.AUDIT_135M_LOGICAL_SLOTS[:12]
    attempts = []
    for block_index, planned_wave in enumerate(plan["waves"]):
        lane = parallel.audit_concurrent_block_slots(
            contract_hash=roster["audit_135m_design_contract_hash"],
            block_index=block_index,
            available_slots=batch_slots,
        )
        wave = parallel.wave_for_retry(
            plan,
            roster,
            planned_wave["group_id"],
            1,
            available_slots=lane,
        )
        for assignment in wave["assigned_cells_in_dispatch_order"]:
            cell = scientific_by_id[assignment["cell_id"]]
            learner_count = parallel._cell_learner_count(cell)
            batch = assignment["dispatch_batch_index"]
            order = assignment["batch_launch_order_index"]
            dispatch_second = batch * 30 + order
            attempts.append(
                {
                    "status": "COMPLETED",
                    "failure_reason": None,
                    "loss": 2.0,
                    "analysis_loss": 2.0,
                    "analysis_loss_kind": "finite_endpoint_nll",
                    "divergence_retained": False,
                    "attempt_id": f"{cell['cell_id']}-attempt-1",
                    "cell_id": cell["cell_id"],
                    "attempt": 1,
                    "group_id": planned_wave["group_id"],
                    "retry_round": 1,
                    "actual_wave_index": block_index,
                    "concurrent_batch_index": 0,
                    "concurrent_batch_slot_set": list(batch_slots),
                    "time_block_index": planned_wave["time_block_index"],
                    "retry_time_block_index": None,
                    "available_slot_set": list(lane),
                    "dispatch_batch_index": batch,
                    "batch_launch_order_index": order,
                    "launch_order_index": assignment["launch_order_index"],
                    "logical_slot": assignment["logical_slot"],
                    "m": learner_count,
                    "gpu_slots": 1,
                    "learner_gpu_slot_map": {
                        str(index): 0 for index in range(learner_count)
                    },
                    "maximum_learners_per_gpu": learner_count,
                    "pairing_identity_hash": cell["pairing_identity_hash"],
                    "attempt_prefix": f"gs://example/{cell['cell_id']}/attempt-1/",
                    "fresh_start": {
                        "same_frozen_initial_model": True,
                        "same_seed_and_data_order": True,
                        "same_command_and_work_budget": True,
                        "resumed": False,
                        "prior_optimizer_state_used": False,
                        "prior_checkpoint_used": False,
                        "prior_tape_used": False,
                        "prior_result_used": False,
                    },
                    "retry_of": None,
                    "retry_reason": None,
                    "retry_authorization": None,
                    "vm_ready_at": "2026-07-17T00:00:00Z",
                    "dispatched_at": (
                        f"2026-07-17T00:00:{dispatch_second:02d}Z"
                    ),
                    "scientific_started_at": (
                        f"2026-07-17T00:00:{dispatch_second + 5:02d}Z"
                    ),
                    "scientific_ended_at": (
                        f"2026-07-17T00:00:{dispatch_second + 20:02d}Z"
                    ),
                    "wave_terminal_prefix_sealed_at": (
                        "2026-07-17T00:01:00Z"
                    ),
                }
            )
    attempts.sort(key=lambda row: (row["actual_wave_index"], row["launch_order_index"]))
    final = parallel.validate_attempt_schedule(
        attempts=attempts,
        roster=roster,
        plan=plan,
        parallel_digest=parallel.parallel_plan_hash(plan),
    )
    assert set(final) == {cell["cell_id"] for cell in roster["launch_cells"]}

    tampered = [dict(row) for row in attempts]
    tampered[0]["available_slot_set"] = list(
        parallel.audit_concurrent_block_slots(
            contract_hash=roster["audit_135m_design_contract_hash"],
            block_index=1,
            available_slots=batch_slots,
        )
    )
    with pytest.raises(
        parallel.ScheduleError, match="deterministic|slot swap|binding fields"
    ):
        parallel.validate_attempt_schedule(
            attempts=tampered,
            roster=roster,
            plan=plan,
            parallel_digest=parallel.parallel_plan_hash(plan),
        )


def test_audit_runtime_authorization_binds_ceiling_and_all_launch_hashes(tmp_path):
    _parent_value, bound, scientific, roster, plan = _bound_and_parallel(tmp_path)
    authorization = {
        "schema": "yeto_audit_135m_runtime_authorization_v1",
        "launch_authorized": True,
        "stage_code": "a1d",
        "roster_hash": parallel.roster_hash(roster),
        "parallel_plan_hash": parallel.parallel_plan_hash(plan),
        "bound_manifest_canonical_sha256": parallel.canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific["randomization_plan_hash"],
        "audit_135m_design_contract_hash": roster[
            "audit_135m_design_contract_hash"
        ],
        "hard_ceiling_usd": 75.0,
        "spot_only": True,
        "maximum_attached_a100_equivalent": 16,
        "max_idle_before_science_seconds": 600,
    }
    assert parallel.multiseed_runtime_authorization_hash(
        stage_code="a1d",
        design_contract_hash=roster["audit_135m_design_contract_hash"],
        roster_digest=parallel.roster_hash(roster),
        parallel_digest=parallel.parallel_plan_hash(plan),
        bound_digest=parallel.canonical_sha256(bound),
        scientific_digest=scientific["randomization_plan_hash"],
        hard_ceiling_usd=75.0,
        authorization=authorization,
    ) == parallel.canonical_sha256(authorization)
    with pytest.raises(parallel.LifecycleError, match="exact runtime authorization"):
        parallel.multiseed_runtime_authorization_hash(
            stage_code="a1d",
            design_contract_hash=roster["audit_135m_design_contract_hash"],
            roster_digest=parallel.roster_hash(roster),
            parallel_digest=parallel.parallel_plan_hash(plan),
            bound_digest=parallel.canonical_sha256(bound),
            scientific_digest=scientific["randomization_plan_hash"],
            hard_ceiling_usd=75.0,
            authorization={**authorization, "hard_ceiling_usd": 75.01},
        )


def test_frozen_authority_hash_drift_fails_closed(tmp_path, monkeypatch):
    changed = tmp_path / "changed-prereg.json"
    changed.write_bytes(audit.PREREG_JSON.read_bytes() + b"\n")
    monkeypatch.setattr(audit, "PREREG_JSON", changed)
    with pytest.raises(audit.AuditContractError, match="JSON bytes differ"):
        audit.load_authority()


def test_a1_paired_boundary_extension_uses_corresponding_geometric_direction(tmp_path):
    selected = _a1_selected()
    selected["H16_mu0"] = audit.A1_GRIDS[(16, 0.0)][0]
    plan = _plan(
        tmp_path,
        "a1x",
        decision_path=_selection(
            tmp_path,
            stage="A1",
            selected_etas=selected,
            boundary_status="REQUIRED",
        ),
    )
    assert len(plan["cells"]) == 4
    for cells in _cells_by_block(plan).values():
        assert {cell["analysis_role"] for cell in cells} == {
            "boundary_control",
            "boundary_method",
        }
        by_role = {cell["analysis_role"]: cell for cell in cells}
        assert by_role["boundary_control"]["eta"] == pytest.approx(0.0109375)
        assert by_role["boundary_method"]["eta"] == pytest.approx(0.00068359375)


def test_selection_eta_outside_registered_grid_or_single_extension_fails(tmp_path):
    selected = _a1_selected()
    selected["H16_mu0"] = 0.019
    with pytest.raises(audit.AuditContractError, match="outside its registry"):
        _plan(
            tmp_path,
            "a1c",
            decision_path=_selection(
                tmp_path,
                stage="A1",
                selected_etas=selected,
            ),
        )
