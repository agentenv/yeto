from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import compare_diloco as compare
from scripts import run_parallel_phase_map as parallel
from scripts import run_phase_map as phase


def _args(tmp_path: Path, design: str, *, token_budget: int | None = None):
    if token_budget is None:
        token_budget = 655_360 if design == "e1" else 1_048_576
    argv = [
        "--study-id",
        f"bp-v2-{design}",
        "--study-phase",
        f"best_paper_v2_{design}",
        "--run-dir",
        str(tmp_path / design),
        "--artifact-uri",
        f"gs://local-contract/{design}",
        "--git-commit",
        "a" * 40,
        "--image-digest",
        "b" * 64,
        "--image-numeric-id",
        "7290368630472593484",
        "--model-path",
        str(tmp_path / "model"),
        "--model-revision",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "--data",
        str(tmp_path / "data.parquet"),
        "--provider-evidence",
        str(tmp_path / "provider.json"),
        "--h",
        "16,256",
        "--mu",
        "0,.9",
        "--eta",
        ".002734375,.0109375,.021875,.04375",
        "--order-seed",
        "20260716",
        "--resource-class",
        "a2-highgpu-4g",
        "--token-budget",
        str(token_budget),
    ]
    if design == "e4":
        argv.extend(("--m", "1,16"))
    for shuffle_seed, training_seed in phase.BEST_PAPER_V2_FRESH_SEED_PAIRS:
        argv.extend(("--seed-pair", f"{shuffle_seed}:{training_seed}"))
    return phase.build_parser().parse_args(argv)


def _bound(args, plan):
    extra_row_hashes = {
        seed: f"{index + 10:064x}"
        for index, (seed, _training) in enumerate(
            phase.BEST_PAPER_V2_FRESH_SEED_PAIRS[1:]
        )
    }
    extra_index_hashes = {
        seed: f"{index + 30:064x}"
        for index, (seed, _training) in enumerate(
            phase.BEST_PAPER_V2_FRESH_SEED_PAIRS[1:]
        )
    }
    return phase.build_bound_manifest(
        args,
        plan,
        model_hash="1" * 64,
        data_hash="2" * 64,
        train_rows_hash="3" * 64,
        development_eval_rows_hash="4" * 64,
        development_eval_packed_hash="5" * 64,
        development_eval_example_ids_hash="6" * 64,
        development_eval_token_ids_hash="7" * 64,
        development_eval_source_indices_hash="8" * 64,
        audit_eval_rows_hash="9" * 64,
        audit_eval_packed_hash="a" * 64,
        audit_eval_example_ids_hash="b" * 64,
        audit_eval_token_ids_hash="c" * 64,
        audit_eval_source_indices_hash="d" * 64,
        audit_access_policy_hash="e" * 64,
        train_pool_source_indices_hash="f" * 64,
        train_source_indices_hash="0" * 64,
        additional_train_rows_hashes=extra_row_hashes,
        additional_train_source_indices_hashes=extra_index_hashes,
    )


def _design(tmp_path: Path, design: str):
    args = _args(tmp_path, design)
    scientific = phase.build_plan(args)
    bound = _bound(args, scientific)
    parent = {"expected_cells": [], "results": []}
    roster = parallel.build_parallel_roster(
        stage_code=design,
        bound_manifest=bound,
        parent_manifest=parent,
        scientific_plan=scientific,
    )
    plan = parallel.build_parallel_plan(roster)
    return args, parent, bound, scientific, roster, plan


def _by_block(plan):
    blocks = {}
    for cell in plan["cells"]:
        blocks.setdefault(cell["randomization"]["block_id"], []).append(cell)
    return blocks


def test_e1_is_eight_fresh_seeds_times_two_atomic_mixed_eta_blocks(tmp_path):
    args = _args(tmp_path, "e1")
    plan = phase.build_plan(args)
    blocks = _by_block(plan)

    assert plan["schema"] == "yeto_multiseed_phase_map_randomization_v2"
    assert plan["atomic_wave_unit"] == "paired_seed_block"
    assert plan["block_fields"] == ["H", "seed", "M"]
    assert len(plan["cells"]) == 48
    assert len(blocks) == 16
    assert {cell["seed"] for cell in plan["cells"]} == {
        seed for seed, _training in phase.BEST_PAPER_V2_FRESH_SEED_PAIRS
    }

    expected = phase.best_paper_v2_arms()
    for block_cells in blocks.values():
        phase.validate_seed_block_cells(block_cells)
        assert {cell["M"] for cell in block_cells} == {4}
        h = block_cells[0]["H"]
        assert {(cell["mu"], cell["eta"]) for cell in block_cells} == set(
            expected[h]
        )
        assert len({cell["pairing_identity_hash"] for cell in block_cells}) == 1
        assert len({cell["pairing_command_hash"] for cell in block_cells}) == 1
        identity = plan["seed_blocks"][block_cells[0]["randomization"]["block_id"]]
        assert identity["data_order"]["shuffle_seed"] == block_cells[0]["seed"]
        assert identity["initialization"]["rank0_process_seeds"]["3"] == (
            block_cells[0]["training_seed"] + 3 * 1009
        )


def test_explicit_block_arm_tuples_are_order_independent_and_canonicalized(tmp_path):
    args = _args(tmp_path, "e1")
    args.block_arm = [
        (256, 0.9, 0.04375),
        (16, 0.9, 0.021875),
        (256, 0.0, 0.04375),
        (16, 0.0, 0.021875),
        (256, 0.9, 0.0109375),
        (16, 0.9, 0.002734375),
    ]
    plan = phase.build_plan(args)
    assert len(plan["cells"]) == 48
    for block in _by_block(plan).values():
        assert {(cell["mu"], cell["eta"]) for cell in block} == set(
            phase.best_paper_v2_arms()[block[0]["H"]]
        )


@pytest.mark.parametrize("drift", ["training_seed", "pairing_hash", "control"])
def test_pairing_invariants_fail_closed_on_one_arm_drift(tmp_path, drift):
    plan = phase.build_plan(_args(tmp_path, "e1"))
    block = copy.deepcopy(next(iter(_by_block(plan).values())))
    if drift == "training_seed":
        block[0]["training_seed"] += 1
    elif drift == "pairing_hash":
        block[0]["pairing_command_hash"] = "0" * 64
    else:
        block[0]["paired_control_id"] = block[0]["cell_id"]
    with pytest.raises(phase.PhaseMapError, match="paired block"):
        phase.validate_seed_block_cells(block)


def test_e4_m_is_a_cell_dimension_and_wires_exact_quorum_and_work(tmp_path):
    plan = phase.build_plan(_args(tmp_path, "e4"))
    assert len(plan["cells"]) == 96
    assert len(_by_block(plan)) == 32
    assert {cell["M"] for cell in plan["cells"]} == {1, 16}

    for learner_count in (1, 16):
        for h in (16, 256):
            cell = next(
                cell
                for cell in plan["cells"]
                if cell["M"] == learner_count and cell["H"] == h
            )
            command = cell["command"]
            assert command[command.index("--settings") + 1] == f"m{learner_count}"
            delays = command[command.index("--learner-push-delay-ms") + 1].split(",")
            assert delays == ["0"] * learner_count
            assert cell["target_work"]["learner_count"] == learner_count
            assert cell["target_work"]["quorum"] == learner_count
            assert cell["target_work"]["outer_steps"] == (
                cell["target_work"]["learner_steps_per_learner"] // h * 4
            )

    for learner_count in (1, 16):
        arm = replace(
            compare.PRESETS[f"m{learner_count}"], strict_quorum=True
        )
        command = compare.syncer_command(arm, 1234, tmp_path, total_steps=8)
        assert command[command.index("--learners") + 1] == str(learner_count)
        assert command[command.index("--quorum") + 1] == str(learner_count)
        assert "--strict-quorum" in command


def test_m16_gpu_packing_is_round_robin_for_one_through_four_gpus():
    expected_maximum = {1: 16, 2: 8, 3: 6, 4: 4}
    for gpu_slots, maximum in expected_maximum.items():
        packing = compare.learner_gpu_packing(16, gpu_slots)
        assert sorted(learner for ids in packing.values() for learner in ids) == list(
            range(16)
        )
        assert max(map(len, packing.values())) == maximum
        assert parallel.learner_gpu_slot_map(16, gpu_slots) == {
            str(learner): learner % gpu_slots for learner in range(16)
        }


def test_e4_rejects_nonintegral_m16_h256_fixed_window_work(tmp_path):
    with pytest.raises(
        phase.PhaseMapError,
        match="not divisible by H=256 for M=16",
    ):
        phase.build_plan(_args(tmp_path, "e4", token_budget=655_360))


def test_parallel_binder_keeps_seed_blocks_atomic_and_deterministic(tmp_path):
    _args_value, parent, bound, scientific, roster, plan = _design(tmp_path, "e4")
    assert len(plan["waves"]) == 32
    assert all(len(wave["assigned_cells_in_dispatch_order"]) == 3 for wave in plan["waves"])
    assert all(
        wave["runtime_pairing_requirements"]["same_pairing_identity_hash"] is True
        for wave in plan["waves"]
    )
    assert plan["capacity"]["maximum_learners_per_gpu_by_shape"]["16"] == {
        "a2-highgpu-4g": 4,
        "a2-highgpu-1g": 16,
    }

    permuted_bound = copy.deepcopy(bound)
    permuted_bound["expected_cells"].reverse()
    permuted_scientific = copy.deepcopy(scientific)
    permuted_scientific["cells"].reverse()
    rebuilt_roster = parallel.build_parallel_roster(
        stage_code="e4",
        bound_manifest=permuted_bound,
        parent_manifest=parent,
        scientific_plan=permuted_scientific,
    )
    rebuilt_plan = parallel.build_parallel_plan(rebuilt_roster)
    assert parallel.canonical_json(rebuilt_roster) == parallel.canonical_json(roster)
    assert parallel.canonical_json(rebuilt_plan) == parallel.canonical_json(plan)


def test_parallel_binder_rejects_same_eta_legacy_wave_substitution(tmp_path):
    _args_value, _parent, _bound_value, _scientific, roster, _plan = _design(
        tmp_path, "e1"
    )
    broken = copy.deepcopy(roster)
    first_group = broken["launch_cells"][0]["block_id"]
    group = [cell for cell in broken["launch_cells"] if cell["block_id"] == first_group]
    treatment = next(cell for cell in group if cell["mu"] == 0.9)
    treatment["eta"] = next(cell for cell in group if cell["mu"] == 0.0)["eta"]
    with pytest.raises(parallel.ScheduleError, match="mixed-eta"):
        parallel.build_parallel_plan(broken)


def test_multiseed_executor_requires_exact_external_runtime_authorization(tmp_path):
    _args_value, _parent, bound, scientific, roster, plan = _design(tmp_path, "e1")
    roster_hash = parallel.roster_hash(roster)
    registry = parallel.CampaignGenerationRegistry(
        stage_code="e1",
        study_id=bound["study_id"],
        roster_digest=roster_hash,
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    with pytest.raises(parallel.LifecycleError, match="runtime authorization"):
        parallel.ParallelWaveExecutor(
            roster=roster,
            parallel_plan=plan,
            scientific_plan=scientific,
            bound_manifest=bound,
            registry=registry,
            campaign_root=tmp_path / "artifacts",
            backend=object(),
        )

    authorization = {
        "schema": "yeto_multiseed_runtime_authorization_v1",
        "launch_authorized": True,
        "stage_code": "e1",
        "best_paper_v2_design_contract_hash": roster[
            "best_paper_v2_design_contract_hash"
        ],
        "roster_hash": roster_hash,
        "parallel_plan_hash": parallel.parallel_plan_hash(plan),
        "bound_manifest_canonical_sha256": parallel.canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific[
            "randomization_plan_hash"
        ],
    }
    executor = parallel.ParallelWaveExecutor(
        roster=roster,
        parallel_plan=plan,
        scientific_plan=scientific,
        bound_manifest=bound,
        registry=registry,
        campaign_root=tmp_path / "artifacts",
        backend=object(),
        runtime_authorization=authorization,
    )
    assert executor.runtime_authorization_hash == parallel.canonical_sha256(
        authorization
    )
    assert executor._common_bindings()[
        "multiseed_runtime_authorization_hash"
    ] == parallel.canonical_sha256(authorization)

    with pytest.raises(parallel.LifecycleError, match="runtime authorization"):
        parallel.ParallelWaveExecutor(
            roster=roster,
            parallel_plan=plan,
            scientific_plan=scientific,
            bound_manifest=bound,
            registry=registry,
            campaign_root=tmp_path / "artifacts",
            backend=object(),
            runtime_authorization={**authorization, "unreviewed_note": "not exact"},
        )


def test_serial_executor_cannot_bypass_the_multiseed_launch_controller(tmp_path):
    with pytest.raises(
        phase.PhaseMapError,
        match="parallel launch controller.*runtime authorization",
    ):
        phase.execute(_args(tmp_path, "e1"))


def test_divergence_is_retained_as_the_frozen_capped_loss_outcome():
    manifest = {
        "analysis_policy": {
            "divergence_loss_cap": 10.0,
            "divergence_is_outcome": True,
        },
        "expected_cells": [
            {
                "cell_id": "c",
                "expected_learner_count": 1,
                "expected_learner_steps": 8,
            }
        ],
        "results": [
            {
                "cell_id": "c",
                "attempt": 1,
                "status": "DIVERGED",
                "loss": None,
                "analysis_loss": 10.0,
                "failure_reason": "scientific_divergence",
                "observed_work": {"learner_step_counts": {"0": 8}},
                "exit_statuses": {"runner": 0, "syncer": 0, "learners": [0]},
            }
        ],
    }
    phase.validate_campaign_work_evidence(manifest)
    manifest["results"][0]["analysis_loss"] = 9.0
    with pytest.raises(phase.WorkEvidenceError, match="capped divergence"):
        phase.validate_campaign_work_evidence(manifest)


def test_parallel_multiseed_analysis_policy_materializes_the_capped_outcome(tmp_path):
    _args_value, _parent, bound, _scientific, _roster, _plan = _design(
        tmp_path, "e1"
    )
    cap = parallel.multiseed_analysis_loss_cap("e1", bound)
    assert cap == 10.0
    assert parallel.multiseed_analysis_outcome_fields(
        status="DIVERGED", loss=None, divergence_loss_cap=cap
    ) == {
        "analysis_loss": 10.0,
        "analysis_loss_kind": "capped_divergence_endpoint_nll",
        "divergence_retained": True,
    }
    assert parallel.multiseed_analysis_outcome_fields(
        status="COMPLETED", loss=2.5, divergence_loss_cap=cap
    ) == {
        "analysis_loss": 2.5,
        "analysis_loss_kind": "finite_endpoint_nll",
        "divergence_retained": False,
    }
    with pytest.raises(parallel.EvidenceError, match="null raw loss"):
        parallel.multiseed_analysis_outcome_fields(
            status="DIVERGED", loss=2.5, divergence_loss_cap=cap
        )


def test_bound_contract_is_complete_but_cannot_self_authorize_launch(tmp_path):
    args = _args(tmp_path, "e1")
    plan = phase.build_plan(args)
    bound = _bound(args, plan)
    assert bound["status"] == "bound_runtime_contract"
    assert bound["launch_authorized"] is False
    assert len(bound["expected_cells"]) == 48
    assert bound["analysis_policy"] == {
        "divergence_loss_cap": 10.0,
        "divergence_is_outcome": True,
        "silent_divergence_exclusion_forbidden": True,
    }
    assert len(bound["pairing"]["seed_blocks"]) == 16
    json.dumps(bound, allow_nan=False)


def test_bound_contract_requires_both_governing_source_documents(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, "e1")
    plan = phase.build_plan(args)
    monkeypatch.setattr(phase, "REPO_ROOT", tmp_path / "missing-source-root")
    with pytest.raises(phase.PhaseMapError, match="source documents are missing"):
        _bound(args, plan)
