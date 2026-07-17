from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import audit_135m_analysis as analysis


def _curve_fixture() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    cells: dict[str, dict[str, object]] = {}
    final: dict[str, dict[str, object]] = {}
    losses = {
        0.5: (2.2, 2.3, 2.4),
        1.0: (2.0, 2.1, 2.2),
        2.0: (2.1, 2.2, 2.3),
        4.0: (2.4, 2.5, 2.6),
    }
    for eta, eta_losses in losses.items():
        for seed, loss in zip((347, 359, 373), eta_losses):
            cell_id = f"cell-{eta}-{seed}"
            cells[cell_id] = {
                "cell_id": cell_id,
                "h": 16,
                "m": 4,
                "mu": 0.0,
                "eta": eta,
                "seed": seed,
                "audit_stage": None if seed == 347 else "A1",
                "audit_phase": None if seed == 347 else "development_initial",
            }
            final[cell_id] = {
                "cell_id": cell_id,
                "attempt": 1,
                "status": "COMPLETED",
                "loss": loss,
            }
    return cells, final


def test_curve_selection_uses_pooled_mean_and_registered_neighbors() -> None:
    cells, final = _curve_fixture()
    evidence = analysis._curve_evidence(
        label="H16_mu0",
        h=16,
        m=4,
        mu=0.0,
        initial_grid=(1.0, 2.0, 4.0),
        allowed_grid=(0.5, 1.0, 2.0, 4.0),
        seeds=(347, 359, 373),
        cells=cells,
        final=final,
        audit_stage="A1",
        phases={"development_initial"},
    )
    assert evidence["selected_eta"] == 1.0
    assert evidence["initial_boundary_winner"] is True
    assert evidence["final_interior_with_worse_neighbors"] is True
    assert evidence["lower_neighbor"]["eta"] == 0.5
    assert evidence["upper_neighbor"]["eta"] == 2.0


def test_scientific_divergence_is_positive_infinity_for_selection() -> None:
    cells, final = _curve_fixture()
    final["cell-1.0-359"].update(status="DIVERGED", loss=None)
    evidence = analysis._curve_evidence(
        label="H16_mu0",
        h=16,
        m=4,
        mu=0.0,
        initial_grid=(1.0, 2.0, 4.0),
        allowed_grid=(0.5, 1.0, 2.0, 4.0),
        seeds=(347, 359, 373),
        cells=cells,
        final=final,
        audit_stage="A1",
        phases={"development_initial"},
    )
    point = next(row for row in evidence["sampled_points"] if row["eta"] == 1.0)
    assert point["pooled_mean"] is None
    assert point["pooled_mean_kind"] == "positive_infinity_scientific_divergence"
    assert evidence["selected_eta"] == 2.0


@pytest.mark.parametrize(
    ("fixed", "tuned", "mean", "expected"),
    [
        ([0.01, 0.02], [-0.005, 0.005], 0.0, "COLLAPSES_WITH_TUNING"),
        ([-0.01, 0.02], [-0.005, 0.005], 0.0, "FIXED_NOT_REPLICATED"),
        ([0.01, 0.02], [0.011, 0.03], 0.02, "SURVIVES_TUNING"),
        ([0.01, 0.02], [-0.03, -0.011], -0.02, "REVERSES_WITH_TUNING"),
    ],
)
def test_registered_classification_table(fixed, tuned, mean, expected) -> None:
    assert (
        analysis._classification(
            fixed_interval=fixed,
            tuned_interval=tuned,
            tuned_mean=mean,
        )
        == expected
    )


def test_holm_intervals_use_stable_p_value_then_label_order() -> None:
    summaries = {
        "b": {
            "student_t_two_sided_p": 0.1,
            "finite_inference_available": True,
            "student_t_df": 4,
            "standard_error": 0.01,
            "mean": 0.0,
        },
        "a": {
            "student_t_two_sided_p": 0.1,
            "finite_inference_available": True,
            "student_t_df": 4,
            "standard_error": 0.01,
            "mean": 0.0,
        },
    }
    adjusted = analysis.holm_intervals(summaries)
    assert adjusted["a"]["holm_rank"] == 1
    assert adjusted["b"]["holm_rank"] == 2
    assert adjusted["a"]["half_width"] > adjusted["b"]["half_width"]


def _a4_precision_manifest(sign: float) -> dict[str, object]:
    cells = []
    results = []
    seed_offsets = {2083: -0.004, 2087: 0.0, 2089: 0.004}
    for m in (1, 4):
        for h, method_mu in ((16, 0.9), (256, 0.5)):
            for seed, offset in seed_offsets.items():
                base = 2.0 + m * 0.01 + h * 0.0001 + offset
                role_losses = {
                    "fixed_control": base,
                    "fixed_method": base + sign * (0.02 + offset),
                    "tuned_control": base + 0.001,
                    "tuned_method": base + 0.001 + sign * (0.003 + offset),
                }
                for role, loss in role_losses.items():
                    cell_id = f"M{m}-H{h}-s{seed}-{role}"
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "audit_stage": "A4",
                            "audit_phase": "confirmation_initial",
                            "m": m,
                            "h": h,
                            "seed": seed,
                            "mu": method_mu if role.endswith("method") else 0.0,
                            "analysis_role": role,
                        }
                    )
                    results.append(
                        {
                            "cell_id": cell_id,
                            "attempt": 1,
                            "status": "COMPLETED",
                            "loss": loss,
                        }
                    )
    return {"expected_cells": cells, "results": results}


def test_a4_precision_trigger_is_sign_blind(tmp_path: Path) -> None:
    outputs = []
    for label, sign in (("positive", 1.0), ("negative", -1.0)):
        manifest = tmp_path / f"{label}.json"
        manifest.write_text(json.dumps(_a4_precision_manifest(sign)))
        evidence = tmp_path / f"{label}-evidence.json"
        trigger = tmp_path / f"{label}-trigger.json"
        result = analysis.precision_a4(
            argparse.Namespace(
                phase_manifest=manifest,
                evidence_output=evidence,
                trigger_output=trigger,
                sealed_at_utc="2026-07-17T01:00:00Z",
            )
        )
        outputs.append((result, json.loads(evidence.read_text())))
    assert outputs[0][0]["expansion_required"] == outputs[1][0]["expansion_required"]
    positive_widths = outputs[0][1]["widths"]
    negative_widths = outputs[1][1]["widths"]
    assert positive_widths.keys() == negative_widths.keys()
    for key in positive_widths:
        assert positive_widths[key]["half_width"] == pytest.approx(
            negative_widths[key]["half_width"]
        )
