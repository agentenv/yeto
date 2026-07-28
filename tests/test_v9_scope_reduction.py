import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "experiment-specs/outer-mup-v9-7b-scope-reduction.json"
GATESIM = ROOT / "experiment-specs/outer-mup-v9-7b-scope-reduction-gatesim.json"


def test_scope_reduction_is_symmetric_and_gatesimmed():
    amendment = json.loads(AMENDMENT.read_text())
    gatesim = json.loads(GATESIM.read_text())
    reduction = amendment["reduction"]
    assert amendment["pre_outcome"] is True
    assert amendment["verification_loss_seen"] is False
    assert amendment["inventory"]["completed_cells"] == 0
    assert amendment["sealed_predictions"] == {
        "path": "experiment-specs/outer-mup-v9-sealed-predictions.json",
        "sha256": "97e02dcad63782978ac51b320621e5a681236518cb0d5db19454b8981549ca9c",
        "untouched": True,
        "targets_referenced_by_stage_7b": ["mu0", "raw"],
    }
    assert len(reduction["retained_cell_ids"]) == 6
    assert all(
        cell_id.endswith("seed907") for cell_id in reduction["retained_cell_ids"]
    )
    assert len(reduction["removed_cell_ids"]) == 6
    assert all(cell_id.endswith("seed901") for cell_id in reduction["removed_cell_ids"])
    assert gatesim["status"] == "PASS"
    assert gatesim["selected_layout"] == "symmetric_6_single_seed"
    selected = gatesim["layouts"][gatesim["selected_layout"]]
    assert selected["cell_count"] == 6
    assert selected["P_eval"] >= gatesim["minimum_required_P_eval"]
    assert (
        selected["P_pass_given_evaluable"]
        >= gatesim["minimum_required_P_pass_given_evaluable"]
    )
