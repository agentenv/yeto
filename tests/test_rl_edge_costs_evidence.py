"""5.7: the published H100 edge-cost table parses and matches its evidence."""
from pathlib import Path

from yeto.rl.engine.recommend import edge_costs_from_table

TABLE = Path(__file__).resolve().parents[1] / "openspec/changes/rl-infra-spec/evidence/edge-costs.json"
PH = "sha256:6039d75ff8514328f4edc2e2d341f2e20f230b299b905a69c6a463ea499e12f3"


def test_edge_cost_table_parses():
    costs = edge_costs_from_table(TABLE)
    assert set(costs) == {(PH, "T2R1S1", "T2R2S0"), (PH, "T2R2S0", "T2R1S1")}
    up = costs[(PH, "T2R1S1", "T2R2S0")]
    dn = costs[(PH, "T2R2S0", "T2R1S1")]
    assert up.cost_lower_s == 146.544 and abs(up.cost_upper_s - 149.744 * 1.2) < 1e-3
    assert dn.cost_lower_s == 3.203 and abs(dn.cost_upper_s - 5.304 * 1.2) < 1e-3
    assert up.recovery_upper_s > 0 and dn.recovery_upper_s > 0
