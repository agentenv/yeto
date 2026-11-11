"""``yeto rollout plan``: the offline fleet estimate (yeto.launch.rollout.plan)."""

from __future__ import annotations

import argparse
import json

from yeto.launch.rollout import entry
from yeto.launch.rollout.plan import SOURCES, PlanInputs, estimate


def test_sixteen_island_defaults():
    r = estimate(PlanInputs())
    assert r["cold_start"]["fleet_tb"] == 24.96 and 25 < r["cold_start"]["minutes"] < 40
    assert r["release"]["fleet_tb_per_step"] == 3.31
    assert r["sandbox"] == {"containers": 320, "hosts": 2, "per_host": 160}
    assert r["cost"]["usd_per_h"] == 16 * 24 + 408
    assert set(SOURCES) == set(r["inputs"])  # every input says where its default comes from


def test_a_store_cap_slows_every_island_and_staleness_warns():
    r = estimate(PlanInputs(store_gb_s=2.0, release_mb_s=1700))
    assert r["cold_start"]["pull_mb_s_per_island"] == 125.0  # 2 GB/s shared by 16
    assert r["release"]["policy_age"] > 4 and any("versions old" in w for w in r["warnings"])


def test_group_balance_warnings():
    short = estimate(PlanInputs(groups_per_island_h=6, groups_per_step=32, step_s=300))
    assert any("trainer waits" in w for w in short["warnings"]) and short["groups"]["trainer_busy"] == 0.25
    long = estimate(PlanInputs(groups_per_island_h=60, groups_per_step=8, step_s=300))
    assert any("backlog grows" in w for w in long["warnings"]) and long["groups"]["backlog_growth_per_h"] == 864.0


def test_cli(capsys):
    parser = argparse.ArgumentParser()
    entry.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["rollout", "plan", "--islands", "8", "--json"])
    assert entry.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inputs"]["islands"] == 8 and out["sandbox"]["hosts"] == 1
    args = parser.parse_args(["rollout", "plan"])
    assert entry.main(args) == 0
    text = capsys.readouterr().out
    assert "cold start" in text and "assumption" in text and "measured" in text
