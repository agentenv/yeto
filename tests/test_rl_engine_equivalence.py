"""Tests for scripts/rl_engine_equivalence.py (openspec rl-engine-ports 6.1)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rl_engine_equivalence.py"
_spec = importlib.util.spec_from_file_location("rl_engine_equivalence", _PATH)
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)


def _row(reward=0.5, loss=0.1, gn=1.0, groups=2, tokens=12, h="h"):
    return {"reward_mean": reward, "loss": loss, "grad_norm": gn,
            "completed_groups": groups, "action_tokens": tokens, "post_sync_hash": h}


def test_dry_run_prints_plan_with_rl_engine_flag(tmp_path, capsys):
    assert eq.main(["--dry-run", "--launch-args", "--config x.json",
                    "--out-dir", str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    engines = [s["engine"] for s in plan["steps"]]
    assert engines == ["legacy", "legacy", "legacy", "ports"]
    for s in plan["steps"]:
        i = s["command"].index("--rl-engine")
        assert s["command"][i + 1] == s["engine"]
        assert "--config" in s["command"]
    assert not list(tmp_path.iterdir())  # nothing run or written


def test_real_mode_requires_launch_args():
    with pytest.raises(SystemExit):
        eq.parse_args([])


def test_extract_rounds_keys_hash_by_policy_version():
    events = [
        {"event": "rl_policy_apply", "island_id": 1, "policy_version": 0,
         "sync/global_policy_hash": "h0"},
        {"event": "rl_local_round", "island_id": 1, "local_round_id": 1, "reward_mean": 0.2,
         "loss": 0.3, "grad_norm": 4.0, "completed_groups": 2, "action_tokens": 9},
        {"event": "rl_policy_apply", "island_id": 1, "policy_version": 1,
         "sync/global_policy_hash": "h1"},
        {"event": "rl_driver_phase", "phase": "train"},
    ]
    rounds = eq.extract_rounds(events)
    assert rounds == {(1, 1): {"reward_mean": 0.2, "loss": 0.3, "grad_norm": 4.0,
                               "completed_groups": 2, "action_tokens": 9,
                               "post_sync_hash": "h1"}}


def test_tolerance_from_repeat_noise_and_nondeterministic_counts():
    reps = [{(0, 1): _row(reward=0.50)}, {(0, 1): _row(reward=0.52)},
            {(0, 1): _row(reward=0.51, tokens=13)}]
    tol = eq.derive_tolerance(reps, factor=2.0, floor=1e-6)
    assert tol["reward_mean"]["tolerance"] == pytest.approx(0.04)
    assert tol["loss"]["tolerance"] == 1e-6
    assert tol["completed_groups"]["kind"] == "exact"
    assert tol["action_tokens"]["kind"] == "nondeterministic"
    assert tol["post_sync_hash"]["kind"] == "exact"


def test_compare_flags_out_of_tolerance_hash_and_missing_rounds():
    reps = [{(0, 1): _row()}, {(0, 1): _row(reward=0.51)}]
    tol = eq.derive_tolerance(reps, factor=1.0, floor=0.0)
    ok = eq.compare(reps[0], {(0, 1): _row(reward=0.505)}, tol)
    assert ok["passed"]
    bad = eq.compare(reps[0], {(0, 1): _row(reward=0.6, h="other"), (0, 2): _row()}, tol)
    assert not bad["passed"]
    assert {f["metric"] for f in bad["failures"]} == {"reward_mean", "post_sync_hash"}
    assert bad["missing_rounds"] == [[0, 2]]


def test_fake_mode_produces_labeled_report(tmp_path):
    out = tmp_path / "eqv"
    assert eq.main(["--fake", "--out-dir", str(out), "--rounds", "3"]) == 0
    text = (out / "report.md").read_text()
    assert text.startswith("# FAKE REPORT")
    assert text.index("## Tolerance") < text.index("## Result: PASS (FAKE)")
    data = json.loads((out / "report.json").read_text())
    assert data["meta"]["fake"] is True and data["passed"]
    keys = {(r["island"], r["round"]) for r in data["rows"]}
    assert keys == {(i, r) for i in (0, 1) for r in (1, 2, 3)}
    hashes = {r["ports"] for r in data["rows"] if r["metric"] == "post_sync_hash" and r["round"] == 3}
    assert len(hashes) == 1  # both islands agree after strict-avg
    legacy = (out / "legacy-0" / "island-0.jsonl").read_text()
    assert '"synthetic_legacy": true' in legacy
