"""Multiprocess CPU fake-island harness (stage 0). No Ray, no GPU."""

from __future__ import annotations

import json

from yeto.rl.engine.fake_islands import FakeIsland, run_fake_islands


def test_quorum_lease_catch_up(tmp_path):
    islands = [
        FakeIsland("fast"),
        FakeIsland("crash", crash_after_rounds=1),
        FakeIsland("slow", work_s=1.2),          # always misses the quorum window
        FakeIsland("late", join_at_version=2),   # joins mid-run: catch-up
    ]
    res = run_fake_islands(islands, tmp_path / "run", rounds=6, theta=1.0,
                           quorum_timeout_s=0.8, lease_s=0.5, hb_s=0.05, deadline_s=60,
                           mode="elastic")
    tape = json.loads((tmp_path / "run" / "tape.json").read_text())
    assert tape["final"] == res["final"]
    ev = res["ledger_events"]
    steps = [e for e in ev if e["kind"] == "outer_step"]
    assert res["final"]["outer_version"] == 6 and len(steps) == 6
    # quorum timeout stepping: some rounds advanced without the slow island
    assert any(e["timed_out"] and "slow" in e["absent"] for e in steps)
    # P4: the slow island's late deltas are carried over with gamma**lag, not dropped
    assert any(e["kind"] == "delta_carried_over" and e["island_id"] == "slow" for e in ev)
    assert any(k.startswith("slow@") for e in steps for k in e["carried_in"])
    assert not any(e["kind"] == "delta_rejected" and e.get("reason") == "stale_base" for e in ev)
    # lease expiry removed the crashed island and dropped its uncommitted delta (if any)
    leaves = [e for e in ev if e["kind"] == "pool_leave"]
    assert [e["island_id"] for e in leaves] == ["crash"] and leaves[0]["reason"] == "lease_expired"
    assert "crash" not in res["final"]["members"]
    # its last delta was accepted but never merged: recorded as dropped_uncommitted
    assert leaves[0]["dropped_uncommitted"] is not None
    assert all(not k.startswith("crash") for e in steps for k in e["raw_weights"])
    # catch-up: late island joined at version >= 2 with zero weight in its first round
    join = next(e for e in ev if e["kind"] == "pool_join" and e["island_id"] == "late")
    assert join["catch_up"] and join["base_version"] >= 2
    late_steps = [e for e in steps if "late" in e["raw_weights"]]
    assert late_steps and late_steps[0]["raw_weights"]["late"] == 0.0
    assert late_steps[0]["base_version"] == join["base_version"]
    assert any(e["raw_weights"]["late"] > 0 for e in late_steps[1:])
    # cross-island samples: all three verdict kinds appear; slow island's stale ones need IS
    verdicts = {e["verdict"] for e in ev if e["kind"] == "sample_verdict"}
    assert {"ACCEPT", "ACCEPT_IS"} <= verdicts
    assert any(e["kind"] == "sample_verdict" and e["island_id"] == "slow" and e["outer_lag"] >= 1
               for e in ev)
    # 0.11: the slow island gets a rollout-only suggestion in the tape (not executed)
    advice = [e for e in res["timeline"] if e["kind"] == "pause_advice"]
    assert [a["target_resource_intent"]["island_id"] for a in advice] == ["slow"]
    assert advice[0]["target_resource_intent"]["action"] == "rollout_only"
    assert "slow" in res["final"]["members"]  # nothing was executed
    # journal replay agrees with the ledger membership
    assert sorted(res["journal_replay"]["members"]) == res["final"]["members"]
    assert res["journal_replay"]["membership_epoch"] == res["final"]["membership_epoch"]


def test_legacy_mode_fixed_roster(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        run_fake_islands([FakeIsland("late", join_at_version=1)], tmp_path / "x")
    res = run_fake_islands([FakeIsland("a"), FakeIsland("b", work_s=0.2)], tmp_path / "ok",
                           rounds=3, quorum_timeout_s=2.0)
    steps = [e for e in res["ledger_events"] if e["kind"] == "outer_step"]
    assert res["failure"] is None and len(steps) == 3
    assert all(e["absent"] == [] and not e["timed_out"] and e["carried_in"] == {} for e in steps)
    assert res["journal_replay"]["members"] == ()  or res["journal_replay"]["members"] == []
    assert not any(e["kind"] == "sample_verdict" and e["verdict"] != "REJECT"
                   for e in res["ledger_events"])
    # a slow member makes the strict round time out: fail closed, no partial step
    bad = run_fake_islands([FakeIsland("a"), FakeIsland("slow", work_s=1.5)], tmp_path / "bad",
                           rounds=3, quorum_timeout_s=0.5)
    assert bad["failure"] and bad["final"]["outer_version"] == 0
    assert not any(e["kind"] == "pause_advice" for e in res["timeline"] + bad["timeline"])
