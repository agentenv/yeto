"""agentic-rollout-utilization 7: generation-stage utilization on the dashboard
(cutoff tally, completion curve, four-phase split, KV/queue, stage-2 carry-over, A/B)."""

import json
import shutil
import subprocess

import pytest

from yeto.dashboard import cli as dash_cli
from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer

from dashboard_helpers import FX, T0, four_island_reducer

SMOKE = FX.parent.parent / "js" / "dashboard_smoke.js"


def _gen_span(rid, start, end):
    # driver monotonic start/end; time_unix written at end
    return {"event": "rl_timeline_span", "island_id": 0, "task": "generate", "rollout_id": rid,
            "start": 100.0, "end": 100.0 + (end - start), "time_unix": end}


def _traj(rid, start, end, **kw):
    r = {"event": "rl_trajectory_reward", "island_id": 0, "rollout_id": rid, "time_unix": end + 50,
         "trajectory_id": f"t{rid}-{start}", "task_id": "x", "reward": 1.0,
         "trajectory_started_at": start, "trajectory_ended_at": end,
         "generation_seconds": 8.0, "tool_seconds": 1.0, "evaluate_time": 0.5, "sandbox_start_seconds": 0.5}
    r.update(kw)
    return r


def util_reducer(*, carry=False, label=None):
    r = Reducer(run="u")
    r.label = label
    recs = [_gen_span(0, T0, T0 + 60), _gen_span(1, T0 + 100, T0 + 150)]
    recs += [_traj(0, T0 + 1, T0 + 20), _traj(0, T0 + 2, T0 + 40), _traj(1, T0 + 101, T0 + 130)]
    if carry:
        recs.append(_traj(1, T0 + 30, T0 + 140, started_rollout_id=0, policy_versions=[[0, 0, 40], [1, 40, 90]]))
    recs.append({"event": "rl_rollout_cutoff", "island_id": 0, "rollout_id": 0, "time_unix": T0 + 70,
                 "submitted_groups": None, "target_groups": 6, "discarded_groups": 6,
                 "discarded_trajectories": 24, "discarded_tokens": None, "filtered_groups": 0})
    recs.append({"event": "rl_rollout_cutoff", "island_id": 0, "rollout_id": 1, "time_unix": T0 + 160,
                 "submitted_groups": 12, "target_groups": 6, "discarded_groups": 6,
                 "discarded_trajectories": 24, "discarded_tokens": 1000, "filtered_groups": 0})
    for i, (used, q) in enumerate([(100, 0), (400, 5), (200, 2)]):
        recs.append({"event": "rl_load_sample", "island_id": 0, "rollout_id": 0, "time_unix": T0 + 10 + 10 * i,
                     "kv_used_tokens": used, "kv_capacity_tokens": 1000, "queued_requests": q,
                     "running_requests": 3, "tool_wait_trajectories": 1})
    r.feed_many(recs)
    return r


def test_cutoff_completion_phases_and_load_per_round():
    u = util_reducer().page_view()["islands"]["0"]["util"]
    r0, r1 = u["rounds"]
    assert r0["cutoff"]["submitted_groups"] is None and r0["cutoff"]["discarded_tokens"] is None  # stays unknown
    assert r1["cutoff"]["submitted_groups"] == 12 and r1["cutoff"]["discarded_tokens"] == 1000
    assert r0["gen_s"] == 60.0 and r0["done"] == [20.0, 40.0] and r1["done"] == [30.0]
    assert r0["phases"] == {"gen": 16.0, "tool": 2.0, "judge": 1.0, "sandbox": 1.0}
    assert r0["tool_share"] == 0.1
    assert r0["peaks"] == {"kv": 0.4, "queued": 5.0}
    assert r1["peaks"] == {"kv": None, "queued": None}  # no load sample inside round 1's span
    assert u["load"][1][1:3] == [0.4, 5.0] and u["load_has_kv"] is True
    assert {t["rid"] for t in u["trajectories"]} == {0, 1} and u["trajectories"][0]["off"] == 1.0
    assert u["carry"] is None  # no stage-2 fields: the block stays hidden


def test_stage_two_carry_over_is_listed():
    carry = util_reducer(carry=True).page_view()["islands"]["0"]["util"]["carry"]
    assert carry == [{"id": f"t1-{T0 + 30}", "task": "x", "from": 0, "to": 1,
                      "versions": [[0, 0, 40], [1, 40, 90]]}]


def test_old_tape_has_no_utilization():
    v = four_island_reducer().page_view()
    assert all(x["util"] is None for x in v["islands"].values())
    assert "compare" not in v


def test_compare_run_is_attached(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, red in ((a, util_reducer()), (b, util_reducer(carry=True))):
        path.write_text("".join(json.dumps(e["record"]) + "\n" for e in red.events))
    import argparse

    p = argparse.ArgumentParser()
    dash_cli.add_parser(p.add_subparsers(dest="cmd"))
    out = tmp_path / "ab.html"
    args = p.parse_args(["dashboard", "export", "--tapes", str(a), "--compare", str(b),
                         "--label", "A 多发 6", "--compare-label", "B 多发 12", "-o", str(out)])
    assert dash_cli.main(args) == 0
    html = out.read_text()
    assert "A 多发 6" in html and "B 多发 12" in html


def _smoke(tmp_path, reducer):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    out = export_html(reducer, tmp_path / "p.html", generated_at=T0)
    res = subprocess.run([node, str(SMOKE), str(out)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)["util"]


def test_page_draws_utilization_and_unknowns(tmp_path):
    r = util_reducer(carry=True, label="A")
    r.compare = [util_reducer(label="B")]
    u = _smoke(tmp_path, r)
    assert u["empty_hidden"] is True and u["body_hidden"] is False
    assert "未知" in u["cut"] and "1000" in u["cut"] and "A" in u["cut"] and "B" in u["cut"]
    assert "全部轮叠加" in u["ctl"] and "叠加" in u["ctl"]  # round switch + A/B mode
    assert u["done_svg"] == 4 and u["cut_lines"] == 4  # 2 rounds x 2 runs, side by side
    assert u["carry_hidden"] is False and "0 → 1" in u["carry"]


def test_page_hides_utilization_for_old_tapes(tmp_path):
    u = _smoke(tmp_path, four_island_reducer())
    assert u["empty_hidden"] is False and u["body_hidden"] is True and u["carry_hidden"] is not False
