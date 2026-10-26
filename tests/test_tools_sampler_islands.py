"""S26 E4/E5 measurement tool (tools/eval/sampler_islands.py) on synthetic tapes."""

from __future__ import annotations

import json
from pathlib import Path

from tools.eval import sampler_islands as tool


def _tape(run: Path, island: int, name: str, rows: list[dict]) -> None:
    path = run / "tape-direct" / "app" / f"l{island}" / "rank0" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + '\n{"event": "rl_cut', encoding="utf-8")  # a cut last line


def _sampler_round(t, rollout_id, generate_s=10.0, groups=32):
    return {"event": "rl_sampler_round", "time_unix": t, "rollout_id": rollout_id,
            "groups": groups, "generate_s": generate_s}


def _trained(t, groups=32):
    return {"event": "rl_round_trained", "time_unix": t, "trained_groups": groups}


def _run(tmp_path: Path, name: str, *, samplers: int, preempt: bool = False) -> Path:
    run = tmp_path / name
    for m in range(samplers):
        rows = [{"event": "rl_sampler_join", "time_unix": 90.0, "incarnation": f"a{m}"},
                _sampler_round(110.0, 0), _sampler_round(120.0, 1),
                {"event": "rl_sampler_admitted", "time_unix": 125.0, "waited_s": 5.0}]
        if preempt and m == 0:
            rows.append({"event": "rl_sampler_leave", "time_unix": 121.0, "reason": "PREEMPTED",
                         "incarnation": "a0", "in_flight_rollout_id": 2})
            _tape(run, m, "rl-island-0.jsonl", rows)
            rows = [{"event": "rl_sampler_join", "time_unix": 181.0, "incarnation": "b0"},
                    _sampler_round(201.0, 2)]
            _tape(run, m, "rl-island-0.inc1.jsonl", rows)
        else:
            _tape(run, m, "rl-island-0.jsonl", rows)
    step = 20.0 / samplers if not preempt else 40.0
    _tape(run, samplers, "rl-island-0.jsonl",
          [_trained(130.0), _trained(130.0 + step), _trained(130.0 + 2 * step)])
    return run


def test_scale_report_counts_rounds_rates_and_preemptions(tmp_path):
    report = tool.scale_report(_run(tmp_path, "n2k", samplers=2, preempt=True), samples_per_group=4)
    assert report["sampler_islands"] == 2 and report["preemptions"] == 1
    s0 = report["samplers"][0]
    assert s0["incarnations"] == ["a0", "b0"] and s0["rounds"] == 3 and s0["groups"] == 96
    assert s0["generate_s"] == 30.0 and s0["backlog_wait_s"] == 5.0
    (p,) = s0["preemptions"]
    assert (p["rejoin_s"], p["first_new_round_s"], p["in_flight_rollout_id"]) == (60.0, 80.0, 2)
    # production: 160 groups from the first round start (100) to the last end (201)
    assert report["production_groups_per_s"] == round(160 / 101, 4)
    assert report["trainer"]["steady_groups_per_s"] == round(64 / 80, 4)
    assert report["consumption_samples_per_s"] == round(64 / 80 * 4, 4)


def test_compare_gives_speedup_for_more_islands_and_a_ratio_for_the_same_size(tmp_path):
    one = tool.scale_report(_run(tmp_path, "n1", samplers=1))
    two = tool.scale_report(_run(tmp_path, "n2", samplers=2))
    killed = tool.scale_report(_run(tmp_path, "n2k", samplers=2, preempt=True))
    up = tool.compare(two, one)
    assert up["speedup"] == 2.0 and up["efficiency"] == 1.0
    same = tool.compare(killed, two)
    assert same["effective_ratio"] == round((64 / 80) / (64 / 20), 4) and "speedup" not in same


def test_missing_generate_seconds_give_no_production_rate(tmp_path):
    run = tmp_path / "old"
    _tape(run, 0, "t.jsonl", [{"event": "rl_sampler_round", "time_unix": 5.0, "groups": 8}])
    report = tool.scale_report(run)
    assert report["samplers"][0]["generate_s"] is None and report["production_groups_per_s"] is None
    assert report["trainer"] is None


def test_latency_report_joins_releases_and_loads_by_seq(tmp_path):
    run = tmp_path / "lat"
    released = [{"event": "rl_policy_released", "time_unix": 100.0 * seq, "seq": seq,
                 "publish_s": 1.0, "release_bytes": 10,
                 "release_timings": {"export_s": 2.0, "commit_weights_s": 0.5}} for seq in (1, 2, 3)]
    _tape(run, 1, "t.jsonl", [_trained(1.0), _trained(2.0)] + released)
    loads = [{"event": "rl_sampler_policy_loaded", "time_unix": 100.0 * seq + 9.0, "seq": seq,
              "weights": "lora", "seen_at": 100.0 * seq + 4.0, "fused_stage": True,
              "follower_timings": {"read_latest_s": 0.1, "load_s": 4.9},
              "load_timings": {"stage_s": 1.0}} for seq in (1, 2, 3)]
    loads.append({"event": "rl_sampler_policy_loaded", "time_unix": 450.0, "seq": 4})  # no release
    _tape(run, 0, "t.jsonl", loads + [_sampler_round(5.0, 0)])
    report = tool.latency_report(run)
    assert report["loads"] == 3
    row = report["rows"][1]
    assert (row["trainer_release_s"], row["wait_s"], row["sampler_load_s"]) == (2.5, 4.0, 5.0)
    assert (row["released_to_loaded_s"], row["end_to_end_s"]) == (9.0, 12.5)
    assert report["summary_after_first"]["end_to_end_s"]["n"] == 2
    assert report["phases"]["sampler.load_s"]["median"] == 4.9
    assert report["phases"]["trainer.export_s"]["n"] == 3


def test_latency_report_of_old_tapes_keeps_unknown_phases_none(tmp_path):
    run = tmp_path / "old"
    _tape(run, 1, "t.jsonl", [_trained(1.0), {"event": "rl_policy_released", "time_unix": 10.0, "seq": 2}])
    _tape(run, 0, "t.jsonl", [{"event": "rl_sampler_policy_loaded", "time_unix": 13.5, "seq": 2},
                              _sampler_round(5.0, 0)])
    (row,) = tool.latency_report(run)["rows"]
    assert row["released_to_loaded_s"] == 3.5
    assert row["wait_s"] is None and row["end_to_end_s"] is None


def test_cli_prints_json(tmp_path, capsys):
    run = _run(tmp_path, "n1", samplers=1)
    assert tool.main(["scale", str(run), "--baseline", str(run)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["compare"]["effective_ratio"] == 1.0


def test_trainer_phase_seconds_and_buffer_wait_share(tmp_path):
    run = tmp_path / "ph"
    marks = [("publish", 0.0), ("generate", 1.0), ("train", 7.0), ("publish", 9.0), ("finish", 10.0)]
    rows = [{"event": "rl_driver_phase", "phase": name, "time_unix": t} for name, t in marks]
    _tape(run, 0, "t.jsonl", rows + [_trained(8.0), _trained(9.5)])
    trainer = tool.scale_report(run)["trainer"]
    assert trainer["phase_s"] == {"publish": 2.0, "generate": 6.0, "train": 2.0}
    assert trainer["buffer_wait_share"] == 0.6


def test_events_only_in_the_launch_log_are_merged(tmp_path):
    run = _run(tmp_path, "log", samplers=1)
    leave = {"event": "rl_sampler_leave", "time_unix": 121.0, "reason": "PREEMPTED",
             "incarnation": "a0", "in_flight_rollout_id": 2}
    rejoin = {"event": "rl_sampler_join", "time_unix": 150.0, "incarnation": "b0"}
    dup = _sampler_round(110.0, 0)
    (run / "launch.ts.log").write_text("\n".join(
        f"2026-10-22T00:00:00Z [p-l0-modal] YETO_RL_EVENT {json.dumps(r)}" for r in (leave, rejoin, dup)))
    s0 = tool.scale_report(run)["samplers"][0]
    assert s0["rounds"] == 2  # the echoed round is not counted twice
    assert s0["preemptions"][0]["rejoin_s"] == 29.0


def test_a_round_cut_by_a_preemption_is_not_a_round(tmp_path):
    run = _run(tmp_path, "cut", samplers=1)
    cut = {"event": "rl_sampler_round", "time_unix": 122.0, "rollout_id": 2, "groups": 0, "generate_s": 3.0}
    (run / "launch.ts.log").write_text(f"t [p-l0-modal] YETO_RL_EVENT {json.dumps(cut)}")
    s0 = tool.scale_report(run)["samplers"][0]
    assert (s0["rounds"], s0["cut_rounds"], s0["cut_generate_s"], s0["generate_s"]) == (2, 1, 3.0, 20.0)
