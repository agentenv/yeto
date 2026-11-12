"""Async GRPO on the dashboard: every panel of a sample-buffer run has data.

One synthetic run (2 buffer trainers, 2 samplers, the NC pool, fleet and an
NVML resource sample) is folded into the reducer; each test reads one panel's
inputs from ``full_view`` (the exported page's data) and requires them non-empty.
Status rules: loading is not stale, a sampler is judged by its own cadence, an
island that left is not stale, samplers finish with their trainers."""

from __future__ import annotations

import json

import pytest

from yeto.dashboard.export import export_view, render_page
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import TapeSource, load_all

T0 = 1_790_000_000.0
ROUNDS = 6


def trainer_records(iid: int, *, t0: float = T0 + 100, rounds: int = ROUNDS) -> list[dict]:
    """A buffer trainer: driver phases, the per-round events a buffer step writes,
    one release per round (``buffer_selected`` = its read position)."""
    out = [{"event": "rl_driver_start", "island_id": iid, "time_unix": t0}]
    t = t0
    for rid in range(rounds):
        out.append({"event": "rl_driver_phase", "island_id": iid, "phase": "generate", "rollout_id": rid,
                    "policy_version": rid, "time_unix": t})
        out.append({"event": "rl_buffer_stale_skipped", "island_id": iid, "rollout_id": rid,
                    "stale_groups": 1 if rid == 3 else 0, "max_policy_age": 2, "time_unix": t + 20,
                    "incomplete_groups": 1, "dropped_small_groups": 0, "carried_over_groups": 1 if rid == 2 else 0,
                    "padded_samples": 2 if rid == 4 else 0})
        out.append({"event": "rl_group_policies", "island_id": iid, "rollout_id": rid,
                    "ages": {"0": 4, "1": 3, "2": 1}, "merged_version": rid, "time_unix": t + 20})
        out.append({"event": "rl_driver_phase", "island_id": iid, "phase": "train", "rollout_id": rid,
                    "time_unix": t + 20})
        out.append({"event": "rl_round_trained", "island_id": iid, "rollout_id": rid, "time_unix": t + 40,
                    "trained_groups": 8, "trained_samples": 32, "reward_mean": 0.3 + 0.05 * rid,
                    "reward_std": 0.4, "reward_p10": 0.0, "reward_p50": 0.25, "reward_p90": 1.0,
                    "grad_norm": 0.5 + 0.01 * rid, "mean_kl": 0.001 * (rid + 1), "tok_per_s": 600.0,
                    "resp_len_mean": 220.0, "resp_len_p95": 400.0, "truncated_frac": 0.0625,
                    "mismatch": {"tis_clipfrac": 0.002, "train_rollout_kl": 0.001 * (rid + 1)},
                    "train_metrics": {"ppo_kl": 0.0, "train_rollout_logprob_abs_diff": 0.013,
                                      "pg_clipfrac": 0.0}})
        out.append({"event": "offpolicy_gate", "island_id": iid, "rollout_id": rid, "time_unix": t + 40,
                    "counts": {"group_dropped": 0, "kept": 8, "unknown_version_dropped": 0}})
        out.append({"event": "rl_driver_phase", "island_id": iid, "phase": "sync", "rollout_id": rid,
                    "time_unix": t + 40})
        out.append({"event": "rl_driver_phase", "island_id": iid, "phase": "publish",
                    "policy_version": rid + 1, "time_unix": t + 42})
        if iid == 0:
            out.append({"event": "rl_policy_released", "island_id": iid, "policy_version": rid + 1,
                        "released_at": t + 44, "buffer_selected": 9 * (rid + 1), "release_bytes": 10,
                        "release_timings": {"upload_s": 2.0}, "origin": {"kind": "merged"},
                        "time_unix": t + 45})
        out.append({"event": "rl_publication", "island_id": iid, "policy_version": rid + 1, "time_unix": t + 45})
        out.append({"event": "rl_round_phases", "island_id": iid, "rollout_id": rid, "time_unix": t + 45,
                    "phase_seconds": {"generate": 20, "train": 20, "sync": 2, "publish": 3}, "wall_seconds": 45})
        t += 45
    out.append({"event": "rl_learner_finalized", "island_id": iid, "time_unix": t + 1})
    return out


def sampler_records(name: str, *, t0: float = T0 + 10, rounds: int = ROUNDS + 2) -> list[dict]:
    """An NC engine sampler: no island_id, only ``sampler_id`` (as on the A cards)."""
    out = [{"event": "rl_sampler_start", "sampler_id": name, "max_backlog_groups": 24, "time_unix": t0,
            "sampling": {"n": 4, "max_tokens": 1024, "temperature": 1.0}},
           {"event": "rl_resource_sample", "sampler_id": name, "available": False,
            "reason": "ModuleNotFoundError: No module named 'pynvml'", "time_unix": t0 + 1}]
    t = t0 + 30
    for rid in range(rounds):
        out.append({"event": "rl_sampler_engine_round", "sampler_id": name, "rollout_id": rid,
                    "policy_version": rid // 2, "groups": 4, "samples": 16, "response_tokens": 3200,
                    "finish": {"length": 1, "stop": 15}, "dropped_groups": 0, "zero_variance_groups": 1,
                    "reward_mean": 0.4, "reward_p10": 0.0, "reward_p50": 0.5, "reward_p90": 1.0,
                    "zero_variance_seen": 1, "truncated": 1, "time_unix": t + 30,
                    "agentic": {"retried_trials": 2, "lost_trials": 1}})
        out.append({"event": "rl_sampler_round", "sampler_id": name, "rollout_id": rid,
                    "policy_version": rid // 2, "version_lag": rid % 2, "groups": 4, "buffer_end": 4 * (rid + 1),
                    "generate_s": 30.0, "time_unix": t + 30})
        if rid % 2 == 1:
            released = T0 + 144 + 45 * (rid // 2)  # trainer 0's release of this version
            out.append({"event": "rl_sampler_policy_loaded", "sampler_id": name, "policy_version": rid // 2 + 1,
                        "released_at": released, "seen_at": released + 10, "buffer_selected": 9,
                        "follower_timings": {"resolve_s": 5.0, "load_s": 3.0}, "time_unix": released + 20})
        t += 35
    out.append({"event": "rl_sampler_backlog_full", "sampler_id": name, "backlog_groups": 25,
                "max_backlog_groups": 24, "rollout_id": rounds, "time_unix": t + 1})
    return out


def pool_records() -> list[dict]:
    return [{"t": T0 + 200, "event": "nc_elastic_node_lost", "ticket": "tk1", "shard": 1, "detail": "unreachable"},
            {"t": T0 + 210, "event": "nc_elastic_ordered", "ticket": "tk2", "shard": 1, "account": "a"},
            {"t": T0 + 260, "event": "nc_elastic_member_ready", "ticket": "tk2", "shard": 1}]


def fleet_records() -> list[dict]:
    return [{"event": "island_ready", "island_id": i, "island": f"run-l{i}-modal", "cloud": "modal", "gpu": "H100",
             "gpus": 1, "time_unix": T0} for i in (0, 1)] + [
        {"event": "cost_tick", "time_unix": T0 + 300, "islands": []}]


def run_reducer(extra: list[dict] = ()) -> Reducer:
    r = Reducer(run="async-t")
    recs = trainer_records(0) + trainer_records(1) + sampler_records("nc-a1") + sampler_records("nc-a2") \
        + pool_records() + fleet_records() + list(extra) + [
            {"event": "rl_resource_sample", "island_id": 0, "time_unix": T0 + 150, "available": True,
             "gpus": [{"mem_used_mb": 40000, "mem_total_mb": 81920, "util_pct": 90}]}]
    for rec in sorted(recs, key=lambda x: x.get("time_unix", x.get("t", 0))):
        r.feed(rec)
    return r


@pytest.fixture(scope="module")
def view():
    return export_view(run_reducer(), generated_at=T0 + 1000)


def _card(view, iid):
    return next(c for c in view["overview"]["islands"] if c["id"] == iid)


def test_islands_and_roles(view):
    ids = {c["id"]: c["role"] for c in view["overview"]["islands"]}
    assert ids == {"0": "trainer", "1": "trainer", "nc-a1": "sampler", "nc-a2": "sampler"}


@pytest.mark.parametrize("key", ["reward", "kl", "kl_tr", "tis_clip", "grad_norm", "resp_len", "tok_s"])
def test_per_round_metric_series_non_empty(view, key):
    assert len(view["overview"]["series"]["0"][key]) == ROUNDS


@pytest.mark.parametrize("key", ["reward", "reward_std", "p10", "p50", "p90", "kl_tr", "tis_clip",
                                 "logprob_diff", "grad_norm", "resp_mean", "resp_p95", "trunc", "tok_s"])
def test_round_records_fields_non_empty(view, key):
    rounds = view["page"]["islands"]["0"]["rounds"]
    assert [r[key] for r in rounds if r[key] is not None] and len(rounds) == ROUNDS


def test_phase_spans_derived_from_driver_phases(view):
    rounds = view["page"]["islands"]["0"]["rounds"]
    assert all(set(r["dur"]) >= {"R", "T", "S"} for r in rounds)
    assert rounds[0]["dur"]["R"] == 20.0 and rounds[0]["dur"]["T"] == 20.0
    assert all("P" in r["phases"] for r in rounds[:-1])


def test_sampler_table(view):
    s = view["page"]["async"]["samplers"]["nc-a1"]
    assert s["n_rounds"] == ROUNDS + 2 and s["policy_version"] == (ROUNDS + 1) // 2
    assert s["lag_hist"] == {"0": 4, "1": 4} and s["lag_max"] == 1
    assert s["zero_variance_seen"] == ROUNDS + 2 and s["zero_variance_dropped"] == ROUNDS + 2
    assert s["truncated"] == ROUNDS + 2 and s["samples"] == 16 * (ROUNDS + 2)
    assert s["reward_mean_last"] == 0.4 and s["generate_p50_s"] == 30.0
    assert s["resource"]["available"] is False
    assert (s["retried_trials"], s["lost_trials"]) == (2 * (ROUNDS + 2), ROUNDS + 2)


def test_trainer_async_table(view):
    t = view["page"]["async"]["trainers"]["0"]
    assert t["stale_groups"] == 1 and t["gate_dropped"] == 0 and t["trained_groups"] == 8 * ROUNDS
    assert (t["incomplete_groups"], t["carried_over_groups"], t["padded_samples"]) == (ROUNDS, 1, 2)
    assert t["age_hist"] == {"0": 4 * ROUNDS, "1": 3 * ROUNDS, "2": ROUNDS}
    assert all(r.get("ages") for r in t["rounds"])


def test_backlog_series(view):
    b = view["page"]["async"]["backlog"]
    assert b["has_reads"] and b["written"] == 2 * 4 * (ROUNDS + 2)
    assert b["read"] == 9 * ROUNDS
    assert b["points"][-1][3] == b["written"] - b["read"]
    assert set(b["reported"]) == {"nc-a1", "nc-a2"}


def test_release_timeline(view):
    rel = view["page"]["async"]["releases"]
    assert [r["policy_version"] for r in rel][:3] == [1, 2, 3]
    first = rel[0]
    assert first["trainer"] == "0" and first["upload_s"] == 2.0
    assert set(first["loads"]) == {"nc-a1", "nc-a2"}
    assert all(v["latency_s"] is not None and v["latency_s"] > 0 for v in first["loads"].values())


def test_nc_pool_events_in_stream_and_timeline(view):
    pool = view["page"]["async"]["pool"]
    assert [p["event"] for p in pool] == ["nc_elastic_node_lost", "nc_elastic_ordered", "nc_elastic_member_ready"]
    assert pool[0]["ts"] == T0 + 200 and pool[0]["shard"] == 1
    types = [e["type"] for e in view["events"]["events"]]
    assert "nc_elastic_node_lost" in types
    assert view["overview"]["counts"]["pool"] == 3


def test_gpu_memory(view):
    assert view["page"]["islands"]["0"]["node_series"]["nvml"][0][1] == pytest.approx(39.06, abs=0.01)
    assert view["page"]["islands"]["0"]["gpu_mem_note"] is None
    assert "pynvml" in view["page"]["islands"]["nc-a1"]["gpu_mem_note"]
    assert "--rl-resource-sample-interval" in view["page"]["islands"]["1"]["gpu_mem_note"]


def test_cost_and_syncerless_page(view):
    assert view["overview"]["cost"]["islands"]
    assert view["overview"]["global_status"]["level"] in ("ok", "warn")


def test_finished_run_statuses(view):
    assert _card(view, "0")["status"] == "done"
    # the samplers run until their trainers end: finished, not waiting / stale
    assert _card(view, "nc-a1")["status"] == "done"
    assert not [a for a in view["overview"]["alerts"] if a["rule"] in ("heartbeat", "startup")]


def test_exported_page_has_async_section_and_nojs_banner(view):
    html = render_page(view)
    assert 'id="async"' in html and 'id="nojs"' in html and "<noscript>" in html
    assert "Quick Look" in html


# ------------------------------------------------------------------ status rules
def _status(r: Reducer, iid: str, now: float) -> dict:
    return next(c for c in r.overview(now=now)["islands"] if c["id"] == iid)


def test_sampler_loading_engine_is_starting_not_stale():
    r = Reducer()
    r.feed({"event": "rl_sampler_start", "sampler_id": "s", "time_unix": T0})
    c = _status(r, "s", T0 + 3 * 3600)
    assert c["status"] == "starting" and c["startup_slow"] is True
    # a slow start is still reported, as an alert
    assert any(a["rule"] == "startup" for a in r.overview(now=T0 + 3 * 3600)["alerts"])


def test_sampler_judged_by_its_generation_cadence():
    r = Reducer()
    r.feed({"event": "rl_sampler_start", "sampler_id": "s", "time_unix": T0})
    r.feed({"event": "rl_sampler_round", "sampler_id": "s", "rollout_id": 0, "groups": 4,
            "generate_s": 600.0, "time_unix": T0 + 600})
    r.feed({"event": "rl_sampler_policy_loaded", "sampler_id": "s", "policy_version": 1,
            "seen_at": T0 + 610, "time_unix": T0 + 760})
    c = _status(r, "s", T0 + 760 + 900)  # generating the next round: 900 s quiet, cadence 600 s
    assert c["status"] == "ok" and c["stale_after_s"] >= 1200
    assert not [a for a in r.overview(now=T0 + 1660)["alerts"] if a["rule"] == "heartbeat"]
    assert _status(r, "s", T0 + 760 + 3000)["status"] == "stale"


def test_trainer_loading_is_not_stale():
    r = Reducer()
    r.feed({"event": "island_ready", "island_id": 0, "island": "run-l0-modal", "time_unix": T0})
    r.feed({"event": "trainer_start_phase", "island_id": 0, "phase": "submitted", "time_unix": T0 + 5})
    assert _status(r, "0", T0 + 2 * 3600)["status"] == "starting"


def test_trainer_quiet_within_its_round_time_is_ok():
    r = Reducer()
    for rec in trainer_records(0, rounds=2)[:-1]:
        r.feed(rec)
    last = r.max_ts
    assert _status(r, "0", last + 100)["status"] == "ok"  # 2 x 45 s + 60 s slack
    assert _status(r, "0", last + 400)["status"] == "stale"


def test_island_leave_is_left_not_stale():
    r = Reducer()
    for rec in trainer_records(1, rounds=2)[:-1]:
        r.feed(rec)
    r.feed({"event": "island_leave", "island_id": 1, "reason_name": "STOPPED", "time_unix": r.max_ts + 1})
    c = _status(r, "1", r.max_ts + 7200)
    assert c["status"] == "left" and c["left"]["reason"] == "STOPPED"
    assert not r.overview(now=r.max_ts + 7200)["alerts"]


def test_fleet_island_done_is_finished():
    r = Reducer()
    r.feed({"event": "island_ready", "island_id": 0, "island": "x", "time_unix": T0})
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 1})
    r.feed({"event": "island_done", "island_id": 0, "island": "x", "time_unix": T0 + 50})
    assert _status(r, "0", T0 + 9999)["status"] == "done"


def test_derived_response_length_from_carry_over():
    r = Reducer()
    r.feed({"event": "rl_round_trained", "island_id": 0, "rollout_id": 0, "trained_samples": 32,
            "trained_groups": 8, "time_unix": T0})
    r.feed({"event": "rl_rollout_carry_over", "island_id": 0, "rollout_id": 0,
            "trained_response_tokens": 6400, "time_unix": T0})
    (row,) = r.round_records(r.islands["0"])
    assert row["resp_mean"] == 200.0 and row["resp_derived"] is True


def test_sampler_id_does_not_override_island_id():
    r = Reducer()
    r.feed({"event": "rl_sampler_round", "island_id": 2, "sampler_id": "x", "rollout_id": 0, "time_unix": T0})
    assert set(r.islands) == {"2"}


def test_pool_tape_file_loads(tmp_path):
    p = tmp_path / "events-pool.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in pool_records()))
    r = Reducer()
    load_all(r, [str(p)])
    assert len(r.page_view()["async"]["pool"]) == 3 and r.max_ts == T0 + 260
    assert not r.islands  # pool events name no island


def test_no_async_key_for_plain_runs():
    r = Reducer()
    r.feed({"event": "rl_local_round", "island_id": 0, "local_round_id": 1, "reward_mean": 0.1, "time_unix": T0})
    assert "async" not in r.page_view()


def test_tape_source_reads_sampler_tape(tmp_path):
    p = tmp_path / "events-nc-a1.abcd.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in sampler_records("nc-a1")))
    r = Reducer()
    TapeSource(p).pump(r)
    assert set(r.islands) == {"nc-a1"}


def test_metrics_layout_containers_and_time_fields(view):
    """The RL metrics layout (SFT-page components) draws from these ids and fields."""
    html = render_page(view)
    for el in ("rKpis", "rNodes", "rFleet", "rCostK", "rCost", "rlDist"):
        assert f'id="{el}"' in html
    rounds = view["page"]["islands"]["0"]["rounds"]
    assert all(r["ts"] and r["step_s"] is None or r["ts"] for r in rounds)
    assert all(r["round_s"] == 45 for r in rounds)
    assert view["page"]["async"]["samplers"]["nc-a1"]["first_ts"] == T0 + 10


def test_eval_points():
    r = Reducer()
    r.feed({"event": "rl_eval", "island_id": 0, "eval/reward_mean": 0.42, "eval/wall_s": 3.0, "time_unix": T0})
    assert r.page_view()["islands"]["0"]["evals"] == [[T0, 0.42, "eval/reward_mean"]]



def _smoke(reducer, tmp_path, now):
    import shutil
    import subprocess
    from pathlib import Path

    from yeto.dashboard.export import export_html

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    out = export_html(reducer, tmp_path / "p.html", generated_at=now)
    res = subprocess.run([node, str(Path(__file__).parent / "js" / "dashboard_smoke.js"), str(out)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_islands_now_note_without_events_says_no_events_yet(tmp_path):
    """Islands known only from fleet.jsonl (no tape event yet): "no events yet",
    not the age of epoch 0 (~20000 d)."""
    r = Reducer(run="empty")
    r.feed({"event": "island_ready", "island_id": 0, "island": "run-l0-modal", "time_unix": T0})
    page = _smoke(r, tmp_path, T0 + 60)
    assert page["nodes_note"].startswith("no events yet") and " d ago" not in page["nodes_note"]


def test_islands_now_note_with_events(tmp_path):
    r = Reducer(run="one")
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0})
    assert _smoke(r, tmp_path, T0)["nodes_note"].startswith("last event ")


def test_a_replacement_that_starts_on_the_latest_release_shows_in_its_row():
    """A replacement island starts on the newest release (rl_sampler_start_on_latest)
    instead of hot-loading it: the release row lists it as a load at start."""
    extra = [{"event": "rl_sampler_start_on_latest", "sampler_id": "nc-b1", "base_version": 2,
              "base_dir": "/x/r2-v2", "time_unix": T0 + 900}]
    rel = {r["policy_version"]: r for r in export_view(run_reducer(extra), generated_at=T0 + 1000)
           ["page"]["async"]["releases"]}
    load = rel[2]["loads"]["nc-b1"]
    assert load["at_start"] is True and load["loaded_at"] == T0 + 900 and load["latency_s"] is None
    assert rel[2]["loads"]["nc-a1"]["at_start"] is False


def test_groups_without_a_policy_identity_are_counted_per_trainer():
    """A relaunched trainer's reader passes over groups of versions no record names
    (rl_buffer_unidentified_policy_skipped) and warns when it keeps none."""
    extra = [{"event": "rl_buffer_unidentified_policy_skipped", "island_id": 0, "rollout_id": 1,
              "unidentified_policy_groups": 4, "versions": [[5, 4]], "time_unix": T0 + 125},
             {"event": "rl_buffer_unidentified_policy_skipped", "island_id": 0, "rollout_id": 2,
              "unidentified_policy_groups": 2, "versions": [[6, 2]], "time_unix": T0 + 165},
             {"event": "rl_buffer_unidentified_policy_starved", "island_id": 0, "rollout_id": 3, "selections": 3,
              "unidentified_policy_groups_total": 9, "time_unix": T0 + 200}]
    trainers = export_view(run_reducer(extra), generated_at=T0 + 1000)["page"]["async"]["trainers"]
    assert trainers["0"]["unidentified_policy_groups"] == 6
    assert trainers["0"]["unidentified_starved"] == {"ts": T0 + 200, "selections": 3, "groups": 9}
    assert [r.get("unidentified_policy_groups") for r in trainers["0"]["rounds"]][:3] == [None, 4, 2]
    assert trainers["1"]["unidentified_policy_groups"] is None and trainers["1"]["unidentified_starved"] is None
