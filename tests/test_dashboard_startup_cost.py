"""fleet-dashboard 8.1-8.3 + rollout-node sampling: startup status and wide
thresholds, GPU+CPU+memory cost, inferred run name, per-node host samples."""

import json
from types import SimpleNamespace

from yeto.dashboard import alerts as A
from yeto.dashboard.cost import island_rate, load_prices
from yeto.dashboard.fleet import meta_from_task
from yeto.dashboard.reducer import Reducer, modal_host_shape
from yeto.dashboard.sources import infer_run_name, load_all
from yeto.modal_runner import HostMemSampler, hostmem_interval_s

from dashboard_helpers import T0

MODAL = {"prices": {"modal:H200": 4.54}, "host_prices": {"modal": {"cpu_core_h": 0.04716, "mem_gib_h": 0.007992}}}


def _startup_reducer(**kw):
    r = Reducer(prices=MODAL, **kw)
    r.feed({"event": "island_ready", "island": "run-a-l0-modal", "island_id": 0, "cloud": "modal",
            "gpu": "H200", "gpus": 16, "cpus": 64, "memory_gib": 1536, "time_unix": T0})
    r.feed({"event": "rl_engine_selected", "island_id": 0, "time_unix": T0 + 120})
    return r


def _hb_alerts(r, now):
    ov = r.overview(now=now)
    return ov["islands"][0], [a for a in ov["alerts"] if a["rule"] in ("heartbeat", "startup")]


def test_starting_island_uses_wide_startup_threshold():
    r = _startup_reducer()
    card, al = _hb_alerts(r, T0 + 900)  # 13 min of weight loading: no heartbeat alert
    assert card["status"] == "starting" and card["starting"] and not al
    card, al = _hb_alerts(r, T0 + 120 + 1300)
    assert card["status"] == "stale" and [a["sev"] for a in al] == [1] and al[0]["rule"] == "startup"
    _, al = _hb_alerts(r, T0 + 120 + 2400)  # 40 min with no event during startup
    assert [a["sev"] for a in al] == [0]


def test_driver_start_switches_back_to_heartbeat_thresholds():
    r = _startup_reducer()
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 800})
    card, al = _hb_alerts(r, T0 + 800 + 400)
    assert not card["starting"] and card["status"] == "stale"
    assert [(a["rule"], a["sev"]) for a in al] == [("heartbeat", 0)]


def test_startup_heartbeat_phase_keeps_starting():
    r = _startup_reducer()
    r.feed({"event": "rl_heartbeat", "island_id": 0, "phase": "startup", "time_unix": T0 + 300})
    assert r.overview(now=T0 + 310)["islands"][0]["status"] == "starting"


def test_startup_thresholds_configurable():
    r = _startup_reducer(thresholds={"startup_warn_s": 100, "startup_severe_s": 200})
    _, al = _hb_alerts(r, T0 + 120 + 250)
    assert [a["sev"] for a in al] == [0]
    assert A.merged_thresholds({"startup_warn_s": 5})["startup_warn_s"] == 5.0


def test_cost_counts_gpu_cpu_memory_modal_2x8():
    r = _startup_reducer()
    row = r.overview(now=T0 + 3600)["cost"]["islands"][0]
    assert abs(row["rate_usd_h"] - 87.93) / 87.93 < 0.05
    assert row["gpu_rate_usd_h"] == 4.54 * 16 and not row["gpu_only"]


def test_cost_marks_gpu_only_when_host_shape_unknown():
    rt = island_rate(MODAL, {"cloud": "modal", "gpu": "H200", "gpus": 8})
    assert rt["gpu_only"] and rt["rate_usd_h"] == 4.54 * 8
    assert island_rate({"prices": {"modal:H200": 4.54}}, {"cloud": "modal", "gpu": "H200", "gpus": 8,
                                                          "cpus": 32, "memory_gib": 768})["gpu_only"]
    assert island_rate(MODAL, {"cloud": "local", "gpu": "x", "gpus": 1})["gpu_only"] is False


def test_example_price_table_has_modal_host_prices():
    hp = load_prices(None)["host_prices"]["modal"]
    assert hp["cpu_core_h"] > 0 and hp["mem_gib_h"] > 0


def test_fleet_meta_records_modal_host_request():
    cfg = SimpleNamespace(gpu="H200!", gpus_per_node=8, num_nodes=2, region=None,
                          cpu_request=32, memory_request_mib=768 * 1024)
    m = meta_from_task(cfg)
    assert m["gpus"] == 16 and m["cpus"] == 64 and m["memory_gib"] == 1536


def test_modal_host_shape_from_meta_args():
    assert modal_host_shape({"gpu": "modal:2x8xh200", "modal_cpu": 32, "modal_memory_gib": 768}) == {
        "cpus": 64, "memory_gib": 1536}
    assert modal_host_shape({"gpu": "modal:8xh100"}) == {"cpus": 32, "memory_gib": 256}
    assert modal_host_shape({"gpu": "nebius:8xH200"}) is None


def _write_run(tmp_path):
    run = tmp_path / "runs" / "run-b"
    (run / "events").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"name": "run-b", "args": {
        "gpu": "modal:2x8xh200", "modal_cpu": 32, "modal_memory_gib": 768}}))
    (run / "fleet.jsonl").write_text(json.dumps({
        "event": "island_ready", "island": "run-b-l0-modal", "island_id": 0, "cloud": "modal",
        "gpu": "H200", "gpus": 16, "time_unix": T0}) + "\n")
    return run


def test_run_name_and_host_shape_inferred_from_run_dir(tmp_path):
    run = _write_run(tmp_path)
    r = Reducer(prices=MODAL)
    load_all(r, [str(run)])
    ov = r.overview(now=T0 + 3600)
    assert ov["run"] == "run-b" and ov["run_inferred"] and ov["run_kind"] == "single_island"
    assert abs(ov["cost"]["islands"][0]["rate_usd_h"] - 87.93) < 0.5
    r2 = Reducer(run="given", prices=MODAL)
    load_all(r2, [str(run)])
    assert r2.overview(now=T0)["run"] == "given" and not r2.overview(now=T0)["run_inferred"]


def test_infer_run_name_fallbacks():
    assert infer_run_name(["/x/tape-direct/yeto-run-c/l0/rank0/rl-island-0.jsonl"]) == "run-c"
    assert infer_run_name(["/x/events/run-d-l0-modal.jsonl"]) == "run-d"
    assert infer_run_name(["/x/rl-island-0.jsonl"]) is None


def test_rollout_node_gpu_sampled_per_node(tmp_path):
    base = tmp_path / "yeto-run-e" / "l0"
    for rank, util in (("0", [90, 80]), ("1", [20, 30])):
        d = base / f"rank{rank}"
        d.mkdir(parents=True)
        (d / f"modal-hostmem-rank{rank}.jsonl").write_text(json.dumps({
            "event": "modal_host_sample", "time_unix": T0, "meminfo_used": 1 << 30, "meminfo_total": 2 << 30,
            "gpu_mem_used_mib": [100 * (int(rank) + 1), 5], "gpu_util_pct": util}) + "\n")
    (base / "rank0" / "rl-island-0.jsonl").write_text(json.dumps(
        {"event": "rl_driver_start", "island_id": 0, "time_unix": T0}) + "\n")
    r = Reducer()
    load_all(r, [str(base)])
    nodes = {n["node"]: n for n in r.overview(now=T0)["islands"][0]["nodes"]}
    assert nodes["1"]["gpu_util_pct"] == 25.0 and nodes["1"]["gpu_mem_used_mib_peak"] == [200, 5]
    assert nodes["0"]["gpu_util_pct"] == 85.0


def test_hostmem_default_on_for_multinode():
    mk = lambda n, env=None: SimpleNamespace(num_nodes=n, envs=env or {})  # noqa: E731
    assert hostmem_interval_s(mk(1)) is None
    assert hostmem_interval_s(mk(2)) == 30.0
    assert hostmem_interval_s(mk(2, {"YETO_MODAL_HOSTMEM_SAMPLE_S": "10"})) == 10.0
    assert hostmem_interval_s(mk(2, {"YETO_MODAL_HOSTMEM_SAMPLE_S": "0"})) is None
    assert hostmem_interval_s(mk(1, {"YETO_MODAL_HOSTMEM_SAMPLE_S": "5"})) == 5.0


def test_hostmem_record_carries_rank(tmp_path):
    rec = HostMemSampler(1, str(tmp_path / "h.jsonl"), node_rank=1).sample()
    assert rec["node_rank"] == 1 and "gpu_util_pct" in rec
