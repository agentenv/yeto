"""SFT adapter: ms-swift / yeto plugin / syncer files -> one reducer view."""

import json
import re
import shutil
import subprocess

import pytest

from yeto.dashboard import sft
from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import discover, load_all

from dashboard_helpers import FX, T0

SMOKE = FX.parent.parent / "js" / "dashboard_smoke.js"


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_run(root, steps=6, island="0", v2=False, torn=False):
    d = root / f"island-{island}"
    _w(d / "logging.jsonl", [{"loss": 0.8 - 0.01 * s, "grad_norm": 1.0, "learning_rate": 1e-6 * s,
                              "iteration": f"{s}/100", "memory(GiB)": 52.0, "elapsed_time": "1h 2m 3s",
                              "train_speed(s/it)": 580.0} for s in range(1, steps + 1)])
    _w(d / "yeto_steps.jsonl", [{"iteration": s, "time": T0 + 570 * s, "target_tokens": 4e6, "sync_s": 0.5,
                                 "export_s": 3.0 if s % 3 == 0 else 0.0, "apply_s": 0.0,
                                 "pushes_sent": 1 if s % 3 == 0 else 0, "applies": 0}
                                for s in range(1, steps + 1)])
    _w(d / "bench_metrics.jsonl", [{"step": s, "tokens": 1e7, "wall_s": 570.0 + s, "tokens_per_s": 1.75e4,
                                    "mfu_6N_pct": 14.5, "world_size": 24} for s in range(1, steps + 1)])
    for rank in (0, 8):
        _w(d / f"health.rank{rank}.jsonl", [{"step": s, "rank": rank, "reserved_peak_gib": 50.0 + rank / 4}
                                            for s in range(1, steps + 1)])
    (d / "args.json").write_text(json.dumps({
        "tensor_model_parallel_size": 4, "expert_model_parallel_size": 8, "pipeline_model_parallel_size": 3,
        "world_size": 24, "train_iters": 100, "global_batch_size": 384,
        "cached_dataset": ["/x/split/island1_v2/shard-1/train" if v2 else "/x/split/island0/shard-1/train"]}))
    if torn:
        with open(d / "logging.jsonl", "a") as fh:
            fh.write('{"loss": 0.1, "iteration": "99')
    s = root / "syncer"
    s.mkdir(parents=True, exist_ok=True)
    (s / "syncer.log").write_text(
        "2026-09-21T13:54:15.176205Z  INFO yeto_syncer: merge thread count merge_threads=32\n"
        "2026-09-21T14:00:00.000000Z  WARN yeto_syncer::server: push rejected: StaleBase learner_id=1 step=3\n"
        "2026-09-21T14:01:00.000000Z  INFO yeto_syncer::server: late push queued for the fragment's next round learner_id=0\n"
        "2026-09-21T14:02:00.000000Z  INFO yeto_syncer::state: background checkpoint written step=16 path=/p seconds=12.5\n"
        "2026-09-21T14:03:00.000000Z  INFO yeto_syncer::server: wire bytes per learner: learner 0 sent=1 recv=2\n")
    (s / "cmdline.txt").write_text("yeto-syncer --port 29400 --learners 1 --total-steps 960 --sync-interval-steps 32\n")
    _w(s / "tape.jsonl", [
        {"kind": "outer_step", "step": 1, "fragment": 0, "step_seconds": 2.0, "timed_out": False, "time_unix": T0 + 2000},
        {"kind": "outer_step", "step": 2, "fragment": 1, "step_seconds": 3.0, "timed_out": True, "time_unix": T0 + 2600},
        {"kind": "delta_rejected", "island_id": 1, "time_unix": T0 + 2600},
        {"step": 1, "fragment": 0, "expected": [0], "responded": [0], "sync/merge_seconds": 4.0, "gnorm": 1.3,
         "time_unix": T0 + 2001}])
    _w(s / "rss.jsonl", [{"event": "sft_syncer_rss", "time_unix": T0 + 3000, "rss_bytes": 9e11}])
    return root


def test_converters():
    r = sft.from_swift_log({"iteration": "8/1092", "loss": 0.75, "elapsed_time": "1d 1h 2m 3s"}, "0")
    assert r["step"] == 8 and r["steps_total"] == 1092 and r["elapsed_s"] == 86400 + 3723
    assert sft.from_swift_log({"loss": 1.0}, "0") is None
    e = sft.from_syncer_log_line("2026-09-21T14:00:00.5Z  WARN yeto_syncer::server: quorum timed out learner_id=2")
    assert e["category"] == "timeout" and e["learner"] == 2 and e["time_unix"] > 0
    assert sft.from_syncer_log_line("2026-09-21T14:00:00Z  INFO x: wire bytes per learner") is None
    assert sft.from_cmdline("a --total-steps 960 --learners=2")["total_steps"] == 960
    assert sft.dataset_tag(["/d/split/island2_v2/shard-3/train"]) == "v2"
    assert sft.dataset_tag(["/d/split/island0/shard-3/train"]) == "v1"


def test_reducer_view(tmp_path):
    root = make_run(tmp_path / "run", torn=True)
    assert any(p.name == "syncer.log" for p in discover([str(root)]))
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    v = r.page_view(now=T0 + 3600)["sft"]
    i = v["islands"][0]
    cols = v["cols"]
    assert i["layout"] == "TP4/EP8/PP3" and i["cards"] == 24 and i["dataset"] == "v1" and i["steps_total"] == 100
    assert len(i["rows"]) == 6  # torn tail line not consumed
    last = dict(zip(cols, i["rows"][-1]))
    assert last["step"] == 6 and last["hbm_gib"] == 52.0 and last["step_s"] == 576.0 and last["tps"] == 1.75e4
    assert i["last_push"]["step"] == 6 and i["last_apply"] is None
    s = v["syncer"]
    assert s["total_steps"] == 960 and s["outer_step"] == 2 and s["eta_ts"] > T0
    cats = [e[1] for e in s["events"]]
    # syncer.log is the authority for rejections / timeouts once seen: tape ones are not double counted
    assert cats.count("rejected") == 1 and "timeout" not in cats and "late" in cats and "checkpoint" in cats
    assert s["merges"][0][3] == 4.0 and s["rss"][0][1] == 9e11
    assert r.counts["sft"] > 0


def test_incremental_no_double_count(tmp_path):
    root = make_run(tmp_path / "run", steps=3)
    from yeto.dashboard.serve import DashboardState

    st = DashboardState(Reducer(), [str(root)])
    st.poll()
    with open(root / "island-0" / "logging.jsonl", "a") as fh:
        fh.write(json.dumps({"loss": 0.5, "iteration": "4/100"}) + "\n")
    st.poll()
    st.poll()
    rows = st.reducer.sft_view(T0)["islands"][0]["rows"]
    assert [x[0] for x in rows] == [1, 2, 3, 4]


def test_bench_restart_continues(tmp_path):
    st = sft.SftState()
    for s in (1, 2, 3):
        st.fold({"event": "sft_train_log", "island_id": "0", "step": s}, None)
    for s in (1, 2, 1):
        st.fold({"event": "sft_bench", "island_id": "0", "seg_step": s, "wall_s": 10.0 * s}, None)
    assert [r[0] for r in st.island_rows(st.islands["0"])] == [1, 2, 3]


def test_bench_restart_after_a_crash_maps_onto_the_resumed_steps():
    # launch 1 logs steps 1..7 and crashes; launch 2 resumes from checkpoint-5
    # and logs 6..9; bench counts 1..7 then 1..4.
    st = sft.SftState()
    for s in [*range(1, 8), *range(6, 10)]:
        st.fold({"event": "sft_train_log", "island_id": "0", "step": s, "loss": float(s)}, None)
    for s in [*range(1, 8), *range(1, 5)]:
        st.fold({"event": "sft_bench", "island_id": "0", "seg_step": s, "wall_s": 100.0 + s}, None)
    bench, offsets = st._bench_by_step(st.islands["0"])
    assert offsets == [0, 5] and sorted(bench) == list(range(1, 10))
    assert bench[6]["wall_s"] == 101.0 and bench[9]["wall_s"] == 104.0  # launch 2's own records
    assert bench[5]["wall_s"] == 105.0  # launch 1's step 5 is kept
    rows = st.island_rows(st.islands["0"])
    assert [r[0] for r in rows] == list(range(1, 10))


def test_bench_of_a_resumed_first_launch_starts_at_the_checkpoint():
    st = sft.SftState()
    for s in (16, 17):
        st.fold({"event": "sft_train_log", "island_id": "0", "step": s}, None)
    for s in (1, 2):
        st.fold({"event": "sft_bench", "island_id": "0", "seg_step": s}, None)
    assert sorted(st._bench_by_step(st.islands["0"])[0]) == [16, 17]


def test_health_of_a_resumed_launch_maps_onto_the_resumed_steps():
    # launch 1 logs steps 1..3 and crashes; launch 2 resumes from checkpoint-2
    # and logs 3..4; every rank's health counts 1..3 then 1..2.
    st = sft.SftState()
    for s in (1, 2, 3, 3, 4):
        st.fold({"event": "sft_train_log", "island_id": "0", "step": s}, None)
    for s in (1, 2, 3, 1, 2):
        st.fold({"event": "sft_bench", "island_id": "0", "seg_step": s, "wall_s": 10.0}, None)
    for launch, peaks in ((1, (11.0, 12.0, 13.0)), (2, (51.0, 52.0))):
        for s, gib in enumerate(peaks, start=1):
            for rank in (0, 8):
                st.fold({"event": "sft_health", "island_id": "0", "seg_step": s, "rank": rank,
                         "reserved_peak_gib": gib + rank / 8}, None)
    hbm = {r[0]: r[12] for r in st.island_rows(st.islands["0"])}
    assert hbm == {1: 12.0, 2: 13.0, 3: 52.0, 4: 53.0}


def test_references_are_not_drawn_unless_configured(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    r = Reducer(run="sft")
    load_all(r, [str(make_run(tmp_path / "run"))])
    assert set(r.sft_view(T0)["refs"]) == {"sft_slow_factor"}
    out = export_html(r, tmp_path / "s.html", generated_at=T0)
    page = json.loads(subprocess.run([node, "-e", SMOKE_SFT, str(out)], capture_output=True, text=True,
                                     timeout=60).stdout)
    assert "reference" not in page["kpis"] and "HBM peak</th>" in page["nodes"] and "NaN" not in page["nodes"]


def test_island_json_fields_are_typed():
    rec = sft.from_island_json({"name": "a", "nodes": "node-a", "steps_total": "<img src=x>", "cards": 8,
                                "layout": ["x"]}, "0")
    assert rec["nodes"] == ["node-a"] and rec["cards"] == 8
    assert "steps_total" not in rec and "layout" not in rec


def test_rl_tape_with_pool_events_has_no_sft():
    """The syncer writes outer_step / pool_join / delta_* without seq for RL runs too."""
    r = Reducer()
    r.feed({"kind": "pool_join", "island_id": 1, "time_unix": T0})
    r.feed({"kind": "outer_step", "step": 1, "fragment": 0, "time_unix": T0 + 1})
    r.feed({"kind": "delta_rejected", "island_id": 1, "time_unix": T0 + 2})
    assert "sft" not in r.page_view()


def test_rl_view_has_no_sft(tmp_path):
    r = Reducer()
    load_all(r, [str(FX / "kill44")])
    assert "sft" not in r.page_view()


def test_rl_run_with_a_syncer_log_has_no_sft(tmp_path):
    """The RL ssh harness keeps syncer.log next to the syncer's event tape: its WARN /
    rejection lines are read, but alone they do not turn an RL run into an SFT page."""
    shutil.copytree(FX / "kill44", tmp_path / "run")
    state = tmp_path / "run" / "state"
    state.mkdir()
    (state / "syncer.log").write_text(
        "2026-09-21T14:00:00Z  WARN yeto_syncer::server: delta rejected learner_id=1\n"
        "2026-09-21T14:00:01Z  INFO yeto_syncer::server: grace window armed step=1 fragment=0 quorum_secs=2.5\n")
    r = Reducer()
    load_all(r, [str(tmp_path / "run")])
    assert r.sft.log_seen and r.sft.log_rounds
    assert "sft" not in r.page_view()
    (state / "island.json").write_text(json.dumps({"name": "island 0"}))  # an SFT-only file
    r2 = Reducer()
    load_all(r2, [str(tmp_path / "run")])
    assert "sft" in r2.page_view()


def test_export_page_renders_sft(tmp_path):
    root = make_run(tmp_path / "run")
    make_run(tmp_path / "run", island="1", v2=True)
    r = Reducer(run="sft", thresholds={"sft_ref_step_s": 571, "sft_hbm_limit_gib": 53.9})
    load_all(r, [str(root)])
    out = export_html(r, tmp_path / "s.html", generated_at=T0)
    html = out.read_text()
    assert not re.search(r"https?://", html)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    res = subprocess.run([node, "-e", SMOKE_SFT, str(out)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    page = json.loads(res.stdout)
    assert "yeto metrics" in page["title"] and page["sft_hidden"] is False and page["rl_hidden"] is True
    assert "TP4/EP8/PP3" in page["nodes"] and "v2" in page["nodes"] and "island 1" in page["nodes"]
    assert "960" in page["kpis"] and "vs reference 571 s" in page["kpis"]
    assert "HBM peak vs 53.9 GiB" in page["nodes"]
    assert "<svg" in page["fleet"] and "p50" in page["dist"] and "Card-hours" in page["cost"]


# Reuse the DOM stub of dashboard_smoke.js, dumping the SFT containers instead.
SMOKE_SFT = (SMOKE.read_text().split("eval(appM[1]);")[0] + """
eval(appM[1]);
process.stdout.write(JSON.stringify({title: els.title.innerHTML, kpis: els.sKpis.innerHTML,
  nodes: els.sNodes.innerHTML, fleet: els.fLoss.innerHTML, tps: els.fTps.innerHTML, sync: els.fSync.innerHTML,
  merge: els.fMerge.innerHTML, round: els.fRound.innerHTML, dist: els.sDist.innerHTML,
  cost: els.sCostK.innerHTML, eval: els.fEval.innerHTML, sft_hidden: els.sftRoot.hidden, rl_hidden: els.rlRoot.hidden}));
""").replace("process.argv[2]", "process.argv[1]")


BINS_JS = """
const fs=require('fs');global.localStorage={getItem(){return null},setItem(){}};global.location={hash:''};
global.document={getElementById(){return null},body:null};
eval(fs.readFileSync(process.argv[1],'utf8')+';global.YE=YetoExt;');
const out=JSON.parse(process.argv[2]).map(v=>YE.bins(v));process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.parametrize("vals", [
    [571, 571, 575, 579, 586, 592], [0.004, 0.004, 0.005, 0.007, 0.008, 0.006],
    [5.0, 5.0], [1.0 + 0.37 * (i % 97) ** 1.5 for i in range(1200)],
])
def test_histogram_bins_cover_samples_and_quantiles(vals):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = FX.parent.parent.parent / "yeto" / "dashboard" / "static" / "sft.js"
    res = subprocess.run([node, "-e", BINS_JS, str(js), json.dumps([vals])], capture_output=True, text=True,
                         timeout=60)
    assert res.returncode == 0, res.stderr
    b = json.loads(res.stdout)[0]
    assert abs(sum(b["pct"]) - 1.0) < 1e-9
    if len(vals) < 500:
        assert b["kept"] == len(vals)  # no tail cut for small samples
    for name, q in b["q"].items():
        if q["bin"] is None:
            assert len(vals) >= 500
            continue
        lo, hi = b["edges"][q["bin"]], b["edges"][q["bin"] + 1]
        assert lo * (1 - 1e-9) <= q["v"] <= hi * (1 + 1e-9), name
        assert b["pct"][q["bin"]] > 0, name  # the quantile's own bin has a bar


def test_kpi_p50_equals_distribution_p50(tmp_path):
    """One quantile rule (nearest rank) on the whole page: 6 steps of 571..576 s -> p50 573 s
    (interpolation would give 573.5 s)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    r = Reducer(run="sft")
    load_all(r, [str(make_run(tmp_path / "run"))])
    out = export_html(r, tmp_path / "s.html", generated_at=T0)
    res = subprocess.run([node, "-e", SMOKE_SFT, str(out)], capture_output=True, text=True, timeout=60)
    page = json.loads(res.stdout)
    kpi = re.search(r"island 0 step p50</div><div class=\"kv\">([^<]+)<", page["kpis"]).group(1)
    dist = re.search(r'<tr class="b"><td>p50</td><td>([^<]+)</td>', page["dist"]).group(1)
    assert kpi == dist == "573 s"


def test_stack_skips_island_without_data(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    root = make_run(tmp_path / "run")
    (root / "island-1").mkdir()
    (root / "island-1" / "island.json").write_text(json.dumps({"island": "1", "name": "slow island 1"}))
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    out = export_html(r, tmp_path / "s.html", generated_at=T0)
    page = json.loads(subprocess.run([node, "-e", SMOKE_SFT, str(out)], capture_output=True, text=True,
                                     timeout=60).stdout)
    assert "slow island 1 (no data)" in page["tps"] and "#f97316" not in page["tps"].split('<div class="lg">')[0]


def _page(tmp_path, root, now):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    out = export_html(r, tmp_path / "s.html", generated_at=now)
    res = subprocess.run([node, "-e", SMOKE_SFT, str(out)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _top_line_ys(svg):
    """y coordinates of the last stroked path (the top of the stack)."""
    d = re.findall(r'<path d="([^"]+)" fill="none"', svg)[-1]
    return [float(p.split(",")[1]) for p in re.split(r"[ML]", d) if p]


def _stack_top(svg):
    """(x, y) of the top of the stack: the last stroked path."""
    d = re.findall(r'<path d="([^"]+)" fill="none"', svg)[-1]
    return [tuple(float(v) for v in p.split(",")) for p in re.split(r"[ML]", d) if p]


def _island_steps(root, island, period, start, tps):
    d = root / f"island-{island}"
    n = len((d / "yeto_steps.jsonl").read_text().splitlines())
    _w(d / "yeto_steps.jsonl", [{"iteration": s, "time": start + period * s, "sync_s": 0.1, "export_s": 0.0,
                                 "apply_s": 0.0} for s in range(1, n + 1)])
    _w(d / "bench_metrics.jsonl", [{"step": s, "tokens": tps * period, "wall_s": period, "tokens_per_s": tps}
                                   for s in range(1, n + 1)])


def test_tps_stack_is_flat_for_islands_with_different_step_periods(tmp_path):
    """Islands stepping every 570 s and every 315 s, half a step apart, at constant tokens/s:
    interpolated on one grid the stacked total is flat (summing raw step points per bucket
    saw-tooths between one and two samples of the fast stepper)."""
    root = make_run(tmp_path / "run", steps=60)
    make_run(tmp_path / "run", steps=110, island="1")
    _island_steps(root, "0", 570.0, T0, 1.75e4)
    _island_steps(root, "1", 315.0, T0 + 140, 1.1e4)
    page = _page(tmp_path, root, T0 + 570 * 59)
    pts = _stack_top(page["tps"].split('<div class="lg">')[0])
    inner = [y for x, y in pts if x > 60]  # past the first grid points, both islands run
    assert max(inner) - min(inner) < 0.5, (min(inner), max(inner))


def test_tps_stack_has_no_ramp_at_the_range_start(tmp_path):
    """Data older than the range anchors its left edge: the first grid point already carries
    every island (the range filter used to drop the samples before t0 -> a ramp from 0)."""
    root = make_run(tmp_path / "run", steps=200)
    make_run(tmp_path / "run", steps=200, island="1")
    _island_steps(root, "0", 570.0, T0, 1.75e4)
    _island_steps(root, "1", 570.0, T0 + 285, 1.1e4)
    page = _page(tmp_path, root, T0 + 570 * 199)  # 24 h range starts around step 48
    pts = _stack_top(page["tps"].split('<div class="lg">')[0])
    ys = [y for _, y in pts]
    assert pts[0][0] <= 47 and max(ys) - min(ys) < 0.5


INTERP_JS = """
const fs=require('fs');global.localStorage={getItem(){return null},setItem(){}};global.location={hash:''};
global.document={getElementById(){return null},body:null};
eval(fs.readFileSync(process.argv[1],'utf8')+';global.YE=YetoExt;');
const a=JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify({stack:YE.interpStack(a.series,a.grid,600,60),
  axes:a.axes.map(function(r){var t=YE.durTicks(r[0],r[1],4),f=YE.durAxis(t);return t.map(f)})}));
"""


def _node_js(src, arg):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = FX.parent.parent.parent / "yeto" / "dashboard" / "static" / "sft.js"
    res = subprocess.run([node, "-e", src, str(js), json.dumps(arg)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_interp_stack_interpolates_holds_and_drops_gaps():
    # A: 100 -> 200 over [0, 600], then a 3 h gap (down), back at 11400
    a = [[0, 100], [600, 200], [1200, 200], [11400, 300], [12000, 300]]
    grid = [300, 900, 1500, 2400, 6000, 10900, 11700]
    out = _node_js(INTERP_JS, {"series": [a], "grid": grid, "axes": []})["stack"][0]
    own = [p[3] for p in out]
    assert own[0] == 150 and own[1] == 200  # linear between samples
    assert own[2] == 200  # one step after the last sample before the gap: still that step
    assert own[3] == 0 and own[4] == 0  # inside the gap: down
    assert own[5] == 300  # within one step before the next sample
    assert own[6] == 300


def test_duration_axis_uses_one_unit_and_round_ticks():
    axes = _node_js(INTERP_JS, {"series": [], "grid": [], "axes": [[250, 550], [0, 1100], [3, 9], [0, 31823],
                                                                    [250, 790]]})["axes"]
    assert axes[0] == ["200 s", "300 s", "400 s", "500 s", "600 s"]
    assert axes[4] == ["4 min", "6 min", "8 min", "10 min", "12 min", "14 min"]  # not 5/7/8 min (rounded s)
    assert all(t == "0" or t.endswith(" min") for t in axes[1]) and "20 min" in axes[1]
    assert axes[2] == ["2 s", "4 s", "6 s", "8 s", "10 s"]
    assert all(t == "0" or t.endswith(" h") for t in axes[3])


def test_push_steps_found_without_pushes_sent(tmp_path):
    """The plugin pushes from a background thread: pushes_sent is 0 on the step
    that exported; export_s still marks the push."""
    root = make_run(tmp_path / "run")
    rows = [json.loads(x) for x in (root / "island-0" / "yeto_steps.jsonl").read_text().splitlines()]
    _w(root / "island-0" / "yeto_steps.jsonl", [dict(x, pushes_sent=0) for x in rows])
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    i = r.page_view(now=T0 + 3600)["sft"]["islands"][0]
    assert i["last_push"]["step"] == 6 and i["last_push"]["export_s"] == 3.0
    page = _page(tmp_path, root, T0 + 570 * 6 + 60)
    assert "island 0 export (no data)" not in page["sync"] and "island 0 export" in page["sync"]


def test_merge_time_is_not_the_round_wall_time(tmp_path):
    """``ms`` (round open-to-done, quorum wait included) is not a merge time."""
    root = make_run(tmp_path / "run")
    _w(root / "syncer" / "tape.jsonl", [
        {"step": 1, "fragment": 0, "expected": [0], "responded": [0], "gnorm": 1.1, "ms": 31227139, "quorum_ms": 31220923, "sync_ms": 4118,
         "time_unix": T0 + 2001},
        {"step": 3, "fragment": 2, "expected": [1], "responded": [1], "gnorm": 1.0, "ms": 341336, "quorum_ms": 41335, "sync_ms": 4200,
         "sync/merge_seconds": 4.2, "time_unix": T0 + 2400}])
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    m = r.page_view(now=T0 + 3600)["sft"]["syncer"]["merges"]
    assert [x[3] for x in m] == [4.118, 4.2]
    assert m[0][5] == pytest.approx(31227.139) and m[0][6] == pytest.approx(31220.923)
    page = _page(tmp_path, root, T0 + 3000)
    ticks = re.findall(r'text-anchor="end">([^<]+)<', page["merge"])
    assert ticks and all("h" not in t for t in ticks)  # merge axis stays at seconds
    assert "quorum wait" in page["round"] and "grace wait" in page["round"] and "(no data)" not in page["round"]


def test_round_wait_split_into_quorum_grace_and_merge(tmp_path):
    """ms = quorum wait + grace wait + merge; the page plots the waits apart from the merge and
    marks the 8.7 h first round off scale instead of flattening every other round."""
    root = make_run(tmp_path / "run")
    rows = [{"step": 1, "fragment": 0, "expected": [0], "responded": [0], "gnorm": 1.1, "ms": 31227139,
             "quorum_ms": 31220923, "sync_ms": 4118, "time_unix": T0 + 600}]
    rows += [{"step": s, "fragment": s % 4, "expected": [0, 1], "responded": [1], "gnorm": 1.0,
              "ms": 474896, "quorum_ms": 5720, "sync_ms": 7283, "time_unix": T0 + 600 + 300 * s} for s in range(2, 14)]
    _w(root / "syncer" / "tape.jsonl", rows)
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    sy = r.page_view(now=T0 + 5000)["sft"]["syncer"]
    c = sy["merge_cols"]
    m = {x[c.index("step")]: x for x in sy["merges"]}
    assert m[2][c.index("quorum_s")] == pytest.approx(5.72)
    assert m[2][c.index("grace_s")] == pytest.approx(474.896 - 5.72 - 7.283)
    assert m[2][c.index("wait_s")] == pytest.approx(474.896 - 7.283)
    assert m[1][c.index("wait_s")] == pytest.approx(31227.139 - 4.118)
    page = _page(tmp_path, root, T0 + 5000)
    assert "1 off scale" in page["round"] and "8.7 h" in page["round"]
    ticks = re.findall(r'text-anchor="end">([^<]+)<', page["round"])
    assert ticks and all(t == "0" or t.endswith(" min") for t in ticks)
    assert "Wait per round" in page["dist"] and "Outer step time" not in page["dist"]


SYNCER_ROUND_LOG = (
    "2026-11-07T02:36:46.696019Z  INFO yeto_syncer::server: grace window armed step=208 fragment=15 "
    "step_secs=Some(590.8) quorum_secs=5.72008911 sync_secs=7.35 grace_ms=600000 grace_floor_ms=0 terminal=false\n"
    "2026-11-07T02:44:35.872551Z  INFO yeto_syncer::server: outer step step=208 fragment=15 "
    "responders=[Member { learner_id: 1, generation: 6 }] gnorm=\"2.0\" ms=474896\n"
    "2026-11-07T02:49:28.467323Z  INFO yeto_syncer::server: grace window armed step=210 fragment=17 "
    "step_secs=Some(581.4) quorum_secs=6.356455203 sync_secs=7.28 grace_ms=600000 grace_floor_ms=0 terminal=false\n"
    "2026-11-07T02:56:11.548825Z  INFO yeto_syncer::server: outer step step=210 fragment=17 "
    "responders=[Member { learner_id: 1, generation: 6 }] gnorm=\"2.7\" ms=409437\n")


def test_syncer_log_round_lines():
    armed, done = (sft.from_syncer_log_line(x) for x in SYNCER_ROUND_LOG.splitlines()[:2])
    assert armed["event"] == "sft_syncer_round" and armed["phase"] == "armed" and armed["step"] == 208
    assert armed["quorum_s"] == pytest.approx(5.72008911)
    assert done["phase"] == "done" and done["fragment"] == 15 and done["round_s"] == pytest.approx(474.896)


def test_round_timing_falls_back_to_syncer_log(tmp_path):
    """A tape merge without ms / quorum_ms takes them from syncer.log; a round only in the log
    still gets a row (merge time unknown)."""
    root = make_run(tmp_path / "run")
    with open(root / "syncer" / "syncer.log", "a") as fh:
        fh.write(SYNCER_ROUND_LOG)
    _w(root / "syncer" / "tape.jsonl", [{"step": 208, "fragment": 15, "expected": [1], "responded": [1],
                                         "sync/merge_seconds": 7.283, "gnorm": 2.0, "time_unix": 1794019475.87}])
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    sy = r.page_view(now=1794020200.0)["sft"]["syncer"]
    c = sy["merge_cols"]
    m = {x[c.index("step")]: x for x in sy["merges"]}
    assert m[208][c.index("round_s")] == pytest.approx(474.896)
    assert m[208][c.index("quorum_s")] == pytest.approx(5.72008911)
    assert m[208][c.index("grace_s")] == pytest.approx(474.896 - 5.72008911 - 7.283)
    assert m[210][c.index("merge_s")] is None and m[210][c.index("round_s")] == pytest.approx(409.437)
    assert m[210][c.index("quorum_s")] == pytest.approx(6.356455203) and m[210][c.index("wait_s")] is None


def _eval_row(step, loss, d_ck, d_base, se=0.0004, prev=None):
    return {"step": step, "ckpt_written": "2026-11-06T04:22:49Z", "loss": loss, "ppl": 2.0,
            "vs_base": {"dloss": d_base, "mean": d_base - 0.009, "se": 0.0015, "better": 1.0},
            "vs_ckpt100": {"dloss": d_ck, "mean": d_ck - 0.001, "se": se, "better": 0.3},
            "vs_prev": None if prev is None else {"dloss": -0.001, "mean": -0.001, "se": 0.0002, "better": 0.6},
            "prev_step": prev, "merges_total": step}


# rows shaped like the run's eval/rows.jsonl: the references are constant (loss - dloss)
CK100, BASE = 0.6723, 0.7558
EVAL_ROWS = [_eval_row(s, l, l - CK100, l - BASE, prev=p) for s, l, p in
             ((32, 0.6824, None), (48, 0.6747, 32), (96, 0.6768, 48), (112, 0.6744, 96))]


def _write_eval(root, rows, extra=""):
    p = root / "eval" / "rows.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + extra)
    return p


def test_eval_rows_dedup_sort_and_bad_lines():
    dup = _eval_row(48, 0.6750, 0.6750 - CK100, 0.6750 - BASE, prev=32)
    text = "\n".join([json.dumps(EVAL_ROWS[2]), "{not json", json.dumps(EVAL_ROWS[1]), "[1, 2]",
                      json.dumps({"loss": 0.7}), json.dumps({"step": True, "loss": 0.7}), "",
                      json.dumps(EVAL_ROWS[0]), json.dumps(dup)]) + "\n"
    rec, bad = sft.from_eval_rows(text)
    assert rec["event"] == "sft_eval" and bad == 4
    rows = rec["rows"]
    assert [r["step"] for r in rows] == [32, 48, 96]
    assert rows[1]["loss"] == 0.6750  # the last line of a step wins
    for r in rows:  # dloss = ckpt loss - reference loss, so the reference is loss - dloss
        assert r["vs_ckpt100"]["ref_loss"] == pytest.approx(CK100)
        assert r["vs_base"]["ref_loss"] == pytest.approx(BASE)
    assert rows[0].get("vs_prev") is None and rows[0]["prev_step"] is None and rows[1]["prev_step"] == 32
    assert sft.from_eval_rows("")[0]["rows"] == []
    nan = sft.eval_row({"step": 8, "loss": float("nan"), "vs_base": {"dloss": "x", "se": 0.1}})
    assert nan["loss"] is None and nan["vs_base"]["dloss"] is None and nan["vs_base"]["ref_loss"] is None
    assert "vs_ckpt100" not in nan  # only the comparisons a row has


def test_eval_in_view_and_rewrite_replaces(tmp_path):
    root = make_run(tmp_path / "run", steps=3)
    p = _write_eval(root, EVAL_ROWS[2:] + EVAL_ROWS[:2], extra='{"step": 9, "loss"')  # torn tail is a bad line
    assert p in discover([str(root)])
    from yeto.dashboard.serve import DashboardState

    st = DashboardState(Reducer(), [str(root)])
    st.poll()
    ev = st.reducer.page_view(now=T0 + 3600)["sft"]["eval"]["rows"]
    assert [r["step"] for r in ev] == [32, 48, 96, 112]
    assert st.reducer.sft.islands.keys() == {"0"}  # eval rows never create an island
    _write_eval(root, [EVAL_ROWS[0], dict(EVAL_ROWS[3], loss=0.6740), EVAL_ROWS[0]])
    st.poll()
    st.poll()
    ev = st.reducer.sft_view(T0)["eval"]["rows"]
    assert [r["step"] for r in ev] == [32, 112] and ev[1]["loss"] == 0.6740
    _write_eval(root, [])
    st.poll()
    assert st.reducer.sft_view(T0)["eval"]["rows"] == []


def test_eval_chart_renders_and_empty_state(tmp_path):
    root = make_run(tmp_path / "run")
    page = _page(tmp_path, root, T0 + 3600)  # no eval file
    assert "no held-out eval yet" in page["eval"] and "<svg" not in page["eval"]
    _write_eval(root, [])
    page = _page(tmp_path, root, T0 + 3600)
    assert "no held-out eval yet" in page["eval"]
    _write_eval(root, EVAL_ROWS)
    svg = _page(tmp_path, root, T0 + 3600)["eval"]
    assert 'class="evloss"' in svg and 'data-ref="ckpt100"' in svg and svg.count('class="eveb"') == 4
    assert svg.count("<circle") == 4
    # base is ~0.08 above while the checkpoints spread ~0.01: noted off scale, not drawn
    assert 'data-ref="base"' not in svg and "base 0.7558" in svg and "off scale" in svg
    ticks = re.findall(r'text-anchor="middle">(\d+)<', svg)
    assert ticks and int(ticks[0]) >= 32 and int(ticks[-1]) <= 112  # x is the outer step
    near = [_eval_row(s, l, l - CK100, l - 0.684) for s, l in ((32, 0.6824), (48, 0.6747))]
    _write_eval(root, near)
    svg = _page(tmp_path, root, T0 + 3600)["eval"]
    assert 'data-ref="base"' in svg and "off scale" not in svg
    _write_eval(root, [EVAL_ROWS[0]])
    svg = _page(tmp_path, root, T0 + 3600)["eval"]
    assert svg.count("<circle") == 1 and 'data-ref="ckpt100"' in svg


TIP_JS = """
const fs=require('fs');global.localStorage={getItem(){return null},setItem(){}};global.location={hash:''};
global.document={getElementById(){return null},body:null};
eval(fs.readFileSync(process.argv[1],'utf8')+';global.YE=YetoExt;');
process.stdout.write(JSON.stringify(JSON.parse(process.argv[2]).map(r=>YE.evalTip(r))));
"""


def test_eval_tooltip_shows_step_loss_and_deltas():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = FX.parent.parent.parent / "yeto" / "dashboard" / "static" / "sft.js"
    rows = sft.from_eval_rows(json.dumps(EVAL_ROWS[3]) + "\n" + json.dumps({"step": 5, "loss": 0.7}) + "\n")[0]["rows"]
    res = subprocess.run([node, "-e", TIP_JS, str(js), json.dumps(rows)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    bare, full = json.loads(res.stdout)
    assert "step 112" in full and "0.6744" in full and "2026-11-06 04:22 UTC" in full
    assert "vs ckpt100<b>\u0394 +0.0021 (per-sample +0.0011 \u00b1 0.0004)" in full
    assert "vs base<b>\u0394 -0.0814 (per-sample -0.0904 \u00b1 0.0015)" in full
    assert "step 5" in bare and "vs " not in bare and "n/a" not in bare
