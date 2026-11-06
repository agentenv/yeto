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


def test_rl_view_has_no_sft(tmp_path):
    r = Reducer()
    load_all(r, [str(FX / "kill44")])
    assert "sft" not in r.page_view()


def test_export_page_renders_sft(tmp_path):
    root = make_run(tmp_path / "run")
    make_run(tmp_path / "run", island="1", v2=True)
    r = Reducer(run="sft")
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
    assert "960" in page["kpis"] and "571" in page["kpis"]
    assert "<svg" in page["fleet"] and "p50" in page["dist"] and "Card-hours" in page["cost"]


# Reuse the DOM stub of dashboard_smoke.js, dumping the SFT containers instead.
SMOKE_SFT = (SMOKE.read_text().split("eval(appM[1]);")[0] + """
eval(appM[1]);
process.stdout.write(JSON.stringify({title: els.title.innerHTML, kpis: els.sKpis.innerHTML,
  nodes: els.sNodes.innerHTML, fleet: els.fLoss.innerHTML, tps: els.fTps.innerHTML, sync: els.fSync.innerHTML,
  merge: els.fMerge.innerHTML, round: els.fRound.innerHTML, dist: els.sDist.innerHTML,
  cost: els.sCostK.innerHTML, sft_hidden: els.sftRoot.hidden, rl_hidden: els.rlRoot.hidden}));
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
    kpi = re.search(r"Fast island step p50</div><div class=\"kv\">([^<]+)<", page["kpis"]).group(1)
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


def test_tps_stack_is_held_on_a_shared_grid(tmp_path):
    """Two islands stepping half a period apart: the stacked total stays flat
    while both run (bucket sums of the raw step points saw-tooth)."""
    root = make_run(tmp_path / "run", steps=12)
    make_run(tmp_path / "run", steps=12, island="1")
    rows = [json.loads(x) for x in (root / "island-1" / "yeto_steps.jsonl").read_text().splitlines()]
    _w(root / "island-1" / "yeto_steps.jsonl", [dict(x, time=x["time"] + 285) for x in rows])
    page = _page(tmp_path, root, T0 + 570 * 12 + 300)
    svg = page["tps"].split('<div class="lg">')[0]
    ys = _top_line_ys(svg)
    both = [y for y in ys if y == min(ys)]
    assert len(set(ys)) <= 3 and len(both) > 0.6 * len(ys)  # 0 / one island / both islands


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
    assert "quorum wait" in page["round"] and "(no data)" not in page["round"]
