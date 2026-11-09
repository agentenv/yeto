"""SFT dashboard: islands resumed from a checkpoint (steps repeat, counters restart) and
stopped / restarting islands (status, active count, HBM bar)."""

import json
import shutil
import subprocess

import pytest

from yeto.dashboard import sft
from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

from dashboard_helpers import T0
from test_dashboard_sft import SMOKE_SFT, make_run

STEP = 330.0


def _a(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def launch(d, steps, *, t_first, loss0, hbm, files=("log", "sync", "bench", "health")):
    """Append one launch of an island: absolute ``steps``, bench / health counting from 1."""
    steps = list(steps)
    if "log" in files:
        _a(d / "logging.jsonl", [{"loss": loss0 + 0.001 * s, "grad_norm": 0.3, "learning_rate": 7e-6,
                                  "iteration": f"{s}/762", "memory(GiB)": 51.0} for s in steps])
    if "sync" in files:
        _a(d / "yeto_steps.jsonl", [{"iteration": s, "time": t_first + STEP * i, "target_tokens": 6.6e5,
                                     "sync_s": 0.01, "export_s": 0.0, "apply_s": 0.0, "pushes_sent": 0,
                                     "applies": 0} for i, s in enumerate(steps)])
    if "bench" in files:
        _a(d / "bench_metrics.jsonl", [{"step": i + 1, "tokens": 5.2e6, "wall_s": STEP + (s % 7),
                                        "tokens_per_s": 1.6e4 + s, "mfu_6N_pct": 13.1} for i, s in enumerate(steps)])
    if "health" in files:
        for rank in (0, 23):
            _a(d / f"health.rank{rank}.jsonl", [{"step": i + 1, "rank": rank, "reserved_peak_gib": hbm + rank / 100}
                                                for i, _ in enumerate(steps)])


def view(root, now):
    r = Reducer(run="sft")
    load_all(r, [str(root)])
    v = r.page_view(now=now)["sft"]
    return v, [dict(zip(v["cols"], row)) for row in v["islands"][0]["rows"]]


def slow5(tmp_path, *, second=range(166, 194)):
    """slow5 on 11-07: launch 1 ran 1..173, the island resumed from checkpoint-165."""
    d = tmp_path / "run" / "island-5"
    launch(d, range(1, 174), t_first=T0, loss0=0.9, hbm=50.0)
    t2 = T0 + STEP * 173 + 3600  # restart + catch-up
    launch(d, second, t_first=t2, loss0=0.5, hbm=52.0)
    return tmp_path / "run", t2


def test_launch_starts_resume_and_clean_continuation():
    seg = sft._segments
    # resume from 165 after running to 173
    assert sft._launch_starts([173, 28], seg(list(range(1, 174)) + list(range(166, 194)))) == [1, 166]
    # a launch that resumed exactly where the previous stopped leaves no backstep in the anchor
    assert sft._launch_starts([6, 4], seg(list(range(1, 11)))) == [1, 7]
    # first launch began at a resumed step
    assert sft._launch_starts([5, 3], seg(list(range(40, 45)) + list(range(43, 46)))) == [40, 43]
    # counter one record ahead of the anchor (file pulled before the other) stays in the same launch
    assert sft._launch_starts([10, 1], seg(list(range(1, 11)))) == [1, 11]
    # no anchor: launches follow each other
    assert sft._launch_starts([3, 2], []) == [1, 4]


def test_resumed_island_rows_follow_the_newest_launch(tmp_path):
    root, t2 = slow5(tmp_path)
    v, rows = view(root, t2 + STEP * 28)
    steps = [r["step"] for r in rows]
    assert steps == list(range(1, 194))  # no duplicates, no fold-back
    ts = [r["ts"] for r in rows]
    assert all(b > a for a, b in zip(ts, ts[1:]))  # time strictly increasing: no backward line in the charts
    by = {r["step"]: r for r in rows}
    assert by[165]["loss"] == pytest.approx(0.9 + 0.165)  # before the checkpoint: launch 1
    assert by[166]["loss"] == pytest.approx(0.5 + 0.166)  # from the checkpoint on: launch 2
    assert by[193]["loss"] == pytest.approx(0.5 + 0.193)
    assert by[166]["ts"] == t2 and by[166]["tps"] == 1.6e4 + 166  # bench segment 2 placed at 166, not 174
    assert by[166]["hbm_gib"] == pytest.approx(52.23) and by[165]["hbm_gib"] == pytest.approx(50.23)
    assert by[100]["tps"] == 1.6e4 + 100
    assert rows[-1]["step"] == 193 and v["islands"][0]["last_ts"] == by[193]["ts"]


def test_mid_catch_up_drops_the_dead_launch_tail(tmp_path):
    """3 steps into the new launch, the dead launch's 169..173 (log, sync, bench, health) are gone."""
    root, t2 = slow5(tmp_path, second=range(166, 169))
    _, rows = view(root, t2 + STEP * 3)
    assert [r["step"] for r in rows][-4:] == [165, 166, 167, 168]
    assert rows[-1]["loss"] == pytest.approx(0.5 + 0.168) and rows[-1]["hbm_gib"] == pytest.approx(52.23)


def test_lagging_stream_tail_is_hidden_until_it_shows_the_restart(tmp_path):
    """yeto_steps (rank0 node) already shows the new launch, logging.jsonl (last stage) not yet."""
    d = tmp_path / "run" / "island-5"
    launch(d, range(1, 174), t_first=T0, loss0=0.9, hbm=50.0)
    t2 = T0 + STEP * 200
    launch(d, range(166, 169), t_first=t2, loss0=0.5, hbm=52.0, files=("sync", "bench", "health"))
    _, rows = view(tmp_path / "run", t2 + STEP * 3)
    assert rows[-1]["step"] == 168 and rows[-1]["ts"] == t2 + 2 * STEP
    assert rows[-1]["loss"] is None  # the new launch's loss has not arrived; the dead launch's is not shown
    # and the other way round: logging.jsonl restarted, yeto_steps not yet
    d2 = tmp_path / "run2" / "island-5"
    launch(d2, range(1, 174), t_first=T0, loss0=0.9, hbm=50.0)
    launch(d2, range(166, 169), t_first=t2, loss0=0.5, hbm=52.0, files=("log",))
    _, rows = view(tmp_path / "run2", t2)
    assert rows[-1]["step"] == 168 and rows[-1]["loss"] == pytest.approx(0.5 + 0.168)


def test_clean_continuation_maps_bench_after_the_previous_launch(tmp_path):
    d = tmp_path / "run" / "island-0"
    launch(d, range(1, 7), t_first=T0, loss0=0.9, hbm=50.0, files=("log", "sync"))
    launch(d, range(7, 11), t_first=T0 + 7 * STEP + 3600, loss0=0.9, hbm=50.0, files=("log", "sync"))
    _a(d / "bench_metrics.jsonl", [{"step": s, "wall_s": 100.0 + s} for s in range(1, 7)] +
       [{"step": s, "wall_s": 200.0 + s} for s in range(1, 5)])
    _, rows = view(tmp_path / "run", T0 + 20 * STEP)
    assert [r["step_s"] for r in rows] == [101, 102, 103, 104, 105, 106, 201, 202, 203, 204]


def test_step_time_never_spans_a_restart(tmp_path):
    d = tmp_path / "run" / "island-1"
    launch(d, range(1, 11), t_first=T0, loss0=0.9, hbm=50.0, files=("log", "sync"))
    launch(d, range(8, 12), t_first=T0 + 20 * STEP, loss0=0.5, hbm=50.0, files=("log", "sync"))
    _, rows = view(tmp_path / "run", T0 + 24 * STEP)
    by = {r["step"]: r for r in rows}
    assert by[7]["step_s"] == STEP and by[8]["step_s"] is None and by[9]["step_s"] == STEP


def test_incremental_resume_never_regresses(tmp_path):
    from yeto.dashboard.serve import DashboardState

    d = tmp_path / "run" / "island-5"
    launch(d, range(1, 174), t_first=T0, loss0=0.9, hbm=50.0)
    st = DashboardState(Reducer(), [str(tmp_path / "run")])
    st.poll()
    t2 = T0 + STEP * 200
    for k in range(166, 176):
        launch(d, [k], t_first=t2 + STEP * (k - 166), loss0=0.5, hbm=52.0)
        st.poll()
        rows = st.reducer.sft_view(t2)["islands"][0]["rows"]
        steps = [r[0] for r in rows]
        # from the new launch's first step on, the dead launch's 166..173 are gone: the last row is always k
        assert steps == list(range(1, k + 1))
        assert rows[-1][2] == pytest.approx(0.5 + 0.001 * k)
    rows = st.reducer.sft_view(t2)["islands"][0]["rows"]
    assert [r[0] for r in rows] == list(range(1, 176))


@pytest.mark.parametrize("age_steps,status", [(1.0, "ok"), (1.5 + 300 / STEP - 0.01, "ok"),
                                              (1.5 + 300 / STEP + 0.05, "stopped"), (30, "stopped")])
def test_status_after_the_last_step(tmp_path, age_steps, status):
    d = tmp_path / "run" / "island-3"
    launch(d, range(1, 21), t_first=T0, loss0=0.9, hbm=50.0, files=("log", "sync"))  # step time from ts: 330 s
    last = T0 + STEP * 19
    v, _ = view(tmp_path / "run", last + age_steps * STEP)
    assert v["islands"][0]["status"] == status


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


def test_page_greys_out_a_stopped_island(tmp_path):
    root = make_run(tmp_path / "run", steps=6)  # island 0: last step at T0 + 3420, 570 s steps
    d = root / "island-1"
    launch(d, range(1, 21), t_first=T0 - 40 * STEP, loss0=0.9, hbm=53.0)  # stopped ~20 steps ago
    page = _page(tmp_path, root, T0 + 3420 + 60)
    assert "Islands active / total</div><div class=\"kv\">1 / 2<" in page["kpis"]
    assert "stopped / restarting" in page["kpis"]
    rows = page["nodes"].split("<tr")
    stopped = next(x for x in rows if "island 1" in x)
    live = next(x for x in rows if "island 0" in x)
    assert ' class="off"' in stopped and "last step 20" in stopped and 'class="bf idle"' in stopped
    assert ' class="off"' not in live and "last step" not in live and "bf idle" not in live


def test_syncer_restart_and_catch_up_log_categories():
    def cat(m):
        return (sft.from_syncer_log_line("2026-11-07T09:42:08.041730Z  INFO yeto_syncer::server: " + m)
                or {}).get("category")

    assert cat('syncer listening port=29400 semantic_profile_sha256="1463"') == "started"
    assert cat("resumed from checkpoint step=272") == "resumed"
    assert cat("initial cut: the learner's send queues are full; a forwarder sends the rest of it learner_id=5"
               " generation=3 fragment=0 version=257") == "catch_up"
    assert cat("background resend started learner_id=1 generation=9 fragments=64 bytes=4 kind=CatchUp("
               "CatchUpTail { rejoin: true }") == "catch_up"
    e = sft.from_syncer_log_line("2026-11-07T10:07:40.111126Z  INFO yeto_syncer::server: background resend caught"
                                 " up; its tail runs learner_id=5 generation=3 fragments=64 sent=127 bytes=961054531584"
                                 " pending=0 seconds=1079.025028366")
    assert e["category"] == "caught_up" and e["learner"] == 5 and e["seconds"] == pytest.approx(1079.025)
    assert cat("background resend drained learner_id=1 generation=9 fragments=64 sent=64 seconds=962.9") == "caught_up"
    assert cat("merge thread count merge_threads=32") is None


def test_stop_thresholds_are_settable():
    r = Reducer(run="sft", thresholds={"sft_stop_factor": 3.0, "sft_stop_slack_s": 0})
    st = r.sft
    for s in (1, 2, 3):
        st.fold({"event": "sft_sync_step", "island_id": "0", "step": s, "time_unix": T0 + 100 * s}, T0 + 100 * s)
    assert r.sft_view(T0 + 300 + 299)["islands"][0]["status"] == "ok"  # 3 x 100 s + 0 s
    assert r.sft_view(T0 + 300 + 301)["islands"][0]["status"] == "stopped"
