"""Live topology of a run with a syncer (yeto/dashboard/topology_diloco.py, alerts_diloco.py,
static/topology_diloco.js), checked against trimmed real tapes:

* ``async-e2e`` / ``async-compaction``: async GRPO (one trainer, two cross-island samplers, no
  syncer): the V1 drawing, its view identical to the one main computed (``*.v1-golden.json``);
* ``run-a``: decoupled DiLoCo, two colocated islands (island 1 relaunched once);
* ``run-b``: decoupled DiLoCo, two fixed-partition islands (2 cards each);
* ``k31-synced-group``: full-parameter synced group, 2 trainers + 2 samplers, shared store,
  syncer tape behind the islands.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from dashboard_helpers import FX

from yeto.dashboard import alerts, alerts_diloco, sources, topology_diloco
from yeto.dashboard.export import export_html, export_view
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.serve import DashboardState
from yeto.dashboard.sources import load_all

D = FX / "diloco"


def load(name: str, *, skip=lambda path: False) -> Reducer:
    r = Reducer(run=name)
    paths = [p for p in sources.discover([str(D / name)]) if not skip(p)]
    load_all(r, [str(p) for p in paths])
    return r


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def diloco(r: Reducer, now: float | None = None) -> dict:
    view = r.page_view(now=now if now is not None else r.max_ts)
    return view["topology"]["diloco"]


# -- which drawing --------------------------------------------------------------------
@pytest.mark.parametrize("name,kind,why", [
    ("async-e2e", "async", "trainers declare no outer sync"),
    ("async-compaction", "async", "trainers declare no outer sync"),
    ("run-a", "island", "declare outer sync"),
    ("run-b", "island", "no sampler island"),
    ("k31-synced-group", "synced_group", "sampler islands 2"),
])
def test_layout_from_the_tapes(name, kind, why):
    r = load(name)
    lay = r.page_view(now=r.max_ts)["layout"]
    assert lay["kind"] == kind and why in lay["reason"], lay


@pytest.mark.parametrize("name", ["async-e2e", "async-compaction"])
def test_async_grpo_view_is_unchanged(name):
    """The V1 topology and alerts of an async GRPO run equal what main computed for the same tapes."""
    golden = json.loads((D / f"{name}.v1-golden.json").read_text())
    r = load(name)
    for when, want in golden.items():
        view = json.loads(json.dumps(r.page_view(now=want["now"]), default=str))
        assert "kind" not in (view.get("topology") or {}) and "diloco" not in (view.get("topology") or {})
        assert view.get("topology") == want["topology"], when
        got = [{k: a[k] for k in ("sev", "rule", "title", "detail", "island")} for a in view["overview"]["alerts"]]
        assert got == want["alerts"], when


def test_a_stray_syncer_tape_does_not_turn_async_into_diloco(tmp_path):
    """The trainers declared no outer sync: a syncer tape next to them changes nothing."""
    r = load("async-e2e")
    for rec in lines(D / "run-a" / "yeto-tape.jsonl")[:3]:
        r.feed(rec, source=str(tmp_path / "yeto-tape.jsonl"))
    assert r.page_view(now=r.max_ts)["layout"]["kind"] == "async"


def test_a_late_syncer_tape_keeps_the_drawing():
    """Run B's islands declare outer sync before any merge: the panel is the island drawing from the
    start, versions stay 0 and read "syncer tape behind" until the tape arrives, then match it."""
    r = load("run-b", skip=lambda p: p.name == "yeto-tape.jsonl")
    lay = r.page_view(now=r.max_ts)["layout"]
    assert lay["kind"] == "island" and "declare outer sync" in lay["reason"]
    d = diloco(r)
    assert d["versions"] == [0, 0, 0, 0] and d["behind"] and d["stuck"] == []
    assert not [a for a in alerts_diloco.evaluate(r, r.max_ts, r.overview(now=r.max_ts)["islands"])
                if a["rule"] in ("diloco_fragment_stuck", "diloco_syncer_idle")]
    for off, rec in enumerate(lines(D / "run-b" / "yeto-tape.jsonl")):
        r.feed(rec, source="late/yeto-tape.jsonl", offset=off)
    d = diloco(r)
    assert r.page_view(now=r.max_ts)["layout"]["kind"] == "island"
    assert d["versions"] == [21, 22, 23, 24] and not d["behind"]


def test_sft_runs_and_runs_without_trainers_get_no_panel():
    r = Reducer(run="sft")
    for rec in lines(D / "run-a" / "yeto-tape.jsonl"):
        r.feed(rec, source="yeto-tape.jsonl")
    assert r.page_view(now=r.max_ts)["layout"]["kind"] == "async"
    assert "topology" not in r.page_view(now=r.max_ts)


# -- syncer, trainers, fragments ---------------------------------------------------------
def test_run_a_syncer_and_trainers_match_the_tapes():
    r = load("run-a")
    merges = lines(D / "run-a" / "yeto-tape.jsonl")
    d = diloco(r)
    assert d["P"] == 4 and d["M"] == 2 and d["N"] == 1 and d["quorum"] == 1
    want = [max(m["step"] for m in merges if m["fragment"] == f) for f in range(4)]
    assert d["versions"] == want == [13, 14, 15, 16]
    sh = d["shards"][0]
    assert (sh["lo"], sh["hi"], sh["step"], sh["merges"]) == (0, 4, 16, 16)
    last = merges[-1]
    assert sh["last"]["step"] == last["step"] and sh["last"]["quorum_s"] == pytest.approx(last["quorum_ms"] / 1000)
    assert sh["last"]["checkpoint_s"] == pytest.approx(last["checkpoint_ms"] / 1000)
    t0, t1 = d["trainers"]
    snap = [x for x in lines(D / "run-a" / "events" / "dlrl-qwen4b-1123-a-l0-modal.jsonl") if x["event"] == "rl_policy_snapshot"][-1]
    assert t0["versions"] == snap["rl/fragment_versions"] and t0["hash"] == snap["rl/policy_hash"][:8]
    assert t0["step"] == 22 and t0["outer"] == 4 and t0["lr"] == 1e-5
    # island 1 stopped at [5, 6, 7, 4]: the merges after its vector are pending for it
    assert t1["versions"] == [5, 6, 7, 4] and t1["pending"] == {"0": 1, "1": 1, "2": 1, "3": 2} and not t1["live"]
    assert t0["island"]["colocated"] and t0["nodes"][0]["roles"] == ["T+R"]
    assert t1["nodes"][0]["incarnation"] == 2  # fleet: island_lost then a relaunched island_ready
    # since anchor: rounds since the fragment was last applied (sync/applied_fragments)
    rounds = [x for x in lines(D / "run-a" / "events" / "dlrl-qwen4b-1123-a-l0-modal.jsonl") if x["event"] == "rl_local_round"]
    since = [0] * 4
    for x in rounds:
        since = [0 if f in x["sync/applied_fragments"] else since[f] + 1 for f in range(4)]
    assert t0["since"] == since
    assert d["H"] == min(x["realized_h"] for x in lines(D / "run-a" / "events" / "dlrl-qwen4b-1123-a-l0-modal.jsonl")
                         if x["event"] == "rl_fragment_push")


def test_run_b_fixed_partition_islands_and_flows():
    r = load("run-b")
    merges = lines(D / "run-b" / "yeto-tape.jsonl")
    d = diloco(r)
    assert d["done"] and d["versions"] == [21, 22, 23, 24]
    for t in d["trainers"]:
        isl = t["island"]
        assert isl["placement"] == "fixed-partition" and isl["exec_mode"] == "partitioned-serial" and not isl["colocated"]
        assert isl["engines"] == 1 and t["nodes"][0]["roles"] == ["T", "R"] and t["nodes"][0]["inferred"]
        assert isl["p50_s"] is None  # the rollout reports group latency 0.0: not timed, not "0 s"
    assert d["trainers"][0]["island"]["served"]["v"] == 29
    # PUSH windows are the responders' wire windows; BCAST windows end when the island received it
    mid = merges[12]["time_unix"] + 1
    d = diloco(r, now=mid)
    pushes = {(f["i"], round(f["t1"], 3)): f for f in d["flows"] if f["k"] == "p"}
    m = merges[12]
    for resp in m["responders"]:
        f = pushes[(str(resp["id"]), round(resp["arrived_unix"], 3))]
        assert f["t0"] == pytest.approx(resp["arrived_unix"] - resp["recv_seconds"]) and f["f"] == m["fragment"]
    bc = [f for f in d["flows"] if f["k"] == "b" and f["t0"] == m["time_unix"]]
    assert {f["i"] for f in bc} == {"0", "1"} and all(f["t1"] >= f["t0"] for f in bc)


def test_synced_group_shared_store_and_syncer_tape_behind():
    r = load("k31-synced-group")
    d = diloco(r)
    assert d["kind"] == "synced_group" and d["P"] == 8 and d["M"] == 2
    assert d["versions"] == [41, 42, 43, 36, 37, 38, 39, 40]
    t0 = d["trainers"][0]
    assert t0["versions"] == [41, 42, 43, 44, 45, 38, 39, 40] and t0["outer"] == 5
    assert d["behind"] and d["stuck"] == [] and t0["pending_n"] == 0  # no negative lag from a stale copy
    assert t0["island"] is None and t0["nodes"][0]["roles"] == ["T"]
    assert not d["follow"] and d["publisher"] == "0" and len(d["sampler_nodes"]) == 2


def test_synced_group_samplers_following_their_trainer(tmp_path):
    """--rl-sampler-follow trainer: each sampler's store ends in trainer<k>; one lane per pair."""
    shutil.copytree(D / "k31-synced-group", tmp_path / "k31")
    for sid, tid in (("2", "0"), ("3", "1")):
        p = tmp_path / "k31" / "events" / f"k31-dlrl-l{sid}-modal.jsonl"
        recs = lines(p)
        for x in recs:
            if x["event"] == "rl_sampler_start":
                x["store"] += f"/trainer{tid}"
        p.write_text("".join(json.dumps(x) + "\n" for x in recs))
    r = Reducer(run="k31")
    load_all(r, [str(tmp_path / "k31")])
    d = diloco(r)
    assert d["follow"] and {k: v["samplers"] for k, v in d["lanes"].items()} == {"0": ["2"], "1": ["3"]}
    assert d["lanes"]["0"]["substore"] == "trainer0" and d["lanes"]["0"]["written"] > 0


def test_syncer_shards_from_their_tapes(tmp_path):
    """Two shard tapes: fragments [0, 2) / [2, 4), steps t = 1 / 2 (mod 2); the shard tape next to
    a syncer tape given as a file is followed too."""
    merges = lines(D / "run-b" / "yeto-tape.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    (out / "yeto-tape.jsonl").write_text("".join(json.dumps(m) + "\n" for m in merges if m["step"] % 2 == 1))
    (out / "yeto-tape.shard1.jsonl").write_text("".join(json.dumps(m) + "\n" for m in merges if m["step"] % 2 == 0))
    assert [p.name for p in sources.discover([str(out / "yeto-tape.jsonl")])] == ["yeto-tape.jsonl", "yeto-tape.shard1.jsonl"]
    r = load("run-b", skip=lambda p: p.name == "yeto-tape.jsonl")
    load_all(r, [str(out / "yeto-tape.jsonl")])
    d = diloco(r)
    assert d["N"] == 2 and [(s["lo"], s["hi"]) for s in d["shards"]] == [(0, 2), (2, 4)]
    assert [s["step"] for s in d["shards"]] == [23, 24] and [s["merges"] for s in d["shards"]] == [12, 12]
    assert {f["sh"] for f in d["flows"] if f["k"] == "p"} <= {0, 1}


def test_syncer_tape_that_appears_after_start_is_followed(tmp_path, monkeypatch):
    run_dir, home = tmp_path / "runs" / "r1", tmp_path / "home"
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: run_dir)
    monkeypatch.setenv("HOME", str(home))
    paths = sources.run_sources("r1")
    shutil.copytree(D / "run-b" / "events", run_dir / "events")
    state = DashboardState(Reducer(run="r1"), paths)
    state.poll()
    assert state.reducer.counts["syncer"] == 0
    (home / "yeto-output").mkdir(parents=True)
    shutil.copy(D / "run-b" / "yeto-tape.jsonl", home / "yeto-output" / "yeto-tape.jsonl")
    state.poll()
    assert state.reducer.counts["syncer"] == len(lines(D / "run-b" / "yeto-tape.jsonl"))


# -- alerts --------------------------------------------------------------------------------
def test_partial_quorum_missed_rounds_are_info():
    r = load("run-a")
    missed = [a for a in r.overview(now=r.max_ts)["alerts"] if a["rule"] == "missed"]
    assert missed and all(a["sev"] == 2 and "partial quorum" in a["detail"] for a in missed)


def test_full_quorum_missed_rounds_still_warn():
    from dashboard_helpers import merge
    r = Reducer(run="q")
    for s in range(1, 4):
        r.feed(dict(merge(s, [0, 1], [0], t=1_790_000_000.0 + s), quorum=2))
    missed = [a for a in alerts.consecutive_missed(r.rounds(), r.thresholds)]
    assert missed and missed[0]["sev"] == 1


def _base(**over) -> dict:
    """A healthy two-trainer view as topology_diloco.view returns it (the fields the rules read)."""
    tr = {"id": "0", "status": "ok", "live": True, "reason": None, "last_event_ts": 990.0, "failure": None,
          "lr_zero": False, "pending": {}, "pending_n": 0, "pending_oldest_ts": None, "phase": "training",
          "phase_raw": "train", "phase_since": 980.0, "island": {"rollout_p50_s": 10.0}}
    d = {"running": True, "done": False, "roster": 2, "M": 2, "quorum": 2, "behind": False, "stuck": [],
         "versions": [5, 6], "period_s": 60.0, "merge_gap_s": 30.0, "last_merge_ts_any": 990.0, "rounds": [],
         "shards": [{"last": {"step": 6, "frag": 1, "ts": 990.0, "attempt": 1, "responded": ["0", "1"],
                              "expected": ["0", "1"], "rejected": 0}}],
         "trainers": [tr, dict(tr, id="1")]}
    for k, v in over.items():
        d[k] = v
    return d


@pytest.mark.parametrize("rule,bad", [
    ("roster", lambda d: d.update(roster=1, trainers=[d["trainers"][0], dict(d["trainers"][1], live=False, status="lost")])),
    ("trainer_lost", lambda d: d["trainers"][1].update(live=False, status="lost")),
    ("rounds_open", lambda d: d.update(rounds=[{"step": 7, "frag": 0, "blocked": True, "attempt": 2, "age_s": 400.0,
                                                "pushed": ["0"], "missing": ["1"]}])),
    ("bcast_backlog", lambda d: d["trainers"][0].update(pending={"1": 2}, pending_n=2, pending_oldest_ts=600.0)),
    ("strict_failures", lambda d: d["trainers"][0].update(failure={"metric": "policy_hash_mismatch_after_apply",
                                                                   "error": "hash differs", "ts": 990.0})),
    ("strict_failures", lambda d: d["trainers"][0].update(lr_zero=True)),
    ("stuck", lambda d: d.update(stuck=[1])),
    ("syncer_idle", lambda d: d.update(last_merge_ts_any=500.0)),
    ("rejected", lambda d: d["shards"][0]["last"].update(rejected=1)),
    ("rollout_stalled", lambda d: d["trainers"][0].update(phase_raw="generate", phase_since=900.0)),
])
def test_each_rule_fires_and_clears(rule, bad):
    fn = getattr(alerts_diloco, rule)
    assert fn(_base(), 1000.0) == []
    d = _base()
    bad(d)
    got = fn(d, 1000.0)
    assert got and all(a["rule"].startswith("diloco_") for a in got)
    assert fn(copy.deepcopy(_base()), 1000.0) == []  # recovered: back to healthy, no alert


def test_bcast_backlog_severity_and_hash_level():
    d = _base()
    d["trainers"][0].update(pending={"1": 1}, pending_n=1, pending_oldest_ts=850.0)
    assert alerts_diloco.bcast_backlog(d, 1000.0)[0]["sev"] == 1
    d["trainers"][0]["pending_oldest_ts"] = 600.0
    assert alerts_diloco.bcast_backlog(d, 1000.0)[0]["sev"] == 0


def test_finished_runs_raise_no_diloco_alert():
    for name in ("run-b", "run-a"):
        r = load(name)
        assert [a for a in r.overview(now=r.max_ts)["alerts"] if a["rule"].startswith("diloco_")] == []


# -- the island table's KL: no placeholder 0 ------------------------------------------------
def test_local_round_loss_kl_does_not_hide_the_train_rollout_kl():
    r = load("run-b")
    rows = r.round_records(r.islands["0"])
    trained = {x["rollout_id"]: x["train_metrics"]["train_rollout_kl"]
               for x in lines(D / "run-b" / "events" / "dlrl-qwen4b-1123-b-l0-modal.jsonl") if x["event"] == "rl_round_trained"}
    for row in rows:
        if row["round"] in trained:
            assert row["kl_tr"] == pytest.approx(trained[row["round"]])


# -- export and the page script ------------------------------------------------------------
def test_export_view_equals_the_page_view():
    r = load("run-b")
    now = r.max_ts
    assert export_view(r, generated_at=now)["page"]["topology"] == r.page_view(now=now)["topology"]


SMOKE = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/)[1];
const app = html.match(/<script>\n([\s\S]*?)\n<\/script>/)[1];
const els = {}, svg = [];
function node(id) {
  return { id, innerHTML: "", textContent: "", hidden: false, style: {}, dataset: {}, children: [], attrs: {},
    listeners: {}, offsetWidth: 200, offsetHeight: 120, clientWidth: Number(process.argv[3] || 1200),
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 560, height: 400 }; },
    querySelectorAll() { return []; } };
}
const made = [];
function mk(id) { if (!els[id]) { els[id] = node(id); if (id === "yeto-data") els[id].textContent = data; } return els[id]; }
global.document = { documentElement: node("html"), getElementById: mk, querySelectorAll: () => [], title: "",
  createElementNS: () => { const n = node(null); svg.push(n); return n; },
  createElement: () => { const n = node(null); made.push(n); return n; } };
global.navigator = {}; global.localStorage = { getItem() { return null; }, setItem() {} };
global.setInterval = () => 0; global.setTimeout = () => 0;
eval(app);
const texts = svg.map((n) => String(n.textContent || "")).filter(Boolean);
const box = (svg.find((n) => /to syncer|to trainers/.test(n.attrs["aria-label"] || "")) || { attrs: {} }).attrs;
console.log(JSON.stringify({ hidden: els.topoBox.hidden, note: els.topoNote.textContent, title: els.topoBox.attrs.title || null,
  texts, label: box["aria-label"] || null, viewBox: box.viewBox || null, kpis: mk("rKpis").innerHTML, nodes: mk("rNodes").innerHTML,
  matrix: made.map((n) => n.innerHTML).filter((h) => /class="fm"/.test(h)).join(""), legend: mk("topoLg").innerHTML }));
"""


def _smoke(tmp_path, reducer, width=1200):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(reducer, tmp_path / "p.html", generated_at=reducer.max_ts)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page), str(width)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_page_draws_the_island_layout(tmp_path):
    out = _smoke(tmp_path, load("run-b"))
    joined = " | ".join(out["texts"])
    assert out["hidden"] is False and out["label"].startswith("islands (trainer + in-island rollout)")
    assert out["viewBox"].startswith("0 0 1480 ") and "layout: island" in out["note"]
    for want in ("Island 0", "Island 1", "Rollout · 1 engine", "round 28", "step 29", "outer v6", "Syncer", "step 24",
                 "serving yeto:29:e5fd32df", "frag [21 22 23 24]", "n0 · l0-modal", "H100 ×2", "T*", "R*"):
        assert want in joined, want
    assert 'class="fm"' in out["matrix"] and ">f3<" in out["matrix"].replace("f3<span", "f3<") and "v24" in out["matrix"]
    assert "Fragment delta PUSH" in out["legend"]
    for want in ("Syncer step / local steps", "Groups trained / rolled out", "in-island rollout engines 2 (fixed-partition)",
                 "Fragment rounds in flight / BCAST pending"):
        assert want in out["kpis"], want
    assert "Syncer RSS" not in out["kpis"] and "samplers 0/0" not in out["kpis"]
    assert "Local step / outer v" in out["nodes"] and "Last PUSH / BCAST applied" in out["nodes"] and "HBM peak" not in out["nodes"]


def test_page_draws_colocated_islands_narrow(tmp_path):
    out = _smoke(tmp_path, load("run-a"), width=420)
    joined = " | ".join(out["texts"])
    assert out["viewBox"].startswith("0 0 400 ") and "T+R" in joined and "#2" in joined
    assert "shared GPU" in out["kpis"]


def test_page_draws_the_synced_group(tmp_path):
    out = _smoke(tmp_path, load("k31-synced-group"))
    joined = " | ".join(out["texts"])
    assert out["label"].startswith("samplers to store to trainers to syncer") and "layout: synced group" in out["note"]
    for want in ("Trainer island 0", "Store", "samplers load trainer 0's weights", "syncer tape behind the islands: no negative lag drawn"):
        assert want in joined, want
    assert "Buffer backlog" in out["kpis"] and "Fragment rounds in flight" in out["kpis"]


def test_page_draws_two_shard_cards(tmp_path):
    merges = lines(D / "run-b" / "yeto-tape.jsonl")
    r = load("run-b", skip=lambda p: p.name == "yeto-tape.jsonl")
    for off, m in enumerate(merges):
        src = "yeto-tape.jsonl" if m["step"] % 2 else "yeto-tape.shard1.jsonl"
        r.feed(m, source=str(tmp_path / src), offset=off)
    joined = " | ".join(_smoke(tmp_path, r)["texts"])
    assert "Syncer shard 0 · fragments 0–1" in joined and "Syncer shard 1 · fragments 2–3" in joined


def test_async_grpo_page_keeps_v1(tmp_path):
    out = _smoke(tmp_path, load("async-e2e"))
    assert out["label"] == "samplers to run store to trainers" and "layout: async-grpo" in (out["title"] or "")
    assert "Buffer backlog" in out["kpis"] and "Syncer step" not in out["kpis"] and "Round / policy" in out["nodes"]
    assert "Fragment delta PUSH" not in out["legend"]


# -- decoupled DiLoCo SFT (topology_sft.py) ------------------------------------------------
SFT = "sft-nemotron"  # 10 ms-swift learner islands, P 64, H 32, trimmed to the last 60 steps / 120 merges


def test_sft_layout_and_its_reason():
    r = load(SFT)
    lay = r.page_view(now=r.max_ts)["layout"]
    assert lay["kind"] == "sft" and "SFT islands 10" in lay["reason"] and "120 syncer merge record(s)" in lay["reason"]
    for name in ("run-a", "run-b", "k31-synced-group", "async-e2e"):  # the RL syncer tape is never an SFT record
        assert load(name).sft.sft_records == 0


def test_sft_without_a_syncer_draws_nothing():
    r = load(SFT, skip=lambda p: "syncer" in p.parts)
    view = r.page_view(now=r.max_ts)
    assert view["layout"]["kind"] == "none" and "topology" not in view and view["sft"]["islands"]


def test_sft_syncer_learners_and_fragments_match_the_tapes():
    r = load(SFT)
    tape = lines(D / SFT / "syncer" / "tape.jsonl")
    merges = [m for m in tape if "fragment" in m and "kind" not in m]
    d = diloco(r)
    assert (d["kind"], d["P"], d["M"], d["N"], d["H"], d["total_steps"]) == ("sft", 64, 10, 1, 32, 1510)
    assert d["versions"] == [max(m["step"] for m in merges if m["fragment"] == f) for f in range(64)]
    assert d["shards"][0]["step"] == 210 and d["roster"] == 10 and not d["done"]
    by = {t["id"]: t for t in d["trainers"]}
    one = by["1"]
    last = merges[-1]
    resp = next(x for x in last["responders"] if x["id"] == 1)
    assert one["contrib"][last["fragment"]] == {"step": 210, "frag": 17, "base": resp["base_version"],
                                                "staleness": resp["staleness"], "tokens": resp["c_tokens"],
                                                "recv_s": resp["recv_seconds"], "ts": resp["arrived_unix"]}
    assert [n["machine"] for n in by["0"]["nodes"]] == ["m01", "m02", "m03"] and by["0"]["nodes"][0]["gpus"] == 8
    assert by["0"]["learner"]["layout"] == "TP4/EP8/PP3" and by["0"]["steps_total"] == 1092 and by["0"]["step"] == 189
    # merged after the island's last applying step (yeto_steps applies > 0) = not applied yet
    steps = lines(D / SFT / "island-0" / "yeto_steps.jsonl")
    applied = max(x["time"] for x in steps if x["applies"])
    want = sum(1 for m in merges if m["time_unix"] > applied)
    assert by["0"]["pending_n"] == want == 2
    # the deltas accepted after the last merge make the next step's round
    acc = sorted({str(x["island_id"]) for x in tape if x.get("kind") == "delta_accepted" and x["time_unix"] > last["time_unix"]})
    assert [(x["step"], x["frag"], x["pushed"]) for x in d["rounds"]] == [(211, 18, acc)] and acc == ["8", "9"]


def test_sft_raises_no_diloco_alert():
    r = load(SFT)
    assert [a for a in r.overview(now=r.max_ts)["alerts"] if a["rule"].startswith("diloco_")] == []


def test_page_draws_the_sft_islands_on_the_sft_page(tmp_path):
    out = _smoke(tmp_path, load(SFT))
    joined = " | ".join(out["texts"])
    assert out["hidden"] is False and out["label"] == "learner islands to syncer" and "layout: sft" in out["note"]
    assert out["viewBox"].startswith("0 0 1480 ")
    for want in ("Island 0", "Learner 0 · fast island", "Learner 9 · slow island 9", "n0 · m01", "card ×8", "step 210",
                 "of 1510 · 120 merges", "H 32", "loss 0.6396"):
        assert want in joined, want
    assert "t210" in out["matrix"] and "base v146" in out["matrix"] and "Island 9" in out["matrix"]
    assert "learner → syncer" in out["legend"]
    assert out["kpis"] == ""  # the SFT page keeps its own cards (sKpis), the RL cards stay empty
