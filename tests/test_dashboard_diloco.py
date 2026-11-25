"""Live topology of a run with a syncer (yeto/dashboard/topology_diloco.py, alerts_diloco.py,
static/topology_diloco.js), checked against trimmed real tapes:

* ``async-e2e`` / ``async-compaction``: async GRPO (one trainer, two cross-island samplers, no
  syncer): the V1 drawing, its view identical to the one main computed (``*.v1-golden.json``);
  their trainer tapes keep ``rl_heartbeat`` / ``trainer_start_phase`` and the earlier launches
  (``events/previous/``), so a live replay sees the trainer as alive as the run did;
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


def tapes(name: str) -> list[str]:
    # ``events/`` as ``run_sources`` passes it: the earlier launches under ``events/previous/<stamp>/``
    return [str(D / name), str(D / name / "events")]


def load(name: str, *, skip=lambda path: False) -> Reducer:
    r = Reducer(run=name)
    paths = [p for p in sources.discover(tapes(name)) if not skip(p)]
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


LIVENESS_RULES = ("heartbeat", "startup", "startup_step")


def _record_ts(rec: dict) -> float | None:
    v = rec.get("time_unix", rec.get("ts"))
    return float(v) if isinstance(v, (int, float)) else None


def live_alerts(paths: list[str], *, extra: tuple = (), after_s: float = 3600, step: float = 30) -> list[tuple]:
    """Replay tapes live (records in time order, the page read every ``step`` s from the first
    record to ``after_s`` after the last); the liveness alerts seen, as (s from the end, title)."""
    recs = []
    for src in (sources.make_source(p) for p in sources.discover(paths)):
        last = 0.0
        for off, rec in src.read_new():
            last = (_record_ts(rec) if isinstance(rec, dict) else None) or last
            recs.append((last, src.name, off, rec, src))
    for off, rec in enumerate(extra):
        recs.append((rec["time_unix"], "extra/yeto-tape.jsonl", off, rec, None))
    recs.sort(key=lambda x: x[0])
    first, end = min(t for t, *_ in recs if t), recs[-1][0]
    r, i, seen, t = Reducer(run="replay"), 0, [], first
    while t <= end + after_s:
        while i < len(recs) and recs[i][0] <= t:
            _, name, off, rec, src = recs[i]
            r.feed(rec, source=name, offset=off, **({} if src is None else {
                "island": src.island, "node": src.node, "island_fixed": src.island_fixed}))
            i += 1
        seen += [(round(t - end), a["title"]) for a in r.overview(now=t)["alerts"] if a["rule"] in LIVENESS_RULES]
        t += step
    assert i == len(recs)
    return seen


@pytest.mark.parametrize("name", ["async-e2e", "async-compaction"])
def test_async_grpo_live_replay_raises_no_liveness_alert(name):
    """Replayed live (records in time order, the page read every 30 s from the first record to an
    hour after the last), a healthy async GRPO run never reads as a stalled or slow-starting
    island: not mid-run, not while the trainer finishes, releases and leaves, not after."""
    assert live_alerts(tapes(name)) == []


def test_a_diloco_trainer_without_heartbeats_is_not_stale_in_its_first_round():
    """Full-parameter DiLoCo islands of older runs write no rl_heartbeat: the first round after
    rl_driver_start (warm-up, the first generation: minutes) is not a heartbeat timeout. Only the
    terminal round, which waits for the slowest island, still reads stale on such a tape..."""
    seen = live_alerts(tapes("g5-fullparam"), after_s=600, step=10)
    assert seen and all(t >= -150 for t, _ in seen), seen
    # ... unless the syncer's config says the run is in its terminal phase (last_step 12, P 4):
    # its terminal grace (120 s) covers the island waiting for the others
    tape = lines(D / "g5-fullparam" / "yeto-tape.jsonl")
    config = {"kind": "syncer_config", "fragments": 4, "owned_fragments": [0, 4], "shard_index": 0,
              "shard_count": 1, "learners": 2, "quorum": 1, "pipeline": 4, "total_steps": 12, "last_step": 12,
              "terminal_grace_ms": 120000, "time_unix": tape[0]["time_unix"] - 1}
    assert live_alerts(tapes("g5-fullparam"), extra=(config,), after_s=600, step=10) == []


def test_a_stray_syncer_tape_does_not_turn_async_into_diloco(tmp_path):
    """The trainers declared no outer sync: a syncer tape next to them changes nothing."""
    r = load("async-e2e")
    for rec in lines(D / "run-a" / "yeto-tape.jsonl")[:3]:
        r.feed(rec, source=str(tmp_path / "yeto-tape.jsonl"))
    assert r.page_view(now=r.max_ts)["layout"]["kind"] == "async"


def test_the_syncers_own_records_add_config_and_links_only():
    """The syncer's config, round open / timeout and connection records (``kind``, no ``step``)
    are not merges, and a ``learner_id`` there is no sign of life of that island: run B's page
    with them on its syncer tape equals the page without them but for the config and the
    trainers' syncer links."""
    before = load("run-b")
    tape = lines(D / "run-b" / "yeto-tape.jsonl")
    first, last = tape[0]["time_unix"], tape[-1]["time_unix"]
    extra = [
        {"kind": "member_connected", "learner_id": 0, "generation": 1, "phase": "init", "time_unix": first - 1},
        {"kind": "member_connected", "learner_id": 1, "generation": 1, "phase": "init", "time_unix": first - 1},
        {"kind": "syncer_config", "fragments": 4, "owned_fragments": [0, 4], "shard_index": 0, "shard_count": 1,
         "learners": 2, "quorum": 1, "pipeline": 2, "total_steps": 24, "time_unix": first - 0.5},
        {"kind": "round_open", "global_step": 1, "fragment": 0, "attempt": 1, "base_version": 0,
         "members": [{"id": 0, "generation": 1}, {"id": 1, "generation": 1}], "quorum": 1, "time_unix": first - 0.2},
        {"kind": "round_attempt_timeout", "global_step": 2, "fragment": 1, "attempt": 1, "reason": "quorum_timeout",
         "next": "new_attempt", "time_unix": first + 1},
        {"kind": "member_disconnected", "learner_id": 1, "generation": 1, "phase": "final", "time_unix": last},
    ]
    after = load("run-b")
    for off, rec in enumerate(extra):
        after.feed(rec, source="extra/yeto-tape.jsonl", offset=off)
    assert after.island_ids() == before.island_ids()
    now = before.max_ts
    a, b = (json.loads(json.dumps(r.page_view(now=now), default=str)) for r in (before, after))
    for key in ("layout", "islands"):
        assert a.get(key) == b.get(key), key
    assert a["overview"]["alerts"] == b["overview"]["alerts"]
    da, db = a["topology"]["diloco"], b["topology"]["diloco"]
    # what they add: the syncer's config and each trainer's connection as the syncer saw it
    assert da["config"] is None and db["config"]["P"] == 4 and db["config"]["tau"] == 2 and db["config"]["T"] == 24
    assert [t["syncer_link"] for t in da["trainers"]] == [None, None]
    assert [(t["syncer_link"] or {}).get("state") for t in db["trainers"]] == ["connected", "disconnected"]
    for d in (da, db):
        d.pop("config")
        for t in d["trainers"]:
            t.pop("syncer_link")
    assert da == db  # every other part of the panel is the P0 one (step 1 and 2 merged already)
    alone = Reducer(run="syncer-only")  # an SFT syncer tape: no RL island appears
    for off, rec in enumerate(extra):
        alone.feed(rec, source="yeto-tape.jsonl", offset=off)
    assert alone.island_ids() == []


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
    for want in ("Trainer island 0", "Store", "samplers load trainer 0's weights", "syncer tape behind the islands",
                 "  no negative lag drawn"):  # one syncer line folded at its " · " to stay inside the card
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


# -- full-parameter islands (g5-fullparam) and samplers following their trainer (f1-follow) ---------
def test_full_parameter_island_rollout_comes_from_the_trainer_side():
    """A full-parameter island writes no rl_local_round: its rollout round and time come from
    rl_round_wall, its groups per round from rl_round_trained; the run total is not reported."""
    r = load("g5-fullparam")
    d = diloco(r)
    assert d["kind"] == "island" and d["done"] and d["versions"] == [9, 10, 11, 12]
    t0, t1 = d["trainers"]
    for t, rnd in ((t0, 71), (t1, 46)):
        isl = t["island"]
        assert isl["groups_src"] == "trained" and isl["groups_total"] is None and isl["round"] == rnd
        assert isl["groups"] == 8 and isl["rollout_p50_s"] > 0 and isl["served"]["v"] is not None
    assert (t0["full_push"]["total"], t1["full_push"]["total"], t1["full_push"]["overwritten"]) == (12, 2, 3)
    assert t0["trained_groups"] + t1["trained_groups"] == 952


def test_a_terminal_full_parameter_round_stops_the_not_advancing_rules():
    r = load("g5-fullparam")
    first_terminal = 1795451975.2608376  # island 0's terminal rl_full_parameter_round
    d = diloco(r, now=first_terminal + 60)
    assert not d["running"] and d["stuck"] == []


def test_page_full_parameter_island_labels_its_gaps(tmp_path):
    out = _smoke(tmp_path, load("g5-fullparam"))
    joined = " | ".join(out["texts"])
    assert "round 71" in joined and "8 groups / round (trainer side)" in joined and "no round yet" not in joined
    assert "run ended · last 7 merge(s) → final cut" in joined and "BCAST pending" not in joined
    assert "2 in run · 3 overwritten" in joined
    assert "Groups trained / rolled out" in out["kpis"] and "952 / —" in out["kpis"]
    assert "not recorded" in out["kpis"] and "not trained yet" not in out["kpis"]
    assert "lag1" not in out["matrix"] and "lag2" not in out["matrix"] and "since anchor" not in out["matrix"]
    # the final-cut ack: an island's anchor below the syncer's committed version is its last applied one
    assert "final v12</div>" in out["matrix"]
    assert "the final version is on the syncer" in out["matrix"]


FP_L0 = D / "g5-fullparam" / "events" / "g5-v1-a-l0-modal.jsonl"


def _with_trajectory_rewards(rewards: list[tuple[float, bool]]) -> tuple[Reducer, int]:
    """g5-fullparam plus rl_trajectory_reward records of island 0's last trained round."""
    r = load("g5-fullparam")
    rid = max(x["rollout_id"] for x in lines(FP_L0) if x["event"] == "rl_round_trained")
    t = r.max_ts
    for i, (reward, aborted) in enumerate(rewards):
        r.feed({"event": "rl_trajectory_reward", "island_id": 0, "rollout_id": rid, "sample_index": i,
                "reward": reward, "aborted": aborted, "time_unix": t - 1}, source=str(FP_L0), offset=10**9 + i)
    return r, rid


def test_full_parameter_reward_reads_the_round_trajectory_mean(tmp_path):
    """A full-parameter island's rl_round_trained has no reward_mean; its own rl_trajectory_reward
    records of the round give the mean (an aborted trajectory is not counted)."""
    r, rid = _with_trajectory_rewards([(1.0, False), (0.0, False), (0.0, False), (1.0, False), (1.0, True)])
    view = r.page_view(now=r.max_ts)
    row = next(x for x in view["islands"]["0"]["rounds"] if x["round"] == rid)
    assert row["reward"] == pytest.approx(0.5) and row["reward_src"] == "trajectories"
    card = next(c for c in view["overview"]["islands"] if c["id"] == "0")
    assert card["reward"] == pytest.approx(0.5)
    assert next(c for c in view["overview"]["islands"] if c["id"] == "1")["reward"] is None
    out = _smoke(tmp_path, r)
    assert "not recorded" not in out["kpis"] and "mean of its trajectory rewards" in out["kpis"]


def test_a_reported_reward_mean_wins_over_the_trajectory_mean():
    r, rid = _with_trajectory_rewards([(1.0, False), (0.0, False)])
    r.page_view(now=r.max_ts)  # the trajectory mean filled in first
    r.feed({"event": "rl_round_trained", "island_id": 0, "rollout_id": rid, "reward_mean": 0.9,
            "time_unix": r.max_ts}, source=str(FP_L0), offset=10**9 + 99)
    row = next(x for x in r.page_view(now=r.max_ts)["islands"]["0"]["rounds"] if x["round"] == rid)
    assert row["reward"] == pytest.approx(0.9) and row["reward_src"] is None


def _without_kl(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    for k in ("train_rollout_kl", "ppo_kl"):
        (rec.get("train_metrics") or {}).pop(k, None)
    return rec


@pytest.mark.parametrize("marker", [
    {"event": "rl_verl_island_start", "island_id": 0},
    {"event": "rl_engine_selected", "island_id": 0, "rl_engine": "ports", "rl_backend": "verl"},
])
def test_a_verl_island_kl_reads_not_reported_by_verl(tmp_path, marker):
    r = load("g5-fullparam", skip=lambda p: p.name == FP_L0.name)
    recs = lines(FP_L0)
    r.feed(dict(marker, time_unix=recs[0]["time_unix"]), source=str(FP_L0), offset=0)
    for off, rec in enumerate(recs, start=1):
        r.feed(_without_kl(rec), source=str(FP_L0), offset=off)
    cards = {c["id"]: c for c in r.page_view(now=r.max_ts)["overview"]["islands"]}
    assert cards["0"]["backend"] == "verl" and cards["1"]["backend"] == "miles"  # its miles_commit
    out = _smoke(tmp_path, r)
    assert out["nodes"].count("not reported by verl") == 1
    # each island's trainer card names its backend: verl, and Megatron for the Miles island
    assert "verl · H100" in out["texts"] and "Megatron · H100" in out["texts"]


def _fullparam_fragments(tmp_path, *, wire=None, overwrite=False) -> Reducer:
    """Two full-parameter islands (strict sweeps through the real syncer; the per-fragment
    rl_fragment_push / rl_fragment_bcast records of FullParameterSync). ``wire`` / ``overwrite``
    mark island 0's last PUSH as encoded ahead and island 1's last BCAST as a partial-quorum
    overwrite, as a background push and an async round write them."""
    src = D / "fullparam-fragments"
    dst = tmp_path / "fullparam-fragments"
    shutil.copytree(src, dst)
    for i, key in ((0, "rl_fragment_push"), (1, "rl_fragment_bcast")):
        path = dst / "events" / f"fullparam-fragments-l{i}-modal.jsonl"
        recs = lines(path)
        last = max(j for j, x in enumerate(recs) if x["event"] == key)
        if key == "rl_fragment_push" and wire is not None:
            recs[last]["wire_bytes"] = wire
        if key == "rl_fragment_bcast" and overwrite:
            recs[last]["overwrite"] = True
        path.write_text("".join(json.dumps(x) + "\n" for x in recs))
    r = Reducer(run="fullparam-fragments")
    load_all(r, [str(p) for p in sources.discover([str(dst), str(dst / "events")])])
    return r


def test_full_parameter_islands_draw_their_per_fragment_push_and_bcast(tmp_path):
    r = _fullparam_fragments(tmp_path)
    d = diloco(r)
    assert d["kind"] == "island" and d["P"] == 6 and d["M"] == 2
    for t in d["trainers"]:
        recs = lines(D / "fullparam-fragments" / "events" / f"fullparam-fragments-l{t['id']}-modal.jsonl")
        push = [x for x in recs if x["event"] == "rl_fragment_push"][-1]
        bcast = [x for x in recs if x["event"] == "rl_fragment_bcast"][-1]
        lp, lb = t["last_push"], t["last_bcast"]
        assert (lp["frag"], lp["step"], lp["bytes"], lp["wire"]) == (
            push["fragment_id"], push["global_step"], push["payload_bytes"], False)
        assert (lb["frag"], lb["v"], lb["bytes"], lb["overwrite"], lb["catch_up"]) == (
            bcast["fragment_id"], bcast["version"], bcast["payload_bytes"], False, False)
        assert lb["received_ts"] == bcast["received_unix"] and t["full_push"] is not None
    flows = d["flows"]
    bcasts = [f for f in flows if f["k"] == "b"]
    # every applied broadcast has its own record (exact window); the last sweep went out as FINAL
    assert sum(1 for f in bcasts if not f["a"]) == 36 and all(f["b"] for f in bcasts if not f["a"])
    assert {f["b"] for f in flows if f["k"] == "p"} <= {32.0, 48.0, 64.0}


def test_a_full_parameter_push_encoded_ahead_counts_its_wire_bytes_and_an_overwrite_shows(tmp_path):
    plain = diloco(_fullparam_fragments(tmp_path / "plain"))
    r = _fullparam_fragments(tmp_path, wire=13, overwrite=True)
    d = diloco(r)
    t0, t1 = d["trainers"]
    assert (t0["last_push"]["bytes"], t0["last_push"]["wire"]) == (13.0, True)
    assert t0["last_push"]["payload_bytes"] == plain["trainers"][0]["last_push"]["bytes"]
    assert (t1["last_bcast"]["overwrite"], t1["bcast_overwrites"], t0["bcast_overwrites"]) == (True, 1, 0)
    step = t0["last_push"]["step"]
    (pf,) = [f for f in d["flows"] if f["k"] == "p" and f["i"] == "0" and f["f"] == t0["last_push"]["frag"]
             and f["t1"] == max(g["t1"] for g in d["flows"] if g["k"] == "p" and g["i"] == "0")]
    assert pf["b"] == 13.0, (step, pf)
    joined = " | ".join(_smoke(tmp_path, r)["texts"])
    assert f"t{step} base v" in joined and "0 MB wire" in joined and " · overwrite" in joined


def test_synced_group_trainers_following_samplers_are_buffer_trainers():
    r = load("f1-follow")
    d = diloco(r)
    assert d["kind"] == "synced_group" and d["follow"] and d["publisher"] is None
    assert {k: v["samplers"] for k, v in d["lanes"].items()} == {"0": ["2"], "1": ["3"]}
    for t in d["trainers"]:  # placement reads "colocated", yet the rollout runs on the samplers
        assert t["island"] is None and [n["roles"] for n in t["nodes"]] == [["T"]]


def test_page_draws_the_follow_lanes(tmp_path):
    out = _smoke(tmp_path, load("f1-follow"))
    joined = " | ".join(out["texts"])
    assert "trainer0/" in joined and "trainer1/" in joined and "sampler(s) on their own store lane trainer0/" in joined
    assert "same GPU" not in joined and "T+R" not in joined and "time-share" not in joined
    assert "not on its tape" in out["nodes"]


def _replayed(name: str, until: float, extra: tuple = ()) -> Reducer:
    """The reducer of ``name`` as a live page saw it at ``until`` (records up to then, plus extra)."""
    recs = []
    for src in (sources.make_source(p) for p in sources.discover(tapes(name))):
        last = 0.0
        for off, rec in src.read_new():
            last = (_record_ts(rec) if isinstance(rec, dict) else None) or last
            if last <= until:
                recs.append((last, src, off, rec))
    r = Reducer(run=name)
    for _, src, off, rec in sorted(recs, key=lambda x: x[0]):
        r.feed(rec, source=src.name, offset=off, island=src.island, node=src.node, island_fixed=src.island_fixed)
    for off, rec in enumerate(extra):
        r.feed(rec, source="yeto-tape.jsonl#p1", offset=off)
    return r


def _run_b_with_p1_records() -> tuple[Reducer, float, int]:
    """Run B as a live page saw it mid-run, with the syncer's P1 records for a round ahead."""
    merges = [m for m in lines(D / "run-b" / "yeto-tape.jsonl") if "step" in m]
    mid = merges[9]["time_unix"]
    ahead = merges[9]["step"] + 3  # no island has pushed it yet
    extra = (
        {"kind": "syncer_config", "fragments": 4, "shard_index": 0, "shard_count": 1, "learners": 2, "quorum": 1,
         "pipeline": 2, "total_steps": 24, "last_step": 24,
         "outer_lr": 0.699999988079071,  # the float32 0.7 as a real syncer tape writes it
         "time_unix": merges[0]["time_unix"] - 9},
        {"kind": "round_open", "global_step": ahead, "fragment": (ahead - 1) % 4, "attempt": 1, "base_version": 0,
         "members": [{"id": 0, "generation": 1}, {"id": 1, "generation": 1}], "quorum": 1, "time_unix": mid - 40},
        {"kind": "round_attempt_timeout", "global_step": ahead, "fragment": (ahead - 1) % 4, "attempt": 1,
         "reason": "quorum_timeout", "next": "new_attempt", "time_unix": mid - 20},
        {"kind": "round_open", "global_step": ahead, "fragment": (ahead - 1) % 4, "attempt": 2, "base_version": 0,
         "members": [{"id": 0, "generation": 1}], "quorum": 1, "time_unix": mid - 20},
        {"kind": "member_disconnected", "learner_id": 1, "generation": 1, "phase": "training", "time_unix": mid - 5},
    )
    return _replayed("run-b", mid, extra), mid, ahead


def test_round_open_shows_a_pull_without_push_and_its_timeouts():
    """P1 syncer records: a round the syncer opened (PULL sent) shows before any island pushed
    (state "pull"), with its attempt and quorum timeouts; the config fills T / tau / lr; a
    disconnected island reads so on its card at once."""
    reducer, mid, ahead = _run_b_with_p1_records()
    d = diloco(reducer, now=mid)
    r = next(x for x in d["rounds"] if x["step"] == ahead)
    assert (r["state"], r["attempt"], r["timeouts"], r["pushed"], r["pulled"]) == ("pull", 2, 1, [], ["0"])
    assert r["since"] == mid - 20 and r["age_s"] == 20  # the current attempt's PULL
    assert (d["config"]["T"], d["config"]["tau"]) == (24, 2) and d["config"]["lr"] == pytest.approx(0.7)
    links = {t["id"]: t["syncer_link"] for t in d["trainers"]}
    assert links["0"] is None and links["1"]["state"] == "disconnected" and links["1"]["since"] == mid - 5


def test_a_round_relaunched_because_its_islands_left_is_not_a_timeout():
    """A ``members_left`` attempt record (every island of the round left, the syncer relaunched
    it at once) is shown as such and never counted as a quorum timeout."""
    reducer, mid, ahead = _run_b_with_p1_records()
    left = {"kind": "round_attempt_timeout", "global_step": ahead, "fragment": (ahead - 1) % 4, "attempt": 2,
            "reason": "members_left", "next": "new_attempt", "time_unix": mid - 10}
    reducer.feed(left, source="yeto-tape.jsonl#p1", offset=99)
    r = next(x for x in diloco(reducer, now=mid)["rounds"] if x["step"] == ahead)
    assert (r["timeouts"], r["members_left"]) == (1, True)  # only the attempt-1 quorum timeout counts


def test_bcast_flows_use_the_syncer_and_island_stamps(tmp_path):
    """Merge bcast_unix starts a BCAST window and the island's received_unix ends it (older tapes:
    the merge time and the apply record minus its queue time)."""
    out = tmp_path / "run-b"
    shutil.copytree(D / "run-b", out)
    rows = lines(out / "yeto-tape.jsonl")
    for m in rows:
        if "step" in m:
            m["bcast_unix"], m["bcast_bytes"] = m["time_unix"] - 0.25, 4096
    (out / "yeto-tape.jsonl").write_text("".join(json.dumps(m) + "\n" for m in rows))
    for f in (out / "events").glob("*.jsonl"):
        recs = lines(f)
        for e in recs:
            if e.get("event") == "rl_fragment_bcast":
                e["received_unix"] = e["time_unix"] - 0.5
                e["applied_unix"] = e["time_unix"]
        f.write_text("".join(json.dumps(e) + "\n" for e in recs))
    r = Reducer(run="stamps")
    load_all(r, [str(p) for p in sources.discover([str(out), str(out / "events")])])
    d = diloco(r)
    stamped = {m["step"]: m["bcast_unix"] for m in rows if "step" in m}
    bflows = [f for f in d["flows"] if f["k"] == "b"]
    assert bflows and all(any(abs(f["t0"] - t) < 1e-6 for t in stamped.values()) for f in bflows)
    applied = [f for f in bflows if not f["a"]]
    assert applied and all(f["t1"] >= f["t0"] + 0.3 for f in applied)
    assert all(f["b"] == 4096 for f in bflows if f["a"])
    lb = next(t["last_bcast"] for t in d["trainers"] if t["last_bcast"])
    assert lb["applied_ts"] - lb["received_ts"] == pytest.approx(0.5)


def test_page_draws_the_pull_the_config_and_the_lost_link(tmp_path):
    reducer, mid, ahead = _run_b_with_p1_records()
    joined = " | ".join(_smoke(tmp_path, reducer)["texts"])
    assert f"t{ahead} f{(ahead - 1) % 4}: PULL out · 20s · attempt 2 · 1 timeout" in joined
    assert "τ 2 · T 24 · lr 0.7" in joined and "link ✕" in joined
