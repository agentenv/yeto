"""Live topology panel (yeto/dashboard/topology.py, static/topology.js): samplers -> run
store -> trainers, built only from tape events the reducer already folded."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from dashboard_helpers import T0

from yeto.dashboard import topology
from yeto.dashboard.export import build_id, export_html, render_page
from yeto.dashboard.reducer import ALIVE_STATUSES, Reducer

FULL = 8_060_901_581


def sampler_start(name: str, t: float, card: str, precision: str, engine: str, tp: int = 8) -> dict:
    return {"event": "rl_sampler_start", "sampler_id": name, "max_backlog_groups": 16, "time_unix": t,
            "provenance": {"card_model": card, "device_family": "cuda", "engine": engine, "precision": precision,
                           "quant_scheme": precision, "topology": f"tp{tp}", "sampler_id": name}}


def group(name: str, t: float, version: int, trials: int = 8) -> dict:
    return {"event": "rl_agentic_group", "sampler_id": name, "written": True, "trials": trials, "kept": trials,
            "behavior_version": version, "version_lag": 0, "time_unix": t}


def sampler_round(name: str, t: float, rid: int, version: int, groups: int = 1) -> dict:
    return {"event": "rl_sampler_round", "sampler_id": name, "rollout_id": rid, "policy_version": version,
            "version_lag": 0, "groups": groups, "buffer_end": rid + 1, "generate_s": 40.0, "time_unix": t}


def trainer_step(t: float, rid: int, *, groups: int = 8) -> list[dict]:
    return [{"event": "rl_driver_phase", "island_id": 0, "phase": "generate", "rollout_id": rid, "time_unix": t},
            {"event": "rl_group_policies", "island_id": 0, "rollout_id": rid,
             "groups": [[f"g{i}", "yeto:0:x", [rid]] for i in range(groups)], "time_unix": t + 100},
            {"event": "rl_driver_phase", "island_id": 0, "phase": "train", "rollout_id": rid, "time_unix": t + 100},
            {"event": "rl_round_trained", "island_id": 0, "rollout_id": rid, "train_step": rid + 1,
             "trained_groups": groups, "trained_samples": groups * 8, "reward_mean": 0.4, "time_unix": t + 120},
            {"event": "rl_round_phases", "island_id": 0, "rollout_id": rid, "time_unix": t + 125,
             "phase_seconds": {"generate": 100, "train": 20, "sync": 2, "publish": 3}, "wall_seconds": 125}]


def release(t: float, version: int, *, delta: bool, upload_s: float, selected: int) -> dict:
    return {"event": "rl_policy_released", "island_id": 0, "policy_version": version, "released_at": t - 1,
            "release_bytes": 500_000_000 if delta else FULL, "delta_ratio": 0.06 if delta else None,
            "buffer_selected": selected, "release_timings": {"upload_s": upload_s}, "weights": "full",
            "time_unix": t}


def loaded(name: str, t: float, version: int, released: float) -> dict:
    return {"event": "rl_sampler_policy_loaded", "sampler_id": name, "policy_version": version,
            "released_at": released, "seen_at": released + 3, "reload_mode": "in_place",
            "follower_timings": {"resolve_s": 12.0, "load_s": 31.0, "verify_s": 0.0}, "time_unix": t}


def drain(name: str, t: float, version: int) -> dict:
    return {"event": "rl_agentic_drain", "sampler_id": name, "policy_version": version, "seq": version + 1,
            "waited_s": 25.0, "in_flight": 16, "timed_out": False, "time_unix": t}


def run(extra=()) -> Reducer:
    """Two samplers (B300 vLLM, MI355X SGLang) and one trainer: 2 steps, a full then a delta
    release; mi355x loads both, b300 drained for the second one but has not loaded it."""
    recs = [sampler_start("b300-0", T0, "NVIDIA B300", "nvfp4", "vllm"),
            sampler_start("mi355x-0", T0, "AMD Instinct MI355X", "mxfp4", "sglang"),
            {"event": "rl_driver_start", "island_id": 0, "time_unix": T0}]
    for k in range(12):
        recs.append(group("b300-0", T0 + 20 + 25 * k, 0 if k < 6 else 1))
        recs.append(group("mi355x-0", T0 + 30 + 40 * k, 0 if k < 4 else 1, trials=7))
        recs.append(sampler_round("b300-0", T0 + 20 + 25 * k, k, 0 if k < 6 else 1))
        recs.append(sampler_round("mi355x-0", T0 + 30 + 40 * k, k, 0 if k < 4 else 1))
    recs += trainer_step(T0 + 10, 0) + trainer_step(T0 + 200, 1)
    recs.append(release(T0 + 150, 1, delta=False, upload_s=45.0, selected=8))
    recs.append(release(T0 + 345, 2, delta=True, upload_s=15.0, selected=16))
    recs += [drain("b300-0", T0 + 180, 0), loaded("b300-0", T0 + 215, 1, T0 + 149),
             drain("mi355x-0", T0 + 185, 0), loaded("mi355x-0", T0 + 220, 1, T0 + 149),
             drain("mi355x-0", T0 + 370, 1), loaded("mi355x-0", T0 + 400, 2, T0 + 344),
             drain("b300-0", T0 + 390, 1)]
    recs += list(extra)
    r = Reducer(run="topo")
    for rec in sorted(recs, key=lambda x: x["time_unix"]):
        r.feed(rec)
    return r


NOW = T0 + 420


def _view(r: Reducer, now: float = NOW) -> dict:
    return r.page_view(now=now)["topology"]


def test_nodes_from_provenance_and_trainer_card():
    t = _view(run())
    by = {s["id"]: s for s in t["samplers"]}
    assert set(by) == {"b300-0", "mi355x-0"}
    b, m = by["b300-0"], by["mi355x-0"]
    assert (b["card"], b["precision"], b["engine"], b["arch"], b["gpus"]) == ("NVIDIA B300", "NVFP4", "vLLM", "Blackwell", 8)
    assert (m["precision"], m["engine"], m["arch"]) == ("MXFP4", "SGLang", "CDNA4")
    assert [x["id"] for x in t["trainers"]] == ["0"]


def test_versions_lag_and_reload_states():
    """LATEST is v2: mi355x loaded it; b300 drained (load in progress) but has not loaded it."""
    t = _view(run())
    by = {s["id"]: s for s in t["samplers"]}
    assert t["latest"] == 2
    assert by["mi355x-0"]["policy_version"] == 2 and by["mi355x-0"]["lag"] == 0 and by["mi355x-0"]["state"] == "sampling"
    b = by["b300-0"]
    assert b["policy_version"] == 1 and b["lag"] == 1
    assert b["state"] == "loading" and b["reload"] == {"target": 2, "since": T0 + 390}


def test_reloading_before_the_drain():
    """Before its drain a sampler behind LATEST is fetching / draining: "reloading" since the release."""
    t = _view(run(), now=T0 + 360)
    b = {s["id"]: s for s in t["samplers"]}["b300-0"]
    assert b["state"] == "reloading" and b["reload"]["since"] == T0 + 345


def test_rates_buffer_and_last_release():
    t = _view(run())
    by = {s["id"]: s for s in t["samplers"]}
    # groups written in the last 5 minutes (NOW - 300, NOW]: b300 k=4..11 (T0+120..T0+295)... counted exactly
    exp_b = sum(1 for k in range(12) if NOW - 300 < T0 + 20 + 25 * k <= NOW)
    exp_m = sum(1 for k in range(12) if NOW - 300 < T0 + 30 + 40 * k <= NOW)
    assert by["b300-0"]["rate_gpm"] == pytest.approx(exp_b / 5)
    assert by["mi355x-0"]["rate_gpm"] == pytest.approx(exp_m / 5)
    rel = t["last_release"]
    assert (rel["v"], rel["kind"], rel["bytes"], rel["upload_s"]) == (2, "delta", 500_000_000, 15.0)
    buf = t["buffer"]
    assert buf["read"] == 16 and buf["backlog"] == buf["written"] - 16


def test_flows_carry_the_tape_windows():
    """Flows replay tape events: groups, reads, releases with their upload window and loads
    with their fetch window; nothing older than WINDOW_S."""
    r = run()
    t = _view(r, now=T0 + 420)
    kinds = {f["k"] for f in t["flows"]}
    assert kinds == {"g", "r", "u", "d", "l"}
    ups = [f for f in t["flows"] if f["k"] == "u"]
    assert [(f["v"], f["kind"], f["u0"]) for f in ups] == [(1, "full", T0 + 150 - 45.0), (2, "delta", T0 + 345 - 15.0)]
    reads = [f for f in t["flows"] if f["k"] == "r"]
    assert [f["m"] for f in reads] == [8, 8]
    ld = [f for f in t["flows"] if f["k"] == "l" and f["s"] == "mi355x-0"][-1]
    assert (ld["v"], ld["seen"], ld["rs"]) == (2, T0 + 347, 12.0)
    g = [f for f in t["flows"] if f["k"] == "g" and f["s"] == "mi355x-0"]
    assert g and all(f["n"] == 7 and f["m"] == 1 for f in g)
    # agentic samplers are not double counted from their rounds
    assert len([f for f in t["flows"] if f["k"] == "g" and f["s"] == "b300-0"]) == 12
    assert t["flows"] == sorted(t["flows"], key=lambda f: f["t"])
    late = _view(r, now=T0 + 420 + topology.WINDOW_S)
    assert all(f["t"] >= T0 + 420 for f in late["flows"])


def test_samplers_without_agentic_groups_use_their_rounds():
    r = Reducer(run="plain")
    r.feed(sampler_start("h100-0", T0, "H100", "fp8", "vllm"))
    for k in range(5):
        r.feed(sampler_round("h100-0", T0 + 30 * (k + 1), k, 0, groups=2))
    t = _view(r, now=T0 + 200)
    s = t["samplers"][0]
    assert s["rate_gpm"] == pytest.approx(10 / 5) and s["arch"] == "Hopper"
    g = [f for f in t["flows"] if f["k"] == "g"]
    assert [f["m"] for f in g] == [2] * 5


def test_status_is_the_shared_island_state():
    """Status / liveness are reducer.island_states verdicts (the rule of the overview cards and
    the Islands now table), for a sampler that left as well as for live nodes."""
    r = run([{"event": "rl_sampler_leave", "sampler_id": "b300-0", "reason": "PREEMPTED", "incarnation": "a1",
              "time_unix": T0 + 410},
             {"event": "rl_sampler_stopped", "sampler_id": "b300-0", "time_unix": T0 + 411}])
    states = r.island_states(NOW)
    t = _view(r)
    for node in t["samplers"] + t["trainers"]:
        assert node["status"] == states[node["id"]]["status"]
        assert node["live"] == states[node["id"]]["alive"] == (node["status"] in ALIVE_STATUSES)
    b = {s["id"]: s for s in t["samplers"]}["b300-0"]
    assert not b["live"] and b["state"] == "offline" and b["reload"] is None


def test_trainer_phase_from_driver_phase():
    t = _view(run(), now=T0 + 330)
    tr = t["trainers"][0]
    assert tr["phase"] == "training" and tr["phase_since"] == T0 + 300


def test_no_panel_without_samplers():
    r = Reducer(run="sync")
    r.feed({"event": "rl_local_round", "island_id": 0, "local_round_id": 1, "time_unix": T0, "reward_mean": 0.1})
    assert "topology" not in r.page_view(now=T0 + 10)


@pytest.mark.parametrize("card,arch", [("A100-SXM4-80GB", "Ampere"), ("NVIDIA A10G", "Ampere"), ("L40S", "Ada"),
                                       ("NVIDIA H100 80GB HBM3", "Hopper"), ("H200", "Hopper"), ("GH200", "Hopper"),
                                       ("B200", "Blackwell"), ("NVIDIA B300", "Blackwell"), ("GB200", "Blackwell"),
                                       ("MI300X", "CDNA3"), ("AMD Instinct MI355X", "CDNA4"),
                                       ("Ascend910B", "Ascend"), (None, None), ("TPU v5e", None)])
def test_arch_of(card, arch):
    assert topology.arch_of(card) == arch


def test_page_ships_the_panel_script():
    html = render_page(None)
    assert "var YetoTopo=" in html and 'id="topoBox"' in html
    assert html.index("var YetoTopo=") < html.index("YetoTopo.render(V,OFFLINE)")
    js = (Path(topology.__file__).parent / "static" / "topology.js").read_text()
    assert "://" not in js.replace('"http:"+"//www.w3.org/2000/svg"', "")  # no URL: the page makes no request
    assert len(build_id()) == 12


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
function mk(id) { if (!els[id]) { els[id] = node(id); if (id === "yeto-data") els[id].textContent = data; } return els[id]; }
global.document = { documentElement: node("html"), getElementById: mk, querySelectorAll: () => [], title: "",
  createElementNS: () => { const n = node(null); svg.push(n); return n; }, createElement: () => node(null) };
global.navigator = {}; global.localStorage = { getItem() { return null; }, setItem() {} };
global.setInterval = () => 0; global.setTimeout = () => 0;
eval(app);
const texts = svg.map((n) => String(n.textContent || "")).filter(Boolean);
console.log(JSON.stringify({ hidden: els.topoBox.hidden, note: els.topoNote.textContent, texts,
  viewBox: (svg.find((n) => n.attrs["aria-label"] === "samplers to run store to trainers") || { attrs: {} }).attrs.viewBox }));
"""


def _smoke(tmp_path, reducer, now, width=1200):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(reducer, tmp_path / "p.html", generated_at=now)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page), str(width)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_exported_page_draws_the_panel(tmp_path):
    out = _smoke(tmp_path, run(), NOW)
    assert out["hidden"] is False and "offline snapshot" in out["note"]
    joined = " | ".join(out["texts"])
    for want in ("NVIDIA B300 · NVFP4 · vLLM", "LATEST v2", "Trainer"):
        assert want in joined, joined
    assert out["viewBox"].startswith("0 0 1200 ")


def test_narrow_layout_and_folding(tmp_path):
    """420 px: the narrow layout; more than 12 live samplers fold into one card per
    card / precision / engine instead of growing without bound."""
    extra = []
    for i in range(14):
        name = f"l40s-{i}"
        extra += [sampler_start(name, T0, "L40S", "bf16", "vllm", tp=1), group(name, T0 + 300 + i, 1)]
    out = _smoke(tmp_path, run(extra), NOW, width=420)
    assert out["viewBox"].startswith("0 0 400 ")
    joined = " | ".join(out["texts"])
    assert "L40S · BF16 ×14" in joined, joined
