"""The sampler's GPU memory guard (yeto.rl.engine.vram_guard) on fake card readings:
the >97% alert (tape, pool event, dashboard) and the in-place reload check (free
memory against the release's staging: in place / empty the cache and read again /
smaller chunks / restart)."""

from __future__ import annotations

import json
import struct

import pytest

from yeto.dashboard.async_rl import AsyncState
from yeto.launch.nc import elastic as el
from yeto.launch.pool.status import Probe
from yeto.rl.adapters.sampler_engines import vllm_expert_reload as ver
from yeto.rl.adapters.sampler_engines.vllm_server import VllmOptions, VllmServerEngine
from yeto.rl.engine import vram_guard as vg
from yeto.rl.engine.release_overlay import overlay_release
from yeto.rl.engine.telemetry import ResourceSampler

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config
from test_rl_k31_vllm_expert_reload import PIECES, _name

TOTAL_MB = 288 * 1024
GIB = 2**30


def _cards(*free_gib, total_mb=TOTAL_MB):
    return [{"index": i, "mem_used_mb": total_mb - f * 1024, "mem_total_mb": total_mb} for i, f in enumerate(free_gib)]


# ------------------------------------------------------------------ alert
def test_a_card_above_97_percent_alerts_once_then_again_after_the_interval_and_clears():
    events, clock = [], [0.0]
    watch = vg.VramWatch(lambda ev, **f: events.append((ev, f)), realert_s=600, clock=lambda: clock[0])
    assert watch.observe(_cards(40, 40)) is None  # 86%
    assert watch.observe(_cards(40, 5)) == vg.VRAM_HIGH_EVENT  # 98.3% on card 1
    ev, f = events[-1]
    assert f["max_frac"] > 0.97 and [c["index"] for c in f["cards"]] == [1] and f["min_free_mib"] == 5 * 1024
    clock[0] = 300
    assert watch.observe(_cards(40, 4)) is None  # still high: no repeat inside realert_s
    clock[0] = 601
    assert watch.observe(_cards(40, 4)) == vg.VRAM_HIGH_EVENT and events[-1][1]["alert"] == 2
    clock[0] = 700
    assert watch.observe(_cards(40, 30)) == vg.VRAM_OK_EVENT and events[-1][1]["high_for_s"] == 700
    assert watch.observe(_cards(40, 30)) is None
    assert watch.observe([{"index": 0, "mem_used_mb": None, "mem_total_mb": None}]) is None  # no reading
    with pytest.raises(ValueError):
        vg.VramWatch(lambda *a, **k: None, high_frac=1.5)


class _Collector:
    source = "fake-smi"

    def __init__(self, readings):
        self.readings = list(readings)

    def open(self):
        pass

    def sample(self):
        return self.readings.pop(0)


def test_the_resource_sampler_feeds_the_watch_from_its_card_samples():
    events = []
    emit = lambda ev, **f: events.append(ev)  # noqa: E731
    sampler = ResourceSampler(emit, collector=_Collector([_cards(2), _cards(50)]), interval_s=3600,
                              vram_watch=vg.VramWatch(emit))
    assert sampler.tick() and sampler.tick()
    assert events == ["rl_resource_sample", vg.VRAM_HIGH_EVENT, "rl_resource_sample", vg.VRAM_OK_EVENT]


def test_the_pool_probe_and_pool_report_a_new_alert_once():
    probe = Probe.parse("YETO_NC_PROBE status=running exit=0 joined=1 age=3 tw=0 tok=- run=- wq=- vh=2 vm=0.981", 0)
    assert probe.vram_alerts == 2 and probe.vram_max_frac == pytest.approx(0.981)
    assert Probe.parse("YETO_NC_PROBE status=running exit=0 joined=1 age=3", 0).vram_alerts is None
    script = el.probe_script("yeto-sampler-nc-0", "/w/events*.jsonl")
    assert "rl_sampler_vram_high" in script and "vh=$vh vm=$vm" in script

    from yeto.launch.pool.island_pool import IslandPool, Member, PoolConfig

    seen = []
    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="nc"), ops=None,
                      events=lambda name, **d: seen.append((name, d)))
    m = Member(account=1, ticket="t1", shard="nc-0")
    pool._vram_seen(m, probe)
    pool._vram_seen(m, probe)  # the same alert count: no second event
    pool._vram_seen(m, Probe(node="ok", vram_alerts=1, vram_max_frac=0.98))  # a new incarnation's tape
    pool._vram_seen(m, Probe(node="ok", vram_alerts=2, vram_max_frac=0.99))
    assert [(n, d["alerts"], d["max_frac"]) for n, d in seen] == [
        ("sampler_vram_high", 2, pytest.approx(0.981)), ("sampler_vram_high", 1, 0.98), ("sampler_vram_high", 2, 0.99)]


def test_the_probe_script_counts_the_alerts_of_the_newest_tape(tmp_path):
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("no bash")
    tape = tmp_path / "events.a.jsonl"
    lines = [{"event": "rl_sampler_join"}, {"event": vg.VRAM_HIGH_EVENT, "max_frac": 0.975},
             {"event": "rl_sampler_round"}, {"event": vg.VRAM_HIGH_EVENT, "max_frac": 0.9812}]
    tape.write_text("".join(json.dumps(x) + "\n" for x in lines))
    script = el.probe_script("c", str(tmp_path / "events*.jsonl"), state_script="echo running 0",
                             logs_script="true")
    import os
    import sys

    shim = tmp_path / "bin"
    shim.mkdir()  # GNU ``stat -c %Y`` on a BSD host (the nodes run Linux)
    (shim / "stat").write_text(f"#!/bin/sh\nexec {sys.executable} -c 'import os,sys; "
                               "print(int(os.stat(sys.argv[-1]).st_mtime))' \"$@\"\n")
    (shim / "stat").chmod(0o755)
    env = {**os.environ, "PATH": f"{shim}:{os.environ.get('PATH', '')}"}

    def run_probe():
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30, env=env).stdout

    out = run_probe()
    probe = Probe.parse(out, 0)
    assert probe.vram_alerts == 2 and probe.vram_max_frac == pytest.approx(0.9812), out
    tape.write_text(json.dumps({"event": "rl_sampler_join"}) + "\n")
    out = run_probe()
    probe = Probe.parse(out, 0)
    assert probe.vram_alerts == 0 and probe.vram_max_frac is None and probe.sampler == "running", out


def test_the_dashboard_shows_the_alert_on_the_sampler_row():
    a = AsyncState()
    a.fold({"event": "rl_sampler_start"}, "nc-0", 1.0)
    a.fold({"event": vg.VRAM_HIGH_EVENT, "max_frac": 0.981, "threshold": 0.97, "cards": [{"index": 3}]}, "nc-0", 2.0)
    a.fold({"event": vg.VRAM_OK_EVENT, "max_frac": 0.95, "threshold": 0.97}, "nc-0", 3.0)
    v = a.sampler["nc-0"]["vram"]
    assert v["alerts"] == 1 and v["max_frac"] == pytest.approx(0.981) and v["high"] is False
    a.fold_pool({"event": "nc_elastic_sampler_vram_high", "shard": "nc-0", "max_frac": 0.981, "alerts": 1})
    assert a.pool[-1]["max_frac"] == pytest.approx(0.981) and a.pool[-1]["alerts"] == 1


# ------------------------------------------------------------------ reload plan
def test_reload_plan_in_place_chunked_or_restart():
    est = vg.StagingEstimate(layer=1, layer_bytes=16 * GIB, experts=8, per_gpu_bytes=4 * GIB)  # 0.5 GiB / expert
    assert vg.reload_plan(10 * GIB, est, reserve_bytes=2 * GIB)["mode"] == vg.SAFE_IN_PLACE
    assert vg.reload_plan(None, est)["mode"] == vg.SAFE_IN_PLACE  # no card reading: as before
    chunked = vg.reload_plan(int(3.6 * GIB), est, reserve_bytes=2 * GIB)
    assert chunked["mode"] == vg.SAFE_CHUNKED and chunked["max_experts"] == 3
    assert vg.reload_plan(int(2.2 * GIB), est, reserve_bytes=2 * GIB)["mode"] == vg.SAFE_RESTART
    # staging of the largest layer, narrowed over TP 8, staging + converted copies
    plan = {1: {"experts": [0, 2], "tensors": {}}, 2: {"experts": [5], "tensors": {}}}
    e = vg.staging_bytes({1: 8 * GIB, 2: 1 * GIB}, plan, tp=8)
    assert (e.layer, e.per_gpu_bytes, e.experts, e.per_expert_bytes) == (1, 2 * GIB, 2, GIB)


def test_a_layer_is_cut_into_expert_chunks_with_their_own_tensors():
    entry = {"experts": [0, 2, 5], "tensors": {_name(1, e, p, k): "f" for e in (0, 2, 5) for p, k in PIECES}}
    chunks = ver.expert_chunks(entry, 2)
    assert [c[0] for c in chunks] == [[0, 2], [5]] and [len(c[1]) for c in chunks] == [12, 6]
    assert all(".experts.5." in n for n in chunks[1][1])
    assert [c[0] for c in ver.expert_chunks(entry, None)] == [[0, 2, 5]]


# ------------------------------------------------------------------ adapter
def _safetensors(path, sizes):
    """A header-only safetensors file (the guard reads the header, never the data)."""
    header, at = {}, 0
    for name, n in sizes.items():
        header[name] = {"dtype": "U8", "shape": [n], "data_offsets": [at, at + n]}
        at += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _setup(tmp_path, readings):
    """Base with layer 1 experts 0..3; a release of experts 0 and 2, 1 GiB per tensor
    (12 GiB in the layer: 3 GiB staging per TP-8 rank, 1.5 GiB per expert)."""
    names = {e: [_name(1, e, p, k) for p, k in PIECES] for e in range(4)}
    rpc = []

    def collective(body):
        rpc.append(body)
        return {"results": [{"seconds": 1.0}] * 8}

    http = _Http({("GET", "/health"): {}, ("POST", "/pause?mode=wait"): {}, ("POST", "/resume"): {},
                  ("POST", "/collective_rpc"): collective})
    config = _config(tmp_path, card_model="MI355X", device_family="amd", in_place_reload=True)
    reads = list(readings)
    engine = VllmServerEngine(config, VllmOptions(platform="rocm", tp=8, expert_reload=True),
                              spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None,
                              run=lambda argv: json.dumps([{"gpu": i} for i in range(8)]),
                              card_memory=lambda: reads.pop(0) if len(reads) > 1 else reads[0])
    base = config.base_dir
    base_map = {n: "model-00001.safetensors" for e in range(4) for n in names[e]}
    (base / ver.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": base_map}))
    _safetensors(base / "model-00001.safetensors", {n: 1 for e in range(4) for n in names[e]})
    release = tmp_path / "rel"
    release.mkdir()
    rel_map = {n: "model-00001.safetensors" for e in (0, 2) for n in names[e]}
    (release / ver.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": rel_map}))
    _safetensors(release / "model-00001.safetensors", {n: GIB for n in rel_map})
    (release / "config.json").write_text((base / "config.json").read_text())
    engine.start(None, 0)
    overlay = overlay_release(base, release, tmp_path / "ov")
    return engine, http, rpc, overlay


def _posts(http):
    return [p for m, p, b in http.calls if m == "POST"]


def test_enough_free_memory_reloads_in_place_without_touching_the_cache(tmp_path):
    engine, http, rpc, overlay = _setup(tmp_path, [_cards(*[20] * 8)])
    report = engine.load(overlay, 1)
    guard = report.detail["memory_guard"]
    assert report.reload_mode == "in_place" and guard["mode"] == "in_place"
    assert guard["staging_bytes"] == 3 * GIB and guard["free_bytes"] == 20 * GIB
    assert [b["method"] for b in rpc] == [ver.RELOAD_METHOD] and "max_experts" not in rpc[0]["kwargs"]


def test_short_memory_empties_the_cache_first_and_reloads_in_place_when_that_freed_enough(tmp_path):
    engine, http, rpc, overlay = _setup(tmp_path, [_cards(20, 20, 4, 20, 20, 20, 20, 20), _cards(*[9] * 8)])
    guard = engine.load(overlay, 1).detail["memory_guard"]
    assert [b["method"] for b in rpc] == [ver.RELEASE_CACHE_METHOD, ver.RELOAD_METHOD]
    assert guard["mode"] == "in_place" and guard["emptied_cache"] and guard["first_free_bytes"] == 4 * GIB
    assert _posts(http) == ["/collective_rpc", "/pause?mode=wait", "/collective_rpc", "/resume"]


def test_still_short_after_the_cache_reloads_in_smaller_chunks(tmp_path):
    engine, http, rpc, overlay = _setup(tmp_path, [_cards(*[4] * 8)])  # 4 - 2 reserve = room for one expert
    report = engine.load(overlay, 1)
    assert report.reload_mode == "in_place" and report.detail["memory_guard"]["mode"] == "chunked"
    assert rpc[-1]["method"] == ver.RELOAD_METHOD and rpc[-1]["kwargs"]["max_experts"] == "1"


def test_no_room_for_one_expert_restarts_instead_of_staging_into_a_full_gpu(tmp_path):
    engine, http, rpc, overlay = _setup(tmp_path, [_cards(*[3] * 8)])
    report = engine.load(overlay, 1)
    assert report.detail["reload"] == "restart" and report.detail["expert_reload_off"] == "gpu_memory"
    assert report.detail["memory_guard"]["mode"] == "restart"
    assert [b["method"] for b in rpc] == [ver.RELEASE_CACHE_METHOD]  # no staging RPC was sent
    assert "/pause?mode=wait" not in _posts(http)
