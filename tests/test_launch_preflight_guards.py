"""openspec change launch-preflight-guards: thread preflight, GPU memory
estimate preflight, per-island override for negative-test runs."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from yeto import island_overrides as io_
from yeto import launch_preflight as lp
from yeto import memory_estimate as me
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401 (autouse fixtures)

# tests/conftest.py replaces count_threads with a fixed reading; keep the real one.
REAL_COUNT_THREADS = lp.count_threads


# --- 1.1 thread counting from a fake /proc -------------------------------------
def _proc(root, pid, *, uid, threads, ppid=1, cmd=b"python x.py"):
    d = root / str(pid)
    d.mkdir()
    (d / "status").write_text(f"Name:\tx\nPPid:\t{ppid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
                              f"Threads:\t{threads}\n")
    (d / "cmdline").write_bytes(cmd.replace(b" ", b"\0"))


def test_count_threads_fake_proc(tmp_path):
    _proc(tmp_path, 10, uid=1000, threads=7)
    _proc(tmp_path, 20, uid=1000, threads=121, cmd=b"/venv/bin/python3 -m sky.server.server --host=127.0.0.1")
    _proc(tmp_path, 21, uid=1000, threads=65, ppid=20, cmd=b"SkyPilot:executor:short:21")
    _proc(tmp_path, 22, uid=1000, threads=5, ppid=21, cmd=b"ssh host")  # descendant of the server
    _proc(tmp_path, 30, uid=0, threads=999)  # other user
    (tmp_path / "self").mkdir()  # non-pid entry
    c = REAL_COUNT_THREADS(tmp_path, uid=1000)
    assert (c.total, c.sky_api, c.sky_api_found, c.processes) == (7 + 121 + 65 + 5, 191, True, 4)


def test_count_threads_without_sky_server(tmp_path):
    _proc(tmp_path, 10, uid=5, threads=3)
    c = REAL_COUNT_THREADS(tmp_path, uid=5)
    assert (c.total, c.sky_api, c.sky_api_found) == (3, 0, False)


# --- 1.2 / 2.4 / 3.1 CLI ------------------------------------------------------------
def _cli(extra=()):
    from yeto.cli import parse_args

    return parse_args(["--gpu", "modal:1xh100", "--model", "org/model", "--data", "org/data", *extra])


def test_cli_defaults():
    a = _cli()
    assert (a.preflight_threads, a.preflight_thread_start, a.preflight_thread_hard,
            a.preflight_thread_wait_s) == ("wait", 2800, 3000, 1800)
    assert (a.preflight_memory, a.preflight_memory_margin) == ("error", 0.9)
    assert a.rl_island_override is None and a.rl_negative_test_run is False


def test_cli_values():
    a = _cli(["--preflight-threads", "error", "--preflight-thread-start", "10",
              "--preflight-thread-hard", "20", "--preflight-thread-wait-s", "5",
              "--preflight-memory", "off", "--preflight-memory-margin", "0.8",
              "--rl-island-override", "1:rl_lr_schedule=constant",
              "--rl-island-override", "0:identity-test-salt=x", "--rl-negative-test-run"])
    assert (a.preflight_threads, a.preflight_thread_start, a.preflight_thread_hard,
            a.preflight_thread_wait_s) == ("error", 10, 20, 5.0)
    assert (a.preflight_memory, a.preflight_memory_margin) == ("off", 0.8)
    assert a.rl_island_override == ["1:rl_lr_schedule=constant", "0:identity-test-salt=x"]
    with pytest.raises(SystemExit):
        _cli(["--preflight-threads", "maybe"])


# --- 1.3 / 1.4 thread preflight --------------------------------------------------
def _targs(**kw):
    base = dict(preflight_threads="wait", preflight_thread_start=2800, preflight_thread_hard=3000,
                preflight_thread_wait_s=1800)
    base.update(kw)
    return SimpleNamespace(**base)


def _counter(values, sky=1800, found=True):
    it = iter(values)
    last = [None]

    def read():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return lp.ThreadCount(last[0], sky if found else 0, found, 10)
    return read


class _Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_threads_below_start_continue_and_record():
    out = io.StringIO()
    rec = lp.thread_preflight(_targs(), counter=_counter([2500]), out=out)
    assert rec["total"] == 2500 and rec["readings"] == [2500] and not rec["waited"]
    assert "threads OK" in out.getvalue()


def test_threads_wait_then_continue():
    clk = _Clock()
    out = io.StringIO()
    rec = lp.thread_preflight(_targs(), counter=_counter([2900, 2850, 2700]), clock=clk,
                              sleep=clk.sleep, out=out)
    assert rec["readings"] == [2900, 2850, 2700] and rec["waited"] and clk.sleeps == [30.0, 30.0]
    assert "WAIT" in out.getvalue()


def test_threads_wait_times_out():
    clk = _Clock()
    with pytest.raises(lp.PreflightError, match="after waiting"):
        lp.thread_preflight(_targs(preflight_thread_wait_s=90), counter=_counter([2900]),
                            clock=clk, sleep=clk.sleep, out=io.StringIO())
    assert clk.sleeps == [30.0, 30.0, 30.0]


def test_threads_error_mode_and_hard_limit():
    with pytest.raises(lp.PreflightError, match="start threshold 2800"):
        lp.thread_preflight(_targs(preflight_threads="error"), counter=_counter([2900]))
    for mode in ("wait", "error"):
        with pytest.raises(lp.PreflightError, match="hard limit 3000"):
            lp.thread_preflight(_targs(preflight_threads=mode), counter=_counter([3100]),
                                sleep=lambda s: pytest.fail("must not wait at the hard limit"))


def test_threads_off_skips_and_warns():
    out = io.StringIO()
    rec = lp.thread_preflight(_targs(preflight_threads="off"),
                              counter=lambda: pytest.fail("off must not read"), out=out)
    assert rec == {"mode": "off", "disabled": True} and "OFF" in out.getvalue()


def test_thread_message_text():
    cfg = lp.ThreadConfig()
    text = lp.thread_message(lp.ThreadCount(2900, 1800, True, 50), cfg, "X")
    assert "user threads 2900, start threshold 2800, hard limit 3000" in text
    assert "SkyPilot API server: 1800 threads (62% of 2900)" in text
    assert "sky api stop && sky api start" in text and "no launch is in progress" in text
    assert "Ray" in text
    text = lp.thread_message(lp.ThreadCount(2900, 0, False, 50), cfg, "X")
    assert "未找到 SkyPilot API 服务" in text and "sky api stop && sky api start" in text


# --- 2.1 memory estimate ----------------------------------------------------------
ARCH = me.ModelArch(params=1e9, layers=10, hidden=1000, vocab=50000, intermediate=4000,
                    heads=10, kv_heads=10, head_dim=100)


def test_estimate_parts_by_hand():
    G = me.GIB
    req = me.MemoryRequest(arch=ARCH, gpu="H100", gpu_mem_gib=80.0, tokens=4000, tuning="full",
                           colocated=False, offload=False)
    e = me.estimate(req)["parts_gib"]
    assert e["weights"] == pytest.approx(2e9 / G, abs=1e-3)
    assert e["grads_optimizer"] == pytest.approx(16e9 / G, abs=1e-3)
    assert e["activations"] == pytest.approx(me.K_ACT_NO_RECOMPUTE * 10 * 4000 * 1000 * 2 / G, abs=1e-3)
    assert e["logits_block"] == pytest.approx(0.5 * 4000 * 50000 * 4 / G, abs=1e-3)
    alloc = (2e9 + 16e9 + me.K_ACT_NO_RECOMPUTE * 8e7 + 4e8) / G
    assert e["fixed_overhead"] == pytest.approx(me.R0_GIB + me.FRAG * alloc, abs=1e-2)
    assert e["inference"] == 0.0
    # TP 2 halves weights, activations, logits; distributed optimizer shards optimizer by DP
    e2 = me.estimate(me.MemoryRequest(arch=ARCH, gpu="H100", gpu_mem_gib=80.0, tokens=4000,
                                      tuning="full", tp=2, dp=2, distributed_optimizer=True,
                                      colocated=False))["parts_gib"]
    assert e2["weights"] == pytest.approx(1e9 / G, abs=1e-3)
    assert e2["grads_optimizer"] == pytest.approx((2e9 + 3e9) / G, abs=1e-3)
    # chunked logits
    e3 = me.estimate(me.MemoryRequest(arch=ARCH, gpu="H100", gpu_mem_gib=80.0, tokens=4000,
                                      logits_chunk=1000))["parts_gib"]
    assert e3["logits_block"] == pytest.approx(0.5 * 1000 * 50000 * 4 / G, abs=1e-3)
    # LoRA: 16 bytes per adapter parameter
    lora = me.lora_params(ARCH, 16, "attention")
    assert lora == 16 * 10 * ((1000 + 1000) * 4)
    # not offloaded engine on the same GPU counts mem_fraction x memory
    e4 = me.estimate(me.MemoryRequest(arch=ARCH, gpu="H100", gpu_mem_gib=80.0, tokens=4000,
                                      colocated=True, offload=False))["parts_gib"]
    assert e4["inference"] == pytest.approx(32.0)


def test_rollout_phase_dominates_small_model():
    r = me.estimate(me.MemoryRequest(arch=ARCH, gpu="H100", gpu_mem_gib=80.0, tokens=100))
    assert r["peak_gib"] == pytest.approx(0.4 * 80 + me.R0_GIB, abs=1e-3)
    assert r["parts_gib"]["rollout_phase_extra"] > 0


def _rl_args(model="Qwen/Qwen3.5-4B", seq=16384, resp=8192, **kw):
    a = dict(training_mode="rl", model=model, seq_len=seq, rollout_max_response_len=resp,
             tuning="lora", lora_r=16, lora_targets="attention", sglang_mem_fraction_static=0.4,
             preflight_memory="error", preflight_memory_margin=0.9, gpu="modal:1xh200")
    a.update(kw)
    return SimpleNamespace(**a)


def test_unknown_model_not_estimated(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent")
    out = io.StringIO()
    rec = lp.memory_preflight(_rl_args(model="org/unknown-model"), [SimpleNamespace(gpu="H100", total_gpus=1)],
                              out=out)
    assert rec["islands"][0]["estimated"] is False and "显存未估算" in out.getvalue()


# --- 2.2 back-test acceptance (calibration runs) -------------------------------------
def test_backtest_run_c_refused_run_d_passes(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent")  # builtin config only
    h200 = SimpleNamespace(gpu="H200", total_gpus=1)
    h100 = SimpleNamespace(gpu="H100", total_gpus=1)
    c = me.estimate_for_launch(_rl_args(), h200, margin=0.9)
    b = me.estimate_for_launch(_rl_args(), h100, margin=0.9)
    d = me.estimate_for_launch(_rl_args(seq=12288, resp=6144), h200, margin=0.9)
    assert not b["fits"] and not c["fits"] and d["fits"]
    assert abs(c["parts_gib"]["logits_block"] - 7.58) / 7.58 <= 0.10
    measured_d = 132.36e9 / me.GIB
    assert abs(d["peak_gib"] - measured_d) / measured_d <= 0.10
    assert any("context 16384 -> " in s["change"] for s in c["suggestions"])
    assert all(s["peak_gib"] <= 0.9 * 139.80 for s in c["suggestions"])


# --- 2.3 / 2.4 launch wiring: refuse with no cloud call -------------------------------
class _Stop(Exception):
    pass


def _launch_args(extra=()):
    from test_rl_algorithm_provenance import _launcher_args

    return _launcher_args("ports", extra, gpu="modal:1xh200")


@pytest.fixture
def no_cloud(monkeypatch):
    """Every path that could reach a cloud raises; returns the call log."""
    import yeto.launcher as launcher
    from yeto import sky_patches

    calls = []

    def boom(name):
        def f(*a, **k):
            calls.append(name)
            raise _Stop(name)
        return f
    monkeypatch.setattr(sky_patches, "install", boom("sky_patches.install"))
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)
    monkeypatch.setattr(launcher, "prepare_verda_islands", boom("verda"))
    monkeypatch.setattr(launcher, "make_syncer_task", boom("make_syncer_task"))
    monkeypatch.setattr(launcher, "_write_run_manifest", lambda a: calls.append("manifest"))
    return calls


@pytest.mark.parametrize("reading,mode,expect", [
    (2500, "wait", "continue"), (2900, "error", "refuse"), (3100, "wait", "refuse"),
    (3100, "error", "refuse")])
def test_launch_thread_preflight_before_any_cloud_call(monkeypatch, no_cloud, reading, mode, expect):
    import yeto.launcher as launcher

    monkeypatch.setattr(lp, "count_threads", lambda *a, **k: lp.ThreadCount(reading, 0, False, 1))
    args = _launch_args(("--preflight-threads", mode))
    if expect == "continue":
        with pytest.raises(_Stop):
            launcher.run(args)
        assert no_cloud == ["sky_patches.install"]
        assert args._launch_preflight_manifest["preflight"]["threads"]["total"] == 2500
    else:
        with pytest.raises(lp.PreflightError):
            launcher.run(args)
        assert no_cloud == []


def test_launch_thread_wait_then_continue(monkeypatch, no_cloud):
    import yeto.launcher as launcher

    vals = iter([2900, 2700])
    monkeypatch.setattr(lp, "count_threads", lambda *a, **k: lp.ThreadCount(next(vals), 0, False, 1))
    monkeypatch.setattr(lp.time, "sleep", lambda s: None)
    with pytest.raises(_Stop):
        launcher.run(_launch_args())
    assert no_cloud == ["sky_patches.install"]


def _memory_launch(monkeypatch, extra, *, seq, resp, tmp_path):
    import yeto.launcher as launcher
    from yeto import runs, sky_patches

    monkeypatch.setattr(runs, "run_dir", lambda name: tmp_path / name)

    calls = []
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent")
    monkeypatch.setattr(sky_patches, "install", lambda: calls.append("install"))
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)

    def no_cloud(*a, **k):
        calls.append("cloud")
        raise _Stop()
    monkeypatch.setattr(launcher, "prepare_verda_islands", no_cloud)
    monkeypatch.setattr(launcher, "make_syncer_task", no_cloud)
    written = []
    monkeypatch.setattr(launcher, "_write_run_manifest",
                        lambda a: written.append(dict(a._launch_preflight_manifest)))
    args = _launch_args(("--seq-len", str(seq), "--rollout-max-response-len", str(resp), *extra))
    args.model = "Qwen/Qwen3.5-4B"
    args.preflight_threads = "off"
    return launcher, args, calls, written


def test_launch_memory_run_c_refused_without_cloud(monkeypatch, tmp_path):
    launcher, args, calls, written = _memory_launch(monkeypatch, (), seq=16384, resp=8192, tmp_path=tmp_path)
    with pytest.raises(lp.PreflightError) as e:
        launcher.run(args)
    assert "logits_block" in str(e.value) and "context 16384 ->" in str(e.value)
    assert "cloud" not in calls and written == []


def test_launch_memory_run_d_passes_and_is_recorded(monkeypatch, tmp_path):
    launcher, args, calls, written = _memory_launch(monkeypatch, (), seq=12288, resp=6144, tmp_path=tmp_path)
    with pytest.raises(_Stop):
        launcher.run(args)
    mem = written[0]["preflight"]["memory"]
    assert mem["mode"] == "error" and mem["islands"][0]["fits"] and "parts_gib" in mem["islands"][0]


def test_launch_memory_off_and_warn(monkeypatch, capsys, tmp_path):
    launcher, args, calls, written = _memory_launch(monkeypatch, ("--preflight-memory", "off"),
                                                    seq=16384, resp=8192, tmp_path=tmp_path)
    with pytest.raises(_Stop):
        launcher.run(args)
    assert written[0]["preflight"]["memory"] == {"mode": "off", "margin": 0.9, "disabled": True,
                                                 "islands": []}
    assert written[0]["preflight"]["threads"]["disabled"] is True
    assert "GPU memory preflight is OFF" in capsys.readouterr().err
    launcher, args, calls, written = _memory_launch(monkeypatch, ("--preflight-memory", "warn"),
                                                    seq=16384, resp=8192, tmp_path=tmp_path)
    with pytest.raises(_Stop):
        launcher.run(args)
    assert written[0]["preflight"]["memory"]["islands"][0]["fits"] is False


def test_uncalibrated_model_over_limit_only_warns(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent")
    out = io.StringIO()
    rec = lp.memory_preflight(_rl_args(model="Qwen/Qwen3-8B", seq=65536, resp=8192),
                              [SimpleNamespace(gpu="L4", total_gpus=1)], out=out)
    assert rec["islands"][0]["fits"] is False and "advisory only" in out.getvalue()


# --- 2.5 / 3.2 dry-run ----------------------------------------------------------------
def _plan(extra=(), gpu="modal:1xh100,modal:1xh100"):
    from test_rl_algorithm_provenance import _launcher_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", extra, gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def test_dry_run_memory_estimate_per_island():
    _, plan = _plan()
    for isl in plan["island_requests"]:
        assert "memory_estimate" in isl and isl["memory_estimate"]["learner_id"] == isl["learner_id"]
    _, plan = _plan(("--preflight-memory", "off"))
    assert all("memory_estimate" not in i for i in plan["island_requests"])


def test_dry_run_override_changes_only_named_island():
    _, base = _plan()
    _, plan = _plan(("--rl-negative-test-run", "--rl-island-override", "1:rl_lr_schedule=constant"))
    i0, i1 = plan["island_requests"]
    b0, b1 = base["island_requests"]
    assert i0["learner_command"] == b0["learner_command"]
    assert "--rl-lr-schedule constant" in i1["learner_command"]
    assert "--rl-lr-schedule" not in b1["learner_command"]
    assert i1["island_overrides"] == [{"island": 1, "key": "rl_lr_schedule", "name": "学习率调度",
                                       "old": "linear", "new": "constant"}]


def test_dry_run_policy_age_override_is_rechecked():
    with pytest.raises(ValueError):
        _plan(("--rl-negative-test-run", "--rl-island-override", "1:rl_max_policy_age=99999"))


def test_island_task_env_only_for_negative_runs():
    import yeto.launcher as launcher

    seen = []

    def factory(args, spec, m, n, addr):
        seen.append(args)
        return SimpleNamespace(envs={"A": "1"})
    args = SimpleNamespace(rl_negative_test_run=False, rl_lr_schedule="auto")
    t = launcher._island_task(factory, args, None, 0, 2, "x", {})
    assert t.envs == {"A": "1"} and seen[0] is args
    args = SimpleNamespace(rl_negative_test_run=True, rl_lr_schedule="auto", rl_max_policy_age=0,
                           rl_island_scheduling="legacy", training_mode="rl", rl_engine="ports")
    ov = {1: {"identity_test_salt": "s1"}}
    t0 = launcher._island_task(factory, args, None, 0, 2, "x", ov)
    t1 = launcher._island_task(factory, args, None, 1, 2, "x", ov)
    assert t0.envs == {"A": "1", io_.NEGATIVE_RUN_ENV: "1"}
    data = json.loads(t1.envs[io_.OVERRIDE_ENV])
    assert data["island"] == 1 and data["overrides"][0]["new"] == "s1"
    assert io_.identity_test_salt(t1.envs) == "s1" and io_.identity_test_salt(t0.envs) is None


# --- 3.1 validation ----------------------------------------------------------------
def test_override_validation():
    a = SimpleNamespace(training_mode="rl", rl_negative_test_run=False,
                        rl_island_override=["1:rl_lr_schedule=constant"])
    with pytest.raises(ValueError, match="--rl-negative-test-run"):
        io_.overrides_of(a, 2)
    a.rl_negative_test_run = True
    assert io_.overrides_of(a, 2) == {1: {"rl_lr_schedule": "constant"}}
    for item, msg in [("2:rl_lr_schedule=constant", "out of range"),
                      ("1:model_revision=abc", "allowed"),
                      ("1:rl_max_carry_lag=3", "syncer"),
                      ("1:rl_lr_schedule=cosine", "must be one of"),
                      ("x:rl_lr_schedule=constant", "ISLAND:KEY=VALUE"),
                      ("1:rl_max_policy_age=-1", ">= 0")]:
        a.rl_island_override = [item]
        with pytest.raises(ValueError, match=msg):
            io_.overrides_of(a, 2)
    a.rl_island_override = []
    assert io_.overrides_of(a, 2) == {}


# --- 3.3 identity salt ---------------------------------------------------------------
def test_identity_salt_changes_contract_only_when_given():
    from yeto.rl.engine.backend_identity import island_contract_sha256

    ident, lr = "a" * 64, "b" * 64
    base = island_contract_sha256(ident, lr)
    assert island_contract_sha256(ident, lr, test_salt=None) == base
    salted = island_contract_sha256(ident, lr, test_salt="s1")
    assert salted != base and salted == island_contract_sha256(ident, lr, test_salt="s1")
    assert salted != island_contract_sha256(ident, lr, test_salt="s2")
    assert island_contract_sha256(ident, None) == ident
    assert island_contract_sha256(ident, None, test_salt="s1") != ident
    assert island_contract_sha256(None, lr, test_salt="s1") is None


# --- 3.4 manifest fields and island tape ---------------------------------------------
def test_manifest_fields_and_tape_event(tmp_path, capsys):
    args = SimpleNamespace(rl_negative_test_run=True, rl_lr_schedule="auto",
                           rl_island_scheduling="elastic", rl_checkpoint_store=None)
    ov = {1: {"rl_lr_schedule": "linear"}}
    m = io_.manifest_fields(args, ov)
    assert m["negative_test"] is True
    assert m["island_overrides"] == [{"island": 1, "key": "rl_lr_schedule", "name": "学习率调度",
                                      "old": "constant", "new": "linear"}]
    assert io_.manifest_fields(SimpleNamespace(rl_negative_test_run=False), {}) == {}
    io_.print_warning(args, ov)
    assert "NEGATIVE-TEST RUN" in capsys.readouterr().err
    env = io_.island_env(args, 1, ov)
    tape = tmp_path / "rl-island-1.jsonl"
    island_args = SimpleNamespace(event_tape=str(tape), rl_resume_store=None,
                                  rl_elastic_checkpoint_store=None)
    slept = []
    io_.island_startup(island_args, 1, env, sleep=slept.append)
    assert slept == [180.0]
    first = json.loads(tape.read_text().splitlines()[0])
    assert first["event"] == "rl_island_override" and first["island_id"] == 1
    assert first["overrides"][0]["new"] == "linear"
    # an island without overrides writes nothing
    tape0 = tmp_path / "rl-island-0.jsonl"
    io_.island_startup(SimpleNamespace(event_tape=str(tape0)), 0, io_.island_env(args, 0, ov),
                       sleep=slept.append)
    assert not tape0.exists() and slept == [180.0]  # the unchanged island does not wait
    args.rl_negative_join_delay_s = 5
    io_.island_startup(island_args, 1, io_.island_env(args, 1, ov), sleep=slept.append)
    assert slept == [180.0, 5.0]
    args.rl_negative_join_delay_s = -1
    with pytest.raises(ValueError, match="join-delay"):
        io_.island_env(args, 1, ov)


def test_manifest_written_by_launcher(monkeypatch, tmp_path):
    import yeto.launcher as launcher
    from yeto import runs

    monkeypatch.setattr(runs, "run_dir", lambda name: tmp_path / name)
    args = SimpleNamespace(training_mode="rl", rl_engine="ports", rl_image=None, source_sha256=None,
                           cluster_prefix="neg", _launch_preflight_manifest={
                               "negative_test": True, "island_overrides": [{"island": 1}],
                               "preflight": {"threads": {"total": 1}}})
    monkeypatch.setattr(launcher, "_miles_overlay_manifest", lambda a, e: {})
    launcher._write_run_manifest(args)
    data = json.loads((tmp_path / "neg" / "run_manifest.json").read_text())
    assert data["negative_test"] is True and data["island_overrides"] == [{"island": 1}]
    assert data["preflight"]["threads"]["total"] == 1


# --- 3.5 dashboard ---------------------------------------------------------------------
def test_reducer_marks_negative_island():
    from yeto.dashboard.reducer import Reducer

    r = Reducer(run="neg")
    r.feed({"event": "rl_island_override", "island_id": 1, "time_unix": 1.0, "negative_test": True,
            "overrides": [{"island": 1, "key": "rl_lr_schedule", "name": "学习率调度",
                           "old": "linear", "new": "constant"}]})
    r.feed({"event": "rl_heartbeat", "island_id": 0, "time_unix": 1.0})
    cards = {c["id"]: c for c in r.overview(now=2.0)["islands"]}
    assert cards["1"]["negative_test"]["label"] == "负例岛：学习率调度 linear→constant"
    assert cards["0"]["negative_test"] is None


def test_page_renders_negative_badge():
    from pathlib import Path

    import yeto.dashboard as d

    js = (Path(d.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert "function negBadge(c)" in js and js.count("negBadge(c)+stBadge(c.status)") == 2


# --- 3.6 resume / merge refusal ---------------------------------------------------------
def test_normal_run_refuses_negative_store(tmp_path):
    store = tmp_path / "store"
    io_.check_store_marker(store, negative=True)
    assert (store / io_.NEGATIVE_MARKER).exists()
    with pytest.raises(ValueError, match="negative-test run"):
        io_.check_store_marker(store, negative=False)
    io_.check_store_marker(store, negative=True)  # negative re-run is fine
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    normal = SimpleNamespace(rl_checkpoint_store=str(store), rl_negative_test_run=False)
    with pytest.raises(ValueError, match="negative-test"):
        io_.check_resume_before_launch(normal, runs_root=runs_root)
    neg = SimpleNamespace(rl_checkpoint_store=str(store), rl_negative_test_run=True)
    io_.check_resume_before_launch(neg, runs_root=runs_root)


def test_normal_run_refuses_store_listed_in_negative_manifest(tmp_path):
    runs_root = tmp_path / "runs"
    (runs_root / "neg").mkdir(parents=True)
    (runs_root / "neg" / "run_manifest.json").write_text(json.dumps(
        {"negative_test": True, "checkpoint_store": "s3://bucket/x", "cluster_prefix": "neg"}))
    args = SimpleNamespace(rl_checkpoint_store="s3://bucket/x", rl_negative_test_run=False)
    with pytest.raises(ValueError, match="'neg'"):
        io_.check_resume_before_launch(args, runs_root=runs_root)
    args.rl_checkpoint_store = "s3://bucket/other"
    io_.check_resume_before_launch(args, runs_root=runs_root, out=io.StringIO())


def test_island_refuses_marked_store(tmp_path):
    store = tmp_path / "store"
    io_.check_store_marker(store, negative=True)
    a = SimpleNamespace(event_tape=None, rl_resume_store=str(store), rl_elastic_checkpoint_store=None)
    with pytest.raises(ValueError, match="negative-test"):
        io_.island_startup(a, 0, {})
    io_.island_startup(a, 0, {io_.NEGATIVE_RUN_ENV: "1"}, sleep=lambda s: pytest.fail("no wait"))


def test_merge_refuses_negative_adapter(tmp_path):
    from yeto.merge import merge_adapter

    run = tmp_path / "run"
    adapter = run / "out" / "adapter"
    adapter.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"negative_test": True}))
    with pytest.raises(ValueError, match="负例运行，禁止导出"):
        merge_adapter(SimpleNamespace(adapter_dir=str(adapter), output_dir=str(tmp_path / "o")))
    (run / "run_manifest.json").write_text(json.dumps({"negative_test": False}))
    (adapter / io_.NEGATIVE_MARKER).write_text("x")
    with pytest.raises(ValueError, match="negative-test run"):
        merge_adapter(SimpleNamespace(adapter_dir=str(adapter), output_dir=str(tmp_path / "o")))
    (adapter / io_.NEGATIVE_MARKER).unlink()
    with pytest.raises(ValueError, match="adapter_config"):
        merge_adapter(SimpleNamespace(adapter_dir=str(adapter), output_dir=str(tmp_path / "o")))


def test_override_env_reaches_ray_workers():
    from yeto.rl.adapters.miles.entry import connect_island_ray

    seen = {}
    ray = SimpleNamespace(init=lambda **kw: seen.update(kw), is_initialized=lambda: False)
    env = {"RAY_ADDRESS": "10.0.0.1:6379", io_.OVERRIDE_ENV: '{"island":1}', io_.NEGATIVE_RUN_ENV: "1"}
    connect_island_ray(environ=env, ray_module=ray)
    assert seen["runtime_env"]["env_vars"][io_.OVERRIDE_ENV] == '{"island":1}'
    assert seen["runtime_env"]["env_vars"][io_.NEGATIVE_RUN_ENV] == "1"
    seen.clear()
    connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.1:6379"}, ray_module=ray)
    assert io_.OVERRIDE_ENV not in seen["runtime_env"]["env_vars"]


# --- 10-09: --rl-max-policy-age needs a tolerant algorithm spec (s18-lpg-age-20261009a) ---
ARU2_M1_SPEC = {"schema": "yeto-rl-algorithm-spec-v2",
                "correction": {"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0},
                "execution": {"max_policy_staleness": 1}}


def _spec_plan(tmp_path, extra, spec=None):
    from test_rl_algorithm_provenance import _launcher_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    if spec is not None:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        extra = (*extra, "--rl-algorithm-spec", str(path))
    args = _launcher_args("ports", extra, gpu="modal:1xh100,modal:1xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan


def test_policy_age_override_refused_with_default_spec(tmp_path):
    args, plan = _spec_plan(tmp_path, ("--rl-negative-test-run", "--rl-island-override",
                                       "1:rl_max_policy_age=1"))
    with pytest.raises(lp.PreflightError, match="max_policy_staleness >= 1"):
        plan(args)
    with pytest.raises(lp.PreflightError, match="tolerates 0"):
        io_.island_args(args, 1, {1: {"rl_max_policy_age": 1}})


def test_policy_age_override_passes_with_aru2_m1_spec(tmp_path):
    args, plan = _spec_plan(tmp_path, ("--rl-negative-test-run", "--rl-island-override",
                                       "1:rl_max_policy_age=1"), spec=ARU2_M1_SPEC)
    p = plan(args)
    assert "--rl-max-policy-age 1" in p["island_requests"][1]["learner_command"]
    assert "--rl-max-policy-age" not in p["island_requests"][0]["learner_command"]


def test_global_policy_age_checked_before_cloud(tmp_path):
    args, _ = _spec_plan(tmp_path, ())
    args.rl_max_policy_age = 1
    with pytest.raises(lp.PreflightError, match="tolerates 0"):
        lp.check_policy_age_spec(args)
    args, _ = _spec_plan(tmp_path, (), spec=ARU2_M1_SPEC)
    args.rl_max_policy_age = 1
    lp.check_policy_age_spec(args)  # passes
    args.rl_max_policy_age = 0
    lp.check_policy_age_spec(args)


def test_launch_refuses_policy_age_without_tolerant_spec_before_cloud(monkeypatch, tmp_path):
    launcher, args, calls, written = _memory_launch(monkeypatch, ("--rl-max-policy-age", "1"),
                                                    seq=12288, resp=6144, tmp_path=tmp_path)
    with pytest.raises(lp.PreflightError, match="max_policy_staleness"):
        launcher.run(args)
    assert "cloud" not in calls and written == []
