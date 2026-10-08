"""fleet-dashboard INFRA-REQ 1.1-1.3 / 2.1-2.2: training metrics, batch summary,
heartbeat and resource-sample events (CPU only)."""

from __future__ import annotations

import json
import math
import threading
from types import SimpleNamespace

import pytest

from test_rl_driver_profiles import _driver, _engine, _events
from test_rl_miles_adapter_trainer_publish import _rank_actor_draining_state_plugin, handle, trainer
from yeto.rl.engine import telemetry
from yeto.rl.engine.driver import BATCH_SUMMARY_KEYS
from yeto.rl.adapters.miles import rollout_meta_hook as rmh


# ---------------------------------------------------------------- 1.1 step_metrics
def _train_with_losses(losses):
    from yeto.rl.adapters.miles import state_plugin

    state_plugin._STEP_LOSSES.clear()
    for item in losses:
        state_plugin._record_step_losses((item, 0.9, "NORMAL"))
    t = trainer(_rank_actor_draining_state_plugin(), [])
    t.train_step(handle())
    return t


def test_step_metrics_fill_round_means_from_loss_dict():
    t = _train_with_losses([
        {"loss": 0.2, "pg_loss": 0.1, "kl_loss": 0.01, "entropy_loss": 0.5, "entropy": 1.0,
         "ppo_kl": 0.003, "ess_ratio": 0.9, "pg_clipfrac": 0.1},
    ])
    m = t.step_metrics()
    assert m.loss == pytest.approx(0.2) and m.pg_loss == pytest.approx(0.1)
    assert m.mean_kl == pytest.approx(0.01)  # kl_loss preferred over ppo_kl
    assert m.ess_ratio == pytest.approx(0.9)
    assert m.lr == pytest.approx(1e-5) and m.train_step == 1
    assert {"kl_loss", "entropy_loss", "entropy", "ppo_kl"} <= set(t.round_metrics())
    # GRPO default keeps the gated fields unchanged (R0)
    assert t.last_step_losses is None and m.clip_fraction is None


def test_step_metrics_without_kl_key_is_none_and_mean_over_steps():
    from yeto.rl.adapters.miles.trainer import mean_step_metrics

    steps = [{"metrics": {"loss": 1.0, "pg_loss": 2.0}}, {"metrics": {"loss": 3.0, "pg_loss": 4.0}}]
    assert mean_step_metrics(steps) == {"loss": 2.0, "pg_loss": 3.0}
    t = _train_with_losses([{"loss": 1.0, "pg_loss": 2.0}])
    m = t.step_metrics()
    assert m.mean_kl is None and m.ess_ratio is None and m.loss == 1.0
    t.train_step(handle())
    assert t.step_metrics().train_step == 2  # cumulative optimizer steps


# ---------------------------------------------------------------- 1.2 rl_round_trained
def test_round_trained_carries_train_fields_and_old_fields(tmp_path):
    driver, _ = _driver(_engine(), tmp_path)
    driver.run()
    trained = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_round_trained"]
    assert len(trained) == 3
    for e in trained:
        for key in ("trained_groups", "trained_samples", "trained_sample_ids_sha256",
                    "train_step", "train_metrics", "step_seconds", "tok_per_s",
                    *BATCH_SUMMARY_KEYS):
            assert key in e
        assert e["train_metrics"] == {} and e["step_seconds"] >= 0
        assert all(e[k] is None for k in BATCH_SUMMARY_KEYS)  # fake: not reported


# ---------------------------------------------------------------- 1.3 batch summary
def _samples():
    out = []
    for i, (reward, length, status) in enumerate(
            [(0.0, 10, "completed"), (1.0, 20, "truncated"), (1.0, 30, "completed"),
             (0.0, 40, "truncated")]):
        out.append(SimpleNamespace(index=i, group_index=0, rollout_id=3,
                                   status=SimpleNamespace(value=status), remove_sample=False,
                                   metadata=None, reward=reward, response_length=length,
                                   weight_versions=None))
    return out


def test_batch_summary_quantiles_and_truncation():
    s = rmh.batch_summary(SimpleNamespace(), _samples())
    assert s["resp_len_mean"] == 25.0
    assert s["resp_len_p95"] == pytest.approx(38.5)  # numpy linear
    assert s["truncated_frac"] == 0.5
    assert (s["reward_p10"], s["reward_p50"], s["reward_p90"]) == (0.0, 0.5, 1.0)
    assert s["adv_mean"] is None and s["adv_std"] is None  # never inferred
    samples = _samples()
    for x, a in zip(samples, [1.0, -1.0, 1.0, -1.0]):
        x.advantage = a
    s = rmh.batch_summary(SimpleNamespace(), samples)
    assert s["adv_mean"] == 0.0 and s["adv_std"] == 1.0


def test_batch_summary_opt_in_only(monkeypatch):
    monkeypatch.delenv(rmh.BATCH_SUMMARY_ENV, raising=False)
    assert not rmh.batch_summary_enabled(SimpleNamespace())
    assert rmh.batch_summary_enabled(SimpleNamespace(yeto_rl_observe_timeline=True))
    monkeypatch.setenv(rmh.BATCH_SUMMARY_ENV, "1")
    assert rmh.batch_summary_enabled(SimpleNamespace())


def test_round_trained_copies_batch_summary(tmp_path):
    engine = _engine()
    original = engine.rollout.generate

    def generate(*a, **kw):
        import dataclasses
        batch = original(*a, **kw)
        return dataclasses.replace(batch, batch_summary={"truncated_frac": 0.25, "reward_p50": 1.0})

    engine.rollout.generate = generate
    driver, _ = _driver(engine, tmp_path, rounds=1)
    driver.run()
    (e,) = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_round_trained"]
    assert e["truncated_frac"] == 0.25 and e["reward_p50"] == 1.0 and e["adv_mean"] is None


# ---------------------------------------------------------------- 2.1 heartbeat
class _Ticks:
    """Fake wait: returns immediately ``n`` times, then reports stop."""

    def __init__(self, n):
        self.n = n
        self.done = threading.Event()

    def __call__(self, event, timeout):
        if self.n <= 0:
            self.done.set()
            return event.wait()  # until stop()
        self.n -= 1
        return event.is_set()


def test_heartbeat_frequency_and_fields_with_fake_clock():
    events, clock = [], iter(range(0, 10_000, 30))
    ticks = _Ticks(10)
    hb = telemetry.HeartbeatThread(
        lambda name, **f: events.append((name, f)),
        lambda: {"phase": "train", "rollout_id": 4, "policy_version": 4},
        interval_s=30, clock=lambda: float(next(clock)), wait=ticks).start()
    assert ticks.done.wait(5)
    hb.stop()
    assert not hb.alive and len(events) == 10  # 5 min of train / 30 s
    name, f = events[-1]
    assert name == "rl_heartbeat" and f["phase"] == "train" and f["rollout_id"] == 4
    assert f["policy_version"] == 4 and f["uptime_s"] == 300 and isinstance(f["pid"], int)


def test_driver_heartbeat_does_not_block_and_stops(tmp_path):
    driver, _ = _driver(_engine(), tmp_path)
    driver.heartbeat_interval_s = 0.005
    original = driver.trainer.train_step

    def slow(batch):
        import time
        time.sleep(0.05)
        return original(batch)

    driver.trainer.train_step = slow
    driver.run()
    before = threading.active_count()
    beats = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_heartbeat"]
    assert beats and {"phase", "rollout_id", "policy_version", "uptime_s", "pid"} <= set(beats[0])
    assert any(b["phase"] == "train" for b in beats)
    assert not any(t.name == "yeto-heartbeat" for t in threading.enumerate())
    assert threading.active_count() == before


def test_default_driver_emits_no_new_periodic_events(tmp_path):
    driver, _ = _driver(_engine(), tmp_path)
    driver.run()
    kinds = {e["event"] for e in _events(tmp_path / "events.jsonl")}
    assert not kinds & {"rl_heartbeat", "rl_resource_sample"}


# ---------------------------------------------------------------- 2.2 resource sample
MB = 2**20


class _FakeNvml:
    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=(h + 1) * 100 * MB, total=1000 * MB)

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=50 + h)

    def nvmlDeviceGetPowerUsage(self, h):
        return 300_000

    def nvmlDeviceGetUUID(self, h):
        if h == 1:
            raise RuntimeError("unsupported")
        return b"GPU-0"


def test_resource_sampler_with_nvml_and_peaks():
    events = []
    ticks = _Ticks(3)
    s = telemetry.ResourceSampler(lambda n, **f: events.append((n, f)), interval_s=30,
                                  nvml_loader=_FakeNvml, rss=lambda: 4096, wait=ticks).start()
    assert ticks.done.wait(5)
    s.stop()
    assert len(events) == 3
    name, f = events[0]
    assert name == "rl_resource_sample" and f["available"] is True and f["cpu_rss_bytes"] == 4096
    assert f["gpus"][0] == {"index": 0, "uuid": "GPU-0", "util_pct": 50, "mem_used_mb": 100.0,
                            "mem_total_mb": 1000.0, "power_w": 300.0}
    assert f["gpus"][1]["uuid"] is None  # unreadable field -> None
    assert s.take_peaks() == {"peak_gpu_mem_bytes": 200 * MB, "peak_cpu_rss_bytes": 4096}
    assert s.take_peaks() == {}
    from yeto.rl.engine.timeline import validate_load_sample
    assert validate_load_sample({"peak_gpu_mem_bytes": 200, "peak_cpu_rss_bytes": 4096}) == []


def test_resource_sampler_without_nvml_writes_one_record(tmp_path):
    def missing():
        raise ImportError("No module named 'pynvml'")

    events = []
    s = telemetry.ResourceSampler(lambda n, **f: events.append((n, f)), interval_s=0.001,
                                  nvml_loader=missing).start()
    s.stop()
    assert len(events) == 1 and events[0][1]["available"] is False and not s.alive
    # through the driver: exactly one record, training continues
    import sys
    driver, _ = _driver(_engine(), tmp_path)
    driver.resource_sample_interval_s = 0.001
    sys.modules["pynvml"] = None  # import fails -> NVML unavailable
    try:
        driver.run()
    finally:
        del sys.modules["pynvml"]
    samples = [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_resource_sample"]
    assert len(samples) == 1 and samples[0]["available"] is False
    assert [e for e in _events(tmp_path / "events.jsonl") if e["event"] == "rl_round_trained"]


def test_learner_switch_defaults(monkeypatch):
    from yeto.rl.adapters.miles.island_entry import apply_ports_infra_switches

    def run(**kw):
        base = dict(rl_observe_timeline=False, rl_heartbeat_interval=None,
                    rl_resource_sample_interval=None)
        ns = SimpleNamespace(**{**base, **kw})
        miles = SimpleNamespace()
        monkeypatch.setattr("yeto.rl.adapters.miles.elastic_hook.apply_recommend_flags",
                            lambda a, m: None)
        apply_ports_infra_switches(ns, miles, environ={})
        return vars(miles)

    assert "yeto_rl_heartbeat_interval_s" not in run()
    got = run(rl_heartbeat_interval=45.0, rl_resource_sample_interval=0)
    assert got["yeto_rl_heartbeat_interval_s"] == 45.0
    assert "yeto_rl_resource_sample_interval_s" not in got
    on = run(rl_observe_timeline=True)
    assert on["yeto_rl_heartbeat_interval_s"] == 30.0
    assert on["yeto_rl_resource_sample_interval_s"] == 60.0


def test_ports_local_round_reaches_wandb_train_axis():
    from yeto.rl.wandb_rl import event_metrics

    m = event_metrics({"event": "rl_local_round", "train_step": 3, "loss": 0.5,
                       "grad_norm": 1.5, "local_round_id": 3, "reward_mean": 0.2})
    assert m["train/step"] == 3 and m["train/loss"] == 0.5 and m["train/grad_norm"] == 1.5
    assert m["rl/reward_mean"] == 0.2
