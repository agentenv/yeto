"""fleet-dashboard 1.2/1.3/2.1/2.2: a tape written by the (fake-engine) ports
driver with the new events, read back by the dashboard reducer."""

from __future__ import annotations

import dataclasses
import sys
import types

from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

from test_rl_driver_profiles import _driver, _engine

MB = 2**20


def _fake_pynvml():
    nv = types.ModuleType("pynvml")
    nv.nvmlInit = lambda: None
    nv.nvmlDeviceGetCount = lambda: 1
    nv.nvmlDeviceGetHandleByIndex = lambda i: i
    nv.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=40 * MB, total=80 * MB)
    nv.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=70)
    nv.nvmlDeviceGetPowerUsage = lambda h: 250_000
    nv.nvmlDeviceGetUUID = lambda h: "GPU-x"
    return nv


def test_driver_tape_with_new_events_feeds_the_reducer(tmp_path, monkeypatch):
    engine = _engine()
    original = engine.rollout.generate

    def generate(*a, **kw):
        import time
        time.sleep(0.03)  # let the periodic threads tick
        return dataclasses.replace(original(*a, **kw),
                                   batch_summary={"resp_len_mean": 12.5, "truncated_frac": 0.0})

    engine.rollout.generate = generate
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    driver, _ = _driver(engine, tmp_path, name="rl-island-0.jsonl")
    driver.heartbeat_interval_s = 0.005
    driver.resource_sample_interval_s = 0.005
    driver.run()

    r = Reducer(run="dash-events")
    load_all(r, [str(tmp_path)])
    (card,) = r.overview()["islands"]
    assert card["heartbeat_seen"] is True and card["round"] == 3
    assert card["resource_available"] is True and card["gpu_util_pct"] == 70
    assert card["mem_pct"] == 50
    series = r.overview()["series"]["0"]
    assert [p[1] for p in series["resp_len"]][-1] == 12.5
    assert series["tok_s"]


def test_driver_tape_without_nvml_shows_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    driver, _ = _driver(_engine(), tmp_path, name="rl-island-0.jsonl")
    driver.resource_sample_interval_s = 0.005
    driver.run()
    r = Reducer(run="dash-events")
    load_all(r, [str(tmp_path)])
    (card,) = r.overview()["islands"]
    assert card["resource_available"] is False and card["gpu_util_pct"] is None
    assert card["heartbeat_seen"] is False  # heartbeat off: old-tape view
