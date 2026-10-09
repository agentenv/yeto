"""fleet-dashboard 8.4: startup heartbeat / resource sampling and sub-step events."""

import threading

import pytest

from yeto.dashboard.reducer import Reducer
from yeto.rl.engine import telemetry as T

from dashboard_helpers import T0


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _stepping_wait():
    """Wait stub: each period returns immediately until stop() (drives ticks deterministically)."""
    gate = threading.Semaphore(0)

    def wait(event, timeout):
        gate.acquire(timeout=2)
        return event.is_set()
    return wait, gate


def test_startup_steps_events_and_heartbeat_phase():
    events, clock = [], _Clock()
    wait, gate = _stepping_wait()

    class NoNvml(Exception):
        pass

    def no_nvml():
        raise NoNvml("no gpu here")

    st = T.StartupTelemetry(lambda e, **f: events.append({"event": e, **f}), heartbeat_interval_s=30,
                            resource_interval_s=60, clock=clock, nvml_loader=no_nvml, wait=wait,
                            labels=lambda: {"island_label": "x"})
    with st:
        gate.release()
        for _ in range(50):
            if any(e["event"] == T.HEARTBEAT_EVENT for e in events):
                break
            threading.Event().wait(0.01)
        clock.t = 160.0
        st.step("ray_connected")
        clock.t = 400.0
        st.step("engine_ready")
        st.step("weights_loaded", note="ok")
        gate.release()
    hb = [e for e in events if e["event"] == T.HEARTBEAT_EVENT]
    assert hb and hb[0]["phase"] == "startup" and hb[0]["startup_step"] == "begin" and hb[0]["rollout_id"] is None
    res = [e for e in events if e["event"] == T.RESOURCE_EVENT]
    assert res == [{"event": T.RESOURCE_EVENT, "available": False, "t": 100.0, "reason": "NoNvml: no gpu here",
                    "phase": "startup", "island_label": "x"}]
    steps = [e for e in events if e["event"] == T.STARTUP_STEP_EVENT]
    assert [(e["step"], e["seconds"], e["step_s"]) for e in steps] == [
        ("ray_connected", 60.0, 60.0), ("engine_ready", 300.0, 240.0), ("weights_loaded", 300.0, 0.0)]
    assert steps[-1]["note"] == "ok" and steps[0]["island_label"] == "x"
    with pytest.raises(ValueError):
        st.step("bogus")


def test_startup_telemetry_off_by_default_writes_nothing():
    events = []
    with T.StartupTelemetry(lambda e, **f: events.append(e)) as st:
        st.step("ray_connected")
    assert events == [] and st.steps[0][0] == "ray_connected"


def _reducer():
    r = Reducer()
    r.feed({"event": "island_ready", "island": "run-a-l0-modal", "island_id": 0, "cloud": "modal",
            "gpu": "H200", "gpus": 8, "time_unix": T0})
    r.feed({"event": "rl_heartbeat", "island_id": 0, "phase": "startup", "startup_step": "begin",
            "rollout_id": None, "time_unix": T0 + 30})
    r.feed({"event": "rl_startup_step", "island_id": 0, "step": "ray_connected", "seconds": 60.0,
            "step_s": 60.0, "time_unix": T0 + 60})
    return r


def test_reducer_tracks_startup_steps_and_stays_starting():
    r = _reducer()
    card = r.overview(now=T0 + 120)["islands"][0]
    assert card["starting"] and card["status"] == "starting" and not card["driver_started"]
    assert card["startup_steps"] == {"ray_connected": {"seconds": 60.0, "step_s": 60.0}}
    assert card["startup_step"] == "ray_connected" and card["startup_step_age_s"] == 60.0
    r.feed({"event": "rl_startup_step", "island_id": 0, "step": "engine_ready", "seconds": 400.0,
            "step_s": 340.0, "time_unix": T0 + 400})
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 900})
    card = r.overview(now=T0 + 950)["islands"][0]
    assert not card["starting"] and card["startup_step_age_s"] is None
    assert list(card["startup_steps"]) == ["ray_connected", "engine_ready"]


def test_slow_startup_step_alert_while_heartbeats_arrive():
    r = _reducer()
    for k in range(1, 40):  # process alive: startup heartbeats every 30 s, step stuck
        r.feed({"event": "rl_heartbeat", "island_id": 0, "phase": "startup", "startup_step": "ray_connected",
                "time_unix": T0 + 60 + 30 * k})
    ov = r.overview(now=T0 + 60 + 30 * 39)
    al = [a for a in ov["alerts"] if a["rule"] == "startup_step"]
    assert len(al) == 1 and al[0]["sev"] == 1 and "engine_ready" in al[0]["detail"]
    assert not [a for a in ov["alerts"] if a["rule"] in ("heartbeat", "startup")]
    early = r.overview(now=T0 + 60 + 600)
    assert not [a for a in early["alerts"] if a["rule"] == "startup_step"]
