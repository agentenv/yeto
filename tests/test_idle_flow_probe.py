"""The A5 idle-flow probe script, locally (short idle points)."""

import importlib.util
import socket
import threading
from pathlib import Path


def _load():
    path = Path(__file__).parent.parent / "scripts" / "idle_flow_probe.py"
    spec = importlib.util.spec_from_file_location("idle_flow_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_against_the_listener_reports_alive_flows():
    mod = _load()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ready, stop = threading.Event(), threading.Event()
    t = threading.Thread(target=mod.listen, args=(port, "127.0.0.1"),
                         kwargs={"ready": ready, "stop": stop}, daemon=True)
    t.start()
    assert ready.wait(5)
    out = mod.probe("127.0.0.1", port, (0.05, 0.2))
    stop.set()
    assert [r["alive"] for r in out["results"]] == [True, True]
    assert out["idle_flow_timeout_s"] is None


def test_a_dropped_flow_sets_the_timeout_to_the_last_surviving_point(monkeypatch):
    mod = _load()
    fake = {0.1: True, 0.2: True, 0.3: False}
    monkeypatch.setattr(mod, "_probe_one", lambda h, p, t: {"idle_s": t, "alive": fake[t],
                                                             "error": None})
    assert mod.probe("h", 1, (0.1, 0.2, 0.3))["idle_flow_timeout_s"] == 0.2
