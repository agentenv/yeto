"""rl-spot-cost-saving 4.1 (minimal): timed in-flight save on a reclaim notice."""

from __future__ import annotations

import json
import signal as _signal
import threading
import time

import pytest

from yeto.cloud import preemption as p
from yeto.rl.adapters.miles import inflight_save as s

AGENTIC = {"trajectory_id": "t1", "group_id": "agentic:t1", "session_ref": "codex-suspended:t1"}
BUFFERED = {"trajectory_id": "s3", "group_id": "g1", "engine_state": {"miles_sample": {}}}


def test_write_in_flight_is_atomic_and_counts_entries(tmp_path):
    commits = []
    out = s.write_in_flight({"entries": [AGENTIC, BUFFERED], "buffer_groups": 1}, tmp_path / "d",
                            "reclaim-1", commit=lambda: commits.append(1))
    doc = json.loads((tmp_path / "d" / "reclaim-1.json").read_text())
    assert doc["schema"] == "yeto.inflight_save/v1" and len(doc["entries"]) == 2
    assert out["entries"] == 2 and out["agentic_entries"] == 1 and out["buffered_entries"] == 1
    assert commits == [1] and out["commit_error"] is None and out["commit_s"] is not None
    assert not list((tmp_path / "d").glob(".*.tmp"))


def test_commit_failure_is_reported_not_raised(tmp_path):
    def bad():
        raise RuntimeError("no volume")

    out = s.write_in_flight({"entries": []}, tmp_path, "x", commit=bad)
    assert out["commit_error"].startswith("RuntimeError") and (tmp_path / "x.json").exists()


def test_rehearsal_sets_last_save_and_emits(tmp_path):
    emitted, lines = [], []
    saver = s.InFlightSaver(lambda t: {"entries": [AGENTIC]}, str(tmp_path),
                            emit=lambda e, **f: emitted.append((e, f)), printer=lines.append)
    assert saver.last_save_s is None
    out = saver.rehearse(3)
    assert out["kind"] == "rehearsal" and out["round"] == 3 and saver.last_save_s == out["total_s"]
    assert (tmp_path / "rehearsal-r3.json").exists()
    assert emitted[0][0] == s.EVENT and lines[0].startswith("[yeto] rl_inflight_save {")


def test_rehearsal_failure_never_raises(tmp_path):
    def broken(t):
        raise RuntimeError("actor gone")

    lines = []
    saver = s.InFlightSaver(broken, str(tmp_path), printer=lines.append)
    assert saver.rehearse(0) is None and saver.last_save_s is None
    assert "actor gone" in lines[0]


def test_emit_wrapper_rehearses_after_each_trained_round(tmp_path):
    class Driver:
        def __init__(self):
            self.events = []

        def emit(self, event, **fields):
            self.events.append(event)

    d = Driver()
    saver = s.InFlightSaver(lambda t: {"entries": []}, str(tmp_path), printer=lambda l: None)
    s.wrap_emit_with_rehearsal(d, saver)
    d.emit("rl_rollout_done", rollout_id=0)
    d.emit("rl_round_trained", rollout_id=0)
    assert [r["round"] for r in saver.results] == [0]
    assert "rl_inflight_save" in d.events and d.events[:2] == ["rl_rollout_done", "rl_round_trained"]


def test_reclaim_without_measurement_skips_save(tmp_path):
    saver = s.InFlightSaver(lambda t: {"entries": [AGENTIC]}, str(tmp_path), printer=lambda l: None)
    events = []
    h = p.ModalExitHandler("0", save=saver.save, leave=lambda: None,
                           emit=lambda e, **f: events.append(f), last_save_s=lambda: saver.last_save_s)
    ev = h.handle()
    assert ev["outcome"] == p.SKIP_NO_MEASUREMENT and not list(tmp_path.glob("reclaim-*.json"))


def test_reclaim_after_rehearsal_saves_within_cap(tmp_path):
    saver = s.InFlightSaver(lambda t: {"entries": [AGENTIC, BUFFERED], "buffer_groups": 1},
                            str(tmp_path), printer=lambda l: None)
    saver.rehearse(1)
    events = []
    h = p.ModalExitHandler("0", save=saver.save, leave=lambda: None,
                           emit=lambda e, **f: events.append((e, f)), last_save_s=lambda: saver.last_save_s)
    ev = h.handle()
    assert ev["saved"] is True and ev["outcome"] == "saved" and ev["handler_s"] <= p.MODAL_HANDLER_CAP_S
    assert ev["leave_confirmed"] is None and ev["dropped_trajectories"] == 0
    files = list(tmp_path.glob("reclaim-*.json"))
    assert len(files) == 1 and len(json.loads(files[0].read_text())["entries"]) == 2


def test_overrunning_save_is_abandoned_without_half_file(tmp_path):
    release = threading.Event()

    def slow(t):
        release.wait(5)
        return {"entries": [AGENTIC]}

    saver = s.InFlightSaver(slow, str(tmp_path), printer=lambda l: None)
    saver.last_save_s = 0.01
    h = p.ModalExitHandler("0", save=saver.save, leave=lambda: None, emit=lambda e, **f: None,
                           last_save_s=lambda: saver.last_save_s, cap_s=5.3, margin_s=5.0)
    t0 = time.monotonic()
    ev = h.handle()
    release.set()
    assert ev["outcome"] == "save timed out" and not ev["saved"] and time.monotonic() - t0 < 2
    assert not list(tmp_path.glob("reclaim-*.json"))


def test_install_is_opt_in(tmp_path):
    assert s.install(object(), object(), island="0", environ={}) is None


def test_run_island_script_routes_inflight_env(monkeypatch):
    from yeto import modal_runner as mr

    seen = []
    monkeypatch.setattr(mr, "_run_forwarding_to_learner", lambda cmd, env, **kw: seen.append(env) or 0)

    class Sig:
        SIGINT, SIGTERM = _signal.SIGINT, _signal.SIGTERM

        @staticmethod
        def signal(num, fn):
            raise AssertionError("not expected")

    env = {"YETO_SPOT_INFLIGHT_SAVE_DIR": "/x"}
    assert mr.run_island_script(["x"], env, popen=lambda c, env, **k: None, signal_mod=Sig) == 0
    assert seen == [env]


@pytest.mark.parametrize("env", [{}, {"YETO_SPOT_INFLIGHT_SAVE_DIR": ""}])
def test_enabled(env):
    assert s.enabled(env) is False
