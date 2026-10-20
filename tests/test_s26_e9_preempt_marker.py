"""S26 E9.1: the Modal reclaim signal reaches a driver inside a Ray actor through
the reclaim marker file (runner writes it; a watcher thread writes the notice)."""

from __future__ import annotations

import io
import json
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from yeto.rl.engine.resume import PREEMPT_WATCH_THREAD, watch_preempt_marker
from yeto.rl.engine.sampler_resume import RECLAIM_MARKER_ENV


def _controller(tmp_path):
    committed = []
    return SimpleNamespace(checkpoint_store=tmp_path / "store", incarnation={"id": "abc", "index": 2},
                           store=SimpleNamespace(commit=lambda: committed.append(1)), committed=committed)


def _wait(cond, limit=5.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_no_marker_path_no_watcher(tmp_path):
    assert watch_preempt_marker(_controller(tmp_path), environ={}) is None


def test_marker_from_the_runner_becomes_the_store_notice_off_the_main_thread(tmp_path):
    from yeto.modal_runner import _run_forwarding_to_learner

    marker = tmp_path / "reclaim"
    env = {RECLAIM_MARKER_ENV: str(marker)}
    controller = _controller(tmp_path)
    out = io.StringIO()
    holder = {}
    # the driver of a verl island runs in a Ray actor thread: start the watcher off the main thread
    t = threading.Thread(target=lambda: holder.update(w=watch_preempt_marker(
        controller, environ=env, poll_s=0.02, out=out)))
    t.start()
    t.join()
    watcher = holder["w"]
    assert watcher is not None and watcher._thread.name == PREEMPT_WATCH_THREAD
    assert not watcher.fired.is_set()

    # the Modal runner (container main process) gets the signal and writes the marker
    handlers = {}
    sig = SimpleNamespace(SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM,
                          signal=lambda num, fn: handlers.__setitem__(num, fn))

    class Proc:
        def wait(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 0

    assert _run_forwarding_to_learner(["true"], env, popen=lambda cmd, env: Proc(), sig=sig) == 0
    assert marker.exists()
    assert _wait(watcher.fired.is_set)
    notices = tmp_path / "store" / "preempt-notices"
    assert _wait(lambda: any(notices.glob("*.json")))
    record = json.loads(next(notices.glob("*.json")).read_text())
    assert record["signal"] == "SIGTERM (reclaim marker)" and record["incarnation"]["id"] == "abc"
    assert record["saved"] is False and controller.committed
    assert "rl_preempt_notice" in out.getvalue()


def test_unreadable_marker_still_writes_the_notice(tmp_path):
    marker = tmp_path / "reclaim"
    marker.write_text("not json")
    controller = _controller(tmp_path)
    watcher = watch_preempt_marker(controller, environ={RECLAIM_MARKER_ENV: str(marker)}, poll_s=0.02,
                                   out=io.StringIO())
    notices = Path(tmp_path / "store" / "preempt-notices")
    assert _wait(lambda: any(notices.glob("*.json")))
    assert json.loads(next(notices.glob("*.json")).read_text())["signal"] == "unknown (reclaim marker)"
    watcher.stop()


def test_launcher_sets_the_marker_path_on_resumable_modal_islands_only():
    from yeto.launcher import _resume_marker_env

    resumable = SimpleNamespace(rl_checkpoint_store="modal-volume://v/run", rl_elastic=False)
    assert _resume_marker_env(resumable, "modal") == {RECLAIM_MARKER_ENV: "/tmp/yeto-reclaim-requested"}
    assert _resume_marker_env(resumable, "nebius") == {}
    assert _resume_marker_env(SimpleNamespace(rl_checkpoint_store=None), "modal") == {}
    # --rl-elastic keeps its own checkpoint store and LEAVE path
    assert _resume_marker_env(SimpleNamespace(rl_checkpoint_store="modal-volume://v/run", rl_elastic=True),
                              "modal") == {}


def test_runner_turns_the_signal_into_the_marker_when_the_launcher_set_it():
    from yeto import modal_runner

    calls = []
    orig = modal_runner._run_forwarding_to_learner
    try:
        modal_runner._run_forwarding_to_learner = lambda cmd, env, **kw: calls.append(env) or 0
        assert modal_runner.run_island_script(["true"], {RECLAIM_MARKER_ENV: "/tmp/x"},
                                              popen=lambda *a, **k: None) == 0
    finally:
        modal_runner._run_forwarding_to_learner = orig
    assert calls


def test_verl_island_in_an_actor_thread_watches_the_marker(tmp_path, monkeypatch):
    """The verl wiring off the main thread: no signal handler, the marker watch is on."""
    import tests.test_rl_verl_round_cuts as vrc
    from yeto.rl.adapters.verl import round_cuts

    marker = tmp_path / "reclaim"
    monkeypatch.setenv(RECLAIM_MARKER_ENV, str(marker))
    plan = vrc._plan(store=tmp_path / "store", state_dir=tmp_path / "state")
    calls = []
    real = round_cuts.wire_round_cuts

    def wire(driver, resume, **kw):
        kw["install_signals"] = True
        out = {}
        t = threading.Thread(target=lambda: out.update(c=real(driver, resume, **kw)))
        t.start()
        t.join()
        calls.append(resume)
        return out["c"]

    monkeypatch.setattr(round_cuts, "wire_round_cuts", wire)
    driver, _h, _c, resume, tape = vrc._island(tmp_path, plan, name="island")
    wired = vrc._events(tape, "rl_round_cuts_wired")[0]
    assert wired["preempt_signals"] == [] and wired["preempt_marker_watch"] is True
    marker.write_text(json.dumps({"signal": int(signal.SIGINT)}))
    notices = Path(resume.controller.checkpoint_store) / "preempt-notices"
    assert _wait(lambda: any(notices.glob("*.json")))
    assert json.loads(next(notices.glob("*.json")).read_text())["signal"] == "SIGINT (reclaim marker)"
