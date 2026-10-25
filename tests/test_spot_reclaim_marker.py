"""rl-spot-cost-saving 4.1 follow-up (S19 s19-agentic5-b-20261010b): reclaim notice as a
marker file watched by a learner thread; log-independent Modal container guard;
near-zero cross-version ratio diagnostics."""

from __future__ import annotations

import json
import math
import signal as _signal
import threading
from types import SimpleNamespace

from yeto.cloud import preemption as p
from yeto.rl.adapters.miles import inflight_save as s

AGENTIC = {"trajectory_id": "t1", "group_id": "agentic:t1", "session_ref": "codex-suspended:t1"}


def test_install_starts_marker_watcher_and_writes_pid(tmp_path, monkeypatch):
    class Driver:
        def emit(self, event, **fields):
            pass

    marker = tmp_path / "marker"
    marker.write_text("stale")
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    env = {s.ENV_DIR: str(tmp_path / "save"), s.ENV_PID_FILE: str(tmp_path / "pid"),
           s.ENV_MARKER: str(marker)}
    h = s.install(Driver(), object(), island="0", environ=env)
    assert h is not None and h.role == "train" and h.marker == str(marker)
    assert started == ["yeto-reclaim-marker"] and not marker.exists()  # stale marker removed
    assert (tmp_path / "pid").read_text().strip().isdigit()


def test_watch_marker_calls_once_when_it_appears(tmp_path):
    marker = tmp_path / "m"
    calls, ticks = [], []

    def sleep(t):
        ticks.append(t)
        if len(ticks) == 3:
            marker.write_text("x")

    s.watch_marker(str(marker), calls.append, sleep=sleep)
    assert calls == [str(marker)] and len(ticks) == 3


def test_marker_in_a_thread_runs_the_reclaim_save(tmp_path):
    """End to end without signals: rehearsal -> marker -> watcher thread -> saved."""
    lines = []
    saver = s.InFlightSaver(lambda t: {"entries": [AGENTIC]}, str(tmp_path / "d"), printer=lines.append)
    saver.rehearse(0)
    h = p.ModalExitHandler("0", save=saver.save, leave=lambda: None, emit=lambda e, **f: None,
                           last_save_s=lambda: saver.last_save_s, role="train")
    marker = tmp_path / "marker"
    th = threading.Thread(target=s.watch_marker, args=(str(marker), lambda path: h.handle()),
                          kwargs={"poll_s": 0.01}, daemon=True)
    th.start()
    marker.write_text("{}")
    th.join(5)
    assert h.event is not None and h.event["outcome"] == "saved"
    steps = [json.loads(line.split(" ", 2)[2])["step"] for line in lines if s.PROGRESS in line]
    assert steps == ["export_start", "export_done", "written"]
    assert list((tmp_path / "d").glob("reclaim-*.json"))


def test_modal_runner_writes_the_marker_and_signals_nobody(tmp_path):
    from yeto import modal_runner as mr

    class P:
        def __init__(self, kw):
            self.kw = kw

        def wait(self):
            return 0

    made, handlers = [], {}

    def popen(cmd, env, **kw):
        made.append(P(kw))
        return made[-1]

    class Sig:
        SIGINT, SIGTERM = _signal.SIGINT, _signal.SIGTERM

        @staticmethod
        def signal(num, fn):
            handlers[num] = fn

    marker = tmp_path / "marker"
    env = {"YETO_SPOT_INFLIGHT_SAVE_DIR": "/yeto-tape/x", "YETO_SPOT_RECLAIM_MARKER": str(marker)}
    assert mr._run_forwarding_to_learner(["x"], env, popen=popen, sig=Sig, clock=lambda: 12.5) == 0
    assert made[-1].kw == {}  # same session, nothing else signalled
    assert not marker.exists()
    handlers[_signal.SIGINT](_signal.SIGINT, None)
    assert json.loads(marker.read_text()) == {"signal": int(_signal.SIGINT), "time": 12.5}


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _poller(lists, expected, claims=None, clock=None):
    from yeto.modal_runner import ContainerSetPoller

    it = iter(lists)
    changes = []
    claims = {} if claims is None else claims
    g = ContainerSetPoller(lambda: next(it), expected, changes.append, attributed=lambda: claims,
                           grace_s=600.0, clock=clock or _Clock())
    return g, changes, claims


def test_container_set_poller_trips_on_a_rescheduled_container():
    g, changes, claims = _poller([set(), {"ta-1"}, {"ta-1"}, None, set(), {"ta-2"}, {"ta-3"}], 1)
    g.poll_once()
    g.poll_once()
    claims["ta-1"] = ("0", "0")  # island 0 printed its container line
    for _ in range(3):
        g.poll_once()
    assert not g.tripped  # gone (killed) is not a change
    g.poll_once()
    assert g.tripped and len(changes) == 1 and "ta-2" in changes[0] and g.expected is False
    assert "an island was rescheduled" in changes[0] and "island 0 rank 0 = ta-1" in changes[0]
    g.poll_once()
    assert len(changes) == 1  # once


def test_container_set_poller_waits_for_all_islands_and_skips_list_errors():
    from yeto.modal_runner import ContainerSetPoller

    def boom():
        raise RuntimeError("cli")

    g = ContainerSetPoller(boom, 2)
    g.poll_once()
    assert not g.tripped
    lists = iter([{"a"}, {"a", "b"}, {"a", "b"}])
    g.list_ids = lambda: next(lists)
    for _ in range(3):
        g.poll_once()
    assert not g.tripped


def test_container_set_poller_late_first_container_is_not_a_reschedule():
    # S27 s27-gr-i: island 0 runs; island 1's input gets a task that never prints its
    # container line, the task leaves, and a later task is island 1's first container.
    g, changes, claims = _poller([{"ta-0"}, {"ta-0", "ta-q"}, {"ta-0", "ta-q"}, {"ta-0"},
                                  {"ta-0", "ta-late"}, {"ta-0", "ta-late"}], 2)
    g.poll_once()
    claims["ta-0"] = ("0", "0")
    for _ in range(4):
        g.poll_once()
    assert not g.tripped
    claims["ta-late"] = ("1", "0")
    g.poll_once()
    assert not g.tripped and changes == []


def test_container_set_poller_two_unclaimed_containers_for_one_free_slot_trip_after_the_grace(capsys):
    clock = _Clock()
    g, changes, claims = _poller([{"ta-0", "ta-a", "ta-b"}] * 3, 2, {"ta-0": ("0", "0")}, clock)
    g.poll_once()
    assert not g.tripped and "WARNING" in capsys.readouterr().out
    clock.t = 600.0
    g.poll_once()
    assert not g.tripped and capsys.readouterr().out == ""  # warned once per container
    clock.t = 601.0
    g.poll_once()
    assert g.tripped and "ta-a, ta-b" in changes[0] and "1 of 2" in changes[0]
    assert "listed for 601 s" in changes[0] and "an island was rescheduled" in changes[0]


def test_container_set_poller_extra_containers_beside_claimed_ones_do_not_trip():
    # S27 s27-ch-d0a: both containers of island 0 claimed and training; two extra
    # containers of the function appear (no island line) and later leave
    clock = _Clock()
    both = {"ta-r0", "ta-r1"}
    lists = [both, both | {"ta-x", "ta-y"}, both | {"ta-x", "ta-y"}, both, both | {"ta-z"}, both | {"ta-z"}]
    g, changes, _ = _poller(lists, 2, {"ta-r0": ("0", "0"), "ta-r1": ("0", "1")}, clock)
    for t in (0.0, 30.0, 500.0, 530.0):
        clock.t = t
        g.poll_once()
    assert not g.tripped and g.extra_since is None  # they left: the grace starts again
    clock.t = 560.0
    g.poll_once()
    clock.t = 1100.0
    g.poll_once()
    assert not g.tripped  # ta-z: 540 s so far


def test_container_set_poller_multi_island_reschedule_after_all_claimed():
    # S27 s27-gr-h: both islands printed their lines; Modal ended island 0's container
    # and a third container appears.
    lists = [{"ta-1"}, {"ta-0", "ta-1"}, {"ta-0", "ta-1"}, {"ta-1", "ta-new"}]
    g, changes, claims = _poller(lists, 2)
    g.poll_once()
    claims["ta-1"] = ("1", "0")
    g.poll_once()
    claims["ta-0"] = ("0", "0")
    g.poll_once()
    assert not g.tripped
    g.poll_once()
    assert g.tripped and "ta-new" in changes[0] and "0 of 2" in changes[0]
    assert "ta-0 no longer listed" in changes[0]


def test_container_set_poller_a_third_container_beside_both_claimed_trips_after_the_grace():
    clock = _Clock()
    lists = [{"ta-0", "ta-1"}, {"ta-0", "ta-1", "ta-new"}, {"ta-0", "ta-1", "ta-new"}]
    g, changes, _ = _poller(lists, 2, {"ta-0": ("0", "0"), "ta-1": ("1", "0")}, clock)
    g.poll_once()
    g.poll_once()
    assert not g.tripped
    clock.t = 601.0
    g.poll_once()
    assert g.tripped and "ta-new" in changes[0] and "0 of 2" in changes[0]


def test_container_set_poller_multi_node_islands_count_every_rank():
    # island 0 has 2 ranks, island 1 has 1: three slots, rank 1 of island 0 is late
    lists = [{"ta-00", "ta-1"}, {"ta-00", "ta-1", "ta-01"}, {"ta-00", "ta-1", "ta-01", "ta-x"},
             {"ta-1", "ta-01", "ta-x"}]
    claims = {"ta-00": ("0", "0"), "ta-1": ("1", "0")}
    g, changes, _ = _poller(lists, 3, claims)
    g.poll_once()
    g.poll_once()
    assert not g.tripped  # ta-01 is the late first container of island 0 rank 1
    g.poll_once()
    assert not g.tripped  # an extra container, every claimed one still listed
    g.poll_once()
    assert g.tripped and "ta-01, ta-x" in changes[0] and "ta-00 no longer listed" in changes[0]


def test_container_set_poller_replacement_of_a_claimed_slot_trips_even_if_old_gone():
    # the S19 case: the old container left, the new one never printed its line
    g, changes, claims = _poller([{"ta-0", "ta-1"}, {"ta-1"}, {"ta-1", "ta-0b"}], 2,
                                 {"ta-0": ("0", "0"), "ta-1": ("1", "0")})
    g.poll_once()
    g.poll_once()
    assert not g.tripped
    g.poll_once()
    assert g.tripped and "ta-0b" in changes[0]


def test_container_set_poller_attribution_error_keeps_slots_free():
    from yeto.modal_runner import ContainerSetPoller

    def bad():
        raise RuntimeError("guards")

    clock = _Clock()
    lists = iter([{"a", "b"}, {"a", "b", "c"}, {"a", "b", "c"}])
    g = ContainerSetPoller(lambda: next(lists), 2, attributed=bad, clock=clock)
    g.poll_once()
    assert not g.tripped
    g.poll_once()
    assert not g.tripped  # nothing claimed, so nothing claimed is lost: the grace applies
    clock.t = 601.0
    g.poll_once()
    assert g.tripped


def test_launcher_attribution_reads_container_id_guard_lines():
    from yeto.modal_runner import ContainerIdGuard, ContainerSetPoller

    guards = [ContainerIdGuard(), ContainerIdGuard()]
    guards[0].feed("[modal-island 0] rank 0 container ta-0\n")
    claims = lambda: {cid: k for g in guards for k, cid in g.first.items()}  # noqa: E731
    lists = iter([{"ta-0", "ta-q"}, {"ta-0", "ta-1"}, {"ta-1", "ta-2"}])
    p = ContainerSetPoller(lambda: next(lists), 2, attributed=claims)
    p.poll_once()
    guards[1].feed("[modal-island 1] rank 0 container ta-1")
    p.poll_once()
    assert not p.tripped
    p.poll_once()
    assert p.tripped and "ta-2" in p.changes[0]


def test_container_set_poller_only_when_no_new_container_is_expected():
    from yeto import launcher

    class Ops:
        def list_container_ids(self):
            return set()

    cfg = SimpleNamespace(num_nodes=1, envs={})
    fixed = SimpleNamespace(no_island_relaunch=True, modal_retries=0, recover_timeout=600)
    poller = launcher.container_set_poller_for(fixed, Ops(), {"l0": cfg}, None, start=False)
    assert poller is not None and poller.expected_count == 1
    relaunch = SimpleNamespace(no_island_relaunch=False, modal_retries=None, recover_timeout=600)
    assert launcher.container_set_poller_for(relaunch, Ops(), {"l0": cfg}, None, start=False) is None
    assert launcher.container_set_poller_for(fixed, None, {"l0": cfg}, None, start=False) is None
    assert launcher.container_set_poller_for(fixed, object(), {"l0": cfg}, None, start=False) is None


def test_list_container_ids_filters_by_app(monkeypatch):
    import subprocess as sp

    from yeto import modal_runner as mr

    rows = [{"container_id": "ta-1", "app_name": "yeto-x"}, {"container_id": "ta-2", "app_name": "other"}]
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, json.dumps(rows), ""))
    assert mr.ModalOps("yeto-x").list_container_ids() == {"ta-1"}
    rows = [{"Container ID": "ta-1", "App Name": "yeto-x", "Start Time": "2026-10-23 18:16:55+00:00"},
            {"Container ID": "ta-q", "App Name": "yeto-x", "Start Time": "Pending"},
            {"Container ID": "ta-n", "App Name": "yeto-x", "Start Time": None}]
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, json.dumps(rows), ""))
    assert mr.ModalOps("yeto-x").list_container_ids() == {"ta-1"}  # queued tasks are left out
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 1, "", "err"))
    assert mr.ModalOps("yeto-x").list_container_ids() is None


def test_near_zero_ratio_examples_are_reported():
    from yeto.rl.adapters.miles import carry_over as co

    sample = SimpleNamespace(tokens=[1, 2, 10, 11, 12], response_length=3, loss_mask=[1, 1, 1],
                             rollout_log_probs=[-0.1, 0.0, -0.2])
    co_versions = [3, 3, 4]
    orig = co.trained_token_versions
    try:
        co.trained_token_versions = lambda smp, cur: list(co_versions)
        out = co.estimate_cross_version_truncation(
            SimpleNamespace(tis_clip=2.0, tis_clip_low=0.0), [sample], 4,
            lambda smp: [-0.1, float("-inf"), -0.2])
    finally:
        co.trained_token_versions = orig
    assert out["cross_version_unscored_samples"] == 0
    assert out["cross_version_ratio_near_zero"] == 1
    ex = out["cross_version_ratio_near_zero_examples"][0]
    assert ex["index"] == 1 and ex["segment_start"] is False and ex["token"] == 11
    assert ex["current_logprob"] == "-inf" and ex["generation_logprob"] == 0.0
    assert out["cross_version_ratio_min"] == 0.0 and math.isclose(out["cross_version_ratio_max"], 1.0)


def test_no_near_zero_keeps_the_key_set():
    from yeto.rl.adapters.miles import carry_over as co

    sample = SimpleNamespace(tokens=[1, 10, 11], response_length=2, loss_mask=[1, 1],
                             rollout_log_probs=[-0.1, -0.2])
    orig = co.trained_token_versions
    try:
        co.trained_token_versions = lambda smp, cur: [3, 4]
        out = co.estimate_cross_version_truncation(SimpleNamespace(), [sample], 4, lambda smp: [-0.1, -0.2])
    finally:
        co.trained_token_versions = orig
    assert "cross_version_ratio_near_zero" not in out and "cross_version_ratio_near_zero_examples" not in out


def test_reclaim_uses_the_export_cached_at_train_start(tmp_path):
    """S19 s19-agentic5-r-20261010c: the rollout actor was gone within 1 s of the
    signal; the reclaim save must not need it while the in-flight set is frozen."""
    calls = []

    def export(t):
        calls.append(t)
        if len(calls) > 1:
            raise RuntimeError("ActorDiedError")
        return {"entries": [AGENTIC]}

    class Driver:
        def emit(self, event, **fields):
            pass

    lines = []
    saver = s.InFlightSaver(export, str(tmp_path), printer=lines.append)
    d = Driver()
    s.wrap_emit_with_rehearsal(d, saver)
    d.emit("rl_driver_phase", phase="train", rollout_id=1)
    assert saver.cached_round == 1 and len(calls) == 1
    out = saver.save(20.0)
    assert out["export_source"] == "cached" and out["cached_round"] == 1 and out["entries"] == 1
    assert len(calls) == 1  # no live export at reclaim
    d.emit("rl_driver_phase", phase="generate", rollout_id=2)
    assert saver.cached is None
    import pytest

    with pytest.raises(RuntimeError):
        saver.save(20.0)  # during a rollout: live export (here: the actor is gone)


def test_rehearsal_never_reads_the_cache(tmp_path):
    exports = []
    saver = s.InFlightSaver(lambda t: exports.append(t) or {"entries": []}, str(tmp_path), printer=lambda l: None)
    saver.cached = {"entries": [AGENTIC]}
    out = saver.rehearse(0)
    assert out["export_source"] == "live" and out["entries"] == 0 and len(exports) == 1


def test_cache_failure_is_printed_and_cleared(tmp_path):
    def boom(t):
        raise RuntimeError("x")

    lines = []
    saver = s.InFlightSaver(boom, str(tmp_path), printer=lines.append)
    saver.cached = {"entries": []}
    saver.refresh_cache(3)
    assert saver.cached is None and "cache_failed" in lines[0]
