"""What the page shows before a run has data: store read failures are visible (status,
alert, global status) instead of a silent "no data"; the Commands box repeats this
dashboard's own arguments; a dashboard started before the run follows its local tapes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yeto.dashboard import cli as dcli
from yeto.dashboard import sources
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.serve import DashboardState
from yeto.dashboard.store_tapes import StoreTapes
from yeto.rl.engine.object_store import DirectoryObjectStore

T0 = 1_790_000_000.0


class Broken:
    uri = "s3://bucket/runs/r"

    def list(self, prefix):
        raise PermissionError("AccessDenied: bad key")


def _sampler_round(t):
    return {"event": "rl_sampler_round", "sampler_id": "s1", "rollout_id": 0, "groups": 4,
            "generate_s": 30.0, "policy_version": 0, "version_lag": 0, "time_unix": t}


def test_failing_store_reaches_the_overview_and_the_alerts():
    reducer = Reducer(run="r")
    state = DashboardState(reducer, [], store=StoreTapes(Broken()))
    state.poll()
    ov = reducer.overview(now=T0)
    assert ov["store"]["failing"] == 1 and "AccessDenied" in ov["store"]["last_error"]
    assert ov["global_status"] == {"level": "bad", "text": "run store unreadable"}
    alert = [a for a in ov["alerts"] if a["rule"] == "store_unreadable"]
    assert alert and alert[0]["sev"] == 1 and "s3://bucket/runs/r" in alert[0]["detail"]
    assert reducer.page_view(now=T0)["overview"]["store"]["errors"] == 1


def test_store_recovers_after_a_clean_poll(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    st = StoreTapes(objects)
    reducer = Reducer(run="r")
    state = DashboardState(reducer, [], store=st)
    st._failed(OSError("transient"))
    assert reducer.overview(now=T0)["store"]["failing"] == 1
    objects.write("tapes/s1/a1/00000000.jsonl", (json.dumps(_sampler_round(T0)) + "\n").encode())
    state.poll()
    status = reducer.overview(now=T0)["store"]
    assert status["failing"] == 0 and status["errors"] == 1 and status["sources"] == 1
    assert status["last_ok_at"] is not None and status["polled"]
    assert not [a for a in reducer.overview(now=T0)["alerts"] if a["rule"] == "store_unreadable"]


def test_no_store_keeps_the_old_overview():
    ov = Reducer(run="r").overview(now=T0)
    assert ov["store"] is None and ov["global_status"]["text"] == "no data"


def _args(**kw):
    base = dict(run="r1", tapes=[], store=None, up_state=None, prices=None, thresholds=None, label=None,
                budget=None, host="127.0.0.1", port=8787, store_poll=10.0, poll=2.0, dashboard_command="serve")
    base.update(kw)
    return argparse.Namespace(**base)


def test_page_commands_repeat_the_real_arguments():
    cmds = dcli.page_commands(_args(port=8797, store="s3://b/runs/r1", budget=50.0, label="r1",
                                    tapes=["/x/events dir"]))
    assert cmds["serve"] == ("yeto dashboard serve --run r1 --store s3://b/runs/r1 --tapes '/x/events dir' "
                             "--label r1 --budget 50 --port 8797")
    assert cmds["export"] == ("yeto dashboard export --run r1 --store s3://b/runs/r1 --tapes '/x/events dir' "
                              "--label r1 --budget 50 -o r1.html")
    assert dcli.page_commands(_args())["serve"] == "yeto dashboard serve --run r1 --port 8787"


def test_reducer_carries_the_commands_into_the_view():
    reducer = dcli._reducer(_args(port=8797))
    assert reducer.overview(now=T0)["commands"]["serve"].endswith("--port 8797")


def test_run_tapes_that_appear_after_start_are_followed(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "r1"
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: run_dir)
    paths = sources.run_sources("r1")
    assert str(run_dir / "events") in paths and str(run_dir / "fleet.jsonl") in paths
    reducer = Reducer(run="r1")
    state = DashboardState(reducer, paths)
    state.poll()
    assert reducer.event_seq == 0
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "events" / "s1.jsonl").write_text(json.dumps(_sampler_round(T0)) + "\n")
    state.poll()
    assert reducer.event_seq == 1


def test_export_of_a_run_with_no_tapes_still_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / "nope")
    monkeypatch.setenv("YETO_STATE_DIR", str(tmp_path / "up"))
    try:
        dcli._paths(_args(dashboard_command="export"))
    except SystemExit as exc:
        assert "no tapes" in str(exc)
    else:
        raise AssertionError("export without any tape must refuse")
    syncer = str(Path(os.path.expanduser("~/yeto-output/yeto-tape.jsonl")))  # followed once it appears
    assert dcli._paths(_args()) == [str(tmp_path / "nope" / "events"), str(tmp_path / "nope" / "fleet.jsonl"),
                                    str(tmp_path / "nope" / "cost-events.jsonl"),
                                    str(tmp_path / "up" / _args().run / "events.jsonl"), syncer]


def test_store_run_name_labels_a_store_only_dashboard():
    assert dcli.store_run_name("s3://bucket/runs/r9") == "r9"
    assert dcli.store_run_name("/data/store/runs/r9/") == "r9"
    assert dcli.store_run_name("s3://bucket/other") is None and dcli.store_run_name(None) is None
    assert dcli._reducer(_args(run=None, store="s3://b/runs/r9")).run_name() == ("r9", False)


def _engine_round(rid, t, *, ok=0, limit=0, aborted=0, retries=None, lost=None, inc="a1"):
    return {"event": "rl_sampler_engine_round", "sampler_id": "s1", "incarnation": inc, "rollout_id": rid,
            "groups": 0, "samples": 0, "dropped_groups": 0, "time_unix": t,
            "agentic": {"trials": 0, "kept": 0, "lost_trials": 0, "retried_trials": 0,
                        "outcomes": {"harness_outcome_ok_total": ok, "harness_outcome_policy_limit_total": limit,
                                     "harness_outcome_aborted_total": aborted},
                        "infra_retries_total": retries or {}, "lost_total": lost or {}}}


def _trial_alerts(*records):
    reducer = Reducer(run="r")
    for i, rec in enumerate(records):
        reducer.feed(rec, source="t", offset=i)
    return [a for a in reducer.overview(now=T0 + 600)["alerts"] if a["rule"] == "trial_fail"]


def test_every_trial_aborted_is_critical():
    # the shape of a real run whose sandbox gateway answered 502: aborted trials, no verdict
    recs = [_engine_round(0, T0, aborted=24, retries={"AgentenvError": 23}),
            _engine_round(1, T0 + 5, aborted=8, retries={"AgentenvError": 31}),
            _engine_round(2, T0 + 10)]
    alerts = _trial_alerts(*recs)
    assert len(alerts) == 1 and alerts[0]["sev"] == 0 and alerts[0]["island"] == "s1"
    assert "32 of 32 trials ended aborted" in alerts[0]["detail"] and "AgentenvError 31" in alerts[0]["detail"]


def test_mostly_aborted_warns_and_healthy_or_few_trials_stay_quiet():
    warn = _trial_alerts(_engine_round(0, T0, ok=3, aborted=7))
    assert len(warn) == 1 and warn[0]["sev"] == 1
    assert not _trial_alerts(_engine_round(0, T0, ok=12, limit=2, aborted=2))
    assert not _trial_alerts(_engine_round(0, T0, aborted=5))  # below trial_fail_min_trials


def test_trial_alert_reads_only_the_current_incarnation():
    recs = [_engine_round(0, T0, aborted=20, inc="a1"),
            {"event": "rl_sampler_start", "sampler_id": "s1", "incarnation": "a2", "time_unix": T0 + 50},
            _engine_round(0, T0 + 60, ok=10, inc="a2")]
    assert not _trial_alerts(*recs)


def _sampler_up(until_s, step=30.0):
    recs = [{"event": "rl_sampler_start", "sampler_id": "s1", "incarnation": "a1", "max_backlog_groups": 16,
             "time_unix": T0}]
    t, rid = T0 + step, 0
    while t <= T0 + until_s:
        recs.append(_engine_round(rid, t, ok=8))
        recs.append({"event": "rl_resource_sample", "sampler_id": "s1", "time_unix": t, "available": True,
                     "gpus": [{"index": 0, "mem_total_mb": 46068.0, "mem_used_mb": 43000.0}]})
        t, rid = t + step, rid + 1
    return recs


def _missing_alerts(recs, now, stop=None):
    reducer = Reducer(run="r")
    for i, rec in enumerate(recs):
        reducer.feed(rec, source="t", offset=i)
    if stop:
        reducer.feed_operator_stop(stop)
    return [a for a in reducer.overview(now=now)["alerts"] if a["rule"] == "trainer_missing"]


def test_samplers_without_a_trainer_warn_then_go_critical():
    assert not _missing_alerts(_sampler_up(900), T0 + 910)  # still within the trainer's startup
    warn = _missing_alerts(_sampler_up(1500), T0 + 1510)
    assert len(warn) == 1 and warn[0]["sev"] == 1 and "nothing trains them" in warn[0]["detail"]
    stop = {"event": "operator_stop", "time_unix": T0 + 60, "cause": "teardown", "marker": "teardown.json (mtime)"}
    crit = _missing_alerts(_sampler_up(2000), T0 + 2010, stop=stop)
    assert len(crit) == 1 and crit[0]["sev"] == 0 and "teardown" in crit[0]["detail"]


def test_no_trainer_alert_once_a_trainer_is_on_the_tapes():
    recs = _sampler_up(2000) + [{"event": "rl_driver_start", "island_id": "trainer", "time_unix": T0 + 1990}]
    assert not _missing_alerts(recs, T0 + 2010)


def _trainer_starting(t0, until, step=30.0):
    """A trainer run up launched after the first group: its start phases and startup
    heartbeats, no rl_driver_start yet (a K3.1 actor init takes about 20 min)."""
    recs = [{"event": "trainer_start_phase", "island_id": "trainer", "phase": "submitted", "time_unix": t0}]
    t, seq = t0 + step, 1
    while t <= until:
        recs.append({"event": "rl_heartbeat", "island_id": "trainer", "phase": "startup", "seq": seq, "time_unix": t})
        t, seq = t + step, seq + 1
    return recs


def test_a_starting_trainer_counts_as_on_the_tapes():
    recs = _sampler_up(2400) + _trainer_starting(T0 + 1300, T0 + 2400)
    assert not _missing_alerts(recs, T0 + 2410)


def test_a_trainer_launch_that_went_silent_or_left_does_not_count():
    silent = _sampler_up(2400) + _trainer_starting(T0 + 300, T0 + 400)  # nothing from it for 33 min
    crit = _missing_alerts(silent, T0 + 2410)
    assert len(crit) == 1 and crit[0]["sev"] == 0
    left = _sampler_up(2400) + _trainer_starting(T0 + 1300, T0 + 1400) + [
        {"event": "island_leave", "island_id": "trainer", "reason": 4, "reason_name": "INFRA_FAILED",
         "error_type": "ImageBuildError", "error": "image build failed", "time_unix": T0 + 1410}]
    assert len(_missing_alerts(left, T0 + 2410)) == 1


OOM = ("Traceback (most recent call last):\n  File \"x.py\", line 1, in f\n"
       "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 47.05 GiB.\n\n"
       "The above exception was the direct cause of the following exception:\n\n"
       "miles.utils.retry_utils.NonRetryableError: All cells failed in this training attempt")


def _trainer_launch(t0, *, leave=True):
    recs = [{"event": "trainer_start_phase", "island_id": 0, "phase": "submitted", "time_unix": t0},
            {"event": "rl_driver_start", "island_id": 0, "time_unix": t0 + 100}]
    if leave:
        recs.append({"event": "island_leave", "island_id": 0, "reason": 4, "reason_name": "INFRA_FAILED",
                     "error_type": "RayTaskError(NonRetryableError)", "error": OOM, "time_unix": t0 + 120})
    return recs


def test_one_record_from_echo_and_store_and_a_restart_reupload_counts_once():
    first = _trainer_launch(T0)
    second = [{"event": "trainer_start_phase", "island_id": 0, "phase": "submitted", "time_unix": T0 + 200},
              {"event": "rl_heartbeat", "island_id": 0, "phase": "startup", "seq": 1, "time_unix": T0 + 230}]
    reducer = Reducer(run="r")
    for i, rec in enumerate(first):  # the launcher's echoed tape
        reducer.feed(rec, source="/runs/r/events/r-l0-modal.jsonl", offset=i)
    for i, rec in enumerate(first):  # the first launch's store upload
        reducer.feed(rec, source="store:trainer/l0-modal-A", offset=i)
    for i, rec in enumerate(first + second):  # the restart uploads its whole tape again
        reducer.feed(rec, source="store:trainer/l0-modal-B", offset=i)
    assert reducer.duplicates == 2 * len(first)
    types = [e["type"] for e in reducer.events]
    assert types.count("island_leave") == 1 and types.count("rl_driver_start") == 1
    card = next(c for c in reducer.overview(now=T0 + 240)["islands"] if c["id"] == "0")
    assert card["status"] == "starting"  # the restart is the current launch, not "left"


def test_trainer_failures_alert_with_the_root_cause():
    reducer = Reducer(run="r")
    for i, rec in enumerate(_trainer_launch(T0)):
        reducer.feed(rec, source="store:trainer/l0-modal-A", offset=i)
    one = [a for a in reducer.overview(now=T0 + 300)["alerts"] if a["rule"] == "trainer_failed"]
    assert len(one) == 1 and one[0]["sev"] == 1
    assert "torch.OutOfMemoryError: CUDA out of memory" in one[0]["detail"] and "INFRA_FAILED" in one[0]["detail"]
    for i, rec in enumerate(_trainer_launch(T0 + 600)):
        reducer.feed(rec, source="store:trainer/l0-modal-B", offset=i)
    loop = [a for a in reducer.overview(now=T0 + 900)["alerts"] if a["rule"] == "trainer_failed"]
    assert len(loop) == 1 and loop[0]["sev"] == 0 and loop[0]["title"] == "trainer 0 keeps failing"
    assert not [a for a in reducer.overview(now=T0 + 8000)["alerts"] if a["rule"] == "trainer_failed"]


def test_a_clean_trainer_leave_is_not_a_failure():
    reducer = Reducer(run="r")
    reducer.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0}, source="s", offset=0)
    reducer.feed({"event": "island_leave", "island_id": 0, "reason_name": "DONE", "time_unix": T0 + 10},
                 source="s", offset=1)
    assert not [a for a in reducer.overview(now=T0 + 20)["alerts"] if a["rule"] == "trainer_failed"]


def test_a_trainer_relaunched_after_its_failures_is_a_warning():
    reducer = Reducer(run="r")
    for i, rec in enumerate(_trainer_launch(T0) + _trainer_launch(T0 + 600)):
        reducer.feed(rec, source=f"store:trainer/l0-modal-{'A' if i < 3 else 'B'}", offset=i)
    later = [{"event": "trainer_start_phase", "island_id": 0, "phase": "submitted", "time_unix": T0 + 900},
             {"event": "rl_heartbeat", "island_id": 0, "phase": "startup", "seq": 1, "time_unix": T0 + 930}]
    for i, rec in enumerate(later):
        reducer.feed(rec, source="store:trainer/l0-modal-C", offset=i)
    alert = [a for a in reducer.overview(now=T0 + 940)["alerts"] if a["rule"] == "trainer_failed"]
    assert len(alert) == 1 and alert[0]["sev"] == 1 and "relaunched" in alert[0]["title"]
    assert "OutOfMemoryError" in alert[0]["detail"]



def test_a_restart_the_echo_delivered_first_is_still_the_current_launch():
    # the echoed tape carries both launches before the store uploads are read
    first = _trainer_launch(T0)
    second = [{"event": "trainer_start_phase", "island_id": 0, "phase": "submitted", "time_unix": T0 + 200},
              {"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 300},
              {"event": "rl_heartbeat", "island_id": 0, "phase": "train", "seq": 1, "time_unix": T0 + 330}]
    reducer = Reducer(run="r")
    for i, rec in enumerate(first + second):
        reducer.feed(rec, source="/runs/r/events/r-l0-modal.jsonl", offset=i)
    for i, rec in enumerate(first):
        reducer.feed(rec, source="store:trainer/l0-modal-A", offset=i)
    for i, rec in enumerate(first + second):  # the restart's upload starts with A's history
        reducer.feed(rec, source="store:trainer/l0-modal-B", offset=i)
    isl = reducer.islands["0"]
    assert isl["launch"] == "l0-modal-B" and isl["launch_ts"] == T0 + 200
    assert reducer.duplicates == len(first) + len(first + second)
    card = next(c for c in reducer.overview(now=T0 + 340)["islands"] if c["id"] == "0")
    assert card["status"] != "left"
    alert = [a for a in reducer.overview(now=T0 + 340)["alerts"] if a["rule"] == "trainer_failed"]
    assert alert and alert[0]["sev"] == 1  # failed earlier, relaunched


def test_a_restart_keeps_the_state_its_own_echoed_records_set():
    first = _trainer_launch(T0)
    second = [{"event": "trainer_start_phase", "island_id": 0, "phase": "submitted", "time_unix": T0 + 200},
              {"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 300},
              {"event": "rl_round_trained", "island_id": 0, "rollout_id": 0, "local_round_id": 0,
               "time_unix": T0 + 330, "reward_mean": 0.4}]
    reducer = Reducer(run="r")
    for i, rec in enumerate(first + second):  # the echo first, then both store uploads
        reducer.feed(rec, source="/runs/r/events/r-l0-modal.jsonl", offset=i)
    for i, rec in enumerate(first):
        reducer.feed(rec, source="store:trainer/l0-modal-A", offset=i)
    for i, rec in enumerate(first + second):
        reducer.feed(rec, source="store:trainer/l0-modal-B", offset=i)
    isl = reducer.islands["0"]
    assert isl["launch"] == "l0-modal-B" and isl["driver_started"]
    card = next(c for c in reducer.overview(now=T0 + 340)["islands"] if c["id"] == "0")
    assert card["status"] == "ok" and not card["starting"]


def _stale_read_records(*, sampler_backlog=17.0, trainer_reads=16):
    recs = [{"event": "rl_sampler_start", "sampler_id": "s1", "incarnation": "a1", "max_backlog_groups": 16,
             "time_unix": T0}]
    for rid in range(9):
        recs.append({"event": "rl_sampler_round", "sampler_id": "s1", "incarnation": "a1", "rollout_id": rid,
                     "groups": 2, "time_unix": T0 + 10 + rid})
    recs.append({"event": "rl_driver_start", "island_id": 0, "time_unix": T0 + 30})
    for k in range(trainer_reads // 8):
        recs.append({"event": "rl_buffer_stale_skipped", "island_id": 0, "rollout_id": k, "stale_groups": 0,
                     "max_policy_age": 12, "time_unix": T0 + 40 + k})
        recs.append({"event": "rl_group_policies", "island_id": 0, "rollout_id": k, "time_unix": T0 + 40 + k,
                     "groups": [[f"g{i}", "yeto:0:x", [0]] for i in range(8)], "ages": {"0": 8}})
    recs.append({"event": "rl_sampler_waiting", "sampler_id": "s1", "incarnation": "a1", "rollout_id": 9,
                 "reason": "backlog_full", "backlog_groups": sampler_backlog, "time_unix": T0 + 60})
    return recs


def test_sampler_blocked_on_a_read_position_the_trainers_passed_warns():
    reducer = Reducer(run="r")
    for i, rec in enumerate(_stale_read_records()):
        reducer.feed(rec, source="t", offset=i)
    ov = reducer.overview(now=T0 + 90)
    alert = [a for a in ov["alerts"] if a["rule"] == "stale_read_position"]
    assert len(alert) == 1 and alert[0]["sev"] == 1 and "counts 17 groups unread (cap 16)" in alert[0]["detail"]
    card = next(c for c in ov["islands"] if c["id"] == "s1")
    assert card["status"] != "done"


def test_no_stale_read_alert_when_the_trainers_are_behind_too():
    reducer = Reducer(run="r")
    for i, rec in enumerate(_stale_read_records(trainer_reads=0)):
        reducer.feed(rec, source="t", offset=i)
    assert not [a for a in reducer.overview(now=T0 + 90)["alerts"] if a["rule"] == "stale_read_position"]


def test_a_new_sampler_incarnation_is_not_done():
    reducer = Reducer(run="r")
    recs = [{"event": "rl_sampler_start", "sampler_id": "s1", "incarnation": "a1", "time_unix": T0},
            {"event": "rl_sampler_round", "sampler_id": "s1", "incarnation": "a1", "rollout_id": 0, "groups": 2,
             "time_unix": T0 + 10},
            {"event": "rl_sampler_stopped", "sampler_id": "s1", "rounds": 1, "time_unix": T0 + 20},
            {"event": "rl_sampler_join", "sampler_id": "s1", "incarnation": "a2", "time_unix": T0 + 100},
            {"event": "rl_sampler_start", "sampler_id": "s1", "incarnation": "a2", "time_unix": T0 + 101},
            {"event": "rl_sampler_waiting", "sampler_id": "s1", "incarnation": "a2", "reason": "backlog_full",
             "backlog_groups": 17, "time_unix": T0 + 130}]
    for i, rec in enumerate(recs):
        reducer.feed(rec, source="t", offset=i)
    card = next(c for c in reducer.overview(now=T0 + 140)["islands"] if c["id"] == "s1")
    assert card["status"] not in ("done", "stopped", "left")


def test_a_release_stuck_uploading_warns_and_is_critical_when_samplers_wait_for_it():
    base = _stale_read_records()
    skip = {"event": "rl_policy_release_skipped", "island_id": 0, "policy_version": 2,
            "reason": "upload_in_flight", "local_version": 2, "time_unix": T0 + 50}
    reducer = Reducer(run="r")
    for i, rec in enumerate(base + [skip]):
        reducer.feed(rec, source="t", offset=i)
    assert not [a for a in reducer.overview(now=T0 + 300)["alerts"] if a["rule"] == "release_stalled"]
    crit = [a for a in reducer.overview(now=T0 + 700)["alerts"] if a["rule"] == "release_stalled"]
    # the sampler's report is older than stale_read_recent_s by now: only the release alert remains
    assert len(crit) == 1 and crit[0]["sev"] == 1 and "v2" in crit[0]["detail"]
    reducer.feed({"event": "rl_sampler_waiting", "sampler_id": "s1", "incarnation": "a1", "rollout_id": 9,
                  "reason": "backlog_full", "backlog_groups": 17, "time_unix": T0 + 690}, source="t", offset=999)
    crit = [a for a in reducer.overview(now=T0 + 700)["alerts"] if a["rule"] == "release_stalled"]
    assert len(crit) == 1 and crit[0]["sev"] == 0 and "nothing is generated" in crit[0]["detail"]
    reducer.feed({"event": "rl_policy_released", "island_id": 0, "policy_version": 1, "buffer_selected": 16,
                  "time_unix": T0 + 710}, source="t", offset=1000)
    assert not [a for a in reducer.overview(now=T0 + 720)["alerts"] if a["rule"] == "release_stalled"]


def test_a_local_round_mean_kl_equal_to_the_train_rollout_kl_is_not_the_ppo_kl():
    trained = {"event": "rl_round_trained", "island_id": 0, "rollout_id": 0, "time_unix": T0 + 1,
               "mean_kl": 0.000487, "train_metrics": {"train_rollout_kl": 0.000487, "entropy": 0.043}}
    local = {"event": "rl_local_round", "island_id": 0, "local_round_id": 1, "time_unix": T0 + 2,
             "mean_kl": 0.000487}
    for order in ((trained, local), (local, trained)):
        reducer = Reducer(run="r")
        for i, rec in enumerate(order):
            reducer.feed(rec, source="t", offset=i)
        s = reducer.series(reducer.islands["0"])
        assert not s.get("kl") and [v for _, v in s["kl_tr"]] == [0.000487]
    reducer = Reducer(run="r")  # a real PPO KL on rl_local_round still shows
    reducer.feed({**local, "mean_kl": 0.02}, source="t", offset=0)
    assert [v for _, v in reducer.series(reducer.islands["0"])["kl"]] == [0.02]
