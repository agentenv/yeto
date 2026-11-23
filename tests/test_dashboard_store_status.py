"""What the page shows before a run has data: store read failures are visible (status,
alert, global status) instead of a silent "no data"; the Commands box repeats this
dashboard's own arguments; a dashboard started before the run follows its local tapes."""

from __future__ import annotations

import argparse
import json

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
    try:
        dcli._paths(_args(dashboard_command="export"))
    except SystemExit as exc:
        assert "no tapes" in str(exc)
    else:
        raise AssertionError("export without any tape must refuse")
    assert dcli._paths(_args()) == [str(tmp_path / "nope" / "events"), str(tmp_path / "nope" / "fleet.jsonl")]
