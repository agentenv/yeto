"""Two concurrent ``yeto run scale --add`` of one run: the run's ``budget.usd`` check
and the scale it allows are one step, so together they never pass the cap."""

from __future__ import annotations

import json
import threading
import time as real_time

from tests.test_run_up import make_run  # noqa: F401 - the fixture
from yeto.rl import up


def test_two_concurrent_scale_adds_do_not_pass_the_run_budget_together(make_run, monkeypatch, tmp_path):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    scale_file = nc_dir / "pool-s.scale.json"
    scale_file.write_text(json.dumps({"target": 1}))
    runs = [make_run(usd=25, nc_usd_per_h=10), make_run(usd=25, nc_usd_per_h=10)]
    for run in runs:
        run.env["YETO_NC_STATE_DIR"] = str(nc_dir)
        monkeypatch.setattr(run, "spend_usd", lambda: 0.0)
    monkeypatch.setattr(up.Run, "pool_doc", lambda self, i=0: {"members": []})

    def pool_scale(argv, **_kw):  # ``pool-scale --add K``: relative to the target it finds, slowly
        add = int(argv[argv.index("--add") + 1])
        target = json.loads(scale_file.read_text())["target"]
        real_time.sleep(0.4)
        scale_file.write_text(json.dumps({"target": target + add}))
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(up.subprocess, "run", pool_scale)
    rcs: list[int] = []
    first = threading.Thread(target=lambda: rcs.append(runs[0].scale(add=1)))
    first.start()
    real_time.sleep(0.15)  # the first scale is inside pool-scale
    rcs.append(runs[1].scale(add=1))
    first.join()
    # 1 -> 2 nodes is $20 of the $25 cap; a third node ($30) must be refused
    assert sorted(rcs) == [0, 2]
    assert json.loads(scale_file.read_text())["target"] == 2


def test_a_breaker_that_fired_while_the_scale_waited_refuses_it(make_run, monkeypatch, tmp_path):
    """The run's breaker check reads the state as it is under the scale lock, not as it
    was when the command opened the run (a scale can wait minutes behind another)."""
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    (nc_dir / "pool-s.scale.json").write_text(json.dumps({"target": 1}))
    run = make_run(usd=1000, nc_usd_per_h=10)
    run.env["YETO_NC_STATE_DIR"] = str(nc_dir)
    monkeypatch.setattr(run, "spend_usd", lambda: 0.0)
    monkeypatch.setattr(up.Run, "pool_doc", lambda self, i=0: {"members": []})
    called = []
    monkeypatch.setattr(up.subprocess, "run", lambda argv, **_kw: called.append(argv))
    other = up.Run(run.cfg, state_root=run.dir.parent)  # ``run up``'s monitor, another process
    other.state["breaker"] = {"at": up._now(), "why": "budget"}
    other.save()
    assert run.scale(add=1) == 2 and not called


def test_a_scale_lock_that_cannot_be_opened_refuses_cleanly(make_run, monkeypatch):
    run = make_run(usd=1000, nc_usd_per_h=10)
    real_open = open

    def no_lock(path, *a, **kw):
        if str(path).endswith("scale.lock"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", no_lock)
    assert run.scale(add=1) == 2
