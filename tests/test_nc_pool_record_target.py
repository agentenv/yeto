"""NC pool: when the pool lowers its own target (a release by a person it detected), it
writes the control file from the file as it is then, not from what it read at the start
of the tick: a request a person wrote meanwhile (target, retire tickets) and the time of
the last manual change are kept."""

from __future__ import annotations

import subprocess

import pytest

from tests.test_nc_elastic import _run
from tests.test_nc_pool_scale_drain import _elastic, _events, _ready
from yeto.launch.nc import pool_run
from yeto.launch.pool.island_pool import GONE, READY


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4,5")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


def test_a_removal_written_while_the_pool_detects_a_console_release_is_not_overwritten(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=3, pool_kw={"scale_down": "fence"})
    released, removed, kept = _ready(pool)
    orders = len(cloud.kinds("order"))
    pool.watch_parallel = 1  # every ticket is read inside the member's own watch step
    cloud.tickets[released.ticket].update(state="ended", reason="released")  # deleted in the NC console
    poll, asked = cloud.poll, []

    def poll_while_a_person_scales(m):
        # the tick read the control file already; a person removes another node while the
        # status call of the released one runs
        if m is released and not asked:
            asked.append(pool_run.scale_pool(cfg, path, remove=[removed.ticket]))
        return poll(m)

    cloud.poll = poll_while_a_person_scales
    pool.tick()
    assert asked == [0] and released.state == GONE
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 1 and doc["retire"] == [removed.ticket] and doc.get("manual_at") is not None
    _run(pool, clock, clock.t + 600)
    assert removed.state == GONE and kept.state == READY and pool.config.target == 1
    assert [e["ticket"] for e in _events(pool, "scaled_down")] == [released.ticket, removed.ticket]
    assert len(cloud.kinds("order")) == orders and pool_run.read_scale(cfg)["target"] == 1


def test_a_detected_release_keeps_the_manual_hold_against_automatic_scaling(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, elastic={"manual_hold_s": 600})
    assert pool_run.scale_pool(cfg, path, add=1) == 0  # a person
    _run(pool, clock, clock.t + 1800)
    gone = _ready(pool)[0]
    assert len(_ready(pool)) == 3
    cloud.tickets[gone.ticket].update(state="ended", reason="released")
    pool.tick()
    pool.save(pool.state_path)
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 2 and doc["source"] == "manual_release" and doc.get("manual_at") is not None
    assert pool_run.scale_pool(cfg, path, 3, source="auto") == 2  # still inside the manual hold
    assert "takes priority over automatic scaling" in capsys.readouterr().out
