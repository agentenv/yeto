"""S26 E3b: island link probe, its island hook, and the launcher's relaunch of
an island whose probe rejected the container's link."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from yeto.wan_probe import EXIT_SLOW_LINK, ProbeResult, probe_link


def _fake_stream(up_s: float, down_s: float):
    def run(addr, upload, download, key, timeout, out):
        out["up"], out["down"] = up_s, down_s
    return run


def test_probe_rates_use_the_slowest_stream_per_direction():
    result = probe_link(("h", 1), streams=4, bytes_per_direction=40_000_000, key=None,
                        stream_fn=_fake_stream(2.0, 0.5))
    assert (result.upload_mbps, result.download_mbps) == (20.0, 80.0)
    assert result.accepted(20.0, 80.0) and not result.accepted(20.1, 0.0)
    assert result.fields()["wan_probe/upload_mbps"] == 20.0


def test_a_failed_stream_rejects_the_link():
    def broken(addr, upload, download, key, timeout, out):
        raise ConnectionError("reset")
    result = probe_link(("h", 1), streams=2, bytes_per_direction=2 << 20, key=None, stream_fn=broken)
    assert result.error == "ConnectionError: reset" and not result.accepted(0.0, 0.0)
    with pytest.raises(ValueError):
        probe_link(("h", 1), streams=0, bytes_per_direction=1, key=None)


def _island_args(tmp_path, **kw):
    return Namespace(syncer="10.0.0.1:29400", wan_streams=8, wan_probe_mib=16,
                     event_tape=str(tmp_path / "tape.jsonl"), learner_id=1, **kw)


def test_island_probe_is_off_by_default_and_exits_75_on_a_slow_link(tmp_path, capsys):
    from yeto.rl.adapters.miles.island_entry import probe_wan_link

    calls = []

    def probe(addr, **kw):
        calls.append((addr, kw["streams"], kw["bytes_per_direction"]))
        return ProbeResult(8, 16 << 20, 16 << 20, 4.0, 1.0)

    probe_wan_link(_island_args(tmp_path), probe=probe)
    assert calls == []
    probe_wan_link(_island_args(tmp_path, wan_probe_min_mbps=2.0, wan_probe_min_down_mbps=10.0), probe=probe)
    with pytest.raises(SystemExit) as stop:
        probe_wan_link(_island_args(tmp_path, wan_probe_min_mbps=8.0), probe=probe)
    assert stop.value.code == EXIT_SLOW_LINK
    assert calls == [(("10.0.0.1", 29400), 8, 16 << 20)] * 2
    records = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines()]
    assert [r["accepted"] for r in records if r["event"] == "rl_wan_probe"] == [True, False]
    assert "[yeto-wan-probe] rejected: up 4.2 MB/s" in capsys.readouterr().out


def test_launcher_flags_reach_the_island_and_bound_the_relaunches():
    from yeto.launch.modal_islands import wan_probe_relaunches
    from yeto.launch.rl_island_task import _wan_transport_flags

    on = Namespace(wan_probe_min_mbps=20.0, wan_probe_min_down_mbps=0.0, wan_probe_mib=128,
                   wan_probe_relaunches=3)
    assert _wan_transport_flags(on) == (" --wan-probe-min-mbps 20.0 --wan-probe-min-down-mbps 0.0"
                                        " --wan-probe-mib 128")
    assert wan_probe_relaunches(on) == 3
    assert wan_probe_relaunches(Namespace(wan_probe_relaunches=3)) == 0  # probe off
    with pytest.raises(ValueError):
        _wan_transport_flags(Namespace(wan_probe_min_mbps=1.0, wan_probe_mib=0))


def test_echo_collector_counts_rejected_probes(tmp_path):
    from yeto.rl.event_echo import PREFIX, TapeCollector

    collector = TapeCollector(tmp_path / "tape.jsonl", fresh=True)
    for accepted, t in ((False, 1), (True, 2), (False, 3)):
        collector.feed(PREFIX + json.dumps({"event": "rl_wan_probe", "accepted": accepted, "island_id": 1, "time_unix": t}) + "\n")
    assert collector.wan_rejects == 2


def _fleet(ops, rejects, relaunches, **kw):
    from test_controller import RUNNING, SYNCER, ImmediateThread

    from yeto.launcher import FleetController

    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(learners={"l0": ("task-l0", 1), "l1": ("task-l1", 2)},
                           syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30,
                           recover_timeout=0, thread_cls=ImmediateThread, fixed_roster=True,
                           no_island_relaunch=True, wan_reject_probe=lambda name: rejects[name],
                           wan_probe_relaunches=relaunches, **kw)


def test_rejected_link_is_relaunched_even_without_island_relaunch():
    from test_controller import FAILED, RUNNING, SUCCEEDED, FakeOps

    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, RUNNING, RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    ops.relaunch_results["l1"] = [7]
    ops.after_relaunch["l1"] = [RUNNING, SUCCEEDED]
    rejects = {"l0": 0, "l1": 1}
    ctl = _fleet(ops, rejects, relaunches=2)
    ctl.run()
    assert ops.relaunch_calls == ["l1"]
    assert ctl.learners["l1"]["wan_relaunches"] == 1 and ctl.learners["l1"]["state"] == "done"


def test_rejected_link_budget_is_bounded():
    from test_controller import FAILED, RUNNING, FakeOps

    from yeto.launcher import FixedRosterIslandAbandoned

    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.status_seq["l1"] = [FAILED]
    rejects = {"l0": 0, "l1": 1}
    ctl = _fleet(ops, rejects, relaunches=0)
    with pytest.raises(FixedRosterIslandAbandoned):
        ctl.run()
    assert ops.relaunch_calls == []


@pytest.mark.parametrize(
    ("up_s", "down_s", "error", "min_up", "min_down", "accepted"),
    [
        (1.0, 1.0, None, 100.0, 100.0, True),    # 100 MB in 1 s each way: exactly at the minimum
        (1.0, 1.0, None, 100.1, 0.0, False),     # upload just below the minimum
        (1.0, 4.0, None, 50.0, 50.0, False),     # fast upload, slow download
        (1.0, 4.0, None, 50.0, 0.0, True),       # download not checked when its minimum is 0
        (0.1, 0.1, "OSError: x", 1.0, 1.0, False),  # any stream error rejects a fast link
    ],
)
def test_slow_container_decision(up_s, down_s, error, min_up, min_down, accepted):
    result = ProbeResult(streams=8, upload_bytes=100_000_000, download_bytes=100_000_000,
                         upload_seconds=up_s, download_seconds=down_s, error=error)
    assert result.accepted(min_up, min_down) is accepted


def test_probe_is_refused_with_the_elastic_syncer():
    from yeto.launch.rl_island_task import _wan_transport_flags

    with pytest.raises(ValueError, match="elastic"):
        _wan_transport_flags(Namespace(wan_probe_min_mbps=20.0, rl_elastic=True))
    assert _wan_transport_flags(Namespace(rl_elastic=True)) == ""  # probe off: no error
