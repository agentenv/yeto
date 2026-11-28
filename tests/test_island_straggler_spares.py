"""A straggler island that loses a node and waits for a spare (yeto.launch.island_straggler
over yeto.launch.island_spares): its mark is cleared when it leaves running, it is never
reported while it recovers and waits for a spare, and the island back on its spare is judged
from its new launch. Over the fake SSH cluster and the fakes of test_ssh_island_spares."""

from __future__ import annotations

from test_island_straggler import Recorder, ScriptedTape, _join, _step
from test_rl_ssh_resident import Clock, _cards, make_ops, wait_for  # noqa: F401 - _cards: autouse
from test_ssh_island_spares import (
    NAMES,
    Cluster,
    _controller,
    _health,  # noqa: F401 - autouse: the fake health checker
    islands_spec,
    make_spares,
    spare,
)

from yeto.launch import island_lifecycle as lc
from yeto.launch import island_straggler as st
from yeto.launch.fleet import RECOVERING, RUNNING
from yeto.rl.engine.object_store import DirectoryObjectStore

OTHER = 9  # an island of another launch the syncer merges each round


def _fleet(tmp_path):
    cluster, clock = Cluster([*NAMES, "b300-3"]), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    store = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "records"))
    tape, log = ScriptedTape(), Recorder()
    ctl = _controller(ops, recover_timeout=600.0, fleet_log=log,
                      recorder=lc.FleetRecorder(store, {"isl": lc.IslandRef(0, "ssh")}),
                      straggler=st.StragglerWatch(tape, cycles=1, log=lambda m: None))
    ctl.ops.sky_ops.t = 0.0
    ctl.ops.now = lambda: clock.t
    return cluster, ops, spares, ctl, store, tape, log, clock


def test_a_straggler_that_loses_a_node_is_cleared_and_not_reported_while_it_waits_for_a_spare(tmp_path):
    cluster, ops, spares, ctl, store, tape, log, clock = _fleet(tmp_path)
    rec = ctl.learners["isl"]
    step = 0

    def merges(*merged, absent=()):
        nonlocal step
        for _ in range(2):
            step += 1
            tape.pending.append(_step(step, list(merged), fragments=1, absent=absent))

    def poll():
        clock.t += 30
        ctl._poll_recorded(rec)
        ctl._poll_stragglers()

    poll()
    assert rec["state"] == RUNNING
    tape.pending += [_join(0), _join(OTHER)]
    merges(0, OTHER)
    poll()
    assert not log.kinds(st.EVENT)
    merges(OTHER, absent=[0])  # stuck while connected: merged without it
    poll()
    assert [e[1] for e in log.kinds(st.EVENT)] == ["isl"] and store.read(0)["straggler"]["action"] == st.WAIT

    spares.records.claim("b300-3", "isl-other")  # no free spare: the island waits
    cluster.lose(NAMES[1])
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(3):  # its source finds the node lost after 2 unanswered checks
        merges(OTHER, absent=[0])
        poll()
        if rec["state"] != RUNNING:
            break
    # the poll that finds the node lost also clears the mark (it left running), and the
    # merges without it meanwhile never reported it a second time
    assert rec["state"] == RECOVERING
    assert [k for k, _, _ in log.events][1:] == ["island_lost", "island_node_lost", "island_waiting_spare",
                                                 st.CLEARED_EVENT]
    (cleared,) = log.kinds(st.CLEARED_EVENT)
    assert cleared[2]["reason"] == "recovering" and store.read(0)["straggler"] is None
    for _ in range(24):  # 12 min of waiting, past the 600 s recover timeout; the others merge on
        merges(OTHER, absent=[0])
        poll()
    assert rec["state"] == RECOVERING
    assert len(log.kinds(st.EVENT)) == 1 and len(log.kinds(st.CLEARED_EVENT)) == 1  # never reported meanwhile
    assert store.read(0)["straggler"] is None

    spares.records.unclaim("b300-3", "isl-other")  # the spare is free: replaced
    for _ in range(40):
        poll()
        if rec["state"] == RUNNING:
            break
    assert rec["state"] == RUNNING and rec["job_id"] == 1
    assert [f["spare"] for k, _, f in log.events if k == "island_node_replaced"] == ["b300-3"]
    # its new launch catches up on the spare while the syncer lists it absent: a new launch
    # is not counted before its first merged push, for up to its recover timeout (600 s)
    for _ in range(15):  # 7.5 min, 15 merge cycles past N = 1
        merges(OTHER, absent=[0])
        poll()
    assert len(log.kinds(st.EVENT)) == 1 and store.read(0)["straggler"] is None
    tape.pending.append(_join(0))  # caught up, then it pushes: counted as usual from here
    merges(0, OTHER)
    poll()
    assert len(log.kinds(st.EVENT)) == 1 and store.read(0)["state"] == lc.RUNNING
    merges(OTHER, absent=[0])  # stuck again on its spare: judged by this launch's merges
    poll()
    assert len(log.kinds(st.EVENT)) == 2 and store.read(0)["straggler"]["last_merged_step"] == step - 2
    assert "catch-up grace" not in log.kinds(st.EVENT)[1][2]["reason"]
    merges(0, OTHER)
    poll()
    assert [e[2]["reason"] for e in log.kinds(st.CLEARED_EVENT)] == ["recovering", "a push of it is merged again"]


def test_a_spare_s_catch_up_past_the_recover_timeout_is_reported_n_cycles_later(tmp_path):
    cluster, ops, _spares, ctl, _store, tape, log, clock = _fleet(tmp_path)
    rec = ctl.learners["isl"]
    step = 0

    def merges(*merged, absent=()):
        nonlocal step
        step += 1
        tape.pending.append(_step(step, list(merged), fragments=1, absent=absent))

    def poll():
        clock.t += 30
        ctl._poll_recorded(rec)
        ctl._poll_stragglers()

    poll()
    tape.pending += [_join(0), _join(OTHER)]
    merges(0, OTHER)
    poll()
    cluster.lose(NAMES[1])  # a free spare: replaced at once
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(10):  # found lost, then replaced (no merge meanwhile: it ran up to its loss)
        poll()
        if rec["state"] == RUNNING and rec["job_id"] == 1:
            break
    assert rec["state"] == RUNNING and rec["job_id"] == 1
    back = clock.t  # its controller holds the new launch running: its grace starts
    tape.pending.append(_join(0))
    while clock.t + 30 < back + 600.0:  # never pushes: not counted for its recover timeout
        merges(OTHER, absent=[0])
        poll()
    assert not log.kinds(st.EVENT)
    merges(OTHER, absent=[0])
    poll()  # the grace runs out: counted from here
    assert not log.kinds(st.EVENT)
    merges(OTHER, absent=[0])
    poll()  # N = 1 cycle later: reported
    (event,) = log.kinds(st.EVENT)
    assert event[2]["missed_cycles"] == 1 and "catch-up grace" in event[2]["reason"]


def test_the_dashboard_with_a_straggler_that_loses_a_node_and_is_replaced(tmp_path):
    """The fleet events of the case above on the dashboard: the badge goes with the clear,
    the spare events leave the card's straggler and lifecycle state as the controller has them."""
    from test_dashboard_fleet_islands import fleet_reducer
    from test_dashboard_island_straggler import _cards as dash_cards
    from test_dashboard_island_straggler import _smoke

    r = fleet_reducer()
    t, name = r.max_ts, "fleet-qa-l1-modal"

    def fleet(event, dt, **fields):
        r.feed({"event": event, "time_unix": t + dt, "island": name, "island_id": 1, **fields})
        return dash_cards(r)["1"]

    card = fleet(st.EVENT, 10, action=st.WAIT, missed_cycles=2, merges=2, fragments=1, newest_step=40,
                 last_merged_step=38, absent=True)
    assert card["straggler"]["missed_cycles"] == 2
    fleet("island_lost", 20, reason="job FAILED")
    fleet("island_node_lost", 20, node="b300-1", rank=1, reason="power")
    fleet("island_waiting_spare", 20, lost="b300-1", reason="no free spare")
    card = fleet(st.CLEARED_EVENT, 20, missed_cycles=2, reason="relaunched")
    assert card["straggler"] is None and card["status"] not in ("ok", "running")
    out = _smoke(tmp_path / "waiting", r)
    assert "straggler" not in out["texts"] and "straggler" not in out["nodes"]
    fleet("island_node_replaced", 400, lost="b300-1", spare="b300-3", resume_step=7)
    fleet("island_launch", 401, incarnation=1)
    card = fleet("island_ready", 402)
    assert card["straggler"] is None
    # its new launch's catch-up outlasted its recover timeout: reported again, by the new launch
    card = fleet(st.EVENT, 500, action=st.WAIT, missed_cycles=2, merges=2, fragments=1, newest_step=52,
                 last_merged_step=None, absent=True)
    assert card["straggler"]["last_merged_step"] is None
    out = _smoke(tmp_path / "again", r)
    assert out["nodes"].count('<span class="wn">straggler</span>') == 1
    card = fleet(st.CLEARED_EVENT, 520, missed_cycles=0, reason="a push of it is merged again")
    assert card["straggler"] is None and r.island("1")["fleet_state"] == "ready"
