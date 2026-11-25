"""Island pool end to end: PoolAvgSync islands (fake engines, IslandDriver) on the
real fragment-round syncer with membership options. CPU, no Ray.

Covers what the retired whole-vector syncer was tested for: a run to the final
cut with pool records and status.json, a late island carried with a discount,
a stale delta refused, a lease expiry followed by an automatic re-JOIN, the
final grace window, and the LEAVE reason on the syncer tape.
"""

from __future__ import annotations

import itertools
import json
import os
import secrets
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from free_ports import free_port as _free_port  # noqa: F401 - imported from here by other tests
from syncer_build import build_syncer
from syncer_port import bound_port
from test_rl_engine_driver import _driver, _engine, _strict_config
from yeto.rl.engine.bridges import PoolAvgSync, RoundCutHooks
from yeto.rl.pool_client import LeaveReason

KEY = f"pool-test-key-{secrets.token_hex(6)}"  # per process: another run's syncer refuses it


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    return build_syncer()  # this process's copy: another process's cargo build never rewrites it


@pytest.fixture(autouse=True)
def _island_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", KEY)


class Syncer:
    def __init__(self, binary: Path, tmp_path: Path, *, learners: int, rounds: int, extra=(),
                 quorum_timeout_s: int = 30):
        self.tape = tmp_path / "syncer" / "tape.jsonl"
        self.tape.parent.mkdir()
        argv = [str(binary), "--port", "0", "--learners", str(learners),
                "--total-steps", str(rounds), "--pipeline", "1", "--grace-ms", "0",
                "--sync-interval-steps", "0", "--delta-correction", "none",
                "--learner-weight", "tokens2-over-steps", "--event-tape", str(self.tape),
                "--quorum-timeout-s", str(quorum_timeout_s), "--final-ack-timeout-s", "30", *extra]
        self.log = (tmp_path / "syncer.log").open("w")
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "YETO_ISLAND_HMAC_KEY": KEY})
        # Islands dial only once the syncer listens: with --quorum 1 an island
        # whose first dial is refused backs off ~2 s, and the other island can
        # run every round (and the final cut) alone in that window. The port is
        # the one this syncer bound (--port 0), never another test's.
        self.port = bound_port(self.proc, log_file=tmp_path / "syncer.log")

    def wait(self, timeout=60.0) -> int:
        try:
            return self.proc.wait(timeout)
        finally:
            self.log.close()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def records(self, kind=None):
        rows = [json.loads(line) for line in self.tape.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if "kind" in r]
        return rows if kind is None else [r for r in rows if r["kind"] == kind]


def _island(tmp_path, syncer, learner_id, rounds, *, delta=1.0, lease_s=30.0):
    engine = _engine(torch.tensor([delta, 2 * delta]))
    config = replace(_strict_config(tmp_path, engine, learner_id=learner_id, rounds=rounds,
                                    tape=f"bridge-{learner_id}.jsonl"),
                     syncer_addr=("127.0.0.1", syncer.port), wan_streams=1)
    sync = PoolAvgSync(config, lease_s=lease_s, base_wait_s=60.0)
    driver = _driver(engine, sync, tmp_path, name=f"island-{learner_id}.jsonl", learner_id=learner_id)
    return sync, driver


def _run(driver, errors):
    def target():
        try:
            driver.run()
        except BaseException as error:  # noqa: BLE001 - reported by the test
            errors.append(error)
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _tape(tmp_path, learner_id):
    return [json.loads(line) for line in (tmp_path / f"island-{learner_id}.jsonl").read_text().splitlines()]


def _run_pool(tmp_path, syncer, islands):
    errors: list[BaseException] = []
    threads = [_run(driver, errors) for _sync, driver in islands]
    for thread in threads:
        thread.join(120)
    assert not any(thread.is_alive() for thread in threads), "an island did not finish"
    assert errors == []
    return syncer.wait()


def test_two_islands_run_to_the_final_cut_with_pool_records(syncer_binary, tmp_path):
    rounds = 3
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds,
                    extra=["--quorum", "1", "--quorum-theta", "1.0", "--max-base-lag", "2",
                           "--island-lease-s", "30", "--final-grace-s", "20"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        assert _run_pool(tmp_path, syncer, islands) == 0
    finally:
        syncer.kill()
    for sync, _driver_ in islands:
        assert sync.base_version == rounds  # each island applied the final cut
    steps = syncer.records("outer_step")
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert all(s["arrived"] == 2 and not s["timed_out"] and s["absent"] == [] for s in steps)
    assert all(abs(sum(w for _i, _b, w in s["weights"]) - 1.0) < 1e-9 for s in steps)
    assert {r["island_id"] for r in syncer.records("pool_join")} == {0, 1}
    assert len(syncer.records("delta_accepted")) == 2 * rounds
    status = json.loads((syncer.tape.parent / "status.json").read_text())
    assert status["schema"] == "yeto.syncer.elastic-status/v1"
    assert status["state"] == "finished" and status["outer_version"] == rounds
    for learner_id in range(2):
        tape = _tape(tmp_path, learner_id)
        assert [e for e in tape if e["event"] == "rl_sync_timing"][-1]["outer_sync"] == "elastic"
        assert sum(e["event"] == "rl_local_round" for e in tape) == rounds


def test_late_island_is_carried_with_a_discount_after_the_soft_deadline(syncer_binary, tmp_path):
    rounds = 3
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds,
                    extra=["--quorum", "2", "--max-base-lag", "2", "--soft-deadline-s", "1",
                           "--carry-gamma", "0.5", "--island-lease-s", "60", "--final-grace-s", "5"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        slow = islands[1][0]
        slow.debug_delay_s = lambda: 2.5  # island 1 misses every soft deadline
        assert _run_pool(tmp_path, syncer, islands) == 0
    finally:
        syncer.kill()
    steps = syncer.records("outer_step")
    assert any(s["timed_out"] and s["absent"] == [1] for s in steps), steps
    carried = syncer.records("delta_carried_over")
    assert carried and all(c["island_id"] == 1 and c["discount"] == 0.5 ** c["lag"] for c in carried)
    assert islands[0][0].base_version == rounds
    # The dashboard folder reads the pool records as before.
    from yeto.rl.engine.merge_metrics import MergeMetricsFolder

    folder = MergeMetricsFolder()
    metrics = [m for m in map(folder.feed, syncer.records()) if m is not None]
    assert len(metrics) == rounds
    assert any(m["merge/carried_weight_share"] > 0 for m in metrics)
    assert any(m["merge/island/0/weight_share"] == 1.0 and m["merge/timed_out"] for m in metrics)


def test_lease_expiry_rejoins_and_the_island_finishes(syncer_binary, tmp_path):
    rounds = 8
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds,
                    extra=["--quorum", "1", "--max-base-lag", "3", "--soft-deadline-s", "2",
                           "--island-lease-s", "1.5", "--final-grace-s", "10"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds, lease_s=1.5) for i in range(2)]
        islands[0][0].debug_delay_s = lambda: 1.0  # keeps the run going while island 1 is silent
        paused = islands[1][0]
        paused.debug_pause = lambda: (1, 4.0)  # after base 1: 4 s without heartbeats or pushes
        assert _run_pool(tmp_path, syncer, islands) == 0
    finally:
        syncer.kill()
    leaves = syncer.records("pool_leave")
    assert any(r["island_id"] == 1 and r["reason"] == "lease_expired" for r in leaves), leaves
    rejoined = [r for r in syncer.records("pool_join") if r["island_id"] == 1 and r["catch_up"]]
    assert rejoined, syncer.records("pool_join")
    tape = _tape(tmp_path, 1)
    assert any(e["event"] == "elastic_rejoin" for e in tape)
    assert any(e["event"] == "elastic_debug_pause" for e in tape)


def test_final_grace_ends_the_run_without_an_island_that_left(syncer_binary, tmp_path):
    rounds = 3
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds,
                    extra=["--quorum", "1", "--max-base-lag", "2", "--island-lease-s", "30",
                           "--soft-deadline-s", "1", "--final-grace-s", "3"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        stayer, quitter = islands[0][0], islands[1][0]
        original, stayer_boundary = quitter.boundary, stayer.boundary
        left = threading.Event()

        def leave_after_first_round(driver, *, rollout_id, stats):
            try:
                original(driver, rollout_id=rollout_id, stats=stats)
                quitter.leave(LeaveReason.PREEMPTED)
            finally:
                left.set()
            raise KeyboardInterrupt("preempted")

        def wait_for_the_quitter(driver, *, rollout_id, stats):
            # With quorum 1 and a base lag of 2 the staying island can run every
            # round before the quitter's first boundary returns; the run would
            # then end before the LEAVE is sent. Hold it after its first round.
            result = stayer_boundary(driver, rollout_id=rollout_id, stats=stats)
            assert left.wait(60), "the quitter never left"
            return result
        quitter.boundary = leave_after_first_round
        stayer.boundary = wait_for_the_quitter
        errors: list[BaseException] = []
        threads = [_run(driver, errors) for _sync, driver in islands]
        for thread in threads:
            thread.join(120)
        assert syncer.wait() == 0
    finally:
        syncer.kill()
    assert islands[0][0].base_version == rounds
    leaves = [r for r in syncer.records("pool_leave") if r["island_id"] == 1]
    assert leaves and leaves[-1]["reason"] == "preempted", syncer.records("pool_leave")
    assert any(isinstance(e, KeyboardInterrupt) for e in errors)


def _goes_silent(sync, *, after_pushes: int, release: threading.Event) -> None:
    """Island ``sync`` pushes ``after_pushes`` times, then holds every later push
    until ``release`` is set while its heartbeats go on (their own thread):
    connected, a member of the pool, and not delivering."""
    pushes = itertools.count()
    sync.debug_delay_s = lambda: 0.0 if next(pushes) < after_pushes else 600.0
    sync._sleep = lambda seconds: release.wait(seconds)


def test_an_island_that_joins_and_stops_pushing_does_not_hold_the_others_merges(syncer_binary, tmp_path):
    """θ = 1.0 (every island answers): island 1 delivers one push, then stays
    connected (heartbeats) without delivering. The round it misses merges island
    0's push at the quorum deadline with island 1 absent, instead of a new attempt
    that captures island 1 again; later rounds do not wait for it."""
    rounds = 5
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds, quorum_timeout_s=2,
                    extra=["--quorum", "1", "--quorum-theta", "1.0", "--max-base-lag", "2",
                           "--island-lease-s", "30", "--final-grace-s", "3"])
    release = threading.Event()
    errors: list[BaseException] = []
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        _goes_silent(islands[1][0], after_pushes=1, release=release)
        threads = [_run(driver, errors) for _sync, driver in islands]
        threads[0].join(90)
        assert not threads[0].is_alive(), "island 0 was held by the silent island"
        assert syncer.wait() == 0
    finally:
        release.set()
        syncer.kill()
    assert islands[0][0].base_version == rounds
    steps = syncer.records("outer_step")
    assert [s["step"] for s in steps] == list(range(1, rounds + 1))
    silent_from = next(i for i, s in enumerate(steps) if s["absent"] == [1])
    assert steps[silent_from]["timed_out"], steps[silent_from]  # merged at the quorum deadline
    assert all(s["absent"] == [] for s in steps[:silent_from])
    later = steps[silent_from + 1:]
    assert later, steps
    # Excused: θ over the islands the round waits for, no deadline wait.
    assert all(s["absent"] == [1] and not s["timed_out"] for s in later), later
    assert all([i for i, _b, _w in s["weights"]] == [0] for s in later)
    assert not [r for r in syncer.records("round_attempt_timeout") if r["next"] == "new_attempt"]
    # The dashboard folder and the SFT dashboard read the deadline merge as a timed-out step.
    from yeto.dashboard.sft import SftState
    from yeto.rl.engine.merge_metrics import MergeMetricsFolder

    folder = MergeMetricsFolder()
    metrics = [m for m in map(folder.feed, syncer.records()) if m is not None]
    assert [m["merge/timed_out"] for m in metrics].count(1) == 1
    assert all(m["merge/island/0/weight_share"] == 1.0 for m in metrics[silent_from:])
    state = SftState()
    for record in syncer.records():
        state.fold(record, record.get("time_unix"))
    assert [row[4] for row in state.outer].count(True) == 1
    assert all(row[3] is not None and row[5] is not None for row in state.outer)
    assert [e[1] for e in state.events].count("timeout") == 1


def test_a_silent_island_that_delivers_again_is_carried_and_waited_for_again(syncer_binary, tmp_path):
    """Island 1 holds one push past the quorum deadline: that round merges without
    it, its push arrives as a late push and is carried into a later round, and the
    rounds after it wait for island 1 again."""
    rounds = 8
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds, quorum_timeout_s=2,
                    extra=["--quorum", "1", "--quorum-theta", "1.0", "--max-base-lag", "4",
                           "--island-lease-s", "30", "--final-grace-s", "10"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        islands[0][0].debug_delay_s = lambda: 0.5  # keeps the run going while island 1 is silent
        pushes = itertools.count()
        islands[1][0].debug_delay_s = lambda: 4.0 if next(pushes) == 1 else 0.0  # its 2nd push only
        assert _run_pool(tmp_path, syncer, islands) == 0
    finally:
        syncer.kill()
    steps = syncer.records("outer_step")
    missed = [s for s in steps if s["absent"] == [1] and s["timed_out"]]
    assert len(missed) == 1, steps  # the deadline merge; later rounds excuse island 1
    # Its held push (from the base of the round it missed) is merged later with a lag.
    carried = [c for c in syncer.records("delta_carried_over") if c["island_id"] == 1]
    assert carried, syncer.records("delta_carried_over")
    late = [s for s in steps if s["step"] > missed[0]["step"]
            and [1, missed[0]["base_version"]] in [[i, b] for i, b, _w in s["weights"]]]
    assert late, steps
    # Delivering again ends the excuse: the rounds after it merge island 1 on time again.
    after = [s for s in steps if s["step"] > late[0]["step"]]
    assert after and all(s["absent"] == [] and 1 in [i for i, _b, _w in s["weights"]] for s in after), after
    assert not [r for r in syncer.records("round_attempt_timeout") if r["next"] == "new_attempt"]


@pytest.mark.parametrize("max_base_lag", ["0", "2"])
def test_a_silent_island_that_comes_back_slower_than_the_others_is_waited_for_again(syncer_binary, tmp_path,
                                                                                     max_base_lag):
    """Island 1 misses one deadline, then keeps pushing a little later than island 0. The
    other island's merges keep their pace: no round but the first missed one (and at most
    one second chance after a refused push) waits for the quorum deadline, and no round
    is retried. With carry (--max-base-lag 2) island 1's late push is carried, which ends
    its excuse, and the later rounds wait for it and merge it again."""
    rounds, timeout_s = 16, 2  # island 0 runs ~6 rounds while island 1 is held
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds, quorum_timeout_s=timeout_s,
                    extra=["--quorum", "1", "--quorum-theta", "1.0", "--max-base-lag", max_base_lag,
                           "--island-lease-s", "30", "--final-grace-s", "10"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        islands[0][0].debug_delay_s = lambda: 0.3
        pushes = itertools.count()
        islands[1][0].debug_delay_s = lambda: 4.0 if next(pushes) == 1 else 0.8  # slower than island 0
        assert _run_pool(tmp_path, syncer, islands) == 0
    finally:
        syncer.kill()
    steps = syncer.records("outer_step")
    assert [s["step"] for s in steps] == list(range(1, rounds + 1))
    timed_out = [s for s in steps if s["timed_out"]]
    assert 1 <= len(timed_out) <= 2 and timed_out[0]["absent"] == [1], steps
    assert all(s["step_seconds"] < timeout_s for s in steps if not s["timed_out"]), steps
    assert not [r for r in syncer.records("round_attempt_timeout") if r["next"] == "new_attempt"]
    if max_base_lag != "0":
        tail = steps[-5:-1]  # the last round may end on the final cut without the slower island
        assert all(s["absent"] == [] and 1 in [i for i, _b, _w in s["weights"]] for s in tail), tail


def test_a_push_merged_with_a_lag_ends_the_excuse_of_an_island_slower_than_the_others(syncer_binary, tmp_path):
    """--max-base-lag 2: island 1's held push and every later one arrive after island 0's
    and answer the open round from an older base: they are merged with a lag, never
    refused, so a merged push is what ends its excuse; the rounds after it wait for
    island 1 again (absent []), they do not keep merging without it."""
    rounds = 12
    syncer = Syncer(syncer_binary, tmp_path, learners=2, rounds=rounds, quorum_timeout_s=2,
                    extra=["--quorum", "1", "--quorum-theta", "1.0", "--max-base-lag", "2",
                           "--island-lease-s", "30", "--final-grace-s", "10"])
    try:
        islands = [_island(tmp_path, syncer, i, rounds) for i in range(2)]
        islands[0][0].debug_delay_s = lambda: 1.0
        pushes = itertools.count()
        islands[1][0].debug_delay_s = lambda: 3.0 if next(pushes) == 1 else 1.3
        assert _run_pool(tmp_path, syncer, islands) == 0
    finally:
        syncer.kill()
    steps = syncer.records("outer_step")
    assert not syncer.records("delta_rejected"), syncer.records("delta_rejected")
    assert syncer.records("delta_carried_over")
    timed_out = [s for s in steps if s["timed_out"]]
    assert len(timed_out) == 1 and timed_out[0]["absent"] == [1], steps
    tail = steps[-4:-1]
    assert all(s["absent"] == [] and 1 in [i for i, _b, _w in s["weights"]] for s in tail), tail


def test_pool_islands_take_round_cuts():
    """R-2: an island in a pool writes round cuts like a strict synced island."""
    assert issubclass(PoolAvgSync, RoundCutHooks)
