"""Island pool end to end: PoolAvgSync islands (fake engines, IslandDriver) on the
real fragment-round syncer with membership options. CPU, no Ray.

Covers what the retired whole-vector syncer was tested for: a run to the final
cut with pool records and status.json, a late island carried with a discount,
a stale delta refused, a lease expiry followed by an automatic re-JOIN, the
final grace window, and the LEAVE reason on the syncer tape.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_rl_engine_driver import _driver, _engine, _strict_config
from yeto.rl.engine.bridges import PoolAvgSync, RoundCutHooks
from yeto.rl.pool_client import LeaveReason

ROOT = Path(__file__).resolve().parent.parent
KEY = "pool-test-key"


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    assert binary.is_file()
    return binary


@pytest.fixture(autouse=True)
def _island_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", KEY)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Syncer:
    def __init__(self, binary: Path, tmp_path: Path, *, learners: int, rounds: int, extra=()):
        self.port = _free_port()
        self.tape = tmp_path / "syncer" / "tape.jsonl"
        self.tape.parent.mkdir()
        argv = [str(binary), "--port", str(self.port), "--learners", str(learners),
                "--total-steps", str(rounds), "--pipeline", "1", "--grace-ms", "0",
                "--sync-interval-steps", "0", "--delta-correction", "none",
                "--learner-weight", "tokens2-over-steps", "--event-tape", str(self.tape),
                "--quorum-timeout-s", "30", "--final-ack-timeout-s", "30", *extra]
        self.log = (tmp_path / "syncer.log").open("w")
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "YETO_ISLAND_HMAC_KEY": KEY})

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
        quitter = islands[1][0]
        original = quitter.boundary

        def leave_after_first_round(driver, *, rollout_id, stats):
            result = original(driver, rollout_id=rollout_id, stats=stats)
            quitter.leave(LeaveReason.PREEMPTED)
            raise KeyboardInterrupt("preempted")
        quitter.boundary = leave_after_first_round
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


def test_pool_islands_take_round_cuts():
    """R-2: an island in a pool writes round cuts like a strict synced island."""
    assert issubclass(PoolAvgSync, RoundCutHooks)
