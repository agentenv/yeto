"""rl-trainer-start-stop L5.5 / L5.7: merge metrics folded from the elastic
syncer tape, and the tape clock fields. No Ray, no GPU."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time

import pytest

from yeto.rl.engine.merge_metrics import MergeMetricsFolder
from yeto.wandb_tape import record_metrics, tape_metrics


def _fold(records):
    folder = MergeMetricsFolder()
    return [m for m in (folder.feed(r) for r in records) if m is not None]


def test_folder_hand_computed_shares_carry_and_rejections():
    # Round v1: island 1 on time (w = 10^2/1 = 100), island 3 late by one
    # (w = 30^2/1 * 0.5 = 450), island 9 not a member, island 2 unknown version.
    records = [
        {"kind": "delta_carried_over", "island_id": 3, "base": 0, "lag": 1, "discount": 0.5},
        {"kind": "delta_accepted", "island_id": 1},
        {"kind": "delta_rejected", "island_id": 9, "reason": "not_member"},
        {"kind": "delta_rejected", "island_id": 2, "reason": "unknown_version"},
        {"kind": "outer_step", "base_version": 1, "arrived": 1, "timed_out": True,
         "raw_weights": [[1, 1, 100.0], [3, 0, 450.0]],
         "weights": [[1, 1, 100 / 550], [3, 0, 450 / 550]], "absent": [2, 3],
         "rejected": [[9, "not_member"], [2, "unknown_version"]],
         "time_unix": 1000.0, "step_seconds": 4.0, "last_arrival_island": 1},
        {"kind": "delta_accepted", "island_id": 1},
        {"kind": "outer_step", "base_version": 2, "arrived": 1, "timed_out": False,
         "raw_weights": [[1, 2, 100.0]], "weights": [[1, 2, 1.0]], "absent": [], "rejected": []},
    ]
    first, second = _fold(records)
    assert first["global_step"] == 2 and first["merge/base_version"] == 1
    assert first["merge/island/1/weight_share"] == pytest.approx(100 / 550)
    assert first["merge/island/3/weight_share"] == pytest.approx(450 / 550)
    assert first["merge/carried_weight_share"] == pytest.approx(450 / 550)
    assert first["merge/submitted"] == 4
    assert first["merge/carried_fraction"] == 0.25
    assert first["merge/carried_discount_mean"] == 0.5
    assert first["merge/rejected_fraction"] == 0.5
    assert first["merge/rejected_fraction/not_member"] == 0.25
    assert first["merge/rejected_fraction/unknown_version"] == 0.25
    assert first["merge/step_seconds"] == 4.0 and first["merge/last_arrival_island"] == 1
    # The folder starts again after each step; absent values stay absent.
    assert second["merge/submitted"] == 1 and second["merge/rejected_fraction"] == 0.0
    assert second["merge/carried_weight_share"] == 0.0
    assert "merge/carried_discount_mean" not in second and "merge/step_seconds" not in second


def test_old_tape_without_rejected_list_counts_rejection_events():
    records = [
        {"kind": "delta_accepted", "island_id": 1},
        {"kind": "delta_rejected", "island_id": 2, "reason": "carry_lag_exceeded"},
        {"kind": "outer_step", "base_version": 0, "arrived": 1, "timed_out": False,
         "raw_weights": [[1, 0, 1.0]], "weights": [[1, 0, 1.0]], "absent": [2]},
    ]
    (m,) = _fold(records)
    assert m["merge/rejected_fraction/carry_lag_exceeded"] == 0.5


def test_wandb_routes_elastic_records_through_the_folder_only():
    folder = MergeMetricsFolder()
    assert record_metrics({"kind": "pool_join", "island_id": 1}, folder) is None
    assert record_metrics({"kind": "delta_accepted", "island_id": 1}, folder) is None
    step = {"kind": "outer_step", "base_version": 0, "weights": [[1, 0, 1.0]]}
    assert record_metrics(step, folder)["merge/island/1/weight_share"] == 1.0
    strict = {"step": 3, "responders": []}
    assert record_metrics(strict, folder) == tape_metrics(strict)


def _wait_tape(tape, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tape.exists():
            rows = [json.loads(line) for line in tape.read_text().splitlines() if line.strip()]
            if predicate(rows):
                return rows
        time.sleep(0.05)
    raise AssertionError(f"tape condition not met: {tape.read_text() if tape.exists() else 'no tape'}")


@pytest.mark.skipif(not os.environ.get("YETO_TEST_ELASTIC_SYNCER"),
                    reason="needs YETO_TEST_ELASTIC_SYNCER (syncer binary with elastic mode)")
def test_real_syncer_tape_merge_shares_and_clock(tmp_path):
    """Real elastic syncer, two client islands (theta 0.5: one arrival steps).

    v0: island 1 alone -> share 1.0. Island 2 then sends on base 0 (lag 1,
    discount 0.5, w = 20^2/2 * 0.5 = 100); island 1 on base 1 (w = 100) ->
    v1 shares 0.5 / 0.5. Island 2 sends on base 7 (unknown_version), then on
    base 2 -> v2 rejected fraction 1/2."""
    from yeto.rl.elastic_client import ElasticClientConfig, ElasticIslandClient

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = "merge-metrics"
    tape = tmp_path / "syncer-tape.jsonl"
    syncer = subprocess.Popen(
        [os.environ["YETO_TEST_ELASTIC_SYNCER"], "--port", str(port), "--learners", "2",
         "--total-steps", "5", "--event-tape", str(tape),
         "--island-scheduling-mode", "elastic", "--quorum-theta", "0.5", "--carry-gamma", "0.5",
         "--soft-deadline-s", "60", "--q-min", "1", "--max-carry-lag", "2",
         "--island-lease-s", "6", "--syncer-epoch", "1"],
        env=dict(os.environ, YETO_ISLAND_HMAC_KEY=key),
        stdout=open(tmp_path / "syncer.log", "w"), stderr=subprocess.STDOUT)
    islands = {i: ElasticIslandClient(ElasticClientConfig(("127.0.0.1", port), i, lease_s=6.0,
                                                          syncer_epoch=1), key.encode())
               for i in (1, 2)}
    started = time.time()
    try:
        for c in islands.values():
            for _ in range(50):
                try:
                    c.join()
                    break
                except OSError:
                    time.sleep(0.1)
        islands[1].elastic_init([0.0] * 4)
        assert islands[1].wait_base(newer_than=None, timeout_s=10.0) is not None
        update = [0.1] * 4
        islands[1].delta_tensor(base_version=0, c_tokens=10, c_steps=1, update=update)
        assert islands[2].wait_base(newer_than=0, timeout_s=10.0) is not None
        islands[2].delta_tensor(base_version=0, c_tokens=20, c_steps=2, update=update)
        _wait_tape(tape, lambda rows: any(r["kind"] == "delta_carried_over" for r in rows))
        islands[1].delta_tensor(base_version=1, c_tokens=10, c_steps=1, update=update)
        assert islands[2].wait_base(newer_than=1, timeout_s=10.0) is not None
        islands[2].delta_tensor(base_version=7, c_tokens=20, c_steps=2, update=update)
        _wait_tape(tape, lambda rows: any(r["kind"] == "delta_rejected" for r in rows))
        islands[2].delta_tensor(base_version=2, c_tokens=20, c_steps=2, update=update)
        rows = _wait_tape(tape, lambda rows: sum(r["kind"] == "outer_step" for r in rows) >= 3)
    finally:
        for c in islands.values():
            c.close()
        syncer.kill()
        syncer.wait(5)

    steps = _fold(rows)
    assert [m["merge/base_version"] for m in steps[:3]] == [0, 1, 2]
    v0, v1, v2 = steps[:3]
    assert v0["merge/island/1/weight_share"] == 1.0 and v0["merge/submitted"] == 1
    assert v1["merge/island/1/weight_share"] == pytest.approx(0.5)
    assert v1["merge/island/2/weight_share"] == pytest.approx(0.5)
    assert v1["merge/carried_weight_share"] == pytest.approx(0.5)
    assert v1["merge/carried_fraction"] == 0.5 and v1["merge/carried_discount_mean"] == 0.5
    assert v2["merge/rejected_fraction/unknown_version"] == 0.5
    assert v2["merge/island/2/weight_share"] == 1.0

    # L5.7 clock: every line has time_unix within the test window; deltas
    # carry their arrival offset; outer steps their duration and last island.
    ended = time.time()
    assert all(started - 1 <= r["time_unix"] <= ended + 1 for r in rows)
    for r in rows:
        if r["kind"] in ("delta_accepted", "delta_carried_over"):
            assert r["arrival_offset_s"] >= 0.0
    outer = [r for r in rows if r["kind"] == "outer_step"][:3]
    assert [r["last_arrival_island"] for r in outer] == [1, 1, 2]
    assert all(r["step_seconds"] >= 0.0 for r in outer)
    assert v1["merge/step_seconds"] == outer[1]["step_seconds"]
