"""The page view of a multi-day run: thousands of rounds, a full load series.

The serve loop rebuilds the view on every poll, so its cost must stay about linear in
the rounds: each round must not rescan the whole load series or every publication.
"""

from __future__ import annotations

from collections import deque

from yeto.dashboard.reducer import Reducer


class _CountingDeque(deque):
    iterations = 0

    def __iter__(self):
        type(self).iterations += 1
        return super().__iter__()


def _long_run(rounds: int, loads: int) -> Reducer:
    r = Reducer(run="long")
    t0 = 1.0e9
    for rid in range(rounds):
        base = t0 + rid * 60
        for task, a, b in (("generate", 0, 30), ("train", 30, 50), ("publish", 50, 55)):
            r.feed({"event": "rl_timeline_span", "island_id": "0", "task": task, "rollout_id": rid,
                    "start": base + a, "end": base + b, "time_unix": base + b})
        r.feed({"event": "rl_publication", "island_id": "0", "policy_version": rid + 1, "time_unix": base + 55})
        r.feed({"event": "rl_trajectory_reward", "island_id": "0", "rollout_id": rid, "trajectory_id": f"t{rid}",
                "reward": 1.0, "trajectory_started_at": base + 1, "trajectory_ended_at": base + 20,
                "time_unix": base + 20})
    for k in range(loads):
        r.feed({"event": "rl_load_sample", "island_id": "0", "kv_used_tokens": k % 7, "kv_capacity_tokens": 10,
                "queued_requests": k % 5, "time_unix": t0 + k * rounds * 60 / loads})
    return r


def test_utilization_does_not_rescan_the_load_series_per_round():
    r = _long_run(200, 2000)
    isl = r.islands["0"]
    isl["load"] = _CountingDeque(isl["load"], maxlen=isl["load"].maxlen)
    _CountingDeque.iterations = 0
    util = r.utilization(isl)
    assert len(util["rounds"]) == 200
    assert _CountingDeque.iterations <= 5, _CountingDeque.iterations


def test_round_peaks_and_publish_spans_match_a_direct_scan():
    r = _long_run(50, 600)
    isl = r.islands["0"]
    rounds = r.round_records(isl)
    util = r.utilization(isl, rounds)
    load = list(isl["load"])
    for row in util["rounds"]:
        a, b = row["gen"]
        win = [x for x in load if a <= x[0] <= b]
        kv = [x[1] / x[2] for x in win if x[1] is not None and x[2]]
        q = [x[3] for x in win if x[3] is not None]
        assert row["peaks"] == {"kv": round(max(kv), 4) if kv else None, "queued": max(q) if q else None}
    for row in rounds:
        rid = row["round"]
        assert row["phases"]["P"] == [1.0e9 + rid * 60 + 50, 1.0e9 + rid * 60 + 55]
