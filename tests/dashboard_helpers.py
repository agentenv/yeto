"""Shared builders for the fleet-dashboard tests."""

from pathlib import Path

from yeto.dashboard.reducer import Reducer

FX = Path(__file__).parent / "fixtures" / "dashboard"
T0 = 1_790_000_000.0


def local_round(island, x, t, **kw):
    rec = {"event": "rl_local_round", "island_id": island, "local_round_id": x, "time_unix": t,
           "reward_mean": 0.2 + 0.01 * x, "grad_norm": 0.8, "action_tokens": 3600,
           "rollout_seconds": 0.5, "train_seconds": 0.5}
    rec.update(kw)
    return rec


def merge(step, expected, responded, *, quorum_ms=1000, t=None, fragment=0):
    return {"step": step, "fragment": fragment, "attempt": 1, "expected": expected,
            "responded": responded, "responders": [{"id": i, "staleness": 0, "contribution": 0.25}
                                                   for i in responded],
            "quorum_ms": quorum_ms, "grace_ms": 0, "sync_ms": 300, "sync/merge_seconds": 0.12,
            "gnorm": 0.4, "time_unix": t}


def four_island_reducer(**kw):
    """4 islands, 3 syncer rounds; island 3 does not push round 2; island 1 resends round 3."""
    r = Reducer(run="t4", **kw)
    for i in range(4):
        for x in range(1, 4):
            r.feed(local_round(i, x, T0 + 10 * x))
    for step in (1, 2, 3):
        pushers = [0, 1, 2] if step == 2 else [0, 1, 2, 3]
        for i in pushers:
            r.feed({"event": "rl_fragment_push", "island_id": i, "global_step": step, "time_unix": T0 + step})
        r.feed(merge(step, [0, 1, 2, 3], pushers, t=T0 + 10 * step + 5))
    r.feed({"event": "rl_pull_resend", "island_id": 1, "global_step": 3, "round_attempt": 1,
            "fragment_id": 0, "pulls_received": 2, "time_unix": T0 + 31})
    return r
