"""Long runs: the dashboard keeps its lists bounded without losing what a view shows.

The NC pool panel used to keep only the first ``POOL_MAX`` pool events: past it (a
long run with a retrying pool) a member_ready, a target change or a node exit was
dropped and the sampler count and exit status froze at their old values."""

from __future__ import annotations

from yeto.dashboard import async_rl
from yeto.dashboard.reducer import Reducer

T0 = 1_790_000_000.0


def test_pool_events_after_the_cap_still_reach_the_view(monkeypatch):
    monkeypatch.setattr(async_rl, "POOL_MAX", 50)
    r = Reducer()
    r.feed({"t": T0, "event": "nc_elastic_target_changed", "old": 0, "new": 1, "source": "config"})
    r.feed({"t": T0 + 1, "event": "nc_elastic_member_ready", "ticket": "tk1", "shard": 0})
    for i in range(200):  # a pool retrying an order every tick
        r.feed({"t": T0 + 10 + i, "event": "nc_elastic_order_refused", "why": "no offer"})
    r.feed({"t": T0 + 500, "event": "nc_elastic_target_changed", "old": 1, "new": 2, "source": "manual"})
    r.feed({"t": T0 + 510, "event": "nc_elastic_member_ready", "ticket": "tk2", "shard": 1})
    r.feed({"t": T0 + 520, "event": "nc_elastic_node_lost", "ticket": "tk1", "shard": 0, "detail": "gone"})
    st = r.async_
    scale = st.pool_scale()
    assert scale["target"] == 2 and scale["ready"] == ["1"]
    assert st.pool_exit("0")["event"] == "nc_elastic_node_lost"
    assert st.pool_exit("1") is None
    assert len(st.pool) <= 50 + 5  # the noise is trimmed, the state events are kept
    assert st.pool[-1]["event"] == "nc_elastic_node_lost"


def test_fleet_records_cost_ticks_and_evals_are_bounded():
    r = Reducer()
    for i in range(3000):
        r.feed({"event": "cost_tick", "time_unix": T0 + i, "total_usd": float(i), "islands": []})
        r.feed({"event": "island_ready", "island": "run-l0-eu", "time_unix": T0 + i})
    fleet = r.fleet_view(now=T0 + 3000)
    assert len(fleet["records"]) == 500 and len(fleet["cost_ticks"]) == 500
    assert fleet["cost_ticks"][-1]["total_usd"] == 2999.0
    assert len(r.fleet_records) <= 500 and len(r.cost_ticks) <= 500


def _pool_outputs(st, shards):
    scale = st.pool_scale()
    return scale, {s: st.pool_exit(s) for s in shards}, sorted(st.pool_shards)


def test_trimmed_pool_matches_an_untrimmed_reference_on_random_streams(monkeypatch):
    """Every output that reads the pool events (sampler count, exit status, the NC
    shards) is the same as with every event kept: only events no view folds are dropped."""
    import random

    evs = [*async_rl.POOL_LEFT, *async_rl.POOL_UP, async_rl.POOL_TARGET, async_rl.POOL_DRAINING,
           async_rl.POOL_STOPPED, "nc_elastic_ordered", "nc_elastic_manual_pause", "nc_elastic_manual_resume",
           "nc_elastic_manual_release_detected", "nc_elastic_order_refused", "nc_elastic_retry",
           "nc_elastic_warning", "nc_elastic_sampler_restart"]
    noise = ["nc_elastic_order_refused", "nc_elastic_retry", "nc_elastic_warning", "nc_elastic_sampler_restart"]
    for seed in range(30):
        rng = random.Random(seed)
        stream = []
        for i in range(3000):
            ev = rng.choice(noise) if rng.random() < 0.8 else rng.choice(evs)
            r = {"t": T0 + i + rng.choice([0, 0, 0, -3]), "event": ev}
            if rng.random() < 0.6:
                r["shard"] = rng.randrange(8)
            if rng.random() < 0.6:
                r["ticket"] = f"tk{rng.randrange(12)}"
            if ev in (async_rl.POOL_TARGET, "nc_elastic_manual_release_detected"):
                r["new"] = rng.randrange(6)
            if ev == "nc_elastic_ordered":
                r["replaces"] = f"tk{rng.randrange(12)}"
            stream.append(r)
        monkeypatch.setattr(async_rl, "POOL_MAX", 10**9)
        ref = async_rl.AsyncState()
        for r in stream:
            ref.fold_pool(r)
        monkeypatch.setattr(async_rl, "POOL_MAX", 100)
        cut = async_rl.AsyncState()
        for r in stream:
            cut.fold_pool(r)
        assert len(cut.pool) < len(ref.pool)
        shards = [str(s) for s in range(8)]
        assert _pool_outputs(cut, shards) == _pool_outputs(ref, shards), seed


def test_a_shard_named_only_by_a_trimmed_event_is_still_an_nc_shard(monkeypatch):
    monkeypatch.setattr(async_rl, "POOL_MAX", 5)
    st = async_rl.AsyncState()
    st.fold_pool({"t": T0, "event": "nc_elastic_sampler_restart", "shard": 7})
    for i in range(20):
        st.fold_pool({"t": T0 + 1 + i, "event": "nc_elastic_order_refused"})
    assert all(p.get("shard") is None for p in st.pool)
    assert st.pool_shards == {"7"}
