"""Account quotas shared by every pool (unified-sampler-pool 6.4): boat, National
Compute accounts, Daytona; adding or removing a pool splits again, never past the limit."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from yeto.launch.pool import quota as q
from yeto.launch.rollout import pools
from yeto.rl import fleet_sandbox as fsb


def test_split_per_unit_never_passes_the_limit():
    for limit, sizes in itertools.product((12, 13, 100), ((1, 1), (2, 3), (4, 1, 2), (5,))):
        claims = {f"p{i}": q.Claim(units=n, weight=1 + i) for i, n in enumerate(sizes)}
        try:
            shares = q.split("x", limit, claims)
        except q.QuotaShort:
            assert sum(c.units * c.weight for c in claims.values()) > limit
            continue
        assert sum(claims[n].units * s for n, s in shares.items()) <= limit
        assert all(s >= claims[n].weight for n, s in shares.items())
    shares = q.split("x", 12, {"a": q.Claim(2, fixed=4), "b": q.Claim(2)})
    assert shares == {"b": 2}, "a's own 4 x 2 is counted; b's 2 processes split the 4 left"
    with pytest.raises(q.QuotaShort) as short:
        q.split("x", 12, {"a": q.Claim(13)})
    assert (short.value.need, short.value.left) == (13, 12)


def test_adding_or_removing_a_boat_pool_splits_the_account_again():
    account = fsb.boat_account({})
    run = {"a": {"processes": 2, "max_in_flight": 20, "explicit": {}},
           "b": {"processes": 2, "max_in_flight": 10, "explicit": {}}}

    def total(shares, param):
        return sum(run[n]["processes"] * s[param] for n, s in shares.items() if param in s)

    before = pools.boat_shares(account, run)
    run["c"] = {"processes": 1, "max_in_flight": 10, "explicit": {}}  # a new agentic pool at run time
    after = pools.boat_shares(account, run)
    assert total(after, "boat_starts_per_minute_n") <= 12 and total(after, "boat_max_active_n") <= 100
    assert after["a"]["boat_starts_per_minute_n"] < before["a"]["boat_starts_per_minute_n"]
    del run["c"], run["b"]  # two pools leave: the rest of the account goes to a
    alone = pools.boat_shares(account, run)
    assert alone["a"]["boat_starts_per_minute_n"] == 6 and alone["a"]["boat_max_active_n"] == 50


def test_the_new_split_takes_effect_without_passing_the_limit_meanwhile():
    # a's 2 processes started at 6 starts/min each (12 total); pool c joins: target split 4/4/4
    now = q.transition(12, {"a": 8, "c": 4}, {"a": 12})
    assert now == {"a": 8, "c": 0}, "c waits for a's processes to restart with their new share"
    assert q.transition(12, {"a": 8, "c": 4}, {"a": 8}) == {"a": 8, "c": 4}
    # a pool removed: its processes hold their share until they end
    assert q.transition(12, {"a": 12}, {"a": 6, "b": 6}) == {"a": 6}
    assert q.transition(12, {"a": 12}, {"a": 6}) == {"a": 12}
    for held in ({"a": 12}, {"a": 3, "b": 9}, {"b": 12}):
        out = q.transition(12, {"a": 6, "b": 3, "c": 3}, held)
        assert sum(max(v, held.get(n, 0)) for n, v in out.items()) + \
            sum(v for n, v in held.items() if n not in out) <= 12


def test_nc_accounts_are_disjoint_and_never_protected():
    shares = q.split_accounts([1, 2, 3, 4, 5], {"i0": 2, "i1": 2})
    assert shares.grants == {"i0": (1, 5), "i1": (4,)} and shares.short == {"i1": 1}
    assert not set(itertools.chain(*shares.grants.values())) & q.NC_PROTECTED
    # a pool that holds accounts keeps them; a new pool gets what is left
    shares = q.split_accounts([1, 4, 5, 6], {"i0": 2, "i1": 1, "new": 2}, held={"i0": [5, 6]})
    assert shares.grants == {"i0": (5, 6), "i1": (1,), "new": (4,)} and shares.short == {"new": 1}
    # a pool removed: its accounts are free for the others
    shares = q.split_accounts([1, 4, 5, 6], {"new": 2}, held={"new": [4]})
    assert shares.grants == {"new": (1, 4)} and shares.free == (5, 6)
    for held, match in (({"a": [2]}, "protected"), ({"a": [9]}, "not allowed"), ({"a": [1], "b": [1]}, "held by")):
        with pytest.raises(ValueError, match=match):
            q.split_accounts([1, 4], {"a": 1, "b": 1}, held=held)


def test_daytona_nodes_floors_then_wants_and_running_nodes_kept():
    shares = q.split_nodes("daytona", 5, {"a": q.NodeWant(4, floor=1), "b": q.NodeWant(4, floor=2)})
    assert shares.total() == 5 and shares.grants["b"] >= 2 and shares.grants["a"] >= 1
    shares = q.split_nodes("daytona", 5, {"a": q.NodeWant(1, held=3), "b": q.NodeWant(4)})
    assert shares.grants == {"a": 3, "b": 2}, "a running node is never taken away by a split"
    with pytest.raises(q.QuotaShort):
        q.split_nodes("daytona", 2, {"a": q.NodeWant(3, held=3)})


def _boat_pool(tmp_path: Path, name: str, *, target: int, mif: int) -> dict:
    spec = {"sampler_id": "x", "store": "/s", "sampling": {"n": 1},
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1, "format": "agentic"},
            "agentic": {"max_in_flight": mif, "task_defaults": {"provider": "boat"}}}
    (tmp_path / f"{name}.spec.json").write_text(json.dumps(spec))
    return {"name": name, "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "gpu": "H100"},
            "spec": str(tmp_path / f"{name}.spec.json"), "run_prefix": "r", "shard_base": name, "target": target,
            "slots": [1, 2, 3], "run": "r"}


def test_run_pools_after_adding_a_pool_sum_within_the_account(tmp_path):
    from yeto.rl import up_consistency

    cfg = {"run": "r", "store": "s3://b/r", "trainer": {"argv": []},
           "samplers": [_boat_pool(tmp_path, "a", target=2, mif=20), _boat_pool(tmp_path, "b", target=1, mif=10)]}
    cfg["samplers"].append(_boat_pool(tmp_path, "c", target=3, mif=5))
    refs = pools.resolve(cfg, tmp_path / "state")
    caps = {r.name: up_consistency.effective_spec(r.load())["breakers"]["harness"]["sandbox_caps"] for r in refs}
    procs = {"a": 2, "b": 1, "c": 3}
    assert sum(procs[n] * c["boat_starts_per_minute_n"] for n, c in caps.items()) <= 12
    assert sum(procs[n] * c["boat_max_active_n"] for n, c in caps.items()) <= 100
    assert sum(procs[n] * c["boat_starts_per_hour_n"] for n, c in caps.items()) <= 60
