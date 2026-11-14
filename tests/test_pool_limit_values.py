"""Price and limit values of a pool config: wrong values are refused before any order."""
from __future__ import annotations

import pytest

from yeto.launch.nc import cli
from yeto.launch.pool import status as el

BASE = {"target": 1, "accounts": [7], "shard_base": "nc-x", "run": "run-x", "spec": "s.json",
        "run_prefix": "run-x", "r2_env_file": "store.env"}


def test_nan_spend_cap_is_refused():
    # json.loads accepts NaN, and "x > nan" is always False: such a cap never refuses an order
    unchecked = el.ElasticConfig(max_run_spend_usd=float("nan"))
    assert el.Budget(spent_ucr=1e15).refusal(unchecked, billed_nodes=0, target=1, hourly_ucr=10**9) is None
    with pytest.raises(ValueError, match="max_run_spend_usd"):
        el.ElasticConfig.from_dict({"max_run_spend_usd": float("nan")})


@pytest.mark.parametrize("key, value", [
    ("max_run_spend_ucr", float("nan")), ("max_run_spend_ucr", -1), ("max_run_spend_ucr", "30"),
    ("max_run_spend_ucr", True), ("max_run_spend_usd", float("inf")), ("max_run_spend_usd", -5.0),
    ("max_run_spend_usd", "30"), ("max_nodes", 0), ("max_nodes", "3"), ("max_nodes", True),
    ("max_nodes", 2.5), ("max_orders", -1), ("max_orders", "10"), ("max_orders", False),
    ("max_sampler_restarts", -1), ("max_sampler_restarts", "2"), ("start_timeout_s", float("nan")),
    ("start_timeout_s", "30"), ("breaker_failures", True),
])
def test_bad_elastic_limits_are_refused(key, value):
    with pytest.raises(ValueError, match=key):
        el.ElasticConfig.from_dict({key: value})


def test_documented_elastic_limits_are_accepted():
    cfg = el.ElasticConfig.from_dict({"max_run_spend_ucr": 2000000000, "max_nodes": 3, "max_orders": 10,
                                      "engine_stall_s": 600, "max_run_spend_usd": 50.5,
                                      "max_sampler_restarts": 0, "max_orders": 0})
    assert cfg.max_run_spend_ucr == 2000000000 and cfg.max_nodes == 3 and cfg.max_orders == 0
    assert el.ElasticConfig.from_dict({"max_run_spend_usd": 0}).max_run_spend_usd == 0


@pytest.mark.parametrize("price", [-3.0, float("nan"), float("inf"), "abc", True])
def test_bad_provider_price_is_refused(price):
    cfg = dict(BASE, provider={"kind": "modal", "price_usd_h": price})
    with pytest.raises(ValueError, match="price_usd_h"):
        cli.pool_config(cfg, allowed={7})


def test_negative_price_disarmed_the_spend_cap():
    # before the fix: a negative node hour made "spent + hourly > cap" never true
    cfg = dict(BASE, provider={"kind": "modal", "price_usd_h": -3.0}, elastic={"max_run_spend_usd": 10})
    with pytest.raises(ValueError, match="price_usd_h"):
        cli.pool_config(cfg, allowed={7})


@pytest.mark.parametrize("price", [3.5, "30", 0])
def test_documented_price_forms_are_accepted(price):
    pc = cli.pool_config(dict(BASE, provider={"kind": "modal", "price_usd_h": price}), allowed={7})
    assert pc.hourly_cost >= 0


@pytest.mark.parametrize("key", ["replace_lead_s", "end_margin_s"])
@pytest.mark.parametrize("value", [float("nan"), -1, "x"])
def test_bad_lead_and_margin_are_refused(key, value):
    with pytest.raises(ValueError, match=key):
        cli.pool_config(dict(BASE, **{key: value}), allowed={7})
