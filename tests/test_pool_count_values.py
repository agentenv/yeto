"""Counts and prices of a run / pool config the earlier checks did not cover: a bool,
a fraction, nan / inf or a negative value is refused at load, before any order."""
from __future__ import annotations

import pytest

from yeto.launch.nc import cli
from yeto.launch.rollout import offers, pools
from yeto.rl.engine import store_dispense

BASE = {"target": 1, "accounts": [7, 8], "shard_base": "nc-x", "run": "run-x", "spec": "s.json",
        "run_prefix": "run-x", "r2_env_file": "store.env"}


def _offers(price):
    return {"offers": [{"resources": {"accelerators": "H100:8"}, "price_usd_h": 6.0},
                       {"resources": {"accelerators": "H200:8"}, "price_usd_h": price}]}


@pytest.mark.parametrize("price", [-3.0, float("nan"), float("inf"), True, "abc"])
def test_bad_later_offer_price_is_refused(price):
    # the pool bills a node at the price of the offer it was launched with: a negative
    # or nan later offer made the run's spend fall (or never compare) and disarmed the cap
    with pytest.raises(ValueError, match=r"offers\[1\].*price_usd_h"):
        offers.offers_from_config(_offers(price))


@pytest.mark.parametrize("price", [-1, float("nan"), False])
def test_bad_single_resources_price_is_refused(price):
    with pytest.raises(ValueError, match="price_usd_h"):
        offers.offers_from_config({"resources": {"accelerators": "H100:8"}, "price_usd_h": price})


@pytest.mark.parametrize("price", [0, 6.32, 11, "11.2", None])
def test_valid_offer_prices_are_accepted(price):
    got = offers.offers_from_config(_offers(price))[1].price_usd_h
    assert got == (None if price is None else float(price))


def _samplers(tmp_path, **entry):
    doc = {"samplers": [dict({"provider": {"kind": "modal"}, "target": 1, "slots": [1], "shard_base": "a"},
                             **entry)]}
    return pools.resolve(doc, tmp_path)


@pytest.mark.parametrize("value", [-5, float("nan"), float("inf"), True, "3"])
def test_bad_sampler_usd_per_h_is_refused(tmp_path, value):
    # samplers[i].usd_per_h prices the run's spend estimate: -5 or nan never trips budget.usd
    with pytest.raises(ValueError, match="usd_per_h"):
        _samplers(tmp_path, usd_per_h=value)


@pytest.mark.parametrize("value", [0, -1, float("nan"), True, "30"])
def test_bad_sampler_interval_is_refused(tmp_path, value):
    with pytest.raises(ValueError, match="interval_s"):
        _samplers(tmp_path, interval_s=value)


def test_documented_sampler_entry_is_accepted(tmp_path):
    ref = _samplers(tmp_path, usd_per_h=0, interval_s=30)[0]
    assert ref.usd_per_h == 0 and ref.interval_s == 30
    assert _samplers(tmp_path, usd_per_h=12.5)[0].usd_per_h == 12.5
    assert _samplers(tmp_path)[0].usd_per_h is None


@pytest.mark.parametrize("target", [1.5, True, float("inf"), float("nan"), "1.5"])
def test_bad_target_is_refused(target):
    with pytest.raises(ValueError, match="target"):
        cli.pool_config(dict(BASE, target=target))


@pytest.mark.parametrize("accounts", [[7.5, 8], [True, 8]])
def test_bad_account_ids_are_refused(accounts):
    with pytest.raises(ValueError, match="accounts"):
        cli.pool_config(dict(BASE, accounts=accounts))


@pytest.mark.parametrize("slots", [2.5, True, float("nan")])
def test_bad_data_slots_are_refused(slots):
    with pytest.raises(ValueError, match="data_slots"):
        cli.pool_config(dict(BASE, data_slots=slots))


def test_documented_counts_are_accepted():
    # the guides write account ids as strings ("<account-id>")
    pc = cli.pool_config(dict(BASE, accounts=["7", "8"], target=2, data_slots=4), allowed={7, 8})
    assert pc.accounts == [7, 8] and pc.target == 2 and pc.data_slots == 4
    assert cli.pool_config(dict(BASE, target=2.0)).target == 2


@pytest.mark.parametrize("backlog", [
    {"synced": False, "explicit_rounds": -1},  # a negative max_backlog_groups
    {"synced": False, "explicit_rounds": 2.5},
    {"synced": True, "explicit_rounds": "3"},  # TypeError when the member spec renders
    {"synced": True, "fragments": float("nan")},  # nan max_backlog_groups
    {"synced": "false"},  # a truthy string: sized as a synced group
    {"synced": True, "trainers": 0},
    {"synced": True, "local_rounds": True},
    {"synced": True, "pipeline": -1},
    {"trainers": 2},  # synced is required
])
def test_bad_backlog_is_refused(backlog):
    with pytest.raises(ValueError, match="backlog"):
        cli.pool_config(dict(BASE, backlog=backlog))


def test_valid_backlog_is_accepted():
    for backlog in ({"synced": False}, {"synced": False, "explicit_rounds": 4},
                    {"synced": True, "trainers": 2, "fragments": 2, "local_rounds": 1, "pipeline": 0,
                     "explicit_rounds": None}):
        cli.pool_config(dict(BASE, backlog=backlog))


@pytest.mark.parametrize("passes", [0, -1, 2.5, True, "2", float("nan")])
def test_bad_dispense_passes_is_refused(passes):
    # 0 was read as 1 (the pool stopped after one pass), 2.5 as 2, -1 refused only on the node
    with pytest.raises(store_dispense.DispenseError, match="passes"):
        store_dispense.check_store_config({"backend": "store", "passes": passes})
    with pytest.raises(ValueError, match="passes"):
        cli.pool_config(dict(BASE, sampler={"dispense": {"backend": "store", "passes": passes}}))


@pytest.mark.parametrize("passes", [1, 1000, None])
def test_documented_passes_are_accepted(passes):
    store_dispense.check_store_config({"backend": "store", "namespace": "task-dispense", "lease_s": 3600,
                                       "max_redo": 2, "passes": passes})
