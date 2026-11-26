"""A sky trainer cluster on a machine the unified price table has no billing row for (a fleet
island moved to an AWS A10G, say) is recorded with the local sky catalog's on-demand price, so
the cost guard and run.budget count it instead of reporting it unpriced. Only the catalog file
on disk is read: sky's catalog functions (which fetch a stale file with no timeout) are never
called. An operator's price table still wins, and a machine with a table row is recorded as
before."""

from __future__ import annotations

import os
import socket
import sys
import types

import pytest

from yeto.cost import billing, prices
from yeto.cost.ledger import Resource
from yeto.gpu_spec import ClusterSpec
from yeto.launch import trainer_provision as tp

HEADER = "InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,Price,SpotPrice,Region,AvailabilityZone\n"
ROWS = ["g5.xlarge,A10G,1.0,4.0,16.0,1.006,0.45,us-west-2,usw2-az1",
        "g5.xlarge,A10G,1.0,4.0,16.0,1.006,0.41,us-west-2,usw2-az2",
        "g5.2xlarge,A10G,1.0,8.0,32.0,1.212,0.52,us-west-2,usw2-az1",
        "g5.4xlarge,A10G,1.0,16.0,64.0,1.624,0.70,us-west-2,usw2-az1",
        "g5.16xlarge,A10G,1.0,64.0,256.0,4.096,1.50,us-east-1,use1-az1",  # another region
        "g5.12xlarge,A10G,4.0,48.0,192.0,5.672,2.10,us-west-2,usw2-az1",  # four cards
        "g6.xlarge,L4,1.0,4.0,16.0,0.805,0.30,us-west-2,usw2-az1"]


def _sky(monkeypatch, path):
    """sky's catalog package with only its path rule usable: every catalog function (they
    fetch a stale file over the network) fails the test if called."""
    def forbidden(*_a, **_kw):
        raise AssertionError("a sky catalog function was called (it may fetch over the network)")

    common = types.SimpleNamespace(get_catalog_path=lambda rel: str(path / rel), read_catalog=forbidden)
    cat = types.SimpleNamespace(get_hourly_cost=forbidden, get_instance_type_for_accelerator=forbidden, common=common)
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(catalog=cat))
    monkeypatch.setitem(sys.modules, "sky.catalog", cat)
    monkeypatch.setitem(sys.modules, "sky.catalog.common", common)


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    """A fresh local AWS catalog file; no network: any connection attempt fails the test."""
    (tmp_path / "aws").mkdir()
    (tmp_path / "aws" / "vms.csv").write_text(HEADER + "\n".join(ROWS) + "\n")
    _sky(monkeypatch, tmp_path)

    def no_network(*_a, **_kw):
        raise AssertionError("network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    return tmp_path / "aws" / "vms.csv"


def _attrs(spec: ClusterSpec, *, spot: bool = False, instance_type: str | None = None, stated=None) -> dict:
    args = types.SimpleNamespace(cluster_prefix="r", training_mode="rl", spot=spot,
                                 learner_instance_type=instance_type, trainer_usd_per_hour=stated)
    prov = tp.TrainerProvision(args, sky=None, specs=[spec], names=["r-l0"], no_sync=True)
    return prov.ledger_attrs(prov.island("r-l0", 0, spec, None, types.SimpleNamespace(envs={})))


def _bill(attrs: dict, rates: billing.Rates | None = None):
    (item,) = billing.resource_costs([Resource("sky_cluster", "r-l0", "r-l0", 0.0, attrs)], 3600.0,
                                     rates=rates).items
    return item


A10G = ClusterSpec("aws", "us-west-2", 1, 1, "A10G")


def test_a_machine_without_a_table_row_records_its_instance_types_on_demand_price(catalog):
    assert prices.node_usd_h("aws", "A10G", 1, spot=False, region="us-west-2") is None  # no billing row
    attrs = _attrs(A10G, instance_type="g5.2xlarge")
    assert attrs["catalog_usd_per_hour"] == pytest.approx(1.212) and "usd_per_hour" not in attrs
    assert attrs["price_note"] == "sky catalog on-demand price of g5.2xlarge in us-west-2 (no price table row)"
    item = _bill(attrs)
    assert item.priced and item.estimated_usd == pytest.approx(1.212) and item.note == attrs["price_note"]


def test_a_spot_node_without_a_named_instance_bills_the_dearest_on_demand_instance(catalog):
    attrs = _attrs(ClusterSpec("aws", "us-west-2", 2, 1, "A10G"), spot=True)
    assert attrs["catalog_usd_per_hour"] == pytest.approx(2 * 1.624)  # two nodes, this region, never under
    assert attrs["price_note"] == ("sky catalog on-demand price of the dearest 1xA10G instance (g5.4xlarge) "
                                   "in us-west-2 (no price table row; spot billed at on-demand)")
    four = _attrs(ClusterSpec("aws", "us-west-2", 1, 4, "A10G"))
    assert four["catalog_usd_per_hour"] == pytest.approx(5.672)


def test_an_operators_price_table_still_wins_over_the_catalog(catalog, tmp_path):
    attrs = _attrs(A10G, instance_type="g5.2xlarge")
    table = tmp_path / "prices.json"
    table.write_text('{"prices": {"aws:A10G": 0.9}}')
    item = _bill(attrs, billing.Rates.from_env(environ={"YETO_DASHBOARD_PRICES": str(table)}))
    assert item.estimated_usd == pytest.approx(0.9) and "sky catalog" not in (item.note or "")
    assert _bill(attrs).estimated_usd == pytest.approx(1.212)  # without one: the catalog's


def test_a_machine_with_a_table_row_is_recorded_as_before(catalog):
    l4 = ClusterSpec("aws", "us-west-2", 1, 1, "L4")
    assert _attrs(l4, instance_type="g6.2xlarge") == {"role": "trainer", "cloud": "aws", "gpu": "L4", "gpus": 1,
                                                     "num_nodes": 1, "region": "us-west-2", "use_spot": False}
    assert _attrs(A10G, stated=2.0) == {"role": "trainer", "cloud": "aws", "gpu": "A10G", "gpus": 1, "num_nodes": 1,
                                        "region": "us-west-2", "use_spot": False, "usd_per_hour": 2.0}


def test_a_stale_catalog_file_is_read_as_it_is_and_dated(catalog):
    os.utime(catalog, (1790000000, 1790000000))  # older than a day: sky would fetch it again
    rate, note = prices.catalog_node_usd_h("aws", "A10G", 1, region="us-west-2", instance_type="g5.xlarge")
    assert rate == pytest.approx(1.006)
    assert note == ("sky catalog on-demand price of g5.xlarge in us-west-2; local catalog file of 2026-09-21, "
                    "not refreshed here")


def test_without_a_local_catalog_the_cluster_stays_unpriced_and_says_so(monkeypatch, tmp_path, capsys):
    _sky(monkeypatch, tmp_path / "missing")
    attrs = _attrs(A10G, instance_type="g5.2xlarge")
    assert "catalog_usd_per_hour" not in attrs and "usd_per_hour" not in attrs
    assert "no local sky catalog for aws: the cost guard reports it unpriced" in capsys.readouterr().out
    assert not _bill(attrs).priced


@pytest.mark.parametrize("content,why", [
    ("InstanceType,Price\ng5.xlarge,1.0\n", "unreadable sky catalog for aws (KeyError)"),  # columns missing
    (HEADER + "g5.xlarge,A10G,one,4.0,16.0,1.006,0.45,us-west-2,usw2-az1\n",
     "unreadable sky catalog for aws (ValueError)"),  # a bad number
    (HEADER + "g5.xlarge,A10G,1.0,4.0,16.0,cheap,0.45,us-west-2,usw2-az1\n",
     "unreadable sky catalog for aws (ValueError)"),
])
def test_a_malformed_catalog_leaves_the_cluster_unpriced_without_raising(monkeypatch, tmp_path, capsys, content, why):
    (tmp_path / "aws").mkdir()
    (tmp_path / "aws" / "vms.csv").write_text(content)
    _sky(monkeypatch, tmp_path)
    assert prices.catalog_node_usd_h("aws", "A10G", 1, region="us-west-2") == (None, why)
    attrs = _attrs(A10G)  # the launch goes on, unpriced and said
    assert "catalog_usd_per_hour" not in attrs and why in capsys.readouterr().out


def test_a_failing_sky_path_rule_leaves_the_cluster_unpriced_without_raising(monkeypatch):
    def broken(_rel):
        raise RuntimeError("sky config unreadable")

    common = types.SimpleNamespace(get_catalog_path=broken)
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(catalog=types.SimpleNamespace(common=common)))
    monkeypatch.setitem(sys.modules, "sky.catalog", types.SimpleNamespace(common=common))
    monkeypatch.setitem(sys.modules, "sky.catalog.common", common)
    assert prices.catalog_node_usd_h("aws", "A10G", 1) == (None, "no sky catalog (RuntimeError)")
    assert "catalog_usd_per_hour" not in _attrs(A10G)


def test_an_unknown_instance_type_or_card_count_stays_unpriced(catalog):
    assert prices.catalog_node_usd_h("aws", "A10G", 1, region="us-west-2", instance_type="nope.large") == (
        None, "sky catalog has no on-demand price for nope.large in us-west-2")
    assert prices.catalog_node_usd_h("aws", "B300", 8, region="us-west-2") == (
        None, "sky catalog has no 8xB300 instance on aws in us-west-2")
