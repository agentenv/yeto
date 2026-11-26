"""A sky trainer cluster on a machine the unified price table has no billing row for (a fleet
island moved to an AWS A10G, say) is recorded with the sky catalog's on-demand price, so the
cost guard and run.budget count it instead of reporting it unpriced. A machine with a table
row is recorded as before (billing reads the row)."""

from __future__ import annotations

import sys
import types

import pytest

from yeto.cost import billing, prices
from yeto.cost.ledger import Resource
from yeto.gpu_spec import ClusterSpec
from yeto.launch import trainer_provision as tp

HOURLY = {("g5.xlarge", "us-west-2"): 1.006, ("g5.2xlarge", "us-west-2"): 1.212, ("g5.4xlarge", "us-west-2"): 1.624}


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    """A local sky catalog (sky.catalog) with three A10G instances in us-west-2."""
    calls = []
    (tmp_path / "aws").mkdir()
    (tmp_path / "aws" / "vms.csv").write_text("InstanceType\n")

    def get_hourly_cost(instance_type, use_spot=False, region=None, zone=None, clouds=None):
        calls.append(("cost", instance_type, use_spot, region, clouds))
        return HOURLY[(instance_type, region)]

    def get_instance_type_for_accelerator(acc, count, use_spot=False, region=None, zone=None, clouds=None, **_kw):
        calls.append(("types", acc, count, use_spot, region, clouds))
        return ([t for t, r in HOURLY if r == region] if (acc, count) == ("A10G", 1) else None), []

    common = types.SimpleNamespace(get_catalog_path=lambda rel: str(tmp_path / rel))
    cat = types.SimpleNamespace(get_hourly_cost=get_hourly_cost,
                                get_instance_type_for_accelerator=get_instance_type_for_accelerator, common=common)
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(catalog=cat))
    monkeypatch.setitem(sys.modules, "sky.catalog", cat)
    monkeypatch.setitem(sys.modules, "sky.catalog.common", common)
    return calls


def _attrs(spec: ClusterSpec, *, spot: bool = False, instance_type: str | None = None, stated=None) -> dict:
    args = types.SimpleNamespace(cluster_prefix="r", training_mode="rl", spot=spot,
                                 learner_instance_type=instance_type, trainer_usd_per_hour=stated)
    prov = tp.TrainerProvision(args, sky=None, specs=[spec], names=["r-l0"], no_sync=True)
    return prov.ledger_attrs(prov.island("r-l0", 0, spec, None, types.SimpleNamespace(envs={})))


A10G = ClusterSpec("aws", "us-west-2", 1, 1, "A10G")


def test_a_machine_without_a_table_row_records_its_instance_types_on_demand_price(catalog):
    assert prices.node_usd_h("aws", "A10G", 1, spot=False, region="us-west-2") is None  # no billing row
    attrs = _attrs(A10G, instance_type="g5.2xlarge")
    assert attrs["usd_per_hour"] == pytest.approx(1.212)
    assert attrs["price_note"] == "sky catalog on-demand price of g5.2xlarge in us-west-2 (no price table row)"
    assert ("cost", "g5.2xlarge", False, "us-west-2", "aws") in catalog
    (item,) = billing.resource_costs([Resource("sky_cluster", "r-l0", "r-l0", 0.0, attrs)], 3600.0).items
    assert item.priced and item.estimated_usd == pytest.approx(1.212) and item.note == attrs["price_note"]


def test_a_spot_node_without_a_named_instance_bills_the_dearest_on_demand_instance(catalog):
    spec = ClusterSpec("aws", "us-west-2", 2, 1, "A10G")
    attrs = _attrs(spec, spot=True)
    assert attrs["usd_per_hour"] == pytest.approx(2 * 1.624)  # two nodes, never under
    assert attrs["price_note"] == ("sky catalog on-demand price of the dearest 1xA10G instance (g5.4xlarge) "
                                   "in us-west-2 (no price table row; spot billed at on-demand)")
    assert all(c[2] is False for c in catalog if c[0] == "cost") and all(c[3] is False for c in catalog if c[0] == "types")


def test_a_machine_with_a_table_row_is_recorded_as_before(catalog):
    l4 = ClusterSpec("aws", "us-west-2", 1, 1, "L4")
    assert _attrs(l4, instance_type="g6.2xlarge") == {"role": "trainer", "cloud": "aws", "gpu": "L4", "gpus": 1,
                                                     "num_nodes": 1, "region": "us-west-2", "use_spot": False}
    assert _attrs(A10G, stated=2.0)["usd_per_hour"] == 2.0  # --trainer-usd-per-hour wins
    assert catalog == []  # the catalog is not read


def test_without_a_local_catalog_the_cluster_stays_unpriced_and_says_so(monkeypatch, tmp_path, capsys):
    common = types.SimpleNamespace(get_catalog_path=lambda rel: str(tmp_path / "missing" / rel))
    cat = types.SimpleNamespace(common=common)
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(catalog=cat))
    monkeypatch.setitem(sys.modules, "sky.catalog", cat)
    monkeypatch.setitem(sys.modules, "sky.catalog.common", common)
    attrs = _attrs(A10G, instance_type="g5.2xlarge")
    assert "usd_per_hour" not in attrs
    assert "no local sky catalog for aws: the cost guard reports it unpriced" in capsys.readouterr().out


def test_an_unknown_instance_type_stays_unpriced(catalog):
    assert prices.catalog_node_usd_h("aws", "A10G", 1, region="us-west-2", instance_type="nope.large") == (
        None, "sky catalog has no on-demand price for nope.large in us-west-2 (KeyError)")
    assert prices.catalog_node_usd_h("aws", "B300", 8, region="us-west-2") == (
        None, "sky catalog has no 8xB300 instance on aws in us-west-2")
