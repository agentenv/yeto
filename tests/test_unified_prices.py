"""The unified machine price table and the provider capabilities read from
``yeto/cloud/capabilities.json``."""

from __future__ import annotations

import datetime as dt
import json
import time
import types

import pytest

from yeto.cost import billing, prices
from yeto.launch.rollout import offers

# -- the table ------------------------------------------------------------------------


def test_every_row_has_a_whole_node_usd_price_a_source_and_a_date():
    rows = prices.rows()
    assert rows
    for r in rows:
        assert r.usd_h >= 0 and r.gpus >= 1 and r.source and dt.date.fromisoformat(r.as_of)
        assert r.grade in prices.GRADES
        assert (r.grade == prices.ILLUSTRATIVE) == ("illustrative" in r.source)
    for h in prices.table().host.values():
        assert h.source and dt.date.fromisoformat(h.as_of)


G = {"grade": "billing"}


@pytest.mark.parametrize("bad, why", [
    ({"provider": "x", "gpu": "H100", "gpus": 8, "usd_h": 1.0, "source": "s", **G}, "as_of is required"),
    ({"provider": "x", "gpu": "H100", "gpus": 8, "usd_h": 1.0, "source": "s", "as_of": "soon", **G}, "ISO date"),
    ({"provider": "x", "gpu": "H100", "gpus": 0, "usd_h": 1.0, "source": "s", "as_of": "2026-01-01", **G}, "gpus"),
    ({"provider": "x", "gpu": "H100", "gpus": 1, "usd_h": -1, "source": "s", "as_of": "2026-01-01", **G}, "usd_h"),
    ({"provider": "x", "gpu": "H100", "gpus": 1, "usd_h": 1, "source": "s", "as_of": "2026-01-01"}, "grade is"),
    ({"provider": "x", "gpu": "H100", "gpus": 1, "usd_h": 1, "source": "s", "as_of": "2026-01-01",
      "grade": "guess"}, "grade must be"),
])
def test_a_row_without_its_source_date_or_a_sane_price_is_refused(bad, why):
    with pytest.raises(prices.PriceTableError, match=why):
        prices.parse({"rows": [bad]})


def test_two_rows_for_the_same_node_are_refused():
    row = {"provider": "x", "gpu": "H100", "gpus": 8, "usd_h": 1.0, "source": "s", "as_of": "2026-01-01", **G}
    with pytest.raises(prices.PriceTableError, match="two rows"):
        prices.parse({"rows": [row, dict(row, gpu="h100")]})


def test_the_syncer_is_priced_per_provider_by_billing_services_rows_with_their_source():
    aws, modal = prices.service(prices.SYNCER, "aws"), prices.service(prices.SYNCER, "modal")
    assert aws.usd_h == prices.syncer_usd_h("aws") == 0.384 and "AWS" in aws.source
    assert modal.usd_h == prices.syncer_usd_h("Modal") == 2.4 and "Modal" in modal.source
    for row in (aws, modal):
        assert row.grade == prices.BILLING and dt.date.fromisoformat(row.as_of)
    assert modal.cited().startswith("syncer@modal 2.4 USD/h (source: ")
    assert prices.service(prices.SYNCER, "nebius") is None and prices.syncer_usd_h("") is None
    # an illustrative row is no price to what bills
    doc = {"services": {"syncer": [{"provider": "x", "usd_h": 1.0, "source": "s", "as_of": "2026-01-01",
                                    "grade": "illustrative"}]}}
    assert prices.parse(doc).services[("syncer", "x")].grade == prices.ILLUSTRATIVE
    assert prices.parse({"services": {"_doc": "x"}}).services == {}


S = {"provider": "aws", **G}


@pytest.mark.parametrize("bad, why", [
    ([{"usd_h": 1.0, "source": "s", **S}], "as_of is required"),
    ([{"usd_h": 1.0, "source": "s", "as_of": "2026-01-01", **G}], "provider is required"),
    ([{"usd_h": -1, "source": "s", "as_of": "2026-01-01", **S}], "usd_h"),
    ([{"usd_h": 1.0, "source": "s", "as_of": "soon", **S}], "ISO date"),
    ([{"usd_h": 1.0, "source": "s", "as_of": "2026-01-01", "provider": "aws", "grade": "guess"}], "grade must be"),
    ([{"usd_h": 1.0, "source": "s", "as_of": "2026-01-01", "cpus": 8, **S}], "unknown keys"),
    ([{"usd_h": 1.0, "source": "s", "as_of": "2026-01-01", **S}] * 2, "two rows for aws"),
    ({"usd_h": 1.0, "source": "s", "as_of": "2026-01-01", **S}, "a list of rows"),
])
def test_a_services_row_without_its_provider_source_date_or_a_sane_price_is_refused(bad, why):
    with pytest.raises(prices.PriceTableError, match=why):
        prices.parse({"services": {"syncer": bad}})


def test_lookup_prefers_the_region_and_spotness_asked_and_never_quotes_spot_for_unknown():
    assert prices.node_usd_h("nebius", "H100", 8, spot=True, region="eu-north1") == pytest.approx(6.32)
    assert prices.node_usd_h("nebius", "H100", 8, spot=False, region="eu-north1") == pytest.approx(36.0)
    assert prices.node_usd_h("nebius", "H100", 8, region="eu-north1") == pytest.approx(36.0)
    assert prices.node_usd_h("modal", "h100!", 2) == pytest.approx(7.9)
    assert prices.node_usd_h("modal", "MI999", 1) is None
    # no region: only the illustrative list row matches, and that is no price to anything that
    # counts money (never the regional spot price either)
    assert prices.node_usd_h("nebius", "H100", 8) is None
    assert prices.node_usd_h("aws", "A100", 1) is None  # the "*" row is illustrative
    assert prices.lookup("nebius", "H100", grade=None).usd_h == pytest.approx(2.95)
    assert prices.lookup("aws", "A100", grade=None).provider == "*"


def test_the_older_tables_are_views_of_the_unified_one():
    assert offers.PRICES_8GPU == {("H100", "eu-north1"): (36.0, 6.32), ("H200", "eu-west1"): (43.2, 11.2),
                                  ("H200", "eu-north1"): (43.2, 16.0), ("H200", "us-central1"): (43.2, 39.92)}
    assert billing.DEFAULT_MODAL_USD_PER_GPU_HOUR == prices.per_gpu_usd_h("modal")
    assert billing.DEFAULT_MODAL_USD_PER_GPU_HOUR["b300"] == pytest.approx(8.5)
    assert (billing.DEFAULT_NC_UCR_PER_GPU_HOUR, billing.DEFAULT_NC_USD_PER_UCR) == (3_000_000, 1e-6)
    from yeto.launch.nc import client

    assert client.hourly_cost_ucr() == 24_000_000 == prices.nc_ucr_per_gpu_h() * prices.nc_node_gpus()
    assert prices.nc_node_usd_h() == pytest.approx(24.0)
    from yeto.launch.rollout.plan import PlanInputs

    assert PlanInputs().island_usd_h == pytest.approx(24.0)


def test_the_dashboard_default_table_keeps_its_prices_and_names_its_source():
    from yeto.dashboard.cost import load_prices, price_per_gpu_h

    table = load_prices(None)
    assert "unified price table" in table["_source"]
    expect = {"modal:H100": 3.95, "modal:H200": 4.54, "nebius:H100": 2.95, "nebius:H200": 3.5, "aws:H100": 6.88,
              "verda:H100": 2.29, "verda:H200": 2.99, "*:A100": 1.79, "modal:B300": 8.5, "modal:L40S": 1.95,
              "modal:L4": 0.8, "modal:A10G": 1.1, "nc:MI355X": 3.0}
    for key, usd in expect.items():
        assert table["prices"][key] == pytest.approx(usd), key
    assert price_per_gpu_h(table, "nc", "MI355X") == pytest.approx(3.0)
    assert table["host_prices"]["modal"] == {"cpu_core_h": 0.04716, "mem_gib_h": 0.007992}
    # the illustrative rows are shown, marked as estimates
    from yeto.dashboard.cost import is_estimate, island_cost

    assert set(table["estimate_keys"]) == {"nebius:H100", "nebius:H200", "aws:H100", "verda:H100", "verda:H200",
                                           "*:A100"}
    assert "estimate" in table["_source"]
    assert is_estimate(table, "nebius", "H100") and is_estimate(table, "gcp", "A100")
    assert not is_estimate(table, "modal", "H100") and not is_estimate(table, "nc", "MI355X")
    isl = {"id": "i", "cloud": "verda", "gpu": "H100", "gpus": 8}
    assert island_cost(isl, table, 0.0)["estimate"] is True
    assert island_cost(dict(isl, cloud="modal"), table, 0.0)["estimate"] is False


def test_a_stale_price_is_named_with_its_source_and_date():
    row = prices.lookup("modal", "H100")
    assert prices.stale([row], today=dt.date.fromisoformat(row.as_of)) == []
    lines = prices.stale([row], today=dt.date.fromisoformat(row.as_of) + dt.timedelta(days=200))
    assert len(lines) == 1 and row.source in lines[0] and row.as_of in lines[0] and "200 days" in lines[0]


# -- trainer and sampler: one price per card --------------------------------------------


@pytest.mark.parametrize("gpu, n", [("L40S", 1), ("H100", 8), ("B300", 8), ("A100-80GB", 2)])
def test_a_modal_trainer_and_a_modal_sampler_pool_of_the_same_gpus_cost_the_same(gpu, n):
    from yeto.launch.nc import cli
    from yeto.launch.pool import modal_provider

    trainer = billing.default_modal_rate(gpu, n)  # the launcher's Modal trainer (modal_runner)
    assert billing.modal_shapes_rate([(gpu, n)]) == pytest.approx(trainer)
    section = {"kind": "modal", "gpu": gpu, "gpus": n}
    assert modal_provider.table_price(section) == pytest.approx(trainer)
    assert cli.pool_price({"shard_base": "s", "provider": section}) == (pytest.approx(trainer), "usd")


def test_the_shape_planner_prices_modal_gpus_from_the_same_rows():
    from yeto.shape import providers as shape

    for gpu in shape.MODAL_PLANNER_GPUS:
        assert shape.MODAL_GPU_USD_PER_SECOND[gpu] * 3600 == pytest.approx(prices.lookup("modal", gpu).usd_per_gpu_h)
    assert shape.MODAL_PRICES_RECORDED == min([prices.lookup("modal", g).as_of for g in shape.MODAL_PLANNER_GPUS]
                                              + [prices.host("modal").as_of])


# -- sky offers: priced by the offer an order lands on ------------------------------------

SPOT = {"infra": "nebius/eu-north1", "accelerators": "H100:8", "use_spot": True}
ONDEMAND = {"infra": "nebius/eu-north1", "accelerators": "H100:8", "use_spot": False}


def test_an_offer_priced_only_by_an_illustrative_row_stays_unpriced_and_a_capped_run_refuses_it(tmp_path):
    """nebius H100:8 without a region matches only the illustrative list row: like on main (no
    price), the offer is unpriced, and a run with a spend cap refuses the pool naming the offer."""
    from yeto.launch.nc import cli
    from yeto.rl import up_consistency

    bare = {"infra": "nebius", "accelerators": "H100:8", "use_spot": False}
    spot = dict(bare, use_spot=True)
    assert [o.price_usd_h for o in offers.offers_from_config({"offers": [{"resources": bare}, {"resources": spot}]})] \
        == [None, None]
    cfg = {"shard_base": "s", "provider": {"kind": "sky", "offers": [{"resources": SPOT}, {"resources": bare}]}}
    assert cli.pool_price(cfg) == (None, "usd")
    assert cli.unpriced_offers(cfg) == ["H100:8@nebius/od"]
    problem = up_consistency.price_problem("spot", None, cfg)
    assert problem is not None and "H100:8@nebius/od" in problem and "no billing price" in problem
    assert up_consistency.price_problem("spot", None, {"shard_base": "s", "provider": {
        "kind": "sky", "offers": [{"resources": SPOT}, {"resources": ONDEMAND}]}}) is None


def test_an_offer_without_a_price_takes_the_table_price_of_its_node():
    got = offers.offers_from_config({"offers": [{"resources": SPOT}, {"resources": ONDEMAND, "price_usd_h": 30}]})
    assert [o.price_usd_h for o in got] == [pytest.approx(6.32), 30]
    one = offers.offers_from_config({"resources": {"cloud": "aws", "accelerators": "L4:1"}})
    assert one[0].price_usd_h is None  # not in the table: unpriced, never $0


def test_a_sky_pool_budgets_its_dearest_offer_not_its_first():
    from yeto.launch.nc import cli

    cfg = {"shard_base": "s", "provider": {"kind": "sky", "offers": [{"resources": SPOT}, {"resources": ONDEMAND}]}}
    assert cli.pool_price(cfg) == (pytest.approx(36.0), "usd")
    unpriced = {"shard_base": "s", "provider": {"kind": "sky", "offers": [
        {"resources": SPOT}, {"resources": {"cloud": "aws", "accelerators": "L4:1"}}]}}
    assert cli.pool_price(unpriced) == (None, "usd")
    rows = cli.pool_price_rows(cfg)
    assert [r.key for r in rows] == ["nebius:H100x8@eu-north1/spot", "nebius:H100x8@eu-north1/od"]
    assert cli.pool_price_rows({"shard_base": "s", "provider": {"kind": "nc"}}) == []


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    from yeto.rl import up

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=time.time, monotonic=time.monotonic, sleep=lambda s: None, gmtime=time.gmtime, strftime=time.strftime))
    pool_cfg = tmp_path / "pool.json"

    def build(provider, members, now):
        pool_cfg.write_text(json.dumps({"shard_base": "s", "provider": provider}))
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
               "trainer": {"argv": ["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"], "usd_per_h": 0},
               "first_group": {"timeout_s": 0, "poll_s": 0}, "budget": {"nc_usd_per_h": 24}}
        run = up.Run(cfg)
        nc_dir = tmp_path / "nc"
        nc_dir.mkdir(exist_ok=True)
        run.env["YETO_NC_STATE_DIR"] = str(nc_dir)
        (nc_dir / "pool-s.json").write_text(json.dumps({"members": members}))
        monkeypatch.setattr(up.time, "time", lambda: now)
        run.state["t0"] = now - 7200
        return run

    return build


def test_an_on_demand_fallback_is_billed_at_its_own_price_by_the_run_as_by_the_pool(make_run):
    """The spot offer failed, the pool ordered the on-demand one: the pool charged that member
    36 USD/h (hourly_ucr); run up counts the same, not the first (spot) offer's 6.32."""
    from yeto.launch.pool.money import to_ucr

    now = 100_000.0
    od, spot = to_ucr(36.0, "usd"), to_ucr(6.32, "usd")
    members = [{"state": "ready", "granted": now - 3600, "hourly_ucr": int(round(od))},
               {"state": "ready", "granted": now - 1800, "hourly_ucr": int(round(spot))},
               {"state": "starting", "granted": now - 600}]  # not priced by the pool yet: the dearest offer
    run = make_run({"kind": "sky", "offers": [{"resources": SPOT}, {"resources": ONDEMAND}]}, members, now)
    assert run.spend_usd() == pytest.approx(36.0 + 6.32 / 2 + 36.0 / 6)
    hours, rate = run.pool_spend_rate(0)
    assert hours * rate == pytest.approx(run.spend_usd())


def test_a_national_compute_pool_keeps_budget_nc_usd_per_h(make_run):
    now = 100_000.0
    run = make_run({"kind": "nc"}, [{"state": "ready", "granted": now - 3600, "hourly_ucr": 99_000_000}], now)
    assert run.spend_usd() == pytest.approx(24.0)


# -- provider capabilities from yeto/cloud/capabilities.json --------------------------


def test_each_kind_reads_its_notice_lease_and_cold_start_from_the_capability_table():
    from yeto.launch.pool import provider

    nc, modal, sky = provider.provider_caps("nc"), provider.provider_caps("modal"), provider.provider_caps("sky")
    assert (nc.lease_bounded, nc.lease_s, nc.renewable, nc.notice_s, nc.cold_start_s, nc.hard_end, nc.currency) == \
        (True, 14400.0, False, 0.0, 3600.0, "lease", "ucr")
    assert (modal.lease_bounded, modal.lease_s, modal.notice_s, modal.cold_start_s, modal.hard_end) == \
        (False, None, 30.0, 900.0, "function_timeout")
    assert (sky.notice_from_cloud, sky.cold_start_s, sky.hard_end) == (True, 1800.0, None)
    assert provider.notice_s_of("sky", "aws") == 120.0 and provider.notice_s_of("sky", "nebius") == 60.0
    assert provider.notice_s_of("sky", "verda") == 0.0 and provider.notice_s_of("sky") == 0.0
    assert provider.notice_s_of("modal") == 30.0
    for kind in ("daytona", "ssh"):
        caps = provider.provider_caps(kind)
        assert (caps.lease_bounded, caps.notice_s, caps.currency) == (False, 0.0, "usd")
    assert provider.provider_caps("ssh").cold_start_s == 300.0


def test_the_registry_and_the_provider_classes_hold_one_copy_of_the_caps():
    from yeto.launch import ssh_sampler_pool
    from yeto.launch.pool import daytona_provider, modal_provider, sky_provider
    from yeto.launch.nc import provider as nc_provider
    from yeto.launch.pool import provider

    classes = {"nc": nc_provider.NCProvider, "daytona": daytona_provider.DaytonaProvider,
               "sky": sky_provider.SkyProvider, "modal": modal_provider.ModalProvider,
               "ssh": ssh_sampler_pool.SshNodeProvider}
    for kind, cls in classes.items():
        assert provider.registered(kind).caps == cls.caps == provider.provider_caps(kind), kind
    assert provider.provider_caps("no-such-kind") == provider.ProviderCaps()


def test_a_capability_row_without_a_source_is_refused(tmp_path):
    from yeto.cloud import capabilities

    table = json.loads(capabilities.TABLE_PATH.read_text())
    del table["providers"]["nc"]["lease_s"]["source"]
    path = tmp_path / "caps.json"
    path.write_text(json.dumps(table))
    with pytest.raises(capabilities.CapabilityTableError, match="providers.nc.lease_s"):
        capabilities.load_table(str(path))


def test_a_sampler_pool_sky_cluster_is_billed_at_its_offer_price_times_its_nodes():
    from yeto.cost.billing import Rates, resource_costs
    from yeto.cost.ledger import Resource

    one = Resource("sky_cluster", "yeto-smp-r-1", "yeto-smp-r-1", 0.0,
                   {"offer": "h100", "price_usd_h": 6.32, "pool": True}, released_at=3600.0)
    group = Resource("sky_cluster", "yeto-smp-r-2", "yeto-smp-r-2", 0.0,
                     {"offer": "h100", "price_usd_h": 6.32, "pool": True, "num_nodes": 2}, released_at=1800.0)
    stated = Resource("sky_cluster", "yeto-smp-r-3", "yeto-smp-r-3", 0.0,
                      {"price_usd_h": 6.32, "usd_per_hour": 9.0, "num_nodes": 2}, released_at=3600.0)
    items = {i.name: i for i in resource_costs([one, group, stated], 7200.0, rates=Rates()).items}
    assert items["yeto-smp-r-1"].priced and items["yeto-smp-r-1"].rate_usd_per_hour == pytest.approx(6.32)
    assert items["yeto-smp-r-1"].estimated_usd == pytest.approx(6.32)
    assert items["yeto-smp-r-2"].rate_usd_per_hour == pytest.approx(12.64)
    assert items["yeto-smp-r-2"].estimated_usd == pytest.approx(6.32)
    assert items["yeto-smp-r-3"].rate_usd_per_hour == pytest.approx(9.0)  # a stated rate wins


def test_a_syncer_vm_is_billed_from_its_order_to_its_release_at_its_syncer_row():
    """A per-island fleet's syncer launch orders a sky VM of its own: billed at the price table's
    syncer row of its cloud, from its order to its confirmed release, never unpriced (0 in the
    run's spend). A ledger written before it carried a price is billed the same way; a cloud
    without a syncer row is billed at the dearest syncer row (conservative, said so)."""
    from yeto.cost.billing import Rates, resource_costs
    from yeto.cost.ledger import Resource

    aws = prices.syncer_usd_h("aws")
    assert aws and aws > 0
    recorded = Resource("sky_cluster", "r-syncer", "r-syncer", 0.0,
                        {"role": "syncer", "cloud": "aws", "usd_per_hour": aws, "price_note": "price table syncer@aws"},
                        released_at=7200.0)
    old = Resource("sky_cluster", "o-syncer", "o-syncer", 0.0, {"role": "syncer"})  # still open
    elsewhere = Resource("sky_cluster", "n-syncer", "n-syncer", 0.0, {"role": "syncer", "cloud": "nebius"},
                         released_at=3600.0)
    summary = resource_costs([recorded, old, elsewhere], 10800.0, rates=Rates())
    items = {i.name: i for i in summary.items}
    assert not summary.unpriced
    assert items["r-syncer"].rate_usd_per_hour == pytest.approx(aws)
    assert items["r-syncer"].estimated_usd == pytest.approx(2 * aws)  # order to release, not to now
    dearest = max(r.usd_h for (name, _p), r in prices.table().services.items() if name == prices.SYNCER)
    # no cloud recorded (an older ledger): not known, so counted at the dearest syncer row
    assert items["o-syncer"].rate_usd_per_hour == pytest.approx(dearest)
    assert items["o-syncer"].estimated_usd == pytest.approx(3 * dearest)  # open: to now
    assert items["n-syncer"].rate_usd_per_hour == pytest.approx(dearest) and dearest >= aws
    assert "dearest syncer row" in items["n-syncer"].note


def test_the_syncer_vm_price_is_never_zero_and_says_when_it_is_conservative():
    rate, note, conservative = prices.syncer_vm_usd_h("aws")
    assert rate == prices.syncer_usd_h("aws") and not conservative and "syncer@aws" in note
    rate, note, conservative = prices.syncer_vm_usd_h("verda")
    assert rate and rate > 0 and conservative and "verda" in note
