"""A Modal app costs its GPUs + the CPU and memory it requests + what it sends out of Modal.

Modal bills each of those apart (modal.com/pricing). The numbers of the reconciliation
tests are the Modal billing report's per-resource rows of a 1xL40S verl trainer app
(8 cores, 64 GiB, publishing its policy to an object store outside Modal every round).
All CPU: fake billing, fake tapes, fake ``modal`` module.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from yeto.cost import billing, egress, prices
from yeto.cost.billing import BillingRow, ModalBilling, resource_costs
from yeto.cost.egress import EgressMeter, Sample
from yeto.cost.guard import CostGuard, GuardConfig, TapeTail
from yeto.cost.ledger import Resource, ResourceLedger

H = 3600.0
T0 = 1_800_000_000.0 - (1_800_000_000.0 % H)  # an hour boundary
CPU_H, MEM_H, EGRESS_GIB = 0.04716, 0.007992, 0.04
GIB = float(1 << 30)
L40S_8C_64G = 1.95 + 8 * CPU_H + 64 * MEM_H  # 2.8388 USD/h


def app(created, released=None, *, egress_events=("rl_publication", "rl_member_publication", "rl_fragment_push"),
        app_id="ap-1", **attrs):
    attrs = {"gpu": "L40S", "gpus": 1, "cpu": 8, "memory_gib": 64, "usd_per_hour": L40S_8C_64G, **attrs}
    if egress_events:
        attrs["egress_events"] = list(egress_events)
    return Resource(kind="modal_app", id=app_id, name="yeto-run", created_at=created, released_at=released,
                    attrs=attrs)


def fixed(rows):
    return ModalBilling(lambda start, end: list(rows))


# -- the price table ---------------------------------------------------------------------


def test_the_modal_host_row_carries_cpu_memory_and_egress_with_source_and_date():
    h = prices.host("modal")
    assert (h.cpu_core_h, h.mem_gib_h, h.egress_gib) == (CPU_H, MEM_H, EGRESS_GIB)
    assert "modal.com/pricing" in h.source and h.as_of == "2026-11-26"
    assert prices.modal_host_usd_h(8, 64) == pytest.approx(8 * CPU_H + 64 * MEM_H)
    # an ephemeral disk bills as memory at 20:1 when that is more than the memory request
    assert prices.modal_billed_memory_gib(64, 3000) == 150 and prices.modal_billed_memory_gib(64, 500) == 64
    assert prices.modal_container_usd_h("L40S", 1, cpu=8, memory_gib=64) == pytest.approx(L40S_8C_64G)
    assert prices.modal_container_usd_h("MI999", 1, cpu=8, memory_gib=64) is None


@pytest.mark.parametrize("bad", [-0.1, "0.04", True])
def test_a_bad_egress_price_is_refused(bad):
    doc = {"rows": [], "host": {"modal": {"cpu_core_h": 1, "mem_gib_h": 1, "egress_gib": bad, "source": "s",
                                          "as_of": "2026-11-26"}}}
    with pytest.raises(prices.PriceTableError, match="egress_gib"):
        prices.parse(doc)


def test_a_host_row_without_egress_has_none():
    doc = {"rows": [], "host": {"x": {"cpu_core_h": 1, "mem_gib_h": 1, "source": "s", "as_of": "2026-11-26"}}}
    assert prices.parse(doc).host["x"].egress_gib is None


# -- list rate: GPU + CPU + memory -----------------------------------------------------------


def test_the_list_rate_of_an_l40s_trainer_matches_its_bill_lines():
    # one billed hour of a 1xL40S trainer app (8 cores, 64 GiB) that ran 58.5 min of it:
    # L40S 1.900, CPU 0.369, memory 0.499 (USD) -> per hour 1.95 / 0.378 / 0.512
    ran_h = 1.900 / 1.95
    assert billing.default_modal_rate("L40S", 1, 8, 64) * ran_h == pytest.approx(1.900 + 0.369 + 0.499, rel=0.01)
    assert billing.default_modal_rate("L40S", 1) == pytest.approx(1.95 + 4 * CPU_H + 32 * MEM_H)


def test_a_ledger_entry_without_a_rate_is_priced_with_its_host_shape():
    r = Resource(kind="modal_app", id="ap-1", name="x", created_at=T0, released_at=T0 + H,
                 attrs={"gpu": "L40S", "gpus": 1, "cpu": 8, "memory_gib": 64})
    (item,) = resource_costs([r], T0 + H).items
    assert item.usd == pytest.approx(L40S_8C_64G) and "GPU + CPU + memory" in item.note


def test_a_price_file_adds_its_modal_host_prices():
    rates = billing.Rates(prices={"prices": {"modal:L40S": 2.0}, "host_prices": {"modal": {"cpu_core_h": 0.1,
                                                                                            "mem_gib_h": 0.01}}})
    assert rates.table_rate("modal", {"gpu": "L40S", "gpus": 1, "cpu": 8, "memory_gib": 64}) == pytest.approx(
        2.0 + 0.8 + 0.64)
    assert rates.table_rate("nebius", {"gpu": "L40S", "gpus": 1}) is None


# -- the billed-hour floor -------------------------------------------------------------------


def test_the_unbilled_tail_never_runs_below_the_app_s_dearest_billed_hour():
    r = app(T0 + 30 * 60, egress_events=())
    rows = [BillingRow("ap-1", "x", T0, 2.5)]  # 2.5 USD over its 30 min in that hour: 5 USD/h
    (item,) = resource_costs([r], T0 + 2 * H, billing=fixed(rows)).items
    assert item.rate_usd_per_hour == pytest.approx(5.0)
    assert item.usd == pytest.approx(2.5 + 5.0) and "above the list rate" in item.note
    # a billed hour cheaper than the list rate leaves the list rate
    (item,) = resource_costs([r], T0 + 2 * H, billing=fixed([BillingRow("ap-1", "x", T0, 0.5)])).items
    assert item.rate_usd_per_hour == pytest.approx(L40S_8C_64G)


def test_a_short_overlap_does_not_set_the_floor():
    r = app(T0 + 50 * 60, egress_events=())  # 10 min in its first hour: too short to read a rate from
    (item,) = resource_costs([r], T0 + 2 * H, billing=fixed([BillingRow("ap-1", "x", T0, 5.0)])).items
    assert item.rate_usd_per_hour == pytest.approx(L40S_8C_64G)


def test_the_floor_leaves_out_billed_egress_when_the_tape_counts_it():
    r = app(T0)
    rows = [BillingRow("ap-1", "x", T0, 6.0, egress=3.0)]  # 3 USD/h without egress
    (item,) = resource_costs([r], T0 + 2 * H, billing=fixed(rows), egress=[]).items
    assert item.rate_usd_per_hour == pytest.approx(3.0)
    # no breakdown: the whole row (high, never low)
    (item,) = resource_costs([r], T0 + 2 * H, billing=fixed([BillingRow("ap-1", "x", T0, 6.0)]), egress=[]).items
    assert item.rate_usd_per_hour == pytest.approx(6.0)
    # an app whose tape egress is not counted keeps billed egress in its floor
    (item,) = resource_costs([app(T0, egress_events=())], T0 + 2 * H, billing=fixed(rows)).items
    assert item.rate_usd_per_hour == pytest.approx(6.0)


# -- egress from the tape --------------------------------------------------------------------


def pub(t, size, event="rl_publication", **extra):
    field = egress.BYTES_FIELD[event]
    return {"event": event, "time_unix": t, field: size, **extra}


def test_tape_egress_is_counted_after_the_newest_billed_hour_only():
    r = app(T0)
    meter = EgressMeter()
    for t in (T0 + 600, T0 + H + 600, T0 + H + 1200):
        meter.observe(pub(t, GIB), t)
    rows = [BillingRow("ap-1", "x", T0, 3.0, egress=0.04)]
    (item,) = resource_costs([r], T0 + 2 * H, billing=fixed(rows), egress=meter.samples).items
    tail_egress = 2 * egress.WIRE_OVERHEAD * EGRESS_GIB
    # billed + the tail hour at the floor without its billed egress (2.96 USD/h) + 2 GiB from the tape
    assert item.usd == pytest.approx(3.0 + 2.96 + tail_egress)
    assert "egress 2.1 GiB" in item.note


def test_tape_egress_counts_only_the_events_the_app_names():
    meter = EgressMeter()
    meter.observe(pub(T0 + 60, GIB), T0 + 60)
    meter.observe(pub(T0 + 120, GIB, "rl_fragment_push"), T0 + 120)
    priced = lambda r: resource_costs([r], T0 + H, egress=meter.samples).items[0].usd  # noqa: E731
    base = L40S_8C_64G
    assert priced(app(T0, egress_events=())) == pytest.approx(base)
    assert priced(app(T0, egress_events=("rl_fragment_push",))) == pytest.approx(
        base + egress.WIRE_OVERHEAD * EGRESS_GIB)
    assert priced(app(T0)) == pytest.approx(base + 2 * egress.WIRE_OVERHEAD * EGRESS_GIB)
    # no meter at all (a caller without tapes): nothing counted
    assert resource_costs([app(T0)], T0 + H).items[0].usd == pytest.approx(base)


def test_a_sample_counts_in_one_app_and_only_inside_its_span():
    meter = EgressMeter()
    meter.observe(pub(T0 + 60, GIB), T0 + 60)
    meter.observe(pub(T0 + 2 * H + 60, GIB), T0 + 2 * H + 60)  # after both apps ended
    a, b = app(T0, T0 + H, app_id="ap-a"), app(T0, T0 + H, app_id="ap-b")
    items = resource_costs([a, b], T0 + 3 * H, egress=meter.samples).items
    assert sum(i.usd for i in items) == pytest.approx(2 * L40S_8C_64G + egress.WIRE_OVERHEAD * EGRESS_GIB)


def test_a_record_without_a_size_counts_the_largest_seen_else_the_fallback():
    samples = [Sample(T0 + 10, "rl_publication", 2 * GIB), Sample(T0 + 20, "rl_publication", None)]
    sized, unsized = egress.bytes_in(samples, ["rl_publication"], T0, T0 + H, None)
    assert sum(sized.values()) == pytest.approx(4 * GIB * egress.WIRE_OVERHEAD) and unsized == 0
    sized, unsized = egress.bytes_in(samples[1:], ["rl_publication"], T0, T0 + H, 3 * GIB)
    assert sum(sized.values()) == pytest.approx(3 * GIB * egress.WIRE_OVERHEAD)
    sized, unsized = egress.bytes_in(samples[1:], ["rl_publication"], T0, T0 + H, None)
    assert sized == {} and unsized == 1
    r = app(T0, egress_fallback_bytes=None)
    (item,) = resource_costs([r], T0 + H, egress=samples[1:]).items
    assert "without a size" in item.note


def test_a_record_mirrored_to_two_tapes_counts_once():
    meter = EgressMeter()
    rec = pub(T0 + 5, GIB, policy_version=3, island_id=0)
    meter.observe(rec, T0 + 5)
    meter.observe(dict(rec), T0 + 5)
    meter.observe(pub(T0 + 5, GIB, policy_version=4, island_id=0), T0 + 5)
    meter.observe({"event": "rl_round_trained", "time_unix": T0}, T0)
    assert len(meter.samples) == 2


@pytest.mark.parametrize("store, shards, want", [
    ("s3://bucket/runs/r", ["0:a"], ["rl_publication", "rl_member_publication", "rl_fragment_push"]),
    ("modal-volume://ckpt", ["0:a"], ["rl_fragment_push"]),
    ("/root/ckpt", ["0:a"], ["rl_fragment_push"]),
    ("s3://bucket/runs/r", None, ["rl_fragment_push"]),
    (None, ["0:a"], ["rl_fragment_push"]),
])
def test_publications_leave_modal_only_for_a_store_outside_it(store, shards, want):
    args = SimpleNamespace(rl_checkpoint_store=store, rl_buffer_shards=shards, rl_island_buffer_shards=None)
    assert egress.egress_events(args) == want


def test_the_fallback_size_is_the_model_at_fp32_and_never_asks_the_hub():
    assert egress.fallback_bytes(SimpleNamespace(model="x/unknown-model", weights_gb=1.5)) == pytest.approx(3e9)
    assert egress.fallback_bytes(SimpleNamespace(model="x/unknown-model-never-on-the-hub", weights_gb=None)) is None


# -- the reconciliation against a bill ---------------------------------------------------------

# a 1xL40S verl trainer app over 59.3 min (20:33:03 -> 21:32:23 of the bill): two billed
# hours, each broken down by resource (USD)
BILL = [(T0, {"L40S": 0.85692, "CPU": 0.16629, "Memory": 0.22500, "Network Egress": 1.13640}),
        (T0 + H, {"L40S": 1.04325, "CPU": 0.20244, "Memory": 0.27392, "Network Egress": 1.83126})]
START, END = T0 + 33 * 60 + 3, T0 + H + 32 * 60 + 23


def bill_rows(upto=None):
    return [BillingRow("ap-1", "x", t, sum(by.values()), by["Network Egress"]) for t, by in BILL
            if upto is None or t + H <= upto]


def tape():
    """Publications every minute carrying, hour by hour, the bytes the bill's egress lines
    came to (74.2 GiB in all; the publications grew later in the run)."""
    meter = EgressMeter()
    for (t0, by), lo, hi in zip(BILL, (START, T0 + H), (T0 + H, END)):
        times = [t for t in range(int(lo) + 30, int(hi), 60)]
        for t in times:
            meter.observe(pub(t, by["Network Egress"] / EGRESS_GIB * GIB / len(times), policy_version=t), t)
    return meter.samples


def test_the_estimate_of_an_egress_heavy_trainer_lands_0_to_20_percent_over_its_bill():
    actual = sum(sum(by.values()) for _t, by in BILL)  # 5.735
    r = app(START, END)
    for rows in ([], bill_rows(upto=billing.floor_hour(END))):  # no bill yet; the closed hour billed
        (item,) = resource_costs([r], END, billing=fixed(rows), egress=tape()).items
        assert 0.0 <= item.usd / actual - 1 <= 0.20, (len(rows), item.usd, actual)
    # main's estimate (the GPU line only, no egress) was 66 % low
    old = Resource(kind="modal_app", id="ap-1", name="x", created_at=START, released_at=END,
                   attrs={"gpu": "L40S", "gpus": 1, "usd_per_hour": 1.95})
    assert resource_costs([old], END).items[0].usd / actual - 1 < -0.6


# -- the production fetch --------------------------------------------------------------------


def test_the_fetch_reads_each_row_s_egress_from_the_workspace_report(monkeypatch):
    seen = {}

    class Report:
        def report(self, **kw):
            seen.update(kw)
            return [SimpleNamespace(object_id="ap-1", description="x", interval_start=T0, cost="2.5",
                                    cost_by_resource={"L40S": 1.5, "Network Egress": "1.0"})]

    class Workspace:
        @staticmethod
        def from_context():
            return SimpleNamespace(billing=Report())

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Workspace=Workspace))
    (row,) = billing.modal_billing_fetch(T0 + 10, T0 + 2 * H)
    assert (row.object_id, row.cost, row.egress, row.interval_start) == ("ap-1", 2.5, 1.0, T0)
    assert seen["resolution"] == "h" and seen["start"].timestamp() == T0


def test_the_fetch_falls_back_to_the_older_report_without_a_breakdown(monkeypatch):
    class Workspace:
        @staticmethod
        def from_context():
            raise RuntimeError("not on this server")

    old = types.SimpleNamespace(workspace_billing_report=lambda **kw: [
        {"object_id": "ap-1", "description": "x", "interval_start": T0, "cost": 2.5}])
    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Workspace=Workspace, billing=old))
    (row,) = billing.modal_billing_fetch(T0, T0 + H)
    assert (row.cost, row.egress) == (2.5, None)


# -- the guard and the fleet -----------------------------------------------------------------


def test_the_guard_counts_the_egress_its_tapes_name(tmp_path):
    led = ResourceLedger(tmp_path / "resources.jsonl")
    led.clock = lambda: T0
    led.created("modal_app", "ap-1", "yeto-run", gpu="L40S", gpus=1, cpu=8, memory_gib=64,
                usd_per_hour=L40S_8C_64G, egress_events=["rl_publication"])
    tapes = tmp_path / "events"
    tapes.mkdir()
    (tapes / "island.jsonl").write_text("".join(json.dumps(pub(T0 + 60 * i, GIB, policy_version=i)) + "\n"
                                                for i in range(1, 11)))
    g = CostGuard("run", GuardConfig(budget_usd=None), ledger=led, tapes=TapeTail([str(tapes / "*.jsonl")]),
                  clock=lambda: T0 + H, log=lambda m: None, events_path=tmp_path / "cost-events.jsonl")
    s = g.tick()
    assert s.total_usd == pytest.approx(L40S_8C_64G + 10 * egress.WIRE_OVERHEAD * EGRESS_GIB)
    assert g.tick().total_usd == pytest.approx(s.total_usd)  # the tape is read once, its samples kept


def test_a_fleet_island_on_modal_is_priced_with_the_host_its_argv_requests():
    from yeto.rl import trainer_worker

    argv = ["--gpu", "modal:1x1xl40s", "--modal-cpu", "8", "--modal-memory-gib=64"]
    assert trainer_worker.table_usd_per_h("modal:1x1xl40s", argv) == pytest.approx(L40S_8C_64G)
    assert trainer_worker.table_usd_per_h("modal:1x1xl40s") == pytest.approx(1.95 + 4 * CPU_H + 32 * MEM_H)
    assert trainer_worker.table_usd_per_h("modal:2x8xh100") == pytest.approx(2 * (8 * 3.95 + 32 * CPU_H
                                                                                 + 256 * MEM_H))
    record = {"usd_per_h": None, "gpu": "modal:1x1xl40s", "argv": argv}
    assert trainer_worker.incarnation_price(record) == pytest.approx(L40S_8C_64G)
