"""Group orders (yeto.launch.pool.group): n nodes of one shape on one fabric, all or none.

A fake provider bills each ticket from its order to its end; a group whose node fails
is released whole and billed only for the time each node was open. The sky, Modal and
inventory providers place a group natively (one multi-node cluster, one clustered
call, n inventory nodes of one fabric)."""

from __future__ import annotations

import pytest

from test_nc_sky_provider import FakeSky
from test_rollout_modal_provider import FakeClient
from test_ssh_sampler_pool import npu_node, provider_section
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.pool.modal_provider import ModalProvider
from yeto.launch.pool.sky_provider import SkyProvider
from yeto.launch.pool import group as g
from yeto.launch.pool.status import LeaseStatus

SHAPE = g.GroupShape("H100", 8)


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProvider:
    """Per-node machines: a ticket bills from its order to its end (``bill``). A node is
    active ``up_after`` seconds after its order; ``fail`` slots end instead (``fail_after``
    seconds), ``refuse`` slots raise at order, ``stuck`` releases fail."""

    kind = "fake"

    def __init__(self, clock, slots=(1, 2, 3, 4), *, up_after=30.0, fail=(), fail_after=60.0, refuse=(),
                 stuck=(), fabrics=("tcp",)):
        self.clock, self.slots, self.fabrics = clock, list(slots), fabrics
        self.up_after, self.fail, self.fail_after, self.refuse, self.stuck = (
            up_after, set(fail), fail_after, set(refuse), set(stuck))
        self.rows: dict[str, dict] = {}
        self.calls: list[tuple] = []

    def order(self, slot):
        self.calls.append(("order", slot))
        if slot in self.refuse:
            raise RuntimeError(f"no capacity for slot {slot}")
        ticket = f"t{slot}"
        self.rows[ticket] = {"slot": slot, "opened": self.clock(), "closed": None}
        return ticket

    def status(self, slot, ticket):
        row = self.rows[ticket]
        age = self.clock() - row["opened"]
        if row["closed"] is not None:
            return LeaseStatus("ended", end_reason="released")
        if slot in self.fail and age >= self.fail_after:
            row["closed"] = self.clock()  # the cloud ended it: billing stops here
            return LeaseStatus("ended", end_reason="lost")
        return LeaseStatus("active" if age >= self.up_after and slot not in self.fail else "waiting")

    def release(self, slot, ticket):
        self.calls.append(("release", ticket))
        if slot in self.stuck:
            raise RuntimeError("delete sent but it still exists")
        self.rows[ticket]["closed"] = self.clock()

    def account_free(self, slot):
        return not any(r["slot"] == slot and r["closed"] is None for r in self.rows.values())

    def bill(self, now, usd_h=10.0):
        """What the cloud charges: every ticket from its order to its end."""
        return {t: usd_h * ((r["closed"] if r["closed"] is not None else now) - r["opened"]) / 3600
                for t, r in self.rows.items()}


def test_a_group_comes_up_whole():
    clock = Clock()
    p = FakeProvider(clock)
    group = g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep, poll_s=10)
    assert group.tickets == ["t1", "t2", "t3"] and not group.native
    assert all(m.ready_at == 30.0 for m in group.members)
    assert not [c for c in p.calls if c[0] == "release"]
    assert group.node_hours(3630.0) == pytest.approx(3 * 3630 / 3600)


def test_a_failed_node_releases_the_whole_group_and_bills_only_the_open_time():
    clock = Clock()
    p = FakeProvider(clock, fail={2}, fail_after=60.0)
    events = []
    with pytest.raises(g.GroupOrderFailed) as err:
        g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep, poll_s=10,
                      on_event=lambda name, fields: events.append(name))
    assert sorted(err.value.released) == ["t1", "t2", "t3"] and err.value.unreleased == []
    assert all(r["closed"] is not None for r in p.rows.values())  # nothing left open
    assert sorted(t for op, t in p.calls if op == "release") == ["t1", "t2", "t3"]  # an ended read is released too
    # every node is billed exactly while it was open: t2 until the cloud ended it, the
    # others until the group released them on the same poll (60 s)
    clock.sleep(3600)  # an hour later nothing has been added to the bill
    assert p.bill(clock()) == pytest.approx({"t1": 10 * 60 / 3600, "t2": 10 * 60 / 3600, "t3": 10 * 60 / 3600})
    assert {m.ticket: m.open_s(clock()) for m in err.value.members} == {"t1": 60.0, "t2": 60.0, "t3": 60.0}
    assert events == ["group_ordered", "group_failed"]


def test_an_order_that_raises_releases_the_nodes_ordered_before_it():
    clock = Clock()
    p = FakeProvider(clock, refuse={3})
    with pytest.raises(g.GroupOrderFailed, match="no capacity"):
        g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep)
    assert p.calls == [("order", 1), ("order", 2), ("order", 3), ("release", "t1"), ("release", "t2")]


def test_a_group_that_does_not_come_up_in_time_is_released():
    clock = Clock()
    p = FakeProvider(clock, up_after=10_000.0)
    with pytest.raises(g.GroupOrderFailed, match="not active after 600 s"):
        g.order_group(p, 2, SHAPE, clock=clock, sleep=clock.sleep, timeout_s=600, poll_s=30)
    assert sorted(t for op, t in p.calls if op == "release") == ["t1", "t2"]
    assert sum(p.bill(clock() + 3600).values()) == pytest.approx(2 * 10 * 600 / 3600)


def test_a_release_that_fails_is_reported_for_cleanup():
    clock = Clock()
    p = FakeProvider(clock, fail={2}, stuck={1})
    with pytest.raises(g.GroupOrderFailed, match="release FAILED") as err:
        g.order_group(p, 2, SHAPE, clock=clock, sleep=clock.sleep)
    assert err.value.unreleased == ["t1"] and err.value.released == ["t2"]


def test_an_interrupt_while_waiting_releases_the_group_and_goes_up():
    clock = Clock()
    p = FakeProvider(clock, up_after=10_000.0)

    def interrupted(s):
        clock.sleep(s)
        if clock() >= 60:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        g.order_group(p, 2, SHAPE, clock=clock, sleep=interrupted, poll_s=30)
    assert sorted(t for op, t in p.calls if op == "release") == ["t1", "t2"]
    assert all(r["closed"] is not None for r in p.rows.values())


def test_too_few_free_slots_or_a_missing_fabric_orders_nothing():
    clock = Clock()
    p = FakeProvider(clock, slots=(1, 2))
    with pytest.raises(g.GroupOrderFailed, match="only 2 free slot"):
        g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep)
    with pytest.raises(g.GroupOrderFailed, match=r"gives \['tcp'\], not ib"):
        g.order_group(p, 2, SHAPE, "ib", clock=clock, sleep=clock.sleep)
    assert p.calls == []
    rdma = FakeProvider(clock, fabrics=("tcp", "ib"))
    assert g.fabric_problem(rdma, "rdma") is None and g.fabric_problem(rdma, "efa") is not None


# -- native group orders ------------------------------------------------------------


def _sky(clock, **kw):
    sky = FakeSky()
    sky.Task.num_nodes = 1
    return sky, SkyProvider("run-g", resources={"cloud": "nebius", "accelerators": "H100:8"}, sky=sky,
                            clock=clock, **{"slots": [1, 2], **kw})


def _launches(sky):
    return [c for c in sky.calls if c[0] == "launch"]


def test_sky_places_one_cluster_of_n_nodes_on_the_best_network_tier():
    from yeto.launch.rollout.offers import Offer

    clock = Clock()
    sky, p = _sky(clock, offers=[Offer("h100", {"cloud": "nebius", "accelerators": "H100:8"}, 36.0)])
    assert p.order_group(2, SHAPE, "rdma") == [(1, "yeto-smp-run-g-1")]
    (_, name, task, kw), = _launches(sky)
    assert task.resources["network_tier"] == "best" and task.num_nodes == 2
    assert kw["down"] and kw["wait_for"] == sky.AutostopWaitFor.NONE  # the pool's idle-autodown keywords
    assert p.hourly_cost(name) == 72.0  # two nodes of the offer


def test_sky_group_through_the_driver_and_a_failed_launch_downs_what_it_left():
    clock = Clock()
    sky, p = _sky(clock, slots=[1])
    launch = sky.launch

    def launched_then_up(task, cluster_name, **kw):
        request = launch(task, cluster_name, **kw)
        assert task.num_nodes == 3
        sky.up(cluster_name)
        return request

    sky.launch = launched_then_up
    group = g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep)
    assert group.native and group.tickets == ["yeto-smp-run-g-1"] and group.members[0].nodes == 3
    assert "network_tier" not in _launches(sky)[0][2].resources  # a tcp group asks for no network tier
    p.release(1, group.tickets[0])

    def half_launched(task, cluster_name, **kw):
        launch(task, cluster_name, **kw)
        raise RuntimeError("capacity: 1 of 3 nodes")

    sky.launch = half_launched
    with pytest.raises(g.GroupOrderFailed, match="capacity"):
        g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep)
    assert ("down", "yeto-smp-run-g-2") in sky.calls and sky.clusters == {}
    assert p.run_tickets() == []


def test_modal_reserves_one_clustered_call():
    client = FakeClient()
    p = ModalProvider("run-m", slots=[1, 2], client=client, price_usd_h=31.6)
    group = g.order_group(p, 2, SHAPE, "rdma", sleep=lambda s: None)
    (ticket,) = group.tickets
    assert group.native and group.members[0].nodes == 2 and client.log == []  # nothing spawned or billed
    assert p.group_of(ticket) == {"size": 2, "rdma": True}
    assert p.hourly_cost(ticket) == pytest.approx(63.2)
    with pytest.raises(RuntimeError, match="single-container calls only"):
        p.spawn(ticket, {"shard": "m-0"})
    single = p.order(2)
    assert p.group_of(single) is None and p.hourly_cost(single) == pytest.approx(31.6)


def _inventory(tmp_path, fabrics):
    nodes = [npu_node(name=f"n{i}", host=f"n{i}", fabric=fab) for i, fab in enumerate(fabrics)]
    conf = sp.parse_config({"provider": provider_section(nodes=[n.name for n in nodes])}, inventory=nodes)
    return sp.SshNodeProvider("r", conf, slots=list(range(len(nodes))), state_path=tmp_path / "s.json")


def test_inventory_picks_free_nodes_of_one_fabric(tmp_path):
    p = _inventory(tmp_path, ["tcp", "roce", "tcp", "roce", "roce"])
    shape = g.GroupShape("910B4", 8)
    assert p.fabrics == ("rdma", "roce", "tcp")
    group = g.order_group(p, 2, shape, "roce", sleep=lambda s: None)
    assert [p.node(m.slot, m.ticket).name for m in group.members] == ["n1", "n3"]
    group2 = g.order_group(p, 1, shape, "rdma", sleep=lambda s: None)
    assert p.node(group2.members[0].slot, group2.members[0].ticket).name == "n4"
    with pytest.raises(g.GroupOrderFailed, match="no 3 free"):
        g.order_group(p, 3, shape, "tcp", sleep=lambda s: None)  # two tcp nodes left: one fabric only
    with pytest.raises(g.GroupOrderFailed, match="no 1 free"):
        g.order_group(p, 1, g.GroupShape("H100", 8), sleep=lambda s: None)  # no such card
    for m in group.members:
        p.release(m.slot, m.ticket)
    assert p.account_free(1) and p.account_free(3)


def test_a_sky_group_stuck_starting_is_downed_although_it_reads_ended():
    """A cluster in INIT past start_timeout_s reads ended/lost but still exists (and bills):
    the group releases it, and its offer cools down so the next attempt takes the next offer."""
    from yeto.launch.rollout.offers import Offer

    clock = Clock()
    offers = [Offer("a", {"cloud": "aws", "accelerators": "H100:8"}, 50.0),
              Offer("b", {"cloud": "gcp", "accelerators": "H100:8"}, 60.0)]
    sky, p = _sky(clock, slots=[1], offers=offers, start_timeout_s=600)
    with pytest.raises(g.GroupOrderFailed, match="ended \\(lost\\)") as err:
        g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep, poll_s=60, timeout_s=3600)
    assert err.value.released == ["yeto-smp-run-g-1"] and err.value.unreleased == []
    assert ("down", "yeto-smp-run-g-1") in sky.calls and sky.clusters == {} and p.run_tickets() == []
    # attempts: the first group sticks in INIT, the second comes up on the other offer
    launch = sky.launch

    def up_on_b(task, cluster_name, **kw):
        request = launch(task, cluster_name, **kw)
        if task.resources["cloud"] == "gcp":
            sky.up(cluster_name)
        return request

    sky.launch = up_on_b
    group = g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep, poll_s=60, timeout_s=3600, attempts=2)
    assert sky.clusters[group.tickets[0]]["status"] == "UP"
    assert [c[2].resources["cloud"] for c in _launches(sky)][-1] == "gcp"


def test_a_native_release_that_fails_never_hides_the_order_error():
    clock = Clock()
    sky, p = _sky(clock, slots=[1])
    launch = sky.launch

    def half_launched(task, cluster_name, **kw):
        launch(task, cluster_name, **kw)
        raise RuntimeError("capacity: 1 of 3 nodes")

    def down_fails(name):
        raise ConnectionError("api server gone")

    sky.launch, sky.down = half_launched, down_fails
    with pytest.raises(g.GroupOrderFailed, match="capacity.*release FAILED") as err:
        g.order_group(p, 3, SHAPE, clock=clock, sleep=clock.sleep)
    assert err.value.unreleased == ["yeto-smp-run-g-1"] and isinstance(err.value.__cause__, g.GroupOrderFailed)


def test_inventory_order_failure_reports_the_nodes_it_could_not_release(tmp_path):
    p = _inventory(tmp_path, ["roce", "roce", "roce"])
    order = p.order

    def third_fails(slot):
        if slot == 2:
            raise RuntimeError("record busy")
        return order(slot)

    release = p.release

    def first_stuck(slot, ticket):
        if slot == 0:
            raise RuntimeError("record busy")
        return release(slot, ticket)

    p.order, p.release = third_fails, first_stuck
    with pytest.raises(g.GroupOrderFailed, match="record busy") as err:
        g.order_group(p, 3, g.GroupShape("910B4", 8), "roce", sleep=lambda s: None)
    assert len(err.value.unreleased) == 1 and len(err.value.released) == 1


def test_sky_rdma_takes_the_clouds_tier_and_refuses_a_shape_without_one():
    from yeto.launch.rollout.offers import Offer

    clock = Clock()
    sky, p = _sky(clock, offers=[Offer("l40s", {"cloud": "nebius", "accelerators": "L40S:1"}, 2.0)])
    with pytest.raises(g.GroupOrderFailed, match="no RDMA network tier"):
        g.order_group(p, 2, g.GroupShape("L40S", 1), "rdma", clock=clock, sleep=clock.sleep)
    assert _launches(sky) == []
    p.order_group(2, g.GroupShape("L40S", 1), "tcp")  # over TCP it is fine, and asks for no tier
    assert "network_tier" not in _launches(sky)[-1][2].resources


def test_a_shape_no_offer_or_function_has_is_refused_before_any_order():
    clock = Clock()
    sky, p = _sky(clock)
    with pytest.raises(g.GroupOrderFailed, match="no offer of 8 x H200"):
        g.order_group(p, 2, g.GroupShape("H200", 8), clock=clock, sleep=clock.sleep)
    assert sky.calls == []
    m = ModalProvider("run-m", slots=[1], client=FakeClient(), config={"gpu": "H100", "gpus": 8})
    with pytest.raises(g.GroupOrderFailed, match="asks for 4 x H100"):
        g.order_group(m, 2, g.GroupShape("H100", 4), sleep=lambda s: None)
    assert m.run_tickets() == []
