"""SkyPilot lease provider skeleton (yeto.launch.nc.sky_provider) over a fake sky
module: the same pool policy (events, budget, breaker, fencing) as National Compute."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, elastic
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY, IslandPool
from yeto.launch.nc.sky_provider import SkyProvider, from_config


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))


class ResourcesUnavailableError(Exception):
    """sky.exceptions.ResourcesUnavailableError's stand-in."""


class RequestId(str):
    """sky.server.common.RequestId: what every sky SDK call returns."""


class FakeSky:
    """The async sky SDK (>= 0.14) over an in-memory cluster table; never a cloud.

    Every call returns a :class:`RequestId` at once; its result (or its failure)
    comes from ``get``, which waits for the request (here: runs it). A launch request
    stays in flight (``api_status`` RUNNING) until ``get`` or ``up`` finishes it, like a
    real provisioning. ``restart_server`` empties the request table (a local API server
    restart), clusters stay. ``autostop(minutes)`` plays sky's skylet idle check."""

    StatusRefreshMode = SimpleNamespace(NONE="NONE", AUTO="AUTO", FORCE="FORCE")
    AutostopWaitFor = SimpleNamespace(JOBS_AND_SSH="jobs_and_ssh", JOBS="jobs", NONE="none")

    def __init__(self):
        self.clusters, self.calls, self.requests = {}, [], {}
        self.fail_status = False
        self.fail_launch = set()  # infra values whose launch fails (at get)
        self.cluster_request = {}

    class Task:
        def __init__(self, name=None, run=None):
            self.name, self.run, self.resources, self.envs, self.secrets = name, run, None, {}, {}

        @classmethod
        def from_yaml(cls, path):
            from pathlib import Path

            task = cls(name=f"yaml:{path}")
            if "run:" in Path(path).read_text():
                task.run = "sampler"
            return task

        def set_resources(self, r):
            self.resources = r

        def update_envs(self, envs):
            self.envs.update(envs)

        def update_secrets(self, secrets):
            self.secrets.update(secrets)

    @staticmethod
    def Resources(**kw):  # noqa: N802 - sky's name
        return dict(kw)

    # -- requests
    def _submit(self, fn, *, done=True):
        rid = RequestId(f"req-{len(self.requests) + 1}")
        self.requests[rid] = {"status": "RUNNING", "fn": fn}
        if done:
            self._run(rid)
        return rid

    def _run(self, rid):
        req = self.requests[rid]
        if req["status"] != "RUNNING":
            return
        try:
            req["result"], req["status"] = req.pop("fn")(), "SUCCEEDED"
        except Exception as error:  # noqa: BLE001
            req["error"], req["status"] = error, "FAILED"

    def get(self, rid):
        if rid not in self.requests:
            raise RuntimeError(f"Failed to get request {rid}: 404 not found")
        self._run(rid)  # waits for it
        req = self.requests[rid]
        if req["status"] == "FAILED":
            raise req["error"]
        if req["status"] == "CANCELLED":
            raise RuntimeError("RequestCancelled")
        return req["result"]

    def api_status(self, ids):
        return [SimpleNamespace(request_id=r, status=self.requests[r]["status"]) for r in ids if r in self.requests]

    def api_cancel(self, ids):
        for r in ids:
            if r in self.requests and self.requests[r]["status"] == "RUNNING":
                self.requests[r]["status"] = "CANCELLED"
        self.calls.append(("cancel", list(ids)))
        return self._submit(lambda: list(ids))

    def restart_server(self):
        self.requests.clear()

    # -- the SDK
    def launch(self, task, cluster_name, **kw):
        self.calls.append(("launch", cluster_name, task, kw))
        failing = task.resources.get("infra") in self.fail_launch
        if not failing:
            labels = dict((task.resources or {}).get("labels") or {})
            self.clusters[cluster_name] = {"name": cluster_name, "status": "INIT", "kw": kw,
                                           "handle": SimpleNamespace(head_ip=None, ssh_user="ubuntu",
                                                                     launched_resources=SimpleNamespace(labels=labels))}

        def provision():
            if failing:
                raise ResourcesUnavailableError(f"no capacity for {task.resources}")
            self.up(cluster_name)
            return 1, self.clusters[cluster_name]["handle"]

        rid = self._submit(provision, done=failing)  # a refused launch fails fast; a good one provisions
        self.cluster_request[cluster_name] = rid
        return rid

    def status(self, names, refresh=None):
        if self.fail_status:
            raise ConnectionError("sky api server down")
        self.calls.append(("status", tuple(names) if names else None, refresh))
        rows = list(self.clusters.values()) if names is None else [self.clusters[n] for n in names
                                                                  if n in self.clusters]
        return self._submit(lambda: rows)

    def down(self, name):
        self.calls.append(("down", name))
        return self._submit(lambda: self.clusters.pop(name, None) and None)

    # -- the cloud / skylet side
    def up(self, name, ip="198.51.100.7"):
        launched = getattr(self.clusters[name].get("handle"), "launched_resources", None)
        self.clusters[name].update(status="UP", handle=SimpleNamespace(head_ip=ip, ssh_user="ubuntu",
                                                                      launched_resources=launched))
        rid = self.cluster_request.get(name)
        if rid in self.requests and self.requests[rid]["status"] == "RUNNING":
            self.requests[rid].update(status="SUCCEEDED", result=(1, self.clusters[name]["handle"]))
            self.requests[rid].pop("fn", None)

    def autostop(self, minutes, *, jobs=0):
        """sky's idle check after ``minutes`` with ``jobs`` running: a cluster whose idle
        timer waits for jobs is idle without one; ``wait_for=none`` only counts time."""
        for name, c in list(self.clusters.items()):
            kw = c.get("kw") or {}
            if not kw.get("down"):
                continue
            idle_limit = kw.get("idle_minutes_to_autostop") or 0
            waits_for_jobs = kw.get("wait_for") != self.AutostopWaitFor.NONE
            if waits_for_jobs and jobs:
                continue
            if minutes >= idle_limit:
                del self.clusters[name]


def marked(name, run, status="UP"):
    """A cluster sky lists whose launched resources carry ``run``'s ownership label."""
    return {"name": name, "status": status,
            "handle": SimpleNamespace(head_ip="198.51.100.8", ssh_user="ubuntu",
                                      launched_resources=SimpleNamespace(labels={"yeto-run": run}))}


def _provider(sky, slots=(1, 2)):
    return SkyProvider("run-s", resources={"cloud": "aws", "accelerators": "H100:8", "use_spot": True},
                       slots=list(slots), sky=sky, host_setup=["sudo docker info"])


def test_sky_provider_maps_cluster_states():
    sky = FakeSky()
    p = _provider(sky)
    name = p.order(1)
    assert name == "yeto-smp-run-s-1" and sky.calls[0][2].resources["use_spot"]
    assert p.status(1, name).state == "waiting" and not p.account_free(1) and p.account_free(2)
    sky.up(name)
    s = p.status(1, name)
    assert s.state == "active" and s.lease_until is None
    assert p.node(1, name).ip == "198.51.100.7"
    sky.clusters[name]["status"] = "STOPPED"
    assert p.status(1, name).end_reason == "preempted"
    del sky.clusters[name]  # the cloud reclaimed it
    assert p.status(1, name) == el.LeaseStatus("ended", end_reason="preempted")
    sky.fail_status = True
    with pytest.raises(ConnectionError):  # a blip to the pool, never an end
        p.status(1, name)


def test_sky_provider_releases_only_its_own_clusters():
    sky = FakeSky()
    p = _provider(sky)
    name = p.order(1)
    sky.clusters["someone-else"] = {"status": "UP"}
    with pytest.raises(PermissionError):
        p.release(1, "someone-else")
    with pytest.raises(PermissionError):
        p.release(1, "yeto-smp-run-s-99")  # right prefix, not in the record
    assert p.release(1, name)["state"] == "ended" and ("down", name) in sky.calls
    assert "someone-else" in sky.clusters and p.run_tickets() == [] and p.account_free(1)


def test_sky_attach_writes_a_direct_alias_and_runs_host_setup(tmp_path):
    sky = FakeSky()
    p = _provider(sky)
    p.ssh_key = "/keys/sky-key"
    lease = SimpleNamespace(ticket="yeto-smp-run-s-1", ip="198.51.100.7", user="ubuntu")
    ran = []
    alias = p.attach(lease, ssh_run=lambda a, c: ran.append((a, c)) or SimpleNamespace(returncode=0))
    text = (cli.tunnel.state_dir() / "ssh_config").read_text()
    assert alias == "yeto-nc-yeto-smp-run-s-1" and "HostName 198.51.100.7" in text and "ProxyCommand" not in text
    assert "IdentityFile /keys/sky-key" in text and ran == [(alias, "sudo docker info")]
    with pytest.raises(RuntimeError, match="host setup"):
        p.attach(lease, ssh_run=lambda a, c: SimpleNamespace(returncode=1))


def test_sky_config_is_validated_and_plugs_into_the_pool_route():
    cfg = {"target": 1, "accounts": [1, 2], "shard_base": "s", "run": "run-s",
           "provider": {"factory": "yeto.launch.nc.sky_provider:from_config",
                        "resources": {"cloud": "gcp"}}}
    ops = cli.pool_ops(cfg)
    assert isinstance(ops.provider, SkyProvider) and ops.provider.slots == [1, 2]
    assert ops.has_fallback_offer() is False  # one offer; with several the pool's ops ask the provider
    with pytest.raises(ValueError, match="unknown keys"):
        from_config(dict(cfg, provider={"resources": {}, "zone": "x"}), "run-s")
    with pytest.raises(ValueError, match="resources"):
        from_config(dict(cfg, provider={}), "run-s")


class SkyOps:
    """PoolOps = SkyProvider (supply) + a fake node route: the pool policy is shared."""

    def __init__(self, provider, sky, clock):
        self.p, self.sky, self.clock = provider, sky, clock
        self.nodes = {}
        self.store = SimpleNamespace(objects=_Objects())

    def order(self, account):
        return self.p.order(account)

    def poll(self, m):
        if m.ticket in self.sky.clusters and self.sky.clusters[m.ticket]["status"] == "INIT":
            self.sky.up(m.ticket)  # sky brings the node up within a tick
        return self.p.status(m.account, m.ticket)

    def start(self, m, shards):
        self.nodes[m.ticket] = {"joined_at": self.clock() + 300, "continues": m.replaces_shard}

    def probe(self, m):
        n = self.nodes.get(m.ticket)
        if n is None or m.ticket not in self.sky.clusters:
            return el.Probe(node=el.PROBE_UNREACHABLE)
        return el.Probe(node="ok", sampler="running", joined=self.clock() >= n["joined_at"])

    def ready(self, m):
        raise AssertionError

    def fence(self, m):
        elastic.fence_shard(self.store, m.shard, by=m.ticket, clock=self.clock)

    def drain(self, m):
        self.p.release(m.account, m.ticket)

    def balance(self):
        return self.p.balance()

    def account_free(self, account):
        return self.p.account_free(account)

    def run_tickets(self):
        return self.p.run_tickets()

    def has_fallback_offer(self):
        return self.p.has_fallback_offer()


class _Objects(dict):
    def read(self, k):
        return self.get(k)

    def write(self, k, v):
        self[k] = v

    def list(self, prefix):
        return sorted(k for k in self if k.startswith(prefix))


def test_pool_over_sky_replaces_a_reclaimed_spot_node_with_the_shared_policy():
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    sky = FakeSky()
    ops = SkyOps(_provider(sky), sky, clock)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-s")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="s", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0,
                                 elastic=el.ElasticConfig(max_orders=3, breaker_failures=2)),
                      ops, clock=clock, log=lambda m: None, events=events)
    for _ in range(10):
        pool.tick()
        t[0] += 60
    first = pool.members[0]
    assert first.state == READY and first.lease_until is None
    del sky.clusters[first.ticket]  # spot capacity reclaimed
    pool.tick()
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert first.state == GONE and "evicted" in names and "shard_fenced" in names
    for _ in range(10):
        t[0] += 60
        pool.tick()
    second = pool.members[1]
    assert second.state == READY and ops.nodes[second.ticket]["continues"] == first.shard
    # the shared budget caps sky orders too
    for _ in range(3):
        sky.clusters.pop(pool.members[-1].ticket, None)
        for _ in range(3):
            t[0] += 60
            pool.tick()
    assert len([c for c in sky.calls if c[0] == "launch"]) == 3
    assert any(e["event"] == "nc_elastic_order_refused" and "max_orders" in e["why"] for e in events.events)


# -- unified-sampler-pool 1.2-1.5: confirmed requests, autostop, patches, reconcile

def _two_offers(sky, clock=None, **kw):
    from yeto.launch.rollout import offers

    two = [{"resources": {"infra": "aws/us-east-1", "accelerators": "L4:1", "use_spot": True}, "price_usd_h": 0.4},
           {"resources": {"infra": "aws/us-west-2", "accelerators": "L4:1", "use_spot": True}, "price_usd_h": 0.5}]
    return SkyProvider("run-s", slots=[1, 2], offers=offers.offers_from_config({"offers": two}), sky=sky,
                       **({"clock": clock} if clock else {}), **kw)


def test_launch_returns_a_request_id_and_a_capacity_failure_raised_at_get_moves_to_the_next_offer():
    sky = FakeSky()
    sky.fail_launch.add("aws/us-east-1")
    p = _two_offers(sky)
    first = p.order(1)  # the submit itself succeeds: the failure is in the request
    assert isinstance(sky.cluster_request[first], RequestId)
    assert p._rows()[first]["launch_request"] == sky.cluster_request[first]
    status = p.status(1, first)  # the request finished: sky.get raises its failure
    assert status.state == "ended" and status.end_reason == "lost"
    assert "ResourcesUnavailableError" in status.detail and "no capacity" in status.detail
    assert p.cursor == 1 and p.has_fallback_offer() and p.offer_breakers()[p.offers[0].key]["fails"] == 1
    assert p.account_free(1)  # no cluster left: nothing bills
    assert p.release(1, first)["state"] == "ended"  # the pool's release of the ended ticket
    second = p.order(1)
    assert sky.calls[-1][2].resources["infra"] == "aws/us-west-2" and p.offer_of(second) == p.offers[1].key
    assert p.status(1, second).state == "waiting"  # its launch is still provisioning
    sky.up(second)
    assert p.status(1, second).state == "active" and p.cursor == 0  # confirmed: back to the cheapest


def test_the_pool_counts_a_failed_launch_against_the_offer_not_its_breaker():
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    sky = FakeSky()
    sky.fail_launch.add("aws/us-east-1")
    ops = SkyOps(_two_offers(sky, clock=clock), sky, clock)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-s")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="s", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0,
                                 elastic=el.ElasticConfig(max_orders=4, breaker_failures=1)),
                      ops, clock=clock, log=lambda m: None, events=events)
    for _ in range(12):
        pool.tick()
        t[0] += 60
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "offer_failed" in names and "breaker_open" not in names
    ready = [m for m in pool.members if m.state == READY]
    assert len(ready) == 1 and ops.p.offer_of(ready[0].ticket) == ops.p.offers[1].key
    assert ("down", pool.members[0].ticket) not in sky.calls  # the failed launch left no cluster


def test_the_watchdog_launch_blocks_on_the_request_and_raises_its_failure():
    sky = FakeSky()
    sky.fail_launch.add("aws/us-east-1")
    p = _two_offers(sky)
    with pytest.raises(ResourcesUnavailableError):
        p.launch_cluster(f"{p.prefix}-1", p.offers[0], slot=1)
    assert p.cursor == 1
    name = p.launch_cluster(f"{p.prefix}-2", p.offers[1], slot=1)
    assert sky.clusters[name]["status"] == "UP" and p.status(1, name).state == "active"


def test_a_cluster_with_no_sky_job_waits_for_nothing_and_lives_to_the_hard_cap():
    sky = FakeSky()
    p = _provider(sky)
    p.hard_cap_minutes = 600
    name = p.order(1)
    kw = sky.calls[0][3]
    assert kw == {"down": True, "idle_minutes_to_autostop": 600, "wait_for": "none"}
    assert "detach_run" not in kw  # not a sky.launch keyword
    sky.autostop(5)  # sky's idle check: no job runs (the sampler is an ssh docker container)
    assert name in sky.clusters
    sky.autostop(599)
    assert name in sky.clusters
    sky.autostop(600)  # the hard cap
    assert name not in sky.clusters
    # the old keywords (down=True alone): idle at once without a job, autodowned within a minute
    sky.clusters["old"] = {"name": "old", "status": "UP", "kw": {"down": True}}
    sky.autostop(1)
    assert "old" not in sky.clusters


def test_a_task_with_its_own_run_is_the_sampler_job_and_goes_down_with_it(tmp_path):
    sky = FakeSky()
    yaml = tmp_path / "sampler.yaml"
    yaml.write_text("run: python -m sampler\n")
    p = from_config({"target": 1, "accounts": [1], "shard_base": "s", "run": "run-s",
                     "provider": {"resources": {"cloud": "aws"}, "task_yaml": str(yaml)}}, "run-s", sky=sky)
    name = p.order(1)
    assert sky.calls[0][3] == {"down": True}
    sky.autostop(5, jobs=1)
    assert name in sky.clusters  # the job runs
    sky.autostop(5, jobs=0)
    assert name not in sky.clusters  # it ended: down


def test_hard_cap_minutes_is_a_pool_config_key():
    cfg = {"target": 1, "accounts": [1], "shard_base": "s", "run": "run-s",
           "provider": {"resources": {"cloud": "aws"}, "hard_cap_minutes": 90}}
    assert from_config(cfg, "run-s", sky=FakeSky()).hard_cap_minutes == 90
    with pytest.raises(ValueError, match="hard_cap_minutes"):
        from_config(dict(cfg, provider={"resources": {"cloud": "aws"}, "hard_cap_minutes": 0}), "run-s",
                    sky=FakeSky())


def test_the_pool_installs_the_nebius_spot_pricing_patch_before_its_first_order(monkeypatch):
    import sys

    from yeto import sky_patches
    from yeto.sky_patches import nebius

    class Compute:
        class FollowsSpotPriceSpec:
            pass

        @staticmethod
        def InstanceSpec(**kw):  # noqa: N802 - the SDK's name
            return dict(kw)

    utils = SimpleNamespace(nebius=SimpleNamespace(compute=lambda: Compute()), launch=lambda: None)
    monkeypatch.setattr(sky_patches, "_STATUS", {})
    monkeypatch.setitem(sys.modules, nebius.TARGET_MODULE, utils)
    monkeypatch.setitem(sys.modules, "sky", SimpleNamespace(__version__="0.14.3"))
    installed = []
    real_install = sky_patches.install
    monkeypatch.setattr(sky_patches, "install", lambda: installed.append(len(sky.calls)) or real_install())
    sky = FakeSky()
    p = SkyProvider("run-s", slots=[1], resources={"infra": "nebius/eu-north1", "accelerators": "L40S:1",
                                                   "use_spot": True}, sky=sky)
    p.order(1)
    p.order(1)
    assert installed == [0]  # once, before the first launch
    assert sky_patches.status()[nebius.TARGET_MODULE] == "applied"
    # what sky's Nebius provisioner then builds for a preemptible VM carries a pricing model
    spec = utils.nebius.compute().InstanceSpec(preemptible="stop", resources="r")
    assert isinstance(spec["follows_spot_price"], Compute.FollowsSpotPriceSpec)


def test_after_an_api_server_restart_clusters_are_reconciled_by_prefix_without_a_second_order():
    t = [0.0]
    sky = FakeSky()
    p = SkyProvider("run-s", resources={"cloud": "aws", "accelerators": "L4:1", "use_spot": True},
                    slots=[1, 2, 3], sky=sky, clock=lambda: t[0])
    up, starting, unborn = p.order(1), p.order(2), p.order(3)
    sky.up(up)
    del sky.clusters[unborn]  # its launch was still queued: sky had not created the cluster yet
    launches = len([c for c in sky.calls if c[0] == "launch"])
    sky.restart_server()  # every request id the record holds is unknown now
    assert p.status(1, up).state == "active"  # UP before the restart: kept, no new order
    lost = p.status(2, starting)
    assert lost.state == "ended" and lost.end_reason == "lost" and "INIT" in lost.detail
    gone = p.status(3, unborn)
    assert gone.state == "ended" and gone.end_reason == "lost" and "before the cluster existed" in gone.detail
    assert sorted(t for _, t in p.run_tickets()) == sorted([up, starting, unborn])  # still ours to release
    assert len([c for c in sky.calls if c[0] == "launch"]) == launches
    assert p.offer_breakers() == {}  # a lost request is not the offer's failure
    for slot, ticket in ((2, starting), (3, unborn)):
        assert p.release(slot, ticket)["state"] == "ended"
    assert starting not in sky.clusters and p.run_tickets() == [(1, up)]
    # a cluster the record never got (a crash between the launch and the record) is adopted
    sky.clusters["yeto-smp-run-s-9"] = marked("yeto-smp-run-s-9", "run-s")
    assert (2, "yeto-smp-run-s-9") in p.run_tickets() and p.status(2, "yeto-smp-run-s-9").state == "active"


def test_release_cancels_a_launch_still_in_flight_before_the_down():
    sky = FakeSky()
    p = _provider(sky)
    name = p.order(1)
    assert p.status(1, name).state == "waiting"
    assert p.release(1, name)["state"] == "ended"
    kinds = [c[0] for c in sky.calls if c[0] in ("cancel", "down")]
    assert kinds == ["cancel", "down"] and name not in sky.clusters
    assert sky.requests[sky.cluster_request[name]]["status"] == "CANCELLED"


def test_status_asks_sky_to_refresh_so_a_reclaimed_spot_vm_reads_gone():
    sky = FakeSky()
    p = _provider(sky)
    name = p.order(1)
    p.status(1, name)
    assert ("status", (name,), "AUTO") in sky.calls


# -- ownership: a positive mark, not a name (REV on #765)
def test_every_launch_carries_the_runs_ownership_label_and_ledger_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    sky = FakeSky()
    p = _provider(sky)
    name = p.order(1)
    task = next(c[2] for c in sky.calls if c[0] == "launch")
    assert task.resources["labels"] == {"yeto-run": "run-s"}
    from yeto.cost.ledger import current

    intents = [r for r in current().records() if r["op"] == "intent"]
    assert intents and intents[0]["name"] == name and intents[0]["attrs"]["run"] == "run-s"


def test_a_label_unsafe_run_id_gets_a_hashed_owner_and_an_unambiguous_prefix(tmp_path):
    sky = FakeSky()
    a = SkyProvider("ABC_2", slots=[1], resources={"infra": "aws"}, sky=sky, record=tmp_path / "a.json")
    b = SkyProvider("abc-2", slots=[1], resources={"infra": "aws"}, sky=sky, record=tmp_path / "b.json")
    assert a.prefix != b.prefix and b.prefix == "yeto-smp-abc-2" and a.prefix.startswith("yeto-smp-abc-2-")
    assert b.owner == "abc-2" and a.owner.startswith("h-") and a.owner != b.owner
    name = a.order(1)
    sky.up(name)
    assert a.run_tickets() == [(1, name)]
    assert b.run_tickets() == []  # never the other run's, whatever its name looks like
    with pytest.raises(PermissionError):
        b.release(1, name)
    assert name in sky.clusters


def test_a_record_of_another_run_is_refused(tmp_path):
    record = tmp_path / "shared.json"
    SkyProvider("run-a", slots=[1], resources={"infra": "aws"}, sky=FakeSky(), record=record).order(1)
    with pytest.raises(PermissionError, match="belongs to run 'run-a'"):
        SkyProvider("run-b", slots=[1], resources={"infra": "aws"}, sky=FakeSky(), record=record).run_tickets()


def test_only_marked_clusters_named_exactly_prefix_n_are_adopted(tmp_path, capsys):
    sky = FakeSky()
    p = SkyProvider("abc", slots=[1, 2], resources={"infra": "aws"}, sky=sky, record=tmp_path / "abc.json")
    sky.clusters["yeto-smp-abc-3"] = marked("yeto-smp-abc-3", "abc")  # ours: a launch the record lost
    sky.clusters["yeto-smp-abc-2-1"] = marked("yeto-smp-abc-2-1", "abc-2")  # run abc-2's
    sky.clusters["yeto-smp-abc-2-5"] = marked("yeto-smp-abc-2-5", "abc")  # marked ours, but not <prefix>-<n>
    sky.clusters["yeto-smp-abc-handmade"] = {"name": "yeto-smp-abc-handmade", "status": "UP"}
    sky.clusters["yeto-smp-abc-9"] = {"name": "yeto-smp-abc-9", "status": "UP"}  # the name, no mark
    sky.clusters["yeto-smp-abc-8"] = marked("yeto-smp-abc-8", "abc-2")  # the name, another run's mark
    assert [t for _, t in p.run_tickets()] == ["yeto-smp-abc-3"]
    assert p.run_tickets() == [(1, "yeto-smp-abc-3")]
    out = capsys.readouterr().out
    assert out.count("yeto-smp-abc-9 has this run's name pattern but no ownership mark") == 1  # once
    assert "yeto-smp-abc-8" in out and "handmade" not in out and "abc-2-1" not in out
    for foreign in ("yeto-smp-abc-2-1", "yeto-smp-abc-2-5", "yeto-smp-abc-handmade", "yeto-smp-abc-9",
                    "yeto-smp-abc-8"):
        with pytest.raises(PermissionError):
            p.release(1, foreign)
    assert p.release(1, "yeto-smp-abc-3")["state"] == "ended"
    assert not [c for c in sky.calls if c[0] == "down" and c[1] != "yeto-smp-abc-3"]
    assert len(sky.clusters) == 5


def test_a_ledger_intent_of_this_run_is_a_mark_and_one_of_another_run_is_not(tmp_path, monkeypatch):
    from yeto.cost.ledger import ResourceLedger
    from yeto.launch.nc.sky_provider import LEDGER_KIND

    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")
    ledger.intent(LEDGER_KIND, "yeto-smp-abc-4", run="abc")  # written before a launch that sky never labelled
    ledger.intent(LEDGER_KIND, "yeto-smp-abc-6", run="abc-other")
    ledger.intent(LEDGER_KIND, "yeto-smp-abc-7")  # no run: no mark
    sky = FakeSky()
    for n in (4, 6, 7):
        sky.clusters[f"yeto-smp-abc-{n}"] = {"name": f"yeto-smp-abc-{n}", "status": "UP"}
    p = SkyProvider("abc", slots=[1, 2], resources={"infra": "aws"}, sky=sky, record=tmp_path / "abc.json")
    assert [t for _, t in p.run_tickets()] == ["yeto-smp-abc-4"]


def test_the_watchdogs_exact_prefix_adopts_only_its_own_runs_clusters(tmp_path):
    """exact_prefix (the spot watchdog: names ``spot-w-<n>``, no run id in them): a cluster an
    earlier process of this run launched is adopted and released; another run's cluster under
    the same pattern, or one opened by hand, is never adopted, released or downed."""
    sky = FakeSky()

    def watchdog(run, where):
        return SkyProvider(run, slots=[1], resources={"infra": "aws", "use_spot": True}, sky=sky,
                           exact_prefix="spot-w", record=tmp_path / f"{where}.json")

    earlier = watchdog("runw", "earlier")
    earlier.launch_cluster("spot-w-1", earlier.offers[0], slot=1)
    other = watchdog("runx", "other")
    other.launch_cluster("spot-w-7", other.offers[0], slot=1)
    sky.clusters["spot-w-debug"] = {"name": "spot-w-debug", "status": "UP"}
    sky.clusters["spot-w-3"] = {"name": "spot-w-3", "status": "UP"}  # by hand, the pattern's name
    mine = watchdog("runw", "mine")  # a new process of run runw; the earlier record is lost
    assert [t for _, t in mine.run_tickets()] == ["spot-w-1"]
    assert mine.release(1, "spot-w-1")["state"] == "ended" and "spot-w-1" not in sky.clusters
    for foreign in ("spot-w-7", "spot-w-debug", "spot-w-3"):
        with pytest.raises(PermissionError):
            mine.release(1, foreign)
        assert foreign in sky.clusters
    assert not [c for c in sky.calls if c[0] == "down" and c[1] != "spot-w-1"]


def test_no_ownership_label_without_a_cloud_the_ledger_intent_is_the_mark(tmp_path, monkeypatch):
    """sky validates labels of cloud-less resources against every enabled cloud (a client-side
    cloud check), so a cloud-less offer gets no label; its ledger intent still marks it."""
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    sky = FakeSky()
    p = SkyProvider("run-s", slots=[1], resources={"accelerators": "L4:1"}, sky=sky, record=tmp_path / "a.json")
    name = p.order(1)
    task = next(c[2] for c in sky.calls if c[0] == "launch")
    assert "labels" not in task.resources
    sky.up(name)
    again = SkyProvider("run-s", slots=[1], resources={"accelerators": "L4:1"}, sky=sky, record=tmp_path / "b.json")
    assert again.run_tickets() == [(1, name)]


@pytest.mark.parametrize("infra", ["aws", "gcp", "nebius", "kubernetes", "runpod", "verda"])
@pytest.mark.parametrize("run", ["run-s", "ABC_2"])
def test_sky_accepts_and_keeps_the_ownership_label_on_every_infra(infra, run):
    """Real sky: the label passes the cloud's label rules and survives the copies the
    optimizer makes (launched_resources is a copy of the requested resources)."""
    sky = pytest.importorskip("sky")
    from yeto.launch.nc.sky_provider import OWNER_LABEL, owner_token

    labels = {OWNER_LABEL: owner_token(run)}
    resources = sky.Resources(infra=infra, labels=labels)
    assert resources.labels == labels and resources.copy(region=None).labels == labels


# -- the run's AWS credentials never reach sky's requests (an R2 store's AWS_PROFILE, found on a real machine)
def _fake_builder(sp):
    def original(*args, **kw):
        return {"AWS_PROFILE": "store-r2", "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE", "SKYPILOT_X": "1"}

    return original, sp._scoped_request_env(original)


def test_the_providers_sky_calls_carry_no_run_aws_credentials():
    from yeto.launch.nc import sky_provider as sp

    original, builder = _fake_builder(sp)
    seen = []

    class Module:
        StatusRefreshMode = SimpleNamespace(AUTO="auto")

        @staticmethod
        def status(names, **kw):
            seen.append(builder())
            return "rid"

    plain = SkyProvider("run-s", slots=[1], resources={"infra": "aws"}, sky=Module())
    assert plain.sky.status(None) == "rid" and plain.sky.StatusRefreshMode.AUTO == "auto"
    assert seen[-1] == {"SKYPILOT_X": "1"}  # the run's AWS keys dropped
    named = SkyProvider("run-s", slots=[1], resources={"infra": "aws"}, sky=Module(), aws_profile="sky-aws")
    named.sky.status(None)
    assert seen[-1] == {"SKYPILOT_X": "1", "AWS_PROFILE": "sky-aws"}
    assert builder() == original()  # outside a provider call: as sky builds it
    assert builder("future", kw=1) == original()  # a builder that grows parameters still works


def test_the_scope_follows_sky_into_threads_with_a_copied_context_and_not_beyond():
    import concurrent.futures
    import contextvars
    import threading

    from yeto.launch.nc import sky_provider as sp

    original, builder = _fake_builder(sp)
    inside, outside = [], []
    started, release = threading.Event(), threading.Event()

    def concurrent_caller():  # another thread, not in any scope, while a scope is active
        started.wait(5)
        outside.append(builder())
        release.set()

    t = threading.Thread(target=concurrent_caller)
    t.start()
    with sp.request_env_scope(sp.aws_request_env(None)):
        started.set()
        release.wait(5)
        ctx = contextvars.copy_context()  # what sky's run_in_parallel hands its workers
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            inside += list(pool.map(lambda _: ctx.copy().run(builder), range(2)))
    t.join(5)
    assert outside == [original()]
    assert inside == [{"SKYPILOT_X": "1"}, {"SKYPILOT_X": "1"}]


def test_real_sky_request_bodies_of_the_provider_drop_the_run_aws_profile(monkeypatch):
    pytest.importorskip("sky")
    from sky.server import common
    from sky.server.requests import payloads

    from yeto.launch.nc import sky_provider as sp

    monkeypatch.setattr(payloads, "request_body_env_vars", payloads.request_body_env_vars)  # restored after
    monkeypatch.setattr(common, "is_api_server_local", lambda *a, **k: True)
    monkeypatch.setenv("AWS_PROFILE", "store-r2")
    sp.install_request_env_scope()
    sp.install_request_env_scope()  # idempotent, self-checked
    assert payloads.RequestBody().env_vars.get("AWS_PROFILE") == "store-r2"  # sky's own forwarding
    p = SkyProvider("run-s", slots=[1], resources={"infra": "aws"}, sky=None)  # the real sky, via the proxy

    def body(*a, **k):
        return payloads.RequestBody().env_vars

    monkeypatch.setattr("sky.status", body)
    assert "AWS_PROFILE" not in p.sky.status(None)
    named = SkyProvider("run-s", slots=[1], resources={"infra": "aws"}, sky=None, aws_profile="sky-aws")
    assert named.sky.status(None)["AWS_PROFILE"] == "sky-aws"


def test_a_sky_that_does_not_use_the_scoped_builder_refuses(monkeypatch):
    pytest.importorskip("sky")
    from sky.server.requests import payloads

    from yeto.launch.nc import sky_provider as sp

    monkeypatch.setattr(payloads, "request_body_env_vars", payloads.request_body_env_vars)
    real_init = payloads.RequestBody.__init__

    def drifted_init(self, **data):  # a sky that stopped building env_vars through the module function
        data.setdefault("env_vars", {})
        real_init(self, **data)

    monkeypatch.setattr(payloads.RequestBody, "__init__", drifted_init)
    with pytest.raises(RuntimeError, match="without payloads.request_body_env_vars"):
        sp.install_request_env_scope()
    monkeypatch.delattr(payloads, "request_body_env_vars")
    with pytest.raises(RuntimeError, match="no payloads.request_body_env_vars"):
        sp.install_request_env_scope()


@pytest.mark.parametrize("env,infra,aws_profile,refused", [
    ({}, "aws", None, []),
    ({"AWS_PROFILE": "store-r2"}, "aws", None, ["AWS_PROFILE"]),
    ({"AWS_PROFILE": "store-r2"}, "aws", "sky-aws", []),
    ({"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"}, "aws", "sky-aws",
     ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
    ({"AWS_ENDPOINT_URL": "https://r2.example"}, "aws", "sky-aws", ["AWS_ENDPOINT_URL"]),
    ({"AWS_CONFIG_FILE": "/c", "AWS_SHARED_CREDENTIALS_FILE": "/s"}, "aws/us-east-2", "sky-aws",
     ["AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE"]),
    ({"AWS_PROFILE": "store-r2", "AWS_ENDPOINT_URL": "https://r2.example"}, None, None,
     ["AWS_PROFILE", "AWS_ENDPOINT_URL"]),  # not pinned: may land on AWS
    ({"AWS_PROFILE": "store-r2", "AWS_ACCESS_KEY_ID": "k"}, "nebius/eu-north1", None, []),  # no AWS call
    ({"AWS_PROFILE": "store-r2"}, "kubernetes", None, []),
])
def test_dry_run_refuses_run_aws_settings_that_would_reach_sky(env, infra, aws_profile, refused):
    from yeto.launch.nc import sky_provider as sp

    resources = {"accelerators": "L4:1", **({"infra": infra} if infra else {})}
    p = SkyProvider("run-s", slots=[1], resources=resources, sky=FakeSky(), aws_profile=aws_profile)
    problems = sp.aws_env_problems(p, env)
    if not refused:
        assert problems == []
    else:
        assert len(problems) == 1 and all(name in problems[0] for name in refused)
        assert "YETO_RL_STORE_S3_PROFILE" in problems[0] and "provider.aws_profile" in problems[0]


def test_the_pool_start_refuses_too_not_only_its_dry_run(monkeypatch):
    from yeto import sky_patches

    monkeypatch.setattr(sky_patches, "require_matching_api_server", lambda out=None: None)
    monkeypatch.setenv("AWS_PROFILE", "store-r2")
    sky = FakeSky()
    p = SkyProvider("run-s", slots=[1], resources={"infra": "aws", "accelerators": "L4:1"}, sky=sky)
    p._real_sky = True
    with pytest.raises(ValueError, match="AWS_PROFILE"):
        p.order(1)
    assert not [c for c in sky.calls if c[0] == "launch"]


def test_the_cleanups_sky_down_of_a_pool_cluster_uses_the_providers_request_env(monkeypatch):
    from yeto.cost import cleanup
    from yeto.launch.nc import sky_provider as sp

    seen = []
    monkeypatch.setattr(sp, "install_request_env_scope", lambda: None)

    class Ops:
        def sky_down(self, cluster):
            seen.append(sp._REQUEST_ENV.get())
            return True

    class Ledger:
        def released(self, *a, **k):
            pass

    def row(**attrs):
        return SimpleNamespace(kind="sky_cluster", id="c-1", name="c-1", key="k", attrs=attrs)

    assert cleanup._release_one(row(pool=True, aws_profile="sky-aws"), Ops(), Ledger(), False, {}).ok
    assert cleanup._release_one(row(pool=True), Ops(), Ledger(), False, {}).ok
    assert cleanup._release_one(row(), Ops(), Ledger(), False, {}).ok  # a launcher cluster: sky's own forwarding
    assert seen == [sp.aws_request_env("sky-aws"), sp.aws_request_env(None), None]
