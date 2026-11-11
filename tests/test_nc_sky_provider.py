"""SkyPilot lease provider skeleton (yeto.launch.nc.sky_provider) over a fake sky
module: the same pool policy (events, budget, breaker, fencing) as National Compute."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, elastic as el
from yeto.launch.nc.lease_pool import GONE, READY, LeasePool, PoolConfig
from yeto.launch.nc.sky_provider import SkyProvider, from_config


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))


class FakeSky:
    """launch / status / down / get over an in-memory cluster table; never a cloud."""

    def __init__(self):
        self.clusters = {}
        self.calls = []
        self.fail_status = False

    class Task:
        def __init__(self, name=None, run=None):
            self.name, self.run, self.resources = name, run, None

        def set_resources(self, r):
            self.resources = r

    @staticmethod
    def Resources(**kw):  # noqa: N802 - sky's name
        return dict(kw)

    def launch(self, task, cluster_name, **kw):
        self.calls.append(("launch", cluster_name, task.resources, kw))
        self.clusters[cluster_name] = {"name": cluster_name, "status": "INIT",
                                       "handle": SimpleNamespace(head_ip=None, ssh_user="ubuntu")}

    def status(self, names):
        if self.fail_status:
            raise ConnectionError("sky api server down")
        return [self.clusters[n] for n in names if n in self.clusters]

    def down(self, name):
        self.calls.append(("down", name))
        self.clusters.pop(name, None)

    @staticmethod
    def get(value):
        return value

    def up(self, name, ip="198.51.100.7"):
        self.clusters[name].update(status="UP", handle=SimpleNamespace(head_ip=ip, ssh_user="ubuntu"))


def _provider(sky, slots=(1, 2)):
    return SkyProvider("run-s", resources={"cloud": "aws", "accelerators": "H100:8", "use_spot": True},
                       slots=list(slots), sky=sky, host_setup=["sudo docker info"])


def test_sky_provider_maps_cluster_states():
    sky = FakeSky()
    p = _provider(sky)
    name = p.order(1)
    assert name == "yeto-smp-run-s-1" and sky.calls[0][2]["use_spot"] and sky.calls[0][3]["detach_run"]
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
        el.fence_shard(self.store, m.shard, by=m.ticket, clock=self.clock)

    def drain(self, m):
        self.p.release(m.account, m.ticket)

    def balance(self):
        return self.p.balance()

    def account_free(self, account):
        return self.p.account_free(account)

    def run_tickets(self):
        return self.p.run_tickets()


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
    pool = LeasePool(PoolConfig(target=1, accounts=[1, 2], shard_base="s", stop_below_ucr=0,
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
