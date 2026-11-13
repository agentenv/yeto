"""Daytona lease provider (yeto.launch.nc.daytona_provider) over a fake Daytona SDK:
spot MI355X sandboxes under the same pool policy (fence, refill, patrol) as National
Compute. No Daytona API is called."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, elastic, pool_run
from yeto.launch.nc.daytona_provider import LABEL_RUN, LABEL_SLOT, SSH_HOST, DaytonaProvider, from_config
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY, IslandPool
from yeto.launch.nc.provider import is_national_compute


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(cli.node_pool, "SSH_CONFIG", tmp_path / "home" / ".ssh" / "config")


class DaytonaNotFoundError(Exception):
    pass


class FakeSdk:
    """The parts of the ``daytona`` package the provider uses: parameter classes record their fields."""

    DaytonaNotFoundError = DaytonaNotFoundError

    @staticmethod
    def GpuType(value):  # noqa: N802 - the SDK's enum name
        return value

    class _Params(dict):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.__dict__.update(kw)

    class CreateSandboxFromImageParams(_Params):
        pass

    class CreateSandboxFromSnapshotParams(_Params):
        pass

    class Resources(_Params):
        pass

    class ListSandboxesQuery(_Params):
        pass


@dataclass
class FakeSandbox:
    name: str
    labels: dict
    state: str = "creating"
    gpu: int = 8
    spot_evicted_at: str | None = None
    queue_timed_out_at: str | None = None
    tokens: list = field(default_factory=list)

    def create_ssh_access(self, expires_in_minutes=None):
        token = f"tok{len(self.tokens) + 1}"
        self.tokens.append((token, expires_in_minutes))
        return SimpleNamespace(token=token, expires_at=None, ssh_command=f"ssh {token}@{SSH_HOST}")


class FakeDaytona:
    """create / get / delete / list over an in-memory table; never a network call."""

    def __init__(self):
        self.boxes: dict[str, FakeSandbox] = {}
        self.calls = []
        self.fail_get = False
        self.fail_create: Exception | None = None
        self.create_leaves_box = False

    def create(self, params, timeout=60):
        self.calls.append(("create", params, timeout))
        if self.create_leaves_box:
            self.boxes[params["name"]] = FakeSandbox(params["name"], dict(params["labels"]))
        if self.fail_create is not None:
            raise self.fail_create
        self.boxes[params["name"]] = FakeSandbox(params["name"], dict(params["labels"]))
        return self.boxes[params["name"]]

    def get(self, name):
        if self.fail_get:
            raise ConnectionError("api down")
        if name not in self.boxes:
            raise DaytonaNotFoundError(name)
        return self.boxes[name]

    def delete(self, sandbox, timeout=60, wait=False):
        self.calls.append(("delete", sandbox.name, wait))
        self.boxes[sandbox.name].state = "destroyed"

    def list(self, query):
        self.calls.append(("list", dict(query["labels"])))
        return [b for b in self.boxes.values()
                if all(b.labels.get(k) == v for k, v in query["labels"].items())]

    def start(self, name):
        self.boxes[name].state = "started"

    def evict(self, name):
        self.boxes[name].state = "destroyed"
        self.boxes[name].spot_evicted_at = "2026-11-11T00:00:00Z"


def _provider(dt, slots=(1, 2), **kw):
    kw.setdefault("image", "rocm/vllm:test")
    return DaytonaProvider("run-d", slots=list(slots), sdk=FakeSdk, client=dt, **kw)


def test_order_creates_spot_mi355x_with_run_labels_and_auto_stop_off():
    dt = FakeDaytona()
    p = _provider(dt, cpu=128, memory_gib=1536, disk_gib=4096)
    name = p.order(2)
    params = dt.calls[0][1]
    assert name == "yeto-smp-run-d-1" and params["name"] == name
    assert params["spot"] is True and params["auto_stop_interval"] == 0 and params["auto_delete_interval"] == 0
    assert params["labels"][LABEL_RUN] == "run-d" and params["labels"][LABEL_SLOT] == "2"
    assert params["resources"]["gpu"] == 8 and params["resources"]["gpu_type"] == ["MI355X"]
    assert params["resources"]["memory"] == 1536 and params["image"] == "rocm/vllm:test"
    assert p.status(2, name).state == "waiting" and not p.account_free(2) and p.account_free(1)
    dt.start(name)
    s = p.status(2, name)
    assert s.state == "active" and s.lease_until is None and s.gpus == 8
    lease = p.node(2, name)
    assert lease.ip == SSH_HOST and lease.user == ""  # the token never goes to the lease file


def test_snapshot_order_carries_no_resources():
    dt = FakeDaytona()
    p = _provider(dt, image=None, snapshot="k31-sampler")
    p.order(1)
    params = dt.calls[0][1]
    assert isinstance(params, FakeSdk.CreateSandboxFromSnapshotParams) and "resources" not in params
    assert params["snapshot"] == "k31-sampler" and params["auto_stop_interval"] == 0


def test_status_maps_eviction_release_and_errors():
    dt = FakeDaytona()
    p = _provider(dt)
    name = p.order(1)
    dt.start(name)
    dt.evict(name)
    assert p.status(1, name) == el.LeaseStatus("ended", end_reason="preempted")
    dt.boxes[name].spot_evicted_at = None
    assert p.status(1, name).end_reason == "released"  # deleted by someone, not evicted
    dt.boxes[name].queue_timed_out_at = "t"
    assert p.status(1, name).end_reason == "queue_timeout"
    for state, reason in (("stopped", "preempted"), ("error", "error"), ("build_failed", "error")):
        dt.boxes[name] = FakeSandbox(name, {LABEL_RUN: "run-d"}, state=state)
        assert p.status(1, name) == el.LeaseStatus("ended", end_reason=reason)
    del dt.boxes[name]  # past Daytona's 24 h retention
    assert p.status(1, name).end_reason == "preempted"
    assert el.classify(p.status(1, name), None, now=0, unreachable_since=None, cfg=el.ElasticConfig(),
                       joined=True).kind == el.EVICTED
    dt.fail_get = True
    with pytest.raises(ConnectionError):  # a blip to the pool, never an end
        p.status(1, name)


def test_create_errors_keep_a_billed_sandbox_and_close_a_missing_one():
    dt = FakeDaytona()
    p = _provider(dt)
    dt.fail_create = RuntimeError("no spot capacity")
    with pytest.raises(RuntimeError, match="spot capacity"):
        p.order(1)
    assert p.run_tickets() == [] and p.account_free(1)
    dt.create_leaves_box = True  # created, but the start wait timed out
    dt.fail_create = TimeoutError("start timeout")
    name = p.order(1)
    assert name == "yeto-smp-run-d-2" and (1, name) in p.run_tickets()


def test_release_deletes_only_this_runs_sandboxes_and_confirms():
    dt = FakeDaytona()
    p = _provider(dt)
    name = p.order(1)
    dt.boxes["someone-else"] = FakeSandbox("someone-else", {}, state="started")
    dt.boxes["yeto-smp-run-d-99"] = FakeSandbox("yeto-smp-run-d-99", {LABEL_RUN: "other-run"}, state="started")
    with pytest.raises(PermissionError):
        p.release(1, "someone-else")
    with pytest.raises(PermissionError):
        p.release(1, "yeto-smp-run-d-99")  # right prefix, another run's label, not in the record
    assert p.release(1, name)["state"] == "ended" and ("delete", name, True) in dt.calls
    assert dt.boxes["someone-else"].state == "started" and p.run_tickets() == [] and p.account_free(1)
    assert p.status(1, name).end_reason == "released"
    p.release(1, name)  # idempotent: an already destroyed sandbox is not deleted again
    assert [c for c in dt.calls if c[0] == "delete"] == [("delete", name, True)]


def test_orphan_sweep_finds_unrecorded_sandboxes_by_label():
    dt = FakeDaytona()
    p = _provider(dt)
    mine = p.order(1)
    # a create whose process died before the record: only the label says it is ours
    dt.boxes["yeto-smp-run-d-7"] = FakeSandbox("yeto-smp-run-d-7", {LABEL_RUN: "run-d", LABEL_SLOT: "2"},
                                               state="started")
    dt.boxes["yeto-smp-run-x-1"] = FakeSandbox("yeto-smp-run-x-1", {LABEL_RUN: "run-x"}, state="started")
    dt.boxes["yeto-smp-run-d-8"] = FakeSandbox("yeto-smp-run-d-8", {LABEL_RUN: "run-d"}, state="destroyed")
    assert sorted(p.run_tickets()) == [(1, mine), (2, "yeto-smp-run-d-7")]
    assert ("list", {LABEL_RUN: "run-d"}) in dt.calls
    assert sorted(p.sweep()) == [mine, "yeto-smp-run-d-7"]
    assert dt.boxes["yeto-smp-run-d-7"].state == "destroyed" and dt.boxes["yeto-smp-run-x-1"].state == "started"
    assert p.run_tickets() == []


def test_attach_writes_a_gateway_alias_and_renews_the_token(tmp_path):
    t = [1000.0]
    dt = FakeDaytona()
    p = _provider(dt, ssh_token_minutes=60, host_setup=["rocm-smi"], clock=lambda: t[0])
    name = p.order(1)
    dt.start(name)
    ran = []
    alias = p.attach(p.node(1, name), ssh_run=lambda a, c: ran.append((a, c)) or SimpleNamespace(returncode=0))
    conf = cli.tunnel.state_dir() / "ssh_config"
    text = conf.read_text()
    assert alias == f"yeto-nc-{name}" and f"HostName {SSH_HOST}" in text and "User tok1" in text
    assert "ProxyCommand" not in text and conf.stat().st_mode & 0o077 == 0 and ran == [(alias, "rocm-smi")]
    assert dt.boxes[name].tokens == [("tok1", 60)]
    # ``ssh <alias>`` resolves only when the user's ssh config includes our file
    assert f"Include {conf}" in (tmp_path / "home" / ".ssh" / "config").read_text().splitlines()
    t[0] += 30 * 60
    p.status(1, name)
    assert "User tok1" in conf.read_text()  # more than a third left
    t[0] += 15 * 60
    p.status(1, name)
    assert "User tok2" in conf.read_text()
    p.release(1, name)
    assert alias not in conf.read_text()
    with pytest.raises(RuntimeError, match="host setup"):
        dt.boxes["yeto-smp-run-d-2"] = FakeSandbox("yeto-smp-run-d-2", {LABEL_RUN: "run-d"}, state="started")
        p.attach(SimpleNamespace(ticket="yeto-smp-run-d-2"), ssh_run=lambda a, c: SimpleNamespace(returncode=1))


def test_missing_api_key_is_an_error(monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    p = DaytonaProvider("run-d", slots=[1], image="x", sdk=FakeSdk)
    with pytest.raises(RuntimeError, match="DAYTONA_API_KEY"):
        p.order(1)


BASE = {"target": 1, "accounts": [1, 2], "shard_base": "s", "run": "run-d"}


@pytest.mark.parametrize("bad, match", [
    ({"gpus": 9}, "gpus"), ({"gpus": 0}, "gpus"), ({"gpus": True}, "gpus"),
    ({"gpu_type": "MI300X"}, "gpu_type"), ({"spot": "yes"}, "spot"),
    ({"snapshot": "s"}, "exactly one"), ({"image": None}, "exactly one"),
    ({"zone": "x"}, "unknown keys"), ({"ttl_minutes": -1}, "ttl_minutes"),
    ({"name_prefix": "Bad_Prefix"}, "name_prefix"), ({"ssh_token_minutes": 1}, "ssh_token_minutes"),
])
def test_config_is_validated(bad, match):
    section = {"kind": "daytona", "image": "rocm/vllm:test", **bad}
    with pytest.raises(ValueError, match=match):
        from_config(dict(BASE, provider=section), "run-d", sdk=FakeSdk, client=FakeDaytona())


def test_kind_daytona_plugs_into_the_pool_route(monkeypatch):
    cfg = dict(BASE, provider={"kind": "daytona", "image": "rocm/vllm:test", "gpus": 8, "spot": True})
    assert not is_national_compute(cfg) and is_national_compute(BASE)
    ops = cli.pool_ops(cfg)
    assert isinstance(ops.provider, DaytonaProvider) and ops.provider.slots == [1, 2]
    assert ops.provider.gpu_type == "MI355X" and ops.provider.spot
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    problems = pool_run.daytona_problems(cfg)
    assert any("DAYTONA_API_KEY" in p for p in problems)
    assert pool_run.daytona_problems(dict(cfg, provider={"kind": "daytona", "image": "x", "gpus": 12})) == [
        "daytona provider: gpus must be an integer 1..8, got 12"]


class DaytonaOps:
    """PoolOps = DaytonaProvider (supply) + a fake node route: the pool policy is shared."""

    def __init__(self, provider, dt, clock):
        self.p, self.dt, self.clock = provider, dt, clock
        self.nodes = {}
        self.store = SimpleNamespace(objects=_Objects())

    def order(self, account):
        return self.p.order(account)

    def poll(self, m):
        box = self.dt.boxes.get(m.ticket)
        if box is not None and box.state == "creating":
            self.dt.start(m.ticket)
        return self.p.status(m.account, m.ticket)

    def start(self, m, shards):
        self.nodes[m.ticket] = {"joined_at": self.clock() + 300, "continues": m.replaces_shard}

    def probe(self, m):
        n = self.nodes.get(m.ticket)
        box = self.dt.boxes.get(m.ticket)
        if n is None or box is None or box.state != "started":
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


class _Objects(dict):
    def read(self, k):
        return self.get(k)

    def write(self, k, v):
        self[k] = v

    def list(self, prefix):
        return sorted(k for k in self if k.startswith(prefix))


def _pool(dt, t, **pc):
    clock = lambda: t[0]  # noqa: E731
    ops = DaytonaOps(_provider(dt), dt, clock)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-d")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="s", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0,
                                 elastic=el.ElasticConfig(max_orders=4, breaker_failures=2), **pc),
                      ops, clock=clock, log=lambda m: None, events=events)
    return pool, ops, events


def test_pool_fences_and_refills_an_evicted_spot_sandbox():
    t = [0.0]
    dt = FakeDaytona()
    pool, ops, events = _pool(dt, t)
    for _ in range(10):
        pool.tick()
        t[0] += 60
    first = pool.members[0]
    assert first.state == READY and first.lease_until is None
    dt.evict(first.ticket)  # Daytona deleted it for on-demand capacity
    pool.tick()
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert first.state == GONE and "evicted" in names and "shard_fenced" in names
    assert any(first.shard in k for k in ops.store.objects)  # the fence that frees its claims
    for _ in range(10):
        t[0] += 60
        pool.tick()
    second = pool.members[1]
    assert second.state == READY and ops.nodes[second.ticket]["continues"] == first.shard
    assert second.ticket != first.ticket and dt.boxes[second.ticket].labels[LABEL_RUN] == "run-d"


def test_pool_adopts_a_labelled_orphan_and_a_stopping_pool_deletes_it():
    t = [0.0]
    dt = FakeDaytona()
    pool, ops, events = _pool(dt, t)
    dt.boxes["yeto-smp-run-d-50"] = FakeSandbox("yeto-smp-run-d-50", {LABEL_RUN: "run-d", LABEL_SLOT: "1"},
                                                state="started")
    pool.tick()
    assert any(e["event"] == "nc_elastic_adopted" and e["ticket"] == "yeto-smp-run-d-50" for e in events.events)
    dt.boxes["yeto-smp-run-d-60"] = FakeSandbox("yeto-smp-run-d-60", {LABEL_RUN: "run-d", LABEL_SLOT: "2"},
                                                state="started")
    pool.stop("test")
    pool._patrol(t[0])
    assert dt.boxes["yeto-smp-run-d-60"].state == "destroyed"
    assert any(e["event"] == "nc_elastic_orphan_lease" for e in events.events)


def test_spot_create_failures_trip_the_breaker():
    t = [0.0]
    dt = FakeDaytona()
    dt.fail_create = RuntimeError("no spot capacity for MI355X")
    pool, ops, events = _pool(dt, t)
    for _ in range(6):
        pool.tick()
        t[0] += 60
    assert len([c for c in dt.calls if c[0] == "create"]) <= 2
    assert not any(b.state != "destroyed" for b in dt.boxes.values())


def test_a_daytona_pool_runs_the_sampler_as_a_process():
    from yeto.launch.daytona_sampler import DaytonaSamplerRoute

    provider = DaytonaProvider("run-x", slots=[0], snapshot="yeto-sampler", sdk=object(), client=object())
    assert isinstance(provider.sampler_route, DaytonaSamplerRoute)


def test_daytona_pool_warnings_name_sandboxes_not_national_compute_leases():
    cfg = dict(BASE, provider={"kind": "daytona", "image": "rocm/vllm:test"})
    warning = pool_run.lease_warning(cfg, "pool.json")
    assert "National Compute" not in warning and "auto-stop off" in warning and "pool-stop --config pool.json" in warning
    banner = pool_run.billing_banner("pool.json", [{"account": 1, "ticket": "yeto-smp-run-d-1"}], national=False)
    assert "sandbox" in banner and "nc down" not in banner and "lease_until" not in banner.split("\n")[2]
    assert "nc down" in pool_run.billing_banner("pool.json", [], national=True)
    assert "National Compute" in pool_run.lease_warning(BASE, "pool.json")


def test_sandboxes_go_to_the_resource_ledger_and_yeto_cleanup_deletes_them(tmp_path, monkeypatch):
    from yeto.cost import billing, cleanup
    from yeto.cost.ledger import LEDGER_ENV, ResourceLedger
    from yeto.launch.pool import provider as providers

    path = tmp_path / "resources.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    dt = FakeDaytona()
    p = _provider(dt, price_usd_h=40.0)
    assert p.hourly_cost() == 40.0
    a, b = p.order(1), p.order(2)
    dt.fail_create = RuntimeError("no capacity")
    with pytest.raises(RuntimeError):
        p.order(1)  # a create that left nothing: closed in the ledger as never created
    p.release(2, b)
    assert providers.ledger_kinds()["daytona_sandbox"].kind == "daytona"
    open_ = ResourceLedger(path).open_resources()
    assert [r.key for r in open_] == [f"daytona_sandbox:{a}"] and open_[0].attrs["usd_per_hour"] == 40.0
    costs = billing.resource_costs(open_, open_[0].created_at + 1800)
    assert costs.items[0].rate_usd_per_hour == 40.0

    deleted = []
    ops = SimpleNamespace(daytona_delete=lambda r: deleted.append(r.id) or p.release(1, r.id))
    assert cleanup.cleanup(ResourceLedger(path), ops, dry_run=True)[0].ok and not deleted
    acts = cleanup.cleanup(ResourceLedger(path), ops)
    assert [x.ok for x in acts] == [True] and deleted == [a]
    assert ResourceLedger(path).open_resources() == []


def test_an_adopted_sandbox_that_vanished_is_released_without_a_delete():
    """A labelled orphan the pool adopted (no local record row) that Daytona then drops
    (evicted past the 24 h it stays readable, or a deleted on-demand sandbox reading not
    found): nothing is left to delete, so its release succeeds instead of refusing it as
    foreign every tick (the member stayed draining, and billed, forever)."""
    t = [0.0]
    dt = FakeDaytona()
    pool, ops, events = _pool(dt, t)
    name = "yeto-smp-run-d-50"
    dt.boxes[name] = FakeSandbox(name, {LABEL_RUN: "run-d", LABEL_SLOT: "1"}, state="started")
    pool.tick()
    assert any(e["event"] == "nc_elastic_adopted" and e["ticket"] == name for e in events.events)
    del dt.boxes[name]
    pool.stop("test")
    member = next(m for m in pool.members if m.ticket == name)
    assert member.state == GONE
    assert not any(e["event"].endswith("drain_failed") for e in events.events)
    assert not [c for c in dt.calls if c[0] == "delete"]
    p = _provider(dt)
    p.release(1, "yeto-smp-run-d-51")  # never existed, never recorded: nothing to delete
    with pytest.raises(PermissionError):
        p.release(1, "yeto-smp-run-x-1")  # another run's name space is still refused


def test_a_transient_get_error_never_reads_as_a_vanished_sandbox(tmp_path):
    """Only an explicit not-found counts as gone: an API outage on release raises and
    leaves the row open (billing still tracked), never marks it released."""
    dt = FakeDaytona()
    p = _provider(dt, record=tmp_path / "rec.json")
    name = "yeto-smp-run-d-60"
    dt.fail_get = True
    with pytest.raises(ConnectionError):
        p.release(1, name)
    assert name not in p._rows()


def test_record_writes_from_two_processes_keep_every_ticket(tmp_path):
    """A manual release in another process while the pool runs: the record's
    read-modify-write holds a file lock, not only the provider's thread lock."""
    import multiprocessing
    import time as real_time

    record = tmp_path / "rec.json"
    ctx = multiprocessing.get_context("fork")

    def work(slot):
        p = _provider(FakeDaytona(), record=record)
        save = p._save

        def slow_save(rows):
            real_time.sleep(0.01)
            save(rows)

        p._save = slow_save
        for i in range(10):
            p._update(f"yeto-smp-run-d-{slot}-{i}", open=False)

    procs = [ctx.Process(target=work, args=(s,)) for s in (1, 2)]
    for q in procs:
        q.start()
    for q in procs:
        q.join(30)
        assert q.exitcode == 0
    assert len(json.loads(record.read_text())) == 20
    assert sorted(f.name for f in tmp_path.iterdir()) == ["rec.json", "rec.lock"]


def _alias_writer(worker: int, rounds: int) -> None:
    p = _provider(FakeDaytona())
    for i in range(rounds):
        p._write_alias(f"yeto-smp-run-d-w{worker}-{i}", f"tok{worker}-{i}", None)


def test_ssh_config_aliases_survive_concurrent_writer_processes():
    """The ssh config is one file for every sandbox and every process on the state dir
    (a pool and a manual command): a read-modify-write held only under a thread lock
    lets one process drop the aliases another just wrote."""
    import multiprocessing  # noqa: PLC0415

    workers, rounds = 6, 25
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_alias_writer, args=(w, rounds)) for w in range(workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    text = (cli.tunnel.state_dir() / "ssh_config").read_text()
    missing = [(w, i) for w in range(workers) for i in range(rounds)
               if f"Host yeto-nc-yeto-smp-run-d-w{w}-{i}\n" not in text]
    assert missing == []
    assert not [f for f in (cli.tunnel.state_dir()).iterdir() if f.name.endswith(".tmp")]


def _nc_alias_writer(rounds: int, ssh_config: str) -> None:
    from yeto.launch.nc import client as ncc  # noqa: PLC0415
    from yeto.launch.nc import node_pool  # noqa: PLC0415

    for i in range(rounds):
        lease = ncc.NodeLease(1, f"nc-{i}", "192.0.2.9", "ubuntu", "", None, None, 8, None)
        node_pool.write_ssh_alias(lease, ssh_config=Path(ssh_config))


def test_ssh_config_aliases_survive_nc_and_daytona_writers(tmp_path):
    """NC (``node_pool.write_ssh_alias``) and Daytona write the same alias file; a
    lock held by only one of them still lets the other drop its aliases."""
    import multiprocessing  # noqa: PLC0415

    rounds = 25
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_alias_writer, args=(0, rounds)),
             ctx.Process(target=_nc_alias_writer, args=(rounds, str(tmp_path / "ssh_cfg")))]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    text = (cli.tunnel.state_dir() / "ssh_config").read_text()
    missing = [i for i in range(rounds) if f"Host yeto-nc-yeto-smp-run-d-w0-{i}\n" not in text]
    missing += [f"nc-{i}" for i in range(rounds) if f"Host yeto-nc-nc-{i}\n" not in text]
    assert missing == []
