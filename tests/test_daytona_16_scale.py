"""16 Daytona samplers at once, without Daytona, a GPU or an object store: the pool's
tick with 16 members, the provider under concurrent starts and rate refusals, 16
release downloads under store throttling, and 16 dispenser holders on one ledger."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from test_nc_daytona_provider import (  # noqa: F401
    DaytonaOps,
    FakeDaytona,
    FakeSdk,
    _Objects,
    _provider,
    _state,  # noqa: F401 - the state-dir fixture (autouse)
)

from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY, IslandPool

N = 16


class SlowOps(DaytonaOps):
    """A fake node route whose ticket read and probe each take one ssh-like round trip."""

    def __init__(self, provider, dt, clock, rtt_s=0.15):
        super().__init__(provider, dt, clock)
        self.rtt_s = rtt_s
        self.inflight = self.peak = 0
        self.lock = threading.Lock()

    def _rt(self):
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        time.sleep(self.rtt_s)
        with self.lock:
            self.inflight -= 1

    def poll(self, m):
        self._rt()
        return super().poll(m)

    def probe(self, m):
        self._rt()
        return super().probe(m)


def _pool16(dt, t, *, parallel=16, rtt_s=0.15):
    clock = lambda: t[0]  # noqa: E731
    ops = SlowOps(_provider(dt, slots=range(1, 2 * N + 1)), dt, clock, rtt_s=rtt_s)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-d")
    pool = IslandPool(PoolConfig(target=N, accounts=list(range(1, 2 * N + 1)), shard_base="s", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0,
                                 elastic=el.ElasticConfig(max_orders=64, breaker_failures=4)),
                      ops, clock=clock, log=lambda m: None, events=events)
    pool.watch_parallel = parallel
    return pool, ops, events


def _run(pool, t, ticks):
    for _ in range(ticks):
        pool.tick()
        t[0] += 60


def test_sixteen_members_are_watched_in_parallel_and_three_reclaims_are_refilled():
    t = [0.0]
    dt = FakeDaytona()
    pool, ops, events = _pool16(dt, t)
    _run(pool, t, 8)
    ready = [m for m in pool.members if m.state == READY]
    assert len(ready) == N and len({m.shard for m in ready}) == N
    # one tick reads 16 tickets and probes 16 nodes: in parallel it costs ~2 round trips, not 32
    pool.tick()
    assert pool.last_tick_s < 8 * ops.rtt_s and ops.peak >= 8, (pool.last_tick_s, ops.peak)
    gone = ready[3], ready[7], ready[11]
    for m in gone:
        dt.evict(m.ticket)
    _run(pool, t, 8)
    assert all(m.state == GONE for m in gone)
    live = [m for m in pool.members if m.state == READY]
    assert len(live) == N
    assert sorted(m.replaces_shard for m in live if m.replaces_shard) == sorted(m.shard for m in gone)
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert names.count("evicted") == 3 and names.count("shard_fenced") >= 3
    pool.stop("test")
    _run(pool, t, 3)
    assert all(m.state == GONE for m in pool.members)
    assert all(b.state == "destroyed" for b in dt.boxes.values())


def test_serial_watch_is_still_available_and_slower():
    t = [0.0]
    dt = FakeDaytona()
    pool, ops, _ = _pool16(dt, t, parallel=1, rtt_s=0.03)
    _run(pool, t, 8)
    pool.tick()
    assert ops.peak == 1 and pool.last_tick_s >= 2 * N * ops.rtt_s * 0.9


def test_a_failed_probe_in_the_parallel_pass_is_an_unreachable_node():
    t = [0.0]
    dt = FakeDaytona()
    pool, ops, events = _pool16(dt, t, rtt_s=0.0)
    _run(pool, t, 8)
    victim = next(m for m in pool.members if m.state == READY)
    real = ops.probe

    def probe(m):
        if m.ticket == victim.ticket:
            raise OSError("ssh: connect timed out")
        return real(m)

    ops.probe = probe
    pool.tick()
    assert victim.unreachable_since is not None and victim.state == READY  # a blip first, not gone yet


# -- the provider under concurrency ------------------------------------------------------------


class RateLimited(Exception):
    status_code = 429


class SlowDaytona(FakeDaytona):
    """create blocks like the SDK's (until started) and the API refuses some calls for rate."""

    def __init__(self, *, create_s=0.0, refuse=0):
        super().__init__()
        self.create_s, self.refuse = create_s, refuse
        self.lock = threading.Lock()
        self.release = threading.Event()

    def _maybe_refuse(self):
        with self.lock:
            if self.refuse > 0:
                self.refuse -= 1
                raise RateLimited("429 Too Many Requests")

    def create(self, params, timeout=60):
        self._maybe_refuse()
        if self.create_s:
            self.release.wait(self.create_s)
        box = super().create(params, timeout)
        box.state = "started"
        return box

    def get(self, name):
        self._maybe_refuse()
        return super().get(name)


def test_order_returns_while_a_slow_create_runs_and_reads_waiting():
    dt = SlowDaytona(create_s=30)
    p = _provider(dt, create_wait_s=0.05)
    t0 = time.monotonic()
    names = [p.order(k) for k in range(1, N + 1)]
    assert time.monotonic() - t0 < 5  # 16 orders do not wait 16 starts
    assert [p.status(1, n).state for n in names] == ["waiting"] * N
    dt.release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(p.status(1, n).state != "active" for n in names):
        time.sleep(0.02)
    assert all(p.status(1, n).state == "active" for n in names)


def test_a_create_that_fails_late_ends_the_ticket_and_frees_the_slot():
    dt = SlowDaytona(create_s=0.2)
    p = _provider(dt, create_wait_s=0.01)
    real = dt.create

    def create(params, timeout=60):
        time.sleep(0.1)
        raise RuntimeError("queue timed out")

    dt.create = create
    name = p.order(3)
    assert p.status(3, name).state == "waiting"
    time.sleep(0.3)
    assert p.status(3, name) == el.LeaseStatus("ended", end_reason="error")
    assert p.account_free(3)
    dt.create = real


def test_rate_refusals_are_retried_with_backoff():
    dt = SlowDaytona(refuse=5)
    slept = []
    p = _provider(dt, sleep=slept.append)
    name = p.order(1)
    assert p.status(1, name).state == "active" and p.rate_limited == 5
    assert len(slept) == 5 and all(0 <= s <= 30 for s in slept)
    dt.refuse = 100
    with pytest.raises(RateLimited):  # bounded: api_attempts, then the pool's blip handling
        p.status(1, name)


def test_concurrent_attaches_keep_every_alias(tmp_path):
    from yeto.launch.nc import cli

    dt = FakeDaytona()
    p = _provider(dt, slots=range(1, N + 1))
    names = [p.order(k) for k in range(1, N + 1)]
    for n in names:
        dt.start(n)
    ok = lambda a, c: SimpleNamespace(returncode=0)  # noqa: E731
    threads = [threading.Thread(target=p.attach, args=(p.node(1, n),), kwargs={"ssh_run": ok}) for n in names]
    [th.start() for th in threads]
    [th.join(10) for th in threads]
    text = (cli.tunnel.state_dir() / "ssh_config").read_text()
    assert all(f"Host yeto-nc-{n}" in text for n in names)
    rows = p._rows()
    assert all(rows[n].get("ssh_expires_at") for n in names)


def test_token_renewals_of_sandboxes_created_together_spread_out():
    t = [0.0]
    dt = FakeDaytona()
    p = _provider(dt, slots=range(1, N + 1), ssh_token_minutes=1440, clock=lambda: t[0])
    names = [p.order(k) for k in range(1, N + 1)]
    for n in names:
        dt.start(n)
        p.attach(p.node(1, n), ssh_run=lambda a, c: SimpleNamespace(returncode=0))
    renewed_at: dict[str, float] = {}
    while t[0] < 24 * 3600 and len(renewed_at) < N:
        t[0] += 600
        for n in names:
            p.status(1, n)
            if n not in renewed_at and len(dt.boxes[n].tokens) > 1:
                renewed_at[n] = t[0]
    assert len(renewed_at) == N
    assert len(set(renewed_at.values())) >= 4  # not one storm of 16 token calls in one tick
    assert max(renewed_at.values()) <= 24 * 3600 * 2 / 3 + 600  # each still before a third is left


# -- 16 release downloads at once -------------------------------------------------------------


def test_sixteen_samplers_download_one_release_through_throttling(tmp_path, monkeypatch):
    from test_rl_object_store import FakeS3

    from yeto.cloud.s3_store import S3ObjectStore, TransferSettings
    from yeto.rl.engine.object_store import DirectoryObjectStore
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore

    monkeypatch.setattr("yeto.cloud.s3_store.time.sleep", lambda s: None)

    class Throttled(FakeS3):
        """Every 7th ranged GET has its body cut (an R2 refusal under load)."""

        def __init__(self):
            super().__init__()
            self.gets = 0

        def get_object(self, Bucket, Key, Range=None, IfMatch=None):
            with self.lock:
                self.gets += 1
                cut = Range is not None and self.gets % 7 == 0
            if cut:
                raise ConnectionResetError("reset by peer")
            return super().get_object(Bucket, Key, Range=Range, IfMatch=IfMatch)

    class Mount(CheckpointStore):
        def __init__(self, root, objects):
            super().__init__(root)
            self.objects = objects

    client = Throttled()
    transfer = TransferSettings(part_bytes=5 << 20, concurrency=4, file_concurrency=4, download_processes=1)
    objects = S3ObjectStore("s3://bucket/run", client=client, transfer=transfer)
    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(Mount(shared, DirectoryObjectStore(shared)))

    def weights(directory):
        for i in range(6):  # 6 expert files of 11 MiB: 3 ranges each
            (directory / f"experts-{i}.safetensors").write_bytes(bytes([i]) * (11 << 20))

    trainer.release(1, "h1", weights)
    for f in shared.rglob("*"):  # the trainer's upload, as objects of the bucket
        if f.is_file():
            client.objects[("bucket", f"run/{f.relative_to(shared).as_posix()}")] = f.read_bytes()
    samplers = [PolicyReleaseStore(Mount(tmp_path / f"m{k}", objects), cache_dir=tmp_path / f"cache{k}")
                for k in range(N)]
    got, errors = {}, []

    def pull(k):
        try:
            got[k] = samplers[k].resolve(samplers[k].latest())
        except Exception as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=pull, args=(k,)) for k in range(N)]
    [th.start() for th in threads]
    [th.join(60) for th in threads]
    assert not errors and len(got) == N
    assert all((got[k] / "experts-5.safetensors").stat().st_size == 11 << 20 for k in range(N))
    assert client.gets > N * 18  # cut bodies were read again, none failed a download


def test_a_failed_release_download_resumes_without_fetching_whole_files_again(tmp_path, monkeypatch):
    from yeto.rl.engine import policy_release
    from yeto.rl.engine.object_store import DirectoryObjectStore
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore

    class Mount(CheckpointStore):
        def __init__(self, root, objects):
            super().__init__(root)
            self.objects = objects

    shared = tmp_path / "bucket"
    trainer = PolicyReleaseStore(Mount(shared, DirectoryObjectStore(shared)))

    def weights(directory):
        for i in range(8):
            (directory / f"experts-{i}.safetensors").write_bytes(bytes([i]) * 1000)

    trainer.release(1, "h1", weights)
    sampler = PolicyReleaseStore(Mount(tmp_path / "m", DirectoryObjectStore(shared)), cache_dir=tmp_path / "cache")
    real, fetched, fail = policy_release.download_file, [], {"at": "experts-6.safetensors"}

    def download(objects, key, path):
        if fail["at"] and key.endswith(fail["at"]):
            raise ConnectionResetError("reset by peer")
        fetched.append(key.rsplit("/", 1)[-1])
        return real(objects, key, path)

    monkeypatch.setattr(policy_release, "download_file", download)
    monkeypatch.setattr(sampler, "_kept_names", lambda: set(), raising=False)
    # one file at a time, so the files before the failure are whole on disk
    monkeypatch.setattr(sampler.store.objects, "transfer", SimpleNamespace(file_concurrency=1), raising=False)
    with pytest.raises(ConnectionResetError):
        sampler.resolve(sampler.latest())
    first = list(fetched)
    assert "experts-5.safetensors" in first
    fail["at"] = None
    fetched.clear()
    directory = sampler.resolve(sampler.latest())
    assert sorted(p.name for p in directory.glob("experts-*")) == [f"experts-{i}.safetensors" for i in range(8)]
    assert not set(fetched) & {n for n in first if n.startswith("experts-")}  # whole files were kept
    assert sampler.resumed_files >= len([n for n in first if n.startswith("experts-")])
    assert not list(directory.rglob("*.yeto-get.tmp"))


# -- 16 dispenser holders on one ledger --------------------------------------------------------


def _counting_store():
    """FakeObjects counting each request kind under a lock (the threads race on ``+= 1``)."""
    from collections import Counter

    from test_store_dispense import FakeObjects

    class Counting(FakeObjects):
        def __init__(self):
            super().__init__()
            self.ops: Counter[str] = Counter()
            self.ops_lock = threading.Lock()
            self.frozen = None  # a listing every scan returns: N holders that scanned at once

        def _count(self, kind):
            with self.ops_lock:
                self.ops[kind] += 1

        def read(self, key):
            self._count("read")
            return super().read(key)

        def write(self, key, data):
            self._count("write")
            return super().write(key, data)

        def exists(self, key):
            self._count("exists")
            return super().exists(key)

        def create(self, key, data):
            won = super().create(key, data)
            self._count("create" if won else "create_412")
            return won

        def list_meta(self, prefix):
            self._count("list")
            return self.frozen if self.frozen is not None else super().list_meta(prefix)

    return Counting()


def test_sixteen_holders_claim_and_settle_each_task_once(tmp_path):
    from test_store_dispense import _open, _rows, _slots, _status, _write

    from yeto.rl.engine.task_dispense import DispenseEnded

    store = _counting_store()
    tasks = 640
    path = _write(tmp_path, _rows(tasks))
    samplers = [_open(path, store, holder=f"s{k}", inc="i", groups=4, config={"lease_s": 10**7}) for k in range(N)]
    got: dict[int, list[int]] = {k: [] for k in range(N)}
    barrier = threading.Barrier(N)

    def run(k):
        barrier.wait()
        try:
            while True:
                spots = _slots(samplers[k](0))
                got[k] += spots
                samplers[k].written(0, spots)
        except DispenseEnded:
            pass

    threads = [threading.Thread(target=run, args=(k,)) for k in range(N)]
    store.ops.clear()
    [th.start() for th in threads]
    [th.join(60) for th in threads]
    ops = dict(store.ops)  # before the status read below
    every = [p for spots in got.values() for p in spots]
    assert sorted(every) == list(range(tasks))
    assert _status(store)["totals"]["trained"] == tasks
    # The request budget, by kind (the listings of idle holders polling a fake clock depend on
    # thread scheduling and are not bounded here; production waits scan_s between them):
    lost = sum(s.stats["claim_lost"] for s in samplers)
    assert ops.get("create") == 2 * tasks, ops  # exactly one claim and one settle per task
    assert ops.get("create_412", 0) == lost, ops  # the only conflicts are claims lost to a race
    assert lost <= tasks * (N - 1), ops  # a holder loses a slot's attempt at most once
    assert not ops.get("read") and not ops.get("write") and not ops.get("exists"), ops  # no re-reads
    assert ops.get("list") == sum(s.stats["scans"] for s in samplers), ops  # one listing per scan


def test_holders_that_scanned_the_same_listing_spread_their_claims(tmp_path):
    """The herd, deterministically: N holders claim in turn from one listing taken before
    any claim. In plan order holder k loses the k*want slots its predecessors took (want*N*(N-1)/2
    lost claims); after its first loss a holder goes on from its own offset."""
    from test_store_dispense import _open, _rows, _write

    store = _counting_store()
    want, tasks = 4, 640
    path = _write(tmp_path, _rows(tasks))
    samplers = [_open(path, store, holder=f"s{k}", inc="i", groups=want, config={"lease_s": 10**7})
                for k in range(N)]
    store.frozen = store.list_meta("")
    store.ops.clear()
    claimed = [samplers[k]._claim(want)[0] for k in range(N)]
    assert all(len(c) == want for c in claimed)
    assert len({p for c in claimed for p in c}) == want * N  # no slot twice
    lost = store.ops["create_412"]
    assert lost == sum(s.stats["claim_lost"] for s in samplers)
    herd = want * N * (N - 1) // 2
    assert lost <= herd // 4, (lost, herd)
    # an uncontended holder keeps the plan order: the first one takes the first want slots
    assert claimed[0] == list(range(want))
