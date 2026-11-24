"""Modal rollout provider (yeto.launch.nc.modal_provider) over a fake Modal client:
the shared pool policy (start, eviction -> fence -> refill, scale down, release)
with function calls instead of nodes. No Modal call is made."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, pool_run, provider as prov
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY, IslandPool
from yeto.launch.nc.modal_provider import (FAILED, RUNNING, SUCCEEDED, ModalPoolOps, ModalProvider, ModalSdkClient,
                                           from_config, offline_problems, store_env)


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("YETO_RESOURCE_LEDGER", raising=False)


class FakeClient:
    def __init__(self):
        self.calls, self.states, self.log, self.fail_status = {}, {}, [], False
        self.n = 0
        self.report_ready = None  # () -> object store: the call writes its ready marker at once

    def spawn(self, payload, *, wanted=None):
        if wanted is not None and not wanted():
            raise RuntimeError("the island was released")
        self.n += 1
        call = f"fc-{self.n}"
        self.calls[call], self.states[call] = payload, RUNNING
        if self.report_ready is not None and payload.get("ready_key"):
            self.report_ready().write(payload["ready_key"], b"{}")
        self.log.append(("spawn", call, payload["shard"]))
        return call

    def status(self, call):
        if self.fail_status:
            raise ConnectionError("modal control plane")
        return self.states[call]

    def cancel(self, call):
        self.log.append(("cancel", call))
        self.states[call] = FAILED


class Objects(dict):
    def read(self, k):
        return self.get(k)

    def write(self, k, v):
        self[k] = v

    def list(self, prefix):
        return sorted(k for k in self if k.startswith(prefix))


def _cfg(tmp_path, **over):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"event_tape": "/work/sampler/events.jsonl", "prompts": {"groups_per_round": 4},
                                "store": "__RUN_PREFIX__"}))
    env = tmp_path / "r2.env"
    env.write_text("R2_BUCKET=b\n")
    env.chmod(0o600)
    cfg = {"target": 1, "accounts": [1, 2, 3], "shard_base": "m", "run": "run-m", "spec": str(spec),
           "run_prefix": "runs-m", "r2_env_file": str(env),
           "provider": {"kind": "modal", "image": "registry.example/sampler@sha256:00", "secret": "store-secret",
                        "bucket": "b", "gpu": "B300", "gpus": 8, "price_usd_h": 50.0}}
    cfg.update(over)
    return cfg


def _ops(tmp_path, client=None, **over):
    cfg = _cfg(tmp_path, **over)
    client = client or FakeClient()
    provider = from_config(cfg, "run-m", client=client)
    store = SimpleNamespace(objects=Objects())
    ops = ModalPoolOps(cfg, provider=provider, fence_store=store)
    if client.report_ready is None:
        client.report_ready = lambda: ops._store.objects
    return ops, client, store


def test_provider_maps_call_states_and_releases_only_its_own_tickets(tmp_path):
    client = FakeClient()
    p = ModalProvider("run-m", slots=[1, 2], client=client)
    ticket = p.order(1)
    assert ticket == "yeto-smp-run-m-1" and client.log == []  # nothing spawned (or billed) by an order
    assert p.status(1, ticket).state == "active" and not p.account_free(1) and p.account_free(2)
    call = p.spawn(ticket, {"shard": "m-0"})
    assert p.status(1, ticket).state == "active"
    end = p.hard_end(p._rows()[ticket])  # its function timeout: spawned_at + timeout_h
    assert end == pytest.approx(p._rows()[ticket]["spawned_at"] + 24 * 3600)
    client.states[call] = SUCCEEDED  # the sampler exited 0 (the run finished): not lost
    assert p.status(1, ticket) == el.LeaseStatus("active", hard_end=end)
    client.states[call] = FAILED  # it crashed, was cancelled outside, or hit the function timeout
    assert p.status(1, ticket) == el.LeaseStatus("ended", end_reason="lost", hard_end=end)
    client.fail_status = True
    with pytest.raises(ConnectionError):  # a blip to the pool, never an end
        p.status(1, ticket)
    client.fail_status = False
    with pytest.raises(PermissionError):
        p.release(1, "yeto-smp-run-m-99")
    with pytest.raises(PermissionError):
        p.spawn("someone-else", {"shard": "x"})
    assert p.release(1, ticket)["state"] == "ended" and p.run_tickets() == [] and p.account_free(1)
    assert p.status(1, ticket).end_reason == "released"


def test_release_cancels_a_running_call_and_confirms(tmp_path):
    client = FakeClient()
    p = ModalProvider("run-m", slots=[1], client=client)
    ticket = p.order(1)
    call = p.spawn(ticket, {"shard": "m-0"})
    assert p.release(1, ticket)["call"] == call and ("cancel", call) in client.log
    client2 = FakeClient()
    p2 = ModalProvider("run-n", slots=[1], client=client2)
    t2 = p2.order(1)
    c2 = p2.spawn(t2, {"shard": "n-0"})
    client2.cancel = lambda call: None  # Modal took the cancel but the call still runs
    p2.sleep, p2.cancel_wait_s = (lambda s: None), 0.0
    with pytest.raises(RuntimeError, match="still runs"):
        p2.release(1, t2)
    assert p2.run_tickets() == [(1, t2)] and client2.states[c2] == RUNNING


def test_release_waits_for_the_asynchronous_cancel():
    """Modal takes a cancel at once but the call reads RUNNING for a few seconds: the
    release polls instead of failing (a real Modal L4 call read RUNNING right after
    cancel and the pool logged nc_elastic_drain_failed, then retried)."""
    now = [0.0]
    client = FakeClient()
    p = ModalProvider("run-m", slots=[1], client=client, clock=lambda: now[0],
                      sleep=lambda s: now.__setitem__(0, now[0] + s), cancel_wait_s=30.0)
    ticket = p.order(1)
    call = p.spawn(ticket, {"shard": "m-0"})
    ends_at = 6.0
    client.cancel = lambda c: client.log.append(("cancel", c))
    status = client.status
    client.status = lambda c: FAILED if now[0] >= ends_at else status(c)
    assert p.release(1, ticket)["state"] == "ended" and ("cancel", call) in client.log
    assert now[0] >= ends_at and p.run_tickets() == []


def test_a_restart_cancels_the_running_call_first(tmp_path):
    client = FakeClient()
    p = ModalProvider("run-m", slots=[1], client=client)
    ticket = p.order(1)
    first = p.spawn(ticket, {"shard": "m-0"})
    second = p.spawn(ticket, {"shard": "m-0"})
    assert client.log == [("spawn", first, "m-0"), ("cancel", first), ("spawn", second, "m-0")]
    assert p.call_id(ticket) == second


def test_the_start_payload_is_the_rendered_member_spec_on_the_run_store(tmp_path):
    ops, client, _ = _ops(tmp_path)
    m = SimpleNamespace(account=1, ticket=ops.order(1), shard="m-0", data_slot=None, replaces_shard="m-9",
                        predecessor_lost=False)
    ops.start(m, ["m-0", "m-1"])
    payload = next(iter(client.calls.values()))
    spec = payload["spec"]
    assert spec["sampler_id"] == "m-0" and spec["buffer_shards"] == ["m-0", "m-1"] and spec["start_on_latest"]
    assert spec["store"] == spec["buffer_root"] == "/tmp/stage/b/runs/runs-m"
    assert payload["bucket"] == "b" and payload["run_prefix"] == "runs-m"
    assert payload["ready_key"] == f"pool-ready/m-0/{m.ticket}.1.json"
    assert ops.probe(m) == el.Probe(node="ok", sampler="running", joined=True)


def test_a_running_call_is_joined_only_once_it_reports_ready(tmp_path):
    """RUNNING is not joined: the pool waits for the call's ready marker, a call
    that never writes it fails the start by start_timeout_s, and a restarted call
    does not inherit the marker of the call before it."""
    client = FakeClient()
    ops, client, store = _ops(tmp_path, client=client)
    client.report_ready = None  # the calls report nothing by themselves
    m = SimpleNamespace(account=1, ticket=ops.order(1), shard="m-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["m-0"])
    assert ops.probe(m) == el.Probe(node="ok", sampler="running", joined=False) and not ops.ready(m)
    key = ops.provider.ready_key(m.ticket)
    store.objects.write(key, b"{}")  # the sampler joined: the body wrote the marker
    assert ops.probe(m).joined and ops.ready(m)
    ops.restart(m, ["m-0"])
    assert ops.provider.ready_key(m.ticket) != key and not ops.probe(m).joined

    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    ops.provider.release(1, m.ticket)
    (tmp_path / "b").mkdir()
    ops2, client2, _ = _ops(tmp_path / "b")
    client2.report_ready = None
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=1, start_timeout_s=600)),
                      ops2, clock=clock, log=lambda m: None, events=events)
    ops2.start_in_background = False
    for _ in range(14):
        pool.tick()
        t[0] += 60
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "member_ready" not in names and "start_timeout" in names


def test_two_first_spawns_on_threads_deploy_the_app_once():
    """The pool starts islands on background threads; a pool of target 2 spawned both at
    once and Modal refused the second, concurrent deploy (start_failed: 'App deploy name
    ... is already taken')."""
    import threading
    import time as _time

    deploys = []

    class App:
        app_id = None

        def deploy(self, name):
            deploys.append(name)
            _time.sleep(0.2)  # the image build: the other thread arrives meanwhile

    fn = SimpleNamespace(spawn=lambda payload: SimpleNamespace(object_id=f"fc-{payload['shard']}"))
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=lambda app, name: fn))
    client = ModalSdkClient({"image": "i", "secret": "s"}, run_id="run-m", modal=modal)
    client.build_app = lambda: App()
    calls = []
    threads = [threading.Thread(target=lambda s=s: calls.append(client.spawn({"shard": s}))) for s in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert deploys == ["yeto-rollout-run-m"] and sorted(calls) == ["fc-a", "fc-b"]


def test_a_reused_container_does_not_report_the_earlier_calls_join(tmp_path, monkeypatch):
    """Modal ran the next call in the same warm container: the earlier call's tape (with
    its rl_sampler_join) was still on disk and the new call reported ready at once,
    while its sampler still waited for the layout. Only a tape this call's sampler
    writes counts."""
    from yeto.launch.nc.modal_provider import rollout_sampler_body  # noqa: PLC0415

    tape = tmp_path / "tape" / "events.jsonl"
    tape.parent.mkdir()
    (tape.parent / "events.old1.jsonl").write_text('{"event": "rl_sampler_join"}\n')  # the earlier call
    objects, polls = Objects(), [0]

    class Proc:
        returncode = 0

        def poll(self):
            polls[0] += 1
            if polls[0] == 3:
                (tape.parent / "events.new2.jsonl").write_text('{"event": "rl_sampler_join"}\n')
            return None if polls[0] < 5 else 0

    seen = []
    payload = {"shard": "m-0", "bucket": "b", "run_prefix": "r", "ready_key": "pool-ready/m-0/t.2.json",
               "spec": {"event_tape": str(tape)}}
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
    real_write = objects.write
    objects.write = lambda k, v: seen.append(polls[0]) or real_write(k, v)
    out = rollout_sampler_body(payload, popen=lambda *a, **k: Proc(), objects=objects, sleep=lambda s: None,
                               root=str(tmp_path / "rollout"))
    assert out["reported_ready"] and seen == [3]  # not at the first poll (the old tape)


def test_a_sampler_that_finished_ends_the_pool_without_a_refill(tmp_path):
    """The syncer told the run's samplers the run finished: the sampler exits 0, the call
    returns. The pool reads a finished sampler (released, nothing ordered again), not a
    lost island (a real Modal run refilled it and the new island found no run)."""
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    ops, client, _ = _ops(tmp_path)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=4)),
                      ops, clock=clock, log=lambda m: None, events=events)
    ops.start_in_background = False
    for _ in range(3):
        pool.tick()
        t[0] += 30
    [m] = [m for m in pool.members if m.state == READY]
    client.states[ops.provider.call_id(m.ticket)] = SUCCEEDED
    for _ in range(4):
        pool.tick()
        t[0] += 30
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "sampler_finished" in names and "evicted" not in names
    assert names.count("ordered") == 1 and pool.finished and not pool.live()


def test_the_call_body_reports_ready_once_the_tape_has_the_join(tmp_path, monkeypatch):
    from yeto.launch.nc.modal_provider import rollout_sampler_body  # noqa: PLC0415

    tape = tmp_path / "tape" / "events.jsonl"
    tape.parent.mkdir()
    objects = Objects()
    polls = [0]

    class Proc:
        returncode = 0

        def poll(self):
            polls[0] += 1
            if polls[0] == 2:  # the engine loaded the model and joined
                (tape.parent / "events.inc1.jsonl").write_text('{"event": "rl_sampler_join"}\n')
            return None if polls[0] < 4 else 0

    payload = {"shard": "m-0", "bucket": "b", "run_prefix": "r", "ready_key": "pool-ready/m-0/t.1.json",
               "spec": {"event_tape": str(tape)}}
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
    out = rollout_sampler_body(payload, popen=lambda *a, **k: Proc(), objects=objects, sleep=lambda s: None,
                               root=str(tmp_path / "rollout"))
    assert out["reported_ready"] and out["rc"] == 0 and "pool-ready/m-0/t.1.json" in objects


def test_store_env_reads_the_secret_names():
    env = store_env({"bucket": "b", "run_prefix": "r", "env": {"X": "1"}}, {"AWS_ENDPOINT_URL": "https://e"})
    assert env["YETO_RL_STORE_S3_URI"] == "s3://b/runs/r" and env["YETO_RL_STORE_S3_CREDENTIALS"] == "aws-env"
    assert env["X"] == "1"
    with pytest.raises(ValueError, match="endpoint"):
        store_env({"bucket": "b", "run_prefix": "r"}, {})


def test_pool_over_modal_refills_a_lost_island_and_scales_down(tmp_path):
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    ops, client, store = _ops(tmp_path)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=2, accounts=[1, 2, 3], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=5)),
                      ops, clock=clock, log=lambda m: None, events=events)
    ops.start_in_background = False  # deterministic: start inside the tick
    for _ in range(6):
        pool.tick()
        t[0] += 60
    ready = [m for m in pool.members if m.state == READY]
    assert len(ready) == 2 and len([x for x in client.log if x[0] == "spawn"]) == 2
    first = ready[0]
    client.states[ops.provider.call_id(first.ticket)] = FAILED  # the container died (preempted host)
    for _ in range(6):
        pool.tick()
        t[0] += 60
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert first.state == GONE and "shard_fenced" in names
    assert len([m for m in pool.members if m.state == READY]) == 2  # refilled
    scale = tmp_path / "scale.json"
    scale.write_text(json.dumps({"run": "run-m", "target": 1, "seq": 1, "retire": []}))
    pool.scale_path, pool.scale_run = scale, "run-m"
    for _ in range(6):
        pool.tick()
        t[0] += 60
    live = [m for m in pool.members if m.state != GONE]
    assert len(live) == 1
    cancelled = {c for kind, c, *_ in client.log if kind == "cancel"}
    assert len(ops.provider.run_tickets()) == 1 and cancelled


TRACEBACK = ['  File "/root/yeto/rl/engine/resume.py", line 372, in store_for',
             "ValueError: store path 's3://b/runs/r/fleet/dispense' is a URI"]


def test_a_failed_call_tells_the_pool_its_samplers_exit_code_and_stderr_tail(tmp_path):
    """A sampler that exits on a configuration error 15 s in: the pool's evicted event
    carries the call's rc and the last stderr lines, not only ``end_reason=lost``."""
    from yeto.launch.nc.modal_provider import failure_detail  # noqa: PLC0415

    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    client = FakeClient()
    client.results = {}
    client.failure = lambda call: failure_detail(client.results.get(call))
    ops, client, _ = _ops(tmp_path, client=client)
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=4)),
                      ops, clock=clock, log=lambda m: None, events=events)
    ops.start_in_background = False
    for _ in range(3):
        pool.tick()
        t[0] += 30
    [m] = [m for m in pool.members if m.state == READY]
    call = ops.provider.call_id(m.ticket)
    client.results[call] = {"rc": 1, "secs": 15.1, "reported_ready": False, "stderr_tail": ["Traceback", *TRACEBACK]}
    client.states[call] = FAILED
    status = ops.provider.status(m.account, m.ticket)
    assert status == el.LeaseStatus("ended", end_reason="lost", hard_end=ops.provider.hard_end(
        ops.provider._rows()[m.ticket]))  # the detail is no part of its identity
    assert status.detail.startswith("sampler exited rc=1 after 15.1 s; stderr tail: Traceback | ")
    assert "is a URI" in status.detail
    pool.tick()
    [evicted] = [e for e in events.events if e["event"].endswith("_evicted")]
    assert evicted["end_reason"] == "lost" and "rc=1" in evicted["detail"] and "is a URI" in evicted["detail"]


def test_failure_detail_reads_a_raised_call_and_bounds_its_length():
    from yeto.launch.nc.modal_provider import FAILURE_DETAIL_CHARS, failure_detail  # noqa: PLC0415

    assert failure_detail({"rc": 0, "secs": 3}) is None and failure_detail(None) is None
    assert failure_detail({"rc": 2, "secs": 1.0}) == "sampler exited rc=2 after 1.0 s"
    assert failure_detail(None, RuntimeError("function timeout")) == "call raised RuntimeError: function timeout"
    long = failure_detail({"rc": 1, "secs": 1, "stderr_tail": ["x" * 5000, "the end"]})
    assert len(long) == FAILURE_DETAIL_CHARS and long.startswith("...") and long.endswith("the end")
    p = ModalProvider("run-m", slots=[1], client=SimpleNamespace(
        status=lambda call: FAILED, spawn=lambda payload: "fc-1",
        failure=lambda call: (_ for _ in ()).throw(ConnectionError("modal"))))
    ticket = p.order(1)
    p.spawn(ticket, {"shard": "m-0"})
    assert p.status(1, ticket).detail == "failure not read: ConnectionError: modal"


def test_the_call_body_returns_the_samplers_stderr_tail(tmp_path, monkeypatch, capsys):
    import io  # noqa: PLC0415

    from yeto.launch.nc.modal_provider import STDERR_TAIL_LINES, rollout_sampler_body  # noqa: PLC0415

    lines = [f"log {i}" for i in range(40)] + TRACEBACK
    got = {}

    class Proc:
        returncode = 1
        stderr = io.StringIO("".join(line + "\n" for line in lines))

        def poll(self):
            return 1

    def popen(argv, **kwargs):
        got.update(kwargs)
        return Proc()

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
    out = rollout_sampler_body({"shard": "m-0", "bucket": "b", "run_prefix": "r", "spec": {}}, popen=popen,
                               objects=Objects(), sleep=lambda s: None, root=str(tmp_path / "rollout"))
    assert got["stderr"] is not None and got["text"]
    assert out["rc"] == 1 and out["stderr_tail"] == lines[-STDERR_TAIL_LINES:]
    assert "log 0" in capsys.readouterr().err, "every line still reaches the container's log"


def test_a_retired_modal_island_drains_then_fences_then_stops(tmp_path):
    """--retire T with the drain scale down: the sampler is asked to claim no new task,
    stays billed until it answers drained, then is fenced and its call cancelled."""
    from yeto.rl.engine import store_dispense  # noqa: PLC0415

    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    ops, client, _ = _ops(tmp_path, sampler={"dispense": {"backend": "store", "namespace": "run-m/dispense"}})
    store = SimpleNamespace(objects=store_dispense.DirObjects(tmp_path / "store"))  # the run store's object API
    ops._store = store
    ops.start_in_background = False
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=2, accounts=[1, 2, 3], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(scale_drain_timeout_s=3600)),
                      ops, clock=clock, log=lambda m: None, events=events)
    for _ in range(4):
        pool.tick()
        t[0] += 60
    victim = [m for m in pool.members if m.state == READY][0]
    scale = tmp_path / "scale.json"
    scale.write_text(json.dumps({"run": "run-m", "target": 1, "seq": 1, "retire": [victim.ticket], "mode": "drain"}))
    pool.scale_path, pool.scale_run = scale, "run-m"
    pool.tick()
    t[0] += 60
    call = ops.provider.call_id(victim.ticket)
    drain = json.loads(store.objects.read(store_dispense.drain_key("run-m/dispense", victim.shard)))
    assert drain["drain"] and victim.retiring_since is not None and client.states[call] == RUNNING
    pool.tick()  # not answered yet: still billed, still running
    assert client.states[call] == RUNNING
    store.objects.write(store_dispense.drained_key("run-m/dispense", victim.shard), json.dumps(
        {"seq": drain["seq"]}).encode())  # the sampler holds no claim any more
    for _ in range(3):
        t[0] += 60
        pool.tick()
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "scale_draining" in names and "shard_fenced" in names
    assert names.index("scale_draining") < names.index("shard_fenced")
    assert victim.state == GONE and ("cancel", call) in client.log
    assert len([m for m in pool.members if m.state != GONE]) == 1


def test_the_billing_banner_of_a_modal_pool_uses_no_national_compute_wording():
    member = {"account": 2, "ticket": "yeto-smp-run-m-1", "shard": "m-0", "state": "ready", "lease_until": None}
    text = pool_run.billing_banner("pool.json", [member], why="test", national=False, kind="modal")
    assert "STILL BILLING: 1 Modal island(s)" in text and "NODES" not in text
    assert "slot 2 ticket yeto-smp-run-m-1" in text and "lease_until" not in text and "account" not in text
    assert "yeto.launch.nc down --account" not in text
    nc = pool_run.billing_banner("pool.json", [dict(member, lease_until=1.0)], national=True, kind="nc")
    assert "NODES STILL BILLING" in nc and "account 2" in nc and "lease_until" in nc


def test_kind_modal_routes_to_its_own_pool_ops_and_dry_run_checks(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    assert prov.provider_kind(cfg) == "modal"
    ops = cli.pool_ops(cfg)
    assert isinstance(ops, ModalPoolOps) and isinstance(ops.provider, ModalProvider)
    bad = _cfg(tmp_path, provider={"kind": "modal", "gpu": "B300"})
    problems = offline_problems(bad)
    assert "modal provider: image is required" in problems and "modal provider: bucket is required" in problems
    assert offline_problems(_cfg(tmp_path, provider={"kind": "modal", "zone": "x"})) == \
        ["modal provider: unknown keys ['zone']"]
    assert "Cloud rollout islands (modal)" in pool_run.lease_warning(cfg, "pool.json")


def test_pool_dry_run_of_a_modal_pool_orders_nothing(tmp_path, capsys):
    cfg = _cfg(tmp_path, provider={"kind": "modal", "gpu": "B300"})
    rc = pool_run.dry_run(cfg, "pool.json")
    out = capsys.readouterr().out
    assert rc == 1 and "problem: modal provider: image is required" in out and "nothing ordered" in out


def test_the_sdk_client_spawns_through_a_deployed_function():
    class FakeModal:
        def __init__(self):
            self.spawned = []
            outer = self

            class Function:
                @staticmethod
                def from_name(app, name):
                    return SimpleNamespace(spawn=lambda payload: outer.spawned.append((app, name, payload))
                                           or SimpleNamespace(object_id="fc-9"))

            class FunctionCall:
                @staticmethod
                def from_id(call_id):
                    def get(timeout=None):
                        raise TimeoutError

                    return SimpleNamespace(get=get, cancel=lambda **kw: outer.spawned.append(("cancel", kw)))

            self.Function, self.FunctionCall = Function, FunctionCall

    modal = FakeModal()
    client = ModalSdkClient({"image": "i", "secret": "s"}, run_id="Run_M", modal=modal)
    client._deployed = True  # the deploy builds an image: not in a CPU test
    assert client.app_name == "yeto-rollout-run-m"
    assert client.spawn({"shard": "m-0"}) == "fc-9" and modal.spawned[0][:2] == ("yeto-rollout-run-m",
                                                                                 "rollout_sampler")
    assert client.status("fc-9") == RUNNING
    client.cancel("fc-9")
    assert modal.spawned[-1] == ("cancel", {"terminate_containers": True})
    # a returned call is SUCCEEDED only when its sampler exited 0 (the body returns its rc)
    results = {"fc-0": {"rc": 0}, "fc-1": {"rc": 1}}
    modal.FunctionCall = SimpleNamespace(from_id=lambda c: SimpleNamespace(get=lambda timeout=None: results[c]))
    assert client.status("fc-0") == SUCCEEDED and client.status("fc-1") == FAILED
    results["fc-1"] = {"rc": 1, "secs": 15.1, "stderr_tail": ["ValueError: bad spec"]}
    assert client.failure("fc-0") is None
    assert client.failure("fc-1") == "sampler exited rc=1 after 15.1 s; stderr tail: ValueError: bad spec"

    def raised(timeout=None):
        raise RuntimeError("container OOM")

    modal.FunctionCall = SimpleNamespace(from_id=lambda c: SimpleNamespace(get=raised))
    assert client.failure("fc-2") == "call raised RuntimeError: container OOM"


def test_concurrent_spawns_keep_every_call_in_the_record(tmp_path):
    """The pool starts Modal islands on background threads (``start_in_background``):
    two spawns at once each rewrite the local record; neither may drop the other's call
    id (a dropped id is a billed GPU call the pool can no longer see or cancel)."""
    import threading
    import time as real_time

    client = FakeClient()
    p = ModalProvider("run-m", slots=[1, 2], client=client)
    tickets = [p.order(1), p.order(2)]
    save = p._save

    def slow_save(rows):  # widens the read-modify-write window of each writer
        real_time.sleep(0.2)
        save(rows)

    p._save = slow_save
    errors: list[BaseException] = []

    def spawn(ticket):
        try:
            p.spawn(ticket, {"shard": ticket})
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=spawn, args=(t,)) for t in tickets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sorted(p.call_id(t) for t in tickets) == sorted(client.calls)


def test_concurrent_orders_from_two_processes_keep_every_ticket(tmp_path):
    """A manual command (another process) may rewrite the record while the pool runs:
    the read-modify-write holds a file lock too, not only a thread lock, and the
    temporary file is per writer (two processes can share a thread id)."""
    import multiprocessing

    record = tmp_path / "modal-calls.json"
    ctx = multiprocessing.get_context("fork")

    def work(slot):
        p = ModalProvider("run-m", slots=[1, 2], client=FakeClient(), record=record)
        save = p._save

        def slow_save(rows):
            time.sleep(0.01)
            save(rows)

        p._save = slow_save
        for _ in range(10):
            p.order(slot)

    procs = [ctx.Process(target=work, args=(s,)) for s in (1, 2)]
    for q in procs:
        q.start()
    for q in procs:
        q.join(30)
        assert q.exitcode == 0
    rows = json.loads(record.read_text())
    assert len(rows) == 20 and sorted(r["n"] for r in rows.values()) == list(range(1, 21))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["modal-calls.json", "modal-calls.lock"]


# -- the pool's deployed app (the app outlived the finished pool: idle GPU billed) ----

class AppClient(FakeClient):
    """A fake client with a deployed app: ``app_id`` once a spawn deployed it, and
    ``stop_app`` like :meth:`ModalSdkClient.stop_app`."""

    def __init__(self, app_id="ap-1", stop_error=None):
        super().__init__()
        self.app_name, self.next_app_id, self.app_id = "yeto-rollout-run-m", app_id, None
        self.stops, self.stop_error = [], stop_error

    def spawn(self, payload, *, wanted=None):
        self.app_id = self.app_id or self.next_app_id  # the first spawn deploys
        return super().spawn(payload, wanted=wanted)

    def stop_app(self, *, expect_id=None, by_name=True):
        if self.app_id is None and not by_name:
            return "not deployed"  # nothing of ours to stop, nothing looked up
        self.stops.append(expect_id)
        if self.stop_error is not None:
            error, self.stop_error = self.stop_error, None
            raise error
        for call, state in self.states.items():  # stopping the app ends every container
            if state == RUNNING:
                self.states[call] = FAILED
        self.app_id = None
        return "stopped"


def _app_pool(tmp_path, client, *, target=1, **over):
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    ops, client, _ = _ops(tmp_path, client=client, **over)
    ops.start_in_background = False
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=target, accounts=[1, 2, 3], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=5)),
                      ops, clock=clock, log=lambda m: None, events=events)

    def ticks(n):
        for _ in range(n):
            pool.tick()
            t[0] += 60

    return pool, ops, events, ticks


def _names(events):
    return [e["event"].removeprefix("nc_elastic_") for e in events.events]


def test_a_finished_pool_stops_its_modal_app(tmp_path):
    """Every sampler exited 0: the pool drains its islands and stops the deployed app, so
    no idle container is left billing until Modal's scale-down."""
    client = AppClient()
    pool, ops, events, ticks = _app_pool(tmp_path, client, target=2)
    ticks(4)
    ready = [m for m in pool.members if m.state == READY]
    assert len(ready) == 2 and client.stops == []
    record = json.loads(ops.provider.app_record.read_text())
    assert record["app_id"] == "ap-1" and record["app_name"] == "yeto-rollout-run-m"
    for m in ready:
        client.states[ops.provider.call_id(m.ticket)] = SUCCEEDED
    ticks(3)
    names = _names(events)
    assert pool.stopped and not pool.live() and "sampler_finished" in names
    assert client.stops == ["ap-1"]  # once, checked against the recorded id
    assert names.index("pool_stopped") < names.index("pool_ended")
    [ended] = [e for e in events.events if e["event"].endswith("pool_ended")]
    assert ended["result"] == "stopped" and ended["app_id"] == "ap-1"
    assert json.loads(ops.provider.app_record.read_text())["stop_result"] == "stopped"
    pool.drain_all("pool-stop", sleep=lambda s: None)  # a later pool-stop: nothing stopped twice
    assert client.stops == ["ap-1"]


def test_a_finished_pool_without_stop_on_finish_stops_the_app_after_its_last_island(tmp_path):
    """``stop_on_finish`` off: the other islands finish on their own; the app stays up
    until the last one ended, then the pool stops it."""
    client = AppClient()
    pool, ops, events, ticks = _app_pool(tmp_path, client, target=2)
    pool.ecfg.stop_on_finish = False
    ticks(4)
    first, second = [m for m in pool.members if m.state == READY]
    client.states[ops.provider.call_id(first.ticket)] = SUCCEEDED
    ticks(2)
    assert pool.finished and not pool.stopped and second.state == READY and client.stops == []
    client.states[ops.provider.call_id(second.ticket)] = SUCCEEDED
    ticks(3)
    assert pool.stopped and not pool.live() and client.stops == ["ap-1"]


def test_replacing_a_lost_island_keeps_the_app(tmp_path):
    """A lost island (its container died) is fenced and refilled into the same app: the
    pool runs on, nothing stops the app."""
    client = AppClient()
    pool, ops, events, ticks = _app_pool(tmp_path, client, target=2)
    ticks(4)
    lost = [m for m in pool.members if m.state == READY][0]
    client.states[ops.provider.call_id(lost.ticket)] = FAILED
    ticks(6)
    names = _names(events)
    assert lost.state == GONE and "shard_fenced" in names
    assert len([m for m in pool.members if m.state == READY]) == 2  # refilled
    assert not pool.stopped and client.stops == [] and "pool_ended" not in names
    assert len([x for x in client.log if x[0] == "spawn"]) == 3


def test_a_sigterm_drain_stops_the_app(tmp_path):
    """SIGTERM / Ctrl-C: the loop's drain_all releases every island and stops the app."""
    client = AppClient()
    pool, ops, events, ticks = _app_pool(tmp_path, client, target=2)
    ticks(4)
    assert len([m for m in pool.members if m.state == READY]) == 2
    left = pool_run._drain_all(pool, "signal SIGTERM", sleep=lambda s: None)
    assert left == [] and client.stops == ["ap-1"] and "pool_ended" in _names(events)


def test_a_failed_app_stop_is_an_event_and_retried_by_the_next_drain(tmp_path):
    client = AppClient(stop_error=ConnectionError("modal control plane"))
    pool, ops, events, ticks = _app_pool(tmp_path, client)
    ticks(3)
    pool.stop("signal SIGTERM")
    assert not pool.live() and client.stops == ["ap-1"] and "pool_end_failed" in _names(events)
    assert "stopped_at" not in json.loads(ops.provider.app_record.read_text())
    pool.drain_all("pool-stop", sleep=lambda s: None)
    assert client.stops == ["ap-1", "ap-1"] and "pool_ended" in _names(events)


def test_pool_stop_after_the_pool_process_died_stops_the_recorded_app(tmp_path):
    """The pool loop was killed (SIGKILL) with an island up: the app stays deployed. A
    pool-stop from another process reads the state file and the app record, releases
    the island and stops the app by name, checked against the recorded id."""
    cfg = _cfg(tmp_path)
    client = AppClient()
    pool, ops, events, ticks = _app_pool(tmp_path, client)
    ticks(3)
    [m] = [m for m in pool.members if m.state == READY]
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    pool.save(state)  # the process dies here: no stop, no app stop
    later = AppClient()  # a new process: no app handle, nothing deployed by it
    later.states = client.states
    provider = from_config(cfg, "run-m", client=later)
    ops2 = ModalPoolOps(cfg, provider=provider, fence_store=SimpleNamespace(objects=Objects()))
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(cfg))
    assert pool_run.stop_pool(cfg, str(path), ops=ops2, sleep=lambda s: None) == 0
    assert later.stops == ["ap-1"] and ("cancel", ops.provider.call_id(m.ticket)) in later.log
    assert json.loads(provider.app_record.read_text())["stopped_why"] == "pool-stop"


def test_no_app_stop_for_a_pool_that_never_deployed(tmp_path):
    """A pool stopped before any spawn: an app of that name is not this pool's."""
    client = AppClient()
    pool, ops, events, ticks = _app_pool(tmp_path, client)
    pool.stop("pool-stop")
    assert client.stops == []
    [ended] = [e for e in events.events if e["event"].endswith("pool_ended")]
    assert ended["result"] == "not deployed"


def test_a_released_island_spawns_no_call(tmp_path):
    """A background start that ran after the pool released its island would spawn a call
    no member drains (and, after the app stop, deploy the app again)."""
    ops, client, _ = _ops(tmp_path)
    ticket = ops.provider.order(1)
    ops.provider.release(1, ticket)
    with pytest.raises(RuntimeError, match="released"):
        ops.provider.spawn(ticket, {"shard": "m-0"})
    assert client.n == 0


class DeployingModal:
    """A fake ``modal`` for :class:`ModalSdkClient`: the app's deploy blocks until
    ``finish`` is set (a slow image build), spawns and stops are recorded."""

    def __init__(self, app_id="ap-5"):
        import threading

        self.started, self.finish = threading.Event(), threading.Event()
        self.spawned, self.stopped, self.apps = [], [], {}
        outer = self

        class App:
            def __init__(self):
                self.app_id = None

            def deploy(self, name):
                outer.started.set()
                assert outer.finish.wait(10), "the test never let the deploy finish"
                self.app_id = app_id
                outer.apps[name] = self

            def stop(self):
                outer.stopped.append(self.app_id)

        def lookup(name):
            if name not in outer.apps:
                raise type("NotFoundError", (Exception,), {})(f"App '{name}' not found")
            return outer.apps[name]

        self.new_app = App
        fn = SimpleNamespace(spawn=lambda payload: outer.spawned.append(payload) or SimpleNamespace(object_id="fc-1"))
        self.Function = SimpleNamespace(from_name=lambda app, name: fn)
        self.App = SimpleNamespace(lookup=lookup)


def test_a_pool_stopped_while_its_app_deploys_stops_the_app_and_spawns_nothing(tmp_path):
    """The pool stopped (process exit) while the first island's background start was
    still deploying the app (a slow image build): the stop saw no app id and logged
    "not deployed"; the deploy then finished, the call was spawned and one GPU container
    billed on. Now the stop waits for the deploy, the start spawns nothing for its
    released island, and the app is stopped."""
    import threading

    modal = DeployingModal()
    cfg = _cfg(tmp_path)
    sdk = ModalSdkClient(cfg["provider"], run_id="run-m", modal=modal)
    sdk.build_app = modal.new_app
    provider = from_config(cfg, "run-m", client=sdk)
    ops = ModalPoolOps(cfg, provider=provider, fence_store=SimpleNamespace(objects=Objects()))
    t = [0.0]
    events = el.EventLog(log=lambda m: None, clock=lambda: t[0], run="run-m")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=4)),
                      ops, clock=lambda: t[0], log=lambda m: None, events=events)
    for _ in range(3):  # ordered, then started on a background thread
        pool.tick()
        t[0] += 30
        if modal.started.is_set():
            break
    assert modal.started.wait(5), "the background start did not reach the deploy"
    threading.Timer(0.3, modal.finish.set).start()  # the image build ends while the stop waits
    pool.stop("process exit")
    assert modal.stopped == ["ap-5"] and modal.spawned == []
    [ended] = [e for e in events.events if e["event"].endswith("pool_ended")]
    assert ended["result"] == "stopped" and ended["app_id"] == "ap-5"
    assert not pool.live()


def test_a_start_that_reaches_the_deploy_after_the_release_deploys_nothing():
    modal = DeployingModal()
    modal.finish.set()
    sdk = ModalSdkClient({"image": "i", "secret": "s"}, run_id="run-m", modal=modal)
    sdk.build_app = modal.new_app
    with pytest.raises(RuntimeError, match="released"):
        sdk.spawn({"shard": "m-0"}, wanted=lambda: False)
    assert not modal.started.is_set() and modal.spawned == [] and not modal.apps


def test_pool_stop_after_a_process_died_during_its_deploy_stops_the_app_by_name(tmp_path):
    """The pool process died during its first deploy: no app id was recorded, but the
    record says a spawn may have deployed. pool-stop stops the app of that name."""
    cfg = _cfg(tmp_path)

    class DiesInDeploy(AppClient):
        def spawn(self, payload, *, wanted=None):
            raise KeyboardInterrupt  # the process ends inside the deploy

    provider = from_config(cfg, "run-m", client=DiesInDeploy())
    ticket = provider.order(1)
    with pytest.raises(KeyboardInterrupt):
        provider.spawn(ticket, {"shard": "m-0"})
    doc = json.loads(provider.app_record.read_text())
    assert doc["intent_at"] is not None and "app_id" not in doc
    modal = DeployingModal()
    modal.finish.set()
    modal.new_app().deploy("yeto-rollout-run-m")  # what the dead process left deployed
    later = from_config(cfg, "run-m", client=ModalSdkClient(cfg["provider"], run_id="run-m", modal=modal))
    out = later.stop_app("pool-stop")
    assert out["result"] == "stopped" and out["app_id"] == "ap-5" and modal.stopped == ["ap-5"]
    assert later.stop_app("pool-stop")["result"] == "already stopped"


def test_the_sdk_client_stops_its_own_app_by_handle_and_a_recorded_one_by_name():
    stopped = []

    class App:
        def __init__(self, app_id):
            self.app_id = app_id

        def deploy(self, name):
            pass

        def stop(self):
            stopped.append(self.app_id)

    class NotFoundError(Exception):
        pass

    apps = {"yeto-rollout-run-m": App("ap-7")}

    def lookup(name):
        if name not in apps:
            raise NotFoundError(f"App '{name}' not found")
        return apps[name]

    modal = SimpleNamespace(App=SimpleNamespace(lookup=lookup))
    own = ModalSdkClient({"image": "i", "secret": "s"}, run_id="run-m", modal=modal)
    own.build_app = lambda: App("ap-3")
    own.deploy()
    assert own.app_id == "ap-3"
    assert own.stop_app(expect_id="ap-3") == "stopped" and stopped == ["ap-3"]  # its handle, no lookup
    assert own.app_id is None and not own._deployed  # a later spawn deploys again
    # another process (pool-stop after a crash): by name, only the recorded deployment
    later = ModalSdkClient({"image": "i", "secret": "s"}, run_id="run-m", modal=modal)
    assert later.stop_app(expect_id="ap-other") == "other deployment" and stopped == ["ap-3"]
    assert later.stop_app(expect_id="ap-7") == "stopped" and stopped == ["ap-3", "ap-7"]
    del apps["yeto-rollout-run-m"]
    assert later.stop_app(expect_id="ap-7") == "absent"


def test_the_idle_window_of_a_pool_container_is_short_by_default_and_a_knob():
    seen = {}

    class App:
        def __init__(self, name):
            pass

        def function(self, **kw):
            seen.update(kw)
            return lambda body: body

    image = SimpleNamespace()
    image.dockerfile_commands = lambda cmds: image
    image.entrypoint = lambda a: image
    image.env = lambda e: image
    image.add_local_python_source = lambda *a, **k: image
    modal = SimpleNamespace(App=App, Image=SimpleNamespace(from_registry=lambda ref: image),
                            Secret=SimpleNamespace(from_name=lambda n: n))
    ModalSdkClient({"image": "i", "secret": "s"}, run_id="run-m", modal=modal).build_app()
    assert seen["scaledown_window"] == 2
    ModalSdkClient({"image": "i", "secret": "s", "scaledown_window_s": 120}, run_id="run-m",
                   modal=modal).build_app()
    assert seen["scaledown_window"] == 120


@pytest.mark.parametrize("value", [1, 3601, 2.5, "60", True])
def test_a_bad_idle_window_is_a_config_error(tmp_path, value):
    cfg = _cfg(tmp_path)
    cfg["provider"]["scaledown_window_s"] = value
    with pytest.raises(ValueError, match="scaledown_window_s"):
        from_config(cfg, "run-m", client=FakeClient())
    assert any("scaledown_window_s" in p for p in offline_problems(cfg))


# -- a call Modal has not scheduled yet (no GPU worker of its shape free) ----------------------
class QueuedClient(FakeClient):
    """Calls wait for a worker until ``placed``: Modal reads them RUNNING and a cancel of an
    unscheduled one is taken but the call keeps reading RUNNING (as INT saw on an L4 pool)."""

    def __init__(self):
        super().__init__()
        self.placed: set[str] = set()

    def spawn(self, payload, *, wanted=None):
        report, self.report_ready = self.report_ready, None  # an unscheduled call reports nothing
        try:
            return super().spawn(payload, wanted=wanted)
        finally:
            self.report_ready = report

    def place(self, call):
        """A worker takes the call: it runs, and its sampler joins."""
        self.placed.add(call)
        key = self.calls[call].get("ready_key")
        if key and self.report_ready is not None:
            self.report_ready().write(key, b"{}")

    def scheduled(self, call):
        return call in self.placed

    def cancel(self, call):
        self.log.append(("cancel", call))
        if call in self.placed:
            self.states[call] = FAILED


def _ledger(tmp_path, monkeypatch):
    from yeto.cost.ledger import ResourceLedger

    path = tmp_path / "resources.jsonl"
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(path))
    return ResourceLedger(path)


def test_an_unscheduled_call_reads_active_but_not_billing_until_modal_places_it(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path, monkeypatch)
    now = [100.0]
    client = QueuedClient()
    p = ModalProvider("run-m", slots=[1], client=client, clock=lambda: now[0], price_usd_h=1.12,
                      config={"gpu": "L4"})
    ticket = p.order(1)
    assert p.status(1, ticket).billing is False  # reserved, nothing spawned
    call = p.spawn(ticket, {"shard": "m-0"})
    waiting = p.status(1, ticket)
    assert waiting.state == "active" and waiting.billing is False
    assert "waits to be scheduled on a L4 worker" in waiting.detail
    (row,) = [r for r in ledger.resources() if r.kind == "modal_call"]
    assert row.attrs["pending"] is True and row.attrs.get("scheduled_at") is None
    now[0] = 500.0
    client.placed.add(call)
    assert p.status(1, ticket).billing is True
    (row,) = [r for r in ledger.resources() if r.kind == "modal_call"]
    assert row.attrs["scheduled_at"] == 500.0 and p._rows()[ticket]["scheduled_at"] == 500.0
    assert p.status(1, ticket).billing is True  # recorded once: no second lookup needed
    second = p.spawn(ticket, {"shard": "m-0"})  # a restart: its new call waits again
    assert p.status(1, ticket).billing is False and p.call_id(ticket) == second


def test_releasing_an_unscheduled_call_ends_it_with_the_cancel(tmp_path, monkeypatch):
    """INT: a call that never left "waiting to be scheduled" read RUNNING after its cancel
    and the drain failed with "cancel sent but it still runs after 60 s"."""
    ledger = _ledger(tmp_path, monkeypatch)
    now = [0.0]
    client = QueuedClient()
    p = ModalProvider("run-m", slots=[1], client=client, clock=lambda: now[0],
                      sleep=lambda s: now.__setitem__(0, now[0] + s), cancel_wait_s=60.0, price_usd_h=1.12)
    ticket = p.order(1)
    call = p.spawn(ticket, {"shard": "m-0"})
    assert p.release(1, ticket)["state"] == "ended" and ("cancel", call) in client.log
    assert now[0] == 0.0 and client.states[call] == RUNNING  # no wait: it never ran
    (row,) = [r for r in ledger.resources() if r.kind == "modal_call"]
    assert row.released_at is not None
    # a scheduled call that keeps running after its cancel is still a drain failure
    t2 = p.order(1)
    c2 = p.spawn(t2, {"shard": "m-1"})
    client.placed.add(c2)
    client.cancel = lambda c: None
    with pytest.raises(RuntimeError, match="still runs after 60 s"):
        p.release(1, t2)


def test_a_pool_does_not_bill_a_call_waiting_for_a_worker_and_times_its_start_out(tmp_path):
    """No capacity: the member is started, not granted and not billed; the pool says so once
    (member_unscheduled), ends the start at start_timeout_s, releases the call without a
    drain failure and orders a replacement. A call placed later bills from then."""
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    client = QueuedClient()
    ops, client, _store = _ops(tmp_path, client=client)
    ops.provider.clock = clock
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2, 3], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0,
                                 elastic=el.ElasticConfig(max_orders=5, start_timeout_s=600)),
                      ops, clock=clock, log=lambda m: None, events=events)
    ops.start_in_background = False
    for _ in range(8):  # 8 min: started, waiting for a worker
        pool.tick()
        t[0] += 60
    first = pool.members[0]
    assert first.started is not None and first.granted is None and pool.budget.spent_ucr == 0
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert names.count("member_unscheduled") == 1
    for _ in range(4):  # past start_timeout_s: the start ends, the call is cancelled, a new one is ordered
        pool.tick()
        t[0] += 60
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "start_timeout" in names and "drain_failed" not in names
    assert first.state == GONE and pool.budget.spent_ucr == 0
    replacement = [m for m in pool.members if m.state != GONE][-1]
    while ops.provider.call_id(replacement.ticket) is None:  # its start spawns the call
        pool.tick()
        t[0] += 60
    pool.tick()  # it waits for a worker, too
    t[0] += 60
    assert replacement.granted is None and pool.budget.spent_ucr == 0
    client.place(ops.provider.call_id(replacement.ticket))  # a worker frees up
    for _ in range(3):
        pool.tick()
        t[0] += 60
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert replacement.granted is not None and "member_scheduled" in names and pool.budget.spent_ucr > 0


def test_the_ledger_bills_a_pool_call_from_its_scheduled_at():
    from yeto.cost.billing import resource_costs
    from yeto.cost.ledger import Resource

    attrs = {"pool": True, "usd_per_hour": 1.12, "run": "r", "ticket": "t"}
    never = Resource("modal_call", "fc-1", "rollout_sampler", 0.0, {**attrs, "pending": True})
    placed = Resource("modal_call", "fc-2", "rollout_sampler", 0.0, {**attrs, "ticket": "u", "pending": True,
                                                                         "scheduled_at": 3600.0})
    legacy = Resource("modal_call", "fc-3", "rollout_sampler", 0.0, {**attrs, "ticket": "v"})
    cost = {i.key.split(":")[-1] if ":" in i.key else i.key: i for i in resource_costs([never, placed, legacy],
                                                                                        7200.0).items}
    by_id = {k: v for k, v in cost.items()}
    usd = {r.id: next(v for k, v in by_id.items() if r.id in k).estimated_usd for r in (never, placed, legacy)}
    assert usd["fc-1"] == 0.0  # 400 s or 2 h waiting for a worker: nothing ran, nothing billed
    assert usd["fc-2"] == pytest.approx(1.12) and usd["fc-3"] == pytest.approx(2.24)


def test_the_sdk_client_reads_scheduled_from_the_call_graph():
    # modal.call_graph.InputInfo / InputStatus (PENDING 0, SUCCESS 1) without the SDK (not in CI)
    pending, success = 0, 1
    graphs = {}

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            return SimpleNamespace(get_call_graph=lambda: graphs[call_id])

    client = ModalSdkClient({"image": "i", "secret": "s"}, run_id="r", modal=SimpleNamespace(FunctionCall=FunctionCall))

    def info(call, task, status):
        return SimpleNamespace(input_id="in-1", function_call_id=call, task_id=task, status=status,
                               function_name="rollout_sampler", module_name="m", children=[])

    graphs.update({"fc-1": [info("fc-1", "", pending)], "fc-2": [info("fc-2", "ta-1", pending)],
                   "fc-3": [info("fc-3", "", success)], "fc-4": []})
    assert client.scheduled("fc-1") is False
    assert client.scheduled("fc-2") is True and client.scheduled("fc-3") is True
    assert client.scheduled("fc-4") is True  # no input info: billed, never free


def test_a_call_near_its_function_timeout_gets_a_successor_before_it_is_retired(tmp_path):
    """The function timeout is the call's hard end: a successor is ordered replace_lead_s before
    it on a free slot, the old call is drained once the successor is ready (make before break),
    and a call that ends at its timeout is an expiry, not an eviction."""
    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    ops, client, _ = _ops(tmp_path, provider={**_cfg(tmp_path)["provider"], "timeout_h": 3})
    ops.provider.clock = clock
    events = el.EventLog(log=lambda m: None, clock=clock, run="run-m")
    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="m", stop_below_ucr=0,
                                 min_balance_to_order_ucr=0, replace_lead_s=1800, end_margin_s=300),
                      ops, clock=clock, log=lambda m: None, events=events)
    ops.start_in_background = False
    for _ in range(3):
        pool.tick()
        t[0] += 60
    [first] = [m for m in pool.members if m.state == READY]
    assert first.lease_until == pytest.approx(ops.provider._rows()[first.ticket]["spawned_at"] + 10800)
    assert len(pool.members) == 1
    t[0] = first.lease_until - 1800 + 1  # inside the lead: a successor on the free slot
    for _ in range(3):
        pool.tick()
        t[0] += 60
    successor = next(m for m in pool.members if m.ticket != first.ticket)
    assert successor.replaces is None or successor.replaces == first.ticket
    assert successor.state == READY and first.state == GONE  # drained after its successor came up
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "evicted" not in names and "lease_end_without_ready_replacement" not in names
    timed_out = el.LeaseStatus("ended", end_reason="lost", hard_end=successor.lease_until)
    assert el.classify(timed_out, None, now=successor.lease_until + 1, unreachable_since=None,
                       cfg=el.ElasticConfig(), joined=True).kind == el.EXPIRED
