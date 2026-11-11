"""Modal rollout provider (yeto.launch.nc.modal_provider) over a fake Modal client:
the shared pool policy (start, eviction -> fence -> refill, scale down, release)
with function calls instead of nodes. No Modal call is made."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, elastic as el, pool_run, provider as prov
from yeto.launch.nc.lease_pool import GONE, READY, LeasePool, PoolConfig
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

    def spawn(self, payload):
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
    client.states[call] = SUCCEEDED  # the sampler exited by itself (or hit the function timeout)
    assert p.status(1, ticket) == el.LeaseStatus("ended", end_reason="lost")
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
    with pytest.raises(RuntimeError, match="still runs"):
        p2.release(1, t2)
    assert p2.run_tickets() == [(1, t2)] and client2.states[c2] == RUNNING


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
    pool = LeasePool(PoolConfig(target=1, accounts=[1], shard_base="m", stop_below_ucr=0,
                                min_balance_to_order_ucr=0, elastic=el.ElasticConfig(max_orders=1, start_timeout_s=600)),
                     ops2, clock=clock, log=lambda m: None, events=events)
    ops2.start_in_background = False
    for _ in range(14):
        pool.tick()
        t[0] += 60
    names = [e["event"].removeprefix("nc_elastic_") for e in events.events]
    assert "member_ready" not in names and "start_timeout" in names


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
    pool = LeasePool(PoolConfig(target=2, accounts=[1, 2, 3], shard_base="m", stop_below_ucr=0,
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
    pool = LeasePool(PoolConfig(target=2, accounts=[1, 2, 3], shard_base="m", stop_below_ucr=0,
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
