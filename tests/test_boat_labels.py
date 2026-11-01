"""CPU tests of the boat sandbox labels and orphan sweeps (S29 D15), on a fake boat client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from yeto.rl.harness.harbor import boat_labels, boat_pool
from yeto.rl.harness.harbor.boat_client import BoatQuota
from yeto.rl.harness.harbor.boat_labels import (
    Labels,
    OrphanWatchdog,
    SandboxInfo,
    labelled_name,
    labels_from_env,
    parse_labels,
    select,
    sweep,
)

from test_harbor_boat import SPEC, FakeBoat, _no_sleep


class LabelBoat(FakeBoat):
    """FakeBoat with names and a listing (the D15 surface)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.names: dict[str, str] = {}
        self.extra: list[SandboxInfo] = []  # sandboxes of other processes / runs
        self.fail_delete: set[str] = set()
        self.list_calls = 0

    def set_name(self, sandbox_id: str, name: str) -> None:
        self.names[sandbox_id] = name

    def list_sandboxes(self) -> list[SandboxInfo]:
        self.list_calls += 1
        own = [SandboxInfo(i, self.names.get(i, ""), "running") for i in sorted(self.live)]
        return own + [b for b in self.extra if b.sandbox_id not in self.deleted]

    def delete(self, sandbox_id: str) -> None:
        if sandbox_id in self.fail_delete:
            raise RuntimeError("HTTP 500")
        super().delete(sandbox_id)


def box(sandbox_id: str, run: str, island: str, inc: str, state: str = "running") -> SandboxInfo:
    return SandboxInfo(sandbox_id, labelled_name(Labels(run, island, inc), "task"), state)


def test_name_round_trip_and_sanitizing():
    name = labelled_name(Labels("s29/run 1", "s29-spot-0", "ab12cd34"), "task__x__env")
    assert name == "ytl1.s29-run-1.s29-spot-0.ab12cd34.task__x__env"
    assert parse_labels(name) == Labels("s29-run-1", "s29-spot-0", "ab12cd34")
    long = labelled_name(Labels("r" * 80, "i" * 80, "c" * 80), "b" * 200)
    assert len(long) == 120 and parse_labels(long) == Labels("r" * 32, "i" * 32, "c" * 32)
    for foreign in ("", None, "task__x__env", "ytl1", "ytl1..a.b", "ytl2.r.i.c.x", "my.ytl1.r.i.c"):
        assert parse_labels(foreign) is None


def test_labels_from_env():
    assert labels_from_env({}) is None
    assert labels_from_env({"YETO_BOAT_RUN_ID": "  "}) is None
    assert labels_from_env({"YETO_BOAT_RUN_ID": "r1", "YETO_BOAT_ISLAND": "s0",
                            "YETO_BOAT_INCARNATION": "c1"}) == Labels("r1", "s0", "c1")


def test_sweep_deletes_only_this_runs_island_and_never_foreign_sandboxes():
    fake = LabelBoat()
    fake.extra = [
        box("old", "r1", "s0", "inc-old"),
        box("mine", "r1", "s0", "inc-new"),          # the current incarnation: kept
        box("other-island", "r1", "s1", "inc-x"),     # another live island of this run: kept
        box("other-run", "r2", "s0", "inc-old"),      # another run: kept
        SandboxInfo("unlabelled", "task__abc__env", "running"),  # not ours: kept
        SandboxInfo("someone", "ytl1-lookalike", "running"),
        box("stopped", "r1", "s0", "inc-old", state="archived"),  # nothing to delete
    ]
    assert sweep(fake, run="r1", island="s0", keep_inc="inc-new") == ["old"]
    assert fake.deleted == ["old"]
    with pytest.raises(ValueError):
        select([], run="")


def test_sweep_survives_a_failed_delete():
    fake = LabelBoat()
    fake.extra = [box("a", "r1", "s0", "old"), box("b", "r1", "s0", "old")]
    fake.fail_delete = {"a"}
    assert sweep(fake, run="r1", island="s0", keep_inc="new") == ["b"]


def test_a_sandbox_is_named_after_run_island_incarnation(monkeypatch):
    monkeypatch.setenv("YETO_BOAT_RUN_ID", "r1")
    monkeypatch.setenv("YETO_BOAT_ISLAND", "s29-spot-0")
    monkeypatch.setenv("YETO_BOAT_INCARNATION", "c0ffee")
    fake = LabelBoat()
    from yeto.rl.harness.harbor.boat_sandbox import BoatSandbox

    sandbox = BoatSandbox(fake, BoatQuota(2, 12), name="task__abc__env", sleep=_no_sleep, poll_interval_s=0.0)

    async def go():
        await sandbox.start(SPEC)
        return dict(fake.names)

    names = asyncio.run(go())
    assert names == {"bx_1": "ytl1.r1.s29-spot-0.c0ffee.task__abc__env"}
    asyncio.run(sandbox.close())


def test_no_run_id_no_label_and_a_client_without_set_name_still_works(monkeypatch):
    monkeypatch.delenv("YETO_BOAT_RUN_ID", raising=False)
    fake = LabelBoat()
    from yeto.rl.harness.harbor.boat_sandbox import BoatSandbox

    async def go(client):
        s = BoatSandbox(client, BoatQuota(2, 12), name="t", sleep=_no_sleep, poll_interval_s=0.0)
        await s.start(SPEC)
        await s.close()

    asyncio.run(go(fake))
    assert fake.names == {}
    monkeypatch.setenv("YETO_BOAT_RUN_ID", "r1")
    plain = FakeBoat()  # no set_name: skipped
    asyncio.run(go(plain))
    assert plain.deleted == ["bx_1"]


def test_sampler_start_labels_the_process_and_sweeps_earlier_incarnations(monkeypatch):
    from yeto.rl.spot_rollout.sampler import label_boat_sandboxes

    fake = LabelBoat()
    fake.extra = [box("pre-1", "r1", "s29-spot-0", "inc0"), box("pre-2", "r1", "s29-spot-0", "inc0"),
                  box("peer", "r1", "s29-spot-1", "incP"), box("foreign", "r9", "s29-spot-0", "inc0")]
    monkeypatch.setattr(boat_labels, "_default_client", lambda: fake)
    for key in ("YETO_BOAT_ISLAND", "YETO_BOAT_INCARNATION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("YETO_BOAT_RUN_ID", raising=False)
    assert label_boat_sandboxes({"sampler_id": "s29-spot-0"}, "inc1") == []  # no run id: off
    assert fake.list_calls == 0
    monkeypatch.setenv("YETO_BOAT_RUN_ID", "r1")
    assert label_boat_sandboxes({"sampler_id": "s29-spot-0"}, "inc1") == ["pre-1", "pre-2"]
    import os

    assert os.environ["YETO_BOAT_ISLAND"] == "s29-spot-0" and os.environ["YETO_BOAT_INCARNATION"] == "inc1"


def test_sampler_start_sweep_errors_do_not_stop_the_sampler(monkeypatch):
    def broken():
        raise RuntimeError("no BOAT_API_KEY")

    monkeypatch.setattr(boat_labels, "_default_client", broken)
    env = {"YETO_BOAT_RUN_ID": "r1", "YETO_BOAT_ISLAND": "s0", "YETO_BOAT_INCARNATION": "c1"}
    assert boat_labels.sweep_stale_incarnations(env) == []


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_watchdog_sweeps_a_superseded_incarnation():
    fake, clock = LabelBoat(), Clock()
    fake.extra = [box("old", "r1", "s0", "a"), box("new", "r1", "s0", "b"), box("x", "r2", "s0", "a")]
    dog = OrphanWatchdog("r1", lease_s=600, check_s=60, client_factory=lambda: fake, clock=clock)
    dog.observe("s0", "a")
    assert dog.poll() == [] and fake.list_calls == 0  # nothing to do: no listing
    dog.observe("s0", "b")  # a newer process of island s0 writes
    clock.t += 1
    assert dog.poll() == []  # throttled by check_s
    clock.t += 60
    assert dog.poll() == ["old"]
    assert fake.deleted == ["old"]


def test_watchdog_sweeps_an_island_silent_past_its_lease_once():
    fake, clock = LabelBoat(), Clock()
    fake.extra = [box("s0-a", "r1", "s0", "a"), box("s1-a", "r1", "s1", "a"), box("r2", "r2", "s0", "a")]
    events = []
    dog = OrphanWatchdog("r1", lease_s=600, check_s=60, client_factory=lambda: fake, clock=clock,
                         emit=lambda name, **kw: events.append((name, kw)))
    dog.observe("s0", "a")
    dog.observe("s1", "a")
    clock.t += 500
    dog.observe("s1", "a")  # s1 is alive
    assert dog.poll() == []
    clock.t += 200  # s0 silent for 700 s > lease
    assert dog.poll() == ["s0-a"]
    assert events and events[0][0] == "rl_boat_orphans_deleted" and events[0][1]["island"] == "s0"
    fake.extra.append(box("s0-a2", "r1", "s0", "a"))
    clock.t += 100
    assert dog.poll() == []  # swept once per (island, incarnation)
    assert "r2" not in fake.deleted


def test_watchdog_never_raises_and_retries_a_supersession():
    clock = Clock()
    calls = []

    def factory():
        calls.append(1)
        raise RuntimeError("no token")

    dog = OrphanWatchdog("r1", lease_s=0, check_s=0, client_factory=factory, clock=clock)
    dog.observe("s0", "a")
    dog.observe("s0", "b")
    assert dog.poll() == [] and dog.poll() == []
    assert len(calls) == 2  # the supersession was kept for the next check


def test_watchdog_from_env():
    assert OrphanWatchdog.from_env({}) is None
    dog = OrphanWatchdog.from_env({"YETO_BOAT_RUN_ID": "r1", "YETO_BOAT_ORPHAN_LEASE_S": "900"})
    assert dog.run == "r1" and dog.lease_s == 900
    assert OrphanWatchdog.from_env({"YETO_BOAT_RUN_ID": "r1"}).lease_s == boat_labels.DEFAULT_ORPHAN_LEASE_S


def test_relay_pump_feeds_the_watchdog_with_item_incarnations():
    from yeto.rl.spot_rollout.relay_buffer import HEADER_SCHEMA, RelayPump, encode_item

    class Buffer:
        read_cursor = 0

        def entries(self):
            return []

        def refresh(self):
            pass

        def __len__(self):
            return 0

    def item(i, inc):
        header = {"schema": HEADER_SCHEMA, "sample_id": f"g{i}", "behavior_version": 0, "tokens": 1,
                  "sampler_round": {"sampler": "s0", "incarnation": inc, "rollout_id": 0}}
        return SimpleNamespace(item_id=i, data=encode_item(header, b"not-a-group"))

    class Consumer:
        def __init__(self):
            self.acked = []

        def take(self, **_):
            return [item(1, "a"), item(2, "b")]

        def ack(self, item_id):
            self.acked.append(item_id)

    seen = []
    dog = SimpleNamespace(observe=lambda s, i: seen.append((s, i)))
    pump = RelayPump(Consumer(), Buffer(), watchdog=dog)
    pump.step()  # the items are rejected (no cuts), but their incarnations were seen
    assert seen == [("s0", "a"), ("s0", "b")]


def test_idle_vm_ttl_default_is_ten_minutes():
    assert boat_pool.DEFAULT_IDLE_VM_TTL_S == 600


def test_spot_plan_env_sets_labels_and_short_ttls():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools/probes/s29_spot/spot_plan.py"
    spec = importlib.util.spec_from_file_location("spot_plan_d15", path)
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    env = plan.env(data="/d", public_ip="1.2.3.4", run_id="r1", island=0)
    assert env["YETO_BOAT_RUN_ID"] == "r1" and env["YETO_BOAT_ISLAND"] == "s29-spot-0"
    assert env["YETO_BOAT_POOL_IDLE_VM_TTL_S"] == "600"
    assert int(env["YETO_BOAT_POOL_IDLE_TTL_S"]) < 600
    assert "YETO_BOAT_RUN_ID" not in plan.env(data="/d", public_ip="1.2.3.4")


def test_d16_the_relay_pump_watchdog_prints_its_deletions(capsys):
    """S29 D16: maybe_start_relay_pump gave the watchdog no emitter, so
    rl_boat_orphans_deleted never reached the trainer log."""
    import json

    from yeto.rl.event_echo import PREFIX
    from yeto.rl.harness.harbor.boat_labels import OrphanWatchdog
    from yeto.rl.spot_rollout.relay_buffer import pump_event_printer

    class _Client:
        def __init__(self):
            from yeto.rl.harness.harbor.boat_labels import SandboxInfo
            self.boxes = [SandboxInfo("bx_1", "ytl1.r1.s29-spot-1.aaa.t", "running")]
            self.deleted = []

        def list_sandboxes(self):
            return [b for b in self.boxes if b.sandbox_id not in self.deleted]

        def delete(self, sandbox_id):
            self.deleted.append(sandbox_id)

    now = [0.0]
    dog = OrphanWatchdog.from_env({"YETO_BOAT_RUN_ID": "r1", "YETO_BOAT_ORPHAN_LEASE_S": "10"},
                                  emit=pump_event_printer(1), client_factory=_Client, clock=lambda: now[0])
    dog.observe("s29-spot-1", "aaa")
    now[0] = 11.0
    assert dog.poll() == ["bx_1"]
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith(PREFIX)][0]
    record = json.loads(line[len(PREFIX):])
    assert record["event"] == "rl_boat_orphans_deleted" and record["island_id"] == 1
    assert record["sandboxes"] == ["bx_1"] and record["reason"] == "island lease expired"
