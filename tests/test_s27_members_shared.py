"""One island member table for the task dispenser, the spot subscribers and
the sample relay (S27 dispatch merge; syncer ``members.rs``).

A sampler process opens three channels to the actor syncer with one id
(``sampler_holder_id`` = ``subscriber_id_for``) and one member incarnation:
TASK_HELLO (rows), SUBSCRIBE (weights) and RELAY_HELLO (samples). When a newer
incarnation of the same sampler joins on any one channel, the old process is
isolated on all three at once: its subscriber gets ``fenced: subscriber``, its
relay producer ``fenced: relay``, its dispenser requests ``fenced: dispense``
and its unreported rows go back to the queue, long before the member lease.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from test_spot_rollout_syncer import KEY, P, Learner, finish, layout, needs_cargo, start_syncer, syncer_binary  # noqa: F401
from test_spot_rollout_syncer import wait_for, wait_log

from yeto.rl.engine import task_dispense as td

FP = bytes(range(32))
ROWS = 40
LEASE_S = 600.0  # isolation must come from the supersede, never from the lease
ISOLATED_WITHIN_S = 3.0


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", KEY)


class Sampler:
    """The three head channels of one sampler process (one member incarnation)."""

    def __init__(self, port: int, sampler_id: str, incarnation: int) -> None:
        from yeto.rl.spot_rollout.release_store import subscriber_id_for

        self.addr = ("127.0.0.1", port)
        self.id = subscriber_id_for(sampler_id)
        assert self.id == td.sampler_holder_id(sampler_id)  # one id on every channel
        self.incarnation = incarnation
        self.task = None
        self.subscriber = None
        self.producer = None

    def open_task(self) -> td.DispenseConnection:
        self.task = td.DispenseConnection(self.addr, learner_id=self.id, incarnation=self.incarnation,
                                          rows=ROWS, fingerprint=FP, key=KEY.encode(), timeout_s=10.0)
        return self.task

    def open_subscriber(self):
        from yeto.rl.spot_rollout.subscriber import SubscriberClient

        self.subscriber = SubscriberClient(self.addr, self.id, layout(), num_streams=1, connect_timeout=20,
                                           incarnation=self.incarnation, heartbeat_s=0.5)
        self.subscriber.start()
        return self.subscriber

    def open_producer(self):
        from yeto.rl.spot_rollout.relay import RelayProducer

        # A fenced producer's connection is closed by the syncer: a send may then
        # fail before the ERROR is read, and the redial is refused (fenced).
        self.producer = RelayProducer(self.addr, 0, self.id, incarnation=self.incarnation, max_attempts=3,
                                      retry_s=0.05)
        return self.producer

    def close(self) -> None:
        for channel in (self.task, self.subscriber, self.producer):
            if channel is not None:
                channel.close()


def _dispense_records(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reclaims(path, sampler: Sampler) -> list[dict]:
    return [r for r in _dispense_records(path) if r.get("kind") == "dispense_reclaim"
            and r.get("island_id") == sampler.id and r.get("incarnation") == f"{sampler.incarnation:016x}"]


@needs_cargo
@pytest.mark.parametrize("first_channel", ["subscribe", "relay", "task"])
def test_a_superseded_sampler_is_isolated_on_all_three_channels_at_once(syncer_binary, tmp_path, first_channel):
    from yeto.rl.spot_rollout.relay import PUT_STORED, RelayConsumer, RelayFenced

    log_path = tmp_path / "task-dispense.jsonl"
    process, port, log = start_syncer(
        syncer_binary, tmp_path, total_steps=P,
        extra=("--dispense", "--dispense-lease-s", str(LEASE_S), "--dispense-log", str(log_path),
               "--max-subscribers", "1", "--sample-relay"))
    gate = threading.Event()
    learner = Learner(port, hold=gate).start()  # the session the subscribers need
    old = Sampler(port, "spot-0", 0x1111_0000_0000_0001)
    new = Sampler(port, "spot-0", 0x2222_0000_0000_0002)
    try:
        # The old process: rows held, weights subscribed, one group stored.
        held = old.open_task().claim(1, 8)
        assert sum(end - start for start, end in held.ranges) == 8
        old.open_subscriber()
        wait_log(log, "subscriber connected")
        assert old.open_producer().put(b"old-group").status == PUT_STORED

        # A newer process of the same sampler opens one channel.
        started = time.monotonic()
        if first_channel == "subscribe":
            new.open_subscriber()
            wait_log(log, "subscriber connected", count=2)
        elif first_channel == "relay":
            assert new.open_producer().put(b"new-group").status == PUT_STORED
        else:
            assert new.open_task().welcome.superseded == old.incarnation

        # All three channels of the old process are isolated within seconds
        # (the member lease is 600 s).
        wait_for(lambda: _reclaims(log_path, old), "the old rows were not reclaimed", timeout=ISOLATED_WITHIN_S)
        reclaim = _reclaims(log_path, old)[0]
        assert reclaim["reason"] == "superseded" and reclaim["positions"] == 8
        wait_for(lambda: old.subscriber.fenced(), "the old subscriber was not fenced", timeout=ISOLATED_WITHIN_S)
        with pytest.raises(RelayFenced, match=r"\(superseded\)$"):
            old.producer.put(b"old-late")
        with pytest.raises(td.DispenseFenced, match=r"\(superseded\)$"):
            old.task.claim(2, 1)
        isolated_s = time.monotonic() - started
        assert isolated_s < ISOLATED_WITHIN_S + 1.0
        # It cannot come back under its incarnation on any channel.
        with pytest.raises(td.DispenseFenced):
            old.open_task()
        with pytest.raises(RelayFenced):
            Sampler(port, "spot-0", old.incarnation).open_producer().put(b"old-again")

        # The new process works on all three; the old rows come first, the
        # group the old process stored stays in the relay.
        task = new.task or new.open_task()
        grant = task.claim(1, 8)
        assert sorted(grant.ranges) == sorted(held.ranges)
        if new.subscriber is None:
            new.open_subscriber()
            wait_log(log, "subscriber connected", count=2)
        if new.producer is None:
            assert new.open_producer().put(b"new-group").status == PUT_STORED
        assert new.subscriber._err is None and not new.subscriber.fenced()
        items = RelayConsumer(("127.0.0.1", port), 0).take(max_items=8, wait_s=0)
        assert sorted(item.data for item in items) == [b"new-group", b"old-group"]
        assert "fenced: subscriber" in log.read_text()
    finally:
        old.close()
        new.close()
        gate.set()
    finish(process, learner)
    # The training island is unaffected: it pushed every step.
    assert learner.pushed == list(range(1, P + 1))


@needs_cargo
def test_a_sampler_process_is_one_member_on_three_channels(syncer_binary, tmp_path):
    """The same incarnation on a second channel is the same member (nothing is
    superseded), and its heartbeats keep the member's rows past the lease."""
    log_path = tmp_path / "task-dispense.jsonl"
    process, port, log = start_syncer(
        syncer_binary, tmp_path, total_steps=P,
        extra=("--dispense", "--dispense-lease-s", "2", "--dispense-log", str(log_path),
               "--max-subscribers", "1", "--sample-relay"))
    gate = threading.Event()
    learner = Learner(port, hold=gate).start()
    sampler = Sampler(port, "spot-1", 0x3333_0000_0000_0003)
    try:
        held = sampler.open_task().claim(1, 4)
        sampler.open_subscriber()  # heartbeats every 0.5 s renew the shared lease
        wait_log(log, "subscriber connected")
        sampler.open_producer().put(b"g")
        time.sleep(4.0)  # twice the lease: the task connection itself sent nothing
        assert not _reclaims(log_path, sampler), "a member that heartbeats on one channel kept its rows"
        report = sampler.task.report(2, [(held.block, 1, held.ranges)])
        assert report.settled == 4
        assert not [r for r in _dispense_records(log_path) if r.get("kind") == "dispense_open"
                    and r.get("superseded")], "one process never supersedes itself"
        # Silent on every channel for the lease: retired (lease expired), not superseded.
        sampler.subscriber.close()
        sampler.producer.close()
        time.sleep(3.0)
        with pytest.raises(td.DispenseFenced, match=r"\(lease expired\)$"):
            sampler.task.heartbeat()
    finally:
        sampler.close()
        gate.set()
    finish(process, learner)


@needs_cargo
def test_a_superseded_task_source_stops_instead_of_superseding_back(syncer_binary, tmp_path):
    """The old process's task source, fenced as superseded, does not reopen under
    a new incarnation (that would supersede the newer process on every channel);
    a lease expiry still reopens (P2 behavior, test_s27_dispense)."""
    from test_s27_dispense import FP as DISPENSE_FP, KEY as DISPENSE_KEY, Syncer

    syncer = Syncer(syncer_binary, tmp_path, lease_s=600.0)
    island = td.sampler_holder_id("spot-2")
    try:
        events: list = []
        connect = td.tcp_connector(("127.0.0.1", syncer.port), learner_id=island, rows=100,
                                   fingerprint=DISPENSE_FP, key=DISPENSE_KEY, timeout_s=5.0)
        old = td.TaskSource(connect=connect, block=4, prefetch_blocks=1, retry_s=0.05,
                            incarnation=lambda: 0x4444, emit=lambda kind, **f: events.append(kind)).start()
        assert len(old.take(4, timeout_s=5)) == 4
        new = td.DispenseConnection(("127.0.0.1", syncer.port), learner_id=island, incarnation=0x5555,
                                    rows=100, fingerprint=DISPENSE_FP, key=DISPENSE_KEY)
        assert new.welcome.superseded == 0x4444
        with pytest.raises(td.DispenseFenced, match=r"\(superseded\)$"):
            while True:
                old.take(4, timeout_s=5)
        assert "rl_dispense_superseded" in events and old.stats["fenced"] == 0
        time.sleep(0.5)
        opens = [r for r in syncer.records() if r.get("kind") == "dispense_open"]
        assert [r["incarnation"] for r in opens] == [f"{0x4444:016x}", f"{0x5555:016x}"]
        new.heartbeat()  # still the island's live incarnation
        old.close(release=False)
        new.close()
    finally:
        syncer.stop()
