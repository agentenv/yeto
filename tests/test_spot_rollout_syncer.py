"""S27 G-EA: receive-only subscribers and the sample relay on the real syncer.

Each test starts the Rust syncer binary (debug build) with one or two raw
f32 learners. Outer lr 1 and momentum 0 with one learner make every merged
fragment equal the learner's target, so the subscriber's bf16 bytes can be
checked exactly (targets are multiples of 0.25 below 64: exact in bf16).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
KEY = "spot-rollout-test-key"
P = 3
NUMEL = 5


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    env = {**os.environ, "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS", "8")}
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True, env=env)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", KEY)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start_syncer(binary: Path, tmp_path: Path, *, total_steps: int, learners: int = 1,
                 extra: tuple[str, ...] = ()) -> tuple[subprocess.Popen, int, Path]:
    port = free_port()
    log = tmp_path / f"syncer-{port}.log"
    command = [str(binary), "--port", str(port), "--learners", str(learners), "--quorum", "1",
               "--grace-ms", "0", "--quorum-timeout-s", "30", "--total-steps", str(total_steps),
               "--pipeline", "1", "--sync-interval-steps", "0", "--delta-correction", "none",
               "--outer-lr", "1", "--outer-momentum", "0",
               "--event-tape", str(tmp_path / f"tape-{port}.jsonl"),
               "--final-state", str(tmp_path / f"final-{port}.bin"), *extra]
    env = {**os.environ, "TOKIO_WORKER_THREADS": "4", "YETO_ISLAND_HMAC_KEY": KEY}
    with log.open("w") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env)
    wait_for(lambda: "syncer listening" in log.read_text(), "syncer did not start", timeout=30)
    return process, port, log


def wait_log(log: Path, text: str, count: int = 1, timeout: float = 30.0) -> None:
    wait_for(lambda: log.read_text().count(text) >= count, f"syncer log never showed {text!r}", timeout)


def layout(numel: int = NUMEL, fragments: int = P):
    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout

    shard = "tp0-of-1.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1"
    out = []
    for fid in range(fragments):
        name = f"actor::{shard}::decoder.layers.{fid}.weight"
        out.append(Fragment(MERGE_AVG, [(name, numel)], identity_shapes={name: (numel,)}))
    return FragmentLayout(out)


def target(fid: int, step: int, numel: int = NUMEL) -> np.ndarray:
    return np.asarray([fid + 0.25 * step + 0.5 * i for i in range(numel)], dtype=np.float32)


def initial(fid: int, numel: int = NUMEL) -> np.ndarray:
    return np.asarray([fid + 0.5 * i for i in range(numel)], dtype=np.float32)


def bf16(values: np.ndarray) -> bytes:
    import torch

    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy().tobytes()


def wait_for(predicate, what: str, timeout: float = 30.0, poll: float = 0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(poll)
    raise TimeoutError(what)


class Learner:
    """One f32 learner that makes fragment ``p`` equal ``target(p, t)`` at step t."""

    def __init__(self, port: int, *, numel: int = NUMEL, fragments: int = P,
                 hold: threading.Event | None = None, on_merge=None) -> None:
        from yeto.protocol import DTYPE_F32, SyncerClient

        self.numel, self.fragments = numel, fragments
        self.client = SyncerClient(("127.0.0.1", port), 0, layout(numel, fragments), dtype=DTYPE_F32,
                                   num_streams=1, connect_timeout=20, max_reconnects=0,
                                   finalization_timeout=60)
        self.held = {fid: (0, initial(fid, numel)) for fid in range(fragments)}
        self.hold = hold
        self.on_merge = on_merge
        self.pushed: list[int] = []
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            client = self.client
            client.start()
            for fid in range(self.fragments):
                client.send_init(fid, initial(fid, self.numel).tobytes())
            pending = []
            while not client.finalizing.is_set():
                client.check_health()
                for update in client.drain_updates():
                    if update.version > self.held[update.fragment_id][0]:
                        values = np.frombuffer(bytes(update.data), dtype=np.float32).copy()
                        self.held[update.fragment_id] = (update.version, values)
                        if self.on_merge is not None:
                            self.on_merge(update.fragment_id, update.version)
                pending.extend(client.drain_pulls())
                still = []
                for permit in pending:
                    step, fid = permit.global_step, permit.fragment_id
                    base = step - self.fragments if step > self.fragments else 0
                    if self.held[fid][0] != base or (self.hold is not None and not self.hold.is_set()):
                        still.append(permit)
                        continue
                    delta = target(fid, step, self.numel) - self.held[fid][1]
                    client.push_fragment(fid, step, permit.round_attempt, base, step, 1, 8, delta.tobytes())
                    self.pushed.append(step)
                pending = still
                time.sleep(0.005)
            manifest, _fragments = client.wait_for_final_fragments(timeout=60)
            client.acknowledge_finalization(manifest)
        except BaseException as error:  # noqa: BLE001 - reported by the test
            self.error = error
        finally:
            self.client.close()

    def start(self) -> "Learner":
        self.thread.start()
        return self

    def join(self, timeout: float = 60.0) -> None:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "learner did not finish"
        if self.error is not None:
            raise self.error


def subscriber(port: int, sub_id: int = 0, *, numel: int = NUMEL, fragments: int = P, lay=None):
    from yeto.protocol import SUBSCRIBER_ID_BIT
    from yeto.rl.spot_rollout.subscriber import SubscriberClient

    return SubscriberClient(("127.0.0.1", port), SUBSCRIBER_ID_BIT | sub_id, lay or layout(numel, fragments),
                            num_streams=2, connect_timeout=20)


def finish(process: subprocess.Popen, learner: Learner) -> None:
    learner.join()
    assert process.wait(timeout=30) == 0


def tape_records(tmp_path: Path, port: int) -> list[dict]:
    path = tmp_path / f"tape-{port}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------- subscriber

@needs_cargo
def test_subscriber_follows_every_merge_and_stays_out_of_the_roster(syncer_binary, tmp_path):
    from yeto.rl.spot_rollout.subscriber import WeightFollower

    total = 4 * P
    process, port, log = start_syncer(syncer_binary, tmp_path, total_steps=total,
                                      extra=("--max-subscribers", "2"))
    seen: list[tuple[int, int, bytes]] = []
    sub = subscriber(port)
    follower = WeightFollower(sub, P, on_update=lambda u: seen.append((u.fragment_id, u.version, bytes(u.data))))
    sub.start()  # before any learner: the syncer holds it until the session exists
    gate = threading.Event()
    learner = Learner(port, hold=gate).start()
    wait_log(log, "subscriber connected")
    gate.set()
    try:
        finish(process, learner)
        wait_for(lambda: sub.run_finished(), "subscriber never heard the run end")
        follower.drain()
    finally:
        sub.close()
    # Every received version carries exactly bf16(global) of that version.
    for fid, version, data in seen:
        expected = initial(fid) if version == 0 else target(fid, version)
        assert data == bf16(expected), (fid, version)
    # Every fragment is followed up to its newest broadcast version (the
    # terminal step is never broadcast; it reaches islands as the final cut).
    newest = {fid: max(t for t in range(1, total) if (t - 1) % P == fid) for fid in range(P)}
    assert {fid: follower.versions[fid] for fid in range(P)} == newest
    assert follower.merged_version() == 3
    text = log.read_text()
    assert "subscriber connected (receive-only; not in the roster)" in text
    # Nothing on the tape names the subscriber id (high bit set).
    for record in tape_records(tmp_path, port):
        dumped = json.dumps(record)
        assert str(0x8000_0000) not in dumped


@needs_cargo
def test_reclaimed_subscriber_does_not_stop_merges_and_a_new_one_catches_up(syncer_binary, tmp_path):
    from yeto.rl.spot_rollout.subscriber import WeightFollower

    total = 6 * P
    process, port, log = start_syncer(syncer_binary, tmp_path, total_steps=total,
                                      extra=("--max-subscribers", "1"))
    gate = threading.Event()
    merged: list[int] = []
    learner = Learner(port, hold=gate, on_merge=lambda fid, v: merged.append(v)).start()
    first = subscriber(port)
    first_follow = WeightFollower(first, P)
    first.start()
    wait_log(log, "subscriber connected")
    gate.set()
    try:
        wait_for(lambda: (first_follow.drain(), first_follow.complete)[1], "first subscriber got no cut")
        # The spot island disappears without a LEAVE (the cloud kills it).
        for sock in list(first._socks):
            sock.close()
        first.close()
        reclaimed_at = len(merged)
        # Merges go on without it.
        wait_for(lambda: len(merged) >= reclaimed_at + P, "merges stopped after the subscriber left")
        second = subscriber(port)  # a replacement incarnation: new generation, same id
        follow = WeightFollower(second, P)
        second.start()
        cut = wait_for(lambda: (follow.drain(), follow.cut())[1], "replacement got no full cut")
        assert min(cut.versions) >= 1  # every fragment at a version merged before it joined
        finish(process, learner)
        wait_for(lambda: second.run_finished(), "replacement never heard the run end")
        follow.drain()
        second.close()
    finally:
        first.close()
        if process.poll() is None:
            process.kill()
    assert learner.pushed == list(range(1, total + 1))  # every round merged


@needs_cargo
def test_a_subscriber_that_never_reads_does_not_slow_the_merges(syncer_binary, tmp_path):
    from yeto.protocol import (
        MSG_DATA_HELLO,
        MSG_SUBSCRIBE,
        SUBSCRIBE_MAC_DOMAIN,
        SUBSCRIBER_ID_BIT,
        DTYPE_BF16,
        encode_data_hello,
        encode_hello,
        seal_hello,
        write_frame,
    )

    numel = 1 << 20  # 2 MiB of bf16 per fragment: the socket buffers fill at once
    total = 5 * P
    process, port, log = start_syncer(syncer_binary, tmp_path, total_steps=total,
                                      extra=("--max-subscribers", "1"))
    lay = layout(numel)
    stuck = socket.create_connection(("127.0.0.1", port))
    stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    body = encode_hello(SUBSCRIBER_ID_BIT | 0, DTYPE_BF16, lay, 1, 77)
    write_frame(stuck, MSG_SUBSCRIBE, seal_hello(body, KEY.encode(), SUBSCRIBE_MAC_DOMAIN))
    data = socket.create_connection(("127.0.0.1", port))
    data.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    write_frame(data, MSG_DATA_HELLO, encode_data_hello(SUBSCRIBER_ID_BIT | 0, 77, 0))
    started = time.monotonic()
    gate = threading.Event()
    threading.Timer(0.0, lambda: (wait_log(log, "subscriber connected"), gate.set())).start()
    learner = Learner(port, numel=numel, hold=gate).start()
    try:
        finish(process, learner)
    finally:
        stuck.close()
        data.close()
    assert learner.pushed == list(range(1, total + 1))
    assert time.monotonic() - started < 60
    assert "subscriber connected" in log.read_text()


@needs_cargo
def test_subscribe_refusals_leave_the_session_running(syncer_binary, tmp_path):
    total = 3 * P

    def refused(client) -> str:
        client.start()
        try:
            wait_for(lambda: client._err is not None, "subscriber was not refused")
            return str(client._err)
        finally:
            client.close()

    # Off by default: SUBSCRIBE is refused.
    process, port, _log = start_syncer(syncer_binary, tmp_path, total_steps=total)
    learner = Learner(port).start()
    assert "without --max-subscribers" in refused(subscriber(port))
    finish(process, learner)

    process, port, log = start_syncer(syncer_binary, tmp_path, total_steps=total,
                                      extra=("--max-subscribers", "1"))
    gate = threading.Event()
    learner = Learner(port, hold=gate).start()
    assert "layout mismatch" in refused(subscriber(port, numel=NUMEL + 1))
    first = subscriber(port, sub_id=7)
    first.start()
    try:
        wait_log(log, "subscriber connected")
        # One slot: another sampler is refused; the same id reconnecting replaces itself.
        assert "subscribers are connected" in refused(subscriber(port, sub_id=8))
        again = subscriber(port, sub_id=7)
        again.start()
        wait_log(log, "subscriber connected", count=2)
        assert again._err is None
        again.close()
    finally:
        first.close()
    gate.set()
    finish(process, learner)


# --------------------------------------------------------------------- relay

@needs_cargo
def test_relay_delivers_in_order_dedupes_resends_and_redelivers_unacked(syncer_binary, tmp_path):
    from yeto.rl.spot_rollout.relay import (
        MSG_RELAY_PUT,
        MSG_RELAY_PUT_ACK,
        PUT_DUPLICATE,
        PUT_FULL,
        PUT_STORED,
        RelayBoard,
        RelayConsumer,
        RelayProducer,
    )

    process, port, log = start_syncer(syncer_binary, tmp_path, total_steps=2 * P,
                                      extra=("--sample-relay", "--sample-relay-max-items", "3"))
    gate = threading.Event()
    learner = Learner(port, hold=gate).start()  # the session runs while the relay is used
    addr = ("127.0.0.1", port)
    try:
        producer = RelayProducer(addr, channel=0, producer=5)
        acks = [producer.put(f"group-{i}".encode()) for i in range(3)]
        assert [a.status for a in acks] == [PUT_STORED] * 3
        assert [a.item_id for a in acks] == [1, 2, 3]
        assert producer.put(b"group-3").status == PUT_FULL  # backpressure: nothing stored
        # A resend of the last sequence number stores nothing twice.
        body = producer.conn.request(MSG_RELAY_PUT, struct.pack("<Q", 3) + b"group-2",
                                     expect=MSG_RELAY_PUT_ACK)
        assert body[8] == PUT_DUPLICATE
        consumer = RelayConsumer(addr, channel=0)
        items = consumer.take(max_items=2, wait_s=0)
        assert [i.data for i in items] == [b"group-0", b"group-1"]
        consumer.ack(items[0].item_id)  # sent with the next take
        assert [i.data for i in consumer.take(max_items=1, wait_s=0)] == [b"group-2"]
        # The consumer process dies before acknowledging items 2 and 3: a new one gets them again.
        consumer.close()
        again = RelayConsumer(addr, channel=0)
        items = again.take(max_items=10, wait_s=0)
        assert [i.data for i in items] == [b"group-1", b"group-2"]
        for item in items:
            again.ack(item.item_id)
        assert again.take(max_items=10, wait_s=0) == []  # acks free the room
        assert producer.put(b"group-3").status == PUT_STORED
        # A long poll returns as soon as an item arrives.
        threading.Timer(0.3, lambda: RelayProducer(addr, channel=0, producer=6).put(b"late")).start()
        late = again.take(max_items=10, wait_s=10)
        assert [i.data for i in late] == [b"group-3"] or [i.data for i in late] == [b"group-3", b"late"]
        # Channels are separate.
        assert RelayConsumer(addr, channel=1).take(wait_s=0) == []
        # Boards: the newest value wins and every client reads it.
        board = RelayBoard(addr, 1, client=0)
        board.set(b"layout-a")
        board.set(b"layout-b")
        assert RelayBoard(addr, 1, client=9).get().data == b"layout-b"
        assert RelayBoard(addr, 2, client=9).get() is None
        # A newer consumer of a channel supersedes the older connection.
        from yeto.rl.spot_rollout.relay import RelaySuperseded

        older = RelayConsumer(addr, channel=3)
        older.take(wait_s=0)
        RelayConsumer(addr, channel=3).take(wait_s=0)
        with pytest.raises(RelaySuperseded):
            older.take(wait_s=0)
    finally:
        gate.set()
        finish(process, learner)
    assert "sample relay enabled" in log.read_text()


@needs_cargo
def test_relay_is_refused_when_off(syncer_binary, tmp_path):
    from yeto.rl.spot_rollout.relay import RelayProducer, RelayRefused

    process, port, _log = start_syncer(syncer_binary, tmp_path, total_steps=P)
    learner = Learner(port).start()
    try:
        with pytest.raises(RelayRefused, match="without --sample-relay"):
            RelayProducer(("127.0.0.1", port), 0, 1, max_attempts=1).put(b"x")
    finally:
        finish(process, learner)
