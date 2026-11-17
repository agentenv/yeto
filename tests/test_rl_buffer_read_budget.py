"""The trainer's payload read budget (CPU; no cloud resource): ``BatchReader``
holds at most ``YETO_SAMPLE_BUFFER_MAX_INFLIGHT_BYTES`` of raw payload bytes
(reads in flight, read and not decoded, read ahead), drops a payload's bytes once
it is decoded, and selects exactly what a reader without the budget selects. The
limits are scaled down: payloads of a few KB, budgets of a few payloads."""

from __future__ import annotations

import threading
import time

import pytest

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

PAD = "p" * 4000  # a prompt that makes each payload ~8 KB


def _group(g: int, version: int) -> list[dict]:
    return [{**sample(2 * g + i, g, version), "prompt": PAD} for i in range(2)]


class Payload(bytes):
    """A payload's bytes that report when they are freed."""

    live: set[str] = set()

    def __del__(self) -> None:
        Payload.live.discard(self.ref)


class Counting(FileSampleBuffer):
    """Counts the reads open at once and the payload bytes alive (not freed yet)."""

    def __init__(self, *args, read_delay_s: float = 0.0, **kw) -> None:
        super().__init__(*args, **kw)
        self.read_delay_s = read_delay_s
        self.lock = threading.Lock()
        self.open = self.peak_open = 0
        self.size_calls = 0
        self.fail: set[str] = set()

    def payload_size(self, payload_ref):
        self.size_calls += 1
        return super().payload_size(payload_ref)

    def read_payload(self, payload_ref):
        with self.lock:
            self.open += 1
            self.peak_open = max(self.peak_open, self.open)
        try:
            time.sleep(self.read_delay_s)
            if payload_ref in self.fail:
                raise ConnectionError("the store did not answer")
            data = Payload(super().read_payload(payload_ref))
            data.ref = payload_ref
            Payload.live.add(payload_ref)
            return data
        finally:
            with self.lock:
                self.open -= 1


def _filled(root, groups: int, version: int = 1, start: int = 0) -> int:
    buffer = FileSampleBuffer(CheckpointStore(root), batch_size=2)
    writer = br.GroupWriter(buffer, "s0")
    for g in range(start, start + groups):
        writer.write(_group(g, version), group_key=f"g{g}")
    return max(buffer.payload_size(e.payload_ref) for e in buffer.entries()[start:])  # the largest


def _reader(root, *, budget, workers: int = 8, prefetch: bool = False, delay: float = 0.0, **kw):
    buffer = Counting(CheckpointStore(root), batch_size=2, read_delay_s=delay)
    return br.BatchReader(buffer, max_policy_age=4, wait_s=0.0, poll_s=0.01, read_workers=workers,
                          prefetch=prefetch, max_inflight_bytes=budget, **kw), buffer


def _take_in_time(reader, *args, timeout_s: float = 30.0, **kw):
    """A take on a thread: a deadlock fails the test instead of hanging it."""
    out: list = []

    def run() -> None:
        try:
            out.append(reader.take(*args, **kw))
        except BaseException as error:  # noqa: BLE001 - raised on the test's thread
            out.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_s)
    assert not thread.is_alive(), "the take did not end (deadlock)"
    if isinstance(out[0], BaseException):
        raise out[0]
    return out[0]


# ---------------------------------------------------------------- the setting
def test_the_budget_setting():
    env = br.MAX_INFLIGHT_BYTES_ENV
    assert br.inflight_bytes_limit({}) is None and br.DEFAULT_MAX_INFLIGHT_BYTES == 0  # off: the reader of main
    assert br.inflight_bytes_limit({env: "0"}) is None  # no limit: as before the budget
    assert br.RECOMMENDED_262K_INFLIGHT_BYTES == 64 << 30
    assert br.inflight_bytes_limit({env: "1000"}) == 1000
    assert br.inflight_bytes_limit({env: "64GiB"}) == 64 << 30
    assert br.inflight_bytes_limit({env: "512 MiB"}) == 512 << 20
    assert br.inflight_bytes_limit({env: "1.5T"}) == 3 << 39
    for bad in ("lots", "-1", "5PB"):
        with pytest.raises(ValueError, match=env):
            br.inflight_bytes_limit({env: bad})


def test_the_reader_takes_the_budget_from_the_environment(tmp_path, monkeypatch):
    _filled(tmp_path, 1)
    monkeypatch.setenv(br.MAX_INFLIGHT_BYTES_ENV, "3MiB")
    assert br.BatchReader(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), max_policy_age=1).budget.limit == 3 << 20
    monkeypatch.setenv(br.MAX_INFLIGHT_BYTES_ENV, "0")
    assert br.BatchReader(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), max_policy_age=1).budget.limit is None


# ---------------------------------------------------------------- the bound
@pytest.mark.parametrize("payloads_in_budget", [1, 2.5, 4])
def test_a_take_holds_at_most_the_budget_and_frees_each_payload_at_its_decode(tmp_path, monkeypatch,
                                                                             payloads_in_budget):
    size = _filled(tmp_path, 8)
    limit = int(size * payloads_in_budget)
    reader, buffer = _reader(tmp_path, budget=limit, delay=0.01)
    alive_at_decode = []
    real = br.decode_group

    def decode(data):
        alive_at_decode.append(len(Payload.live))
        return real(data)

    monkeypatch.setattr(br, "decode_group", decode)
    Payload.live.clear()
    groups, report = _take_in_time(reader, 0, 8, current_version=1)
    assert [pos for pos, _ in groups] == list(range(8))
    assert reader.budget.peak <= max(limit, size) and reader.budget.used == 0
    most = max(1, int(payloads_in_budget))
    assert max(alive_at_decode) <= most  # never more than the budget's payloads alive at once
    assert buffer.peak_open <= most
    assert Payload.live == set()  # every payload's bytes are gone once the round is decoded
    assert report.read_s >= 0 and report.decode_s >= 0


def test_without_the_budget_a_take_reads_the_round_at_once_as_before(tmp_path):
    _filled(tmp_path, 8)
    reader, buffer = _reader(tmp_path, budget=None, delay=0.02)
    Payload.live.clear()
    groups, _ = _take_in_time(reader, 0, 8, current_version=1)
    assert [pos for pos, _ in groups] == list(range(8))
    assert reader.budget.limit is None and buffer.size_calls == 0  # no HEAD per payload
    assert buffer.peak_open == 8


def test_the_budget_changes_nothing_the_take_selects(tmp_path):
    """Same buffer, same rounds: stale groups, a damaged payload and the carried
    group come out the same with a budget of one payload as without one."""
    root = tmp_path / "buf"
    buffer = FileSampleBuffer(CheckpointStore(root), batch_size=2)
    writer = br.GroupWriter(buffer, "s0")
    for g, version in enumerate((1, 3, 3, 2, 4, 4, 4, 4, 4, 4)):
        writer.write(_group(g, version), group_key=f"g{g}")
    damaged = buffer.entries()[5].payload_ref
    (root / "sample-buffer" / damaged).write_bytes(b"not a payload")
    size = max(buffer.payload_size(e.payload_ref) for e in buffer.entries())

    def rounds(budget):
        reader, _ = _reader(root, budget=budget, group_size=2)
        out = []
        for rid in range(3):
            groups, report = _take_in_time(reader, rid, 2, current_version=5)
            out.append(([(p, [s["index"] for s in g]) for p, g in groups],
                        report.stale_groups, report.damaged_payload_groups, report.start, report.end))
        return out

    assert rounds(size) == rounds(None)


def test_a_payload_larger_than_the_budget_is_read_alone(tmp_path, capsys):
    size = _filled(tmp_path, 3)
    reader, buffer = _reader(tmp_path, budget=size // 2)
    groups, _ = _take_in_time(reader, 0, 3, current_version=1)
    assert len(groups) == 3 and buffer.peak_open == 1 and reader.budget.used == 0
    assert "above the read budget" in capsys.readouterr().out


def test_a_take_without_read_workers_counts_its_reads_too(tmp_path):
    _filled(tmp_path, 4)
    reader, _ = _reader(tmp_path, budget=1, workers=1)
    groups, _ = _take_in_time(reader, 0, 4, current_version=1)
    assert len(groups) == 4 and reader.budget.used == 0


def test_a_buffer_that_cannot_size_a_payload_is_counted_when_read(tmp_path):
    size = _filled(tmp_path, 4)

    class NoSize(Counting):
        payload_size = None  # not callable: the reader cannot ask

    reader = br.BatchReader(NoSize(CheckpointStore(tmp_path), batch_size=2), max_policy_age=1, wait_s=0.0,
                            poll_s=0.0, prefetch=False, max_inflight_bytes=size)
    groups, _ = _take_in_time(reader, 0, 4, current_version=1)
    assert len(groups) == 4 and reader.budget.used == 0 and reader.budget.peak >= size


def test_a_take_that_raises_gives_its_budget_back(tmp_path):
    size = _filled(tmp_path, 4)
    reader, buffer = _reader(tmp_path, budget=2 * size)
    buffer.fail = {buffer.entries()[1].payload_ref}
    with pytest.raises(ConnectionError):
        _take_in_time(reader, 0, 4, current_version=1)
    deadline = time.monotonic() + 10
    while reader.budget.used and time.monotonic() < deadline:  # abandoned reads release when they end
        time.sleep(0.01)
    assert reader.budget.used == 0
    buffer.fail = set()
    groups, _ = _take_in_time(reader, 0, 4, current_version=1)  # the groups are selected again
    assert [pos for pos, _ in groups] == [0, 1, 2, 3] and reader.budget.used == 0


# ---------------------------------------------------------------- prefetch under the same budget
def test_the_read_ahead_stops_at_the_budget_and_the_take_reads_the_rest(tmp_path):
    size = _filled(tmp_path, 4)
    reader, buffer = _reader(tmp_path, budget=int(2.5 * size), prefetch=True)
    reader.wait_s = 30.0
    first, _ = _take_in_time(reader, 0, 4, current_version=1)
    assert [pos for pos, _ in first] == [0, 1, 2, 3]
    _filled(tmp_path, 4, version=2, start=4)  # the next round arrives while the trainer trains
    deadline = time.monotonic() + 10
    while (reader._prefetcher is not None and reader._prefetcher.is_alive()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(reader._reads) == 2  # two payloads fit; the third would pass the budget
    assert 0 < reader.budget.used <= 2.5 * size
    second, report = _take_in_time(reader, 1, 4, current_version=2)
    assert [pos for pos, _ in second] == [4, 5, 6, 7]
    assert report.prefetched == 2 and reader.budget.used == 0
    reader._stop_prefetching()


def test_a_read_ahead_the_take_does_not_choose_gives_its_bytes_back(tmp_path):
    size = _filled(tmp_path, 2)
    reader, buffer = _reader(tmp_path, budget=4 * size, prefetch=True)
    reader.wait_s = 30.0
    _take_in_time(reader, 0, 2, current_version=1)
    _filled(tmp_path, 2, version=2, start=2)
    deadline = time.monotonic() + 10
    while len(reader._reads) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert 0 < reader.budget.used <= 2 * size
    # a version jump ages the read-ahead groups out: never chosen, their bytes released
    _filled(tmp_path, 2, version=9, start=4)
    reader.max_policy_age = 1
    groups, report = _take_in_time(reader, 1, 2, current_version=9)
    assert [pos for pos, _ in groups] == [4, 5] and report.stale_groups == 2
    assert reader.budget.used == 0


def test_a_reset_gives_the_read_ahead_budget_back(tmp_path):
    size = _filled(tmp_path, 2)
    reader, _ = _reader(tmp_path, budget=4 * size, prefetch=True)
    reader.wait_s = 30.0
    _take_in_time(reader, 0, 2, current_version=1)
    _filled(tmp_path, 2, version=2, start=2)
    deadline = time.monotonic() + 10
    while len(reader._reads) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    reader.reset(0)
    assert reader.budget.used == 0 and reader._reads == {}


# ---------------------------------------------------------------- the budget itself
def test_byte_budget_turns_and_room():
    budget = br.ByteBudget(10)
    first, second = budget.ticket(), budget.ticket()
    budget.acquire(first, 8)
    assert not budget.try_acquire(1)  # a take's read is waiting its turn: no read ahead jumps it
    done = threading.Event()
    threading.Thread(target=lambda: (budget.acquire(second, 5), done.set()), daemon=True).start()
    assert not done.wait(0.1)  # 8 + 5 > 10: it waits
    budget.release(8)
    assert done.wait(5) and budget.used == 5 and budget.waits == 1
    assert budget.try_acquire(5) and not budget.try_acquire(1)
    budget.release(10)
    assert budget.try_acquire(50) and budget.used == 50  # alone: a payload above the limit is never refused
    assert br.ByteBudget(None).try_acquire(1 << 50)


# ---------------------------------------------------------------- v2 payloads decode in the read threads
@pytest.mark.parametrize("codec", ["raw", "zstd"])
def test_v2_payloads_decode_in_the_read_threads_and_v1_on_the_callers(tmp_path, monkeypatch, codec):
    import numpy

    from yeto.rl.engine.buffer_payload import PAYLOAD_CODEC_ENV

    threads: list[str] = []
    real = br.decode_group
    monkeypatch.setattr(br, "decode_group", lambda data: (threads.append(threading.current_thread().name),
                                                          real(data))[1])

    def write(root, codec_now):
        monkeypatch.setenv(PAYLOAD_CODEC_ENV, codec_now)
        buffer = FileSampleBuffer(CheckpointStore(root), batch_size=2)
        writer = br.GroupWriter(buffer, "s0")
        for g in range(4):
            group = _group(g, 1)
            group = [{**s, "arr": numpy.arange(1000 * (g + 1), dtype=numpy.int32)} for s in group]
            writer.write(group, group_key=f"g{g}")

    write(tmp_path / "v2", codec)
    write(tmp_path / "v1", "json")
    out = {}
    for name in ("v2", "v1"):
        threads.clear()
        reader, _ = _reader(tmp_path / name, budget=1 << 20)
        groups, _ = _take_in_time(reader, 0, 4, current_version=1)
        out[name] = groups
        assert reader.budget.used == 0
        on_workers = all(t.startswith("buffer-read") for t in threads)
        assert len(threads) == 4 and on_workers == (name == "v2")
    for (p2, g2), (p1, g1) in zip(out["v2"], out["v1"]):
        assert p2 == p1 and [numpy.array_equal(a["arr"], b["arr"]) for a, b in zip(g2, g1)] == [True, True]


def test_a_damaged_v2_payload_decoded_in_a_read_thread_is_passed_over(tmp_path, monkeypatch):
    from yeto.rl.engine.buffer_payload import PAYLOAD_CODEC_ENV

    monkeypatch.setenv(PAYLOAD_CODEC_ENV, "zstd")
    root = tmp_path / "buf"
    _filled(root, 3)
    buffer = FileSampleBuffer(CheckpointStore(root), batch_size=2)
    path = root / "sample-buffer" / buffer.entries()[1].payload_ref
    path.write_bytes(path.read_bytes()[:40])  # v2 magic, cut short
    reader, _ = _reader(root, budget=1 << 20)
    groups, report = _take_in_time(reader, 0, 2, current_version=1)
    assert [pos for pos, _ in groups] == [0, 2] and report.damaged_payload_groups == 1
    assert reader.budget.used == 0



# ---------------------------------------------------------------- no deadlock behind the read-ahead (R2 F2)
def _settled_prefetch(reader) -> None:
    deadline = time.monotonic() + 10
    while reader._prefetcher is not None and reader._prefetcher.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)


def _unblock(reader) -> None:
    """Let a hung take end so the interpreter can exit (pool workers are joined at exit)."""
    with reader.budget._cond:
        reader.budget.limit = None
        reader.budget._cond.notify_all()


def _take_or_unblock(reader, *args, **kw):
    try:
        return _take_in_time(reader, *args, timeout_s=15.0, **kw)
    except AssertionError:
        _unblock(reader)
        raise


def test_a_take_read_after_a_failed_read_ahead_does_not_wait_behind_the_full_budget(tmp_path):
    """A read ahead fails once (a store hiccup), the read-ahead then fills the budget with
    the groups after it: the take reads the failed group itself, before the groups whose
    read-ahead holds the room."""
    size = _filled(tmp_path, 4)
    reader, buffer = _reader(tmp_path, budget=3 * size, prefetch=True)
    reader.wait_s, reader.poll_s = 30.0, 0.05
    first, _ = _take_in_time(reader, 0, 4, current_version=1)
    assert [p for p, _ in first] == [0, 1, 2, 3]
    ref = "payloads/s0-g4.bin"
    buffer.fail = {ref}
    _filled(tmp_path, 1, version=1, start=4)
    deadline = time.monotonic() + 10
    while not (ref in reader._reads and reader._reads[ref].done()) and time.monotonic() < deadline:
        time.sleep(0.01)
    buffer.fail = set()
    _filled(tmp_path, 3, version=1, start=5)  # the read-ahead fills the budget
    _settled_prefetch(reader)
    assert reader.budget.used + size > reader.budget.limit  # the failed group's read finds no room
    groups, report = _take_or_unblock(reader, 1, 4, current_version=1)
    assert [p for p, _ in groups] == [4, 5, 6, 7] and report.prefetched >= 2
    assert reader.budget.used == 0 and reader.budget.urged >= 1


def test_a_take_read_of_a_group_the_read_ahead_passed_over_does_not_wait_behind_it(tmp_path):
    """The read-ahead's floor is this take's + 1. A basis version that did not move
    between two takes (no new publication) leaves a group at the old floor to the
    take, ahead of the read-ahead groups that fill the budget."""
    size = _filled(tmp_path, 4, version=5)
    reader, _ = _reader(tmp_path, budget=3 * size, prefetch=True)  # max_policy_age=4
    reader.wait_s, reader.poll_s = 30.0, 0.05
    _take_in_time(reader, 0, 4, current_version=5)  # floor 1, read-ahead floor 2
    _filled(tmp_path, 1, version=1, start=4)  # at the floor, still in the window
    _filled(tmp_path, 3, version=5, start=5)
    _settled_prefetch(reader)
    assert "payloads/s0-g4.bin" not in reader._reads and len(reader._reads) >= 2
    assert reader.budget.used + size > reader.budget.limit  # the passed-over group's read finds no room
    groups, _ = _take_or_unblock(reader, 1, 4, current_version=5)
    assert [p for p, _ in groups] == [4, 5, 6, 7] and reader.budget.used == 0


def test_a_deferred_group_judged_again_does_not_wait_behind_the_read_ahead(tmp_path):
    """A group whose lease the last take could not read waits for the next take; that
    take reads it first, before the groups the read-ahead holds the room for."""
    from yeto.rl.engine.incarnation_lease import LeaseUnknown

    class Gate:
        def __init__(self) -> None:
            self.unreadable: set[int] = set()

        def verdict(self, entry, position):
            if position in self.unreadable:
                raise LeaseUnknown("lease unreadable")
            return None

    size = _filled(tmp_path, 7)
    gate = Gate()
    reader, _ = _reader(tmp_path, budget=int(2.5 * size), prefetch=True, group_size=4, incarnation_gate=gate,
                        one_group_per_task=False)
    reader.wait_s, reader.poll_s = 30.0, 0.05
    gate.unreadable = {3}
    first, report = _take_in_time(reader, 0, 3, current_version=1)  # 3 groups of 2: the fill defers 3
    assert [p for p, _ in first] == [0, 1, 2] and report.end == 3 and [p for p, _ in reader._deferred] == [3]
    gate.unreadable = set()
    _settled_prefetch(reader)
    assert sorted(reader._reads) == ["payloads/s0-g4.bin", "payloads/s0-g5.bin"]
    assert reader.budget.used + size > reader.budget.limit
    groups, _ = _take_or_unblock(reader, 1, 3, current_version=1)
    assert [p for p, _ in groups][:3] == [3, 4, 5] and reader.budget.used == 0


def test_byte_budget_urge_and_abandon():
    budget = br.ByteBudget(10)
    held = budget.ticket()
    assert budget.acquire(held, 10)
    late, early = budget.ticket(), budget.ticket()
    out = {}
    threads = [threading.Thread(target=lambda t=t, k=k: out.__setitem__(k, budget.acquire(t, 5)), daemon=True)
               for k, t in (("late", late), ("early", early))]
    for t in threads:
        t.start()
    time.sleep(0.05)
    assert out == {}  # no room
    budget.urge(early)  # the take waits for it: past its turn and the room
    threads[1].join(5)
    assert out == {"early": True} and budget.used == 15 and budget.urged == 1
    budget.abandon(late)
    threads[0].join(5)
    assert out["late"] is False and budget.used == 15  # abandoned: no room taken
    budget.release(15)
    assert budget.try_acquire(1)  # every ticket took its turn
