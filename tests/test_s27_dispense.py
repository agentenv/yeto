"""S27 P2 task dispenser: wire golden bytes, the island client, and simulated
islands of different speeds against the real Rust syncer (no GPU, no Ray).

Covers ``syncer/src/dispense.rs`` + ``dispense_service.rs`` (via the binary),
``yeto/rl/engine/task_dispense.py`` and ``tools/eval/dispense_check.py``."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tools.eval.dispense_check import check, check_tapes, read_records
from yeto.rl.engine import task_dispense as td
from yeto.rl.engine.data_split import pass_order

ROOT = Path(__file__).resolve().parent.parent
KEY = b"dispense-test-key"
FP = hashlib.sha256(b"rows").digest()


def _cargo_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.cargo/bin'}:{env.get('PATH', '')}"
    env.setdefault("TOKIO_WORKER_THREADS", "4")
    return env


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True, env=_cargo_env())
    assert binary.exists()
    return binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Syncer:
    """The real syncer with a dispenser; no learner ever connects (the
    dispense connections do not need a learner session)."""

    def __init__(self, binary: Path, tmp: Path, *extra: str, lease_s: float = 2.0, port: int | None = None,
                 key: bytes | None = KEY, learners: int = 4, name: str = "syncer") -> None:
        self.port = port or free_port()
        self.log = tmp / "task-dispense.jsonl"
        args = [str(binary), "--port", str(self.port), "--learners", str(learners), "--quorum", "1",
                "--total-steps", "1000", "--dispense", "--dispense-lease-s", str(lease_s),
                "--dispense-log", str(self.log), *extra]
        if key:
            args += ["--island-hmac-key", key.decode()]
        self.out = open(tmp / f"{name}.out", "w")
        self.proc = subprocess.Popen(args, stdout=self.out, stderr=subprocess.STDOUT, env=_cargo_env())
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"syncer exited {self.proc.returncode}: {(tmp / f'{name}.out').read_text()[-2000:]}")
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("syncer did not listen")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.out.close()

    def records(self) -> list[dict]:
        return read_records([self.log])


def source(port: int, learner_id: int, *, block: int, rows: int = 1000, key: bytes | None = KEY,
           fingerprint: bytes = FP, prefetch: int = 2, events: list | None = None,
           address: tuple[str, int] | None = None, retry_s: float = 0.1) -> td.TaskSource:
    connect = td.tcp_connector(address or ("127.0.0.1", port), learner_id=learner_id, rows=rows,
                               fingerprint=fingerprint, key=key, timeout_s=5.0)
    emit = (lambda kind, **f: events.append((kind, f))) if events is not None else None
    return td.TaskSource(connect=connect, block=block, prefetch_blocks=prefetch, emit=emit,
                         retry_s=retry_s).start()


def crash(src: td.TaskSource) -> None:
    """Kill an island process: no release, no more heartbeats."""
    with src._cv:
        src._stopped = True
        src._cv.notify_all()
    src._thread.join(timeout=5)
    if src._conn is not None:
        src._conn.close()


# ------------------------------------------------------------------ golden


def test_wire_bytes_match_the_rust_golden_vectors():
    """Same vectors as ``dispense_service::tests::wire_golden_vectors``."""
    body = td.encode_task_hello(3, 0x0102030405060708, 480, b"\xab" * 32)
    assert body.hex() == "0100030000000807060504030201e001000000000000" + "ab" * 32
    sealed = td.seal_task_hello(body, b"k")
    assert sealed[len(body):].hex() == "78a9ab168d688c2f84a9ed2ac1f6ed21a55667af74969a9266cc7944a832fac9"
    assert td.seal_task_hello(body, None) == body
    welcome = td.decode_welcome(bytes.fromhex(
        "0100e0930400000000000000000000000000000001000001090000000000000006000000000000002800000000000000"))
    assert welcome == td.Welcome(lease_s=300.0, passes=None, max_block=65536, reconnect=False,
                                 superseded=9, reclaimed=6, next=40)
    grant = td.decode_grant(bytes.fromhex(
        "000700000000000000000000000200000004000000000000000a0000000000000014000000000000001600000000000000"))
    assert grant == td.Grant(td.GRANT, 7, 0.0, ((4, 10), (20, 22)))
    assert td.decode_grant(bytes.fromhex("010000000000000000e803000000000000")) == td.Grant(td.WAIT, 0, 1.0, ())
    assert td.decode_grant(bytes.fromhex("0200000000000000000000000000000000")).status == td.END
    report = td.encode_report(5, [(7, td.TRAINED, [(4, 6)]), (8, td.REQUEUE, [])])
    assert report.hex() == ("05000000000000000200000007000000000000000101000000040000000000000006"
                            "000000000000000800000000000000" "0300000000")
    assert td.encode_claim(4, 16).hex() == "040000000000000010000000"
    assert td.decode_ack(bytes.fromhex("020000000000000001000000000000000000000000000000")) == td.Ack(2, 1, 0)
    with pytest.raises(td.DispenseError):
        td.decode_grant(bytes.fromhex("0700000000000000000000000000000000"))


def test_ranges_and_position_map():
    assert td.ranges_of([5, 1, 2, 3, 9, 2]) == [(1, 4), (5, 6), (9, 10)]
    pmap = td.PositionMap(5, seed=11)
    assert [pmap.row(p) for p in range(5)] == [0, 1, 2, 3, 4], "pass 0 is data order"
    assert [pmap.row(p) for p in range(5, 10)] == pass_order(5, seed=11, data_pass=1)
    assert sorted(pmap.row(p) for p in range(10, 15)) == [0, 1, 2, 3, 4]
    assert pmap.data_pass(14) == 2
    with pytest.raises(ValueError):
        td.PositionMap(0, seed=1)


def test_file_fingerprint_counts_rows_and_hashes_bytes(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')
    rows, digest = td.file_fingerprint(path)
    assert rows == 2 and digest == hashlib.sha256(path.read_bytes()).digest()


# --------------------------------------------------------- against the syncer


def test_islands_of_different_speeds_share_one_sequence(syncer_binary, tmp_path):
    """3 islands, 1x / 2x / 4x speed: no position goes to two islands, the
    fast island trains about 4x the slow one, nobody waits for anybody."""
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        speeds = {0: 0.004, 1: 0.008, 2: 0.016}
        sources = {i: source(syncer.port, i, block=8) for i in speeds}
        trained: dict[int, list[int]] = {i: [] for i in speeds}
        stop = time.monotonic() + 3.0
        rng = random.Random(0)

        def island(i: int) -> None:
            while time.monotonic() < stop:
                positions = sources[i].take(8, timeout_s=10)
                time.sleep(speeds[i])
                dropped = [p for p in positions if rng.random() < 0.1]
                requeue = [p for p in positions if p not in dropped and rng.random() < 0.05]
                done = [p for p in positions if p not in dropped and p not in requeue]
                sources[i].report(trained=done, dropped=dropped, requeue=requeue)
                trained[i].extend(done)

        threads = [threading.Thread(target=island, args=(i,)) for i in speeds]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        for src in sources.values():
            src.close(release=True)
        flat = [p for ps in trained.values() for p in ps]
        assert len(flat) == len(set(flat)), "a position was trained by two islands"
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["held_at_end"] == 0
        assert summary["trained"] == len(flat)
        fast, slow = len(trained[0]), len(trained[2])
        assert fast > 2.5 * slow, (fast, slow)
        assert summary["islands"]["0"]["granted"] > summary["islands"]["2"]["granted"]
    finally:
        syncer.stop()


def test_a_crashed_island_loses_its_rows_to_the_others_after_the_lease(syncer_binary, tmp_path):
    syncer = Syncer(syncer_binary, tmp_path, lease_s=1.5)
    try:
        a = source(syncer.port, 0, block=4)
        b = source(syncer.port, 1, block=4)
        b_taken = b.take(4)
        b.report(trained=b_taken[:1])
        assert b.flush(5)
        time.sleep(0.3)
        held_by_b = sorted(b._owner) + [p for p, _ in b._ready]  # taken + prefetched
        crash(b)
        got: list[int] = []
        deadline = time.monotonic() + 10
        while not set(held_by_b) <= set(got) and time.monotonic() < deadline:
            positions = a.take(4, timeout_s=5)
            a.report(trained=positions)
            got.extend(positions)
        assert set(held_by_b) <= set(got), "the crashed island's rows never reached island 0"
        a.close()
        records = syncer.records()
        reclaims = [r for r in records if r["kind"] == "dispense_reclaim"]
        assert reclaims and reclaims[0]["reason"] == "lease_expired" and reclaims[0]["island_id"] == 1
        summary = check(records)
        assert summary["ok"], summary["problems"]
        assert summary["islands"]["1"]["trained"] == 1
    finally:
        syncer.stop()


def test_a_restarted_island_supersedes_its_old_process_at_once(syncer_binary, tmp_path):
    syncer = Syncer(syncer_binary, tmp_path, lease_s=60)
    try:
        a = source(syncer.port, 0, block=4)
        old = source(syncer.port, 1, block=4)
        old.take(4)
        time.sleep(0.2)
        held = sorted(old._owner) + [p for p, _ in old._ready]
        crash(old)
        events: list = []
        new = source(syncer.port, 1, block=4, events=events)  # same island, new process
        opened = [f for kind, f in events if kind == "rl_dispense_open"][0]
        assert opened["superseded"] == f"{old.incarnation:016x}" and opened["reclaimed"] == len(held)
        # no lease wait: the old rows go first to whoever claims next
        got = new.take(len(held), timeout_s=5) + a.take(4, timeout_s=5)
        assert set(held) <= set(got)
        new.report(trained=got[:len(held)])
        a.report(trained=got[len(held):])
        new.close()
        a.close()
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["islands"]["1"]["incarnations"] == 2
    finally:
        syncer.stop()


def test_a_new_island_joins_a_running_pool_and_gets_fresh_rows(syncer_binary, tmp_path):
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        a = source(syncer.port, 0, block=5, prefetch=1)
        first = a.take(5)
        late = source(syncer.port, 7, block=5, prefetch=1)  # an id the launch never named
        theirs = late.take(5)
        assert not set(first) & set(theirs)
        a.report(trained=first)
        late.report(trained=theirs)
        a.close()
        late.close()
        summary = check(syncer.records())
        assert summary["ok"] and "7" in summary["islands"]
    finally:
        syncer.stop()


def test_limited_passes_end_the_data_after_every_row(syncer_binary, tmp_path):
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", "--dispense-max-redo", "1")
    try:
        sources = [source(syncer.port, i, block=3, rows=20) for i in range(2)]
        seen: list[int] = []
        ended = []

        def island(src: td.TaskSource, requeue_first: bool) -> None:
            first = True
            while True:
                try:
                    positions = src.take(3, timeout_s=10)
                except td.DispenseEnded:
                    ended.append(True)
                    return
                if first and requeue_first:
                    src.report(requeue=positions)  # stale: trained again elsewhere
                    first = False
                    continue
                src.report(trained=positions)
                seen.extend(positions)

        threads = [threading.Thread(target=island, args=(s, i == 0)) for i, s in enumerate(sources)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert len(ended) == 2
        assert sorted(seen) == list(range(20)), "every row trained exactly once (the requeued ones too)"
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["given_out"] == 20 and summary["trained"] == 20
        assert any(r["kind"] == "dispense_end" for r in syncer.records())
        for s in sources:
            s.close()
    finally:
        syncer.stop()


def test_refusals_wrong_key_and_other_data(syncer_binary, tmp_path):
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        good = source(syncer.port, 0, block=2, rows=10)
        with pytest.raises(td.DispenseError, match="HMAC"):
            source(syncer.port, 1, block=2, rows=10, key=b"wrong")
        with pytest.raises(td.DispenseMismatch):
            source(syncer.port, 1, block=2, rows=11)
        with pytest.raises(td.DispenseMismatch):
            source(syncer.port, 1, block=2, rows=10, fingerprint=b"\x01" * 32)
        good.close()
    finally:
        syncer.stop()


class Proxy:
    """A TCP relay the test can cut (head unreachable) and restore."""

    def __init__(self, target_port: int) -> None:
        self.target = ("127.0.0.1", target_port)
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.port = self.listener.getsockname()[1]
        self.up = threading.Event()
        self.up.set()
        self.conns: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            if not self.up.is_set():
                client.close()
                continue
            upstream = socket.create_connection(self.target)
            self.conns += [client, upstream]
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    @staticmethod
    def _pipe(a: socket.socket, b: socket.socket) -> None:
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass

    def cut(self) -> None:
        self.up.clear()
        for s in self.conns:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.conns.clear()

    def restore(self) -> None:
        self.up.set()


def test_head_unreachable_island_trains_on_from_its_blocks_then_waits(syncer_binary, tmp_path):
    """Short outage (< lease): prefetched blocks keep the island going, the
    queued reports reach the dispenser once (same seq), nothing is fenced.
    Long outage (> lease): the island stops taking at its own lease deadline,
    is fenced on return and continues with a new incarnation."""
    syncer = Syncer(syncer_binary, tmp_path, lease_s=2.0)
    proxy = Proxy(syncer.port)
    try:
        events: list = []
        src = source(syncer.port, 0, block=4, prefetch=3, events=events,
                     address=("127.0.0.1", proxy.port), retry_s=0.1)
        first = src.take(4, timeout_s=5)
        time.sleep(0.3)  # prefetch fills
        proxy.cut()
        offline = src.take(4, timeout_s=1)  # from a prefetched block, head down
        src.report(trained=first + offline)
        time.sleep(0.4)
        proxy.restore()
        assert src.flush(10)
        assert src.stats["fenced"] == 0 and src.stats["trained"] == 8
        # long outage: take() blocks once the lease runs out on the island's clock
        proxy.cut()
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            while True:
                src.take(1, timeout_s=3.5 - (time.monotonic() - started))
        assert time.monotonic() - started < 3.6
        assert 1.2 < src._clock() - src._contact, "stopped at the lease margin, not before"
        time.sleep(1.0)  # the dispenser expires the lease meanwhile
        proxy.restore()
        after = src.take(4, timeout_s=10)
        assert src.stats["fenced"] == 1 and any(kind == "rl_dispense_fenced" for kind, _ in events)
        src.report(trained=after)
        src.close()
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["islands"]["0"]["incarnations"] == 2
    finally:
        proxy.cut()
        syncer.stop()


# ---------------------------------------------------------- checkpoint resume


def test_a_resumed_syncer_returns_to_the_checkpoint_cut(syncer_binary, tmp_path):
    """The dispenser record is a checkpoint sidecar: a learner drives a few
    rounds (each writes a checkpoint), the syncer dies, and the resumed
    syncer restores the record of the last checkpoint: every process of the
    record is fenced, its unreported rows come first, trained rows stay
    trained."""
    torch = pytest.importorskip("torch")
    from tests.test_integration import ToyLearner, DIM
    from yeto.fragments import build_layout

    port = free_port()
    ckpt = tmp_path / "state.ckpt"
    common = ("--checkpoint-path", str(ckpt), "--checkpoint-every", "1", "--grace-ms", "50")
    syncer = Syncer(syncer_binary, tmp_path, *common, port=port, key=None, learners=1, lease_s=60)
    try:
        src = source(port, 0, block=4, rows=50, key=None, prefetch=1)
        positions = src.take(4)
        src.report(trained=positions[:2], dropped=positions[2:3])  # position 3 stays held
        assert src.flush(5)
        layout = build_layout([("model.embed.weight", DIM // 4), ("model.body.weight", DIM)], 2)
        learner = ToyLearner(0, port, torch.randn(DIM + DIM // 4), layout)
        learner.start()
        deadline = time.monotonic() + 30
        while not (Path(str(ckpt) + ".dispense").exists() and any(
                r["kind"] == "dispense_checkpoint" for r in syncer.records())):
            assert time.monotonic() < deadline, "no checkpoint with a dispenser record"
            time.sleep(0.05)
        cut = [r for r in syncer.records() if r["kind"] == "dispense_checkpoint"][-1]
        syncer.stop()  # SIGKILL
        learner.join(timeout=10)
        before = syncer.records()
        resumed = Syncer(syncer_binary, tmp_path, *common, "--resume", port=free_port(), key=None,
                         learners=1, lease_s=60, name="resumed")
        try:
            restore = [r for r in resumed.records() if r["kind"] == "dispense_restore"]
            assert restore and restore[-1]["next"] == cut["next"]
            with pytest.raises(td.DispenseFenced):
                td.DispenseConnection(("127.0.0.1", resumed.port), learner_id=0,
                                      incarnation=src.incarnation, rows=50, fingerprint=FP, key=None)
            again = source(resumed.port, 0, block=8, rows=50, key=None, prefetch=1)
            got = again.take(8)
            held_at_cut = {3} | set(range(4, cut["next"]))  # held (incl. prefetch) at the cut
            assert set(got[:len(held_at_cut)]) == held_at_cut, (got, cut)
            assert not set(got) & set(positions[:3]), "trained / dropped rows are never given out again"
            again.report(trained=got)
            again.close()
            summary = check(resumed.records())
            assert summary["ok"], summary["problems"]
            assert summary["restores"] == 1
            assert len(before) < len(resumed.records())
        finally:
            resumed.stop()
    finally:
        syncer.stop()


# --------------------------------------------------------------- the checker


def _grant(island, inc, block, ranges):
    return {"kind": "dispense_grant", "island_id": island, "incarnation": inc, "block": block,
            "ranges": ranges}


def test_checker_catches_a_double_grant_and_a_double_training():
    base = [{"kind": "dispense_open", "island_id": 0, "incarnation": "a", "rows": 10},
            {"kind": "dispense_open", "island_id": 1, "incarnation": "b", "rows": 10},
            _grant(0, "a", 1, [[0, 4]])]
    assert check(base)["ok"]
    double = base + [_grant(1, "b", 2, [[2, 6]])]
    result = check(double)
    assert not result["ok"] and "while island 0 holds it" in result["problems"][0]
    trained_twice = base + [
        {"kind": "dispense_report", "island_id": 0, "incarnation": "a", "block": 1, "report": "trained",
         "ranges": [[0, 4]], "redo_exhausted": []},
        _grant(1, "b", 2, [[0, 2]]),
        {"kind": "dispense_report", "island_id": 1, "incarnation": "b", "block": 2, "report": "trained",
         "ranges": [[0, 2]], "redo_exhausted": []}]
    result = check(trained_twice)
    assert not result["ok"]
    assert any("granted again" in p for p in result["problems"])
    assert any("trained twice" in p for p in result["problems"])


def test_checker_compares_tape_rows_with_trained_reports():
    log = [{"kind": "dispense_open", "island_id": 0, "incarnation": "a", "rows": 4},
           _grant(0, "a", 1, [[0, 6]]),
           {"kind": "dispense_report", "island_id": 0, "incarnation": "a", "block": 1, "report": "trained",
            "ranges": [[0, 6]], "redo_exhausted": []}]
    summary = check(log)
    rows = [td.PositionMap(4, seed=3).row(p) for p in range(6)]
    tape = [{"event": "rl_driver_start", "learner_id": 0},
            {"event": "rl_round_data_rows", "rows": [[r, r, 1] for r in rows]}]
    assert check_tapes(summary, tape, seed=3) == []
    extra = tape + [{"event": "rl_round_data_rows", "rows": [[3, 3, 1], [3, 3, 1]]}]
    assert check_tapes(summary, extra, seed=3)
