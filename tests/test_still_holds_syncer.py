"""``still_holds`` on the syncer data path: TASK_HOLDS / TASK_HELD, the task
source's check before a group is written, and the WorkContract cases run over
the wire against the real syncer (cases without clock steps; the in-process
Rust driver ``syncer/src/work_contract.rs`` runs them all)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tests.test_dispense_syncer import FP, KEY, Syncer, syncer_binary  # noqa: F401 - the fixture
from yeto.rl.engine import task_dispense as td

CASES = Path(__file__).parent / "fixtures" / "work_contract_cases.json"


def test_wire_bytes_match_the_rust_golden_vectors():
    """Same vectors as ``dispense_service::tests::holds_golden_vectors``."""
    assert td.encode_holds([3, 1 << 40]).hex() == "0200000003000000000000000000000000010000"
    assert td.decode_held(bytes.fromhex("020000000100")) == [True, False]
    with pytest.raises(td.DispenseError):
        td.decode_held(bytes.fromhex("0200000001"))


class WireContract:
    """The WorkContract over TCP: one DispenseConnection per member."""

    def __init__(self, port: int, rows: int) -> None:
        self.port, self.rows = port, rows
        self.address = ("127.0.0.1", port)
        self.ids: dict[str, int] = {}
        self.conns: dict[str, td.DispenseConnection] = {}
        self.seq: dict[str, int] = {}
        self.held: dict[str, dict[int, int]] = {}
        self.fenced: set[str] = set()

    def _id(self, m: str) -> tuple[int, int]:
        holder, inc = m.split("#")
        lid = self.ids.setdefault(holder, td.sampler_holder_id(f"wire-{holder}"))
        return lid, int(inc)

    def _next(self, m: str) -> int:
        self.seq[m] = self.seq.get(m, 0) + 1
        return self.seq[m]

    def join(self, m: str) -> int:
        lid, inc = self._id(m)
        conn = td.DispenseConnection(self.address, learner_id=lid, incarnation=inc, rows=self.rows,
                                     fingerprint=FP, key=KEY, timeout_s=5.0)
        self.conns[m] = conn
        return conn.welcome.reclaimed

    def _fence(self, m: str) -> None:
        self.fenced.add(m)
        self.held.pop(m, None)

    def claim(self, m: str, n: int):
        if m in self.fenced:
            return "fenced", []
        try:
            grant = self.conns[m].claim(self._next(m), n)
        except (td.DispenseFenced, ConnectionError):
            self._fence(m)
            return "fenced", []
        if grant.status == td.WAIT:
            return "wait", []
        if grant.status == td.END:
            return "end", []
        items = [p for s, e in grant.ranges for p in range(s, e)]
        for p in items:
            self.held.setdefault(m, {})[p] = grant.block
        return "items", items

    def report(self, m: str, items, kind: int) -> bool:
        held = self.held.get(m, {})
        if m in self.fenced or not all(i in held for i in items):
            return False
        by_block: dict[int, list[int]] = {}
        for i in items:
            by_block.setdefault(held.pop(i), []).append(i)
        try:
            self.conns[m].report(self._next(m), [(b, kind, td.ranges_of(ps)) for b, ps in by_block.items()])
        except (td.DispenseFenced, ConnectionError):
            self._fence(m)
            return False
        return True

    def holds(self, m: str, item: int) -> bool:
        if m in self.fenced:
            return False
        try:
            return self.conns[m].holds([item])[0]
        except (td.DispenseFenced, ConnectionError):
            self._fence(m)
            return False

    def retire(self, m: str, mode: str = "retire") -> dict:
        lid, inc = self._id(m)
        return td.retire_member(self.address, lid, None if mode != "retire" else inc, key=KEY, mode=mode)

    def close(self) -> None:
        for conn in self.conns.values():
            conn.close()


KINDS = {"done": td.TRAINED, "drop": td.DROPPED, "fail": td.REQUEUE, "release": td.RELEASE}


def _wire_cases():
    cases = json.loads(CASES.read_text())["cases"]
    return [c for c in cases if not any(s["op"] in ("advance", "beat") for s in c["steps"])]


@pytest.mark.parametrize("case", _wire_cases(), ids=lambda c: c["name"])
def test_the_real_syncer_conforms_over_the_wire(syncer_binary, tmp_path, case):  # noqa: F811
    config = case["config"]
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", str(config["passes"]),
                    "--dispense-max-redo", str(config["max_redo"]), lease_s=600.0)
    c = WireContract(syncer.port, config["rows"])
    try:
        for index, step in enumerate(case["steps"]):
            where = f"{case['name']} step {index}: {step}"
            op, m = step["op"], step.get("m")
            if op == "totals":
                continue  # the in-process drivers check the totals
            if op == "join":
                assert c.join(m) == step["expect"], where
            elif op == "claim":
                status, items = c.claim(m, step["n"])
                want = step["expect"]
                if isinstance(want, dict):
                    assert status == "items" and sorted(items) == sorted(want["items"]), (where, status, items)
                else:
                    assert status == want, (where, status, items)
            elif op in KINDS:
                ok = c.report(m, step["items"], KINDS[op])
                assert ("ok" if ok else "not_held") == step["expect"], where
            elif op == "holds":
                assert c.holds(m, step["item"]) is step["expect"], where
            elif op in ("fence", "retire"):
                answer = c.retire(m)
                if op == "retire":
                    assert answer["reclaimed"] == step["expect"], where
            elif op in ("drain", "undrain"):
                c.retire(m, mode=op)
            elif op == "drained":
                assert (c.retire(m, mode="drain")["status"] == "drained") is step["expect"], where
            else:
                raise AssertionError(f"unknown op {op!r} ({where})")
    finally:
        c.close()
        syncer.stop()


def test_the_task_source_checks_still_holds_before_a_write(syncer_binary, tmp_path):  # noqa: F811
    syncer = Syncer(syncer_binary, tmp_path, lease_s=600.0)
    island = td.sampler_holder_id("holds-1")
    try:
        connect = td.tcp_connector(("127.0.0.1", syncer.port), learner_id=island, rows=100,
                                   fingerprint=FP, key=KEY, timeout_s=5.0)
        source = td.TaskSource(connect=connect, block=4, prefetch_blocks=1, retry_s=0.05,
                               incarnation=lambda: 0x9999).start()
        rows = td.DispensedRows(source, td.PositionMap(100, seed=0))
        taken = source.take(4, timeout_s=5)
        assert all(rows.still_holds(p) for p in taken)
        assert not source.still_holds(99), "never taken here"
        source.report(trained=[taken[0]])
        assert source.flush(5)
        assert not source.still_holds(taken[0]), "reported: settled"
        assert td.retire_member(("127.0.0.1", syncer.port), island, key=KEY)["reclaimed"] >= 3
        started = time.monotonic()
        assert not source.still_holds(taken[1]), "retired by the controller: given to others"
        assert time.monotonic() - started < 5
        source.close(release=False)
    finally:
        syncer.stop()
