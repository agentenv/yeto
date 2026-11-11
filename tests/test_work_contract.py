"""WorkContract conformance (``yeto.rl.engine.work_contract``) of the store
ledger dispenser: the cases of ``tests/fixtures/work_contract_cases.json``,
which ``syncer/src/work_contract.rs`` runs against the syncer dispenser too."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.engine_sampler import unique_prompt_ids
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY
from yeto.rl.engine.work_contract import ClaimResult, Member, WorkContract

CASES = Path(__file__).parent / "fixtures" / "work_contract_cases.json"
NAMESPACE = "run/dispense"


class ServerObjects:
    """An object store with exclusive create and one server clock."""

    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.data: dict[str, tuple[bytes, float]] = {}
        self.lock = threading.Lock()

    def read(self, key):
        item = self.data.get(key)
        return None if item is None else item[0]

    def write(self, key, data):
        with self.lock:
            self.data[key] = (bytes(data), self.now)

    def create(self, key, data):
        with self.lock:
            if key in self.data:
                return False
            self.data[key] = (bytes(data), self.now)
            return True

    def exists(self, key):
        return key in self.data

    def list_meta(self, prefix):
        with self.lock:
            return [(k, t) for k, (_, t) in self.data.items() if k.startswith(prefix)], self.now


class StoreContract:
    """The WorkContract over :class:`sd.StoreDispensedPrompts` (one per member)
    sharing one ledger; ``fence`` moves the holder's incarnation lease."""

    def __init__(self, tmp_path: Path, config: dict) -> None:
        self.objects = ServerObjects()
        rows = [{"text": f"q{i}", "label": "", "metadata": {"task_id": f"t{i}"}} for i in range(config["rows"])]
        self.path = tmp_path / "prompts.jsonl"
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.config = {"backend": "store", "namespace": NAMESPACE, "lease_s": float(config["lease_s"]),
                       "max_redo": int(config["max_redo"]), "passes": int(config["passes"]), "seed": 0,
                       "scan_s": 0.0, "max_scan_s": 0.0}
        self.leases: dict[str, str] = {}  # holder -> the incarnation its incarnation lease names
        self.members: dict[Member, sd.StoreDispensedPrompts] = {}
        self.drains: dict[str, str] = {}  # holder -> the open drain request's seq

    def _prompts(self, m: Member) -> sd.StoreDispensedPrompts:
        if m not in self.members:
            raise AssertionError(f"{m} never joined")
        return self.members[m]

    def join(self, m: Member) -> int:
        self.leases[m.holder] = m.incarnation
        prompts = sd.StoreDispensedPrompts.open(
            self.path, self.config, store=self.objects, holder=m.holder, incarnation=m.incarnation,
            input_key="text", label_key="label", groups_per_round=1, task_key="metadata", renew=False,
            lease_of_holder=self.leases.get, fenced=lambda: self.leases.get(m.holder) != m.incarnation)
        self.members[m] = prompts
        return prompts.recover()["released"]

    def claim(self, m: Member, n: int) -> ClaimResult:
        status, slots = self._prompts(m).poll(n)
        return ClaimResult(status, tuple(slots))  # type: ignore[arg-type]

    def _report(self, m: Member, items, **how) -> bool:
        prompts = self._prompts(m)
        items = [int(i) for i in items]
        if not all(prompts.still_holds(i) for i in items):
            for i in items:  # a sampler drops a group it no longer holds
                prompts._end(i, fail="release", reason="not_held")
            return False
        keep = set(prompts._held) - set(items)
        if how.get("drop"):
            for i in items:
                prompts._end(i, done="dropped", reason="test")
        elif how.get("release"):
            for i in items:
                prompts._end(i, fail="release", reason="test")
        elif how.get("fail"):
            prompts.written(0, [], requeue=items, keep=keep)
        else:
            prompts.written(0, items, keep=keep)
        return True

    def done(self, m, items):
        return self._report(m, items)

    def drop(self, m, items):
        return self._report(m, items, drop=True)

    def fail(self, m, items):
        return self._report(m, items, fail=True)

    def release(self, m, items):
        return self._report(m, items, release=True)

    def beat(self, m: Member) -> None:
        self._prompts(m).renew()

    def still_holds(self, m: Member, item: int) -> bool:
        return self._prompts(m).still_holds(item)

    def fence(self, m: Member) -> None:
        self.leases[m.holder] = f"{m.incarnation}-fenced"

    def retire(self, holder: str, incarnation: str | None = None, reason: str = "scale_down") -> int:
        return sd.retire(self.objects, NAMESPACE, holder, incarnation, reason=reason)["released"]

    def drain(self, m: Member, on: bool = True) -> None:
        seq = sd.request_drain(self.objects, NAMESPACE, m.holder, on=on, seq=f"d{len(self.drains)}-{on}")
        if on:
            self.drains[m.holder] = seq
        else:
            self.drains.pop(m.holder, None)

    def drained(self, m: Member) -> bool:
        seq = self.drains.get(m.holder)
        return seq is not None and sd.drain_answered(self.objects, NAMESPACE, m.holder, seq)

    def advance(self, seconds: float) -> None:
        self.objects.now += float(seconds)

    def totals(self) -> dict[str, int]:
        totals = sd.status(self.objects, NAMESPACE, details=False)["totals"]
        return {k: totals[k] for k in ("trained", "dropped", "exhausted")}


def _member(text: str) -> Member:
    holder, inc = text.split("#")
    return Member(holder, inc)


def run_case(contract, case) -> None:
    for index, step in enumerate(case["steps"]):
        where = f"{case['name']} step {index}: {step}"
        op = step["op"]
        if op == "advance":
            contract.advance(step["s"])
            continue
        if op == "totals":
            assert contract.totals() == step["expect"], where
            continue
        m = _member(step["m"])
        if op == "join":
            assert contract.join(m) == step["expect"], where
        elif op == "claim":
            got = contract.claim(m, step["n"])
            want = step["expect"]
            if isinstance(want, dict):
                assert got.status == "items" and sorted(got.items) == sorted(want["items"]), (where, got)
            else:
                assert got.status == want, (where, got)
        elif op in ("done", "drop", "fail", "release"):
            ok = getattr(contract, op)(m, step["items"])
            assert ("ok" if ok else "not_held") == step["expect"], where
        elif op == "holds":
            assert contract.still_holds(m, step["item"]) is step["expect"], where
        elif op == "beat":
            contract.beat(m)
        elif op == "fence":
            contract.fence(m)
        elif op in ("drain", "undrain"):
            contract.drain(m, op == "drain")
        elif op == "drained":
            assert contract.drained(m) is step["expect"], where
        elif op == "retire":
            assert contract.retire(m.holder, m.incarnation) == step["expect"], where
        else:
            raise AssertionError(f"unknown op {op!r} ({where})")


def _cases():
    return json.loads(CASES.read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_store_ledger_conforms(tmp_path, case):
    contract = StoreContract(tmp_path, case["config"])
    assert isinstance(contract, WorkContract)
    run_case(contract, case)


def test_case_table_covers_every_operation():
    ops = {step["op"] for case in _cases() for step in case["steps"]}
    assert ops >= {"join", "claim", "done", "drop", "fail", "release", "beat", "holds", "fence", "retire",
                   "advance", "totals", "drain", "undrain", "drained"}
    names = {case["name"] for case in _cases()}
    assert len(names) == len(_cases())


def test_epoch_boundary_repeat_gets_its_own_prompt_id(tmp_path):
    contract = StoreContract(tmp_path, {"rows": 3, "passes": 2, "max_redo": 1, "lease_s": 900})
    m = Member("s0", "1")
    contract.join(m)
    prompts = contract.members[m]
    prompts.groups_per_round = 4
    items = prompts(0)
    assert [it.metadata[DISPENSE_POSITION_KEY] for it in items] == [0, 1, 2, 3]
    rows = [it.metadata["row"] for it in items]
    assert len(set(rows)) == 3  # slot 3 (pass 1) repeats a row of pass 0
    ids = [it.prompt_id for it in unique_prompt_ids(items)]
    assert len(set(ids)) == 4 and ids[3] == f"row{rows[3]}-p3"


def test_retire_without_incarnation_retires_every_holding_incarnation(tmp_path):
    contract = StoreContract(tmp_path, {"rows": 4, "passes": 1, "max_redo": 1, "lease_s": 900})
    a, b = Member("s0", "1"), Member("s1", "1")
    contract.join(a)
    contract.join(b)
    assert contract.claim(a, 2).items == (0, 1)
    report = sd.retire(contract.objects, NAMESPACE, "s0", reason="evicted")
    assert report == {"holder": "s0", "incarnations": ["1"], "released": 2, "slots": [0, 1]}
    assert contract.claim(a, 1).status == "fenced"
    assert contract.claim(b, 4).items == (0, 1, 2, 3)
    fails = [k for k in contract.objects.data if "/fail/" in k]
    body = json.loads(contract.objects.read(fails[0]))
    assert body["reason"] == "retired" and body["retire_reason"] == "evicted"


def test_retire_cli_on_a_local_store(tmp_path, capsys):
    root = tmp_path / "store"
    objects = sd.DirObjects(root)
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps({"text": f"q{i}", "metadata": {"task_id": f"t{i}"}}) + "\n" for i in range(2)))
    prompts = sd.StoreDispensedPrompts.open(path, {"namespace": NAMESPACE, "scan_s": 0, "max_scan_s": 0},
                                            store=objects, holder="nc-1", incarnation="x", input_key="text",
                                            label_key=None, groups_per_round=1, task_key="metadata", renew=False)
    assert prompts.poll(1) == ("items", [0])
    assert sd.main(["retire", "--store", str(root), "--namespace", NAMESPACE, "--holder", "nc-1"]) == 0
    assert json.loads(capsys.readouterr().out)["released"] == 1
    assert prompts.poll(1) == ("fenced", [])
    assert not prompts.still_holds(0)
