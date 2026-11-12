"""The ticket ledger and the lease file under concurrent writers (the pool's ticket
prefetch threads, a manual CLI on the same state dir): no row is lost, and a ledger
that cannot be read is never taken for an empty one."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from yeto.launch.nc import cli
from yeto.launch.nc import client as ncc

REPO = Path(__file__).resolve().parents[1]


def test_a_writer_that_read_the_ledger_before_another_writer_does_not_drop_its_row(tmp_path):
    """A: load | B: load, append t-b, write | A: append t-a, write -> t-b is gone."""
    path = tmp_path / "tickets.json"
    loaded, b_done = threading.Event(), threading.Event()

    class Slow(ncc.Ledger):
        def load(self):
            rows = super().load()
            loaded.set()
            b_done.wait(1.0)  # B finishes here without a lock; with one, B waits for A
            return rows

    def b():
        loaded.wait(5)
        ncc.Ledger(path).add(1, "t-b")
        b_done.set()

    thread = threading.Thread(target=b)
    thread.start()
    Slow(path).add(1, "t-a")
    thread.join(10)
    assert sorted(r["ticket"] for r in ncc.Ledger(path).load()) == ["t-a", "t-b"]


def test_sixteen_threads_adding_and_marking_tickets_lose_nothing(tmp_path):
    path = tmp_path / "tickets.json"
    errors = []

    def work(i):
        try:
            for j in range(20):
                ledger = ncc.Ledger(path)  # cli.ledger() builds one per call
                ledger.add(i, f"t{i}-{j}", run="run-a")
                ledger.mark(f"t{i}-{j}", ip=f"192.0.2.{i}", ended=j % 2 == 0)
        except BaseException as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    rows = ncc.Ledger(path).load()
    assert errors == []
    assert sorted(r["ticket"] for r in rows) == sorted(f"t{i}-{j}" for i in range(16) for j in range(20))
    assert all(r.get("ip") for r in rows)
    assert len(ncc.Ledger(path).open_rows()) == 16 * 10
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "tickets.lock") == ["tickets.json"]


WRITER = """
import sys
from pathlib import Path
from yeto.launch.nc import cli, client as ncc
kind, path, who, n = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
for j in range(n):
    if kind == "ledger":
        ncc.Ledger(path).add(int(who), f"t{who}-{j}")
        ncc.Ledger(path).mark(f"t{who}-{j}", ip="192.0.2.1")
    else:
        cli.save_lease(ncc.NodeLease(int(who), f"t{who}-{j}", "192.0.2.1", "ubuntu", "k", None, None, 8, 1))
"""


def _processes(kind, path, tmp_path, *, procs=6, n=40):
    env = {**os.environ, "PYTHONPATH": str(REPO), "YETO_NC_STATE_DIR": str(tmp_path)}
    running = [subprocess.Popen([sys.executable, "-c", WRITER, kind, str(path), str(i), str(n)], env=env,
                                stderr=subprocess.PIPE, text=True) for i in range(procs)]
    failures = [(p.returncode, err[-400:]) for p in running for err in [p.communicate(timeout=300)[1]]
                if p.returncode != 0]
    return failures, sorted(f"t{i}-{j}" for i in range(procs) for j in range(n))


def test_several_processes_writing_one_ledger_lose_nothing(tmp_path):
    path = tmp_path / "tickets.json"
    failures, expected = _processes("ledger", path, tmp_path)
    assert failures == []
    rows = ncc.Ledger(path).load()
    assert sorted(r["ticket"] for r in rows) == expected and all(r.get("ip") for r in rows)


def test_several_processes_saving_leases_lose_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path))
    failures, expected = _processes("leases", tmp_path / "leases.json", tmp_path)
    assert failures == []
    assert sorted(json.loads((tmp_path / "leases.json").read_text())) == expected
    assert cli.load_lease("t0-0").ip == "192.0.2.1"


def test_a_ledger_that_cannot_be_read_is_an_error_and_is_not_overwritten(tmp_path):
    path = tmp_path / "tickets.json"
    assert ncc.Ledger(path).load() == []  # no file: no tickets yet
    ncc.Ledger(path).add(1, "t1", run="run-a")
    good = path.read_text()
    path.write_text(good[: len(good) // 2])  # a torn file
    ledger = ncc.Ledger(path)
    for read in (ledger.load, ledger.open_rows, lambda: ledger.owns(1, "t1"), lambda: ledger.run_rows("run-a")):
        with pytest.raises(ncc.NcError, match="tickets.json"):
            read()
    with pytest.raises(ncc.NcError):
        ledger.add(1, "t2")  # would have written a ledger holding t2 alone
    with pytest.raises(ncc.NcError):
        ledger.mark("t1", ended=True)
    assert path.read_text() == good[: len(good) // 2]
    path.write_text('{"not": "a list"}')
    with pytest.raises(ncc.NcError):
        ledger.load()
