"""The syncer's tape in the run store (tape_upload.syncer_uploaders, LocalSyncer.start_store_upload)
and the dashboard reading it from there alone (store_tapes, topology_diloco.shard_count)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dashboard_helpers import FX

from yeto.dashboard import topology_diloco
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all
from yeto.dashboard.store_tapes import StoreTapes
from yeto.launch.local_syncer import LocalSyncer
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.tape_upload import shard_tape, syncer_island, syncer_uploaders

RUN_B = FX / "diloco" / "run-b"
REPO = Path(__file__).resolve().parents[1]


def merges() -> list[str]:
    return [x for x in (RUN_B / "yeto-tape.jsonl").read_text().splitlines() if x.strip()]


def write_shards(tape: Path) -> None:
    """run B's merges split the way two shards write them: steps t = 1 / 2 (mod 2)."""
    tape.parent.mkdir(parents=True, exist_ok=True)
    for i in (0, 1):
        shard_tape(tape, i).write_text("".join(x + "\n" for x in merges() if json.loads(x)["step"] % 2 == (1 - i)))


def store_reducer(root: Path) -> Reducer:
    """run B's island tapes from disk, the syncer only from the store."""
    r = Reducer(run="b")
    load_all(r, [str(RUN_B / "events"), str(RUN_B / "fleet.jsonl")])
    StoreTapes(DirectoryObjectStore(root)).pump(r)
    return r


def test_names_and_shard_paths():
    assert [syncer_island(i) for i in range(3)] == ["syncer", "syncer-shard1", "syncer-shard2"]
    assert shard_tape("/x/yeto-tape.jsonl", 0) == Path("/x/yeto-tape.jsonl")
    assert shard_tape("/x/yeto-tape.jsonl", 2) == Path("/x/yeto-tape.shard2.jsonl")


def test_the_dashboard_sees_the_sharded_syncer_from_the_store_alone(tmp_path):
    tape, root = tmp_path / "out" / "yeto-tape.jsonl", tmp_path / "store"
    write_shards(tape)
    ups = syncer_uploaders(DirectoryObjectStore(root), tape, shards=2, incarnation="h-1", interval_s=3600,
                           from_end=False)
    assert all(up.stop(timeout_s=5) for up in ups)
    assert sorted(p.parent.parent.name for p in root.rglob("*.jsonl")) == ["syncer", "syncer-shard1"]
    r = store_reducer(root)
    assert r.counts["syncer"] == len(merges()) and topology_diloco.shard_count(r) == 2
    d = r.page_view(now=r.max_ts)["topology"]["diloco"]
    assert d["versions"] == [21, 22, 23, 24] and [(s["lo"], s["hi"], s["step"]) for s in d["shards"]] == [(0, 2, 23), (2, 4, 24)]


def test_local_tape_and_its_upload_count_once(tmp_path):
    tape, root = tmp_path / "out" / "yeto-tape.jsonl", tmp_path / "store"
    write_shards(tape)
    for up in syncer_uploaders(DirectoryObjectStore(root), tape, shards=2, incarnation="h-1", interval_s=3600,
                               from_end=False):
        up.stop(timeout_s=5)
    r = store_reducer(root)
    load_all(r, [str(tape), str(shard_tape(tape, 1))])
    assert len(r.merges) == len(merges()) and r.duplicates >= len(merges())


def test_from_end_skips_what_an_earlier_job_left(tmp_path):
    tape, root = tmp_path / "yeto-tape.jsonl", tmp_path / "store"
    lines = merges()
    tape.write_text("".join(x + "\n" for x in lines[:5]))
    (up,) = syncer_uploaders(DirectoryObjectStore(root), tape, incarnation="h-2", interval_s=3600)
    with open(tape, "a") as fh:
        fh.write(lines[5] + "\n")
    assert up.stop(timeout_s=5)
    got = [x for p in sorted(root.rglob("*.jsonl")) for x in p.read_text().splitlines()]
    assert got == [lines[5]]


def test_a_failing_store_never_reaches_the_syncer(tmp_path):
    class Down:
        def write(self, key, data):
            raise ConnectionError("store unreachable")

    tape = tmp_path / "yeto-tape.jsonl"
    tape.write_text(merges()[0] + "\n")
    logs: list[str] = []
    (up,) = syncer_uploaders(Down(), tape, incarnation="h", interval_s=3600, from_end=False, log=logs.append)
    assert up.flush() == 0 and up.failures == 1 and up.pending_bytes() > 0  # kept on the local tape
    assert logs and "ConnectionError" in logs[0]
    up.stop(timeout_s=0)


def _args(**kw) -> argparse.Namespace:
    base = {"rl_checkpoint_store": "s3://bucket/runs/r1", "syncer_shards": 1, "training_mode": "rl"}
    base.update(kw)
    return argparse.Namespace(**base)


def _local(tmp_path, monkeypatch) -> LocalSyncer:
    monkeypatch.setattr("yeto.launch.local_syncer.syncer_command", lambda *a, **k: "true")
    monkeypatch.setattr("yeto.launch.local_syncer.syncer_event_tape", lambda args: str(tmp_path / "yeto-tape.jsonl"))
    return LocalSyncer(_args(), 2, log_file=str(tmp_path / "syncer.log"))


def test_local_syncer_uploads_only_to_an_object_store(tmp_path, monkeypatch):
    monkeypatch.delenv("YETO_RL_TAPE_UPLOAD_S", raising=False)
    sy = _local(tmp_path, monkeypatch)
    root = DirectoryObjectStore(tmp_path / "store")
    sy.start_store_upload(_args(rl_checkpoint_store="modal-volume://vol/runs/r1"), objects=root)
    sy.start_store_upload(_args(rl_checkpoint_store=None), objects=root)
    assert sy.store_uploaders == ()
    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "0")
    sy.start_store_upload(_args(), objects=root)
    assert sy.store_uploaders == ()
    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "3600")
    monkeypatch.setattr("yeto.launch.syncer.syncer_shard_count", lambda args: 2)
    sy.start_store_upload(_args(), objects=root)
    assert [u.island for u in sy.store_uploaders] == ["syncer", "syncer-shard1"]
    (tmp_path / "yeto-tape.jsonl").write_text(merges()[0] + "\n")
    sy.stop()  # drains: the line written after the start is in the store
    assert sy.store_uploaders == () and len(list((tmp_path / "store" / "tapes" / "syncer").rglob("*.jsonl"))) == 1


def test_follow_runs_beside_a_syncer_until_stopped(tmp_path):
    tape, root = tmp_path / "yeto-tape.jsonl", tmp_path / "store"
    tape.write_text(merges()[0] + "\n")
    env = dict(os.environ, PYTHONPATH=str(REPO), YETO_RL_TAPE_UPLOAD_S="0.2")
    proc = subprocess.Popen([sys.executable, "-m", "yeto.rl.engine.tape_upload", "follow", "--tape", str(tape),
                             "--store", str(root), "--incarnation", "c1"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not list(root.rglob("*.jsonl")):
            time.sleep(0.1)
    finally:
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert [p.relative_to(root).parts[:3] for p in root.rglob("*.jsonl")] == [("tapes", "syncer", "c1")]
