"""The on-site syncer service with the real Rust syncer: the syncer process is killed while
a learner merges, the service restarts it (as ``yeto fleet service-run --role syncer``
does: ``LocalSyncer`` under :func:`yeto.launch.fleet_syncer.serve`), the restarted syncer
resumes from its checkpoint on the same address, its record is written again, and the
merge goes on to the run's last step: every outer step once, in order, none skipped. CPU
only; a port of 8880-8899."""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import threading
import time
import types

import pytest
import torch

from test_integration import DIM, ToyLearner
from test_rl_full_parameter_async import needs_cargo
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from yeto.final_marker import read_checkpoint_global_step
from yeto.fragments import build_layout
from yeto.launch import fleet_syncer
from yeto.launch import syncer as syncer_mod
from yeto.launch.local_syncer import LocalSyncer
from yeto.protocol import DTYPE_F32
from yeto.rl import island_launches
from yeto.rl.engine.object_store import DirectoryObjectStore


def _port() -> int:
    for port in range(8880, 8900):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    pytest.skip("no free port in 8880-8899")


class ServiceSyncer(LocalSyncer):
    """``LocalSyncer`` with a small SFT syncer command: a checkpoint every outer step, and
    ``--resume`` when the checkpoint exists (the launch's ``_resume_if_exists``)."""

    def __init__(self, binary, tmp, port: int, total: int) -> None:  # noqa: D107 - the base needs launch args
        self.checkpoint, self.tape = tmp / "state.ckpt", tmp / "tape.jsonl"
        self.binary = str(binary)
        self.log_file = str(tmp / "syncer.log")
        self.event_tape = str(self.tape)
        self.command = (f"exec {binary} --port {port} --learners 1 --quorum 1 --grace-ms 20 "
                        "--sync-interval-steps 0 --min-round-interval-ms 50 "
                        f"--total-steps {total} --checkpoint-path {self.checkpoint} --checkpoint-every 1 "
                        f"--event-tape {self.tape} --shutdown-timeout-s 5 "
                        + syncer_mod._resume_if_exists(str(self.checkpoint)))
        self.proc, self._reaped = None, False
        self.tape_forwarder = self.tape_run = None
        self._log_offset = self._event_offset = 0
        self.store_uploaders = ()


def _steps(tape) -> list[int]:
    if not tape.exists():
        return []
    out = []
    for line in tape.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if "step" in row:
                out.append(int(row["step"]))
    return out


def _command_of(pid: int) -> str:
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout


@needs_cargo
def test_a_killed_syncer_is_restarted_from_its_checkpoint_and_the_merge_goes_on(  # noqa: PLR0915
        syncer_binary, tmp_path, monkeypatch):  # noqa: F811
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    port, total = _port(), 14
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(syncer_mod, "_head_ports", lambda args: [port])
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects)
    args = types.SimpleNamespace(cluster_prefix="r", rl_checkpoint_store=str(tmp_path / "store"),
                                 rl_syncer_epoch=4, rl_sync_max_learners=4)
    syncer = ServiceSyncer(syncer_binary, tmp_path, port, total)
    syncer.start()
    extra = {"service": "127.0.0.1", "key_id": fleet_syncer.key_id("e2e-key")}
    result: dict = {}
    serving = threading.Thread(target=lambda: result.update(code=fleet_syncer.serve(
        args, f"127.0.0.1:{port}", 1, clusters=[], head_mode=True, local_syncer=syncer,
        sleep=lambda s: time.sleep(0.2), record_extra=extra)), daemon=True)
    serving.start()
    layout = build_layout([("model.embed.weight", DIM // 4), ("model.body.weight", DIM)], 2)
    target = torch.ones(DIM + DIM // 4)
    learners = []
    try:
        deadline = time.monotonic() + 60
        while island_launches.read_syncer_record(objects) is None:
            assert time.monotonic() < deadline, "the syncer was never recorded"
            time.sleep(0.1)
        first_record = island_launches.read_syncer_record(objects)
        assert first_record["service"] == "127.0.0.1" and first_record["syncer_epoch"] == 4
        learners.append(ToyLearner(0, port, target, layout, dtype=DTYPE_F32))
        learners[-1].start()
        while len(_steps(syncer.tape)) < 4:
            assert time.monotonic() < deadline, (tmp_path / "syncer.log").read_text()[-2000:]
            time.sleep(0.05)
        # kill the syncer process itself: one PID, checked to be the syncer
        victim = syncer.proc.pid
        assert "yeto-syncer" in _command_of(victim), _command_of(victim)
        import os

        os.kill(victim, signal.SIGKILL)
        killed_at = time.time()
        while syncer.proc.pid == victim or syncer.proc.poll() is not None:
            assert time.monotonic() < deadline + 30, "the service did not restart the syncer"
            time.sleep(0.05)
        resumed_from = read_checkpoint_global_step(syncer.checkpoint)
        before = _steps(syncer.tape)
        assert resumed_from >= 4 and before[:resumed_from] == list(range(1, resumed_from + 1))
        while not any(m.get("at", 0) > killed_at for m in [island_launches.read_syncer_record(objects)]):
            assert time.monotonic() < deadline + 60, "the record was not written again after the restart"
            time.sleep(0.1)
        again = island_launches.read_syncer_record(objects)
        assert again["addr"] == first_record["addr"] and again["syncer_epoch"] == 4 and again["service"] == "127.0.0.1"
        # the island redials the same address (its client re-JOINs); one whose redials ran out starts again
        learners[0].join(timeout=60)
        if learners[0].final_manifest is None:
            learners.append(ToyLearner(0, port, target, layout, dtype=DTYPE_F32))
            learners[-1].start()
            learners[-1].join(timeout=60)
        serving.join(timeout=120)
        assert not serving.is_alive(), (tmp_path / "syncer.log").read_text()[-3000:]
        assert result["code"] == 0  # the run finished
        done = [learner for learner in learners if learner.final_manifest is not None]
        assert len(done) == 1 and done[0].final_manifest.global_step == total
        log = (tmp_path / "syncer.log").read_text()
        assert "resumed from checkpoint" in log
        steps = _steps(syncer.tape)
        after = steps[len(before):]
        # the merge goes on from the checkpoint: the next step, every step once, to the last
        assert after[0] == resumed_from + 1 and after == list(range(resumed_from + 1, total + 1))
        assert read_checkpoint_global_step(syncer.checkpoint) == total
    finally:
        syncer.stop()
        for learner in learners:
            learner.join(timeout=5)
