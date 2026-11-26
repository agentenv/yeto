"""The on-site syncer service with the real Rust syncer: the syncer process is killed while
a learner merges, the service restarts it (as ``yeto fleet service-run --role syncer``
does: ``LocalSyncer`` under :func:`yeto.launch.fleet_syncer.serve`), the restarted syncer
resumes from its checkpoint on the same address, its record is written again, and the
merge goes on to the run's last step: every outer step once, in order, none skipped. CPU
only; a port of 8880-8899."""

from __future__ import annotations

import json
import pathlib
import signal
import socket
import subprocess
import threading
import time
import types

import pytest
import torch

from test_fleet_service import _env  # noqa: F401 - autouse: fake secrets, env restored
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

    def start_store_upload(self, args, objects=None) -> None:  # no run store tape upload here (a local store)
        return None

    def start_log_forwarder(self) -> None:
        return None


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


def _argv_of(pid: int) -> list[str]:
    """The process's argv, whole: ``/proc/<pid>/cmdline`` (NUL-separated), else ``ps -ww``
    (no column cut; split at whitespace, which the test's paths never hold)."""
    import os

    proc = f"/proc/{pid}/cmdline"
    if os.path.exists(proc):
        with open(proc, "rb") as handle:
            return [a.decode() for a in handle.read().split(b"\0") if a]
    out = subprocess.run(["ps", "-ww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True).stdout
    return out.split()


def _is_binary(pid: int, binary) -> bool:
    """``pid`` runs ``binary`` (its argv[0] is that file, by realpath)."""
    import os

    argv = _argv_of(pid)
    return bool(argv) and os.path.realpath(argv[0]) == os.path.realpath(str(binary))


def _dead(pid: int) -> bool:
    """``pid`` is gone or a zombie (it writes nothing more)."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return not out or out.startswith("Z")


def _long_path_copy(binary, tmp_path):
    """A copy of the syncer binary under a path longer than any ``ps`` column (CI's runner
    paths are: a cut command line must not hide which process this is)."""
    import os
    import shutil

    where = tmp_path / ("long-" + "d" * 60) / ("e" * 60) / ("f" * 60) / ("g" * 60)
    where.mkdir(parents=True)
    copy = where / pathlib.Path(str(binary)).name
    shutil.copyfile(binary, copy)  # the bytes only: a system binary's file flags are not copied
    os.chmod(copy, 0o755)
    assert len(str(copy)) > 260
    return copy


def test_the_process_check_reads_the_whole_command_line(tmp_path):
    """A process whose binary sits under a long path is still told by its argv[0] (no cut)."""
    import os
    import shutil

    sleeper = _long_path_copy(shutil.which("sleep"), tmp_path)
    proc = subprocess.Popen([str(sleeper), "30"])
    try:
        assert _is_binary(proc.pid, sleeper), _argv_of(proc.pid)
        assert _argv_of(proc.pid)[0] == str(sleeper) and len(_argv_of(proc.pid)[0]) > 260
        assert not _is_binary(proc.pid, shutil.which("sleep") + "-other")
        assert not _is_binary(os.getpid(), sleeper)
    finally:
        proc.kill()
        proc.wait()


@needs_cargo
@pytest.mark.parametrize("long_path", [False, True], ids=["short-path", "long-path"])
def test_a_killed_syncer_is_restarted_from_its_checkpoint_and_the_merge_goes_on(  # noqa: PLR0915
        syncer_binary, tmp_path, monkeypatch, long_path):  # noqa: F811
    from yeto import runs

    if long_path:  # the binary under a path as long as a CI runner's, or longer
        syncer_binary = _long_path_copy(syncer_binary, tmp_path)

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
        assert _is_binary(victim, syncer_binary), _argv_of(victim)
        import os

        os.kill(victim, signal.SIGKILL)
        killed_at = time.time()
        while not _dead(victim):  # what it left is read before its restart can merge on
            assert time.monotonic() < deadline + 30, "the syncer did not die"
            time.sleep(0.01)
        resumed_from = read_checkpoint_global_step(syncer.checkpoint)
        before = _steps(syncer.tape)
        while syncer.proc.pid == victim or syncer.proc.poll() is not None:
            assert time.monotonic() < deadline + 30, "the service did not restart the syncer"
            time.sleep(0.05)
        # the syncer writes a step's checkpoint before its tape line (the line carries the checkpoint's
        # time): a SIGKILL between the two leaves the checkpoint one step past the tape
        assert before == list(range(1, len(before) + 1)) and resumed_from in (len(before), len(before) + 1)
        assert resumed_from >= 4
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


@needs_cargo
def test_the_whole_service_process_stopped_and_started_again_goes_on_at_the_same_term(  # noqa: PLR0915
        syncer_binary, tmp_path, monkeypatch):  # noqa: F811
    """``yeto fleet service-run --role syncer`` as the unit runs it, twice on one host: the
    first service process ends on SIGTERM (a stop or restart of the unit), the second
    resumes the syncer from its checkpoint, at the same term, and the merge goes on to the
    run's last step: every outer step once, in order, none merged twice."""
    import os

    from test_fleet_island_launches import _prepared
    from test_fleet_service import onsite_config
    from yeto.rl import fleet_service as fs
    from yeto.rl import up

    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "service-restart-e2e-key")
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    port, total = _port(), 14
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(syncer_mod, "_head_ports", lambda args: [port])
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects)
    # this syncer's checkpoint (ServiceSyncer keeps it in tmp_path): the restart finds it there
    monkeypatch.setattr(fs, "_checkpoints", lambda args, home: [str(tmp_path / "state.ckpt")])
    # the fleet's last record is another host's syncer's (term 3): this service starts at 4
    island_launches.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=3, learners=1, max_learners=4,
                                        run="r", extra={"service": "10.0.0.9", "service_node": "site-x"})
    config = str(onsite_config(tmp_path, host="127.0.0.1"))
    layout = build_layout([("model.embed.weight", DIM // 4), ("model.body.weight", DIM)], 2)
    target = torch.ones(DIM + DIM // 4)
    learners: list = []
    records: list[dict] = []
    logs: list[str] = []
    done = threading.Event()

    def service(**kw):
        return fs.run_syncer(config, objects=objects, parse=_prepared, node="site-s", log=logs.append,
                             workdir=str(tmp_path / "work"),
                             make_syncer=lambda args, n: ServiceSyncer(syncer_binary, tmp_path, port, total), **kw)

    def keep_learning():
        """The island: it dials the syncer once it is recorded, redials it across the restart,
        and one whose redials ran out starts again (as its launch would relaunch it)."""
        while island_launches.read_syncer_record(objects) is None or \
                island_launches.read_syncer_record(objects).get("service") != "127.0.0.1":
            if done.wait(0.1):
                return
        while not done.is_set():
            if not learners or (not learners[-1].is_alive() and learners[-1].final_manifest is None):
                learners.append(ToyLearner(0, port, target, layout, dtype=DTYPE_F32))
                learners[-1].start()
            if learners[-1].final_manifest is not None:
                return
            done.wait(0.2)

    def stop_service_at(steps: int):
        deadline = time.monotonic() + 90
        while len(_steps(tmp_path / "tape.jsonl")) < steps and time.monotonic() < deadline:
            time.sleep(0.05)
        records.append(island_launches.read_syncer_record(objects))
        os.kill(os.getpid(), signal.SIGTERM)  # this process: the service, as systemd stops the unit

    island = threading.Thread(target=keep_learning, daemon=True)
    island.start()
    previous = signal.signal(signal.SIGTERM, fs._exit_on_sigterm)
    try:
        threading.Thread(target=stop_service_at, args=(4,), daemon=True).start()
        with pytest.raises(SystemExit) as stopped:
            service()
        assert stopped.value.code == 128 + signal.SIGTERM
        signal.signal(signal.SIGTERM, previous)
        before = _steps(tmp_path / "tape.jsonl")
        resumed_from = read_checkpoint_global_step(tmp_path / "state.ckpt")
        assert resumed_from >= 4 and before == list(range(1, len(before) + 1))
        assert fleet_syncer.read_service_status(objects)["state"] == "running"  # a stop of the unit: no end
        assert service() == 0  # started again: resumes, merges to the end, the run is finished
    finally:
        signal.signal(signal.SIGTERM, previous)
        done.set()
        island.join(timeout=30)
        for learner in learners:
            learner.join(timeout=5)
    (first,) = records
    assert first["syncer_epoch"] == 4 and first["service_generation"] == 1
    # the second service: the same host, so the same term; a new lease token (its generation) fences the first
    runs_at = [m for m in logs if "runs on this host at" in m]
    assert len(runs_at) == 2 and "(epoch 4, generation 1," in runs_at[0] and "(epoch 4, generation 2," in runs_at[1]
    assert "resumed from checkpoint" in (tmp_path / "syncer.log").read_text()
    steps = _steps(tmp_path / "tape.jsonl")
    assert steps == list(range(1, total + 1))  # every outer step once, in order: none skipped, none merged twice
    assert steps[len(before)] == resumed_from + 1
    assert read_checkpoint_global_step(tmp_path / "state.ckpt") == total
    assert fleet_syncer.read_service_status(objects)["state"] == "finished"
    finals = [learner for learner in learners if learner.final_manifest is not None]
    assert len(finals) == 1 and finals[0].final_manifest.global_step == total
