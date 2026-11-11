"""A syncer killed with SIGKILL and restarted with ``--resume`` keeps merging
for the islands that stayed alive. Real Rust syncer, real full-parameter
islands (the verl CPU harness: engine role stream, lazy capture, background
pushes, the paired WAN pipeline), no Ray.

The islands keep their processes across the syncer restart: each redials the
same address, the resumed syncer re-sends its cut and the PULLs of its open
rounds, and the rounds after the checkpointed step close and reach the final
cut. A push queued just before the kill dies with the old syncer; its island
must answer the resumed syncer's PULL of that round by replaying the retained
cut, also when that cut is no longer the island's newest (lazy) capture.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time

import pytest

from process_groups import group_popen, kill_group
from test_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    SHAPES, _layout, _plan, _run_island, needs_cargo, syncer_binary,
)
from test_rl_algorithm_provenance import _launcher_args
from yeto.final_marker import read_checkpoint_global_step
from yeto.rl.engine.full_parameter_stream import LAZY_CAPTURE_ENV

ROUNDS = 6


def _syncer_command(binary, tmp_path, *, fragments: int, rounds: int, port: int) -> tuple[str, str]:
    """The launcher's async syncer command (it adds ``--resume`` once the
    checkpoint exists, as the syncer wrapper does after a crash)."""
    from yeto import launcher
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", str(fragments), "--total-steps", str(rounds),
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download"),
        gpu="modal:2xh100,modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    command = launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"]
    command = (command.replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    assert "--resume" in command and "--checkpoint-every 1" in command
    return command, launcher.rl_syncer_profile(args, 2).sha256


def _start(command: str, log):
    return group_popen(["bash", "-c", command], stdout=log, stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "info")})


@needs_cargo
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy-capture", "whole-capture"])
def test_live_islands_keep_merging_after_a_kill_9_and_resume(syncer_binary, tmp_path, monkeypatch, lazy):
    from free_ports import free_port

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "kill-resume-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    monkeypatch.setenv(LAZY_CAPTURE_ENV, "1" if lazy else "0")
    fragments = len(_layout().fragments.fragments)
    total = ROUNDS * fragments
    port = free_port()
    command, profile = _syncer_command(syncer_binary, tmp_path, fragments=fragments, rounds=ROUNDS, port=port)
    checkpoint = tmp_path / "state.ckpt"
    log1, log2 = (tmp_path / "syncer-1.log").open("w"), (tmp_path / "syncer-2.log").open("w")
    first = _start(command, log1)
    second = None
    results: dict = {}
    threads = [threading.Thread(target=_run_island, args=(
        i, _plan(i, port, profile, rounds=ROUNDS, fragments=fragments, async_rounds=True,
                 matrix_merge="rda", pipeline=2, final_cut="download"), None, results), daemon=True)
        for i in range(2)]
    try:
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 60
        while not (checkpoint.exists() and read_checkpoint_global_step(checkpoint) >= fragments + 1):
            assert time.monotonic() < deadline, "no merges before the kill"
            assert first.poll() is None, (tmp_path / "syncer-1.log").read_text()[-3000:]
            time.sleep(0.01)
        os.killpg(first.pid, signal.SIGKILL)  # kill -9: no stop checkpoint, no goodbye
        first.wait(timeout=10)
        resumed_at = read_checkpoint_global_step(checkpoint)
        assert resumed_at < total
        time.sleep(1.0)  # the islands train on and push into the dead link
        second = _start(command, log2)
        for thread in threads:
            thread.join(timeout=150)
        try:
            code = second.wait(timeout=30)
        except subprocess.TimeoutExpired:
            code = None  # reported after the islands' own errors
    finally:
        for process in (first, second):
            if process is not None:
                kill_group(process)
        log1.close()
        log2.close()
    log = (tmp_path / "syncer-2.log").read_text()
    assert f"resumed from checkpoint step={resumed_at}" in log
    for learner_id in range(2):
        result = results.get(learner_id)
        assert result is not None and not isinstance(result, BaseException), (learner_id, result, log[-3000:])
        acks = [fields for event, fields in result["events"] if event == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}], (learner_id, log[-3000:])
    assert code == 0, log[-3000:]
    assert read_checkpoint_global_step(checkpoint) == total
    for name in SHAPES:  # both downloaded the same final cut
        assert results[0]["values"][name].equal(results[1]["values"][name]), name
