"""Dense strict sweep against the real Rust syncer (P0-1.7b.1, Q7 item 4).

The syncer runs the exact actor command the launcher generates for
``--rl-sync-preset full-parameter``; the client binds the profile hash the
launcher sends islands. One learner sweeps P=2 fragments once: the syncer
must accept that HELLO, merge the sweep and finalize ``init + delta``; a
HELLO with any other profile hash must be refused.
"""

from __future__ import annotations

import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, DTYPE_F32_PUSH_BF16, SyncerClient

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
FRAGMENTS = 2
NUMEL = 3
pytestmark = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "dense-sweep-test-key")


def _dense_args(*extra: str, total_steps: int = 1):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", str(FRAGMENTS), "--total-steps", str(total_steps), *extra),
        gpu="modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _start_syncer(binary: Path, tmp_path: Path, *extra: str,
                  total_steps: int = 1) -> tuple[subprocess.Popen, int, str]:
    """The launcher's dense actor syncer, relocated to a free port and tmp_path."""
    from yeto import launcher

    args = _dense_args(*extra, total_steps=total_steps)
    command = launcher.role_syncer_commands(args, 1, binary=str(binary))["actor"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = (command
               .replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    assert "--policy-sweep-fragments 2" in command and "--require-profile-binding" in command
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    return process, port, launcher.rl_syncer_profile(args, 1).sha256


def _layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_RDA, [(f"actor.w{i}", NUMEL)]) for i in range(FRAGMENTS)])


def _client(port: int, profile_hash: str, dtype: int = DTYPE_F32) -> SyncerClient:
    return SyncerClient(("127.0.0.1", port), 0, _layout(), dtype=dtype, num_streams=1,
                        connect_timeout=10, max_reconnects=0, finalization_timeout=20,
                        syncer_profile_hash=bytes.fromhex(profile_hash))


def _f32(values: list[float]) -> bytes:
    return torch.tensor(values, dtype=torch.float32).numpy().tobytes()


def _stop(process: subprocess.Popen) -> str:
    if process.poll() is None:
        process.terminate()
    out, _ = process.communicate(timeout=10)
    return out


def _wait(predicate, what: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError(what)


def test_launcher_dense_syncer_runs_one_strict_sweep(syncer_binary, tmp_path):
    process, port, profile_hash = _start_syncer(syncer_binary, tmp_path)
    client = _client(port, profile_hash)
    initial = {fid: [float(fid)] * NUMEL for fid in range(FRAGMENTS)}
    delta = {fid: [0.5 + fid] * NUMEL for fid in range(FRAGMENTS)}
    try:
        client.start()
        for fid in range(FRAGMENTS):
            client.send_init(fid, _f32(initial[fid]))
        pushed: list[int] = []

        def next_pull():
            client.check_health()
            return next(iter(client.drain_pulls()), None)

        # Serial rounds (pipeline 1): one PULL per fragment of the local step,
        # every push against the same anchor (base version 0, local step 1).
        while len(pushed) < FRAGMENTS:
            permit = _wait(next_pull, f"syncer did not permit fragment {len(pushed)}")
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 0, 1, 1, 1, _f32(delta[permit.fragment_id]))
            pushed.append(permit.fragment_id)
        assert pushed == list(range(FRAGMENTS))
        manifest, finals = client.wait_for_final_fragments(timeout=20)
        assert list(manifest.versions) == [1, 2]
        for fragment in finals:
            merged = torch.frombuffer(bytearray(fragment.data), dtype=torch.float32)
            expected = torch.tensor(initial[fragment.fragment_id]) + torch.tensor(
                delta[fragment.fragment_id])
            assert torch.equal(merged[-NUMEL:], expected), fragment.fragment_id
        client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        client.close()
        log = _stop(process)
    assert "syncer semantic profile" not in log


def test_launcher_dense_syncer_refuses_another_profile_hash(syncer_binary, tmp_path):
    process, port, profile_hash = _start_syncer(syncer_binary, tmp_path)
    wrong = ("0" if profile_hash[0] != "0" else "1") + profile_hash[1:]
    client = _client(port, wrong)
    try:
        with pytest.raises(Exception):
            client.start()
            _wait(lambda: client.check_health(), "client stayed healthy", timeout=10)
    finally:
        client.close()
        log = _stop(process)
    assert "syncer semantic profile does not match" in log


STREAMED = ("--rl-sync-fragment-pipeline", str(FRAGMENTS), "--rl-sync-async-checkpoint")


def _run_sweeps(binary: Path, tmp_path: Path, extra: tuple[str, ...], sweeps: int,
                dtype: int = DTYPE_F32):
    """Drive ``sweeps`` strict sweeps; return (max PULLs seen at once, finals, checkpoint)."""
    process, port, profile_hash = _start_syncer(binary, tmp_path, *extra, total_steps=sweeps)
    client = _client(port, profile_hash, dtype)
    concurrent = 0
    try:
        client.start()
        for fid in range(FRAGMENTS):
            client.send_init(fid, _f32([float(fid) + 0.125 * i for i in range(NUMEL)]))
        for sweep in range(1, sweeps + 1):
            permits: dict[int, object] = {}

            def gather():
                client.check_health()
                for permit in client.drain_pulls():
                    permits[permit.fragment_id] = permit
                return any(not getattr(p, "pushed", False) for p in permits.values())

            while len(permits) < FRAGMENTS:
                _wait(gather, f"no PULL in sweep {sweep}")
                time.sleep(0.2)  # let every PULL the profile permits arrive
                gather()
                concurrent = max(concurrent, sum(not getattr(p, "pushed", False)
                                                 for p in permits.values()))
                # Push only what is permitted now; the serial profile then
                # releases the next fragment's PULL after this commit.
                for fid, permit in sorted(permits.items()):
                    if getattr(permit, "pushed", False):
                        continue
                    base = 0 if sweep == 1 else (sweep - 2) * FRAGMENTS + fid + 1
                    delta = [0.5 + fid + 0.03125 * sweep + 0.001 * i for i in range(NUMEL)]
                    client.push_fragment(fid, permit.global_step, permit.round_attempt,
                                         base, sweep, 1, 7, _f32(delta))
                    permit.pushed = True
        manifest, finals = client.wait_for_final_fragments(timeout=30)
        result = {f.fragment_id: bytes(f.data) for f in finals}
        client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        client.close()
        log = _stop(process)
    assert "syncer semantic profile" not in log, log
    return concurrent, list(manifest.versions), result, (tmp_path / "state.ckpt").read_bytes(), log


def test_streamed_sweep_matches_the_serial_sweep_bit_for_bit(syncer_binary, tmp_path):
    """E3: fragment pipelining + async checkpoint changes timing, not results."""
    serial_dir, streamed_dir = tmp_path / "serial", tmp_path / "streamed"
    serial_dir.mkdir()
    streamed_dir.mkdir()
    serial = _run_sweeps(syncer_binary, serial_dir, (), sweeps=3)
    streamed = _run_sweeps(syncer_binary, streamed_dir, STREAMED, sweeps=3)
    assert serial[0] == 1, "the serial profile must permit one fragment at a time"
    assert streamed[0] == FRAGMENTS, "the streamed profile must permit a whole sweep at once"
    assert serial[1] == streamed[1] == [5, 6]
    assert serial[2] == streamed[2]
    assert serial[3] == streamed[3]
    assert "--async-checkpoint" not in serial[4]


def test_streamed_flags_bind_the_profile_and_are_full_parameter_only():
    serial = _dense_args()
    streamed = _dense_args(*STREAMED)
    assert serial.pipeline == 1 and streamed.pipeline == FRAGMENTS
    from yeto import launcher

    command = launcher.role_syncer_commands(streamed, 1, binary="yeto-syncer")["actor"]
    assert f"--pipeline {FRAGMENTS}" in command and "--async-checkpoint" in command
    assert (launcher.rl_syncer_profile(serial, 1).sha256
            != launcher.rl_syncer_profile(streamed, 1).sha256)
    with pytest.raises(ValueError, match="--rl-sync-fragment-pipeline between 1 and --fragments"):
        _dense_args("--rl-sync-fragment-pipeline", str(FRAGMENTS + 1))


def test_bf16_pushes_stay_within_bf16_error_of_the_f32_sweep(syncer_binary, tmp_path):
    """E3: bf16 outer deltas change results only by bf16 rounding of each delta."""
    exact_dir, bf16_dir = tmp_path / "f32", tmp_path / "bf16"
    exact_dir.mkdir()
    bf16_dir.mkdir()
    exact = _run_sweeps(syncer_binary, exact_dir, STREAMED, sweeps=3)
    rounded = _run_sweeps(syncer_binary, bf16_dir, (*STREAMED, "--rl-sync-push-dtype", "bf16"),
                          sweeps=3, dtype=DTYPE_F32_PUSH_BF16)
    assert exact[1] == rounded[1] == [5, 6]  # versions are unchanged
    for fid, data in exact[2].items():
        want = torch.frombuffer(bytearray(data), dtype=torch.float32)[-NUMEL:]
        got = torch.frombuffer(bytearray(rounded[2][fid]), dtype=torch.float32)[-NUMEL:]
        # Three sweeps, each delta < 2 in magnitude, bf16 error <= 2^-8 relative.
        assert torch.allclose(got, want, rtol=0, atol=3 * 2.0 / 256), fid
        assert not torch.equal(got, want), "bf16 pushes must actually round"


def test_sweep_checkpoint_interval_reaches_the_syncer_and_its_profile():
    every = 2 * FRAGMENTS
    args = _dense_args("--rl-sync-checkpoint-every", str(every), total_steps=4)
    from yeto import launcher

    command = launcher.role_syncer_commands(args, 1, binary="yeto-syncer")["actor"]
    assert f"--checkpoint-every {every}" in command
    assert launcher.rl_syncer_profile(args, 1).checkpoint_every == every
    with pytest.raises(ValueError, match="--rl-sync-checkpoint-every 1 or a multiple"):
        _dense_args("--rl-sync-checkpoint-every", str(FRAGMENTS + 1))


def test_bf16_push_flag_reaches_the_island_command():
    from yeto.launch.rl_island_task import _rl_trainable_island_flags

    args = _dense_args("--rl-sync-push-dtype", "bf16")
    assert "--sweep-push-dtype bf16" in _rl_trainable_island_flags(args, 0, 1)
    assert "--sweep-push-dtype" not in _rl_trainable_island_flags(_dense_args(), 0, 1)
