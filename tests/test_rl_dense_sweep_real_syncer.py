"""Dense strict sweep against the real Rust syncer (P0-1.7b.1, Q7 item 4).

The syncer runs the exact actor command the launcher generates for
``--rl-sync-preset dense-full``; the client binds the profile hash the
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
from yeto.protocol import DTYPE_F32, SyncerClient

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


def _dense_args():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "dense-full", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", str(FRAGMENTS), "--total-steps", "1"), gpu="modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _start_syncer(binary: Path, tmp_path: Path) -> tuple[subprocess.Popen, int, str]:
    """The launcher's dense actor syncer, relocated to a free port and tmp_path."""
    from yeto import launcher

    args = _dense_args()
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


def _client(port: int, profile_hash: str) -> SyncerClient:
    return SyncerClient(("127.0.0.1", port), 0, _layout(), dtype=DTYPE_F32, num_streams=1,
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
