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
from yeto.protocol import (
    DTYPE_F32,
    DTYPE_F32_PUSH_BF16,
    DTYPE_F32_PUSH_I8,
    DTYPE_FLAG_BCAST_DELTA,
    SyncerClient,
)

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


def _client(port: int, profile_hash: str, dtype: int = DTYPE_F32, *,
            max_reconnects: int = 0, push_i8_block: int = 128) -> SyncerClient:
    return SyncerClient(("127.0.0.1", port), 0, _layout(), dtype=dtype, num_streams=1,
                        connect_timeout=10, max_reconnects=max_reconnects, finalization_timeout=20,
                        syncer_profile_hash=bytes.fromhex(profile_hash), push_i8_block=push_i8_block)


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
                dtype: int = DTYPE_F32, *, after_sweep=None, max_reconnects: int = 0,
                push_i8_block: int = 128, updates: dict | None = None, sink=None):
    """Drive ``sweeps`` strict sweeps; return (max PULLs seen at once, finals, checkpoint).

    ``updates`` collects every BCAST the client accepted, as
    ``{(fragment_id, version): f32 bytes}``; ``after_sweep(client, sweep)``
    runs once all pushes of a sweep are queued."""
    process, port, profile_hash = _start_syncer(binary, tmp_path, *extra, total_steps=sweeps)
    client = _client(port, profile_hash, dtype, max_reconnects=max_reconnects,
                     push_i8_block=push_i8_block)
    if sink is not None:
        client.install_inbound_chunk_sink(sink)
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
            if after_sweep is not None:
                after_sweep(client, sweep)
        manifest, finals = client.wait_for_final_fragments(timeout=30)
        if updates is not None:
            updates.update({(u.fragment_id, u.version): bytes(u.data) for u in client.drain_updates()})
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


def _tail(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.float32)


def test_int8_pushes_stay_within_their_block_error_of_the_f32_sweep(syncer_binary, tmp_path):
    """E3: int8 outer deltas change results only by half a scale step per delta."""
    exact_dir, int8_dir = tmp_path / "f32", tmp_path / "int8"
    exact_dir.mkdir()
    int8_dir.mkdir()
    exact = _run_sweeps(syncer_binary, exact_dir, STREAMED, sweeps=3)
    quantized = _run_sweeps(syncer_binary, int8_dir,
                            (*STREAMED, "--rl-sync-push-dtype", "int8", "--rl-sync-push-int8-block", "32"),
                            sweeps=3, dtype=DTYPE_F32_PUSH_I8, push_i8_block=32)
    assert exact[1] == quantized[1] == [5, 6]
    for fid, data in exact[2].items():
        want, got = _tail(data)[-NUMEL:], _tail(quantized[2][fid])[-NUMEL:]
        # Three sweeps; each delta's absmax < 2, so error <= 3 * 2 / 254.
        assert torch.allclose(got, want, rtol=0, atol=3 * 2.0 / 254), fid
        assert not torch.equal(got, want), "int8 pushes must actually quantize"


def _downlink_run(binary, tmp_path, full_every: int, sweeps: int, **kwargs):
    updates: dict = {}
    result = _run_sweeps(binary, tmp_path,
                         (*STREAMED, "--rl-sync-bcast-delta-full-every", str(full_every)),
                         sweeps=sweeps, dtype=DTYPE_F32 | DTYPE_FLAG_BCAST_DELTA,
                         updates=updates, **kwargs)
    return result, updates


def test_bf16_downlink_tracks_the_exact_broadcasts_and_leaves_the_merge_exact(syncer_binary, tmp_path):
    """E3: BCAST deltas change only what islands hold between f32 refreshes;
    the syncer's masters, versions, FINAL and checkpoint stay bit-identical."""
    exact_dir, delta_dir = tmp_path / "f32", tmp_path / "delta"
    exact_dir.mkdir()
    delta_dir.mkdir()
    exact_updates: dict = {}
    exact = _run_sweeps(syncer_binary, exact_dir, STREAMED, sweeps=4, updates=exact_updates)
    delta, delta_updates = _downlink_run(syncer_binary, delta_dir, 3, sweeps=4)
    assert exact[1] == delta[1] == [7, 8]
    assert exact[2] == delta[2], "FINAL must be the exact f32 masters"
    assert exact[3] == delta[3], "checkpoints must not depend on the downlink"
    assert set(exact_updates) == set(delta_updates)
    # Fragment 0 publishes versions 1, 3, 5, 7: FULL, DELTA, DELTA, FULL (M=3).
    kinds = {}
    for key, data in sorted(delta_updates.items()):
        want, got = _tail(exact_updates[key]), _tail(data)
        kinds[key] = "full" if torch.equal(got, want) else "delta"
        # one bf16 rounding of (theta - anchor), |theta - anchor| < 4 here
        assert torch.allclose(got, want, rtol=0, atol=4.0 / 256), key
    assert [kinds[(0, v)] for v in (0, 1, 3, 5, 7)] == ["full", "full", "delta", "delta", "full"]
    assert [kinds[(1, v)] for v in (0, 2, 4, 6)] == ["full", "full", "delta", "delta"]


def test_bf16_downlink_anchor_mismatch_reconnects_and_resyncs_with_f32(syncer_binary, tmp_path):
    """E3 fallback: a learner whose anchor drifted fails the digest check,
    reconnects, receives the shared anchor as FULL, and the run finishes
    with the same exact result."""
    exact_dir, delta_dir = tmp_path / "f32", tmp_path / "delta"
    exact_dir.mkdir()
    delta_dir.mkdir()
    exact = _run_sweeps(syncer_binary, exact_dir, STREAMED, sweeps=4)
    corrupted, reconnects = [], []

    def corrupt(client, sweep):
        if sweep == 4:
            reconnects.append(client._reconnects_used)
        if sweep != 1:
            return
        _wait(lambda: client._downlink.version(0) == 1, "fragment 0 version 1 never arrived")
        held = client._downlink._held[0]
        held.values = held.values.copy()
        held.values[0] += 1e-3
        corrupted.append(client.connection_generation)

    delta, _ = _downlink_run(syncer_binary, delta_dir, 100, sweeps=4, after_sweep=corrupt,
                             max_reconnects=5)
    assert corrupted and reconnects and reconnects[0] >= 1, "the digest mismatch must reconnect"
    assert exact[1] == delta[1] and exact[2] == delta[2] and exact[3] == delta[3]


def test_bf16_downlink_and_int8_flags_reach_syncer_and_island_commands():
    from yeto import launcher
    from yeto.launch.rl_island_task import _rl_trainable_island_flags

    args = _dense_args("--rl-sync-push-dtype", "int8", "--rl-sync-push-int8-block", "64",
                       "--rl-sync-bcast-delta-full-every", "4")
    island = _rl_trainable_island_flags(args, 0, 1)
    assert "--sweep-push-dtype int8" in island and "--sweep-push-int8-block 64" in island
    command = launcher.role_syncer_commands(args, 1, binary="yeto-syncer")["actor"]
    assert "--bcast-delta-full-every 4" in command
    plain = _dense_args()
    assert "--bcast-delta" not in launcher.role_syncer_commands(plain, 1, binary="yeto-syncer")["actor"]
    with pytest.raises(ValueError, match="--rl-sync-bcast-delta-full-every 0"):
        _dense_args("--rl-sync-bcast-delta-full-every", "1")
    with pytest.raises(ValueError, match="--rl-sync-push-int8-block only with"):
        _dense_args("--rl-sync-push-int8-block", "64")


class _BytesSink:
    """Minimal transactional inbound sink: assembles the f32 fragment bytes."""

    def begin_message(self, message_id, total_bytes):
        return {"buf": bytearray(), "bound": None}

    def bind_fragment(self, tx, msg_type, fragment_id, version, payload_bytes):
        tx["bound"] = payload_bytes

    def stage_chunk(self, tx, payload):
        return bytes(payload)

    def consume_staged_chunk(self, tx, offset, token, nbytes):
        self.consume_chunk(tx, offset, token)

    def consume_chunk(self, tx, offset, payload):
        assert offset == len(tx["buf"])
        tx["buf"] += bytes(payload)

    def finish_message(self, tx):
        import hashlib

        from yeto.protocol import StreamedInboundPayload

        assert len(tx["buf"]) == tx["bound"]
        return StreamedInboundPayload(bytes(tx["buf"]), hashlib.sha256(tx["buf"]).digest())

    def abort_message(self, tx, staged_tokens):
        tx["buf"].clear()


def test_bf16_downlink_through_a_streaming_sink_matches_the_buffered_path(syncer_binary, tmp_path):
    """E3: the Miles path streams BCAST chunks into a transactional sink; the
    downlink wrapper must hand it the same f32 anchor bytes."""
    buffered_dir, streamed_dir = tmp_path / "buffered", tmp_path / "streamed"
    buffered_dir.mkdir()
    streamed_dir.mkdir()
    buffered, buffered_updates = _downlink_run(syncer_binary, buffered_dir, 3, sweeps=3)
    streamed, streamed_updates = _downlink_run(syncer_binary, streamed_dir, 3, sweeps=3,
                                               sink=_BytesSink())
    assert buffered[2] == streamed[2]
    assert buffered_updates == streamed_updates


def _overlap_island_run(binary: Path, tmp_path: Path, extra: tuple[str, ...], dtype: int,
                        sweeps: int, progress: dict[tuple[int, int], list[float]]):
    """One overlap island (E3 item 3) against the real syncer.

    Round s trains before sweep s-1 comes back; at boundary s the island waits
    for G_{s-1}, rebases ``theta <- G_{s-1} + (theta - base)`` and pushes
    ``theta - G_{s-1}``. Returns (FINAL bytes, the island's anchors per sweep,
    redials used)."""
    process, port, profile_hash = _start_syncer(binary, tmp_path, *extra, total_steps=sweeps)
    client = _client(port, profile_hash, dtype)
    init = {fid: torch.tensor([float(fid) + 0.125 * i for i in range(NUMEL)]) for fid in range(FRAGMENTS)}
    anchors: dict[int, dict[int, torch.Tensor]] = {}
    held: dict[tuple[int, int], torch.Tensor] = {}
    try:
        client.start()
        for fid in range(FRAGMENTS):
            client.send_init(fid, _f32(init[fid].tolist()))
        anchor = dict(init)  # what the last push was measured against
        theta = {fid: init[fid] + torch.tensor(progress[(1, fid)]) for fid in range(FRAGMENTS)}
        for sweep in range(1, sweeps + 1):
            if sweep > 1:
                # Round `sweep` trained from `base` while sweep-1 was in flight.
                base = dict(theta)
                theta = {fid: base[fid] + torch.tensor(progress[(sweep, fid)]) for fid in range(FRAGMENTS)}

                def got_sweep():
                    client.check_health()
                    held.update({(u.fragment_id, u.version): _tail(bytes(u.data))
                                 for u in client.drain_updates()})
                    return all((fid, (sweep - 2) * FRAGMENTS + fid + 1) in held for fid in range(FRAGMENTS))

                _wait(got_sweep, f"sweep {sweep - 1} never came back")
                anchor = {fid: held[(fid, (sweep - 2) * FRAGMENTS + fid + 1)] for fid in range(FRAGMENTS)}
                anchors[sweep - 1] = anchor
                theta = {fid: anchor[fid] + (theta[fid] - base[fid]) for fid in range(FRAGMENTS)}
            permits: dict[int, object] = {}

            def gather():
                client.check_health()
                permits.update({p.fragment_id: p for p in client.drain_pulls()})
                return len(permits) == FRAGMENTS

            _wait(gather, f"no PULLs for sweep {sweep}")
            for fid, permit in sorted(permits.items()):
                base_version = 0 if sweep == 1 else (sweep - 2) * FRAGMENTS + fid + 1
                client.push_fragment(fid, permit.global_step, permit.round_attempt, base_version,
                                     sweep, 1, 7, _f32((theta[fid] - anchor[fid]).tolist()))
        manifest, finals = client.wait_for_final_fragments(timeout=30)
        result = {f.fragment_id: _tail(bytes(f.data)) for f in finals}
        client.acknowledge_finalization(manifest)
        redials = client._reconnects_used
        assert process.wait(timeout=15) == 0
    finally:
        client.close()
        log = _stop(process)
    assert "syncer semantic profile" not in log, log
    return init, result, anchors, redials


def test_overlap_with_bf16_downlink_counts_each_round_once_without_a_resync(syncer_binary, tmp_path):
    """E3: overlap + bf16 BCAST deltas. The island rebases on a bf16-tracked
    anchor, but it pushes ``theta - anchor`` against that same anchor, so the
    syncer's f32 masters still get every round's progress exactly once; the
    downlink error stays out of the masters and never triggers a resync."""
    sweeps = 4
    progress = {(s, fid): [0.5 + fid + 0.03125 * s + 0.001 * i for i in range(NUMEL)]
                for s in range(1, sweeps + 1) for fid in range(FRAGMENTS)}
    exact_dir, delta_dir = tmp_path / "f32", tmp_path / "delta"
    exact_dir.mkdir()
    delta_dir.mkdir()
    init, exact, exact_anchors, exact_redials = _overlap_island_run(
        syncer_binary, exact_dir, STREAMED, DTYPE_F32, sweeps, progress)
    _, delta, delta_anchors, delta_redials = _overlap_island_run(
        syncer_binary, delta_dir, (*STREAMED, "--rl-sync-bcast-delta-full-every", "3"),
        DTYPE_F32 | DTYPE_FLAG_BCAST_DELTA, sweeps, progress)
    assert exact_redials == delta_redials == 0, "no DownlinkResync expected"
    for fid in range(FRAGMENTS):
        want = init[fid] + sum(torch.tensor(progress[(s, fid)]) for s in range(1, sweeps + 1))
        # (anchor + p) - anchor in f32 is p up to an ulp of the anchor: no bf16 error.
        assert torch.allclose(exact[fid], want, rtol=0, atol=1e-5), fid
        assert torch.allclose(delta[fid], want, rtol=0, atol=1e-5), fid
        for sweep, anchors in exact_anchors.items():
            got = delta_anchors[sweep][fid]
            # one bf16 rounding of (G - previous anchor); |G - previous anchor| < 4
            assert torch.allclose(got, anchors[fid], rtol=0, atol=4.0 / 256), (sweep, fid)
    assert any(not torch.equal(delta_anchors[s][f], exact_anchors[s][f])
               for s in delta_anchors for f in range(FRAGMENTS)), "the downlink must actually round"
