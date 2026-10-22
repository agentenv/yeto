"""S26 E3 sync options: bf16 push transcoding, session binding, preset scoping. CPU only."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_rl_algorithm_provenance import (  # noqa: F401
    _fake_sky,
    _island_run,
    _launcher_args,
    _no_modal_listing,
)
from test_rl_dense_sweep_real_syncer import syncer_binary  # noqa: F401
from test_rl_full_parameter_sync import _role_stream_config

from yeto.downlink import (
    HEADER,
    KIND_DELTA,
    KIND_FULL,
    DownlinkAnchors,
    DownlinkChunkSink,
    DownlinkResync,
)
from yeto.protocol import (
    DTYPE_BF16,
    DTYPE_F32,
    DTYPE_F32_PUSH_BF16,
    DTYPE_F32_PUSH_I8,
    DTYPE_FLAG_BCAST_DELTA,
    _f32_parts_to_bf16,
    _f32_parts_to_i8,
    bulk_dtype,
    f32_bytes_to_bf16,
    f32_bytes_to_i8,
    i8_nbytes,
    push_wire_dtype,
    valid_session_dtype,
)


def test_push_bf16_session_keeps_bulk_payloads_f32():
    assert bulk_dtype(DTYPE_F32_PUSH_BF16) == DTYPE_F32
    assert push_wire_dtype(DTYPE_F32_PUSH_BF16) == DTYPE_BF16
    assert push_wire_dtype(DTYPE_F32) == DTYPE_F32


def test_f32_to_bf16_matches_torch_round_to_nearest_even_across_split_parts():
    values = torch.randn(50_001) * torch.logspace(-20, 20, 50_001)
    values[:3] = torch.tensor([0.0, -0.0, float("inf")])
    raw = values.numpy().tobytes()
    want = values.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes()
    assert f32_bytes_to_bf16(raw) == want
    parts = [raw[:5], raw[5:4099], raw[4099:4100], raw[4100:]]  # split mid-value
    assert b"".join(_f32_parts_to_bf16(parts)) == want
    with pytest.raises(ValueError, match="mid-value"):
        list(_f32_parts_to_bf16([raw[:6]]))


def test_push_dtype_is_bound_into_the_role_stream_session_contract():
    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        role_stream_session_contract_hash,
    )

    exact = _role_stream_config("actor", 30100, sweep=True)
    rounded = replace(exact, push_dtype="bf16")
    assert (role_stream_session_contract_hash("1" * 64, exact)
            != role_stream_session_contract_hash("1" * 64, rounded))


@pytest.mark.parametrize(("flag", "value"), (
    ("--rl-sync-fragment-pipeline", "2"),
    ("--rl-sync-push-dtype", "bf16"),
    ("--rl-sync-checkpoint-every", "4"),
))
def test_e3_options_only_apply_to_the_full_parameter_preset(flag, value):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (flag, value), gpu="modal:2xh100")
    args.controller = "local"
    with pytest.raises(ValueError, match=f"{flag} only applies to --rl-sync-preset full-parameter"):
        _prepare_rl_args(args)


# -- S26 E3 third batch: int8 pushes, bf16 downlink ---------------------------

def _decode_i8(wire: bytes, numel: int) -> np.ndarray:
    block = int(np.frombuffer(wire, "<u4", 1)[0])
    at, out = 4, []
    for start in range(0, numel, block):
        n = min(block, numel - start)
        scale = np.frombuffer(wire, "<f4", 1, at)[0]
        out.append(np.frombuffer(wire, np.int8, n, at + 4).astype(np.float32) * scale)
        at += 4 + n
    assert at == len(wire)
    return np.concatenate(out)


@pytest.mark.parametrize("block", (32, 128, 1000))
def test_int8_push_error_is_at_most_half_a_scale_step_per_block(block):
    rng = np.random.default_rng(0)
    values = (rng.standard_normal(10_007) * np.repeat(rng.uniform(1e-6, 3, 11), 10_007 // 11 + 1)[:10_007]
              ).astype(np.float32)
    values[:block] = 0.0  # an all-zero block
    wire = f32_bytes_to_i8(values.tobytes(), block)
    assert len(wire) == i8_nbytes(values.size, block)
    decoded = _decode_i8(wire, values.size)
    for start in range(0, values.size, block):
        chunk = values[start:start + block]
        bound = np.abs(chunk).max() / 254 * (1 + 1e-6)
        assert np.abs(decoded[start:start + block] - chunk).max() <= bound
    raw = values.tobytes()
    parts = [raw[:3], raw[3:4 * block + 5], raw[4 * block + 5:]]  # split mid-value and mid-block
    assert b"".join(_f32_parts_to_i8(parts, block)) == wire
    with pytest.raises(ValueError, match="mid-value"):
        list(_f32_parts_to_i8([raw[:6]], block))
    with pytest.raises(ValueError, match="block size"):
        f32_bytes_to_i8(raw, 16)


def test_session_dtype_flags_match_the_syncer():
    assert valid_session_dtype(DTYPE_F32_PUSH_I8 | DTYPE_FLAG_BCAST_DELTA)
    assert valid_session_dtype(DTYPE_F32 | DTYPE_FLAG_BCAST_DELTA)
    assert not valid_session_dtype(DTYPE_BF16 | DTYPE_FLAG_BCAST_DELTA)
    assert bulk_dtype(DTYPE_F32_PUSH_I8 | DTYPE_FLAG_BCAST_DELTA) == DTYPE_F32


def _syncer_side(anchor: np.ndarray | None, theta: np.ndarray, version: int, base: int):
    """Mirror of syncer/src/downlink.rs ``publish`` (DELTA when an anchor exists)."""
    if anchor is None:
        new = theta.copy()
        return KIND_FULL, theta.tobytes(), new, HEADER.pack(KIND_FULL, version, hashlib.sha256(new.tobytes()).digest())
    delta = torch.from_numpy(theta - anchor).to(torch.bfloat16)
    new = anchor + delta.float().numpy()
    wire = delta.view(torch.int16).numpy().astype("<i2").tobytes()
    return KIND_DELTA, wire, new, HEADER.pack(KIND_DELTA, base, hashlib.sha256(new.tobytes()).digest())


class _Collect:
    def begin_message(self, message_id, total):
        return bytearray()

    def bind_fragment(self, tx, msg_type, fid, version, payload_bytes):
        self.bound = payload_bytes

    def consume_chunk(self, tx, offset, payload):
        assert offset == len(tx)
        tx += bytes(payload)

    def finish_message(self, tx):
        return bytes(tx)

    def abort_message(self, tx, tokens):
        self.aborted = True


def test_downlink_sink_decodes_split_chunks_into_the_shared_anchor():
    rng = np.random.default_rng(1)
    numel = 1001
    anchors = DownlinkAnchors(1)
    inner = _Collect()
    sink = DownlinkChunkSink(inner, anchors, [numel], bcast_type=5)
    syncer_anchor = None
    theta = rng.standard_normal(numel).astype(np.float32)
    for version in range(1, 5):
        theta = theta + 0.01 * rng.standard_normal(numel).astype(np.float32)
        _kind, wire, syncer_anchor, header = _syncer_side(syncer_anchor, theta, version, version - 1)
        payload = header + wire
        tx = sink.begin_message(version, len(payload))
        sink.bind_fragment(tx, 5, 0, version, len(payload))
        assert inner.bound == 4 * numel
        cuts = [0, 7, 40, 41, 42, 777, 1501, len(payload)]  # inside header, odd byte splits
        for a, b in itertools.pairwise(cuts):
            if a < b:
                sink.consume_staged_chunk(tx, a, sink.stage_chunk(tx, payload[a:b]), b - a)
        assert sink.finish_message(tx) == syncer_anchor.tobytes()
        assert anchors.version(0) == version
        np.testing.assert_allclose(syncer_anchor, theta, rtol=0, atol=2e-2 / 128)


def test_downlink_delta_without_its_anchor_or_with_a_drifted_one_asks_for_a_resync():
    numel = 64
    theta = np.linspace(-1, 1, numel, dtype=np.float32)
    anchors = DownlinkAnchors(1)
    _kind, wire, held, header = _syncer_side(None, theta, 1, 1)
    anchors.decode_whole(0, 1, header + wire, numel)
    _kind, wire, _new, header = _syncer_side(held, theta + 0.5, 3, 2)  # learner holds 1, not 2
    with pytest.raises(DownlinkResync, match="needs anchor version 2"):
        anchors.decode_whole(0, 3, header + wire, numel)
    drifted = held.copy()
    drifted[5] += 1e-3
    _kind, wire, _new, header = _syncer_side(drifted, theta + 0.5, 2, 1)
    with pytest.raises(DownlinkResync, match="digest mismatch"):
        anchors.decode_whole(0, 2, header + wire, numel)
    assert anchors.version(0) == 1, "a failed delta must not move the anchor"


def test_int8_and_downlink_options_are_bound_into_the_role_stream_contract():
    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        role_stream_session_contract_hash,
    )

    exact = _role_stream_config("actor", 30100, sweep=True)
    hashes = {role_stream_session_contract_hash("1" * 64, config) for config in (
        exact,
        replace(exact, push_dtype="int8"),
        replace(exact, push_dtype="int8", push_int8_block=64),
        replace(exact, bcast_delta=True),
    )}
    assert len(hashes) == 4
    assert exact.wire_dtype_label == "fp32"  # unchanged for existing runs
    assert replace(exact, push_dtype="int8", bcast_delta=True).session_dtype == (
        DTYPE_F32_PUSH_I8 | DTYPE_FLAG_BCAST_DELTA)
    with pytest.raises(ValueError, match="int8 block"):
        replace(exact, push_int8_block=16)


def test_strict_avg_downlink_reaches_the_syncer_and_the_island():
    from yeto import launcher

    args = _launcher_args("ports", ("--rl-sync-bcast-delta-full-every", "8"), gpu="modal:2xh100")
    args.controller = "local"
    run = _island_run(args)
    assert " --bcast-delta " in run
    command = launcher.role_syncer_commands(args, 1, binary="yeto-syncer")["actor"]
    assert "--bcast-delta-full-every 8" in command
    plain = _launcher_args("ports", (), gpu="modal:2xh100")
    plain.controller = "local"
    assert "--bcast-delta" not in _island_run(plain)


@pytest.mark.parametrize(("extra", "match"), (
    (("--rl-sync-bcast-delta-full-every", "1"), "0 \\(off\\) or >= 2"),
    (("--rl-sync-preset", "decoupled", "--local-rl-rounds-per-sync", "2",
      "--rl-sync-bcast-delta-full-every", "4"), "only applies to --rl-sync-preset strict-avg"),
    (("--rl-sync-push-dtype", "int8"), "--rl-sync-push-dtype only applies"),
))
def test_e3_downlink_and_int8_scope(extra, match):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", extra, gpu="modal:2xh100")
    args.controller = "local"
    with pytest.raises(ValueError, match=match):
        _prepare_rl_args(args)


def _run_bridge_pair(binary, tmp_path, rounds, *, full_every):
    """Two fake islands through ``StrictRlBridge`` (the verl/Miles LoRA path)."""
    import subprocess

    from test_rl_integration import _BridgeThread, _FakeMiles, _port, _start

    from yeto.rl.bridge import BridgeConfig, StrictRlBridge

    port = _port()
    process = _start(binary, port, tmp_path / "state.ckpt", rounds=rounds)
    if full_every:
        process.kill()
        process.wait()
        argv = [*process.args, "--bcast-delta-full-every", str(full_every)]
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    runtimes = (_FakeMiles([0.1, 0.3], 0), _FakeMiles([0.3, 0.5], 1))
    threads = []
    try:
        for learner_id, runtime in enumerate(runtimes):
            config = BridgeConfig(
                syncer_addr=("127.0.0.1", port), learner_id=learner_id, global_rounds=rounds,
                groups_per_round=1, samples_per_group=2, local_optimizer_steps=1, wan_streams=0,
                expected_specs=runtime.current.specs,
                base_model_revision=runtime.current.base_model_revision,
                lora_config_hash=runtime.current.lora_config_hash,
                layout_hash=runtime.current.layout_hash,
                event_tape=str(tmp_path / f"island-{learner_id}.jsonl"),
                bcast_delta=bool(full_every))
            threads.append(_BridgeThread(StrictRlBridge(runtime, config)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive() and thread.error is None, thread.error
        assert process.wait(timeout=10) == 0
        return [next(iter(t.result.tensors.values())) for t in threads], runtimes
    finally:
        for thread in threads:
            thread.bridge.client.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_strict_bridge_with_bf16_downlink_finishes_on_the_exact_f32_cut(syncer_binary, tmp_path):  # noqa: F811
    """E3, verl/LoRA path: islands train from the bf16-delta anchor, the syncer
    keeps exact masters, and the final cut matches the exact f32 run (up to the
    f32 rounding of ``local - anchor`` on a different anchor)."""
    binary = syncer_binary
    exact_dir, delta_dir = tmp_path / "f32", tmp_path / "delta"
    exact_dir.mkdir()
    delta_dir.mkdir()
    exact, _ = _run_bridge_pair(binary, exact_dir, 4, full_every=0)
    delta, runtimes = _run_bridge_pair(binary, delta_dir, 4, full_every=3)
    for want, got in zip(exact, delta, strict=True):
        assert torch.allclose(want, got, rtol=0, atol=1e-6), (want, got)
    assert all(runtime.applied[-1] == 4 for runtime in runtimes)


def test_verl_island_accepts_the_downlink_flag():
    from yeto.rl.adapters.verl.island_entry import parse_args as verl_parse_args

    args = verl_parse_args([
        "--model", "m", "--model-revision", "r", "--data", "d", "--learner-id", "0",
        "--reward-function", "f", "--global-rounds", "1", "--groups-per-round", "1",
        "--samples-per-group", "1", "--rollout-max-response-len", "8", "--event-tape", "t",
        "--inner-lr", "1e-6", "--seq-len", "8", "--seed", "0", "--bcast-delta"])
    assert args[0].bcast_delta is True


def test_syncer_disk_resources_only_carry_set_options():
    from types import SimpleNamespace

    import pytest

    from yeto.launch.syncer import syncer_disk_resources

    assert syncer_disk_resources(SimpleNamespace()) == {}
    assert syncer_disk_resources(SimpleNamespace(syncer_disk_gib=None, syncer_disk_tier=None)) == {}
    assert syncer_disk_resources(SimpleNamespace(syncer_disk_gib=600, syncer_disk_tier="ultra")) == {
        "disk_size": 600, "disk_tier": "ultra"}
    assert syncer_disk_resources(SimpleNamespace(syncer_instance_type="r6i.16xlarge")) == {
        "instance_type": "r6i.16xlarge"}
    with pytest.raises(ValueError, match="--syncer-disk-gib"):
        syncer_disk_resources(SimpleNamespace(syncer_disk_gib=0))
