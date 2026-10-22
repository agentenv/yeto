"""Tensor-aligned push quantization blocks (session dtype flag 0x40).

Without the flag, int8/int4 blocks run across a whole fragment, so a tensor
replicated on several ranks (norms, routers, TP replicas) lands in different
blocks next to different neighbours and quantizes differently on each rank.
With the flag the blocks restart at every tensor boundary.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_p3_int4_push import _decode_i4
from test_rl_dense_sweep_real_syncer import (  # noqa: F401
    FRAGMENTS,
    _dense_args,
    _start_syncer,
    _stop,
    _wait,
    syncer_binary,
)

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import (
    DTYPE_F32,
    DTYPE_F32_PUSH_BF16,
    DTYPE_F32_PUSH_I4,
    DTYPE_F32_PUSH_I8,
    DTYPE_F32_PUSH_Q4,
    DTYPE_FLAG_BCAST_DELTA,
    DTYPE_FLAG_TENSOR_BLOCKS,
    FULL_PARAMETER_PUSH_DTYPES,
    Q4_BLOCK,
    SyncerClient,
    _f32_parts_to_i4,
    _f32_parts_to_i8,
    _f32_parts_to_q4,
    _q4_nbytes,
    bulk_dtype,
    f32_bytes_to_i4,
    f32_bytes_to_i8,
    f32_bytes_to_q4,
    i4_bytes_to_f32,
    i4_nbytes,
    i8_nbytes,
    session_base,
    tensor_blocks,
    valid_session_dtype,
)
from yeto.push_error_feedback import PushErrorFeedback
from yeto.tensor_io import dequantize_q4

TENSOR = np.linspace(-0.3, 0.7, 70, dtype=np.float32)  # the replicated tensor
LOUD = np.full(45, 50.0, np.float32)
QUIET = np.full(300, 1e-4, np.float32)
QUIET_TAIL = QUIET[: LOUD.size]
CODECS = ("int8", "int4", "int4-log")


def _decode(codec: str, wire: bytes, segments: list[int] | None, numel: int) -> np.ndarray:
    """Independent reference decoder: one segment at a time, as the syncer does."""
    sizes = segments or [numel]
    at, out = 0, []
    if codec == "int4-log":
        for n in sizes:
            end = at + _q4_nbytes(n)
            out.append(dequantize_q4(wire[at:end], n).numpy())
            at = end
        return np.concatenate(out)
    block = int(np.frombuffer(wire, "<u4", 1)[0])
    if codec == "int4":
        for n in sizes:
            end = at + i4_nbytes(n, block) - 4
            out.append(_decode_i4(wire[:4] + wire[4 + at:4 + end], n))
            at = end
        return np.concatenate(out)
    at = 4
    for n in sizes:
        for start in range(0, n, block):
            m = min(block, n - start)
            scale = np.frombuffer(wire, "<f4", 1, at)[0]
            out.append(np.frombuffer(wire, np.int8, m, at + 4).astype(np.float32) * scale)
            at += 4 + m
    assert at == len(wire)
    return np.concatenate(out)


def _encode(codec: str, values: np.ndarray, segments: list[int] | None) -> bytes:
    raw = values.astype("<f4").tobytes()
    if codec == "int8":
        return f32_bytes_to_i8(raw, 128, segments=segments)
    if codec == "int4":
        return f32_bytes_to_i4(raw, 128, segments=segments)
    return f32_bytes_to_q4(raw, segments=segments)


def _rank_view(codec: str, before: np.ndarray, after: np.ndarray, aligned: bool) -> np.ndarray:
    """What the syncer would decode for TENSOR on a rank with these neighbours."""
    values = np.concatenate([before, TENSOR, after])
    segments = [before.size, TENSOR.size, after.size]
    wire = _encode(codec, values, segments if aligned else None)
    decoded = _decode(codec, wire, segments if aligned else None, values.size)
    if not aligned:  # legacy wire: blocks run across the fragment
        decoded = _decode_whole(codec, wire, values.size)
    return decoded[before.size:before.size + TENSOR.size]


def _decode_whole(codec: str, wire: bytes, numel: int) -> np.ndarray:
    if codec == "int4-log":
        return dequantize_q4(wire, numel).numpy()
    return _decode(codec, wire, None, numel)


@pytest.mark.parametrize("codec", CODECS)
def test_replicated_tensor_quantizes_identically_with_tensor_blocks(codec):
    left = _rank_view(codec, LOUD, QUIET, aligned=True)
    right = _rank_view(codec, QUIET, QUIET_TAIL, aligned=True)
    assert np.array_equal(left, right)


@pytest.mark.xfail(strict=True, reason="legacy layout: blocks cross tensor boundaries (the drift this flag fixes)")
@pytest.mark.parametrize("codec", CODECS)
def test_replicated_tensor_drifts_across_ranks_without_tensor_blocks(codec):
    left = _rank_view(codec, LOUD, QUIET, aligned=False)
    right = _rank_view(codec, QUIET, QUIET_TAIL, aligned=False)
    assert np.array_equal(left, right)


@pytest.mark.parametrize("codec", CODECS)
def test_legacy_wire_is_unchanged_when_segments_are_absent(codec):
    values = np.concatenate([LOUD, TENSOR, QUIET])
    legacy = _encode(codec, values, None)
    single = _encode(codec, values, [values.size])
    assert legacy == single  # one segment is the original whole-fragment layout


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("sizes", ([1], [3, 130, 1, 257], [64, 64], [300, 5]))
def test_round_trip_stays_within_each_tensors_own_bound(codec, sizes):
    rng = np.random.default_rng(len(sizes))
    parts = [(rng.standard_normal(n) * 10.0 ** rng.integers(-4, 2)).astype(np.float32) for n in sizes]
    values = np.concatenate(parts)
    decoded = _decode(codec, _encode(codec, values, sizes), sizes, values.size)
    at = 0
    for part in parts:
        got = decoded[at:at + part.size]
        absmax = np.abs(part).max()
        if codec == "int4-log":
            kept = got != 0
            assert np.all(np.abs(got[kept] - part[kept]) <= np.abs(part[kept]) * (2 ** 0.5 - 1) * (1 + 1e-6))
            assert np.all(np.abs(part[~kept]) <= absmax * 2 ** -6.5 * (1 + 1e-6))
        else:
            bound = absmax / (254 if codec == "int8" else 14)
            assert np.abs(got - part).max() <= bound * (1 + 1e-5)
        at += part.size


@pytest.mark.parametrize("codec", CODECS)
def test_wire_size_counts_one_extra_tail_block_per_tensor(codec):
    sizes = [3, 130, 1, 257]
    total = sum(sizes)
    wire = _encode(codec, np.ones(total, np.float32), sizes)
    expected = {"int8": i8_nbytes(total, 128, sizes), "int4": i4_nbytes(total, 128, sizes),
                "int4-log": _q4_nbytes(total, sizes)}[codec]
    assert len(wire) == expected
    whole = {"int8": i8_nbytes(total, 128), "int4": i4_nbytes(total, 128), "int4-log": _q4_nbytes(total)}[codec]
    extra_blocks = sum(-(-n // 128) for n in sizes) - -(-total // 128)
    if codec == "int4-log":
        extra_blocks = sum(-(-n // Q4_BLOCK) for n in sizes) - -(-total // Q4_BLOCK)
    assert 0 < expected - whole <= extra_blocks * (4 + (Q4_BLOCK // 2 if codec == "int4-log" else 128))
    # A GLM-scale fragment (a few hundred tensors) pays well under 0.1 percent.
    big = [4096 * 4096] * 8 + [4096] * 400
    assert (i8_nbytes(sum(big), 128, big) - i8_nbytes(sum(big), 128)) / i8_nbytes(sum(big), 128) < 1e-3


@pytest.mark.parametrize("codec", CODECS)
def test_streaming_matches_one_shot_for_any_part_split(codec):
    sizes = [3, 130, 1, 257, 128, 5]
    rng = np.random.default_rng(7)
    raw = (rng.standard_normal(sum(sizes)) * 3).astype("<f4").tobytes()
    stream = {"int8": lambda p: _f32_parts_to_i8(p, 128, sizes),
              "int4": lambda p: _f32_parts_to_i4(p, 128, None, sizes),
              "int4-log": lambda p: _f32_parts_to_q4(p, sizes)}[codec]
    one_shot = _encode(codec, np.frombuffer(raw, "<f4"), sizes)
    for step in (1, 7, 100, 4 * 128, 4 * 129, len(raw)):
        parts = [raw[i:i + step] for i in range(0, len(raw), step)]
        assert b"".join(stream(parts)) == one_shot, step
    with pytest.raises(ValueError, match="mid-value"):
        list(stream([raw[:6]]))
    with pytest.raises(ValueError, match="longer than"):
        list(stream([raw + b"\0\0\0\0"]))


def test_dtype_flag_is_valid_only_on_block_quantized_push_dtypes():
    for base in (DTYPE_F32_PUSH_I8, DTYPE_F32_PUSH_I4, DTYPE_F32_PUSH_Q4):
        for extra in (0, DTYPE_FLAG_BCAST_DELTA):
            dtype = base | DTYPE_FLAG_TENSOR_BLOCKS | extra
            assert valid_session_dtype(dtype) and tensor_blocks(dtype)
            assert session_base(dtype) == base and bulk_dtype(dtype) == DTYPE_F32
    for base in (DTYPE_F32, DTYPE_F32_PUSH_BF16, 2, 3):
        assert not valid_session_dtype(base | DTYPE_FLAG_TENSOR_BLOCKS)
    assert not tensor_blocks(DTYPE_F32_PUSH_I8)


def test_error_feedback_residual_follows_the_tensor_blocks():
    sizes = [70, 45, 91]
    rng = np.random.default_rng(3)
    delta = (rng.standard_normal(sum(sizes)) * np.repeat([1.0, 40.0, 0.01], sizes)).astype(np.float32)
    feedback = PushErrorFeedback(128, "f32")
    raw = delta.tobytes()
    parts = [raw[i:i + 50] for i in range(0, len(raw), 50)]
    sent = b"".join(feedback.parts(0, 2, 2, delta.size, parts, sizes))
    assert sent == raw
    feedback.commit(0, 2)
    wire = f32_bytes_to_i4(raw, 128, segments=sizes)
    expected = delta - i4_bytes_to_f32(wire, delta.size, sizes)
    assert np.array_equal(feedback._residuals[0][1], expected)  # noqa: SLF001
    legacy = delta - i4_bytes_to_f32(f32_bytes_to_i4(raw, 128), delta.size)
    assert not np.array_equal(expected, legacy)


def test_tensor_blocks_are_the_default_for_quantized_pushes_and_opt_out_keeps_the_old_contract():
    from test_rl_full_parameter_sync import _role_stream_config

    from yeto.rl.adapters.miles.full_parameter.role_stream import role_stream_session_contract_hash

    def contract(config):
        return role_stream_session_contract_hash("1" * 64, config)

    exact = _role_stream_config("actor", 30100, sweep=True)
    for codec, base in (("int8", DTYPE_F32_PUSH_I8), ("int4", DTYPE_F32_PUSH_I4),
                        ("int4-log", DTYPE_F32_PUSH_Q4)):
        default = replace(exact, push_dtype=codec)
        legacy = replace(default, push_tensor_blocks=False)
        assert default.session_dtype == base | DTYPE_FLAG_TENSOR_BLOCKS
        assert legacy.session_dtype == base
        assert default.wire_dtype_label == legacy.wire_dtype_label + "-tb"
        assert not legacy.wire_dtype_label.endswith("-tb")
        # A run resumed with the other choice presents another session
        # contract, which a syncer checkpoint refuses.
        assert contract(default) != contract(legacy)
    for codec in ("f32", "bf16"):  # nothing to restart: contract and dtype unchanged
        config = replace(exact, push_dtype=codec)
        assert config.session_dtype == FULL_PARAMETER_PUSH_DTYPES[codec]
        assert contract(config) == contract(replace(config, push_tensor_blocks=False))


def test_resumed_session_is_refused_when_its_recorded_tensor_block_choice_changes(
        syncer_binary, tmp_path):  # noqa: F811
    """Real syncer: a checkpoint written under one contract refuses the other."""
    from test_rl_full_parameter_sync import _role_stream_config

    from yeto.rl.adapters.miles.full_parameter.role_stream import role_stream_session_contract_hash

    exact = _role_stream_config("actor", 30100, sweep=True)
    layout = FragmentLayout([Fragment(MERGE_RDA, [(f"w{i}", 3)]) for i in range(FRAGMENTS)])

    def start(config):
        process, port, profile_hash = _start_syncer(syncer_binary, tmp_path)
        client = SyncerClient(("127.0.0.1", port), 0, layout, dtype=config.session_dtype, num_streams=1,
                              connect_timeout=10, max_reconnects=0, finalization_timeout=20,
                              syncer_profile_hash=bytes.fromhex(profile_hash),
                              session_contract_hash=role_stream_session_contract_hash("1" * 64, config))
        return process, client

    legacy = replace(exact, push_dtype="int8", push_tensor_blocks=False)
    default = replace(legacy, push_tensor_blocks=True)
    process, client = start(legacy)
    try:  # first run: writes the checkpoint under the legacy contract
        client.start()
        for fid in range(FRAGMENTS):
            client.send_init(fid, np.zeros(3, "<f4").tobytes())
        _wait(lambda: (tmp_path / "state.ckpt").exists(), "no checkpoint written")
    finally:
        client.close()
        _stop(process)
    for config, accepted in ((default, False), (legacy, True)):
        process, client = start(config)
        try:
            client.start()
            for fid in range(FRAGMENTS):
                client.send_init(fid, np.zeros(3, "<f4").tobytes())
            if accepted:
                _wait(lambda: client.drain_pulls() or True, "resume stalled", timeout=5)
            else:
                with pytest.raises(Exception):
                    _wait(lambda: client.check_health(), "client stayed healthy", timeout=10)
                # The refusal ends the syncer; let it exit and print the
                # error before _stop, or SIGTERM can race the message.
                process.wait(timeout=10)
        finally:
            client.close()
            log = _stop(process)
        assert ("session contract hash does not match" in log) is (not accepted), log


def test_opt_out_flag_reaches_the_island_and_is_quantized_dtype_only():
    from yeto.launch.rl_island_task import _rl_trainable_island_flags  # noqa: F401

    with pytest.raises(ValueError, match="only with --rl-sync-push-dtype int8, int4 or int4-log"):
        _dense_args("--rl-sync-push-no-tensor-blocks")
    args = _dense_args("--rl-sync-push-dtype", "int8", "--rl-sync-push-no-tensor-blocks")
    assert args.rl_sync_push_no_tensor_blocks
    assert not _dense_args("--rl-sync-push-dtype", "int8").rl_sync_push_no_tensor_blocks


# ---- real Rust syncer -------------------------------------------------------

_SIZES = ([LOUD.size, TENSOR.size, QUIET.size], [QUIET.size, TENSOR.size, QUIET_TAIL.size])
_NUMEL = sum(_SIZES[0])


def _sweep_through_real_syncer(binary, tmp_path, dtype):
    """One strict sweep; fragment 0 = [LOUD, TENSOR, QUIET], fragment 1 = [QUIET, TENSOR, QUIET_TAIL].

    Returns the decoded delta (final minus init) of each fragment."""
    layout = FragmentLayout([
        Fragment(MERGE_RDA, [(f"w{i}.{j}", n) for j, n in enumerate(_SIZES[i])]) for i in range(FRAGMENTS)])
    deltas = [np.concatenate([LOUD, TENSOR, QUIET]), np.concatenate([QUIET, TENSOR, QUIET_TAIL])]
    process, port, profile_hash = _start_syncer(binary, tmp_path)
    client = SyncerClient(("127.0.0.1", port), 0, layout, dtype=dtype, num_streams=1,
                          connect_timeout=10, max_reconnects=0, finalization_timeout=20,
                          syncer_profile_hash=bytes.fromhex(profile_hash))
    try:
        client.start()
        for fid in range(FRAGMENTS):
            client.send_init(fid, np.zeros(_NUMEL, "<f4").tobytes())
        pushed = set()

        def next_pull():
            client.check_health()
            return next(iter(client.drain_pulls()), None)

        while len(pushed) < FRAGMENTS:
            permit = _wait(next_pull, "no PULL")
            fid = permit.fragment_id
            raw = deltas[fid].astype("<f4").tobytes()
            if fid == 0:  # one-shot path
                client.push_fragment(fid, permit.global_step, permit.round_attempt, 0, 1, 1, 1, raw)
            else:  # streamed path, split mid-value
                client.push_fragment_parts(fid, permit.global_step, permit.round_attempt, 0, 1, 1, 1,
                                           [raw[i:i + 37] for i in range(0, len(raw), 37)])
            pushed.add(fid)
        manifest, finals = client.wait_for_final_fragments(timeout=20)
        result = {f.fragment_id: torch.frombuffer(bytearray(f.data), dtype=torch.float32).numpy()[-_NUMEL:]
                  for f in finals}
        client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        client.close()
        _stop(process)
    return deltas, result


@pytest.mark.parametrize(("codec", "base"), (("int8", DTYPE_F32_PUSH_I8), ("int4", DTYPE_F32_PUSH_I4),
                                             ("int4-log", DTYPE_F32_PUSH_Q4)))
def test_real_syncer_decodes_tensor_blocks_and_keeps_the_replica_identical(
        syncer_binary, tmp_path, codec, base):  # noqa: F811
    deltas, finals = _sweep_through_real_syncer(syncer_binary, tmp_path, base | DTYPE_FLAG_TENSOR_BLOCKS)
    for fid in range(FRAGMENTS):
        wire = _encode(codec, deltas[fid], _SIZES[fid])
        expected = _decode(codec, wire, _SIZES[fid], _NUMEL)
        assert np.allclose(finals[fid], expected, rtol=0, atol=1e-6 * np.abs(deltas[fid]).max()), fid
    start = (LOUD.size, QUIET.size)
    assert np.array_equal(finals[0][start[0]:start[0] + TENSOR.size],
                          finals[1][start[1]:start[1] + TENSOR.size])


def test_real_syncer_refuses_the_flag_on_an_unquantized_session(syncer_binary, tmp_path):  # noqa: F811
    process, port, profile_hash = _start_syncer(syncer_binary, tmp_path)
    layout = FragmentLayout([Fragment(MERGE_RDA, [(f"w{i}", 3)]) for i in range(FRAGMENTS)])
    with pytest.raises(ValueError, match="unsupported session dtype"):
        SyncerClient(("127.0.0.1", port), 0, layout, dtype=DTYPE_F32 | DTYPE_FLAG_TENSOR_BLOCKS)
    _stop(process)
