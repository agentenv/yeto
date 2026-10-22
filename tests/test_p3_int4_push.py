"""S27 P3: int4 pushes of full-parameter outer deltas. CPU only.

Two int4 push dtypes keep INIT/BCAST/FINAL f32 and quantize only the outer
gradient:

* ``int4`` (session dtype 7): linear symmetric int4, per-block absmax scale,
  nearest or stochastic rounding.
* ``int4-log`` (session dtype 6): the SFT q4 codec (256-value blocks, log2
  levels).

These tests bound both codecs' round-trip error, account wire sizes, merge
mixed-precision deltas, and compare the codecs (with and without island-side
error feedback) on a tiny model.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_rl_dense_sweep_real_syncer import (  # noqa: F401
    NUMEL,
    STREAMED,
    _dense_args,
    _run_sweeps,
    _tail,
    syncer_binary,
)
from test_rl_full_parameter_sync import _role_stream_config

from yeto.protocol import (
    DEFAULT_I4_BLOCK,
    DEFAULT_I8_BLOCK,
    DTYPE_F32,
    DTYPE_F32_PUSH_I4,
    DTYPE_F32_PUSH_Q4,
    DTYPE_FLAG_BCAST_DELTA,
    FULL_PARAMETER_PUSH_DTYPES,
    Q4_BLOCK,
    _f32_parts_to_i4,
    _f32_parts_to_q4,
    _q4_nbytes,
    bulk_dtype,
    f32_bytes_to_bf16,
    f32_bytes_to_i4,
    f32_bytes_to_i8,
    f32_bytes_to_q4,
    i4_nbytes,
    i8_nbytes,
    valid_session_dtype,
)
from yeto.push_error_feedback import PushErrorFeedback
from yeto.tensor_io import dequantize_q4, quantize_q4

# int4-log: nearest level in log2 space errs by at most a factor sqrt(2); a
# value below scale * 2**-6.5 is dropped to zero.
Q4_REL = math.sqrt(2) - 1
Q4_FLOOR = 2**-6.5


def _spread(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.repeat(rng.uniform(1e-6, 3, 13), n // 13 + 1)[:n]
    values = (rng.standard_normal(n) * scales).astype(np.float32)
    values[:Q4_BLOCK] = 0.0  # an all-zero block
    return values


def _decode_i4(wire: bytes, numel: int) -> np.ndarray:
    block = int(np.frombuffer(wire, "<u4", 1)[0])
    at, out = 4, []
    for start in range(0, numel, block):
        n = min(block, numel - start)
        scale = np.frombuffer(wire, "<f4", 1, at)[0]
        body = np.frombuffer(wire, np.uint8, -(-n // 2), at + 4)
        nibbles = np.stack([body & 0x0F, body >> 4], axis=1).reshape(-1)[:n].astype(np.int8)
        out.append(np.where(nibbles > 7, nibbles - 16, nibbles).astype(np.float32) * scale)
        at += 4 + body.size
    assert at == len(wire)
    return np.concatenate(out)


def _decode(codec: str, values: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    raw = values.astype("<f4").tobytes()
    if codec == "f32":
        return values.astype(np.float32)
    if codec == "bf16":
        bits = np.frombuffer(f32_bytes_to_bf16(raw), "<u2").astype(np.uint32) << 16
        return bits.view(np.float32)
    if codec == "int8":
        wire = f32_bytes_to_i8(raw, DEFAULT_I8_BLOCK)
        out, at = [], 4
        for start in range(0, values.size, DEFAULT_I8_BLOCK):
            n = min(DEFAULT_I8_BLOCK, values.size - start)
            scale = np.frombuffer(wire, "<f4", 1, at)[0]
            out.append(np.frombuffer(wire, np.int8, n, at + 4).astype(np.float32) * scale)
            at += 4 + n
        return np.concatenate(out)
    if codec == "int4-log":
        return dequantize_q4(f32_bytes_to_q4(raw), values.size).numpy()
    assert codec == "int4"
    return _decode_i4(f32_bytes_to_i4(raw, DEFAULT_I4_BLOCK, rng=rng), values.size)


def _wire_nbytes(codec: str, numel: int) -> int:
    return {"f32": 4 * numel, "bf16": 2 * numel,
            "int8": i8_nbytes(numel, DEFAULT_I8_BLOCK),
            "int4": i4_nbytes(numel, DEFAULT_I4_BLOCK),
            "int4-log": _q4_nbytes(numel)}[codec]


def test_int4_push_sessions_keep_bulk_f32_and_compose_with_the_downlink():
    assert FULL_PARAMETER_PUSH_DTYPES["int4"] == DTYPE_F32_PUSH_I4 == 7
    assert FULL_PARAMETER_PUSH_DTYPES["int4-log"] == DTYPE_F32_PUSH_Q4 == 6
    for base in (DTYPE_F32_PUSH_I4, DTYPE_F32_PUSH_Q4):
        for dtype in (base, base | DTYPE_FLAG_BCAST_DELTA):
            assert valid_session_dtype(dtype)
            assert bulk_dtype(dtype) == DTYPE_F32
    assert not valid_session_dtype(8)


@pytest.mark.parametrize("numel", (1, 63, 64, 65, 10_007))
@pytest.mark.parametrize("block", (64, 128, 250, 256))
def test_linear_int4_round_trip_is_within_half_a_scale_step(numel, block):
    values = _spread(numel)
    wire = f32_bytes_to_i4(values.tobytes(), block)
    assert len(wire) == i4_nbytes(numel, block)
    decoded = _decode_i4(wire, numel)
    for start in range(0, numel, block):
        chunk = values[start:start + block]
        absmax = np.abs(chunk).max()
        assert np.abs(decoded[start:start + block] - chunk).max() <= absmax / 14 * (1 + 1e-6)
        if absmax:
            assert np.abs(decoded[start:start + block]).max() == pytest.approx(absmax, rel=1e-6)


def test_linear_int4_stochastic_rounding_is_unbiased_and_seeded():
    values = np.full(256, 0.3, np.float32)
    values[0] = 7.0  # scale = 1, so 0.3 rounds to 0 or 1
    nearest = _decode_i4(f32_bytes_to_i4(values.tobytes(), 256), 256)
    assert np.all(nearest[1:] == 0.0)
    draws = np.stack([
        _decode_i4(f32_bytes_to_i4(values.tobytes(), 256, rng=np.random.default_rng(seed)), 256)
        for seed in range(200)
    ])
    assert set(np.unique(draws[:, 1:])) == {0.0, 1.0}
    assert draws[:, 1:].mean() == pytest.approx(0.3, abs=0.01)
    again = f32_bytes_to_i4(values.tobytes(), 256, rng=np.random.default_rng(5))
    assert again == f32_bytes_to_i4(values.tobytes(), 256, rng=np.random.default_rng(5))


def test_linear_int4_streaming_matches_one_shot_and_rejects_bad_input():
    raw = _spread(3 * 128 + 17).tobytes()
    parts = [raw[:3], raw[3:4 * 128 + 5], raw[4 * 128 + 5:]]  # split mid-value/mid-block
    assert b"".join(_f32_parts_to_i4(parts, 128)) == f32_bytes_to_i4(raw, 128)
    with pytest.raises(ValueError, match="mid-value"):
        list(_f32_parts_to_i4([raw[:6]], 128))
    with pytest.raises(ValueError, match="non-finite"):
        f32_bytes_to_i4(np.array([1.0, np.inf], np.float32).tobytes(), 128)
    for block in (62, 65, 258):
        with pytest.raises(ValueError, match="int4 block size"):
            f32_bytes_to_i4(raw, block)


@pytest.mark.parametrize("numel", (1, Q4_BLOCK - 1, Q4_BLOCK, Q4_BLOCK + 1, 10_007))
def test_log_q4_round_trip_stays_within_its_log_level_bounds(numel):
    values = _spread(numel)
    wire = f32_bytes_to_q4(values.tobytes())
    assert len(wire) == _q4_nbytes(numel)
    # The shared encoder is the SFT codec, byte for byte.
    assert wire == quantize_q4(torch.from_numpy(values))
    decoded = dequantize_q4(wire, numel).numpy()
    for start in range(0, numel, Q4_BLOCK):
        x, d = values[start:start + Q4_BLOCK], decoded[start:start + Q4_BLOCK]
        absmax = np.abs(x).max()
        kept = d != 0
        assert np.all(np.sign(d[kept]) == np.sign(x[kept]))
        assert np.all(np.abs(d[kept] - x[kept]) <= np.abs(x[kept]) * Q4_REL * (1 + 1e-6))
        assert np.all(np.abs(x[~kept]) <= absmax * Q4_FLOOR * (1 + 1e-6))
        if absmax:
            assert np.abs(d).max() == absmax  # the absmax value is exact


def test_log_q4_streaming_matches_one_shot_and_rejects_bad_input():
    raw = _spread(3 * Q4_BLOCK + 17).tobytes()
    parts = [raw[:3], raw[3:4 * Q4_BLOCK + 5], raw[4 * Q4_BLOCK + 5:]]
    assert b"".join(_f32_parts_to_q4(parts)) == f32_bytes_to_q4(raw)
    with pytest.raises(ValueError, match="mid-value"):
        list(_f32_parts_to_q4([raw[:6]]))
    with pytest.raises(ValueError, match="non-finite"):
        f32_bytes_to_q4(np.array([1.0, np.nan], np.float32).tobytes())


def test_wire_size_accounting_per_push_dtype():
    numel = 1_000_000_000
    sizes = {codec: _wire_nbytes(codec, numel) for codec in FULL_PARAMETER_PUSH_DTYPES}
    assert sizes["bf16"] * 2 == sizes["f32"]
    # int8: 1 byte/value + 4-byte scale per 128 values + 4-byte header.
    assert sizes["int8"] == 4 + numel + 4 * -(-numel // 128)
    # int4: 0.5 byte/value + 4-byte scale per block + 4-byte header.
    assert sizes["int4"] == 4 + numel // 2 + 4 * -(-numel // 128)  # 4.25 bits/value
    assert i4_nbytes(numel, 256) == 4 + numel // 2 + 4 * -(-numel // 256)  # 4.125
    assert i4_nbytes(numel, 64) == 4 + numel // 2 + 4 * -(-numel // 64)  # 4.5
    # int4-log: 4.125 bits/value.
    assert sizes["int4-log"] == numel // 2 + 4 * -(-numel // 256)
    assert sizes["f32"] / sizes["int4"] == pytest.approx(32 / 4.25, rel=1e-6)
    assert sizes["f32"] / sizes["int4-log"] == pytest.approx(32 / 4.125)


def test_merging_islands_with_different_push_precisions_keeps_the_per_island_bounds():
    """Session dtype is fixed per session, but a merge is linear in the decoded
    deltas: each island's quantization error stays its own, scaled by its weight."""
    numel = 4 * Q4_BLOCK
    codecs = ("bf16", "int8", "int4", "int4-log")
    islands = {codec: _spread(numel, seed) for seed, codec in enumerate(codecs)}
    weights = {"bf16": 0.4, "int8": 0.3, "int4": 0.2, "int4-log": 0.1}
    exact = sum(weights[c] * v for c, v in islands.items())
    merged = sum(weights[c] * _decode(c, v) for c, v in islands.items())

    def block_absmax(values: np.ndarray, block: int) -> np.ndarray:
        return np.repeat(np.abs(values).reshape(-1, block).max(axis=1), block)

    bound = (weights["bf16"] * np.abs(islands["bf16"]) * 2**-8
             + weights["int8"] * block_absmax(islands["int8"], 128) / 254
             + weights["int4"] * block_absmax(islands["int4"], DEFAULT_I4_BLOCK) / 14
             + weights["int4-log"] * block_absmax(islands["int4-log"], Q4_BLOCK) * Q4_REL)
    assert np.all(np.abs(merged - exact) <= bound * (1 + 1e-5) + 1e-12)


def _tiny_multi_island_run(codec: str, *, error_feedback: str = "off", stochastic: bool = False,
                           seed: int = 0, islands: int = 3, outer_rounds: int = 6,
                           inner_steps: int = 8) -> dict[str, float]:
    """Tiny full-parameter multi-island run on CPU: every island trains from
    the shared weights, pushes its outer delta through ``codec`` and the outer
    step averages the decoded deltas (outer lr 1, no momentum). With
    ``error_feedback`` (``bf16``/``f32``, int4 only) each island runs a
    ``PushErrorFeedback`` over the whole model as one fragment.

    Returns the held-out loss, the mean per-push relative error and the
    relative error of each island's summed pushes (drift over the run).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed) if stochastic else None
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    teacher = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    data_gen = torch.Generator().manual_seed(seed + 1)
    held_out = torch.randn(512, 16, generator=data_gen)
    with torch.no_grad():
        held_target = teacher(held_out)
    global_flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    numel = global_flat.numel()
    feedback = [PushErrorFeedback(DEFAULT_I4_BLOCK, error_feedback) if error_feedback != "off" else None
                for _ in range(islands)]
    sent = [np.zeros(numel) for _ in range(islands)]
    true = [np.zeros(numel) for _ in range(islands)]
    push_errors = []
    for outer in range(outer_rounds):
        decoded = []
        for island in range(islands):
            torch.nn.utils.vector_to_parameters(global_flat.clone(), model.parameters())
            opt = torch.optim.SGD(model.parameters(), lr=0.05)
            for _ in range(inner_steps):
                x = torch.randn(64, 16, generator=data_gen)
                with torch.no_grad():
                    y = teacher(x)
                opt.zero_grad()
                torch.nn.functional.mse_loss(model(x), y).backward()
                opt.step()
            delta = (torch.nn.utils.parameters_to_vector(model.parameters()).detach() - global_flat).numpy()
            if feedback[island] is None:
                got = _decode(codec, delta, rng)
            else:
                assert codec == "int4" and not stochastic
                raw = b"".join(feedback[island].parts(0, outer + 1, 1, numel, [delta.tobytes()]))
                feedback[island].commit(0, outer + 1)
                got = _decode(codec, np.frombuffer(raw, "<f4"))
            push_errors.append(np.linalg.norm(got - delta) / np.linalg.norm(delta))
            sent[island] += got
            true[island] += delta
            decoded.append(torch.from_numpy(got.copy()))
        global_flat = global_flat + torch.stack(decoded).mean(dim=0)
    torch.nn.utils.vector_to_parameters(global_flat, model.parameters())
    with torch.no_grad():
        loss = torch.nn.functional.mse_loss(model(held_out), held_target).item()
    drift = np.mean([np.linalg.norm(s - t) / np.linalg.norm(t) for s, t in zip(sent, true)])
    return {"loss": loss, "push_error": float(np.mean(push_errors)), "drift": float(drift)}


def test_tiny_model_compares_push_codecs():
    runs = {codec: _tiny_multi_island_run(codec) for codec in FULL_PARAMETER_PUSH_DTYPES}
    runs["int4+ef-bf16"] = _tiny_multi_island_run("int4", error_feedback="bf16")
    runs["int4+ef-f32"] = _tiny_multi_island_run("int4", error_feedback="f32")
    runs["int4-sr"] = _tiny_multi_island_run("int4", stochastic=True)
    exact = runs["f32"]["loss"]
    err = {name: run["push_error"] for name, run in runs.items()}
    assert err["f32"] == 0.0
    assert 0 < err["bf16"] < err["int8"] < err["int4"] < err["int4-log"]
    # Linear int4 at the default block beats the log codec per push.
    assert err["int4"] < 0.15 < err["int4-log"]
    # Nearest rounding has lower per-push error than stochastic rounding.
    assert err["int4"] < err["int4-sr"]
    # Error feedback trades per-push error for almost no drift over the run.
    # A bf16 residual keeps almost all of the f32 residual's benefit.
    assert runs["int4+ef-f32"]["drift"] < runs["int4"]["drift"] / 10
    assert runs["int4+ef-bf16"]["drift"] < runs["int4"]["drift"] / 10
    for name, run in runs.items():
        assert run["loss"] == pytest.approx(exact, rel=0.1), name


def test_int4_options_are_bound_into_the_role_stream_contract_and_cli():
    from yeto.launch.rl_island_task import _rl_trainable_island_flags
    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        role_stream_session_contract_hash,
    )
    from yeto.rl.engine.run_flags import rl_flags

    exact = _role_stream_config("actor", 30100, sweep=True)
    linear = replace(exact, push_dtype="int4")
    log = replace(exact, push_dtype="int4-log")
    assert linear.wire_dtype_label == "fp32-push-int4-b128"
    assert replace(linear, push_int4_block=64, push_int4_stochastic=True).wire_dtype_label == (
        "fp32-push-int4-b64-sr")
    assert log.wire_dtype_label == "fp32-push-q4"
    assert replace(linear, bcast_delta=True).session_dtype == DTYPE_F32_PUSH_I4 | DTYPE_FLAG_BCAST_DELTA
    assert log.session_dtype == DTYPE_F32_PUSH_Q4
    configs = (exact, replace(exact, push_dtype="int8"), linear, log,
               replace(linear, push_int4_block=256), replace(linear, push_int4_stochastic=True))
    assert len({role_stream_session_contract_hash("1" * 64, c) for c in configs}) == len(configs)
    with pytest.raises(ValueError, match="push dtype must be one of"):
        replace(exact, push_dtype="int2")
    with pytest.raises(ValueError, match="int4 block size"):
        replace(linear, push_int4_block=65)
    with pytest.raises(ValueError, match="int4 options need"):
        replace(exact, push_int4_stochastic=True)
    with pytest.raises(ValueError, match="int4 options need"):
        replace(exact, push_int4_error_feedback="bf16")
    with pytest.raises(ValueError, match="needs nearest rounding"):
        replace(linear, push_int4_stochastic=True, push_int4_error_feedback="bf16")
    ef = replace(linear, push_int4_error_feedback="bf16")
    assert ef.wire_dtype_label == "fp32-push-int4-b128-ef-bf16"
    assert len({role_stream_session_contract_hash("1" * 64, c) for c in (
        linear, ef, replace(linear, push_int4_error_feedback="f32"))}) == 3
    choices = {flag.option: flag.kwargs["choices"] for flag in rl_flags()
               if flag.option in ("--rl-sync-push-dtype", "--sweep-push-dtype")}
    assert choices == dict.fromkeys(("--rl-sync-push-dtype", "--sweep-push-dtype"),
                                    tuple(FULL_PARAMETER_PUSH_DTYPES))
    island = _rl_trainable_island_flags(_dense_args(
        "--rl-sync-push-dtype", "int4", "--rl-sync-push-int4-block", "64",
        "--rl-sync-push-int4-stochastic"), 0, 1)
    assert "--sweep-push-dtype int4 " in island + " "
    assert "--sweep-push-int4-block 64" in island and "--sweep-push-int4-stochastic" in island
    island = _rl_trainable_island_flags(_dense_args(
        "--rl-sync-push-dtype", "int4", "--rl-sync-push-int4-error-feedback", "bf16"), 0, 1)
    assert "--sweep-push-int4-error-feedback bf16" in island
    island = _rl_trainable_island_flags(_dense_args("--rl-sync-push-dtype", "int4-log"), 0, 1)
    assert "--sweep-push-dtype int4-log" in island and "--sweep-push-int4-block" not in island
    for extra, match in ((("--rl-sync-push-int4-block", "64"), "only with --rl-sync-push-dtype int4"),
                         (("--rl-sync-push-int4-stochastic",), "only with --rl-sync-push-dtype int4"),
                         (("--rl-sync-push-dtype", "int4", "--rl-sync-push-int4-block", "300"),
                          "even, between 64 and 256"),
                         (("--rl-sync-push-int4-error-feedback", "bf16"), "only with --rl-sync-push-dtype int4"),
                         (("--rl-sync-push-dtype", "int4", "--rl-sync-push-int4-stochastic",
                           "--rl-sync-push-int4-error-feedback", "f32"), "without --rl-sync-push-int4-stochastic")):
        with pytest.raises(ValueError, match=match):
            _dense_args(*extra)


@pytest.mark.parametrize(("name", "dtype", "atol_per_sweep"), (
    ("int4", DTYPE_F32_PUSH_I4, 2.0 / 14),
    ("int4-log", DTYPE_F32_PUSH_Q4, 2.0 * Q4_REL),
))
def test_int4_pushes_stay_within_their_codec_error_of_the_f32_sweep(
        syncer_binary, tmp_path, name, dtype, atol_per_sweep):  # noqa: F811
    """End to end through the real syncer: int4 sessions decode the deltas
    into f32 masters; FINAL differs from the f32 sweep by the codec bound only."""
    exact_dir, quant_dir = tmp_path / "f32", tmp_path / name
    exact_dir.mkdir()
    quant_dir.mkdir()
    exact = _run_sweeps(syncer_binary, exact_dir, STREAMED, sweeps=3)
    quantized = _run_sweeps(syncer_binary, quant_dir, (*STREAMED, "--rl-sync-push-dtype", name),
                            sweeps=3, dtype=dtype)
    assert exact[1] == quantized[1]
    for fid, data in exact[2].items():
        want, got = _tail(data)[-NUMEL:], _tail(quantized[2][fid])[-NUMEL:]
        # Three sweeps; each delta's absmax < 2.
        assert torch.allclose(got, want, rtol=0, atol=3 * atol_per_sweep), fid


def _push(feedback: PushErrorFeedback, fragment_id: int, step: int, delta: np.ndarray,
          *, chunk: int = 1000, commit: bool = True) -> np.ndarray:
    raw = delta.astype("<f4").tobytes()
    parts = [raw[i:i + chunk] for i in range(0, len(raw), chunk)]  # splits mid-value/mid-block
    sent = np.frombuffer(b"".join(feedback.parts(fragment_id, step, 2, delta.size, parts)), "<f4")
    if commit:
        feedback.commit(fragment_id, step)
    return _decode("int4", sent.copy())


@pytest.mark.parametrize("residual_dtype", ("bf16", "f32"))
def test_error_feedback_carries_the_residual_and_bounds_the_summed_error(residual_dtype):
    feedback = PushErrorFeedback(DEFAULT_I4_BLOCK, residual_dtype)
    rng = np.random.default_rng(0)
    deltas = [(rng.standard_normal(1001) * 0.01).astype(np.float32) for _ in range(20)]
    received = [_push(feedback, 1, 1 + 2 * k, d) for k, d in enumerate(deltas)]
    plain = [_decode("int4", d) for d in deltas]
    true_sum = np.sum(deltas, axis=0)
    ef_drift = np.linalg.norm(np.sum(received, axis=0) - true_sum)
    plain_drift = np.linalg.norm(np.sum(plain, axis=0) - true_sum)
    assert ef_drift < plain_drift / 3
    # Telescoping: the summed error is just the last residual (one push's
    # rounding), up to the residual's own bf16 rounding.
    _, stored = feedback._residuals[1]
    residual = stored if residual_dtype == "f32" else (stored.astype(np.uint32) << 16).view(np.float32)
    assert ef_drift == pytest.approx(np.linalg.norm(residual), rel=0.05)


def test_error_feedback_replays_failed_pushes_and_drops_residuals_after_skipped_rounds():
    feedback = PushErrorFeedback(DEFAULT_I4_BLOCK)
    rng = np.random.default_rng(1)
    d1, d2 = ((rng.standard_normal(300) * 0.01).astype(np.float32) for _ in range(2))
    _push(feedback, 0, 1, d1)
    first = _push(feedback, 0, 3, d2, commit=False)
    feedback.discard(0)  # the push was not queued
    assert np.array_equal(_push(feedback, 0, 3, d2), first)  # replay sends the same bytes
    # Fragment 0 skips step 5 (catch-up): step 7 must not carry step 3's residual.
    assert np.array_equal(_push(feedback, 0, 7, d1), _decode("int4", d1))
    with pytest.raises(RuntimeError, match="no staged residual"):
        feedback.commit(0, 9)
    with pytest.raises(ValueError, match="expected 300"):
        list(feedback.parts(0, 9, 2, 300, [d1[:10].tobytes()]))
    feedback.reset()
    assert feedback.resident_bytes() == {}
    with pytest.raises(ValueError, match="residual dtype"):
        PushErrorFeedback(DEFAULT_I4_BLOCK, "fp8")


@pytest.mark.parametrize(("residual_dtype", "width"), (("bf16", 2), ("f32", 4)))
def test_error_feedback_memory_is_each_ranks_own_shard(residual_dtype, width):
    """Fragments map to one owner rank each; the residual held for a rank is
    its own shard's values times the residual width, never the full model."""
    numels = {0: 1000, 1: 700, 2: 1000, 3: 300}
    owner = {0: 0, 1: 0, 2: 1, 3: 1}  # fragment -> owner rank
    feedback = PushErrorFeedback(DEFAULT_I4_BLOCK, residual_dtype)
    for fragment_id, numel in numels.items():
        _push(feedback, fragment_id, 1 + fragment_id, np.full(numel, 0.01, np.float32))
    per_rank: dict[int, int] = {}
    for fragment_id, nbytes in feedback.resident_bytes().items():
        per_rank[owner[fragment_id]] = per_rank.get(owner[fragment_id], 0) + nbytes
    assert per_rank == {0: 1700 * width, 1: 1300 * width}
    # A push in flight stages one more copy of that fragment only.
    list(feedback.parts(2, 5, 2, numels[2], [np.zeros(numels[2], np.float32).tobytes()]))
    assert feedback.resident_bytes()[2] == 2 * numels[2] * width
    assert feedback.resident_bytes()[0] == numels[0] * width


class _FakeClient:
    """Consumes push parts like ``SyncerClient`` and records what it sent."""

    def __init__(self) -> None:
        self.outcomes: list[object] = []
        self.sent: list[np.ndarray] = []

    def push_fragment_parts(self, *args):
        parts = args[-1]
        data = b"".join(bytes(part) for part in parts)
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if isinstance(outcome, BaseException):
            raise outcome
        self.sent.append(np.frombuffer(data, "<f4").copy())
        return outcome


def _ef_role_stream(delta: np.ndarray):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        MilesFullParameterRoleStream,
        RoleSubmissionCandidate,
    )

    config = replace(_role_stream_config("actor", 30100, sweep=True),
                     push_dtype="int4", push_int4_error_feedback="bf16")
    stream = MilesFullParameterRoleStream(config)
    client = _FakeClient()
    stream.client = client
    fragment = SimpleNamespace(numel=delta.size)
    stream.adapter = SimpleNamespace(
        layout=SimpleNamespace(fragments=SimpleNamespace(fragments=[fragment, fragment])),
        delta_parts_from_authoritative=lambda *_a, **_k: lambda: iter([delta.tobytes()]),
    )
    cut, anchor = object(), object()
    stream._pinned_cuts[id(cut)] = (cut, 1)
    stream._pinned_anchors[id(anchor)] = (anchor, 1)

    def candidate(step: int):
        from yeto.protocol import PullRequest

        permit = PullRequest(0, step, 1)
        stream._pending[0] = permit
        return RoleSubmissionCandidate("actor", permit, max(0, step - 2), step, 1, 7, delta.size * 4,
                                       cut, anchor)

    return stream, client, candidate


def test_role_stream_pushes_through_error_feedback_and_commits_only_queued_pushes():
    from yeto.protocol import PartialMessageGenerationLost

    delta = (np.random.default_rng(3).standard_normal(500) * 0.01).astype(np.float32)
    stream, client, candidate = _ef_role_stream(delta)
    assert stream.error_feedback is not None
    assert stream.push_candidate(candidate(1)) is not None
    assert np.array_equal(client.sent[0], delta)  # nothing carried yet
    residual = delta - _decode("int4", delta)
    client.outcomes = [PartialMessageGenerationLost(connection_generation=1, connection_epoch=1,
                                                    queued_chunks=1, operation="PUSH_FRAGMENT"),
                       False]
    with pytest.raises(PartialMessageGenerationLost):
        stream.push_candidate(candidate(3))
    assert stream.push_candidate(candidate(3)) is None
    assert stream.push_candidate(candidate(3)) is not None
    # The committed push carried step 1's residual (stored as bf16).
    assert np.allclose(client.sent[-1], delta + residual, rtol=0, atol=np.abs(residual).max() / 128)
    assert stream.error_feedback.resident_bytes() == {0: delta.size * 2}
