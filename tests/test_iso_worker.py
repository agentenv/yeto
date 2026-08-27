from __future__ import annotations

import struct
import subprocess
import sys

import pytest
import torch

from yeto.iso_worker import (
    FRAME_HEADER,
    FRAME_MAGIC,
    ISO_OPCODE,
    MAX_U64,
    PROTOCOL_VERSION,
    IsoResponse,
    Status,
    encode_request,
    iso_flatten_spectrum,
    read_response,
)


F32_ORACLE_RTOL = 2e-5
F32_ORACLE_ATOL = 2e-5


def _f64_oracle(matrix: torch.Tensor) -> torch.Tensor:
    work = matrix.double()
    u, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    sigma_mean = singular_values.mean()
    cutoff = singular_values.max() * 1.0e-10
    flattened = torch.where(
        singular_values > cutoff,
        sigma_mean,
        torch.zeros_like(singular_values),
    )
    return ((u * flattened.unsqueeze(0)) @ vh).float()


def _assert_matches_oracle(matrix: torch.Tensor) -> torch.Tensor:
    actual = iso_flatten_spectrum(matrix, device="cpu")
    expected = _f64_oracle(matrix)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=F32_ORACLE_RTOL,
        atol=F32_ORACLE_ATOL,
    )
    return actual


def test_diagonal_flattens_retained_singular_values():
    matrix = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    actual = _assert_matches_oracle(matrix)
    torch.testing.assert_close(actual, torch.diag(torch.tensor([2.0, 2.0])))


def test_flatten_never_calls_svd(monkeypatch: pytest.MonkeyPatch):
    def forbidden_svd(*args, **kwargs):
        raise AssertionError("iso_flatten_spectrum must not decompose via SVD")

    monkeypatch.setattr(torch.linalg, "svd", forbidden_svd)
    result = iso_flatten_spectrum(torch.eye(2, dtype=torch.float32), device="cpu")

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.eye(2))


@pytest.mark.parametrize(
    "matrix",
    [
        torch.tensor(
            [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 0.25]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[1.0, 2.0, 0.5, -2.0], [3.0, -1.0, 4.0, 0.25]],
            dtype=torch.float32,
        ),
    ],
    ids=["tall", "wide"],
)
def test_tall_and_wide_matrices_match_f64_oracle(matrix: torch.Tensor):
    _assert_matches_oracle(matrix)


@pytest.mark.parametrize("shape", [(96, 160), (160, 96)], ids=["wide", "tall"])
def test_random_matrices_match_f64_oracle(shape: tuple[int, int]):
    generator = torch.Generator().manual_seed(20260827)
    matrix = torch.randn(shape, generator=generator, dtype=torch.float32)
    _assert_matches_oracle(matrix.contiguous())


def test_rank_deficient_matrix_preserves_rank():
    matrix = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    actual = _assert_matches_oracle(matrix)
    expected = torch.zeros_like(matrix)
    expected[0, 0] = 4.0 / 3.0
    expected[1, 1] = 4.0 / 3.0
    torch.testing.assert_close(
        actual,
        expected,
        rtol=F32_ORACLE_RTOL,
        atol=F32_ORACLE_ATOL,
    )
    assert int(torch.linalg.matrix_rank(actual.double()).item()) == 2


def test_zero_matrix_stays_zero():
    matrix = torch.zeros((5, 3), dtype=torch.float32)
    actual = _assert_matches_oracle(matrix)
    assert torch.equal(actual, matrix)


def test_already_weighted_average_is_the_only_svd_input():
    learner_a = torch.tensor(
        [[4.0, 1.0, 0.0], [0.0, 2.0, -1.0], [1.0, 0.0, 3.0]],
        dtype=torch.float32,
    )
    learner_b = torch.tensor(
        [[-1.0, 0.0, 2.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]],
        dtype=torch.float32,
    )
    weight_a, weight_b = 2.0, 5.0
    merged = (weight_a * learner_a + weight_b * learner_b) / (weight_a + weight_b)

    actual = _assert_matches_oracle(merged.contiguous())
    torch.testing.assert_close(
        actual,
        _f64_oracle(merged),
        rtol=F32_ORACLE_RTOL,
        atol=F32_ORACLE_ATOL,
    )


def _start_cpu_worker() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "yeto.iso_worker", "--device", "cpu"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _exchange(proc: subprocess.Popen[bytes], request: bytes) -> IsoResponse:
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(request)
    proc.stdin.flush()
    return read_response(proc.stdout)


def test_request_uses_exact_coordinated_48_byte_header():
    matrix = torch.tensor([[1.0]], dtype=torch.float32)
    encoded = encode_request(9, matrix)
    assert FRAME_HEADER.size == 48
    assert encoded[:48] == struct.pack(
        "<8sIIQQQQ",
        FRAME_MAGIC,
        PROTOCOL_VERSION,
        ISO_OPCODE,
        9,
        1,
        1,
        4,
    )
    assert encoded[48:] == struct.pack("<f", 1.0)


def test_protocol_roundtrip_is_persistent_and_echoes_request_ids():
    proc = _start_cpu_worker()
    try:
        # This is exactly the startup probe the Rust coordinator sends.
        first = torch.tensor([[1.0]], dtype=torch.float32)
        first_response = _exchange(proc, encode_request(17, first))
        assert first_response.request_id == 17
        assert first_response.status == Status.OK
        assert first_response.error is None
        assert first_response.matrix is not None
        torch.testing.assert_close(
            first_response.matrix,
            _f64_oracle(first),
            rtol=F32_ORACLE_RTOL,
            atol=F32_ORACLE_ATOL,
        )

        # A second request through the same process proves the worker remains
        # alive and frame boundaries are deterministic.
        second = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        second_response = _exchange(proc, encode_request(18, second))
        assert second_response.request_id == 18
        assert second_response.status == Status.OK
        assert second_response.matrix is not None
        torch.testing.assert_close(
            second_response.matrix,
            _f64_oracle(second),
            rtol=F32_ORACLE_RTOL,
            atol=F32_ORACLE_ATOL,
        )

        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
        assert proc.stdout is not None
        assert proc.stdout.read() == b""
        assert proc.stderr is not None
        assert proc.stderr.read() == b""
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_protocol_rejects_nonfinite_input_and_continues():
    proc = _start_cpu_worker()
    try:
        invalid = torch.tensor([[1.0, float("nan")]], dtype=torch.float32)
        error = _exchange(proc, encode_request(41, invalid, check_finite=False))
        assert error.request_id == 41
        assert error.status == Status.NONFINITE_INPUT
        assert error.matrix is None
        assert error.error == "matrix contains NaN or Inf"

        valid = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        recovered = _exchange(proc, encode_request(42, valid))
        assert recovered.request_id == 42
        assert recovered.status == Status.OK
        assert recovered.matrix is not None
        torch.testing.assert_close(
            recovered.matrix,
            _f64_oracle(valid),
            rtol=F32_ORACLE_RTOL,
            atol=F32_ORACLE_ATOL,
        )

        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_protocol_rejects_wrong_payload_length_and_continues():
    proc = _start_cpu_worker()
    try:
        malformed = FRAME_HEADER.pack(
            FRAME_MAGIC,
            PROTOCOL_VERSION,
            ISO_OPCODE,
            51,
            1,
            2,
            4,
        ) + struct.pack("<f", 1.0)
        error = _exchange(proc, malformed)
        assert error.request_id == 51
        assert (error.rows, error.cols) == (1, 2)
        assert error.status == Status.INVALID_LENGTH
        assert error.error is not None
        assert "expected exactly 8" in error.error

        overflow = FRAME_HEADER.pack(
            FRAME_MAGIC,
            PROTOCOL_VERSION,
            ISO_OPCODE,
            52,
            MAX_U64,
            2,
            0,
        )
        overflow_error = _exchange(proc, overflow)
        assert overflow_error.request_id == 52
        assert (overflow_error.rows, overflow_error.cols) == (MAX_U64, 2)
        assert overflow_error.status == Status.INVALID_SHAPE
        assert overflow_error.error == "matrix byte length overflows unsigned 64-bit framing"

        valid = torch.tensor([[1.0]], dtype=torch.float32)
        recovered = _exchange(proc, encode_request(53, valid))
        assert recovered.request_id == 53
        assert recovered.status == Status.OK
        assert recovered.matrix is not None
        assert torch.equal(recovered.matrix, valid)

        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
