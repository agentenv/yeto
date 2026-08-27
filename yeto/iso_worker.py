"""Persistent Torch worker for full-matrix IsoLoCo spectrum flattening.

The worker reads and writes binary frames on stdin/stdout.  All integers and
f32 values are little-endian; no text is ever written to stdout.

Every request and response starts with the same 48-byte little-endian header::

    bytes[8] magic = b"YETOISO1"
    u32 protocol_version = 1
    u32 code
    u64 request_id
    u64 rows
    u64 cols
    u64 payload_bytes

Request payload::

    code = 1
    f32[rows * cols] row_major_matrix

Response payload::

    code = 0 on success, nonzero on error
    bytes[...] result_or_utf8_error

On success, ``code`` is zero and the trailing payload is one row-major
f32 matrix.  On error, the trailing payload is a bounded UTF-8 diagnostic.
Error responses echo request ID, rows, and columns whenever a complete header
was available.  Fully received malformed frames are rejected without
terminating the persistent worker; a truncated or over-limit stream is fatal
because its next frame boundary cannot be recovered safely.
"""

from __future__ import annotations

import argparse
import enum
import struct
import sys
from dataclasses import dataclass
from typing import BinaryIO, Sequence

import torch


PROTOCOL_VERSION = 1
FRAME_MAGIC = b"YETOISO1"
ISO_OPCODE = 1

FRAME_HEADER = struct.Struct("<8sIIQQQQ")
F32_BYTES = 4
MAX_U64 = (1 << 64) - 1
MAX_ERROR_BYTES = 4096

# Qwen3.8-27B's largest Iso matrix is about 680 MiB in f32.  The limit is a
# guard against corrupt frame prefixes and remains configurable for other
# models.
DEFAULT_MAX_FRAME_BYTES = 2 * 1024**3


class Status(enum.IntEnum):
    """Response status codes used by the binary protocol."""

    OK = 0
    MALFORMED_FRAME = 1
    UNSUPPORTED_PROTOCOL = 2
    INVALID_SHAPE = 3
    INVALID_LENGTH = 4
    NONFINITE_INPUT = 5
    COMPUTE_ERROR = 6
    NONFINITE_OUTPUT = 7
    FRAME_TOO_LARGE = 8


class WorkerRequestError(ValueError):
    """A recoverable error in one completely received request frame."""

    def __init__(
        self,
        status: Status,
        message: str,
        request_id: int = 0,
        rows: int = 0,
        cols: int = 0,
    ):
        super().__init__(message)
        self.status = status
        self.request_id = request_id
        self.rows = rows
        self.cols = cols


class TruncatedStreamError(EOFError):
    """The stream ended in the middle of a frame."""


@dataclass(frozen=True)
class IsoResponse:
    """Decoded response returned by :func:`read_response`."""

    request_id: int
    status: Status
    rows: int
    cols: int
    matrix: torch.Tensor | None = None
    error: str | None = None


def _check_request_id(request_id: int) -> None:
    if not isinstance(request_id, int) or isinstance(request_id, bool):
        raise TypeError("request_id must be an integer")
    if not 0 <= request_id <= MAX_U64:
        raise ValueError("request_id must fit in an unsigned 64-bit integer")


def _matrix_numel(rows: int, cols: int, *, request_id: int) -> int:
    if rows <= 0 or cols <= 0:
        raise WorkerRequestError(
            Status.INVALID_SHAPE,
            f"matrix dimensions must be positive, got rows={rows}, cols={cols}",
            request_id,
            rows,
            cols,
        )
    if rows > MAX_U64 // cols or rows * cols > MAX_U64 // F32_BYTES:
        raise WorkerRequestError(
            Status.INVALID_SHAPE,
            "matrix byte length overflows unsigned 64-bit framing",
            request_id,
            rows,
            cols,
        )
    numel = rows * cols
    if numel > torch.iinfo(torch.int64).max:
        raise WorkerRequestError(
            Status.INVALID_SHAPE,
            "matrix element count exceeds signed 64-bit tensor indexing",
            request_id,
            rows,
            cols,
        )
    return numel


def iso_flatten_spectrum(
    matrix: torch.Tensor,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Apply exact full-matrix Iso spectrum flattening with Torch SVD.

    The wire input and all spectral math are f32: this calls
    ``torch.linalg.svd(..., full_matrices=False)`` on the configured device
    without promoting the matrix.  If ``A = U diag(s) Vh`` and
    ``k = min(rows, cols)``, retained singular values (strictly greater than
    ``1e-10 * max(s)``) are replaced by ``mean(s)``.  Singular values at or
    below the cutoff remain zero, so the numerical rank cannot increase.  The
    returned tensor is contiguous, row-major CPU f32.
    """

    if matrix.dtype != torch.float32:
        raise TypeError(f"matrix must have dtype torch.float32, got {matrix.dtype}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got ndim={matrix.ndim}")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"matrix dimensions must be positive, got {tuple(matrix.shape)}")
    if not matrix.is_contiguous():
        raise ValueError("matrix must be contiguous row-major f32")

    target = torch.device(device)
    with torch.inference_mode():
        work = matrix.detach().to(device=target, dtype=torch.float32)
        if not bool(torch.isfinite(work).all().item()):
            raise WorkerRequestError(Status.NONFINITE_INPUT, "matrix contains NaN or Inf")

        # Do not substitute a Gram approximation or blockwise decomposition:
        # Iso semantics require one thin SVD of the complete canonical matrix.
        u, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
        sigma_mean = singular_values.mean()
        sigma_max = singular_values.max()
        cutoff = sigma_max * 1.0e-10
        flattened = torch.where(
            singular_values > cutoff,
            sigma_mean,
            torch.zeros_like(singular_values),
        )
        result = (u * flattened.unsqueeze(0)) @ vh

        if not bool(torch.isfinite(result).all().item()):
            raise WorkerRequestError(Status.NONFINITE_OUTPUT, "Iso result contains NaN or Inf")
        return result.to(device="cpu", dtype=torch.float32).contiguous()


def encode_request(
    request_id: int,
    matrix: torch.Tensor,
    *,
    check_finite: bool = True,
) -> bytes:
    """Encode one request frame for clients and tests.

    Production callers may stream the same documented header and matrix bytes
    directly to avoid materializing this convenience function's combined
    ``bytes`` object.
    """

    _check_request_id(request_id)
    if matrix.dtype != torch.float32:
        raise TypeError(f"matrix must have dtype torch.float32, got {matrix.dtype}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got ndim={matrix.ndim}")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"matrix dimensions must be positive, got {tuple(matrix.shape)}")
    if matrix.device.type != "cpu":
        raise ValueError("encode_request requires a CPU tensor")
    if not matrix.is_contiguous():
        raise ValueError("matrix must be contiguous row-major f32")
    if check_finite and not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("matrix contains NaN or Inf")

    rows, cols = matrix.shape
    payload_len = matrix.numel() * F32_BYTES
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        PROTOCOL_VERSION,
        ISO_OPCODE,
        request_id,
        rows,
        cols,
        payload_len,
    )
    matrix_bytes = matrix.numpy().tobytes(order="C")
    return header + matrix_bytes


def _read_exact(stream: BinaryIO, size: int, *, allow_clean_eof: bool = False) -> bytearray | None:
    """Read exactly ``size`` bytes without assuming one ``read`` fills them."""

    if size < 0:
        raise ValueError("read size must be non-negative")
    if size == 0:
        return bytearray()

    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    readinto = getattr(stream, "readinto", None)
    while offset < size:
        if callable(readinto):
            count = readinto(view[offset:])
            chunk = None
        else:
            chunk = stream.read(size - offset)
            count = len(chunk) if chunk else 0
        if not count:
            if offset == 0 and allow_clean_eof:
                return None
            raise TruncatedStreamError(f"stream ended after {offset} of {size} required bytes")
        if chunk is not None:
            view[offset : offset + count] = chunk
        offset += count
    return data


def _write_all(stream: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None:
            # Buffered binary streams conventionally return the byte count, but
            # accepting None keeps the helper compatible with write-all streams.
            return
        if written <= 0:
            raise BrokenPipeError("binary output stream accepted no bytes")
        offset += written


def _bounded_error_bytes(message: str) -> bytes:
    encoded = message.encode("utf-8", errors="strict")
    if len(encoded) <= MAX_ERROR_BYTES:
        return encoded
    # Decode/re-encode after slicing so a multibyte code point is never split.
    return encoded[:MAX_ERROR_BYTES].decode("utf-8", errors="ignore").encode("utf-8")


def _write_error(
    stream: BinaryIO,
    request_id: int,
    rows: int,
    cols: int,
    status: Status,
    message: str,
) -> None:
    error_bytes = _bounded_error_bytes(message)
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        PROTOCOL_VERSION,
        int(status),
        request_id,
        rows,
        cols,
        len(error_bytes),
    )
    _write_all(stream, header)
    _write_all(stream, error_bytes)
    stream.flush()


def _write_success(stream: BinaryIO, request_id: int, matrix: torch.Tensor) -> None:
    if matrix.dtype != torch.float32 or matrix.device.type != "cpu" or not matrix.is_contiguous():
        raise RuntimeError("worker produced a non-contiguous or non-CPU f32 result")
    rows, cols = matrix.shape
    matrix_view = memoryview(matrix.numpy()).cast("B")
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        PROTOCOL_VERSION,
        int(Status.OK),
        request_id,
        rows,
        cols,
        len(matrix_view),
    )
    _write_all(stream, header)
    _write_all(stream, matrix_view)
    stream.flush()


def _parse_request(header: bytearray, payload: bytearray) -> tuple[int, int, int, torch.Tensor]:
    magic, version, opcode, request_id, rows, cols, payload_len = FRAME_HEADER.unpack(header)
    if magic != FRAME_MAGIC:
        raise WorkerRequestError(
            Status.MALFORMED_FRAME, "invalid request magic", request_id, rows, cols
        )
    if version != PROTOCOL_VERSION:
        raise WorkerRequestError(
            Status.UNSUPPORTED_PROTOCOL,
            f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}",
            request_id,
            rows,
            cols,
        )
    if opcode != ISO_OPCODE:
        raise WorkerRequestError(
            Status.UNSUPPORTED_PROTOCOL,
            f"unsupported opcode {opcode}; expected {ISO_OPCODE}",
            request_id,
            rows,
            cols,
        )

    numel = _matrix_numel(rows, cols, request_id=request_id)
    expected = numel * F32_BYTES
    if payload_len != expected or len(payload) != expected:
        raise WorkerRequestError(
            Status.INVALID_LENGTH,
            f"request payload has declared/actual length {payload_len}/{len(payload)}, expected exactly {expected}",
            request_id,
            rows,
            cols,
        )

    matrix = torch.frombuffer(payload, dtype=torch.float32, count=numel).reshape(rows, cols)
    return request_id, rows, cols, matrix


def read_response(
    stream: BinaryIO,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> IsoResponse:
    """Read and strictly decode one response frame from ``stream``."""

    header = _read_exact(stream, FRAME_HEADER.size)
    assert header is not None
    magic, version, raw_status, request_id, rows, cols, payload_size = FRAME_HEADER.unpack(header)
    if payload_size > max_frame_bytes:
        raise ValueError(f"response frame has {payload_size} bytes, limit is {max_frame_bytes}")
    payload = _read_exact(stream, payload_size)
    assert payload is not None
    if magic != FRAME_MAGIC:
        raise ValueError("invalid response magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported response protocol version {version}")
    try:
        status = Status(raw_status)
    except ValueError as exc:
        raise ValueError(f"unknown response status {raw_status}") from exc

    if status != Status.OK:
        if payload_size > MAX_ERROR_BYTES:
            raise ValueError(
                f"error response has {payload_size} bytes, limit is {MAX_ERROR_BYTES}"
            )
        try:
            error = bytes(payload).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("error response is not valid UTF-8") from exc
        return IsoResponse(request_id, status, rows, cols, error=error)

    try:
        numel = _matrix_numel(rows, cols, request_id=request_id)
    except WorkerRequestError as exc:
        raise ValueError(str(exc)) from exc
    expected_bytes = numel * F32_BYTES
    if len(payload) != expected_bytes:
        raise ValueError(f"response matrix has {len(payload)} bytes, expected {expected_bytes}")
    matrix = torch.frombuffer(payload, dtype=torch.float32, count=numel).reshape(rows, cols).clone()
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("response matrix contains NaN or Inf")
    return IsoResponse(request_id, status, rows, cols, matrix=matrix)


def serve(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    *,
    device: str | torch.device,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> int:
    """Serve requests until clean EOF; return a process-style exit code."""

    if sys.byteorder != "little":
        raise RuntimeError("the Iso worker currently requires a little-endian host")
    if max_frame_bytes < F32_BYTES:
        raise ValueError("max_frame_bytes must permit at least one f32 value")

    target = torch.device(device)
    # Resolve device errors before accepting the first potentially large frame.
    torch.empty(0, dtype=torch.float32, device=target)

    while True:
        request_id = 0
        rows = 0
        cols = 0
        header_received = False
        matrix: torch.Tensor | None = None
        try:
            header = _read_exact(input_stream, FRAME_HEADER.size, allow_clean_eof=True)
            if header is None:
                return 0
            header_received = True
            _, _, _, request_id, rows, cols, payload_size = FRAME_HEADER.unpack(header)
            if payload_size > max_frame_bytes:
                _write_error(
                    output_stream,
                    request_id,
                    rows,
                    cols,
                    Status.FRAME_TOO_LARGE,
                    f"request frame has {payload_size} bytes, limit is {max_frame_bytes}",
                )
                return 2
            payload = _read_exact(input_stream, payload_size)
            assert payload is not None
        except TruncatedStreamError as exc:
            if header_received:
                _write_error(
                    output_stream,
                    request_id,
                    rows,
                    cols,
                    Status.INVALID_LENGTH,
                    str(exc),
                )
            print(f"iso-worker: truncated input: {exc}", file=sys.stderr, flush=True)
            return 2

        try:
            request_id, rows, cols, matrix = _parse_request(header, payload)
            result = iso_flatten_spectrum(matrix, device=target)
        except WorkerRequestError as exc:
            _write_error(
                output_stream,
                request_id,
                rows,
                cols,
                exc.status,
                str(exc),
            )
            matrix = None
            del payload
            continue
        except RuntimeError as exc:
            # RuntimeError includes backend/device failures such as CUDA OOM.
            # The complete response remains framed, and stderr retains details
            # without corrupting stdout's binary protocol.
            message = f"{type(exc).__name__}: {exc}"
            print(f"iso-worker request {request_id}: {message}", file=sys.stderr, flush=True)
            _write_error(
                output_stream, request_id, rows, cols, Status.COMPUTE_ERROR, message
            )
            matrix = None
            del payload
            continue

        _write_success(output_stream, request_id, result)
        # Do not retain a complete matrix payload/result while blocking for the
        # next request in this persistent process.
        del payload, matrix, result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent full-matrix Torch Iso worker")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device used for f32 SVD (default: cuda:0; use cpu for tests)",
    )
    parser.add_argument(
        "--max-frame-bytes",
        type=int,
        default=DEFAULT_MAX_FRAME_BYTES,
        help=f"maximum accepted request payload bytes (default: {DEFAULT_MAX_FRAME_BYTES})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return serve(
            sys.stdin.buffer,
            sys.stdout.buffer,
            device=args.device,
            max_frame_bytes=args.max_frame_bytes,
        )
    except BrokenPipeError:
        print("iso-worker: output pipe closed", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"iso-worker: fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
