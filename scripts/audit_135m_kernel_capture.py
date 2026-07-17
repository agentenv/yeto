#!/usr/bin/env python3
"""Stream and seal the registered full finite pseudo-gradient kernel.

The syncer writes an exact state checkpoint immediately before every captured
outer commit.  For the registered ``mu=0`` cells, the difference between two
successive states is the applied outer update and therefore ``-eta`` times the
production merged pseudo-gradient.  This sidecar keeps only two state cuts in
memory, deletes the unused learner candidates as they arrive, and appends each
reconstructed update to a per-fragment scratch matrix.

After the final syncer checkpoint is available, the scratch matrices are read
once in coordinate blocks.  FFT autocorrelation yields every finite lag, not a
truncated lag summary.  The raw energy-weighted kernel is projected to a
positive-semidefinite correlation matrix by eigenvalue clipping and diagonal
renormalization.  Both raw and regularized kernels, exact coverage, ordered
update hashes, and ``V_H`` are sealed in one compact JSON artifact; all large
capture and scratch files are then removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CKPT_MAGIC = 0xD170_5A7E
SCHEMA = "audit_135m_finite_kernel_capture_v1"
PSD_EIGENVALUE_FLOOR = 1.0e-8


class KernelCaptureError(RuntimeError):
    """A capture is incomplete, reordered, or numerically invalid."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _state_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise KernelCaptureError(f"malformed capture-state name: {path.name}") from exc


def parse_state(path: Path) -> tuple[int, list[np.ndarray]]:
    """Return ``(global_step, fragment parameter arrays)`` from a syncer state."""

    data = path.read_bytes()
    offset = 0
    if len(data) < 16:
        raise KernelCaptureError(f"truncated syncer checkpoint: {path}")
    (magic,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if magic != CKPT_MAGIC:
        raise KernelCaptureError(f"bad syncer checkpoint magic in {path}")
    (global_step,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    (fragment_count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    fragments: list[np.ndarray] = []
    for fragment in range(fragment_count):
        if offset + 16 > len(data):
            raise KernelCaptureError(
                f"truncated fragment header {fragment} in {path}"
            )
        _version, numel = struct.unpack_from("<QQ", data, offset)
        offset += 16
        byte_count = int(numel) * 4
        if byte_count <= 0 or offset + 2 * byte_count > len(data):
            raise KernelCaptureError(
                f"invalid fragment payload {fragment} in {path}"
            )
        params = np.frombuffer(
            data, dtype="<f4", count=int(numel), offset=offset
        ).copy()
        offset += byte_count
        offset += byte_count  # Momentum buffer; not needed for registered mu=0.
        fragments.append(params)
    return int(global_step), fragments


def _load_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise KernelCaptureError(
                    f"{path}:{line_number}: invalid capture index JSON"
                ) from exc
            if not isinstance(value, dict):
                raise KernelCaptureError(
                    f"{path}:{line_number}: capture index row is not an object"
                )
            rows.append(value)
    return rows


def _load_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise KernelCaptureError(
                    f"{path}:{line_number}: invalid {label} JSON"
                ) from exc
            if not isinstance(value, dict):
                raise KernelCaptureError(
                    f"{path}:{line_number}: {label} row is not an object"
                )
            rows.append(value)
    return rows


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _fft_dot_autocorrelation(
    matrix_path: Path,
    *,
    rows: int,
    columns: int,
    coordinate_chunk: int,
) -> np.ndarray:
    if matrix_path.stat().st_size != rows * columns * 4:
        raise KernelCaptureError(
            f"scratch matrix size differs for {matrix_path.name}"
        )
    matrix = np.memmap(
        matrix_path, dtype="<f4", mode="r", shape=(rows, columns)
    )
    nfft = _next_power_of_two(2 * rows)
    autocorrelation = np.zeros(rows, dtype=np.float64)
    for start in range(0, columns, coordinate_chunk):
        stop = min(columns, start + coordinate_chunk)
        block = np.asarray(matrix[:, start:stop], dtype=np.float64)
        transformed = np.fft.rfft(block, n=nfft, axis=0)
        recovered = np.fft.irfft(
            transformed * np.conjugate(transformed), n=nfft, axis=0
        )
        autocorrelation += recovered[:rows].sum(axis=1, dtype=np.float64)
    del matrix
    return autocorrelation


def _regularize_kernel(raw_rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    count = int(raw_rho.shape[0])
    indices = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :])
    toeplitz = raw_rho[indices]
    toeplitz = (toeplitz + toeplitz.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(toeplitz)
    clipped_count = int(np.count_nonzero(eigenvalues < PSD_EIGENVALUE_FLOOR))
    clipped = np.maximum(eigenvalues, PSD_EIGENVALUE_FLOOR)
    regularized = (eigenvectors * clipped) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(regularized), PSD_EIGENVALUE_FLOOR))
    regularized = regularized / diagonal[:, None] / diagonal[None, :]
    regularized = (regularized + regularized.T) * 0.5
    rho = np.array(
        [
            1.0
            if lag == 0
            else float(np.diag(regularized, k=lag).mean(dtype=np.float64))
            for lag in range(count)
        ],
        dtype=np.float64,
    )
    rho[0] = 1.0
    return rho, regularized, clipped_count


def _kernel_summary(
    matrix_path: Path,
    *,
    fragment: int,
    update_count: int,
    numel: int,
    norms: Sequence[float],
    coordinate_chunk: int,
) -> dict[str, Any]:
    if len(norms) != update_count or any(
        not math.isfinite(value) or value <= 0.0 for value in norms
    ):
        raise KernelCaptureError(
            f"fragment {fragment} update norms are incomplete/nonpositive"
        )
    numerator = _fft_dot_autocorrelation(
        matrix_path,
        rows=update_count,
        columns=numel,
        coordinate_chunk=coordinate_chunk,
    )
    norm_array = np.asarray(norms, dtype=np.float64)
    denominator = np.array(
        [
            float(np.dot(norm_array[lag:], norm_array[: update_count - lag]))
            for lag in range(update_count)
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(numerator)) or np.any(denominator <= 0.0):
        raise KernelCaptureError(
            f"fragment {fragment} finite-kernel accumulators are invalid"
        )
    raw_rho = numerator / denominator
    raw_rho[0] = 1.0
    regularized_rho, regularized_matrix, clipped_count = _regularize_kernel(raw_rho)
    weights = np.arange(update_count, 0, -1, dtype=np.float64)
    raw_v = float(update_count + 2.0 * np.dot(weights[1:], raw_rho[1:]))
    regularized_v = float(np.ones(update_count) @ regularized_matrix @ np.ones(update_count))
    if not math.isfinite(regularized_v) or regularized_v <= 0.0:
        raise KernelCaptureError(f"fragment {fragment} regularized V is not positive")
    return {
        "fragment": fragment,
        "update_count": update_count,
        "parameter_count": numel,
        "energy_weighted_rho_raw": [float(value) for value in raw_rho],
        "energy_weighted_rho_psd": [float(value) for value in regularized_rho],
        "lag_pair_counts": [update_count - lag for lag in range(update_count)],
        "dot_numerators": [float(value) for value in numerator],
        "norm_product_denominators": [float(value) for value in denominator],
        "raw_v": raw_v,
        "regularized_v": regularized_v,
        "minimum_raw_toeplitz_eigenvalue": float(
            np.linalg.eigvalsh(raw_rho[
                np.abs(
                    np.arange(update_count)[:, None]
                    - np.arange(update_count)[None, :]
                )
            ]).min()
        ),
        "psd_eigenvalue_floor": PSD_EIGENVALUE_FLOOR,
        "psd_clipped_eigenvalue_count": clipped_count,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_outer_steps <= 0 or args.fragment_count <= 0:
        raise KernelCaptureError("expected steps/fragments must be positive")
    if args.expected_outer_steps % args.fragment_count:
        raise KernelCaptureError("outer steps do not divide evenly across fragments")
    expected_per_fragment = args.expected_outer_steps // args.fragment_count
    capture_dir = args.capture_dir.resolve()
    states_dir = capture_dir / "states"
    candidates_dir = capture_dir / "candidates"
    scratch = args.scratch_dir.resolve()
    if scratch.exists():
        raise KernelCaptureError(f"refusing to reuse kernel scratch directory: {scratch}")
    scratch.mkdir(parents=True)

    previous_step: int | None = None
    previous_fragments: list[np.ndarray] | None = None
    fragment_numel: list[int] | None = None
    update_counts = [0] * args.fragment_count
    update_norms: list[list[float]] = [[] for _ in range(args.fragment_count)]
    update_registry: list[dict[str, Any]] = []
    matrix_paths = [scratch / f"fragment-{index:04d}.f32" for index in range(args.fragment_count)]
    matrix_handles = [path.open("xb") for path in matrix_paths]
    started_at = time.time()
    last_status = 0.0

    def delete_candidates() -> None:
        if candidates_dir.is_dir():
            for candidate in candidates_dir.glob("*.f32"):
                candidate.unlink(missing_ok=True)

    def record_transition(
        *,
        step: int,
        before: Sequence[np.ndarray],
        after: Sequence[np.ndarray],
    ) -> None:
        if len(before) != args.fragment_count or len(after) != args.fragment_count:
            raise KernelCaptureError("capture state fragment count differs")
        changed = [
            fragment
            for fragment, (left, right) in enumerate(zip(before, after))
            if left.shape != right.shape or not np.array_equal(left, right)
        ]
        if len(changed) != 1:
            raise KernelCaptureError(
                f"capture transition at step {step} changed {changed}, expected one fragment"
            )
        fragment = changed[0]
        if before[fragment].shape != after[fragment].shape:
            raise KernelCaptureError(f"fragment shape changed at step {step}")
        update = np.subtract(
            after[fragment], before[fragment], dtype=np.float32
        )
        update_bytes = update.astype("<f4", copy=False).tobytes(order="C")
        matrix_handles[fragment].write(update_bytes)
        norm = float(np.linalg.norm(update.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 0.0:
            raise KernelCaptureError(f"step {step} has a nonpositive update norm")
        update_counts[fragment] += 1
        update_norms[fragment].append(norm)
        update_registry.append(
            {
                "step": step,
                "fragment": fragment,
                "fragment_update_index": update_counts[fragment] - 1,
                "applied_update_f32_sha256": hashlib.sha256(update_bytes).hexdigest(),
                "applied_update_norm": norm,
            }
        )

    try:
        while True:
            delete_candidates()
            files = (
                sorted(states_dir.glob("state_before_step_*.ckpt"), key=_state_step)
                if states_dir.is_dir()
                else []
            )
            progressed = False
            for state_path in files:
                step = _state_step(state_path)
                if previous_step is not None and step <= previous_step:
                    state_path.unlink(missing_ok=True)
                    continue
                if previous_step is not None and step != previous_step + 1:
                    raise KernelCaptureError(
                        f"capture state sequence jumped from {previous_step} to {step}"
                    )
                global_step, fragments = parse_state(state_path)
                if global_step != step - 1:
                    raise KernelCaptureError(
                        f"state-before-step {step} reports global step {global_step}"
                    )
                if len(fragments) != args.fragment_count:
                    raise KernelCaptureError("capture checkpoint fragment count differs")
                if fragment_numel is None:
                    fragment_numel = [int(value.size) for value in fragments]
                    expected_scratch = (
                        sum(fragment_numel) * expected_per_fragment * 4
                    )
                    free = shutil.disk_usage(scratch).free
                    state_bytes = state_path.stat().st_size
                    required = int(expected_scratch * 1.10 + state_bytes * 3 + 10e9)
                    if free < required:
                        raise KernelCaptureError(
                            "insufficient disk for lossless finite-kernel scratch: "
                            f"need {required}, have {free}"
                        )
                elif [int(value.size) for value in fragments] != fragment_numel:
                    raise KernelCaptureError("capture fragment layout changed")
                if previous_fragments is not None and previous_step is not None:
                    record_transition(
                        step=previous_step,
                        before=previous_fragments,
                        after=fragments,
                    )
                previous_step = step
                previous_fragments = fragments
                state_path.unlink(missing_ok=True)
                progressed = True
            if args.done_file.is_file() and not progressed:
                if previous_step is None or previous_fragments is None:
                    raise KernelCaptureError("capture completed without a state sequence")
                if previous_step != args.expected_outer_steps:
                    raise KernelCaptureError(
                        f"last captured step is {previous_step}, expected {args.expected_outer_steps}"
                    )
                final_global_step, final_fragments = parse_state(args.final_checkpoint)
                if final_global_step != args.expected_outer_steps:
                    raise KernelCaptureError(
                        "final checkpoint global step differs from registered outer work"
                    )
                record_transition(
                    step=previous_step,
                    before=previous_fragments,
                    after=final_fragments,
                )
                break
            now = time.monotonic()
            if now - last_status >= 60.0:
                _write_json_atomic(
                    args.status,
                    {
                        "schema": "audit_135m_finite_kernel_capture_status_v1",
                        "phase": "STREAMING",
                        "last_state_step": previous_step,
                        "update_counts": update_counts,
                        "expected_outer_steps": args.expected_outer_steps,
                        "loss_exposed": False,
                    },
                )
                last_status = now
            time.sleep(args.poll_seconds)
    finally:
        for handle in matrix_handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    if fragment_numel is None:
        raise KernelCaptureError("capture fragment layout was never observed")
    if update_counts != [expected_per_fragment] * args.fragment_count:
        raise KernelCaptureError(
            f"per-fragment update coverage differs: {update_counts}"
        )
    if [row["step"] for row in update_registry] != list(
        range(1, args.expected_outer_steps + 1)
    ):
        raise KernelCaptureError("ordered update registry is incomplete/reordered")

    index_path = capture_dir / "index.jsonl"
    index_rows = _load_index(index_path)
    expected_index_rows = args.expected_outer_steps * args.learner_count
    if len(index_rows) != expected_index_rows:
        raise KernelCaptureError(
            f"capture index has {len(index_rows)} rows, expected {expected_index_rows}"
        )
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in index_rows:
        if row.get("schema") != "syncer_probe_capture_v1":
            raise KernelCaptureError("capture index schema differs")
        by_step.setdefault(int(row["step"]), []).append(row)
    if set(by_step) != set(range(1, args.expected_outer_steps + 1)):
        raise KernelCaptureError("capture index step coverage differs")
    for update in update_registry:
        rows = by_step[int(update["step"])]
        if (
            len(rows) != args.learner_count
            or {int(row["learner_id"]) for row in rows}
            != set(range(args.learner_count))
            or {int(row["fragment"]) for row in rows}
            != {int(update["fragment"])}
        ):
            raise KernelCaptureError(
                f"capture index learner/fragment coverage differs at step {update['step']}"
            )

    tape_rows = _load_jsonl_objects(args.event_tape, "event tape")
    tape_projection = [
        {"step": int(row["step"]), "fragment": int(row["fragment"])}
        for row in tape_rows
    ]
    update_projection = [
        {"step": int(row["step"]), "fragment": int(row["fragment"])}
        for row in update_registry
    ]
    if tape_projection != update_projection:
        raise KernelCaptureError(
            "capture state transitions do not match the registered event-tape order"
        )

    fragments = [
        _kernel_summary(
            matrix_paths[fragment],
            fragment=fragment,
            update_count=expected_per_fragment,
            numel=fragment_numel[fragment],
            norms=update_norms[fragment],
            coordinate_chunk=args.coordinate_chunk,
        )
        for fragment in range(args.fragment_count)
    ]
    payload = {
        "schema": SCHEMA,
        "status": "SEALED",
        "loss_exposed": False,
        "estimator": {
            "pseudo_gradient_reconstruction": (
                "mu=0 applied_update=-eta*production_merged_pseudo_gradient"
            ),
            "rho": "sum_dot_over_sum_norm_products_per_exact_finite_lag",
            "psd_regularization": (
                "symmetric_toeplitz_eigendecomposition_clip_1e-8_"
                "diagonal_renormalization_then_lag_diagonal_average"
            ),
            "fft_coordinate_chunk": args.coordinate_chunk,
        },
        "outer_eta": args.outer_eta,
        "pseudo_gradient_scale_from_applied_update": -1.0 / args.outer_eta,
        "expected_outer_steps": args.expected_outer_steps,
        "observed_outer_steps": len(update_registry),
        "fragment_count": args.fragment_count,
        "learner_count": args.learner_count,
        "updates_per_fragment": expected_per_fragment,
        "fragment_parameter_counts": fragment_numel,
        "ordered_update_registry": update_registry,
        "ordered_update_registry_hash": canonical_sha256(update_registry),
        "capture_index_row_count": len(index_rows),
        "capture_index_sha256": sha256_file(index_path),
        "event_tape_sha256": sha256_file(args.event_tape),
        "event_tape_step_fragment_hash": canonical_sha256(tape_projection),
        "fragments": fragments,
        "K_H": sum(row["update_count"] for row in fragments),
        "V_H_raw": sum(float(row["raw_v"]) for row in fragments),
        "V_H_psd": sum(float(row["regularized_v"]) for row in fragments),
        "state_transition_replay_exact": True,
        "all_registered_updates_covered": True,
        "candidate_payloads_used_by_estimator": False,
        "large_capture_cleanup_complete": True,
        "wall_seconds": time.time() - started_at,
    }
    payload["capture_canonical_sha256"] = canonical_sha256(payload)
    if args.output.exists():
        raise KernelCaptureError(f"refusing to overwrite kernel capture: {args.output}")
    _write_json_atomic(args.output, payload)
    shutil.rmtree(capture_dir, ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    _write_json_atomic(
        args.status,
        {
            "schema": "audit_135m_finite_kernel_capture_status_v1",
            "phase": "SEALED",
            "output_sha256": sha256_file(args.output),
            "loss_exposed": False,
        },
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--event-tape", type=Path, required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-outer-steps", type=int, required=True)
    parser.add_argument("--fragment-count", type=int, default=4)
    parser.add_argument("--learner-count", type=int, required=True)
    parser.add_argument("--outer-eta", type=float, required=True)
    parser.add_argument("--coordinate-chunk", type=int, default=32768)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not math.isfinite(args.outer_eta) or args.outer_eta <= 0.0:
            raise KernelCaptureError("outer eta must be finite and positive")
        if args.coordinate_chunk <= 0 or args.poll_seconds <= 0.0:
            raise KernelCaptureError("chunk/poll settings must be positive")
        result = capture(args)
    except (KernelCaptureError, OSError, ValueError, KeyError) as exc:
        print(f"finite-kernel capture error: {exc}", file=__import__("sys").stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "capture_canonical_sha256": result["capture_canonical_sha256"],
                "K_H": result["K_H"],
                "V_H_psd": result["V_H_psd"],
                "loss_exposed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
