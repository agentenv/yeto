#!/usr/bin/env python3
"""Fail-closed CPU replay and retained-tape audit for CRP-SGD.

CRP is evaluated only from exact f32 stock and proposal directions.  The
legacy BCMP shadow tape is audited separately because its scalar summaries do
not identify vector sums or transverse projections.  Historical PTI geometry
can be screened from exact syncer checkpoint differences; this is explicitly
off-policy direction evidence, not a finite-loss claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# Frozen before reading any empirical score produced by this script.
CRP_INDIVIDUAL_MAX_RATIO = 1.0 / 20.0
CRP_PULSE_MIN_RATIO = 1.0 / 20.0
CRP_PULSE_MAX_RATIO = 1.0 / 8.0
CRP_MAX_AGE = 8
CRP_MIN_CONTRIBUTORS = 2
CRP_MIN_OPPORTUNITIES = 32
CRP_MIN_OPPORTUNITIES_PER_FRAGMENT = 8
CRP_MIN_MEAN_SCORE = 0.001
CRP_MIN_POSITIVE_FRAGMENTS = 3
CRP_MIN_POSITIVE_PULSE_FRACTION = 0.60
CRP_MIN_ACTION_FRACTION = 0.25
CRP_BOOTSTRAP_BLOCK_LENGTH = 4
CRP_BOOTSTRAP_REPLICATES = 4096
CRP_BOOTSTRAP_ALPHA = 0.05
CRP_BOOTSTRAP_SEED = 0x435250

PTI_COEFFICIENTS = (
    -1.0 / 4.0,
    -1.0 / 8.0,
    -1.0 / 16.0,
    -1.0 / 32.0,
    0.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
)
PTI_INTERLOCK_LENGTH = 3
PTI_TRANSVERSE_NORM_SQ_MIN = 2.0**-40

CKPT_MAGIC = 0xD170_5A7E
EXACT_SCHEMA = "crp_exact_vectors_v1"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def policy_contract() -> dict:
    return {
        "individual_max_ratio_strict": CRP_INDIVIDUAL_MAX_RATIO,
        "pulse_min_ratio_inclusive": CRP_PULSE_MIN_RATIO,
        "pulse_max_ratio_inclusive": CRP_PULSE_MAX_RATIO,
        "max_source_age_inclusive": CRP_MAX_AGE,
        "min_contributors": CRP_MIN_CONTRIBUTORS,
        "decision_gates": {
            "min_opportunities": CRP_MIN_OPPORTUNITIES,
            "min_opportunities_per_fragment": CRP_MIN_OPPORTUNITIES_PER_FRAGMENT,
            "min_mean_score_strict": CRP_MIN_MEAN_SCORE,
            "min_positive_fragments": CRP_MIN_POSITIVE_FRAGMENTS,
            "min_positive_pulse_fraction": CRP_MIN_POSITIVE_PULSE_FRACTION,
            "min_action_fraction": CRP_MIN_ACTION_FRACTION,
            "bootstrap": {
                "alpha": CRP_BOOTSTRAP_ALPHA,
                "block_length": CRP_BOOTSTRAP_BLOCK_LENGTH,
                "replicates": CRP_BOOTSTRAP_REPLICATES,
                "seed": CRP_BOOTSTRAP_SEED,
            },
        },
        "timing": [
            "resolve previous same-fragment residual against current stock",
            "a failed resolution clears the existing bank",
            "form a pulse only from residuals admitted before this boundary",
            "append a newly admitted residual only after the pulse decision",
            "seal the current residual for the next same-fragment boundary",
        ],
        "fallback": "return the original stock f32 byte object without re-encoding",
    }


def _dot64(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    return float(np.einsum("i,i->", a, b, dtype=np.float64, optimize=False))


def _norm64(a: np.ndarray) -> float:
    value = _dot64(a, a)
    return math.sqrt(max(value, 0.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na = _norm64(a)
    nb = _norm64(b)
    if na == 0.0 or nb == 0.0:
        return None
    value = _dot64(a, b) / (na * nb)
    return min(1.0, max(-1.0, value))


def _f32le(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes(order="C")


@dataclass(frozen=True)
class ExactVector:
    raw: bytes
    values: np.ndarray
    sha256: str
    valid: bool = True
    error: str | None = None

    @classmethod
    def from_raw(
        cls, raw: bytes, expected_sha256: str | None = None, numel: int | None = None
    ) -> "ExactVector":
        digest = sha256_bytes(raw)
        errors: list[str] = []
        if expected_sha256 is not None and digest != expected_sha256:
            errors.append("sha256_mismatch")
        if len(raw) % 4:
            errors.append("byte_length_not_divisible_by_four")
            values = np.empty(0, dtype=np.float32)
        else:
            values = np.frombuffer(raw, dtype="<f4")
            if numel is not None and values.size != numel:
                errors.append("numel_mismatch")
            if not np.isfinite(values).all():
                errors.append("nonfinite")
        return cls(raw, values, digest, not errors, ",".join(errors) or None)


@dataclass(frozen=True)
class ExactEvent:
    event_id: str
    sequence: int
    fragment: int
    stock: ExactVector
    proposal: ExactVector | None


@dataclass
class _Pending:
    source_event_id: str
    source_sequence: int
    source_ordinal: int
    stock: np.ndarray
    proposal: np.ndarray
    residual: np.ndarray
    ratio: float


@dataclass
class _BankEntry:
    source_event_id: str
    source_sequence: int
    source_ordinal: int
    residual: np.ndarray
    score: float


@dataclass
class _FragmentState:
    ordinal: int = 0
    pending: _Pending | None = None
    bank: list[_BankEntry] = field(default_factory=list)


@dataclass(frozen=True)
class CRPAction:
    raw: bytes
    values: np.ndarray
    stock_raw: bytes
    fallback: bool
    reason: str
    pulse_ratio: float
    contributor_count: int
    admitted_event_id: str | None
    resolution_score: float | None
    expired_count: int

    @property
    def bit_identical_to_stock(self) -> bool:
        return self.raw is self.stock_raw


class CRPEngine:
    """Causal CRP state machine with byte-identical stock fallback."""

    def __init__(self) -> None:
        self._states: dict[int, _FragmentState] = defaultdict(_FragmentState)

    def _fallback(
        self,
        event: ExactEvent,
        reason: str,
        *,
        admitted_event_id: str | None = None,
        score: float | None = None,
        expired: int = 0,
    ) -> CRPAction:
        # Do not call tobytes(): object identity proves there was no conversion.
        return CRPAction(
            raw=event.stock.raw,
            values=event.stock.values,
            stock_raw=event.stock.raw,
            fallback=True,
            reason=reason,
            pulse_ratio=0.0,
            contributor_count=0,
            admitted_event_id=admitted_event_id,
            resolution_score=score,
            expired_count=expired,
        )

    def process(self, event: ExactEvent) -> CRPAction:
        state = self._states[event.fragment]
        state.ordinal += 1
        ordinal = state.ordinal

        if not event.stock.valid or event.stock.values.size == 0:
            state.pending = None
            state.bank.clear()
            return self._fallback(event, f"invalid_stock:{event.stock.error}")
        if event.proposal is None or not event.proposal.valid:
            state.pending = None
            state.bank.clear()
            error = "missing" if event.proposal is None else event.proposal.error
            return self._fallback(event, f"invalid_proposal:{error}")
        if event.proposal.values.shape != event.stock.values.shape:
            state.pending = None
            state.bank.clear()
            return self._fallback(event, "proposal_shape_mismatch")

        stock = event.stock.values.astype(np.float64, copy=False)
        proposal = event.proposal.values.astype(np.float64, copy=False)
        stock_norm = _norm64(stock)
        if stock_norm == 0.0:
            state.pending = None
            state.bank.clear()
            return self._fallback(event, "zero_stock_norm")

        admitted: _BankEntry | None = None
        resolution_score: float | None = None
        resolution_failed = False
        if state.pending is not None:
            pending = state.pending
            stock_cos = _cosine(pending.stock, stock)
            proposal_cos = _cosine(pending.proposal, stock)
            if stock_cos is None or proposal_cos is None:
                resolution_failed = True
            else:
                resolution_score = proposal_cos - stock_cos
                if (
                    math.isfinite(resolution_score)
                    and pending.ratio < CRP_INDIVIDUAL_MAX_RATIO
                    and resolution_score > 0.0
                ):
                    admitted = _BankEntry(
                        pending.source_event_id,
                        pending.source_sequence,
                        pending.source_ordinal,
                        pending.residual,
                        resolution_score,
                    )
                else:
                    resolution_failed = True
            if resolution_failed:
                state.bank.clear()

        before_expiry = len(state.bank)
        state.bank[:] = [
            entry
            for entry in state.bank
            if ordinal - entry.source_ordinal <= CRP_MAX_AGE
        ]
        expired = before_expiry - len(state.bank)

        pulse_raw: bytes | None = None
        pulse_values: np.ndarray | None = None
        pulse_ratio = 0.0
        contributors = 0
        pulse_bank_count = len(state.bank)
        if pulse_bank_count >= CRP_MIN_CONTRIBUTORS:
            bank_sum = np.sum(
                np.stack([entry.residual for entry in state.bank]),
                axis=0,
                dtype=np.float64,
            )
            stock_sq = _dot64(stock, stock)
            transverse = bank_sum - (_dot64(bank_sum, stock) / stock_sq) * stock
            transverse_norm = _norm64(transverse)
            raw_ratio = transverse_norm / stock_norm
            if math.isfinite(raw_ratio) and raw_ratio >= CRP_PULSE_MIN_RATIO:
                pulse_ratio = min(raw_ratio, CRP_PULSE_MAX_RATIO)
                correction = transverse * (pulse_ratio / raw_ratio)
                pulse_values = np.asarray(stock + correction, dtype="<f4")
                pulse_raw = _f32le(pulse_values)
                contributors = len(state.bank)
                state.bank.clear()

        if admitted is not None:
            # The newly resolved residual is deliberately too new for this
            # boundary's pulse.  This is the t -> t+1 -> t+2 causal delay.
            state.bank.append(admitted)

        residual = proposal - stock
        residual_norm = _norm64(residual)
        ratio = residual_norm / stock_norm
        if math.isfinite(ratio) and np.isfinite(residual).all():
            state.pending = _Pending(
                event.event_id,
                event.sequence,
                ordinal,
                stock.copy(),
                proposal.copy(),
                residual.copy(),
                ratio,
            )
        else:
            state.pending = None
            state.bank.clear()
            return self._fallback(event, "nonfinite_current_residual")

        admitted_id = None if admitted is None else admitted.source_event_id
        if pulse_raw is None or pulse_values is None:
            if resolution_failed:
                reason = "resolution_rejected_bank_cleared"
            elif pulse_bank_count < CRP_MIN_CONTRIBUTORS:
                reason = "insufficient_contributors"
            else:
                reason = "projected_ratio_below_minimum"
            return self._fallback(
                event,
                reason,
                admitted_event_id=admitted_id,
                score=resolution_score,
                expired=expired,
            )

        return CRPAction(
            raw=pulse_raw,
            values=pulse_values,
            stock_raw=event.stock.raw,
            fallback=False,
            reason="pulse",
            pulse_ratio=pulse_ratio,
            contributor_count=contributors,
            admitted_event_id=admitted_id,
            resolution_score=resolution_score,
            expired_count=expired,
        )


def read_exact_index(index: Path) -> tuple[list[ExactEvent], list[dict]]:
    root = index.parent
    events: list[ExactEvent] = []
    provenance: list[dict] = []
    seen_ids: set[str] = set()
    previous_sequence: int | None = None
    with index.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != EXACT_SCHEMA:
                raise ValueError(
                    f"{index}:{line_number}: schema must be {EXACT_SCHEMA}"
                )
            event_id = str(row["event_id"])
            sequence = int(row["sequence"])
            fragment = int(row["fragment"])
            numel = int(row["numel"])
            if event_id in seen_ids:
                raise ValueError(
                    f"{index}:{line_number}: duplicate event_id {event_id}"
                )
            if previous_sequence is not None and sequence <= previous_sequence:
                raise ValueError(
                    f"{index}:{line_number}: sequence is not strictly increasing"
                )
            if fragment < 0 or numel <= 0:
                raise ValueError(f"{index}:{line_number}: invalid fragment or numel")
            seen_ids.add(event_id)
            previous_sequence = sequence

            def load(field: str) -> tuple[Path, ExactVector]:
                path = Path(row[field])
                path = path if path.is_absolute() else root / path
                raw = path.read_bytes()
                vector = ExactVector.from_raw(raw, str(row[f"{field}_sha256"]), numel)
                return path, vector

            stock_path, stock = load("stock_f32le")
            proposal_path, proposal = load("proposal_f32le")
            events.append(ExactEvent(event_id, sequence, fragment, stock, proposal))
            provenance.append(
                {
                    "event_id": event_id,
                    "stock_path": str(stock_path),
                    "stock_sha256": stock.sha256,
                    "proposal_path": str(proposal_path),
                    "proposal_sha256": proposal.sha256,
                }
            )
    if not events:
        raise ValueError(f"{index}: no records")
    return events, provenance


def replay_exact_crp(index: Path) -> dict:
    events, provenance = read_exact_index(index)
    engine = CRPEngine()
    actions = []
    reasons: Counter[str] = Counter()
    pulses = 0
    for event in events:
        action = engine.process(event)
        reasons[action.reason] += 1
        pulses += int(not action.fallback)
        actions.append(
            {
                "event_id": event.event_id,
                "sequence": event.sequence,
                "fragment": event.fragment,
                "stock_sha256": event.stock.sha256,
                "action_sha256": sha256_bytes(action.raw),
                "bit_identical_stock_fallback": action.bit_identical_to_stock,
                "reason": action.reason,
                "pulse_ratio": action.pulse_ratio,
                "contributors": action.contributor_count,
                "admitted_event_id": action.admitted_event_id,
                "resolution_score": action.resolution_score,
                "expired_count": action.expired_count,
            }
        )
    return {
        "schema": "crp_exact_replay_result_v1",
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "decision": "REPLAYED",
        "identifiable": True,
        "policy": policy_contract(),
        "input": {
            "index": str(index),
            "index_sha256": sha256_file(index),
            "events": len(events),
            "vector_provenance": provenance,
        },
        "summary": {
            "pulses": pulses,
            "fallbacks": len(events) - pulses,
            "action_fraction": pulses / len(events),
            "reasons": dict(sorted(reasons.items())),
        },
        "actions": actions,
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def audit_bcmp_scalar(paths: list[Path]) -> dict:
    schema_counts: Counter[str] = Counter()
    candidate_rows: dict[tuple[str, str], dict] = {}
    resolutions: dict[tuple[str, str], dict] = {}
    files = []
    for path in sorted(paths, key=str):
        files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        for row in _iter_jsonl(path):
            schema = str(row.get("schema"))
            schema_counts[schema] += 1
            if schema == "bcmp_shadow_v1":
                event_id = str(row["event_id"])
                for candidate in ("ray", "slab", "reset"):
                    candidate_rows[event_id, candidate] = row
            elif schema == "bcmp_shadow_resolution_v1":
                resolutions[str(row["event_id"]), str(row["candidate"])] = row

    joined = sorted(set(candidate_rows) & set(resolutions))
    summaries: dict[str, dict] = {}
    for candidate in ("ray", "slab", "reset"):
        rows = [
            (candidate_rows[key], resolutions[key])
            for key in joined
            if key[1] == candidate
        ]
        tiny = 0
        positive_dot = 0
        tiny_positive_dot = 0
        ratios: list[float] = []
        for shadow, resolution in rows:
            stock_norm = float(shadow["stock_total_step_l2"])
            residual_norm = float(resolution["direction_l2"])
            ratio = math.inf if stock_norm <= 0.0 else residual_norm / stock_norm
            ratios.append(ratio)
            is_tiny = math.isfinite(ratio) and ratio < CRP_INDIVIDUAL_MAX_RATIO
            is_positive_dot = float(resolution["future_gradient_dot"]) > 0.0
            tiny += int(is_tiny)
            positive_dot += int(is_positive_dot)
            tiny_positive_dot += int(is_tiny and is_positive_dot)
        finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
        summaries[candidate] = {
            "joined_scalar_resolutions": len(rows),
            "residual_ratio_below_1_over_20": tiny,
            "future_residual_dot_positive": positive_dot,
            "both_proxy_conditions": tiny_positive_dot,
            "residual_ratio_min": min(finite_ratios) if finite_ratios else None,
            "residual_ratio_max": max(finite_ratios) if finite_ratios else None,
            "nonfinite_residual_ratios": len(ratios) - len(finite_ratios),
            "warning": (
                "future_gradient_dot is a descriptive residual proxy, not CRP's "
                "normalized cosine gain z"
            ),
        }

    missing = [
        "stock direction vector G_t bytes",
        "proposal direction vector Q_t or residual r_t bytes",
        "dtype, tensor layout, and accumulation order for those vectors",
        "residual-residual cross terms needed to norm a bank sum",
        "later stock vectors needed to project the bank transversely",
        "merged production-boundary identity and order (the rows are learner-local)",
    ]
    return {
        "schema": "crp_retained_tape_audit_v1",
        "decision": "UNIDENTIFIABLE",
        "identifiable": False,
        "policy": policy_contract(),
        "input": {"files": files, "schema_counts": dict(sorted(schema_counts.items()))},
        "scalar_coverage": {
            "shadow_candidate_pairs": len(candidate_rows),
            "resolution_pairs": len(resolutions),
            "joined_pairs": len(joined),
            "by_candidate": summaries,
        },
        "missing_capabilities": missing,
        "conclusion": (
            "Scalar norms, residual/future-gradient dots, and cosines do not identify "
            "CRP bank sums, projections, pulse bytes, or normalized gain z. No CRP "
            "action or empirical CRP score was reconstructed."
        ),
    }


@dataclass(frozen=True)
class CheckpointFragment:
    version: int
    params: np.ndarray


@dataclass(frozen=True)
class Checkpoint:
    global_step: int
    fragments: tuple[CheckpointFragment, ...]


def parse_checkpoint(path: Path) -> Checkpoint:
    raw = path.read_bytes()
    view = memoryview(raw)
    offset = 0

    def take(size: int, description: str) -> memoryview:
        nonlocal offset
        if offset + size > len(view):
            raise ValueError(f"{path}: truncated {description} at byte {offset}")
        chunk = view[offset : offset + size]
        offset += size
        return chunk

    magic = struct.unpack("<I", take(4, "magic"))[0]
    if magic != CKPT_MAGIC:
        raise ValueError(f"{path}: bad checkpoint magic 0x{magic:08x}")
    global_step = struct.unpack("<Q", take(8, "global step"))[0]
    fragment_count = struct.unpack("<I", take(4, "fragment count"))[0]
    fragments = []
    for fragment in range(fragment_count):
        version, numel = struct.unpack("<QQ", take(16, f"fragment {fragment} header"))
        params = np.frombuffer(
            take(numel * 4, f"fragment {fragment} params"), dtype="<f4"
        )
        take(numel * 4, f"fragment {fragment} momentum")
        if not np.isfinite(params).all():
            raise ValueError(f"{path}: fragment {fragment} has nonfinite params")
        fragments.append(CheckpointFragment(version, params))
    ledger_count = struct.unpack("<I", take(4, "ledger count"))[0]
    take(ledger_count * 28, "ledger")
    if offset != len(view):
        metadata_size = struct.unpack("<I", take(4, "metadata size"))[0]
        metadata = bytes(take(metadata_size, "metadata"))
        json.loads(metadata.decode("utf-8"))
    if offset != len(view):
        raise ValueError(f"{path}: trailing bytes")
    return Checkpoint(global_step, tuple(fragments))


def _unique_capture_rows(index: Path) -> list[dict]:
    unique: dict[tuple[int, int], dict] = {}
    for row in _iter_jsonl(index):
        if row.get("schema") != "syncer_probe_capture_v1":
            raise ValueError(f"{index}: unexpected schema {row.get('schema')}")
        key = (int(row["step"]), int(row["fragment"]))
        material = {
            "step": key[0],
            "fragment": key[1],
            "current_fragment_version": int(row["current_fragment_version"]),
            "state_checkpoint": str(row["state_checkpoint"]),
        }
        previous = unique.setdefault(key, material)
        if previous != material:
            raise ValueError(f"{index}: conflicting duplicate capture row for {key}")
    return sorted(unique.values(), key=lambda row: (row["step"], row["fragment"]))


def materialize_factual_directions(
    capture: Path,
) -> tuple[dict[int, list[np.ndarray]], dict]:
    index = capture / "index.jsonl"
    rows = _unique_capture_rows(index)
    root = capture
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["fragment"]].append(row)

    directions: dict[int, list[np.ndarray]] = {}
    fragment_meta: dict[str, dict] = {}
    checkpoint_hashes: dict[str, str] = {}
    for fragment, fragment_rows in sorted(grouped.items()):
        stream = []
        for current, following in zip(fragment_rows, fragment_rows[1:]):
            if following["current_fragment_version"] != current["step"]:
                raise ValueError(
                    f"{index}: fragment {fragment} next version "
                    f"{following['current_fragment_version']} != prior step {current['step']}"
                )
            current_path = root / current["state_checkpoint"]
            following_path = root / following["state_checkpoint"]
            for path in (current_path, following_path):
                checkpoint_hashes.setdefault(str(path), sha256_file(path))
            before = parse_checkpoint(current_path)
            after = parse_checkpoint(following_path)
            if before.global_step != current["step"] - 1:
                raise ValueError(
                    f"{current_path}: unexpected global step {before.global_step}"
                )
            if after.global_step != following["step"] - 1:
                raise ValueError(
                    f"{following_path}: unexpected global step {after.global_step}"
                )
            if fragment >= len(before.fragments) or fragment >= len(after.fragments):
                raise ValueError(
                    f"{index}: fragment {fragment} missing from checkpoint"
                )
            a = before.fragments[fragment]
            b = after.fragments[fragment]
            if a.version != current["current_fragment_version"]:
                raise ValueError(f"{current_path}: fragment version mismatch")
            if b.version != following["current_fragment_version"]:
                raise ValueError(f"{following_path}: fragment version mismatch")
            if a.params.shape != b.params.shape:
                raise ValueError(f"{index}: fragment {fragment} shape changed")
            # The captured checkpoint at the next same-fragment boundary is
            # the first post-commit state for this fragment.  Subtract in f32
            # to materialize the realized production parameter displacement.
            displacement = np.subtract(b.params, a.params, dtype=np.float32)
            if not np.isfinite(displacement).all():
                raise ValueError(f"{index}: nonfinite displacement")
            stream.append(displacement)
        directions[fragment] = stream
        fragment_meta[str(fragment)] = {
            "snapshots": len(fragment_rows),
            "directions": len(stream),
            "numel": int(stream[0].size) if stream else None,
            "first_step": fragment_rows[0]["step"],
            "last_step": fragment_rows[-1]["step"],
        }
    derived_chain = hashlib.sha256()
    for fragment in sorted(directions):
        for ordinal, direction in enumerate(directions[fragment]):
            raw = _f32le(direction)
            derived_chain.update(struct.pack("<II", fragment, ordinal))
            derived_chain.update(hashlib.sha256(raw).digest())
    return directions, {
        "capture": str(capture),
        "index_sha256": sha256_file(index),
        "unique_boundaries": len(rows),
        "fragments": fragment_meta,
        "checkpoint_count": len(checkpoint_hashes),
        "checkpoint_set_sha256": sha256_bytes(
            canonical_json(checkpoint_hashes).encode("utf-8")
        ),
        "derived_direction_chain_sha256": derived_chain.hexdigest(),
    }


def _pti_scores(stream: list[np.ndarray]) -> tuple[list[dict], dict]:
    outcomes: list[dict] = []
    skipped = Counter()
    histories = {
        coefficient: deque(maxlen=PTI_INTERLOCK_LENGTH)
        for coefficient in PTI_COEFFICIENTS
    }
    eligible_counts = Counter()
    post_warmup_opportunities = 0
    for index in range(1, len(stream) - 1):
        previous = stream[index - 1].astype(np.float64, copy=False)
        current = stream[index].astype(np.float64, copy=False)
        following = stream[index + 1].astype(np.float64, copy=False)
        norms = (_norm64(previous), _norm64(current), _norm64(following))
        if any(norm == 0.0 or not math.isfinite(norm) for norm in norms):
            skipped["zero_or_nonfinite_norm"] += 1
            for history in histories.values():
                history.clear()
            continue
        previous_unit = previous / norms[0]
        current_unit = current / norms[1]
        following_unit = following / norms[2]
        previous_current = min(1.0, max(-1.0, _dot64(previous_unit, current_unit)))
        current_following = min(1.0, max(-1.0, _dot64(current_unit, following_unit)))
        previous_following = min(1.0, max(-1.0, _dot64(previous_unit, following_unit)))
        transverse_sq = max(0.0, 1.0 - previous_current * previous_current)
        if transverse_sq < PTI_TRANSVERSE_NORM_SQ_MIN:
            skipped["degenerate_transverse"] += 1
            for history in histories.values():
                history.clear()
            continue
        transverse_following = (
            previous_following - previous_current * current_following
        ) / math.sqrt(transverse_sq)
        scores = {}
        eligible = {}
        post_warmup = len(histories[PTI_COEFFICIENTS[0]]) == PTI_INTERLOCK_LENGTH
        for coefficient in PTI_COEFFICIENTS:
            history = histories[coefficient]
            eligible[coefficient] = (
                coefficient != 0.0
                and post_warmup
                and all(score > 0.0 for score in history)
            )
            eligible_counts[coefficient] += int(eligible[coefficient])
            score = (
                current_following + coefficient * transverse_following
            ) / math.sqrt(1.0 + coefficient * coefficient) - current_following
            scores[coefficient] = score
        post_warmup_opportunities += int(post_warmup)
        outcomes.append(
            {
                "direction_index": index,
                "stock_next_cosine": current_following,
                "scores": scores,
                "post_warmup": post_warmup,
                "eligible_before_score": eligible,
            }
        )
        for coefficient, score in scores.items():
            histories[coefficient].append(score)
    return outcomes, {
        "skipped": dict(sorted(skipped.items())),
        "post_warmup_opportunities": post_warmup_opportunities,
        "eligible_counts": eligible_counts,
    }


def screen_pti_captures(captures: list[Path]) -> dict:
    per_capture = []
    aggregate: dict[float, list[tuple[int, float, bool, bool]]] = defaultdict(list)
    total_by_fragment = Counter()
    for capture in captures:
        directions, provenance = materialize_factual_directions(capture)
        capture_rows = []
        for fragment, stream in sorted(directions.items()):
            outcomes, meta = _pti_scores(stream)
            total_by_fragment[fragment] += len(outcomes)
            for outcome in outcomes:
                for coefficient, score in outcome["scores"].items():
                    aggregate[coefficient].append(
                        (
                            fragment,
                            score,
                            bool(outcome["eligible_before_score"][coefficient]),
                            bool(outcome["post_warmup"]),
                        )
                    )
            capture_rows.append(
                {
                    "fragment": fragment,
                    "valid_shadow_scores": len(outcomes),
                    "skipped": meta["skipped"],
                    "post_warmup_opportunities": meta["post_warmup_opportunities"],
                    "eligible_counts": {
                        format(coefficient, ".8g"): int(
                            meta["eligible_counts"][coefficient]
                        )
                        for coefficient in PTI_COEFFICIENTS
                        if coefficient != 0.0
                    },
                }
            )
        per_capture.append({"provenance": provenance, "fragments": capture_rows})

    coefficient_results = {}
    for coefficient in PTI_COEFFICIENTS:
        rows = aggregate[coefficient]
        scores = [score for _, score, _, _ in rows]
        by_fragment: dict[int, list[float]] = defaultdict(list)
        for fragment, score, _, _ in rows:
            by_fragment[fragment].append(score)
        post_warmup_count = sum(post_warmup for _, _, _, post_warmup in rows)
        eligible_count = sum(eligible for _, _, eligible, _ in rows)
        coefficient_results[format(coefficient, ".8g")] = {
            "scores": len(scores),
            "mean_cosine_gain": math.fsum(scores) / len(scores) if scores else None,
            "positive_score_fraction": (
                sum(score > 0.0 for score in scores) / len(scores) if scores else None
            ),
            "positive_fragment_count": sum(
                math.fsum(values) / len(values) > 0.0 for values in by_fragment.values()
            ),
            "fragment_mean_cosine_gain": {
                str(fragment): math.fsum(values) / len(values)
                for fragment, values in sorted(by_fragment.items())
            },
            "post_warmup_opportunities": post_warmup_count,
            "interlock_eligible_count": eligible_count,
            "interlock_eligible_fraction": (
                eligible_count / post_warmup_count if post_warmup_count else None
            ),
        }
    return {
        "schema": "pti_historical_direction_screen_v1",
        "decision": "DIRECTION_SCREEN_ONLY",
        "identifiable": True,
        "causal_loss_claim": False,
        "coefficients": list(PTI_COEFFICIENTS),
        "interlock_length": PTI_INTERLOCK_LENGTH,
        "transverse_norm_sq_min": PTI_TRANSVERSE_NORM_SQ_MIN,
        "input": per_capture,
        "valid_scores_by_fragment": {
            str(fragment): count
            for fragment, count in sorted(total_by_fragment.items())
        },
        "coefficient_results": coefficient_results,
        "limitations": [
            "checkpoint differences identify realized factual directions, not losses",
            "the source capture has no sealed CRN k=0/k=8 outcome bundle",
            "the checked-in PTI proposal does not freeze a tie-break among eligible coefficients",
            "eligibility is therefore reported per coefficient; no composite PTI action is invented",
            "historical stock directions are off-policy after the first hypothetical non-stock action",
        ],
    }


def mstp_audit() -> dict:
    return {
        "decision": "UNIDENTIFIABLE",
        "identifiable": False,
        "missing_capabilities": [
            "joined exact anchor, H/2, and H parameter arrays",
            "exact Adam first-moment and metric arrays at midpoint and endpoint",
            "per-parameter optimizer clocks, LR mass, and decay accounting",
            "exact committed responder order and production RDA parity proof",
            "sealed full CRN restore/evaluation bundle for a finite-loss claim",
        ],
    }


def retained_evidence_report(bcmp: list[Path], pti_captures: list[Path]) -> dict:
    return {
        "schema": "crp_optimizer_retained_evidence_report_v1",
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "preregistered_policy_sha256": sha256_bytes(
            canonical_json(policy_contract()).encode("utf-8")
        ),
        "crp": audit_bcmp_scalar(bcmp),
        "pti": screen_pti_captures(pti_captures)
        if pti_captures
        else {
            "decision": "UNIDENTIFIABLE",
            "identifiable": False,
            "reason": "no exact factual-direction capture supplied",
        },
        "mstp": mstp_audit(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--exact-crp-index", type=Path)
    mode.add_argument("--audit-retained", action="store_true")
    parser.add_argument("--bcmp", type=Path, action="append", default=[])
    parser.add_argument("--pti-capture", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.exact_crp_index is not None:
        result = replay_exact_crp(args.exact_crp_index)
    else:
        if not args.bcmp:
            raise SystemExit("--audit-retained requires at least one --bcmp JSONL")
        result = retained_evidence_report(args.bcmp, args.pti_capture)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
