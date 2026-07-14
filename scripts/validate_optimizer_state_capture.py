#!/usr/bin/env python3
"""Fail-closed validation for a completed optimizer-state capture campaign.

The validator deliberately treats the learner manifests as indexes, not as
proof that the indexed files are present.  It verifies every enumerated
artifact through :func:`yeto.optimizer_state_capture.load_capture`, checks the
manifest and artifact sidecars, rejects extra/missing/temporary files, and
then validates cross-learner Richardson/push linkage.

Successful validation writes a stable evidence set below the arm directory:

``optimizer_state_capture_validation.json``
    Human- and machine-readable validation summary.

``optimizer_state_capture_validation.json.sha256``
    Checksum sidecar for the summary.

``optimizer_state_capture_artifacts.sha256``
    A portable ``sha256sum`` manifest whose paths are relative to the arm
    directory.  It covers every learner artifact plus the authoritative
    syncer response transcript.  A teardown gate should re-run this manifest
    from that directory immediately before the final artifact sync and VM
    deletion.

``optimizer_state_capture_committed_boundaries.json`` and its checksum sidecar
    A canonical, fail-closed projection of each authoritative syncer commit to
    its exact ordered responder, push-candidate, and Richardson artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from yeto.optimizer_state_capture import (
    CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL,
    CAPTURE_PROFILE_FULL,
    CAPTURE_PROFILES,
    CaptureIntegrityError,
    load_capture,
)
from yeto.protocol import DTYPE_F32


SCHEMA = "yeto.optimizer-state-capture-validation"
SCHEMA_VERSION = 1
BOUNDARY_INDEX_SCHEMA = "yeto.optimizer-state-capture-committed-boundaries"
BOUNDARY_INDEX_SCHEMA_VERSION = 1
CAPTURE_SCHEMA = "yeto.optimizer-state-capture"
CAPTURE_SCHEMA_VERSION = 1
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_F64_BITS = re.compile(r"[0-9a-f]{16}\Z")
CAPTURE_DIRECTORY = re.compile(r"optimizer_state_capture_learner_([0-9]+)\Z")
FINAL_LIFECYCLE_STATES = frozenset({"pushed", "superseded_unpushed", "closed_unpushed"})
ALLOWED_TERMINAL_DROP_REASONS = frozenset(
    {
        "hmc_incomplete_at_close",
        "midpoint_incomplete_at_close",
        "window_unpushed_at_close",
    }
)


class ValidationError(ValueError):
    """A capture cannot be accepted as complete and internally consistent."""


@dataclass(frozen=True)
class Expectations:
    """Frozen campaign geometry and resource caps supplied by the run spec."""

    learner_ids: tuple[int, ...]
    fragments: int
    window_steps: int
    every: int
    max_hmc_events: int
    max_midpoint_windows: int
    max_bytes: int
    min_joined_boundaries: int = 0
    min_joined_per_fragment: int = 0
    strict_writer: bool = False
    background_writer: bool = False
    background_writer_max_items: int = 0
    background_writer_max_bytes: int = 0
    capture_profile: str = CAPTURE_PROFILE_FULL

    def __post_init__(self) -> None:
        if not self.learner_ids:
            raise ValidationError("expected learner list must not be empty")
        if len(set(self.learner_ids)) != len(self.learner_ids):
            raise ValidationError("expected learner ids must be unique")
        if any(learner_id < 0 for learner_id in self.learner_ids):
            raise ValidationError("expected learner ids must be non-negative")
        if self.fragments < 1:
            raise ValidationError("expected fragment count must be positive")
        if self.window_steps < 2 or self.window_steps % 2:
            raise ValidationError("expected H must be even and at least two")
        if self.every < 1:
            raise ValidationError("expected capture cadence must be positive")
        if (
            min(
                self.max_hmc_events,
                self.max_midpoint_windows,
                self.max_bytes,
                self.min_joined_boundaries,
                self.min_joined_per_fragment,
            )
            < 0
        ):
            raise ValidationError("capture caps and thresholds must be non-negative")
        required_total = self.fragments * self.min_joined_per_fragment
        if self.min_joined_boundaries < required_total:
            raise ValidationError(
                "minimum joined-boundary total must be at least fragments times "
                f"the per-fragment minimum ({required_total})"
            )
        if self.background_writer and (
            self.background_writer_max_items < 1 or self.background_writer_max_bytes < 1
        ):
            raise ValidationError(
                "expected background writer item and byte caps must be positive"
            )
        if self.capture_profile not in CAPTURE_PROFILES:
            raise ValidationError(
                f"unknown expected capture profile {self.capture_profile!r}"
            )
        if (
            self.capture_profile == CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL
            and self.max_hmc_events != 0
        ):
            raise ValidationError(
                "crp_pti_directional validation requires max_hmc_events=0"
            )

    def as_json(self) -> dict[str, Any]:
        value = {
            "learner_ids": list(self.learner_ids),
            "rank": 0,
            "fragments": self.fragments,
            "window_steps": self.window_steps,
            "every": self.every,
            "max_hmc_events": self.max_hmc_events,
            "max_midpoint_windows": self.max_midpoint_windows,
            "max_bytes": self.max_bytes,
            "min_joined_boundaries": self.min_joined_boundaries,
            "min_joined_per_fragment": self.min_joined_per_fragment,
            "strict_writer": self.strict_writer,
        }
        if self.background_writer:
            value.update(
                {
                    "background_writer": True,
                    "background_writer_max_items": self.background_writer_max_items,
                    "background_writer_max_bytes": self.background_writer_max_bytes,
                }
            )
        if self.capture_profile != CAPTURE_PROFILE_FULL:
            value["capture_profile"] = self.capture_profile
        return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """An arm-relative reference to one already-validated immutable object."""

    path: str
    sha256: str

    def as_json(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ResponderRef:
    """One commit responder in the syncer's exact production merge order."""

    responder_index: int
    learner_id: int
    source_attempt_event_seq: int
    attempt_serial: int
    window_uuid: str
    base_version: int
    local_step: int
    c_steps: int
    c_tokens: int
    weight_f64_bits: str
    payload_sha256: str
    push: ArtifactRef
    richardson: ArtifactRef

    def as_json(self) -> dict[str, Any]:
        return {
            "responder_index": self.responder_index,
            "learner_id": self.learner_id,
            "source_attempt_event_seq": self.source_attempt_event_seq,
            "attempt_serial": self.attempt_serial,
            "window_uuid": self.window_uuid,
            "base_version": self.base_version,
            "local_step": self.local_step,
            "c_steps": self.c_steps,
            "c_tokens": self.c_tokens,
            "weight_f64_bits": self.weight_f64_bits,
            "payload_sha256": self.payload_sha256,
            "push": self.push.as_json(),
            "richardson": self.richardson.as_json(),
        }


@dataclass(frozen=True, slots=True)
class CommittedBoundary:
    """A commit normalized only after every authoritative join has passed."""

    capture_session_uuid: str
    commit_event_seq: int
    fragment_id: int
    request_global_step: int
    committed_fragment_version: int
    broadcast_payload_sha256: str
    responders: tuple[ResponderRef, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "capture_session_uuid": self.capture_session_uuid,
            "commit_event_seq": self.commit_event_seq,
            "fragment_id": self.fragment_id,
            "request_global_step": self.request_global_step,
            "committed_fragment_version": self.committed_fragment_version,
            "broadcast_payload_sha256": self.broadcast_payload_sha256,
            "responders": [responder.as_json() for responder in self.responders],
        }


@dataclass(frozen=True, slots=True)
class CommittedBoundaryIndex:
    """Checksummable projection bound to the exact validated source tree."""

    source_transcript: ArtifactRef
    capture_session_uuid: str
    layout_sha256: str
    source_tree_manifest_sha256: str
    expected: Expectations
    boundaries: tuple[CommittedBoundary, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": BOUNDARY_INDEX_SCHEMA,
            "schema_version": BOUNDARY_INDEX_SCHEMA_VERSION,
            "source_transcript": self.source_transcript.as_json(),
            "capture_session_uuid": self.capture_session_uuid,
            "layout_sha256": self.layout_sha256,
            "source_tree_manifest_sha256": self.source_tree_manifest_sha256,
            "expected": self.expected.as_json(),
            "boundary_count": len(self.boundaries),
            "boundaries": [boundary.as_json() for boundary in self.boundaries],
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.as_json(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CampaignValidation:
    summary: dict[str, Any]
    tree_manifest_text: str
    boundary_index: CommittedBoundaryIndex


@dataclass(frozen=True, slots=True)
class _CandidateRef:
    metadata: Mapping[str, Any]
    push: ArtifactRef
    richardson: ArtifactRef


@dataclass(frozen=True, slots=True)
class _TranscriptValidation:
    summary: dict[str, Any]
    joined_keys: tuple[tuple[int, int], ...]
    joined_by_fragment: dict[int, int]
    sha256: str
    boundaries: tuple[CommittedBoundary, ...]


@dataclass
class LearnerValidation:
    learner_id: int
    manifest: dict[str, Any]
    manifest_sha256: str
    layout_sha256: str
    envelopes: list[dict[str, Any]]
    artifact_entries: list[dict[str, Any]]
    lifecycle_by_uuid: dict[str, dict[str, Any]] | None
    tree_entries: dict[str, str]

    def summary(self) -> dict[str, Any]:
        counters = self.manifest["counters"]
        kinds = Counter(envelope["kind"] for envelope in self.envelopes)
        lifecycle_states = Counter(
            row["status"] for row in (self.lifecycle_by_uuid or {}).values()
        )
        return {
            "learner_id": self.learner_id,
            "capture_dir": f"optimizer_state_capture_learner_{self.learner_id}",
            "manifest_path": (
                f"optimizer_state_capture_learner_{self.learner_id}/manifest.json"
            ),
            "manifest_sha256": self.manifest_sha256,
            "closed": counters["closed"],
            "layout_sha256": self.layout_sha256,
            "artifact_bytes": counters["artifact_bytes"],
            "artifact_counts": dict(sorted(kinds.items())),
            "verified_artifacts": len(self.envelopes),
            "drop_counts": counters["drop_counts"],
            "lifecycle_status_counts": dict(sorted(lifecycle_states.items())),
        }


def _fail(message: str, *, context: str | None = None) -> None:
    if context:
        raise ValidationError(f"{context}: {message}")
    raise ValidationError(message)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("must be a JSON object", context=context)
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("must be a JSON array", context=context)
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("must be an integer", context=context)
    if minimum is not None and value < minimum:
        _fail(f"must be at least {minimum}", context=context)
    return value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        _fail("must be a boolean", context=context)
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("must be a non-empty string", context=context)
    return value


def _require_sha256(value: Any, context: str) -> str:
    text = _require_string(value, context)
    if HEX_SHA256.fullmatch(text) is None:
        _fail("must be a lowercase SHA-256 digest", context=context)
    return text


def _require_transcript_weight(
    row: Mapping[str, Any], *, c_steps: int, c_tokens: int, context: str
) -> str:
    """Validate the redundant decimal/binary weight against production math."""

    bits = _require_string(row.get("weight_f64_bits"), f"{context}.weight_f64_bits")
    if HEX_F64_BITS.fullmatch(bits) is None:
        _fail(
            "must be exactly 16 lowercase hexadecimal f64 bits",
            context=f"{context}.weight_f64_bits",
        )
    decoded = struct.unpack(">d", bytes.fromhex(bits))[0]
    if not math.isfinite(decoded) or decoded < 0.0:
        _fail(
            "must encode a finite non-negative f64",
            context=f"{context}.weight_f64_bits",
        )
    weight = row.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        _fail("must be a finite JSON number", context=f"{context}.weight")
    weight = float(weight)
    if not math.isfinite(weight):
        _fail("must be a finite JSON number", context=f"{context}.weight")
    if struct.pack(">d", weight).hex() != bits:
        _fail(
            "decimal weight does not round-trip to weight_f64_bits",
            context=context,
        )
    try:
        token_count = float(c_tokens)
        expected = 0.0 if c_steps == 0 else token_count * token_count / float(c_steps)
    except OverflowError as exc:
        raise ValidationError(f"{context}: counters overflow f64 weight math") from exc
    if not math.isfinite(expected) or struct.pack(">d", expected).hex() != bits:
        _fail(
            "weight_f64_bits differs from token-weighted production formula",
            context=context,
        )
    return bits


def _require_window_uuid(value: Any, context: str) -> str:
    text = _require_string(value, context)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as exc:
        raise ValidationError(f"{context}: must be a canonical UUID") from exc
    if str(parsed) != text:
        _fail("must be a lowercase canonical UUID", context=context)
    return text


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot decode {path}: {exc}") from exc
    return _require_mapping(value, str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_sidecar(
    path: Path, sidecar: Path, expected_digest: str | None = None
) -> str:
    if not path.is_file() or path.is_symlink():
        _fail("missing regular data file", context=str(path))
    if not sidecar.is_file() or sidecar.is_symlink():
        _fail("missing regular checksum sidecar", context=str(sidecar))
    actual = _sha256_file(path)
    if expected_digest is not None and actual != expected_digest:
        _fail(
            f"digest {actual} does not match manifest digest {expected_digest}",
            context=str(path),
        )
    try:
        raw = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(
            f"cannot decode checksum sidecar {sidecar}: {exc}"
        ) from exc
    expected = f"{actual}  {path.name}\n"
    if raw != expected:
        _fail("malformed or mismatched checksum sidecar", context=str(sidecar))
    return actual


def _relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"capture path escapes arm directory: {path}") from exc
    return relative.as_posix()


def _assert_finite(value: Any, context: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            _fail("contains a non-finite tensor value", context=context)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{context}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        _fail("contains a non-finite numeric value", context=context)
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    _fail(f"contains unsupported value type {type(value).__name__}", context=context)


def _artifact_basename(value: Any, context: str) -> str:
    name = _require_string(value, context)
    candidate = Path(name)
    if candidate.name != name or name in (".", "..") or not name.endswith(".pt"):
        _fail("must be one safe .pt basename", context=context)
    return name


def _metadata_int(
    metadata: Mapping[str, Any], key: str, context: str, *, minimum: int | None = None
) -> int:
    if key not in metadata:
        _fail(f"missing metadata field {key}", context=context)
    return _require_int(metadata[key], f"{context}.{key}", minimum=minimum)


def _validate_richardson(
    envelope: Mapping[str, Any], expectations: Expectations, context: str
) -> None:
    metadata = _require_mapping(envelope.get("metadata"), f"{context}.metadata")
    payload = _require_mapping(envelope.get("payload"), f"{context}.payload")
    h = _metadata_int(metadata, "window_steps", context, minimum=2)
    if h != expectations.window_steps:
        _fail(f"window H {h} != expected {expectations.window_steps}", context=context)
    if h % 2:
        _fail("window H is odd", context=context)
    midpoint_steps = _metadata_int(metadata, "accepted_midpoint_steps", context)
    endpoint_steps = _metadata_int(metadata, "accepted_endpoint_steps", context)
    if midpoint_steps != h // 2 or endpoint_steps != h:
        _fail(
            f"accepted H/2,H counters are {midpoint_steps},{endpoint_steps}; "
            f"expected {h // 2},{h}",
            context=context,
        )
    reset_step = _metadata_int(metadata, "reset_local_step", context, minimum=0)
    midpoint_step = _metadata_int(metadata, "midpoint_local_step", context, minimum=0)
    endpoint_step = _metadata_int(metadata, "endpoint_local_step", context, minimum=0)
    if midpoint_step - reset_step != h // 2 or endpoint_step - reset_step != h:
        _fail("local-step boundaries are not exact H/2 and H", context=context)
    if metadata.get("state_boundary") != "post_optimizer_step_pre_broadcast":
        _fail("unexpected state boundary", context=context)
    history = _require_list(
        payload.get("step_history"), f"{context}.payload.step_history"
    )
    if len(history) != h:
        _fail(f"step history has {len(history)} rows; expected {h}", context=context)
    for offset, raw_row in enumerate(history):
        row = _require_mapping(raw_row, f"{context}.payload.step_history[{offset}]")
        if row.get("accepted_optimizer_step") is not True:
            _fail("step history contains an unaccepted update", context=context)
        before = _require_int(
            row.get("local_step_before_update"),
            f"{context}.payload.step_history[{offset}].local_step_before_update",
            minimum=0,
        )
        if before != reset_step + offset:
            _fail("step history is not contiguous from reset", context=context)
    snapshots: list[dict[str, Any]] = []
    for snapshot in ("anchor", "midpoint", "endpoint"):
        snapshots.append(
            _require_mapping(payload.get(snapshot), f"{context}.payload.{snapshot}")
        )

    if expectations.capture_profile == CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL:
        expected_payload_keys = {
            "anchor",
            "midpoint",
            "endpoint",
            "step_history",
            "lr_mass_first_by_group",
            "lr_mass_second_by_group",
        }
        if set(payload) != expected_payload_keys:
            _fail(
                "crp_pti_directional payload has fields outside its closed "
                f"direction-evidence schema: {sorted(set(payload) ^ expected_payload_keys)!r}",
                context=context,
            )
        expected_order: list[str] | None = None
        expected_shape: tuple[int, ...] | None = None
        for name, snapshot in zip(("anchor", "midpoint", "endpoint"), snapshots):
            snapshot_context = f"{context}.payload.{name}"
            if set(snapshot) != {"tensor_order", "parameters_f32"}:
                _fail(
                    "directional snapshot must contain exactly tensor_order and "
                    "parameters_f32",
                    context=snapshot_context,
                )
            order = _require_list(
                snapshot.get("tensor_order"), f"{snapshot_context}.tensor_order"
            )
            if not order:
                _fail("tensor order must not be empty", context=snapshot_context)
            if any(not isinstance(item, str) or not item for item in order):
                _fail(
                    "tensor order contains a non-string or empty name",
                    context=snapshot_context,
                )
            if len(set(order)) != len(order):
                _fail("tensor order contains duplicates", context=snapshot_context)
            parameters = snapshot.get("parameters_f32")
            if not isinstance(parameters, torch.Tensor):
                _fail("parameters_f32 must be a tensor", context=snapshot_context)
            if parameters.dtype != torch.float32 or parameters.ndim != 1:
                _fail(
                    "parameters_f32 must be one flat float32 tensor",
                    context=snapshot_context,
                )
            if parameters.numel() < 1:
                _fail("parameters_f32 must not be empty", context=snapshot_context)
            if expected_order is None:
                expected_order = list(order)
                expected_shape = tuple(parameters.shape)
            elif (
                list(order) != expected_order
                or tuple(parameters.shape) != expected_shape
            ):
                _fail(
                    "directional snapshots change tensor order or flat shape",
                    context=snapshot_context,
                )


def _f32_wire_sha256(value: Any, context: str) -> str:
    if not isinstance(value, torch.Tensor):
        _fail("f32 endpoint must be a tensor", context=context)
    if value.dtype != torch.float32 or value.ndim != 1 or not value.is_contiguous():
        _fail("f32 endpoint must be a contiguous flat float32 tensor", context=context)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _retry_identity(metadata: Mapping[str, Any], context: str) -> str:
    key = {
        name: metadata.get(name)
        for name in (
            "window_uuid",
            "pull_global_step",
            "base_version",
            "learner_id",
            "rank",
            "fragment_id",
        )
    }
    for name in key:
        if key[name] is None:
            _fail(f"missing push retry key {name}", context=context)
    raw = (json.dumps(key, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_push_candidate(
    envelope: Mapping[str, Any], expectations: Expectations, context: str
) -> None:
    metadata = _require_mapping(envelope.get("metadata"), f"{context}.metadata")
    if metadata.get("wire_codec") != "f32":
        _fail("push candidate wire codec must be f32", context=context)
    if metadata.get("push_site") != "immediately_before_client.push_fragment":
        _fail("unexpected push capture site", context=context)
    base_version = _metadata_int(metadata, "base_version", context, minimum=0)
    fragment_version = _metadata_int(metadata, "fragment_version", context, minimum=0)
    if base_version != fragment_version:
        _fail("push base_version differs from fragment_version", context=context)
    if (
        _metadata_int(metadata, "c_steps", context, minimum=0)
        != expectations.window_steps
    ):
        _fail("push c_steps differs from expected H", context=context)
    _require_window_uuid(metadata.get("window_uuid"), f"{context}.window_uuid")
    retry = _require_sha256(metadata.get("retry_identity"), f"{context}.retry_identity")
    if retry != _retry_identity(metadata, context):
        _fail("push retry identity does not match immutable retry key", context=context)
    _metadata_int(metadata, "retry_ordinal", context, minimum=1)
    _metadata_int(metadata, "attempt_serial", context, minimum=1)
    _metadata_int(metadata, "pull_global_step", context, minimum=0)
    _metadata_int(metadata, "local_step", context, minimum=0)
    _metadata_int(metadata, "c_tokens", context, minimum=0)
    _require_sha256(metadata.get("payload_sha256"), f"{context}.payload_sha256")


def _validate_generic_metadata(
    envelope: Mapping[str, Any],
    entry: Mapping[str, Any],
    learner_id: int,
    layout_sha256: str,
    expectations: Expectations,
    context: str,
) -> None:
    metadata = _require_mapping(envelope.get("metadata"), f"{context}.metadata")
    if _metadata_int(metadata, "learner_id", context) != learner_id:
        _fail("artifact learner id differs from directory", context=context)
    if _metadata_int(metadata, "rank", context) != 0:
        _fail("artifact was not captured by rank zero", context=context)
    fragment_id = _metadata_int(metadata, "fragment_id", context, minimum=0)
    if fragment_id >= expectations.fragments:
        _fail("artifact fragment id is outside expected layout", context=context)
    fragment_version = _metadata_int(metadata, "fragment_version", context, minimum=0)
    if (
        fragment_id != entry["fragment_id"]
        or fragment_version != entry["fragment_version"]
    ):
        _fail("artifact metadata differs from manifest index", context=context)
    if "layout_sha256" in metadata:
        artifact_layout = _require_sha256(
            metadata["layout_sha256"], f"{context}.metadata.layout_sha256"
        )
        if artifact_layout != layout_sha256:
            _fail("artifact layout digest differs from manifest", context=context)
    if (
        envelope.get("kind") == "adamw_first_gradient"
        and metadata.get("gradient_boundary")
        != "post_allreduce_post_clip_pre_optimizer_step"
    ):
        _fail("unexpected first-gradient boundary", context=context)


def _validate_lifecycles(
    manifest: Mapping[str, Any],
    envelopes: Sequence[dict[str, Any]],
    learner_id: int,
    expectations: Expectations,
    context: str,
) -> dict[str, dict[str, Any]] | None:
    if "window_lifecycles" not in manifest:
        return None
    rows = _require_list(manifest["window_lifecycles"], f"{context}.window_lifecycles")
    lifecycle_by_uuid: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row_context = f"{context}.window_lifecycles[{index}]"
        row = _require_mapping(raw, row_context)
        window_uuid = _require_window_uuid(
            row.get("window_uuid"), f"{row_context}.window_uuid"
        )
        if window_uuid in lifecycle_by_uuid:
            _fail("duplicate lifecycle window_uuid", context=row_context)
        fragment = _require_int(
            row.get("fragment_id"), f"{row_context}.fragment_id", minimum=0
        )
        if fragment >= expectations.fragments:
            _fail("lifecycle fragment is outside expected layout", context=row_context)
        _require_int(
            row.get("fragment_version"), f"{row_context}.fragment_version", minimum=0
        )
        h = _require_int(
            row.get("window_steps"), f"{row_context}.window_steps", minimum=2
        )
        if h != expectations.window_steps:
            _fail(
                f"lifecycle H {h} != expected {expectations.window_steps}",
                context=row_context,
            )
        status = _require_string(row.get("status"), f"{row_context}.status")
        if status not in FINAL_LIFECYCLE_STATES:
            _fail(
                f"non-final or unknown lifecycle status {status!r}", context=row_context
            )
        attempts = _require_int(
            row.get("push_attempts"), f"{row_context}.push_attempts", minimum=0
        )
        enqueued = _require_int(
            row.get("enqueued_pushes", attempts if status == "pushed" else 0),
            f"{row_context}.enqueued_pushes",
            minimum=0,
        )
        if enqueued > attempts:
            _fail("enqueued push count exceeds recorded attempts", context=row_context)
        if status == "pushed" and enqueued < 1:
            _fail("pushed lifecycle has no enqueued push", context=row_context)
        if status != "pushed" and enqueued != 0:
            _fail("unpushed lifecycle reports an enqueued push", context=row_context)
        lifecycle_by_uuid[window_uuid] = row

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    richardson_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    push_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for envelope in envelopes:
        metadata = envelope["metadata"]
        window_uuid = metadata.get("window_uuid")
        if window_uuid is None:
            continue
        window_uuid = _require_window_uuid(
            window_uuid, f"{context}.{envelope['kind']}.window_uuid"
        )
        lifecycle = lifecycle_by_uuid.get(window_uuid)
        if lifecycle is None:
            _fail(
                f"artifact references unknown window_uuid {window_uuid}",
                context=context,
            )
        for field in ("fragment_id", "fragment_version"):
            if metadata.get(field) != lifecycle.get(field):
                _fail(f"artifact {field} differs from lifecycle", context=context)
        if metadata.get("learner_id") != learner_id:
            _fail("artifact learner differs from lifecycle directory", context=context)
        by_window[window_uuid].append(envelope)
        if envelope["kind"] == "richardson_window":
            richardson_by_window[window_uuid].append(envelope)
        if envelope["kind"] == "push_candidate":
            push_by_window[window_uuid].append(envelope)

    for window_uuid, lifecycle in lifecycle_by_uuid.items():
        richardson = richardson_by_window.get(window_uuid, [])
        pushes = push_by_window.get(window_uuid, [])
        if lifecycle["status"] == "pushed":
            if len(richardson) != 1:
                _fail(
                    "pushed lifecycle must link exactly one Richardson artifact",
                    context=window_uuid,
                )
            if len(pushes) != lifecycle["push_attempts"]:
                _fail(
                    "push artifact count differs from lifecycle push_attempts",
                    context=window_uuid,
                )
            rich_metadata = richardson[0]["metadata"]
            rich_endpoint = richardson[0]["payload"]["endpoint"]["parameters_f32"]
            endpoint_digest = _f32_wire_sha256(
                rich_endpoint, f"{context}.{window_uuid}.endpoint"
            )
            if lifecycle.get("expected_f32_payload_sha256") != endpoint_digest:
                _fail(
                    "lifecycle payload digest differs from exact Richardson endpoint",
                    context=window_uuid,
                )
            for lifecycle_key, metadata_key in (
                ("endpoint_local_step", "endpoint_local_step"),
                ("endpoint_tokens_total", "endpoint_tokens_total"),
                ("c_steps", "accepted_endpoint_steps"),
            ):
                if lifecycle.get(lifecycle_key) != rich_metadata.get(metadata_key):
                    _fail(
                        f"lifecycle {lifecycle_key} differs from Richardson endpoint",
                        context=window_uuid,
                    )
            for push in pushes:
                metadata = push["metadata"]
                expected_fields = {
                    "base_version": lifecycle["fragment_version"],
                    "local_step": lifecycle["endpoint_local_step"],
                    "c_steps": lifecycle["c_steps"],
                    "c_tokens": lifecycle["c_tokens"],
                    "payload_sha256": lifecycle["expected_f32_payload_sha256"],
                }
                for key, expected in expected_fields.items():
                    if metadata.get(key) != expected:
                        _fail(f"push {key} differs from lifecycle", context=window_uuid)
            last_push = pushes[-1]["metadata"]
            if lifecycle.get("last_retry_identity") != last_push.get("retry_identity"):
                _fail(
                    "lifecycle last retry identity differs from final artifact",
                    context=window_uuid,
                )
            if lifecycle.get("last_pull_global_step") != last_push.get(
                "pull_global_step"
            ):
                _fail(
                    "lifecycle last pull step differs from final artifact",
                    context=window_uuid,
                )
        elif len(richardson) > 1:
            _fail("lifecycle links multiple Richardson artifacts", context=window_uuid)

    retries_raw = manifest.get("push_retries", {})
    retries = _require_mapping(retries_raw, f"{context}.push_retries")
    push_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for envelope in envelopes:
        if envelope["kind"] == "push_candidate":
            push_groups[envelope["metadata"]["retry_identity"]].append(
                envelope["metadata"]
            )
    if set(retries) != set(push_groups):
        _fail("manifest push_retries keys differ from push artifacts", context=context)
    immutable_fields = (
        "window_uuid",
        "pull_global_step",
        "base_version",
        "learner_id",
        "rank",
        "fragment_id",
        "local_step",
        "c_steps",
        "c_tokens",
        "wire_codec",
        "payload_sha256",
    )
    for retry_identity, metadata_rows in push_groups.items():
        retry = _require_mapping(
            retries[retry_identity], f"{context}.push_retries.{retry_identity}"
        )
        attempts = _require_int(
            retry.get("attempts"),
            f"{context}.push_retries.{retry_identity}.attempts",
            minimum=1,
        )
        candidate = _require_mapping(
            retry.get("candidate"), f"{context}.push_retries.{retry_identity}.candidate"
        )
        if attempts != len(metadata_rows):
            _fail("retry attempt count differs from artifacts", context=retry_identity)
        ordinals = sorted(row["retry_ordinal"] for row in metadata_rows)
        if ordinals != list(range(1, attempts + 1)):
            _fail("retry ordinals are not exactly 1..attempts", context=retry_identity)
        first = metadata_rows[0]
        expected_candidate = {key: first[key] for key in immutable_fields}
        if candidate != expected_candidate:
            _fail(
                "retry candidate differs from artifact metadata", context=retry_identity
            )
        for row in metadata_rows[1:]:
            if {key: row[key] for key in immutable_fields} != expected_candidate:
                _fail(
                    "retry artifacts changed immutable fields", context=retry_identity
                )
    return lifecycle_by_uuid


def _validate_learner(
    arm_dir: Path, learner_id: int, expectations: Expectations
) -> LearnerValidation:
    capture_dir = arm_dir / f"optimizer_state_capture_learner_{learner_id}"
    context = f"learner {learner_id}"
    if not capture_dir.is_dir() or capture_dir.is_symlink():
        _fail(
            "capture directory is missing or is not a regular directory",
            context=context,
        )
    manifest_path = capture_dir / "manifest.json"
    manifest_sidecar = capture_dir / "manifest.json.sha256"
    manifest_digest = _verify_sidecar(manifest_path, manifest_sidecar)
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != CAPTURE_SCHEMA:
        _fail("unsupported capture schema", context=context)
    if manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        _fail("unsupported capture schema version", context=context)
    if manifest.get("kind") != "run_manifest":
        _fail("manifest kind is not run_manifest", context=context)

    config = _require_mapping(manifest.get("config"), f"{context}.config")
    raw_capture_profile = config.get("capture_profile", CAPTURE_PROFILE_FULL)
    actual_capture_profile = _require_string(
        raw_capture_profile, f"{context}.config.capture_profile"
    )
    if actual_capture_profile not in CAPTURE_PROFILES:
        _fail("unknown capture profile", context=context)
    if actual_capture_profile != expectations.capture_profile:
        _fail(
            "config capture_profile differs from expected mode",
            context=context,
        )
    if actual_capture_profile == CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL:
        if config.get("scientific_scope") != "crp_pti_direction_evidence_only":
            _fail("directional profile has incorrect scientific scope", context=context)
        if config.get("capture_v2_restore_complete") is not False:
            _fail(
                "directional profile must explicitly disclaim capture-v2 restore authority",
                context=context,
            )
    expected_config = {
        "learner_id": learner_id,
        "rank": 0,
        "every": expectations.every,
        "max_hmc_events": expectations.max_hmc_events,
        "max_midpoint_windows": expectations.max_midpoint_windows,
        "max_bytes": expectations.max_bytes,
    }
    actual_background_writer = config.get("background_writer", False)
    if actual_background_writer is not False and actual_background_writer is not True:
        _fail("config background_writer must be boolean when present", context=context)
    if actual_background_writer != expectations.background_writer:
        _fail(
            "config background_writer differs from expected mode",
            context=context,
        )
    writer_counts: dict[str, int] | None = None
    if expectations.background_writer:
        expected_config.update(
            {
                "background_writer": True,
                "background_writer_max_items": (
                    expectations.background_writer_max_items
                ),
                "background_writer_max_bytes": (
                    expectations.background_writer_max_bytes
                ),
            }
        )
    for key, expected in expected_config.items():
        if isinstance(expected, bool):
            actual = _require_bool(config.get(key), f"{context}.config.{key}")
        else:
            actual = _require_int(config.get(key), f"{context}.config.{key}", minimum=0)
        if actual != expected:
            _fail(f"config {key}={actual} != expected {expected}", context=context)
    if config.get("optimizer_class") != "torch.optim.adamw.AdamW":
        _fail("capture optimizer is not native torch.optim.AdamW", context=context)
    layout_sha256 = _require_sha256(
        config.get("layout_sha256"), f"{context}.config.layout_sha256"
    )

    counters = _require_mapping(manifest.get("counters"), f"{context}.counters")
    if not _require_bool(counters.get("closed"), f"{context}.counters.closed"):
        _fail("capture manifest is not closed", context=context)
    if (
        _require_int(
            counters.get("pending_raw_bytes"),
            f"{context}.counters.pending_raw_bytes",
            minimum=0,
        )
        != 0
    ):
        _fail("capture retains pending raw tensors", context=context)
    artifact_bytes = _require_int(
        counters.get("artifact_bytes"), f"{context}.counters.artifact_bytes", minimum=0
    )
    if artifact_bytes > expectations.max_bytes:
        _fail("artifact byte counter exceeds configured cap", context=context)
    hmc_admitted = _require_int(
        counters.get("hmc_events_admitted"),
        f"{context}.counters.hmc_events_admitted",
        minimum=0,
    )
    midpoint_admitted = _require_int(
        counters.get("midpoint_windows_admitted"),
        f"{context}.counters.midpoint_windows_admitted",
        minimum=0,
    )
    if hmc_admitted > expectations.max_hmc_events:
        _fail("HMC admission counter exceeds configured cap", context=context)
    if midpoint_admitted > expectations.max_midpoint_windows:
        _fail("midpoint admission counter exceeds configured cap", context=context)
    drop_counts = _require_mapping(
        counters.get("drop_counts"), f"{context}.counters.drop_counts"
    )
    for reason, count in drop_counts.items():
        _require_string(reason, f"{context}.counters.drop_counts key")
        _require_int(count, f"{context}.counters.drop_counts.{reason}", minimum=1)
    if expectations.strict_writer:
        unexpected_drops = {
            reason: count
            for reason, count in drop_counts.items()
            if reason not in ALLOWED_TERMINAL_DROP_REASONS
        }
        if unexpected_drops:
            _fail(
                "strict writer contract rejects non-terminal drops: "
                f"{dict(sorted(unexpected_drops.items()))}",
                context=context,
            )
    if expectations.background_writer:
        reserved = _require_int(
            counters.get("background_artifact_reserved_bytes"),
            f"{context}.counters.background_artifact_reserved_bytes",
            minimum=0,
        )
        if reserved != 0:
            _fail(
                "background capture retains reserved artifact bytes",
                context=context,
            )
        writer = _require_mapping(
            manifest.get("background_writer"), f"{context}.background_writer"
        )
        if writer.get("state") != "closed":
            _fail("background writer is not closed", context=context)
        for key, expected in (
            ("max_items", expectations.background_writer_max_items),
            ("max_bytes", expectations.background_writer_max_bytes),
        ):
            if (
                _require_int(
                    writer.get(key), f"{context}.background_writer.{key}", minimum=1
                )
                != expected
            ):
                _fail(f"background writer {key} differs from expected", context=context)
        writer_counts = {
            key: _require_int(
                writer.get(key), f"{context}.background_writer.{key}", minimum=0
            )
            for key in (
                "accepted_items",
                "accepted_bytes",
                "completed_items",
                "completed_bytes",
                "abandoned_items",
                "abandoned_bytes",
                "reserved_items",
                "reserved_bytes",
                "queued_items",
                "queued_bytes",
                "in_flight_items",
                "in_flight_bytes",
            )
        }
        if (
            writer_counts["accepted_items"] != writer_counts["completed_items"]
            or writer_counts["accepted_bytes"] != writer_counts["completed_bytes"]
        ):
            _fail(
                "background writer did not complete every accepted byte",
                context=context,
            )
        for key in (
            "abandoned_items",
            "abandoned_bytes",
            "reserved_items",
            "reserved_bytes",
            "queued_items",
            "queued_bytes",
            "in_flight_items",
            "in_flight_bytes",
        ):
            if writer_counts[key] != 0:
                _fail(f"background writer terminal {key} is nonzero", context=context)
        if writer.get("worker_alive") is not False:
            _fail("background writer worker is still alive", context=context)
        if any(
            writer.get(key) is not None
            for key in ("failure_sequence", "failure_type", "failure_message")
        ):
            _fail("background writer reports a terminal failure", context=context)

    raw_entries = _require_list(manifest.get("artifacts"), f"{context}.artifacts")
    if writer_counts is not None and writer_counts["completed_items"] != len(
        raw_entries
    ):
        _fail(
            "background writer completed-item count differs from artifact index",
            context=context,
        )
    entries: list[dict[str, Any]] = []
    expected_names = {"manifest.json", "manifest.json.sha256"}
    seen_names: set[str] = set()
    envelopes: list[dict[str, Any]] = []
    tree_entries = {
        _relative_posix(manifest_path, arm_dir): manifest_digest,
        _relative_posix(manifest_sidecar, arm_dir): _sha256_file(manifest_sidecar),
    }
    indexed_bytes = 0
    for index, raw_entry in enumerate(raw_entries):
        entry_context = f"{context}.artifacts[{index}]"
        entry = _require_mapping(raw_entry, entry_context)
        name = _artifact_basename(entry.get("path"), f"{entry_context}.path")
        if name in seen_names:
            _fail("duplicate artifact path", context=entry_context)
        seen_names.add(name)
        expected_names.update({name, f"{name}.sha256"})
        digest = _require_sha256(entry.get("sha256"), f"{entry_context}.sha256")
        byte_count = _require_int(
            entry.get("bytes"), f"{entry_context}.bytes", minimum=1
        )
        sidecar_bytes = _require_int(
            entry.get("sidecar_bytes"), f"{entry_context}.sidecar_bytes", minimum=1
        )
        kind = _require_string(entry.get("kind"), f"{entry_context}.kind")
        allowed_kinds = {"adamw_first_gradient", "richardson_window", "push_candidate"}
        if kind not in allowed_kinds:
            _fail(f"unknown artifact kind {kind!r}", context=entry_context)
        if (
            expectations.capture_profile == CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL
            and kind == "adamw_first_gradient"
        ):
            _fail(
                "crp_pti_directional profile forbids first-gradient artifacts",
                context=entry_context,
            )
        fragment_id = _require_int(
            entry.get("fragment_id"), f"{entry_context}.fragment_id", minimum=0
        )
        if fragment_id >= expectations.fragments:
            _fail(
                "manifest artifact fragment is outside expected layout",
                context=entry_context,
            )
        fragment_version = _require_int(
            entry.get("fragment_version"),
            f"{entry_context}.fragment_version",
            minimum=0,
        )
        normalized = {
            **entry,
            "path": name,
            "sha256": digest,
            "bytes": byte_count,
            "sidecar_bytes": sidecar_bytes,
            "kind": kind,
            "fragment_id": fragment_id,
            "fragment_version": fragment_version,
        }
        artifact_path = capture_dir / name
        artifact_sidecar = capture_dir / f"{name}.sha256"
        actual_digest = _verify_sidecar(artifact_path, artifact_sidecar, digest)
        if artifact_path.stat().st_size != byte_count:
            _fail("artifact size differs from manifest", context=str(artifact_path))
        if artifact_sidecar.stat().st_size != sidecar_bytes:
            _fail(
                "artifact sidecar size differs from manifest",
                context=str(artifact_sidecar),
            )
        try:
            envelope = load_capture(artifact_path)
        except (CaptureIntegrityError, OSError, ValueError, TypeError) as exc:
            raise ValidationError(
                f"{artifact_path}: load_capture rejected artifact: {exc}"
            ) from exc
        envelope = _require_mapping(envelope, str(artifact_path))
        if envelope.get("kind") != kind:
            _fail(
                "artifact kind differs from manifest index", context=str(artifact_path)
            )
        _validate_generic_metadata(
            envelope,
            normalized,
            learner_id,
            layout_sha256,
            expectations,
            str(artifact_path),
        )
        _assert_finite(envelope, str(artifact_path))
        if kind == "richardson_window":
            _validate_richardson(envelope, expectations, str(artifact_path))
        elif kind == "push_candidate":
            _validate_push_candidate(envelope, expectations, str(artifact_path))
        entries.append(normalized)
        envelopes.append(envelope)
        indexed_bytes += byte_count + sidecar_bytes
        tree_entries[_relative_posix(artifact_path, arm_dir)] = actual_digest
        tree_entries[_relative_posix(artifact_sidecar, arm_dir)] = _sha256_file(
            artifact_sidecar
        )

    actual_names: set[str] = set()
    for child in capture_dir.iterdir():
        if child.is_symlink() or not child.is_file():
            _fail(
                "capture directory contains a symlink or non-file entry",
                context=str(child),
            )
        if ".tmp-" in child.name or child.name.endswith(".tmp"):
            _fail("capture directory contains a temporary file", context=str(child))
        actual_names.add(child.name)
    missing = sorted(expected_names - actual_names)
    unlisted = sorted(actual_names - expected_names)
    if missing or unlisted:
        _fail(
            f"exact file-set mismatch missing={missing!r} unlisted={unlisted!r}",
            context=context,
        )
    if indexed_bytes != artifact_bytes:
        _fail(
            f"indexed artifact bytes {indexed_bytes} != counter {artifact_bytes}",
            context=context,
        )
    lifecycle_by_uuid = _validate_lifecycles(
        manifest, envelopes, learner_id, expectations, context
    )
    return LearnerValidation(
        learner_id=learner_id,
        manifest=manifest,
        manifest_sha256=manifest_digest,
        layout_sha256=layout_sha256,
        envelopes=envelopes,
        artifact_entries=entries,
        lifecycle_by_uuid=lifecycle_by_uuid,
        tree_entries=tree_entries,
    )


def _load_transcript(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file() or path.is_symlink():
        _fail(
            "response transcript is missing or is not a regular file", context=str(path)
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read response transcript {path}: {exc}") from exc
    if not raw or not raw.endswith(b"\n"):
        _fail(
            "response transcript is empty or lacks its final newline", context=str(path)
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"response transcript is not UTF-8: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            _fail(
                "response transcript contains an empty line",
                context=f"{path}:{line_number}",
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_json_no_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValidationError(
                f"cannot decode response transcript {path}:{line_number}: {exc}"
            ) from exc
        rows.append(_require_mapping(value, f"{path}:{line_number}"))
    return rows, hashlib.sha256(raw).hexdigest()


def _candidate_attempt_refs(
    arm_dir: Path,
    learners: Sequence[LearnerValidation],
) -> dict[tuple[int, int], _CandidateRef]:
    attempts: dict[tuple[int, int], _CandidateRef] = {}
    for learner in learners:
        if learner.lifecycle_by_uuid is None:
            _fail(
                "authoritative transcript validation requires lifecycle-aware manifests"
            )
        capture_relative = f"optimizer_state_capture_learner_{learner.learner_id}"
        artifact_pairs = list(
            zip(learner.envelopes, learner.artifact_entries, strict=True)
        )
        richardson_by_window: dict[str, ArtifactRef] = {}
        for envelope, entry in artifact_pairs:
            if envelope["kind"] != "richardson_window":
                continue
            window_uuid = envelope["metadata"]["window_uuid"]
            if window_uuid in richardson_by_window:
                _fail(
                    "multiple Richardson artifacts share a learner window",
                    context=f"learner={learner.learner_id} window={window_uuid}",
                )
            path = f"{capture_relative}/{entry['path']}"
            # The entry path is already a validated basename.  Rechecking the
            # assembled path makes the normalized arm-relative contract explicit.
            _relative_posix(arm_dir / path, arm_dir)
            richardson_by_window[window_uuid] = ArtifactRef(
                path=path,
                sha256=entry["sha256"],
            )
        serials: list[int] = []
        for envelope, entry in artifact_pairs:
            if envelope["kind"] != "push_candidate":
                continue
            metadata = envelope["metadata"]
            window_uuid = metadata["window_uuid"]
            richardson = richardson_by_window.get(window_uuid)
            if richardson is None:
                _fail(
                    "push candidate lacks linked Richardson artifact",
                    context=f"learner={learner.learner_id} window={window_uuid}",
                )
            serial = metadata["attempt_serial"]
            serials.append(serial)
            key = (learner.learner_id, serial)
            if key in attempts:
                _fail("duplicate learner attempt_serial artifact", context=str(key))
            path = f"{capture_relative}/{entry['path']}"
            _relative_posix(arm_dir / path, arm_dir)
            attempts[key] = _CandidateRef(
                metadata=metadata,
                push=ArtifactRef(path=path, sha256=entry["sha256"]),
                richardson=richardson,
            )
        expected_serials = list(range(1, len(serials) + 1))
        if serials != expected_serials:
            _fail(
                f"learner attempt_serial sequence {serials!r} is not contiguous "
                f"{expected_serials!r}",
                context=f"learner {learner.learner_id}",
            )
        manifest_serial = _require_int(
            learner.manifest["counters"].get("push_attempt_serial"),
            f"learner {learner.learner_id}.counters.push_attempt_serial",
            minimum=0,
        )
        if manifest_serial != len(serials):
            _fail(
                "manifest push_attempt_serial differs from push artifacts",
                context=f"learner {learner.learner_id}",
            )
    return attempts


def _match_primary_attempt(
    row: Mapping[str, Any], candidate: Mapping[str, Any], context: str
) -> None:
    exact_fields = {
        "learner_id": "learner_id",
        "fragment_id": "fragment_id",
        "request_global_step": "pull_global_step",
        "base_version": "base_version",
        "local_step": "local_step",
        "c_steps": "c_steps",
        "c_tokens": "c_tokens",
        "window_uuid": "window_uuid",
        "attempt_serial": "attempt_serial",
    }
    for transcript_key, candidate_key in exact_fields.items():
        transcript_value = row.get(transcript_key)
        candidate_value = candidate.get(candidate_key)
        if (
            type(transcript_value) is not type(candidate_value)
            or transcript_value != candidate_value
        ):
            _fail(
                f"transcript {transcript_key} differs from learner candidate",
                context=context,
            )
    if _require_int(row.get("wire_dtype"), f"{context}.wire_dtype") != DTYPE_F32:
        _fail("transcript attempt is not f32 wire dtype", context=context)
    payload_digest = candidate["payload_sha256"]
    if row.get("declared_payload_sha256") != payload_digest:
        _fail("declared payload digest differs from learner candidate", context=context)
    if row.get("received_payload_sha256") != payload_digest:
        _fail("received payload digest differs from learner candidate", context=context)
    if row.get("payload_digest_match") is not True:
        _fail("syncer did not verify the learner payload digest", context=context)


def _match_responder_to_attempt(
    responder: Mapping[str, Any], attempt: Mapping[str, Any], context: str
) -> None:
    for key in (
        "learner_id",
        "connection_id",
        "window_uuid",
        "attempt_serial",
        "base_version",
        "local_step",
        "c_steps",
        "c_tokens",
        "weight_f64_bits",
        "received_payload_sha256",
    ):
        responder_value = responder.get(key)
        attempt_value = attempt.get(key)
        if (
            type(responder_value) is not type(attempt_value)
            or responder_value != attempt_value
        ):
            _fail(
                f"commit responder {key} differs from source attempt", context=context
            )


def _validate_response_transcript(
    arm_dir: Path,
    transcript_path: Path,
    learners: Sequence[LearnerValidation],
    expectations: Expectations,
) -> _TranscriptValidation:
    transcript_path = transcript_path.resolve()
    rows, transcript_digest = _load_transcript(transcript_path)
    relative_path = _relative_posix(transcript_path, arm_dir)
    if not rows:
        _fail("response transcript has no header", context=str(transcript_path))
    header = rows[0]
    if header.get("schema") != "syncer_response_transcript_header_v1":
        _fail("first transcript row is not the v1 header", context=str(transcript_path))
    if (
        _require_int(header.get("event_seq"), "response transcript header event_seq")
        != 1
    ):
        _fail("response transcript header event_seq must be one")
    session = _require_string(
        header.get("capture_session_uuid"), "response transcript capture_session_uuid"
    )

    primary_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    primary_by_event: dict[int, dict[str, Any]] = {}
    commits: list[dict[str, Any]] = []
    allowed_schemas = {
        "syncer_response_transcript_header_v1",
        "syncer_push_attempt_v1",
        "syncer_round_commit_v1",
    }
    for index, row in enumerate(rows):
        context = f"{transcript_path}:{index + 1}"
        event_seq = _require_int(
            row.get("event_seq"), f"{context}.event_seq", minimum=1
        )
        expected_event_seq = index + 1
        if event_seq != expected_event_seq:
            _fail(
                f"event_seq {event_seq} is not contiguous expected {expected_event_seq}",
                context=context,
            )
        if row.get("capture_session_uuid") != session:
            _fail("capture session changes within transcript", context=context)
        schema = row.get("schema")
        if schema not in allowed_schemas:
            _fail(f"unknown response transcript schema {schema!r}", context=context)
        if index == 0:
            continue
        if schema == "syncer_response_transcript_header_v1":
            _fail("response transcript contains a second header", context=context)
        if schema == "syncer_round_commit_v1":
            commits.append(row)
            continue

        learner_id = _require_int(
            row.get("learner_id"), f"{context}.learner_id", minimum=0
        )
        if learner_id not in expectations.learner_ids:
            _fail("attempt belongs to an unexpected learner", context=context)
        fragment_id = _require_int(
            row.get("fragment_id"), f"{context}.fragment_id", minimum=0
        )
        if fragment_id >= expectations.fragments:
            _fail("attempt fragment is outside expected layout", context=context)
        serial = _require_int(
            row.get("attempt_serial"), f"{context}.attempt_serial", minimum=1
        )
        c_steps = _require_int(row.get("c_steps"), f"{context}.c_steps", minimum=0)
        c_tokens = _require_int(row.get("c_tokens"), f"{context}.c_tokens", minimum=0)
        _require_window_uuid(row.get("window_uuid"), f"{context}.window_uuid")
        _require_sha256(
            row.get("declared_payload_sha256"), f"{context}.declared_payload_sha256"
        )
        _require_sha256(
            row.get("received_payload_sha256"), f"{context}.received_payload_sha256"
        )
        _require_transcript_weight(
            row,
            c_steps=c_steps,
            c_tokens=c_tokens,
            context=context,
        )
        source_seq = row.get("source_attempt_event_seq")
        if source_seq is None:
            key = (learner_id, serial)
            if key in primary_by_key:
                _fail("duplicate primary learner attempt_serial", context=context)
            primary_by_key[key] = row
            primary_by_event[event_seq] = row
        else:
            source_seq = _require_int(
                source_seq, f"{context}.source_attempt_event_seq", minimum=1
            )
            if source_seq >= event_seq or source_seq not in primary_by_event:
                _fail(
                    "follow-up disposition references no earlier primary attempt",
                    context=context,
                )
            source = primary_by_event[source_seq]
            for key in (
                "learner_id",
                "fragment_id",
                "request_global_step",
                "base_version",
                "local_step",
                "c_steps",
                "c_tokens",
                "window_uuid",
                "attempt_serial",
                "declared_payload_sha256",
                "received_payload_sha256",
            ):
                row_value = row.get(key)
                source_value = source.get(key)
                if (
                    type(row_value) is not type(source_value)
                    or row_value != source_value
                ):
                    _fail(f"follow-up disposition changes {key}", context=context)

    candidates = _candidate_attempt_refs(arm_dir, learners)
    if set(primary_by_key) != set(candidates):
        _fail(
            "primary transcript attempts differ from learner artifacts: "
            f"transcript_only={sorted(set(primary_by_key) - set(candidates))!r} "
            f"learner_only={sorted(set(candidates) - set(primary_by_key))!r}"
        )
    transcript_serials: dict[int, list[int]] = defaultdict(list)
    for key, row in primary_by_key.items():
        transcript_serials[key[0]].append(key[1])
        _match_primary_attempt(
            row,
            candidates[key].metadata,
            f"learner={key[0]} attempt_serial={key[1]}",
        )
    for learner_id, serials in transcript_serials.items():
        expected_serials = list(range(1, len(serials) + 1))
        if serials != expected_serials:
            _fail(
                f"transcript attempt serials {serials!r} are not monotone contiguous "
                f"{expected_serials!r}",
                context=f"learner {learner_id}",
            )

    expected_responders = list(sorted(expectations.learner_ids))
    joined: list[tuple[int, int]] = []
    normalized_boundaries: list[CommittedBoundary] = []
    seen_commits: set[tuple[int, int]] = set()
    for commit in commits:
        event_seq = commit["event_seq"]
        context = f"transcript commit event_seq={event_seq}"
        fragment_id = _require_int(
            commit.get("fragment_id"), f"{context}.fragment_id", minimum=0
        )
        if fragment_id >= expectations.fragments:
            _fail("commit fragment is outside expected layout", context=context)
        request_step = _require_int(
            commit.get("request_global_step"),
            f"{context}.request_global_step",
            minimum=0,
        )
        commit_key = (fragment_id, request_step)
        if commit_key in seen_commits:
            _fail("duplicate round commit identity", context=context)
        seen_commits.add(commit_key)
        committed_fragment_version = _require_int(
            commit.get("committed_fragment_version"),
            f"{context}.committed_fragment_version",
            minimum=0,
        )
        if committed_fragment_version != request_step:
            _fail(
                "committed fragment version differs from request step", context=context
            )
        if _require_int(commit.get("wire_dtype"), f"{context}.wire_dtype") != DTYPE_F32:
            _fail("commit is not f32 wire dtype", context=context)
        broadcast_payload_sha256 = _require_sha256(
            commit.get("broadcast_payload_sha256"),
            f"{context}.broadcast_payload_sha256",
        )
        responders = _require_list(commit.get("responders"), f"{context}.responders")
        responder_count = _require_int(
            commit.get("responder_count"), f"{context}.responder_count", minimum=0
        )
        if responder_count != len(responders):
            _fail("responder_count differs from responder array", context=context)
        responder_ids = [
            _require_int(
                _require_mapping(row, f"{context}.responders[{index}]").get(
                    "learner_id"
                ),
                f"{context}.responders[{index}].learner_id",
                minimum=0,
            )
            for index, row in enumerate(responders)
        ]
        if responder_ids != expected_responders:
            _fail(
                f"commit responders {responder_ids!r} != expected {expected_responders!r}",
                context=context,
            )
        normalized_responders: list[ResponderRef] = []
        for index, raw_responder in enumerate(responders):
            responder_context = f"{context}.responders[{index}]"
            responder = _require_mapping(raw_responder, responder_context)
            responder_index = _require_int(
                responder.get("responder_index"),
                f"{responder_context}.responder_index",
                minimum=0,
            )
            if responder_index != index:
                _fail(
                    "responder_index is not exact merge order",
                    context=responder_context,
                )
            source_seq = _require_int(
                responder.get("source_attempt_event_seq"),
                f"{responder_context}.source_attempt_event_seq",
                minimum=1,
            )
            if source_seq >= event_seq or source_seq not in primary_by_event:
                _fail(
                    "responder references no previous primary attempt",
                    context=responder_context,
                )
            source = primary_by_event[source_seq]
            if source.get("disposition") != "admitted_pending":
                _fail(
                    "responder source attempt was not admitted",
                    context=responder_context,
                )
            if source.get("fragment_id") != fragment_id:
                _fail(
                    "responder source fragment differs from commit",
                    context=responder_context,
                )
            if source.get("request_global_step") != request_step:
                _fail(
                    "responder source request step differs from commit",
                    context=responder_context,
                )
            _match_responder_to_attempt(responder, source, responder_context)
            candidate_key = (responder["learner_id"], responder["attempt_serial"])
            if candidate_key not in candidates:
                _fail(
                    "responder has no learner capture candidate",
                    context=responder_context,
                )
            candidate = candidates[candidate_key]
            _match_primary_attempt(source, candidate.metadata, responder_context)
            attempt_serial = _require_int(
                responder.get("attempt_serial"),
                f"{responder_context}.attempt_serial",
                minimum=1,
            )
            base_version = _require_int(
                responder.get("base_version"),
                f"{responder_context}.base_version",
                minimum=0,
            )
            local_step = _require_int(
                responder.get("local_step"),
                f"{responder_context}.local_step",
                minimum=0,
            )
            c_steps = _require_int(
                responder.get("c_steps"), f"{responder_context}.c_steps", minimum=0
            )
            c_tokens = _require_int(
                responder.get("c_tokens"), f"{responder_context}.c_tokens", minimum=0
            )
            weight_f64_bits = _require_transcript_weight(
                responder,
                c_steps=c_steps,
                c_tokens=c_tokens,
                context=responder_context,
            )
            normalized_responders.append(
                ResponderRef(
                    responder_index=responder_index,
                    learner_id=responder["learner_id"],
                    source_attempt_event_seq=source_seq,
                    attempt_serial=attempt_serial,
                    window_uuid=responder["window_uuid"],
                    base_version=base_version,
                    local_step=local_step,
                    c_steps=c_steps,
                    c_tokens=c_tokens,
                    weight_f64_bits=weight_f64_bits,
                    payload_sha256=_require_sha256(
                        responder.get("received_payload_sha256"),
                        f"{responder_context}.received_payload_sha256",
                    ),
                    push=candidate.push,
                    richardson=candidate.richardson,
                )
            )
        normalized_boundaries.append(
            CommittedBoundary(
                capture_session_uuid=session,
                commit_event_seq=event_seq,
                fragment_id=fragment_id,
                request_global_step=request_step,
                committed_fragment_version=committed_fragment_version,
                broadcast_payload_sha256=broadcast_payload_sha256,
                responders=tuple(normalized_responders),
            )
        )
        joined.append(commit_key)

    joined.sort()
    joined_by_fragment = Counter(fragment_id for fragment_id, _ in joined)
    if len(joined) < expectations.min_joined_boundaries:
        _fail(
            f"only {len(joined)} transcript-committed boundaries; expected at least "
            f"{expectations.min_joined_boundaries}"
        )
    for fragment_id in range(expectations.fragments):
        if joined_by_fragment[fragment_id] < expectations.min_joined_per_fragment:
            _fail(
                f"fragment {fragment_id} has {joined_by_fragment[fragment_id]} committed "
                f"boundaries; expected at least {expectations.min_joined_per_fragment}"
            )
    transcript_summary = {
        "path": relative_path,
        "sha256": transcript_digest,
        "capture_session_uuid": session,
        "events": len(rows),
        "primary_attempts": len(primary_by_key),
        "commits": len(commits),
    }
    return _TranscriptValidation(
        summary=transcript_summary,
        joined_keys=tuple(joined),
        joined_by_fragment=dict(joined_by_fragment),
        sha256=transcript_digest,
        boundaries=tuple(normalized_boundaries),
    )


def _validate_campaign_full(
    arm_dir: Path,
    expectations: Expectations,
    response_transcript: Path | None = None,
) -> CampaignValidation:
    """Run the only trusted validation path and retain its normalized join."""

    arm_dir = arm_dir.resolve(strict=True)
    if not arm_dir.is_dir():
        _fail("arm path is not a directory", context=str(arm_dir))
    expected_dirs = {
        f"optimizer_state_capture_learner_{learner_id}"
        for learner_id in expectations.learner_ids
    }
    actual_dirs = {
        child.name
        for child in arm_dir.iterdir()
        if child.is_dir() and CAPTURE_DIRECTORY.fullmatch(child.name)
    }
    if actual_dirs != expected_dirs:
        _fail(
            f"capture learner directories differ: expected={sorted(expected_dirs)!r} "
            f"actual={sorted(actual_dirs)!r}"
        )
    learners = [
        _validate_learner(arm_dir, learner_id, expectations)
        for learner_id in expectations.learner_ids
    ]
    layouts = {learner.layout_sha256 for learner in learners}
    if len(layouts) != 1:
        _fail("learner layout digests differ")
    layout_sha256 = next(iter(layouts))
    response_transcript = (
        response_transcript or arm_dir / "syncer_response_transcript.jsonl"
    )
    transcript = _validate_response_transcript(
        arm_dir, response_transcript, learners, expectations
    )
    all_tree_entries: dict[str, str] = {}
    for learner in learners:
        overlap = set(all_tree_entries).intersection(learner.tree_entries)
        if overlap:
            _fail(f"duplicate tree paths across learners: {sorted(overlap)!r}")
        all_tree_entries.update(learner.tree_entries)
    transcript_path = response_transcript.resolve(strict=True)
    transcript_relative = _relative_posix(transcript_path, arm_dir)
    if transcript_relative in all_tree_entries:
        _fail("response transcript collides with a capture artifact path")
    all_tree_entries[transcript_relative] = transcript.sha256
    tree_text = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(all_tree_entries.items())
    )
    tree_digest = hashlib.sha256(tree_text.encode("ascii")).hexdigest()
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "expected": expectations.as_json(),
        "learners": [learner.summary() for learner in learners],
        "response_transcript": transcript.summary,
        "join_mode": "authoritative_syncer_commit",
        "join_key_fields": ["fragment_id", "request_global_step"],
        "joined_boundaries": len(transcript.joined_keys),
        "joined_by_fragment": {
            str(fragment_id): transcript.joined_by_fragment.get(fragment_id, 0)
            for fragment_id in range(expectations.fragments)
        },
        "joined_keys": [
            {"fragment_id": fragment_id, "boundary": boundary}
            for fragment_id, boundary in transcript.joined_keys
        ],
        "unlisted_artifacts": [],
        "missing_artifacts": [],
        "temporary_files": [],
        "errors": [],
        "tree_manifest_path": "optimizer_state_capture_artifacts.sha256",
        "tree_manifest_sha256": tree_digest,
        "tree_entries": len(all_tree_entries),
    }
    boundary_index = CommittedBoundaryIndex(
        source_transcript=ArtifactRef(
            path=transcript_relative,
            sha256=transcript.sha256,
        ),
        capture_session_uuid=transcript.summary["capture_session_uuid"],
        layout_sha256=layout_sha256,
        source_tree_manifest_sha256=tree_digest,
        expected=expectations,
        boundaries=transcript.boundaries,
    )
    return CampaignValidation(
        summary=summary,
        tree_manifest_text=tree_text,
        boundary_index=boundary_index,
    )


def validate_campaign(
    arm_dir: Path,
    expectations: Expectations,
    response_transcript: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate one completed arm and return its summary and tree-manifest text."""

    result = _validate_campaign_full(
        arm_dir,
        expectations,
        response_transcript=response_transcript,
    )
    return result.summary, result.tree_manifest_text


def extract_committed_boundary_index(
    arm_dir: Path,
    expectations: Expectations,
    response_transcript: Path | None = None,
) -> CommittedBoundaryIndex:
    """Validate a complete campaign and return its authoritative joined index."""

    return _validate_campaign_full(
        arm_dir,
        expectations,
        response_transcript=response_transcript,
    ).boundary_index


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_and_sidecar(path: Path, summary: Mapping[str, Any]) -> None:
    raw = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    _atomic_write(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )


def _write_committed_boundary_index(path: Path, index: CommittedBoundaryIndex) -> str:
    """Atomically write a canonical boundary index and return its SHA-256."""

    raw = index.canonical_bytes()
    _atomic_write(path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    _atomic_write(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def validate_and_write(
    arm_dir: Path,
    expectations: Expectations,
    *,
    response_transcript: Path | None = None,
    output: Path | None = None,
    tree_manifest: Path | None = None,
    boundary_index: Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically write PASS/FAIL evidence.

    A failure summary is retained for diagnosis, but a stale tree manifest is
    removed so it cannot satisfy a completion gate from a previous attempt.
    """

    arm_dir = arm_dir.resolve()
    output = output or arm_dir / "optimizer_state_capture_validation.json"
    tree_manifest = (
        tree_manifest or arm_dir / "optimizer_state_capture_artifacts.sha256"
    )
    boundary_index = (
        boundary_index or arm_dir / "optimizer_state_capture_committed_boundaries.json"
    )
    try:
        result = _validate_campaign_full(
            arm_dir, expectations, response_transcript=response_transcript
        )
    except (ValidationError, OSError) as exc:
        tree_manifest.unlink(missing_ok=True)
        boundary_index.unlink(missing_ok=True)
        boundary_index.with_suffix(boundary_index.suffix + ".sha256").unlink(
            missing_ok=True
        )
        failure = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "expected": expectations.as_json(),
            "errors": [str(exc)],
        }
        _write_json_and_sidecar(output, failure)
        raise
    summary = result.summary
    _atomic_write(tree_manifest, result.tree_manifest_text.encode("ascii"))
    summary["tree_manifest_path"] = os.path.relpath(tree_manifest, arm_dir)
    summary["tree_manifest_sha256"] = _sha256_file(tree_manifest)
    boundary_digest = _write_committed_boundary_index(
        boundary_index, result.boundary_index
    )
    summary["committed_boundary_index_path"] = os.path.relpath(boundary_index, arm_dir)
    summary["committed_boundary_index_sha256"] = boundary_digest
    _write_json_and_sidecar(output, summary)
    return summary


def _parse_learner_ids(value: str) -> tuple[int, ...]:
    try:
        learner_ids = tuple(int(part) for part in value.split(",") if part != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected learners must be comma-separated non-negative integers"
        ) from exc
    if not learner_ids or any(learner_id < 0 for learner_id in learner_ids):
        raise argparse.ArgumentTypeError(
            "expected learners must be comma-separated non-negative integers"
        )
    if len(set(learner_ids)) != len(learner_ids):
        raise argparse.ArgumentTypeError("expected learner ids must be unique")
    return learner_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", required=True, type=Path)
    parser.add_argument("--expected-learners", required=True, type=_parse_learner_ids)
    parser.add_argument("--expected-fragments", required=True, type=int)
    parser.add_argument("--expected-h", required=True, type=int)
    parser.add_argument("--expected-every", required=True, type=int)
    parser.add_argument(
        "--expected-capture-profile",
        choices=sorted(CAPTURE_PROFILES),
        default=CAPTURE_PROFILE_FULL,
    )
    parser.add_argument("--expected-max-hmc-events", required=True, type=int)
    parser.add_argument("--expected-max-midpoint-windows", required=True, type=int)
    parser.add_argument("--expected-max-bytes", required=True, type=int)
    parser.add_argument("--min-joined-boundaries", type=int, default=0)
    parser.add_argument("--min-joined-per-fragment", type=int, default=0)
    parser.add_argument(
        "--response-transcript",
        type=Path,
        default=None,
        help="authoritative syncer transcript (default: ARM_DIR/syncer_response_transcript.jsonl)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tree-manifest", type=Path, default=None)
    parser.add_argument("--boundary-index", type=Path, default=None)
    parser.add_argument(
        "--strict-writer",
        action="store_true",
        help="fail on every capture drop except incomplete shutdown-tail state",
    )
    parser.add_argument(
        "--expected-background-writer",
        action="store_true",
        help="require a drained successful bounded background writer in every manifest",
    )
    parser.add_argument("--expected-background-writer-max-items", type=int, default=0)
    parser.add_argument("--expected-background-writer-max-bytes", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        expectations = Expectations(
            learner_ids=args.expected_learners,
            fragments=args.expected_fragments,
            window_steps=args.expected_h,
            every=args.expected_every,
            max_hmc_events=args.expected_max_hmc_events,
            max_midpoint_windows=args.expected_max_midpoint_windows,
            max_bytes=args.expected_max_bytes,
            min_joined_boundaries=args.min_joined_boundaries,
            min_joined_per_fragment=args.min_joined_per_fragment,
            strict_writer=args.strict_writer,
            background_writer=args.expected_background_writer,
            background_writer_max_items=(args.expected_background_writer_max_items),
            background_writer_max_bytes=(args.expected_background_writer_max_bytes),
            capture_profile=args.expected_capture_profile,
        )
        summary = validate_and_write(
            args.arm_dir,
            expectations,
            response_transcript=args.response_transcript,
            output=args.output,
            tree_manifest=args.tree_manifest,
            boundary_index=args.boundary_index,
        )
    except (ValidationError, OSError) as exc:
        print(f"optimizer capture validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        "optimizer capture validation passed: "
        f"joined={summary['joined_boundaries']} mode={summary['join_mode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
