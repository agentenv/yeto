"""Core protocol and evaluation helpers for online action probing.

The action-probe service is deliberately separate from the training and
syncer processes.  A request contains the complete current LoRA state and
five *already constructed* trial fragments. ``A0`` is always the configured
fallback and ``A1`` through ``A4`` are alternatives from one declared action
family. The Python side never reconstructs merge or outer-optimizer math.

Frames use a small dependency-free wire format::

    8-byte magic | u32 JSON length | u64 payload length | JSON | payload

Integers are network byte order.  Tensor payloads are contiguous little-
endian f32.  The JSON header names every byte range and carries a SHA-256 for
each tensor/action.  This makes malformed or partially written requests fail
closed before a model is touched.

Evaluate request contract (``protocol=yeto-action-probe-v1``):

* Header identity: ``request_id``, ``run_uuid``, ``step``, ``fragment_id``,
  ``base_version``, ``state_epoch``, ``fragment_versions``, ``layout_hash``,
  ``anchor_manifest_sha256``, and ``probe_config_sha256``.
* ``state.tensors`` describes the complete named LoRA state in payload order.
* ``fragment.tensor_names`` is the exact ordered model fragment.
  ``fragment.action_family`` is ``leave_one_out`` (also the default for
  pre-family v1 frames) or ``step_scale``. Its five ``actions`` are A0-A4 f32
  ranges plus SHA-256 and family-specific safety metadata. No merge math is
  reconstructed here.
* The binary payload is all complete-state tensors followed by A0-A4.

Successful responses return per-panel losses, paired-LCB diagnostics,
selected action and selected-action digest, worker/timing data, and state,
anchor, layout/probe, and trial digests. Any malformed request, timeout,
worker error, unsafe action, or confidence failure selects local A0; the
future Rust client must still verify request/action digests before commit.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
import struct
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import build_packed_dataset
from .losses import sft_loss


PROTOCOL = "yeto-action-probe-v1"
FRAME_MAGIC = b"YETOAP01"
FRAME_PREFIX = struct.Struct("!8sIQ")
MAX_HEADER_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024
ACTION_NAMES = ("A0", "A1", "A2", "A3", "A4")
BASELINE_ACTION = "A0"
LEAVE_ONE_OUT_ACTION_FAMILY = "leave_one_out"
STEP_SCALE_ACTION_FAMILY = "step_scale"
SUPPORTED_ACTION_FAMILIES = (
    LEAVE_ONE_OUT_ACTION_FAMILY,
    STEP_SCALE_ACTION_FAMILY,
)
# Short alias for protocol consumers that treat the tuple as wire vocabulary.
ACTION_FAMILIES = SUPPORTED_ACTION_FAMILIES
MIN_SELECTED_MASS = 0.70
MIN_NORM_MULTIPLIER = 0.5
MAX_NORM_MULTIPLIER = 2.0
MAX_STEP_NORM_RELATIVE_ERROR = 0.01
CANONICALIZATION = "yeto-messages-tools-v1"
MANIFEST_SCHEMA = "disjoint_hf_holdout_v1"


class ActionProbeError(RuntimeError):
    """Base class for errors that must result in an A0 fallback."""


class ProtocolError(ActionProbeError):
    """Malformed or inconsistent wire data."""


class ManifestError(ActionProbeError):
    """The anchor manifest or its materialized data failed validation."""


class EvaluationError(ActionProbeError):
    """A request cannot be evaluated without violating fail-closed rules."""


class RequestValidationError(EvaluationError):
    """A model-incompatible request rejected before any model mutation."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"header is not finite JSON: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Frame:
    """One decoded wire frame.

    ``digest`` covers the exact prefix, JSON bytes, and binary payload.  Two
    semantically equivalent JSON objects with different wire encodings are
    intentionally not exact retries.
    """

    header: dict[str, Any]
    payload: bytes
    digest: str


def encode_frame(header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    if not isinstance(header, Mapping):
        raise ProtocolError("frame header must be a JSON object")
    if not isinstance(payload, bytes):
        raise ProtocolError("frame payload must be bytes")
    header_bytes = _json_bytes(header)
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ProtocolError(
            f"frame header is {len(header_bytes)} bytes, limit is {MAX_HEADER_BYTES}"
        )
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"frame payload is {len(payload)} bytes, limit is {MAX_PAYLOAD_BYTES}"
        )
    prefix = FRAME_PREFIX.pack(FRAME_MAGIC, len(header_bytes), len(payload))
    return prefix + header_bytes + payload


def decode_frame(data: bytes) -> Frame:
    """Decode exactly one in-memory frame; trailing bytes are rejected."""

    if len(data) < FRAME_PREFIX.size:
        raise ProtocolError("truncated frame prefix")
    magic, header_len, payload_len = FRAME_PREFIX.unpack_from(data)
    _validate_frame_lengths(magic, header_len, payload_len)
    expected = FRAME_PREFIX.size + header_len + payload_len
    if len(data) != expected:
        raise ProtocolError(f"frame has {len(data)} bytes, expected exactly {expected}")
    return _decode_frame_parts(
        data[: FRAME_PREFIX.size],
        data[FRAME_PREFIX.size : FRAME_PREFIX.size + header_len],
        data[FRAME_PREFIX.size + header_len :],
    )


def _validate_frame_lengths(magic: bytes, header_len: int, payload_len: int) -> None:
    if magic != FRAME_MAGIC:
        raise ProtocolError(f"bad frame magic {magic!r}")
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"invalid frame header length {header_len}")
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"invalid frame payload length {payload_len}")


def _decode_frame_parts(prefix: bytes, header_bytes: bytes, payload: bytes) -> Frame:
    def object_without_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ProtocolError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str):
        raise ProtocolError(f"non-finite JSON constant {value}")

    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except ProtocolError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ProtocolError(f"invalid frame JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ProtocolError("frame JSON must be an object")
    return Frame(
        header=header,
        payload=payload,
        digest=_sha256(prefix + header_bytes + payload),
    )


def _recv_exact(
    sock: socket.socket,
    size: int,
    *,
    clean_eof: bool = False,
    deadline: float | None = None,
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("frame receive deadline exceeded")
            sock.settimeout(remaining)
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            if not chunks and clean_eof:
                raise EOFError
            raise ProtocolError(
                f"connection closed after {len(chunks)} of {size} bytes"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def recv_frame(sock: socket.socket, *, timeout_s: float | None = None) -> Frame:
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    prefix = _recv_exact(sock, FRAME_PREFIX.size, clean_eof=True, deadline=deadline)
    magic, header_len, payload_len = FRAME_PREFIX.unpack(prefix)
    _validate_frame_lengths(magic, header_len, payload_len)
    header_bytes = _recv_exact(sock, header_len, deadline=deadline)
    payload = _recv_exact(sock, payload_len, deadline=deadline) if payload_len else b""
    return _decode_frame_parts(prefix, header_bytes, payload)


def send_frame(
    sock: socket.socket, header: Mapping[str, Any], payload: bytes = b""
) -> None:
    sock.sendall(encode_frame(header, payload))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise ProtocolError(f"expected a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point():
        raise ProtocolError(f"tensor dtype must be floating point, got {tensor.dtype}")
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    if not bool(torch.isfinite(flat).all()):
        raise ProtocolError("tensor contains NaN or Inf")
    return flat.view(torch.uint8).numpy().tobytes()


def tensor_mapping_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """Canonical digest of a named f32 tensor mapping."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        shape = [int(dim) for dim in tensor.shape]
        raw = _tensor_bytes(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_json_bytes({"shape": shape}))
        digest.update(b"\0")
        digest.update(raw)
    return digest.hexdigest()


def probe_config_digest(config: Mapping[str, Any]) -> str:
    """Digest the complete static evaluator/model configuration."""

    return _sha256(_json_bytes(config))


def _normalize_trial(
    trial: torch.Tensor | Mapping[str, torch.Tensor],
    fragment_names: Sequence[str],
) -> torch.Tensor:
    if isinstance(trial, torch.Tensor):
        return trial.detach().reshape(-1)
    if not isinstance(trial, Mapping):
        raise ProtocolError("trial must be a flat tensor or a named tensor mapping")
    missing = [name for name in fragment_names if name not in trial]
    extra = sorted(set(trial) - set(fragment_names))
    if missing or extra:
        raise ProtocolError(f"trial names mismatch: missing={missing}, extra={extra}")
    return torch.cat([trial[name].detach().reshape(-1) for name in fragment_names])


def build_evaluate_frame(
    *,
    request_id: str,
    run_uuid: str,
    step: int,
    fragment_id: int,
    base_version: int,
    state_epoch: int,
    fragment_versions: Sequence[int],
    layout_hash: str,
    anchor_manifest_sha256: str,
    probe_config_sha256: str,
    current_state: Mapping[str, torch.Tensor],
    fragment_names: Sequence[str],
    trials: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    action_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    action_family: str = LEAVE_ONE_OUT_ACTION_FAMILY,
) -> bytes:
    """Build one canonical evaluate frame for a future syncer client.

    Rust owns action construction.  This helper exists for tests and Python
    clients and intentionally accepts exact trial parameter tensors rather
    than candidate deltas or merge metadata.
    """

    if not request_id or not run_uuid:
        raise ProtocolError("request_id and run_uuid must be non-empty")
    if action_family not in SUPPORTED_ACTION_FAMILIES:
        raise ProtocolError(
            f"action_family must be one of {list(SUPPORTED_ACTION_FAMILIES)}"
        )
    if set(trials) != set(ACTION_NAMES):
        raise ProtocolError(f"trials must contain exactly {list(ACTION_NAMES)}")
    if action_family == STEP_SCALE_ACTION_FAMILY and action_metadata is None:
        raise ProtocolError("step_scale actions require explicit action_metadata")
    state_names = sorted(current_state)
    if not state_names:
        raise ProtocolError("current_state is empty")
    if not fragment_names or len(set(fragment_names)) != len(fragment_names):
        raise ProtocolError("fragment_names must be a non-empty unique sequence")
    missing = [name for name in fragment_names if name not in current_state]
    if missing:
        raise ProtocolError(
            f"fragment tensors are missing from current_state: {missing}"
        )

    payload = bytearray()
    state_specs = []
    for name in state_names:
        tensor = current_state[name]
        raw = _tensor_bytes(tensor)
        spec = {
            "name": name,
            "shape": [int(dim) for dim in tensor.shape],
            "offset": len(payload),
            "nbytes": len(raw),
            "sha256": _sha256(raw),
        }
        payload.extend(raw)
        state_specs.append(spec)

    fragment_numel = sum(int(current_state[name].numel()) for name in fragment_names)
    action_specs = []
    for action in ACTION_NAMES:
        trial = _normalize_trial(trials[action], fragment_names)
        if int(trial.numel()) != fragment_numel:
            raise ProtocolError(
                f"{action} has {trial.numel()} values, expected {fragment_numel}"
            )
        raw = _tensor_bytes(trial)
        if action_metadata is None:
            action_index = ACTION_NAMES.index(action)
            metadata = {
                "eligible": True,
                "omitted_responder_id": None
                if action == BASELINE_ACTION
                else action_index - 1,
                "selected_mass": 1.0 if action == BASELINE_ACTION else 0.75,
                "norm_multiplier": 1.0,
                "step_norm_ratio": 1.0,
                "ineligible_reason": None,
            }
        else:
            if set(action_metadata) != set(ACTION_NAMES):
                raise ProtocolError(
                    f"action_metadata must contain exactly {list(ACTION_NAMES)}"
                )
            metadata = dict(action_metadata[action])
        action_specs.append(
            {
                **metadata,
                "name": action,
                "offset": len(payload),
                "nbytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
        payload.extend(raw)

    header = {
        "protocol": PROTOCOL,
        "type": "evaluate",
        "request_id": request_id,
        "run_uuid": run_uuid,
        "step": int(step),
        "fragment_id": int(fragment_id),
        "base_version": int(base_version),
        "state_epoch": int(state_epoch),
        "fragment_versions": [int(version) for version in fragment_versions],
        "layout_hash": layout_hash,
        "anchor_manifest_sha256": anchor_manifest_sha256,
        "probe_config_sha256": probe_config_sha256,
        "dtype": "f32le",
        "state": {
            "tensors": state_specs,
            "sha256": tensor_mapping_digest(current_state),
        },
        "fragment": {
            "action_family": action_family,
            "tensor_names": list(fragment_names),
            "numel": fragment_numel,
            "actions": action_specs,
        },
    }
    return encode_frame(header, bytes(payload))


@dataclass(frozen=True)
class EvaluateRequest:
    request_id: str
    run_uuid: str
    step: int
    fragment_id: int
    base_version: int
    state_epoch: int
    fragment_versions: tuple[int, ...]
    layout_hash: str
    anchor_manifest_sha256: str
    probe_config_sha256: str
    current_state: OrderedDict[str, torch.Tensor]
    current_state_digest: str
    action_family: str
    fragment_names: tuple[str, ...]
    trials: dict[str, torch.Tensor]
    action_digests: dict[str, str]
    action_eligibility: dict[str, bool]
    action_metadata: dict[str, dict[str, Any]]
    request_digest: str


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProtocolError(f"{field} is not hexadecimal") from exc
    return value.lower()


def _require_finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ProtocolError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be a finite number")
    return result


def _slice_f32(
    payload: bytes, spec: Mapping[str, Any], expected_offset: int
) -> tuple[torch.Tensor, int, str]:
    if not isinstance(spec, Mapping):
        raise ProtocolError("tensor descriptor must be an object")
    offset = _require_int(spec.get("offset"), "tensor.offset")
    nbytes = _require_int(spec.get("nbytes"), "tensor.nbytes", minimum=4)
    if offset != expected_offset:
        raise ProtocolError(
            f"payload range starts at {offset}, expected {expected_offset}"
        )
    if nbytes % 4:
        raise ProtocolError(f"f32 payload length {nbytes} is not divisible by 4")
    end = offset + nbytes
    if end > len(payload):
        raise ProtocolError(
            f"payload range [{offset}, {end}) exceeds {len(payload)} bytes"
        )
    raw = payload[offset:end]
    digest = _require_digest(spec.get("sha256"), "tensor.sha256")
    if _sha256(raw) != digest:
        raise ProtocolError("tensor SHA-256 mismatch")
    # bytearray gives torch a writable, owned backing store; clone severs the
    # final view and makes the request independent of the socket frame.
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    if not bool(torch.isfinite(tensor).all()):
        raise ProtocolError("tensor payload contains NaN or Inf")
    return tensor, end, digest


def parse_evaluate_request(frame: Frame) -> EvaluateRequest:
    header = frame.header
    if header.get("protocol") != PROTOCOL or header.get("type") != "evaluate":
        raise ProtocolError("not an action-probe evaluate request")
    if header.get("dtype") != "f32le":
        raise ProtocolError("only little-endian f32 payloads are supported")
    if sys.byteorder != "little":
        raise ProtocolError("f32le action probing requires a little-endian host")

    request_id = header.get("request_id")
    run_uuid = header.get("run_uuid")
    layout_hash = header.get("layout_hash")
    for value, field in (
        (request_id, "request_id"),
        (run_uuid, "run_uuid"),
        (layout_hash, "layout_hash"),
    ):
        if not isinstance(value, str) or not value:
            raise ProtocolError(f"{field} must be a non-empty string")
        if len(value) > 256:
            raise ProtocolError(f"{field} exceeds the 256-character limit")
    layout_hash = _require_digest(layout_hash, "layout_hash")
    versions = header.get("fragment_versions")
    if not isinstance(versions, list) or not versions:
        raise ProtocolError("fragment_versions must be a non-empty list")
    fragment_versions = tuple(
        _require_int(value, f"fragment_versions[{index}]")
        for index, value in enumerate(versions)
    )
    fragment_id = _require_int(header.get("fragment_id"), "fragment_id")
    if fragment_id >= len(fragment_versions):
        raise ProtocolError(
            f"fragment_id {fragment_id} is outside {len(fragment_versions)} versions"
        )
    base_version = _require_int(header.get("base_version"), "base_version")
    if base_version != fragment_versions[fragment_id]:
        raise ProtocolError(
            "base_version must equal fragment_versions[fragment_id] for the current state"
        )

    state = header.get("state")
    if not isinstance(state, Mapping) or not isinstance(state.get("tensors"), list):
        raise ProtocolError("state.tensors must be a list")
    if not state["tensors"]:
        raise ProtocolError("state.tensors is empty")
    current_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for index, spec in enumerate(state["tensors"]):
        if not isinstance(spec, Mapping):
            raise ProtocolError(f"state.tensors[{index}] must be an object")
        name = spec.get("name")
        shape = spec.get("shape")
        if not isinstance(name, str) or not name or name in current_state:
            raise ProtocolError(f"invalid or duplicate state tensor name {name!r}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in shape
            )
        ):
            raise ProtocolError(f"state tensor {name!r} has invalid shape {shape!r}")
        flat, offset, _ = _slice_f32(frame.payload, spec, offset)
        numel = math.prod(shape)
        if flat.numel() != numel:
            raise ProtocolError(
                f"state tensor {name!r} has {flat.numel()} values, shape requires {numel}"
            )
        current_state[name] = flat.view(shape)

    expected_state_digest = _require_digest(state.get("sha256"), "state.sha256")
    actual_state_digest = tensor_mapping_digest(current_state)
    if actual_state_digest != expected_state_digest:
        raise ProtocolError("complete state SHA-256 mismatch")

    fragment = header.get("fragment")
    if not isinstance(fragment, Mapping):
        raise ProtocolError("fragment must be an object")
    action_family = fragment.get("action_family", LEAVE_ONE_OUT_ACTION_FAMILY)
    if action_family not in SUPPORTED_ACTION_FAMILIES:
        raise ProtocolError(
            f"fragment.action_family must be one of {list(SUPPORTED_ACTION_FAMILIES)}"
        )
    fragment_names_value = fragment.get("tensor_names")
    if not isinstance(fragment_names_value, list) or not fragment_names_value:
        raise ProtocolError("fragment.tensor_names must be a non-empty list")
    fragment_names = tuple(fragment_names_value)
    if any(not isinstance(name, str) or not name for name in fragment_names):
        raise ProtocolError("fragment.tensor_names contains an invalid name")
    if len(set(fragment_names)) != len(fragment_names):
        raise ProtocolError("fragment.tensor_names contains duplicates")
    missing = [name for name in fragment_names if name not in current_state]
    if missing:
        raise ProtocolError(
            f"fragment tensors are absent from complete state: {missing}"
        )
    fragment_numel = sum(int(current_state[name].numel()) for name in fragment_names)
    if (
        _require_int(fragment.get("numel"), "fragment.numel", minimum=1)
        != fragment_numel
    ):
        raise ProtocolError("fragment.numel does not match fragment tensor shapes")

    action_specs = fragment.get("actions")
    if not isinstance(action_specs, list) or len(action_specs) != len(ACTION_NAMES):
        raise ProtocolError(
            f"fragment.actions must contain exactly {len(ACTION_NAMES)} actions"
        )
    trials: dict[str, torch.Tensor] = {}
    action_digests: dict[str, str] = {}
    action_eligibility: dict[str, bool] = {}
    action_metadata: dict[str, dict[str, Any]] = {}
    step_scales: dict[str, float] = {}
    for index, spec in enumerate(action_specs):
        if not isinstance(spec, Mapping):
            raise ProtocolError(f"fragment.actions[{index}] must be an object")
        action = spec.get("name")
        if action not in ACTION_NAMES or action in trials:
            raise ProtocolError(f"invalid or duplicate action name {action!r}")
        flat, offset, digest = _slice_f32(frame.payload, spec, offset)
        if flat.numel() != fragment_numel:
            raise ProtocolError(
                f"{action} has {flat.numel()} values, expected {fragment_numel}"
            )
        trials[action] = flat
        action_digests[action] = digest
        eligible = spec.get("eligible")
        if not isinstance(eligible, bool):
            raise ProtocolError(f"{action}.eligible must be boolean")
        omitted = spec.get("omitted_responder_id")
        if omitted is not None:
            omitted = _require_int(omitted, f"{action}.omitted_responder_id")
        selected_mass = _require_finite_float(
            spec.get("selected_mass"), f"{action}.selected_mass"
        )
        norm_multiplier = _require_finite_float(
            spec.get("norm_multiplier"), f"{action}.norm_multiplier"
        )
        step_norm_ratio = _require_finite_float(
            spec.get("step_norm_ratio"), f"{action}.step_norm_ratio"
        )
        reason = spec.get("ineligible_reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 256):
            raise ProtocolError(f"{action}.ineligible_reason is invalid")
        if not 0 <= selected_mass <= 1:
            raise ProtocolError(f"{action}.selected_mass must be in [0, 1]")
        if norm_multiplier <= 0 or step_norm_ratio <= 0:
            raise ProtocolError(
                f"{action} norm_multiplier and step_norm_ratio must be positive"
            )
        if action == BASELINE_ACTION:
            if not eligible or omitted is not None:
                raise ProtocolError("A0 must be eligible and omit no responder")
        if action_family == LEAVE_ONE_OUT_ACTION_FAMILY:
            if "step_scale" in spec:
                raise ProtocolError(
                    f"{action}.step_scale is invalid for leave_one_out actions"
                )
            if action != BASELINE_ACTION and eligible:
                if omitted is None:
                    raise ProtocolError(f"{action} must identify its omitted responder")
                if selected_mass < MIN_SELECTED_MASS:
                    raise ProtocolError(
                        f"{action} is marked eligible below minimum selected mass"
                    )
                if not (MIN_NORM_MULTIPLIER <= norm_multiplier <= MAX_NORM_MULTIPLIER):
                    raise ProtocolError(
                        f"{action} is marked eligible with unsafe norm multiplier"
                    )
                if abs(step_norm_ratio - 1.0) > MAX_STEP_NORM_RELATIVE_ERROR + 1e-12:
                    raise ProtocolError(
                        f"{action} is marked eligible with unsafe step norm ratio"
                    )
        else:
            if omitted is not None:
                raise ProtocolError(
                    f"{action} must not omit a responder for step_scale actions"
                )
            if selected_mass != 1.0:
                raise ProtocolError(
                    f"{action}.selected_mass must be exactly 1 for step_scale actions"
                )
            step_scale = _require_finite_float(
                spec.get("step_scale"), f"{action}.step_scale"
            )
            if step_scale <= 0:
                raise ProtocolError(f"{action}.step_scale must be positive")
            step_scales[action] = step_scale
        action_eligibility[action] = eligible
        action_metadata[action] = {
            "eligible": eligible,
            "omitted_responder_id": omitted,
            "selected_mass": selected_mass,
            "norm_multiplier": norm_multiplier,
            "step_norm_ratio": step_norm_ratio,
            "ineligible_reason": reason,
        }
        if action_family == STEP_SCALE_ACTION_FAMILY:
            action_metadata[action]["step_scale"] = step_scales[action]
    if set(trials) != set(ACTION_NAMES):
        raise ProtocolError(f"actions must be exactly {list(ACTION_NAMES)}")
    if action_family == LEAVE_ONE_OUT_ACTION_FAMILY:
        omitted_responders = [
            action_metadata[action]["omitted_responder_id"]
            for action in ACTION_NAMES[1:]
        ]
        if any(responder is None for responder in omitted_responders) or len(
            set(omitted_responders)
        ) != len(omitted_responders):
            raise ProtocolError("A1-A4 must identify four distinct omitted responders")
    elif len(step_scales) != len(ACTION_NAMES) or len(set(step_scales.values())) != len(
        ACTION_NAMES
    ):
        raise ProtocolError("A0-A4 must have five unique step scales")
    if offset != len(frame.payload):
        raise ProtocolError(
            f"payload has {len(frame.payload) - offset} unclaimed bytes"
        )

    return EvaluateRequest(
        request_id=request_id,
        run_uuid=run_uuid,
        step=_require_int(header.get("step"), "step"),
        fragment_id=fragment_id,
        base_version=base_version,
        state_epoch=_require_int(header.get("state_epoch"), "state_epoch"),
        fragment_versions=fragment_versions,
        layout_hash=layout_hash,
        anchor_manifest_sha256=_require_digest(
            header.get("anchor_manifest_sha256"), "anchor_manifest_sha256"
        ),
        probe_config_sha256=_require_digest(
            header.get("probe_config_sha256"), "probe_config_sha256"
        ),
        current_state=current_state,
        current_state_digest=actual_state_digest,
        action_family=action_family,
        fragment_names=fragment_names,
        trials=trials,
        action_digests=action_digests,
        action_eligibility=action_eligibility,
        action_metadata=action_metadata,
        request_digest=frame.digest,
    )


def canonical_anchor_row(row: Any, *, context: str = "row") -> dict[str, Any]:
    if not isinstance(row, Mapping):
        try:
            row = dict(row)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{context}: expected an object row") from exc
    try:
        plain = json.loads(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{context}: row is not finite JSON: {exc}") from exc
    messages = plain.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ManifestError(f"{context}: expected a non-empty messages list")
    canonical = {"messages": messages}
    if plain.get("tools"):
        canonical["tools"] = plain["tools"]
    return canonical


def canonical_anchor_hash(row: Mapping[str, Any]) -> str:
    return _sha256(_canonical_anchor_payload(row))


def _canonical_anchor_payload(row: Mapping[str, Any]) -> bytes:
    return json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        f"{path}:{line_number}: malformed JSON: {exc}"
                    ) from exc
                rows.append(canonical_anchor_row(row, context=f"{path}:{line_number}"))
    except OSError as exc:
        raise ManifestError(f"cannot read anchor data {path}: {exc}") from exc
    if not rows:
        raise ManifestError(f"anchor data {path} is empty")
    return rows


@dataclass(frozen=True)
class AnchorManifest:
    path: Path
    data_path: Path
    rows: tuple[dict[str, Any], ...]
    manifest_sha256: str
    data_sha256: str
    canonical_hashes: tuple[str, ...]
    raw: dict[str, Any]


def load_anchor_manifest(path: str | Path) -> AnchorManifest:
    """Load and independently verify a disjoint-holdout anchor manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
        manifest = json.loads(raw_bytes)
    except OSError as exc:
        raise ManifestError(
            f"cannot read anchor manifest {manifest_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"anchor manifest is malformed JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("anchor manifest must be a JSON object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"unsupported anchor manifest schema {manifest.get('schema')!r}"
        )
    if manifest.get("canonicalization") != CANONICALIZATION:
        raise ManifestError(
            f"unsupported canonicalization {manifest.get('canonicalization')!r}"
        )
    if manifest.get("verified_zero_overlap") is not True:
        raise ManifestError(
            "anchor manifest does not assert verified_zero_overlap=true"
        )
    if manifest.get("overlap_count") != 0:
        raise ManifestError(
            f"anchor manifest overlap_count is {manifest.get('overlap_count')!r}"
        )

    def resolve_artifact(value: str) -> Path:
        candidate = Path(value).expanduser()
        candidates = (
            [candidate, manifest_path.parent / candidate.name]
            if candidate.is_absolute()
            else [
                manifest_path.parent / candidate,
                manifest_path.parent / candidate.name,
            ]
        )
        for item in candidates:
            resolved = item.resolve()
            if resolved.is_file():
                return resolved
        return candidates[0].resolve()

    output_path = manifest.get("output_path")
    output_digest = _require_digest(manifest.get("output_sha256"), "output_sha256")
    if not isinstance(output_path, str) or not output_path:
        raise ManifestError("anchor manifest output_path must be non-empty")
    data_path = resolve_artifact(output_path)
    if not data_path.is_file():
        raise ManifestError(f"anchor data file is missing: {data_path}")
    actual_data_digest = _sha256_file(data_path)
    if actual_data_digest != output_digest:
        raise ManifestError(
            f"anchor data SHA-256 mismatch: manifest={output_digest}, actual={actual_data_digest}"
        )

    rows = _read_jsonl(data_path)
    selected_count = manifest.get("selected_count")
    if selected_count != len(rows):
        raise ManifestError(
            f"anchor row count mismatch: manifest={selected_count}, actual={len(rows)}"
        )
    expected_hashes = manifest.get("selected_canonical_hashes")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(rows):
        raise ManifestError("selected_canonical_hashes does not match selected_count")
    actual_hashes = tuple(canonical_anchor_hash(row) for row in rows)
    if tuple(expected_hashes) != actual_hashes:
        raise ManifestError("anchor canonical row hashes do not match the manifest")
    if len(set(actual_hashes)) != len(actual_hashes):
        raise ManifestError("anchor manifest contains duplicate canonical examples")

    exclusions = manifest.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise ManifestError("anchor manifest must include canonical exclusion evidence")
    excluded_hashes: set[str] = set()
    for index, record in enumerate(exclusions):
        if not isinstance(record, Mapping):
            raise ManifestError(f"exclusions[{index}] must be an object")
        exclusion_path_value = record.get("path")
        if not isinstance(exclusion_path_value, str) or not exclusion_path_value:
            raise ManifestError(f"exclusions[{index}].path must be non-empty")
        exclusion_path = resolve_artifact(exclusion_path_value)
        if not exclusion_path.is_file():
            raise ManifestError(f"exclusion file is missing: {exclusion_path}")
        expected_file_digest = _require_digest(
            record.get("sha256"), f"exclusions[{index}].sha256"
        )
        actual_file_digest = _sha256_file(exclusion_path)
        if actual_file_digest != expected_file_digest:
            raise ManifestError(f"exclusion file SHA-256 mismatch for {exclusion_path}")
        exclusion_rows = _read_jsonl(exclusion_path)
        canonical_payload = b"".join(
            _canonical_anchor_payload(row) + b"\n" for row in exclusion_rows
        )
        expected_canonical_digest = _require_digest(
            record.get("canonical_sha256"),
            f"exclusions[{index}].canonical_sha256",
        )
        if _sha256(canonical_payload) != expected_canonical_digest:
            raise ManifestError(
                f"canonical exclusion SHA-256 mismatch for {exclusion_path}"
            )
        excluded_hashes.update(canonical_anchor_hash(row) for row in exclusion_rows)
    overlap = set(actual_hashes) & excluded_hashes
    if overlap:
        raise ManifestError(
            f"anchor data overlaps {len(overlap)} canonical exclusion examples"
        )
    expected_unique_excluded = manifest.get("unique_excluded_canonical_count")
    if expected_unique_excluded is not None and expected_unique_excluded != len(
        excluded_hashes
    ):
        raise ManifestError(
            "unique_excluded_canonical_count does not match verified exclusions"
        )

    return AnchorManifest(
        path=manifest_path,
        data_path=data_path,
        rows=tuple(rows),
        manifest_sha256=_sha256(raw_bytes),
        data_sha256=actual_data_digest,
        canonical_hashes=actual_hashes,
        raw=manifest,
    )


def build_anchor_panels(
    manifest: AnchorManifest,
    tokenizer,
    *,
    seq_len: int,
    panels: int,
    blocks_per_panel: int,
    train_on: str = "assistant",
    device: torch.device | str = "cpu",
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], str]:
    """Materialize deterministic, static teacher-forced anchor panels.

    Canonical rows are assigned round-robin to panels before token packing,
    so no conversation/source row contributes to more than one statistical
    panel. A panel then takes its first ``blocks_per_panel`` packed blocks
    that contain positive teacher-forced target weight after the causal
    shift. This keeps the paired confidence unit at conversation-cluster
    granularity rather than pretending adjacent token blocks are independent.
    """

    if seq_len <= 1 or panels < 2 or blocks_per_panel <= 0:
        raise ManifestError(
            "seq_len > 1, panels >= 2, and blocks_per_panel > 0 are required"
        )
    device = torch.device(device)
    materialized = []
    digest = hashlib.sha256()
    digest.update(manifest.manifest_sha256.encode("ascii"))
    digest.update(f"\n{seq_len}\n{panels}\n{blocks_per_panel}\n{train_on}\n".encode())
    for panel_id in range(panels):
        panel_rows = list(manifest.rows[panel_id::panels])
        if not panel_rows:
            raise ManifestError(
                f"anchor has {len(manifest.rows)} rows, fewer than {panels} panels"
            )
        try:
            panel_dataset = build_packed_dataset(
                panel_rows,
                tokenizer,
                learner_id=0,
                num_learners=1,
                seq_len=seq_len,
                max_rows=len(panel_rows),
                train_on=train_on,
            )
        except ValueError as exc:
            raise ManifestError(
                f"anchor panel {panel_id} cannot form a block: {exc}"
            ) from exc
        target_block_indices = [
            index
            for index in range(len(panel_dataset))
            if float(panel_dataset.weights[index, 1:].sum().item()) > 0.0
        ]
        if len(target_block_indices) < blocks_per_panel:
            raise ManifestError(
                f"anchor panel {panel_id} has only {len(target_block_indices)} "
                "packed blocks with teacher-forced target tokens; "
                f"need {blocks_per_panel}"
            )
        selected_indices = target_block_indices[:blocks_per_panel]
        ids = panel_dataset.blocks[selected_indices].contiguous()
        weights = panel_dataset.weights[selected_indices].contiguous()
        digest.update(ids.view(torch.uint8).numpy().tobytes())
        digest.update(weights.view(torch.uint8).numpy().tobytes())
        materialized.append(
            (
                ids.to(device=device, non_blocking=False),
                weights.to(device=device, non_blocking=False),
            )
        )
    # The outer tuple prevents accidental panel replacement. Tensor contents
    # are treated as immutable by the evaluator.
    return tuple((ids, weights) for ids, weights in materialized), digest.hexdigest()


def evaluate_panel_losses(
    model,
    panels: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_function: str = "cross_entropy",
) -> list[float]:
    """Return one teacher-forced loss/token value per anchor panel."""

    model.eval()
    losses = []
    with torch.inference_mode():
        for panel_id, (input_ids, weights) in enumerate(panels):
            output = model(input_ids=input_ids, use_cache=False)
            loss, token_count = sft_loss(
                output.logits, input_ids, loss_function, weights
            )
            count = float(token_count.detach().item())
            value = float(loss.detach().item()) / max(count, 1.0)
            if count <= 0 or not math.isfinite(value):
                raise EvaluationError(
                    f"panel {panel_id} produced invalid loss={value}, tokens={count}"
                )
            losses.append(value)
    return losses


@dataclass(frozen=True)
class SelectionConfig:
    min_gain: float = 0.00025
    lcb_z: float = 2.365
    min_win_rate: float = 0.75
    min_panels: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError("min_gain must be finite and >= 0")
        if not math.isfinite(self.lcb_z) or self.lcb_z < 0:
            raise ValueError("lcb_z must be finite and >= 0")
        if not 0 <= self.min_win_rate <= 1:
            raise ValueError("min_win_rate must be in [0, 1]")
        if self.min_panels < 2:
            raise ValueError("min_panels must be >= 2")


@dataclass(frozen=True)
class ActionStatistic:
    action: str
    mean_gain: float
    standard_error: float
    lcb: float
    win_rate: float
    wins: int
    panels: int
    eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "mean_gain": self.mean_gain,
            "standard_error": self.standard_error,
            "lcb": self.lcb,
            "win_rate": self.win_rate,
            "wins": self.wins,
            "panels": self.panels,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class SelectionResult:
    selected_action: str
    fallback_reason: str | None
    statistics: tuple[ActionStatistic, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "fallback_reason": self.fallback_reason,
            "statistics": [stat.as_dict() for stat in self.statistics],
        }


def select_paired_lcb(
    losses_by_action: Mapping[str, Sequence[float]],
    config: SelectionConfig = SelectionConfig(),
    *,
    eligible_actions: Sequence[str] | None = None,
    baseline_losses_by_action: Mapping[str, Sequence[float]] | None = None,
    action_multipliers: Mapping[str, float] | None = None,
) -> SelectionResult:
    """Select an action by deterministic paired-panel confidence bounds.

    Gain is ``loss(A0) - loss(Ai)``.  A nonbaseline action must have a
    positive LCB, mean gain at least ``min_gain``, and the configured panel
    win rate.  Any malformed/non-finite input returns A0 rather than raising.
    Ties are broken by LCB, mean gain, and win rate. Scalar actions may also
    provide their multipliers, in which case ties prefer the multiplier
    closest to 1.0 and then the lower multiplier. Other action families retain
    the fixed action order.
    """

    try:
        if set(losses_by_action) != set(ACTION_NAMES):
            raise ValueError("loss map does not contain exactly A0-A4")
        normalized = {
            action: tuple(float(value) for value in losses_by_action[action])
            for action in ACTION_NAMES
        }
        panel_count = len(normalized[BASELINE_ACTION])
        if panel_count < config.min_panels:
            raise ValueError("insufficient panels")
        if any(len(values) != panel_count for values in normalized.values()):
            raise ValueError("actions have different panel counts")
        if any(
            not math.isfinite(value) or value < 0
            for values in normalized.values()
            for value in values
        ):
            raise ValueError("loss map contains a negative, NaN, or Inf value")
        allowed = (
            set(ACTION_NAMES[1:]) if eligible_actions is None else set(eligible_actions)
        )
        if not allowed <= set(ACTION_NAMES[1:]):
            raise ValueError("eligible_actions contains an invalid action")
        normalized_multipliers = None
        if action_multipliers is not None:
            if set(action_multipliers) != set(ACTION_NAMES[1:]):
                raise ValueError("action_multipliers must contain A1-A4")
            normalized_multipliers = {
                action: float(action_multipliers[action])
                for action in ACTION_NAMES[1:]
            }
            if any(
                not math.isfinite(multiplier) or multiplier <= 0
                for multiplier in normalized_multipliers.values()
            ):
                raise ValueError("action_multipliers contains an invalid value")
            if len(set(normalized_multipliers.values())) != len(
                normalized_multipliers
            ):
                raise ValueError("action_multipliers contains duplicates")
        if baseline_losses_by_action is None:
            paired_baselines = {
                action: normalized[BASELINE_ACTION] for action in ACTION_NAMES[1:]
            }
        else:
            if set(baseline_losses_by_action) != set(ACTION_NAMES[1:]):
                raise ValueError("paired baseline map must contain A1-A4")
            paired_baselines = {
                action: tuple(
                    float(value) for value in baseline_losses_by_action[action]
                )
                for action in ACTION_NAMES[1:]
            }
            if any(len(values) != panel_count for values in paired_baselines.values()):
                raise ValueError("paired baselines have different panel counts")
            if any(
                not math.isfinite(value) or value < 0
                for values in paired_baselines.values()
                for value in values
            ):
                raise ValueError("paired baseline contains invalid loss")
    except (TypeError, ValueError, OverflowError):
        return SelectionResult(BASELINE_ACTION, "invalid_losses", ())

    stats = []
    try:
        for action in ACTION_NAMES[1:]:
            gains = [
                base - trial
                for base, trial in zip(paired_baselines[action], normalized[action])
            ]
            if any(not math.isfinite(gain) for gain in gains):
                raise ValueError("paired gain overflowed")
            mean_gain = statistics.fmean(gains)
            standard_error = statistics.stdev(gains) / math.sqrt(panel_count)
            lcb = mean_gain - config.lcb_z * standard_error
            if not all(
                math.isfinite(value) for value in (mean_gain, standard_error, lcb)
            ):
                raise ValueError("paired statistic is not finite")
            wins = sum(gain > 0 for gain in gains)
            win_rate = wins / panel_count
            eligible = (
                action in allowed
                and lcb > 0
                and mean_gain >= config.min_gain
                and win_rate >= config.min_win_rate
            )
            stats.append(
                ActionStatistic(
                    action=action,
                    mean_gain=mean_gain,
                    standard_error=standard_error,
                    lcb=lcb,
                    win_rate=win_rate,
                    wins=wins,
                    panels=panel_count,
                    eligible=eligible,
                )
            )
    except (OverflowError, statistics.StatisticsError, ValueError):
        return SelectionResult(BASELINE_ACTION, "invalid_losses", ())
    eligible = [stat for stat in stats if stat.eligible]
    if not eligible:
        return SelectionResult(BASELINE_ACTION, "no_action_passed", tuple(stats))
    if normalized_multipliers is not None:
        winner = max(
            eligible,
            key=lambda stat: (
                stat.lcb,
                stat.mean_gain,
                stat.win_rate,
                -abs(normalized_multipliers[stat.action] - 1.0),
                -normalized_multipliers[stat.action],
            ),
        )
    else:
        action_rank = {action: index for index, action in enumerate(ACTION_NAMES)}
        winner = max(
            eligible,
            key=lambda stat: (
                stat.lcb,
                stat.mean_gain,
                stat.win_rate,
                -action_rank[stat.action],
            ),
        )
    return SelectionResult(winner.action, None, tuple(stats))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ActionProbeReplica:
    """One persistent model replica and immutable anchor-panel cache."""

    def __init__(
        self,
        model,
        panels: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        anchor_manifest_sha256: str,
        anchor_tensors_sha256: str,
        probe_config_sha256: str,
        layout_hash: str,
        fragment_layout: Mapping[int, Sequence[str]],
        device: torch.device | str,
        loss_function: str = "cross_entropy",
    ):
        self.model = model
        self.panels = tuple(panels)
        self.anchor_manifest_sha256 = _require_digest(
            anchor_manifest_sha256, "anchor_manifest_sha256"
        )
        self.anchor_tensors_sha256 = _require_digest(
            anchor_tensors_sha256, "anchor_tensors_sha256"
        )
        self.probe_config_sha256 = _require_digest(
            probe_config_sha256, "probe_config_sha256"
        )
        self.layout_hash = _require_digest(layout_hash, "layout_hash")
        self.fragment_layout = {
            int(fragment_id): tuple(names)
            for fragment_id, names in fragment_layout.items()
        }
        if sorted(self.fragment_layout) != list(range(len(self.fragment_layout))):
            raise EvaluationError("fragment layout IDs must be contiguous from zero")
        self.device = torch.device(device)
        self.loss_function = loss_function
        self.params = OrderedDict(
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        if not self.params:
            raise EvaluationError("model has no trainable LoRA parameters")
        layout_names = [
            name
            for fragment_id in sorted(self.fragment_layout)
            for name in self.fragment_layout[fragment_id]
        ]
        if len(layout_names) != len(set(layout_names)) or set(layout_names) != set(
            self.params
        ):
            raise EvaluationError(
                "fragment layout must cover every trainable tensor exactly once"
            )
        non_f32 = [
            name for name, param in self.params.items() if param.dtype != torch.float32
        ]
        if non_f32:
            raise EvaluationError(
                "exact f32 restoration requires fp32 trainable parameters; "
                f"non-f32 examples: {non_f32[:4]}"
            )
        self.model.eval()

    def _validate_state(self, request: EvaluateRequest) -> None:
        expected = set(self.params)
        provided = set(request.current_state)
        if expected != provided:
            raise RequestValidationError(
                "complete LoRA state names mismatch: "
                f"missing={sorted(expected - provided)[:8]}, "
                f"extra={sorted(provided - expected)[:8]}"
            )
        for name, param in self.params.items():
            if tuple(param.shape) != tuple(request.current_state[name].shape):
                raise RequestValidationError(
                    f"state shape mismatch for {name}: model={tuple(param.shape)}, "
                    f"request={tuple(request.current_state[name].shape)}"
                )

    def _apply_state(self, state: Mapping[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.params.items():
                param.copy_(state[name].to(device=self.device, dtype=torch.float32))

    def _apply_fragment(self, names: Sequence[str], flat: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for name in names:
                param = self.params[name]
                end = offset + param.numel()
                param.copy_(
                    flat[offset:end]
                    .view_as(param)
                    .to(device=self.device, dtype=torch.float32)
                )
                offset = end
        if offset != flat.numel():
            raise EvaluationError(
                f"fragment consumed {offset} values from a {flat.numel()}-value trial"
            )

    def _state_is_exact(self, state: Mapping[str, torch.Tensor]) -> bool:
        for name, param in self.params.items():
            expected = state[name].to(device=self.device, dtype=torch.float32)
            if not torch.equal(param.detach(), expected):
                return False
        return True

    def evaluate(
        self, request: EvaluateRequest, actions: Sequence[str]
    ) -> dict[str, Any]:
        if request.anchor_manifest_sha256 != self.anchor_manifest_sha256:
            raise RequestValidationError("request targets a different anchor manifest")
        if request.probe_config_sha256 != self.probe_config_sha256:
            raise RequestValidationError(
                "request targets a different probe configuration"
            )
        if request.layout_hash != self.layout_hash:
            raise RequestValidationError("request targets a different fragment layout")
        expected_fragment_names = self.fragment_layout.get(request.fragment_id)
        if expected_fragment_names != request.fragment_names:
            raise RequestValidationError(
                f"fragment {request.fragment_id} tensor order does not match the evaluator layout"
            )
        if not actions or any(action not in ACTION_NAMES for action in actions):
            raise RequestValidationError(f"invalid action assignment {list(actions)!r}")
        if len(set(actions)) != len(actions):
            raise RequestValidationError("action assignment contains duplicates")
        self._validate_state(request)
        started = time.perf_counter()
        action_results = {}
        primary_error: Exception | None = None
        restore_error: Exception | None = None
        try:
            self._apply_state(request.current_state)
            if not self._state_is_exact(request.current_state):
                raise EvaluationError("complete LoRA state did not apply exactly")
            current_fragment = torch.cat(
                [
                    request.current_state[name].reshape(-1)
                    for name in request.fragment_names
                ]
            )
            for action in actions:
                action_started = time.perf_counter()
                self._apply_fragment(request.fragment_names, request.trials[action])
                _sync_device(self.device)
                eval_started = time.perf_counter()
                eval_finished = eval_started
                try:
                    losses = evaluate_panel_losses(
                        self.model,
                        self.panels,
                        loss_function=self.loss_function,
                    )
                    _sync_device(self.device)
                    eval_finished = time.perf_counter()
                finally:
                    # Restore before recording or moving to another trial. A
                    # second full-state restore in the outer finally protects
                    # against a partially applied fragment.
                    self._apply_fragment(request.fragment_names, current_fragment)
                    for name in request.fragment_names:
                        expected = request.current_state[name].to(
                            device=self.device, dtype=torch.float32
                        )
                        if not torch.equal(self.params[name].detach(), expected):
                            raise EvaluationError(
                                f"fragment restoration was not exact for {name}"
                            )
                action_results[action] = {
                    "panel_losses": losses,
                    "eval_ms": (eval_finished - eval_started) * 1000.0,
                    "total_ms": (time.perf_counter() - action_started) * 1000.0,
                    "trial_sha256": request.action_digests[action],
                }
        except Exception as exc:
            primary_error = exc
        finally:
            restore_started = time.perf_counter()
            try:
                self._apply_state(request.current_state)
                _sync_device(self.device)
                if not self._state_is_exact(request.current_state):
                    raise EvaluationError("model state restoration was not exact")
            except (
                Exception
            ) as exc:  # preserve a restoration failure over an eval failure
                restore_error = exc
            restore_ms = (time.perf_counter() - restore_started) * 1000.0
        if restore_error is not None:
            raise EvaluationError(
                f"failed to restore current state: {restore_error}"
            ) from restore_error
        if primary_error is not None:
            raise primary_error
        return {
            "actions": action_results,
            "state_sha256": request.current_state_digest,
            "state_restored": True,
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "anchor_tensors_sha256": self.anchor_tensors_sha256,
            "probe_config_sha256": self.probe_config_sha256,
            "restore_ms": restore_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }
