"""Canonical exact tensor packs built on :mod:`yeto.capture_v2_store`.

The codec snapshots named tensors into one immutable CAS payload and publishes
one immutable CAS manifest describing how to decode it.  It is deliberately a
storage codec, not a learner hook: callers decide which tensors and clocks form
one causally valid capture boundary.

Trainable tensors are required to be fp32.  Optimizer tensors retain their
exact supported dtype, including scalar step tensors.  Plain clocks are
non-negative signed-64-bit integers.  Tensor payload bytes use canonical
little-endian, contiguous storage and are ordered by category and ASCII name.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ManifestRef,
    ObjectRef,
)


SCHEMA = "yeto.capture-v2-tensor-pack"
SCHEMA_VERSION = 1
PAYLOAD_ROLE = "tensor-pack/payload"
BYTE_ORDER = "little"
TENSOR_ORDER = "trainable-then-optimizer-by-ascii-name"

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,511}\Z")
_MANIFEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CLOCK = 2**63 - 1
_KINDS = ("trainable", "optimizer")
_KIND_ORDER = {kind: index for index, kind in enumerate(_KINDS)}

_DTYPE_TO_CODE = {
    torch.bool: "bool",
    torch.uint8: "u8",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
_DTYPE_BYTES = {
    code: torch.empty((), dtype=dtype).element_size()
    for code, dtype in _CODE_TO_DTYPE.items()
}


class TensorPackError(CaptureStoreError):
    """A tensor pack cannot be encoded or fails exact validation."""


@dataclass(frozen=True)
class TensorPackRef:
    """Immutable identities returned after publishing a tensor pack."""

    manifest: ManifestRef
    payload: ObjectRef
    payload_inserted: bool


@dataclass
class DecodedTensorPack:
    """Fresh, independent CPU tensors reconstructed from a verified pack."""

    manifest_id: str
    manifest_sha256: str
    payload: ObjectRef
    trainable: dict[str, torch.Tensor]
    optimizer: dict[str, torch.Tensor]
    clocks: dict[str, int]
    metadata: dict[str, Any]


def _validate_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or _NAME_RE.fullmatch(value) is None:
        raise TensorPackError(f"{context} must be a safe non-empty ASCII name")
    if value.startswith("/") or any(
        component in ("", ".", "..") for component in value.split("/")
    ):
        raise TensorPackError(f"{context} contains an unsafe path component")
    return value


def _validate_exact_int(value: Any, context: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise TensorPackError(f"{context} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise TensorPackError(f"{context} exceeds {maximum}")
    return value


def _ordered_items(values: Mapping[str, Any], context: str) -> list[tuple[str, Any]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{context} must be a mapping")
    items: list[tuple[str, Any]] = []
    for name, value in values.items():
        items.append((_validate_name(name, f"{context} name"), value))
    return sorted(items, key=lambda item: item[0])


def _snapshot_tensor(
    tensor: torch.Tensor, *, kind: str, name: str
) -> tuple[torch.Tensor, str, bytes]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{kind} tensor {name!r} must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise TensorPackError(f"{kind} tensor {name!r} must use strided layout")
    if tensor.device.type == "meta":
        raise TensorPackError(f"{kind} tensor {name!r} cannot be a meta tensor")
    if kind == "trainable" and tensor.dtype != torch.float32:
        raise TensorPackError(
            f"trainable tensor {name!r} must be fp32, got {tensor.dtype}"
        )
    try:
        dtype_code = _DTYPE_TO_CODE[tensor.dtype]
    except KeyError as exc:
        raise TensorPackError(
            f"{kind} tensor {name!r} has unsupported dtype {tensor.dtype}"
        ) from exc

    # ``copy=True`` is essential: serialization must not retain an alias to
    # a live parameter, gradient, or optimizer state tensor.
    snapshot = tensor.detach().to(device="cpu", copy=True).contiguous()
    raw = snapshot.reshape(-1).view(torch.uint8).numpy().tobytes()
    return snapshot, dtype_code, raw


def _descriptor(
    *, kind: str, name: str, dtype: str, shape: list[int], offset: int, raw: bytes
) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "offset": offset,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _clock_rows(clocks: Mapping[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in _ordered_items(clocks, "clock"):
        rows.append(
            {
                "name": name,
                "value": _validate_exact_int(
                    value, f"clock {name!r}", maximum=_MAX_CLOCK
                ),
            }
        )
    return rows


def _snapshot_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    try:
        raw = (
            json.dumps(
                dict(metadata),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        value = json.loads(raw)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TensorPackError(f"metadata is not canonical JSON data: {exc}") from exc
    if not isinstance(value, dict):  # Defensive; a Mapping always encodes as an object.
        raise TensorPackError("metadata must encode as a JSON object")
    return value


def publish_tensor_pack(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    trainable: Mapping[str, torch.Tensor],
    optimizer: Mapping[str, torch.Tensor],
    clocks: Mapping[str, int],
    metadata: Mapping[str, Any] | None = None,
) -> TensorPackRef:
    """Snapshot, encode, and atomically publish one canonical tensor pack.

    Mapping insertion order, tensor strides, device, and aliases do not affect
    the content identity.  At least one named trainable fp32 tensor is required.
    """

    if sys.byteorder != BYTE_ORDER:
        raise TensorPackError("tensor-pack v1 requires a little-endian host")
    if (
        not isinstance(manifest_id, str)
        or _MANIFEST_ID_RE.fullmatch(manifest_id) is None
    ):
        raise TensorPackError("manifest_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")

    trainable_items = _ordered_items(trainable, "trainable")
    optimizer_items = _ordered_items(optimizer, "optimizer")
    clock_rows = _clock_rows(clocks)
    metadata_value = _snapshot_metadata(metadata)
    if not trainable_items:
        raise TensorPackError("tensor pack requires at least one trainable tensor")

    payload = bytearray()
    descriptors: list[dict[str, Any]] = []
    for kind, items in (
        ("trainable", trainable_items),
        ("optimizer", optimizer_items),
    ):
        for name, tensor in items:
            snapshot, dtype_code, raw = _snapshot_tensor(tensor, kind=kind, name=name)
            descriptors.append(
                _descriptor(
                    kind=kind,
                    name=name,
                    dtype=dtype_code,
                    shape=list(snapshot.shape),
                    offset=len(payload),
                    raw=raw,
                )
            )
            payload.extend(raw)

    payload_result = store.put_bytes(bytes(payload))
    pack_metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "byte_order": BYTE_ORDER,
        "tensor_order": TENSOR_ORDER,
        "tensors": descriptors,
        "clocks": clock_rows,
        "metadata": metadata_value,
    }
    manifest = store.publish_manifest(
        manifest_id,
        [ManifestEntry(PAYLOAD_ROLE, payload_result.ref)],
        metadata=pack_metadata,
    )
    return TensorPackRef(manifest, payload_result.ref, payload_result.inserted)


def _shape(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise TensorPackError(f"{context} shape must be an array")
    return [
        _validate_exact_int(dimension, f"{context} shape dimension")
        for dimension in value
    ]


def _numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _decode_tensor(raw: bytes, dtype_code: str, shape: list[int]) -> torch.Tensor:
    dtype = _CODE_TO_DTYPE[dtype_code]
    if not raw:
        return torch.empty(shape, dtype=dtype)
    # A private bytearray followed by clone gives the returned tensor its own
    # storage.  No decoded tensors alias the payload buffer or one another.
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)


def _validate_clocks(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        raise TensorPackError("tensor-pack clocks must be an array")
    clocks: dict[str, int] = {}
    previous: str | None = None
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != {"name", "value"}:
            raise TensorPackError(f"tensor-pack clock row {index} is malformed")
        name = _validate_name(row["name"], f"clock row {index} name")
        if previous is not None and name <= previous:
            raise TensorPackError("tensor-pack clocks are not strictly name-sorted")
        previous = name
        clocks[name] = _validate_exact_int(
            row["value"], f"clock {name!r}", maximum=_MAX_CLOCK
        )
    return clocks


def _load_payload_bytes(store: CaptureObjectStore, ref: ObjectRef) -> bytes:
    path = store.verify_object(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TensorPackError(f"cannot read tensor-pack payload {path}: {exc}") from exc
    if len(raw) != ref.bytes:
        raise TensorPackError(
            f"tensor-pack payload size mismatch: expected {ref.bytes}, got {len(raw)}"
        )
    actual = hashlib.sha256(raw).hexdigest()
    if actual != ref.sha256:
        raise TensorPackError(
            f"tensor-pack payload SHA-256 mismatch: expected {ref.sha256}, got {actual}"
        )
    return raw


def load_tensor_pack(
    store: CaptureObjectStore, ref: TensorPackRef | ManifestRef | str
) -> DecodedTensorPack:
    """Load one pack only after strict CAS, schema, and per-tensor validation."""

    if sys.byteorder != BYTE_ORDER:
        raise TensorPackError("tensor-pack v1 requires a little-endian host")
    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, TensorPackRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    rows = manifest["objects"]
    if len(rows) != 1 or rows[0]["role"] != PAYLOAD_ROLE:
        raise TensorPackError(
            f"tensor-pack manifest must reference exactly one {PAYLOAD_ROLE!r} object"
        )
    payload_ref = ObjectRef(rows[0]["sha256"], rows[0]["bytes"])
    if isinstance(ref, TensorPackRef) and payload_ref != ref.payload:
        raise TensorPackError(
            "tensor-pack reference disagrees with its manifest payload"
        )

    metadata = manifest["metadata"]
    expected_metadata_keys = {
        "schema",
        "schema_version",
        "byte_order",
        "tensor_order",
        "tensors",
        "clocks",
        "metadata",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
        raise TensorPackError("tensor-pack manifest metadata fields are malformed")
    if metadata["schema"] != SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise TensorPackError("tensor-pack manifest uses an unsupported schema")
    if metadata["byte_order"] != BYTE_ORDER:
        raise TensorPackError("tensor-pack manifest uses an unsupported byte order")
    if metadata["tensor_order"] != TENSOR_ORDER:
        raise TensorPackError("tensor-pack manifest uses an unsupported tensor order")
    if not isinstance(metadata["metadata"], dict):
        raise TensorPackError("tensor-pack user metadata must be an object")

    raw_payload = _load_payload_bytes(store, payload_ref)
    descriptor_rows = metadata["tensors"]
    if not isinstance(descriptor_rows, list) or not descriptor_rows:
        raise TensorPackError("tensor-pack tensors must be a non-empty array")

    tensors: dict[str, dict[str, torch.Tensor]] = {
        "trainable": {},
        "optimizer": {},
    }
    expected_descriptor_keys = {
        "kind",
        "name",
        "dtype",
        "shape",
        "offset",
        "bytes",
        "sha256",
    }
    parsed_rows: list[tuple[dict[str, Any], str, str]] = []
    previous_order: tuple[int, str] | None = None
    for index, row in enumerate(descriptor_rows):
        context = f"tensor-pack tensor row {index}"
        if not isinstance(row, dict) or set(row) != expected_descriptor_keys:
            raise TensorPackError(f"{context} fields are malformed")
        kind = row["kind"]
        if not isinstance(kind, str) or kind not in _KIND_ORDER:
            raise TensorPackError(f"{context} has unsupported kind {kind!r}")
        name = _validate_name(row["name"], f"{context} name")
        order = (_KIND_ORDER[kind], name)
        if previous_order is not None and order <= previous_order:
            raise TensorPackError(
                "tensor-pack tensors are not strictly canonical-ordered"
            )
        previous_order = order
        parsed_rows.append((row, kind, name))

    cursor = 0
    for index, (row, kind, name) in enumerate(parsed_rows):
        context = f"tensor-pack tensor row {index}"
        dtype_code = row["dtype"]
        if not isinstance(dtype_code, str) or dtype_code not in _CODE_TO_DTYPE:
            raise TensorPackError(f"{context} has unsupported dtype {dtype_code!r}")
        if kind == "trainable" and dtype_code != "f32":
            raise TensorPackError(f"{context} trainable dtype must be f32")
        shape = _shape(row["shape"], context)
        offset = _validate_exact_int(row["offset"], f"{context} offset")
        byte_count = _validate_exact_int(row["bytes"], f"{context} bytes")
        if offset != cursor:
            raise TensorPackError(
                f"{context} is not contiguous: expected offset {cursor}, got {offset}"
            )
        expected_bytes = _numel(shape) * _DTYPE_BYTES[dtype_code]
        if byte_count != expected_bytes:
            raise TensorPackError(
                f"{context} byte count mismatch: expected {expected_bytes}, "
                f"got {byte_count}"
            )
        end = offset + byte_count
        if end > len(raw_payload):
            raise TensorPackError(f"{context} exceeds the payload boundary")
        digest = row["sha256"]
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise TensorPackError(f"{context} SHA-256 is malformed")
        tensor_raw = raw_payload[offset:end]
        actual_digest = hashlib.sha256(tensor_raw).hexdigest()
        if actual_digest != digest:
            raise TensorPackError(
                f"{context} SHA-256 mismatch: expected {digest}, got {actual_digest}"
            )
        tensors[kind][name] = _decode_tensor(tensor_raw, dtype_code, shape)
        cursor = end

    if not tensors["trainable"]:
        raise TensorPackError("tensor pack contains no trainable tensors")
    if cursor != len(raw_payload):
        raise TensorPackError(
            f"tensor-pack payload has {len(raw_payload) - cursor} trailing bytes"
        )
    clocks = _validate_clocks(metadata["clocks"])
    manifest_sha256 = (
        ref.manifest.sha256
        if isinstance(ref, TensorPackRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return DecodedTensorPack(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha256,
        payload=payload_ref,
        trainable=tensors["trainable"],
        optimizer=tensors["optimizer"],
        clocks=clocks,
        metadata=dict(metadata["metadata"]),
    )
