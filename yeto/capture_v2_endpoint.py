"""Strict learner-endpoint restore manifests for capture-v2.

This module binds already-published tensor packs and opaque exact-state objects
into one content-addressed learner endpoint.  It defines storage identity and
cross-reference rules only.  It does not decide when a learner is causally
safe to capture, mutate a model or optimizer, or claim any replay outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ManifestRef,
    ObjectRef,
)
from .capture_v2_tensor_pack import (
    DecodedTensorPack,
    TensorPackRef,
    load_tensor_pack,
)


SCHEMA = "yeto.capture-v2-learner-endpoint"
SCHEMA_VERSION = 1
FUTURE_GROUP_COUNT = 8
MODES = frozenset({"train", "eval"})

MODEL_BUFFERS_ROLE = "model/buffers"
PYTHON_RNG_ROLE = "rng/python"
NUMPY_RNG_ROLE = "rng/numpy"
TORCH_CPU_RNG_ROLE = "rng/torch-cpu"

_MAX_COUNTER = 2**63 - 1


class EndpointManifestError(CaptureStoreError):
    """A learner endpoint is malformed or has an invalid cross-reference."""


@dataclass(frozen=True)
class FutureGroupRefs:
    """Available future-group objects and their explicit completeness state.

    ``complete`` requires all indices 0 through 7 and no reason.  ``incomplete``
    requires fewer than eight available refs and a non-empty reason.
    """

    state: str
    refs: Mapping[int, ObjectRef]
    reason: str | None = None


@dataclass(frozen=True)
class EndpointRestoreRef:
    """Content identity of one published learner endpoint manifest."""

    manifest: ManifestRef


@dataclass(frozen=True)
class EndpointRngRefs:
    """Verified opaque objects holding exact RNG serialization bytes."""

    python: ObjectRef
    numpy: ObjectRef
    torch_cpu: ObjectRef
    torch_cuda: tuple[ObjectRef, ...]


@dataclass
class LoadedLearnerEndpoint:
    """A strictly validated endpoint and freshly decoded tensor packs."""

    manifest_id: str
    manifest_sha256: str
    mode: str
    fragment_versions: tuple[int, ...]
    fragments: dict[int, DecodedTensorPack]
    model_buffers: ObjectRef
    scheduler: dict[str, Any]
    scaler: dict[str, Any] | None
    rng: EndpointRngRefs
    future_groups: FutureGroupRefs


def _fragment_payload_role(fragment_id: int) -> str:
    return f"fragments/{fragment_id}/tensor-pack-payload"


def _cuda_rng_role(device_index: int) -> str:
    return f"rng/torch-cuda/{device_index}"


def _future_group_role(group_index: int) -> str:
    return f"future-groups/{group_index}"


def _exact_nonnegative_int(
    value: Any, context: str, *, maximum: int = _MAX_COUNTER
) -> int:
    if type(value) is not int or value < 0:
        raise EndpointManifestError(f"{context} must be a non-negative integer")
    if value > maximum:
        raise EndpointManifestError(f"{context} exceeds {maximum}")
    return value


def _exact_ref(value: Any, context: str) -> ObjectRef:
    if not isinstance(value, ObjectRef):
        raise TypeError(f"{context} must be an ObjectRef")
    return value


def _snapshot_json_object(
    value: Mapping[str, Any] | None, context: str, *, allow_none: bool
) -> dict[str, Any] | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{context} must be a mapping")
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    try:
        raw = (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        result = json.loads(raw)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EndpointManifestError(
            f"{context} is not canonical JSON data: {exc}"
        ) from exc
    if not isinstance(result, dict):  # Defensive; Mapping encodes as an object.
        raise EndpointManifestError(f"{context} must encode as a JSON object")
    return result


def _indexed_refs(
    value: Mapping[int, ObjectRef],
    context: str,
    *,
    require_contiguous: bool,
    maximum_exclusive: int | None = None,
) -> dict[int, ObjectRef]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    result: dict[int, ObjectRef] = {}
    for index, ref in value.items():
        index = _exact_nonnegative_int(index, f"{context} index")
        if maximum_exclusive is not None and index >= maximum_exclusive:
            raise EndpointManifestError(
                f"{context} index {index} is outside [0, {maximum_exclusive})"
            )
        result[index] = _exact_ref(ref, f"{context} {index}")
    result = dict(sorted(result.items()))
    if require_contiguous and list(result) != list(range(len(result))):
        raise EndpointManifestError(f"{context} indices must be contiguous from zero")
    return result


def _future_rows(future: FutureGroupRefs) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(future, FutureGroupRefs):
        raise TypeError("future_groups must be FutureGroupRefs")
    refs = _indexed_refs(
        future.refs,
        "future-group refs",
        require_contiguous=False,
        maximum_exclusive=FUTURE_GROUP_COUNT,
    )
    if future.state == "complete":
        if list(refs) != list(range(FUTURE_GROUP_COUNT)):
            raise EndpointManifestError(
                f"complete future groups require exactly {FUTURE_GROUP_COUNT} refs"
            )
        if future.reason is not None:
            raise EndpointManifestError("complete future groups cannot have a reason")
    elif future.state == "incomplete":
        if len(refs) >= FUTURE_GROUP_COUNT:
            raise EndpointManifestError(
                "incomplete future groups must have fewer than eight refs"
            )
        if (
            not isinstance(future.reason, str)
            or not future.reason.strip()
            or len(future.reason) > 1024
        ):
            raise EndpointManifestError(
                "incomplete future groups require a non-empty reason"
            )
    else:
        raise EndpointManifestError(
            "future-group state must be 'complete' or 'incomplete'"
        )
    return (
        [{"index": index, "role": _future_group_role(index)} for index in refs],
        future.reason,
    )


def _fragment_pack_rows(
    store: CaptureObjectStore, fragment_packs: Mapping[int, TensorPackRef]
) -> tuple[list[dict[str, Any]], dict[int, ObjectRef]]:
    if not isinstance(fragment_packs, Mapping):
        raise TypeError("fragment_packs must be a mapping")
    packs: dict[int, TensorPackRef] = {}
    for fragment_id, pack in fragment_packs.items():
        fragment_id = _exact_nonnegative_int(fragment_id, "fragment id")
        if not isinstance(pack, TensorPackRef):
            raise TypeError(f"fragment {fragment_id} pack must be a TensorPackRef")
        packs[fragment_id] = pack
    packs = dict(sorted(packs.items()))
    if not packs:
        raise EndpointManifestError("learner endpoint requires at least one fragment")
    if list(packs) != list(range(len(packs))):
        raise EndpointManifestError("fragment ids must be contiguous from zero")

    rows: list[dict[str, Any]] = []
    payloads: dict[int, ObjectRef] = {}
    for fragment_id, pack in packs.items():
        decoded = load_tensor_pack(store, pack)
        if decoded.payload != pack.payload:
            raise EndpointManifestError(
                f"fragment {fragment_id} tensor-pack payload identity mismatch"
            )
        payloads[fragment_id] = pack.payload
        rows.append(
            {
                "fragment_id": fragment_id,
                "tensor_pack_manifest_id": pack.manifest.manifest_id,
                "tensor_pack_manifest_sha256": pack.manifest.sha256,
                "tensor_pack_manifest_bytes": pack.manifest.bytes,
                "payload_role": _fragment_payload_role(fragment_id),
            }
        )
    return rows, payloads


def publish_learner_endpoint(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    fragment_packs: Mapping[int, TensorPackRef],
    fragment_versions: Sequence[int],
    mode: str,
    model_buffers: ObjectRef,
    scheduler: Mapping[str, Any],
    scaler: Mapping[str, Any] | None,
    python_rng: ObjectRef,
    numpy_rng: ObjectRef,
    torch_cpu_rng: ObjectRef,
    torch_cuda_rng: Mapping[int, ObjectRef],
    future_groups: FutureGroupRefs,
) -> EndpointRestoreRef:
    """Validate all cross-references, then publish one endpoint manifest."""

    if not isinstance(mode, str) or mode not in MODES:
        raise EndpointManifestError(f"mode must be one of {sorted(MODES)}")
    fragment_rows, fragment_payloads = _fragment_pack_rows(store, fragment_packs)
    if isinstance(fragment_versions, (str, bytes)) or not isinstance(
        fragment_versions, Sequence
    ):
        raise TypeError("fragment_versions must be a sequence")
    versions = [
        _exact_nonnegative_int(value, f"fragment version {index}")
        for index, value in enumerate(fragment_versions)
    ]
    if len(versions) != len(fragment_rows):
        raise EndpointManifestError(
            "fragment_versions length must equal the number of fragment packs"
        )

    model_buffers = _exact_ref(model_buffers, "model_buffers")
    python_rng = _exact_ref(python_rng, "python_rng")
    numpy_rng = _exact_ref(numpy_rng, "numpy_rng")
    torch_cpu_rng = _exact_ref(torch_cpu_rng, "torch_cpu_rng")
    cuda_refs = _indexed_refs(torch_cuda_rng, "torch CUDA RNG", require_contiguous=True)
    scheduler_value = _snapshot_json_object(
        scheduler, "scheduler metadata", allow_none=False
    )
    scaler_value = _snapshot_json_object(scaler, "scaler metadata", allow_none=True)
    future_rows, future_reason = _future_rows(future_groups)
    future_refs = _indexed_refs(
        future_groups.refs,
        "future-group refs",
        require_contiguous=False,
        maximum_exclusive=FUTURE_GROUP_COUNT,
    )

    entries: list[ManifestEntry] = [
        ManifestEntry(_fragment_payload_role(index), fragment_payloads[index])
        for index in fragment_payloads
    ]
    entries.extend(
        [
            ManifestEntry(MODEL_BUFFERS_ROLE, model_buffers),
            ManifestEntry(PYTHON_RNG_ROLE, python_rng),
            ManifestEntry(NUMPY_RNG_ROLE, numpy_rng),
            ManifestEntry(TORCH_CPU_RNG_ROLE, torch_cpu_rng),
        ]
    )
    entries.extend(
        ManifestEntry(_cuda_rng_role(index), ref) for index, ref in cuda_refs.items()
    )
    entries.extend(
        ManifestEntry(_future_group_role(index), ref)
        for index, ref in future_refs.items()
    )

    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "fragment_versions": versions,
        "fragments": fragment_rows,
        "model_buffers_role": MODEL_BUFFERS_ROLE,
        "scheduler": scheduler_value,
        "scaler": scaler_value,
        "rng": {
            "python_role": PYTHON_RNG_ROLE,
            "numpy_role": NUMPY_RNG_ROLE,
            "torch_cpu_role": TORCH_CPU_RNG_ROLE,
            "torch_cuda_roles": [_cuda_rng_role(index) for index in cuda_refs],
        },
        "future_groups": {
            "state": future_groups.state,
            "required": FUTURE_GROUP_COUNT,
            "available": future_rows,
            "reason": future_reason,
        },
    }
    manifest = store.publish_manifest(manifest_id, entries, metadata=metadata)
    return EndpointRestoreRef(manifest)


def _parse_fragment_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EndpointManifestError("endpoint fragments must be a non-empty array")
    expected_keys = {
        "fragment_id",
        "tensor_pack_manifest_id",
        "tensor_pack_manifest_sha256",
        "tensor_pack_manifest_bytes",
        "payload_role",
    }
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise EndpointManifestError(f"endpoint fragment row {index} is malformed")
        fragment_id = _exact_nonnegative_int(
            row["fragment_id"], f"endpoint fragment row {index} id"
        )
        if fragment_id != index:
            raise EndpointManifestError(
                "endpoint fragment ids must be canonical and contiguous from zero"
            )
        if row["payload_role"] != _fragment_payload_role(fragment_id):
            raise EndpointManifestError(
                f"endpoint fragment {fragment_id} payload role is noncanonical"
            )
        # ManifestRef performs strict id, digest, and byte-count validation.
        try:
            ManifestRef(
                row["tensor_pack_manifest_id"],
                row["tensor_pack_manifest_sha256"],
                row["tensor_pack_manifest_bytes"],
                False,
            )
        except CaptureStoreError as exc:
            raise EndpointManifestError(
                f"endpoint fragment {fragment_id} manifest reference is malformed: {exc}"
            ) from exc
        rows.append(row)
    return rows


def _parse_rng(value: Any) -> list[str]:
    expected_keys = {
        "python_role",
        "numpy_role",
        "torch_cpu_role",
        "torch_cuda_roles",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise EndpointManifestError("endpoint RNG fields are malformed")
    if value["python_role"] != PYTHON_RNG_ROLE:
        raise EndpointManifestError("endpoint Python RNG role is noncanonical")
    if value["numpy_role"] != NUMPY_RNG_ROLE:
        raise EndpointManifestError("endpoint NumPy RNG role is noncanonical")
    if value["torch_cpu_role"] != TORCH_CPU_RNG_ROLE:
        raise EndpointManifestError("endpoint Torch CPU RNG role is noncanonical")
    cuda_roles = value["torch_cuda_roles"]
    if not isinstance(cuda_roles, list):
        raise EndpointManifestError("endpoint Torch CUDA RNG roles must be an array")
    expected_cuda = [_cuda_rng_role(index) for index in range(len(cuda_roles))]
    if cuda_roles != expected_cuda:
        raise EndpointManifestError(
            "endpoint Torch CUDA RNG roles must be canonical and contiguous"
        )
    return cuda_roles


def _parse_future(value: Any) -> tuple[str, list[dict[str, Any]], str | None]:
    if not isinstance(value, dict) or set(value) != {
        "state",
        "required",
        "available",
        "reason",
    }:
        raise EndpointManifestError("endpoint future-group fields are malformed")
    if type(value["required"]) is not int or value["required"] != FUTURE_GROUP_COUNT:
        raise EndpointManifestError(
            f"endpoint future groups must require exactly {FUTURE_GROUP_COUNT}"
        )
    rows = value["available"]
    if not isinstance(rows, list):
        raise EndpointManifestError("endpoint available future groups must be an array")
    indices: list[int] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"index", "role"}:
            raise EndpointManifestError(
                f"endpoint future-group row {position} is malformed"
            )
        index = _exact_nonnegative_int(
            row["index"], f"endpoint future-group row {position} index"
        )
        if index >= FUTURE_GROUP_COUNT:
            raise EndpointManifestError(
                f"endpoint future-group index {index} is outside [0, 8)"
            )
        if position and index <= indices[-1]:
            raise EndpointManifestError(
                "endpoint future-group indices must be strictly sorted"
            )
        if row["role"] != _future_group_role(index):
            raise EndpointManifestError(
                f"endpoint future-group {index} role is noncanonical"
            )
        indices.append(index)

    state = value["state"]
    reason = value["reason"]
    if state == "complete":
        if indices != list(range(FUTURE_GROUP_COUNT)):
            raise EndpointManifestError(
                f"complete future groups require exactly {FUTURE_GROUP_COUNT} refs"
            )
        if reason is not None:
            raise EndpointManifestError("complete future groups cannot have a reason")
    elif state == "incomplete":
        if len(indices) >= FUTURE_GROUP_COUNT:
            raise EndpointManifestError(
                "incomplete future groups must have fewer than eight refs"
            )
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
            raise EndpointManifestError(
                "incomplete future groups require a non-empty reason"
            )
    else:
        raise EndpointManifestError(
            "future-group state must be 'complete' or 'incomplete'"
        )
    return state, rows, reason


def _object_ref_by_role(objects: Mapping[str, ObjectRef], role: str) -> ObjectRef:
    try:
        return objects[role]
    except KeyError as exc:
        raise EndpointManifestError(
            f"endpoint is missing object role {role!r}"
        ) from exc


def load_learner_endpoint(
    store: CaptureObjectStore, ref: EndpointRestoreRef | ManifestRef | str
) -> LoadedLearnerEndpoint:
    """Strictly verify an endpoint, every object, and every tensor-pack link."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, EndpointRestoreRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    expected_metadata_keys = {
        "schema",
        "schema_version",
        "mode",
        "fragment_versions",
        "fragments",
        "model_buffers_role",
        "scheduler",
        "scaler",
        "rng",
        "future_groups",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
        raise EndpointManifestError("endpoint manifest metadata fields are malformed")
    if metadata["schema"] != SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise EndpointManifestError("endpoint manifest uses an unsupported schema")
    mode = metadata["mode"]
    if not isinstance(mode, str) or mode not in MODES:
        raise EndpointManifestError(f"endpoint mode must be one of {sorted(MODES)}")
    if metadata["model_buffers_role"] != MODEL_BUFFERS_ROLE:
        raise EndpointManifestError("endpoint model-buffer role is noncanonical")
    if not isinstance(metadata["scheduler"], dict):
        raise EndpointManifestError("endpoint scheduler metadata must be an object")
    if metadata["scaler"] is not None and not isinstance(metadata["scaler"], dict):
        raise EndpointManifestError(
            "endpoint scaler metadata must be an object or null"
        )

    fragment_rows = _parse_fragment_rows(metadata["fragments"])
    versions_value = metadata["fragment_versions"]
    if not isinstance(versions_value, list):
        raise EndpointManifestError("endpoint fragment_versions must be an array")
    versions = tuple(
        _exact_nonnegative_int(value, f"endpoint fragment version {index}")
        for index, value in enumerate(versions_value)
    )
    if len(versions) != len(fragment_rows):
        raise EndpointManifestError(
            "endpoint fragment_versions length must equal fragment count"
        )
    cuda_roles = _parse_rng(metadata["rng"])
    future_state, future_rows, future_reason = _parse_future(metadata["future_groups"])

    object_rows = manifest["objects"]
    objects = {
        row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in object_rows
    }
    expected_roles = [row["payload_role"] for row in fragment_rows]
    expected_roles.extend(
        [
            MODEL_BUFFERS_ROLE,
            PYTHON_RNG_ROLE,
            NUMPY_RNG_ROLE,
            TORCH_CPU_RNG_ROLE,
        ]
    )
    expected_roles.extend(cuda_roles)
    expected_roles.extend(row["role"] for row in future_rows)
    actual_roles = [row["role"] for row in object_rows]
    if actual_roles != expected_roles:
        raise EndpointManifestError(
            "endpoint object roles differ from the canonical referenced role sequence"
        )

    decoded_fragments: dict[int, DecodedTensorPack] = {}
    for row in fragment_rows:
        fragment_id = row["fragment_id"]
        tensor_manifest_ref = ManifestRef(
            row["tensor_pack_manifest_id"],
            row["tensor_pack_manifest_sha256"],
            row["tensor_pack_manifest_bytes"],
            False,
        )
        decoded = load_tensor_pack(store, tensor_manifest_ref)
        endpoint_payload = _object_ref_by_role(objects, row["payload_role"])
        if decoded.payload != endpoint_payload:
            raise EndpointManifestError(
                f"fragment {fragment_id} tensor-pack payload cross-reference mismatch"
            )
        decoded_fragments[fragment_id] = decoded

    future_refs = {
        row["index"]: _object_ref_by_role(objects, row["role"]) for row in future_rows
    }
    rng = EndpointRngRefs(
        python=_object_ref_by_role(objects, PYTHON_RNG_ROLE),
        numpy=_object_ref_by_role(objects, NUMPY_RNG_ROLE),
        torch_cpu=_object_ref_by_role(objects, TORCH_CPU_RNG_ROLE),
        torch_cuda=tuple(_object_ref_by_role(objects, role) for role in cuda_roles),
    )
    manifest_sha256 = (
        ref.manifest.sha256
        if isinstance(ref, EndpointRestoreRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedLearnerEndpoint(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha256,
        mode=mode,
        fragment_versions=versions,
        fragments=decoded_fragments,
        model_buffers=_object_ref_by_role(objects, MODEL_BUFFERS_ROLE),
        scheduler=dict(metadata["scheduler"]),
        scaler=None if metadata["scaler"] is None else dict(metadata["scaler"]),
        rng=rng,
        future_groups=FutureGroupRefs(future_state, future_refs, future_reason),
    )
