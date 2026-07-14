"""Strict capture-v2 syncer-boundary manifests and reconstruction contract.

The schema binds one syncer fragment commit to verified learner endpoint
manifests and exact CAS objects.  Reconstruction remains caller-owned; this
module additionally verifies the captured memoryless f32 outer update before
accepting callback outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .capture_v2_endpoint import (
    INPUT_PROVENANCE_ROLE,
    EndpointIdentity,
    EndpointRestoreRef,
    InputProvenance,
    LoadedLearnerEndpoint,
    load_learner_endpoint,
)
from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ManifestRef,
    ObjectRef,
)


SCHEMA = "yeto.capture-v2-syncer-boundary"
SCHEMA_VERSION = 2

PRE_FRAGMENT_ROLE = "syncer/pre-fragment"
STOCK_PSEUDO_GRADIENT_ROLE = "syncer/stock-pseudo-gradient"
POST_FRAGMENT_ROLE = "syncer/post-fragment"
OUTER_STATE_ROLE = "syncer/outer-state"
BROADCAST_ROLE = "syncer/broadcast"

_MAX_COUNTER = 2**63 - 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_F64_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONFIG_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class SyncerBoundaryError(CaptureStoreError):
    """A syncer boundary or one of its cross-references is invalid."""


class ReconstructionMismatchError(SyncerBoundaryError):
    """A reconstruction callback did not reproduce captured exact bytes."""


@dataclass(frozen=True)
class BoundaryConfig:
    """Named merge/outer algorithm with canonical JSON parameters."""

    name: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class FlatF32FragmentFormat:
    """Canonical coordinate layout for every fragment-shaped boundary object."""

    fragment_numel: int
    tensor_layout_sha256: str


@dataclass(frozen=True)
class SyncerBoundaryIdentity:
    """Causal identity and version transition of one fragment commit."""

    capture_session_uuid: str
    commit_id: str
    commit_seq: int
    fragment_id: int
    pre_fragment_version: int
    post_fragment_version: int


@dataclass(frozen=True)
class ResponderEndpointRef:
    """One learner endpoint and the exact syncer-received response identity."""

    endpoint: EndpointRestoreRef
    weight_f64_bits: str
    payload: ObjectRef


@dataclass(frozen=True)
class SyncerBoundaryRef:
    """Content identity of one published syncer-boundary manifest."""

    manifest: ManifestRef


@dataclass
class LoadedResponder:
    """A responder row cross-checked against its loaded learner endpoint."""

    responder_index: int
    endpoint: LoadedLearnerEndpoint
    endpoint_ref: EndpointRestoreRef
    weight_f64_bits: str
    payload: ObjectRef


@dataclass(frozen=True)
class LoadedSyncerBoundary:
    """Strictly validated syncer boundary and all learner endpoint links."""

    manifest_id: str
    manifest_sha256: str
    identity: SyncerBoundaryIdentity
    responders: tuple[LoadedResponder, ...]
    merge_config: BoundaryConfig
    outer_config: BoundaryConfig
    fragment_format: FlatF32FragmentFormat
    pre_fragment: ObjectRef
    stock_pseudo_gradient: ObjectRef
    post_fragment: ObjectRef
    outer_state: ObjectRef
    broadcast: ObjectRef


@dataclass(frozen=True)
class ReconstructionRequest:
    """Verified opaque inputs exposed to reconstruction code."""

    identity: SyncerBoundaryIdentity
    responders: tuple[LoadedResponder, ...]
    merge_config: BoundaryConfig
    outer_config: BoundaryConfig
    fragment_format: FlatF32FragmentFormat
    pre_fragment: bytes
    stock_pseudo_gradient: bytes
    outer_state: bytes


@dataclass(frozen=True)
class ReconstructionOutput:
    """Exact bytes produced by a reconstruction callback."""

    post_fragment: bytes
    broadcast: bytes


class ReconstructionCallback(Protocol):
    """Opaque reconstruction function; no algorithm is prescribed here."""

    def __call__(self, request: ReconstructionRequest) -> ReconstructionOutput: ...


def _exact_nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise SyncerBoundaryError(f"{context} must be a non-negative integer")
    if value > _MAX_COUNTER:
        raise SyncerBoundaryError(f"{context} exceeds {_MAX_COUNTER}")
    return value


def _canonical_uuid(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise SyncerBoundaryError(f"{context} must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise SyncerBoundaryError(f"{context} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise SyncerBoundaryError(f"{context} must be a canonical UUID string")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SyncerBoundaryError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _f64_bits(value: Any, context: str) -> str:
    if not isinstance(value, str) or _F64_BITS_RE.fullmatch(value) is None:
        raise SyncerBoundaryError(f"{context} must be exactly 16 lowercase hex digits")
    return value


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SyncerBoundaryError(f"{context} is malformed")
    return value


def _fragment_format_value(value: FlatF32FragmentFormat) -> dict[str, Any]:
    if not isinstance(value, FlatF32FragmentFormat):
        raise TypeError("fragment_format must be FlatF32FragmentFormat")
    numel = _exact_nonnegative_int(value.fragment_numel, "fragment_numel")
    if numel == 0:
        raise SyncerBoundaryError("fragment_numel must be strictly positive")
    return {
        "encoding": "ieee754-f32le-flat",
        "fragment_numel": numel,
        "tensor_layout_sha256": _sha256(
            value.tensor_layout_sha256, "tensor_layout_sha256"
        ),
    }


def _parse_fragment_format(value: Any) -> FlatF32FragmentFormat:
    if not isinstance(value, dict) or set(value) != {
        "encoding",
        "fragment_numel",
        "tensor_layout_sha256",
    }:
        raise SyncerBoundaryError("fragment_format fields are malformed")
    if value["encoding"] != "ieee754-f32le-flat":
        raise SyncerBoundaryError("fragment_format encoding is noncanonical")
    numel = _exact_nonnegative_int(value["fragment_numel"], "fragment_numel")
    if numel == 0:
        raise SyncerBoundaryError("fragment_numel must be strictly positive")
    return FlatF32FragmentFormat(
        fragment_numel=numel,
        tensor_layout_sha256=_sha256(
            value["tensor_layout_sha256"], "tensor_layout_sha256"
        ),
    )


def _validate_fragment_object_lengths(
    fragment_format: FlatF32FragmentFormat,
    *refs: tuple[ObjectRef, str],
) -> None:
    expected_bytes = fragment_format.fragment_numel * 4
    for ref, context in refs:
        if not isinstance(ref, ObjectRef):
            raise TypeError(f"{context} must be an ObjectRef")
        if ref.bytes != expected_bytes:
            raise SyncerBoundaryError(
                f"{context} bytes must equal fragment_numel * 4 ({expected_bytes})"
            )


def _responder_payload_role(responder_index: int) -> str:
    return f"responders/{responder_index}/payload"


def _snapshot_json_object(value: Mapping[str, Any], context: str) -> dict[str, Any]:
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
        raise SyncerBoundaryError(
            f"{context} is not canonical JSON data: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise SyncerBoundaryError(f"{context} must encode as a JSON object")
    return result


def _config_value(value: BoundaryConfig, context: str) -> dict[str, Any]:
    if not isinstance(value, BoundaryConfig):
        raise TypeError(f"{context} must be BoundaryConfig")
    if not isinstance(value.name, str) or _CONFIG_NAME_RE.fullmatch(value.name) is None:
        raise SyncerBoundaryError(f"{context} name is malformed")
    return {
        "name": value.name,
        "parameters": _snapshot_json_object(value.parameters, f"{context} parameters"),
    }


def _parse_config(value: Any, context: str) -> BoundaryConfig:
    if not isinstance(value, dict) or set(value) != {"name", "parameters"}:
        raise SyncerBoundaryError(f"{context} fields are malformed")
    name = value["name"]
    if not isinstance(name, str) or _CONFIG_NAME_RE.fullmatch(name) is None:
        raise SyncerBoundaryError(f"{context} name is malformed")
    parameters = value["parameters"]
    if not isinstance(parameters, dict):
        raise SyncerBoundaryError(f"{context} parameters must be an object")
    return BoundaryConfig(name, dict(parameters))


def _identity_value(identity: SyncerBoundaryIdentity) -> dict[str, Any]:
    if not isinstance(identity, SyncerBoundaryIdentity):
        raise TypeError("identity must be SyncerBoundaryIdentity")
    pre_version = _exact_nonnegative_int(
        identity.pre_fragment_version, "pre_fragment_version"
    )
    post_version = _exact_nonnegative_int(
        identity.post_fragment_version, "post_fragment_version"
    )
    if post_version <= pre_version:
        raise SyncerBoundaryError(
            "post_fragment_version must be strictly newer than pre_fragment_version"
        )
    return {
        "capture_session_uuid": _canonical_uuid(
            identity.capture_session_uuid, "capture_session_uuid"
        ),
        "commit_id": _identifier(identity.commit_id, "commit_id"),
        "commit_seq": _exact_nonnegative_int(identity.commit_seq, "commit_seq"),
        "fragment_id": _exact_nonnegative_int(identity.fragment_id, "fragment_id"),
        "pre_fragment_version": pre_version,
        "post_fragment_version": post_version,
    }


def _parse_identity(value: Any) -> SyncerBoundaryIdentity:
    expected_keys = {
        "capture_session_uuid",
        "commit_id",
        "commit_seq",
        "fragment_id",
        "pre_fragment_version",
        "post_fragment_version",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SyncerBoundaryError("syncer-boundary identity fields are malformed")
    identity = SyncerBoundaryIdentity(
        capture_session_uuid=_canonical_uuid(
            value["capture_session_uuid"], "boundary capture_session_uuid"
        ),
        commit_id=_identifier(value["commit_id"], "boundary commit_id"),
        commit_seq=_exact_nonnegative_int(value["commit_seq"], "boundary commit_seq"),
        fragment_id=_exact_nonnegative_int(
            value["fragment_id"], "boundary fragment_id"
        ),
        pre_fragment_version=_exact_nonnegative_int(
            value["pre_fragment_version"], "boundary pre_fragment_version"
        ),
        post_fragment_version=_exact_nonnegative_int(
            value["post_fragment_version"], "boundary post_fragment_version"
        ),
    )
    if identity.post_fragment_version <= identity.pre_fragment_version:
        raise SyncerBoundaryError(
            "boundary post_fragment_version must be strictly newer than pre_fragment_version"
        )
    return identity


def _endpoint_identity_value(identity: EndpointIdentity) -> dict[str, Any]:
    return {
        "capture_session_uuid": identity.capture_session_uuid,
        "learner_id": identity.learner_id,
        "rank": identity.rank,
        "local_step": identity.local_step,
        "active_fragment_id": identity.active_fragment_id,
        "window_uuid": identity.window_uuid,
    }


def _provenance_value(provenance: InputProvenance) -> dict[str, Any]:
    return {
        "role": INPUT_PROVENANCE_ROLE,
        "sha256": provenance.object.sha256,
        "bytes": provenance.object.bytes,
        "source_commit": provenance.source_commit,
        "image_id": provenance.image_id,
        "model_sha256": provenance.model_sha256,
        "data_sha256": provenance.data_sha256,
        "config_sha256": provenance.config_sha256,
    }


def _same_canonical_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _responder_rows(
    store: CaptureObjectStore,
    responders: Sequence[ResponderEndpointRef],
    identity: SyncerBoundaryIdentity,
) -> tuple[list[dict[str, Any]], dict[int, ObjectRef]]:
    if isinstance(responders, (str, bytes)) or not isinstance(responders, Sequence):
        raise TypeError("responders must be a sequence")
    loaded: list[tuple[LoadedLearnerEndpoint, ResponderEndpointRef]] = []
    for responder in responders:
        if not isinstance(responder, ResponderEndpointRef):
            raise TypeError("responders must contain ResponderEndpointRef values")
        endpoint = load_learner_endpoint(store, responder.endpoint)
        if endpoint.identity.capture_session_uuid != identity.capture_session_uuid:
            raise SyncerBoundaryError(
                "responder endpoint capture_session_uuid differs from boundary"
            )
        if endpoint.identity.active_fragment_id != identity.fragment_id:
            raise SyncerBoundaryError(
                "responder endpoint active_fragment_id differs from boundary fragment"
            )
        if (
            endpoint.fragment_versions[identity.fragment_id]
            != identity.pre_fragment_version
        ):
            raise SyncerBoundaryError(
                "responder endpoint fragment version differs from boundary pre-version"
            )
        _f64_bits(responder.weight_f64_bits, "responder weight_f64_bits")
        if not isinstance(responder.payload, ObjectRef):
            raise TypeError("responder payload must be an ObjectRef")
        store.verify_object(responder.payload)
        loaded.append((endpoint, responder))
    if not loaded:
        raise SyncerBoundaryError("syncer boundary requires at least one responder")
    loaded.sort(key=lambda item: (item[0].identity.learner_id, item[0].identity.rank))
    learner_ids = [item[0].identity.learner_id for item in loaded]
    if len(set(learner_ids)) != len(learner_ids):
        raise SyncerBoundaryError(
            "syncer responders must have unique learner_id values"
        )

    rows: list[dict[str, Any]] = []
    payloads: dict[int, ObjectRef] = {}
    for responder_index, (endpoint, responder) in enumerate(loaded):
        payload_role = _responder_payload_role(responder_index)
        payloads[responder_index] = responder.payload
        rows.append(
            {
                "responder_index": responder_index,
                "endpoint_manifest_id": responder.endpoint.manifest.manifest_id,
                "endpoint_manifest_sha256": responder.endpoint.manifest.sha256,
                "endpoint_manifest_bytes": responder.endpoint.manifest.bytes,
                "endpoint_identity": _endpoint_identity_value(endpoint.identity),
                "input_provenance": _provenance_value(endpoint.input_provenance),
                "weight_f64_bits": responder.weight_f64_bits,
                "payload_role": payload_role,
                "payload_sha256": responder.payload.sha256,
                "payload_bytes": responder.payload.bytes,
            }
        )
    return rows, payloads


def publish_syncer_boundary(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    identity: SyncerBoundaryIdentity,
    responders: Sequence[ResponderEndpointRef],
    fragment_format: FlatF32FragmentFormat,
    pre_fragment: ObjectRef,
    stock_pseudo_gradient: ObjectRef,
    post_fragment: ObjectRef,
    outer_state: ObjectRef,
    broadcast: ObjectRef,
    merge_config: BoundaryConfig,
    outer_config: BoundaryConfig,
) -> SyncerBoundaryRef:
    """Cross-check every input and atomically publish one syncer boundary."""

    identity_data = _identity_value(identity)
    responder_rows, responder_payloads = _responder_rows(store, responders, identity)
    fragment_format_data = _fragment_format_value(fragment_format)
    _validate_fragment_object_lengths(
        fragment_format,
        (pre_fragment, "pre_fragment"),
        (stock_pseudo_gradient, "stock_pseudo_gradient"),
        (post_fragment, "post_fragment"),
        (broadcast, "broadcast"),
    )
    if not isinstance(outer_state, ObjectRef):
        raise TypeError("outer_state must be an ObjectRef")
    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "identity": identity_data,
        "responders": responder_rows,
        "merge_config": _config_value(merge_config, "merge_config"),
        "outer_config": _config_value(outer_config, "outer_config"),
        "fragment_format": fragment_format_data,
        "broadcast": {
            "role": BROADCAST_ROLE,
            "sha256": broadcast.sha256,
            "bytes": broadcast.bytes,
        },
    }
    manifest = store.publish_manifest(
        manifest_id,
        [
            ManifestEntry(PRE_FRAGMENT_ROLE, pre_fragment),
            ManifestEntry(STOCK_PSEUDO_GRADIENT_ROLE, stock_pseudo_gradient),
            ManifestEntry(POST_FRAGMENT_ROLE, post_fragment),
            ManifestEntry(OUTER_STATE_ROLE, outer_state),
            ManifestEntry(BROADCAST_ROLE, broadcast),
            *(
                ManifestEntry(_responder_payload_role(index), payload)
                for index, payload in responder_payloads.items()
            ),
        ],
        metadata=metadata,
    )
    return SyncerBoundaryRef(manifest)


def _parse_responder_rows(
    store: CaptureObjectStore,
    value: Any,
    identity: SyncerBoundaryIdentity,
    objects: Mapping[str, ObjectRef],
) -> tuple[LoadedResponder, ...]:
    if not isinstance(value, list) or not value:
        raise SyncerBoundaryError("boundary responders must be a non-empty array")
    expected_keys = {
        "responder_index",
        "endpoint_manifest_id",
        "endpoint_manifest_sha256",
        "endpoint_manifest_bytes",
        "endpoint_identity",
        "input_provenance",
        "weight_f64_bits",
        "payload_role",
        "payload_sha256",
        "payload_bytes",
    }
    result: list[LoadedResponder] = []
    previous_key: tuple[int, int] | None = None
    learner_ids: set[int] = set()
    for index, row in enumerate(value):
        context = f"boundary responder row {index}"
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise SyncerBoundaryError(f"{context} fields are malformed")
        if type(row["responder_index"]) is not int or row["responder_index"] != index:
            raise SyncerBoundaryError("boundary responder indices must be contiguous")
        try:
            endpoint_manifest = ManifestRef(
                row["endpoint_manifest_id"],
                row["endpoint_manifest_sha256"],
                row["endpoint_manifest_bytes"],
                False,
            )
        except CaptureStoreError as exc:
            raise SyncerBoundaryError(
                f"{context} endpoint manifest reference is malformed: {exc}"
            ) from exc
        endpoint_ref = EndpointRestoreRef(endpoint_manifest)
        endpoint = load_learner_endpoint(store, endpoint_ref)
        expected_identity = _endpoint_identity_value(endpoint.identity)
        if not isinstance(row["endpoint_identity"], dict) or not _same_canonical_json(
            row["endpoint_identity"], expected_identity
        ):
            raise SyncerBoundaryError(
                f"{context} endpoint identity cross-reference mismatch"
            )
        expected_provenance = _provenance_value(endpoint.input_provenance)
        if not isinstance(row["input_provenance"], dict) or not _same_canonical_json(
            row["input_provenance"], expected_provenance
        ):
            raise SyncerBoundaryError(
                f"{context} input provenance cross-reference mismatch"
            )
        if endpoint.identity.capture_session_uuid != identity.capture_session_uuid:
            raise SyncerBoundaryError(
                f"{context} capture_session_uuid differs from boundary"
            )
        if endpoint.identity.active_fragment_id != identity.fragment_id:
            raise SyncerBoundaryError(
                f"{context} active_fragment_id differs from boundary"
            )
        if (
            endpoint.fragment_versions[identity.fragment_id]
            != identity.pre_fragment_version
        ):
            raise SyncerBoundaryError(
                f"{context} fragment version differs from boundary pre-version"
            )
        key = (endpoint.identity.learner_id, endpoint.identity.rank)
        if previous_key is not None and key <= previous_key:
            raise SyncerBoundaryError(
                "boundary responders are not strictly learner/rank ordered"
            )
        previous_key = key
        if endpoint.identity.learner_id in learner_ids:
            raise SyncerBoundaryError("boundary responders repeat learner_id")
        learner_ids.add(endpoint.identity.learner_id)
        payload_role = _responder_payload_role(index)
        if row["payload_role"] != payload_role:
            raise SyncerBoundaryError(f"{context} payload role is noncanonical")
        try:
            payload = ObjectRef(row["payload_sha256"], row["payload_bytes"])
        except CaptureStoreError as exc:
            raise SyncerBoundaryError(
                f"{context} payload reference is malformed: {exc}"
            ) from exc
        if payload != objects[payload_role]:
            raise SyncerBoundaryError(
                f"{context} payload metadata/object cross-reference mismatch"
            )
        result.append(
            LoadedResponder(
                responder_index=index,
                endpoint=endpoint,
                endpoint_ref=endpoint_ref,
                weight_f64_bits=_f64_bits(
                    row["weight_f64_bits"], f"{context} weight_f64_bits"
                ),
                payload=payload,
            )
        )
    return tuple(result)


def _object_map(
    manifest: Mapping[str, Any], responder_count: int
) -> dict[str, ObjectRef]:
    rows = manifest["objects"]
    expected_roles = [
        PRE_FRAGMENT_ROLE,
        STOCK_PSEUDO_GRADIENT_ROLE,
        POST_FRAGMENT_ROLE,
        OUTER_STATE_ROLE,
        BROADCAST_ROLE,
        *(_responder_payload_role(index) for index in range(responder_count)),
    ]
    if [row["role"] for row in rows] != expected_roles:
        raise SyncerBoundaryError(
            "syncer-boundary object roles differ from the canonical role order"
        )
    return {row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in rows}


def load_syncer_boundary(
    store: CaptureObjectStore, ref: SyncerBoundaryRef | ManifestRef | str
) -> LoadedSyncerBoundary:
    """Strictly verify one syncer boundary and every endpoint cross-reference."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, SyncerBoundaryRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    expected_metadata_keys = {
        "schema",
        "schema_version",
        "identity",
        "responders",
        "merge_config",
        "outer_config",
        "fragment_format",
        "broadcast",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
        raise SyncerBoundaryError("syncer-boundary metadata fields are malformed")
    if metadata["schema"] != SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise SyncerBoundaryError("syncer-boundary uses an unsupported schema")
    identity = _parse_identity(metadata["identity"])
    responder_value = metadata["responders"]
    if not isinstance(responder_value, list) or not responder_value:
        raise SyncerBoundaryError("boundary responders must be a non-empty array")
    objects = _object_map(manifest, len(responder_value))
    responders = _parse_responder_rows(store, responder_value, identity, objects)
    merge_config = _parse_config(metadata["merge_config"], "merge_config")
    outer_config = _parse_config(metadata["outer_config"], "outer_config")
    fragment_format = _parse_fragment_format(metadata["fragment_format"])
    _validate_fragment_object_lengths(
        fragment_format,
        (objects[PRE_FRAGMENT_ROLE], "pre_fragment"),
        (objects[STOCK_PSEUDO_GRADIENT_ROLE], "stock_pseudo_gradient"),
        (objects[POST_FRAGMENT_ROLE], "post_fragment"),
        (objects[BROADCAST_ROLE], "broadcast"),
    )

    broadcast_metadata = metadata["broadcast"]
    if not isinstance(broadcast_metadata, dict) or set(broadcast_metadata) != {
        "role",
        "sha256",
        "bytes",
    }:
        raise SyncerBoundaryError("syncer-boundary broadcast fields are malformed")
    if broadcast_metadata["role"] != BROADCAST_ROLE:
        raise SyncerBoundaryError("syncer-boundary broadcast role is noncanonical")
    try:
        explicit_broadcast = ObjectRef(
            broadcast_metadata["sha256"], broadcast_metadata["bytes"]
        )
    except CaptureStoreError as exc:
        raise SyncerBoundaryError(
            f"syncer-boundary broadcast reference is malformed: {exc}"
        ) from exc
    if explicit_broadcast != objects[BROADCAST_ROLE]:
        raise SyncerBoundaryError(
            "syncer-boundary broadcast SHA/bytes cross-reference mismatch"
        )

    manifest_sha256 = (
        ref.manifest.sha256
        if isinstance(ref, SyncerBoundaryRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedSyncerBoundary(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha256,
        identity=identity,
        responders=responders,
        merge_config=merge_config,
        outer_config=outer_config,
        fragment_format=fragment_format,
        pre_fragment=objects[PRE_FRAGMENT_ROLE],
        stock_pseudo_gradient=objects[STOCK_PSEUDO_GRADIENT_ROLE],
        post_fragment=objects[POST_FRAGMENT_ROLE],
        outer_state=objects[OUTER_STATE_ROLE],
        broadcast=objects[BROADCAST_ROLE],
    )


def _read_exact_object(
    store: CaptureObjectStore, ref: ObjectRef, context: str
) -> bytes:
    path = store.verify_object(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SyncerBoundaryError(
            f"cannot read {context} object {path}: {exc}"
        ) from exc
    if len(raw) != ref.bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise SyncerBoundaryError(f"{context} object changed after verification")
    return raw


def memoryless_outer_update_f32le(
    pre_fragment: bytes, stock_pseudo_gradient: bytes, lr_f64_bits: str
) -> bytes:
    """Reproduce the syncer's coordinate-order f32 memoryless outer update."""

    if type(pre_fragment) is not bytes or type(stock_pseudo_gradient) is not bytes:
        raise TypeError("memoryless outer update inputs must have exact type bytes")
    if len(pre_fragment) != len(stock_pseudo_gradient) or len(pre_fragment) % 4:
        raise SyncerBoundaryError(
            "memoryless outer update requires equal whole-f32 input lengths"
        )
    bits = _f64_bits(lr_f64_bits, "outer lr_f64_bits")
    lr64 = struct.unpack(">d", bytes.fromhex(bits))[0]
    if not math.isfinite(lr64) or lr64 <= 0.0:
        raise SyncerBoundaryError("outer lr_f64_bits must decode to finite positive f64")
    try:
        lr32 = struct.unpack("<f", struct.pack("<f", lr64))[0]
        pre = struct.unpack(f"<{len(pre_fragment) // 4}f", pre_fragment)
        stock = struct.unpack(
            f"<{len(stock_pseudo_gradient) // 4}f", stock_pseudo_gradient
        )
        if not all(math.isfinite(value) for value in (*pre, *stock)):
            raise SyncerBoundaryError(
                "memoryless outer update inputs must contain finite f32 values"
            )
        result = []
        for parameter, direction in zip(pre, stock, strict=True):
            product = struct.unpack("<f", struct.pack("<f", lr32 * direction))[0]
            updated = struct.unpack("<f", struct.pack("<f", parameter - product))[0]
            result.append(updated)
        return struct.pack(f"<{len(result)}f", *result)
    except (OverflowError, struct.error) as exc:
        raise SyncerBoundaryError(
            "memoryless outer update contains invalid or non-finite f32 arithmetic"
        ) from exc


def _outer_lr_bits(config: BoundaryConfig) -> str:
    if config.name != "nesterov" or not isinstance(config.parameters, dict):
        raise ReconstructionMismatchError(
            "factual memoryless update requires nesterov outer_config"
        )
    if set(config.parameters) != {"lr_f64_bits", "momentum_f64_bits"}:
        raise ReconstructionMismatchError(
            "factual memoryless update requires exact nesterov config fields"
        )
    if config.parameters["momentum_f64_bits"] != "0000000000000000":
        raise ReconstructionMismatchError(
            "factual memoryless update requires exact positive-zero momentum"
        )
    return _f64_bits(config.parameters["lr_f64_bits"], "outer lr_f64_bits")


def verify_reconstruction(
    store: CaptureObjectStore,
    boundary: LoadedSyncerBoundary | SyncerBoundaryRef | ManifestRef | str,
    callback: ReconstructionCallback,
) -> ReconstructionOutput:
    """Require a callback to reproduce exact post-fragment and broadcast bytes."""

    # Loaded structures are convenient inspection views, not authority.  Always
    # reload their content-addressed manifest immediately before constructing a
    # callback request so caller mutation after an earlier load cannot alter
    # verified reconstruction inputs.
    loaded = load_syncer_boundary(
        store,
        boundary.manifest_sha256
        if isinstance(boundary, LoadedSyncerBoundary)
        else boundary,
    )
    request = ReconstructionRequest(
        identity=loaded.identity,
        responders=loaded.responders,
        merge_config=loaded.merge_config,
        outer_config=loaded.outer_config,
        fragment_format=loaded.fragment_format,
        pre_fragment=_read_exact_object(store, loaded.pre_fragment, "pre-fragment"),
        stock_pseudo_gradient=_read_exact_object(
            store, loaded.stock_pseudo_gradient, "stock-pseudo-gradient"
        ),
        outer_state=_read_exact_object(store, loaded.outer_state, "outer-state"),
    )
    expected_post = _read_exact_object(store, loaded.post_fragment, "post-fragment")
    expected_broadcast = _read_exact_object(store, loaded.broadcast, "broadcast")
    factual_post = memoryless_outer_update_f32le(
        request.pre_fragment,
        request.stock_pseudo_gradient,
        _outer_lr_bits(loaded.outer_config),
    )
    if factual_post != expected_post:
        raise ReconstructionMismatchError(
            "captured post-fragment does not reproduce the factual memoryless outer update"
        )
    if expected_broadcast != expected_post:
        raise ReconstructionMismatchError(
            "captured broadcast differs from the factual memoryless post-fragment"
        )
    result = callback(request)
    if not isinstance(result, ReconstructionOutput):
        raise ReconstructionMismatchError(
            "reconstruction callback must return ReconstructionOutput"
        )
    if not isinstance(result.post_fragment, bytes) or not isinstance(
        result.broadcast, bytes
    ):
        raise ReconstructionMismatchError("reconstruction outputs must be exact bytes")

    if result.post_fragment != expected_post:
        raise ReconstructionMismatchError(
            "reconstruction post-fragment bytes do not match the captured object"
        )
    if result.broadcast != expected_broadcast:
        raise ReconstructionMismatchError(
            "reconstruction broadcast bytes do not match the captured object"
        )
    return result
