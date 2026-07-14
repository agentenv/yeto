from __future__ import annotations

import copy
import hashlib
import struct
from collections import OrderedDict

import pytest
import torch

from yeto.capture_v2_endpoint import (
    EndpointIdentity,
    FutureGroupRefs,
    InputProvenance,
    publish_learner_endpoint,
)
from yeto.capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ObjectRef,
)
from yeto.capture_v2_syncer import (
    BoundaryConfig,
    FlatF32FragmentFormat,
    ReconstructionMismatchError,
    ReconstructionOutput,
    ResponderEndpointRef,
    SyncerBoundaryError,
    SyncerBoundaryIdentity,
    load_syncer_boundary,
    memoryless_outer_update_f32le,
    publish_syncer_boundary,
    verify_reconstruction,
)
from yeto.capture_v2_tensor_pack import publish_tensor_pack


SESSION = "12345678-1234-5678-9234-567812345678"
PRE_VERSION = 17
FRAGMENT_ID = 1


def _object(store: CaptureObjectStore, raw: bytes) -> ObjectRef:
    return store.put_bytes(raw).ref


def _window_uuid(learner_id: int) -> str:
    return f"00000000-0000-4000-8000-{learner_id + 1:012d}"


def _learner_endpoint(
    store: CaptureObjectStore,
    learner_id: int,
    *,
    session: str = SESSION,
    active_fragment_id: int = FRAGMENT_ID,
    fragment_versions=(5, PRE_VERSION),
):
    packs = {}
    for fragment_id in range(2):
        packs[fragment_id] = publish_tensor_pack(
            store,
            f"learner-{learner_id}-fragment-{fragment_id}",
            fragment_id=fragment_id,
            trainable={
                f"learner.{learner_id}.layer.{fragment_id}.weight": torch.tensor(
                    [learner_id + fragment_id + 0.25], dtype=torch.float32
                )
            },
            optimizer={
                f"learner.{learner_id}.layer.{fragment_id}.weight/exp_avg": (
                    torch.tensor([learner_id + fragment_id + 1.0])
                )
            },
            clocks={"optimizer_steps": 31 + learner_id},
            metadata={"fragment_id": fragment_id, "learner_id": learner_id},
        )
    provenance_object = _object(store, f"provenance learner {learner_id}".encode())
    return publish_learner_endpoint(
        store,
        f"endpoint-learner-{learner_id}",
        identity=EndpointIdentity(
            capture_session_uuid=session,
            learner_id=learner_id,
            rank=0,
            local_step=100 + learner_id,
            active_fragment_id=active_fragment_id,
            window_uuid=_window_uuid(learner_id),
        ),
        input_provenance=InputProvenance(
            object=provenance_object,
            source_commit="a" * 40,
            image_id="gcp:yeto-a100-v1",
            model_sha256="b" * 64,
            data_sha256=hashlib.sha256(f"shard-{learner_id}".encode()).hexdigest(),
            config_sha256="d" * 64,
        ),
        fragment_packs=packs,
        fragment_versions=list(fragment_versions),
        mode="train",
        model_buffers=_object(store, f"buffers {learner_id}".encode()),
        scheduler={"last_epoch": 41, "base_lrs": [0.28]},
        scaler=None,
        python_rng=_object(store, f"python rng {learner_id}".encode()),
        numpy_rng=_object(store, f"numpy rng {learner_id}".encode()),
        torch_cpu_rng=_object(store, f"torch cpu rng {learner_id}".encode()),
        torch_cuda_rng={0: _object(store, f"torch cuda rng {learner_id}".encode())},
        future_groups=FutureGroupRefs(
            "incomplete", {}, "future groups not retained for boundary fixture"
        ),
    )


def _responder(store: CaptureObjectStore, endpoint, learner_id: int):
    return ResponderEndpointRef(
        endpoint=endpoint,
        weight_f64_bits=struct.pack(">d", 128.0 + learner_id).hex(),
        payload=_object(store, f"push payload {learner_id}".encode()),
    )


def _boundary_arguments(store: CaptureObjectStore):
    endpoints = {
        learner_id: _learner_endpoint(store, learner_id) for learner_id in range(3)
    }
    pre = struct.pack("<4f", 1.0, -2.0, 3.0, -4.0)
    stock = struct.pack("<4f", 0.25, -0.5, 1.25, -1.5)
    outer = b"outer-state exact bytes"
    post = memoryless_outer_update_f32le(pre, stock, struct.pack(">d", 0.28).hex())
    broadcast = post
    return {
        "identity": SyncerBoundaryIdentity(
            capture_session_uuid=SESSION,
            commit_id="step-00000042-fragment-0001",
            commit_seq=9,
            fragment_id=FRAGMENT_ID,
            pre_fragment_version=PRE_VERSION,
            post_fragment_version=42,
        ),
        "responders": [
            _responder(store, endpoints[2], 2),
            _responder(store, endpoints[0], 0),
            _responder(store, endpoints[1], 1),
        ],
        "fragment_format": FlatF32FragmentFormat(4, "e" * 64),
        "pre_fragment": _object(store, pre),
        "stock_pseudo_gradient": _object(store, stock),
        "post_fragment": _object(store, post),
        "outer_state": _object(store, outer),
        "broadcast": _object(store, broadcast),
        "merge_config": BoundaryConfig(
            "rda",
            OrderedDict([("epsilon_f64_bits", "3cb0000000000000"), ("weighted", True)]),
        ),
        "outer_config": BoundaryConfig(
            "nesterov",
            OrderedDict(
                [
                    ("lr_f64_bits", struct.pack(">d", 0.28).hex()),
                    ("momentum_f64_bits", "0000000000000000"),
                ]
            ),
        ),
        "_raw": {
            "pre": pre,
            "stock": stock,
            "outer": outer,
            "post": post,
            "broadcast": broadcast,
        },
        "_endpoints": endpoints,
    }


def _publish_boundary(store: CaptureObjectStore, manifest_id="syncer-boundary-9"):
    arguments = _boundary_arguments(store)
    internal = {key: arguments.pop(key) for key in ("_raw", "_endpoints")}
    boundary = publish_syncer_boundary(store, manifest_id, **arguments)
    return boundary, arguments, internal


def _entries(manifest: dict):
    return [
        ManifestEntry(row["role"], ObjectRef(row["sha256"], row["bytes"]))
        for row in manifest["objects"]
    ]


def _reseal(store, boundary, metadata, manifest_id, entries=None):
    manifest = store.load_manifest(boundary.manifest)
    return store.publish_manifest(
        manifest_id,
        _entries(manifest) if entries is None else entries,
        metadata=metadata,
    )


def test_boundary_is_canonical_deterministic_and_cross_wires_endpoints(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _boundary_arguments(store)
    raw = arguments.pop("_raw")
    arguments.pop("_endpoints")
    first = publish_syncer_boundary(store, "syncer-boundary-9", **arguments)

    reversed_arguments = dict(arguments)
    reversed_arguments["responders"] = list(reversed(arguments["responders"]))
    reversed_arguments["merge_config"] = BoundaryConfig(
        "rda",
        OrderedDict(reversed(list(arguments["merge_config"].parameters.items()))),
    )
    reversed_arguments["outer_config"] = BoundaryConfig(
        "nesterov",
        OrderedDict(reversed(list(arguments["outer_config"].parameters.items()))),
    )
    second = publish_syncer_boundary(store, "syncer-boundary-9", **reversed_arguments)

    assert second.manifest.sha256 == first.manifest.sha256
    assert second.manifest.inserted is False
    loaded = load_syncer_boundary(store, first)
    assert loaded.manifest_id == "syncer-boundary-9"
    assert loaded.identity == arguments["identity"]
    assert [row.endpoint.identity.learner_id for row in loaded.responders] == [0, 1, 2]
    assert [row.responder_index for row in loaded.responders] == [0, 1, 2]
    assert [
        store.object_path(row.payload.sha256).read_bytes() for row in loaded.responders
    ] == [b"push payload 0", b"push payload 1", b"push payload 2"]
    assert [row.endpoint.identity.local_step for row in loaded.responders] == [
        100,
        101,
        102,
    ]
    assert all(
        row.endpoint.identity.capture_session_uuid == SESSION
        for row in loaded.responders
    )
    assert loaded.merge_config == BoundaryConfig(
        "rda", {"epsilon_f64_bits": "3cb0000000000000", "weighted": True}
    )
    assert loaded.outer_config.name == "nesterov"
    assert loaded.broadcast == arguments["broadcast"]

    manifest = store.load_manifest(first.manifest)
    assert [row["role"] for row in manifest["objects"]] == [
        "syncer/pre-fragment",
        "syncer/stock-pseudo-gradient",
        "syncer/post-fragment",
        "syncer/outer-state",
        "syncer/broadcast",
        "responders/0/payload",
        "responders/1/payload",
        "responders/2/payload",
    ]
    assert manifest["metadata"]["broadcast"] == {
        "role": "syncer/broadcast",
        "sha256": hashlib.sha256(raw["broadcast"]).hexdigest(),
        "bytes": len(raw["broadcast"]),
    }
    for responder in manifest["metadata"]["responders"]:
        assert (
            responder["endpoint_identity"]["learner_id"] == responder["responder_index"]
        )
        assert responder["input_provenance"]["source_commit"] == "a" * 40
        assert responder["input_provenance"]["image_id"] == "gcp:yeto-a100-v1"
        payload_row = next(
            row
            for row in manifest["objects"]
            if row["role"] == responder["payload_role"]
        )
        assert responder["payload_sha256"] == payload_row["sha256"]
        assert responder["payload_bytes"] == payload_row["bytes"]
    store.audit()


def test_opaque_reconstruction_callback_must_reproduce_both_exact_outputs(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    boundary, _arguments, internal = _publish_boundary(store)
    seen = {}

    def reconstruct(request):
        seen["request"] = request
        post = memoryless_outer_update_f32le(
            request.pre_fragment,
            request.stock_pseudo_gradient,
            request.outer_config.parameters["lr_f64_bits"],
        )
        return ReconstructionOutput(post, post)

    result = verify_reconstruction(store, boundary, reconstruct)
    assert result == ReconstructionOutput(
        internal["_raw"]["post"], internal["_raw"]["broadcast"]
    )
    request = seen["request"]
    assert request.identity.fragment_id == FRAGMENT_ID
    assert [row.endpoint.identity.learner_id for row in request.responders] == [0, 1, 2]
    assert request.pre_fragment == internal["_raw"]["pre"]
    assert request.stock_pseudo_gradient == internal["_raw"]["stock"]
    assert request.outer_state == internal["_raw"]["outer"]


def test_reconstruction_reloads_authority_after_config_and_responder_mutation(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    boundary, _arguments, internal = _publish_boundary(store)
    loaded = load_syncer_boundary(store, boundary)
    authoritative_weight = loaded.responders[0].weight_f64_bits
    authoritative_payload = loaded.responders[0].payload

    loaded.merge_config.parameters["weighted"] = False
    loaded.outer_config.parameters["lr_f64_bits"] = "3ff0000000000000"
    loaded.responders[0].weight_f64_bits = "0000000000000000"
    loaded.responders[0].payload = _object(store, b"mutated responder payload view")
    seen = {}

    def reconstruct(request):
        seen["request"] = request
        post = memoryless_outer_update_f32le(
            request.pre_fragment,
            request.stock_pseudo_gradient,
            request.outer_config.parameters["lr_f64_bits"],
        )
        return ReconstructionOutput(post, post)

    result = verify_reconstruction(store, loaded, reconstruct)

    assert result.post_fragment == internal["_raw"]["post"]
    request = seen["request"]
    assert request.merge_config.parameters["weighted"] is True
    assert (
        request.outer_config.parameters["lr_f64_bits"] == struct.pack(">d", 0.28).hex()
    )
    assert [row.endpoint.identity.learner_id for row in request.responders] == [0, 1, 2]
    assert request.responders[0].weight_f64_bits == authoritative_weight
    assert request.responders[0].payload == authoritative_payload


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong-post", "post-fragment bytes do not match"),
        ("wrong-broadcast", "broadcast bytes do not match"),
        ("wrong-return-type", "must return ReconstructionOutput"),
        ("non-bytes", "outputs must be exact bytes"),
    ],
)
def test_reconstruction_mismatches_fail_closed(tmp_path, case, message):
    store = CaptureObjectStore(tmp_path / case)
    boundary, _arguments, internal = _publish_boundary(store)

    def reconstruct(_request):
        if case == "wrong-post":
            return ReconstructionOutput(b"wrong", internal["_raw"]["broadcast"])
        if case == "wrong-broadcast":
            return ReconstructionOutput(internal["_raw"]["post"], b"wrong")
        if case == "wrong-return-type":
            return (internal["_raw"]["post"], internal["_raw"]["broadcast"])
        if case == "non-bytes":
            return ReconstructionOutput(
                bytearray(internal["_raw"]["post"]),
                internal["_raw"]["broadcast"],
            )
        raise AssertionError(case)  # pragma: no cover

    with pytest.raises(ReconstructionMismatchError, match=message):
        verify_reconstruction(store, boundary, reconstruct)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra-field", "metadata fields are malformed"),
        ("boolean-schema", "unsupported schema"),
        ("invalid-session", "canonical UUID"),
        ("nonnewer-version", "strictly newer"),
        ("boolean-commit", "non-negative integer"),
        ("responder-index", "indices must be contiguous"),
        ("responder-order", "not strictly learner/rank ordered"),
        ("identity-cross-wire", "identity cross-reference mismatch"),
        ("provenance-cross-wire", "provenance cross-reference mismatch"),
        ("endpoint-cross-wire", "identity cross-reference mismatch"),
        ("weight-bits", "16 lowercase hex"),
        ("payload-sha", "lowercase SHA-256"),
        ("payload-ff", "payload metadata/object cross-reference mismatch"),
        ("digest-only-payload", "fields are malformed"),
        ("merge-config-extra", "fields are malformed"),
        ("outer-config-array", "parameters must be an object"),
        ("broadcast-sha", "broadcast SHA/bytes cross-reference mismatch"),
    ],
)
def test_resealed_boundary_schema_and_cross_wire_mutations_fail_closed(
    tmp_path, case, message
):
    store = CaptureObjectStore(tmp_path / case)
    boundary, _arguments, _internal = _publish_boundary(store)
    metadata = copy.deepcopy(store.load_manifest(boundary.manifest)["metadata"])

    if case == "extra-field":
        metadata["unexpected"] = 1
    elif case == "boolean-schema":
        metadata["schema_version"] = True
    elif case == "invalid-session":
        metadata["identity"]["capture_session_uuid"] = "not-a-uuid"
    elif case == "nonnewer-version":
        metadata["identity"]["post_fragment_version"] = PRE_VERSION
    elif case == "boolean-commit":
        metadata["identity"]["commit_seq"] = True
    elif case == "responder-index":
        metadata["responders"][0]["responder_index"] = True
    elif case == "responder-order":
        original_payloads = [
            {
                key: row[key]
                for key in ("payload_role", "payload_sha256", "payload_bytes")
            }
            for row in metadata["responders"]
        ]
        metadata["responders"][0], metadata["responders"][1] = (
            metadata["responders"][1],
            metadata["responders"][0],
        )
        metadata["responders"][0]["responder_index"] = 0
        metadata["responders"][1]["responder_index"] = 1
        for index, payload in enumerate(original_payloads):
            metadata["responders"][index].update(payload)
    elif case == "identity-cross-wire":
        metadata["responders"][0]["endpoint_identity"]["local_step"] += 1
    elif case == "provenance-cross-wire":
        metadata["responders"][0]["input_provenance"]["config_sha256"] = "f" * 64
    elif case == "endpoint-cross-wire":
        source = metadata["responders"][1]
        target = metadata["responders"][0]
        for key in (
            "endpoint_manifest_id",
            "endpoint_manifest_sha256",
            "endpoint_manifest_bytes",
        ):
            target[key] = source[key]
    elif case == "weight-bits":
        metadata["responders"][0]["weight_f64_bits"] = "ABC"
    elif case == "payload-sha":
        metadata["responders"][0]["payload_sha256"] = "A" * 64
    elif case == "payload-ff":
        metadata["responders"][0]["payload_sha256"] = "f" * 64
    elif case == "digest-only-payload":
        metadata["responders"][0].pop("payload_role")
        metadata["responders"][0].pop("payload_bytes")
    elif case == "merge-config-extra":
        metadata["merge_config"]["unexpected"] = 1
    elif case == "outer-config-array":
        metadata["outer_config"]["parameters"] = []
    elif case == "broadcast-sha":
        metadata["broadcast"]["sha256"] = "f" * 64
    else:  # pragma: no cover
        raise AssertionError(case)

    resealed = _reseal(store, boundary, metadata, f"resealed-{case}")
    with pytest.raises(SyncerBoundaryError, match=message):
        load_syncer_boundary(store, resealed)


def test_object_role_order_and_extra_objects_are_rejected(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    boundary, _arguments, _internal = _publish_boundary(store)
    manifest = store.load_manifest(boundary.manifest)
    metadata = copy.deepcopy(manifest["metadata"])
    entries = _entries(manifest)

    reordered = _reseal(
        store,
        boundary,
        metadata,
        "reordered-roles",
        [entries[1], entries[0], *entries[2:]],
    )
    with pytest.raises(SyncerBoundaryError, match="canonical role order"):
        load_syncer_boundary(store, reordered)

    extra_ref = _object(store, b"unexpected boundary object")
    extra = _reseal(
        store,
        boundary,
        metadata,
        "extra-role",
        [*entries, ManifestEntry("syncer/unexpected", extra_ref)],
    )
    with pytest.raises(SyncerBoundaryError, match="canonical role order"):
        load_syncer_boundary(store, extra)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-object", "regular non-symlink"),
        ("corrupt-object", "CAS SHA-256 mismatch"),
        ("missing-endpoint", "missing regular manifest"),
        ("corrupt-endpoint", "manifest SHA-256 mismatch"),
        ("corrupt-boundary", "manifest SHA-256 mismatch"),
    ],
)
def test_missing_and_corrupt_objects_or_manifests_fail_closed(tmp_path, case, message):
    store = CaptureObjectStore(tmp_path / case)
    boundary, arguments, _internal = _publish_boundary(store)
    if case == "missing-object":
        store.object_path(arguments["outer_state"].sha256).unlink()
    elif case == "corrupt-object":
        path = store.object_path(arguments["broadcast"].sha256)
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
    elif case == "missing-endpoint":
        endpoint = arguments["responders"][0].endpoint
        store.manifest_path(endpoint.manifest.sha256).unlink()
    elif case == "corrupt-endpoint":
        endpoint = arguments["responders"][0].endpoint
        path = store.manifest_path(endpoint.manifest.sha256)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    elif case == "corrupt-boundary":
        path = store.manifest_path(boundary.manifest.sha256)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    else:  # pragma: no cover
        raise AssertionError(case)

    with pytest.raises(CaptureStoreError, match=message):
        load_syncer_boundary(store, boundary)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong-session", "capture_session_uuid differs"),
        ("wrong-fragment", "active_fragment_id differs"),
        ("wrong-pre-version", "fragment version differs"),
        ("duplicate-learner", "unique learner_id"),
        ("no-responders", "at least one responder"),
        ("invalid-config", "not canonical JSON data"),
    ],
)
def test_publication_rejects_causally_mismatched_responders(tmp_path, case, message):
    store = CaptureObjectStore(tmp_path / case)
    arguments = _boundary_arguments(store)
    arguments.pop("_raw")
    endpoints = arguments.pop("_endpoints")
    if case == "wrong-session":
        wrong = _learner_endpoint(
            store, 9, session="87654321-4321-4765-8765-123456789abc"
        )
        arguments["responders"] = [_responder(store, wrong, 9)]
    elif case == "wrong-fragment":
        wrong = _learner_endpoint(store, 9, active_fragment_id=0)
        arguments["responders"] = [_responder(store, wrong, 9)]
    elif case == "wrong-pre-version":
        wrong = _learner_endpoint(store, 9, fragment_versions=(5, 16))
        arguments["responders"] = [_responder(store, wrong, 9)]
    elif case == "duplicate-learner":
        arguments["responders"] = [
            _responder(store, endpoints[0], 0),
            _responder(store, endpoints[0], 0),
        ]
    elif case == "no-responders":
        arguments["responders"] = []
    elif case == "invalid-config":
        arguments["outer_config"] = BoundaryConfig("nesterov", {"lr": float("nan")})
    else:  # pragma: no cover
        raise AssertionError(case)

    manifest_count = len(list(store.manifests_dir.iterdir()))
    with pytest.raises(SyncerBoundaryError, match=message):
        publish_syncer_boundary(store, f"invalid-{case}", **arguments)
    assert len(list(store.manifests_dir.iterdir())) == manifest_count
