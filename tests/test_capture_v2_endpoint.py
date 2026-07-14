from __future__ import annotations

import copy
from collections import OrderedDict

import pytest
import torch

from yeto.capture_v2_endpoint import (
    FUTURE_GROUP_COUNT,
    EndpointIdentity,
    EndpointManifestError,
    FutureGroupRefs,
    InputProvenance,
    load_learner_endpoint,
    publish_learner_endpoint,
)
from yeto.capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ObjectRef,
)
from yeto.capture_v2_tensor_pack import publish_tensor_pack


def _object(store: CaptureObjectStore, label: str) -> ObjectRef:
    return store.put_bytes(f"opaque exact state: {label}".encode()).ref


def _fragment_pack(store: CaptureObjectStore, fragment_id: int):
    return publish_tensor_pack(
        store,
        f"fragment-pack-{fragment_id}",
        trainable={
            f"model.layer.{fragment_id}.weight": torch.tensor(
                [fragment_id + 0.25, fragment_id - 0.5], dtype=torch.float32
            )
        },
        optimizer={
            f"model.layer.{fragment_id}.weight/exp_avg": torch.tensor(
                [fragment_id + 1.0, fragment_id + 2.0], dtype=torch.float32
            ),
            f"model.layer.{fragment_id}.weight/step": torch.tensor(
                fragment_id + 7, dtype=torch.int64
            ),
        },
        clocks={"optimizer_steps": fragment_id + 7},
        metadata={"fragment_id": fragment_id},
    )


def _endpoint_arguments(
    store: CaptureObjectStore,
    *,
    fragment_count: int = 3,
    cuda_count: int = 2,
    future_state: str = "complete",
    future_indices=range(FUTURE_GROUP_COUNT),
    future_reason: str | None = None,
):
    packs = {index: _fragment_pack(store, index) for index in range(fragment_count)}
    future_refs = {
        index: _object(store, f"future-group-{index}") for index in future_indices
    }
    return {
        "identity": EndpointIdentity(
            capture_session_uuid="12345678-1234-5678-9234-567812345678",
            learner_id=4,
            rank=0,
            local_step=42,
            active_fragment_id=0,
            window_uuid="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        ),
        "input_provenance": InputProvenance(
            object=_object(store, "input-provenance"),
            source_commit="a" * 40,
            image_id="gcp:yeto-a100-v1",
            model_sha256="b" * 64,
            data_sha256="c" * 64,
            config_sha256="d" * 64,
        ),
        "fragment_packs": packs,
        "fragment_versions": [101 + index for index in range(fragment_count)],
        "mode": "train",
        "model_buffers": _object(store, "model-buffers"),
        "scheduler": OrderedDict(
            [("last_epoch", 41), ("base_lrs", [0.28]), ("step_count", 42)]
        ),
        "scaler": {"enabled": False, "scale": 1.0},
        "python_rng": _object(store, "python-rng"),
        "numpy_rng": _object(store, "numpy-rng"),
        "torch_cpu_rng": _object(store, "torch-cpu-rng"),
        "torch_cuda_rng": {
            index: _object(store, f"torch-cuda-rng-{index}")
            for index in range(cuda_count)
        },
        "future_groups": FutureGroupRefs(future_state, future_refs, future_reason),
    }


def _manifest_entries(store: CaptureObjectStore, manifest: dict):
    return [
        ManifestEntry(row["role"], ObjectRef(row["sha256"], row["bytes"]))
        for row in manifest["objects"]
    ]


def _reseal_endpoint(
    store: CaptureObjectStore,
    endpoint,
    metadata: dict,
    *,
    manifest_id: str,
    entries=None,
):
    manifest = store.load_manifest(endpoint.manifest)
    return store.publish_manifest(
        manifest_id,
        _manifest_entries(store, manifest) if entries is None else entries,
        metadata=metadata,
    )


def test_complete_endpoint_is_canonical_deterministic_and_cross_verified(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _endpoint_arguments(store)
    first = publish_learner_endpoint(store, "endpoint-0042", **arguments)

    reversed_arguments = dict(arguments)
    reversed_arguments["fragment_packs"] = OrderedDict(
        reversed(list(arguments["fragment_packs"].items()))
    )
    reversed_arguments["torch_cuda_rng"] = OrderedDict(
        reversed(list(arguments["torch_cuda_rng"].items()))
    )
    reversed_arguments["future_groups"] = FutureGroupRefs(
        "complete",
        OrderedDict(reversed(list(arguments["future_groups"].refs.items()))),
    )
    reversed_arguments["scheduler"] = OrderedDict(
        reversed(list(arguments["scheduler"].items()))
    )
    second = publish_learner_endpoint(store, "endpoint-0042", **reversed_arguments)

    assert second.manifest.sha256 == first.manifest.sha256
    assert second.manifest.inserted is False
    loaded = load_learner_endpoint(store, first)
    assert loaded.manifest_id == "endpoint-0042"
    assert loaded.manifest_sha256 == first.manifest.sha256
    assert loaded.identity == arguments["identity"]
    assert loaded.input_provenance == arguments["input_provenance"]
    assert loaded.mode == "train"
    assert loaded.fragment_versions == (101, 102, 103)
    assert list(loaded.fragments) == [0, 1, 2]
    for fragment_id, pack in loaded.fragments.items():
        assert pack.metadata["fragment_id"] == fragment_id
        assert list(pack.trainable) == [f"model.layer.{fragment_id}.weight"]
        assert pack.optimizer[f"model.layer.{fragment_id}.weight/step"].item() == (
            fragment_id + 7
        )
    assert loaded.model_buffers == arguments["model_buffers"]
    assert loaded.scheduler == {
        "base_lrs": [0.28],
        "last_epoch": 41,
        "step_count": 42,
    }
    assert loaded.scaler == {"enabled": False, "scale": 1.0}
    assert loaded.rng.python == arguments["python_rng"]
    assert loaded.rng.numpy == arguments["numpy_rng"]
    assert loaded.rng.torch_cpu == arguments["torch_cpu_rng"]
    assert loaded.rng.torch_cuda == tuple(arguments["torch_cuda_rng"].values())
    assert loaded.future_groups.state == "complete"
    assert loaded.future_groups.reason is None
    assert list(loaded.future_groups.refs) == list(range(FUTURE_GROUP_COUNT))

    manifest = store.load_manifest(first.manifest)
    assert [row["role"] for row in manifest["objects"]] == [
        "fragments/0/tensor-pack-payload",
        "fragments/1/tensor-pack-payload",
        "fragments/2/tensor-pack-payload",
        "model/buffers",
        "provenance/input",
        "rng/python",
        "rng/numpy",
        "rng/torch-cpu",
        "rng/torch-cuda/0",
        "rng/torch-cuda/1",
        *(f"future-groups/{index}" for index in range(FUTURE_GROUP_COUNT)),
    ]
    audit = store.audit()
    assert audit.manifests == 4
    assert audit.references == 3 + len(manifest["objects"])
    assert audit.unique_objects == len(manifest["objects"])
    assert audit.deduplicated_bytes == sum(
        pack.payload.bytes for pack in arguments["fragment_packs"].values()
    )


def test_explicit_incomplete_future_groups_round_trip_without_fake_refs(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _endpoint_arguments(
        store,
        fragment_count=1,
        cuda_count=0,
        future_state="incomplete",
        future_indices=(0, 3, 7),
        future_reason="capture budget exhausted before five future groups arrived",
    )
    endpoint = publish_learner_endpoint(store, "endpoint-incomplete", **arguments)
    loaded = load_learner_endpoint(store, endpoint)

    assert loaded.rng.torch_cuda == ()
    assert loaded.future_groups.state == "incomplete"
    assert list(loaded.future_groups.refs) == [0, 3, 7]
    assert loaded.future_groups.reason == (
        "capture budget exhausted before five future groups arrived"
    )
    metadata = store.load_manifest(endpoint.manifest)["metadata"]
    assert metadata["future_groups"] == {
        "state": "incomplete",
        "required": 8,
        "available": [
            {"index": 0, "role": "future-groups/0"},
            {"index": 3, "role": "future-groups/3"},
            {"index": 7, "role": "future-groups/7"},
        ],
        "reason": "capture budget exhausted before five future groups arrived",
    }
    store.audit()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("complete-seven", "require exactly 8 refs"),
        ("incomplete-eight", "fewer than eight refs"),
        ("complete-reason", "cannot have a reason"),
        ("incomplete-no-reason", "require a non-empty reason"),
        ("invalid-state", "must be 'complete' or 'incomplete'"),
        ("index-eight", r"outside \[0, 8\)"),
    ],
)
def test_future_group_publication_count_and_state_rules_fail_closed(
    tmp_path, case, message
):
    store = CaptureObjectStore(tmp_path / case)
    if case == "complete-seven":
        state, indices, reason = "complete", range(7), None
    elif case == "incomplete-eight":
        state, indices, reason = "incomplete", range(8), "marked partial"
    elif case == "complete-reason":
        state, indices, reason = "complete", range(8), "should not exist"
    elif case == "incomplete-no-reason":
        state, indices, reason = "incomplete", range(7), None
    elif case == "invalid-state":
        state, indices, reason = "unknown", range(7), "unknown state"
    elif case == "index-eight":
        state, indices, reason = "incomplete", (0, 8), "bad index"
    else:  # pragma: no cover
        raise AssertionError(case)
    arguments = _endpoint_arguments(
        store,
        fragment_count=1,
        future_state=state,
        future_indices=indices,
        future_reason=reason,
    )
    with pytest.raises(EndpointManifestError, match=message):
        publish_learner_endpoint(store, f"bad-{case}", **arguments)
    # Only the input tensor-pack manifest exists; no endpoint was published.
    assert len(list(store.manifests_dir.iterdir())) == 1


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra-field", "metadata fields are malformed"),
        ("boolean-schema", "unsupported schema"),
        ("wrong-mode", "mode must be one of"),
        ("short-versions", "length must equal fragment count"),
        ("boolean-version", "must be a non-negative integer"),
        ("noncontiguous-fragments", "canonical and contiguous"),
        ("scheduler-array", "scheduler metadata must be an object"),
        ("scaler-array", "scaler metadata must be an object or null"),
        ("identity-extra-field", "identity fields are malformed"),
        ("invalid-window-uuid", "canonical UUID"),
        ("active-fragment-outside", "must identify one endpoint fragment"),
        ("invalid-source-commit", "lowercase 40-hex commit"),
        ("provenance-object-mismatch", "cross-reference mismatch"),
        ("cuda-role-gap", "canonical and contiguous"),
        ("future-required-seven", "must require exactly 8"),
        ("future-complete-seven", "require exactly 8 refs"),
        ("future-incomplete-eight", "fewer than eight refs"),
    ],
)
def test_resealed_endpoint_schema_mutations_fail_closed(tmp_path, case, message):
    store = CaptureObjectStore(tmp_path / case)
    arguments = _endpoint_arguments(store, fragment_count=2)
    endpoint = publish_learner_endpoint(store, "valid-endpoint", **arguments)
    metadata = copy.deepcopy(store.load_manifest(endpoint.manifest)["metadata"])

    if case == "extra-field":
        metadata["unexpected"] = True
    elif case == "boolean-schema":
        metadata["schema_version"] = True
    elif case == "wrong-mode":
        metadata["mode"] = "predict"
    elif case == "short-versions":
        metadata["fragment_versions"].pop()
    elif case == "boolean-version":
        metadata["fragment_versions"][0] = True
    elif case == "noncontiguous-fragments":
        metadata["fragments"][1]["fragment_id"] = 0
    elif case == "scheduler-array":
        metadata["scheduler"] = []
    elif case == "scaler-array":
        metadata["scaler"] = []
    elif case == "identity-extra-field":
        metadata["identity"]["unexpected"] = 1
    elif case == "invalid-window-uuid":
        metadata["identity"]["window_uuid"] = "not-a-uuid"
    elif case == "active-fragment-outside":
        metadata["identity"]["active_fragment_id"] = 2
    elif case == "invalid-source-commit":
        metadata["input_provenance"]["source_commit"] = "A" * 40
    elif case == "provenance-object-mismatch":
        metadata["input_provenance"]["sha256"] = "f" * 64
    elif case == "cuda-role-gap":
        metadata["rng"]["torch_cuda_roles"][1] = "rng/torch-cuda/3"
    elif case == "future-required-seven":
        metadata["future_groups"]["required"] = 7
    elif case == "future-complete-seven":
        metadata["future_groups"]["available"].pop()
    elif case == "future-incomplete-eight":
        metadata["future_groups"]["state"] = "incomplete"
        metadata["future_groups"]["reason"] = "claimed incomplete"
    else:  # pragma: no cover
        raise AssertionError(case)

    resealed = _reseal_endpoint(
        store, endpoint, metadata, manifest_id=f"resealed-{case}"
    )
    with pytest.raises(EndpointManifestError, match=message):
        load_learner_endpoint(store, resealed)


def test_fragment_tensor_pack_payload_cross_reference_mismatch_is_rejected(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _endpoint_arguments(store, fragment_count=2)
    endpoint = publish_learner_endpoint(store, "valid-endpoint", **arguments)
    metadata = copy.deepcopy(store.load_manifest(endpoint.manifest)["metadata"])
    source = metadata["fragments"][1]
    target = metadata["fragments"][0]
    for key in (
        "tensor_pack_manifest_id",
        "tensor_pack_manifest_sha256",
        "tensor_pack_manifest_bytes",
    ):
        target[key] = source[key]
    resealed = _reseal_endpoint(
        store, endpoint, metadata, manifest_id="cross-reference-mismatch"
    )

    with pytest.raises(EndpointManifestError, match="cross-reference mismatch"):
        load_learner_endpoint(store, resealed)


def test_extra_or_reordered_object_roles_are_rejected(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _endpoint_arguments(store, fragment_count=1)
    endpoint = publish_learner_endpoint(store, "valid-endpoint", **arguments)
    manifest = store.load_manifest(endpoint.manifest)
    metadata = copy.deepcopy(manifest["metadata"])
    entries = _manifest_entries(store, manifest)

    extra = _object(store, "unexpected-object")
    with_extra = _reseal_endpoint(
        store,
        endpoint,
        metadata,
        manifest_id="extra-object-role",
        entries=[*entries, ManifestEntry("unexpected/object", extra)],
    )
    with pytest.raises(EndpointManifestError, match="object roles differ"):
        load_learner_endpoint(store, with_extra)

    reordered = _reseal_endpoint(
        store,
        endpoint,
        metadata,
        manifest_id="reordered-object-roles",
        entries=[entries[1], entries[0], *entries[2:]],
    )
    with pytest.raises(EndpointManifestError, match="object roles differ"):
        load_learner_endpoint(store, reordered)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-object", "regular non-symlink"),
        ("corrupt-object", "CAS SHA-256 mismatch"),
        ("missing-pack-manifest", "missing regular manifest"),
        ("corrupt-pack-manifest", "manifest SHA-256 mismatch"),
        ("corrupt-endpoint-manifest", "manifest SHA-256 mismatch"),
    ],
)
def test_missing_and_corrupt_cross_references_fail_before_restore(
    tmp_path, case, message
):
    store = CaptureObjectStore(tmp_path / case)
    arguments = _endpoint_arguments(store, fragment_count=1)
    endpoint = publish_learner_endpoint(store, "valid-endpoint", **arguments)

    if case == "missing-object":
        store.object_path(arguments["python_rng"].sha256).unlink()
    elif case == "corrupt-object":
        path = store.object_path(arguments["model_buffers"].sha256)
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
    elif case == "missing-pack-manifest":
        pack = arguments["fragment_packs"][0]
        store.manifest_path(pack.manifest.sha256).unlink()
    elif case == "corrupt-pack-manifest":
        pack = arguments["fragment_packs"][0]
        path = store.manifest_path(pack.manifest.sha256)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    elif case == "corrupt-endpoint-manifest":
        path = store.manifest_path(endpoint.manifest.sha256)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    else:  # pragma: no cover
        raise AssertionError(case)

    with pytest.raises(CaptureStoreError, match=message):
        load_learner_endpoint(store, endpoint)


def test_invalid_endpoint_inputs_are_rejected_without_an_endpoint_manifest(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _endpoint_arguments(store, fragment_count=2)
    original_manifest_count = len(list(store.manifests_dir.iterdir()))

    invalid_cases = [
        ({"fragment_versions": [1]}, "length must equal"),
        ({"fragment_versions": [True, 2]}, "non-negative integer"),
        ({"mode": "predict"}, "mode must be one of"),
        ({"scheduler": {"loss": float("nan")}}, "not canonical JSON data"),
        (
            {
                "identity": EndpointIdentity(
                    capture_session_uuid="12345678-1234-5678-9234-567812345678",
                    learner_id=4,
                    rank=0,
                    local_step=42,
                    active_fragment_id=0,
                    window_uuid="AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE",
                )
            },
            "canonical UUID",
        ),
        (
            {
                "input_provenance": InputProvenance(
                    object=arguments["input_provenance"].object,
                    source_commit="short",
                    image_id="gcp:yeto-a100-v1",
                    model_sha256="b" * 64,
                    data_sha256="c" * 64,
                    config_sha256="d" * 64,
                )
            },
            "lowercase 40-hex commit",
        ),
        (
            {
                "fragment_packs": {
                    0: arguments["fragment_packs"][0],
                    2: arguments["fragment_packs"][1],
                }
            },
            "contiguous from zero",
        ),
    ]
    for index, (changes, message) in enumerate(invalid_cases):
        attempt = dict(arguments)
        attempt.update(changes)
        with pytest.raises((EndpointManifestError, TypeError), match=message):
            publish_learner_endpoint(store, f"invalid-{index}", **attempt)
    assert len(list(store.manifests_dir.iterdir())) == original_manifest_count
