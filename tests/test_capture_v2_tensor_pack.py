from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import pytest
import torch

import yeto.capture_v2_tensor_pack as tensor_pack_mod
from yeto.capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
)
from yeto.capture_v2_tensor_pack import (
    PAYLOAD_ROLE,
    TensorPackError,
    load_tensor_pack,
    publish_tensor_pack,
)


def _raw(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def _sample_inputs():
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not matrix.is_contiguous()
    trainable = OrderedDict(
        [
            ("model.layer.9.weight", matrix),
            ("model.layer.1.bias", torch.tensor([-0.0, 2.5], dtype=torch.float32)),
        ]
    )
    optimizer = OrderedDict(
        [
            ("model.layer.9.weight/exp_avg_sq", torch.arange(12).reshape(4, 3)),
            ("model.layer.1.bias/step", torch.tensor(17.0, dtype=torch.float32)),
            (
                "model.layer.1.bias/exp_avg",
                torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
            ),
        ]
    )
    clocks = OrderedDict([("tokens_total", 8192), ("local_step", 41)])
    return trainable, optimizer, clocks


def _publish_sample(store: CaptureObjectStore, manifest_id: str = "capture-0001"):
    trainable, optimizer, clocks = _sample_inputs()
    return publish_tensor_pack(
        store,
        manifest_id,
        fragment_id=3,
        trainable=trainable,
        optimizer=optimizer,
        clocks=clocks,
        metadata={"boundary": "post-step"},
    )


def _reseal_metadata(
    store: CaptureObjectStore,
    pack,
    metadata: dict,
    *,
    manifest_id: str,
    role: str = PAYLOAD_ROLE,
):
    return store.publish_manifest(
        manifest_id,
        [ManifestEntry(role, pack.payload)],
        metadata=metadata,
    )


def test_pack_is_deterministic_and_round_trips_exact_named_tensors(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    trainable, optimizer, clocks = _sample_inputs()
    first = publish_tensor_pack(
        store,
        "capture-0001",
        fragment_id=3,
        trainable=trainable,
        optimizer=optimizer,
        clocks=clocks,
        metadata={"boundary": "post-step"},
    )
    second = publish_tensor_pack(
        store,
        "capture-0001",
        fragment_id=3,
        trainable=OrderedDict(
            (name, tensor.detach().contiguous().clone())
            for name, tensor in reversed(list(trainable.items()))
        ),
        optimizer=OrderedDict(
            (name, tensor.detach().contiguous().clone())
            for name, tensor in reversed(list(optimizer.items()))
        ),
        clocks=OrderedDict(reversed(list(clocks.items()))),
        metadata={"boundary": "post-step"},
    )

    assert first.payload == second.payload
    assert first.manifest.sha256 == second.manifest.sha256
    assert first.payload_inserted is True
    assert second.payload_inserted is False
    assert second.manifest.inserted is False

    loaded = load_tensor_pack(store, first)
    assert loaded.manifest_id == "capture-0001"
    assert loaded.manifest_sha256 == first.manifest.sha256
    assert loaded.fragment_id == 3
    assert list(loaded.trainable) == sorted(trainable)
    assert list(loaded.optimizer) == sorted(optimizer)
    assert loaded.clocks == {"local_step": 41, "tokens_total": 8192}
    assert loaded.metadata == {"boundary": "post-step"}
    for name, expected in trainable.items():
        assert loaded.trainable[name].dtype == expected.dtype
        assert loaded.trainable[name].shape == expected.shape
        assert torch.equal(loaded.trainable[name], expected)
        assert _raw(loaded.trainable[name]) == _raw(expected)
    for name, expected in optimizer.items():
        assert loaded.optimizer[name].dtype == expected.dtype
        assert loaded.optimizer[name].shape == expected.shape
        assert torch.equal(loaded.optimizer[name], expected)
        assert _raw(loaded.optimizer[name]) == _raw(expected)

    manifest = store.load_manifest(first.manifest)
    assert manifest["metadata"]["fragment_id"] == 3
    descriptors = manifest["metadata"]["tensors"]
    assert [(row["kind"], row["name"]) for row in descriptors] == [
        *(("trainable", name) for name in sorted(trainable)),
        *(("optimizer", name) for name in sorted(optimizer)),
    ]
    payload = store.object_path(first.payload.sha256).read_bytes()
    for row in descriptors:
        tensor_raw = payload[row["offset"] : row["offset"] + row["bytes"]]
        assert hashlib.sha256(tensor_raw).hexdigest() == row["sha256"]

    audit = store.audit()
    assert audit.manifests == 1
    assert audit.references == 1
    assert audit.logical_bytes == first.payload.bytes
    assert audit.physical_bytes == first.payload.bytes


def test_round_trip_preserves_supported_optimizer_dtype_bits_and_shapes(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    fp32_bits = torch.tensor(
        [0x00000000, -0x80000000, 0x7FC00001, 0x3F800000], dtype=torch.int32
    ).view(torch.float32)
    trainable = {
        "fragment.edge_bits": fp32_bits,
        "fragment.empty": torch.empty((0, 3), dtype=torch.float32),
    }
    optimizer = {
        "state/bool": torch.tensor([True, False], dtype=torch.bool),
        "state/u8": torch.tensor([0, 255], dtype=torch.uint8),
        "state/i8": torch.tensor([-128, 127], dtype=torch.int8),
        "state/i16": torch.tensor([-32768, 32767], dtype=torch.int16),
        "state/i32": torch.tensor([-123456, 123456], dtype=torch.int32),
        "state/i64_clock": torch.tensor(2**40, dtype=torch.int64),
        "state/f16": torch.tensor([-1.5, 2.25], dtype=torch.float16),
        "state/bf16": torch.tensor([-1.5, 2.25], dtype=torch.bfloat16),
        "state/f32": torch.tensor([-1.5, 2.25], dtype=torch.float32),
        "state/f64_empty": torch.empty((2, 0), dtype=torch.float64),
    }
    ref = publish_tensor_pack(
        store,
        "dtype-bits",
        fragment_id=0,
        trainable=trainable,
        optimizer=optimizer,
        clocks={"optimizer_steps": 2**40},
    )
    loaded = load_tensor_pack(store, ref.manifest.sha256)

    for category, expected_values in (
        (loaded.trainable, trainable),
        (loaded.optimizer, optimizer),
    ):
        for name, expected in expected_values.items():
            actual = category[name]
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert _raw(actual) == _raw(expected)
    assert loaded.clocks == {"optimizer_steps": 2**40}


def test_encode_and_decode_are_independent_of_all_source_and_result_aliases(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    shared = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    original = shared.clone()
    ref = publish_tensor_pack(
        store,
        "aliases",
        fragment_id=0,
        trainable={"fragment.a": shared, "fragment.b": shared},
        optimizer={"state/exp_avg": shared},
        clocks={"step": 1},
    )

    shared.add_(1000)
    first = load_tensor_pack(store, ref)
    second = load_tensor_pack(store, ref)
    assert torch.equal(first.trainable["fragment.a"], original)
    assert torch.equal(first.trainable["fragment.b"], original)
    assert torch.equal(first.optimizer["state/exp_avg"], original)
    pointers = {
        first.trainable["fragment.a"].data_ptr(),
        first.trainable["fragment.b"].data_ptr(),
        first.optimizer["state/exp_avg"].data_ptr(),
        second.trainable["fragment.a"].data_ptr(),
    }
    assert len(pointers) == 4

    first.trainable["fragment.a"].add_(10)
    assert torch.equal(first.trainable["fragment.b"], original)
    assert torch.equal(first.optimizer["state/exp_avg"], original)
    assert torch.equal(second.trainable["fragment.a"], original)


def test_payload_and_manifest_byte_corruption_fail_before_decode(tmp_path):
    payload_store = CaptureObjectStore(tmp_path / "payload-cas")
    payload_pack = _publish_sample(payload_store)
    payload_path = payload_store.object_path(payload_pack.payload.sha256)
    payload_raw = bytearray(payload_path.read_bytes())
    payload_raw[len(payload_raw) // 2] ^= 1
    payload_path.write_bytes(payload_raw)
    with pytest.raises(CaptureStoreError, match="CAS SHA-256 mismatch"):
        load_tensor_pack(payload_store, payload_pack)

    manifest_store = CaptureObjectStore(tmp_path / "manifest-cas")
    manifest_pack = _publish_sample(manifest_store)
    manifest_path = manifest_store.manifest_path(manifest_pack.manifest.sha256)
    manifest_raw = bytearray(manifest_path.read_bytes())
    manifest_raw[len(manifest_raw) // 2] ^= 1
    manifest_path.write_bytes(manifest_raw)
    with pytest.raises(CaptureStoreError, match="manifest SHA-256 mismatch"):
        load_tensor_pack(manifest_store, manifest_pack)


def test_valid_cas_payload_with_stale_per_tensor_hash_fails_closed(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    pack = _publish_sample(store)
    base_manifest = store.load_manifest(pack.manifest)
    changed = bytearray(store.object_path(pack.payload.sha256).read_bytes())
    changed[0] ^= 1
    changed_payload = store.put_bytes(bytes(changed))
    resealed = store.publish_manifest(
        "stale-tensor-hash",
        [ManifestEntry(PAYLOAD_ROLE, changed_payload.ref)],
        metadata=copy.deepcopy(base_manifest["metadata"]),
    )

    with pytest.raises(TensorPackError, match="tensor row 0 SHA-256 mismatch"):
        load_tensor_pack(store, resealed)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra-metadata-field", "metadata fields are malformed"),
        ("boolean-schema", "unsupported schema"),
        ("missing-fragment-id", "metadata fields are malformed"),
        ("boolean-fragment-id", "must be a non-negative integer"),
        ("wrong-order-contract", "unsupported tensor order"),
        ("extra-descriptor-field", "row 0 fields are malformed"),
        ("unhashable-kind", "unsupported kind"),
        ("unhashable-dtype", "unsupported dtype"),
        ("non-f32-trainable", "trainable dtype must be f32"),
        ("boolean-shape", "shape dimension must be a non-negative integer"),
        ("offset-gap", "is not contiguous"),
        ("wrong-byte-count", "byte count mismatch"),
        ("reordered-tensors", "not strictly canonical-ordered"),
        ("boolean-clock", "must be a non-negative integer"),
        ("reordered-clocks", "not strictly name-sorted"),
    ],
)
def test_resealed_schema_mutations_fail_closed(tmp_path, case, message):
    store = CaptureObjectStore(tmp_path / case)
    pack = _publish_sample(store)
    metadata = copy.deepcopy(store.load_manifest(pack.manifest)["metadata"])

    if case == "extra-metadata-field":
        metadata["unexpected"] = 1
    elif case == "boolean-schema":
        metadata["schema_version"] = True
    elif case == "missing-fragment-id":
        metadata.pop("fragment_id")
    elif case == "boolean-fragment-id":
        metadata["fragment_id"] = True
    elif case == "wrong-order-contract":
        metadata["tensor_order"] = "dictionary-order"
    elif case == "extra-descriptor-field":
        metadata["tensors"][0]["unexpected"] = 1
    elif case == "unhashable-kind":
        metadata["tensors"][0]["kind"] = []
    elif case == "unhashable-dtype":
        metadata["tensors"][0]["dtype"] = []
    elif case == "non-f32-trainable":
        metadata["tensors"][0]["dtype"] = "f64"
    elif case == "boolean-shape":
        metadata["tensors"][0]["shape"][0] = True
    elif case == "offset-gap":
        metadata["tensors"][0]["offset"] = 1
    elif case == "wrong-byte-count":
        metadata["tensors"][0]["bytes"] += 1
    elif case == "reordered-tensors":
        metadata["tensors"][0], metadata["tensors"][1] = (
            metadata["tensors"][1],
            metadata["tensors"][0],
        )
    elif case == "boolean-clock":
        metadata["clocks"][0]["value"] = True
    elif case == "reordered-clocks":
        metadata["clocks"].reverse()
    else:  # pragma: no cover - keeps future parameter additions explicit
        raise AssertionError(case)

    resealed = _reseal_metadata(store, pack, metadata, manifest_id=f"resealed-{case}")
    with pytest.raises(TensorPackError, match=message):
        load_tensor_pack(store, resealed)


def test_manifest_with_wrong_payload_role_fails_closed(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    pack = _publish_sample(store)
    metadata = copy.deepcopy(store.load_manifest(pack.manifest)["metadata"])
    wrong_role = _reseal_metadata(
        store,
        pack,
        metadata,
        manifest_id="wrong-payload-role",
        role="tensor-pack/not-payload",
    )
    with pytest.raises(TensorPackError, match="exactly one"):
        load_tensor_pack(store, wrong_role)


def test_invalid_sources_and_clocks_publish_nothing(tmp_path):
    cases = [
        {
            "trainable": {"fragment": torch.ones(1, dtype=torch.float16)},
            "optimizer": {},
            "clocks": {},
            "message": "must be fp32",
        },
        {
            "trainable": {"worker/../fragment": torch.ones(1, dtype=torch.float32)},
            "optimizer": {},
            "clocks": {},
            "message": "unsafe path component",
        },
        {
            "trainable": {"fragment": torch.ones(1, dtype=torch.float32)},
            "optimizer": {"state/complex": torch.ones(1, dtype=torch.complex64)},
            "clocks": {},
            "message": "unsupported dtype",
        },
        {
            "trainable": {"fragment": torch.ones(1, dtype=torch.float32)},
            "optimizer": {},
            "clocks": {"step": True},
            "message": "non-negative integer",
        },
    ]
    for index, case in enumerate(cases):
        store = CaptureObjectStore(tmp_path / f"cas-{index}")
        with pytest.raises((TensorPackError, TypeError), match=case["message"]):
            publish_tensor_pack(
                store,
                f"invalid-{index}",
                fragment_id=0,
                trainable=case["trainable"],
                optimizer=case["optimizer"],
                clocks=case["clocks"],
            )
        assert list(store.objects_dir.iterdir()) == []
        assert list(store.manifests_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("manifest_id", "metadata", "message"),
    [
        ("../unsafe", {}, "manifest_id must match"),
        ("nonfinite", {"loss": float("nan")}, "not canonical JSON data"),
        ("not-json", {"tensor": torch.ones(1)}, "not canonical JSON data"),
    ],
)
def test_invalid_identity_or_metadata_is_rejected_before_payload_insert(
    tmp_path, manifest_id, metadata, message
):
    store = CaptureObjectStore(tmp_path / manifest_id.replace("/", "_"))
    with pytest.raises((TensorPackError, TypeError), match=message):
        publish_tensor_pack(
            store,
            manifest_id,
            fragment_id=0,
            trainable={"fragment": torch.ones(1, dtype=torch.float32)},
            optimizer={},
            clocks={},
            metadata=metadata,
        )
    assert list(store.objects_dir.iterdir()) == []
    assert list(store.manifests_dir.iterdir()) == []


def test_load_rejects_a_non_little_endian_host_before_decoding(tmp_path, monkeypatch):
    store = CaptureObjectStore(tmp_path / "cas")
    pack = _publish_sample(store)

    monkeypatch.setattr(tensor_pack_mod.sys, "byteorder", "big")
    with pytest.raises(TensorPackError, match="requires a little-endian host"):
        load_tensor_pack(store, pack)
