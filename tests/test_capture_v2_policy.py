from __future__ import annotations

import copy
import hashlib
import struct

import pytest
import torch

from yeto.capture_v2_endpoint import (
    EndpointIdentity,
    FutureGroupRefs,
    InputProvenance,
    publish_learner_endpoint,
)
from yeto.capture_v2_policy import (
    CAPABILITIES,
    PolicyContractError,
    load_policy_definition,
    load_policy_outcome,
    load_sealed_outer_action,
    publish_policy_definition,
    publish_policy_outcome,
    publish_sealed_outer_action,
)
from yeto.capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ObjectRef,
)
from yeto.capture_v2_syncer import (
    BoundaryConfig,
    ResponderEndpointRef,
    SyncerBoundaryIdentity,
    load_syncer_boundary,
    publish_syncer_boundary,
)
from yeto.capture_v2_tensor_pack import publish_tensor_pack


SESSION = "12345678-1234-5678-9234-567812345678"


def _object(store: CaptureObjectStore, raw: bytes) -> ObjectRef:
    return store.put_bytes(raw).ref


def _endpoint(
    store: CaptureObjectStore,
    *,
    complete_future_groups: bool = False,
    cuda_rng_count: int = 0,
):
    pack = publish_tensor_pack(
        store,
        "policy-fixture-fragment",
        trainable={"model.layer.weight": torch.tensor([1.0, -2.0])},
        optimizer={"model.layer.weight/exp_avg": torch.tensor([0.25, -0.5])},
        clocks={"optimizer_steps": 7},
        metadata={"fragment_id": 0},
    )
    future_groups = {
        index: _object(store, f"future group {index}".encode())
        for index in range(8 if complete_future_groups else 0)
    }
    return publish_learner_endpoint(
        store,
        "policy-fixture-endpoint",
        identity=EndpointIdentity(
            capture_session_uuid=SESSION,
            learner_id=0,
            rank=0,
            local_step=100,
            active_fragment_id=0,
            window_uuid="00000000-0000-4000-8000-000000000001",
        ),
        input_provenance=InputProvenance(
            object=_object(store, b"endpoint provenance"),
            source_commit="a" * 40,
            image_id="gcp:yeto-a100-v1",
            model_sha256="b" * 64,
            data_sha256="c" * 64,
            config_sha256="d" * 64,
        ),
        fragment_packs={0: pack},
        fragment_versions=[7],
        mode="train",
        model_buffers=_object(store, b"model buffers"),
        scheduler={"last_epoch": 7},
        scaler=None,
        python_rng=_object(store, b"python rng"),
        numpy_rng=_object(store, b"numpy rng"),
        torch_cpu_rng=_object(store, b"torch cpu rng"),
        torch_cuda_rng={
            index: _object(store, f"torch cuda rng {index}".encode())
            for index in range(cuda_rng_count)
        },
        future_groups=FutureGroupRefs(
            "complete" if complete_future_groups else "incomplete",
            future_groups,
            None
            if complete_future_groups
            else "future groups unavailable in policy fixture",
        ),
    )


def _boundary(
    store: CaptureObjectStore,
    *,
    weight_f64_bits=None,
    complete_future_groups: bool = False,
    cuda_rng_count: int = 0,
    outer_lr_f64_bits=struct.pack(">d", 0.28).hex(),
):
    endpoint = _endpoint(
        store,
        complete_future_groups=complete_future_groups,
        cuda_rng_count=cuda_rng_count,
    )
    pre = _object(store, b"pre fragment")
    outer = _object(store, b"outer state")
    post = _object(store, b"post fragment")
    broadcast = _object(store, b"broadcast fragment")
    responder = ResponderEndpointRef(
        endpoint=endpoint,
        weight_f64_bits=(
            struct.pack(">d", 128.0).hex()
            if weight_f64_bits is None
            else weight_f64_bits
        ),
        payload_sha256=hashlib.sha256(b"responder payload").hexdigest(),
    )
    return publish_syncer_boundary(
        store,
        "policy-fixture-boundary",
        identity=SyncerBoundaryIdentity(
            capture_session_uuid=SESSION,
            commit_id="step-00000008-fragment-0000",
            commit_seq=8,
            fragment_id=0,
            pre_fragment_version=7,
            post_fragment_version=8,
        ),
        responders=[responder],
        pre_fragment=pre,
        post_fragment=post,
        outer_state=outer,
        broadcast=broadcast,
        merge_config=BoundaryConfig("rda", {"weighted": True}),
        outer_config=BoundaryConfig(
            "nesterov",
            {
                **(
                    {"lr_f64_bits": outer_lr_f64_bits}
                    if outer_lr_f64_bits is not None
                    else {}
                ),
                "momentum_f64_bits": struct.pack(">d", 0.9).hex(),
            },
        ),
    )


def _policy(store: CaptureObjectStore, capabilities=CAPABILITIES, *, suffix=""):
    source = _object(store, f"policy source {suffix}".encode())
    config = _object(store, f"policy config {suffix}".encode())
    ref = publish_policy_definition(
        store,
        f"policy-definition{suffix}",
        policy_id=f"candidate{suffix}" if suffix else "candidate",
        policy_version="1.2.0",
        source_commit="e" * 40,
        source=source,
        config=config,
        capabilities=capabilities,
    )
    return ref, source, config


def _action_arguments(
    store: CaptureObjectStore,
    *,
    capabilities=CAPABILITIES,
    required_capabilities=("global_boundary_state",),
    boundary_kwargs=None,
):
    boundary = _boundary(store, **(boundary_kwargs or {}))
    policy, _source, config = _policy(store, capabilities)
    stock = _object(store, b"stock pseudo-gradient")
    selected = _object(store, b"selected nonstock pseudo-gradient")
    result = _object(store, b"resulting fragment")
    return {
        "policy": policy,
        "boundary": boundary,
        "fragment_id": 0,
        "required_capabilities": list(required_capabilities),
        "stock_pseudo_gradient": stock,
        "selected_pseudo_gradient": selected,
        "outer_lr_f64_bits": struct.pack(">d", 0.28).hex(),
        "resulting_fragment": result,
        "decision": _object(store, b"decision record"),
        "config_sha256": config.sha256,
        "action_kind": "nonstock",
        "action_reason": "candidate supplied a sealed pseudo-gradient",
        "fallback_reason": None,
    }


def _publish_action(store: CaptureObjectStore, manifest_id="sealed-action"):
    arguments = _action_arguments(store)
    action = publish_sealed_outer_action(store, manifest_id, **arguments)
    return action, arguments


def _entries(manifest: dict):
    return [
        ManifestEntry(row["role"], ObjectRef(row["sha256"], row["bytes"]))
        for row in manifest["objects"]
    ]


def _reseal(store, original, metadata, manifest_id, entries=None):
    manifest_ref = original.manifest
    manifest = store.load_manifest(manifest_ref)
    return store.publish_manifest(
        manifest_id,
        _entries(manifest) if entries is None else entries,
        metadata=metadata,
    )


def test_frozen_policy_is_canonical_deterministic_and_cas_backed(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    source = _object(store, b"source tree archive")
    config = _object(store, b"canonical config bytes")
    first = publish_policy_definition(
        store,
        "policy-definition",
        policy_id="midpoint-adam-v2",
        policy_version="2.0.1",
        source_commit="a" * 40,
        source=source,
        config=config,
        capabilities=reversed(CAPABILITIES),
    )
    second = publish_policy_definition(
        store,
        "policy-definition",
        policy_id="midpoint-adam-v2",
        policy_version="2.0.1",
        source_commit="a" * 40,
        source=source,
        config=config,
        capabilities=CAPABILITIES,
    )

    assert second.manifest.sha256 == first.manifest.sha256
    assert second.manifest.inserted is False
    loaded = load_policy_definition(store, first)
    assert loaded.policy_id == "midpoint-adam-v2"
    assert loaded.policy_version == "2.0.1"
    assert loaded.source_commit == "a" * 40
    assert loaded.source == source
    assert loaded.config == config
    assert loaded.capabilities == CAPABILITIES
    manifest = store.load_manifest(first.manifest)
    assert [row["role"] for row in manifest["objects"]] == [
        "policy/source",
        "policy/config",
    ]
    store.audit()


def test_policy_may_declare_the_empty_capability_set(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    policy, _source, _config = _policy(store, capabilities=())

    assert load_policy_definition(store, policy).capabilities == ()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra-field", "metadata fields are malformed"),
        ("boolean-schema", "unsupported schema"),
        ("policy-id", "canonical lowercase identifier"),
        ("policy-version", "canonical MAJOR.MINOR.PATCH"),
        ("source-commit", "lowercase 40-hex commit"),
        ("capability-object", "capabilities must be an array"),
        ("capability-duplicate", "duplicate capabilities"),
        ("capability-order", "not in canonical order"),
        ("capability-unknown", "unknown capability"),
        ("source-cross-wire", "metadata cross-reference mismatch"),
    ],
)
def test_resealed_policy_schema_mutations_fail_closed(tmp_path, case, message):
    store = CaptureObjectStore(tmp_path / case)
    policy, _source, _config = _policy(store)
    metadata = copy.deepcopy(store.load_manifest(policy.manifest)["metadata"])
    if case == "extra-field":
        metadata["unexpected"] = 1
    elif case == "boolean-schema":
        metadata["schema_version"] = True
    elif case == "policy-id":
        metadata["policy_id"] = "Uppercase"
    elif case == "policy-version":
        metadata["policy_version"] = "01.2.0"
    elif case == "source-commit":
        metadata["source_commit"] = "A" * 40
    elif case == "capability-object":
        metadata["capabilities"] = {name: True for name in CAPABILITIES}
    elif case == "capability-duplicate":
        metadata["capabilities"].append(metadata["capabilities"][0])
    elif case == "capability-order":
        metadata["capabilities"].reverse()
    elif case == "capability-unknown":
        metadata["capabilities"][0] = "omniscience"
    elif case == "source-cross-wire":
        metadata["source"]["sha256"] = "f" * 64
    else:  # pragma: no cover
        raise AssertionError(case)
    resealed = _reseal(store, policy, metadata, f"resealed-{case}")
    with pytest.raises(PolicyContractError, match=message):
        load_policy_definition(store, resealed)


def test_nonstock_action_is_canonical_and_contains_no_outcome_fields(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store)
    first = publish_sealed_outer_action(store, "sealed-action", **arguments)
    reversed_arguments = dict(arguments)
    reversed_arguments["required_capabilities"] = list(
        reversed(arguments["required_capabilities"])
    )
    second = publish_sealed_outer_action(store, "sealed-action", **reversed_arguments)

    assert second.manifest.sha256 == first.manifest.sha256
    assert second.manifest.inserted is False
    loaded = load_sealed_outer_action(store, first)
    assert loaded.policy.policy_id == "candidate"
    assert loaded.boundary.identity.fragment_id == 0
    assert loaded.required_capabilities == ("global_boundary_state",)
    assert loaded.stock_pseudo_gradient == arguments["stock_pseudo_gradient"]
    assert loaded.selected_pseudo_gradient == arguments["selected_pseudo_gradient"]
    assert loaded.selected_pseudo_gradient != loaded.stock_pseudo_gradient
    assert loaded.decision == arguments["decision"]
    assert loaded.decision_sha256 == arguments["decision"].sha256
    assert loaded.action_kind == "nonstock"
    assert loaded.fallback_reason is None
    manifest = store.load_manifest(first.manifest)
    assert [row["role"] for row in manifest["objects"]] == [
        "action/stock-pseudo-gradient",
        "action/selected-pseudo-gradient",
        "action/resulting-fragment",
        "action/decision",
    ]
    assert not any("loss" in key or "outcome" in key for key in manifest["metadata"])
    store.audit()


def test_stock_fallback_requires_the_exact_stock_object(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store)
    arguments.update(
        action_kind="stock_fallback",
        selected_pseudo_gradient=arguments["stock_pseudo_gradient"],
        resulting_fragment=load_syncer_boundary(
            store, arguments["boundary"]
        ).post_fragment,
        fallback_reason="candidate capability unavailable at this boundary",
    )
    action = publish_sealed_outer_action(store, "stock-fallback", **arguments)
    loaded = load_sealed_outer_action(store, action)
    assert loaded.selected_pseudo_gradient == loaded.stock_pseudo_gradient
    assert loaded.fallback_reason == (
        "candidate capability unavailable at this boundary"
    )


@pytest.mark.parametrize(
    ("kind", "same_object", "fallback", "message"),
    [
        ("stock_fallback", False, "fell back", "must reference the exact stock"),
        ("stock_fallback", True, None, "non-empty bounded string"),
        ("nonstock", True, None, "must differ from the stock"),
        ("nonstock", False, "not allowed", "cannot contain fallback_reason"),
        ("unknown", False, None, "must be 'stock_fallback' or 'nonstock'"),
    ],
)
def test_action_kind_and_fallback_contract_fails_closed(
    tmp_path, kind, same_object, fallback, message
):
    store = CaptureObjectStore(tmp_path / kind)
    arguments = _action_arguments(store)
    arguments["action_kind"] = kind
    arguments["selected_pseudo_gradient"] = (
        arguments["stock_pseudo_gradient"]
        if same_object
        else arguments["selected_pseudo_gradient"]
    )
    arguments["fallback_reason"] = fallback
    if kind == "stock_fallback":
        arguments["resulting_fragment"] = load_syncer_boundary(
            store, arguments["boundary"]
        ).post_fragment
    with pytest.raises(PolicyContractError, match=message):
        publish_sealed_outer_action(store, "invalid-action", **arguments)


def test_missing_policy_capability_rejects_publication_and_resealed_action(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store, capabilities=())
    with pytest.raises(PolicyContractError, match="missing required capabilities"):
        publish_sealed_outer_action(store, "missing-capability", **arguments)

    arguments["required_capabilities"] = []
    valid = publish_sealed_outer_action(store, "valid-capability", **arguments)
    metadata = copy.deepcopy(store.load_manifest(valid.manifest)["metadata"])
    metadata["required_capabilities"].append("global_boundary_state")
    resealed = _reseal(store, valid, metadata, "resealed-missing-capability")
    with pytest.raises(PolicyContractError, match="missing required capabilities"):
        load_sealed_outer_action(store, resealed)


@pytest.mark.parametrize(
    "capability",
    [
        "same_fragment_history",
        "midpoint_adam",
        "model_autograd",
        "proposal_stream",
    ],
)
def test_capabilities_without_v1_boundary_evidence_fail_closed(tmp_path, capability):
    store = CaptureObjectStore(tmp_path / capability)
    arguments = _action_arguments(store, required_capabilities=(capability,))

    with pytest.raises(PolicyContractError, match="no v1 boundary evidence schema"):
        publish_sealed_outer_action(store, "unsupported-capability", **arguments)


def test_worker_restore_requires_at_least_one_cuda_rng_object(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(
        store,
        required_capabilities=("worker_restore",),
        boundary_kwargs={"cuda_rng_count": 0},
    )

    with pytest.raises(PolicyContractError, match="at least one CUDA RNG object"):
        publish_sealed_outer_action(store, "zero-cuda-rng", **arguments)


def test_crn_train_k8_requires_complete_canonical_future_groups(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(
        store,
        required_capabilities=("crn_train_k8",),
        boundary_kwargs={"complete_future_groups": False},
    )

    with pytest.raises(
        PolicyContractError, match="complete canonical future groups 0..7"
    ):
        publish_sealed_outer_action(store, "incomplete-future-groups", **arguments)


def test_supported_capabilities_are_proved_by_complete_boundary_evidence(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(
        store,
        required_capabilities=reversed(
            ("global_boundary_state", "worker_restore", "crn_train_k8")
        ),
        boundary_kwargs={"complete_future_groups": True, "cuda_rng_count": 1},
    )

    action = publish_sealed_outer_action(store, "proved-capabilities", **arguments)

    assert load_sealed_outer_action(store, action).required_capabilities == (
        "global_boundary_state",
        "worker_restore",
        "crn_train_k8",
    )


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, -1.0]
)
def test_action_outer_lr_must_decode_to_finite_positive_f64(tmp_path, value):
    store = CaptureObjectStore(tmp_path / struct.pack(">d", value).hex())
    arguments = _action_arguments(store)
    arguments["outer_lr_f64_bits"] = struct.pack(">d", value).hex()

    with pytest.raises(PolicyContractError, match="finite positive f64"):
        publish_sealed_outer_action(store, "invalid-action-lr", **arguments)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "must contain lr_f64_bits"),
        (True, "16 lowercase hex digits"),
        (struct.pack(">d", float("nan")).hex(), "finite positive f64"),
    ],
)
def test_boundary_outer_lr_must_be_present_typed_and_finite(tmp_path, value, message):
    store = CaptureObjectStore(tmp_path / str(value))
    arguments = _action_arguments(store, boundary_kwargs={"outer_lr_f64_bits": value})

    with pytest.raises(PolicyContractError, match=message):
        publish_sealed_outer_action(store, "invalid-boundary-lr", **arguments)


def test_action_outer_lr_bits_must_exactly_equal_boundary(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store)
    arguments["outer_lr_f64_bits"] = struct.pack(">d", 0.1).hex()

    with pytest.raises(PolicyContractError, match="differs from syncer boundary"):
        publish_sealed_outer_action(store, "cross-wired-lr", **arguments)


def test_decision_requires_exact_cas_bytes_not_an_arbitrary_digest(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store)
    arguments["decision"] = "f" * 64

    with pytest.raises(TypeError, match="decision must be an ObjectRef"):
        publish_sealed_outer_action(store, "arbitrary-decision-digest", **arguments)


def test_stock_fallback_rejects_unrelated_resulting_fragment(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store)
    arguments.update(
        action_kind="stock_fallback",
        selected_pseudo_gradient=arguments["stock_pseudo_gradient"],
        fallback_reason="attempted fallback with an unrelated result",
    )

    with pytest.raises(PolicyContractError, match="exact boundary post_fragment"):
        publish_sealed_outer_action(store, "unrelated-stock-result", **arguments)


def test_nonstock_result_derivation_remains_policy_owned_and_identity_bound(tmp_path):
    """V1 binds candidate bytes but cannot prove policy-owned optimizer math."""

    store = CaptureObjectStore(tmp_path / "cas")
    arguments = _action_arguments(store)
    boundary = load_syncer_boundary(store, arguments["boundary"])
    assert arguments["resulting_fragment"] != boundary.post_fragment

    action = publish_sealed_outer_action(store, "policy-owned-result", **arguments)

    assert (
        load_sealed_outer_action(store, action).resulting_fragment
        == arguments["resulting_fragment"]
    )


@pytest.mark.parametrize(
    "weight", [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, -1.0]
)
def test_sealed_action_rejects_nonfinite_nonpositive_or_negative_zero_weights(
    tmp_path, weight
):
    store = CaptureObjectStore(tmp_path / struct.pack(">d", weight).hex())
    boundary = _boundary(store, weight_f64_bits=struct.pack(">d", weight).hex())
    policy, _source, config = _policy(store)
    stock = _object(store, b"stock")
    selected = _object(store, b"selected")
    with pytest.raises(PolicyContractError, match="finite strictly positive"):
        publish_sealed_outer_action(
            store,
            "invalid-weight-action",
            policy=policy,
            boundary=boundary,
            fragment_id=0,
            required_capabilities=[],
            stock_pseudo_gradient=stock,
            selected_pseudo_gradient=selected,
            outer_lr_f64_bits=struct.pack(">d", 0.28).hex(),
            resulting_fragment=_object(store, b"result"),
            decision=_object(store, b"decision"),
            config_sha256=config.sha256,
            action_kind="nonstock",
            action_reason="would otherwise select candidate",
            fallback_reason=None,
        )


def test_sealed_action_load_rejects_resealed_negative_zero_boundary_weight(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    action, arguments = _publish_action(store)

    boundary_manifest = store.load_manifest(arguments["boundary"].manifest)
    boundary_metadata = copy.deepcopy(boundary_manifest["metadata"])
    boundary_metadata["responders"][0]["weight_f64_bits"] = struct.pack(
        ">d", -0.0
    ).hex()
    resealed_boundary = store.publish_manifest(
        "resealed-negative-zero-boundary",
        _entries(boundary_manifest),
        metadata=boundary_metadata,
    )

    action_manifest = store.load_manifest(action.manifest)
    action_metadata = copy.deepcopy(action_manifest["metadata"])
    action_metadata["boundary"] = {
        "manifest_id": resealed_boundary.manifest_id,
        "sha256": resealed_boundary.sha256,
        "bytes": resealed_boundary.bytes,
    }
    resealed_action = store.publish_manifest(
        "resealed-negative-zero-action",
        _entries(action_manifest),
        metadata=action_metadata,
    )

    with pytest.raises(PolicyContractError, match="finite strictly positive"):
        load_sealed_outer_action(store, resealed_action)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra-outcome-field", "contain outcome fields"),
        ("boolean-schema", "unsupported schema"),
        ("boolean-fragment", "non-negative integer"),
        ("boundary-cross-wire", "fragment_id differs"),
        ("config-cross-wire", "differs from policy config"),
        ("required-object", "must be an array"),
        ("required-order", "not in canonical order"),
        ("unsupported-capability", "no v1 boundary evidence schema"),
        ("worker-restore-no-cuda", "at least one CUDA RNG object"),
        ("crn-incomplete", "complete canonical future groups 0..7"),
        ("selected-cross-wire", "must differ from the stock"),
        ("stock-result-not-boundary", "exact boundary post_fragment"),
        ("stock-cross-wire", "metadata/object cross-reference mismatch"),
        ("decision-object-cross-wire", "metadata/object cross-reference mismatch"),
        ("decision-digest-mismatch", "differs from decision object"),
        ("decision-sha", "lowercase SHA-256"),
        ("outer-lr", "16 lowercase hex"),
        ("outer-lr-nan", "finite positive f64"),
        ("outer-lr-mismatch", "differs from syncer boundary"),
        ("reason-type", "non-empty bounded string"),
    ],
)
def test_resealed_action_schema_role_and_cross_wire_mutations_fail_closed(
    tmp_path, case, message
):
    store = CaptureObjectStore(tmp_path / case)
    action, _arguments = _publish_action(store)
    manifest = store.load_manifest(action.manifest)
    metadata = copy.deepcopy(manifest["metadata"])
    entries = _entries(manifest)
    if case == "extra-outcome-field":
        metadata["k8_loss"] = 1.0
    elif case == "boolean-schema":
        metadata["schema_version"] = True
    elif case == "boolean-fragment":
        metadata["fragment_id"] = True
    elif case == "boundary-cross-wire":
        metadata["fragment_id"] = 1
    elif case == "config-cross-wire":
        metadata["config_sha256"] = "f" * 64
    elif case == "required-object":
        metadata["required_capabilities"] = {"same_fragment_history": True}
    elif case == "required-order":
        metadata["required_capabilities"] = [
            "crn_train_k8",
            "global_boundary_state",
        ]
    elif case == "unsupported-capability":
        metadata["required_capabilities"] = ["same_fragment_history"]
    elif case == "worker-restore-no-cuda":
        metadata["required_capabilities"] = ["worker_restore"]
    elif case == "crn-incomplete":
        metadata["required_capabilities"] = ["crn_train_k8"]
    elif case == "selected-cross-wire":
        metadata["selected_pseudo_gradient"]["sha256"] = metadata[
            "stock_pseudo_gradient"
        ]["sha256"]
        metadata["selected_pseudo_gradient"]["bytes"] = metadata[
            "stock_pseudo_gradient"
        ]["bytes"]
        entries[1] = ManifestEntry(entries[1].role, entries[0].object)
    elif case == "stock-result-not-boundary":
        metadata["action_kind"] = "stock_fallback"
        metadata["fallback_reason"] = "resealed as invalid stock fallback"
        metadata["selected_pseudo_gradient"]["sha256"] = metadata[
            "stock_pseudo_gradient"
        ]["sha256"]
        metadata["selected_pseudo_gradient"]["bytes"] = metadata[
            "stock_pseudo_gradient"
        ]["bytes"]
        entries[1] = ManifestEntry(entries[1].role, entries[0].object)
    elif case == "stock-cross-wire":
        metadata["stock_pseudo_gradient"]["sha256"] = "f" * 64
    elif case == "decision-object-cross-wire":
        metadata["decision"]["sha256"] = "f" * 64
        metadata["decision_sha256"] = "f" * 64
    elif case == "decision-digest-mismatch":
        metadata["decision_sha256"] = "f" * 64
    elif case == "decision-sha":
        metadata["decision_sha256"] = "A" * 64
    elif case == "outer-lr":
        metadata["outer_lr_f64_bits"] = "ABC"
    elif case == "outer-lr-nan":
        metadata["outer_lr_f64_bits"] = struct.pack(">d", float("nan")).hex()
    elif case == "outer-lr-mismatch":
        metadata["outer_lr_f64_bits"] = struct.pack(">d", 0.1).hex()
    elif case == "reason-type":
        metadata["action_reason"] = 123
    else:  # pragma: no cover
        raise AssertionError(case)
    resealed = _reseal(store, action, metadata, f"resealed-{case}", entries)
    with pytest.raises(PolicyContractError, match=message):
        load_sealed_outer_action(store, resealed)


def test_action_object_role_order_and_corruption_fail_closed(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    action, arguments = _publish_action(store)
    manifest = store.load_manifest(action.manifest)
    metadata = copy.deepcopy(manifest["metadata"])
    entries = _entries(manifest)
    reordered = _reseal(
        store,
        action,
        metadata,
        "reordered-action-objects",
        [entries[1], entries[0], entries[2], entries[3]],
    )
    with pytest.raises(PolicyContractError, match="canonical order"):
        load_sealed_outer_action(store, reordered)

    path = store.object_path(arguments["selected_pseudo_gradient"].sha256)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(CaptureStoreError, match="CAS SHA-256 mismatch"):
        load_sealed_outer_action(store, action)


def test_decision_object_corruption_fails_closed(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    action, arguments = _publish_action(store)
    path = store.object_path(arguments["decision"].sha256)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)

    with pytest.raises(CaptureStoreError, match="CAS SHA-256 mismatch"):
        load_sealed_outer_action(store, action)


def test_outcome_is_separate_append_only_and_does_not_change_action_bytes(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    action, _arguments = _publish_action(store)
    action_path = store.manifest_path(action.manifest.sha256)
    action_bytes_before = action_path.read_bytes()
    outcome = publish_policy_outcome(
        store,
        "outcome-0001",
        action=action,
        k0=_object(store, b"k0 evaluation artifact"),
        k0_loss=1.25,
        k8=_object(store, b"k8 evaluation artifact"),
        k8_loss=0.875,
        evaluation=_object(store, b"heldout evaluation artifact"),
        evaluation_loss=0.8125,
    )

    assert action_path.read_bytes() == action_bytes_before
    assert action.manifest.sha256 == hashlib.sha256(action_bytes_before).hexdigest()
    loaded = load_policy_outcome(store, outcome)
    assert loaded.action.manifest_sha256 == action.manifest.sha256
    assert loaded.k0_loss == 1.25
    assert loaded.k8_loss == 0.875
    assert loaded.evaluation_loss == 0.8125
    manifest = store.load_manifest(outcome.manifest)
    assert [row["role"] for row in manifest["objects"]] == [
        "outcome/k0",
        "outcome/k8",
        "outcome/evaluation",
    ]
    assert "outcome" not in store.load_manifest(action.manifest)["metadata"]
    store.audit()


@pytest.mark.parametrize(
    "loss", [float("nan"), float("inf"), float("-inf"), 1, True, "1.0"]
)
def test_outcome_publication_rejects_nonfinite_or_wrong_typed_losses(tmp_path, loss):
    store = CaptureObjectStore(tmp_path / str(loss).replace("/", "_"))
    action, _arguments = _publish_action(store)
    manifest_count = len(list(store.manifests_dir.iterdir()))
    with pytest.raises(PolicyContractError, match="finite JSON float"):
        publish_policy_outcome(
            store,
            "invalid-outcome",
            action=action,
            k0=_object(store, b"k0"),
            k0_loss=loss,
            k8=_object(store, b"k8"),
            k8_loss=0.9,
            evaluation=_object(store, b"evaluation"),
            evaluation_loss=0.8,
        )
    assert len(list(store.manifests_dir.iterdir())) == manifest_count


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra-field", "metadata fields are malformed"),
        ("boolean-schema", "unsupported schema"),
        ("boolean-loss", "finite JSON float"),
        ("action-cross-wire", "manifest id mismatch"),
        ("object-cross-wire", "metadata/object cross-reference mismatch"),
        ("object-order", "canonical order"),
    ],
)
def test_resealed_outcome_schema_cross_wire_and_order_fail_closed(
    tmp_path, case, message
):
    store = CaptureObjectStore(tmp_path / case)
    action, _arguments = _publish_action(store)
    outcome = publish_policy_outcome(
        store,
        "valid-outcome",
        action=action,
        k0=_object(store, b"k0"),
        k0_loss=1.0,
        k8=_object(store, b"k8"),
        k8_loss=0.9,
        evaluation=_object(store, b"evaluation"),
        evaluation_loss=0.8,
    )
    manifest = store.load_manifest(outcome.manifest)
    metadata = copy.deepcopy(manifest["metadata"])
    entries = _entries(manifest)
    if case == "extra-field":
        metadata["unexpected"] = 1
    elif case == "boolean-schema":
        metadata["schema_version"] = True
    elif case == "boolean-loss":
        metadata["k0"]["loss"] = True
    elif case == "action-cross-wire":
        metadata["action"]["manifest_id"] = "different-action-id"
    elif case == "object-cross-wire":
        metadata["k0"]["sha256"] = "f" * 64
    elif case == "object-order":
        entries = [entries[1], entries[0], entries[2]]
    else:  # pragma: no cover
        raise AssertionError(case)
    resealed = _reseal(store, outcome, metadata, f"resealed-{case}", entries)
    with pytest.raises((PolicyContractError, CaptureStoreError), match=message):
        load_policy_outcome(store, resealed)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-object", "regular non-symlink"),
        ("corrupt-object", "CAS SHA-256 mismatch"),
        ("missing-action", "missing regular manifest"),
        ("corrupt-action", "manifest SHA-256 mismatch"),
        ("corrupt-outcome", "manifest SHA-256 mismatch"),
    ],
)
def test_outcome_missing_or_corrupt_objects_and_action_fail_closed(
    tmp_path, case, message
):
    store = CaptureObjectStore(tmp_path / case)
    action, _arguments = _publish_action(store)
    k0 = _object(store, b"k0")
    outcome = publish_policy_outcome(
        store,
        "valid-outcome",
        action=action,
        k0=k0,
        k0_loss=1.0,
        k8=_object(store, b"k8"),
        k8_loss=0.9,
        evaluation=_object(store, b"evaluation"),
        evaluation_loss=0.8,
    )
    if case == "missing-object":
        store.object_path(k0.sha256).unlink()
    elif case == "corrupt-object":
        path = store.object_path(k0.sha256)
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
    elif case == "missing-action":
        store.manifest_path(action.manifest.sha256).unlink()
    elif case == "corrupt-action":
        path = store.manifest_path(action.manifest.sha256)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    elif case == "corrupt-outcome":
        path = store.manifest_path(outcome.manifest.sha256)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    else:  # pragma: no cover
        raise AssertionError(case)
    with pytest.raises(CaptureStoreError, match=message):
        load_policy_outcome(store, outcome)
