"""Adversarial authority tests for the capture-v2/PTI adapter."""

from __future__ import annotations

import hashlib
import math
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
    load_sealed_outer_action,
    publish_policy_definition,
    publish_sealed_outer_action,
)
from yeto.capture_v2_pti_adapter import (
    CaptureV2PTIPolicyState,
    IDENTIFIED,
    PTIAdapterError,
    UNIDENTIFIABLE,
    load_authoritative_pti_action,
    process_authoritative_boundary,
)
from yeto.capture_v2_store import CaptureObjectStore
from yeto.capture_v2_syncer import (
    BoundaryConfig,
    FlatF32FragmentFormat,
    ResponderEndpointRef,
    SyncerBoundaryError,
    SyncerBoundaryIdentity,
    memoryless_outer_update_f32le,
    publish_syncer_boundary,
)
from yeto.capture_v2_tensor_pack import publish_tensor_pack
from yeto.pti_sgd import PTIEvent, encode_f32le


SESSION = "12345678-1234-5678-9234-567812345678"
OTHER_SESSION = "87654321-4321-4765-8234-567812345678"
LR_BITS = struct.pack(">d", 0.28).hex()
LAYOUT_SHA = hashlib.sha256(b"model.weight:f32:2").hexdigest()


def _object(store, raw: bytes):
    return store.put_bytes(raw).ref


def _endpoint(store, *, session: str, pre_version: int, suffix: str):
    pack = publish_tensor_pack(
        store,
        f"pack-{suffix}",
        fragment_id=0,
        trainable={"model.weight": torch.tensor([1.0, -2.0])},
        optimizer={"model.weight/exp_avg": torch.tensor([0.0, 0.0])},
        clocks={"optimizer_steps": pre_version},
        metadata={"suffix": suffix},
    )
    return publish_learner_endpoint(
        store,
        f"endpoint-{suffix}",
        identity=EndpointIdentity(
            capture_session_uuid=session,
            learner_id=0,
            rank=0,
            local_step=pre_version,
            active_fragment_id=0,
            window_uuid=f"00000000-0000-4000-8000-{pre_version + 1:012d}",
        ),
        input_provenance=InputProvenance(
            object=_object(store, f"provenance-{suffix}".encode()),
            source_commit="a" * 40,
            image_id="capture-v2-test",
            model_sha256="b" * 64,
            data_sha256="c" * 64,
            config_sha256="d" * 64,
        ),
        fragment_packs={0: pack},
        fragment_versions=[pre_version],
        mode="train",
        model_buffers=_object(store, f"buffers-{suffix}".encode()),
        scheduler={"last_epoch": pre_version},
        scaler=None,
        python_rng=_object(store, f"py-{suffix}".encode()),
        numpy_rng=_object(store, f"np-{suffix}".encode()),
        torch_cpu_rng=_object(store, f"cpu-{suffix}".encode()),
        torch_cuda_rng={0: _object(store, f"cuda-{suffix}".encode())},
        future_groups=FutureGroupRefs("incomplete", {}, "not required by PTI"),
    )


def _boundary(
    store,
    *,
    sequence: int,
    pre_version: int,
    stock_values: tuple[float, float],
    session: str = SESSION,
    pre_values: tuple[float, float] = (1.0, -2.0),
    merge_config=BoundaryConfig("rda", {"weighted": True}),
    lr_bits: str = LR_BITS,
    momentum_bits: str = "0000000000000000",
    layout_sha: str = LAYOUT_SHA,
    fragment_numel: int = 2,
    captured_pre_values: tuple[float, float] | None = None,
    captured_stock_values: tuple[float, float] | None = None,
    wrong_post: bool = False,
    wrong_broadcast: bool = False,
    suffix: str | None = None,
):
    suffix = suffix or f"{session[:8]}-{sequence}-{pre_version}-{stock_values[0]}"
    endpoint = _endpoint(
        store, session=session, pre_version=pre_version, suffix=suffix
    )
    pre_raw = encode_f32le(pre_values)
    stock_raw = encode_f32le(stock_values)
    factual_stock = encode_f32le(captured_stock_values or stock_values)
    factual_pre = encode_f32le(captured_pre_values or pre_values)
    post_raw = memoryless_outer_update_f32le(factual_pre, factual_stock, LR_BITS)
    if wrong_post:
        post_raw = encode_f32le((9.0, 9.0))
    broadcast_raw = encode_f32le((8.0, 8.0)) if wrong_broadcast else post_raw
    return publish_syncer_boundary(
        store,
        f"boundary-{suffix}",
        identity=SyncerBoundaryIdentity(
            capture_session_uuid=session,
            commit_id=f"commit-{sequence}-{suffix}",
            commit_seq=sequence,
            fragment_id=0,
            pre_fragment_version=pre_version,
            post_fragment_version=pre_version + 1,
        ),
        responders=[
            ResponderEndpointRef(
                endpoint=endpoint,
                weight_f64_bits=struct.pack(">d", 128.0).hex(),
                payload=_object(store, f"payload-{suffix}".encode()),
            )
        ],
        fragment_format=FlatF32FragmentFormat(fragment_numel, layout_sha),
        pre_fragment=_object(store, pre_raw),
        stock_pseudo_gradient=_object(store, stock_raw),
        post_fragment=_object(store, post_raw),
        outer_state=_object(store, b""),
        broadcast=_object(store, broadcast_raw),
        merge_config=merge_config,
        outer_config=BoundaryConfig(
            "nesterov",
            {"lr_f64_bits": lr_bits, "momentum_f64_bits": momentum_bits},
        ),
    )


def _policy(store, *, capabilities=CAPABILITIES, suffix="main"):
    source = _object(store, f"pti source {suffix}".encode())
    config = _object(store, f"pti config {suffix}".encode())
    return publish_policy_definition(
        store,
        f"pti-policy-{suffix}",
        policy_id=f"pti-{suffix}",
        policy_version="1.0.0",
        source_commit="e" * 40,
        source=source,
        config=config,
        capabilities=capabilities,
    )


def test_authoritative_boundary_derives_exact_stock_fallback_and_proof(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    state = CaptureV2PTIPolicyState()
    policy = _policy(store)
    boundary = _boundary(
        store, sequence=7, pre_version=40, stock_values=(1.0, 0.0)
    )

    result = process_authoritative_boundary(
        store,
        state=state,
        policy=policy,
        boundary=boundary,
        action_manifest_id="action-7",
    )

    assert result.status == IDENTIFIED
    assert result.reason == "warmup"
    loaded = load_authoritative_pti_action(store, result.action)
    assert loaded.action.action_kind == "stock_fallback"
    assert loaded.action.stock_pseudo_gradient == loaded.action.selected_pseudo_gradient
    assert loaded.action.resulting_fragment == loaded.action.boundary.post_fragment
    assert result.pti_result.action.raw == store.object_path(
        loaded.action.stock_pseudo_gradient.sha256
    ).read_bytes()


def test_causal_history_opens_nonstock_and_computes_f32_result(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    state = CaptureV2PTIPolicyState()
    policy = _policy(store)
    results = []
    for sequence, degrees in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        radians = math.radians(degrees)
        boundary = _boundary(
            store,
            sequence=sequence,
            pre_version=sequence,
            stock_values=(math.cos(radians), math.sin(radians)),
            suffix=f"seq-{sequence}",
        )
        results.append(
            process_authoritative_boundary(
                store,
                state=state,
                policy=policy,
                boundary=boundary,
                action_manifest_id=f"action-{sequence}",
            )
        )

    final = results[-1]
    assert [result.status for result in results] == [IDENTIFIED] * 5
    assert final.pti_result.action.used_nonstock is True
    loaded = load_authoritative_pti_action(store, final.action).action
    assert loaded.action_kind == "nonstock"
    assert loaded.selected_pseudo_gradient != loaded.stock_pseudo_gradient
    expected = memoryless_outer_update_f32le(
        store.object_path(loaded.boundary.pre_fragment.sha256).read_bytes(),
        final.pti_result.action.raw,
        LR_BITS,
    )
    assert store.object_path(loaded.resulting_fragment.sha256).read_bytes() == expected


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"merge_config": BoundaryConfig("avg", {"weighted": True})}, "merge"),
        ({"lr_bits": struct.pack(">d", 0.2800001).hex()}, "outer_lr"),
        ({"momentum_bits": struct.pack(">d", 0.9).hex()}, "outer_momentum"),
        ({"wrong_post": True}, "post_mismatch"),
        ({"wrong_broadcast": True}, "broadcast_mismatch"),
        ({"captured_stock_values": (0.0, 1.0)}, "post_mismatch"),
        ({"captured_pre_values": (4.0, -3.0)}, "post_mismatch"),
    ],
)
def test_unsupported_or_false_boundary_abstains_without_pti_action(
    tmp_path, overrides, reason
):
    store = CaptureObjectStore(tmp_path / "cas")
    state = CaptureV2PTIPolicyState()
    result = process_authoritative_boundary(
        store,
        state=state,
        policy=_policy(store),
        boundary=_boundary(
            store,
            sequence=0,
            pre_version=0,
            stock_values=(1.0, 0.0),
            **overrides,
        ),
        action_manifest_id="must-not-exist",
    )

    assert result.status == UNIDENTIFIABLE
    assert reason in result.reason
    assert result.action is None
    assert state.ledger == ()
    assert state.capture_session_uuid is None


def test_missing_capability_and_session_sequence_layout_relabels_abstain(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    missing = tuple(name for name in CAPABILITIES if name != "same_fragment_history")
    first_state = CaptureV2PTIPolicyState()
    missing_result = process_authoritative_boundary(
        store,
        state=first_state,
        policy=_policy(store, capabilities=missing, suffix="missing"),
        boundary=_boundary(
            store, sequence=0, pre_version=0, stock_values=(1.0, 0.0)
        ),
        action_manifest_id="missing-action",
    )
    assert missing_result.status == UNIDENTIFIABLE
    assert "same_fragment_history" in missing_result.reason

    state = CaptureV2PTIPolicyState()
    policy = _policy(store, suffix="continuity")
    valid = process_authoritative_boundary(
        store,
        state=state,
        policy=policy,
        boundary=_boundary(
            store,
            sequence=3,
            pre_version=10,
            stock_values=(1.0, 0.0),
            suffix="valid",
        ),
        action_manifest_id="valid-action",
    )
    assert valid.status == IDENTIFIED
    head = state.ledger_head
    cases = (
        _boundary(
            store,
            sequence=4,
            pre_version=11,
            stock_values=(1.0, 0.1),
            session=OTHER_SESSION,
            suffix="other-session",
        ),
        _boundary(
            store,
            sequence=5,
            pre_version=11,
            stock_values=(1.0, 0.1),
            suffix="sequence-gap",
        ),
        _boundary(
            store,
            sequence=4,
            pre_version=11,
            stock_values=(1.0, 0.1),
            layout_sha="a" * 64,
            suffix="layout-change",
        ),
    )
    for index, boundary in enumerate(cases):
        abstained = process_authoritative_boundary(
            store,
            state=state,
            policy=policy,
            boundary=boundary,
            action_manifest_id=f"abstain-{index}",
        )
        assert abstained.status == UNIDENTIFIABLE
        assert abstained.action is None
        assert state.ledger_head == head


def test_api_rejects_caller_event_injection_and_boundary_relabels(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    kwargs = {
        "store": store,
        "state": CaptureV2PTIPolicyState(),
        "policy": _policy(store),
        "boundary": _boundary(
            store, sequence=0, pre_version=0, stock_values=(1.0, 0.0)
        ),
        "action_manifest_id": "action",
    }
    injected = PTIEvent(99, 99, 99, (2,), "0" * 64, b"12345678")
    with pytest.raises(TypeError, match="unexpected keyword"):
        process_authoritative_boundary(**kwargs, event=injected)
    with pytest.raises(TypeError, match="unexpected keyword"):
        process_authoritative_boundary(**kwargs, sequence=99)


def test_decision_crosswire_is_rejected_even_when_generic_action_is_valid(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    policy = _policy(store)
    boundary_a = _boundary(
        store,
        sequence=0,
        pre_version=0,
        stock_values=(1.0, 0.0),
        suffix="a",
    )
    boundary_b = _boundary(
        store,
        sequence=10,
        pre_version=10,
        stock_values=(0.0, 1.0),
        suffix="b",
    )
    action_a = process_authoritative_boundary(
        store,
        state=CaptureV2PTIPolicyState(),
        policy=policy,
        boundary=boundary_a,
        action_manifest_id="action-a",
    )
    action_b = process_authoritative_boundary(
        store,
        state=CaptureV2PTIPolicyState(),
        policy=policy,
        boundary=boundary_b,
        action_manifest_id="action-b",
    )
    loaded_a = load_sealed_outer_action(store, action_a.action)
    crosswired = publish_sealed_outer_action(
        store,
        "crosswired",
        policy=loaded_a.policy_ref,
        boundary=loaded_a.boundary_ref,
        fragment_id=loaded_a.fragment_id,
        required_capabilities=loaded_a.required_capabilities,
        stock_pseudo_gradient=loaded_a.stock_pseudo_gradient,
        selected_pseudo_gradient=loaded_a.selected_pseudo_gradient,
        outer_lr_f64_bits=loaded_a.outer_lr_f64_bits,
        resulting_fragment=loaded_a.resulting_fragment,
        decision=action_b.decision,
        config_sha256=loaded_a.config_sha256,
        action_kind=loaded_a.action_kind,
        action_reason=loaded_a.action_reason,
        fallback_reason=loaded_a.fallback_reason,
    )
    load_sealed_outer_action(store, crosswired)
    with pytest.raises(PTIAdapterError, match="cross-wired"):
        load_authoritative_pti_action(store, crosswired)


def test_fragment_numel_and_byte_lengths_are_authoritative(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    with pytest.raises(SyncerBoundaryError, match=r"fragment_numel \* 4"):
        _boundary(
            store,
            sequence=0,
            pre_version=0,
            stock_values=(1.0, 0.0),
            fragment_numel=3,
        )
