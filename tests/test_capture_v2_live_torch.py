from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from yeto.capture_v2_endpoint import (
    EndpointIdentity,
    InputProvenance,
    load_learner_endpoint,
)
from yeto.capture_v2_live_torch import (
    CapturedFutureGroup,
    LiveTorchCaptureError,
    hash_live_torch_state,
    publish_live_torch_endpoint,
    restore_live_torch_endpoint,
)
from yeto.capture_v2_store import CaptureObjectStore


SESSION = "12345678-1234-5678-9234-567812345678"
WINDOW = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


class _ToyLearner(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(
            torch.tensor([offset + 1.0, offset - 2.0], dtype=torch.float32)
        )
        self.right = torch.nn.Parameter(
            torch.tensor([[offset + 0.5], [offset - 0.25]], dtype=torch.float32)
        )
        self.register_buffer(
            "running", torch.tensor([offset + 3.0], dtype=torch.float64)
        )
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))

    def forward(self) -> torch.Tensor:
        return self.left.square().sum() + self.right.square().sum()


def _branch(offset: float):
    model = _ToyLearner(offset)
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameters["left"]], "lr": 0.03},
            {"params": [parameters["right"]], "lr": 0.01},
        ],
        betas=(0.8, 0.91),
        weight_decay=0.02,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 3)
    )
    return model, parameters, optimizer, scheduler


def _advance(model, optimizer, scheduler) -> None:
    optimizer.zero_grad(set_to_none=True)
    model().backward()
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        model.running.add_(0.125)
        model.counter.add_(1)


def _identity() -> EndpointIdentity:
    return EndpointIdentity(
        capture_session_uuid=SESSION,
        learner_id=2,
        rank=0,
        local_step=17,
        active_fragment_id=1,
        window_uuid=WINDOW,
    )


def _provenance(store: CaptureObjectStore) -> InputProvenance:
    return InputProvenance(
        object=store.put_bytes(b"sealed input provenance").ref,
        source_commit="a" * 40,
        image_id="gcp:yeto-a100-v1",
        model_sha256="b" * 64,
        data_sha256="c" * 64,
        config_sha256="d" * 64,
    )


def _future(count: int = 8) -> tuple[CapturedFutureGroup, ...]:
    return tuple(
        CapturedFutureGroup(
            group_id=f"actual-batch-{index}",
            data_iterator_position=500 + index,
            content=f"exact materialized group {index}".encode(),
        )
        for index in range(count)
    )


def _publish(store, model, parameters, optimizer, scheduler, **changes):
    arguments = {
        "identity": _identity(),
        "input_provenance": _provenance(store),
        "model": model,
        "trainable_parameters": parameters,
        # Deliberately not ASCII-sorted: fragment membership/order comes from
        # the production layout while tensor-pack bytes remain canonical.
        "fragment_parameter_names": {0: ("right",), 1: ("left",)},
        "fragment_versions": (11, 12),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": None,
        "future_groups": _future(),
    }
    arguments.update(changes)
    return publish_live_torch_endpoint(store, "live-endpoint-17", **arguments)


def test_live_endpoint_round_trip_restores_all_mutable_state_and_rng(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)
    source_model, source_parameters, source_optimizer, source_scheduler = _branch(0.0)
    _advance(source_model, source_optimizer, source_scheduler)
    source_model.eval()
    endpoint = _publish(
        store,
        source_model,
        source_parameters,
        source_optimizer,
        source_scheduler,
    )
    source_hash = hash_live_torch_state(
        authority_sha256=endpoint.manifest.sha256,
        model=source_model,
        trainable_parameters=source_parameters,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        scaler=None,
    )
    expected_python = random.getstate()
    expected_numpy = copy.deepcopy(np.random.get_state())
    expected_torch = torch.get_rng_state().clone()

    restored_model, restored_parameters, restored_optimizer, restored_scheduler = (
        _branch(90.0)
    )
    _advance(restored_model, restored_optimizer, restored_scheduler)
    restored_model.train()
    random.seed(180)
    np.random.seed(181)
    torch.manual_seed(182)
    receipt = restore_live_torch_endpoint(
        store,
        endpoint,
        model=restored_model,
        trainable_parameters=restored_parameters,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
    )

    assert receipt.endpoint == endpoint
    assert receipt.state_sha256 == source_hash
    assert [group.group_index for group in receipt.future_groups] == list(range(8))
    assert [group.data_iterator_position for group in receipt.future_groups] == list(
        range(500, 508)
    )
    assert [group.content for group in receipt.future_groups] == [
        f"exact materialized group {index}".encode() for index in range(8)
    ]
    assert restored_model.training is False
    assert (
        dict(restored_model.named_buffers()).keys()
        == dict(source_model.named_buffers()).keys()
    )
    for name in source_parameters:
        assert torch.equal(source_parameters[name], restored_parameters[name])
        source_state = source_optimizer.state[source_parameters[name]]
        restored_state = restored_optimizer.state[restored_parameters[name]]
        assert source_state.keys() == restored_state.keys()
        for key in source_state:
            assert torch.equal(source_state[key], restored_state[key])
            assert source_state[key].device == restored_state[key].device
    for name, source in source_model.named_buffers():
        assert torch.equal(source, dict(restored_model.named_buffers())[name])
    assert (
        source_optimizer.param_groups[0]["lr"]
        == restored_optimizer.param_groups[0]["lr"]
    )
    assert (
        source_optimizer.param_groups[1]["lr"]
        == restored_optimizer.param_groups[1]["lr"]
    )
    assert source_scheduler.state_dict() == restored_scheduler.state_dict()
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.get_rng_state(), expected_torch)

    assert (
        hash_live_torch_state(
            authority_sha256=endpoint.manifest.sha256,
            model=restored_model,
            trainable_parameters=restored_parameters,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=None,
        )
        == receipt.state_sha256
    )
    with torch.no_grad():
        restored_parameters["left"].add_(1.0)
    assert (
        hash_live_torch_state(
            authority_sha256=endpoint.manifest.sha256,
            model=restored_model,
            trainable_parameters=restored_parameters,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=None,
        )
        != receipt.state_sha256
    )


def test_live_endpoint_marks_missing_future_groups_incomplete_without_fake_refs(
    tmp_path,
):
    store = CaptureObjectStore(tmp_path / "cas")
    model, parameters, optimizer, scheduler = _branch(0.0)
    endpoint = _publish(
        store,
        model,
        parameters,
        optimizer,
        scheduler,
        future_groups=_future(3),
        incomplete_future_reason="only three actual groups were materialized",
    )
    loaded = load_learner_endpoint(store, endpoint)

    assert loaded.future_groups.state == "incomplete"
    assert list(loaded.future_groups.refs) == [0, 1, 2]
    assert loaded.future_groups.reason == "only three actual groups were materialized"


def test_live_endpoint_refuses_to_label_incomplete_future_as_complete(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    model, parameters, optimizer, scheduler = _branch(0.0)

    with pytest.raises(LiveTorchCaptureError, match="explicit incomplete reason"):
        _publish(
            store,
            model,
            parameters,
            optimizer,
            scheduler,
            future_groups=_future(7),
        )


def test_unsupported_optimizer_state_fails_closed_before_endpoint_publication(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    model, parameters, optimizer, scheduler = _branch(0.0)
    optimizer.state[parameters["left"]]["unsupported"] = 1.25

    with pytest.raises(LiveTorchCaptureError, match="unsupported type float"):
        _publish(store, model, parameters, optimizer, scheduler)
    # An immutable, unreachable pack for the already validated fragment may
    # exist, but no endpoint authority is published around partial state.
    assert all(
        b"yeto.capture-v2-learner-endpoint" not in path.read_bytes()
        for path in store.manifests_dir.glob("*.json")
    )


def test_restore_rejects_cross_wired_optimizer_topology_before_mutation(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    model, parameters, optimizer, scheduler = _branch(0.0)
    _advance(model, optimizer, scheduler)
    endpoint = _publish(store, model, parameters, optimizer, scheduler)

    target, target_parameters, target_optimizer, target_scheduler = _branch(50.0)
    # Swap parameter ownership between the two groups without changing the
    # optimizer class or number of groups.
    target_optimizer.param_groups[0]["params"] = [target_parameters["right"]]
    target_optimizer.param_groups[1]["params"] = [target_parameters["left"]]
    before = {name: value.detach().clone() for name, value in target_parameters.items()}

    with pytest.raises(LiveTorchCaptureError, match="parameters or config keys"):
        restore_live_torch_endpoint(
            store,
            endpoint,
            model=target,
            trainable_parameters=target_parameters,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            scaler=None,
        )
    for name, value in target_parameters.items():
        assert torch.equal(value, before[name])
