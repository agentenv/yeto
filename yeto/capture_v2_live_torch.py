"""Live PyTorch producer and exact-restore bridge for capture-v2.

The lower capture-v2 modules deliberately define immutable storage and
authority contracts without touching a live learner.  This module is the
narrow bridge from a single-process PyTorch learner to those contracts.  It
captures only state that can be restored and verified exactly; unsupported
optimizer state or non-JSON scheduler/scaler state fails closed.

The bridge does not decide *when* a boundary is causally safe and does not
advance a data iterator.  Callers must stop mutation at an admitted boundary
and provide the already materialized future-group bytes in exact consumption
order.  Publication performs all snapshots synchronously before returning.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .capture_v2_endpoint import (
    FUTURE_GROUP_COUNT,
    EndpointIdentity,
    EndpointRestoreRef,
    FutureGroupRefs,
    InputProvenance,
    LoadedFutureGroupEnvelope,
    LoadedLearnerEndpoint,
    load_future_group_envelope,
    load_learner_endpoint,
    publish_future_group_envelope,
    publish_learner_endpoint,
)
from .capture_v2_store import CaptureObjectStore, CaptureStoreError, ObjectRef
from .capture_v2_tensor_pack import TensorPackRef, publish_tensor_pack


SCHEMA = "yeto.capture-v2-live-torch"
SCHEMA_VERSION = 1
_STATE_HASH_DOMAIN = b"yeto.capture-v2-live-torch-state-v1\x00"
_SAFE_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class LiveTorchCaptureError(CaptureStoreError):
    """Live state cannot be represented or restored exactly."""


@dataclass(frozen=True)
class CapturedFutureGroup:
    """One already materialized group in exact future consumption order."""

    group_id: str
    data_iterator_position: int
    content: bytes


@dataclass(frozen=True)
class RestoredTorchEndpoint:
    """Verified restore identity and exact future groups for a fresh branch."""

    endpoint: EndpointRestoreRef
    state_sha256: str
    future_groups: tuple[LoadedFutureGroupEnvelope, ...]


def _canonical_json(value: Any, context: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LiveTorchCaptureError(
            f"{context} is not canonical JSON data: {exc}"
        ) from exc


def _json_snapshot(value: Any, context: str) -> Any:
    raw = _canonical_json(value, context)
    return json.loads(raw)


def _class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_state_key(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SAFE_KEY_RE.fullmatch(value) is None:
        raise LiveTorchCaptureError(
            f"{context} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
        )
    return value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    snapshot = value.detach().to(device="cpu", copy=True).contiguous()
    return snapshot.reshape(-1).view(torch.uint8).numpy().tobytes()


def _torch_object_bytes(value: Any, context: str) -> bytes:
    handle = io.BytesIO()
    try:
        torch.save(value, handle)
        raw = handle.getvalue()
        # ``weights_only`` prevents an endpoint object from becoming arbitrary
        # code execution authority during restore.  Verify that the exact
        # bytes we are about to publish are accepted by that restricted path.
        torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LiveTorchCaptureError(
            f"{context} cannot be represented by restricted torch serialization: {exc}"
        ) from exc
    return raw


def _load_torch_object(store: CaptureObjectStore, ref: ObjectRef, context: str) -> Any:
    path = store.verify_object(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LiveTorchCaptureError(f"cannot read {context}: {exc}") from exc
    if len(raw) != ref.bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise LiveTorchCaptureError(f"{context} changed after CAS verification")
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LiveTorchCaptureError(
            f"cannot decode restricted {context}: {exc}"
        ) from exc


def _named_parameters(
    value: Mapping[str, torch.nn.Parameter],
) -> dict[str, torch.nn.Parameter]:
    if not isinstance(value, Mapping):
        raise TypeError("trainable_parameters must be a mapping")
    result: dict[str, torch.nn.Parameter] = {}
    for name, parameter in value.items():
        if not isinstance(name, str) or not name:
            raise LiveTorchCaptureError(
                "trainable parameter names must be non-empty strings"
            )
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"trainable parameter {name!r} must be a Parameter")
        if not parameter.requires_grad:
            raise LiveTorchCaptureError(
                f"trainable parameter {name!r} does not require gradients"
            )
        if parameter.dtype != torch.float32:
            raise LiveTorchCaptureError(
                f"trainable parameter {name!r} must be fp32, got {parameter.dtype}"
            )
        if parameter.layout != torch.strided or parameter.device.type == "meta":
            raise LiveTorchCaptureError(
                f"trainable parameter {name!r} must be a materialized strided tensor"
            )
        if name in result:
            raise LiveTorchCaptureError(f"duplicate trainable parameter name {name!r}")
        result[name] = parameter
    if not result:
        raise LiveTorchCaptureError("at least one trainable parameter is required")
    return result


def _fragment_names(
    value: Mapping[int, Sequence[str]], parameters: Mapping[str, torch.nn.Parameter]
) -> dict[int, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError("fragment_parameter_names must be a mapping")
    result: dict[int, tuple[str, ...]] = {}
    observed: list[str] = []
    for fragment_id, names in value.items():
        if type(fragment_id) is not int or fragment_id < 0:
            raise LiveTorchCaptureError("fragment ids must be non-negative integers")
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
            raise TypeError(f"fragment {fragment_id} names must be a sequence")
        row = tuple(names)
        if not row:
            raise LiveTorchCaptureError(f"fragment {fragment_id} cannot be empty")
        if any(not isinstance(name, str) or name not in parameters for name in row):
            raise LiveTorchCaptureError(
                f"fragment {fragment_id} contains an unknown parameter name"
            )
        result[fragment_id] = row
        observed.extend(row)
    result = dict(sorted(result.items()))
    if list(result) != list(range(len(result))):
        raise LiveTorchCaptureError("fragment ids must be contiguous from zero")
    if len(observed) != len(set(observed)):
        raise LiveTorchCaptureError(
            "a trainable parameter appears in multiple fragments"
        )
    if set(observed) != set(parameters):
        missing = sorted(set(parameters) - set(observed))
        raise LiveTorchCaptureError(
            f"fragment layout does not cover every trainable parameter: {missing[:3]}"
        )
    return result


def _optimizer_topology(
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
) -> tuple[dict[int, str], dict[str, Any]]:
    by_identity = {id(parameter): name for name, parameter in parameters.items()}
    if len(by_identity) != len(parameters):
        raise LiveTorchCaptureError("trainable parameter mapping aliases one Parameter")
    seen: set[str] = set()
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for parameter in group.get("params", ()):
            try:
                name = by_identity[id(parameter)]
            except KeyError as exc:
                raise LiveTorchCaptureError(
                    f"optimizer group {group_index} contains an unbound parameter"
                ) from exc
            if name in seen:
                raise LiveTorchCaptureError(
                    f"optimizer parameter {name!r} appears in multiple groups"
                )
            seen.add(name)
            names.append(name)
        config = {key: item for key, item in group.items() if key != "params"}
        groups.append(
            {
                "names": names,
                "config": _json_snapshot(config, f"optimizer group {group_index}"),
            }
        )
    if seen != set(parameters):
        raise LiveTorchCaptureError("optimizer does not own every trainable parameter")
    if any(id(parameter) not in by_identity for parameter in optimizer.state):
        raise LiveTorchCaptureError(
            "optimizer contains state for a parameter outside the capture mapping"
        )
    return by_identity, {
        "class": _class_name(optimizer),
        "groups": groups,
    }


def _pack_fragment(
    store: CaptureObjectStore,
    *,
    manifest_id: str,
    fragment_id: int,
    names: Sequence[str],
    parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    optimizer_metadata: Mapping[str, Any],
    scheduler: object,
    scaler: object | None,
) -> TensorPackRef:
    optimizer_tensors: dict[str, torch.Tensor] = {}
    clocks: dict[str, int] = {}
    state_rows: list[dict[str, Any]] = []
    for name in names:
        parameter = parameters[name]
        state = optimizer.state.get(parameter, {})
        if not isinstance(state, Mapping):
            raise LiveTorchCaptureError(
                f"optimizer state for parameter {name!r} must be a mapping"
            )
        for key, item in sorted(state.items(), key=lambda pair: str(pair[0])):
            key = _safe_state_key(key, f"optimizer state key for {name!r}")
            pack_name = f"{name}/optimizer-state/{key}"
            if isinstance(item, torch.Tensor):
                optimizer_tensors[pack_name] = item
                state_rows.append(
                    {
                        "parameter": name,
                        "key": key,
                        "kind": "tensor",
                        "pack_name": pack_name,
                        "device": str(item.device),
                    }
                )
            elif type(item) is int and 0 <= item <= 2**63 - 1:
                clocks[pack_name] = item
                state_rows.append(
                    {
                        "parameter": name,
                        "key": key,
                        "kind": "clock",
                        "pack_name": pack_name,
                        "device": None,
                    }
                )
            else:
                raise LiveTorchCaptureError(
                    f"optimizer state {name!r}/{key} has unsupported type "
                    f"{type(item).__name__}; only tensors and non-negative int clocks "
                    "have exact restore authority"
                )
    metadata = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "parameters": list(names),
        "parameter_devices": {name: str(parameters[name].device) for name in names},
        "optimizer": optimizer_metadata,
        "optimizer_states": state_rows,
        "scheduler_class": _class_name(scheduler),
        "scaler_class": None if scaler is None else _class_name(scaler),
    }
    return publish_tensor_pack(
        store,
        manifest_id,
        fragment_id=fragment_id,
        trainable={name: parameters[name] for name in names},
        optimizer=optimizer_tensors,
        clocks=clocks,
        metadata=metadata,
    )


def _rng_objects(
    store: CaptureObjectStore,
) -> tuple[ObjectRef, ObjectRef, ObjectRef, dict[int, ObjectRef]]:
    python_ref = store.put_bytes(
        _torch_object_bytes(random.getstate(), "Python RNG state")
    ).ref
    numpy_state = np.random.get_state()
    numpy_value = {
        "algorithm": numpy_state[0],
        "keys": numpy_state[1].astype("<u4", copy=False).tobytes().hex(),
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian_f64_bits": np.float64(numpy_state[4]).tobytes().hex(),
    }
    numpy_ref = store.put_bytes(_canonical_json(numpy_value, "NumPy RNG state")).ref
    torch_cpu_ref = store.put_bytes(_tensor_bytes(torch.get_rng_state())).ref
    cuda_refs = {
        index: store.put_bytes(_tensor_bytes(torch.cuda.get_rng_state(index))).ref
        for index in range(torch.cuda.device_count())
    }
    return python_ref, numpy_ref, torch_cpu_ref, cuda_refs


def publish_live_torch_endpoint(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    identity: EndpointIdentity,
    input_provenance: InputProvenance,
    model: torch.nn.Module,
    trainable_parameters: Mapping[str, torch.nn.Parameter],
    fragment_parameter_names: Mapping[int, Sequence[str]],
    fragment_versions: Sequence[int],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
    future_groups: Sequence[CapturedFutureGroup],
    incomplete_future_reason: str | None = None,
) -> EndpointRestoreRef:
    """Synchronously snapshot and publish one exact live learner endpoint.

    The caller owns causal admission and must prevent concurrent mutation for
    the duration of this call.  The function never invents absent future
    groups: exactly eight yields ``complete``; fewer requires an explicit
    reason and is durably marked ``incomplete``.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if scheduler is None or not callable(getattr(scheduler, "state_dict", None)):
        raise TypeError("scheduler must provide state_dict")
    if scaler is not None and not callable(getattr(scaler, "state_dict", None)):
        raise TypeError("scaler must provide state_dict")
    parameters = _named_parameters(trainable_parameters)
    fragments = _fragment_names(fragment_parameter_names, parameters)
    _, optimizer_metadata = _optimizer_topology(optimizer, parameters)
    scheduler_state = _json_snapshot(scheduler.state_dict(), "scheduler state")
    scaler_state = (
        None if scaler is None else _json_snapshot(scaler.state_dict(), "scaler state")
    )

    packs = {
        fragment_id: _pack_fragment(
            store,
            manifest_id=f"{manifest_id}.fragment-{fragment_id}",
            fragment_id=fragment_id,
            names=names,
            parameters=parameters,
            optimizer=optimizer,
            optimizer_metadata=optimizer_metadata,
            scheduler=scheduler,
            scaler=scaler,
        )
        for fragment_id, names in fragments.items()
    }
    live_buffers = dict(sorted(model.named_buffers()))
    for name, value in live_buffers.items():
        if value.layout != torch.strided or value.device.type == "meta":
            raise LiveTorchCaptureError(
                f"model buffer {name!r} must be a materialized strided tensor"
            )
    buffers = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tensors": {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in live_buffers.items()
        },
        "devices": {name: str(value.device) for name, value in live_buffers.items()},
    }
    model_buffers = store.put_bytes(_torch_object_bytes(buffers, "model buffers")).ref
    python_rng, numpy_rng, torch_cpu_rng, torch_cuda_rng = _rng_objects(store)

    if isinstance(future_groups, (str, bytes)) or not isinstance(
        future_groups, Sequence
    ):
        raise TypeError("future_groups must be a sequence")
    groups = tuple(future_groups)
    if len(groups) > FUTURE_GROUP_COUNT:
        raise LiveTorchCaptureError("capture-v2 permits exactly eight future groups")
    refs: dict[int, ObjectRef] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, CapturedFutureGroup):
            raise TypeError(f"future group {index} must be CapturedFutureGroup")
        refs[index] = publish_future_group_envelope(
            store,
            capture_session_uuid=identity.capture_session_uuid,
            window_uuid=identity.window_uuid,
            learner_id=identity.learner_id,
            rank=identity.rank,
            group_index=index,
            group_id=group.group_id,
            data_iterator_position=group.data_iterator_position,
            content=group.content,
        )
    if len(groups) == FUTURE_GROUP_COUNT:
        if incomplete_future_reason is not None:
            raise LiveTorchCaptureError(
                "complete future groups cannot have an incomplete reason"
            )
        future = FutureGroupRefs("complete", refs)
    else:
        if (
            not isinstance(incomplete_future_reason, str)
            or not incomplete_future_reason.strip()
        ):
            raise LiveTorchCaptureError(
                "fewer than eight future groups require an explicit incomplete reason"
            )
        future = FutureGroupRefs("incomplete", refs, incomplete_future_reason)

    return publish_learner_endpoint(
        store,
        manifest_id,
        identity=identity,
        input_provenance=input_provenance,
        fragment_packs=packs,
        fragment_versions=fragment_versions,
        mode="train" if model.training else "eval",
        model_buffers=model_buffers,
        scheduler=scheduler_state,
        scaler=scaler_state,
        python_rng=python_rng,
        numpy_rng=numpy_rng,
        torch_cpu_rng=torch_cpu_rng,
        torch_cuda_rng=torch_cuda_rng,
        future_groups=future,
    )


def _load_buffers(
    store: CaptureObjectStore,
    endpoint: LoadedLearnerEndpoint,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    value = _load_torch_object(store, endpoint.model_buffers, "model buffers")
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "schema_version", "tensors", "devices"}
        or value["schema"] != SCHEMA
        or value["schema_version"] != SCHEMA_VERSION
        or not isinstance(value["tensors"], dict)
        or not isinstance(value["devices"], dict)
    ):
        raise LiveTorchCaptureError(
            "model-buffer object has malformed live-torch fields"
        )
    tensors = value["tensors"]
    devices = value["devices"]
    if any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in tensors.items()
    ) or set(devices) != set(tensors):
        raise LiveTorchCaptureError(
            "model-buffer object must contain aligned tensor/device mappings"
        )
    targets = dict(model.named_buffers())
    if set(tensors) != set(targets):
        raise LiveTorchCaptureError("model-buffer names differ from the restore target")
    for name, source in tensors.items():
        target = targets[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise LiveTorchCaptureError(
                f"model buffer {name!r} shape or dtype differs from the restore target"
            )
        if devices[name] != str(target.device):
            raise LiveTorchCaptureError(
                f"model buffer {name!r} device differs from the restore target"
            )
    return tensors


def _load_rng(
    store: CaptureObjectStore, endpoint: LoadedLearnerEndpoint
) -> tuple[Any, tuple[Any, ...], torch.Tensor, tuple[torch.Tensor, ...]]:
    python_state = _load_torch_object(store, endpoint.rng.python, "Python RNG state")
    try:
        random.Random().setstate(python_state)
    except Exception as exc:
        raise LiveTorchCaptureError(f"invalid Python RNG state: {exc}") from exc

    path = store.verify_object(endpoint.rng.numpy)
    try:
        numpy_raw = path.read_bytes()
        numpy_value = json.loads(numpy_raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveTorchCaptureError(f"invalid NumPy RNG state: {exc}") from exc
    if numpy_raw != _canonical_json(numpy_value, "NumPy RNG state") or set(
        numpy_value
    ) != {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian_f64_bits",
    }:
        raise LiveTorchCaptureError("NumPy RNG state is noncanonical or malformed")
    try:
        keys_raw = bytes.fromhex(numpy_value["keys"])
        if len(keys_raw) % 4:
            raise ValueError("key bytes are not uint32-aligned")
        keys = np.frombuffer(keys_raw, dtype="<u4").astype(np.uint32, copy=True)
        cached_raw = bytes.fromhex(numpy_value["cached_gaussian_f64_bits"])
        if len(cached_raw) != 8:
            raise ValueError("cached Gaussian is not one f64")
        cached = float(np.frombuffer(cached_raw, dtype=np.float64)[0])
        numpy_state = (
            numpy_value["algorithm"],
            keys,
            numpy_value["position"],
            numpy_value["has_gauss"],
            cached,
        )
        probe = np.random.RandomState()
        probe.set_state(numpy_state)
    except (TypeError, ValueError, KeyError) as exc:
        raise LiveTorchCaptureError(f"invalid NumPy RNG fields: {exc}") from exc

    def rng_tensor(ref: ObjectRef, context: str) -> torch.Tensor:
        rng_path = store.verify_object(ref)
        raw = rng_path.read_bytes()
        if len(raw) != ref.bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise LiveTorchCaptureError(f"{context} changed after CAS verification")
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()

    cpu = rng_tensor(endpoint.rng.torch_cpu, "Torch CPU RNG state")
    if cpu.numel() != torch.get_rng_state().numel():
        raise LiveTorchCaptureError("Torch CPU RNG state has the wrong byte count")
    if len(endpoint.rng.torch_cuda) != torch.cuda.device_count():
        raise LiveTorchCaptureError(
            "captured Torch CUDA RNG count differs from the restore process"
        )
    cuda = tuple(
        rng_tensor(ref, f"Torch CUDA RNG state {index}")
        for index, ref in enumerate(endpoint.rng.torch_cuda)
    )
    for index, value in enumerate(cuda):
        if value.numel() != torch.cuda.get_rng_state(index).numel():
            raise LiveTorchCaptureError(
                f"Torch CUDA RNG state {index} has the wrong byte count"
            )
    return python_state, numpy_state, cpu, cuda


def _metadata_from_packs(
    endpoint: LoadedLearnerEndpoint,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    common: dict[str, Any] | None = None
    states: dict[str, Any] = {}
    parameter_devices: dict[str, str] = {}
    observed_names: list[str] = []
    for fragment_id, pack in endpoint.fragments.items():
        metadata = pack.metadata
        if not isinstance(metadata, dict) or set(metadata) != {
            "schema",
            "schema_version",
            "parameters",
            "parameter_devices",
            "optimizer",
            "optimizer_states",
            "scheduler_class",
            "scaler_class",
        }:
            raise LiveTorchCaptureError(
                f"fragment {fragment_id} lacks canonical live-torch metadata"
            )
        if metadata["schema"] != SCHEMA or metadata["schema_version"] != SCHEMA_VERSION:
            raise LiveTorchCaptureError(
                f"fragment {fragment_id} has unsupported live-torch metadata"
            )
        shared = {
            "optimizer": metadata["optimizer"],
            "scheduler_class": metadata["scheduler_class"],
            "scaler_class": metadata["scaler_class"],
        }
        if common is None:
            common = shared
        elif common != shared:
            raise LiveTorchCaptureError(
                "fragment packs disagree on live restore topology"
            )
        names = metadata["parameters"]
        if (
            not isinstance(names, list)
            or len(names) != len(set(names))
            or sorted(names) != sorted(pack.trainable)
        ):
            raise LiveTorchCaptureError(
                f"fragment {fragment_id} parameter metadata is noncanonical"
            )
        observed_names.extend(names)
        devices = metadata["parameter_devices"]
        if (
            not isinstance(devices, dict)
            or set(devices) != set(names)
            or any(not isinstance(device, str) for device in devices.values())
        ):
            raise LiveTorchCaptureError(
                f"fragment {fragment_id} parameter devices are malformed"
            )
        parameter_devices.update(devices)
        rows = metadata["optimizer_states"]
        if not isinstance(rows, list):
            raise LiveTorchCaptureError("optimizer state metadata must be an array")
        referenced_tensors: set[str] = set()
        referenced_clocks: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "parameter",
                "key",
                "kind",
                "pack_name",
                "device",
            }:
                raise LiveTorchCaptureError("optimizer state row is malformed")
            name = row["parameter"]
            key = row["key"]
            pack_name = row["pack_name"]
            if name not in pack.trainable or not isinstance(key, str):
                raise LiveTorchCaptureError("optimizer state row is cross-wired")
            state_id = f"{name}\x00{key}"
            if state_id in states:
                raise LiveTorchCaptureError("optimizer state row is duplicated")
            if row["kind"] == "tensor":
                if pack_name not in pack.optimizer or not isinstance(
                    row["device"], str
                ):
                    raise LiveTorchCaptureError(
                        "optimizer tensor state row is cross-wired"
                    )
                states[state_id] = (name, key, pack.optimizer[pack_name], row["device"])
                referenced_tensors.add(pack_name)
            elif row["kind"] == "clock":
                if pack_name not in pack.clocks or row["device"] is not None:
                    raise LiveTorchCaptureError(
                        "optimizer clock state row is cross-wired"
                    )
                states[state_id] = (name, key, pack.clocks[pack_name], None)
                referenced_clocks.add(pack_name)
            else:
                raise LiveTorchCaptureError("optimizer state kind is unsupported")
        if referenced_tensors != set(pack.optimizer) or referenced_clocks != set(
            pack.clocks
        ):
            raise LiveTorchCaptureError(
                f"fragment {fragment_id} has unreferenced optimizer state"
            )
    if len(observed_names) != len(set(observed_names)):
        raise LiveTorchCaptureError("fragment packs duplicate a trainable parameter")
    if common is None:
        raise LiveTorchCaptureError("endpoint has no live-torch fragment metadata")
    return common, states, parameter_devices


def _validate_restore_topology(
    common: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
    parameters: Mapping[str, torch.nn.Parameter],
) -> None:
    _, current = _optimizer_topology(optimizer, parameters)
    captured_optimizer = common["optimizer"]
    if (
        not isinstance(captured_optimizer, dict)
        or set(captured_optimizer) != {"class", "groups"}
        or captured_optimizer["class"] != current["class"]
        or not isinstance(captured_optimizer["groups"], list)
        or len(captured_optimizer["groups"]) != len(current["groups"])
    ):
        raise LiveTorchCaptureError(
            "optimizer class or group count differs from capture"
        )
    for index, (captured_group, current_group) in enumerate(
        zip(captured_optimizer["groups"], current["groups"], strict=True)
    ):
        if (
            not isinstance(captured_group, dict)
            or set(captured_group) != {"names", "config"}
            or captured_group["names"] != current_group["names"]
            or not isinstance(captured_group["config"], dict)
            or set(captured_group["config"]) != set(current_group["config"])
        ):
            raise LiveTorchCaptureError(
                f"optimizer group {index} parameters or config keys differ from capture"
            )
    if common["scheduler_class"] != _class_name(scheduler):
        raise LiveTorchCaptureError("scheduler class differs from capture")
    scaler_class = None if scaler is None else _class_name(scaler)
    if common["scaler_class"] != scaler_class:
        raise LiveTorchCaptureError("scaler presence or class differs from capture")


def _frame(digest: Any, label: str, raw: bytes) -> None:
    label_raw = label.encode("utf-8")
    digest.update(len(label_raw).to_bytes(8, "big"))
    digest.update(label_raw)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def hash_live_torch_state(
    *,
    authority_sha256: str,
    model: torch.nn.Module,
    trainable_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
) -> str:
    """Hash all mutable branch state represented by a live endpoint.

    The endpoint authority digest binds immutable frozen model/input state;
    trainable parameters, buffers, optimizer, scheduler/scaler, mode, and all
    RNG streams are hashed from the live branch.  The result is stable under
    repeated read-only calls and changes after a represented state mutation.
    """

    if (
        not isinstance(authority_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
    ):
        raise LiveTorchCaptureError(
            "authority_sha256 must be a lowercase SHA-256 digest"
        )
    parameters = _named_parameters(trainable_parameters)
    digest = hashlib.sha256(_STATE_HASH_DOMAIN)
    _frame(digest, "authority", authority_sha256.encode())
    for name, tensor in sorted(parameters.items()):
        descriptor = _canonical_json(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            f"parameter {name!r} descriptor",
        )
        _frame(digest, f"parameter/{name}/descriptor", descriptor)
        _frame(digest, f"parameter/{name}/bytes", _tensor_bytes(tensor))
    for name, tensor in sorted(model.named_buffers()):
        descriptor = _canonical_json(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            f"buffer {name!r} descriptor",
        )
        _frame(digest, f"buffer/{name}/descriptor", descriptor)
        _frame(digest, f"buffer/{name}/bytes", _tensor_bytes(tensor))
    _, topology = _optimizer_topology(optimizer, parameters)
    _frame(
        digest, "optimizer/topology", _canonical_json(topology, "optimizer topology")
    )
    for name, parameter in sorted(parameters.items()):
        state = optimizer.state.get(parameter, {})
        for key, value in sorted(state.items(), key=lambda pair: str(pair[0])):
            key = _safe_state_key(key, f"optimizer state key for {name!r}")
            if isinstance(value, torch.Tensor):
                _frame(
                    digest,
                    f"optimizer/{name}/{key}/descriptor",
                    _canonical_json(
                        {
                            "device": str(value.device),
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                        },
                        "optimizer tensor descriptor",
                    ),
                )
                _frame(digest, f"optimizer/{name}/{key}/bytes", _tensor_bytes(value))
            elif type(value) is int and 0 <= value <= 2**63 - 1:
                _frame(
                    digest,
                    f"optimizer/{name}/{key}/clock",
                    str(value).encode(),
                )
            else:
                raise LiveTorchCaptureError(
                    f"optimizer state {name!r}/{key} is not hashable by capture-v2"
                )
    _frame(
        digest, "scheduler", _canonical_json(scheduler.state_dict(), "scheduler state")
    )
    _frame(
        digest,
        "scaler",
        _canonical_json(
            None if scaler is None else scaler.state_dict(), "scaler state"
        ),
    )
    _frame(digest, "mode", b"train" if model.training else b"eval")
    _frame(digest, "rng/python", _canonical_json(random.getstate(), "Python RNG state"))
    numpy_state = np.random.get_state()
    _frame(
        digest,
        "rng/numpy",
        _canonical_json(
            {
                "algorithm": numpy_state[0],
                "keys": numpy_state[1].astype("<u4", copy=False).tobytes().hex(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian_f64_bits": np.float64(numpy_state[4]).tobytes().hex(),
            },
            "NumPy RNG state",
        ),
    )
    _frame(digest, "rng/torch-cpu", _tensor_bytes(torch.get_rng_state()))
    for index in range(torch.cuda.device_count()):
        _frame(
            digest,
            f"rng/torch-cuda/{index}",
            _tensor_bytes(torch.cuda.get_rng_state(index)),
        )
    return digest.hexdigest()


def restore_live_torch_endpoint(
    store: CaptureObjectStore,
    endpoint_ref: EndpointRestoreRef,
    *,
    model: torch.nn.Module,
    trainable_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
) -> RestoredTorchEndpoint:
    """Restore one verified endpoint into a fresh compatible PyTorch branch.

    Every object, topology, name, shape, dtype, class, CUDA RNG count, and
    future-group envelope is validated before live state is mutated.  A
    failure raises and yields no receipt; the caller must discard that fresh
    branch.
    """

    if not isinstance(endpoint_ref, EndpointRestoreRef):
        raise TypeError("endpoint_ref must be EndpointRestoreRef")
    endpoint = load_learner_endpoint(store, endpoint_ref)
    parameters = _named_parameters(trainable_parameters)
    common, optimizer_states, parameter_devices = _metadata_from_packs(endpoint)
    captured_names = {
        name for pack in endpoint.fragments.values() for name in pack.trainable
    }
    if captured_names != set(parameters):
        raise LiveTorchCaptureError(
            "captured trainable parameter names differ from the restore target"
        )
    for pack in endpoint.fragments.values():
        for name, source in pack.trainable.items():
            target = parameters[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise LiveTorchCaptureError(
                    f"trainable parameter {name!r} shape or dtype differs from capture"
                )
            if parameter_devices[name] != str(target.device):
                raise LiveTorchCaptureError(
                    f"trainable parameter {name!r} device differs from capture"
                )
    _validate_restore_topology(common, optimizer, scheduler, scaler, parameters)
    buffers = _load_buffers(store, endpoint, model)
    python_rng, numpy_rng, torch_cpu_rng, torch_cuda_rng = _load_rng(store, endpoint)
    future = tuple(
        load_future_group_envelope(store, endpoint.future_groups.refs[index])
        for index in sorted(endpoint.future_groups.refs)
    )
    prepared_optimizer_states: list[tuple[str, str, Any]] = []
    for name, key, value, device in optimizer_states.values():
        if device is None:
            prepared_optimizer_states.append((name, key, value))
            continue
        try:
            target_device = torch.device(device)
            if target_device.type == "meta":
                raise ValueError("meta state is not restorable")
            prepared = value.to(target_device, copy=True)
        except Exception as exc:
            raise LiveTorchCaptureError(
                f"optimizer state {name!r}/{key} has an unavailable device: {exc}"
            ) from exc
        prepared_optimizer_states.append((name, key, prepared))

    # All fallible authority and topology validation is complete.  The caller
    # supplied a fresh branch; any unexpected device/application failure below
    # invalidates that branch and produces no restore receipt.
    try:
        with torch.no_grad():
            for pack in endpoint.fragments.values():
                for name, source in pack.trainable.items():
                    parameters[name].copy_(source, non_blocking=False)
            targets = dict(model.named_buffers())
            for name, source in buffers.items():
                targets[name].copy_(source, non_blocking=False)
        optimizer.state.clear()
        captured_groups = common["optimizer"]["groups"]
        for group, captured_group in zip(
            optimizer.param_groups, captured_groups, strict=True
        ):
            for key, value in captured_group["config"].items():
                group[key] = value
        for name, key, value in prepared_optimizer_states:
            state = optimizer.state.setdefault(parameters[name], {})
            state[key] = value
        scheduler.load_state_dict(endpoint.scheduler)
        if scaler is not None:
            if endpoint.scaler is None:
                raise LiveTorchCaptureError("captured scaler state is absent")
            scaler.load_state_dict(endpoint.scaler)
        model.train(endpoint.mode == "train")
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_cpu_rng)
        for index, value in enumerate(torch_cuda_rng):
            torch.cuda.set_rng_state(value, index)
    except Exception as exc:
        raise LiveTorchCaptureError(
            f"validated endpoint could not be applied to fresh branch: {exc}"
        ) from exc

    # Loading APIs may legally accept subsets or normalize values.  Capture-v2
    # restore authority is stricter: the represented live state must now equal
    # every verified source byte/value, not merely have been accepted.
    for pack in endpoint.fragments.values():
        for name, source in pack.trainable.items():
            if _tensor_bytes(parameters[name]) != _tensor_bytes(source):
                raise LiveTorchCaptureError(
                    f"trainable parameter {name!r} differs after restore"
                )
    for name, source in buffers.items():
        if _tensor_bytes(dict(model.named_buffers())[name]) != _tensor_bytes(source):
            raise LiveTorchCaptureError(f"model buffer {name!r} differs after restore")
    _, restored_topology = _optimizer_topology(optimizer, parameters)
    if restored_topology != common["optimizer"]:
        raise LiveTorchCaptureError("optimizer topology differs after restore")
    for name, key, expected in prepared_optimizer_states:
        actual = optimizer.state.get(parameters[name], {}).get(key)
        if isinstance(expected, torch.Tensor):
            if (
                not isinstance(actual, torch.Tensor)
                or actual.device != expected.device
                or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or _tensor_bytes(actual) != _tensor_bytes(expected)
            ):
                raise LiveTorchCaptureError(
                    f"optimizer state {name!r}/{key} differs after restore"
                )
        elif actual != expected or type(actual) is not type(expected):
            raise LiveTorchCaptureError(
                f"optimizer state {name!r}/{key} differs after restore"
            )
    if (
        _json_snapshot(scheduler.state_dict(), "restored scheduler state")
        != endpoint.scheduler
    ):
        raise LiveTorchCaptureError("scheduler state differs after restore")
    restored_scaler = None if scaler is None else scaler.state_dict()
    if _json_snapshot(restored_scaler, "restored scaler state") != endpoint.scaler:
        raise LiveTorchCaptureError("scaler state differs after restore")
    if model.training != (endpoint.mode == "train"):
        raise LiveTorchCaptureError("model mode differs after restore")
    if random.getstate() != python_rng:
        raise LiveTorchCaptureError("Python RNG state differs after restore")
    actual_numpy = np.random.get_state()
    if (
        actual_numpy[0] != numpy_rng[0]
        or not np.array_equal(actual_numpy[1], numpy_rng[1])
        or actual_numpy[2:] != numpy_rng[2:]
    ):
        raise LiveTorchCaptureError("NumPy RNG state differs after restore")
    if not torch.equal(torch.get_rng_state(), torch_cpu_rng):
        raise LiveTorchCaptureError("Torch CPU RNG state differs after restore")
    for index, expected in enumerate(torch_cuda_rng):
        if not torch.equal(torch.cuda.get_rng_state(index), expected):
            raise LiveTorchCaptureError(
                f"Torch CUDA RNG state {index} differs after restore"
            )

    state_sha256 = hash_live_torch_state(
        authority_sha256=endpoint_ref.manifest.sha256,
        model=model,
        trainable_parameters=parameters,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    return RestoredTorchEndpoint(endpoint_ref, state_sha256, future)
