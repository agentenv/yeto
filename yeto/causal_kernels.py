"""Correctness-gated causal-LM attention and training-kernel policies.

The native path is always available and remains the default. Optional CUDA
kernels are selected explicitly, pinned to versions exercised by the A100
benchmark, and rejected before model loading when their contract cannot be
met. This module deliberately contains no silent fallback: an explicit
request either resolves exactly or raises an actionable error.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MethodType

import torch

from .kernel_deps import (
    FLASH_ATTN_VERSION,
    LIGER_KERNEL_VERSION,
    LIGER_QWEN2_SOURCE_SHA256,
    PEFT_VERSION,
)

ATTENTION_BACKENDS = ("auto", "sdpa", "flash-attn-2")
KERNEL_BACKENDS = ("native", "liger")

NATIVE_LAYER_BACKEND = "transformers-native"
NATIVE_LOSS_IMPLEMENTATION = "torch-cross-entropy"
FUSED_LINEAR_CE_IMPLEMENTATION = "liger-fused-linear-cross-entropy"

_HF_ATTENTION_NAMES = {
    "sdpa": "sdpa",
    "flash-attn-2": "flash_attention_2",
}
_DISPLAY_ATTENTION_NAMES = {value: key for key, value in _HF_ATTENTION_NAMES.items()}
_PACKAGE_INSTALL_HINTS = {
    "liger-kernel": "pip install -e '.[a100-liger]'",
    "peft": "pip install -e '.[a100-liger]'",
    "flash-attn": "the pinned --no-build-isolation command in docs/A100_KERNELS.md",
}

_TENSOR_DIGEST_CHUNK_BYTES = 4 * 1024 * 1024
_STRUCTURAL_CONTAINER_TYPES = (dict, list, tuple, set, frozenset, bytearray)


class KernelIsolationError(RuntimeError):
    """A rejected direct forward binding and its verified rollback status.

    ``process_state_poisoned`` is true when the pre-binding model/process state
    could not be reproduced exactly without retaining a second copy of model
    tensors. Every instance of this exception is fatal for in-process
    continuation because it is created only after the binding transaction has
    begun. ``rollback_complete`` remains evidence about the observed state, not
    permission to run another model or benchmark arm in the same process.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_invariants: list[str],
        rollback_report: dict,
    ) -> None:
        super().__init__(message)
        self.failed_invariants = tuple(failed_invariants)
        self.rollback_report = rollback_report
        self.rollback_complete = bool(rollback_report["complete"])
        self.process_state_poisoned = not self.rollback_complete
        self.fatal = True


class _IsolationViolation(Exception):
    """Internal control flow carrying a contract failure to one rollback site."""

    def __init__(
        self,
        failed_invariants: list[str],
        cause: Exception | None = None,
    ) -> None:
        super().__init__(str(failed_invariants))
        self.failed_invariants = failed_invariants
        self.cause = cause


@dataclass(frozen=True)
class _ContainerFingerprint:
    name: str
    object_id: int
    object_type: str
    sha256: str


@dataclass(frozen=True)
class _NamespaceSnapshot:
    bindings: dict[str, object]
    containers: tuple[_ContainerFingerprint, ...]
    custom_objects: tuple[_ContainerFingerprint, ...]


@dataclass(frozen=True)
class _ModuleSnapshot:
    name: str
    module: torch.nn.Module
    module_type: type
    namespace: _NamespaceSnapshot


@dataclass(frozen=True)
class _ClassSnapshot:
    module_class: type
    namespace: _NamespaceSnapshot


@dataclass(frozen=True)
class _MutableAttributeSnapshot:
    name: str
    value: object
    object_id: int
    object_type: str
    structural_sha256: str


@dataclass(frozen=True)
class _TensorSnapshot:
    category: str
    name: str
    tensor: torch.Tensor
    object_id: int
    object_type: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    stride: tuple[int, ...]
    storage_offset: int
    data_ptr: int
    storage_nbytes: int
    requires_grad: bool
    version: int
    is_contiguous: bool
    content_sha256: str
    python_namespace: _NamespaceSnapshot
    hook_state: tuple[_MutableAttributeSnapshot, ...]
    gradient: _TensorSnapshot | None


@dataclass(frozen=True)
class _ConfigSnapshot:
    path: str
    value: object
    object_id: int
    object_type: str
    structural_sha256: str


@dataclass(frozen=True)
class _ModelSnapshot:
    modules: tuple[_ModuleSnapshot, ...]
    module_classes: tuple[_ClassSnapshot, ...]
    parameters: tuple[_TensorSnapshot, ...]
    buffers: tuple[_TensorSnapshot, ...]
    config_like_objects: tuple[_ConfigSnapshot, ...]


@dataclass(frozen=True)
class _ProcessSnapshot:
    qwen2_module: _NamespaceSnapshot
    qwen2_class: _NamespaceSnapshot
    functional_cross_entropy: object
    cpu_rng_state: torch.Tensor
    cuda_initialized: bool
    cuda_rng_states: tuple[torch.Tensor, ...]
    backend_flags: tuple[tuple[str, object], ...]


def _qualified_type(value_or_type) -> str:
    value_type = value_or_type if isinstance(value_or_type, type) else type(value_or_type)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _hash_field(digest, tag: str, payload: bytes | memoryview = b"") -> None:
    encoded_tag = tag.encode("utf-8")
    digest.update(len(encoded_tag).to_bytes(4, "big"))
    digest.update(encoded_tag)
    payload_size = payload.nbytes if isinstance(payload, memoryview) else len(payload)
    digest.update(payload_size.to_bytes(8, "big"))
    digest.update(payload)


def _structural_update(
    digest,
    value,
    *,
    leaf_identity: bool,
    seen: dict[int, int],
    traverse_object_namespaces: bool = False,
) -> None:
    """Hash built-in container structure without calling user equality methods."""
    if value is None:
        _hash_field(digest, "none")
        return
    if isinstance(value, bool):
        _hash_field(digest, "bool", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _hash_field(digest, "int", str(value).encode())
        return
    if isinstance(value, float):
        _hash_field(digest, "float", value.hex().encode())
        return
    if isinstance(value, complex):
        _hash_field(
            digest,
            "complex",
            f"{value.real.hex()}:{value.imag.hex()}".encode(),
        )
        return
    if isinstance(value, str):
        _hash_field(digest, "str", value.encode("utf-8", errors="surrogatepass"))
        return
    if isinstance(value, bytes):
        _hash_field(digest, "bytes", value)
        return
    if isinstance(value, bytearray):
        if leaf_identity:
            _hash_field(digest, "bytearray-object-id", str(id(value)).encode())
        _hash_field(digest, f"bytearray:{_qualified_type(value)}", bytes(value))
        return
    if isinstance(value, torch.Tensor):
        # Registered parameters and buffers receive a streaming content digest
        # separately. Container structure records their identity/metadata so a
        # registry or alias change is still visible without hashing every large
        # tensor once per module namespace that references it.
        payload = (
            f"{_qualified_type(value)}:{id(value)}:{tuple(value.shape)}:"
            f"{value.dtype}:{value.device}:{value.layout}:"
            f"{getattr(value, '_version', 'unavailable')}"
        )
        _hash_field(digest, "tensor-reference", payload.encode())
        return

    if isinstance(value, _STRUCTURAL_CONTAINER_TYPES):
        object_id = id(value)
        if object_id in seen:
            _hash_field(digest, "container-reference", str(seen[object_id]).encode())
            return
        seen[object_id] = len(seen)
        _hash_field(digest, "container-type", _qualified_type(value).encode())
        if leaf_identity:
            _hash_field(digest, "container-object-id", str(object_id).encode())
        _hash_field(digest, "container-length", str(len(value)).encode())
        if isinstance(value, dict):
            for key, item in value.items():
                _structural_update(
                    digest,
                    key,
                    leaf_identity=leaf_identity,
                    seen=seen,
                    traverse_object_namespaces=traverse_object_namespaces,
                )
                _structural_update(
                    digest,
                    item,
                    leaf_identity=leaf_identity,
                    seen=seen,
                    traverse_object_namespaces=traverse_object_namespaces,
                )
        elif isinstance(value, (set, frozenset)):
            item_digests = []
            for item in value:
                item_digest = hashlib.sha256()
                _structural_update(
                    item_digest,
                    item,
                    leaf_identity=leaf_identity,
                    seen=dict(seen),
                    traverse_object_namespaces=traverse_object_namespaces,
                )
                item_digests.append(item_digest.digest())
            for item_digest in sorted(item_digests):
                _hash_field(digest, "set-item", item_digest)
        else:
            for item in value:
                _structural_update(
                    digest,
                    item,
                    leaf_identity=leaf_identity,
                    seen=seen,
                    traverse_object_namespaces=traverse_object_namespaces,
                )
        return

    if (
        traverse_object_namespaces
        and not isinstance(value, (torch.nn.Module, type))
        and not inspect.ismodule(value)
    ):
        try:
            object_namespace = vars(value)
        except TypeError:
            object_namespace = None
        if isinstance(object_namespace, dict):
            object_id = id(value)
            if object_id in seen:
                _hash_field(
                    digest,
                    "object-reference",
                    str(seen[object_id]).encode(),
                )
                return
            seen[object_id] = len(seen)
            _hash_field(digest, "object-type", _qualified_type(value).encode())
            if leaf_identity:
                _hash_field(digest, "object-id", str(object_id).encode())
            _structural_update(
                digest,
                object_namespace,
                leaf_identity=leaf_identity,
                seen=seen,
                traverse_object_namespaces=True,
            )
            return

    leaf_type = _qualified_type(value)
    if leaf_identity:
        payload = f"{leaf_type}:{id(value)}"
    else:
        try:
            representation = repr(value)
        except Exception as exc:  # pragma: no cover - defensive fail-closed path
            raise RuntimeError(
                f"could not structurally fingerprint config leaf {leaf_type}"
            ) from exc
        payload = f"{leaf_type}:{representation}"
    _hash_field(digest, "leaf", payload.encode("utf-8", errors="backslashreplace"))


def _structural_sha256(
    value,
    *,
    leaf_identity: bool,
    traverse_object_namespaces: bool = False,
) -> str:
    digest = hashlib.sha256()
    _structural_update(
        digest,
        value,
        leaf_identity=leaf_identity,
        seen={},
        traverse_object_namespaces=traverse_object_namespaces,
    )
    return digest.hexdigest()


def _container_fingerprints(namespace: dict[str, object]) -> tuple[_ContainerFingerprint, ...]:
    fingerprints = []
    for name, value in namespace.items():
        # This is the interpreter's enormous shared builtins table, not Qwen2
        # state. Binding identity is still checked with the rest of the module
        # namespace; recursively hashing it would add noise and startup cost.
        if name == "__builtins__":
            continue
        if isinstance(value, _STRUCTURAL_CONTAINER_TYPES):
            fingerprints.append(
                _ContainerFingerprint(
                    name=name,
                    object_id=id(value),
                    object_type=_qualified_type(value),
                    sha256=_structural_sha256(
                        value,
                        leaf_identity=True,
                        traverse_object_namespaces=True,
                    ),
                )
            )
    return tuple(fingerprints)


def _custom_object_fingerprints(
    namespace: dict[str, object],
) -> tuple[_ContainerFingerprint, ...]:
    fingerprints = []
    for name, value in namespace.items():
        if isinstance(value, (_STRUCTURAL_CONTAINER_TYPES, torch.Tensor, torch.nn.Module, type)):
            continue
        if inspect.ismodule(value):
            continue
        try:
            object_namespace = vars(value)
        except TypeError:
            continue
        if not isinstance(object_namespace, dict):
            continue
        fingerprints.append(
            _ContainerFingerprint(
                name=name,
                object_id=id(value),
                object_type=_qualified_type(value),
                sha256=_structural_sha256(
                    value,
                    leaf_identity=True,
                    traverse_object_namespaces=True,
                ),
            )
        )
    return tuple(fingerprints)


def _capture_namespace(namespace: dict[str, object]) -> _NamespaceSnapshot:
    bindings = dict(namespace)
    return _NamespaceSnapshot(
        bindings=bindings,
        containers=_container_fingerprints(bindings),
        custom_objects=_custom_object_fingerprints(bindings),
    )


def _namespace_is_identical(
    before: _NamespaceSnapshot,
    after: _NamespaceSnapshot,
    *,
    allowed_added: frozenset[str] = frozenset(),
) -> bool:
    expected_keys = before.bindings.keys() | allowed_added
    return (
        after.bindings.keys() == expected_keys
        and all(
            after.bindings[name] is value
            for name, value in before.bindings.items()
        )
    )


def _namespace_containers_are_identical(
    before: _NamespaceSnapshot,
    after: _NamespaceSnapshot,
    *,
    allowed_added: frozenset[str] = frozenset(),
) -> bool:
    actual = tuple(
        item for item in after.containers if item.name not in allowed_added
    )
    return before.containers == actual


def _namespace_custom_objects_are_identical(
    before: _NamespaceSnapshot,
    after: _NamespaceSnapshot,
    *,
    allowed_added: frozenset[str] = frozenset(),
) -> bool:
    actual = tuple(
        item for item in after.custom_objects if item.name not in allowed_added
    )
    return before.custom_objects == actual


def _iter_tensor_digest_chunks(tensor: torch.Tensor, max_elements: int):
    """Yield logical-order tensor slices bounded by ``max_elements``."""
    stack = [tensor.detach()]
    while stack:
        current = stack.pop()
        if current.numel() <= max_elements or current.ndim == 0:
            yield current
            continue
        split_dimension = next(
            (dimension for dimension, size in enumerate(current.shape) if size > 1),
            None,
        )
        if split_dimension is None:  # pragma: no cover - impossible for numel > 1
            yield current
            continue
        dimension_size = current.shape[split_dimension]
        elements_per_index = current.numel() // dimension_size
        indices_per_chunk = max(1, max_elements // max(1, elements_per_index))
        slices = [
            current.narrow(
                split_dimension,
                start,
                min(indices_per_chunk, dimension_size - start),
            )
            for start in range(0, dimension_size, indices_per_chunk)
        ]
        stack.extend(reversed(slices))


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    if tensor.device.type == "meta":
        raise RuntimeError("cannot attest tensor contents on the meta device")
    if tensor.layout is not torch.strided:
        raise RuntimeError(
            f"cannot attest unsupported tensor layout {tensor.layout}; "
            "the isolated Qwen2 lane requires strided parameters and buffers"
        )
    max_elements = max(1, _TENSOR_DIGEST_CHUNK_BYTES // max(1, tensor.element_size()))
    digest = hashlib.sha256()
    _hash_field(digest, "element-count", str(tensor.numel()).encode())
    for chunk in _iter_tensor_digest_chunks(tensor, max_elements):
        cpu_chunk = chunk.contiguous().to(device="cpu")
        byte_view = cpu_chunk.reshape(-1).view(torch.uint8).numpy()
        _hash_field(digest, "tensor-chunk", memoryview(byte_view))
    return digest.hexdigest()


def _capture_tensor_hook_state(
    tensor: torch.Tensor,
) -> tuple[_MutableAttributeSnapshot, ...]:
    snapshots = []
    for name in ("_backward_hooks", "_post_accumulate_grad_hooks"):
        value = getattr(tensor, name, None)
        snapshots.append(
            _MutableAttributeSnapshot(
                name=name,
                value=value,
                object_id=id(value),
                object_type=_qualified_type(value),
                structural_sha256=_structural_sha256(
                    value,
                    leaf_identity=True,
                    traverse_object_namespaces=True,
                ),
            )
        )
    return tuple(snapshots)


def _capture_tensor(
    category: str,
    name: str,
    tensor: torch.Tensor,
    content_cache: dict[int, str],
    *,
    capture_gradient: bool = False,
) -> _TensorSnapshot:
    try:
        version = int(tensor._version)
    except Exception as exc:
        raise RuntimeError(f"could not read mutation version for {category} {name!r}") from exc
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    content_sha256 = content_cache.get(id(tensor))
    if content_sha256 is None:
        content_sha256 = _tensor_content_sha256(tensor)
        content_cache[id(tensor)] = content_sha256
    gradient = None
    if capture_gradient:
        actual_gradient = tensor.grad
        if actual_gradient is not None:
            gradient = _capture_tensor(
                "gradient",
                f"{name}.grad",
                actual_gradient,
                content_cache,
            )
    return _TensorSnapshot(
        category=category,
        name=name,
        tensor=tensor,
        object_id=id(tensor),
        object_type=_qualified_type(tensor),
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        layout=str(tensor.layout),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        data_ptr=int(tensor.data_ptr()),
        storage_nbytes=storage_nbytes,
        requires_grad=bool(tensor.requires_grad),
        version=version,
        is_contiguous=bool(tensor.is_contiguous()),
        content_sha256=content_sha256,
        python_namespace=_capture_namespace(vars(tensor)),
        hook_state=_capture_tensor_hook_state(tensor),
        gradient=gradient,
    )


def _named_tensors(model, method_name: str):
    method = getattr(model, method_name)
    try:
        return tuple(method(recurse=True, remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        return tuple(method(recurse=True))


def _is_config_like(name: str, value: object) -> bool:
    if value is None:
        return False
    if name == "config" or name.endswith("_config"):
        return True
    try:
        return callable(getattr(value, "to_dict", None))
    except Exception as exc:
        raise RuntimeError(
            f"could not inspect possible config-like attribute {name!r}"
        ) from exc


def _capture_config(path: str, config: object) -> _ConfigSnapshot:
    try:
        to_dict = getattr(config, "to_dict", None)
    except Exception as exc:
        raise RuntimeError(
            f"could not inspect config-like object at {path!r}"
        ) from exc
    try:
        structural_value = to_dict() if callable(to_dict) else dict(vars(config))
        structural_sha256 = _structural_sha256(
            structural_value,
            leaf_identity=False,
            traverse_object_namespaces=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not structurally fingerprint config-like object at {path!r}"
        ) from exc
    return _ConfigSnapshot(
        path=path,
        value=config,
        object_id=id(config),
        object_type=_qualified_type(config),
        structural_sha256=structural_sha256,
    )


def _capture_config_like_objects(
    modules: tuple[_ModuleSnapshot, ...],
) -> tuple[_ConfigSnapshot, ...]:
    snapshots = []
    for module in modules:
        for name, value in module.namespace.bindings.items():
            if not _is_config_like(name, value):
                continue
            path = f"{module.name}.{name}" if module.name else name
            snapshots.append(_capture_config(path, value))
    if not any(item.path == "config" for item in snapshots):
        raise RuntimeError("the isolated fused-linear-CE model exposes no root config")
    return tuple(snapshots)


def _capture_model_state(model) -> _ModelSnapshot:
    modules = tuple(model.named_modules())
    module_snapshots = tuple(
        _ModuleSnapshot(
            name=name,
            module=module,
            module_type=type(module),
            namespace=_capture_namespace(vars(module)),
        )
        for name, module in modules
    )
    module_classes = []
    seen_classes = set()
    for _name, module in modules:
        module_class = type(module)
        if id(module_class) in seen_classes:
            continue
        seen_classes.add(id(module_class))
        module_classes.append(
            _ClassSnapshot(
                module_class=module_class,
                namespace=_capture_namespace(dict(vars(module_class))),
            )
        )
    content_cache: dict[int, str] = {}
    parameters = tuple(
        _capture_tensor(
            "parameter",
            name,
            tensor,
            content_cache,
            capture_gradient=True,
        )
        for name, tensor in _named_tensors(model, "named_parameters")
    )
    buffers = tuple(
        _capture_tensor("buffer", name, tensor, content_cache)
        for name, tensor in _named_tensors(model, "named_buffers")
    )
    return _ModelSnapshot(
        modules=module_snapshots,
        module_classes=tuple(module_classes),
        parameters=parameters,
        buffers=buffers,
        config_like_objects=_capture_config_like_objects(module_snapshots),
    )


def _capture_backend_flags() -> tuple[tuple[str, object], ...]:
    return (
        (
            "deterministic_algorithms",
            bool(torch.are_deterministic_algorithms_enabled()),
        ),
        (
            "deterministic_algorithms_warn_only",
            bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        ),
        (
            "fill_uninitialized_memory",
            bool(torch.utils.deterministic.fill_uninitialized_memory),
        ),
        ("float32_matmul_precision", torch.get_float32_matmul_precision()),
        ("cudnn_enabled", bool(torch.backends.cudnn.enabled)),
        ("cudnn_benchmark", bool(torch.backends.cudnn.benchmark)),
        ("cudnn_deterministic", bool(torch.backends.cudnn.deterministic)),
        ("cudnn_allow_tf32", bool(torch.backends.cudnn.allow_tf32)),
        ("cuda_matmul_allow_tf32", bool(torch.backends.cuda.matmul.allow_tf32)),
        (
            "cuda_matmul_allow_fp16_reduced_precision_reduction",
            bool(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
        ),
        (
            "cuda_matmul_allow_bf16_reduced_precision_reduction",
            bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        ),
        ("flash_sdp_enabled", bool(torch.backends.cuda.flash_sdp_enabled())),
        (
            "mem_efficient_sdp_enabled",
            bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        ),
        ("math_sdp_enabled", bool(torch.backends.cuda.math_sdp_enabled())),
        ("cudnn_sdp_enabled", bool(torch.backends.cuda.cudnn_sdp_enabled())),
    )


def _capture_process_state(modeling_qwen2, expected_class) -> _ProcessSnapshot:
    cuda_initialized = bool(torch.cuda.is_initialized())
    cuda_rng_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if cuda_initialized
        else ()
    )
    return _ProcessSnapshot(
        qwen2_module=_capture_namespace(vars(modeling_qwen2)),
        qwen2_class=_capture_namespace(dict(vars(expected_class))),
        functional_cross_entropy=torch.nn.functional.cross_entropy,
        cpu_rng_state=torch.get_rng_state().clone(),
        cuda_initialized=cuda_initialized,
        cuda_rng_states=cuda_rng_states,
        backend_flags=_capture_backend_flags(),
    )


def _require_a100(device) -> None:
    try:
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:
        raise RuntimeError("could not verify the CUDA device for the A100 kernel lane") from exc
    if "A100" not in name or capability != (8, 0):
        raise RuntimeError(
            f"the optimized kernel lane is scoped to A100/SM80; found "
            f"{name!r} with capability {capability}"
        )


def _require_exact_package(distribution: str, import_name: str, required: str) -> None:
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        hint = _PACKAGE_INSTALL_HINTS.get(distribution, f"install {distribution}=={required}")
        raise RuntimeError(
            f"{distribution}=={required} is required; use {hint} on the A100 node"
        ) from exc
    if installed != required:
        raise RuntimeError(
            f"{distribution}=={required} is required, but version {installed} is installed"
        )
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(
            f"{distribution}=={required} is installed but {import_name!r} is not importable"
        )


def attention_load_kwargs(requested: str, device, dtype: torch.dtype) -> dict[str, str]:
    """Map a public attention selection to Hugging Face load kwargs."""
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(
            f"unknown attention backend {requested!r}; choose from {ATTENTION_BACKENDS}"
        )
    if requested == "auto":
        return {}
    if requested == "flash-attn-2":
        if device.type != "cuda":
            raise RuntimeError("--attention-backend flash-attn-2 requires CUDA")
        if dtype is not torch.bfloat16:
            raise RuntimeError(
                "--attention-backend flash-attn-2 requires BF16 in the A100 lane; "
                "use --attention-backend sdpa for FP32"
            )
        _require_a100(device)
        _require_exact_package("flash-attn", "flash_attn", FLASH_ATTN_VERSION)
    return {"attn_implementation": _HF_ATTENTION_NAMES[requested]}


def resolved_attention_backend(model, requested: str) -> str:
    """Read and verify the attention implementation selected by Transformers."""
    config = getattr(model, "config", None)
    resolved = getattr(config, "_attn_implementation", None)
    if resolved is None:
        resolved = getattr(config, "attn_implementation", None)
    if isinstance(resolved, dict):
        normalized = {
            key: _DISPLAY_ATTENTION_NAMES.get(value, value or "unknown")
            for key, value in resolved.items()
        }
        resolved_values = set(normalized.values())
        display = str(normalized)
    else:
        display = _DISPLAY_ATTENTION_NAMES.get(resolved, resolved or "unknown")
        resolved_values = {display}
    if requested != "auto" and resolved_values != {requested}:
        raise RuntimeError(
            f"requested attention backend {requested!r}, but the loaded model resolved {display!r}"
        )
    return str(display)


def validate_kernel_request(
    kernel_backend: str,
    loss_function: str,
    device,
    dtype: torch.dtype,
    base_quantization: str = "none",
    tuning: str | None = None,
    shard: str | None = None,
) -> None:
    """Reject unsupported optimized-kernel combinations before model loading."""
    if kernel_backend not in KERNEL_BACKENDS:
        raise ValueError(f"unknown kernel backend {kernel_backend!r}; choose from {KERNEL_BACKENDS}")
    if kernel_backend == "native":
        return
    if device.type != "cuda":
        raise RuntimeError("--kernel-backend liger fused-linear-CE requires CUDA")
    _require_a100(device)
    if dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("the A100 Liger fused-linear-CE lane supports BF16 and FP32 only")
    if loss_function != "cross_entropy":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE supports only the built-in "
            "cross_entropy loss; "
            "use --kernel-backend native for custom, pickled, or RL losses"
        )
    if base_quantization != "none":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE does not support a quantized base; "
            "use --base-quantization none or --kernel-backend native"
        )
    if tuning != "lora":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE is production-approved only "
            "for --tuning lora; use --kernel-backend native for full tuning"
        )
    if shard != "ddp":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE is production-approved only "
            "for --shard ddp; use --kernel-backend native until FSDP has separate "
            "CUDA parity evidence"
        )
    _require_exact_package("liger-kernel", "liger_kernel", LIGER_KERNEL_VERSION)
    _require_exact_package("peft", "peft", PEFT_VERSION)


def validate_lora_production_envelope(model) -> dict:
    """Require the shared LoRA state contract used by production and evidence."""
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("the production LoRA model exposes no output embedding head")

    adapted_names = [
        name for name, _parameter in output_head.named_parameters()
        if "lora_" in name
    ]
    adapted_modules = [
        name or "<root>"
        for name, module in output_head.named_modules()
        if {
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
        }
        & vars(module).keys()
    ]
    trainable_head_names = [
        name
        for name, parameter in output_head.named_parameters()
        if parameter.requires_grad
    ]
    if adapted_names or adapted_modules or trainable_head_names:
        raise RuntimeError(
            "the production LoRA profile requires a frozen, unadapted lm_head; "
            f"adapted_parameters={adapted_names[:5]} "
            f"adapted_modules={adapted_modules[:5]} "
            f"trainable={trainable_head_names[:5]}"
        )

    trainable_dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        dtype_name = str(parameter.dtype).removeprefix("torch.")
        trainable_dtype_counts[dtype_name] = (
            trainable_dtype_counts.get(dtype_name, 0) + parameter.numel()
        )
    if set(trainable_dtype_counts) != {"float32"}:
        raise RuntimeError(
            "the production LoRA profile requires FP32 trainable adapters; "
            f"found {trainable_dtype_counts}"
        )

    return {
        "output_head": {
            "frozen": True,
            "adapted": False,
            "parameter_count": sum(
                parameter.numel() for parameter in output_head.parameters()
            ),
        },
        "trainable_dtype_counts": trainable_dtype_counts,
    }


def _installed_qwen2_source_sha256(lce_forward) -> str:
    source_file = inspect.getsourcefile(lce_forward)
    if source_file is None:
        raise RuntimeError(
            "the pinned fused Qwen2 forward has no inspectable source file"
        )
    try:
        source = Path(source_file).read_bytes()
    except Exception as exc:
        raise RuntimeError(
            "could not read the installed fused Qwen2 source module"
        ) from exc
    return hashlib.sha256(source).hexdigest()


def _qwen2_fused_linear_ce_forward_function():
    """Return the fused Qwen2 forward after source and ABI verification."""
    try:
        from liger_kernel.transformers import functional as liger_functional
        from liger_kernel.transformers.model.qwen2 import lce_forward
    except Exception as exc:
        raise RuntimeError(
            "could not import Liger's hash-locked fused Qwen2 forward"
        ) from exc

    forward_identity = (
        getattr(lce_forward, "__module__", None),
        getattr(lce_forward, "__name__", None),
        getattr(lce_forward, "__qualname__", None),
    )
    expected_identity = (
        "liger_kernel.transformers.model.qwen2",
        "lce_forward",
        "lce_forward",
    )
    if forward_identity != expected_identity:
        raise RuntimeError(
            "the fused Qwen2 forward identity does not match the approved "
            f"module/name/qualname {expected_identity}; found {forward_identity}"
        )

    actual_source_sha256 = _installed_qwen2_source_sha256(lce_forward)
    if actual_source_sha256 != LIGER_QWEN2_SOURCE_SHA256:
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 source SHA-256 does not match "
            f"the approved {LIGER_QWEN2_SOURCE_SHA256}; found "
            f"{actual_source_sha256}"
        )

    try:
        forward_signature = inspect.signature(lce_forward)
        fused_signature = inspect.signature(
            liger_functional.liger_fused_linear_cross_entropy
        )
    except Exception as exc:
        raise RuntimeError(
            "could not inspect the hash-locked fused Qwen2 loss contract"
        ) from exc

    parameters = forward_signature.parameters
    first_parameter = next(iter(parameters.values()), None)
    if (
        first_parameter is None
        or first_parameter.name != "self"
        or first_parameter.kind
        is not inspect.Parameter.POSITIONAL_OR_KEYWORD
    ):
        raise RuntimeError(
            "the hash-locked fused Qwen2 forward has no ordinary self parameter"
        )
    for name in ("labels", "use_cache", "skip_logits"):
        parameter = parameters.get(name)
        if (
            parameter is None
            or parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            or parameter.default is not None
        ):
            raise RuntimeError(
                "the hash-locked fused Qwen2 forward does not expose the exact "
                f"optional {name!r} keyword contract"
            )
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        raise RuntimeError(
            "the hash-locked fused Qwen2 forward cannot pass loss keywords"
        )

    accum_dtype = fused_signature.parameters.get("accum_dtype")
    if accum_dtype is None or accum_dtype.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION}'s fused-linear-CE primitive does not "
            "explicitly support accum_dtype; refusing to let FP32 accumulation "
            "be silently filtered"
        )
    return lce_forward


def require_liger_model_support(config) -> str:
    """Require the one model family approved for isolated fused linear CE."""

    model_type = getattr(config, "model_type", None)
    if model_type != "qwen2":
        raise RuntimeError(
            "the isolated Liger fused-linear-CE lane supports only Hugging Face "
            f"Qwen2/Qwen2.5 (model_type='qwen2'); found {model_type!r}"
        )
    _qwen2_fused_linear_ce_forward_function()
    return str(model_type)


def _tensor_registries_are_identical(
    before: tuple[_TensorSnapshot, ...], after: tuple[_TensorSnapshot, ...]
) -> bool:
    return len(before) == len(after) and all(
        actual.category == expected.category
        and actual.name == expected.name
        and actual.tensor is expected.tensor
        and actual.object_type == expected.object_type
        for expected, actual in zip(before, after, strict=True)
    )


def _tensor_metadata(snapshot: tuple[_TensorSnapshot, ...]) -> tuple:
    return tuple(
        (
            item.category,
            item.name,
            item.shape,
            item.dtype,
            item.device,
            item.layout,
            item.stride,
            item.storage_offset,
            item.data_ptr,
            item.storage_nbytes,
            item.requires_grad,
            item.is_contiguous,
        )
        for item in snapshot
    )


def _tensor_versions(snapshot: tuple[_TensorSnapshot, ...]) -> tuple:
    return tuple((item.category, item.name, item.version) for item in snapshot)


def _tensor_contents(snapshot: tuple[_TensorSnapshot, ...]) -> tuple:
    return tuple(
        (item.category, item.name, item.content_sha256) for item in snapshot
    )


def _tensor_python_namespaces_are_identical(
    before: tuple[_TensorSnapshot, ...],
    after: tuple[_TensorSnapshot, ...],
) -> bool:
    return len(before) == len(after) and all(
        actual.category == expected.category
        and actual.name == expected.name
        and _namespace_is_identical(
            expected.python_namespace,
            actual.python_namespace,
        )
        and _namespace_containers_are_identical(
            expected.python_namespace,
            actual.python_namespace,
        )
        and _namespace_custom_objects_are_identical(
            expected.python_namespace,
            actual.python_namespace,
        )
        for expected, actual in zip(before, after, strict=True)
    )


def _mutable_attribute_snapshots_are_identical(
    before: tuple[_MutableAttributeSnapshot, ...],
    after: tuple[_MutableAttributeSnapshot, ...],
) -> bool:
    return len(before) == len(after) and all(
        actual.name == expected.name
        and actual.value is expected.value
        and actual.object_id == expected.object_id
        and actual.object_type == expected.object_type
        and actual.structural_sha256 == expected.structural_sha256
        for expected, actual in zip(before, after, strict=True)
    )


def _tensor_hook_states_are_identical(
    before: tuple[_TensorSnapshot, ...],
    after: tuple[_TensorSnapshot, ...],
) -> bool:
    return len(before) == len(after) and all(
        actual.category == expected.category
        and actual.name == expected.name
        and _mutable_attribute_snapshots_are_identical(
            expected.hook_state,
            actual.hook_state,
        )
        for expected, actual in zip(before, after, strict=True)
    )


def _parameter_gradient_presence(
    snapshot: tuple[_TensorSnapshot, ...],
) -> tuple[tuple[str, bool], ...]:
    return tuple((item.name, item.gradient is not None) for item in snapshot)


def _parameter_gradient_bindings_are_identical(
    before: tuple[_TensorSnapshot, ...],
    after: tuple[_TensorSnapshot, ...],
) -> bool:
    if len(before) != len(after):
        return False
    for expected_parameter, actual_parameter in zip(before, after, strict=True):
        expected = expected_parameter.gradient
        actual = actual_parameter.gradient
        if expected is None or actual is None:
            if expected is not actual:
                return False
            continue
        if (
            actual.tensor is not expected.tensor
            or actual.object_type != expected.object_type
        ):
            return False
    return True


def _parameter_gradients(
    snapshot: tuple[_TensorSnapshot, ...],
) -> tuple[_TensorSnapshot, ...]:
    return tuple(
        item.gradient for item in snapshot if item.gradient is not None
    )


def _config_like_objects_are_identical(
    before: tuple[_ConfigSnapshot, ...],
    after: tuple[_ConfigSnapshot, ...],
) -> bool:
    return len(before) == len(after) and all(
        actual.path == expected.path
        and actual.value is expected.value
        and actual.object_id == expected.object_id
        and actual.object_type == expected.object_type
        and actual.structural_sha256 == expected.structural_sha256
        for expected, actual in zip(before, after, strict=True)
    )


def _model_state_invariants(
    before: _ModelSnapshot,
    after: _ModelSnapshot,
    *,
    allow_forward_addition: bool,
) -> dict[str, bool]:
    expected_layout = tuple(
        (item.name, id(item.module)) for item in before.modules
    )
    actual_layout = tuple((item.name, id(item.module)) for item in after.modules)
    module_layout_unchanged = actual_layout == expected_layout
    after_modules = {id(item.module): item for item in after.modules}
    module_types_unchanged = all(
        id(item.module) in after_modules
        and after_modules[id(item.module)].module_type is item.module_type
        for item in before.modules
    )

    root_before = before.modules[0]
    root_after = after_modules.get(id(root_before.module))
    allowed_added = frozenset({"forward"}) if allow_forward_addition else frozenset()
    instance_bindings_unchanged = root_after is not None and _namespace_is_identical(
        root_before.namespace,
        root_after.namespace,
        allowed_added=allowed_added,
    )
    instance_containers_unchanged = (
        root_after is not None
        and _namespace_containers_are_identical(
            root_before.namespace,
            root_after.namespace,
            allowed_added=allowed_added,
        )
    )
    nested_bindings_unchanged = all(
        id(item.module) in after_modules
        and _namespace_is_identical(
            item.namespace,
            after_modules[id(item.module)].namespace,
        )
        for item in before.modules[1:]
    )
    nested_containers_unchanged = all(
        id(item.module) in after_modules
        and _namespace_containers_are_identical(
            item.namespace,
            after_modules[id(item.module)].namespace,
        )
        for item in before.modules[1:]
    )
    instance_custom_objects_unchanged = (
        root_after is not None
        and _namespace_custom_objects_are_identical(
            root_before.namespace,
            root_after.namespace,
            allowed_added=allowed_added,
        )
    )
    nested_custom_objects_unchanged = all(
        id(item.module) in after_modules
        and _namespace_custom_objects_are_identical(
            item.namespace,
            after_modules[id(item.module)].namespace,
        )
        for item in before.modules[1:]
    )

    expected_classes = tuple(id(item.module_class) for item in before.module_classes)
    actual_classes = tuple(id(item.module_class) for item in after.module_classes)
    module_class_layout_unchanged = actual_classes == expected_classes
    after_classes = {
        id(item.module_class): item for item in after.module_classes
    }
    module_class_bindings_unchanged = all(
        id(item.module_class) in after_classes
        and _namespace_is_identical(
            item.namespace,
            after_classes[id(item.module_class)].namespace,
        )
        for item in before.module_classes
    )
    module_class_containers_unchanged = all(
        id(item.module_class) in after_classes
        and _namespace_containers_are_identical(
            item.namespace,
            after_classes[id(item.module_class)].namespace,
        )
        for item in before.module_classes
    )
    module_class_custom_objects_unchanged = all(
        id(item.module_class) in after_classes
        and _namespace_custom_objects_are_identical(
            item.namespace,
            after_classes[id(item.module_class)].namespace,
        )
        for item in before.module_classes
    )

    before_gradients = _parameter_gradients(before.parameters)
    after_gradients = _parameter_gradients(after.parameters)
    return {
        "module_layout_unchanged": module_layout_unchanged,
        "module_types_unchanged": module_types_unchanged,
        "instance_bindings_unchanged": instance_bindings_unchanged,
        "instance_container_state_unchanged": instance_containers_unchanged,
        "instance_custom_object_state_unchanged": (
            instance_custom_objects_unchanged
        ),
        "nested_module_bindings_unchanged": nested_bindings_unchanged,
        "nested_container_state_unchanged": nested_containers_unchanged,
        "nested_custom_object_state_unchanged": (
            nested_custom_objects_unchanged
        ),
        "module_class_layout_unchanged": module_class_layout_unchanged,
        "module_class_bindings_unchanged": module_class_bindings_unchanged,
        "module_class_container_state_unchanged": (
            module_class_containers_unchanged
        ),
        "module_class_custom_object_state_unchanged": (
            module_class_custom_objects_unchanged
        ),
        "config_like_identity_type_and_structure_unchanged": (
            _config_like_objects_are_identical(
                before.config_like_objects,
                after.config_like_objects,
            )
        ),
        "parameter_registry_and_identity_unchanged": (
            _tensor_registries_are_identical(before.parameters, after.parameters)
        ),
        "parameter_metadata_unchanged": (
            _tensor_metadata(before.parameters) == _tensor_metadata(after.parameters)
        ),
        "parameter_versions_unchanged": (
            _tensor_versions(before.parameters) == _tensor_versions(after.parameters)
        ),
        "parameter_contents_unchanged": (
            _tensor_contents(before.parameters) == _tensor_contents(after.parameters)
        ),
        "parameter_python_attribute_state_unchanged": (
            _tensor_python_namespaces_are_identical(
                before.parameters,
                after.parameters,
            )
        ),
        "parameter_hook_state_unchanged": (
            _tensor_hook_states_are_identical(
                before.parameters,
                after.parameters,
            )
        ),
        "parameter_gradient_presence_unchanged": (
            _parameter_gradient_presence(before.parameters)
            == _parameter_gradient_presence(after.parameters)
        ),
        "parameter_gradient_identity_unchanged": (
            _parameter_gradient_bindings_are_identical(
                before.parameters,
                after.parameters,
            )
        ),
        "parameter_gradient_metadata_unchanged": (
            _tensor_metadata(before_gradients)
            == _tensor_metadata(after_gradients)
        ),
        "parameter_gradient_versions_unchanged": (
            _tensor_versions(before_gradients)
            == _tensor_versions(after_gradients)
        ),
        "parameter_gradient_contents_unchanged": (
            _tensor_contents(before_gradients)
            == _tensor_contents(after_gradients)
        ),
        "parameter_gradient_python_attribute_state_unchanged": (
            _tensor_python_namespaces_are_identical(
                before_gradients,
                after_gradients,
            )
        ),
        "parameter_gradient_hook_state_unchanged": (
            _tensor_hook_states_are_identical(
                before_gradients,
                after_gradients,
            )
        ),
        "buffer_registry_and_identity_unchanged": (
            _tensor_registries_are_identical(before.buffers, after.buffers)
        ),
        "buffer_metadata_unchanged": (
            _tensor_metadata(before.buffers) == _tensor_metadata(after.buffers)
        ),
        "buffer_versions_unchanged": (
            _tensor_versions(before.buffers) == _tensor_versions(after.buffers)
        ),
        "buffer_contents_unchanged": (
            _tensor_contents(before.buffers) == _tensor_contents(after.buffers)
        ),
        "buffer_python_attribute_state_unchanged": (
            _tensor_python_namespaces_are_identical(
                before.buffers,
                after.buffers,
            )
        ),
        "buffer_hook_state_unchanged": (
            _tensor_hook_states_are_identical(
                before.buffers,
                after.buffers,
            )
        ),
    }


def _process_state_invariants(
    before: _ProcessSnapshot, after: _ProcessSnapshot
) -> dict[str, bool]:
    cuda_rng_states_unchanged = (
        before.cuda_initialized == after.cuda_initialized
        and len(before.cuda_rng_states) == len(after.cuda_rng_states)
        and all(
            torch.equal(expected, actual)
            for expected, actual in zip(
                before.cuda_rng_states,
                after.cuda_rng_states,
                strict=True,
            )
        )
    )
    return {
        "qwen2_module_bindings_unchanged": _namespace_is_identical(
            before.qwen2_module, after.qwen2_module
        ),
        "qwen2_module_container_state_unchanged": (
            _namespace_containers_are_identical(
                before.qwen2_module, after.qwen2_module
            )
        ),
        "qwen2_module_custom_object_state_unchanged": (
            _namespace_custom_objects_are_identical(
                before.qwen2_module, after.qwen2_module
            )
        ),
        "qwen2_class_bindings_unchanged": _namespace_is_identical(
            before.qwen2_class, after.qwen2_class
        ),
        "qwen2_class_container_state_unchanged": (
            _namespace_containers_are_identical(
                before.qwen2_class, after.qwen2_class
            )
        ),
        "qwen2_class_custom_object_state_unchanged": (
            _namespace_custom_objects_are_identical(
                before.qwen2_class, after.qwen2_class
            )
        ),
        "torch_cross_entropy_global_unchanged": (
            after.functional_cross_entropy is before.functional_cross_entropy
        ),
        "cpu_rng_state_unchanged": torch.equal(
            before.cpu_rng_state,
            after.cpu_rng_state,
        ),
        "cuda_initialization_state_unchanged": (
            before.cuda_initialized == after.cuda_initialized
        ),
        "cuda_rng_states_unchanged": cuda_rng_states_unchanged,
        "deterministic_cudnn_and_sdpa_backend_flags_unchanged": (
            before.backend_flags == after.backend_flags
        ),
    }


def _restore_owner_bindings(owner, before: _NamespaceSnapshot) -> None:
    if inspect.ismodule(owner):
        namespace = vars(owner)
        for name in tuple(namespace):
            if name not in before.bindings:
                del namespace[name]
        namespace.update(before.bindings)
        return

    current_names = set(vars(owner))
    for name in current_names - before.bindings.keys():
        delattr(owner, name)
    for name, value in before.bindings.items():
        if vars(owner).get(name) is not value:
            setattr(owner, name, value)


def _restore_model_bindings(before: _ModelSnapshot) -> None:
    # Restore exact module classes first so a third-party replacement cannot
    # intercept the namespace restoration through custom attribute behavior.
    for item in before.modules:
        if type(item.module) is not item.module_type:
            object.__setattr__(item.module, "__class__", item.module_type)
    # A shallow namespace restore reverses added/removed/rebound attributes,
    # including the approved temporary root forward. It deliberately does not
    # pretend to reverse in-place tensor/container mutation; the post-rollback
    # content/structure fingerprints detect that and poison the process.
    for item in before.modules:
        namespace = object.__getattribute__(item.module, "__dict__")
        namespace.clear()
        namespace.update(item.namespace.bindings)


def _restore_module_class_bindings(before: _ModelSnapshot) -> None:
    for item in before.module_classes:
        _restore_owner_bindings(item.module_class, item.namespace)


def _restore_parameter_gradients(before: _ModelSnapshot) -> None:
    for item in before.parameters:
        expected_gradient = (
            None if item.gradient is None else item.gradient.tensor
        )
        if item.tensor.grad is not expected_gradient:
            item.tensor.grad = expected_gradient


def _restore_tensor_mutable_bindings(before: _ModelSnapshot) -> None:
    snapshots = (
        *before.parameters,
        *before.buffers,
        *_parameter_gradients(before.parameters),
    )
    restored = set()
    for item in snapshots:
        if id(item.tensor) in restored:
            continue
        restored.add(id(item.tensor))
        namespace = vars(item.tensor)
        namespace.clear()
        namespace.update(item.python_namespace.bindings)
        for hook in item.hook_state:
            if getattr(item.tensor, hook.name, None) is not hook.value:
                setattr(item.tensor, hook.name, hook.value)


def _restore_process_runtime_state(before: _ProcessSnapshot) -> None:
    flags = dict(before.backend_flags)
    torch.use_deterministic_algorithms(
        bool(flags["deterministic_algorithms"]),
        warn_only=bool(flags["deterministic_algorithms_warn_only"]),
    )
    torch.utils.deterministic.fill_uninitialized_memory = bool(
        flags["fill_uninitialized_memory"]
    )
    torch.set_float32_matmul_precision(str(flags["float32_matmul_precision"]))
    torch.backends.cudnn.enabled = bool(flags["cudnn_enabled"])
    torch.backends.cudnn.benchmark = bool(flags["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(flags["cudnn_deterministic"])
    torch.backends.cudnn.allow_tf32 = bool(flags["cudnn_allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(
        flags["cuda_matmul_allow_tf32"]
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
        flags["cuda_matmul_allow_fp16_reduced_precision_reduction"]
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
        flags["cuda_matmul_allow_bf16_reduced_precision_reduction"]
    )
    torch.backends.cuda.enable_flash_sdp(bool(flags["flash_sdp_enabled"]))
    torch.backends.cuda.enable_mem_efficient_sdp(
        bool(flags["mem_efficient_sdp_enabled"])
    )
    torch.backends.cuda.enable_math_sdp(bool(flags["math_sdp_enabled"]))
    torch.backends.cuda.enable_cudnn_sdp(bool(flags["cudnn_sdp_enabled"]))
    torch.set_rng_state(before.cpu_rng_state)
    if before.cuda_initialized:
        torch.cuda.set_rng_state_all(list(before.cuda_rng_states))


def _rollback_isolation_state(
    *,
    model,
    modeling_qwen2,
    expected_class,
    process_before: _ProcessSnapshot,
    model_before: _ModelSnapshot,
) -> dict:
    restore_errors = []
    for label, operation in (
        (
            "rng_and_backend_state",
            lambda: _restore_process_runtime_state(process_before),
        ),
        (
            "qwen2_module_bindings",
            lambda: _restore_owner_bindings(
                modeling_qwen2, process_before.qwen2_module
            ),
        ),
        (
            "qwen2_class_bindings",
            lambda: _restore_owner_bindings(
                expected_class, process_before.qwen2_class
            ),
        ),
        (
            "torch_cross_entropy_global",
            lambda: setattr(
                torch.nn.functional,
                "cross_entropy",
                process_before.functional_cross_entropy,
            ),
        ),
        (
            "module_class_bindings",
            lambda: _restore_module_class_bindings(model_before),
        ),
        ("model_bindings_and_types", lambda: _restore_model_bindings(model_before)),
        (
            "parameter_gradient_bindings",
            lambda: _restore_parameter_gradients(model_before),
        ),
        (
            "tensor_python_and_hook_bindings",
            lambda: _restore_tensor_mutable_bindings(model_before),
        ),
    ):
        try:
            operation()
        except Exception as exc:  # pragma: no cover - defensive poison path
            restore_errors.append(f"{label}: {type(exc).__name__}: {exc}")

    process_invariants = {}
    model_invariants = {}
    verification_errors = []
    try:
        process_after = _capture_process_state(modeling_qwen2, expected_class)
        process_invariants = _process_state_invariants(process_before, process_after)
    except Exception as exc:  # pragma: no cover - defensive poison path
        verification_errors.append(
            f"process verification: {type(exc).__name__}: {exc}"
        )
    try:
        model_after = _capture_model_state(model)
        model_invariants = _model_state_invariants(
            model_before,
            model_after,
            allow_forward_addition=False,
        )
    except Exception as exc:
        verification_errors.append(
            f"model verification: {type(exc).__name__}: {exc}"
        )

    complete = (
        not restore_errors
        and not verification_errors
        and bool(process_invariants)
        and bool(model_invariants)
        and all(process_invariants.values())
        and all(model_invariants.values())
    )
    return {
        "attempted": True,
        "complete": complete,
        "process_state_poisoned": not complete,
        "restore_errors": restore_errors,
        "verification_errors": verification_errors,
        "process_invariants": process_invariants,
        "model_invariants": model_invariants,
    }


def _raise_isolation_error(
    *,
    model,
    modeling_qwen2,
    expected_class,
    process_before: _ProcessSnapshot,
    model_before: _ModelSnapshot,
    failed_invariants: list[str],
    cause: Exception | None = None,
) -> None:
    rollback_report = _rollback_isolation_state(
        model=model,
        modeling_qwen2=modeling_qwen2,
        expected_class=expected_class,
        process_before=process_before,
        model_before=model_before,
    )
    if rollback_report["complete"]:
        suffix = (
            "pre-binding process and model state was restored and verified, but "
            "the post-binding failure is fatal and the process must terminate"
        )
    else:
        suffix = (
            "rollback could not reproduce the exact pre-binding state; the process "
            "is poisoned and must terminate"
        )
    error = KernelIsolationError(
        "the isolated fused-linear-CE binding violated its state contract "
        f"({failed_invariants}); {suffix}",
        failed_invariants=failed_invariants,
        rollback_report=rollback_report,
    )
    if cause is not None:
        raise error from cause
    raise error


def apply_liger_fused_linear_ce(model) -> dict:
    """Patch only one loaded Qwen2 model instance's loss-producing forward.

    Every layer kernel remains the Transformers implementation. Across the
    explicitly attested surfaces, the only intended state change is the
    imported fused ``forward`` bound to this model instance after its source
    module and ABI are verified. Attestation covers
    registered parameters, gradients, and buffers; concrete module instances
    and their exact classes; built-in containers; config-like objects; and
    custom Python objects with inspectable ``__dict__`` state. Opaque leaves
    are covered by binding identity. Tensor contents use bounded streaming
    SHA-256 chunks, not a retained duplicate of model weights.

    A rejected binding restores restorable Python bindings and module types,
    then verifies the exact pre-binding fingerprints. If an in-place
    tensor/container mutation cannot be reversed without a full model copy,
    the raised ``KernelIsolationError`` marks the process poisoned and callers
    must exit.
    """
    _require_exact_package("liger-kernel", "liger_kernel", LIGER_KERNEL_VERSION)
    model_type = require_liger_model_support(getattr(model, "config", None))
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception as exc:
        raise RuntimeError("could not import the pinned Transformers Qwen2 model") from exc

    expected_class = modeling_qwen2.Qwen2ForCausalLM
    if type(model) is not expected_class:
        raise RuntimeError(
            "the isolated fused-linear-CE patch must run on an unwrapped, native "
            f"Qwen2ForCausalLM instance before PEFT; found {type(model)!r}"
        )
    if "forward" in vars(model):
        raise RuntimeError(
            "the isolated fused-linear-CE patch requires an unmodified model "
            "instance with no pre-existing forward override"
        )

    expected_fused_forward = _qwen2_fused_linear_ce_forward_function()
    process_before = _capture_process_state(modeling_qwen2, expected_class)
    model_before = _capture_model_state(model)

    try:
        try:
            model.forward = MethodType(expected_fused_forward, model)
        except Exception as exc:
            raise _IsolationViolation(
                ["direct_forward_binding_completed"],
                cause=exc,
            )

        process_after = _capture_process_state(modeling_qwen2, expected_class)
        model_after = _capture_model_state(model)

        invariants = {
            **_process_state_invariants(process_before, process_after),
            **_model_state_invariants(
                model_before,
                model_after,
                allow_forward_addition=True,
            ),
        }
        patched_forward = vars(model).get("forward")
        forward_is_instance_bound = (
            patched_forward is not None
            and getattr(patched_forward, "__self__", None) is model
        )
        forward_binding_matches_imported_qwen2_function = (
            forward_is_instance_bound
            and getattr(patched_forward, "__func__", None)
            is expected_fused_forward
        )
        fused_forward_keyword_contract = False
        if callable(patched_forward):
            patched_signature = inspect.signature(patched_forward)
            has_forward_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in patched_signature.parameters.values()
            )
            fused_forward_keyword_contract = (
                {"labels", "use_cache", "skip_logits"}
                <= patched_signature.parameters.keys()
                and has_forward_kwargs
            )

        invariants["forward_is_instance_bound"] = forward_is_instance_bound
        invariants["forward_binding_matches_imported_qwen2_function"] = (
            forward_binding_matches_imported_qwen2_function
        )
        invariants["fused_forward_keyword_contract"] = (
            fused_forward_keyword_contract
        )
        failed_invariants = [
            name for name, passed in invariants.items() if not passed
        ]
        if failed_invariants:
            raise _IsolationViolation(failed_invariants)

        report = {
            "model_type": model_type,
            "layer_backend": NATIVE_LAYER_BACKEND,
            "loss_backend": "liger",
            "loss_implementation": FUSED_LINEAR_CE_IMPLEMENTATION,
            "patch_scope": "model-instance-forward-only",
            "binding_implementation": "types.MethodType",
            "forward_function": (
                "liger_kernel.transformers.model.qwen2.lce_forward"
            ),
            "forward_identity": {
                "module": expected_fused_forward.__module__,
                "name": expected_fused_forward.__name__,
                "qualname": expected_fused_forward.__qualname__,
            },
            "forward_source_sha256": LIGER_QWEN2_SOURCE_SHA256,
            "invariants": invariants,
            "trust_boundary": {
                "supply_chain": (
                    "exact-package-version-and-approved-qwen2-module-sha256"
                ),
                "attestation_purpose": "accidental-state-and-api-drift-detection",
                "identity_semantics": "binding-drift-only-not-semantic-integrity",
                "not_a_sandbox_for": [
                    "malicious-or-compromised-python-dependencies",
                    "import-time-code",
                    "concurrent-mutation",
                    "native-extensions",
                    "function-code-tampering",
                    "builtins",
                    "arbitrary-inherited-framework-globals",
                ],
            },
            "state_attestation": {
                "tensor_content_digest": "sha256",
                "tensor_digest_chunk_bytes": _TENSOR_DIGEST_CHUNK_BYTES,
                "retains_duplicate_model_tensors": False,
                "tensor_python_and_hook_state": (
                    "binding-and-structural-sha256"
                ),
                "config_containers_and_inspectable_objects": (
                    "structural-sha256"
                ),
                "opaque_python_leaves": "binding-identity",
                "rng_state": "cpu-and-all-visible-cuda-generators",
                "backend_flags": "deterministic-cudnn-matmul-and-sdpa",
                "failure_contract": (
                    "verified-rollback-evidence-and-fatal-process-exit"
                ),
            },
        }
    except _IsolationViolation as violation:
        _raise_isolation_error(
            model=model,
            modeling_qwen2=modeling_qwen2,
            expected_class=expected_class,
            process_before=process_before,
            model_before=model_before,
            failed_invariants=violation.failed_invariants,
            cause=violation.cause,
        )
    except Exception as exc:
        _raise_isolation_error(
            model=model,
            modeling_qwen2=modeling_qwen2,
            expected_class=expected_class,
            process_before=process_before,
            model_before=model_before,
            failed_invariants=["post_binding_validation_completed"],
            cause=exc,
        )
    return report


def binary_mask_labels(
    input_ids: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a binary per-token weight mask to labels with ignore_index."""
    if input_ids.shape != weights.shape:
        raise ValueError(
            f"input IDs and weights must have the same shape, got {input_ids.shape} and {weights.shape}"
        )
    is_binary = torch.logical_or(weights == 0, weights == 1)
    if not bool(is_binary.all().item()):
        raise ValueError(
            "--kernel-backend liger fused-linear-CE requires binary 0/1 token weights; "
            "fractional weights require --kernel-backend native"
        )
    labels = input_ids.masked_fill(weights != 1, -100)
    target_tokens = (labels[:, 1:] != -100).sum()
    return labels, target_tokens


def liger_sft_forward(model, input_ids: torch.Tensor, weights: torch.Tensor):
    """Run Liger fused linear CE and return a local token-sum loss."""
    labels, target_tokens = binary_mask_labels(input_ids, weights)
    output = model(
        input_ids=input_ids,
        labels=labels,
        num_items_in_batch=1,
        accum_dtype=torch.float32,
        skip_logits=True,
        use_cache=False,
    )
    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("the fused-linear-CE forward returned no loss")
    if loss.ndim != 0:
        raise RuntimeError(
            f"the fused-linear-CE forward returned a non-scalar loss with shape {loss.shape}"
        )
    if getattr(output, "logits", None) is not None:
        raise RuntimeError(
            "the requested Liger fused-loss path materialized logits; "
            "this model/version combination is unsupported"
        )
    # Liger 0.8.0 selects reduction='sum' when num_items_in_batch is given,
    # then divides by that value. Passing one therefore preserves Yeto's
    # local token-SUM contract exactly.
    return loss, target_tokens
