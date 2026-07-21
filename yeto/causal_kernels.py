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

import torch

from .kernel_deps import FLASH_ATTN_VERSION, LIGER_KERNEL_VERSION

ATTENTION_BACKENDS = ("auto", "sdpa", "flash-attn-2")
KERNEL_BACKENDS = ("native", "liger")

NATIVE_LAYER_BACKEND = "transformers-native"
NATIVE_LOSS_IMPLEMENTATION = "torch-cross-entropy"
FUSED_LINEAR_CE_IMPLEMENTATION = "liger-fused-linear-cross-entropy"

_QWEN2_APPLY_CONTROLS = (
    ("rope", True),
    ("cross_entropy", False),
    ("fused_linear_cross_entropy", True),
    ("rms_norm", True),
    ("swiglu", True),
    ("model", None),
)

_HF_ATTENTION_NAMES = {
    "sdpa": "sdpa",
    "flash-attn-2": "flash_attention_2",
}
_DISPLAY_ATTENTION_NAMES = {value: key for key, value in _HF_ATTENTION_NAMES.items()}
_PACKAGE_INSTALL_HINTS = {
    "liger-kernel": "pip install -e '.[a100-liger]'",
    "flash-attn": "the pinned --no-build-isolation command in docs/A100_KERNELS.md",
}

_TENSOR_DIGEST_CHUNK_BYTES = 4 * 1024 * 1024
_STRUCTURAL_CONTAINER_TYPES = (dict, list, tuple, set, frozenset, bytearray)


class KernelIsolationError(RuntimeError):
    """A rejected third-party apply call and its verified rollback status.

    ``process_state_poisoned`` is true when the pre-apply model/process state
    could not be reproduced exactly without retaining a second copy of model
    tensors.  Callers must terminate the process in that case; continuing with
    another model would make the isolation claim unverifiable.
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
        self.fatal = self.process_state_poisoned


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


@dataclass(frozen=True)
class _ModuleSnapshot:
    name: str
    module: torch.nn.Module
    module_type: type
    namespace: _NamespaceSnapshot


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


@dataclass(frozen=True)
class _ConfigSnapshot:
    object_id: int
    object_type: str
    structural_sha256: str


@dataclass(frozen=True)
class _ModelSnapshot:
    modules: tuple[_ModuleSnapshot, ...]
    parameters: tuple[_TensorSnapshot, ...]
    buffers: tuple[_TensorSnapshot, ...]
    config: _ConfigSnapshot


@dataclass(frozen=True)
class _ProcessSnapshot:
    qwen2_module: _NamespaceSnapshot
    qwen2_class: _NamespaceSnapshot
    functional_cross_entropy: object


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
                    digest, key, leaf_identity=leaf_identity, seen=seen
                )
                _structural_update(
                    digest, item, leaf_identity=leaf_identity, seen=seen
                )
        elif isinstance(value, (set, frozenset)):
            item_digests = []
            for item in value:
                item_digest = hashlib.sha256()
                _structural_update(
                    item_digest,
                    item,
                    leaf_identity=leaf_identity,
                    seen={},
                )
                item_digests.append(item_digest.digest())
            for item_digest in sorted(item_digests):
                _hash_field(digest, "set-item", item_digest)
        else:
            for item in value:
                _structural_update(
                    digest, item, leaf_identity=leaf_identity, seen=seen
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


def _structural_sha256(value, *, leaf_identity: bool) -> str:
    digest = hashlib.sha256()
    _structural_update(
        digest,
        value,
        leaf_identity=leaf_identity,
        seen={},
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
                    sha256=_structural_sha256(value, leaf_identity=True),
                )
            )
    return tuple(fingerprints)


def _capture_namespace(namespace: dict[str, object]) -> _NamespaceSnapshot:
    bindings = dict(namespace)
    return _NamespaceSnapshot(
        bindings=bindings,
        containers=_container_fingerprints(bindings),
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
    before: _NamespaceSnapshot, after: _NamespaceSnapshot
) -> bool:
    return before.containers == after.containers


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


def _capture_tensor(
    category: str,
    name: str,
    tensor: torch.Tensor,
    content_cache: dict[int, str],
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
    )


def _named_tensors(model, method_name: str):
    method = getattr(model, method_name)
    try:
        return tuple(method(recurse=True, remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        return tuple(method(recurse=True))


def _capture_config(config) -> _ConfigSnapshot:
    if config is None:
        raise RuntimeError("the isolated fused-linear-CE model exposes no config")
    to_dict = getattr(config, "to_dict", None)
    try:
        structural_value = to_dict() if callable(to_dict) else dict(vars(config))
        structural_sha256 = _structural_sha256(
            structural_value,
            leaf_identity=False,
        )
    except Exception as exc:
        raise RuntimeError("could not structurally fingerprint the model config") from exc
    return _ConfigSnapshot(
        object_id=id(config),
        object_type=_qualified_type(config),
        structural_sha256=structural_sha256,
    )


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
    content_cache: dict[int, str] = {}
    parameters = tuple(
        _capture_tensor("parameter", name, tensor, content_cache)
        for name, tensor in _named_tensors(model, "named_parameters")
    )
    buffers = tuple(
        _capture_tensor("buffer", name, tensor, content_cache)
        for name, tensor in _named_tensors(model, "named_buffers")
    )
    return _ModelSnapshot(
        modules=module_snapshots,
        parameters=parameters,
        buffers=buffers,
        config=_capture_config(getattr(model, "config", None)),
    )


def _capture_process_state(modeling_qwen2, expected_class) -> _ProcessSnapshot:
    return _ProcessSnapshot(
        qwen2_module=_capture_namespace(vars(modeling_qwen2)),
        qwen2_class=_capture_namespace(dict(vars(expected_class))),
        functional_cross_entropy=torch.nn.functional.cross_entropy,
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


def _qwen2_fused_linear_ce_apply_function():
    """Return the pinned public Qwen2 apply function after an exact ABI check."""
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        from liger_kernel.transformers import functional as liger_functional
    except Exception as exc:
        raise RuntimeError(
            "could not import Liger's public Qwen2 kernel apply function"
        ) from exc

    signature = inspect.signature(apply_liger_kernel_to_qwen2)
    parameters = tuple(signature.parameters.values())
    expected_names = tuple(name for name, _default in _QWEN2_APPLY_CONTROLS)
    if tuple(parameter.name for parameter in parameters) != expected_names:
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 apply function does not expose "
            f"the exact controls {expected_names}; found {tuple(signature.parameters)}"
        )
    for parameter, (_name, expected_default) in zip(
        parameters, _QWEN2_APPLY_CONTROLS, strict=True
    ):
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise RuntimeError(
                f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 control "
                f"{parameter.name!r} has unsupported kind {parameter.kind}"
            )
        if parameter.default != expected_default:
            raise RuntimeError(
                f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 control "
                f"{parameter.name!r} has unexpected default {parameter.default!r}; "
                f"expected {expected_default!r}"
            )
    fused_signature = inspect.signature(
        liger_functional.liger_fused_linear_cross_entropy
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
    return apply_liger_kernel_to_qwen2


def require_liger_model_support(config) -> str:
    """Require the one model family approved for isolated fused linear CE."""

    model_type = getattr(config, "model_type", None)
    if model_type != "qwen2":
        raise RuntimeError(
            "the isolated Liger fused-linear-CE lane supports only Hugging Face "
            f"Qwen2/Qwen2.5 (model_type='qwen2'); found {model_type!r}"
        )
    _qwen2_fused_linear_ce_apply_function()
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
            root_before.namespace, root_after.namespace
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

    config_unchanged = before.config == after.config
    return {
        "module_layout_unchanged": module_layout_unchanged,
        "module_types_unchanged": module_types_unchanged,
        "instance_bindings_unchanged": instance_bindings_unchanged,
        "instance_container_state_unchanged": instance_containers_unchanged,
        "nested_module_bindings_unchanged": nested_bindings_unchanged,
        "nested_container_state_unchanged": nested_containers_unchanged,
        "model_config_identity_type_and_structure_unchanged": config_unchanged,
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
    }


def _process_state_invariants(
    before: _ProcessSnapshot, after: _ProcessSnapshot
) -> dict[str, bool]:
    return {
        "qwen2_module_bindings_unchanged": _namespace_is_identical(
            before.qwen2_module, after.qwen2_module
        ),
        "qwen2_module_container_state_unchanged": (
            _namespace_containers_are_identical(
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
        "torch_cross_entropy_global_unchanged": (
            after.functional_cross_entropy is before.functional_cross_entropy
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
        ("model_bindings_and_types", lambda: _restore_model_bindings(model_before)),
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
        suffix = "pre-apply process and model state was restored and verified"
    else:
        suffix = (
            "rollback could not reproduce the exact pre-apply state; the process "
            "is poisoned and must terminate"
        )
    error = KernelIsolationError(
        "the isolated fused-linear-CE apply call violated its state contract "
        f"({failed_invariants}); {suffix}",
        failed_invariants=failed_invariants,
        rollback_report=rollback_report,
    )
    if cause is not None:
        raise error from cause
    raise error


def apply_liger_fused_linear_ce(model) -> dict:
    """Patch only one loaded Qwen2 model instance's loss-producing forward.

    Every layer kernel remains the Transformers implementation. The only
    accepted state change is a new ``forward`` binding on this exact model
    instance. Bindings, module types, built-in container structure, config,
    and registered tensor metadata/content are attested before and after the
    call. Tensor contents use bounded streaming SHA-256 chunks, not a retained
    duplicate of model weights.

    A rejected call restores restorable Python bindings and module types, then
    verifies the exact pre-apply fingerprints. If an in-place tensor/container
    mutation cannot be reversed without a full model copy, the raised
    ``KernelIsolationError`` marks the process poisoned and callers must exit.
    """
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

    apply_fn = _qwen2_fused_linear_ce_apply_function()
    inherited_forward = model.forward
    process_before = _capture_process_state(modeling_qwen2, expected_class)
    model_before = _capture_model_state(model)

    try:
        apply_fn(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            model=model,
        )
    except Exception as exc:
        _raise_isolation_error(
            model=model,
            modeling_qwen2=modeling_qwen2,
            expected_class=expected_class,
            process_before=process_before,
            model_before=model_before,
            failed_invariants=["third_party_apply_completed"],
            cause=exc,
        )

    try:
        process_after = _capture_process_state(modeling_qwen2, expected_class)
        model_after = _capture_model_state(model)
    except Exception as exc:
        _raise_isolation_error(
            model=model,
            modeling_qwen2=modeling_qwen2,
            expected_class=expected_class,
            process_before=process_before,
            model_before=model_before,
            failed_invariants=["post_apply_state_was_fully_observable"],
            cause=exc,
        )

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
        and getattr(patched_forward, "__func__", None)
        is not getattr(inherited_forward, "__func__", None)
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
    invariants["fused_forward_keyword_contract"] = fused_forward_keyword_contract
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        _raise_isolation_error(
            model=model,
            modeling_qwen2=modeling_qwen2,
            expected_class=expected_class,
            process_before=process_before,
            model_before=model_before,
            failed_invariants=failed_invariants,
        )

    return {
        "model_type": model_type,
        "layer_backend": NATIVE_LAYER_BACKEND,
        "loss_backend": "liger",
        "loss_implementation": FUSED_LINEAR_CE_IMPLEMENTATION,
        "patch_scope": "model-instance-forward-only",
        "apply_function": (
            "liger_kernel.transformers.apply_liger_kernel_to_qwen2"
        ),
        "apply_controls": {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": False,
            "swiglu": False,
        },
        "invariants": invariants,
        "state_attestation": {
            "tensor_content_digest": "sha256",
            "tensor_digest_chunk_bytes": _TENSOR_DIGEST_CHUNK_BYTES,
            "retains_duplicate_model_tensors": False,
            "config_and_builtin_containers": "structural-sha256",
            "failure_contract": (
                "verified-rollback-or-fatal-poisoned-process"
            ),
        },
    }


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
