"""Publish fast path (rl-publish-fastpath): hash the policy where it lives.

On a single island without an outer sync the driver never needs the LoRA
tensors themselves: publication goes trainer -> engines through Miles'
``update_weights``, and the driver only checks identities.  Shipping the whole
adapter (11.2 GB for Flash-Next) through Ray to the driver twice per round and
hashing it three times there was ~60% of a round (S16 FN 2x8 run).

This module computes, in ONE call over the canonical tensors, exactly the two
digests the driver used to compute itself:

* ``policy_tensor_hash`` -- byte-identical to :func:`yeto.rl.core.policy_tensor_hash`;
* ``payload_hash`` / ``payload_bytes`` -- byte-identical to
  :func:`yeto.rl.adapters.miles.publish.payload_digest`.

The two SHA-256 streams run in two threads (``hashlib`` releases the GIL on
large buffers), so the wall time is about one pass.  The definitions of the
hashes do not change: tape values stay identical (tests pin this).

:class:`TrainerResidentState` is the driver-side handle: identity fields and
the digests, no tensors.  Anything that really needs tensors (``.tensors``,
``to_lora()``, ``policy_hash()``) triggers a full export through the
``materialize`` callback, which re-checks the tensor hash, so a consumer this
change did not anticipate stays correct (only slower).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from .trainable_state import (
    LAYOUT_FRAGMENTS,
    LAYOUT_LORA,
    TrainableState,
    UnsupportedLayoutError,
    require_supported_layout,
)

# Layouts a trainer-resident handle can stand for.  ``fragments`` (full
# parameters, P0-D1) never materializes: its tensors stay chunked in the trainer.
RESIDENT_LAYOUTS = frozenset({LAYOUT_LORA, LAYOUT_FRAGMENTS})

POLICY_TENSOR_PREFIX = b"yeto-rl-policy-tensors-v1\0"
PAYLOAD_PREFIX = b"yeto-rl-publication-payload-v1\0"


class PolicyDigestError(RuntimeError):
    """The trainer's weights no longer match a resident state's digest."""


class WeightsChanged(PolicyDigestError):
    """The trainer's weights version moved since a resident state was exported."""


@dataclass(frozen=True)
class PolicyDigest:
    policy_tensor_hash: str
    payload_hash: str
    payload_bytes: int
    # (name, shape) in canonical (sorted) order; enough to rebuild the layout hash
    specs: tuple[tuple[str, tuple[int, ...]], ...]
    hash_seconds: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "policy_tensor_hash": self.policy_tensor_hash,
            "payload_hash": self.payload_hash,
            "payload_bytes": int(self.payload_bytes),
            "specs": [[name, list(shape)] for name, shape in self.specs],
            "hash_seconds": float(self.hash_seconds),
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> "PolicyDigest":
        return cls(
            policy_tensor_hash=str(raw["policy_tensor_hash"]),
            payload_hash=str(raw["payload_hash"]),
            payload_bytes=int(raw["payload_bytes"]),
            specs=tuple((str(n), tuple(int(d) for d in s)) for n, s in raw["specs"]),
            hash_seconds=float(raw.get("hash_seconds", 0.0)),
        )


def _raw(tensor: Any, name: str) -> memoryview:
    import torch

    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name!r} must be a floating-point torch.Tensor")
    value = tensor.detach()
    if value.device.type != "cpu" or value.dtype != torch.float32 or not value.is_contiguous():
        value = value.to(device="cpu", dtype=torch.float32).contiguous()
    return memoryview(value.numpy()).cast("B")


def _tensor_stream(names, tensors, *, base_model_revision, lora_config_hash, layout_hash) -> str:
    digest = hashlib.sha256()
    digest.update(POLICY_TENSOR_PREFIX)
    digest.update(base_model_revision.encode("ascii"))
    digest.update(lora_config_hash.encode("ascii"))
    digest.update(layout_hash.encode("ascii"))
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(_raw(tensors[name], name))
    return digest.hexdigest()


def _payload_stream(names, tensors) -> tuple[str, int]:
    digest = hashlib.sha256(PAYLOAD_PREFIX)
    total = 0
    for name in names:
        raw = _raw(tensors[name], name)
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(json.dumps(list(tensors[name].shape)).encode("ascii"))
        digest.update(raw)
        total += raw.nbytes
    return digest.hexdigest(), total


def digest_canonical_tensors(
    tensors: Mapping[str, Any],
    *,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str,
    parallel: bool = True,
) -> PolicyDigest:
    """Both publication digests of canonical CPU f32 tensors in one call.

    The caller (the trainer's export) has already checked the values are
    finite; this function only hashes.
    """

    import time

    started = time.monotonic()
    names = tuple(sorted(tensors))
    if not names:
        raise ValueError("canonical LoRA state is empty")
    tensor_kwargs = dict(base_model_revision=base_model_revision,
                         lora_config_hash=lora_config_hash, layout_hash=layout_hash)
    if parallel:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="yeto-digest") as pool:
            tensor_future = pool.submit(_tensor_stream, names, tensors, **tensor_kwargs)
            payload_future = pool.submit(_payload_stream, names, tensors)
            tensor_hash = tensor_future.result()
            payload_hash, payload_bytes = payload_future.result()
    else:
        tensor_hash = _tensor_stream(names, tensors, **tensor_kwargs)
        payload_hash, payload_bytes = _payload_stream(names, tensors)
    specs = tuple((name, tuple(int(d) for d in tensors[name].shape)) for name in names)
    return PolicyDigest(tensor_hash, payload_hash, payload_bytes, specs,
                        hash_seconds=round(time.monotonic() - started, 3))


class NonFiniteTensor(ValueError):
    """A tensor staged by :func:`stage_f32_on_host` holds NaN or Inf (``name`` says which)."""

    def __init__(self, name: str):
        super().__init__(f"{name!r} contains NaN or Inf")
        self.name = name


class HostStaging:
    """One reusable pinned host buffer for :func:`stage_f32_on_host` (E1).

    Allocating pinned memory costs milliseconds per call; the policy keeps the
    same size every round, so the owner (the trainer process) keeps one of
    these and the buffer is reallocated only when the policy grows."""

    def __init__(self, *, pin: bool = True) -> None:
        self._pin = pin
        self._buffer = None

    def flat(self, numel: int):
        import torch

        if self._buffer is None or self._buffer.numel() < numel:
            pin = self._pin and torch.cuda.is_available()
            self._buffer = torch.empty(numel, dtype=torch.float32, pin_memory=pin)
        return self._buffer[:numel]


def stage_f32_on_host(tensors: Mapping[str, Any], staging: HostStaging | None = None) -> dict[str, Any]:
    """Canonical CPU f32 contiguous copies of ``tensors`` (sorted by name), bit for bit
    the values of ``t.detach().to("cpu", torch.float32).contiguous()`` per tensor.

    E1: with ``staging`` and every tensor on one device, the f32 conversion and the
    finite check run on that device (the GPU) and the copies go to the pinned host
    buffer asynchronously, with ONE synchronization for all of them (the per-tensor
    path waits once per tensor: ~400 waits for a Qwen3-0.6B LoRA). No extra device
    buffer is held: a tensor already f32 and contiguous is copied as it is. The
    returned tensors are views of ``staging``'s buffer and are valid until the next
    call with the same ``staging``: hash them, do not keep them (``torch.save`` of a
    view would also write the whole buffer). Without ``staging`` (or with tensors on
    several devices) each tensor is copied on its own, as before. Raises
    :class:`NonFiniteTensor` naming the first non-finite tensor in name order, like
    the per-tensor check."""

    import torch

    names = sorted(tensors)
    values = [tensors[n].detach() for n in names]
    devices = {v.device for v in values}
    if staging is None or len(devices) != 1:
        out = {}
        for name, value in zip(names, values):
            value = value.to(device="cpu", dtype=torch.float32).contiguous().clone()
            if not torch.isfinite(value).all().item():
                raise NonFiniteTensor(name)
            out[name] = value
        return out
    flat = staging.flat(sum(v.numel() for v in values))
    out, finite, offset = {}, [], 0
    for name, value in zip(names, values):
        source = value.to(dtype=torch.float32).contiguous()
        finite.append(torch.isfinite(source).all())
        view = flat[offset:offset + source.numel()].view(source.shape)
        view.copy_(source, non_blocking=True)
        out[name] = view
        offset += source.numel()
    device = next(iter(devices))
    ok = bool(torch.stack(finite).all().item()) if finite else True
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()  # the copies to the host are complete
    if not ok:
        for name, check in zip(names, finite):
            if not bool(check.item()):
                raise NonFiniteTensor(name)
    return out


def layout_hash_of(specs: Sequence[tuple[str, tuple[int, ...]]]) -> str:
    """The canonical layout hash rebuilt from (name, shape) only."""

    from math import prod

    from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash

    return canonical_layout_hash(tuple(
        CanonicalTensorSpec(name, tuple(shape), "float32", int(prod(shape)))
        for name, shape in sorted(specs)
    ))


@dataclass(frozen=True)
class TrainerResidentState:
    """A policy that stays in the trainer: identities and digests, no tensors.

    Duck-types the parts of :class:`TrainableState` the single-island publish
    path reads.  ``materialize(policy_version)`` exports the full state (used
    only by consumers that need tensors); ``recheck()`` returns the digest of
    the trainer's CURRENT weights (the publisher's "trainer still holds what
    is being published" check, by trainer weights version).
    """

    layout: str
    base_model_revision: str
    config_hash: str
    layout_hash: str
    policy_version: int
    digest: PolicyDigest
    materialize_fn: Callable[[int], TrainableState] = field(repr=False, compare=False)
    # trainer weights version at export: {"process_id", "version"} (design D3)
    weights_mark: Mapping[str, Any] | None = None
    # (state) -> None, raises when the trainer no longer holds this policy
    holds_fn: Callable[[Any], None] | None = field(default=None, repr=False, compare=False)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_supported_layout(self.layout, RESIDENT_LAYOUTS)

    # -- identities (no tensors needed) -----------------------------------
    def policy_tensor_hash(self) -> str:
        return self.digest.policy_tensor_hash

    @property
    def payload_digest(self) -> tuple[str, int]:
        return self.digest.payload_hash, self.digest.payload_bytes

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.digest.specs)

    def with_version(self, policy_version: int) -> "TrainerResidentState":
        return replace(self, policy_version=int(policy_version), _cache={})

    def check_holds(self) -> None:
        """Raise if the trainer's weights changed since this handle's export
        (weights-version comparison, see design D3)."""

        if self.holds_fn is None:
            raise WeightsChanged("resident state has no trainer weights check")
        self.holds_fn(self)

    # -- lazy full state ----------------------------------------------------
    def materialize(self) -> TrainableState:
        if self.layout == LAYOUT_FRAGMENTS:
            raise UnsupportedLayoutError(self.layout, frozenset({LAYOUT_LORA}))
        state = self._cache.get("state")
        if state is None:
            state = self.materialize_fn(self.policy_version)
            if state.policy_tensor_hash() != self.digest.policy_tensor_hash:
                raise PolicyDigestError(
                    "trainer weights changed after the resident digest was taken")
            self._cache["state"] = state
        return state

    @property
    def tensors(self) -> Mapping[str, Any]:
        return self.materialize().tensors

    def to_lora(self):
        return self.materialize().to_lora()

    def policy_hash(self) -> str:
        return self.materialize().policy_hash()


def is_resident(state: Any) -> bool:
    return isinstance(state, TrainerResidentState)
