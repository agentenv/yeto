"""Canonical PEFT LoRA representation used at every RL synchronization boundary."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..fragments import MERGE_AVG, Fragment, FragmentLayout
from ..protocol import layout_fingerprint

_PEFT_LORA_NAME = re.compile(r"\.lora_(?:A|B)\.weight\Z")


@dataclass(frozen=True, order=True)
class CanonicalTensorSpec:
    name: str
    shape: tuple[int, ...]
    numel: int

    def __post_init__(self) -> None:
        if not self.name or not _PEFT_LORA_NAME.search(self.name):
            raise ValueError(f"not a canonical PEFT LoRA tensor name: {self.name!r}")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError(f"invalid shape for {self.name!r}: {self.shape}")
        if math.prod(self.shape) != self.numel:
            raise ValueError(f"shape/numel mismatch for {self.name!r}")


@dataclass(frozen=True)
class PolicyIdentity:
    version: int
    policy_hash: str

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("policy version must be non-negative")
        if len(self.policy_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.policy_hash
        ):
            raise ValueError("policy hash must be a lowercase SHA256")


@dataclass(frozen=True)
class CanonicalLoraState:
    identity: PolicyIdentity
    layout_fingerprint: str
    specs: tuple[CanonicalTensorSpec, ...]
    tensors: Mapping[str, torch.Tensor]

    @property
    def policy_version(self) -> int:
        return self.identity.version

    @property
    def policy_hash(self) -> str:
        return self.identity.policy_hash


def canonical_specs(tensors: Mapping[str, torch.Tensor]) -> tuple[CanonicalTensorSpec, ...]:
    if not tensors:
        raise ValueError("canonical LoRA state is empty")
    return tuple(
        CanonicalTensorSpec(name, tuple(int(dim) for dim in tensor.shape), tensor.numel())
        for name, tensor in sorted(tensors.items())
    )


def build_avg_layout(specs: Sequence[CanonicalTensorSpec]) -> FragmentLayout:
    ordered = sorted(specs, key=lambda spec: spec.name)
    if not ordered:
        raise ValueError("canonical LoRA layout is empty")
    if len({spec.name for spec in ordered}) != len(ordered):
        raise ValueError("canonical LoRA tensor names must be unique")
    return FragmentLayout(
        [
            Fragment(
                merge_mode=MERGE_AVG,
                tensors=[(spec.name, spec.numel) for spec in ordered],
                identity_shapes={spec.name: spec.shape for spec in ordered},
            )
        ]
    )


def _canonical_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name!r} must be a floating-point torch.Tensor")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name!r} contains NaN or Inf")
    return value.clone()


def normalize_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: _canonical_tensor(tensor, name)
        for name, tensor in sorted(tensors.items())
    }


def flat_tensor(
    tensors: Mapping[str, torch.Tensor],
    specs: Sequence[CanonicalTensorSpec] | None = None,
) -> torch.Tensor:
    specs = tuple(sorted(specs or canonical_specs(tensors)))
    if set(tensors) != {spec.name for spec in specs}:
        raise ValueError("tensor names do not match canonical specs")
    values = []
    for spec in specs:
        tensor = _canonical_tensor(tensors[spec.name], spec.name)
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(f"shape mismatch for {spec.name!r}")
        values.append(tensor.reshape(-1))
    return torch.cat(values)


def tensors_from_flat(
    flat: torch.Tensor, specs: Sequence[CanonicalTensorSpec]
) -> dict[str, torch.Tensor]:
    specs = tuple(specs)
    if specs != tuple(sorted(specs)):
        raise ValueError("canonical LoRA specs are not sorted")
    flat = _canonical_tensor(flat.reshape(-1), "flat LoRA policy")
    expected = sum(spec.numel for spec in specs)
    if flat.numel() != expected:
        raise ValueError(f"flat LoRA policy has {flat.numel()} values, expected {expected}")
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for spec in specs:
        result[spec.name] = flat[offset : offset + spec.numel].reshape(spec.shape).clone()
        offset += spec.numel
    return result


def policy_sha256(
    tensors: Mapping[str, torch.Tensor],
    specs: Sequence[CanonicalTensorSpec] | None = None,
) -> tuple[str, str]:
    specs = tuple(sorted(specs or canonical_specs(tensors)))
    layout = build_avg_layout(specs)
    fingerprint = layout_fingerprint(layout)
    digest = hashlib.sha256()
    digest.update(b"yeto-rl-policy-v1\0")
    digest.update(fingerprint)
    # numpy is a transitive torch dependency and gives an explicit little-endian encoding.
    data = flat_tensor(tensors, specs).numpy().astype("<f4", copy=False).tobytes()
    digest.update(data)
    return fingerprint.hex(), digest.hexdigest()


def canonical_state(
    policy_version: int,
    tensors: Mapping[str, torch.Tensor],
    *,
    expected_specs: Sequence[CanonicalTensorSpec] | None = None,
    expected_layout_fingerprint: str | None = None,
) -> CanonicalLoraState:
    normalized = normalize_tensors(tensors)
    specs = canonical_specs(normalized)
    if expected_specs is not None and specs != tuple(expected_specs):
        raise ValueError("canonical LoRA names or shapes changed")
    fingerprint, digest = policy_sha256(normalized, specs)
    if expected_layout_fingerprint is not None and fingerprint != expected_layout_fingerprint:
        raise ValueError("canonical LoRA layout fingerprint changed")
    return CanonicalLoraState(
        identity=PolicyIdentity(policy_version, digest),
        layout_fingerprint=fingerprint,
        specs=specs,
        tensors=normalized,
    )


def policy_delta(local: CanonicalLoraState, base: CanonicalLoraState) -> torch.Tensor:
    if local.specs != base.specs or local.layout_fingerprint != base.layout_fingerprint:
        raise ValueError("local and base LoRA layouts differ")
    delta = flat_tensor(local.tensors, local.specs) - flat_tensor(base.tensors, base.specs)
    if not torch.isfinite(delta).all().item():
        raise ValueError("local LoRA delta contains NaN or Inf")
    return delta.contiguous()
