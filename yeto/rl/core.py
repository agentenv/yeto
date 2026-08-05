"""Canonical PEFT LoRA values at the Yeto/Miles synchronization boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..fragments import MERGE_AVG, Fragment, FragmentLayout
from ..protocol import layout_fingerprint

_PEFT_LORA_NAME = re.compile(r"\.lora_(?:A|B)\.weight\Z")


class StrictRlInvariantError(RuntimeError):
    """A deterministic INIT strict-run invariant violation."""

    def __init__(self, metric: str, message: str) -> None:
        super().__init__(message)
        self.metric = metric


@dataclass(frozen=True, order=True)
class CanonicalTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        if not self.name or not _PEFT_LORA_NAME.search(self.name):
            raise ValueError(f"not a canonical PEFT LoRA tensor name: {self.name!r}")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError(f"invalid shape for {self.name!r}: {self.shape}")
        if self.dtype != "float32":
            raise ValueError(f"canonical LoRA dtype must be float32, got {self.dtype!r}")
        if math.prod(self.shape) != self.numel:
            raise ValueError(f"shape/numel mismatch for {self.name!r}")


@dataclass(frozen=True)
class CanonicalLoraState:
    base_model_revision: str
    lora_config_hash: str
    layout_hash: str
    policy_version: int
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.base_model_revision):
            raise ValueError("base model revision must be an immutable commit")
        for name, value in (
            ("LoRA config", self.lora_config_hash),
            ("layout", self.layout_hash),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} hash must be a lowercase SHA256")
        if self.policy_version < 0:
            raise ValueError("policy version must be non-negative")

    @property
    def specs(self) -> tuple[CanonicalTensorSpec, ...]:
        return canonical_specs(self.tensors)


def canonical_specs(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[CanonicalTensorSpec, ...]:
    if not tensors:
        raise ValueError("canonical LoRA state is empty")
    return tuple(
        CanonicalTensorSpec(
            name,
            tuple(int(dim) for dim in tensor.shape),
            "float32",
            tensor.numel(),
        )
        for name, tensor in sorted(tensors.items())
    )


def build_avg_layout(specs: Sequence[CanonicalTensorSpec]) -> FragmentLayout:
    ordered = tuple(sorted(specs))
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


def build_rl_fragment_layout(
    specs: Sequence[CanonicalTensorSpec],
    num_fragments: int,
) -> FragmentLayout:
    """Build the deterministic all-AVG binpack used by decoupled RL."""

    ordered = tuple(sorted(specs, key=lambda spec: (-spec.numel, spec.name)))
    if num_fragments < 2:
        raise ValueError("decoupled RL requires at least 2 fragments")
    if num_fragments > len(ordered):
        raise ValueError("decoupled RL fragments exceed canonical tensor count")
    if len({spec.name for spec in ordered}) != len(ordered):
        raise ValueError("canonical LoRA tensor names must be unique")

    bins: list[list[CanonicalTensorSpec]] = [[] for _ in range(num_fragments)]
    sizes = [0] * num_fragments
    for spec in ordered:
        fragment_id = min(range(num_fragments), key=lambda index: sizes[index])
        bins[fragment_id].append(spec)
        sizes[fragment_id] += spec.numel
    return FragmentLayout(
        [
            Fragment(
                merge_mode=MERGE_AVG,
                tensors=[(spec.name, spec.numel) for spec in members],
                identity_shapes={spec.name: spec.shape for spec in members},
            )
            for members in bins
        ]
    )


def canonical_layout_hash(specs: Sequence[CanonicalTensorSpec]) -> str:
    """Semantic hash persisted by the syncer and rebuilt by the exporter."""

    return layout_fingerprint(build_avg_layout(specs)).hex()


def canonical_lora_config_hash(
    *, rank: int, target_modules: Sequence[str]
) -> str:
    if rank <= 0 or not target_modules:
        raise ValueError("canonical LoRA config requires rank and target modules")
    payload = {
        "bias": "none",
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "r": rank,
        "target_modules": sorted(set(target_modules)),
        "task_type": "CAUSAL_LM",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name!r} must be a floating-point torch.Tensor")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name!r} contains NaN or Inf")
    return value.clone()


def canonical_state(
    policy_version: int,
    tensors: Mapping[str, torch.Tensor],
    *,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str | None = None,
    expected_specs: Sequence[CanonicalTensorSpec] | None = None,
) -> CanonicalLoraState:
    normalized = {
        name: _canonical_tensor(tensor, name)
        for name, tensor in sorted(tensors.items())
    }
    specs = canonical_specs(normalized)
    if expected_specs is not None and specs != tuple(expected_specs):
        raise ValueError("canonical LoRA names, shapes, or dtypes changed")
    actual_layout_hash = canonical_layout_hash(specs)
    if layout_hash is not None and layout_hash != actual_layout_hash:
        raise ValueError("canonical LoRA layout hash changed")
    return CanonicalLoraState(
        base_model_revision,
        lora_config_hash,
        actual_layout_hash,
        policy_version,
        normalized,
    )


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
    flat: torch.Tensor,
    specs: Sequence[CanonicalTensorSpec],
) -> dict[str, torch.Tensor]:
    specs = tuple(specs)
    if specs != tuple(sorted(specs)):
        raise ValueError("canonical LoRA specs are not sorted")
    flat = _canonical_tensor(flat.reshape(-1), "flat LoRA policy")
    expected = sum(spec.numel for spec in specs)
    if flat.numel() != expected:
        raise ValueError(
            f"flat LoRA policy has {flat.numel()} values, expected {expected}"
        )
    tensors = {}
    offset = 0
    for spec in specs:
        tensors[spec.name] = flat[
            offset : offset + spec.numel
        ].reshape(spec.shape).clone()
        offset += spec.numel
    return tensors


def policy_delta(local: CanonicalLoraState, base: CanonicalLoraState) -> torch.Tensor:
    if local.policy_version != base.policy_version:
        raise ValueError("local and base policy versions differ")
    if local.specs != base.specs:
        raise ValueError("local and base LoRA layouts differ")
    if (
        local.base_model_revision,
        local.lora_config_hash,
        local.layout_hash,
    ) != (
        base.base_model_revision,
        base.lora_config_hash,
        base.layout_hash,
    ):
        raise ValueError("local and base canonical LoRA identities differ")
    delta = flat_tensor(local.tensors, local.specs) - flat_tensor(
        base.tensors, base.specs
    )
    if not torch.isfinite(delta).all().item():
        raise ValueError("local LoRA delta contains NaN or Inf")
    return delta.contiguous()


def policy_hash(state: CanonicalLoraState) -> str:
    digest = hashlib.sha256()
    digest.update(b"yeto-rl-policy-v1\0")
    digest.update(state.base_model_revision.encode("ascii"))
    digest.update(state.lora_config_hash.encode("ascii"))
    digest.update(state.layout_hash.encode("ascii"))
    digest.update(state.policy_version.to_bytes(8, "little"))
    for spec in state.specs:
        digest.update(spec.name.encode("utf-8"))
        digest.update(_canonical_tensor(state.tensors[spec.name], spec.name).numpy().tobytes())
    return digest.hexdigest()


def policy_tensor_hash(state: CanonicalLoraState) -> str:
    """Hash a complete canonical LoRA policy independently of local progress."""

    digest = hashlib.sha256()
    digest.update(b"yeto-rl-policy-tensors-v1\0")
    digest.update(state.base_model_revision.encode("ascii"))
    digest.update(state.lora_config_hash.encode("ascii"))
    digest.update(state.layout_hash.encode("ascii"))
    for spec in state.specs:
        digest.update(spec.name.encode("utf-8"))
        digest.update(
            _canonical_tensor(state.tensors[spec.name], spec.name).numpy().tobytes()
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicySnapshot:
    rollout_id: int
    fragment_versions: tuple[int, ...]
    policy_hash: str

    def __post_init__(self) -> None:
        if self.rollout_id < 0:
            raise ValueError("rollout id must be non-negative")
        if len(self.fragment_versions) < 2 or any(
            version < 0 for version in self.fragment_versions
        ):
            raise ValueError("policy snapshot requires non-negative fragment versions")
        if not re.fullmatch(r"[0-9a-f]{64}", self.policy_hash):
            raise ValueError("policy snapshot hash must be a lowercase SHA256")

    @classmethod
    def create(
        cls,
        rollout_id: int,
        state: CanonicalLoraState,
        fragment_versions: Sequence[int],
    ) -> PolicySnapshot:
        return cls(
            rollout_id,
            tuple(fragment_versions),
            policy_tensor_hash(state),
        )

    @property
    def token(self) -> str:
        return f"yeto:{self.rollout_id}:{self.policy_hash}"


def parse_policy_snapshot_token(token: object) -> tuple[int, str]:
    match = re.fullmatch(r"yeto:(0|[1-9][0-9]*):([0-9a-f]{64})", str(token))
    if match is None:
        raise ValueError(f"invalid policy snapshot token {token!r}")
    return int(match.group(1)), match.group(2)


@dataclass(frozen=True)
class LocalRoundStats:
    island_id: int
    local_round_id: int
    base_policy_version: int
    active_groups: int
    completed_groups: int
    cancelled_groups: int
    completed_trajectories: int
    action_tokens: int
    tool_wait_seconds: float
    group_p50_seconds: float
    group_p95_seconds: float
    group_p99_seconds: float
    reward_mean: float
    reward_std: float
    zero_variance_group_ratio: float
    mean_kl: float | None
    ess_ratio: float | None
    clip_fraction: float | None
    delta_l2_norm: float
    rollout_seconds: float
    train_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "island_id",
            "local_round_id",
            "base_policy_version",
            "active_groups",
            "completed_groups",
            "cancelled_groups",
            "completed_trajectories",
            "action_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "reward_mean",
            "reward_std",
            "tool_wait_seconds",
            "group_p50_seconds",
            "group_p95_seconds",
            "group_p99_seconds",
            "zero_variance_group_ratio",
            "delta_l2_norm",
            "rollout_seconds",
            "train_seconds",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("mean_kl", "ess_ratio", "clip_fraction"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
