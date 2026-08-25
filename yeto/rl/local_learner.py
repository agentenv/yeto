"""Role-qualified full-parameter boundary between local RL and DiLoCo."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Protocol

import torch

from ..fragments import MERGE_AVG, Fragment, FragmentLayout
from .contracts import LocalStepReceipt, TrainerUpdateManifest

_ROLES_BY_ALGORITHM = {
    "grpo": frozenset({"actor"}),
    "sao": frozenset({"actor", "critic"}),
}
_PARAMETER_NAME = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,511}\Z")
_SHARD_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, order=True)
class ComponentIdentity:
    """Immutable identity of one trainable algorithm component."""

    role: str
    model_revision: str
    config_hash: str

    def __post_init__(self) -> None:
        if self.role not in {"actor", "critic"}:
            raise ValueError(f"unsupported trainable role: {self.role!r}")
        if not _GIT_REVISION.fullmatch(self.model_revision):
            raise ValueError("component model revision must be an immutable commit")
        if not _SHA256.fullmatch(self.config_hash):
            raise ValueError("component config hash must be a lowercase SHA256")


@dataclass(frozen=True, order=True)
class ParameterSpec:
    """One canonical FP32 parameter at the synchronization boundary."""

    role: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    shard_id: str = "global"

    def __post_init__(self) -> None:
        if self.role not in {"actor", "critic"}:
            raise ValueError(f"unsupported trainable role: {self.role!r}")
        if not _PARAMETER_NAME.fullmatch(self.name) or "::" in self.name:
            raise ValueError(f"invalid canonical parameter name: {self.name!r}")
        if not _SHARD_ID.fullmatch(self.shard_id):
            raise ValueError(f"invalid parameter shard identity: {self.shard_id!r}")
        if not self.shape or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError(f"invalid shape for {self.wire_name!r}: {self.shape}")
        if self.dtype != "float32":
            raise ValueError("synchronization parameter dtype must be float32")
        if math.prod(self.shape) != self.numel:
            raise ValueError(f"shape/numel mismatch for {self.wire_name!r}")

    @property
    def wire_name(self) -> str:
        return f"{self.role}::{self.shard_id}::{self.name}"


@dataclass(frozen=True)
class ParameterLayout:
    """Deterministic role-qualified layout shared by dense and PULSE exchange."""

    algorithm: str
    components: tuple[ComponentIdentity, ...]
    specs: tuple[ParameterSpec, ...]
    fragments: FragmentLayout
    layout_hash: str
    fragment_strategy: str
    _specs_by_wire_name: Mapping[str, ParameterSpec] = field(
        repr=False,
        compare=False,
    )

    @classmethod
    def create(
        cls,
        *,
        algorithm: str,
        components: Sequence[ComponentIdentity],
        specs: Sequence[ParameterSpec],
        num_fragments: int,
        fragment_strategy: str = "balanced",
    ) -> ParameterLayout:
        if algorithm not in _ROLES_BY_ALGORITHM:
            raise ValueError(f"unsupported local RL algorithm: {algorithm!r}")
        ordered_components = tuple(sorted(components))
        ordered_specs = tuple(sorted(specs))
        component_roles = {component.role for component in ordered_components}
        spec_roles = {spec.role for spec in ordered_specs}
        expected_roles = _ROLES_BY_ALGORITHM[algorithm]
        if (
            component_roles != expected_roles
            or spec_roles != expected_roles
            or len(ordered_components) != len(expected_roles)
        ):
            raise ValueError(
                f"{algorithm} requires exactly these trainable roles: "
                f"{sorted(expected_roles)}"
            )
        wire_names = [spec.wire_name for spec in ordered_specs]
        if not wire_names or len(set(wire_names)) != len(wire_names):
            raise ValueError("parameter layout must contain unique parameters")
        if fragment_strategy not in {"balanced", "owner_affine"}:
            raise ValueError("unsupported parameter fragment strategy")
        owners = sorted({(spec.role, spec.shard_id) for spec in ordered_specs})
        minimum_fragments = 1 if fragment_strategy == "balanced" else len(owners)
        if num_fragments < minimum_fragments or num_fragments > len(ordered_specs):
            raise ValueError("parameter fragment count is outside the layout")

        if fragment_strategy == "balanced":
            bins = _binpack_parameter_specs(ordered_specs, num_fragments)
        else:
            bins = _owner_affine_parameter_bins(ordered_specs, num_fragments)
        fragments = FragmentLayout(
            [
                Fragment(
                    merge_mode=MERGE_AVG,
                    tensors=[(spec.wire_name, spec.numel) for spec in members],
                    identity_shapes={spec.wire_name: spec.shape for spec in members},
                )
                for members in bins
            ]
        )
        payload = {
            "schema": 1 if fragment_strategy == "balanced" else 2,
            "algorithm": algorithm,
            "components": [asdict(component) for component in ordered_components],
            "specs": [asdict(spec) for spec in ordered_specs],
            "fragments": [
                [name for name, _ in fragment.tensors]
                for fragment in fragments.fragments
            ],
        }
        if fragment_strategy != "balanced":
            payload["fragment_strategy"] = fragment_strategy
        layout_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            algorithm,
            ordered_components,
            ordered_specs,
            fragments,
            layout_hash,
            fragment_strategy,
            {spec.wire_name: spec for spec in ordered_specs},
        )

    def fragment_specs(self, fragment_id: int) -> tuple[ParameterSpec, ...]:
        if not 0 <= fragment_id < self.fragments.num_fragments:
            raise ValueError("parameter fragment ID is outside the layout")
        return tuple(
            self._specs_by_wire_name[name]
            for name, _ in self.fragments.fragments[fragment_id].tensors
        )

    def fragment_owner(self, fragment_id: int) -> tuple[str, str]:
        """Return the unique role/topology owner of an owner-affine fragment."""

        if self.fragment_strategy != "owner_affine":
            raise ValueError("parameter layout is not owner-affine")
        owners = {
            (spec.role, spec.shard_id) for spec in self.fragment_specs(fragment_id)
        }
        if len(owners) != 1:
            raise RuntimeError("owner-affine parameter fragment mixes owners")
        return next(iter(owners))


def dense_sweep_session_contract_hash(
    layout: ParameterLayout,
    *,
    policy_rounds: int,
    learner_generations: Mapping[int, int],
    training_contract_hash: str = "0" * 64,
) -> bytes:
    """Hash the complete semantic identity of one strict dense-H=1 run."""

    if (
        isinstance(policy_rounds, bool)
        or not isinstance(policy_rounds, int)
        or policy_rounds < 1
    ):
        raise ValueError("dense policy rounds must be positive")
    if not isinstance(learner_generations, Mapping) or not learner_generations:
        raise ValueError("dense learner generations must be a non-empty mapping")
    if (
        not isinstance(training_contract_hash, str)
        or len(training_contract_hash) != 64
        or any(value not in "0123456789abcdef" for value in training_contract_hash)
    ):
        raise ValueError("dense training contract hash must be a lowercase SHA256")
    roster = []
    for learner_id in sorted(learner_generations):
        generation = learner_generations[learner_id]
        if (
            isinstance(learner_id, bool)
            or not isinstance(learner_id, int)
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or learner_id < 0
            or generation < 0
        ):
            raise ValueError("dense learner roster entries must be non-negative integers")
        roster.append({"learner_id": learner_id, "generation": generation})
    if [entry["learner_id"] for entry in roster] != list(range(len(roster))):
        raise ValueError("dense learner roster must contain contiguous IDs from zero")

    payload = {
        "schema": 2,
        "parameter_layout_hash": layout.layout_hash,
        "training_contract_hash": training_contract_hash,
        "components": [asdict(component) for component in layout.components],
        "profile": {
            "policy_sweep_fragments": layout.fragments.num_fragments,
            "policy_rounds": policy_rounds,
            "wire_dtype": "fp32",
            "outer_lr": 1,
            "outer_momentum": 0,
            "learner_weight": "equal",
            "quorum": "full",
            "membership": "fixed",
            "pipeline": 1,
            "local_horizon": 1,
            "delta_correction": "none",
            "learner_generations": roster,
        },
    }
    return hashlib.sha256(
        b"yeto-dense-sweep-session-v1\0"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()


def _binpack_parameter_specs(
    specs: Sequence[ParameterSpec],
    num_fragments: int,
) -> list[list[ParameterSpec]]:
    bins: list[list[ParameterSpec]] = [[] for _ in range(num_fragments)]
    sizes = [0] * num_fragments
    for spec in sorted(
        specs,
        key=lambda value: (
            -value.numel,
            value.role,
            value.shard_id,
            value.name,
        ),
    ):
        fragment_id = min(range(num_fragments), key=lambda index: sizes[index])
        bins[fragment_id].append(spec)
        sizes[fragment_id] += spec.numel
    return bins


def _owner_affine_parameter_bins(
    specs: Sequence[ParameterSpec],
    num_fragments: int,
) -> list[list[ParameterSpec]]:
    """Allocate deterministic balanced bins without crossing topology owners."""

    by_owner: dict[tuple[str, str], list[ParameterSpec]] = {}
    for spec in specs:
        by_owner.setdefault((spec.role, spec.shard_id), []).append(spec)
    owners = tuple(sorted(by_owner))
    counts = {owner: 1 for owner in owners}
    totals = {owner: sum(spec.numel for spec in by_owner[owner]) for owner in owners}
    remaining = num_fragments - len(owners)
    while remaining:
        eligible = [owner for owner in owners if counts[owner] < len(by_owner[owner])]
        if not eligible:
            raise RuntimeError("owner-affine fragment allocation is incomplete")
        # Split the owner with the largest current average bin. ``owners`` is
        # already canonical, and max() retains the first item on an exact tie.
        owner = max(
            eligible,
            key=lambda value: Fraction(totals[value], counts[value]),
        )
        counts[owner] += 1
        remaining -= 1

    bins: list[list[ParameterSpec]] = []
    for owner in owners:
        bins.extend(_contiguous_parameter_bins(by_owner[owner], counts[owner]))
    if len(bins) != num_fragments or any(not members for members in bins):
        raise RuntimeError("owner-affine fragment allocation produced empty bins")
    return bins


def _contiguous_parameter_bins(
    specs: Sequence[ParameterSpec],
    num_fragments: int,
) -> list[list[ParameterSpec]]:
    """Split canonical owner specs into balanced contiguous fragments.

    Owner ranks stream and apply their fragments independently.  Keeping the
    concatenated wire-name order identical to the rank's canonical manifest
    lets Miles validate that a plan is both exact and replayable without a
    second name-index structure.
    """

    ordered = sorted(specs)
    bins = []
    cursor = 0
    remaining_total = sum(spec.numel for spec in ordered)
    for fragment_index in range(num_fragments):
        fragments_left = num_fragments - fragment_index
        if fragments_left == 1:
            bins.append(ordered[cursor:])
            break
        target = Fraction(remaining_total, fragments_left)
        start = cursor
        current = 0
        # Retain at least one spec for every later fragment.  Choose the
        # nearest side of the target at each canonical boundary.
        while cursor < len(ordered) - (fragments_left - 1):
            next_size = ordered[cursor].numel
            if current and abs(current - target) <= abs(current + next_size - target):
                break
            current += next_size
            cursor += 1
        if cursor == start:
            current += ordered[cursor].numel
            cursor += 1
        bins.append(ordered[start:cursor])
        remaining_total -= current
    if len(bins) != num_fragments or any(not members for members in bins):
        raise RuntimeError("contiguous parameter fragmentation produced empty bins")
    return bins


@dataclass(frozen=True)
class ParameterFragmentCut:
    fragment_id: int
    flat: torch.Tensor = field(repr=False, compare=False)
    payload_hash: str


@dataclass(frozen=True)
class ParameterCut:
    """One complete immutable policy cut at a safe learner boundary."""

    policy_version: int
    layout_hash: str
    fragments: tuple[ParameterFragmentCut, ...]
    policy_hash: str


@dataclass(frozen=True)
class DenseUpdateFragment:
    fragment_id: int
    target_minus_base: torch.Tensor = field(repr=False, compare=False)
    payload_hash: str
    l2_norm: float


@dataclass(frozen=True)
class DenseTrainerUpdate:
    """Complete dense target-minus-base update emitted by a local learner."""

    manifest: TrainerUpdateManifest
    receipt: LocalStepReceipt
    fragments: tuple[DenseUpdateFragment, ...]


class FullParameterProvider(Protocol):
    """Miles-facing provider; implementations own sharding and safe application."""

    def at_safe_boundary(self) -> bool: ...

    def read_parameters(
        self,
        specs: tuple[ParameterSpec, ...],
    ) -> Mapping[str, torch.Tensor]: ...

    def apply_complete_cut(
        self,
        layout: ParameterLayout,
        cut: ParameterCut,
    ) -> None: ...


def _flat_fragment(
    values: Mapping[str, torch.Tensor],
    specs: tuple[ParameterSpec, ...],
) -> torch.Tensor:
    expected = {spec.wire_name for spec in specs}
    if set(values) != expected:
        raise ValueError("provider parameter names do not match the fragment")
    flat = torch.empty(sum(spec.numel for spec in specs), dtype=torch.float32)
    offset = 0
    for spec in specs:
        value = values[spec.wire_name]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{spec.wire_name!r} must be a floating-point tensor")
        canonical = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(canonical.shape) != spec.shape:
            raise ValueError(f"shape mismatch for {spec.wire_name!r}")
        if not torch.isfinite(canonical).all().item():
            raise ValueError(f"{spec.wire_name!r} contains NaN or Inf")
        flat[offset : offset + spec.numel].copy_(canonical.reshape(-1))
        offset += spec.numel
    return flat


def _tensor_hash(domain: bytes, fragment_id: int, value: torch.Tensor) -> str:
    canonical = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(fragment_id.to_bytes(8, "little"))
    digest.update(memoryview(canonical.numpy()).cast("B"))
    return digest.hexdigest()


def _cut_hash(
    layout_hash: str,
    policy_version: int,
    fragments: Sequence[ParameterFragmentCut],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"yeto-full-parameter-cut-v1\0")
    digest.update(layout_hash.encode("ascii"))
    digest.update(policy_version.to_bytes(8, "little"))
    for fragment in fragments:
        digest.update(bytes.fromhex(fragment.payload_hash))
    return digest.hexdigest()


def _dense_payload_hash(fragments: Sequence[DenseUpdateFragment]) -> str:
    digest = hashlib.sha256()
    digest.update(b"yeto-dense-trainer-update-v1\0")
    for fragment in fragments:
        digest.update(bytes.fromhex(fragment.payload_hash))
    return digest.hexdigest()


def make_parameter_cut(
    layout: ParameterLayout,
    *,
    policy_version: int,
    values: Mapping[str, torch.Tensor],
) -> ParameterCut:
    """Create one complete canonical cut from role/shard-qualified values."""

    if (
        isinstance(policy_version, bool)
        or not isinstance(policy_version, int)
        or policy_version < 0
    ):
        raise ValueError("parameter-cut policy version must be non-negative")
    expected_names = {spec.wire_name for spec in layout.specs}
    if set(values) != expected_names:
        raise ValueError("parameter-cut values do not match the complete layout")
    fragments = []
    for fragment_id in range(layout.fragments.num_fragments):
        specs = layout.fragment_specs(fragment_id)
        flat = _flat_fragment(
            {spec.wire_name: values[spec.wire_name] for spec in specs},
            specs,
        )
        payload_hash = _tensor_hash(b"yeto-parameter-fragment-v1\0", fragment_id, flat)
        fragments.append(ParameterFragmentCut(fragment_id, flat, payload_hash))
    return ParameterCut(
        policy_version,
        layout.layout_hash,
        tuple(fragments),
        _cut_hash(layout.layout_hash, policy_version, fragments),
    )


def parameter_cut_from_fragment_flats(
    layout: ParameterLayout,
    *,
    policy_version: int,
    fragments: Mapping[int, torch.Tensor],
) -> ParameterCut:
    """Build a complete cut from already-flat fragment payloads.

    Unlike :func:`make_parameter_cut`, this path never reconstructs named
    tensors and then flattens them again.  It is the materialized reference
    backend for lossless syncer broadcasts and the byte-equivalence oracle for
    the reference-backed Miles transport.
    """

    if (
        isinstance(policy_version, bool)
        or not isinstance(policy_version, int)
        or policy_version < 0
    ):
        raise ValueError("parameter-cut policy version must be non-negative")
    expected_ids = set(range(layout.fragments.num_fragments))
    if set(fragments) != expected_ids:
        raise ValueError("parameter-cut fragments do not cover the complete layout")
    cuts = []
    for fragment_id in range(layout.fragments.num_fragments):
        value = fragments[fragment_id]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("parameter-cut fragment must be a floating-point tensor")
        flat = (
            value.detach()
            .reshape(-1)
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        if (
            flat.numel() != layout.fragments.fragments[fragment_id].numel
            or not torch.isfinite(flat).all().item()
        ):
            raise ValueError("parameter-cut fragment payload is malformed")
        payload_hash = _tensor_hash(
            b"yeto-parameter-fragment-v1\0",
            fragment_id,
            flat,
        )
        cuts.append(ParameterFragmentCut(fragment_id, flat, payload_hash))
    return ParameterCut(
        policy_version,
        layout.layout_hash,
        tuple(cuts),
        _cut_hash(layout.layout_hash, policy_version, cuts),
    )


def parameter_values(
    layout: ParameterLayout,
    cut: ParameterCut,
) -> dict[str, torch.Tensor]:
    """Materialize a validated complete cut as named CPU FP32 tensors."""

    _validate_parameter_cut(layout, cut)
    values = {}
    for fragment in cut.fragments:
        offset = 0
        for spec in layout.fragment_specs(fragment.fragment_id):
            values[spec.wire_name] = (
                fragment.flat[offset : offset + spec.numel].reshape(spec.shape).clone()
            )
            offset += spec.numel
    return values


def advance_parameter_cut_version(
    layout: ParameterLayout,
    cut: ParameterCut,
    *,
    target_policy_version: int,
) -> ParameterCut:
    """Promote an unchanged complete payload to a newer committed policy.

    This is the exact one-learner DiLoCo merge result: the accepted local cut
    becomes the next global policy without copying its immutable fragments.
    """

    _validate_parameter_cut(layout, cut)
    if (
        isinstance(target_policy_version, bool)
        or not isinstance(target_policy_version, int)
        or target_policy_version <= cut.policy_version
    ):
        raise ValueError("promoted parameter cut must advance its policy version")
    return ParameterCut(
        target_policy_version,
        layout.layout_hash,
        cut.fragments,
        _cut_hash(layout.layout_hash, target_policy_version, cut.fragments),
    )


def _validate_parameter_cut(layout: ParameterLayout, cut: ParameterCut) -> None:
    if cut.layout_hash != layout.layout_hash:
        raise ValueError("parameter cut layout identity changed")
    if [fragment.fragment_id for fragment in cut.fragments] != list(
        range(layout.fragments.num_fragments)
    ):
        raise ValueError("parameter cut is not a complete ordered fragment sweep")
    for fragment in cut.fragments:
        specs = layout.fragment_specs(fragment.fragment_id)
        expected = sum(spec.numel for spec in specs)
        if (
            fragment.flat.device.type != "cpu"
            or fragment.flat.dtype != torch.float32
            or not fragment.flat.is_contiguous()
            or fragment.flat.numel() != expected
            or not torch.isfinite(fragment.flat).all().item()
            or fragment.payload_hash
            != _tensor_hash(
                b"yeto-parameter-fragment-v1\0",
                fragment.fragment_id,
                fragment.flat,
            )
        ):
            raise ValueError("parameter cut fragment is malformed")
    if cut.policy_hash != _cut_hash(
        cut.layout_hash,
        cut.policy_version,
        cut.fragments,
    ):
        raise ValueError("parameter cut policy hash is malformed")


def make_dense_trainer_update(
    layout: ParameterLayout,
    anchor: ParameterCut,
    local: ParameterCut,
    receipt: LocalStepReceipt,
    *,
    learner_id: int,
    learner_generation: int,
    target_policy_version: int,
) -> DenseTrainerUpdate:
    """Encode one complete local-minus-anchor dense DiLoCo update."""

    _validate_parameter_cut(layout, anchor)
    _validate_parameter_cut(layout, local)
    if local.policy_version != anchor.policy_version:
        raise ValueError("local cut must retain its global anchor version")
    if (
        receipt.algorithm != layout.algorithm
        or receipt.learner_id != learner_id
        or receipt.learner_generation != learner_generation
        or receipt.base_policy_version != anchor.policy_version
        or receipt.base_policy_hash != anchor.policy_hash
        or receipt.parameter_layout_hash != layout.layout_hash
        or not receipt.optimizer_step_succeeded
    ):
        raise ValueError("local-step receipt does not bind this learner anchor")
    if target_policy_version <= anchor.policy_version:
        raise ValueError("dense update target must follow its anchor")
    updates = []
    payload_bytes = 0
    for base_fragment, local_fragment in zip(
        anchor.fragments,
        local.fragments,
        strict=True,
    ):
        delta = local_fragment.flat - base_fragment.flat
        if not torch.isfinite(delta).all().item():
            raise ValueError("dense trainer update contains NaN or Inf")
        payload_hash = _tensor_hash(
            b"yeto-dense-update-fragment-v1\0",
            base_fragment.fragment_id,
            delta,
        )
        payload_bytes += delta.numel() * delta.element_size()
        updates.append(
            DenseUpdateFragment(
                base_fragment.fragment_id,
                delta,
                payload_hash,
                float(delta.norm().item()),
            )
        )
    manifest = TrainerUpdateManifest(
        exchange_mode="dense",
        learner_id=learner_id,
        learner_generation=learner_generation,
        base_policy_version=anchor.policy_version,
        target_policy_version=target_policy_version,
        parameter_layout_hash=layout.layout_hash,
        payload_hash=_dense_payload_hash(updates),
        payload_bytes=payload_bytes,
        fragment_count=len(updates),
        complete=True,
    )
    return DenseTrainerUpdate(manifest, receipt, tuple(updates))


def _validate_dense_update(
    layout: ParameterLayout,
    update: DenseTrainerUpdate,
) -> None:
    if [fragment.fragment_id for fragment in update.fragments] != list(
        range(layout.fragments.num_fragments)
    ):
        raise ValueError("dense update is not a complete ordered fragment sweep")
    payload_bytes = 0
    for fragment in update.fragments:
        specs = layout.fragment_specs(fragment.fragment_id)
        expected = sum(spec.numel for spec in specs)
        value = fragment.target_minus_base
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or value.numel() != expected
            or not torch.isfinite(value).all().item()
            or fragment.payload_hash
            != _tensor_hash(
                b"yeto-dense-update-fragment-v1\0",
                fragment.fragment_id,
                value,
            )
            or not math.isclose(
                fragment.l2_norm,
                float(value.norm().item()),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("dense update fragment is malformed")
        payload_bytes += value.numel() * value.element_size()
    if (
        update.manifest.fragment_count != len(update.fragments)
        or update.manifest.payload_bytes != payload_bytes
        or update.manifest.payload_hash != _dense_payload_hash(update.fragments)
    ):
        raise ValueError("dense update manifest is malformed")


def reconstruct_dense_target(
    layout: ParameterLayout,
    anchor: ParameterCut,
    update: DenseTrainerUpdate,
) -> ParameterCut:
    """Reconstruct and hash a complete target cut from a dense update."""

    _validate_parameter_cut(layout, anchor)
    if (
        update.manifest.exchange_mode != "dense"
        or update.manifest.base_policy_version != anchor.policy_version
        or update.manifest.parameter_layout_hash != layout.layout_hash
        or len(update.fragments) != len(anchor.fragments)
    ):
        raise ValueError("dense update does not bind the supplied anchor")
    _validate_dense_update(layout, update)
    fragments = []
    for base_fragment, delta_fragment in zip(
        anchor.fragments,
        update.fragments,
        strict=True,
    ):
        if base_fragment.fragment_id != delta_fragment.fragment_id:
            raise ValueError("dense update fragment order changed")
        flat = base_fragment.flat + delta_fragment.target_minus_base
        payload_hash = _tensor_hash(
            b"yeto-parameter-fragment-v1\0",
            base_fragment.fragment_id,
            flat,
        )
        fragments.append(
            ParameterFragmentCut(base_fragment.fragment_id, flat, payload_hash)
        )
    policy_version = update.manifest.target_policy_version
    return ParameterCut(
        policy_version,
        layout.layout_hash,
        tuple(fragments),
        _cut_hash(layout.layout_hash, policy_version, fragments),
    )


class DenseDiLoCoConnector:
    """Exports dense local deltas without coupling DiLoCo to GRPO or SAO."""

    def __init__(
        self,
        provider: FullParameterProvider,
        layout: ParameterLayout,
        *,
        learner_id: int,
        learner_generation: int,
        initial_policy_version: int,
    ) -> None:
        for name, value in (
            ("learner_id", learner_id),
            ("learner_generation", learner_generation),
            ("initial_policy_version", initial_policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.provider = provider
        self.layout = layout
        self.learner_id = learner_id
        self.learner_generation = learner_generation
        self.current_policy_version = initial_policy_version

    def capture(self, *, policy_version: int | None = None) -> ParameterCut:
        self._require_safe_boundary()
        version = (
            self.current_policy_version if policy_version is None else policy_version
        )
        if version != self.current_policy_version:
            raise ValueError("capture version does not match the installed policy")
        values = {}
        for fragment_id in range(self.layout.fragments.num_fragments):
            specs = self.layout.fragment_specs(fragment_id)
            values.update(self.provider.read_parameters(specs))
        return make_parameter_cut(
            self.layout,
            policy_version=version,
            values=values,
        )

    def export_dense_update(
        self,
        anchor: ParameterCut,
        receipt: LocalStepReceipt,
        *,
        target_policy_version: int,
    ) -> DenseTrainerUpdate:
        self._validate_cut(anchor)
        if (
            receipt.algorithm != self.layout.algorithm
            or receipt.learner_id != self.learner_id
            or receipt.learner_generation != self.learner_generation
            or receipt.base_policy_version != anchor.policy_version
            or receipt.base_policy_hash != anchor.policy_hash
            or receipt.parameter_layout_hash != self.layout.layout_hash
            or not receipt.optimizer_step_succeeded
        ):
            raise ValueError("local-step receipt does not bind this learner anchor")
        if target_policy_version <= anchor.policy_version:
            raise ValueError("dense update target must follow its anchor")
        local = self.capture(policy_version=anchor.policy_version)
        return self.export_dense_update_from_cut(
            anchor,
            local,
            receipt,
            target_policy_version=target_policy_version,
        )

    def export_dense_update_from_cut(
        self,
        anchor: ParameterCut,
        local: ParameterCut,
        receipt: LocalStepReceipt,
        *,
        target_policy_version: int,
    ) -> DenseTrainerUpdate:
        """Export a dense update from two already captured safe-boundary cuts."""

        return make_dense_trainer_update(
            self.layout,
            anchor,
            local,
            receipt,
            learner_id=self.learner_id,
            learner_generation=self.learner_generation,
            target_policy_version=target_policy_version,
        )

    def reconstruct_target(
        self,
        anchor: ParameterCut,
        update: DenseTrainerUpdate,
    ) -> ParameterCut:
        return reconstruct_dense_target(self.layout, anchor, update)

    def apply_global_cut(
        self,
        cut: ParameterCut,
        *,
        expected_base_version: int,
    ) -> None:
        self._require_safe_boundary()
        self._validate_cut(cut)
        if expected_base_version != self.current_policy_version:
            raise ValueError("global cut base version is stale")
        if cut.policy_version <= expected_base_version:
            raise ValueError("global cut must advance the installed policy")
        self.provider.apply_complete_cut(self.layout, cut)
        self.current_policy_version = cut.policy_version

    def _require_safe_boundary(self) -> None:
        if not self.provider.at_safe_boundary():
            raise RuntimeError("parameter synchronization requires a safe boundary")

    def _validate_cut(self, cut: ParameterCut) -> None:
        _validate_parameter_cut(self.layout, cut)

    def _validate_update(self, update: DenseTrainerUpdate) -> None:
        _validate_dense_update(self.layout, update)
