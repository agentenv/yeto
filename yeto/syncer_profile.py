"""Canonical cross-language identity for a Yeto syncer launch.

The syncer hashes the same fixed-width encoding from its parsed ``Config``.
SAO clients carry that digest in HELLO, so a runtime contract describing one
schedule cannot silently connect to a server launched with another.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

SYNCER_SEMANTIC_PROFILE_SCHEMA = "yeto.syncer-semantic-profile.v1"
_DOMAIN = b"yeto-syncer-semantic-profile-v1\0"
_FIELDS = {
    "schema",
    "learners",
    "quorum",
    "grace_ms",
    "grace_gamma",
    "grace_tau",
    "pipeline",
    "min_round_interval_ms",
    "sync_interval_steps",
    "delta_correction",
    "quorum_timeout_s",
    "final_ack_timeout_s",
    "total_steps",
    "policy_sweep_fragments",
    "outer_lr",
    "outer_momentum",
    "checkpoint_enabled",
    "checkpoint_every",
    "resume",
    "mark_final_checkpoint",
    "learner_budget_steps",
    "max_base_lag",
    "learner_weight",
    "require_profile_binding",
}


def _integer(name: str, value: object, *, bits: int, minimum: int = 0) -> int:
    maximum = (1 << bits) - 1
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"syncer profile {name} must be in [{minimum}, {maximum}]")
    return value


def _optional_integer(
    name: str,
    value: object,
    *,
    bits: int,
    minimum: int = 0,
) -> int | None:
    if value is None:
        return None
    return _integer(name, value, bits=bits, minimum=minimum)


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"syncer profile {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"syncer profile {name} must be finite")
    return result


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"syncer profile {name} must be boolean")
    return value


def _f32(name: str, value: object) -> float:
    result = _number(name, value)
    try:
        result = struct.unpack("<f", struct.pack("<f", result))[0]
    except OverflowError as error:
        raise ValueError(f"syncer profile {name} must fit f32") from error
    if not math.isfinite(result):
        raise ValueError(f"syncer profile {name} must fit finite f32")
    return result


def _encode_optional(encoded: bytearray, value: int | None, fmt: str) -> None:
    encoded.append(value is not None)
    if value is not None:
        encoded.extend(struct.pack(fmt, value))


@dataclass(frozen=True)
class SyncerSemanticProfile:
    """Parsed server semantics, excluding ports and concrete output paths."""

    learners: int
    quorum: int
    grace_ms: int
    grace_gamma: float
    grace_tau: float
    pipeline: int
    min_round_interval_ms: int
    sync_interval_steps: float
    delta_correction: str
    quorum_timeout_s: int
    final_ack_timeout_s: int
    total_steps: int
    policy_sweep_fragments: int | None
    outer_lr: float
    outer_momentum: float
    checkpoint_enabled: bool
    checkpoint_every: int
    resume: bool
    mark_final_checkpoint: bool
    learner_budget_steps: int | None
    max_base_lag: int | None
    learner_weight: str
    require_profile_binding: bool

    @classmethod
    def from_mapping(cls, raw: object) -> SyncerSemanticProfile:
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError(
                "syncer semantic profile fields do not match the v1 schema"
            )
        if raw["schema"] != SYNCER_SEMANTIC_PROFILE_SCHEMA:
            raise ValueError("unsupported syncer semantic profile schema")
        delta_correction = raw["delta_correction"]
        if delta_correction not in {"heloco", "none"}:
            raise ValueError("syncer profile delta_correction is invalid")
        learner_weight = raw["learner_weight"]
        if learner_weight not in {"tokens2-over-steps", "equal"}:
            raise ValueError("syncer profile learner_weight is invalid")
        profile = cls(
            learners=_integer("learners", raw["learners"], bits=32, minimum=1),
            quorum=_integer("quorum", raw["quorum"], bits=32, minimum=1),
            grace_ms=_integer("grace_ms", raw["grace_ms"], bits=64),
            grace_gamma=_number("grace_gamma", raw["grace_gamma"]),
            grace_tau=_number("grace_tau", raw["grace_tau"]),
            pipeline=_integer("pipeline", raw["pipeline"], bits=32, minimum=1),
            min_round_interval_ms=_integer(
                "min_round_interval_ms", raw["min_round_interval_ms"], bits=64
            ),
            sync_interval_steps=_number(
                "sync_interval_steps", raw["sync_interval_steps"]
            ),
            delta_correction=delta_correction,
            quorum_timeout_s=_integer(
                "quorum_timeout_s", raw["quorum_timeout_s"], bits=64, minimum=1
            ),
            final_ack_timeout_s=_integer(
                "final_ack_timeout_s",
                raw["final_ack_timeout_s"],
                bits=64,
                minimum=1,
            ),
            total_steps=_integer("total_steps", raw["total_steps"], bits=64, minimum=1),
            policy_sweep_fragments=_optional_integer(
                "policy_sweep_fragments",
                raw["policy_sweep_fragments"],
                bits=32,
                minimum=1,
            ),
            outer_lr=_f32("outer_lr", raw["outer_lr"]),
            outer_momentum=_f32("outer_momentum", raw["outer_momentum"]),
            checkpoint_enabled=_boolean(
                "checkpoint_enabled", raw["checkpoint_enabled"]
            ),
            checkpoint_every=_integer(
                "checkpoint_every", raw["checkpoint_every"], bits=64
            ),
            resume=_boolean("resume", raw["resume"]),
            mark_final_checkpoint=_boolean(
                "mark_final_checkpoint", raw["mark_final_checkpoint"]
            ),
            learner_budget_steps=_optional_integer(
                "learner_budget_steps",
                raw["learner_budget_steps"],
                bits=64,
                minimum=1,
            ),
            max_base_lag=_optional_integer(
                "max_base_lag", raw["max_base_lag"], bits=64
            ),
            learner_weight=learner_weight,
            require_profile_binding=_boolean(
                "require_profile_binding", raw["require_profile_binding"]
            ),
        )
        profile._validate_general()
        return profile

    def _validate_general(self) -> None:
        if self.quorum > self.learners:
            raise ValueError("syncer profile quorum exceeds learners")
        if not 0.0 <= self.grace_gamma < 1.0 or self.grace_tau < 0.0:
            raise ValueError("syncer profile adaptive grace values are invalid")
        if self.sync_interval_steps < 0.0:
            raise ValueError("syncer profile sync_interval_steps is negative")
        if self.outer_lr <= 0.0 or not 0.0 <= self.outer_momentum < 1.0:
            raise ValueError("syncer profile outer optimizer values are invalid")
        if self.mark_final_checkpoint and not self.checkpoint_enabled:
            raise ValueError("syncer profile final marker requires checkpointing")
        if self.resume and not self.checkpoint_enabled:
            raise ValueError("syncer profile resume requires checkpointing")
        if self.policy_sweep_fragments is not None:
            fragments = self.policy_sweep_fragments
            if self.pipeline != 1 or self.total_steps % fragments:
                raise ValueError("syncer profile policy sweep schedule is invalid")

    def validate_sao(self) -> None:
        """Enforce the paired actor/critic fail-closed server profile."""

        if self.policy_sweep_fragments is not None:
            raise ValueError("SAO syncer profile rejects legacy policy-sweep mode")
        if self.learner_budget_steps is not None:
            raise ValueError("SAO syncer profile rejects learner-budget cutoff mode")
        if not self.require_profile_binding:
            raise ValueError("SAO syncer profile must require HELLO profile binding")
        if self.quorum != self.learners:
            raise ValueError("SAO syncer profile requires quorum equal to learners")
        if self.grace_ms != 0:
            raise ValueError("SAO syncer profile requires grace_ms=0")
        if not self.checkpoint_enabled or self.checkpoint_every != 1 or not self.resume:
            raise ValueError(
                "SAO syncer profile requires checkpointing every round with resume"
            )
        if self.max_base_lag != 0:
            raise ValueError("SAO syncer profile requires max_base_lag=0")

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(_DOMAIN)
        encoded.extend(struct.pack("<IIQ", self.learners, self.quorum, self.grace_ms))
        encoded.extend(struct.pack("<dd", self.grace_gamma, self.grace_tau))
        encoded.extend(
            struct.pack(
                "<IQd",
                self.pipeline,
                self.min_round_interval_ms,
                self.sync_interval_steps,
            )
        )
        encoded.append(self.delta_correction == "heloco")
        encoded.extend(
            struct.pack(
                "<QQQ",
                self.quorum_timeout_s,
                self.final_ack_timeout_s,
                self.total_steps,
            )
        )
        _encode_optional(encoded, self.policy_sweep_fragments, "<I")
        encoded.extend(struct.pack("<ff", self.outer_lr, self.outer_momentum))
        encoded.append(self.checkpoint_enabled)
        encoded.extend(struct.pack("<Q", self.checkpoint_every))
        encoded.append(self.resume)
        encoded.append(self.mark_final_checkpoint)
        _encode_optional(encoded, self.learner_budget_steps, "<Q")
        _encode_optional(encoded, self.max_base_lag, "<Q")
        encoded.append(0 if self.learner_weight == "tokens2-over-steps" else 1)
        encoded.append(self.require_profile_binding)
        return bytes(encoded)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
