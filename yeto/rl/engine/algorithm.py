"""``AlgorithmSpec``: yeto-owned algorithm description with a canonical SHA256.

Pure Python; no torch/ray/miles imports. The spec is translated to engine
arguments only by adapters; ``to_legacy_argv`` documents the legacy mapping
used by ``yeto.rl.learner.build_miles_argv``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

ALGORITHM_SPEC_SCHEMA = "yeto-rl-algorithm-spec-v1"
BOUNDED_NONZERO_STD_FILTER = "yeto.rl.filters.bounded_nonzero_reward_std"
STOCK_NONZERO_STD_FILTER = (
    "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
)
SUPPORTED_ADVANTAGE_ESTIMATORS = frozenset({"grpo"})
SUPPORTED_LOSSES = frozenset({"policy_loss"})
SUPPORTED_FILTERS = frozenset({BOUNDED_NONZERO_STD_FILTER})


class AlgorithmSpecError(ValueError):
    """An algorithm description is malformed or unsupported."""


@dataclass(frozen=True)
class AlgorithmSpec:
    advantage_estimator: str = "grpo"
    loss: str = "policy_loss"
    kl_coef: float | None = None
    dynamic_sampling_filter: str | None = None
    dynamic_sampling_max_replacements: int | None = None

    def __post_init__(self) -> None:
        if self.advantage_estimator not in SUPPORTED_ADVANTAGE_ESTIMATORS:
            raise AlgorithmSpecError(
                f"unsupported advantage estimator {self.advantage_estimator!r}; "
                f"supported: {sorted(SUPPORTED_ADVANTAGE_ESTIMATORS)}"
            )
        if self.loss not in SUPPORTED_LOSSES:
            raise AlgorithmSpecError(
                f"unsupported loss {self.loss!r}; supported: {sorted(SUPPORTED_LOSSES)}"
            )
        if self.kl_coef is not None:
            if isinstance(self.kl_coef, bool) or not isinstance(self.kl_coef, (int, float)):
                raise AlgorithmSpecError("kl_coef must be a number")
            if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
                raise AlgorithmSpecError("kl_coef must be finite and non-negative")
            object.__setattr__(self, "kl_coef", float(self.kl_coef))
        if (
            self.dynamic_sampling_filter is not None
            and self.dynamic_sampling_filter not in SUPPORTED_FILTERS
        ):
            raise AlgorithmSpecError(
                f"unsupported dynamic sampling filter {self.dynamic_sampling_filter!r}"
            )
        limit = self.dynamic_sampling_max_replacements
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise AlgorithmSpecError(
                    "dynamic_sampling_max_replacements must be a non-negative int"
                )
            if self.dynamic_sampling_filter is None:
                raise AlgorithmSpecError(
                    "dynamic_sampling_max_replacements requires a dynamic sampling filter"
                )

    # -- identity ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"schema": ALGORITHM_SPEC_SCHEMA, **asdict(self)}

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AlgorithmSpec":
        if not isinstance(payload, Mapping):
            raise AlgorithmSpecError("algorithm spec must be an object")
        data = dict(payload)
        schema = data.pop("schema", ALGORITHM_SPEC_SCHEMA)
        if schema != ALGORITHM_SPEC_SCHEMA:
            raise AlgorithmSpecError(f"unknown algorithm spec schema {schema!r}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise AlgorithmSpecError(f"unknown algorithm spec fields: {unknown}")
        return cls(**data)

    # -- legacy mapping ---------------------------------------------------
    @classmethod
    def from_legacy_args(cls, args: Any) -> "AlgorithmSpec":
        """Build the spec equivalent to a legacy learner/launcher namespace.

        Legacy always uses GRPO (``--advantage-estimator grpo``); the stock
        nonzero-std filter is normalized to the bounded one exactly as
        ``yeto.launcher`` does when a replacement bound is set.
        """

        flt = getattr(args, "dynamic_sampling_filter_path", None)
        limit = getattr(args, "dynamic_sampling_max_replacements", None)
        if flt == STOCK_NONZERO_STD_FILTER and limit is not None:
            flt = BOUNDED_NONZERO_STD_FILTER
        return cls(
            advantage_estimator="grpo",
            kl_coef=getattr(args, "kl_coef", None),
            dynamic_sampling_filter=flt,
            dynamic_sampling_max_replacements=limit,
        )

    def to_legacy_argv(self) -> list[str]:
        """Miles argv fragment emitted by legacy ``build_miles_argv`` for this spec."""

        argv = ["--advantage-estimator", self.advantage_estimator]
        if self.kl_coef is not None:
            argv += ["--kl-coef", str(self.kl_coef)]
        if self.dynamic_sampling_filter is not None:
            argv += ["--dynamic-sampling-filter-path", self.dynamic_sampling_filter]
        return argv

    def to_legacy_runtime_attrs(self) -> dict[str, Any]:
        """Attributes legacy sets on the Miles namespace (read by the filter)."""

        return {
            "yeto_rl_dynamic_sampling_max_replacements": (
                self.dynamic_sampling_max_replacements
            )
        }
