"""yeto reward post-processing dispatcher (change ``rl-algo-grpo-knobs``, design D4/D5).

This is the only ``--custom-reward-post-process-path`` of the ports path::

    --custom-reward-post-process-path yeto.rl.algos.reward_pipeline.post_process

Miles (``miles/ray/rollout/train_data_conversion.py::_post_process_rewards``)
calls ``post_process(args, samples)`` *instead of* its built-in reward
normalization, so the dispatcher re-implements that built-in path exactly
(``grpo_default``) and composes it with yeto stages::

    post_process(args, samples)
      cfg    = pipeline_config(args)                 # args.yeto_algo_plugins (D5), hash-checked
      raw    = [s.get_reward_value(args) for s]      # as Miles
      shaped = raw
      for stage in cfg.reward_shapers:               # REWARD_SHAPERS, in order
          shaped = stage(args, samples, shaped, params)
      groups = reward_groups(args, samples)          # _reward_group_segments w/o prompt_group_sizes
      adv    = ADV_TRANSFORMS[cfg.advantage_transform](args, samples, shaped, groups, params)
      return shaped, adv

With no shaper and ``grpo_default`` the output is element-wise identical to
Miles' built-in ``_post_process_rewards`` (``tests/test_rl_reward_pipeline_equivalence.py``
compares with ``torch.equal`` and pins the source hash of
``train_data_conversion.py``; a Miles upgrade fails the test and asks for a
re-review).

Frozen interface (consumed by ``rl-algo-seq-and-adv``)
------------------------------------------------------

* ``post_process(args, samples) -> (rewards, advantages_input)``: the Miles
  entry point (both lists have ``len(samples)`` entries).
* ``REWARD_SHAPERS: dict[str, StageDef]`` and ``ADV_TRANSFORMS: dict[str, StageDef]``:
  module-level registries. A stage is
  ``fn(args, samples, rewards, groups, params) -> list[float]`` for
  transforms and ``fn(args, samples, rewards, params) -> list[float]`` for
  shapers; ``params`` is the stage's JSON parameters from the algorithm spec
  (without ``name``).
* ``register_reward_shaper(name, fn, validate=...)`` /
  ``register_advantage_transform(name, fn, validate=...)``: ``validate(params,
  spec) -> list[str]`` returns problems (field-named) checked before any GPU
  process (``yeto.rl.algos.grpo_knobs`` calls it from the spec rejection
  matrix).
* ``reward_groups(args, samples) -> list[list[int]]`` and
  ``rollout_segments(samples, rows) -> list[(rollout_key, rows)]``: the Miles
  grouping and multi-segment merging, for transforms that need them.
* ``PIPELINE_ATTR`` (``args.yeto_algo_plugins``) and
  ``plugins_payload(config) / read_plugins(args)``: the runtime-attrs
  channel (D5).

Registering a new advantage transform (P2)
------------------------------------------

1. Add ``fn`` and register it in this module (``register_advantage_transform``)
   with a ``validate`` for its parameters. Registration in this module keeps
   the dispatcher's PluginRef (the source SHA256 of this file) covering it,
   so the new stage enters the algorithm hash.
2. Select it from the spec: ``advantage.transform = "<name>"`` (and
   ``advantage.transform_params`` for parameters) plus
   ``advantage.reward_postprocess = dispatcher_ref()``.
3. Map any new mechanism (``register_mechanism``) and keep it undeclared
   until its GPU smoke passes.
4. Add a CPU test: with the transform's degenerate parameters (if it has
   any) it must equal ``grpo_default`` element-wise, and its own numerics
   must match a hand computation.

Import-light: no torch/miles at import time (torch is imported on use).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DISPATCHER_PATH = "yeto.rl.algos.reward_pipeline.post_process"
PIPELINE_ATTR = "yeto_algo_plugins"
PLUGINS_SCHEMA = "yeto-algo-plugins-v1"
DEFAULT_TRANSFORM = "grpo_default"
# Estimators Miles normalizes in _post_process_rewards (train_data_conversion.py:284).
NORMALIZED_ESTIMATORS = ("grpo", "gspo", "reinforce_plus_plus_baseline")
# Estimators that divide by the group std (train_data_conversion.py:259).
STD_ESTIMATORS = ("grpo", "gspo")


class RewardPipelineError(RuntimeError):
    """The dispatcher cannot run with this configuration / batch."""


@dataclass(frozen=True)
class StageDef:
    name: str
    fn: Callable[..., list[float]]
    validate: Callable[[Mapping[str, Any], Any], list[str]]


REWARD_SHAPERS: dict[str, StageDef] = {}
ADV_TRANSFORMS: dict[str, StageDef] = {}


def _no_params(kind: str, name: str):
    def validate(params: Mapping[str, Any], _spec: Any) -> list[str]:
        if params:
            return [f"{kind} {name!r} takes no parameters, got {sorted(params)}"]
        return []

    return validate


def register_reward_shaper(name: str, fn, *, validate=None) -> StageDef:
    if name in REWARD_SHAPERS:
        raise ValueError(f"reward shaper {name!r} already registered")
    stage = StageDef(name, fn, validate or _no_params("reward shaper", name))
    REWARD_SHAPERS[name] = stage
    return stage


def register_advantage_transform(name: str, fn, *, validate=None) -> StageDef:
    if name in ADV_TRANSFORMS:
        raise ValueError(f"advantage transform {name!r} already registered")
    stage = StageDef(name, fn, validate or _no_params("advantage transform", name))
    ADV_TRANSFORMS[name] = stage
    return stage


# --------------------------------------------------------------------------
# runtime attrs channel (design D5)
# --------------------------------------------------------------------------


def canonical_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def plugins_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """``{"config": ..., "sha256": ...}`` for ``args.yeto_algo_plugins``."""

    config = {"schema": PLUGINS_SCHEMA, **dict(config)}
    return {"config": config, "sha256": canonical_sha256(config)}


def read_plugins(args: Any, *, required: bool = True) -> dict[str, Any] | None:
    """Read and hash-check ``args.yeto_algo_plugins``; returns its config."""

    payload = getattr(args, PIPELINE_ATTR, None)
    if payload is None:
        if required:
            raise RewardPipelineError(
                f"args.{PIPELINE_ATTR} is missing: the yeto algorithm plugins run only "
                "with the configuration the ports adapter derives from the AlgorithmSpec"
            )
        return None
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping) or set(payload) != {"config", "sha256"}:
        raise RewardPipelineError(f"args.{PIPELINE_ATTR} must be {{config, sha256}}")
    config = payload["config"]
    actual = canonical_sha256(config)
    if actual != payload["sha256"]:
        raise RewardPipelineError(
            f"args.{PIPELINE_ATTR} hash mismatch: declared {payload['sha256']}, "
            f"config hashes to {actual} (plugin configuration was modified)"
        )
    if config.get("schema") != PLUGINS_SCHEMA:
        raise RewardPipelineError(f"args.{PIPELINE_ATTR}: unknown schema {config.get('schema')!r}")
    return dict(config)


def pipeline_config(args: Any) -> dict[str, Any]:
    config = read_plugins(args)
    pipeline = config.get("reward_pipeline")
    if not isinstance(pipeline, Mapping):
        raise RewardPipelineError(
            f"args.{PIPELINE_ATTR} has no reward_pipeline section; the dispatcher is only "
            "emitted when the spec selects a reward shaper or advantage transform"
        )
    return dict(pipeline)


# --------------------------------------------------------------------------
# Miles grouping (train_data_conversion.py:189-221, 238-248)
# --------------------------------------------------------------------------


def reward_groups(args: Any, samples: Sequence[Any], *, warn=None) -> list[list[int]]:
    """``_reward_group_segments`` without the multi-LoRA ``prompt_group_sizes`` branch."""

    group_indices = [getattr(s, "group_index", None) for s in samples]
    if all(g is not None for g in group_indices):
        by_group: dict[int, list[int]] = {}
        for row, group_index in enumerate(group_indices):
            by_group.setdefault(int(group_index), []).append(row)
        return list(by_group.values())
    expected = args.n_samples_per_prompt * args.rollout_batch_size
    if len(samples) == expected:
        n = args.n_samples_per_prompt
        return [list(range(start, start + n)) for start in range(0, len(samples), n)]
    (warn or _warn_fallback)(args, samples)
    return [list(range(len(samples)))]


def rollout_key(sample: Any, row: int):
    if getattr(sample, "rollout_id", None) is not None:
        return sample.rollout_id
    if getattr(sample, "index", None) is not None:
        return sample.index
    return ("row", row)


def rollout_segments(samples: Sequence[Any], rows: Sequence[int]):
    """[(rollout_key, [rows])] in first-seen order (multi-segment merging)."""

    merged: dict[Any, list[int]] = {}
    for row in rows:
        merged.setdefault(rollout_key(samples[row], row), []).append(row)
    return list(merged.items())


def shared_rollout_rewards(rewards: Sequence[float], segments) -> list[float]:
    """One reward per rollout; siblings must agree (Miles raises the same error)."""

    shared = []
    for key, rows in segments:
        sibling = [rewards[r] for r in rows]
        if any(reward != sibling[0] for reward in sibling[1:]):
            raise ValueError(
                f"all samples in rollout {key!r} must share one reward; "
                f"rows {rows} have rewards {sibling}"
            )
        shared.append(sibling[0])
    return shared


FALLBACK_EVENT = "rl_reward_group_fallback"


def _warn_fallback(args: Any, samples: Sequence[Any]) -> None:
    """Whole-batch fallback: ports' bounded filter always sets group_index."""

    event = {
        "event": FALLBACK_EVENT,
        "samples": len(samples),
        "expected_fixed_fanout": int(args.n_samples_per_prompt * args.rollout_batch_size),
        "missing_group_index": sum(getattr(s, "group_index", None) is None for s in samples),
        "detail": "no group_index and not a fixed fan-out batch: the whole batch is one "
        "reward group (Miles built-in behaviour kept for equivalence)",
    }
    emit_event(args, event)


def emit_event(args: Any, event: dict[str, Any]) -> None:
    logger.warning("%s", json.dumps(event, sort_keys=True))
    if getattr(args, "yeto_rl_event_tape", None) and getattr(args, "yeto_rl_learner_id", None) is not None:
        from yeto.rl.miles import _append_rl_event

        _append_rl_event(args, event)


# --------------------------------------------------------------------------
# advantage transforms
# --------------------------------------------------------------------------


def grpo_default(args, samples, rewards, groups, params) -> list[float]:
    """Exact ``_post_process_rewards`` normalization (``_normalize_rewards_by_rollout``)."""

    if not (args.advantage_estimator in list(NORMALIZED_ESTIMATORS) and args.rewards_normalization):
        return rewards
    if not samples:
        return []
    import torch

    normalized = torch.empty(len(rewards), dtype=torch.float)
    for rows in groups:
        segments = rollout_segments(samples, rows)
        shared = shared_rollout_rewards(rewards, segments)
        values = torch.tensor(shared, dtype=torch.float)
        centered = values - values.mean()
        if args.advantage_estimator in list(STD_ESTIMATORS) and args.grpo_std_normalization and len(values) > 1:
            std = values.std()
            if std > 0:
                centered = centered / (std + 1e-6)
        for (_, seg_rows), value in zip(segments, centered.tolist(), strict=True):
            for row in seg_rows:
                normalized[row] = value
    return normalized.tolist()


register_advantage_transform(DEFAULT_TRANSFORM, grpo_default)


# --------------------------------------------------------------------------
# reward shapers
# --------------------------------------------------------------------------


def overlong_penalty_value(length: int, max_length: int, cache_length: int) -> float:
    """DAPO soft overlong punishment (design D6)."""

    if length <= max_length - cache_length:
        return 0.0
    if length <= max_length:
        return (max_length - cache_length - length) / cache_length
    return -1.0


def _validate_overlong(params: Mapping[str, Any], spec: Any) -> list[str]:
    problems = []
    unknown = sorted(set(params) - {"max_length", "cache_length"})
    if unknown:
        problems.append(f"overlong_penalty: unknown parameters {unknown}")
    lmax, lcache = params.get("max_length"), params.get("cache_length")
    for key, value in (("max_length", lmax), ("cache_length", lcache)):
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"overlong_penalty.{key} must be an integer, got {value!r}")
    if problems:
        return problems
    if not 0 < lcache <= lmax:
        problems.append(
            f"overlong_penalty requires 0 < cache_length <= max_length, got "
            f"cache_length={lcache}, max_length={lmax}"
        )
    return problems


def overlong_penalty(args, samples, rewards, params) -> list[float]:
    """Add the DAPO penalty of each rollout's merged response length.

    Multi-segment rollouts (same rollout key) use the sum of their segments'
    ``response_length`` so every sibling gets the same shaped reward (the
    one-reward-per-rollout check keeps holding).
    """

    lmax, lcache = int(params["max_length"]), int(params["cache_length"])
    totals: dict[Any, int] = {}
    keys = [rollout_key(s, row) for row, s in enumerate(samples)]
    for key, sample in zip(keys, samples):
        totals[key] = totals.get(key, 0) + int(sample.response_length)
    return [
        float(r) + overlong_penalty_value(totals[k], lmax, lcache)
        for r, k in zip(rewards, keys, strict=True)
    ]


register_reward_shaper("overlong_penalty", overlong_penalty, validate=_validate_overlong)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

RAW_REWARD_METADATA_KEY = "yeto_raw_reward"


def _check_multi_lora(args: Any) -> None:
    if getattr(args, "multi_lora", False):
        raise RewardPipelineError(
            "the yeto reward dispatcher cannot see Miles' multi-LoRA prompt_group_sizes; "
            "multi-LoRA runs cannot use reward shaping / advantage transforms"
        )


def post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Miles ``--custom-reward-post-process-path`` entry."""

    _check_multi_lora(args)
    cfg = pipeline_config(args)
    raw = [sample.get_reward_value(args) for sample in samples]
    shaped = raw
    shapers = cfg.get("reward_shapers") or []
    for stage_cfg in shapers:
        params = dict(stage_cfg)
        name = params.pop("name")
        if name not in REWARD_SHAPERS:
            raise RewardPipelineError(f"unknown reward shaper {name!r} (known {sorted(REWARD_SHAPERS)})")
        shaped = REWARD_SHAPERS[name].fn(args, samples, shaped, params)
        if len(shaped) != len(samples):
            raise RewardPipelineError(f"reward shaper {name!r} returned {len(shaped)} rewards")
    if shapers:
        for sample, value in zip(samples, raw, strict=True):
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata[RAW_REWARD_METADATA_KEY] = value
        emit_event(args, reward_summary_event(raw, shaped))
    name = cfg.get("advantage_transform", DEFAULT_TRANSFORM)
    if name not in ADV_TRANSFORMS:
        raise RewardPipelineError(f"unknown advantage transform {name!r} (known {sorted(ADV_TRANSFORMS)})")
    params = dict(cfg.get("advantage_params") or {})
    groups = reward_groups(args, samples)
    advantages = ADV_TRANSFORMS[name].fn(args, samples, shaped, groups, params)
    if len(advantages) != len(samples):
        raise RewardPipelineError(f"advantage transform {name!r} returned {len(advantages)} values")
    return shaped, advantages


REWARD_SUMMARY_EVENT = "rl_reward_shaping"


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return {"mean": None, "min": None, "max": None}
    return {"mean": sum(finite) / len(finite), "min": min(finite), "max": max(finite)}


def reward_summary_event(raw: Sequence[float], shaped: Sequence[float]) -> dict[str, Any]:
    """Raw vs shaped reward summary (task 6.2)."""

    penalised = sum(1 for r, s in zip(raw, shaped) if float(r) != float(s))
    return {
        "event": REWARD_SUMMARY_EVENT,
        "samples": len(raw),
        "shaped_samples": penalised,
        "raw_reward": _stats(raw),
        "shaped_reward": _stats(shaped),
    }


def dispatcher_ref():
    """The PluginRef of this dispatcher (source SHA256 of this module)."""

    from yeto.rl.engine.algorithm import PluginRef

    return PluginRef.from_path(DISPATCHER_PATH)
