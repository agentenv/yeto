"""Sequence-level ratio and advantage variants (change ``rl-algo-seq-and-adv``).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. Through
the P0 registry (never by editing the shared files) it adds:

* spec fields
  - ``advantage.gamma`` (design D4): REINFORCE++ discount, default 1.0 (not
    emitted); ``--gamma`` maps to it. Only 1.0 is open;
  - ``advantage.gdpo`` (design D6): ``{"components": [{"name", "weight"}],
    "whiten": true}``; components are sorted by name and enter the hash;
* advantage transforms of the 1b dispatcher (design D5): ``maxrl``, ``mapo``,
  ``gdpo`` (``advantage.transform``), computed per rollout entry (multi-segment
  rollouts merged, the result broadcast back to every segment);
* mechanisms (``features`` dimension): ``maxrl`` / ``mapo`` (binary reward
  required), ``gdpo``. GSPO / REINFORCE++ / REINFORCE++-baseline are P0's
  ``advantage_estimators`` mechanisms; none is declared by the Miles adapter
  before its G1 smoke;
* rejections (pre-GPU): GSPO clip hint (engine default 0.2 vs paper 3e-4/4e-4),
  GSPO + advantage transform, transform/estimator mismatch, MaxRL/MAPO with the
  overlong soft penalty, non-default gamma, GDPO declaration consistency;
* gradient rules (design D2/D8): GSPO fully clipped round; GDPO / REINFORCE++
  where the reported statistics say no advantage is non-zero;
* runtime attrs ``yeto_rl_seq_adv`` (GDPO components for the Miles process).

Transform module identity: the 1b dispatcher's pipeline-plugins rule requires
this module's PluginRef (``yeto.rl.algos.seq_adv.<stage>``, path chosen by
1b) in ``spec.plugins``; ``grpo_knobs.with_pipeline_plugins(spec)`` adds it.

Import-light: no torch/miles at import time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl.engine.algorithm import (
    AlgorithmSpecError,
    register_field,
    register_gradient_rule,
    register_launch_check,
    register_mechanism,
    register_rejection,
    register_runtime_attrs,
)
from yeto.rl.engine.miles_adapter.algorithm_flags import _float, _num, _value_row, register_flag

from . import reward_pipeline as rp

MODULE = "yeto.rl.algos.seq_adv"
SEQ_ADV_ATTR = "yeto_rl_seq_adv"
SEQ_ADV_SCHEMA = "yeto-rl-seq-adv-v1"
REWARD_COMPONENTS_KEY = "yeto_reward_components"
TRANSFORM_EVENT = "rl_advantage_transform"
GROUP_TRANSFORMS = ("maxrl", "mapo", "gdpo")
BINARY_TRANSFORMS = ("maxrl", "mapo")
SUPPORTED_GAMMAS = (1.0,)
# D2: fully clipped when the token-weighted clip fraction reaches 1 (float slack).
FULL_CLIP_THRESHOLD = 1.0 - 1e-9
# Keys the adapter may use for the GSPO clip fraction (Miles loss dict key and
# its logged name); the P0 ``masked_fraction`` contract carries the value.
CLIPFRAC_KEYS = ("pg_clipfrac", "train/pg_clipfrac")
STD_EPS = 1e-6  # Miles _post_process_rewards: centered / (std + 1e-6)


class AdvantageTransformError(rp.RewardPipelineError):
    """A transform input violates its contract; the round fails."""


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


def _parse_gamma(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise AlgorithmSpecError(f"{path} must be a finite number in (0, 1], got {value!r}")
    return value


def _parse_gdpo(path: str, value: Any) -> tuple | None:
    """Canonical ``(("components", ((name, weight), ...)), ("whiten", bool))``."""

    if value is None:
        return None
    if isinstance(value, tuple):
        value = {k: v for k, v in value}
        if isinstance(value.get("components"), tuple):
            value["components"] = [{"name": n, "weight": w} for n, w in value["components"]]
    if not isinstance(value, Mapping) or not set(value) <= {"components", "whiten"} \
            or "components" not in value:
        raise AlgorithmSpecError(f"{path} must be an object {{components: [...], whiten: true}}")
    whiten = value.get("whiten", True)
    if not isinstance(whiten, bool):
        raise AlgorithmSpecError(f"{path}.whiten must be a boolean, got {whiten!r}")
    components = value["components"]
    if isinstance(components, (str, bytes, Mapping)) or not isinstance(components, Sequence):
        raise AlgorithmSpecError(f"{path}.components must be a list of {{name, weight}}")
    if not components:
        raise AlgorithmSpecError(f"{path}.components is empty; declare at least one component")
    out = []
    for i, item in enumerate(components):
        if not isinstance(item, Mapping) or set(item) != {"name", "weight"}:
            raise AlgorithmSpecError(f"{path}.components[{i}] must be exactly {{name, weight}}")
        name, weight = item["name"], item["weight"]
        if not isinstance(name, str) or not name:
            raise AlgorithmSpecError(f"{path}.components[{i}].name must be a non-empty string")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or not math.isfinite(float(weight)):
            raise AlgorithmSpecError(
                f"{path}.components[{i}].weight ({name!r}) must be a finite number, got {weight!r}"
            )
        out.append((name, float(weight)))
    names = [n for n, _ in out]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise AlgorithmSpecError(f"{path}.components names repeat: {duplicates}")
    return (("components", tuple(sorted(out))), ("whiten", whiten))


def _gdpo_json(value: tuple | None) -> dict | None:
    if value is None:
        return None
    data = dict(value)
    return {
        "components": [{"name": n, "weight": w} for n, w in data["components"]],
        "whiten": data["whiten"],
    }


register_field("advantage", "gamma", default=1.0, parse=_parse_gamma)
register_field("advantage", "gdpo", default=None, parse=_parse_gdpo, to_json=_gdpo_json)
# --gamma: default 1.0 is not emitted (argv snapshots unchanged); --lambd stays unmapped.
register_flag(_value_row("--gamma", "advantage.gamma", _float, emit=_num, default=1.0))


def gdpo_components(spec) -> tuple[tuple[str, float], ...]:
    value = spec.advantage.gdpo
    return () if value is None else dict(value)["components"]


# --------------------------------------------------------------------------
# mechanisms
# --------------------------------------------------------------------------


def _transform(spec) -> str:
    return spec.advantage.transform


register_mechanism("features", "maxrl", lambda s: _transform(s) == "maxrl",
                   requires_binary_reward=True)
register_mechanism("features", "mapo", lambda s: _transform(s) == "mapo",
                   requires_binary_reward=True)
register_mechanism("features", "gdpo", lambda s: _transform(s) == "gdpo")

# Mechanisms of this change as (dimension, name); ``--rl-allow-unverified-mechanism``
# takes them qualified, ``"dimension:name"`` (task 6.1). A transform additionally needs the dispatcher
# (``custom_reward_postprocess``).
MECHANISMS = {
    "gspo": ("advantage_estimators", "gspo"),
    "reinforce_plus_plus": ("advantage_estimators", "reinforce_plus_plus"),
    "reinforce_plus_plus_baseline": ("advantage_estimators", "reinforce_plus_plus_baseline"),
    "maxrl": ("features", "maxrl"),
    "mapo": ("features", "mapo"),
    "gdpo": ("features", "gdpo"),
}


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------


def _reject_gspo_clip(s) -> str | None:
    # P0 ``sequence_ratio_without_clip`` refuses the spec; this adds the hint (2.1).
    if s.advantage.estimator == "gspo" and (s.loss.eps_clip is None or s.loss.eps_clip_high is None):
        return (
            "GSPO clips the sequence-level ratio: give loss.eps_clip and loss.eps_clip_high "
            "explicitly. yeto sets no default; the engine default (Miles --eps-clip 0.2) is a "
            "token-level value, while the GSPO paper uses 3e-4 (low) / 4e-4 (high)"
        )
    return None


def _reject_gspo_transform(s) -> str | None:
    if s.advantage.estimator == "gspo" and _transform(s) != rp.DEFAULT_TRANSFORM:
        return (
            f"GSPO with advantage.transform={_transform(s)!r} is expressible but not opened "
            "(unverified combination); use advantage.transform='grpo_default' with GSPO"
        )
    return None


def _reject_transform_estimator(s) -> str | None:
    if _transform(s) in GROUP_TRANSFORMS and s.advantage.estimator != "grpo":
        return (
            f"advantage.transform={_transform(s)!r} is a group-relative transform and only "
            f"combines with advantage.estimator='grpo', got {s.advantage.estimator!r}"
        )
    return None


def _reject_binary_with_overlong(s) -> str | None:
    shapers = [dict(i)["name"] for i in s.advantage.reward_shapers]
    if _transform(s) in BINARY_TRANSFORMS and "overlong_penalty" in shapers:
        return (
            f"advantage.transform={_transform(s)!r} requires binary {{0,1}} rewards; the "
            "overlong_penalty reward shaper makes them non-binary. Drop one of them "
            "(sampling.overlong_filter keeps rewards binary)"
        )
    return None


def _reject_gamma(s) -> str | None:
    gamma = s.advantage.gamma
    if gamma == 1.0:
        return None
    if s.advantage.estimator != "reinforce_plus_plus":
        return (
            f"advantage.gamma={gamma} only applies to advantage.estimator='reinforce_plus_plus' "
            f"(Miles reads --gamma there), got {s.advantage.estimator!r}; drop it"
        )
    return (
        f"advantage.gamma={gamma} is expressible but not opened (unverified); supported "
        f"values: {list(SUPPORTED_GAMMAS)}"
    )


def _reject_gdpo(s) -> str | None:
    selected = _transform(s) == "gdpo"
    declared = s.advantage.gdpo is not None
    if selected and not declared:
        return (
            "advantage.transform='gdpo' needs advantage.gdpo {components: [{name, weight}], "
            "whiten: true} declaring the reward vector"
        )
    if declared and not selected:
        return "advantage.gdpo requires advantage.transform='gdpo'"
    if declared and not dict(s.advantage.gdpo)["whiten"]:
        return (
            "advantage.gdpo.whiten=false is expressible but not opened (the GDPO paper "
            "whitens the batch); use whiten=true"
        )
    return None


register_rejection("seq_adv_gspo_clip_hint", _reject_gspo_clip)
register_rejection("seq_adv_gspo_transform", _reject_gspo_transform)
register_rejection("seq_adv_transform_estimator", _reject_transform_estimator)
register_rejection("seq_adv_binary_overlong", _reject_binary_with_overlong)
register_rejection("seq_adv_gamma", _reject_gamma)
register_rejection("seq_adv_gdpo", _reject_gdpo)


# --------------------------------------------------------------------------
# runtime attrs (GDPO components reach the Miles rollout process)
# --------------------------------------------------------------------------


def seq_adv_config(spec) -> dict[str, Any] | None:
    if _transform(spec) != "gdpo" or spec.advantage.gdpo is None:
        return None
    return {
        "schema": SEQ_ADV_SCHEMA,
        "gdpo": _gdpo_json(spec.advantage.gdpo),
        "algorithm_spec_sha256": spec.sha256(),
    }


def runtime_attrs(spec) -> dict[str, Any]:
    config = seq_adv_config(spec)
    if config is None:
        return {}
    return {SEQ_ADV_ATTR: {"config": config, "sha256": rp.canonical_sha256(config)}}


def read_seq_adv_config(args: Any) -> dict[str, Any]:
    payload = getattr(args, SEQ_ADV_ATTR, None)
    if payload is None:
        raise AdvantageTransformError(
            f"args.{SEQ_ADV_ATTR} is missing: GDPO runs only with the configuration the ports "
            "adapter derives from the AlgorithmSpec"
        )
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    if not isinstance(payload, Mapping) or set(payload) != {"config", "sha256"}:
        raise AdvantageTransformError(f"args.{SEQ_ADV_ATTR} must be {{config, sha256}}")
    actual = rp.canonical_sha256(payload["config"])
    if actual != payload["sha256"]:
        raise AdvantageTransformError(
            f"args.{SEQ_ADV_ATTR} hash mismatch: declared {payload['sha256']}, config hashes "
            f"to {actual}"
        )
    config = dict(payload["config"])
    if config.get("schema") != SEQ_ADV_SCHEMA:
        raise AdvantageTransformError(f"args.{SEQ_ADV_ATTR}: unknown schema {config.get('schema')!r}")
    return config


register_runtime_attrs("seq_adv", runtime_attrs)


def launch_problems(spec, values) -> list[str]:
    """Checks needing the run configuration (alignment A4)."""

    problems = []
    cp = int(values.get("context_parallel_size", 1) or 1)
    if spec.advantage.estimator == "gspo" and cp != 1:
        problems.append(
            f"GSPO with context parallel size {cp}: the sequence-level ratio gathers every "
            "sequence across CP ranks; that layout is not verified (rl-infra-spec A4). Use "
            "context parallel size 1"
        )
    return problems


register_launch_check("seq_adv", launch_problems)


# --------------------------------------------------------------------------
# numerics (pure python float64 would differ from Miles' float32; the
# transforms use torch float32 exactly like grpo_default so MAPO at p=0.5 is
# element-wise Miles' GRPO normalization)
# --------------------------------------------------------------------------


def _group_normalize(values):
    """Miles GRPO: centered / (std + 1e-6) if G > 1 and std > 0 else centered."""

    centered = values - values.mean()
    if len(values) > 1:
        std = values.std()
        if std > 0:
            centered = centered / (std + STD_EPS)
    return centered


def maxrl_values(rewards):
    """A = (r - mean) / mean; 0 for G=1 or mean 0 (explicit branches)."""

    import torch

    values = torch.as_tensor(rewards, dtype=torch.float)
    mean = values.mean()
    if len(values) <= 1 or not mean > 0:
        return torch.zeros_like(values)
    return (values - mean) / mean


def mapo_values(rewards):
    """A = (1-lam)(r-mu)/sigma + lam (r-mu)/mu, lam = 1 - 4p(1-p) (design D5)."""

    import torch

    values = torch.as_tensor(rewards, dtype=torch.float)
    if len(values) <= 1:
        return torch.zeros_like(values)
    mean = values.mean()
    p = float(mean)
    lam = 1.0 - 4.0 * p * (1.0 - p)
    first = _group_normalize(values)
    if lam == 0.0:
        return first  # p = 0.5: exactly Miles' GRPO normalization
    second = (values - mean) / mean if mean > 0 else torch.zeros_like(values)
    return (1.0 - lam) * first + lam * second


def _whiten(values):
    """Batch whitening: subtract the mean, divide by (std + 1e-6) when std > 0."""

    return _group_normalize(values)


def _entries(samples, rows):
    return rp.rollout_segments(samples, rows)


def _require_binary(name: str, samples, rewards) -> None:
    bad = [(i, r) for i, r in enumerate(rewards) if float(r) not in (0.0, 1.0)]
    if bad:
        shown = ", ".join(f"sample {i}: {r!r}" for i, r in bad[:8])
        raise AdvantageTransformError(
            f"advantage.transform={name!r} requires binary {{0,1}} rewards after reward "
            f"shaping; got {len(bad)} other value(s) ({shown})"
        )


def _broadcast(n: int, per_group) -> list[float]:
    out: list[float | None] = [None] * n
    for segments, values in per_group:
        for (_, seg_rows), value in zip(segments, values, strict=True):
            for row in seg_rows:
                out[row] = float(value)
    if any(v is None for v in out):
        raise AdvantageTransformError("advantage transform left samples without a value")
    return out  # type: ignore[return-value]


def _check_finite(name: str, values: Sequence[float]) -> None:
    bad = [i for i, v in enumerate(values) if not math.isfinite(v)]
    if bad:
        raise AdvantageTransformError(f"advantage transform {name!r} produced non-finite values at {bad}")


def _summary(name, per_group_rewards, advantages) -> dict[str, Any]:
    groups = len(per_group_rewards)
    return {
        "event": TRANSFORM_EVENT,
        "transform": name,
        "groups": groups,
        "entries": sum(len(g) for g in per_group_rewards),
        "all_zero_groups": sum(1 for g in per_group_rewards if all(float(r) == 0.0 for r in g)),
        "all_one_groups": sum(1 for g in per_group_rewards if all(float(r) == 1.0 for r in g)),
        "constant_groups": sum(1 for g in per_group_rewards if len(set(map(float, g))) <= 1),
        "samples": len(advantages),
        "nonzero_advantages": sum(1 for a in advantages if a != 0.0),
    }


def _current_round_id() -> int | None:
    """Training rollout id of the round being post-processed, or None.

    ``Sample.rollout_id`` is a per-trajectory key (multi-segment merging), not
    the training round; the round is read from the policy token the driver
    publishes before each rollout (``yeto:<rollout_id>:<hash>``, INFRA sink).
    No sink (no ``YETO_ROLLOUT_META_SINK`` and no Ray): None.
    """

    import os

    from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook

    if not os.environ.get(hook.META_SINK_ENV):
        try:
            import ray
        except ImportError:
            return None
        if not ray.is_initialized():
            return None
    token = hook.current_policy_token()
    if not token:
        return None
    parts = token.split(":")
    if len(parts) != 3 or parts[0] != "yeto" or not parts[1].isdigit():
        raise AdvantageTransformError(f"unexpected policy token {token!r}")
    return int(parts[1])


def _report_round(args, name, per_group_rewards, advantages) -> None:
    """Event + this rollout's non-zero advantage count for the driver (INFRA R2)."""

    summary = _summary(name, per_group_rewards, advantages)
    round_id = _current_round_id()
    summary["rollout_id"] = round_id
    if round_id is not None:
        from yeto.rl.engine.miles_adapter.rollout_meta_hook import record_round_metadata

        record_round_metadata(args, round_id,
                              nonzero_advantages=int(summary["nonzero_advantages"]))
    rp.emit_event(args, summary)


def _group_transform(name, fn, args, samples, rewards, groups) -> list[float]:
    _require_binary(name, samples, rewards)
    per_group, per_group_rewards = [], []
    for rows in groups:
        segments = _entries(samples, rows)
        shared = rp.shared_rollout_rewards(rewards, segments)
        per_group.append((segments, fn(shared).tolist()))
        per_group_rewards.append(shared)
    out = _broadcast(len(samples), per_group)
    _check_finite(name, out)
    _report_round(args, name, per_group_rewards, out)
    return out


def maxrl(args, samples, rewards, groups, params) -> list[float]:
    return _group_transform("maxrl", maxrl_values, args, samples, rewards, groups)


def mapo(args, samples, rewards, groups, params) -> list[float]:
    return _group_transform("mapo", mapo_values, args, samples, rewards, groups)


def reward_components(samples, rows, names: Sequence[str]) -> list[list[float]]:
    """Per-row component vectors in ``names`` order; strict (design D6)."""

    expected = set(names)
    out = []
    for row in rows:
        sample = samples[row]
        metadata = getattr(sample, "metadata", None)
        vector = metadata.get(REWARD_COMPONENTS_KEY) if isinstance(metadata, Mapping) else None
        where = f"sample {row} (index {getattr(sample, 'index', None)!r})"
        if not isinstance(vector, Mapping):
            raise AdvantageTransformError(
                f"{where}: sample.metadata[{REWARD_COMPONENTS_KEY!r}] is missing; the reward "
                f"function must write {{name: float}} for components {sorted(expected)}"
            )
        missing = sorted(expected - set(vector))
        extra = sorted(set(vector) - expected)
        if missing:
            raise AdvantageTransformError(f"{where}: reward vector lacks components {missing}")
        if extra:
            raise AdvantageTransformError(
                f"{where}: reward vector has undeclared components {extra} "
                f"(declared {sorted(expected)})"
            )
        values = []
        for name in names:
            value = vector[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise AdvantageTransformError(
                    f"{where}: reward component {name!r} must be a finite number, got {value!r}"
                )
            values.append(float(value))
        out.append(values)
    return out


def gdpo(args, samples, rewards, groups, params) -> list[float]:
    import torch

    config = read_seq_adv_config(args)["gdpo"]
    names = [c["name"] for c in config["components"]]
    weights = torch.tensor([c["weight"] for c in config["components"]], dtype=torch.float)
    per_group, per_group_rewards, entry_values = [], [], []
    for rows in groups:
        segments = _entries(samples, rows)
        vectors = []
        for key, seg_rows in segments:
            comps = reward_components(samples, seg_rows, names)
            if any(c != comps[0] for c in comps[1:]):
                raise AdvantageTransformError(
                    f"all samples in rollout {key!r} must share one reward vector; rows "
                    f"{list(seg_rows)} have {comps}"
                )
            vectors.append(comps[0])
        matrix = torch.tensor(vectors, dtype=torch.float).reshape(len(vectors), len(names))
        normalized = torch.stack(
            [_group_normalize(matrix[:, k]) for k in range(len(names))], dim=1
        )
        combined = (normalized * weights).sum(dim=1)
        per_group.append(segments)
        entry_values.append(combined)
        per_group_rewards.append(rp.shared_rollout_rewards(rewards, segments))
    if not per_group:
        return []
    flat = torch.cat(entry_values)
    if config.get("whiten", True):
        flat = _whiten(flat)
    values, start = [], 0
    for segments in per_group:
        values.append((segments, flat[start:start + len(segments)].tolist()))
        start += len(segments)
    out = _broadcast(len(samples), values)
    _check_finite("gdpo", out)
    _report_round(args, "gdpo", per_group_rewards, out)
    return out


def _validate_gdpo(params: Mapping[str, Any], spec: Any) -> list[str]:
    if params:
        return [f"advantage transform 'gdpo' takes no transform_params (use advantage.gdpo), "
                f"got {sorted(params)}"]
    return []


rp.register_advantage_transform("maxrl", maxrl)
rp.register_advantage_transform("mapo", mapo)
rp.register_advantage_transform("gdpo", gdpo, validate=_validate_gdpo)


# --------------------------------------------------------------------------
# zero-gradient rules (design D2/D8)
# --------------------------------------------------------------------------


def aggregate_clipfrac(values: Sequence[Any], weights: Sequence[Any] | None = None) -> float | None:
    """Round clip fraction from per-mini-batch values (token-weighted, D2).

    ``weights`` are the mini-batches' loss-token counts (equal weights when
    omitted). Any missing / non-finite value -> None (unknown keeps the
    strict R0 rule).
    """

    values = list(values)
    if not values:
        return None
    if weights is None:
        weights = [1.0] * len(values)
    weights = list(weights)
    if len(weights) != len(values):
        return None
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        if value is None or weight is None:
            return None
        value, weight = float(value), float(weight)
        if not (math.isfinite(value) and math.isfinite(weight)) or weight < 0:
            return None
        total += value * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return total / weight_sum


def clipfrac_from_losses(step_losses: Sequence[Mapping[str, Any]],
                         token_counts: Sequence[Any] | None = None) -> float | None:
    """``masked_fraction`` for GSPO from Miles' per-step loss dicts (``pg_clipfrac``)."""

    values = []
    for losses in step_losses or ():
        found = None
        if isinstance(losses, Mapping):
            for key in CLIPFRAC_KEYS:
                if losses.get(key) is not None:
                    raw = losses[key]
                    found = float(raw.item() if hasattr(raw, "item") else raw)
                    break
        values.append(found)
    return aggregate_clipfrac(values, token_counts)


def gspo_gradient_rule(spec, batch_summary, step_metrics=None):
    """D2: a GSPO round whose every sequence was clipped expects no gradient."""

    if spec.advantage.estimator != "gspo":
        return None
    masked = getattr(step_metrics, "masked_fraction", None)
    if masked is None:
        return None
    masked = float(masked)
    if math.isfinite(masked) and masked >= FULL_CLIP_THRESHOLD:
        return False
    return None


def _batch_value(batch_summary, name):
    value = getattr(batch_summary, name, None)
    if value is None and isinstance(batch_summary, Mapping):
        value = batch_summary.get(name)
    return value


def gdpo_expects_gradient(batch_summary) -> bool:
    """D8: non-zero advantages reported by the dispatcher; unknown -> True."""

    nonzero = _batch_value(batch_summary, "nonzero_advantages")
    if nonzero is None:
        return True
    return int(nonzero) > 0


def rpp_expects_gradient(batch_summary, *, reward_kl: bool) -> bool | None:
    """D8: REINFORCE++ advantages (before whitening) are not all equal.

    From group statistics: some group varies or group means differ -> True;
    missing statistics -> True. With identical rewards everywhere the only
    remaining source is a reward-side KL, whose size is not reported to the
    driver (it is exactly 0 on round 0, when the LoRA B matrices are 0 and the
    policy equals the reference): that case returns None (no verdict; the R0
    rule applies) with a reward KL, False without one.
    """

    groups = list(getattr(batch_summary, "groups", batch_summary) or ())
    if not groups:
        return True
    stds = [getattr(g, "reward_std", None) for g in groups]
    means = [getattr(g, "reward_mean", None) for g in groups]
    if any(v is None or not math.isfinite(float(v)) for v in stds + means):
        return True
    if any(float(s) > 0 for s in stds):
        return True
    if len({float(m) for m in means}) > 1:
        return True
    return None if reward_kl else False


def advantage_gradient_rule(spec, batch_summary, step_metrics=None):
    """GDPO / REINFORCE++ verdict (D8): ``False`` relaxes, ``True`` requires.

    The P0 hook currently honours only ``False`` (relax); ``True`` takes
    effect once the hook may tighten (infra-drafts/2a-shared.patch). Until
    then the P0 default (``any(reward_std > 0)``) is the stricter bound
    only where it already expects a gradient.
    """

    if _transform(spec) == "gdpo":
        return gdpo_expects_gradient(batch_summary)
    if spec.advantage.estimator == "reinforce_plus_plus":
        reward_kl = spec.kl.placement == "reward" and (spec.kl.coef or 0.0) > 0
        return rpp_expects_gradient(batch_summary, reward_kl=reward_kl)
    return None


def expects_gradient(spec, batch_summary, step_metrics=None) -> bool:
    """Full D2/D8 rule of this change (independent of the relax-only hook).

    ``AlgorithmSpec.expects_gradient`` applies the P0 default first
    (``any(reward_std > 0)``) and lets rules only relax it; GDPO (components
    may vary while the scalar reward is constant) and REINFORCE++ (group
    means may differ) can need *more* than the default -- see the interface
    request in progress.md.
    """

    groups = getattr(batch_summary, "groups", batch_summary)
    if _transform(spec) == "gdpo":
        return gdpo_expects_gradient(batch_summary)
    if spec.advantage.estimator == "reinforce_plus_plus":
        reward_kl = spec.kl.placement == "reward" and (spec.kl.coef or 0.0) > 0
        verdict = rpp_expects_gradient(batch_summary, reward_kl=reward_kl)
        if verdict is not None:
            return verdict
    expected = any(g.reward_std > 0 for g in groups)
    if expected and gspo_gradient_rule(spec, batch_summary, step_metrics) is False:
        return False
    return expected


register_gradient_rule("seq_adv_gspo_full_clip", gspo_gradient_rule,
                       mechanism="advantage_estimators:gspo")
register_gradient_rule("seq_adv_gdpo_nonzero", advantage_gradient_rule, mechanism="features:gdpo")
register_gradient_rule("seq_adv_rpp_not_all_equal", advantage_gradient_rule,
                       mechanism="advantage_estimators:reinforce_plus_plus")
