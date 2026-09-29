"""GRPO-family knobs on the ports path (change ``rl-algo-grpo-knobs``).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. It adds
(through the P0 registry, never by editing the shared files):

* spec fields
  - ``advantage.reward_shapers``: ordered list of ``{"name": <REWARD_SHAPERS key>, **params}``
    (``overlong_penalty`` with ``max_length``/``cache_length``, design D6);
  - ``advantage.transform`` / ``advantage.transform_params``: the
    ``ADV_TRANSFORMS`` key (default ``grpo_default``) and its parameters
    (P2 ``rl-algo-seq-and-adv`` registers more keys);
  - ``loss.constant_denominator``: Dr.GRPO constant denominator (design D2);
  - ``kl.ref_model``: ``{"source", "revision"}`` reference model identity (design D3);
* rejection rules (pre-GPU, field-named): dispatcher selection, stage
  parameters, constant aggregation, KL reference identity, over-sampling;
* :func:`plugins_config` -- the JSON configuration the dispatcher / reducer /
  overlong filter read from ``args.yeto_algo_plugins`` (design D5);
* :func:`launch_problems` -- the checks that need the run configuration
  (rollout batch size, generation limit, CP, multi-LoRA).

Every field is default-off: a spec that selects none of them keeps its hash
and argv byte-identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl.engine.algorithm import (
    AlgorithmSpecError,
    register_field,
    register_mechanism,
    register_rejection,
)

from . import reward_pipeline as rp

REDUCER_PATH = "yeto.rl.algos.reducers.constant_denominator_reducer"


# --------------------------------------------------------------------------
# field parsers (stored hashable: tuples; to_json gives the JSON form)
# --------------------------------------------------------------------------


def _scalar(path: str, value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise AlgorithmSpecError(f"{path} must be finite, got {value!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise AlgorithmSpecError(f"{path} must be a JSON scalar, got {value!r}")


def _params(path: str, value: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, tuple):
        value = dict(value)
    if not isinstance(value, Mapping):
        raise AlgorithmSpecError(f"{path} must be an object, got {value!r}")
    return tuple(sorted((str(k), _scalar(f"{path}.{k}", v)) for k, v in value.items()))


def _parse_shapers(path: str, value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise AlgorithmSpecError(f"{path} must be a list of {{name, ...params}} objects")
    out = []
    for i, item in enumerate(value):
        item = _params(f"{path}[{i}]", item)
        names = [v for k, v in item if k == "name"]
        if not names or not isinstance(names[0], str):
            raise AlgorithmSpecError(f"{path}[{i}].name is required (one of {sorted(rp.REWARD_SHAPERS)})")
        if names[0] not in rp.REWARD_SHAPERS:
            raise AlgorithmSpecError(
                f"{path}[{i}].name {names[0]!r} is not a registered reward shaper "
                f"(one of {sorted(rp.REWARD_SHAPERS)})"
            )
        out.append(item)
    if len({dict(i)["name"] for i in out}) != len(out):
        raise AlgorithmSpecError(f"{path} lists the same shaper twice")
    return tuple(out)


def _shapers_json(value: tuple) -> list[dict[str, Any]]:
    return [dict(item) for item in value]


def _parse_transform(path: str, value: Any) -> str:
    if not isinstance(value, str) or value not in rp.ADV_TRANSFORMS:
        raise AlgorithmSpecError(
            f"{path} must be one of {sorted(rp.ADV_TRANSFORMS)}, got {value!r}"
        )
    return value


def _parse_transform_params(path: str, value: Any) -> tuple:
    return () if value is None else _params(path, value)


def _parse_denominator(path: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise AlgorithmSpecError(f"{path} must be a finite number > 0, got {value!r}")
    return value


def _parse_ref_model(path: str, value: Any) -> tuple | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = dict(value)
    if not isinstance(value, Mapping) or set(value) != {"source", "revision"}:
        raise AlgorithmSpecError(f"{path} must be an object with exactly source and revision")
    for key in ("source", "revision"):
        if not isinstance(value[key], str) or not value[key]:
            raise AlgorithmSpecError(f"{path}.{key} must be a non-empty string")
    return (("revision", value["revision"]), ("source", value["source"]))


def _dict_or_none(value):
    return None if value is None else dict(value)


register_field("advantage", "reward_shapers", default=(), parse=_parse_shapers, to_json=_shapers_json)
register_field("advantage", "transform", default=rp.DEFAULT_TRANSFORM, parse=_parse_transform)
register_field("advantage", "transform_params", default=(), parse=_parse_transform_params,
               to_json=lambda v: dict(v))
register_field("loss", "constant_denominator", default=None, parse=_parse_denominator)
register_field("kl", "ref_model", default=None, parse=_parse_ref_model, to_json=_dict_or_none)


# --------------------------------------------------------------------------
# mechanisms (features dimension; declared only after G1, design D1)
# --------------------------------------------------------------------------


def uses_pipeline(spec) -> bool:
    return bool(spec.advantage.reward_shapers) or spec.advantage.transform != rp.DEFAULT_TRANSFORM


for _shaper in ("overlong_penalty",):
    register_mechanism(
        "features", _shaper,
        lambda s, n=_shaper: any(dict(i)["name"] == n for i in s.advantage.reward_shapers),
    )
register_mechanism("features", "kl_loss_ref_model", lambda s: s.kl.ref_model is not None)


# --------------------------------------------------------------------------
# rejection matrix additions (design D1-D7)
# --------------------------------------------------------------------------


def _reject_pipeline(s) -> str | None:
    ref = s.advantage.reward_postprocess
    if ref is not None and ref.path != rp.DISPATCHER_PATH:
        return (
            f"advantage.reward_postprocess {ref.path!r}: the ports path uses only the yeto "
            f"dispatcher {rp.DISPATCHER_PATH!r}; select advantage.reward_shapers / "
            "advantage.transform instead"
        )
    if uses_pipeline(s) and ref is None:
        return (
            "advantage.reward_shapers / advantage.transform need the dispatcher: set "
            f"advantage.reward_postprocess to {{path: {rp.DISPATCHER_PATH!r}, sha256: <source sha256>}}"
        )
    if ref is not None and not uses_pipeline(s):
        return (
            "advantage.reward_postprocess selects the dispatcher but no reward shaper or "
            "non-default advantage.transform is configured; drop it (the built-in Miles "
            "normalization is used)"
        )
    if s.advantage.transform_params and s.advantage.transform == rp.DEFAULT_TRANSFORM:
        return "advantage.transform_params given for advantage.transform='grpo_default' (takes none)"
    return None


def _reject_stage_params(s) -> str | None:
    problems = []
    for item in s.advantage.reward_shapers:
        params = dict(item)
        name = params.pop("name")
        problems += [f"advantage.reward_shapers[{name}]: {p}" for p in rp.REWARD_SHAPERS[name].validate(params, s)]
    stage = rp.ADV_TRANSFORMS[s.advantage.transform]
    problems += [f"advantage.transform[{stage.name}]: {p}"
                 for p in stage.validate(dict(s.advantage.transform_params), s)]
    return "; ".join(problems) or None


def _reject_constant(s) -> str | None:
    constant = s.loss.aggregation == "constant"
    denominator = s.loss.constant_denominator
    if s.loss.reducer is not None and s.loss.reducer.path != REDUCER_PATH:
        return (
            f"loss.reducer {s.loss.reducer.path!r}: the only pg_loss reducer on the ports path is "
            f"the vendored Dr.GRPO reducer {REDUCER_PATH!r} (with loss.aggregation='constant')"
        )
    if constant and denominator is None:
        return "loss.aggregation='constant' requires loss.constant_denominator (a finite number > 0)"
    if not constant and denominator is not None:
        return "loss.constant_denominator requires loss.aggregation='constant'"
    if constant and (s.loss.reducer is None or s.loss.reducer.path != REDUCER_PATH):
        return (
            f"loss.aggregation='constant' uses the yeto reducer {REDUCER_PATH!r} as "
            "loss.reducer (vendored Dr.GRPO reducer, denominator from the spec)"
        )
    if not constant and s.loss.reducer is not None and s.loss.reducer.path == REDUCER_PATH:
        return f"loss.reducer {REDUCER_PATH!r} requires loss.aggregation='constant'"
    return None


def _reject_ref_model(s) -> str | None:
    if s.kl.placement == "loss" and s.kl.ref_model is None:
        return (
            "kl.placement='loss' loads a reference model; kl.ref_model {source, revision} is "
            "required so the reference identity enters the algorithm hash"
        )
    if s.kl.placement != "loss" and s.kl.ref_model is not None:
        return "kl.ref_model applies to kl.placement='loss' only"
    return None


def _reject_over_sampling(s) -> str | None:
    if s.sampling.over_sampling_batch_size is not None and s.sampling.filter is None:
        return (
            "sampling.over_sampling_batch_size requires a dynamic sampling filter "
            "(sampling.filter); over-sampling only replaces filtered groups"
        )
    return None


def required_plugins(spec) -> tuple:
    """PluginRefs a spec must list in ``plugins`` (review F1/F2)."""

    from yeto.rl.engine.algorithm import PluginRef

    refs = list(rp.stage_plugin_refs()) if uses_pipeline(spec) else []
    if spec.sampling.overlong_filter:
        refs.append(PluginRef.from_path(rp.SAMPLE_FILTERS_PATH))
    return tuple(sorted(refs, key=lambda r: r.path))


def with_pipeline_plugins(spec):
    """``spec`` with the required pipeline/sample-filter PluginRefs in ``plugins``."""

    required = {r.path: r for r in required_plugins(spec)}
    kept = [p for p in spec.plugins if p.path not in required]
    return spec.replace(plugins=[*kept, *required.values()])


def _reject_plugins(s) -> str | None:
    required = required_plugins(s)
    listed = {p.path: p.sha256 for p in s.plugins}
    missing = [r.path for r in required if r.path not in listed]
    stale = [r.path for r in required if r.path in listed and listed[r.path] != r.sha256]
    problems = []
    if missing:
        problems.append(
            f"plugins must list the code of every reward-pipeline / sample-filter module "
            f"{missing} (grpo_knobs.with_pipeline_plugins adds them)"
        )
    if stale:
        problems.append(f"plugins {stale}: source hash differs from the current module source")
    return "; ".join(problems) or None


register_rejection("grpo_knobs_reward_pipeline", _reject_pipeline)
register_rejection("grpo_knobs_pipeline_plugins", _reject_plugins)
register_rejection("grpo_knobs_stage_params", _reject_stage_params)
register_rejection("grpo_knobs_constant_aggregation", _reject_constant)
register_rejection("grpo_knobs_kl_ref_model", _reject_ref_model)
register_rejection("grpo_knobs_over_sampling", _reject_over_sampling)


# --------------------------------------------------------------------------
# runtime attrs (design D5) and run-configuration checks
# --------------------------------------------------------------------------


def plugins_config(spec) -> dict[str, Any] | None:
    """Configuration read by the yeto plugins in Miles processes (None: none used)."""

    config: dict[str, Any] = {}
    if uses_pipeline(spec):
        config["reward_pipeline"] = {
            "reward_shapers": _shapers_json(spec.advantage.reward_shapers),
            "advantage_transform": spec.advantage.transform,
            "advantage_params": dict(spec.advantage.transform_params),
            "pipeline_sha256": rp.pipeline_sha256(),
        }
    if spec.loss.aggregation == "constant":
        config["reducer"] = {"denominator": spec.loss.constant_denominator}
    if spec.sampling.overlong_filter:
        config["overlong_filter"] = True
        config["sample_filters_sha256"] = rp.plugin_sha(rp.SAMPLE_FILTERS_PATH)
    if not config:
        return None
    config["algorithm_spec_sha256"] = spec.sha256()
    return config


def runtime_attrs(spec) -> dict[str, Any]:
    """``{yeto_algo_plugins: {config, sha256}}`` or ``{}`` (default GRPO: nothing)."""

    config = plugins_config(spec)
    return {} if config is None else {rp.PIPELINE_ATTR: rp.plugins_payload(config)}


def launch_problems(
    spec,
    *,
    rollout_batch_size: int | None = None,
    rollout_max_response_len: int | None = None,
    context_parallel_size: int | None = 1,
    multi_lora: bool = False,
) -> list[str]:
    """Checks needing the run configuration (before any GPU process).

    A value the adapter does not provide (None) skips its check.
    """

    problems = []
    for item in spec.advantage.reward_shapers:
        if dict(item)["name"] not in rp.REWARD_SHAPERS:
            problems.append(f"reward shaper {dict(item)['name']!r} is not registered")
    if spec.advantage.transform not in rp.ADV_TRANSFORMS:
        problems.append(f"advantage transform {spec.advantage.transform!r} is not registered")
    over = spec.sampling.over_sampling_batch_size
    if over is not None and rollout_batch_size is not None and over < rollout_batch_size:
        problems.append(
            f"sampling.over_sampling_batch_size={over} must be >= the rollout batch size "
            f"{rollout_batch_size}"
        )
    for item in spec.advantage.reward_shapers:
        params = dict(item)
        if (params["name"] == "overlong_penalty" and rollout_max_response_len is not None
                and params["max_length"] > rollout_max_response_len):
            problems.append(
                f"overlong_penalty.max_length={params['max_length']} exceeds the generation "
                f"limit rollout_max_response_len={rollout_max_response_len}"
            )
    if (spec.loss.aggregation == "constant" and context_parallel_size is not None
            and context_parallel_size != 1):
        problems.append(
            f"loss.aggregation='constant' requires context parallel size 1, got {context_parallel_size}"
        )
    if multi_lora and uses_pipeline(spec):
        problems.append(
            "multi-LoRA: the reward dispatcher cannot see prompt_group_sizes; reward shapers / "
            "advantage transforms are refused"
        )
    return problems


def check_ref_model(spec, base_model_revision: str | None, *, base_model: str | None = None,
                    ref_load_override: str | None = None) -> None:
    """Tasks 3.2/3.3: the KL reference must be this island's base model.

    ``--ref-load`` is resolved from the pinned base model (``--model`` at
    ``--model-revision``, ``run_config._resolve_ref_load``) unless
    ``--megatron-ref-load`` overrides it with an unverifiable local
    checkpoint, which is refused with KL loss. Revisions compare
    case-insensitively (the learner lower-cases ``model_revision``).
    """

    ref = spec.kl.ref_model
    if ref is None:
        return
    ref = dict(ref)
    if base_model_revision is None or ref["revision"].lower() != str(base_model_revision).lower():
        raise AlgorithmSpecError(
            f"kl.ref_model.revision {ref['revision']!r} does not match this island's "
            f"base_model_revision {base_model_revision!r}; the reference model is the base"
        )
    if base_model is not None and ref["source"] != base_model:
        raise AlgorithmSpecError(
            f"kl.ref_model.source {ref['source']!r} does not match this island's base model "
            f"{base_model!r} (--ref-load is resolved from --model)"
        )
    if ref_load_override:
        raise AlgorithmSpecError(
            f"kl.placement='loss' with --megatron-ref-load {ref_load_override!r}: the reference "
            "checkpoint cannot be bound to kl.ref_model {source, revision}; drop the override"
        )


def island_problems(spec, island) -> list[str]:
    try:
        check_ref_model(spec, island.get("base_model_revision"),
                        base_model=island.get("base_model"),
                        ref_load_override=island.get("ref_load_override"))
    except AlgorithmSpecError as exc:
        return [str(exc)]
    return []


def overlong_gradient_rule(spec, batch_summary, step_metrics=None):
    """Task 6.5: with overlong filtering, a round whose non-zero-variance groups
    had *all* their samples filtered expects no gradient.

    Needs ``GroupMetadata.filtered_samples`` (rollout metadata written by the
    shared hook, 1b-hook.patch); without it (None) the rule abstains and the
    strict default applies.
    """

    if not spec.sampling.overlong_filter:
        return None
    groups = getattr(batch_summary, "groups", batch_summary)
    live = [g for g in groups if g.reward_std > 0]
    if not live:
        return None
    for g in live:
        filtered = getattr(g, "filtered_samples", None)
        if filtered is None or filtered < len(g.sample_ids):
            return None
    return False


def _launch_check(spec, values):
    return launch_problems(
        spec,
        rollout_batch_size=values.get("rollout_batch_size"),
        rollout_max_response_len=values.get("rollout_max_response_len"),
        context_parallel_size=values.get("context_parallel_size", 1),
        multi_lora=bool(values.get("multi_lora", False)),
    )


# P0 extension hooks (algo-cap ebd436b, merged from infra-drafts/1b-shared.patch).
from yeto.rl.engine.algorithm import (  # noqa: E402
    register_gradient_rule,
    register_island_check,
    register_launch_check,
    register_runtime_attrs,
)

from yeto.rl.engine.algorithm import register_pipeline_plugin_module  # noqa: E402

register_pipeline_plugin_module("yeto.rl.algos.reward_pipeline")
register_pipeline_plugin_module("yeto.rl.algos.sample_filters")
register_runtime_attrs("grpo_knobs", runtime_attrs)
register_launch_check("grpo_knobs", _launch_check)
register_island_check("grpo_knobs_ref_model", island_problems)
register_gradient_rule("grpo_knobs_overlong_filter", overlong_gradient_rule,
                       mechanism="features:overlong_filter")


# --------------------------------------------------------------------------
# task 8.3: mechanisms whose single-GPU smoke (G1) passed
# (openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1/g1_report.json).
# overlong_filter is NOT declared (its hook wiring, 1b-hook.patch, is not merged
# and it has no G1). Declaring means only "G1 passed", never "improves training".
# --------------------------------------------------------------------------

G1_EVIDENCE = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1"
# integ-decl: the only allowed reducer is claimed by the constant-denominator
# aggregation (P0 register_named_reducer), so declaring loss_aggregations:
# constant admits it without the generic features:custom_pg_loss_reducer.
from yeto.rl.engine.algorithm import register_named_reducer  # noqa: E402

# Pinned to the reducer source the drgrpo G1 ran (evidence
# 2026-09-29-algo1b-g1/out/drgrpo/algorithm_spec.json).
REDUCER_SOURCE_SHA256 = "253856acaefbea8de936b03ea89bf5c50039719732f21709aa66358f9f78e4ed"
register_named_reducer(REDUCER_PATH, mechanisms=("loss_aggregations:constant",),
                       sha256=REDUCER_SOURCE_SHA256)

G1_DECLARED: dict[str, dict[str, frozenset[str]]] = {
    # G1 run name -> mechanisms declared from it (dimension -> names). Only
    # mechanisms shown to take effect on the GPU are declared (coordinator
    # decision): clip_higher / dual_clip (clipfrac 0) and over_sampling (no
    # replacement) wait for a G1 that triggers them.
    # custom_pg_loss_reducer is declared by ALGO-CAP as "only the Dr.GRPO reducer"
    # (grpo_knobs_constant_aggregation refuses any other loss.reducer).
    # drgrpo: only the constant aggregation (token and no_grpo_std_normalization
    # were withdrawn after review: no isolated evidence; see evidence/2026-09-29-algo1b-g1c).
    "drgrpo": {"loss_aggregations": frozenset({"constant"})},
    "kl_k3": {"features": frozenset({"kl_loss_ref_model"}), "kl_placements": frozenset({"loss"})},
    "entropy": {"features": frozenset({"entropy_bonus"})},
    "overlong_penalty": {"features": frozenset({"overlong_penalty"}),
                         "reward_postprocessors": frozenset({"custom_reward_postprocess"})},
    "clip_higher": {"features": frozenset({"clip_higher", "eps_clip"})},  # g1b run A-r1 (clipfrac > 0)
}


def declared_mechanisms() -> dict[str, frozenset[str]]:
    """Union of G1_DECLARED per dimension (to merge into the engine declaration)."""

    out: dict[str, set[str]] = {}
    for dims in G1_DECLARED.values():
        for dim, names in dims.items():
            out.setdefault(dim, set()).update(names)
    return {d: frozenset(n) for d, n in out.items()}


def merge_declared(capabilities):
    """``capabilities`` with the G1-declared mechanisms added (union per dimension)."""

    import dataclasses

    extra = declared_mechanisms()
    if not extra:
        return capabilities
    return dataclasses.replace(capabilities, **{
        dim: frozenset(getattr(capabilities, dim)) | names for dim, names in extra.items()})
