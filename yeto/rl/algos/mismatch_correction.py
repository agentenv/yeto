"""Train/inference mismatch observation and correction (change
``rl-algo-mismatch-correction``, design D2-D7).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. It
adds, on top of the P0 ``CorrectionSpec`` (``method`` none/tis/opsm/custom):

* named correction mechanisms, detected on the spec, so that an engine can
  declare them one by one (``EngineCapabilities.corrections``):

  ==================  ===============================================  =========
  mechanism           spec                                             masks
  ==================  ===============================================  =========
  mismatch_observe    method=custom, function=:data:`OBSERVE_PATH`     no
  tis (P0 built-in)   method=tis, tis_clip, tis_clip_low               no
  icepop              method=custom, function=:data:`ICEPOP_PATH`      yes
  opsm_trainer        OPSM, opsm_old_logprob_source=trainer            yes
  opsm_rollout        OPSM, opsm_old_logprob_source=rollout            yes
  mis                 method=custom, function=:data:`MIS_PATH`,        no
                      mis_mode truncate|clip
  mis_mask            same, mis_mode=mask                              yes
  ==================  ===============================================  =========

* extension fields of the ``correction`` group: ``opsm_old_logprob_source``
  and the MIS fields ``mis_level`` / ``mis_mode`` / ``mis_lower_bound`` /
  ``mis_upper_bound`` / ``mis_batch_normalize``;
* rejection rules (explicit thresholds, bound order, OPSM logprob source,
  observe-only purity, MIS field consistency) that fail before any GPU
  process exists;
* MIS parameters as Miles namespace attributes (``register_runtime_attrs``;
  ``mis.py`` reads ``args.tis_mode`` etc., which have no CLI flag).

Every threshold is explicit (design D3): nothing inherits a Miles default.
``execution.max_policy_staleness`` stays 0 (P0 rejection matrix): these
mechanisms correct numeric train/inference differences under the *same*
policy version (the serial driver's policy token guarantees
pi_behav = pi_old weights, design D1); they are not an off-policy license.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from yeto.rl.engine import algorithm as _algorithm
from yeto.rl.engine.algorithm import (
    AlgorithmSpec,
    AlgorithmSpecError,
    PluginRef,
    _boolean,
    _choice,
    _number,
    register_field,
    register_mechanism,
    register_rejection,
    register_runtime_attrs,
)

OBSERVE_PATH = "yeto.rl.algos.mismatch_observe.observe_mismatch"
ICEPOP_PATH = "miles.backends.training_utils.loss_hub.corrections.icepop_function"
MIS_PATH = "yeto.rl.algos.vendor.miles_mis.compute_mis_weights_with_cp"
# SHA256 of miles/backends/training_utils/loss_hub/corrections.py at
# MILES_NEXT_COMMIT (checked against the source by
# tests/test_rl_mismatch_observe.py in miles-next-venv). The Miles module is
# not importable in the yeto CPU venv, so the IcePop PluginRef is pinned here.
ICEPOP_SOURCE_SHA256 = "971ccb0bf00b43b0582839c5b8dc05e91162c878ab7ec0ca878e3b1e668f5318"

OPSM_SOURCES = ("trainer", "rollout")
MIS_LEVELS = ("token", "sequence", "geometric")
MIS_MODES = ("truncate", "clip", "mask")
MIS_FIELDS = ("mis_level", "mis_mode", "mis_lower_bound", "mis_upper_bound", "mis_batch_normalize")

CORRECTION_MECHANISMS = (
    "mismatch_observe",
    "tis",
    "icepop",
    "opsm_trainer",
    "opsm_rollout",
    "mis",
    "mis_mask",
)
# Mechanisms whose masking may legitimately zero a whole round (design D7).
MASKING_MECHANISMS = frozenset({"icepop", "opsm_trainer", "opsm_rollout", "mis_mask"})


# --------------------------------------------------------------------------
# spec helpers
# --------------------------------------------------------------------------


def _function(spec: AlgorithmSpec) -> str | None:
    function = spec.correction.function
    return function.path if function is not None else None


def _uses_opsm(spec: AlgorithmSpec) -> bool:
    # ``opsm_delta`` marks OPSM (P0: method='opsm'; 1a-shared.patch lets it
    # combine with tis/custom, and this detection covers both forms).
    return spec.correction.opsm_delta is not None


def selected_corrections(spec: AlgorithmSpec) -> tuple[str, ...]:
    """Named correction mechanisms of ``spec`` (see module table)."""

    out = []
    c = spec.correction
    fn = _function(spec)
    if c.method == "tis":
        out.append("tis")
    if fn == OBSERVE_PATH:
        out.append("mismatch_observe")
    if fn == ICEPOP_PATH:
        out.append("icepop")
    if fn == MIS_PATH:
        out.append("mis_mask" if c.mis_mode == "mask" else "mis")
    if _uses_opsm(spec):
        out.append(f"opsm_{c.opsm_old_logprob_source}")
    return tuple(out)


def icepop_ref() -> PluginRef:
    return PluginRef(ICEPOP_PATH, ICEPOP_SOURCE_SHA256)


def observe_ref() -> PluginRef:
    return PluginRef.from_path(OBSERVE_PATH)


def mis_ref() -> PluginRef:
    return PluginRef.from_path(MIS_PATH)


def observe_spec(**overrides: Any) -> AlgorithmSpec:
    """Default GRPO + observe-only mismatch metrics."""

    return AlgorithmSpec(
        correction={"method": "custom", "function": observe_ref().to_dict(),
                    "mismatch_metrics": True},
        **overrides,
    )


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


def _optional_choice(options):
    def parse(path: str, value: Any):
        return None if value is None else _choice(path, value, options)

    return parse


def _positive(path: str, value: Any):
    return _number(path, value, low=0.0, low_open=True)


# With 1a-shared.patch (FieldDef.always_emit) the source is written to the
# canonical JSON whenever OPSM is selected, even at its default "trainer".
_ALWAYS_EMIT = (
    {"always_emit": lambda group: group.opsm_delta is not None}
    if "always_emit" in inspect.signature(register_field).parameters else {}
)
register_field(
    "correction", "opsm_old_logprob_source", default="trainer",
    parse=lambda path, value: _choice(path, value, OPSM_SOURCES), **_ALWAYS_EMIT,
)
register_field("correction", "mis_level", default=None, parse=_optional_choice(MIS_LEVELS))
register_field("correction", "mis_mode", default=None, parse=_optional_choice(MIS_MODES))
register_field("correction", "mis_lower_bound", default=None, parse=_positive)
register_field("correction", "mis_upper_bound", default=None, parse=_positive)
register_field("correction", "mis_batch_normalize", default=False, parse=_boolean)


# --------------------------------------------------------------------------
# mechanisms (design D7/D8)
# --------------------------------------------------------------------------

# With 1a-shared.patch these functions no longer also require the generic
# ("corrections", "custom") mechanism, so an engine can declare them alone.
if hasattr(_algorithm, "register_named_correction_function"):
    # P0 accepts only yeto. paths; IcePop (a miles. path) keeps requiring
    # corrections:custom as well.
    for _path in (OBSERVE_PATH, MIS_PATH):
        _algorithm.register_named_correction_function(_path)

for _name in CORRECTION_MECHANISMS:
    if _name == "tis":
        continue  # P0 built-in ("corrections", "tis"), masks nothing
    register_mechanism(
        "corrections", _name,
        lambda spec, n=_name: n in selected_corrections(spec),
        masks_tokens=_name in MASKING_MECHANISMS,
    )


# --------------------------------------------------------------------------
# rejections (spec: 阈值必须显式给出 / 至多一种修正方式 / OPSM 来源)
# --------------------------------------------------------------------------


def _reject_bounds(spec: AlgorithmSpec) -> str | None:
    c = spec.correction
    fn = _function(spec)
    if fn == ICEPOP_PATH and (c.tis_clip is None or c.tis_clip_low is None):
        missing = [n for n in ("tis_clip", "tis_clip_low") if getattr(c, n) is None]
        return (
            f"IcePop needs explicit interval bounds: missing "
            f"{', '.join('correction.' + m for m in missing)} (no engine default is inherited)"
        )
    if (c.method == "tis" or fn == ICEPOP_PATH) and c.tis_clip is not None \
            and c.tis_clip_low is not None and not c.tis_clip_low < c.tis_clip:
        what = "IcePop interval" if fn == ICEPOP_PATH else "TIS clip"
        return (
            f"{what}: correction.tis_clip_low={c.tis_clip_low} must be smaller than "
            f"correction.tis_clip={c.tis_clip}"
        )
    return None


def _reject_observe(spec: AlgorithmSpec) -> str | None:
    c = spec.correction
    if _function(spec) != OBSERVE_PATH:
        return None
    problems = []
    if c.tis_clip is not None or c.tis_clip_low is not None:
        problems.append("correction.tis_clip/tis_clip_low (observe-only weights are 1)")
    if _uses_opsm(spec):
        problems.append("OPSM (observe-only cannot be combined with any correction)")
    if not c.mismatch_metrics:
        problems.append("correction.mismatch_metrics must be true (--get-mismatch-metrics)")
    if problems:
        return "observe-only mismatch mode: " + "; ".join(problems)
    return None


def _reject_opsm_source(spec: AlgorithmSpec) -> str | None:
    c = spec.correction
    source = c.opsm_old_logprob_source
    if not _uses_opsm(spec):
        if source != "trainer":
            return (
                f"correction.opsm_old_logprob_source={source!r} requires OPSM "
                "(correction.opsm_delta)"
            )
        return None
    note = (
        "--use-rollout-logprobs replaces pi_old for the PPO ratio as well, not only for OPSM"
    )
    if source == "rollout" and not c.use_rollout_logprobs:
        return (
            "correction.opsm_old_logprob_source='rollout' is translated to "
            f"--use-rollout-logprobs: set correction.use_rollout_logprobs=true ({note})"
        )
    if source == "trainer" and c.use_rollout_logprobs:
        return (
            "correction.use_rollout_logprobs=true makes OPSM use the rollout logprobs: set "
            f"correction.opsm_old_logprob_source='rollout' explicitly ({note})"
        )
    if source == "rollout" and c.method in ("tis", "custom"):
        return (
            "OPSM with opsm_old_logprob_source='rollout' cannot be combined with "
            f"correction.method={c.method!r}: {note}, which makes the importance weights "
            "rollout vs rollout; choose opsm_old_logprob_source='trainer'"
        )
    return None


def _reject_mis(spec: AlgorithmSpec) -> str | None:
    c = spec.correction
    given = [n for n in MIS_FIELDS if getattr(c, n) != AlgorithmSpec.default_at(f"correction.{n}")]
    if _function(spec) != MIS_PATH:
        if given:
            return (
                f"{', '.join('correction.' + n for n in given)} require the MIS function "
                f"(correction.function={MIS_PATH})"
            )
        return None
    problems = []
    for name in ("mis_level", "mis_mode", "mis_upper_bound"):
        if getattr(c, name) is None:
            problems.append(f"correction.{name} is required")
    if c.mis_mode in ("clip", "mask") and c.mis_lower_bound is None:
        problems.append(f"correction.mis_lower_bound is required for mis_mode={c.mis_mode!r}")
    if c.mis_mode == "truncate" and c.mis_lower_bound is not None:
        problems.append("correction.mis_lower_bound is unused by mis_mode='truncate'; drop it")
    if (c.mis_lower_bound is not None and c.mis_upper_bound is not None
            and not c.mis_lower_bound < c.mis_upper_bound):
        problems.append(
            f"correction.mis_lower_bound={c.mis_lower_bound} must be smaller than "
            f"correction.mis_upper_bound={c.mis_upper_bound}"
        )
    if c.mis_batch_normalize and c.mis_level == "geometric":
        problems.append("correction.mis_batch_normalize supports mis_level token|sequence only")
    if c.tis_clip is not None or c.tis_clip_low is not None:
        problems.append("MIS uses mis_lower_bound/mis_upper_bound, not correction.tis_clip*")
    if problems:
        return "MIS: " + "; ".join(problems)
    return None


register_rejection("correction_bounds", _reject_bounds)
register_rejection("mismatch_observe_only", _reject_observe)
register_rejection("opsm_logprob_source", _reject_opsm_source)
register_rejection("mis_fields", _reject_mis)


# --------------------------------------------------------------------------
# MIS parameters: Miles namespace attributes (register_runtime_attrs)
# --------------------------------------------------------------------------


def mis_config(spec: AlgorithmSpec) -> dict[str, Any]:
    """Namespace attributes ``mis.compute_mis_weights`` reads (RS/veto off).

    Set by the adapter on the parsed Miles namespace
    (``MilesLaunchArgs.runtime_attrs``), i.e. after Miles' own
    ``--custom-config-path`` handling, so they cannot be overridden there.
    Empty unless the MIS function is selected.
    """

    if _function(spec) != MIS_PATH:
        return {}
    c = spec.correction
    return {
        "rs_level": c.mis_level,
        "rs_lower_bound": None,
        "rs_upper_bound": None,
        "rs_veto_threshold": None,
        "tis_batch_normalize": bool(c.mis_batch_normalize),
        "tis_level": c.mis_level,
        "tis_lower_bound": c.mis_lower_bound,
        "tis_mode": c.mis_mode,
        "tis_upper_bound": c.mis_upper_bound,
        "use_rs": False,
    }


register_runtime_attrs("mismatch_mis", mis_config)


# --------------------------------------------------------------------------
# masked fraction (design D7): derived from the Miles reported-loss metrics
# --------------------------------------------------------------------------


def masked_fraction_from_metrics(spec: AlgorithmSpec, metrics: Mapping[str, Any]) -> float | None:
    """Fraction of loss tokens a masking mechanism zeroed this round, or None.

    ``metrics`` is the Miles reported-loss dict of the round (keys as Miles
    logs them, with or without a ``train/`` prefix). IcePop: ``tis_clipfrac``
    (tokens whose ratio is outside the interval). MIS mask mode:
    ``mis_tis_mask_fraction_low + mis_tis_mask_fraction_high``. OPSM reports
    ``opsm_clipfrac``, which is not a fraction of tokens (Miles
    ``compute_opsm_mask`` sums 1/len per masked sequence), so OPSM yields None
    and the stricter default gradient expectation applies. None also when a
    key is missing. With several masking mechanisms the result is None.
    """

    def get(key: str) -> float | None:
        for candidate in (key, f"train/{key}"):
            if candidate in metrics and metrics[candidate] is not None:
                return float(metrics[candidate])
        return None

    masking = [m for m in selected_corrections(spec) if m in MASKING_MECHANISMS]
    if len(masking) != 1:
        return None
    (name,) = masking
    if name == "icepop":
        value = get("tis_clipfrac")
    elif name == "mis_mask":
        low, high = get("mis_tis_mask_fraction_low"), get("mis_tis_mask_fraction_high")
        value = None if low is None or high is None else low + high
    else:
        value = None
    if value is None or value != value:  # NaN
        return None
    return min(max(value, 0.0), 1.0)
