"""Which part of a vision-language checkpoint is the text model (S27 G-EZ).

A VL checkpoint wraps a text causal LM next to a vision tower and one or more
projectors. RL on the text model must train the text parameters and must not
train the rest, so the rule has to say three things, not one:

* ``text_prefix``: the prefix every text tensor carries in the VL checkpoint;
  stripping it gives the name the text model itself uses;
* ``exclude_prefixes``: whole modules that are not the text model (vision
  tower, projectors). These are excluded, not frozen: nothing reads verl's
  ``freeze_vision_tower`` in fork acad9875, so a tensor that is present is a
  tensor that trains;
* ``exclude_text_subprefixes``: blocks inside the text model that RL has no
  loss for (a multi-token-prediction head). Left in, such a block would only
  decay and then be published.

The rule also names the nested config key holding the text config, and the
config fields that must be turned off so the text config does not declare a
block whose weights were excluded.

Nothing here renames an architecture to get a checkpoint past a check. The
point is the opposite: the selection is explicit and verifiable, so
``adapters/<backend>/nemotron_h.py:check_provider`` keeps refusing a VL
provider and keeps its value as a safety net. What reaches the island is a
text provider whose vision tower and MTP block do not exist.

This module imports no torch and no backend: the checkpoint converter, both
backend adapters and the pre-step correctness check share this one definition.

* :func:`rule_for_config`: the rule for an HF ``config.json`` (or ``None``);
* :func:`text_config`: the text config to write, with excluded blocks off;
* :func:`plan_subset`: source name -> text name over an index ``weight_map``,
  plus what was excluded and why (:class:`TextSubsetError` on an unclassified
  name);
* :func:`check_loaded`: the loaded model's parameters against the plan.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

KIND_TEXT = "text"
KIND_EXCLUDED_MODULE = "excluded_module"
KIND_EXCLUDED_TEXT_BLOCK = "excluded_text_block"


class TextSubsetError(ValueError):
    """A checkpoint does not match the rule (raised before any engine starts)."""


@dataclass(frozen=True)
class TextSubsetRule:
    """The rule for one VL checkpoint family. Data, not behaviour."""

    outer_model_type: str
    text_prefix: str
    text_config_key: str
    exclude_prefixes: tuple[str, ...]
    exclude_text_subprefixes: tuple[str, ...]
    text_config_off: Mapping[str, Any]

    def as_data(self) -> dict[str, Any]:
        """The rule as plain data (hashed into a run manifest by callers)."""
        return {
            "schema": "yeto-text-subset-v1",
            "outer_model_type": self.outer_model_type,
            "text_prefix": self.text_prefix,
            "text_config_key": self.text_config_key,
            "exclude_prefixes": list(self.exclude_prefixes),
            "exclude_text_subprefixes": list(self.exclude_text_subprefixes),
            "text_config_off": dict(self.text_config_off),
        }


# Nemotron-3.5-Super-VL-120B-A12B-BF16: outer ``nemotron_h_omni`` wrapping a
# ``nemotron_h`` text LM (``NemotronHForCausalLM``), a radio vision tower and
# two projectors. The index of that checkpoint carries exactly four top-level
# prefixes: ``language_model.`` (42683 keys), ``vision_model.`` (391),
# ``mlp1.`` (3), ``vision_projector.`` (2). The same prefix rule is what the
# vLLM text-only load on npu1 n4 uses (``survey/ascend_nemotron_serve.md``).
NEMOTRON_H_OMNI = TextSubsetRule(
    outer_model_type="nemotron_h_omni",
    text_prefix="language_model.",
    text_config_key="llm_config",
    exclude_prefixes=("vision_model.", "mlp1.", "vision_projector."),
    exclude_text_subprefixes=("mtp.",),
    # The text config declares one next-token-prediction block
    # (``num_nextn_predict_layers`` 1, ``mtp_layers_block_type``
    # ['attention','moe']); its weights are excluded, so the config must not
    # declare it either.
    text_config_off={"num_nextn_predict_layers": 0, "mtp_layers_block_type": []},
)

RULES: tuple[TextSubsetRule, ...] = (NEMOTRON_H_OMNI,)


@dataclass
class TextSubset:
    """A planned selection over one checkpoint index."""

    rule: TextSubsetRule
    renames: dict[str, str] = field(default_factory=dict)
    excluded: dict[str, list[str]] = field(default_factory=dict)

    @property
    def text_tensors(self) -> int:
        return len(self.renames)

    @property
    def excluded_tensors(self) -> int:
        return sum(len(v) for v in self.excluded.values())

    def summary(self) -> dict[str, Any]:
        return {
            "rule": self.rule.as_data(),
            "text_tensors": self.text_tensors,
            "excluded_tensors": self.excluded_tensors,
            "excluded_by_prefix": {k: len(v) for k, v in sorted(self.excluded.items())},
        }


def rule_for_config(config: Mapping[str, Any]) -> TextSubsetRule | None:
    """The rule for this HF config, or ``None`` when it is not a wrapped checkpoint."""
    model_type = str(config.get("model_type", ""))
    for rule in RULES:
        if rule.outer_model_type == model_type:
            return rule
    return None


def text_config(config: Mapping[str, Any], rule: TextSubsetRule) -> dict[str, Any]:
    """The text ``config.json`` to write: the nested text config, excluded blocks off.

    Raises when the nested config is missing, or when it does not name an
    architecture of its own (the text model must be identified by the
    checkpoint, never by this module).
    """
    nested = config.get(rule.text_config_key)
    if not isinstance(nested, Mapping):
        raise TextSubsetError(
            f"config has no '{rule.text_config_key}' mapping, so the text config is unknown")
    out = dict(nested)
    if not out.get("architectures"):
        raise TextSubsetError(
            f"'{rule.text_config_key}' names no architectures; refusing to invent one")
    if not out.get("model_type"):
        raise TextSubsetError(f"'{rule.text_config_key}' has no model_type")
    for key, value in rule.text_config_off.items():
        if key in out:
            out[key] = value
    for carry in ("dtype", "torch_dtype"):
        if carry in config and carry not in out:
            out[carry] = config[carry]
    return out


def classify(name: str, rule: TextSubsetRule) -> tuple[str, str]:
    """``(kind, matched_prefix)`` for one checkpoint tensor name.

    Raises :class:`TextSubsetError` when the name matches nothing the rule
    names: an unclassified tensor must stop the run, because silently dropping
    it trains the wrong model and silently keeping it trains too much.
    """
    for prefix in rule.exclude_prefixes:
        if name.startswith(prefix):
            return KIND_EXCLUDED_MODULE, prefix
    if name.startswith(rule.text_prefix):
        rest = name[len(rule.text_prefix):]
        for sub in rule.exclude_text_subprefixes:
            if rest.startswith(sub):
                return KIND_EXCLUDED_TEXT_BLOCK, rule.text_prefix + sub
        return KIND_TEXT, rule.text_prefix
    raise TextSubsetError(
        f"tensor {name!r} matches neither the text prefix {rule.text_prefix!r} nor any "
        f"excluded prefix {rule.exclude_prefixes + rule.exclude_text_subprefixes!r}")


def plan_subset(names: Iterable[str], rule: TextSubsetRule) -> TextSubset:
    """Plan the selection over checkpoint tensor names (an index ``weight_map``)."""
    subset = TextSubset(rule=rule)
    for name in names:
        kind, prefix = classify(name, rule)
        if kind == KIND_TEXT:
            subset.renames[name] = name[len(rule.text_prefix):]
        else:
            subset.excluded.setdefault(prefix, []).append(name)
    if not subset.renames:
        raise TextSubsetError(
            f"no tensor carries the text prefix {rule.text_prefix!r}: wrong rule for this checkpoint")
    collisions = len(subset.renames) - len(set(subset.renames.values()))
    if collisions:
        raise TextSubsetError(f"stripping {rule.text_prefix!r} collides on {collisions} name(s)")
    return subset


def check_loaded(
    loaded: Mapping[str, Any],
    subset: TextSubset,
    *,
    expected_params: int | None = None,
    shapes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a built model's parameters against the plan. Raise on any mismatch.

    ``loaded`` maps the model's own parameter names to anything with a
    ``shape`` (a tensor) or to a shape sequence. One missing or one extra name
    fails: this is the check that a rename-only workaround would pass and a
    correct selection must not.

    ``shapes`` (optional) maps the model's names to the shape the source
    checkpoint recorded, so a silently reshaped parameter also fails.
    """
    want = set(subset.renames.values())
    have = set(loaded)
    # An excluded module present in the model is reported before the generic
    # set difference: it is the specific failure this check exists for.
    for prefix in list(subset.rule.exclude_prefixes) + [
        subset.rule.text_prefix + sub for sub in subset.rule.exclude_text_subprefixes
    ]:
        bare = prefix[len(subset.rule.text_prefix):] if prefix.startswith(
            subset.rule.text_prefix) else prefix
        present = sorted(n for n in have if n.startswith(bare) or n.startswith(prefix))
        if present:
            raise TextSubsetError(
                f"excluded module {prefix!r} is present in the model as {present[:3]}: "
                "it would train and be published")
    missing = sorted(want - have)
    extra = sorted(have - want)
    if missing or extra:
        raise TextSubsetError(
            f"parameter set does not match the plan: {len(missing)} missing, {len(extra)} extra; "
            f"first missing {missing[:3]}, first extra {extra[:3]}")
    total = 0
    bad_shapes: list[str] = []
    for name, value in loaded.items():
        shape = tuple(getattr(value, "shape", value) or ())
        count = 1
        for dim in shape:
            count *= int(dim)
        total += count
        if shapes is not None and name in shapes:
            if tuple(int(d) for d in shapes[name]) != tuple(int(d) for d in shape):
                bad_shapes.append(name)
    if bad_shapes:
        raise TextSubsetError(
            f"{len(bad_shapes)} parameter(s) have a shape the checkpoint does not record: "
            f"{bad_shapes[:3]}")
    if expected_params is not None and total != expected_params:
        raise TextSubsetError(
            f"trainable parameters {total} != expected {expected_params} "
            f"(difference {total - expected_params})")
    return {"tensors": len(have), "params": total, **subset.summary()}
