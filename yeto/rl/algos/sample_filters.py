"""overlong filtering for the shared ``record_trained_groups`` hook
(change ``rl-algo-grpo-knobs`` design D7; rl-infra-spec alignment A2/F5).

``yeto.rl.engine.miles_adapter.rollout_meta_hook.record_trained_groups`` is the
single ``--rollout-sample-filter-path`` of the ports path, shared with the
rl-infra-spec 3.6 ledger. The hook calls :func:`apply_sample_filters` *before*
recording the kept groups (patch ``infra-drafts/1b-hook.patch``); nothing else
may open another ``--rollout-sample-filter-path``.

Field contract (alignment A2/F5):

* a filtered sample keeps its place in its group (its reward still enters the
  group's advantage statistics) but gets ``sample.remove_sample = True`` -- Miles
  zeroes its loss mask (``train_data_conversion.py:99-100``);
* ``sample.metadata["yeto_filtered_by"] = "overlong_filter"`` records why;
* ``args._yeto_sample_filter_counts = {"overlong_filter": n}`` is read by
  ``build_metadata`` and shipped as rollout metadata
  ``filtered_samples = {"overlong_filter": n}`` plus per-group
  ``filtered_samples`` (int). The infra ledger records these samples with
  the *terminal* state ``filtered`` (reason = the key). This module never
  touches over-sampling: Miles does not return surplus completed groups to its
  buffer (``sglang_rollout.py:505-510``: once ``data`` is full, further kept
  groups are simply not added; only samples aborted under ``--partial-rollout``
  go back), so over-sampling leaves no non-terminal remainder here.

Default configuration (no ``overlong_filter``): no sample is touched and the
counts attribute is not set (the hook behaves exactly as before).

Fail-closed (review F3): on a ports run (``yeto_rl_expected_algorithm_sha256``
set on the Miles namespace) whose runtime attrs did not reach this process
(neither ``yeto_algo_plugins`` nor the always-present
``yeto_rl_dynamic_sampling_max_replacements``), the hook raises instead of
silently skipping a filter the spec may enable. The plugins payload is also
checked against the expected algorithm hash (``read_plugins``) and against
this module's source hash.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .reward_pipeline import (
    SAMPLE_FILTERS_PATH,
    RewardPipelineError,
    expected_algorithm_sha256,
    plugin_sha,
    read_plugins,
    runtime_attrs_delivered,
)

FILTER_COUNTS_ATTR = "_yeto_sample_filter_counts"
FILTERED_BY_KEY = "yeto_filtered_by"
OVERLONG_FILTER = "overlong_filter"


def _flat(group: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in group:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def _status(sample: Any) -> str:
    status = getattr(sample, "status", None)
    return str(getattr(status, "value", status) or "")


def overlong_filter(data: Sequence[Sequence[Any]]) -> int:
    """Mark truncated samples ``remove_sample=True``; returns the count."""

    removed = 0
    for group in data:
        for sample in _flat(group):
            if _status(sample) == "truncated" and not getattr(sample, "remove_sample", False):
                sample.remove_sample = True
                if getattr(sample, "metadata", None) is None:
                    sample.metadata = {}
                sample.metadata[FILTERED_BY_KEY] = OVERLONG_FILTER
                removed += 1
    return removed


def apply_sample_filters(args: Any, data: Sequence[Sequence[Any]]) -> dict[str, int] | None:
    """Run the spec-selected sample filters on the kept groups (in place)."""

    if expected_algorithm_sha256(args) is not None and not runtime_attrs_delivered(args):
        raise RewardPipelineError(
            "sample filters: this ports run's runtime attrs (yeto_algo_plugins) did not reach "
            "the Miles rollout process; refusing to skip a filter the spec may enable"
        )
    config = read_plugins(args, required=False)
    if not config or not config.get("overlong_filter"):
        return None
    actual = plugin_sha(SAMPLE_FILTERS_PATH)
    if config.get("sample_filters_sha256") != actual:
        raise RewardPipelineError(
            f"sample filters source {actual} differs from the spec's "
            f"{config.get('sample_filters_sha256')}"
        )
    counts = {OVERLONG_FILTER: overlong_filter(data)}
    setattr(args, FILTER_COUNTS_ATTR, counts)
    return counts


def group_filtered_samples(group: Sequence[Any]) -> int:
    return sum(1 for s in _flat(group) if getattr(s, "remove_sample", False))


def metadata_fields(args: Any, trained_groups: Sequence[Sequence[Any]] = ()) -> dict[str, Any]:
    """Extra rollout-metadata fields (empty when no filter ran)."""

    counts = getattr(args, FILTER_COUNTS_ATTR, None)
    if not counts:
        return {}
    return {"filtered_samples": dict(counts)}


def reset(args: Any) -> None:
    if hasattr(args, FILTER_COUNTS_ATTR):
        setattr(args, FILTER_COUNTS_ATTR, None)
