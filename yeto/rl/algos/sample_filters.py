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
  the *terminal* state ``filtered`` (reason = the key). Over-sampling leftovers
  that Miles keeps in its buffer are *not* filtered here; the ledger records
  them as the non-terminal ``carried_over``.

Default configuration (no ``args.yeto_algo_plugins`` or no
``overlong_filter``): no sample is touched and the counts attribute is not set
(the hook behaves exactly as before).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .reward_pipeline import read_plugins

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

    config = read_plugins(args, required=False)
    if not config or not config.get("overlong_filter"):
        return None
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
