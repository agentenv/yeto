"""Per-outer-step merge metrics from the elastic syncer tape (rl-trainer-start-stop D9, L5.5).

The elastic syncer (``syncer/src/elastic.rs``) writes one JSONL record per
event, keyed by ``kind``. :class:`MergeMetricsFolder` reads those records in
order and returns one flat metrics dict per ``outer_step``:

* ``merge/island/<id>/weight_share`` - the island's normalized merge weight
  (``w = c_tokens**2 / c_steps``, carried deltas discounted by
  ``carry_gamma**lag``), summed over its entries in the step.
* ``merge/carried_weight_share`` - the share of the step's weight that comes
  from carried (late) deltas; ``merge/carried_fraction`` - carried deltas
  over all deltas submitted in the round; ``merge/carried_discount_mean``.
* ``merge/rejected_fraction`` and ``merge/rejected_fraction/<reason>`` -
  rejected deltas over all deltas submitted in the round.
* Timing from the tape clock (L5.7) when present: ``merge/step_seconds``,
  ``merge/last_arrival_island``, ``merge/time_unix``.

"Submitted in the round" counts every ``delta_accepted``,
``delta_carried_over`` and ``delta_rejected`` since the previous
``outer_step``. A carried delta is counted in the round where it arrived,
which is the step that merges it. Values that the tape does not hold stay
absent (never 0).
"""

from __future__ import annotations

from collections import Counter
from typing import Any


class MergeMetricsFolder:
    """Fold elastic tape records into one metrics dict per outer step."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._accepted = 0
        self._carried_discounts: list[float] = []
        self._rejected: Counter[str] = Counter()

    def feed(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Consume one tape record; return the metrics on ``outer_step``."""
        kind = record.get("kind")
        if kind == "delta_accepted":
            self._accepted += 1
        elif kind == "delta_carried_over":
            discount = record.get("discount")
            self._carried_discounts.append(float(discount) if discount is not None else float("nan"))
        elif kind == "delta_rejected":
            self._rejected[str(record.get("reason", "unknown"))] += 1
        elif kind == "outer_step":
            metrics = self._step_metrics(record)
            self._reset()
            return metrics
        return None

    def _step_metrics(self, rec: dict[str, Any]) -> dict[str, Any]:
        base_version = int(rec.get("base_version", 0))
        metrics: dict[str, Any] = {
            "global_step": base_version + 1,
            "merge/base_version": base_version,
            "merge/arrived": int(rec.get("arrived", 0)),
            "merge/timed_out": int(bool(rec.get("timed_out", False))),
        }
        shares: dict[int, float] = {}
        carried_share = 0.0
        carried_in_step = 0
        for island, entry_base, weight in rec.get("weights") or []:
            shares[int(island)] = shares.get(int(island), 0.0) + float(weight)
            if int(entry_base) < base_version:
                carried_share += float(weight)
                carried_in_step += 1
        for island, share in sorted(shares.items()):
            metrics[f"merge/island/{island}/weight_share"] = share
        metrics["merge/carried_weight_share"] = carried_share
        metrics["merge/carried_in_step"] = carried_in_step

        rejected = self._rejected
        if "rejected" in rec:  # the step's own list wins over the counted events
            rejected = Counter(str(reason) for _island, reason in rec["rejected"])
        carried = len(self._carried_discounts)
        submitted = self._accepted + carried + sum(rejected.values())
        metrics["merge/submitted"] = submitted
        if submitted:
            metrics["merge/carried_fraction"] = carried / submitted
            metrics["merge/rejected_fraction"] = sum(rejected.values()) / submitted
            for reason, count in sorted(rejected.items()):
                metrics[f"merge/rejected_fraction/{reason}"] = count / submitted
        if self._carried_discounts:
            metrics["merge/carried_discount_mean"] = sum(self._carried_discounts) / carried

        for source, name in (("step_seconds", "merge/step_seconds"),
                             ("last_arrival_island", "merge/last_arrival_island"),
                             ("time_unix", "merge/time_unix")):
            if rec.get(source) is not None:
                metrics[name] = rec[source]
        return metrics
