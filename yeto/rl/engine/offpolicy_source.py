"""Per-source off-policy metrics: the ``offpolicy_source`` event
(rl-hetero-sampler-engines D4, S25 N2.5; grouping per A-INF 4.1).

yeto computes no importance weight for training. The training backend logs, per
consumed sample, its recomputed log-probs next to the rollout log-probs (the
pass-through of ``rl-offpolicy-correction-passthrough``). After each round the
trainer adapter hands those recorded values here, one :class:`SourceSample` per
consumed sample, and this module only groups them by sampler source and
summarises them. The ratio is ``exp(train_logp - rollout_logp)`` per token, as
in :mod:`.mismatch_criteria`.

One entry per provenance key (``Provenance.key()``: device family, card,
engine@version, precision, topology); samples without provenance group under
``unknown``. Per entry:

* ``samples``, ``tokens``;
* ``ratio_p50`` / ``ratio_p90`` / ``ratio_p99`` / ``ratio_max`` over tokens;
* ``tis_clipfrac``: share of tokens with ratio above ``tis_upper``, per-sample
  mean then mean over samples (the :mod:`.mismatch_criteria` convention);
* ``abs_diff_mean`` (per-sample mean, then mean) and ``abs_diff_p99`` (tokens);
* ``k3`` (same convention as :mod:`.mismatch_criteria`);
* ``finish``: count per ``finish_reason``; ``truncated_share``: share ending
  with ``length``; ``loop_abort_share``;
* ``by_length``: the same ``tis_clipfrac`` / ``abs_diff_mean`` / ``samples``
  per response-length bucket (``offpolicy_correction.length_bucket``).

Counts add up to the round's consumed samples (``consumed`` in the event).
Pure Python: no torch.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .mismatch_criteria import DEFAULT_TIS_UPPER, K3_TOKEN_CLAMP, LOG_RATIO_CLAMP
from .offpolicy_correction import length_bucket
from .sampler_engine import FINISH_LENGTH, FINISH_LOOP_ABORT, Provenance

OFFPOLICY_SOURCE_EVENT = "offpolicy_source"
SCHEMA = "yeto-offpolicy-source-v1"


@dataclass(frozen=True)
class SourceSample:
    """One consumed sample: its source record, the backend's recomputed
    log-probs and the rollout log-probs (same length; response tokens only),
    and the completion's ``finish_reason`` (None: not recorded)."""

    provenance: Mapping[str, str] | None
    train_logprobs: Sequence[float]
    rollout_logprobs: Sequence[float]
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.train_logprobs) != len(self.rollout_logprobs):
            raise ValueError(f"train log-probs ({len(self.train_logprobs)}) and rollout "
                             f"log-probs ({len(self.rollout_logprobs)}) differ in length")


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Nearest-rank quantile of sorted values (None when empty)."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[min(rank, len(sorted_values)) - 1])


def _diff(train: float, rollout: float) -> float:
    d = float(train) - float(rollout)
    if not math.isfinite(d):
        return 0.0  # as mismatch_criteria: a non-finite difference counts as 0
    return d


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


class _Group:
    def __init__(self) -> None:
        self.samples = 0
        self.ratios: list[float] = []
        self.abs_diffs: list[float] = []
        self.sample_clip: list[float] = []
        self.sample_abs: list[float] = []
        self.sample_k3: list[float] = []
        self.finish: Counter[str] = Counter()

    def add(self, sample: SourceSample, tis_upper: float) -> tuple[float | None, float | None]:
        self.samples += 1
        self.finish[sample.finish_reason or "unknown"] += 1
        diffs = [_diff(t, r) for t, r in zip(sample.train_logprobs, sample.rollout_logprobs)]
        if not diffs:
            return None, None
        clipped = [max(-LOG_RATIO_CLAMP, min(LOG_RATIO_CLAMP, d)) for d in diffs]
        ratios = [math.exp(d) for d in clipped]
        k3 = [max(-K3_TOKEN_CLAMP, min(K3_TOKEN_CLAMP, math.exp(d) - 1 - d)) for d in clipped]
        self.ratios += ratios
        self.abs_diffs += [abs(d) for d in diffs]
        clip = sum(1 for r in ratios if r > tis_upper) / len(ratios)
        absd = sum(abs(d) for d in diffs) / len(diffs)
        self.sample_clip.append(clip)
        self.sample_abs.append(absd)
        self.sample_k3.append(sum(k3) / len(k3))
        return clip, absd

    def summary(self) -> dict[str, Any]:
        ratios = sorted(self.ratios)
        abs_diffs = sorted(self.abs_diffs)
        return {
            "samples": self.samples,
            "tokens": len(ratios),
            "ratio_p50": _quantile(ratios, 0.50),
            "ratio_p90": _quantile(ratios, 0.90),
            "ratio_p99": _quantile(ratios, 0.99),
            "ratio_max": ratios[-1] if ratios else None,
            "tis_clipfrac": _mean(self.sample_clip),
            "abs_diff_mean": _mean(self.sample_abs),
            "abs_diff_p99": _quantile(abs_diffs, 0.99),
            "k3": _mean(self.sample_k3),
            "finish": dict(sorted(self.finish.items())),
            "truncated_share": self.finish[FINISH_LENGTH] / self.samples if self.samples else None,
            "loop_abort_share": (self.finish[FINISH_LOOP_ABORT] / self.samples
                                 if self.samples else None),
        }


def offpolicy_source_event(samples: Iterable[SourceSample], *, rollout_id: int,
                           trainer_version: int, backend: str,
                           tis_upper: float = DEFAULT_TIS_UPPER) -> dict[str, Any]:
    """The ``offpolicy_source`` event fields of one training round (module doc)."""
    groups: dict[str, _Group] = defaultdict(_Group)
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"clip": [], "abs": [], "n": []}))
    records: dict[str, dict[str, str]] = {}
    consumed = 0
    for sample in samples:
        consumed += 1
        provenance = Provenance.from_record(sample.provenance)
        key = provenance.key()
        records.setdefault(key, {f: getattr(provenance, f) for f in Provenance.KEY_FIELDS})
        clip, absd = groups[key].add(sample, float(tis_upper))
        bucket = buckets[key][length_bucket(len(sample.rollout_logprobs))]
        bucket["n"].append(1.0)
        if clip is not None:
            bucket["clip"].append(clip)
            bucket["abs"].append(absd)
    sources = []
    for key in sorted(groups):
        entry = {"key": key, **records[key], **groups[key].summary()}
        entry["by_length"] = {
            name: {"samples": len(b["n"]), "tis_clipfrac": _mean(b["clip"]),
                   "abs_diff_mean": _mean(b["abs"])}
            for name, b in sorted(buckets[key].items())}
        sources.append(entry)
    return {"schema": SCHEMA, "rollout_id": int(rollout_id), "trainer_version": int(trainer_version),
            "backend": str(backend), "tis_upper": float(tis_upper), "consumed": consumed,
            "sources": sources}


def payload_source_sample(sample: Mapping[str, Any], *, train_logprobs: Sequence[float] = (),
                          rollout_logprobs: Sequence[float] | None = None) -> SourceSample:
    """A :class:`SourceSample` from a buffer payload sample (``metadata.provenance``
    and ``metadata.finish_reason``, as the engine sampler island writes them).

    Without ``train_logprobs`` the sample counts in ``samples`` and ``finish``
    only (its ratio fields stay empty): the reader side of a backend whose
    training process does not hand its log-probs back (Miles).
    """
    metadata = sample.get("metadata") or {}
    rollout: list[float] = []
    if train_logprobs:
        rollout = list(sample.get("rollout_log_probs") or () if rollout_logprobs is None else rollout_logprobs)
    return SourceSample(provenance=metadata.get("provenance"), train_logprobs=list(train_logprobs),
                        rollout_logprobs=rollout, finish_reason=metadata.get("finish_reason"))


def flat_metrics(event: Mapping[str, Any], prefix: str = OFFPOLICY_SOURCE_EVENT) -> dict[str, float]:
    """The numeric per-source fields of an event as ``<prefix>/<key>/<field>`` metrics."""
    out: dict[str, float] = {}
    for entry in event["sources"]:
        for name in ("samples", "tokens", "ratio_p50", "ratio_p99", "tis_clipfrac", "abs_diff_mean",
                     "k3", "truncated_share", "loop_abort_share"):
            value = entry.get(name)
            if isinstance(value, (int, float)):
                out[f"{prefix}/{entry['key']}/{name}"] = float(value)
    return out


def masked_source_samples(train_logprobs: Sequence[Sequence[float]], rollout_logprobs: Sequence[Sequence[float]],
                          masks: Sequence[Sequence[float]], *,
                          provenances: Sequence[Mapping[str, str] | None] | None = None,
                          finish_reasons: Sequence[str | None] | None = None) -> list[SourceSample]:
    """One :class:`SourceSample` per row of a padded trainer batch (nested lists,
    e.g. ``tensor.tolist()``), keeping only the tokens whose mask is set.

    ``provenances`` / ``finish_reasons``: per row, or None when the batch does
    not carry them (the rows then group under ``unknown``).
    """
    rows = len(masks)
    if len(train_logprobs) != rows or len(rollout_logprobs) != rows:
        raise ValueError(f"{len(train_logprobs)} train rows, {len(rollout_logprobs)} rollout rows, {rows} mask rows")
    for name, values in (("provenances", provenances), ("finish_reasons", finish_reasons)):
        if values is not None and len(values) != rows:
            raise ValueError(f"{len(values)} {name} for {rows} rows")
    out = []
    for row in range(rows):
        keep = [i for i, m in enumerate(masks[row]) if m]
        out.append(SourceSample(
            provenance=None if provenances is None else provenances[row],
            train_logprobs=[float(train_logprobs[row][i]) for i in keep],
            rollout_logprobs=[float(rollout_logprobs[row][i]) for i in keep],
            finish_reason=None if finish_reasons is None else finish_reasons[row]))
    return out
