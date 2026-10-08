"""R-D5a chain-count judge for Codex GPU runs (rl-codex-harness-rollout design 9.2).

The GPU 9.2 hard criterion used to be stated only in prose: every trajectory
has ``chains_total == 1``.  With CompactionRL rollout compaction on
(``YETO_CODEX_COMPACTIONRL``, progress.md "S13 Codex 压缩接线") a trajectory is
deliberately split into one sample per compaction segment, so the criterion
becomes, per trajectory:

- ``chains_total == num_segments == compactions + 1 == number of samples``;
- ``chain_index`` / ``segment_index`` are exactly ``0..n-1``;
- ``chain_break_reason`` is ``compaction_window`` on every sample but the first
  (None there).

With compaction off the criterion is unchanged: ``chains_total == 1`` on every
sample (and one sample per trajectory).  Aborted (infrastructure) samples are
judged the same way: the generate wrapper keeps an aborted compacted
trajectory as its single segment-0 sample, which satisfies the off rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

COMPACTION_BREAK_REASON = "compaction_window"


def _key(meta: Mapping[str, Any], position: int) -> str:
    trajectory = meta.get("trajectory_id")
    return str(trajectory) if trajectory not in (None, "") else f"<no trajectory_id #{position}>"


def judge_chain_counts(
    samples_metadata: Iterable[Mapping[str, Any]],
    *,
    compaction_enabled: bool,
) -> list[str]:
    """Return the violations (empty list = criterion holds)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for position, meta in enumerate(samples_metadata):
        groups.setdefault(_key(meta, position), []).append(meta)
    problems: list[str] = []
    for trajectory, metas in groups.items():
        totals = {m.get("chains_total") for m in metas}
        if not compaction_enabled:
            if totals != {1} or len(metas) != 1:
                problems.append(
                    f"{trajectory}: chains_total {sorted(map(str, totals))} over {len(metas)} sample(s); "
                    "expected exactly one sample with chains_total==1"
                )
            continue
        # Aborted compacted trajectories collapse to one sample (no segment keys).
        if len(metas) == 1 and metas[0].get("num_segments") is None:
            if totals != {1}:
                problems.append(f"{trajectory}: single sample with chains_total {sorted(map(str, totals))}")
            continue
        n = len(metas)
        segments = {m.get("num_segments") for m in metas}
        compactions = {m.get("compactions") for m in metas}
        if totals != {n} or segments != {n}:
            problems.append(
                f"{trajectory}: {n} sample(s) but chains_total {sorted(map(str, totals))}, "
                f"num_segments {sorted(map(str, segments))}"
            )
        if compactions != {n - 1}:
            problems.append(
                f"{trajectory}: compactions {sorted(map(str, compactions))} != num_segments-1 ({n - 1})"
            )
        order = sorted(metas, key=lambda m: (m.get("chain_index") is None, m.get("chain_index") or 0))
        if [m.get("chain_index") for m in order] != list(range(n)) or [
            m.get("segment_index") for m in order
        ] != list(range(n)):
            problems.append(f"{trajectory}: chain_index/segment_index are not 0..{n - 1}")
            continue
        reasons = [m.get("chain_break_reason") for m in order]
        if reasons != [None] + [COMPACTION_BREAK_REASON] * (n - 1):
            problems.append(f"{trajectory}: chain_break_reason {reasons}")
    return problems
