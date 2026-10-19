"""Keep ABORTED samples out of group baselines (S25 N10.3b).

An ABORTED trajectory (infrastructure failure, misaligned tokens) is never
trained: it carries no loss tokens and reward 0.  A group-relative estimator
(GRPO) still averages every row of a prompt group, so a placeholder row
would pull its siblings' baseline toward 0 and inflate their advantages.

The backends drop it differently, with the same invariant: an ABORTED
sample never enters a group mean.

* Miles: the dynamic-sampling filter ``check_no_aborted`` drops a group that
  holds an ABORTED sample, and Miles samples a new group in its place
  (``harbor_agent.train_args``).
* verl: a batch has no per-sample drop, so the placeholder stays in the batch
  and :func:`isolate_rows` gives it a group of its own before the advantage
  step (``adapters.verl.advantage``).  Its siblings keep their group.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

ISOLATED = "yeto-isolated"


def isolate_rows(group_ids: Sequence[Hashable], excluded: Sequence[bool]) -> list[Hashable]:
    """``group_ids`` with each excluded row moved to a one-row group of its own.

    The new id ``(ISOLATED, row)`` cannot equal an original id (verl uses
    strings), so no other row joins it.
    """
    if len(group_ids) != len(excluded):
        raise ValueError(f"{len(group_ids)} group ids but {len(excluded)} exclusion flags")
    return [(ISOLATED, row) if out else gid for row, (gid, out) in enumerate(zip(group_ids, excluded))]
