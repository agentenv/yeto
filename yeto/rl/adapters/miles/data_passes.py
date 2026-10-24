"""Miles: every pass over an island's rows after the first is reshuffled (S27 task assignment).

The rule is ``data_split.pass_order``: pass 0 is the island's rows in data
order, pass ``p >= 1`` a shuffle seeded by the run seed and ``p``. Miles'
``RolloutDataSource.get_samples`` starts the next epoch in the same order
unless ``--rollout-shuffle`` (which also shuffles epoch 0), so on an island
with a data slot the adapter wraps ``get_samples`` in the rollout executor:
a draw that reaches the end of the rows takes the rest of the pass, starts
the next epoch in its pass order and draws the remaining prompts there. Each
part is an ordinary Miles draw (it never wraps), so the group and sample
counters move as Miles moves them. ``epoch_id`` is the pass number, and a seek
to a cursor puts the rows in that pass's order (:func:`order_for_epoch`).

The wrapper runs inside the rollout executor actor; the pool installs it with
``__ray_call__`` before the first draw (``MilesRolloutPool``). It does nothing
when Miles shuffles itself (``--rollout-shuffle``) or the source has no dataset.
"""

from __future__ import annotations

from typing import Any

from yeto.rl.engine.data_split import pass_order

_ATTR = "_yeto_data_passes"


class DataPasses:
    """The pass order over one Miles data source's dataset (module doc)."""

    def __init__(self, source: Any, *, seed: int) -> None:
        self.source = source
        self.seed = int(seed)
        self._draw = source.get_samples
        source.get_samples = self.get_samples
        # a seek before the install may already have moved the source to a later pass
        self.order(int(source.epoch_id))

    def order(self, epoch: int) -> None:
        dataset = self.source.dataset
        rows = dataset.origin_samples
        dataset.samples = [rows[i] for i in pass_order(len(rows), seed=self.seed, data_pass=epoch)]
        dataset.epoch_id = int(epoch)

    def get_samples(self, num_samples: int) -> list:
        source, wanted = self.source, int(num_samples)
        if not len(source.dataset):
            raise ValueError("the island's rows are empty (every prompt was filtered out)")
        groups: list = []
        while True:
            left = len(source.dataset) - int(source.sample_offset)
            if wanted <= left:
                return groups + self._draw(wanted)
            if left > 0:
                groups += self._draw(left)
                wanted -= left
            source.epoch_id = int(source.epoch_id) + 1
            source.sample_offset = 0
            self.order(source.epoch_id)


def install(executor: Any, *, seed: int) -> bool:
    """Runs INSIDE the rollout executor actor: wrap its data source once
    (idempotent). False when Miles orders the rows itself or there is no dataset."""
    source = getattr(executor, "data_source", None)
    source = getattr(source, "__self__", source)
    if source is None:
        return False
    if getattr(source, _ATTR, None) is not None:
        return True
    if (getattr(source, "dataset", None) is None
            or getattr(getattr(source, "args", None), "rollout_shuffle", False)):
        return False
    setattr(source, _ATTR, DataPasses(source, seed=seed))
    return True


def order_for_epoch(source: Any, epoch: int) -> bool:
    """After a seek: put the source's rows in ``epoch``'s pass order when the pass
    order is installed; False when it is not (Miles' own shuffle rule applies)."""
    passes = getattr(source, _ATTR, None)
    if passes is None:
        return False
    passes.order(int(epoch))
    return True
