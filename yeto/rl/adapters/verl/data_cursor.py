"""verl island data cursor (RolloutPool ``data_cursor`` / ``seek_data_cursor``).

S17 N16: V2 relaunched island 1 at v2 in a new container; verl's train
dataloader started at offset 0 again, so from round 2 on it re-drew the prompts
of its own round 0 (same ``prompts_sha256``). The neutral driver can move the
data source by whole rounds (``bridges.whole_round_restart_cursor``) only when
the pool reports and seeks a cursor; this module gives the verl pool one.

The cursor is ``{"sample_offset": prompts drawn from the train dataloader}``.
verl's sync trainer draws prompts only through ``trainer._fetch_one_gen_batch``
(``gen_batch_size`` prompts per call, re-iterating the dataloader at epoch end),
so the counter wraps that method; seeking forward draws and discards whole
chunks through the same method, which reproduces the epoch/shuffle order of an
uninterrupted run. Seeking backwards is refused (a fresh dataloader cannot be
rewound without verl's own checkpoint).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CURSOR_KEY = "sample_offset"


class VerlDataCursor:
    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.drawn = 0
        original = trainer._fetch_one_gen_batch
        if getattr(original, "_yeto_cursor", None) is not None:
            raise RuntimeError("verl trainer data cursor installed twice")

        def counted(*args: Any, **kwargs: Any) -> Any:
            batch = original(*args, **kwargs)
            self.drawn += len(batch)
            return batch

        counted._yeto_cursor = self  # type: ignore[attr-defined]
        self._draw = counted
        trainer._fetch_one_gen_batch = counted

    def cursor(self) -> dict[str, int]:
        return {CURSOR_KEY: int(self.drawn)}

    def _chunk(self) -> int:
        data = self.trainer.config.data
        get = data.get if hasattr(data, "get") else (lambda k, d=None: getattr(data, k, d))
        return int(get("gen_batch_size", None) or get("train_batch_size"))

    def seek(self, cursor: Mapping[str, int]) -> dict[str, int]:
        if set(cursor) != {CURSOR_KEY}:
            raise ValueError(f"verl data cursor is {{{CURSOR_KEY!r}}} only, got {dict(cursor)}")
        target = int(cursor[CURSOR_KEY])
        if target < self.drawn:
            raise ValueError(f"verl data cursor cannot move back from {self.drawn} to {target}")
        chunk = self._chunk()
        if (target - self.drawn) % chunk:
            raise ValueError(f"verl data cursor seek {self.drawn}->{target} is not a whole number "
                             f"of {chunk}-prompt dataloader fetches")
        while self.drawn < target:
            self._draw()
        return self.cursor()
