"""Golden fixture: Python ``island_ledger.merge_weight`` (tokens^2 / steps) is
locked to the Rust ``merge::learner_weight``
(syncer/src/merge.rs::tests::python_merge_weight_golden_matches).

Regenerate: python tests/test_rl_inter_island_rust_golden.py --write
"""
from __future__ import annotations

import sys
from pathlib import Path

from yeto.rl.engine import island_ledger

REPO = Path(__file__).resolve().parents[1]
WEIGHT_FIXTURE = REPO / "syncer/tests/fixtures/merge_weight_golden.txt"
WEIGHT_CASES = ((0, 0), (0, 4), (7, 0), (1, 1), (37, 4), (91, 4), (50, 5), (4096, 3),
                (1 << 20, 7), (123457, 13), (2**32 - 1, 1), (5, 2**32 + 9), (10**9, 999983))


def render_weights() -> str:
    """"<c_tokens> <c_steps> => <weight repr>"; read by
    syncer/src/merge.rs::tests::python_merge_weight_golden_matches."""
    return "".join(f"{t} {c} => {island_ledger.merge_weight(t, c)!r}\n" for t, c in WEIGHT_CASES)


def test_merge_weight_fixture_is_current():
    """Python's merge_weight copy (island-local ledger, no Rust build) is
    locked to the Rust learner_weight through this fixture: the Rust test
    replays it bit for bit."""
    assert WEIGHT_FIXTURE.read_text() == render_weights()


if __name__ == "__main__" and "--write" in sys.argv:
    WEIGHT_FIXTURE.write_text(render_weights())
