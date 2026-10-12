"""rl-code-quality-cleanup 2.6: the three sync modes' weight / staleness /
finish rules are written in one table (engine/bridges.py docstring) and the
same table is in the spec; this test stops the two from drifting."""
from __future__ import annotations

from pathlib import Path

from yeto.rl.engine import bridges

REPO = Path(__file__).resolve().parents[1]
HEADER = "| mode | merge weight | staleness | finish |"


def _rows(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(HEADER)
    out = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        out.append(line.strip())
    return out


def test_bridges_docstring_has_the_sync_mode_table():
    rows = _rows(bridges.__doc__)
    assert rows[0] == HEADER
    assert [r.split("|")[1].split()[0] for r in rows[2:]] == ["strict", "decoupled", "elastic"]


def test_spec_table_matches_the_docstring():
    spec = REPO / "openspec/changes/rl-code-quality-cleanup/specs/rl-codebase-consistency/spec.md"
    if not spec.exists():
        archived = sorted(REPO.glob("openspec/specs/rl-codebase-consistency/spec.md"))
        if not archived:
            import pytest

            pytest.skip("rl-codebase-consistency spec not present")
        spec = archived[0]
    assert _rows(spec.read_text()) == _rows(bridges.__doc__)
