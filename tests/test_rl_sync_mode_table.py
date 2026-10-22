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
    assert [r.split("|")[1].split()[0] for r in rows[2:]] == ["strict", "decoupled", "island"]


def test_spec_table_matches_the_docstring():
    spec = REPO / "openspec/changes/rl-code-quality-cleanup/specs/rl-codebase-consistency/spec.md"
    if not spec.exists():
        archived = sorted(REPO.glob("openspec/specs/rl-codebase-consistency/spec.md"))
        if not archived:
            import pytest

            pytest.skip("rl-codebase-consistency spec not present")
        spec = archived[0]
    assert _rows(spec.read_text()) == _rows(bridges.__doc__)


def test_no_second_bridge_or_critic_sync_in_production_code():
    """2.1/2.7: the dead elastic bridge, its factory and the in-process
    two-role critic average are gone; PPO-family critic sync is
    DualStrictAvgSync only.  (StrictRlBridge stays until the legacy engine is
    deleted, task 1.10; StrictAvgSync delegates to it.)"""
    hits = []
    for path in sorted((REPO / "yeto").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in ("class ElasticRlBridge", "def make_island_bridge", "class TwoRoleStrictAvg"):
            if name in text:
                hits.append(f"{path.relative_to(REPO)}: {name}")
    assert hits == []
    assert hasattr(bridges.DualStrictAvgSync, "_critic_catchup")
