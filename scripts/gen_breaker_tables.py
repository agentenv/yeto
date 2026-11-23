#!/usr/bin/env python3
"""Regenerate the reference tables of docs/BREAKERS.md from the breaker registry.

Usage: python3 scripts/gen_breaker_tables.py   (rewrites the generated block of docs/BREAKERS.md)
The tables are ``yeto breaker list --markdown``; the text around the block is written by hand.
A test (tests/test_breakers_docs.py) fails when the tables drift from the registry.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yeto.breakers.cli import render_markdown  # noqa: E402
from yeto.breakers.registry import default_registry  # noqa: E402
from yeto.breakers.spec import LAYERS  # noqa: E402

OUT = ROOT / "docs" / "BREAKERS.md"
BEGIN = "<!-- BEGIN GENERATED: scripts/gen_breaker_tables.py -->"
END = "<!-- END GENERATED -->"


def tables() -> str:
    return render_markdown(default_registry(), LAYERS)


def render(text: str) -> str:
    """``text`` with its generated block replaced by the current tables."""
    head, sep, rest = text.partition(BEGIN)
    _, sep2, tail = rest.partition(END)
    if not sep or not sep2:
        raise ValueError(f"{OUT.name}: no generated block ({BEGIN} ... {END})")
    return f"{head}{BEGIN}\n\n{tables()}\n\n{END}{tail}"


def main() -> None:
    OUT.write_text(render(OUT.read_text()))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
