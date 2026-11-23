"""docs/BREAKERS.md: its reference tables are the registry's (scripts/gen_breaker_tables.py)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location("gen_breaker_tables", ROOT / "scripts" / "gen_breaker_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_tables_match_the_registry():
    gen = _gen()
    text = gen.OUT.read_text()
    assert gen.render(text) == text, "docs/BREAKERS.md is stale: run python3 scripts/gen_breaker_tables.py"


def test_every_registered_entry_is_in_the_docs():
    from yeto.breakers.registry import default_registry

    text = _gen().OUT.read_text()
    missing = [b.name for b in default_registry() if f"`{b.name}`" not in text]
    assert not missing, f"docs/BREAKERS.md does not list {missing}"


# `<layer>.<word>` spellings in the guide's breaker section that name a file or a config key
NOT_BREAKERS = {"run.json", "trainer.argv", "sampler.dispense", "sampler.breakers"}


def test_customer_guide_names_only_registered_entries():
    """Every `<layer>.<entry>` the guide's breaker section names is a registry entry."""
    from yeto.breakers.registry import default_registry
    from yeto.breakers.spec import LAYERS

    guide = (ROOT / "docs" / "KIMI_K31_ASYNC_RL_GUIDE.md").read_text()
    section = guide[guide.index("## 7. Breakers"):guide.index("## 8. ")]
    names = set(re.findall(rf"`((?:{'|'.join(LAYERS)})\.[a-z0-9_]+)`", section))
    registry = default_registry()
    unknown = sorted(n for n in names - NOT_BREAKERS if n not in registry)
    assert not unknown, f"the guide names breakers the registry does not have: {unknown}"
