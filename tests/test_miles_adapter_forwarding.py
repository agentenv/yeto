"""yeto-framework-decoupling 5.1: the old ``yeto.rl.engine.miles_adapter`` path
forwards to ``yeto.rl.adapters.miles`` and yields the very same objects."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

NEW = Path(__file__).resolve().parents[1] / "yeto" / "rl" / "adapters" / "miles"
OLD = NEW.parents[1] / "engine" / "miles_adapter"
# Modules that existed at the move; modules added later live only at the new path.
MODULES = sorted(p.stem for p in OLD.glob("*.py") if p.stem != "__init__")


def test_every_forwarder_has_a_moved_module():
    assert len(MODULES) == 28
    assert all((NEW / f"{m}.py").is_file() for m in MODULES)


@pytest.mark.parametrize("name", MODULES)
def test_old_path_is_the_same_module_object(name):
    new = importlib.import_module(f"yeto.rl.adapters.miles.{name}")
    old = importlib.import_module(f"yeto.rl.engine.miles_adapter.{name}")
    assert old is new
    assert old.__file__ == new.__file__ and "/adapters/miles/" in new.__file__


def test_package_reexports_loop_runner():
    from yeto.rl.adapters.miles import LoopRunner as new
    from yeto.rl.engine.miles_adapter import LoopRunner as old
    assert old is new
