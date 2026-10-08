"""yeto-framework-decoupling 5.1: the old ``yeto.rl.engine.miles_adapter`` path
forwards to ``yeto.rl.adapters.miles`` and yields the very same objects."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

NEW = Path(__file__).resolve().parents[1] / "yeto" / "rl" / "adapters" / "miles"
MODULES = sorted(p.stem for p in NEW.glob("*.py") if p.stem != "__init__")


def test_every_adapter_module_has_a_forwarder():
    old = NEW.parents[1] / "engine" / "miles_adapter"
    assert sorted(p.stem for p in old.glob("*.py") if p.stem != "__init__") == MODULES


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
