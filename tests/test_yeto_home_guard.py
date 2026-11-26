"""The session keeps out of the developer's real ~/.yeto (tests/yeto_home_guard.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yeto_home_guard as guard


def test_every_state_path_resolves_under_the_session_root():
    from yeto import runs, sandbox
    from yeto.launch.nc import tunnel
    from yeto.rl import up

    root = Path(os.environ[guard.ROOT_ENV])
    for path in (runs.RUNS_DIR, up.STATE_ROOT, tunnel.state_dir(), sandbox.state_root()):
        assert root in Path(path).parents
    for name in guard.STATE_ENV:
        assert root in Path(os.environ[name]).parents
    assert "YETO_RESOURCE_LEDGER" not in os.environ


def test_a_write_under_the_real_root_is_refused_and_recorded(tmp_path, monkeypatch):
    real = tmp_path / ".yeto"
    (real / "runs").mkdir(parents=True)
    monkeypatch.setitem(guard._state, "guarded", (str(real),))
    monkeypatch.setitem(guard._state, "violations", [])
    with pytest.raises(PermissionError, match="real ~/.yeto"):
        (real / "runs" / "r" / "resources.jsonl").parent.mkdir()
    with pytest.raises(PermissionError):
        (real / "runs" / "x.json").write_text("{}")
    (real / "runs").mkdir(exist_ok=True)  # writes nothing: not refused
    assert (real / "runs").is_dir() and not list((real / "runs").iterdir())
    assert [event for _test, event, _path in guard._state["violations"]] == ["os.mkdir", "open"]
    assert (tmp_path / "elsewhere.json").write_text("{}") == 2  # outside the root: untouched
