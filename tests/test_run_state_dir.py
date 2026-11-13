"""Read-only ``yeto run status`` / ``run down`` on a run that was never brought up (a typo
in the name) say so and exit 2 without creating ``<state root>/<run>/``; ``run up`` still
creates it. ``YETO_STATE_DIR`` moves the state root."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

from yeto.rl import up

CONFIG = '{"run": "typo", "store": "s3://b/x", "trainer": {"argv": []}, "pool": {"config": "p.json"}}'


def _config(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(CONFIG)
    return str(path)


@pytest.mark.parametrize("command", ["status", "down"])
def test_read_only_command_on_unknown_run_is_rc2_and_creates_nothing(tmp_path, monkeypatch, capsys, command):
    root = tmp_path / "rl-up"
    monkeypatch.setattr(up, "STATE_ROOT", root)
    assert up.main(types.SimpleNamespace(config=_config(tmp_path), rl_command=command)) == 2
    err = capsys.readouterr().err
    assert "typo" in err and "state directory" in err and "misspelled" in err
    assert not (root / "typo").exists()


def test_open_run_for_up_still_creates_state_dir(tmp_path, monkeypatch):
    root = tmp_path / "rl-up"
    monkeypatch.setattr(up, "STATE_ROOT", root)
    run = up.open_run(_config(tmp_path))
    assert run.dir == root / "typo" and run.dir.is_dir()


def test_read_only_open_run_on_existing_run_works(tmp_path, monkeypatch):
    root = tmp_path / "rl-up"
    (root / "typo").mkdir(parents=True)
    monkeypatch.setattr(up, "STATE_ROOT", root)
    assert up.open_run(_config(tmp_path), create=False).dir == root / "typo"


def _state_root_in_child(env: dict) -> str:
    code = "from yeto.rl import up; print(up.STATE_ROOT)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return out.stdout.strip()


def test_yeto_state_dir_overrides_state_root(tmp_path):
    env = dict(os.environ, YETO_STATE_DIR=str(tmp_path / "custom"))
    assert _state_root_in_child(env) == str(tmp_path / "custom")


def test_state_root_default_is_under_home(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "YETO_STATE_DIR"}
    env["HOME"] = str(tmp_path)
    assert _state_root_in_child(env) == str(tmp_path / ".yeto" / "rl-up")
