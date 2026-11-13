"""``yeto run status`` on a run that was never brought up here (a typo in the name) says so
and exits 2 without creating ``<state root>/<run>/``; ``run up`` still creates it. ``run
down`` without local state (another control host, HOME wiped) warns, creates nothing under
the state root, and still runs every stop step the config drives, verdict NOT verified.
``YETO_STATE_DIR`` moves the state root."""

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


def test_status_on_unknown_run_is_rc2_and_creates_nothing(tmp_path, monkeypatch, capsys):
    root = tmp_path / "rl-up"
    monkeypatch.setattr(up, "STATE_ROOT", root)
    assert up.main(types.SimpleNamespace(config=_config(tmp_path), rl_command="status")) == 2
    err = capsys.readouterr().err
    assert "typo" in err and "state directory" in err and "misspelled" in err
    assert not (root / "typo").exists()


def _mock_down(monkeypatch, calls):
    monkeypatch.setattr(up.Run, "_cmd", lambda self, argv, *, log, timeout=None: calls.append(list(argv)) or 0)
    monkeypatch.setattr(up.Run, "sandbox_hosts", lambda self: [])
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    monkeypatch.setenv("YETO_RUN_DOWN_GRACE_S", "0")


def _steps(calls):
    return {name for argv in calls for name in ("pool-stop", "pool-status", "teardown") if name in argv} | {
        "trainer-down" for argv in calls if "down" in argv and "--config" not in argv}


def test_down_without_local_state_still_stops_what_the_config_names(tmp_path, monkeypatch, capsys):
    root = tmp_path / "rl-up"
    monkeypatch.setattr(up, "STATE_ROOT", root)
    monkeypatch.setattr(up.tempfile, "mkdtemp", lambda prefix="": str((tmp_path / "tmp").mkdir() or tmp_path / "tmp"))
    path = tmp_path / "run.json"
    path.write_text(CONFIG.replace('"pool"', '"sandbox": {"envs": "e.json"}, "pool"'))
    calls: list[list[str]] = []
    _mock_down(monkeypatch, calls)
    assert up.main(types.SimpleNamespace(config=str(path), rl_command="down")) == 1  # NOT verified
    cap = capsys.readouterr()
    err = cap.err + cap.out
    assert "no local state for typo" in err and "stopping what the config names" in err
    assert "NOT verified" in err and "no local state" in err.split("NOT verified")[1]
    assert not root.exists()
    assert _steps(calls) == {"trainer-down", "pool-stop", "pool-status", "teardown"}
    assert any(a[a.index("--config") + 1] == "p.json" for a in calls if "pool-stop" in a)


def test_down_with_local_state_runs_the_same_steps(tmp_path, monkeypatch, capsys):
    root = tmp_path / "rl-up"
    (root / "typo").mkdir(parents=True)
    monkeypatch.setattr(up, "STATE_ROOT", root)
    path = tmp_path / "run.json"
    path.write_text(CONFIG.replace('"pool"', '"sandbox": {"envs": "e.json"}, "pool"'))
    calls: list[list[str]] = []
    _mock_down(monkeypatch, calls)
    assert up.main(types.SimpleNamespace(config=str(path), rl_command="down")) == 0
    assert "no local state" not in capsys.readouterr().err
    assert _steps(calls) == {"trainer-down", "pool-stop", "pool-status", "teardown"}


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
