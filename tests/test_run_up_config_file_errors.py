"""``yeto run`` on a fresh control host: a missing or broken run config, or a state
directory it cannot create, ends with one line naming the file and exit code 2, not
a Python traceback."""

from __future__ import annotations

import types

import pytest

from yeto.rl import up


def _args(config: str, command: str = "status"):
    return types.SimpleNamespace(config=config, rl_command=command)


@pytest.mark.parametrize("content, words", [
    (None, "not found"),
    ('{"run": "x",', "not valid JSON"),
    ('{"run": "x", "bogus": 1, "store": "s", "trainer": {}, "pool": {}}', "unknown keys"),
])
def test_bad_run_config_is_one_line_and_rc2(tmp_path, monkeypatch, capsys, content, words):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    path = tmp_path / "run.json"
    if content is not None:
        path.write_text(content)
    assert up.main(_args(str(path))) == 2
    err = capsys.readouterr().err
    assert str(path) in err and words in err


def test_unwritable_state_dir_is_one_line_and_rc2(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(up, "STATE_ROOT", blocker / "rl-up")  # under a file: mkdir fails
    path = tmp_path / "run.json"
    path.write_text('{"run": "x", "store": "s3://b/x", "trainer": {"argv": []}, "pool": {"config": "p.json"}}')
    assert up.main(_args(str(path))) == 2
    assert "state directory" in capsys.readouterr().err
