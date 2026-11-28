"""The session keeps out of the developer's real ~/.yeto (tests/yeto_home_guard.py)."""

from __future__ import annotations

import os
import sys
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


def test_the_real_yeto_output_is_guarded_in_this_process(tmp_path, monkeypatch):
    output = tmp_path / "home" / guard.OUTPUT_DIR
    output.mkdir(parents=True)
    monkeypatch.setitem(guard._state, "guarded", (str(tmp_path / "home" / ".yeto"), str(output)))
    monkeypatch.setitem(guard._state, "violations", [])
    with pytest.raises(PermissionError, match="real ~/yeto-output"):
        (output / "rl-island-0.jsonl").write_text("{}")
    assert [event for _test, event, _path in guard._state["violations"]] == ["open"]
    homes = {os.environ.get("HOME") or "", guard.pwd.getpwuid(os.getuid()).pw_dir} - {""}
    assert {os.path.join(os.path.abspath(home), guard.OUTPUT_DIR) for home in homes} <= set(guard._real_roots())


def test_a_child_that_names_the_real_yeto_output_is_refused(tmp_path, monkeypatch):
    import subprocess

    home = tmp_path / "home"
    output = home / guard.OUTPUT_DIR
    monkeypatch.setitem(guard._state, "guarded", (str(home / ".yeto"), str(output)))
    monkeypatch.setitem(guard._state, "violations", [])
    monkeypatch.setenv("HOME", str(home))
    command = "yeto-syncer --dispense --dispense-log ~/yeto-output/task-dispense.jsonl"
    with pytest.raises(PermissionError, match="real ~/yeto-output"):
        subprocess.run(["bash", "-c", f"echo {command}"], check=True)
    with pytest.raises(PermissionError):
        subprocess.Popen([sys.executable, "-c", "pass", f"--event-tape={output}/rl-island-0.jsonl"])
    with pytest.raises(PermissionError):
        subprocess.run("echo $HOME/yeto-output/x ${HOME}/yeto-output/y", shell=True, check=True)
    assert [event for _test, event, _path in guard._state["violations"]] == ["subprocess"] * 3
    assert not output.exists()
    # paths of another home, or under tmp_path, are the test's own
    elsewhere = {**os.environ, "HOME": str(tmp_path / "node-home")}
    subprocess.run(["bash", "-c", f"echo {command}"], env=elsewhere, check=True, capture_output=True)
    subprocess.run(["echo", str(tmp_path / guard.OUTPUT_DIR), "~/yeto-outputs", "node:yeto-output/"],
                   check=True, capture_output=True)
    # another machine's ~/yeto-output: an rsync or ssh argument, or a remote spec in a shell line
    subprocess.run(["true", "node:~/yeto-output/task-dispense.jsonl"], check=True)
    subprocess.run(["bash", "-c", "true node:~/yeto-output/rl-island-0.jsonl"], check=True)
    ssh = tmp_path / "ssh"  # runs its command line on the node, not here
    ssh.write_text("#!/bin/sh\nexit 0\n")
    ssh.chmod(0o755)
    subprocess.run([str(ssh), "node", "mkdir -p ~/yeto-output && cat ~/yeto-output/yeto-tape.jsonl"], check=True)
    assert len(guard._state["violations"]) == 3
    with pytest.raises(PermissionError):  # past a wrapper, the shell still runs it here
        subprocess.run(["timeout", "60", "bash", "-c", f"echo {command}"], check=True)


def test_a_local_file_tool_that_writes_the_real_yeto_output_by_absolute_path_is_refused(tmp_path, monkeypatch):
    import subprocess

    home = tmp_path / "home"
    output = home / guard.OUTPUT_DIR
    monkeypatch.setitem(guard._state, "guarded", (str(home / ".yeto"), str(output)))
    monkeypatch.setitem(guard._state, "violations", [])
    source = tmp_path / "tape.jsonl"
    source.write_text("{}\n")
    for argv in (["mkdir", "-p", str(output)],
                 ["cp", str(source), f"{output}/yeto-tape.jsonl"],
                 ["rsync", "-a", f"{source}", f"{output}/"],
                 ["/bin/mkdir", str(output / "sub")],
                 ["timeout", "30", "cp", str(source), str(output)],
                 ["dd", f"if={source}", f"of={output}/x"],
                 ["install", "-D", f"--target-directory={output}", str(source)]):
        with pytest.raises(PermissionError, match="real ~/yeto-output"):
            subprocess.run(argv, check=True)
    assert len(guard._state["violations"]) == 7 and not output.exists()
    # the same tools on paths of the test, of another machine, or a name that only starts the same
    subprocess.run(["mkdir", "-p", str(tmp_path / "out" / guard.OUTPUT_DIR)], check=True)
    subprocess.run(["cp", str(source), str(tmp_path / "out" / guard.OUTPUT_DIR)], check=True)
    subprocess.run(["mkdir", "-p", f"{output}s"], check=True)
    rsync = tmp_path / "rsync"  # a fake: the remote side of the copy is the node's
    rsync.write_text("#!/bin/sh\nexit 0\n")
    rsync.chmod(0o755)
    subprocess.run([str(rsync), "-az", f"node:{output}/", str(tmp_path / "copy")], check=True)
    subprocess.run([str(rsync), "-az", str(source), f"node:{output}/"], check=True)
    assert len(guard._state["violations"]) == 7 and not output.exists()
